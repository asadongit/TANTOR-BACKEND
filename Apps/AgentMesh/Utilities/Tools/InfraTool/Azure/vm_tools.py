import logging
from typing import List, Optional, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

# --- Absolute Import Fix for Registry Discovery ---
from Apps.AgentMesh.Utilities.Tools.InfraTool.Azure.auth_utils import AzureProvider, save_credentials
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry

logger = logging.getLogger(__name__)

# --- Tool Definitions ---

@tool
async def azure_configure_auth(
    subscription_id: str,
    client_id: str,
    client_secret: str,
    tenant_id: str,
    config: RunnableConfig = None
) -> str:
    """
    Configure Azure credentials for the current conversation thread.
    This is required before calling any other Azure tools.
    """
    thread_id = AzureProvider.get_thread_id(config)
    
    # Save to persistent database (UPSERT logic handled in provider)
    await save_credentials(thread_id, {
        "subscription_id": subscription_id,
        "client_id": client_id,
        "client_secret": client_secret,
        "tenant_id": tenant_id
    })
    
    return f"Azure credentials successfully configured and PERSISTED for thread '{thread_id}'."

@tool
async def azure_list_vms(resource_group: str, config: RunnableConfig = None) -> str:
    """
    List Virtual Machines in a specific Azure Resource Group.
    If the user hasn't provided a resource group name, you must ask them for one first.
    """
    try:
        compute_client = await AzureProvider.get_compute_client(config)
        if resource_group:
            logger.info(f"Listing VMs in resource group: {resource_group}")
            vms = compute_client.virtual_machines.list(resource_group)
        else:
            logger.info("Listing all VMs in subscription")
            vms = compute_client.virtual_machines.list_all()
        
        vm_list = []
        for vm in vms:
            vm_list.append(f"- Name: {vm.name}, Location: {vm.location}, Size: {vm.hardware_profile.vm_size}")
            
        if not vm_list:
            location_str = f" in resource group '{resource_group}'" if resource_group else ""
            return f"No Virtual Machines found{location_str} in this subscription."
            
        header = f"Azure Virtual Machines in '{resource_group}':\n" if resource_group else "All Azure Virtual Machines:\n"
        return header + "\n".join(vm_list)
    except Exception as e:
        return f"Error listing VMs: {str(e)}"

@tool
async def azure_create_vm(
    vm_name: str,
    resource_group: str,
    location: str = "eastus",
    vm_size: str = "Standard_B1s",
    admin_username: str = "azureuser",
    admin_password: Optional[str] = None,
    public_ip_name: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Create a new Azure Virtual Machine with automatic Networking.
    Supports custom admin credentials and using existing Public IPs.
    """
    try:
        if not admin_password:
            raise ValueError("A VM admin password must be supplied securely")
        clients = await AzureProvider.get_clients(config)
        resource_client = clients["resource"]
        network_client = clients["network"]
        compute_client = clients["compute"]
        
        # 1. Create Resource Group
        logger.info(f"Creating Resource Group: {resource_group}")
        resource_client.resource_groups.create_or_update(resource_group, {"location": location})
        
        # 2. Virtual Network
        vnet_name = f"{vm_name}-vnet"
        vnet_poller = network_client.virtual_networks.begin_create_or_update(
            resource_group, vnet_name,
            {
                "location": location,
                "address_space": {"address_prefixes": ["10.0.0.0/16"]}
            }
        )
        vnet_poller.result()
        
        # 3. Subnet
        subnet_name = f"{vm_name}-subnet"
        subnet_poller = network_client.subnets.begin_create_or_update(
            resource_group, vnet_name, subnet_name,
            {"address_prefix": "10.0.1.0/24"}
        )
        subnet = subnet_poller.result()
        
        # 3.5. Network Security Group (NSG) - REQUIRED for Standard SKU
        nsg_name = f"{vm_name}-nsg"
        logger.info(f"Creating/Updating NSG: {nsg_name}")
        nsg_poller = network_client.network_security_groups.begin_create_or_update(
            resource_group, nsg_name,
            {
                "location": location,
                "security_rules": [
                    {
                        "name": "AllowSSHInbound",
                        "protocol": "Tcp",
                        "source_port_range": "*",
                        "destination_port_range": "22",
                        "source_address_prefix": "*",
                        "destination_address_prefix": "*",
                        "access": "Allow",
                        "priority": 1000,
                        "direction": "Inbound",
                    },
                    {
                        "name": "AllowHTTPInbound",
                        "protocol": "Tcp",
                        "source_port_range": "*",
                        "destination_port_range": "80",
                        "source_address_prefix": "*",
                        "destination_address_prefix": "*",
                        "access": "Allow",
                        "priority": 1001,
                        "direction": "Inbound",
                    }
                ]
            }
        )
        nsg = nsg_poller.result()
        
        # 4. Public IP Management
        public_ip = None
        ip_name = "N/A"
        
        if public_ip_name:
            if public_ip_name.lower() == "none":
                logger.info("Skipping Public IP assignment.")
            else:
                try:
                    logger.info(f"Retrieving existing Public IP: {public_ip_name}")
                    public_ip = network_client.public_ip_addresses.get(resource_group, public_ip_name)
                    ip_name = public_ip_name
                except Exception as e:
                    logger.error(f"Existing Public IP '{public_ip_name}' not found: {e}")
                    raise ValueError(f"Target Public IP '{public_ip_name}' not found in resource group '{resource_group}'.")
        
        # Create new Public IP if not provided and not explicitly skipped
        if not public_ip and (not public_ip_name or public_ip_name.lower() != "none"):
            ip_name = f"{vm_name}-ip"
            logger.info(f"Creating NEW Public IP (STANDARD SKU): {ip_name}")
            ip_poller = network_client.public_ip_addresses.begin_create_or_update(
                resource_group, ip_name,
                {
                    "location": location,
                    "sku": {"name": "Standard"},
                    "public_ip_allocation_method": "Static" # Standard SKU MUST be Static
                }
            )
            public_ip = ip_poller.result()
        
        # 5. Network Interface (NIC)
        nic_name = f"{vm_name}-nic"
        ip_config = {
            "name": f"{vm_name}-ip-config",
            "subnet": {"id": subnet.id}
        }
        
        if public_ip:
            ip_config["public_ip_address"] = {"id": public_ip.id}
            
        nic_poller = network_client.network_interfaces.begin_create_or_update(
            resource_group, nic_name,
            {
                "location": location,
                "ip_configurations": [ip_config],
                "network_security_group": {"id": nsg.id}
            }
        )
        nic = nic_poller.result()
        
        # 6. VM Parameters (Ubuntu 22.04 LTS Default)
        vm_params = {
            "location": location,
            "hardware_profile": {"vm_size": vm_size},
            "storage_profile": {
                "image_reference": {
                    "publisher": "Canonical",
                    "offer": "0001-com-ubuntu-server-jammy",
                    "sku": "22_04-lts",
                    "version": "latest"
                },
                "os_disk": {
                    "name": f"{vm_name}-osdisk",
                    "caching": "ReadWrite",
                    "create_option": "FromImage"
                }
            },
            "os_profile": {
                "computer_name": vm_name,
                "admin_username": admin_username,
                "admin_password": admin_password
            },
            "network_profile": {
                "network_interfaces": [{"id": nic.id}]
            }
        }
        
        logger.info(f"Starting VM creation: {vm_name}")
        vm_poller = compute_client.virtual_machines.begin_create_or_update(resource_group, vm_name, vm_params)
        vm_poller.result()
        
        return f"Successfully created VM '{vm_name}' in resource group '{resource_group}' at {location}. Resources: [NIC: {nic_name}, Public IP: {ip_name}, NSG: {nsg_name}]"
    except Exception as e:
        return f"Failed to create VM: {str(e)}"

@tool
async def azure_start_vm(
    vm_name: str,
    resource_group: str,
    config: RunnableConfig = None
) -> str:
    """
    Start a powered-off (deallocated) Azure Virtual Machine.
    """
    try:
        compute_client = await AzureProvider.get_compute_client(config)
        logger.info(f"Starting VM: {vm_name}")
        poller = compute_client.virtual_machines.begin_start(resource_group, vm_name)
        poller.result()
        return f"Virtual Machine '{vm_name}' successfully started."
    except Exception as e:
        return f"Error starting VM: {str(e)}"

@tool
async def azure_stop_vm(
    vm_name: str,
    resource_group: str,
    config: RunnableConfig = None
) -> str:
    """
    Stop and Deallocate an Azure Virtual Machine (saves costs).
    """
    try:
        compute_client = await AzureProvider.get_compute_client(config)
        logger.info(f"Stopping (deallocating) VM: {vm_name}")
        poller = compute_client.virtual_machines.begin_deallocate(resource_group, vm_name)
        poller.result()
        return f"Virtual Machine '{vm_name}' successfully stopped and deallocated."
    except Exception as e:
        return f"Error stopping VM: {str(e)}"

@tool
async def azure_delete_vm(
    vm_name: str,
    resource_group: str,
    config: RunnableConfig = None
) -> str:
    """
    Permanently delete an Azure Virtual Machine.
    """
    try:
        compute_client = await AzureProvider.get_compute_client(config)
        logger.info(f"Deleting VM: {vm_name}")
        poller = compute_client.virtual_machines.begin_delete(resource_group, vm_name)
        poller.result()
        return f"Virtual Machine '{vm_name}' successfully deleted."
    except Exception as e:
        return f"Error deleting VM: {str(e)}"

# --- Registry Integration ---

def register_vm_tools():
    """ Registers Azure VM tools with rich metadata for the Frontend UI. """
    
    registry.register(
        azure_configure_auth,
        ToolMeta(
            name="azure_configure_auth",
            display_name="Azure: Configure Credentials",
            description="Initialize Azure credentials for this conversation thread.",
            category="Infrastructure",
            package="azure",
            version="1.2.0",
            allow_duplicate=False,
            ai_visible_fields=["subscription_id", "client_id", "client_secret", "tenant_id"],
            configs=[
                {
                    "name": "subscription_id",
                    "type": "string",
                    "description": "Azure Subscription ID",
                    "required": True
                },
                {
                    "name": "tenant_id",
                    "type": "string",
                    "description": "Azure Tenant ID (Directory ID)",
                    "required": True
                },
                {
                    "name": "client_id",
                    "type": "string",
                    "description": "Service Principal Client ID (App ID)",
                    "required": True
                },
                {
                    "name": "client_secret",
                    "type": "password",
                    "description": "Service Principal Client Secret",
                    "required": True
                }
            ]
        )
    )

    registry.register(
        azure_list_vms,
        ToolMeta(
            name="azure_list_vms",
            display_name="Azure: List Virtual Machines",
            description="Fetch a list of all VMs or filter by a specific resource group.",
            category="Infrastructure",
            package="azure",
            version="1.2.0",
            configs=[
                {
                    "name": "location",
                    "type": "string",
                    "description": "Azure Region",
                    "options": ["eastus", "westus", "westeurope", "southeastasia", "centralus"],
                    "default": "eastus"
                },
                {
                    "name": "vm_size",
                    "type": "string",
                    "description": "Azure VM Size (e.g., Standard_B1s)",
                    "default": "Standard_B1s"
                },
                {
                    "name": "admin_username",
                    "type": "string",
                    "description": "Admin username for the VM",
                    "default": "azureuser"
                },
                {
                    "name": "admin_password",
                    "type": "password",
                    "description": "Admin password for the VM",
                    "required": True
                },
                {
                    "name": "resource_group",
                    "type": "string",
                    "description": "The Azure Resource Group name to list VMs from.",
                    "required": True
                },
                {
                    "name": "public_ip_name",
                    "type": "string",
                    "description": "Optional: Name of an existing Public IP to use, or 'None' to skip.",
                    "required": False
                }
            ]
        )
    )

    registry.register(
        azure_create_vm,
        ToolMeta(
            name="azure_create_vm",
            display_name="Azure: Create Virtual Machine",
            description="Provision a new Azure VM with automatic networking.",
            category="Infrastructure",
            package="azure",
            version="1.2.0",
            configs=[
                {
                    "name": "location",
                    "type": "string",
                    "description": "Azure Region",
                    "options": ["eastus", "westus", "westeurope", "southeastasia", "centralus"],
                    "default": "eastus"
                },
                {
                    "name": "vm_size",
                    "type": "string",
                    "description": "Azure VM Size (e.g., Standard_B1s)",
                    "default": "Standard_B1s"
                },
                {
                    "name": "admin_username",
                    "type": "string",
                    "description": "Admin username for the VM",
                    "default": "azureuser"
                },
                {
                    "name": "admin_password",
                    "type": "password",
                    "description": "Admin password for the VM",
                    "required": True
                },
                {
                    "name": "resource_group",
                    "type": "string",
                    "description": "The Azure Resource Group name",
                    "required": True
                },
                {
                    "name": "public_ip_name",
                    "type": "string",
                    "description": "Optional: Existing Public IP to use, or 'None' to skip.",
                    "required": False
                }
            ]
        )
    )

    registry.register(
        azure_start_vm,
        ToolMeta(
            name="azure_start_vm",
            display_name="Azure: Start Virtual Machine",
            description="Power on an existing deallocated VM.",
            category="Infrastructure",
            package="azure",
            version="1.2.0"
        )
    )

    registry.register(
        azure_stop_vm,
        ToolMeta(
            name="azure_stop_vm",
            display_name="Azure: Stop Virtual Machine",
            description="Shut down and deallocate a VM to save costs.",
            category="Infrastructure",
            package="azure",
            version="1.2.0"
        )
    )

    registry.register(
        azure_delete_vm,
        ToolMeta(
            name="azure_delete_vm",
            display_name="Azure: Delete Virtual Machine",
            description="Permanently delete a Virtual Machine.",
            category="Infrastructure",
            package="azure",
            version="1.2.0"
        )
    )

# Automatically register on import
register_vm_tools()
