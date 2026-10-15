import logging
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

# --- Absolute Import Fix for Registry Discovery ---
from Apps.AgentMesh.Utilities.Tools.InfraTool.Azure.auth_utils import AzureProvider
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry

logger = logging.getLogger(__name__)

# --- Tool Definitions ---

@tool
async def azure_list_public_ips(resource_group: str, config: RunnableConfig = None) -> str:
    """
    List all Public IP Addresses in a specific Azure Resource Group.
    If the user hasn't provided a resource group name, you must ask them.
    """
    try:
        network_client = await AzureProvider.get_network_client(config)
        logger.info(f"Listing Public IPs in resource group: {resource_group}")
        ips = network_client.public_ip_addresses.list(resource_group)
        
        ip_list = []
        for ip in ips:
            ip_val = ip.ip_address if ip.ip_address else "Allocating..."
            ip_list.append(f"- Name: {ip.name}, IP: {ip_val}, Location: {ip.location}, Method: {ip.public_ip_allocation_method}")
            
        if not ip_list:
            return f"No Public IP Addresses found in resource group '{resource_group}'."
            
        return f"Public IPs in '{resource_group}':\n" + "\n".join(ip_list)
    except Exception as e:
        return f"Error listing Public IPs: {str(e)}"

@tool
async def azure_associate_public_ip(
    nic_name: str,
    public_ip_name: str,
    resource_group: str,
    config: RunnableConfig = None
) -> str:
    """
    Associate an existing Public IP address with an existing Network Interface Card (NIC).
    """
    try:
        network_client = await AzureProvider.get_network_client(config)
        
        # 1. Get the NIC and its config
        logger.info(f"Fetching NIC: {nic_name}")
        nic = network_client.network_interfaces.get(resource_group, nic_name)
        
        # 2. Get the Public IP
        logger.info(f"Fetching Public IP: {public_ip_name}")
        public_ip = network_client.public_ip_addresses.get(resource_group, public_ip_name)
        
        # 3. Update the NIC configuration
        if not nic.ip_configurations:
            return f"NIC '{nic_name}' has no IP configurations."
            
        # Update the primary IP configuration
        nic.ip_configurations[0].public_ip_address = public_ip
        
        # 4. Save the NIC
        logger.info(f"Updating NIC '{nic_name}' with Public IP '{public_ip_name}'")
        poller = network_client.network_interfaces.begin_create_or_update(resource_group, nic_name, nic)
        poller.result()
        
        return f"Successfully associated Public IP '{public_ip_name}' to NIC '{nic_name}'."
    except Exception as e:
        return f"Failed to associate Public IP: {str(e)}"

# --- Registry Integration ---

def register_ip_tools():
    """ Registers Azure Public IP tools. """
    
    registry.register(
        azure_list_public_ips,
        ToolMeta(
            name="azure_list_public_ips",
            display_name="Azure: List Public IPs",
            description="Fetch a list of all Public IP addresses in a specific resource group.",
            category="Infrastructure",
            package="azure",
            version="1.2.0",
            configs=[
                {
                    "name": "resource_group",
                    "type": "string",
                    "description": "Azure Resource Group name",
                    "required": True
                }
            ]
        )
    )

    registry.register(
        azure_associate_public_ip,
        ToolMeta(
            name="azure_associate_public_ip",
            display_name="Azure: Associate Public IP",
            description="Link an existing Public IP to an existing Network Interface (NIC).",
            category="Infrastructure",
            package="azure",
            version="1.2.0",
            configs=[
                {
                    "name": "nic_name",
                    "type": "string",
                    "description": "The name of the target NIC",
                    "required": True
                },
                {
                    "name": "public_ip_name",
                    "type": "string",
                    "description": "The name of the Public IP to attach",
                    "required": True
                },
                {
                    "name": "resource_group",
                    "type": "string",
                    "description": "Azure Resource Group name",
                    "required": True
                }
            ]
        )
    )

# Automatically register on import
register_ip_tools()
