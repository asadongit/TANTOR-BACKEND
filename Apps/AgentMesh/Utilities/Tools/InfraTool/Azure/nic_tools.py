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
async def azure_list_nics(resource_group: str, config: RunnableConfig = None) -> str:
    """
    List all Network Interface Cards (NICs) in a specific Azure Resource Group.
    If the user hasn't provided a resource group name, you must ask them.
    """
    try:
        network_client = await AzureProvider.get_network_client(config)
        logger.info(f"Listing NICs in resource group: {resource_group}")
        nics = network_client.network_interfaces.list(resource_group)
        
        nic_list = []
        for nic in nics:
            private_ip = nic.ip_configurations[0].private_ip_address if nic.ip_configurations else "N/A"
            nic_list.append(f"- Name: {nic.name}, Location: {nic.location}, Private IP: {private_ip}")
            
        if not nic_list:
            return f"No NICs found in resource group '{resource_group}'."
            
        return f"Network Interfaces in '{resource_group}':\n" + "\n".join(nic_list)
    except Exception as e:
        return f"Error listing NICs: {str(e)}"

# --- Registry Integration ---

def register_nic_tools():
    """ Registers Azure NIC tools. """
    
    registry.register(
        azure_list_nics,
        ToolMeta(
            name="azure_list_nics",
            display_name="Azure: List NICs",
            description="Fetch a list of all Network Interfaces in a specific resource group.",
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

# Automatically register on import
register_nic_tools()
