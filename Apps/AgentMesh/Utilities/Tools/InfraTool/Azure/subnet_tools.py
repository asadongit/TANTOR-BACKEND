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
async def azure_list_subnets(vnet_name: str, resource_group: str, config: RunnableConfig = None) -> str:
    """
    List all subnets within a specific Virtual Network (VNet).
    If the user hasn't specified a VNet name or resource group, you must ask them.
    """
    try:
        network_client = await AzureProvider.get_network_client(config)
        logger.info(f"Listing subnets in VNet: {vnet_name} (RG: {resource_group})")
        subnets = network_client.subnets.list(resource_group, vnet_name)
        
        subnet_list = []
        for subnet in subnets:
            subnet_list.append(f"- Name: {subnet.name}, Address Prefix: {subnet.address_prefix}")
            
        if not subnet_list:
            return f"No subnets found for VNet '{vnet_name}' in resource group '{resource_group}'."
            
        return f"Subnets in VNet '{vnet_name}':\n" + "\n".join(subnet_list)
    except Exception as e:
        return f"Error listing subnets: {str(e)}"

# --- Registry Integration ---

def register_subnet_tools():
    """ Registers Azure Subnet tools. """
    
    registry.register(
        azure_list_subnets,
        ToolMeta(
            name="azure_list_subnets",
            display_name="Azure: List Subnets",
            description="Fetch a list of all subnets in a specific VNet.",
            category="Infrastructure",
            package="azure",
            version="1.2.0",
            configs=[
                {
                    "name": "vnet_name",
                    "type": "string",
                    "description": "Azure Virtual Network name",
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
register_subnet_tools()
