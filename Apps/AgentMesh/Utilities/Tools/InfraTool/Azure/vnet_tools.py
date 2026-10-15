import logging
from typing import Optional
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

# --- Absolute Import Fix for Registry Discovery ---
from Apps.AgentMesh.Utilities.Tools.InfraTool.Azure.auth_utils import AzureProvider
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry

logger = logging.getLogger(__name__)

# --- Tool Definitions ---

@tool
async def azure_list_vnets(resource_group: str, config: RunnableConfig = None) -> str:
    """
    List all Virtual Networks in a specific Azure Resource Group.
    If the user hasn't provided a resource group name, you must ask them for one first.
    """
    try:
        network_client = await AzureProvider.get_network_client(config)
        logger.info(f"Listing VNets in resource group: {resource_group}")
        vnets = network_client.virtual_networks.list(resource_group)
        
        vnet_list = []
        for vnet in vnets:
            vnet_list.append(f"- Name: {vnet.name}, Location: {vnet.location}, Address Space: {vnet.address_space.address_prefixes}")
            
        if not vnet_list:
            return f"No Virtual Networks found in resource group '{resource_group}'."
            
        return f"Virtual Networks in '{resource_group}':\n" + "\n".join(vnet_list)
    except Exception as e:
        return f"Error listing VNets: {str(e)}"

# --- Registry Integration ---

def register_vnet_tools():
    """ Registers Azure VNet tools. """
    
    registry.register(
        azure_list_vnets,
        ToolMeta(
            name="azure_list_vnets",
            display_name="Azure: List Virtual Networks",
            description="Fetch a list of all VNets in a specific resource group.",
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
register_vnet_tools()
