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
async def azure_list_resource_groups(config: RunnableConfig = None) -> str:
    """
    List all Azure Resource Groups in the current subscription.
    """
    try:
        resource_client = await AzureProvider.get_resource_client(config)
        groups = resource_client.resource_groups.list()
        
        group_list = []
        for group in groups:
            group_list.append(f"- Name: {group.name}, Location: {group.location}")
            
        if not group_list:
            return "No resource groups found in this subscription."
            
        return "Current Azure Resource Groups:\n" + "\n".join(group_list)
    except Exception as e:
        return f"Error listing resource groups: {str(e)}"

@tool
async def azure_create_resource_group(resource_group: str, location: str = "eastus", config: RunnableConfig = None) -> str:
    """
    Create a new Azure Resource Group in a specific location.
    """
    try:
        resource_client = await AzureProvider.get_resource_client(config)
        logger.info(f"Creating Resource Group: {resource_group} in {location}")
        resource_client.resource_groups.create_or_update(resource_group, {"location": location})
        return f"Successfully created resource group '{resource_group}' in '{location}'."
    except Exception as e:
        return f"Error creating resource group: {str(e)}"

@tool
async def azure_list_resources(config: RunnableConfig = None) -> str:
    """
    List all resources in the configured Azure subscription across all resource groups.
    """
    try:
        resource_client = await AzureProvider.get_resource_client(config)
        resources = resource_client.resources.list()
        
        resource_list = []
        for resource in resources:
            resource_list.append(f"- Name: {resource.name}, Type: {resource.type}, Location: {resource.location}")
            
        if not resource_list:
            return "No resources found in this subscription."
            
        return "Current Azure Resources:\n" + "\n".join(resource_list)
    except Exception as e:
        return f"Error listing resources: {str(e)}"

@tool
async def azure_delete_resource_group(resource_group: str, config: RunnableConfig = None) -> str:
    """
    Delete an entire Azure Resource Group and everything inside it.
    Use with extreme caution as this is irreversible.
    """
    try:
        resource_client = await AzureProvider.get_resource_client(config)
        logger.warning(f"Deleting resource group: {resource_group}")
        # Start deletion - this can take a long time
        resource_client.resource_groups.begin_delete(resource_group)
        return f"Deletion of resource group '{resource_group}' has been initiated."
    except Exception as e:
        return f"Error deleting resource group: {str(e)}"

@tool
async def azure_delete_resource(
    resource_name: str,
    resource_group: str,
    resource_type: str,
    api_version: str = "2021-04-01",
    config: RunnableConfig = None
) -> str:
    """
    Delete a specific Azure resource by its name, group, and type.
    Resource type example: 'Microsoft.Compute/virtualMachines' or 'Microsoft.Network/virtualNetworks'.
    """
    try:
        resource_client = await AzureProvider.get_resource_client(config)
        
        # Parse type into provider and type name (e.g. Microsoft.Compute/virtualMachines)
        parts = resource_type.split('/')
        provider = parts[0]
        type_name = "/".join(parts[1:])
        
        logger.warning(f"Deleting resource: {resource_name} of type {resource_type}")
        poller = resource_client.resources.begin_delete(
            resource_group_name=resource_group,
            resource_provider_namespace=provider,
            parent_resource_path="",
            resource_type=type_name,
            resource_name=resource_name,
            api_version=api_version
        )
        poller.result()
        return f"Successfully deleted {resource_type} '{resource_name}'."
    except Exception as e:
        return f"Failed to delete resource: {str(e)}"

# --- Registry Integration ---

def register_resource_tools():
    """ Registers general Azure Resource tools. """
    
    registry.register(
        azure_list_resources,
        ToolMeta(
            name="azure_list_resources",
            display_name="Azure: List Resources",
            description="Fetch a list of all resources in the subscription.",
            category="Infrastructure",
            package="azure",
            version="1.2.0"
        )
    )

    registry.register(
        azure_delete_resource_group,
        ToolMeta(
            name="azure_delete_resource_group",
            display_name="Azure: Delete Resource Group",
            description="Permanently remove a resource group and all its contents.",
            category="Infrastructure",
            package="azure",
            version="1.2.0"
        )
    )

    registry.register(
        azure_list_resource_groups,
        ToolMeta(
            name="azure_list_resource_groups",
            display_name="Azure: List Resource Groups",
            description="Fetch a list of all resource groups in the subscription.",
            category="Infrastructure",
            package="azure",
            version="1.2.0"
        )
    )

    registry.register(
        azure_create_resource_group,
        ToolMeta(
            name="azure_create_resource_group",
            display_name="Azure: Create Resource Group",
            description="Create a new resource group in a specified location.",
            category="Infrastructure",
            package="azure",
            version="1.2.0"
        )
    )

    registry.register(
        azure_delete_resource,
        ToolMeta(
            name="azure_delete_resource",
            display_name="Azure: Delete Resource",
            description="Generic tool to delete any specific Azure resource.",
            category="Infrastructure",
            package="azure",
            version="1.2.0"
        )
    )

# Automatically register on import
register_resource_tools()
