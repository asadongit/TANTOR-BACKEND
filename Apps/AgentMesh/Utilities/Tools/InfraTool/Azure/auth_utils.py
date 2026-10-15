import os
import logging
from typing import Optional, Dict
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.network import NetworkManagementClient
from langchain_core.runnables import RunnableConfig
from Apps.AgentMesh.Utilities.State.ToolStateManager import ToolStateManager

logger = logging.getLogger(__name__)

# --- Persistent Credential Helpers (Wrappers for ToolStateManager) ---

async def save_credentials(thread_id: str, creds: Dict[str, str]):
    """Saves or updates Azure credentials for a specific thread in DB."""
    logger.info(f"Saving Azure credentials to DB for thread: {thread_id}")
    await ToolStateManager.save_state(thread_id, "azure_creds", creds)

async def get_credentials(thread_id: str) -> Optional[Dict[str, str]]:
    """Retrieves credentials for a specific thread from DB."""
    return await ToolStateManager.get_state(thread_id, "azure_creds")

# --- Azure Provider Logic ---

class AzureProvider:
    @staticmethod
    def get_thread_id(config: RunnableConfig) -> str:
        """Extracts thread_id from LangChain config with mandatory security checks."""
        if not config:
            raise ValueError("Configuration missing: Tools must be run within a valid graph context.")
        
        # Determine if config is a dictionary or a RunnableConfig object
        if isinstance(config, dict):
            configurable = config.get("configurable", {})
        else:
            configurable = getattr(config, "configurable", {})
            
        # Extract thread_id from common locations
        thread_id = None
        if isinstance(configurable, dict):
            thread_id = configurable.get("thread_id") or configurable.get("context", {}).get("thread_id")
        
        # 🛡️ SECURITY: Fail if thread_id is missing or unsafe
        if not thread_id or thread_id == "default_thread":
            logger.error("Security Bypass Attempt: Tool invoked without a unique thread identity.")
            raise ValueError("Authentication Denied: A unique, non-default Thread ID is required for security isolation.")
        
        logger.debug(f"Resolved Secure Thread ID: {thread_id}")
        return thread_id

    @classmethod
    async def get_clients(cls, config: RunnableConfig):
        """Returns initialized Azure management clients for the current thread."""
        thread_id = cls.get_thread_id(config)
        creds_data = await get_credentials(thread_id)

        if not creds_data:
            raise ValueError(f"Azure credentials NOT configured for thread {thread_id}. Run 'azure_configure_auth' first.")

        # In production, we use ClientSecretCredential for automation/service principals
        credential = ClientSecretCredential(
            tenant_id=creds_data['tenant_id'],
            client_id=creds_data['client_id'],
            client_secret=creds_data['client_secret']
        )
        
        subscription_id = creds_data['subscription_id']
        
        return {
            "compute": ComputeManagementClient(credential, subscription_id),
            "resource": ResourceManagementClient(credential, subscription_id),
            "network": NetworkManagementClient(credential, subscription_id),
            "subscription_id": subscription_id
        }

    @classmethod
    async def get_compute_client(cls, config: RunnableConfig):
        clients = await cls.get_clients(config)
        return clients["compute"]

    @classmethod
    async def get_resource_client(cls, config: RunnableConfig):
        clients = await cls.get_clients(config)
        return clients["resource"]

    @classmethod
    async def get_network_client(cls, config: RunnableConfig):
        clients = await cls.get_clients(config)
        return clients["network"]
