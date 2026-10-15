import logging
from typing import Optional, Dict, Any
from langchain_core.runnables import RunnableConfig
from Apps.AgentMesh.Utilities.State.ToolStateManager import ToolStateManager

logger = logging.getLogger(__name__)

# --- Persistent Configuration Helpers ---

async def save_gsheet_config(thread_id: str, config_data: Dict[str, Any]):
    """Saves the Google Sheet configuration (link, tab, etc) for a specific thread."""
    logger.info(f"Saving Google Sheet config for thread: {thread_id}")
    await ToolStateManager.save_state(thread_id, "gsheet_config", config_data)

async def get_gsheet_config(thread_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves the Google Sheet configuration for a specific thread."""
    return await ToolStateManager.get_state(thread_id, "gsheet_config")

class GoogleSheetsProvider:
    @staticmethod
    def get_thread_id(config: RunnableConfig) -> str:
        """Extracts thread_id from LangChain config with security checks."""
        if not config:
            raise ValueError("Configuration missing: Tool must be run within a valid graph context.")
        
        configurable = config.get("configurable", {}) if isinstance(config, dict) else getattr(config, "configurable", {})
        
        thread_id = None
        if isinstance(configurable, dict):
            thread_id = configurable.get("thread_id") or configurable.get("context", {}).get("thread_id")
        
        if not thread_id or thread_id == "default_thread":
            raise ValueError("Security Isolation: A unique Thread ID is required to access session-specific Google Sheets.")
        
        return thread_id

    @classmethod
    async def get_config(cls, thread_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves config from DB."""
        return await ToolStateManager.get_state(thread_id, "gsheet_config")
