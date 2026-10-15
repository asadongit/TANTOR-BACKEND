import logging
import json
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilySearch

from Apps.AgentMesh.Utilities.State.ToolStateManager import ToolStateManager
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry

logger = logging.getLogger(__name__)

# --- Persistent Credential Helpers ---

async def save_tavily_credentials(thread_id: str, api_key: str):
    """Saves or updates Tavily credentials for a specific thread in DB."""
    logger.info(f"Saving Tavily credentials to DB for thread: {thread_id}")
    await ToolStateManager.save_state(thread_id, "tavily_creds", {"api_key": api_key})

async def get_tavily_credentials(thread_id: str) -> Optional[str]:
    """Retrieves Tavily credentials for a specific thread from DB."""
    creds = await ToolStateManager.get_state(thread_id, "tavily_creds")
    return creds.get("api_key") if creds else None

# --- Helper ---

def get_thread_id(config: RunnableConfig) -> str:
    """Extracts thread_id from LangChain config with mandatory security checks."""
    if not config:
        raise ValueError("Configuration missing: Tools must be run within a valid graph context.")
    
    configurable = config.get("configurable", {}) if isinstance(config, dict) else getattr(config, "configurable", {})
    
    thread_id = None
    if isinstance(configurable, dict):
        thread_id = configurable.get("thread_id") or configurable.get("context", {}).get("thread_id")
    
    if not thread_id or thread_id == "default_thread":
        logger.error("Security Bypass Attempt: Tool invoked without a unique thread identity.")
        raise ValueError("Authentication Denied: A unique, non-default Thread ID is required for security isolation.")
    
    return thread_id

# --- Tool Definition ---

@tool
async def tavily_search(
    query: str,
    api_key: Optional[str] = None,
    max_results: int = 5,
    search_depth: str = "basic",
    include_answer: bool = False,
    config: RunnableConfig = None
) -> str:
    """
    Search the web using Tavily for real-time, accurate, and factual results.
    Useful for finding current news, facts, and detailed information.
    """
    try:
        thread_id = get_thread_id(config)
        
        # 1. Manage API Key (Use provided or fetch existing)
        if api_key:
            await save_tavily_credentials(thread_id, api_key)
        else:
            api_key = await get_tavily_credentials(thread_id)

        if not api_key:
            return "Error: Tavily API Key is missing. Please provide it in the tool configuration or conversation context."

        # 2. Execute Search
        search_tool = TavilySearch(
            tavily_api_key=api_key,
            max_results=max_results,
            search_depth=search_depth,
            include_answer=include_answer
        )
        
        results = await search_tool.ainvoke({"query": query})
        return str(results)

    except Exception as e:
        logger.error(f"Error in tavily_search: {str(e)}")
        return f"Error performing Tavily search: {str(e)}"

# --- Registry Integration ---

def register_tavily_tools():
    """Registers Tavily search tool with combined metadata."""
    
    registry.register(
        tavily_search,
        ToolMeta(
            name="tavily_search",
            display_name="Tavily: Web Search",
            description="Perform high-quality web searches for real-time information and facts.",
            category="Search",
            package="tavily",
            version="1.0.0",
            # Fields visible to the AI Agent
            ai_visible_fields=["query", "max_results", "search_depth", "include_answer"],
            # Configuration fields for the Human in the UI
            configs=[
                {"name": "api_key", "type": "password", "description": "Tavily API Key (Optional if already configured)", "required": False}
            ]
        )
    )

# Automatically register on import
register_tavily_tools()
