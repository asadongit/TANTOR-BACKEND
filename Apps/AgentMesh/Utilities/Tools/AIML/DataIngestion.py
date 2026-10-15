import requests
import logging
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry
from Apps.AgentMesh.Config import settings
from Apps.AgentMesh.Utilities.State.ToolStateManager import ToolStateManager

logger = logging.getLogger(__name__)

# --- Persistent Session Helpers (Wrappers for ToolStateManager) ---

async def get_aiml_session(thread_id: str) -> Optional[str]:
    """Helper to retrieve the session_id for a specific thread from DB."""
    state = await ToolStateManager.get_state(thread_id, "aiml_session")
    return state.get("session_id") if state else None

async def save_aiml_session(thread_id: str, session_id: str):
    """Helper to save the session_id for a specific thread to DB."""
    await ToolStateManager.save_state(thread_id, "aiml_session", {"session_id": session_id})

class DataIngestionInput(BaseModel):
    """Input schema for Data Ingestion API."""
    table_name: str = Field(..., description="The name of the table to ingest (e.g., 'loan_dataset_10k').")
    project_name: str = Field(..., description="The name of the project (e.g., 'prj_savemodel').")
    profiling_required: bool = Field(default=False, description="Whether profiling is required for the dataset.")
    user_name: str = Field(..., description="The user name performing the ingestion.")
    domain: List[str] = Field(default=["string"], description="List of domains associated with the data.")
    category: List[str] = Field(default=["string"], description="List of categories for the data.")
    sub_category: List[str] = Field(default=["string"], description="List of sub-categories for the data.")
    tag: List[str] = Field(default=["string"], description="List of tags for the data.")
    sub_tag: List[str] = Field(default=["string"], description="List of sub-tags for the data.")
    policy_id: List[int] = Field(default=[621], description="List of policy IDs associated with the ingestion.")

def ensure_list(data: Any, element_type: type = str) -> List[Any]:
    """Helper to ensure we have a list of the correct type, even if it's a comma-separated string."""
    if isinstance(data, list):
        return [element_type(x) for x in data]
    if isinstance(data, str):
        if not data or data.lower() == "string":
            return []
        # Support comma-separated strings from the UI configs
        return [element_type(x.strip()) for x in data.split(",") if x.strip()]
    return []

@tool("data_ingestion", args_schema=DataIngestionInput)
async def data_ingestion(
    table_name: str,
    project_name: str,
    user_name: str,
    config: RunnableConfig,
    profiling_required: bool = False,
    domain: Any = ["string"],
    category: Any = ["string"],
    sub_category: Any = ["string"],
    tag: Any = ["string"],
    sub_tag: Any = ["string"],
    policy_id: Any = [621]
) -> str:
    """
    Ingests data into the system and returns the dataset schema.
    This tool allows the agent to understand the structure and metadata of the ingested dataset.
    Automatically saves the session_id to the thread session for future tools.
    """
    # 1. Resolve Thread ID for session management
    thread_id = config.get("configurable", {}).get("thread_id", "default")

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    
    # Process inputs to ensure they are lists
    payload = {
        "table_name": table_name,
        "project_name": project_name,
        "profiling_required": profiling_required,
        "user_name": user_name,
        "domain": ensure_list(domain),
        "category": ensure_list(category),
        "sub_category": ensure_list(sub_category),
        "tag": ensure_list(tag),
        "sub_tag": ensure_list(sub_tag),
        "policy_id": ensure_list(policy_id, int)
    }

    try:
        logger.info(f"Triggering Data Ingestion for thread {thread_id}, table: {table_name}")
        response = requests.post(settings.DATA_INGESTION_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        # 2. Extract and SAVE the session_id for future tools in this thread (PERSISTENT)
        session_id = result.get("session_id")
        if session_id:
            await save_aiml_session(thread_id, session_id)
            logger.info(f"Successfully saved session {session_id} to DB for thread {thread_id}")

        return f"Data Ingestion Successful. Response: {result}"

    except requests.exceptions.RequestException as e:
        logger.error(f"Data Ingestion failed: {str(e)}")
        return f"Error: Data Ingestion failed. Reason: {str(e)}"
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        return f"Error: An unexpected error occurred: {str(e)}"

# Register with AgentMesh Tool Registry
registry.register(
    data_ingestion,
    ToolMeta(
        name="data_ingestion",
        display_name="Data Ingestion Tool",
        description="Ingest data from a specified table and project. Automatically shares session_id with future tools.",
        category="AI ML",
        package="AI ML",
        version="1.1.0",
        ai_visible_fields=["table_name", "project_name"],
        configs=[
            {"name": "table_name", "type": "string", "description": "Table Name", "required": True, "default": ""},
            {"name": "project_name", "type": "string", "description": "Project Name", "required": True, "default": ""},
            {"name": "profiling_required", "type": "boolean", "description": "Profiling Required", "required": False, "default": False},
            {"name": "user_name", "type": "string", "description": "User Name", "required": True, "default": ""},
            {"name": "domain", "type": "string", "description": "Domain (comma separated)", "required": False, "default": ""},
            {"name": "category", "type": "string", "description": "Category (comma separated)", "required": False, "default": ""},
            {"name": "sub_category", "type": "string", "description": "Sub Category (comma separated)", "required": False, "default": ""},
            {"name": "tag", "type": "string", "description": "Tag (comma separated)", "required": False, "default": ""},
            {"name": "sub_tag", "type": "string", "description": "Sub Tag (comma separated)", "required": False, "default": ""},
            {"name": "policy_id", "type": "string", "description": "Policy IDs (comma separated)", "required": False, "default": ""}
        ]
    )
)
