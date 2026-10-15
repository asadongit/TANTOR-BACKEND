import logging
from typing import Optional, Any, Dict
from sqlalchemy import select, delete
from Database.Database import AsyncSessionLocal
from Database.Models import ToolStateModel
from Apps.AgentMesh.Utilities.Security import protect_sensitive_data, reveal_sensitive_data

def process_state_data(data: Any, mode: str = "encrypt") -> Any:
    """Recursively encrypts or decrypts all strings within a nested data structure."""
    if mode == "encrypt":
        return protect_sensitive_data(data, force=True)
    if mode == "decrypt":
        return reveal_sensitive_data(data)
    raise ValueError("mode must be either encrypt or decrypt")

logger = logging.getLogger(__name__)

class ToolStateManager:
    """
    Utility to manage persistent tool state (session IDs, credentials) in PostgreSQL.
    This replaces in-memory global dictionaries to ensure scalability and durability.
    """
    
    @staticmethod
    async def save_state(thread_id: str, tool_key: str, state_data: Any):
        """Persistent save or update of tool state associated with a thread."""
        async with AsyncSessionLocal() as session:
            try:
                # 1. Check if state already exists for this thread and tool
                query = select(ToolStateModel).where(
                    ToolStateModel.thread_id == thread_id,
                    ToolStateModel.tool_key == tool_key
                )
                result = await session.execute(query)
                tool_state = result.scalar_one_or_none()

                # Encrypt data before saving
                encrypted_data = process_state_data(state_data, mode="encrypt")

                if tool_state:
                    # Update existing state
                    tool_state.state = encrypted_data
                    logger.info(f"Updated persistent state for {tool_key} in thread {thread_id}")
                else:
                    # Create new state entry
                    new_state = ToolStateModel(
                        thread_id=thread_id,
                        tool_key=tool_key,
                        state=encrypted_data
                    )
                    session.add(new_state)
                    logger.info(f"Created new persistent state for {tool_key} in thread {thread_id}")

                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Error saving tool state to DB: {str(e)}")
                raise

    @staticmethod
    async def get_state(thread_id: str, tool_key: str) -> Optional[Any]:
        """Fetch tool state from persistent database."""
        async with AsyncSessionLocal() as session:
            try:
                query = select(ToolStateModel).where(
                    ToolStateModel.thread_id == thread_id,
                    ToolStateModel.tool_key == tool_key
                )
                result = await session.execute(query)
                tool_state = result.scalar_one_or_none()
                
                if tool_state:
                    # Decrypt before returning
                    return process_state_data(tool_state.state, mode="decrypt")
                return None
            except Exception as e:
                logger.error(f"Error fetching tool state from DB: {str(e)}")
                return None

    @staticmethod
    async def delete_state(thread_id: str, tool_key: str):
        """Clear state for a specific tool and thread."""
        async with AsyncSessionLocal() as session:
            try:
                stmt = delete(ToolStateModel).where(
                    ToolStateModel.thread_id == thread_id,
                    ToolStateModel.tool_key == tool_key
                )
                await session.execute(stmt)
                await session.commit()
                logger.info(f"Deleted state for {tool_key} in thread {thread_id}")
            except Exception as e:
                await session.rollback()
                logger.error(f"Error deleting tool state from DB: {str(e)}")
