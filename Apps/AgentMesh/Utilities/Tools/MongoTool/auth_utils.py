import os
import logging
from typing import Optional, Dict, Any
from langchain_core.runnables import RunnableConfig
from pymongo import MongoClient
from Apps.AgentMesh.Utilities.State.ToolStateManager import ToolStateManager

logger = logging.getLogger(__name__)

# --- Persistent Credential Helpers ---

async def save_mongo_credentials(thread_id: str, creds: Dict[str, Any]):
    """Saves or updates MongoDB credentials for a specific thread in DB."""
    logger.info(f"Saving MongoDB credentials to DB for thread: {thread_id}")
    await ToolStateManager.save_state(thread_id, "mongo_creds", creds)

async def get_mongo_credentials(thread_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves MongoDB credentials for a specific thread from DB."""
    return await ToolStateManager.get_state(thread_id, "mongo_creds")

# --- MongoDB Provider Logic ---

class MongoProvider:
    _clients: Dict[str, MongoClient] = {}

    @staticmethod
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

    @classmethod
    async def get_mongo_credentials(cls, thread_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves MongoDB credentials for a specific thread from DB."""
        return await ToolStateManager.get_state(thread_id, "mongo_creds")

    @classmethod
    async def get_client(cls, config: RunnableConfig) -> MongoClient:
        """Establishes or retrieves a cached MongoClient connection for the thread."""
        thread_id = cls.get_thread_id(config)
        creds = await cls.get_mongo_credentials(thread_id)

        if not creds:
            raise ValueError(f"MongoDB credentials NOT configured for thread {thread_id}. Run 'mongo_configure_auth' first.")

        host = creds.get('host', 'localhost')
        port = creds.get('port', 27017)
        user = creds.get('user')
        password = creds.get('password')
        auth_source = creds.get('auth_source', 'admin')

        # Use connection caching key
        cache_key = f"{host}||{port}||{user}||{auth_source}"
        if cache_key in cls._clients:
            try:
                # Ping to check if connection is still alive
                cls._clients[cache_key].admin.command('ping')
                return cls._clients[cache_key]
            except Exception:
                logger.warning(f"Cached MongoDB client for {cache_key} was disconnected. Reconnecting...")
                try:
                    cls._clients[cache_key].close()
                except Exception:
                    pass
                cls._clients.pop(cache_key, None)

        try:
            if user and password:
                client = MongoClient(
                    host=host,
                    port=int(port),
                    username=user,
                    password=password,
                    authSource=auth_source,
                    serverSelectionTimeoutMS=5000,
                )
            else:
                client = MongoClient(
                    host=host,
                    port=int(port),
                    serverSelectionTimeoutMS=5000,
                )
            
            # Verify connectivity
            client.admin.command('ping')
            cls._clients[cache_key] = client
            return client
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB ({host}:{port}): {str(e)}")
            raise ConnectionError(f"MongoDB connection failed: {str(e)}")
