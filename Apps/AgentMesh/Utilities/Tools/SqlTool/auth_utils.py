import os
import logging
import psycopg
from urllib.parse import quote_plus
from typing import Optional, Dict, Any
from langchain_core.runnables import RunnableConfig
from Apps.AgentMesh.Utilities.State.ToolStateManager import ToolStateManager

logger = logging.getLogger(__name__)

# --- Persistent Credential Helpers ---

async def save_sql_credentials(thread_id: str, creds: Dict[str, Any]):
    """Saves or updates SQL credentials for a specific thread in DB."""
    logger.info(f"Saving SQL credentials to DB for thread: {thread_id}")
    await ToolStateManager.save_state(thread_id, "sql_creds", creds)

async def get_sql_credentials(thread_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves SQL credentials for a specific thread from DB."""
    return await ToolStateManager.get_state(thread_id, "sql_creds")

# --- SQL Provider Logic ---

class SqlProvider:
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
    async def get_sql_credentials(cls, thread_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves SQL credentials for a specific thread from DB."""
        return await ToolStateManager.get_state(thread_id, "sql_creds")

    @classmethod
    async def get_connection(cls, config: RunnableConfig):
        """Establishes connection to the configured database for the current thread."""
        thread_id = cls.get_thread_id(config)
        creds = await cls.get_sql_credentials(thread_id)

        if not creds:
            raise ValueError(f"SQL credentials NOT configured for thread {thread_id}. Run 'sql_configure_auth' first.")

        db_type = creds.get('db_type', 'postgres').lower()
        host = creds.get('host')
        user = creds.get('user')
        password = creds.get('password')
        database = creds.get('database')
        port = creds.get('port')

        from sqlalchemy import create_engine

        try:
            if db_type == 'oracle':
                import oracledb
                # Enable thin mode explicitly (default in oracledb 2.x, but good to ensure)
                oracledb.defaults.config_dir = None
                
                # Format: oracle+oracledb://user:password@host:port/service_name
                port_val = port or 1521
                # Check if database represents a service name or SID. 
                # Standard service name connection url:
                url = f"oracle+oracledb://{quote_plus(str(user))}:{quote_plus(str(password))}@{host}:{port_val}/{quote_plus(str(database))}"
                engine = create_engine(url)
                conn = engine.raw_connection()
                return conn
            else:
                port_val = port or 5432
                url = f"postgresql+psycopg://{quote_plus(str(user))}:{quote_plus(str(password))}@{host}:{port_val}/{quote_plus(str(database))}"
                engine = create_engine(url)
                conn = engine.raw_connection()
                # Ensure autocommit or transaction management works
                conn.autocommit = True
                return conn
        except Exception as e:
            logger.error(f"Failed to connect to database ({db_type}): {str(e)}")
            raise ConnectionError(f"Database connection failed: {str(e)}")

