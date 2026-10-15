import logging
import json
from typing import List, Optional, Dict, Any
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from Apps.AgentMesh.Utilities.Tools.SqlTool.auth_utils import SqlProvider, save_sql_credentials
from Apps.AgentMesh.Utilities.Tools.SqlTool.utils import is_query_safe, attempt_sql_self_heal, is_write_query_safe
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry

logger = logging.getLogger(__name__)

# --- Tool Definitions ---

@tool
async def sql_configure_auth(
    host: str,
    user: str,
    password: str,
    database: str,
    schema_name: str,
    table_name: str,
    port: int = 5432,
    db_type: str = "postgres",
    config: RunnableConfig = None
) -> str:
    """
    Configure database credentials AND target table for the current thread.
    Supports both PostgreSQL and Oracle databases.
    """
    thread_id = SqlProvider.get_thread_id(config)
    
    await save_sql_credentials(thread_id, {
        "host": host,
        "user": user,
        "password": password,
        "database": database,
        "schema_name": schema_name,
        "table_name": table_name,
        "port": port,
        "db_type": db_type.lower()
    })
    
    return f"SQL Configured: Agent is now LOCKED to {db_type} table '{schema_name}.{table_name}' for thread '{thread_id}'."

@tool
async def sql_db_inspector(
    sample_limit: int = 5,
    config: RunnableConfig = None
) -> str:
    """
    Inspect the CURRENT configured table's schema and sample data.
    The table is predefined by the human in the configuration step.
    """
    try:
        thread_id = SqlProvider.get_thread_id(config)
        creds = await SqlProvider.get_sql_credentials(thread_id)
        db_type = creds.get('db_type', 'postgres').lower()
        schema_name = creds.get('schema_name')
        table_name = creds.get('table_name')

        conn = await SqlProvider.get_connection(config)
        cursor = conn.cursor()
        
        tables = [t.strip() for t in table_name.split(",") if t.strip()]
        if not tables:
            conn.close()
            return "No tables configured for database inspection."

        # If '*' or 'any' is configured, list available tables
        if len(tables) == 1 and (tables[0] == "*" or tables[0].lower() == "any"):
            if db_type == 'oracle':
                cursor.execute("""
                    SELECT table_name FROM all_tables 
                    WHERE owner = :owner 
                    ORDER BY table_name
                """, {"owner": schema_name.upper()})
            else:
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = %s 
                    ORDER BY table_name
                """, (schema_name,))
            all_tabs = [r[0] for r in cursor.fetchall()]
            conn.close()
            return f"Database type: {db_type}. Available tables under schema '{schema_name}':\n{json.dumps(all_tabs, indent=2)}"

        results = {}

        for table in tables:
            # 1. Fetch Columns
            if db_type == 'oracle':
                cursor.execute("""
                    SELECT column_name, data_type, nullable
                    FROM all_tab_columns
                    WHERE owner = :owner AND table_name = :table_name
                    ORDER BY column_id
                """, {"owner": schema_name.upper(), "table_name": table.upper()})
            else:
                cursor.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                """, (schema_name, table))
            columns = cursor.fetchall()
            
            if not columns:
                results[table] = {"error": "Table not found."}
                continue
                
            enriched = {}
            quoted_table = f'"{schema_name}"."{table}"'
            
            for col_name, data_type, nullable in columns:
                enriched[col_name] = {"type": data_type, "nullable": nullable}
                quoted_col = f'"{col_name}"'
                
                # Enrich categorical
                if data_type.lower() in ('text', 'varchar', 'character varying', 'char', 'string', 'varchar2'):
                    try:
                        if db_type == 'oracle':
                            cursor.execute(f'SELECT DISTINCT {quoted_col} FROM {quoted_table} WHERE ROWNUM <= 21')
                        else:
                            cursor.execute(f'SELECT DISTINCT {quoted_col} FROM {quoted_table} LIMIT 21')
                        distinct = cursor.fetchall()
                        if len(distinct) <= 20:
                            enriched[col_name]["distinct_values"] = [str(r[0]) for r in distinct]
                    except:
                        conn.rollback()
                
                # Enrich numeric ranges
                elif any(t in data_type.lower() for t in ('int', 'numeric', 'date', 'timestamp', 'float', 'double')):
                    try:
                        cursor.execute(f"SELECT MIN({quoted_col}), MAX({quoted_col}) FROM {quoted_table}")
                        min_val, max_val = cursor.fetchone()
                        enriched[col_name]["range"] = {"min": str(min_val), "max": str(max_val)}
                    except:
                        conn.rollback()

            # 2. Fetch Sample Rows
            try:
                if db_type == 'oracle':
                    cursor.execute(f"SELECT * FROM {quoted_table} WHERE ROWNUM <= {sample_limit}")
                else:
                    cursor.execute(f"SELECT * FROM {quoted_table} LIMIT {sample_limit}")
                rows = cursor.fetchall()
                col_names = [desc[0] for desc in cursor.description]
                samples = [dict(zip(col_names, row)) for row in rows]
            except Exception as e:
                samples = []
                conn.rollback()
                
            results[table] = {
                "schema": enriched,
                "sample_data": samples
            }
            
        conn.close()
        return f"Database context for schema '{schema_name}' tables:\n{json.dumps(results, indent=2, default=str)}"
        
    except Exception as e:
        return f"Error inspecting database: {str(e)}"

@tool
async def sql_query_executor(
    sql_query: str,
    config: RunnableConfig = None
) -> str:
    """
    Execute a SELECT query against the CURRENT configured table or system views. 
    The table is locked to the one provided by the human during configuration.
    """
    try:
        thread_id = SqlProvider.get_thread_id(config)
        creds = await SqlProvider.get_sql_credentials(thread_id)
        db_type = creds.get('db_type', 'postgres').lower()
        schema_name = creds.get('schema_name')
        table_name = creds.get('table_name')

        conn = await SqlProvider.get_connection(config)
        cursor = conn.cursor()
        
        # 1. Security Check
        safe, msg = is_query_safe(sql_query, table_name)
        if not safe:
            conn.close()
            return f"Security Block: {msg}"
            
        # 2. Execution with Self-Healing
        final_sql = sql_query
        execution_success = False
        last_error = ""
        
        # Get column names for healing logic
        if db_type == 'oracle':
            cursor.execute("""
                SELECT column_name FROM all_tab_columns 
                WHERE owner = :owner AND table_name = :table_name
            """, {"owner": schema_name.upper(), "table_name": table_name.upper()})
        else:
            cursor.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_schema = %s AND table_name = %s
            """, (schema_name, table_name))
        all_cols = [r[0] for r in cursor.fetchall()]

        for attempt in range(2):
            try:
                cursor.execute(final_sql)
                execution_success = True
                break
            except Exception as e:
                last_error = str(e)
                conn.rollback()
                
                if attempt == 0:
                    healed = attempt_sql_self_heal(final_sql, last_error, schema_name, table_name, all_cols)
                    if healed and healed != final_sql:
                        final_sql = healed
                    else:
                        break
                else:
                    break
        
        if not execution_success:
            conn.close()
            return f"SQL Execution Failed: {last_error}\nLast Query Attempted: {final_sql}"
            
        # 3. Process Results
        data = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description] if cursor.description else []
        conn.close()
        
        if not data:
            return "Query executed successfully. Result: 0 rows returned."
            
        results = [dict(zip(col_names, row)) for row in data[:100]] # Limit to 100 for tool output
        
        output = {
            "query_used": final_sql,
            "row_count": len(data),
            "data": results,
            "note": "Output limited to first 100 rows." if len(data) > 100 else ""
        }
        
        return json.dumps(output, indent=2, default=str)
        
    except Exception as e:
        return f"System Error in SQL Executor: {str(e)}"

@tool
async def sql_write_executor(
    sql_query: str,
    config: RunnableConfig = None
) -> str:
    """
    Execute a mutating write query (INSERT, UPDATE, DELETE) against the configured table.
    This tool should not be called dynamically by the AI agent and is designed for direct node/tool executions.
    """
    try:
        thread_id = SqlProvider.get_thread_id(config)
        creds = await SqlProvider.get_sql_credentials(thread_id)
        table_name = creds.get('table_name')

        conn = await SqlProvider.get_connection(config)
        cursor = conn.cursor()
        
        # 1. Security Check
        safe, msg = is_write_query_safe(sql_query, table_name)
        if not safe:
            conn.close()
            return f"Security Block: {msg}"
            
        # 2. Execution
        try:
            cursor.execute(sql_query)
            
            # Check if this is a query that returns rows (e.g. SELECT)
            is_select = cursor.description is not None
            
            if is_select:
                data = cursor.fetchall()
                col_names = [desc[0] for desc in cursor.description] if cursor.description else []
                results = [dict(zip(col_names, row)) for row in data[:100]]
                conn.close()
                return json.dumps({
                    "query_used": sql_query,
                    "status": "success",
                    "row_count": len(data),
                    "data": results,
                    "note": "Output limited to first 100 rows." if len(data) > 100 else ""
                }, indent=2, default=str)
            else:
                row_count = cursor.rowcount
                if hasattr(conn, "commit"):
                    conn.commit()
                conn.close()
                
                return json.dumps({
                    "query_used": sql_query,
                    "status": "success",
                    "rows_affected": row_count
                }, indent=2)
            
        except Exception as e:
            if hasattr(conn, "rollback"):
                conn.rollback()
            conn.close()
            return f"SQL Execution Failed: {str(e)}"
            
    except Exception as e:
        return f"System Error in SQL Write Executor: {str(e)}"

# --- Registry Integration ---

def register_sql_tools():
    """ Registers SQL tools with metadata for the Frontend UI. """
    
    registry.register(
        sql_configure_auth,
        ToolMeta(
            name="sql_configure_auth",
            display_name="SQL: Configure Credentials",
            description="Initialize Database credentials for this conversation thread (Postgres or Oracle).",
            category="Database",
            package="sql",
            version="1.0.0",
            allow_duplicate=False,
            ai_visible_fields=[], # Hide all from AI; Human must fill this in the UI
            configs=[
                {"name": "db_type", "type": "select", "options": ["postgres", "oracle"], "description": "Database Type", "required": True, "default": "postgres"},
                {"name": "host", "type": "string", "description": "DB Host", "required": True},
                {"name": "user", "type": "string", "description": "DB Username", "required": True},
                {"name": "password", "type": "password", "description": "DB Password", "required": True},
                {"name": "database", "type": "string", "description": "Database/Service Name", "required": True},
                {"name": "schema_name", "type": "string", "description": "Target Schema (e.g. public or admin)", "required": True, "default": "public"},
                {"name": "table_name", "type": "string", "description": "Target Table Name(s) (comma-separated for multiple, or * for monitoring)", "required": True},
                {"name": "port", "type": "number", "description": "DB Port (default: 5432 for Postgres, 1521 for Oracle)", "required": False}
            ]
        )
    )

    registry.register(
        sql_db_inspector,
        ToolMeta(
            name="sql_db_inspector",
            display_name="SQL: DB Inspector",
            description="Extract metadata from the human-configured table.",
            category="Database",
            package="sql",
            version="1.0.0",
            ai_visible_fields=["sample_limit"]
        )
    )

    registry.register(
        sql_query_executor,
        ToolMeta(
            name="sql_query_executor",
            display_name="SQL: Query Executor",
            description="Safely execute analytical SQL queries on the locked table.",
            category="Database",
            package="sql",
            version="1.0.0",
            ai_visible_fields=["sql_query"]
        )
    )

    registry.register(
        sql_write_executor,
        ToolMeta(
            name="sql_write_executor",
            display_name="SQL: Write Executor",
            description="Execute row mutations (INSERT, UPDATE, DELETE) directly.",
            category="Database",
            package="sql",
            version="1.0.0",
            ai_visible_fields=[], # Hide completely from LLM agent
            configs=[
                {"name": "sql_query", "type": "string", "description": "SQL Write Query (INSERT, UPDATE, DELETE)", "required": True}
            ]
        )
    )

# Automatically register on import
register_sql_tools()
