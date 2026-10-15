"""
Unified DB Node Builder
Generic node for performing operations on SQL (Postgres, SQLite, MySQL) and MongoDB databases.

Supports connection pooling, thread-safe asynchronous file locking for SQLite,
and automatic parameterized query generation to prevent SQL injection.
"""

import json
import logging
import asyncio
import re
from typing import Callable, Any, Dict, List, Optional, Tuple
from datetime import datetime
from collections import OrderedDict

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from .utils import get_nested_val, interpolate, get_production_data

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Thread-safe global manager for database connection pools and locks with LRU eviction."""
    MAX_CACHE_SIZE = 10
    _pg_pools: OrderedDict = OrderedDict()
    _mongo_clients: OrderedDict = OrderedDict()
    _sqlite_locks: OrderedDict = OrderedDict()
    _global_lock = asyncio.Lock()

    @classmethod
    async def get_pg_pool(cls, dsn: str) -> Any:
        import asyncpg
        async with cls._global_lock:
            if dsn in cls._pg_pools:
                cls._pg_pools.move_to_end(dsn)
            else:
                if len(cls._pg_pools) >= cls.MAX_CACHE_SIZE:
                    oldest_dsn, pool = cls._pg_pools.popitem(last=False)
                    logger.info(f"Evicting PostgreSQL pool for: {oldest_dsn[:30]}...")
                    task = asyncio.create_task(pool.close())
                    if not hasattr(cls, "_bg_tasks"):
                        cls._bg_tasks = set()
                    cls._bg_tasks.add(task)
                    task.add_done_callback(cls._bg_tasks.discard)
                
                cls._pg_pools[dsn] = await asyncpg.create_pool(
                    dsn,
                    min_size=1,
                    max_size=10,
                    command_timeout=15.0
                )
            return cls._pg_pools[dsn]

    @classmethod
    async def get_mongo_client(cls, dsn: str) -> Any:
        import pymongo
        async with cls._global_lock:
            if dsn in cls._mongo_clients:
                cls._mongo_clients.move_to_end(dsn)
            else:
                if len(cls._mongo_clients) >= cls.MAX_CACHE_SIZE:
                    oldest_dsn, client = cls._mongo_clients.popitem(last=False)
                    logger.info(f"Evicting MongoDB client for: {oldest_dsn[:30]}...")
                    await asyncio.to_thread(client.close)
                
                cls._mongo_clients[dsn] = pymongo.MongoClient(
                    dsn,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=5000,
                    maxPoolSize=20,
                    waitQueueTimeoutMS=10000
                )
            return cls._mongo_clients[dsn]

    @classmethod
    async def get_sqlite_lock(cls, path: str) -> asyncio.Lock:
        async with cls._global_lock:
            if path in cls._sqlite_locks:
                cls._sqlite_locks.move_to_end(path)
            else:
                if len(cls._sqlite_locks) >= cls.MAX_CACHE_SIZE:
                    cls._sqlite_locks.popitem(last=False)
                cls._sqlite_locks[path] = asyncio.Lock()
            return cls._sqlite_locks[path]


def _resolve_template(template: Any, context: Dict[str, Any]) -> Any:
    """
    Recursively resolve {{context.xxx}} placeholders in a document template.
    """
    if isinstance(template, str):
        if template.strip() == "{{__NOW__}}":
            return datetime.now().isoformat()
        
        single_var = re.fullmatch(r"\{\{([\w. ]+)\}\}", template.strip())
        if single_var:
            var_name = single_var.group(1)
            val = get_nested_val(context, var_name)
            if val is not None:
                return val
            val = _search_in_items(context, var_name)
            if val is not None:
                return val
        
        result = interpolate(template, context)
        if "{{" in result:
            items_ctx = _build_items_context(context)
            if items_ctx:
                result = interpolate(result, items_ctx)
        return result
    
    elif isinstance(template, dict):
        return {k: _resolve_template(v, context) for k, v in template.items()}
    
    elif isinstance(template, list):
        return [_resolve_template(item, context) for item in template]
    
    return template


def _search_in_items(context: Dict[str, Any], key: str) -> Any:
    """Search for a key inside items[] or aggregated_results[] dicts."""
    key_lower = key.lower().replace(" ", "_")
    for list_key in ("items", "aggregated_results"):
        items = context.get(list_key)
        if isinstance(items, list):
            for item in reversed(items):
                if isinstance(item, dict):
                    if key in item:
                        return item[key]
                    for k, v in item.items():
                        if isinstance(k, str) and k.lower().replace(" ", "_") == key_lower:
                            return v
    return None


def _build_items_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten items[] and aggregated_results[] dicts into a lookup dict."""
    merged = {}
    for list_key in ("items", "aggregated_results"):
        items = context.get(list_key)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    merged.update(item)
    return merged


def parameterize_query(query_template: str, context: Dict[str, Any], engine: str) -> Tuple[str, list]:
    """
    Extracts {{...}} placeholders from SQL query template, resolves their values,
    and returns a parameterized SQL query and parameters list to prevent SQL injection.
    """
    params = []
    
    def replacer(match):
        end_idx = match.end()
        trailing = query_template[end_idx:end_idx+10].lower()
        has_text_cast = trailing.startswith("::text") or trailing.startswith("::varchar")

        var_name = match.group(1).strip()
        val = get_nested_val(context, var_name)
        if val is None:
            val = _search_in_items(context, var_name)

        # If placeholder is followed by ::text or ::varchar and val is primitive (float/int/bool), convert to str
        if val is not None and has_text_cast and isinstance(val, (int, float, bool)):
            val = str(val)
            
        # Automatically serialize dicts, lists, and custom objects (like SmartInput) for SQL parameters
        if val is not None and not isinstance(val, (str, int, float, bool)):
            try:
                val = json.dumps(val, default=str)
            except Exception:
                val = str(val)

        params.append(val)
        
        param_idx = len(params)
        if engine in ("postgresql", "postgres"):
            return f"${param_idx}"
        elif engine == "oracle":
            return f":{param_idx}"
        else:
            return "?"

    parameterized_sql = re.sub(r"\{\{([\w. ]+)\}\}", replacer, query_template)
    return parameterized_sql, params


async def execute_db_query_core(
    db_engine: str,
    connection_string: str,
    database: str,
    collection_name: str,
    operation: str,
    document: Any,
    query: Any,
    limit: int,
    is_test: bool = False
) -> Tuple[str, Any]:
    """
    Core executor for MongoDB, PostgreSQL, and SQLite operations.
    Can be used by both the workflow execution node and validation/testing endpoints.
    """
    import asyncio
    import re
    db_engine = db_engine.lower()

    # Enforce safety limits to avoid memory exhaustion
    if is_test:
        limit = min(limit or 200, 200)
    else:
        limit = min(limit or 5000, 5000)

    # --- MongoDB Execution ---
    if db_engine == "mongodb":
        if not connection_string or not database or not collection_name:
            raise ValueError("MongoDB requires connection string, database name, and collection name.")

        client = await ConnectionManager.get_mongo_client(connection_string)
        db = client[database]
        coll = db[collection_name]

        if is_test:
            import json
            def parse_json_val(val):
                if isinstance(val, str):
                    try:
                        return json.loads(val)
                    except Exception:
                        return val
                return val
            document = parse_json_val(document)
            query = parse_json_val(query) or {}

        if operation == "insert_one":
            if not isinstance(document, dict):
                raise ValueError("insert_one requires a dictionary document.")
            insert_doc = {k: v for k, v in document.items() if v is not None or k != "_id"}
            res = await asyncio.to_thread(coll.insert_one, insert_doc)
            result_data = {"inserted_id": str(res.inserted_id), "acknowledged": res.acknowledged}
            summary = f"MongoDB insert_one successful: _id={res.inserted_id}"
            return summary, result_data

        elif operation == "insert_many":
            if not isinstance(document, list):
                raise ValueError("insert_many requires a list of documents.")
            res = await asyncio.to_thread(coll.insert_many, document)
            result_data = {"inserted_ids": [str(oid) for oid in res.inserted_ids], "count": len(res.inserted_ids)}
            summary = f"MongoDB insert_many successful: Inserted {len(res.inserted_ids)} documents"
            return summary, result_data

        elif operation == "find_one":
            doc = await asyncio.to_thread(coll.find_one, query)
            if doc:
                doc["_id"] = str(doc["_id"])
            result_data = doc
            summary = f"MongoDB find_one: {'Found' if doc else 'Not found'}"
            return summary, result_data

        elif operation == "find_many":
            limit_val = int(limit)
            cursor = coll.find(query)
            if is_test:
                cursor = cursor.limit(limit_val)
            else:
                cursor = cursor.limit(limit_val).batch_size(min(100, limit_val)).max_time_ms(10000)
            docs = await asyncio.to_thread(list, cursor)
            for doc in docs:
                doc["_id"] = str(doc["_id"])
            result_data = docs
            summary = f"MongoDB find_many: Found {len(docs)} documents"
            return summary, result_data

        elif operation == "update_one":
            if not document:
                raise ValueError("update_one requires an update document.")
            update_doc = document if any(k.startswith("$") for k in document.keys()) else {"$set": document}
            res = await asyncio.to_thread(coll.update_one, query, update_doc)
            result_data = {
                "matched_count": res.matched_count,
                "modified_count": res.modified_count,
                "acknowledged": res.acknowledged
            }
            summary = f"MongoDB update_one: Matched {res.matched_count}, Modified {res.modified_count}"
            return summary, result_data

        elif operation == "delete_one":
            res = await asyncio.to_thread(coll.delete_one, query)
            result_data = {"deleted_count": res.deleted_count, "acknowledged": res.acknowledged}
            summary = f"MongoDB delete_one: Deleted {res.deleted_count}"
            return summary, result_data

        else:
            raise ValueError(f"Unknown MongoDB operation: {operation}")

    # --- PostgreSQL Execution (asyncpg) ---
    elif db_engine in ("postgresql", "postgres"):
        if not connection_string:
            raise ValueError("PostgreSQL requires a connection string.")

        # asyncpg requires postgresql:// or postgres:// scheme (no SQLAlchemy +asyncpg driver suffix)
        if connection_string.startswith("postgresql+asyncpg://"):
            connection_string = connection_string.replace("postgresql+asyncpg://", "postgresql://", 1)

        pool = await ConnectionManager.get_pg_pool(connection_string)
        
        if is_test:
            sql, params = parameterize_query(query, {}, "postgresql")
        else:
            sql, params = query if isinstance(query, tuple) else (query, [])

        async with pool.acquire() as conn:
            if operation in ("query", "select"):
                records = await conn.fetch(sql, *params)
                result_data = [dict(r) for r in records]
                result_data = result_data[:limit]
                summary = f"Postgres query: Returned {len(result_data)} rows"
                return summary, result_data
            else:
                status = await conn.execute(sql, *params)
                result_data = {"status": status}
                summary = f"Postgres execute: {status}"
                return summary, result_data

    # --- SQLite Execution (aiosqlite) ---
    elif db_engine == "sqlite":
        if not connection_string:
            raise ValueError("SQLite requires a db file path (connection string).")

        import aiosqlite
        db_lock = await ConnectionManager.get_sqlite_lock(connection_string)
        
        if is_test:
            sql, params = parameterize_query(query, {}, "sqlite")
        else:
            sql, params = query if isinstance(query, tuple) else (query, [])

        async with db_lock:
            async with aiosqlite.connect(connection_string) as conn:
                conn.row_factory = aiosqlite.Row
                if operation in ("query", "select"):
                    async with conn.execute(sql, params) as cursor:
                        rows = await cursor.fetchall()
                        result_data = [dict(r) for r in rows]
                        result_data = result_data[:limit]
                        summary = f"SQLite query: Returned {len(result_data)} rows"
                        return summary, result_data
                else:
                    async with conn.execute(sql, params) as cursor:
                        result_data = {"rows_affected": cursor.rowcount}
                        summary = f"SQLite execute: Affected {cursor.rowcount} rows"
                    await conn.commit()
                    return summary, result_data

    else:
        raise ValueError(f"Unsupported database engine: {db_engine}")


def build_db_node(node_config: NodeConfig) -> Callable:
    """
    Builds a database execution node for PostgreSQL, SQLite, and MongoDB.
    """
    import os

    db_engine = getattr(node_config, "db_engine", "mongodb") or "mongodb"
    db_engine = db_engine.lower()
    
    connection_string = getattr(node_config, "db_connection_string", None)
    database = getattr(node_config, "db_database", None)
    collection_name = getattr(node_config, "db_collection", None) or ""
    operation = getattr(node_config, "db_operation", "query") or "query"
    doc_template = getattr(node_config, "db_document", None)
    query_template = getattr(node_config, "db_query", None)
    output_key = getattr(node_config, "db_output_key", "db_result") or "db_result"
    limit = getattr(node_config, "db_limit", 100) or 100
    node_id = node_config.id

    # Fallback to env variables if not set
    if db_engine == "mongodb":
        if not connection_string:
            connection_string = os.environ.get("MONGODB_URI", "")
        if not database:
            database = os.environ.get("MONGODB_DATABASE", "")
    elif db_engine in ("postgresql", "postgres"):
        if not connection_string:
            connection_string = os.environ.get("DATABASE_URL", "")

    async def execute_db(state: AgentState):
        ctx = get_production_data(state)

        # Resolve dynamic configurations using state context
        resolved_conn = interpolate(connection_string, ctx) if connection_string else ""
        resolved_db = interpolate(database, ctx) if database else ""
        resolved_coll = interpolate(collection_name, ctx) if collection_name else ""

        if db_engine == "mongodb":
            if not resolved_conn or not resolved_db or not resolved_coll:
                error_msg = f"MongoDB Node {node_id}: Missing connection, database, or collection configurations."
                logger.error(error_msg)
                return {"messages": [AIMessage(content=f"[ERROR] {error_msg}")], "active_agent": node_id}
        elif db_engine in ("postgresql", "postgres", "sqlite"):
            if not resolved_conn:
                error_msg = f"{db_engine.capitalize()} Node {node_id}: Missing connection configuration."
                logger.error(error_msg)
                return {"messages": [AIMessage(content=f"[ERROR] {error_msg}")], "active_agent": node_id}

        # Resolve templates
        document = _resolve_template(doc_template, ctx) if doc_template else None
        
        if db_engine in ("postgresql", "postgres", "sqlite"):
            sql, params = parameterize_query(query_template, ctx, db_engine)
            query_input = (sql, params)
        else:
            query_input = _resolve_template(query_template, ctx) if query_template else {}

        try:
            summary, result_data = await execute_db_query_core(
                db_engine=db_engine,
                connection_string=resolved_conn,
                database=resolved_db,
                collection_name=resolved_coll,
                operation=operation,
                document=document,
                query=query_input,
                limit=limit,
                is_test=False
            )
        except Exception as e:
            error_msg = f"{db_engine.capitalize()} Node {node_id} failed: {str(e)}"
            logger.exception(error_msg)
            return {"messages": [AIMessage(content=f"[ERROR] {error_msg}")], "active_agent": node_id}

        return {
            "messages": [AIMessage(content=summary, name=f"{db_engine.capitalize()}_{node_id}")],
            "active_agent": node_id,
            "context": {**state.get("context", {}), output_key: result_data, f"db_result_{node_id}": result_data}
        }

    return RunnableLambda(execute_db).with_retry(stop_after_attempt=3)
