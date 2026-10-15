import logging
import json
from typing import List, Optional, Dict, Any
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from bson import json_util

from Apps.AgentMesh.Utilities.Tools.MongoTool.auth_utils import MongoProvider, save_mongo_credentials
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry

logger = logging.getLogger(__name__)

# --- Helper functions for Schema Inferences & Formatting ---

def infer_schema_from_documents(docs: List[Dict[str, Any]]) -> Dict[str, str]:
    """Helper to infer schema types from a list of MongoDB documents."""
    schema = {}
    for doc in docs:
        for key, value in doc.items():
            if key not in schema:
                schema[key] = type(value).__name__
            elif schema[key] == "NoneType" and value is not None:
                schema[key] = type(value).__name__
    return schema

# --- Tool Definitions ---

@tool
async def mongo_configure_auth(
    host: str,
    database: str,
    collection_name: str,
    port: int = 27017,
    user: Optional[str] = None,
    password: Optional[str] = None,
    auth_source: str = "admin",
    config: RunnableConfig = None
) -> str:
    """
    Configure database credentials AND target collection for the current thread.
    Supports secure isolation per session.
    """
    thread_id = MongoProvider.get_thread_id(config)
    
    await save_mongo_credentials(thread_id, {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "database": database,
        "collection_name": collection_name,
        "auth_source": auth_source
    })
    
    return f"MongoDB Configured: Agent is now locked to database '{database}', collection '{collection_name}' for thread '{thread_id}'."

@tool
async def mongo_db_inspector(
    sample_limit: int = 5,
    config: RunnableConfig = None
) -> str:
    """
    Inspect the CURRENT configured collection's schema and sample documents.
    If '*' or 'any' was configured as the collection name, it will list all available collections.
    """
    try:
        thread_id = MongoProvider.get_thread_id(config)
        creds = await MongoProvider.get_mongo_credentials(thread_id)
        database_name = creds.get('database')
        target_collection = creds.get('collection_name')

        client = await MongoProvider.get_client(config)
        db = client[database_name]

        # If wildcard '*', return a list of all collections in the db
        if target_collection in ("*", "any"):
            col_list = db.list_collection_names()
            return f"Database: {database_name}. Available collections:\n{json.dumps(col_list, indent=2)}"

        cols = [c.strip() for c in target_collection.split(",") if c.strip()]
        if not cols:
            return "No collections configured for inspection."

        results = {}
        for col_name in cols:
            if col_name not in db.list_collection_names():
                results[col_name] = {"error": "Collection not found in database."}
                continue

            collection = db[col_name]
            
            # Fetch sample documents
            sample_docs = list(collection.find().limit(sample_limit))
            
            # Infer schema types
            schema = infer_schema_from_documents(sample_docs)
            
            # Serialize output using bson json_util to correctly format ObjectId, DateTime, etc.
            serialized_samples = json.loads(json_util.dumps(sample_docs))
            
            results[col_name] = {
                "inferred_schema": schema,
                "sample_data": serialized_samples
            }

        return f"Database context for database '{database_name}':\n{json.dumps(results, indent=2)}"

    except Exception as e:
        return f"Error inspecting MongoDB: {str(e)}"

@tool
async def mongo_query_executor(
    operation: str,
    collection_name: Optional[str] = None,
    filter_query: Optional[str] = None,
    projection: Optional[str] = None,
    sort: Optional[str] = None,
    limit: int = 100,
    pipeline: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Execute a read-only MongoDB query (find, count_documents, aggregate) against the configured collection(s).
    Takes JSON strings for queries and parameters.
    """
    try:
        thread_id = MongoProvider.get_thread_id(config)
        creds = await MongoProvider.get_mongo_credentials(thread_id)
        database_name = creds.get('database')
        target_collection = creds.get('collection_name')

        # 1. Security & Collection Selection Checks
        op_lower = operation.lower().strip()
        if op_lower not in ("find", "count_documents", "aggregate"):
            return f"Security Block: Operation '{operation}' is not supported or prohibited. Only read-only operations ('find', 'count_documents', 'aggregate') are permitted."

        allowed_cols = [c.strip() for c in target_collection.split(",") if c.strip()]
        is_wildcard = len(allowed_cols) == 1 and allowed_cols[0] in ("*", "any")

        col_to_use = None
        if collection_name:
            col_cleaned = collection_name.strip()
            if not is_wildcard and col_cleaned not in allowed_cols:
                return f"Security Block: Collection '{col_cleaned}' is not authorized. Allowed collection list: {target_collection}"
            col_to_use = col_cleaned
        else:
            if not is_wildcard and len(allowed_cols) == 1:
                col_to_use = allowed_cols[0]
            else:
                return f"Error: Multiple collections are configured ({target_collection}). You must specify the target 'collection_name' to query."

        client = await MongoProvider.get_client(config)
        db = client[database_name]
        collection = db[col_to_use]

        # Deserialize JSON filters/projections using bson's json_util to support native MongoDB types
        def parse_json_param(param_str: Optional[str]) -> Optional[Dict[str, Any]]:
            if not param_str or not param_str.strip():
                return None
            try:
                return json_util.loads(param_str)
            except Exception as pe:
                raise ValueError(f"Failed to parse query parameter JSON: {str(pe)}")

        # 2. Execution logic
        if op_lower == "count_documents":
            parsed_filter = parse_json_param(filter_query) or {}
            count = collection.count_documents(parsed_filter)
            return json.dumps({"count": count})

        elif op_lower == "find":
            parsed_filter = parse_json_param(filter_query) or {}
            parsed_projection = parse_json_param(projection)
            parsed_sort = parse_json_param(sort)

            cursor = collection.find(parsed_filter, parsed_projection)
            if parsed_sort:
                # Support sorting format: list of tuples/lists or dictionary
                if isinstance(parsed_sort, dict):
                    sort_list = list(parsed_sort.items())
                else:
                    sort_list = parsed_sort
                cursor = cursor.sort(sort_list)

            results = list(cursor.limit(limit))
            serialized = json.loads(json_util.dumps(results))
            return json.dumps({
                "row_count": len(results),
                "data": serialized
            }, indent=2)

        elif op_lower == "aggregate":
            if not pipeline or not pipeline.strip():
                return "Error: Aggregate operation requires a 'pipeline' parameter."
            
            parsed_pipeline = json_util.loads(pipeline)
            if not isinstance(parsed_pipeline, list):
                return "Error: Pipeline must be a valid JSON array."

            # Mutation check for aggregate stages
            forbidden_stages = ("$out", "$merge")
            for stage in parsed_pipeline:
                if not isinstance(stage, dict):
                    continue
                for stage_name in stage.keys():
                    if stage_name in forbidden_stages:
                        return f"Security Block: Aggregation stage '{stage_name}' is mutating and strictly prohibited."

            cursor = collection.aggregate(parsed_pipeline)
            results = list(cursor)
            # Limit count manually to prevent huge context output
            results_limited = results[:limit]
            serialized = json.loads(json_util.dumps(results_limited))

            return json.dumps({
                "row_count": len(results),
                "data": serialized,
                "note": "Output capped to limit." if len(results) > limit else ""
            }, indent=2)

    except Exception as e:
        return f"Error executing MongoDB query: {str(e)}"

@tool
async def mongo_write_executor(
    operation: str,
    collection_name: Optional[str] = None,
    document: Optional[str] = None,
    filter_query: Optional[str] = None,
    update_query: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Execute a writing or mutating MongoDB operation (insert, update, delete) against the configured collection(s).
    This tool should not be called dynamically by the AI agent and is designed for direct node/tool executions.
    """
    try:
        thread_id = MongoProvider.get_thread_id(config)
        creds = await MongoProvider.get_mongo_credentials(thread_id)
        database_name = creds.get('database')
        target_collection = creds.get('collection_name')

        op_lower = operation.lower().strip()
        allowed_ops = ("insert_one", "insert_many", "update_one", "update_many", "delete_one", "delete_many")
        if op_lower not in allowed_ops:
            return f"Error: Write operation '{operation}' is not supported. Supported operations: {allowed_ops}"

        allowed_cols = [c.strip() for c in target_collection.split(",") if c.strip()]
        is_wildcard = len(allowed_cols) == 1 and allowed_cols[0] in ("*", "any")

        col_to_use = None
        if collection_name:
            col_cleaned = collection_name.strip()
            if not is_wildcard and col_cleaned not in allowed_cols:
                return f"Security Block: Collection '{col_cleaned}' is not authorized. Allowed collection list: {target_collection}"
            col_to_use = col_cleaned
        else:
            if not is_wildcard and len(allowed_cols) == 1:
                col_to_use = allowed_cols[0]
            else:
                return f"Error: Multiple collections are configured ({target_collection}). You must specify the target 'collection_name' to mutate."

        client = await MongoProvider.get_client(config)
        db = client[database_name]
        collection = db[col_to_use]

        # Helper to parse parameter inputs safely
        def parse_json_param(param_str: Optional[str], param_name: str) -> Optional[Any]:
            if not param_str or not param_str.strip():
                return None
            try:
                return json_util.loads(param_str)
            except Exception as pe:
                raise ValueError(f"Failed to parse '{param_name}' JSON: {str(pe)}")

        parsed_doc = parse_json_param(document, "document")
        parsed_filter = parse_json_param(filter_query, "filter_query")
        parsed_update = parse_json_param(update_query, "update_query")

        # 2. Execution Logic
        result_data = {}
        if op_lower == "insert_one":
            if not parsed_doc:
                return "Error: insert_one requires a valid 'document' input."
            res = collection.insert_one(parsed_doc)
            result_data = {"acknowledged": res.acknowledged, "inserted_id": str(res.inserted_id)}

        elif op_lower == "insert_many":
            if not parsed_doc:
                return "Error: insert_many requires a valid list of documents in 'document' parameter."
            if not isinstance(parsed_doc, list):
                return "Error: insert_many 'document' parameter must be a JSON array."
            res = collection.insert_many(parsed_doc)
            result_data = {"acknowledged": res.acknowledged, "inserted_ids": [str(id) for id in res.inserted_ids]}

        elif op_lower == "update_one":
            if not parsed_filter or not parsed_update:
                return "Error: update_one requires both 'filter_query' and 'update_query' parameters."
            res = collection.update_one(parsed_filter, parsed_update)
            result_data = {
                "acknowledged": res.acknowledged,
                "matched_count": res.matched_count,
                "modified_count": res.modified_count,
                "upserted_id": str(res.upserted_id) if res.upserted_id else None
            }

        elif op_lower == "update_many":
            if not parsed_filter or not parsed_update:
                return "Error: update_many requires both 'filter_query' and 'update_query' parameters."
            res = collection.update_many(parsed_filter, parsed_update)
            result_data = {
                "acknowledged": res.acknowledged,
                "matched_count": res.matched_count,
                "modified_count": res.modified_count,
                "upserted_id": str(res.upserted_id) if res.upserted_id else None
            }

        elif op_lower == "delete_one":
            if not parsed_filter:
                return "Error: delete_one requires a 'filter_query' parameter."
            res = collection.delete_one(parsed_filter)
            result_data = {"acknowledged": res.acknowledged, "deleted_count": res.deleted_count}

        elif op_lower == "delete_many":
            if not parsed_filter:
                return "Error: delete_many requires a 'filter_query' parameter."
            res = collection.delete_many(parsed_filter)
            result_data = {"acknowledged": res.acknowledged, "deleted_count": res.deleted_count}

        return json.dumps({
            "operation": operation,
            "collection": col_to_use,
            "result": result_data
        }, indent=2)

    except Exception as e:
        return f"Error executing MongoDB write query: {str(e)}"

# --- Registry Integration ---

def register_mongo_tools():
    """ Registers MongoDB tools with metadata for the Frontend UI. """
    
    registry.register(
        mongo_configure_auth,
        ToolMeta(
            name="mongo_configure_auth",
            display_name="MongoDB: Configure Credentials",
            description="Initialize MongoDB connection credentials and target collection for this thread.",
            category="Database",
            package="mongodb",
            version="1.0.0",
            allow_duplicate=False,
            ai_visible_fields=[], # Hide credentials from AI; Human must fill this in UI
            configs=[
                {"name": "host", "type": "string", "description": "MongoDB Host", "required": True, "default": "localhost"},
                {"name": "port", "type": "number", "description": "MongoDB Port", "required": True, "default": 27017},
                {"name": "user", "type": "string", "description": "DB Username (Optional)", "required": False},
                {"name": "password", "type": "password", "description": "DB Password (Optional)", "required": False},
                {"name": "database", "type": "string", "description": "Database Name", "required": True},
                {"name": "collection_name", "type": "string", "description": "Target Collection Name (use * or any for wildcard, or comma-separated list)", "required": True},
                {"name": "auth_source", "type": "string", "description": "Authentication Source Database", "required": True, "default": "admin"}
            ]
        )
    )

    registry.register(
        mongo_db_inspector,
        ToolMeta(
            name="mongo_db_inspector",
            display_name="MongoDB: Collection Inspector",
            description="Inspect schema fields, structure, and sample data from the configured collection.",
            category="Database",
            package="mongodb",
            version="1.0.0",
            ai_visible_fields=["sample_limit"]
        )
    )

    registry.register(
        mongo_query_executor,
        ToolMeta(
            name="mongo_query_executor",
            display_name="MongoDB: Query Executor",
            description="Safely run read-only find, count, or aggregate queries on authorized collections.",
            category="Database",
            package="mongodb",
            version="1.0.0",
            ai_visible_fields=["operation", "collection_name", "filter_query", "projection", "sort", "limit", "pipeline"]
        )
    )

    registry.register(
        mongo_write_executor,
        ToolMeta(
            name="mongo_write_executor",
            display_name="MongoDB: Write Executor",
            description="Execute document mutations (insert, update, delete) directly.",
            category="Database",
            package="mongodb",
            version="1.0.0",
            ai_visible_fields=[], # HIDE completely from the LLM agent
            configs=[
                {"name": "operation", "type": "select", "options": ["insert_one", "insert_many", "update_one", "update_many", "delete_one", "delete_many"], "description": "Write Operation", "required": True},
                {"name": "collection_name", "type": "string", "description": "Target Collection Name", "required": False},
                {"name": "document", "type": "string", "description": "Document(s) to insert (JSON object or JSON array)", "required": False},
                {"name": "filter_query", "type": "string", "description": "Filter Query for updates/deletes (JSON object)", "required": False},
                {"name": "update_query", "type": "string", "description": "Update operation (e.g. {'$set': {'status': 'active'}}) (JSON object)", "required": False}
            ]
        )
    )

# Auto-register on load
register_mongo_tools()

