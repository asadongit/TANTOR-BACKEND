"""
graph_tools.py - Production-Grade Granular Neo4j Graph Tool Suite.

Exposes four decoupled tools matching professional-grade workflows (like n8n and OCI tools):
1. configure_neo4j_auth: Session-based database credentials setup.
2. get_graph_schema: Retrives labels, relationships, property keys, and sample paths (cached).
3. fuzzy_search_graph: Case-insensitive node property matching to locate correct names and labels.
4. execute_graph_query: Secure Cypher query generator (with LLM keywords) and direct execution engine.
"""

import os
import json
import logging
import re
import time
import asyncio
import hashlib
from collections import OrderedDict
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage

from neo4j import AsyncGraphDatabase
from neo4j.graph import Node, Relationship

from Apps.AgentMesh.Core.ToolRegistry import registry
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Utilities.LLM.llm_config import get_llm_provider
from Apps.AgentMesh.Utilities.State.ToolStateManager import ToolStateManager

logger = logging.getLogger(__name__)


# ==========================================
# 1. State / Thread Authentication Manager
# ==========================================
class Neo4jAuthStateProvider:
    @staticmethod
    def get_thread_id(config: RunnableConfig) -> str:
        if not config:
            return "default_thread"
        if isinstance(config, dict):
            configurable = config.get("configurable", {})
        else:
            configurable = getattr(config, "configurable", {})
        
        thread_id = None
        if isinstance(configurable, dict):
            thread_id = (
                configurable.get("thread_id")
                or configurable.get("context", {}).get("thread_id")
            )
        return thread_id or "default_thread"

    @classmethod
    async def get_credentials(cls, config: RunnableConfig) -> Dict[str, str]:
        thread_id = cls.get_thread_id(config)
        creds = await ToolStateManager.get_state(thread_id, "neo4j_creds") or {}
        
        # Priority: Env Vars -> Thread State -> Default Fallbacks
        uri = os.getenv("NEO4J_URI") or creds.get("uri") or "bolt://localhost:7687"
        user = os.getenv("NEO4J_USER") or creds.get("user") or "neo4j"
        password = os.getenv("NEO4J_PASSWORD") or creds.get("password")
        database = creds.get("database") or os.getenv("NEO4J_DATABASE") or "neo4j"
        
        return {"uri": uri, "user": user, "password": password, "database": database}


# ==========================================
# 2. Connection Pool Manager
# ==========================================
class Neo4jConnectionManager:
    _drivers: OrderedDict = OrderedDict()
    _max_drivers: int = 20
    _lock = asyncio.Lock()

    @classmethod
    async def get_driver(cls, uri: str, user: str, password: str):
        if not password:
            raise ValueError("Neo4j credentials are not configured")
        cache_key = hashlib.sha256(
            f"{uri}\0{user}\0{password}".encode("utf-8")
        ).hexdigest()
        async with cls._lock:
            # If driver exists, move to end of OrderedDict to mark as recently used
            if cache_key in cls._drivers:
                driver = cls._drivers.pop(cache_key)
                cls._drivers[cache_key] = driver
                return driver

            logger.info(f"Opening fresh Neo4j driver connection to: {uri}")
            try:
                driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
                # Only verify connectivity during fresh pool checkout to avoid latency on hot path
                await driver.verify_connectivity()
                
                cls._drivers[cache_key] = driver
                
                # Check for pool capacity eviction
                if len(cls._drivers) > cls._max_drivers:
                    oldest_key, oldest_driver = cls._drivers.popitem(last=False)
                    logger.info(f"Evicting connection for {oldest_key.split('||')[0]} from connection pool.")
                    try:
                        await oldest_driver.close()
                    except Exception as ce:
                        logger.warning(f"Error closing evicted Neo4j driver: {str(ce)}")
                
                return driver
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j database at {uri}: {str(e)}")
                raise e


# ==========================================
# 3. Dynamic Schema Caching Engine
# ==========================================
# 3. Dynamic Schema Caching Engine & Filters
# ==========================================
def filter_schema_by_keywords(schema_data: Dict[str, Any], keywords: List[str]) -> str:
    labels = schema_data.get("labels", [])
    rels = schema_data.get("rels", [])
    props = schema_data.get("props", [])
    paths = schema_data.get("paths", [])
    
    if not keywords:
        filtered_labels = labels[:50]
        filtered_rels = rels[:50]
        filtered_props = props[:50]
        filtered_paths = paths[:10]
    else:
        # Normalize keywords: replace "-" and "_" with a space
        keywords_lower = [k.lower().replace("-", " ").replace("_", " ").strip() for k in keywords]
        
        # Helper to normalize database labels/keys/values to match normalized keywords
        def normalize_val(val: str) -> str:
            return val.lower().replace("-", " ").replace("_", " ").strip()
        
        filtered_labels = [
            l for l in labels
            if any(k in normalize_val(l) for k in keywords_lower)
        ]
        filtered_rels = [
            r for r in rels
            if any(k in normalize_val(r) for k in keywords_lower)
        ]
        filtered_props = [
            p for p in props
            if any(k in normalize_val(p) for k in keywords_lower)
        ]
        
        # Include default subsets if no matches
        if not filtered_labels:
            filtered_labels = labels[:15]
        if not filtered_rels:
            filtered_rels = rels[:15]
        if not filtered_props:
            filtered_props = props[:15]
            
        filtered_paths = []
        for path in paths:
            if any(k in normalize_val(path) for k in keywords_lower):
                filtered_paths.append(path)
        if not filtered_paths:
            filtered_paths = paths[:10]
            
    return f"""
NODE LABELS (Filtered): {filtered_labels if filtered_labels else "None found"}
RELATIONSHIP TYPES (Filtered): {filtered_rels if filtered_rels else "None found"}
PROPERTY KEYS (Filtered): {filtered_props if filtered_props else "None found"}
COMMON TRAVERSAL SCHEMAS (Filtered):
{chr(10).join(f"- {p}" for p in filtered_paths) if filtered_paths else "- No matching relationships found."}
""".strip()


class DynamicSchemaEngine:
    _schema_cache: Dict[str, Dict[str, Any]] = {}
    CACHE_TTL_SECONDS = 3600
    _cache_lock = asyncio.Lock()

    @classmethod
    async def get_raw_schema(cls, driver, uri: str, force_refresh: bool = False, database: str = "neo4j") -> Dict[str, Any]:
        current_time = time.time()
        cache_key = f"{uri}||{database}"
        async with cls._cache_lock:
            if not force_refresh and cache_key in cls._schema_cache:
                cache_entry = cls._schema_cache[cache_key]
                if current_time - cache_entry["timestamp"] < cls.CACHE_TTL_SECONDS:
                    return cache_entry["schema_data"]

            try:
                async with driver.session(database=database) as session:
                    labels_res = await session.run("CALL db.labels() YIELD label RETURN label ORDER BY label LIMIT 1000")
                    labels = [r["label"] async for r in labels_res]
                    
                    rels_res = await session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType LIMIT 1000")
                    rels = [r["relationshipType"] async for r in rels_res]
                    
                    props_res = await session.run("CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey ORDER BY propertyKey LIMIT 1000")
                    props = [r["propertyKey"] async for r in props_res]
                    
                    paths_res = await session.run("""
                        MATCH (n)-[r]->(m) 
                        RETURN labels(n)[0] as source_label, type(r) as rel_type, labels(m)[0] as target_label 
                        LIMIT 50
                    """)
                    paths = []
                    async for r in paths_res:
                        s = r["source_label"] or "Entity"
                        t = r["rel_type"] or "RELATED_TO"
                        o = r["target_label"] or "Entity"
                        paths.append(f"(:`{s}`)-[:`{t}`]->(:`{o}`)")

                schema_data = {
                    "labels": labels,
                    "rels": rels,
                    "props": props,
                    "paths": paths
                }
                
                if len(cls._schema_cache) >= 100:
                    cls._schema_cache.pop(next(iter(cls._schema_cache)))

                cls._schema_cache[cache_key] = {
                    "schema_data": schema_data,
                    "timestamp": current_time
                }
                return schema_data
            except Exception as e:
                logger.error(f"Error retrieving database schema: {str(e)}")
                return {"labels": [], "rels": [], "props": [], "paths": []}

    @classmethod
    async def get_schema(cls, driver, uri: str, force_refresh: bool = False, database: str = "neo4j") -> str:
        schema_data = await cls.get_raw_schema(driver, uri, force_refresh, database)
        labels = schema_data.get("labels", [])[:50]
        rels = schema_data.get("rels", [])[:50]
        props = schema_data.get("props", [])[:50]
        paths = schema_data.get("paths", [])[:10]
        
        return f"""
NODE LABELS (Top 50): {labels if labels else "None found"}
RELATIONSHIP TYPES (Top 50): {rels if rels else "None found"}
PROPERTY KEYS (Top 50): {props if props else "None found"}
COMMON TRAVERSAL SCHEMAS (Samples):
{chr(10).join(f"- {p}" for p in paths) if paths else "- No active relationships found."}
""".strip()


# ==========================================
# 4. Result Serializers & Truncation Guards
# ==========================================
def serialize_graph_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def serialize_val(val):
        if isinstance(val, Node):
            return {"__METADATA_LABELS__": list(val.labels), **dict(val)}
        elif isinstance(val, Relationship):
            return {"__METADATA_TYPE__": val.type, "__METADATA_PROPERTIES__": dict(val)}
        elif isinstance(val, list):
            return [serialize_val(item) for item in val]
        elif isinstance(val, dict):
            return {k: serialize_val(v) for k, v in val.items()}
        return val
    return [serialize_val(record) for record in records]


def format_graph_results(results: List[Dict[str, Any]], max_records: int = 8, max_chars: int = 10000) -> str:
    if not results:
        return "GRAPH_RESULTS: No records matched."
    capped_results = results[:max_records]
    json_res = json.dumps(capped_results, indent=2, default=str)
    if len(json_res) > max_chars:
        json_res = json_res[:max_chars] + "\n... [TRUNCATED FOR CONTEXT WINDOW LIMITS]"
    return f"GRAPH_RESULTS: {json_res}"


def is_read_only_query(cypher: str) -> bool:
    """Verifies that the Cypher query contains no database mutation keywords."""
    forbidden = ["CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP", "CREATE INDEX", "CREATE CONSTRAINT"]
    upper_query = cypher.upper()
    
    # Tokenize query to prevent catching properties like `set_value` or `node_name_delete`
    tokens = re.findall(r"\b[A-Z_]+\b", upper_query)
    for kw in forbidden:
        if kw in tokens:
            return False
    return True


# ==========================================
# 5. Core Internal Processing Utilities
# ==========================================
async def extract_keywords_llm(question: str, llm, limit: int = 5) -> List[str]:
    """Uses LLM to extract primary queryable keywords/nouns from the user question."""
    extraction_prompt = (
        f"You are a highly precise keyword extractor. Extract up to {limit} queryable nouns, entity names, "
        f"or technical terms from the following question. Output the result strictly as a clean, "
        f"comma-separated list of terms with no other formatting, introductions, or markdown.\n"
        f"Question: {question}\n"
        f"Keywords:"
    )
    try:
        response = await llm.ainvoke([
            SystemMessage(content="Extract important search terms only."),
            HumanMessage(content=extraction_prompt)
        ])
        raw = response.content.strip().strip('`').strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.splitlines()[1:-1]).strip()
        return [kw.strip() for kw in raw.split(',') if kw.strip()][:limit]
    except Exception as e:
        logger.error(f"Failed keyword extraction: {str(e)}")
        # Simple fallback token splitter
        return [w for w in re.findall(r"\b\w{3,}\b", question.lower())][:limit]


async def _run_anchor_expand_fallback(question_text: str, keywords: List[str], driver, database: str = "neo4j") -> List[Dict[str, Any]]:
    """Uses extracted entity fallback if the first query fails or returns empty."""
    if not keywords:
        return []
    
    entity_name = keywords[0]
    normalized_entity = entity_name.lower().replace("-", " ").replace("_", " ").strip()
    fallback_cypher = f"""
    MATCH (n)-[r]-(m)
    WHERE replace(replace(toLower(n.name), '-', ' '), '_', ' ') CONTAINS "{normalized_entity}"
       OR replace(replace(toLower(m.name), '-', ' '), '_', ' ') CONTAINS "{normalized_entity}"
       OR ANY(l IN labels(n) WHERE replace(replace(toLower(l), '-', ' '), '_', ' ') CONTAINS "{normalized_entity}")
       OR ANY(l IN labels(m) WHERE replace(replace(toLower(l), '-', ' '), '_', ' ') CONTAINS "{normalized_entity}")
       OR ANY(k IN keys(n) WHERE replace(replace(toLower(k), '-', ' '), '_', ' ') CONTAINS "{normalized_entity}")
       OR ANY(k IN keys(m) WHERE replace(replace(toLower(k), '-', ' '), '_', ' ') CONTAINS "{normalized_entity}")
       OR ANY(tag IN n.tags WHERE replace(replace(toLower(tag), '-', ' '), '_', ' ') CONTAINS "{normalized_entity}")
       OR ANY(tag IN m.tags WHERE replace(replace(toLower(tag), '-', ' '), '_', ' ') CONTAINS "{normalized_entity}")
    RETURN n, r, m
    LIMIT 10
    """
    try:
        async with driver.session(database=database) as session:
            res = await session.run(fallback_cypher)
            records = [dict(record) async for record in res]
            return serialize_graph_records(records)
    except Exception as e:
        logger.error(f"Anchor Expand Fallback execution failed: {str(e)}")
        return []


# ==========================================
# 6. Exponent / Decoupled Tools Definitions
# ==========================================
class Neo4jAuthInput(BaseModel):
    uri: str = Field(description="Neo4j Connection URI (e.g. 'bolt://localhost:7687')")
    user: str = Field(description="Database user name")
    password: str = Field(description="Database user password", repr=False)
    database: Optional[str] = Field(default="neo4j", description="Target database name (optional)")

@tool("configure_neo4j_auth", args_schema=Neo4jAuthInput)
async def configure_neo4j_auth(uri: str, user: str, password: str, database: Optional[str] = "neo4j", config: RunnableConfig = None) -> str:
    """
    Configure and save Neo4j database credentials dynamically for this thread session.
    """
    try:
        thread_id = Neo4jAuthStateProvider.get_thread_id(config)
        creds = {"uri": uri, "user": user, "password": password, "database": database}
        await ToolStateManager.save_state(thread_id, "neo4j_creds", creds)
        return f"Successfully updated Neo4j database authentication credentials and database target to '{database}' for thread: {thread_id}."
    except Exception as e:
        logger.error(f"Failed to save Neo4j auth state: {str(e)}")
        return f"Error setting Neo4j credentials: {str(e)}"


class Neo4jSchemaInput(BaseModel):
    refresh_cache: bool = Field(default=False, description="Force a full schema check and ignore cached data")

@tool("get_graph_schema", args_schema=Neo4jSchemaInput)
async def get_graph_schema(refresh_cache: bool = False, config: RunnableConfig = None) -> str:
    """
    Retrieve active Node Labels, Relationship Types, and active connectivity paths in Neo4j.
    """
    try:
        creds = await Neo4jAuthStateProvider.get_credentials(config)
        driver = await Neo4jConnectionManager.get_driver(creds["uri"], creds["user"], creds["password"])
        return await DynamicSchemaEngine.get_schema(driver, creds["uri"], force_refresh=refresh_cache, database=creds["database"])
    except Exception as e:
        return f"ERROR: Failed to retrieve active database structure: {str(e)}"


class Neo4jFuzzySearchInput(BaseModel):
    search_term: str = Field(description="The entity, noun, name, or partial keyword to locate (e.g. 'mahindra')")
    limit: int = Field(default=10, description="Max matching nodes to return")

@tool("fuzzy_search_graph", args_schema=Neo4jFuzzySearchInput)
async def fuzzy_search_graph(search_term: str, limit: int = 10, config: RunnableConfig = None) -> str:
    """
    Perform a case-insensitive fuzzy name property search to locate correct nodes, properties, and labels.
    """
    try:
        creds = await Neo4jAuthStateProvider.get_credentials(config)
        driver = await Neo4jConnectionManager.get_driver(creds["uri"], creds["user"], creds["password"])
        
        # Normalize search term by replacing "-" and "_" with a space
        normalized_term = search_term.lower().replace("-", " ").replace("_", " ").strip()
        
        search_query = f"""
        MATCH (n)
        WHERE replace(replace(toLower(n.name), '-', ' '), '_', ' ') CONTAINS "{normalized_term}"
           OR replace(replace(toLower(n.title), '-', ' '), '_', ' ') CONTAINS "{normalized_term}"
           OR replace(replace(toLower(n.description), '-', ' '), '_', ' ') CONTAINS "{normalized_term}"
           OR ANY(l IN labels(n) WHERE replace(replace(toLower(l), '-', ' '), '_', ' ') CONTAINS "{normalized_term}")
           OR ANY(k IN keys(n) WHERE replace(replace(toLower(k), '-', ' '), '_', ' ') CONTAINS "{normalized_term}")
           OR ANY(tag IN n.tags WHERE replace(replace(toLower(tag), '-', ' '), '_', ' ') CONTAINS "{normalized_term}")
        RETURN labels(n) as node_labels, n.name as matched_name, properties(n) as all_properties
        LIMIT {limit}
        """
        async with driver.session(database=creds["database"]) as session:
            res = await session.run(search_query)
            records = [dict(record) async for record in res]
            return format_graph_results(records)
    except Exception as e:
        return f"ERROR: Fuzzy search failed: {str(e)}"


class Neo4jExecuteInput(BaseModel):
    question: Optional[str] = Field(None, description="Natural language question to translate and execute (e.g. 'Show revenue metric of Mahindra')")
    cypher_query: Optional[str] = Field(None, description="Direct Cypher query statement to execute directly (e.g. 'MATCH (n) RETURN n LIMIT 5')")

@tool("execute_graph_query", args_schema=Neo4jExecuteInput)
async def execute_graph_query(question: Optional[str] = None, cypher_query: Optional[str] = None, config: RunnableConfig = None) -> str:
    """
    Execute a read-only Cypher query directly, or provide a natural language question which is automatically translated and run.
    """
    input_str = question or cypher_query
    if not input_str:
        return "ERROR: Missing query payload. You must supply a 'question' or a 'cypher_query'."
    
    # 1. Auth Setup
    try:
        creds = await Neo4jAuthStateProvider.get_credentials(config)
        driver = await Neo4jConnectionManager.get_driver(creds["uri"], creds["user"], creds["password"])
    except Exception as e:
        return f"ERROR: Database connection failed: {str(e)}"

    # 2. Check direct execution path
    is_direct_cypher = False
    input_strip = input_str.strip()
    upper_input = input_strip.upper()
    
    starts_with_cypher = any(upper_input.startswith(kw) for kw in ["MATCH", "RETURN", "WITH", "CALL", "CREATE", "MERGE"])
    has_cypher_structure = any(kw in upper_input for kw in ["MATCH ", "MERGE ", "UNWIND ", "OPTIONAL MATCH"])
    
    if starts_with_cypher or has_cypher_structure:
        is_direct_cypher = True

    if is_direct_cypher:
        # Strict security validation
        if not is_read_only_query(input_strip):
            return "ERROR: Security violation. Dangerous mutating actions (CREATE, DELETE, MERGE, SET, etc.) are strictly prohibited."
        
        try:
            async with driver.session(database=creds["database"]) as session:
                res = await session.run(input_strip)
                records = [dict(record) async for record in res]
                serialized = serialize_graph_records(records)
                return format_graph_results(serialized)
        except Exception as e:
            return f"ERROR: Cypher statement execution failed: {str(e)}"

    # 3. LLM Translation Path (Token-Optimized with LLM Keywords + Cached Schema)
    keywords = []
    try:
        llm = get_llm_provider()
        keywords = await extract_keywords_llm(input_strip, llm, limit=5)
        schema_data = await DynamicSchemaEngine.get_raw_schema(driver, creds["uri"], database=creds["database"])
        schema_info = filter_schema_by_keywords(schema_data, keywords)
    except Exception as e:
        return f"ERROR: Preparation failed: {str(e)}"

    system_prompt = f"""You are a precise, expert Neo4j Cypher query generator.
Given a database schema, primary search keywords, and a natural language question, generate a single, syntactically correct read-only Cypher query.

### Primary Keywords Detected:
{', '.join(keywords) if keywords else "None"}

### Database Schema:
{schema_info}

### Cypher Generation Rules:
1. Fuzzy Name Filtering:
   Always use case-insensitive fuzzy matching on name/title properties.
   Use `toLower(n.name) CONTAINS "value"` instead of strict equals matches (`n.name = "value"`).
2. Undirected Traversal:
   Always use undirected relationship traversals (e.g. `-[r]-` or `-[:HAS_METRIC]-`). Never use directed arrows (`->` or `<-`) unless specifically required, to avoid getting empty returns from flipped schema directions.
3. Answer Extraction:
   Prefer returning full nodes (e.g. `RETURN n` or `RETURN n, m`) if the target property keys are not perfectly obvious.
4. Output format:
   Output ONLY the raw executable Cypher query. Do not wrap it in markdown code blocks (do NOT use ```cypher), and do not include any introductions, explanations, or markdown syntax.
5. No Relationship Wildcards:
   Do NOT guess or invent relationship types. Only use relationship types explicitly defined in the Database Schema. Never generate wildcard names like `[:HAS_*]` or query non-existent relationship types.
6. Temporal/Quarterly Synonyms:
   When searching for periodic entities, generate Cypher clauses that check multiple name variations (e.g. `toLower(n.name) CONTAINS "q3 fy24" OR toLower(n.name) CONTAINS "q3 fy2024" OR (toLower(n.name) CONTAINS "q3" AND toLower(n.name) CONTAINS "2024")`) to ensure spelling or formatting differences do not cause empty query results.
"""
    
    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Question: {input_strip}")
        ])
        generated_cypher = response.content.strip()
        
        if generated_cypher.startswith("```"):
            lines = generated_cypher.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            generated_cypher = "\n".join(lines).strip()
            
        logger.info(f"LLM Generated Cypher: {generated_cypher}")

        # Security check on generated query
        if not is_read_only_query(generated_cypher):
            return "ERROR: Security violation. The generated query attempted a mutating operation which is blocked."

        # Execute
        async with driver.session(database=creds["database"]) as session:
            res = await session.run(generated_cypher)
            records = [dict(record) async for record in res]
            results = serialize_graph_records(records)
            
            # Fallback if empty
            if not results:
                logger.info("Main Cypher query returned no records. Initiating Anchor-and-Expand fallback...")
                fallback_results = await _run_anchor_expand_fallback(input_strip, keywords, driver, database=creds["database"])
                return format_graph_results(fallback_results)

            return format_graph_results(results)

    except Exception as e:
        logger.error(f"Error executing/generating query: {str(e)}")
        # Execute Fallback
        try:
            fallback_results = await _run_anchor_expand_fallback(input_strip, keywords, driver, database=creds["database"])
            if fallback_results:
                return format_graph_results(fallback_results)
        except Exception:
            pass
        return f"ERROR: Graph query execution failed: {str(e)}"


# ==========================================
# 7. Core Tools Multi-Registration Binding
# ==========================================
def register_granular_graph_tools():
    """Registers our granular 4-tool Graph Suite into the AgentMesh Core registry."""
    # 1. Auth Setup
    registry.register(
        configure_neo4j_auth,
        ToolMeta(
            name="configure_neo4j_auth",
            display_name="Neo4j: Configure Authentication",
            description="Dynamically configure and save Neo4j Connection credentials for this thread session.",
            category="Graph Database",
            package="built-in",
            version="1.3.0",
            configs=[
                {"name": "uri", "type": "string", "description": "Neo4j Connection URI (bolt/neo4j)", "required": True, "default": "bolt://localhost:7687"},
                {"name": "user", "type": "string", "description": "Username (default: neo4j)", "required": True, "default": "neo4j"},
                {"name": "password", "type": "string", "description": "Password", "required": True, "default": ""},
                {"name": "database", "type": "string", "description": "Target Database (default: neo4j)", "required": False, "default": "neo4j"}
            ]
        )
    )

    # 2. Schema Retriever
    registry.register(
        get_graph_schema,
        ToolMeta(
            name="get_graph_schema",
            display_name="Neo4j: Get Database Schema",
            description="Explore active node labels, relationship types, and active connectivity paths in Neo4j.",
            category="Graph Database",
            package="built-in",
            version="1.3.0",
            ai_visible_fields=["refresh_cache"]
        )
    )

    # 3. Fuzzy search locator
    registry.register(
        fuzzy_search_graph,
        ToolMeta(
            name="fuzzy_search_graph",
            display_name="Neo4j: Fuzzy Search Entity",
            description="Find exact entity names, node properties, and labels using fuzzy case-insensitive keyword searches.",
            category="Graph Database",
            package="built-in",
            version="1.3.0",
            ai_visible_fields=["search_term"]
        )
    )

    # 4. Executor
    registry.register(
        execute_graph_query,
        ToolMeta(
            name="execute_graph_query",
            display_name="Neo4j: Execute Search",
            description="Securely query graph database using a natural language question (translated to Cypher) or a raw read-only Cypher statement.",
            category="Graph Database",
            package="built-in",
            version="1.3.0",
            ai_visible_fields=["question", "cypher_query"]
        )
    )
    logger.info("Successfully registered the granular 4-tool Graph suite in AgentMesh Core.")


# Execute Registration automatically
try:
    register_granular_graph_tools()
except Exception as re:
    logger.error(f"Failed to auto-register modular graph tools: {str(re)}")

