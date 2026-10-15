import logging
import os
import json
import re
from typing import Callable, Dict, Any
from langchain_core.runnables import RunnableLambda
from neo4j import AsyncGraphDatabase

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState

logger = logging.getLogger(__name__)

def build_network_graph_node(node_config: NodeConfig) -> Callable:
    """
    Builds a node that ingests transaction data into a Neo4j Graph Database.
    """
    
    target_key = getattr(node_config, 'graph_target_key', 'latest_message')
    uri = getattr(node_config, 'graph_neo4j_uri', None) or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = getattr(node_config, 'graph_neo4j_user', None) or os.environ.get("NEO4J_USERNAME", os.environ.get("NEO4J_USER", "neo4j"))
    password = getattr(node_config, 'graph_neo4j_password', None) or os.environ.get("NEO4J_PASSWORD")

    # Sanitize dynamic graph collection name for cypher queries (no injection)
    raw_col = getattr(node_config, 'graph_collection', 'FraudFlow') or 'FraudFlow'
    collection = re.sub(r'[^a-zA-Z0-9_]', '', raw_col)
    if not collection:
        collection = 'FraudFlow'

    def resolve_path(data: Dict[str, Any], path: str, default: Any = None) -> Any:
        if not path:
            return default
        parts = path.split('.')
        current = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    async def network_graph_node(state: AgentState) -> Dict[str, Any]:
        if not password:
            raise ValueError("Neo4j credentials are not configured")
        logger.info(f"[{node_config.id}] Running network_graph_node. Target key: {target_key}")
        
        def is_valid_data(val):
            if isinstance(val, dict):
                return True
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                return True
            return False

        # 1. Extract transaction details from all possible pipeline sources
        data = None
        context = state.get("context", {})

        # Priority 1a: cleaned_transaction.cleaned_data (from data_cleansing_node)
        cleaned_tx = context.get("cleaned_transaction")
        if isinstance(cleaned_tx, dict) and "cleaned_data" in cleaned_tx:
            cd = cleaned_tx["cleaned_data"]
            if isinstance(cd, dict):
                data = cd
                logger.info(f"[{node_config.id}] Data from context.cleaned_transaction.cleaned_data")

        # Priority 1b: context_data.cleaned_transaction.cleaned_data (alternative path)
        if not is_valid_data(data):
            ctx_data = context.get("context_data", {})
            if isinstance(ctx_data, dict):
                for sub_key in ["cleaned_transaction", "enriched_transaction", "canonical_transaction"]:
                    sub = ctx_data.get(sub_key)
                    if isinstance(sub, dict):
                        if "cleaned_data" in sub and isinstance(sub["cleaned_data"], dict):
                            data = sub["cleaned_data"]
                        elif is_valid_data(sub):
                            data = sub
                        if is_valid_data(data):
                            logger.info(f"[{node_config.id}] Data from context.context_data.{sub_key}")
                            break

        # Priority 2: enriched_transaction
        if not is_valid_data(data):
            enriched = context.get("enriched_transaction")
            if is_valid_data(enriched):
                data = enriched
                logger.info(f"[{node_config.id}] Data from context.enriched_transaction")

        # Priority 3: canonical_transaction
        if not is_valid_data(data):
            canonical = context.get("canonical_transaction")
            if is_valid_data(canonical):
                data = canonical
                logger.info(f"[{node_config.id}] Data from context.canonical_transaction")

        # Priority 4: nodes from context (any node's output that has valid data)
        if not is_valid_data(data):
            nodes = context.get("nodes", {})
            for nid, ndata in nodes.items():
                if is_valid_data(ndata):
                    data = ndata
                    logger.info(f"[{node_config.id}] Data from context.nodes.{nid}")
                    break

        # Priority 5: latest_api_result (ML predictions)
        if not is_valid_data(data):
            latest_api = context.get("latest_api_result")
            if is_valid_data(latest_api):
                data = latest_api
                logger.info(f"[{node_config.id}] Data from context.latest_api_result")

        # Priority 6: target_key from state or context
        if not is_valid_data(data):
            val = state.get(target_key)
            if not is_valid_data(val):
                val = context.get(target_key)
            if is_valid_data(val):
                data = val
                logger.info(f"[{node_config.id}] Data from state/context['{target_key}']")

        # Priority 7: last message as JSON (fallback)
        if not is_valid_data(data) and state.get("messages"):
            try:
                parsed = json.loads(state["messages"][-1].content)
                if is_valid_data(parsed):
                    data = parsed
                    logger.info(f"[{node_config.id}] Data from latest_message")
            except Exception:
                pass

        if not is_valid_data(data):
            logger.warning(f"No valid transaction data found. Skipping Neo4j ingestion.")
            return {}

        # 2. Convert to list of transactions
        transactions = []
        if isinstance(data, list):
            transactions = data
        elif isinstance(data, dict):
            if "cleaned_data" in data and isinstance(data["cleaned_data"], dict):
                transactions = [data["cleaned_data"]]
            elif "predictions" in data and isinstance(data["predictions"], list):
                transactions = data["predictions"]
            elif "data" in data and isinstance(data["data"], list):
                transactions = data["data"]
            elif "transactions" in data and isinstance(data["transactions"], list):
                transactions = data["transactions"]
            else:
                transactions = [data]
            
        try:
            driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        except Exception as e:
            logger.error(f"[{node_config.id}] Failed to create Neo4j driver: {e}")
            return {}
        
        ingestion_mode = getattr(node_config, 'graph_ingestion_mode', 'visual')
        cypher_query = getattr(node_config, 'graph_cypher_query', '')
        nodes_schema = getattr(node_config, 'graph_nodes_schema', []) or []
        relations_schema = getattr(node_config, 'graph_relations_schema', []) or []

        async def ingest_batch_visual(tx, batch_items):
            # Group nodes by label
            nodes_by_label = {}
            for tx_data in batch_items:
                for node_def in nodes_schema:
                    lbl = re.sub(r'[^a-zA-Z0-9_]', '', node_def.get("label", ""))
                    id_p = node_def.get("id_path", "")
                    if not lbl or not id_p:
                        continue
                    node_id = resolve_path(tx_data, id_p)
                    if node_id is None:
                        continue
                    node_id = str(node_id)
                    
                    props = {"id": node_id}
                    for prop_def in node_def.get("properties", []):
                        prop_name = re.sub(r'[^a-zA-Z0-9_]', '', prop_def.get("name", ""))
                        prop_path = prop_def.get("path", "")
                        if not prop_name or not prop_path:
                            continue
                        prop_val = resolve_path(tx_data, prop_path)
                        if prop_val is not None:
                            props[prop_name] = prop_val
                    
                    if lbl not in nodes_by_label:
                        nodes_by_label[lbl] = {}
                    # De-duplicate nodes by ID in this batch
                    nodes_by_label[lbl][node_id] = props

            # Batch merge nodes
            for lbl, nodes_dict in nodes_by_label.items():
                props_list = list(nodes_dict.values())
                if not props_list:
                    continue
                # UNWIND properties list and merge dynamic attributes
                query = f"""
                UNWIND $props AS p
                MERGE (n:{lbl}:{collection} {{id: p.id}})
                SET n += p
                """
                await tx.run(query, props=props_list)

            # Group relationships by (src_lbl, rel_type, tgt_lbl)
            relations_by_signature = {}
            for tx_data in batch_items:
                for rel_def in relations_schema:
                    src_lbl = re.sub(r'[^a-zA-Z0-9_]', '', rel_def.get("source_label", ""))
                    tgt_lbl = re.sub(r'[^a-zA-Z0-9_]', '', rel_def.get("target_label", ""))
                    rel_t = re.sub(r'[^a-zA-Z0-9_]', '', rel_def.get("type", ""))
                    src_id_p = rel_def.get("source_id_path", "")
                    tgt_id_p = rel_def.get("target_id_path", "")
                    
                    if not src_lbl or not tgt_lbl or not rel_t or not src_id_p or not tgt_id_p:
                        continue
                    
                    src_id = resolve_path(tx_data, src_id_p)
                    tgt_id = resolve_path(tx_data, tgt_id_p)
                    if src_id is None or tgt_id is None:
                        continue
                    
                    sig = (src_lbl, rel_t, tgt_lbl)
                    if sig not in relations_by_signature:
                        relations_by_signature[sig] = set()
                    relations_by_signature[sig].add((str(src_id), str(tgt_id)))

            # Batch merge relationships
            for (src_lbl, rel_type, tgt_lbl), rels_set in relations_by_signature.items():
                rels_list = [{"src_id": r[0], "tgt_id": r[1]} for r in rels_set]
                if not rels_list:
                    continue
                query = f"""
                UNWIND $rels AS r
                MATCH (s:{src_lbl}:{collection} {{id: r.src_id}})
                MATCH (t:{tgt_lbl}:{collection} {{id: r.tgt_id}})
                MERGE (s)-[:{rel_type}]->(t)
                """
                await tx.run(query, rels=rels_list)

        async def execute_cypher_concurrently(session, batch_items):
            import asyncio
            tasks = []
            for tx_data in batch_items:
                params = {**tx_data, "item": tx_data}
                tasks.append(session.run(cypher_query, **params))
            await asyncio.gather(*tasks)

        try:
            async with driver.session() as session:
                valid_dict_transactions = [tx for tx in transactions if isinstance(tx, dict)]
                if valid_dict_transactions:
                    if ingestion_mode == 'cypher':
                        await execute_cypher_concurrently(session, valid_dict_transactions)
                    else:
                        await session.execute_write(ingest_batch_visual, valid_dict_transactions)
            logger.info(f"[{node_config.id}] Successfully ingested {len(transactions)} transaction(s) into Neo4j in mode '{ingestion_mode}' using collection label '{collection}'.")
        except Exception as e:
            logger.error(f"[{node_config.id}] Neo4j ingestion failed: {e}")
        finally:
            await driver.close()

        return {}

    return RunnableLambda(network_graph_node)
