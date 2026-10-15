"""
Ingestion Node Builder
Handles automatic document ingestion into vector stores.
"""

import logging
from typing import Callable

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Core.Services.IngestionService import IngestionService

logger = logging.getLogger(__name__)


def build_ingestion_node(node_config: NodeConfig) -> Callable:
    """
    Build an ingestion node that automatically ingests incoming message content
    into a specified vector collection.
    """
    async def execute_ingestion(state: AgentState):
        context = state.get("context", {})
        data_type = getattr(node_config, "data_type", "text_rag")

        if data_type == "precomputed_vector":
            # Native vector insertion path (bypasses LLM chunking)
            collection_name = node_config.vector_collection or "agent_memory"
            
            batch_vectors = context.get("batch_vectors")
            # Pull in any payload built by the Payload Builder node
            context_payload = context.get("payload", {})
            if isinstance(context_payload, str):
                import json as _json
                try:
                    context_payload = _json.loads(context_payload)
                except Exception:
                    context_payload = {}
            logger.info(f"Ingestion node {node_config.id}: data_type=precomputed_vector, batch_vectors count={len(batch_vectors) if batch_vectors else 0}, context_payload={context_payload}")
            if not batch_vectors:
                # Fallback to single vector logic
                vector_key = getattr(node_config, "vector_context_key", "signature_vector")
                payload_key = getattr(node_config, "payload_context_key", "reference_id")
                vector_data = context.get(vector_key)
                if not vector_data:
                    return {"messages": [AIMessage(content=f"[ERROR] Ingestion node {node_config.id}: Vector array not found at context key '{vector_key}' or 'batch_vectors'")], "active_agent": node_config.id}
                    
                payload_data = context.get(payload_key)
                payload_dict = payload_data if isinstance(payload_data, dict) else {payload_key: payload_data} if payload_data else {}
                batch_vectors = [{"vector": vector_data, "payload": payload_dict}]
                
            if not isinstance(batch_vectors, list) or len(batch_vectors) == 0:
                return {"messages": [AIMessage(content=f"[ERROR] Ingestion node {node_config.id}: Empty or invalid batch_vectors array.")], "active_agent": node_config.id}
                
            try:
                import uuid
                from qdrant_client.models import PointStruct, Distance, VectorParams
                from Apps.AgentMesh.Utilities.Tools.RagTools.VectorDatabaseModules.Qdrant_instance import qdrant_manager
                
                client = qdrant_manager._get_client()
                
                # Check if collection exists using try/except for compatibility
                # with Qdrant client/server version mismatches
                collection_exists = False
                try:
                    client.get_collection(collection_name)
                    collection_exists = True
                except Exception:
                    collection_exists = False
                
                if not collection_exists:
                    vector_size = len(batch_vectors[0]["vector"])
                    logger.info(f"Ingestion node {node_config.id}: Collection '{collection_name}' not found. Creating it with size {vector_size}.")
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                    )
                
                # Filter by target_vector_type if configured (allows parallel segregation)
                target_type = getattr(node_config, "target_vector_type", None)
                if target_type:
                    batch_vectors = [v for v in batch_vectors if v.get("type") == target_type]
                    if not batch_vectors:
                        logger.info(f"Ingestion node {node_config.id}: No vectors matched target_type '{target_type}'. Skipping.")
                        return {"messages": [AIMessage(content=f"Ingestion node {node_config.id}: Skipped (no matching types).")], "active_agent": node_config.id}

                points = []
                for item in batch_vectors:
                    # Merge: context_payload (from Payload Builder) + top-level 'type' from item
                    merged_payload = {**context_payload}
                    # Add the type field (Signature/Stamp) stored at top level of each batch item
                    if item.get("type"):
                        merged_payload["type"] = item["type"]
                    # Override with any explicit payload already inside the item (highest priority)
                    if item.get("payload"):
                        merged_payload.update(item["payload"])
                    points.append(
                        PointStruct(
                            id=uuid.uuid4().hex,
                            vector=item["vector"],
                            payload=merged_payload
                        )
                    )
                logger.info(f"Ingestion node {node_config.id}: Upserting {len(points)} points into '{collection_name}'. Sample payload: {points[0].payload if points else {}}")
                
                client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                
                ref_id = batch_vectors[0].get("payload", {}).get("reference_id", "unknown")
                
                sig_count = sum(1 for v in batch_vectors if v.get("type", "").lower() == "signature" or v.get("payload", {}).get("type", "").lower() == "signature")
                stamp_count = sum(1 for v in batch_vectors if v.get("type", "").lower() == "stamp" or v.get("payload", {}).get("type", "").lower() == "stamp")
                
                if sig_count == 0 and stamp_count == 0:
                    doc_types = f"{len(batch_vectors)} item(s)"
                else:
                    doc_types = f"{sig_count} Signature(s) and {stamp_count} Stamp(s)"
                
                success_msg = f"✅ Reference Image Stored in Collection: {collection_name}"
                logger.info(f"Ingestion node {node_config.id}: Successfully uploaded {doc_types} for '{ref_id}' into '{collection_name}'.")
                
                return {
                    "messages": [
                        AIMessage(content=success_msg, name=node_config.id, response_metadata={"active_agent": node_config.id})
                    ],
                    "active_agent": node_config.id
                }
            except Exception as e:
                logger.error(f"Ingestion node {node_config.id} failed vector insertion: {e}")
                return {"error": str(e), "active_agent": node_config.id}
        
        # 1. Resolve Data Source (Original Text/RAG path)
        # Priority: ingestion_source_key -> latest_ocr_output -> latest_message
        source_key = getattr(node_config, "ingestion_source_key", None)
        content = None
        
        if source_key and source_key in context:
            content = context[source_key]
            logger.info(f"Ingestion node {node_config.id}: Ingesting from context key '{source_key}'")
        elif "latest_ocr_output" in context and context["latest_ocr_output"]:
            content = context["latest_ocr_output"]
            logger.info(f"Ingestion node {node_config.id}: Auto-detected OCR output for ingestion")
        else:
            messages = state.get("messages", [])
            if messages:
                content = messages[-1].content
                logger.info(f"Ingestion node {node_config.id}: Ingesting latest message content")
        
        if not content:
            logger.warning(f"Ingestion node {node_config.id}: No content found to ingest")
            return {"active_agent": node_config.id}
        
        # 2. Ingest into vector store
        try:
            count = await IngestionService.ingest_text(
                text=str(content),
                collection_name=node_config.vector_collection or "agent_memory",
                vector_store_type=node_config.vector_store or "chroma",
                chunk_size=node_config.chunk_size or 1000,
                chunk_overlap=getattr(node_config, "chunk_overlap", 100),
                chunking_strategy=getattr(node_config, "chunking_strategy", "recursive")
            )
            
            logger.info(f"Ingestion node {node_config.id}: Ingested {count} chunks")
            
            return {
                "messages": [
                    AIMessage(content=f"Successfully vectorized {count} chunks of data into '{node_config.vector_collection}'.", response_metadata={"active_agent": node_config.id})
                ],
                "active_agent": node_config.id
            }
        except Exception as e:
            logger.error(f"Ingestion node {node_config.id} failed: {e}")
            return {
                "error": str(e),
                "active_agent": node_config.id
            }
    
    return RunnableLambda(execute_ingestion)
