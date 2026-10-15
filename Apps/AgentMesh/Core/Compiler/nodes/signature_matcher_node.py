import logging
from typing import Callable

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Utilities.Tools.RagTools.VectorDatabaseModules.Qdrant_instance import qdrant_manager
from qdrant_client.http.models import Distance

logger = logging.getLogger(__name__)

def build_signature_matcher_node(node_config: NodeConfig) -> Callable:
    """
    Signature Matcher Node.
    Takes a generated vector for a signature and compares it against Qdrant Reference DB.
    """
    match_threshold = getattr(node_config, "match_threshold", 0.75)
    vector_collection = getattr(node_config, "vector_collection", "ReferenceSignatures")
    distance_metric = getattr(node_config, "distance_metric", "cosine")
    target_context_key = getattr(node_config, "target_context_key", "signature_ref_id")
    enable_fallback_search = getattr(node_config, "enable_fallback_search", True)  # default ON

    async def execute_match(state: AgentState):
        ctx = state.get("context", {})
        node_id = node_config.id
        
        batch_vectors = ctx.get("batch_vectors", [])
        
        # Look for the target reference ID in context (payload or top-level)
        reference_id = ctx.get("payload", {}).get(target_context_key) or ctx.get(target_context_key)
        if not reference_id:
            reference_id = ctx.get("payload", {}).get("reference_id") or ctx.get("reference_id")
        
        # Filter for signature vectors
        sig_vectors = [v for v in batch_vectors if v.get("type") == "Signature"]
        
        if not sig_vectors:
            # Fallback: maybe just 'signature_vector' is in context directly
            sig_vec = ctx.get("signature_vector")
            if sig_vec:
                sig_vectors = [{"vector": sig_vec}]
            else:
                msg = f"Signature Matcher {node_id}: No signature vectors found in context."
                return {"messages": [AIMessage(content=f"[ERROR] {msg}")], "active_agent": node_id}

        try:
            client = qdrant_manager._get_client()
            
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue
            
            match_results = []
            
            for idx, sv in enumerate(sig_vectors):
                query_vector = sv.get("vector")
                if not query_vector:
                    continue
                    
                best_match = None
                
                # Step 1: Attempt 1:1 Match (if Reference ID exists)
                if reference_id:
                    filter_condition = Filter(
                        must=[FieldCondition(key="reference_id", match=MatchValue(value=reference_id))]
                    )
                    search_result = client.query_points(
                        collection_name=vector_collection,
                        query=query_vector,
                        query_filter=filter_condition,
                        limit=1
                    ).points
                    if search_result:
                        best_match = search_result[0]
                        
                # Step 2: Attempt 1:N Fallback Match (if 1:1 failed and fallback is enabled)
                if not best_match and enable_fallback_search:
                    search_result = client.query_points(
                        collection_name=vector_collection,
                        query=query_vector,
                        limit=1
                    ).points
                    if search_result:
                        best_match = search_result[0]
                
                if best_match:
                    # In Qdrant, Cosine similarity score ranges usually 0 to 1
                    score = best_match.score
                    # Determine if it's a match based on distance metric
                    # For Cosine/Dot: higher is better. For Euclidean: lower is better.
                    is_match = False
                    if distance_metric in ["cosine", "dot"]:
                        is_match = score >= match_threshold
                    else: # euclidean
                        is_match = score <= match_threshold
                        
                    result_data = {
                        "signature_index": idx,
                        "best_score": score,
                        "reference_payload": best_match.payload,
                        "is_match": is_match,
                        "threshold_used": match_threshold,
                        "metric": distance_metric
                    }
                    match_results.append(result_data)
                else:
                    match_results.append({
                        "signature_index": idx,
                        "error": "No reference vectors found in DB."
                    })
            
            summary_lines = [f"✅ Signature Matcher checked {len(sig_vectors)} signatures."]
            for res in match_results:
                if "error" in res:
                    summary_lines.append(f" - Sig {res['signature_index']}: {res['error']}")
                else:
                    match_str = "MATCH" if res['is_match'] else "NO MATCH"
                    ref_id = res['reference_payload'].get('reference_id', 'Unknown') if res['reference_payload'] else 'Unknown'
                    summary_lines.append(f" - Sig {res['signature_index']}: {match_str} (Score: {res['best_score']:.4f}, Ref: {ref_id})")
            
            summary = "\n".join(summary_lines)
            logger.info(summary)
            
            return {
                "messages": [AIMessage(content=summary, name=f"SigMatch_{node_id}")],
                "active_agent": node_id,
                "context": {
                    **ctx,
                    "signature_match_results": match_results
                }
            }
            
        except Exception as e:
            error_msg = f"Signature Matcher Node {node_id} failed: {str(e)}"
            logger.exception(error_msg)
            return {"messages": [AIMessage(content=f"[ERROR] {error_msg}")], "active_agent": node_id}
            
    return RunnableLambda(execute_match)
