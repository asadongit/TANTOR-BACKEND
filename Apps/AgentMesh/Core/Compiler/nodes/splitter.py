"""
Splitter Node Builder (Production Grade - May 2026)
Handles iteration over lists/arrays with n8n-compatible logic.
Supports batching, custom output mapping, and smart discovery.
"""

import logging
from typing import Callable, Any, List

from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from .utils import node_id_to_key, get_nested_val

logger = logging.getLogger(__name__)


def build_splitter_node(node_config: NodeConfig) -> Callable:
    """
    Build a splitter node that iterates over list items.
    
    Production Features:
    - destination_field: Map current item to a custom context key (e.g. 'user' instead of 'item')
    - smart_discovery: Automatically finds lists in payload if field is not specified
    - batch_size: Supports chunking for high-throughput processing
    """
    async def execute_splitter(state: AgentState):
        ctx = state.get("context", {})
        payload = ctx.get("payload", {})
        
        # 1. Determine the source list (supports comma-separated fallback keys)
        raw_field = node_config.split_field or "items"
        if isinstance(raw_field, str) and raw_field.startswith("{{") and raw_field.endswith("}}"):
            raw_field = raw_field[2:-2].strip()
        
        # Support comma-separated fallback: "key1, key2" tries key1 first, then key2
        candidate_fields = [f.strip() for f in raw_field.split(",") if f.strip()]
        
        field = candidate_fields[0] if candidate_fields else "items"
        source_list = None
        
        for candidate in candidate_fields:
            potential = get_nested_val(payload, candidate) or get_nested_val(ctx, candidate)
            if potential is not None and (not isinstance(potential, list) or len(potential) > 0):
                field = candidate
                source_list = potential
                if len(candidate_fields) > 1:
                    logger.info(f"Splitter {node_config.id}: Resolved field '{candidate}' from fallback list {candidate_fields}")
                break
            
        logger.debug(f"[SPLITTER DEBUG] resolved_field={field}, payload_keys={list(payload.keys()) if payload else []}, ctx_keys={list(ctx.keys()) if ctx else []}")
        
        # 2. Smart discovery logic (Enterprise Fallback)
        if source_list is None or (isinstance(source_list, list) and len(source_list) == 0):
            # Configurable discovery keys — set domain-specific keys in the UI
            # (e.g., ["latest_ocr_output", "ocr_like_results"] for document pipelines)
            configured_discovery = getattr(node_config, "discovery_keys", None) or []
            if isinstance(configured_discovery, str):
                configured_discovery = [k.strip() for k in configured_discovery.split(",") if k.strip()]
            # Generic defaults that work for any workflow
            generic_defaults = ["file_list", "uploaded_files", "data", "results", "tasks", "items", "rows", "records"]
            alternatives = list(configured_discovery) + [k for k in generic_defaults if k not in configured_discovery]
            for alt in alternatives:
                potential = payload.get(alt) or ctx.get(alt)
                if isinstance(potential, list) and len(potential) > 0:
                    field = alt
                    source_list = potential
                    logger.info(f"Splitter {node_config.id}: Auto-discovered list in field '{field}'")
                    break
        
        # 3. Handle non-list types (wrap in list for single-item processing)
        if source_list is not None and not isinstance(source_list, list):
            source_list = [source_list]
            
        # 4. No list found - Exit loop
        if source_list is None or len(source_list) == 0:
            logger.warning(f"Splitter {node_config.id}: No list found to split. Exiting.")
            return {"active_agent": node_config.id, "intent": "done"}
        
        # 5. Track progress using sticky context keys
        progress_key = f"split_{node_id_to_key(node_config.id)}_idx"
        current_index = ctx.get(progress_key, 0)
        batch_size = max(1, node_config.batch_size or 1)
        
        if batch_size == 1 and len(source_list) > 100:
            logger.warning(
                f"Splitter {node_config.id}: Large list of {len(source_list)} items is being processed with batch_size=1. "
                "Consider increasing batch_size in the workflow configuration to reduce graph execution cycles."
            )
        
        # Get the slice for this iteration
        batch = source_list[current_index : current_index + batch_size]
        
        # 6. Check for completion
        if not batch:
            logger.info(f"Splitter {node_config.id}: Loop complete ({len(source_list)} items).")
            return {
                "active_agent": node_config.id,
                "intent": "done",
                "context": {**ctx, progress_key: 0} # Reset for potential re-run
            }
            
        # 7. Prepare the iteration state
        new_index = current_index + batch_size
        is_last = new_index >= len(source_list)
        
        # Map output to destination field (n8n style)
        dest_key = getattr(node_config, "destination_field", "item") or "item"
        item_to_process = batch[0] if batch_size == 1 else batch
        
        # Update context
        new_ctx = {
            **ctx,
            progress_key: new_index,
            dest_key: item_to_process,
            "current_batch": batch,
            "loop_metadata": {
                "index": current_index,
                "batch_index": current_index // batch_size,
                "total": len(source_list),
                "is_last": is_last,
                "source_field": field,
                "destination_field": dest_key
            }
        }

        # If the item is a dictionary with a file_path key (e.g. from multi_file_ingestion), expose latest_file_path
        if isinstance(item_to_process, dict):
            if "file_path" in item_to_process:
                new_ctx["latest_file_path"] = item_to_process["file_path"]
            if "file_name" in item_to_process:
                new_ctx["latest_file_name"] = item_to_process["file_name"]
        
        logger.info(f"[SPLITTER] Processing {dest_key} {current_index+1}/{len(source_list)} (is_last: {is_last})")
        
        return {
            "active_agent": node_config.id,
            "intent": "loop",
            "context": new_ctx
        }
        
    executor = RunnableLambda(execute_splitter)
    setattr(executor, "name", node_config.id)
    return executor
