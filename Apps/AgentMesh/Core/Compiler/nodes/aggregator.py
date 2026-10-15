"""
Aggregator Node Builder (Industry Grade - May 2026)
Collects multiple items from a loop and groups them into a single payload.
Supports list appending, object merging, and string concatenation.
"""

import logging
from typing import Callable, Any, List, Dict
import json

from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from .utils import node_id_to_key
from Apps.AgentMesh.Core.RedisManager import redis_manager

logger = logging.getLogger(__name__)

# Process-level cache for ultra-fast in-memory access
_AGGREGATOR_CACHE: Dict[str, List[Any]] = {}
_AGGREGATOR_CACHE_MAX = 5000  # Evict oldest when exceeding this

# CRASH-RECOVERY TRADE-OFF WARNING:
# To prevent database serialization bloat of O(n) loop records inside checkpoints,
# intermediate aggregator results are cached in local memory / Redis.
# If the backend crashes/restarts during an active loop execution and Redis is unavailable,
# the intermediate state is unrecoverable, requiring the loop to restart.
# This trade-off is accepted to enable high-throughput scaling (10K+ loops).

def _evict_cache_if_needed():
    while len(_AGGREGATOR_CACHE) > _AGGREGATOR_CACHE_MAX:
        _AGGREGATOR_CACHE.pop(next(iter(_AGGREGATOR_CACHE)), None)

async def _get_bucket(thread_id: str, node_id: str, storage_key: str, ctx: dict) -> list:
    cache_key = f"aggregator:{thread_id}:{node_id}"
    
    # 1. Try local process-level cache
    if cache_key in _AGGREGATOR_CACHE:
        return _AGGREGATOR_CACHE[cache_key]
        
    # 2. Try Redis cache if available
    if redis_manager.available and redis_manager.client:
        try:
            val = await redis_manager.client.get(cache_key)
            if val:
                bucket = json.loads(val)
                _AGGREGATOR_CACHE[cache_key] = bucket
                _evict_cache_if_needed()
                return bucket
        except Exception as e:
            logger.warning(f"Failed to fetch aggregator bucket from Redis: {e}")
            
    # 3. Fallback to graph checkpoint context (durability fallback)
    bucket = ctx.get(storage_key, [])
    if not isinstance(bucket, list):
        bucket = []
    _AGGREGATOR_CACHE[cache_key] = bucket
    _evict_cache_if_needed()
    return bucket

async def _save_bucket(thread_id: str, node_id: str, bucket: list):
    cache_key = f"aggregator:{thread_id}:{node_id}"
    _AGGREGATOR_CACHE[cache_key] = bucket
    _evict_cache_if_needed()
    
    if redis_manager.available and redis_manager.client:
        try:
            await redis_manager.client.set(cache_key, json.dumps(bucket), ex=3600)
        except Exception as e:
            logger.warning(f"Failed to save aggregator bucket to Redis: {e}")

async def _clear_bucket(thread_id: str, node_id: str):
    cache_key = f"aggregator:{thread_id}:{node_id}"
    _AGGREGATOR_CACHE.pop(cache_key, None)
    if redis_manager.available and redis_manager.client:
        try:
            await redis_manager.client.delete(cache_key)
        except Exception as e:
            logger.warning(f"Failed to delete aggregator bucket from Redis: {e}")


def build_aggregator_node(node_config: NodeConfig) -> Callable:
    """
    Build an aggregator node that collects data across multiple execution passes.
    
    Production Features:
    - action: 'append' (list), 'merge' (dict), 'concatenate' (string)
    - field_to_aggregate: The path to the data to collect (defaults to 'item')
    - destination_field: Where to store the final result (defaults to 'aggregated_results')
    - smart_completion: Automatically detects loop completion via loop_metadata
    """
    async def execute_aggregator(state: AgentState):
        ctx = state.get("context", {})
        loop_metadata = ctx.get("loop_metadata", {})
        thread_id = ctx.get("thread_id", "default")
        
        # 1. Configurable parameters
        action = getattr(node_config, "action", "append") or "append"
        source_field = getattr(node_config, "field_to_aggregate", "item") or "item"
        dest_field = getattr(node_config, "destination_field", "aggregated_results") or "aggregated_results"
        
        # 2. Extract current value to aggregate
        current_val = ctx.get(source_field)
        if current_val is None and "." in source_field:
            parts = source_field.split(".")
            current_val = ctx
            for p in parts:
                if isinstance(current_val, dict):
                    current_val = current_val.get(p)
                else:
                    current_val = None
                    break
        
        # When the named field holds nothing, fall back to keys that earlier
        # workflows happened to use. Say so — collecting something other than
        # what was asked for, silently, is how a loop ends up gathering OCR text
        # where a configured field was expected and nothing reports a problem.
        if current_val is None:
            for guess in ("latest_classification", "item"):
                if ctx.get(guess) is not None:
                    current_val = ctx.get(guess)
                    logger.warning(
                        f"Aggregator {node_config.id}: '{source_field}' held nothing, so "
                        f"'{guess}' was collected instead. Set Field to Aggregate to the key "
                        f"the previous node actually writes."
                    )
                    break

        # A raw file item carries no extracted content, so prefer whatever the
        # reading step produced — again, announced rather than assumed.
        if current_val is None or (isinstance(current_val, dict) and "file_path" in current_val and "item_type" in current_val and "class" not in current_val):
            for guess in ("vision_result", "latest_ocr_output"):
                if ctx.get(guess):
                    current_val = ctx.get(guess)
                    logger.warning(
                        f"Aggregator {node_config.id}: '{source_field}' did not hold extracted "
                        f"content, so '{guess}' was collected instead."
                    )
                    break

        if current_val is None:
            logger.warning(
                f"Aggregator {node_config.id}: nothing to collect this pass — "
                f"'{source_field}' is absent and no fallback matched. Available keys: "
                f"{sorted(k for k in ctx if not k.startswith('_'))[:12]}"
            )

        # 3. Handle persistent storage key (with process + Redis caching)
        storage_key = f"agg_{node_id_to_key(node_config.id)}_bucket"
        bucket = await _get_bucket(thread_id, node_config.id, storage_key, ctx)
        
        # Copy to avoid mutating shared state unexpectedly
        bucket = list(bucket)
        
        # 4. Perform Aggregation Action
        group_by = getattr(node_config, "group_by", None)
        if current_val is not None:
            if action in ["append", "group_by"] or group_by:
                bucket.append(current_val)
            elif action == "merge" and isinstance(current_val, dict):
                if not bucket: bucket = [{}]
                bucket[0].update(current_val)
            elif action == "concatenate":
                separator = getattr(node_config, "separator", "\\n") or "\\n"
                separator = separator.replace("\\n", "\n").replace("\\t", "\t")
                if not bucket: bucket = [""]
                bucket[0] = (bucket[0] + separator + str(current_val)).strip()
        
        # 5. Detect Completion
        is_last = loop_metadata.get("is_last", False)
        
        # 6. Build Context Update
        updates = {}
        
        if is_last:
            logger.info(f"Aggregator {node_config.id}: Loop finished. Releasing {len(bucket)} items to '{dest_field}'.")
            
            if action == "group_by" or group_by:
                key_field = group_by if isinstance(group_by, str) else "class"
                grouped = {}
                for item in bucket:
                    item_key = "Unclassified"
                    if isinstance(item, dict):
                        item_key = item.get(key_field) or item.get("class") or item.get("document_type") or "Unclassified"
                    if item_key not in grouped:
                        grouped[item_key] = []
                    grouped[item_key].append(item)
                final_result = grouped
            else:
                final_result = bucket
                if action in ["merge", "concatenate"] and bucket:
                    final_result = bucket[0]
            
            updates[dest_field] = final_result
            # Clear ephemeral storage cache
            await _clear_bucket(thread_id, node_config.id)
            # Clear checkpoint context database bucket
            updates[storage_key] = []
        else:
            logger.debug(f"Aggregator {node_config.id}: Collected item {loop_metadata.get('index', 0)+1}. Waiting for more.")
            await _save_bucket(thread_id, node_config.id, bucket)
            # Optimization: keep checkpoint context empty until completion
            updates[storage_key] = []

        return {
            "intent": "done" if is_last else "loop",
            "active_agent": node_config.id,
            "context": {**ctx, **updates}
        }

    executor = RunnableLambda(execute_aggregator)
    setattr(executor, "name", node_config.id)
    return executor
