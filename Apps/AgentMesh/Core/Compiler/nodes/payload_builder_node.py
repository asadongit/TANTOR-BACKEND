"""
Payload Builder Node
A generic node that builds a complex JSON payload from context templates.
This node is used to map dynamic variables from AgentMesh context
into a structured JSON document, often used right before an API Webhook node.
"""

import json
import logging
from typing import Callable, Any, Dict

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from .utils import get_nested_val, interpolate

logger = logging.getLogger(__name__)

def _resolve_template(template: Any, context: Dict[str, Any]) -> Any:
    """
    Recursively resolve {{variable}} placeholders in a document template.
    Uses centralized secure interpolate() for string rendering.
    Unresolvable variables become None (null in JSON) for single-variable references.
    """
    if isinstance(template, str):
        if "{{" not in template:
            return template

        import re
        # Single variable check: {{xxx}} -> preserve raw type if resolved
        single_var = re.fullmatch(r"\{\{([\w. ]+)\}\}", template.strip())
        if single_var:
            var_name = single_var.group(1).strip()
            if var_name.startswith("context."):
                var_name = var_name[8:]
            val = get_nested_val(context, var_name)
            return val

        # Multi-variable or inline text -> use central sandboxed interpolate()
        return interpolate(template, context)

    elif isinstance(template, dict):
        resolved_dict = {}
        for k, v in template.items():
            resolved_key = _resolve_template(k, context) if isinstance(k, str) and "{{" in k else k
            if not isinstance(resolved_key, (str, int, float, bool)):
                resolved_key = str(resolved_key) if resolved_key is not None else k
            resolved_dict[resolved_key] = _resolve_template(v, context)
        return resolved_dict

    elif isinstance(template, list):
        return [_resolve_template(item, context) for item in template]

    return template

def build_payload_builder_node(node_config: NodeConfig) -> Callable:
    """
    Build a Payload Builder execution node.
    """
    node_id = node_config.id
    payload_template = getattr(node_config, "payload_template", None)
    destination_key = getattr(node_config, "payload_destination_key", "webhook_payload") or "webhook_payload"

    async def execute_payload_builder(state: AgentState):
        ctx = state.get("context", {})

        if not payload_template:
            logger.warning(f"[PAYLOAD_BUILDER] Node {node_id}: No payload_template provided. Skipping.")
            return {
                "active_agent": node_id,
                "context": ctx
            }

        logger.info(f"[PAYLOAD_BUILDER] Node {node_id}: Building payload to '{destination_key}'")

        try:
            # ── Auto-collect structured agent outputs ──
            # Gathers keys starting with auto_collect_prefix into a single dict 
            # so templates can reference {{all_agent_outputs}}.
            # Default to cd_output_ for backward compatibility with older workflows.
            prefix = getattr(node_config, "auto_collect_prefix", None) or "cd_output_"
            agent_outputs = {k: v for k, v in ctx.items()
                            if k.startswith(prefix) and isinstance(v, dict)}
            if agent_outputs:
                ctx["all_agent_outputs"] = agent_outputs
                logger.info(f"[PAYLOAD_BUILDER] Auto-collected {len(agent_outputs)} agent outputs with prefix '{prefix}'")

            # Strip binary fields from large data arrays to prevent multi-MB payloads
            STRIP_KEYS = {"img_base64", "file_path", "image_base64", "base64", "image_data"}
            
            # Allow configurable data keys to clean (defaults to common OCR/document keys)
            data_keys_to_clean = getattr(node_config, "clean_binary_from", None) or ["latest_ocr_output", "ocr_like_results"]
            if isinstance(data_keys_to_clean, str):
                data_keys_to_clean = [k.strip() for k in data_keys_to_clean.split(",") if k.strip()]

            for key in data_keys_to_clean:
                target_data = ctx.get(key)
                if isinstance(target_data, list):
                    cleaned_data = [
                        {k: v for k, v in page.items() if k not in STRIP_KEYS}
                        if isinstance(page, dict) else page
                        for page in target_data
                    ]
                    ctx[key] = cleaned_data
                    logger.info(f"[PAYLOAD_BUILDER] Stripped binary fields from {len(target_data)} items in '{key}'")
            
            # Note: We no longer auto-derive workflow-specific fields like 'customer_folder_name' or 'cd_customer_id'.
            # The Payload Builder is completely generic. All custom payload structures should be 
            # defined via {{context.variable}} interpolation within the Payload Template itself.

            # Interpolate the template with workflow context
            if isinstance(payload_template, str):
                try:
                    parsed_template = json.loads(payload_template)
                    built_payload = _resolve_template(parsed_template, ctx)
                except json.JSONDecodeError:
                    built_payload = _resolve_template(payload_template, ctx)
            else:
                built_payload = _resolve_template(payload_template, ctx)

            # Store in context
            ctx[destination_key] = built_payload
            
            # ── Auto-inject into batch_vectors if present ──
            if "batch_vectors" in ctx and isinstance(ctx["batch_vectors"], list):
                for item in ctx["batch_vectors"]:
                    # Inject the built payload, but preserve existing image data, types, and reference_ids
                    item_type = item.get("type", "unknown")
                    existing_payload = item.get("payload", {})
                    
                    # Merge existing payload with the new built payload
                    item_payload = {**existing_payload, **(built_payload if isinstance(built_payload, dict) else {})}
                    item_payload["type"] = item_type
                    item["payload"] = item_payload
                logger.info(f"[PAYLOAD_BUILDER] Auto-injected payload into {len(ctx['batch_vectors'])} batch_vectors.")
            
            ocr_len = len(built_payload.get("ocr_data", [])) if isinstance(built_payload, dict) and built_payload.get("ocr_data") else 0
            has_latest = "latest_ocr_output" in ctx
            summary = f"Successfully built JSON payload. latest_ocr_output present: {has_latest}. ocr_data len: {ocr_len}"
            logger.info(f"[PAYLOAD_BUILDER] Node {node_id}: {summary}")

            return {
                "active_agent": node_id,
                "context": ctx
            }

        except Exception as e:
            error_msg = f"Payload Builder Node {node_id} failed: {str(e)}"
            logger.exception(error_msg)
            return {
                "messages": [AIMessage(content=f"[ERROR] {error_msg}")],
                "active_agent": node_id,
                "error": {"message": str(e), "type": type(e).__name__},
                "error_source": node_id
            }

    executor = RunnableLambda(execute_payload_builder)
    setattr(executor, "name", node_id)
    return executor
