"""
PII Redaction Node (Enterprise)
Generic, reusable privacy filter for any AgentMesh workflow.

Architecture:
  Engine (HOW to mask) is decoupled from Target (WHAT to mask).
  Any engine can be combined with any target for maximum flexibility.

Engines:
  - local       : DataFog library (fast, destructive, no vault)
  - docbot      : External DocBot API (enterprise, vaulted, reversible)

Targets:
  - latest_message         : Mask the last message in the conversation
  - context_variable       : Mask a specific context variable (JSON-safe)
  - targeted_replacement   : Precision mask known values in a target text
                             (uses upstream identifiers, zero false positives)
"""

import json
import logging
import re
from typing import Callable, Any, Dict, List, Optional, Tuple

import httpx
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Regex Layer (pre-processing, works with any engine)
# ============================================================================

def _apply_custom_regex(text: str, patterns: List[Dict[str, str]]) -> str:
    """
    Apply user-defined regex patterns as a pre-processing step.
    Runs BEFORE the main engine, so the engine never sees matching data.
    """
    for item in patterns:
        label = item.get("label", "REDACTED")
        pattern = item.get("pattern")
        if pattern:
            try:
                text = re.sub(pattern, f"[{label}]", text)
            except re.error as e:
                logger.error(f"PII Node: Invalid regex pattern '{pattern}': {e}")
    return text


# ============================================================================
# Engine: Local (DataFog) — Destructive, no vault
# ============================================================================

def _mask_text_local(text: str, datafog_engine: str = "regex") -> str:
    """Mask PII using DataFog library. Destructive — original data is lost."""
    try:
        import datafog
        return datafog.sanitize(text, engine=datafog_engine)
    except Exception as e:
        logger.warning(f"PII Node: DataFog {datafog_engine} failed: {e}. Returning original text.")
        return text


def _mask_value_local(val: Any, datafog_engine: str = "regex",
                      custom_patterns: List[Dict[str, str]] = None) -> Any:
    """
    Recursively mask PII in a value (string, dict, or list).
    Preserves JSON structure — only string values are masked.
    """
    if isinstance(val, str):
        masked = _mask_text_local(val, datafog_engine)
        if custom_patterns:
            masked = _apply_custom_regex(masked, custom_patterns)
        return masked
    elif isinstance(val, dict):
        return {k: _mask_value_local(v, datafog_engine, custom_patterns) for k, v in val.items()}
    elif isinstance(val, list):
        return [_mask_value_local(item, datafog_engine, custom_patterns) for item in val]
    return val


# ============================================================================
# Engine: DocBot (External API) — Enterprise, vaulted, reversible
# ============================================================================

async def _mask_text_docbot(text: str, docbot_url: str,
                            method: str = "local") -> Dict[str, Any]:
    """
    Call DocBot's /pii/mask endpoint.
    
    DocBot detects PII using its own internal regex + NLP engines,
    encrypts the original values, stores them in a secure vault,
    and returns reversible tokens like [PAN_a3f82b1c].
    
    Args:
        text: The raw text to scan and mask.
        docbot_url: Base URL of the DocBot service (e.g. http://localhost:8000).
        method: DocBot detection method — 'local' (regex) or 'hybrid' (regex + LLM).
    
    Returns:
        Dict with 'masked_text', 'vault_token', and 'mask_ids'.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {
            "text": text,
            "method": method,
            "save_to_vault": True
        }
        
        # User requested: fully accept url sent from config of pii node (ui) and do not add anything to it.
        response = await client.post(docbot_url, json=payload)
        response.raise_for_status()
        return response.json()


async def _mask_value_docbot(val: Any, docbot_url: str, method: str = "local",
                             custom_patterns: List[Dict[str, str]] = None) -> Any:
    """
    Recursively mask PII in a value using DocBot.
    Only string values are sent to DocBot; JSON structure is preserved.
    """
    if isinstance(val, str):
        if custom_patterns:
            val = _apply_custom_regex(val, custom_patterns)
        try:
            result = await _mask_text_docbot(val, docbot_url, method)
            return result.get("masked_text", val)
        except Exception as e:
            logger.error(f"PII Node: DocBot masking failed for string value: {e}")
            return val
    elif isinstance(val, dict):
        masked = {}
        for k, v in val.items():
            masked[k] = await _mask_value_docbot(v, docbot_url, method, custom_patterns)
        return masked
    elif isinstance(val, list):
        return [await _mask_value_docbot(item, docbot_url, method, custom_patterns) for item in val]
    return val


# ============================================================================
# Unified Masking Dispatcher
# ============================================================================

async def _mask_text(text: str, engine: str, custom_patterns: List = None,
                     datafog_engine: str = "regex",
                     docbot_url: str = "", docbot_method: str = "local") -> str:
    """
    Unified text masking: applies custom regex first, then dispatches
    to the selected engine.
    """
    if custom_patterns:
        text = _apply_custom_regex(text, custom_patterns)

    if engine == "docbot":
        try:
            result = await _mask_text_docbot(text, docbot_url, docbot_method)
            return result.get("masked_text", text)
        except Exception as e:
            logger.error(f"PII Node: DocBot engine failed: {e}. Returning regex-cleaned text.")
            return text
    else:
        # Local engine (DataFog)
        return _mask_text_local(text, datafog_engine)


# ============================================================================
# Node Builder
# ============================================================================

def build_pii_node(node_config: NodeConfig) -> Callable:
    """
    Build an enterprise PII Redaction node.
    
    Engine (HOW — who does the masking):
        - local   : DataFog library (fast, destructive, no vault)
        - docbot  : External DocBot API (enterprise, vaulted, reversible)
    
    Target (WHAT — which data to mask):
        - latest_message       : The last message in the conversation
        - context_variable     : A specific key in workflow context (JSON-safe)
        - targeted_replacement : Precision find-and-replace of known identifiers
                                 in a rendered text field (zero false positives)
    
    Properties (from node_config):
        - pii_engine            : "local" | "docbot"
        - pii_target            : "latest_message" | "context_variable" | "targeted_replacement"
        - pii_custom_regex      : list of {label, pattern} for pre-processing
        - datafog_engine        : "regex" | "gliner" (only for local engine)
        - docbot_url            : DocBot API base URL (only for docbot engine)
        - docbot_method         : "local" | "hybrid" (only for docbot engine)
        - context_variable_key  : which context key to mask (for context_variable target)
        - identifiers_input_key : context key with identifiers (for targeted_replacement)
        - rendered_input_key    : context key with text to mask (for targeted_replacement)
    """

    async def execute_pii(state: AgentState):
        # ── Read configuration ──
        engine = getattr(node_config, "pii_engine", "local") or "local"
        target = getattr(node_config, "pii_target", "latest_message") or "latest_message"
        custom_patterns = getattr(node_config, "pii_custom_regex", []) or []

        # Local engine settings
        datafog_engine = getattr(node_config, "datafog_engine", "regex") or "regex"

        # DocBot engine settings
        docbot_url = getattr(node_config, "docbot_url", "http://localhost:8000") or "http://localhost:8000"
        docbot_method = getattr(node_config, "docbot_method", "local") or "local"

        # Target-specific keys
        context_variable_key = getattr(node_config, "context_variable_key", "") or ""
        identifiers_input_key = getattr(node_config, "identifiers_input_key", "extracted_identifiers")
        rendered_input_key = getattr(node_config, "rendered_input_key", "ocr_like_results")

        node_id = node_config.id

        logger.info(
            f"[PII] Node {node_id}: Engine={engine}, Target={target}, "
            f"Custom Patterns={len(custom_patterns)}"
        )

        updates: Dict[str, Any] = {"active_agent": node_id}

        # ════════════════════════════════════════════════════════════
        # Target: Latest Message
        # ════════════════════════════════════════════════════════════
        if target == "latest_message":
            messages = state.get("messages", [])
            if not messages:
                logger.warning(f"[PII] Node {node_id}: No messages to mask.")
                updates["messages"] = [AIMessage(
                    content="PII Node: No messages found to mask.",
                    name=f"PII_{node_id}"
                )]
                return updates

            last_msg = messages[-1]
            original_content = last_msg.content

            # Try to parse as JSON to preserve structure
            try:
                data = json.loads(original_content)
                if engine == "docbot":
                    masked_data = await _mask_value_docbot(
                        data, docbot_url, docbot_method, custom_patterns
                    )
                else:
                    masked_data = _mask_value_local(data, datafog_engine, custom_patterns)
                new_content = json.dumps(masked_data, indent=2)
            except (json.JSONDecodeError, TypeError):
                # Plain text masking
                new_content = await _mask_text(
                    original_content, engine, custom_patterns,
                    datafog_engine, docbot_url, docbot_method
                )

            updates["messages"] = [AIMessage(content=new_content, name="PII_Filter")]

        # ════════════════════════════════════════════════════════════
        # Target: Context Variable
        # ════════════════════════════════════════════════════════════
        elif target == "context_variable":
            ctx = state.get("context", {})

            if context_variable_key:
                # Mask a specific key
                target_val = ctx.get(context_variable_key)
                if target_val is None:
                    logger.warning(
                        f"[PII] Node {node_id}: Context key '{context_variable_key}' not found."
                    )
                    updates["messages"] = [AIMessage(
                        content=f"PII Node: Key '{context_variable_key}' not found in context.",
                        name=f"PII_{node_id}"
                    )]
                    return updates

                if engine == "docbot":
                    masked_val = await _mask_value_docbot(
                        target_val, docbot_url, docbot_method, custom_patterns
                    )
                else:
                    masked_val = _mask_value_local(target_val, datafog_engine, custom_patterns)

                updates["context"] = {**ctx, context_variable_key: masked_val}
            else:
                # Mask entire context
                if engine == "docbot":
                    masked_ctx = await _mask_value_docbot(
                        ctx, docbot_url, docbot_method, custom_patterns
                    )
                else:
                    masked_ctx = _mask_value_local(ctx, datafog_engine, custom_patterns)
                updates["context"] = masked_ctx

            updates["messages"] = [AIMessage(
                content=f"PII Node: Sanitized context data ({engine}).",
                name=f"PII_{node_id}"
            )]

        # ════════════════════════════════════════════════════════════
        # Target: Targeted Replacement (Precision Mode)
        # ════════════════════════════════════════════════════════════
        elif target == "targeted_replacement":
            ctx = state.get("context", {})
            identifiers = ctx.get(identifiers_input_key, [])
            ocr_results = ctx.get(rendered_input_key, [])

            if not identifiers:
                logger.warning(
                    f"[PII] Node {node_id}: No identifiers found at "
                    f"'{identifiers_input_key}' for targeted replacement. Passing through unmasked data."
                )
                updates["context"] = {
                    **ctx,
                    "latest_ocr_output": ocr_results
                }
                updates["messages"] = [AIMessage(
                    content="PII Node: No identifiers found for targeted masking. Passed through unmasked data.",
                    name=f"PII_{node_id}"
                )]
                return updates

            if not ocr_results:
                logger.warning(
                    f"[PII] Node {node_id}: No rendered data found at "
                    f"'{rendered_input_key}' for targeted replacement."
                )
                updates["messages"] = [AIMessage(
                    content="PII Node: No rendered text found for targeted masking.",
                    name=f"PII_{node_id}"
                )]
                return updates

            # Build replacement map from known identifiers
            replacement_map: Dict[str, str] = {}
            masked_count = 0

            for ident in identifiers:
                val = ident.get("value")
                if not val or len(str(val)) < 3:
                    continue

                original_str = str(val)

                try:
                    if engine == "docbot":
                        docbot_res = await _mask_text_docbot(
                            original_str, docbot_url, docbot_method
                        )
                        masked_str = docbot_res.get("masked_text", original_str)
                    else:
                        masked_str = _mask_text_local(original_str, datafog_engine)

                    if masked_str != original_str:
                        replacement_map[original_str] = masked_str
                        ident["masked_value"] = masked_str
                        masked_count += 1
                except Exception as e:
                    label = ident.get("label", ident.get("concept", "unknown"))
                    logger.error(
                        f"[PII] Node {node_id}: Failed to mask identifier "
                        f"'{label}': {e}"
                    )

            # Apply replacements to the rendered AI summaries
            for ocr_res in ocr_results:
                summary = ocr_res.get("ai_summary", "")
                for original_val, masked_val in replacement_map.items():
                    summary = summary.replace(original_val, masked_val)
                ocr_res["ai_summary"] = summary

            updates["context"] = {
                **ctx,
                rendered_input_key: ocr_results,
                identifiers_input_key: identifiers,
                "latest_ocr_output": ocr_results  # Ensure payload_builder can find it for JSON uploads
            }
            updates["messages"] = [AIMessage(
                content=(
                    f"PII Node: Masked {masked_count} sensitive identifiers "
                    f"using {engine} engine."
                ),
                name=f"PII_{node_id}"
            )]

        else:
            logger.error(f"[PII] Node {node_id}: Unknown target '{target}'.")
            updates["messages"] = [AIMessage(
                content=f"PII Node: Unknown target '{target}'.",
                name=f"PII_{node_id}"
            )]
        return updates

    executor = RunnableLambda(execute_pii).with_retry(stop_after_attempt=3)
    setattr(executor, "name", node_config.id)
    return executor
