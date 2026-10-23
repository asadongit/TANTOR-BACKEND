"""
Jev (System One by TypeSafe AI) Node Builder.
Evaluates typed questions (choice, noul, score) against unstructured state or collections of JSON records.
Returns structured, calibrated categorical decisions without generative hallucination.
Supports high-concurrency evaluation and chunk-based streaming.
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

import httpx
from langchain_core.messages import AIMessage
from jinja2.sandbox import SandboxedEnvironment

from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig

logger = logging.getLogger(__name__)

# Live thread progress registry for external UI / dashboard observers
_JEV_PROGRESS_REGISTRY: Dict[str, Dict[str, Any]] = {}

def get_jev_progress(thread_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve live evaluation progress for an active session thread."""
    if not thread_id:
        return None
    return _JEV_PROGRESS_REGISTRY.get(thread_id)

def set_jev_progress(thread_id: str, data: Dict[str, Any]) -> None:
    """Publish live evaluation progress for an active session thread."""
    if thread_id:
        _JEV_PROGRESS_REGISTRY[thread_id] = {
            **data,
            "timestamp": time.time(),
        }

_jinja_env = SandboxedEnvironment()

def _interpolate(template: str, scope: Dict[str, Any]) -> str:
    """Renders a Jinja2 template string safely."""
    try:
        return _jinja_env.from_string(template).render(**scope)
    except Exception:
        return template

DEFAULT_TYPESAFE_API_URL = "https://api.typesafe.ai/v1/systemone"
VALID_QUESTION_TYPES = {"choice", "noul", "score"}


def _norm_key(k: Any) -> str:
    """Normalize a key string by removing non-alphanumeric characters and lowercasing."""
    return re.sub(r"[^a-zA-Z0-9]", "", str(k).lower())


def _format_item_state(item: Any, fields: Optional[List[str]], template: Optional[str]) -> str:
    """Formats a single item/record into an informative state string for Jev."""
    if not isinstance(item, dict):
        return str(item)

    if template and template.strip():
        try:
            return _interpolate(template, {**item, "item": item})
        except Exception:
            pass

    clean_fields = [f.strip() for f in fields if f and str(f).strip()] if fields else []
    if clean_fields:
        parts = []
        for f in clean_fields:
            nf = _norm_key(f)
            val = None
            f_name = f
            for k, v in item.items():
                if _norm_key(k) == nf:
                    val = v
                    f_name = k
                    break
            if val is not None and str(val).strip() and str(val).lower() != "nan":
                parts.append(f"{f_name}: {val}")
        if parts:
            return " | ".join(parts)

    # General security finding priority attributes across scanners
    priority_keys = [
        "vulnerability_name", "plugin_name", "vulnerability", "title", "name", "rule",
        "affected_package", "package", "component", "dependency", "software",
        "affected_file", "path", "file", "target", "binary",
        "related_cve", "cve", "cve_id", "related_cve_alias", "cpe", "cpe_id", "cwe_id",
        "cwe_description", "description", "synopsis", "summary", "details",
        "plugin_output", "plugin_family", "family", "application_patch", "application_name",
        "os", "os_flavour", "os_falvour", "port", "protocol"
    ]
    parts = []
    seen_keys = set()
    for pk in priority_keys:
        npk = _norm_key(pk)
        for k, v in item.items():
            if _norm_key(k) == npk and k not in seen_keys and v is not None and str(v).strip() and str(v).lower() != "nan":
                parts.append(f"{k}: {v}")
                seen_keys.add(k)
                break

    if parts:
        return " | ".join(parts)

    return json.dumps(item, default=str)[:1000]


async def _evaluate_state_with_jev(
    client: httpx.AsyncClient,
    api_url: str,
    api_key: str,
    model: str,
    state_str: str,
    questions: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Call TypeSafe AI Jev endpoint for a given state string and question set.
    Includes validation and automatic exponential backoff retry on HTTP 429 and 529.
    """
    headers = {
        "Content-Type": "application/json",
    }
    clean_key = str(api_key or "").strip()
    if clean_key:
        headers["Authorization"] = f"Bearer {clean_key}"

    # Validate and build questions payload
    payload_questions = {}
    for q_id, q_cfg in questions.items():
        if not isinstance(q_cfg, dict):
            continue
        q_type = str(q_cfg.get("type", "")).strip().lower()
        if q_type not in VALID_QUESTION_TYPES:
            logger.warning(f"Jev Node: Skipping question '{q_id}' with unsupported type '{q_type}'")
            continue

        qd = dict(q_cfg)
        qd["type"] = q_type
        payload_questions[q_id] = qd

    if not payload_questions:
        raise ValueError("No valid questions (choice, noul, score) provided for Jev evaluation.")

    payload = {
        "model": model,
        "state": state_str,
        "questions": payload_questions,
    }

    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            response = await client.post(api_url, json=payload, headers=headers)
            if response.status_code in (429, 529):
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Jev API returned status {response.status_code}. Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                await asyncio.sleep(delay)
                continue

            response.raise_for_status()
            data = response.json()
            return data.get("answers", {})

        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Jev API network error ({exc}). Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
            await asyncio.sleep(delay)

    raise RuntimeError(f"All {max_retries} retries exhausted for Jev API call.")


def _extract_answers_from_response(
    answers: Dict[str, Any],
    questions: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Dynamically extract typed answers and confidence scores from TypeSafe response.
    Embeds flat, clean fields using the exact canvas question ID as the key name.
    """
    extracted = {}
    for q_id, q_cfg in questions.items():
        ans = answers.get(q_id)
        if not ans or not isinstance(ans, dict):
            continue

        q_type = ans.get("type") or q_cfg.get("type")
        if q_type == "choice":
            choice_val = ans.get("choice")
            extracted[q_id] = choice_val if choice_val is not None else "Unclassified"
            if q_id == "vulnerability_category" and "category" not in questions:
                extracted["category"] = extracted[q_id]
            if "confidence" in ans and ans["confidence"] is not None:
                try:
                    conf_f = float(ans["confidence"])
                    extracted[f"{q_id}_confidence"] = conf_f
                    if q_id in ("category", "vulnerability_category"):
                        extracted["confidence"] = conf_f
                        extracted["category_confidence"] = conf_f
                except (ValueError, TypeError):
                    pass

        elif q_type == "noul":
            if "noul" in ans and ans["noul"] is not None:
                try:
                    extracted[q_id] = float(ans["noul"])
                except (ValueError, TypeError):
                    extracted[q_id] = ans.get("noul")

        elif q_type == "score":
            if "score" in ans and ans["score"] is not None:
                try:
                    extracted[q_id] = float(ans["score"])
                except (ValueError, TypeError):
                    extracted[q_id] = ans.get("score")
            if "confidence" in ans and ans["confidence"] is not None:
                try:
                    extracted[f"{q_id}_confidence"] = float(ans["confidence"])
                except (ValueError, TypeError):
                    pass

    return extracted


def build_jev_node(node_config: NodeConfig) -> Callable:
    """
    Build a Jev (System One by TypeSafe AI) node runner.
    Takes data from context (single item or list of records), evaluates configured questions
    via TypeSafe AI Jev, tags rows with the results, and saves the structured dataset.
    """
    model = getattr(node_config, "jev_model", "jev-latest") or "jev-latest"
    api_key_env = getattr(node_config, "jev_api_key_env", "TYPESAFE_API_KEY") or "TYPESAFE_API_KEY"
    api_key_override = getattr(node_config, "jev_api_key", None)
    raw_api_url = getattr(node_config, "jev_api_url", None)
    if not raw_api_url or "chat/completions" in str(raw_api_url).lower() or ("localhost" in str(raw_api_url).lower() and "8000" in str(raw_api_url)):
        api_url = DEFAULT_TYPESAFE_API_URL
    else:
        api_url = str(raw_api_url).strip()

    input_key = getattr(node_config, "jev_input_key", "excel_json_data") or "excel_json_data"
    output_key = getattr(node_config, "jev_output_key", "excel_json_data") or "excel_json_data"
    mode = getattr(node_config, "jev_mode", "items") or "items"
    item_fields = getattr(node_config, "jev_item_fields", None)
    state_template = getattr(node_config, "jev_state_template", None)
    questions = getattr(node_config, "jev_questions", {}) or {}
    timeout_sec = float(getattr(node_config, "jev_timeout_seconds", 15.0) or 15.0)
    concurrency = int(getattr(node_config, "jev_concurrency", 25) or 25)
    batch_progress_size = int(getattr(node_config, "jev_batch_progress_size", 100) or 100)
    stream_progress = bool(getattr(node_config, "jev_stream_progress", False))

    async def jev_node(state: AgentState) -> Dict[str, Any]:
        context: Dict[str, Any] = state.get("context", {}) or {}

        # 1. Guard: Validate questions exist from canvas
        if not questions:
            err = (
                f"Jev Node '{node_config.id}': No evaluation questions configured. "
                f"Configure at least one question (choice/noul/score) in the node settings on the canvas."
            )
            logger.error(err)
            return {
                "context": dict(context),
                "messages": [
                    AIMessage(
                        content=f"⚠️ {err}",
                        name=node_config.id,
                        response_metadata={"active_agent": node_config.id},
                    )
                ],
                "active_agent": node_config.id,
                "error": {"message": err, "type": "ConfigurationError"},
                "error_source": node_config.id,
            }

        # 2. Resolve raw input data from context
        raw_data = context.get(input_key)
        if raw_data is None:
            for fallback_k in ("excel_json_data_path", "excel_json_data", "latest_excel_json", "items", "rows", "records", "data"):
                if fallback_k in context and context[fallback_k] is not None:
                    raw_data = context[fallback_k]
                    logger.debug(f"Jev Node '{node_config.id}': fell back to context key '{fallback_k}'")
                    break

        json_file_path = None
        if isinstance(raw_data, str) and (raw_data.endswith(".json") or os.path.exists(raw_data)):
            json_file_path = raw_data
            try:
                with open(json_file_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
            except Exception as e:
                logger.error(f"Jev Node '{node_config.id}': failed to read JSON file '{json_file_path}': {e}")
                raw_data = []

        if raw_data is None:
            logger.warning(f"Jev Node '{node_config.id}': no data found at input_key '{input_key}'. Returning unchanged state.")
            return {
                "context": context,
                "messages": [
                    AIMessage(
                        content=f"Jev Node '{node_config.id}': No input data found under '{input_key}'.",
                        name=node_config.id,
                        response_metadata={"active_agent": node_config.id},
                    )
                ],
            }

        is_list = isinstance(raw_data, list)
        items_to_process = raw_data if is_list else [raw_data]
        total_items = len(items_to_process)

        if total_items == 0:
            logger.info(f"Jev Node '{node_config.id}': empty dataset to process.")
            return {
                "context": context,
                "messages": [
                    AIMessage(
                        content=f"Jev Node '{node_config.id}': Dataset is empty.",
                        name=node_config.id,
                        response_metadata={"active_agent": node_config.id},
                    )
                ],
            }

        # 3. Resolve API credentials and endpoint
        resolved_api_key = api_key_override or os.environ.get(api_key_env, "")

        # Fallback: if configured env var is empty or legacy OPENAI_API_KEY, check standard TypeSafe keys
        if not resolved_api_key or not resolved_api_key.strip() or api_key_env == "OPENAI_API_KEY":
            for fallback_env in ("TYPESAFE_API_KEY", "JEV_API_KEY", "TYPE_SAFE_API_KEY", "TYPESAFE_KEY"):
                val = os.environ.get(fallback_env, "")
                if val and val.strip():
                    resolved_api_key = val.strip()
                    logger.info(f"Jev Node '{node_config.id}': using TypeSafe API key from fallback environment variable '{fallback_env}'.")
                    break

        use_live_api = bool(resolved_api_key and resolved_api_key.strip())
        effective_api_url = api_url

        if not use_live_api:
            logger.warning(
                f"Jev Node '{node_config.id}': No TypeSafe API key found in env var '{api_key_env}' or TYPESAFE_API_KEY. "
                f"Items will be marked as 'unclassified'."
            )

        # 4. Processing logic with concurrency semaphore and batch progress
        semaphore = asyncio.Semaphore(max(1, concurrency))
        progress_messages: List[AIMessage] = []
        thread_id = str(context.get("thread_id") or "").strip()

        if thread_id and total_items > 0:
            set_jev_progress(thread_id, {
                "status": "evaluating",
                "completed": 0,
                "total": total_items,
                "percent": 0.0,
                "node_id": node_config.id,
            })

        async def process_single_item(item: Any, http_client: httpx.AsyncClient) -> Any:
            state_str = _format_item_state(item, item_fields, state_template)

            eval_result: Dict[str, Any] = {}
            eval_status = "evaluated"
            eval_error = None

            if use_live_api:
                async with semaphore:
                    try:
                        eval_result = await _evaluate_state_with_jev(
                            http_client, effective_api_url, resolved_api_key, model, state_str, questions
                        )
                    except Exception as exc:
                        eval_status = "error"
                        eval_error = str(exc)
                        logger.warning(f"Jev API evaluation failed for item ({exc}). Marked with error status.")
                        eval_result = {}
            else:
                eval_status = "unclassified"
                eval_error = f"API key not found in {api_key_env}"

            extracted = _extract_answers_from_response(eval_result, questions)

            if isinstance(item, dict):
                tagged = dict(item)
                tagged.update(extracted)
                tagged["jev_status"] = eval_status
                if eval_error:
                    tagged["jev_error"] = eval_error
                tagged["jev_evaluations"] = eval_result
                return tagged
            else:
                res = {
                    "original": item,
                    **extracted,
                    "jev_status": eval_status,
                    "jev_evaluations": eval_result,
                }
                if eval_error:
                    res["jev_error"] = eval_error
                return res

        limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)
        tagged_records: List[Any] = []
        completed_count = 0

        async with httpx.AsyncClient(timeout=timeout_sec, limits=limits) as http_client:
            chunk_size = max(1, batch_progress_size)
            for chunk_start in range(0, total_items, chunk_size):
                chunk = items_to_process[chunk_start : chunk_start + chunk_size]
                chunk_results = await asyncio.gather(
                    *[process_single_item(it, http_client) for it in chunk],
                    return_exceptions=True
                )

                clean_chunk = []
                for r in chunk_results:
                    if isinstance(r, Exception):
                        logger.error(f"Jev Node row processing exception: {r}")
                        clean_chunk.append({"jev_status": "error", "jev_error": str(r)})
                    else:
                        clean_chunk.append(r)

                tagged_records.extend(clean_chunk)
                completed_count += len(clean_chunk)

                if thread_id:
                    pct = round((completed_count / total_items) * 100, 1)
                    set_jev_progress(thread_id, {
                        "status": "evaluating",
                        "completed": completed_count,
                        "total": total_items,
                        "percent": pct,
                        "node_id": node_config.id,
                        "records": list(tagged_records),
                    })
                    if stream_progress and total_items > chunk_size:
                        progress_messages.append(
                            AIMessage(
                                content=json.dumps({
                                    "type": "jev_progress",
                                    "node_id": node_config.id,
                                    "completed": completed_count,
                                    "total": total_items,
                                    "percent": pct,
                                }),
                                name=node_config.id,
                            )
                        )

        if thread_id and total_items > 0:
            set_jev_progress(thread_id, {
                "status": "completed",
                "completed": total_items,
                "total": total_items,
                "percent": 100.0,
                "node_id": node_config.id,
                "records": list(tagged_records),
            })

        # 5. Compile summary statistics
        summary_payload: Dict[str, Any] = {
            "total_items": total_items,
            "model": model,
            "live_api": use_live_api,
            "questions": list(questions.keys()),
        }

        # Tally metrics for choice questions (e.g. category)
        breakdowns: Dict[str, Dict[str, int]] = {}
        for q_id, q_cfg in questions.items():
            if q_cfg.get("type") == "choice":
                counts: Dict[str, int] = {}
                for rec in tagged_records:
                    val = str(rec.get(q_id, "Unclassified"))
                    counts[val] = counts.get(val, 0) + 1
                breakdowns[q_id] = counts

        summary_payload["breakdowns"] = breakdowns

        # Format summary message
        summary_lines = [
            f"**Jev Evaluation Completed**",
            f"• **Total Items Evaluated:** {total_items}",
            f"• **Model:** {model}",
        ]
        for q_id, counts in breakdowns.items():
            dist_str = ", ".join(f"{k}: {v}" for k, v in counts.items())
            summary_lines.append(f"• **{q_id} Breakdown:** {dist_str}")

        summary_msg = "\n".join(summary_lines)

        final_output = tagged_records if is_list else (tagged_records[0] if tagged_records else {})

        # 6. Save tagged JSON to disk if input was a file path
        output_context_val = final_output
        tagged_path = None
        if json_file_path:
            tagged_path = json_file_path
            if not json_file_path.endswith("_tagged.json"):
                tagged_path = json_file_path.replace(".json", "_tagged.json")
            try:
                with open(tagged_path, "w", encoding="utf-8") as f:
                    json.dump(final_output, f, indent=2, default=str)
                logger.info(f"Jev Node '{node_config.id}': persisted tagged findings to '{tagged_path}'")
                output_context_val = tagged_path
            except Exception as e:
                logger.error(f"Jev Node '{node_config.id}': failed to save tagged findings to '{tagged_path}': {e}")
                output_context_val = tagged_path

        logger.info(f"Jev Node '{node_config.id}' finished successfully: {total_items} items processed.")

        # 7. Build updated context
        new_context = dict(context)
        new_context.pop("context", None)
        new_context.pop("data", None)
        new_context.pop("msg", None)
        new_context.pop("excel_json_data_path", None)
        new_context.pop("latest_excel_json", None)
        new_context.pop("extracted_file_path", None)
        new_context.pop("latest_file_path", None)
        new_context.pop("vapt_validation_preview", None)
        new_context[output_key] = output_context_val
        new_context[f"{output_key}_summary"] = summary_payload
        new_context["jev_summary"] = summary_payload

        all_messages = progress_messages + [
            AIMessage(
                content=summary_msg,
                name=node_config.id,
                response_metadata={"active_agent": node_config.id},
            )
        ]

        return {
            "context": new_context,
            "messages": all_messages,
            "active_agent": node_config.id,
        }

    return jev_node
