"""
Laya (local / on-prem decision model) Node Builder.

Drop-in alternative to the Jev node for air-gapped or on-prem deployments.  Laya
answers the same typed questions (choice, noul, score) over HTTP against a Laya
service reachable by ``laya_endpoint`` -- either a loopback sidecar or a Laya
instance running on another host.

Design notes
------------
* **Same input/output contract as jev_node.**  Rows carry ``jev_status`` and
  ``jev_evaluations`` so a workflow can swap one node for the other on the
  canvas without touching downstream consumers, and the VAPT dashboard keeps
  working unchanged.
* **State building and answer extraction are shared with jev_node**, not
  re-implemented.  That guarantees the two backends produce identical row
  shapes and identical gating semantics; only the transport differs.
* **Tighter context budget.**  The English Laya checkpoint reads about 512
  tokens in total, including the question and its options, so
  ``laya_max_state_chars`` leaves room for the options rather than letting the
  long state push them out.
* Laya returns ``action.act_probability`` on every answer; it is captured as
  ``laya_act_probability`` so a cascade can later escalate low-act rows to Jev.

Status values (identical to jev_node):
  ``classified`` | ``needs_review`` | ``insufficient_state`` | ``error``
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from langchain_core.messages import AIMessage

from .vapt_sanitizer import (
    VaptSanitizer,
    COMPACT_VAPT_QUESTIONS,
    CATEGORY_CANONICAL_MAP,
)

from Apps.AgentMesh.Core.Compiler.nodes.jev_node import (
    PRIORITY_FIELDS,
    _as_float,
    _build_question_payload,
    _extract_answers_from_response,
    _interpolate,
    _is_blank,
    _is_output_column,
    _norm_key,
    _resolve_fields,
    _source_record,
    _write_checkpoint,
    set_jev_progress,
)
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig

logger = logging.getLogger(__name__)

DEFAULT_LAYA_ENDPOINT = "http://127.0.0.1:8770"
LAYA_PREDICT_PATH = "/predict"
LAYA_API_PREDICT_PATH = "/api/predict"

# The English checkpoint reads roughly 512 tokens in total.  Reserve headroom
# for the question instructions and option descriptions, otherwise the tail of
# the options is cut off and the model answers against an incomplete option list.
LAYA_DEFAULT_MAX_STATE_CHARS = 1200


class LayaAPIError(RuntimeError):
    """A non-retryable or exhausted Laya service failure, with upstream detail."""

    def __init__(self, message: str, status_code: Optional[int] = None, detail: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail

    @property
    def is_fatal(self) -> bool:
        """Errors that will repeat identically for every remaining row."""
        return self.status_code in (400, 401, 403, 413, 415, 422)

    @property
    def is_service_unavailable(self) -> bool:
        """A connection-level failure: the service is down or unreachable."""
        return self.status_code is None


_RETRYABLE_STATUS = {429, 500, 502, 503, 504, 529}


def _normalise_endpoint(endpoint: str) -> str:
    """Return a predict URL for either a base URL or a full /predict URL."""
    raw = str(endpoint or "").strip() or DEFAULT_LAYA_ENDPOINT
    if not raw.startswith(("http://", "https://")):
        raw = "http://" + raw
    raw = raw.rstrip("/")
    if raw.endswith(LAYA_PREDICT_PATH) or raw.endswith(LAYA_API_PREDICT_PATH):
        return raw
    return raw + LAYA_PREDICT_PATH


def _error_detail(response: "httpx.Response") -> Any:
    """Extract the service's error body, bounded, for logging."""
    try:
        body = response.json()
    except Exception:
        body = None
    if isinstance(body, dict) and "detail" in body:
        detail = body["detail"]
    else:
        detail = body if body is not None else response.text
    text = str(detail)
    return text[:500] + ("…" if len(text) > 500 else "")


async def _predict_with_laya(
    client: httpx.AsyncClient,
    predict_url: str,
    api_token: str,
    state: Any,
    questions: Dict[str, Dict[str, Any]],
    lang: Optional[str] = "en",
    max_retries: int = 3,
    base_delay: float = 1.0,
    service_down: Optional[Dict[str, bool]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Call the Laya ``/predict`` endpoint for one state.

    Returns ``(answers, meta)`` where meta carries ``usage``, ``latency_ms`` and
    ``version`` as reported by the service.

    Retries on 429/5xx and on transport faults.  A connection-level failure is
    raised as a LayaAPIError with ``status_code=None`` so the node can tell
    "service is down" apart from "this row was rejected".
    """
    headers = {"Content-Type": "application/json"}
    clean_token = str(api_token or "").strip()
    if clean_token:
        headers["Authorization"] = f"Bearer {clean_token}"

    payload: Dict[str, Any] = {"state": state, "questions": questions}
    if lang:
        payload["lang"] = lang

    attempts = max(1, int(max_retries))
    last_exc: Optional[BaseException] = None

    for attempt in range(attempts):
        # Another in-flight row may have already proven the service unreachable.
        # Fail immediately rather than repeating the whole retry ladder.
        if service_down and service_down.get("flag"):
            raise LayaAPIError(
                "Laya service unavailable: skipped, service already proven unreachable",
                status_code=None,
            )
        try:
            response = await client.post(predict_url, json=payload, headers=headers)

            if response.status_code in _RETRYABLE_STATUS:
                detail = _error_detail(response)
                if attempt == attempts - 1:
                    raise LayaAPIError(
                        f"Laya service returned {response.status_code} after {attempts} attempts: {detail}",
                        status_code=response.status_code,
                        detail=detail,
                    )
                retry_after = response.headers.get("retry-after")
                delay = base_delay * (2 ** attempt)
                try:
                    if retry_after:
                        delay = max(delay, float(retry_after))
                except (ValueError, TypeError):
                    pass
                logger.warning(
                    f"Laya service returned {response.status_code}. Retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{attempts})..."
                )
                await asyncio.sleep(min(delay, 60.0))
                continue

            if response.status_code >= 400:
                detail = _error_detail(response)
                raise LayaAPIError(
                    f"Laya service rejected the request ({response.status_code}): {detail}",
                    status_code=response.status_code,
                    detail=detail,
                )

            data = response.json()
            if not isinstance(data, dict) or "answers" not in data:
                raise LayaAPIError(
                    "Laya service returned a response without an 'answers' field.",
                    status_code=response.status_code,
                    detail=str(data)[:500],
                )

            meta = {
                "usage": data.get("usage", {}) or {},
                "latency_ms": data.get("latency_ms"),
                "version": data.get("version"),
                "model": data.get("model"),
            }
            return data.get("answers", {}) or {}, meta

        except LayaAPIError:
            raise
        except (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.ProtocolError,
            httpx.PoolTimeout,
        ) as exc:
            last_exc = exc
            if attempt == attempts - 1:
                break
            delay = min(base_delay * (2 ** attempt), 60.0)
            logger.warning(
                f"Laya transport error ({type(exc).__name__}: {exc}). "
                f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{attempts})..."
            )
            await asyncio.sleep(delay)

    if service_down is not None:
        service_down["flag"] = True
    raise LayaAPIError(
        f"Laya service unreachable after {attempts} attempts: {type(last_exc).__name__}: {last_exc}",
        status_code=None,
        detail=str(last_exc),
    )


def _prepare_laya_questions(
    questions: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Validate questions for Laya and report ones whose criteria lack descriptions.

    Laya matches the state against the *option descriptions*, so a bare list of
    option names classifies noticeably worse than a described mapping.  This does
    not change the payload -- it surfaces the weakness in the log so it can be
    fixed on the canvas rather than silently degrading accuracy.
    """
    payload_questions = _build_question_payload(questions)
    undescribed: List[str] = []
    for q_id, q_cfg in payload_questions.items():
        if str(q_cfg.get("type")).lower() != "choice":
            continue
        criteria = q_cfg.get("criteria")
        if isinstance(criteria, list):
            undescribed.append(q_id)
        elif isinstance(criteria, dict):
            if all(not str(v).strip() for v in criteria.values()):
                undescribed.append(q_id)
    return payload_questions, undescribed


def _build_laya_item_state(
    item: Any,
    fields: Optional[List[str]],
    template: Optional[str] = None,
    max_chars: Optional[int] = None,
    tag_target_key: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build the Laya state for one record in a strictly generic, schema-agnostic manner.
    Works for any dataset (VAPT scans from any scanner, tickets, logs, fraud, generic JSON).
    """
    meta: Dict[str, Any] = {"fields_sent": [], "fields_missing": [], "usable": True}

    if not isinstance(item, dict):
        text = str(item).strip()
        if max_chars and len(text) > max_chars:
            text = text[:max_chars]
        meta["fields_sent"] = ["<scalar>"]
        meta["usable"] = bool(text)
        return text, meta

    if template and str(template).strip():
        rendered = _interpolate(template, {**item, "item": item})
        text = str(rendered).strip()
        if max_chars and len(text) > max_chars:
            text = text[:max_chars]
        meta["fields_sent"] = ["<template>"]
        meta["usable"] = bool(text)
        return text, meta

    redirect_cols = {str(tag_target_key).strip()} if tag_target_key else set()
    state: Dict[str, Any] = {}

    if fields:
        chosen, missing = _resolve_fields(item, fields)
        meta["fields_missing"] = missing
        for col in chosen:
            if _is_output_column(col) or col in redirect_cols:
                continue
            val = item.get(col)
            if _is_blank(val):
                continue
            state[col] = str(val).strip()
    else:
        # Schema-agnostic resolution:
        # Prioritize informative columns if present, then all other row fields
        present_keys = [k for k in item.keys() if not _is_output_column(k) and k not in redirect_cols]
        seen = set()
        ordered_keys = []
        norm_map = {_norm_key(k): k for k in present_keys}
        for pk in PRIORITY_FIELDS:
            actual = norm_map.get(_norm_key(pk))
            if actual is not None and actual not in seen:
                ordered_keys.append(actual)
                seen.add(actual)
        for k in present_keys:
            if k not in seen:
                ordered_keys.append(k)
                seen.add(k)

        for col in ordered_keys:
            val = item.get(col)
            if _is_blank(val):
                continue
            state[col] = str(val).strip()

    if not state:
        meta["usable"] = False
        return {}, meta

    if max_chars:
        budget = max_chars
        trimmed: Dict[str, Any] = {}
        for col, text in state.items():
            if budget <= 0:
                break
            if len(text) > budget:
                text = text[:budget]
            trimmed[col] = text
            budget -= len(text)
        state = trimmed

    meta["fields_sent"] = list(state.keys())
    meta["usable"] = bool(state)
    return state, meta


def build_laya_node(node_config: NodeConfig) -> Callable:
    """
    Build a Laya node runner.

    Takes data from context (single item or list of records), evaluates the
    configured questions via a Laya service, tags rows with the results, and
    saves the structured dataset.  Produces the same row shape as the Jev node.
    """
    predict_url = _normalise_endpoint(
        getattr(node_config, "laya_endpoint", DEFAULT_LAYA_ENDPOINT) or DEFAULT_LAYA_ENDPOINT
    )

    api_token_env = getattr(node_config, "laya_api_token_env", "LAYA_API_TOKEN") or "LAYA_API_TOKEN"
    api_token_override = getattr(node_config, "laya_api_token", None) or getattr(node_config, "laya_api_key", None)
    lang = getattr(node_config, "laya_lang", "en")

    input_key = getattr(node_config, "jev_input_key", "excel_json_data") or "excel_json_data"
    output_key = getattr(node_config, "jev_output_key", "excel_json_data") or "excel_json_data"
    item_fields = getattr(node_config, "jev_item_fields", None)
    state_template = getattr(node_config, "jev_state_template", None)
    raw_questions = getattr(node_config, "laya_questions", None) or getattr(node_config, "jev_questions", None) or {}
    if not raw_questions or "insufficient_info" in (raw_questions.get("vulnerability_category", {}).get("options", [])):
        questions = COMPACT_VAPT_QUESTIONS
    else:
        questions = raw_questions

    tag_target_key = getattr(node_config, "jev_tag_target_key", None) or None
    tag_key_mode = str(getattr(node_config, "jev_tag_key_mode", "replace") or "replace").strip().lower()
    if tag_key_mode not in ("replace", "preserve"):
        tag_key_mode = "replace"
    def _parse_gate(val: Any, default: float = 0.60) -> float:
        if val in (None, ""):
            return default
        try:
            f = float(val)
            if f > 1.0:
                f = f / 100.0
            return max(0.0, min(1.0, f))
        except (ValueError, TypeError):
            return default

    confidence_gate = _parse_gate(
        getattr(node_config, "jev_confidence_gate", None)
        or getattr(node_config, "confidence_threshold", None)
        or getattr(node_config, "jev_confidence_threshold", None),
        0.60
    )
    top_prob_gate = _parse_gate(
        getattr(node_config, "jev_top_prob_gate", None)
        or getattr(node_config, "top_prob_threshold", None)
        or getattr(node_config, "jev_top_prob_threshold", None)
        or getattr(node_config, "top_probability_threshold", None),
        0.60
    )
    raw_margin_gate = getattr(node_config, "jev_margin_gate", None)
    margin_gate = _as_float(raw_margin_gate) if raw_margin_gate not in (None, "") else None

    raw_concurrency = getattr(node_config, "laya_concurrency", None)
    if raw_concurrency in (None, "", 0, "0"):
        raw_concurrency = getattr(node_config, "jev_concurrency", None)
    try:
        concurrency = int(raw_concurrency) if raw_concurrency not in (None, "") else 10
    except (TypeError, ValueError):
        concurrency = 10
    timeout_sec = float(getattr(node_config, "laya_timeout_seconds", 60.0) or 60.0)
    max_retries = int(getattr(node_config, "laya_max_retries", 3) or 3)
    retry_base_delay = float(getattr(node_config, "laya_retry_base_delay", 1.0) or 1.0)

    batch_progress_size = int(getattr(node_config, "jev_batch_progress_size", None) or getattr(node_config, "laya_batch_progress_size", None) or 10)
    stream_progress = bool(getattr(node_config, "jev_stream_progress", False))
    checkpoint_every_chunk = bool(getattr(node_config, "jev_checkpoint_every_chunk", True))
    abort_on_fatal = bool(getattr(node_config, "laya_abort_on_fatal_error", True))

    raw_max_chars = getattr(node_config, "laya_max_state_chars", LAYA_DEFAULT_MAX_STATE_CHARS)
    try:
        max_state_chars = int(raw_max_chars) if raw_max_chars not in (None, "", 0, "0") else None
    except (TypeError, ValueError):
        max_state_chars = LAYA_DEFAULT_MAX_STATE_CHARS

    async def laya_node(state: AgentState) -> Dict[str, Any]:
        context: Dict[str, Any] = state.get("context", {}) or {}

        # 1. Validate question configuration
        if not questions:
            err = (
                f"Laya Node '{node_config.id}': No evaluation questions configured. "
                f"Configure at least one question (choice/noul/score) in the node settings."
            )
            logger.error(err)
            return {
                "context": dict(context),
                "messages": [AIMessage(content=f"⚠️ {err}", name=node_config.id)],
                "active_agent": node_config.id,
                "error": {"message": err, "type": "ConfigurationError"},
                "error_source": node_config.id,
            }

        try:
            payload_questions, undescribed = _prepare_laya_questions(questions)
        except ValueError as exc:
            err = f"Laya Node '{node_config.id}': {exc}"
            logger.error(err)
            return {
                "context": dict(context),
                "messages": [AIMessage(content=f"⚠️ {err}", name=node_config.id)],
                "active_agent": node_config.id,
                "error": {"message": err, "type": "ConfigurationError"},
                "error_source": node_config.id,
            }

        if undescribed:
            logger.warning(
                f"Laya Node '{node_config.id}': choice question(s) {undescribed} have no option "
                f"descriptions. Laya matches the state against those descriptions, so bare option "
                f"names classify noticeably worse. Describe each option on the canvas."
            )

        # 2. Resolve input
        raw_data = context.get(input_key)
        if raw_data is None:
            for fallback_k in (
                "excel_json_data_path", "excel_json_data", "latest_excel_json",
                "items", "rows", "records", "data",
            ):
                if fallback_k in context and context[fallback_k] is not None:
                    raw_data = context[fallback_k]
                    logger.debug(f"Laya Node '{node_config.id}': fell back to context key '{fallback_k}'")
                    break

        json_file_path = None
        if isinstance(raw_data, str) and (raw_data.endswith(".json") or os.path.exists(raw_data)):
            json_file_path = raw_data
            try:
                with open(json_file_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
            except Exception as e:
                logger.error(f"Laya Node '{node_config.id}': failed to read JSON file '{json_file_path}': {e}")
                raw_data = []

        if raw_data is None:
            logger.warning(
                f"Laya Node '{node_config.id}': no data found at input_key '{input_key}'. "
                f"Returning unchanged state."
            )
            return {
                "context": context,
                "messages": [
                    AIMessage(
                        content=f"Laya Node '{node_config.id}': No input data found under '{input_key}'.",
                        name=node_config.id,
                    )
                ],
            }

        is_list = isinstance(raw_data, list)
        items_to_process = raw_data if is_list else [raw_data]
        total_items = len(items_to_process)

        if total_items == 0:
            logger.info(f"Laya Node '{node_config.id}': empty dataset to process.")
            return {
                "context": context,
                "messages": [
                    AIMessage(content=f"Laya Node '{node_config.id}': Dataset is empty.", name=node_config.id)
                ],
            }

        # 3. Resolve token and report the field set once per run
        resolved_token = str(api_token_override or "").strip()
        if not resolved_token:
            resolved_token = str(os.environ.get(api_token_env, "") or "").strip()
        if not resolved_token:
            resolved_token = str(os.environ.get("LAYA_API_TOKEN", "") or "").strip()

        if not resolved_token:
            logger.info(
                f"Laya Node '{node_config.id}': no token in '{api_token_env}'; calling the service "
                f"unauthenticated. Set a token if the Laya endpoint is network-reachable."
            )

        sample_dict = next((it for it in items_to_process if isinstance(it, dict)), {})
        if sample_dict:
            _, sample_meta = _build_laya_item_state(
                sample_dict,
                item_fields,
                state_template,
                max_chars=max_state_chars,
                tag_target_key=(tag_target_key if tag_key_mode == "replace" else None),
            )
            logger.info(
                f"Laya Node '{node_config.id}': state columns for classification: {sample_meta['fields_sent']} "
                f"(budget {max_state_chars} chars, usable={sample_meta['usable']})"
            )

        # 4. Processing
        semaphore = asyncio.Semaphore(max(1, concurrency))
        progress_messages: List[AIMessage] = []
        thread_id = str(context.get("thread_id") or "").strip()

        checkpoint_path = None
        if json_file_path:
            checkpoint_path = (
                json_file_path
                if json_file_path.endswith("_tagged.json")
                else json_file_path.replace(".json", "_tagged.json")
            )

        tagged_records: List[Any] = []
        completed_count = 0
        status_counts: Dict[str, int] = {}
        total_input_tokens = 0
        latencies_ms: List[float] = []
        service_versions: set = set()
        fatal_error: Optional[LayaAPIError] = None
        aborted = False
        started_at = time.perf_counter()

        # Set once the service has been proven unreachable.  Concurrent rows then
        # fail immediately instead of each running the full retry ladder, which
        # otherwise costs concurrency x attempts before the abort takes effect.
        service_down = {"flag": False}

        def _new_row(item: Any) -> Dict[str, Any]:
            return _source_record(item)

        async def process_single_item(item: Any, http_client: httpx.AsyncClient) -> Dict[str, Any]:
            nonlocal total_input_tokens

            # 1. Deterministic pre-filter: check if row is empty/summary/disclaimer
            if VaptSanitizer.is_unusable_row(item):
                row = _new_row(item)
                row["category"] = "insufficient_info"
                row["category_confidence"] = 1.0
                row["vulnerability_category"] = "insufficient_info"
                row["vulnerability_category_confidence"] = 1.0
                row["vulnerability_category_margin"] = 1.0
                row["jev_status"] = "insufficient_info"
                row["jev_fields_sent"] = []
                row["jev_evaluations"] = {}
                return row

            # 2. Build model state (strip bias columns like IP/Host/Severity, enrich ports)
            if item_fields or state_template:
                state_value, state_meta = _build_laya_item_state(
                    item,
                    item_fields,
                    state_template,
                    max_chars=max_state_chars,
                    tag_target_key=(tag_target_key if tag_key_mode == "replace" else None),
                )
            else:
                clean_state = VaptSanitizer.prepare_model_state(item, max_output_chars=max_state_chars or 400)
                state_value = clean_state
                state_meta = {"fields_sent": list(clean_state.keys()), "usable": bool(clean_state)}

            row = _new_row(item)
            row["jev_fields_sent"] = state_meta["fields_sent"]

            if not state_meta["usable"]:
                row["jev_status"] = "insufficient_info"
                row["category"] = "insufficient_info"
                row["vulnerability_category"] = "insufficient_info"
                row["jev_error"] = (
                    "No usable state fields resolved for this row; "
                    "classification would be a guess."
                )
                row["jev_evaluations"] = {}
                return row

            async with semaphore:
                if service_down["flag"]:
                    row["jev_status"] = "error"
                    row["jev_error"] = "Laya service unavailable: skipped, service already proven unreachable"
                    row["jev_error_kind"] = "service_unavailable"
                    row["jev_evaluations"] = {}
                    if abort_on_fatal:
                        raise LayaAPIError("Laya service unavailable (short-circuited)", status_code=None)
                    return row
                try:
                    answers, meta = await _predict_with_laya(
                        http_client,
                        predict_url,
                        resolved_token,
                        state_value,
                        payload_questions,
                        lang=lang,
                        max_retries=max_retries,
                        base_delay=retry_base_delay,
                        service_down=service_down,
                    )
                except LayaAPIError as exc:
                    row["jev_status"] = "error"
                    row["jev_evaluations"] = {}
                    if exc.is_service_unavailable:
                        service_down["flag"] = True
                        row["jev_error"] = f"Laya service unavailable: {exc}"
                        row["jev_error_kind"] = "service_unavailable"
                    else:
                        row["jev_error"] = str(exc)
                        row["jev_error_status_code"] = exc.status_code
                    if abort_on_fatal and (exc.is_fatal or exc.is_service_unavailable):
                        raise
                    return row
                except Exception as exc:  # defensive: never lose the row
                    logger.exception(f"Laya Node '{node_config.id}': unexpected row failure")
                    row["jev_status"] = "error"
                    row["jev_error"] = f"{type(exc).__name__}: {exc}"
                    row["jev_evaluations"] = {}
                    return row

            usage = meta.get("usage") or {}
            total_input_tokens += int(usage.get("input_tokens") or 0)
            latency = meta.get("latency_ms")
            if latency is not None:
                try:
                    latencies_ms.append(float(latency))
                except (TypeError, ValueError):
                    pass
            if meta.get("version"):
                service_versions.add(str(meta["version"]))

            # Apply dual OR decision gating (confidence < threshold or top_prob < threshold)
            row = VaptSanitizer.apply_decision_gates(
                row,
                answers,
                confidence_threshold=confidence_gate,
                top_prob_threshold=top_prob_gate,
            )

            # Laya-specific: the act-or-escalate head probability
            act_probs: List[float] = []
            for ans in answers.values():
                if isinstance(ans, dict) and isinstance(ans.get("action"), dict):
                    p = _as_float(ans["action"].get("act_probability"))
                    if p is not None:
                        act_probs.append(p)
            if act_probs:
                row["laya_act_probability"] = round(min(act_probs), 4)

            row["jev_evaluations"] = answers
            if tag_target_key and tag_target_key != "category":
                row[tag_target_key] = row.get("category")
            return row

        limits = httpx.Limits(
            max_connections=max(1, concurrency) * 2,
            max_keepalive_connections=max(1, concurrency),
        )
        chunk_size = max(1, batch_progress_size)

        if thread_id and total_items > 0:
            set_jev_progress(thread_id, {
                "status": "evaluating",
                "backend": "laya",
                "endpoint": predict_url,
                "completed": 0,
                "total": total_items,
                "percent": 0.0,
                "node_id": node_config.id,
                "checkpoint_path": checkpoint_path,
            })

        async with httpx.AsyncClient(timeout=timeout_sec, limits=limits) as http_client:
            for chunk_start in range(0, total_items, chunk_size):
                chunk = items_to_process[chunk_start : chunk_start + chunk_size]
                chunk_results = await asyncio.gather(
                    *[process_single_item(it, http_client) for it in chunk],
                    return_exceptions=True,
                )

                clean_chunk: List[Dict[str, Any]] = []
                for original, result in zip(chunk, chunk_results):
                    if isinstance(result, BaseException):
                        if isinstance(result, LayaAPIError):
                            fatal_error = result
                        else:
                            logger.error(
                                f"Laya Node row processing exception: "
                                f"{type(result).__name__}: {result}"
                            )
                        row = _new_row(original)
                        row["jev_status"] = "error"
                        row["jev_error"] = f"{type(result).__name__}: {result}"
                        row["jev_evaluations"] = {}
                        clean_chunk.append(row)
                    else:
                        clean_chunk.append(result)

                tagged_records.extend(clean_chunk)
                completed_count += len(clean_chunk)
                for rec in clean_chunk:
                    st = str(rec.get("jev_status") or "unknown")
                    status_counts[st] = status_counts.get(st, 0) + 1

                if checkpoint_path and checkpoint_every_chunk:
                    _write_checkpoint(checkpoint_path, tagged_records)

                if thread_id:
                    pct = round((completed_count / total_items) * 100, 1)
                    set_jev_progress(thread_id, {
                        "status": "evaluating",
                        "backend": "laya",
                        "completed": completed_count,
                        "total": total_items,
                        "percent": pct,
                        "node_id": node_config.id,
                        "checkpoint_path": checkpoint_path,
                        "status_counts": dict(status_counts),
                        "records": tagged_records,
                    })


                if fatal_error is not None:
                    aborted = True
                    logger.error(
                        f"Laya Node '{node_config.id}': aborting after fatal service error "
                        f"({fatal_error.status_code}); {completed_count} rows processed, "
                        f"{total_items - completed_count} not attempted. "
                        f"Checkpoint: {checkpoint_path or '(none)'}"
                    )
                    break

        elapsed_s = max(time.perf_counter() - started_at, 1e-6)

        if thread_id and total_items > 0:
            set_jev_progress(thread_id, {
                "status": "aborted" if aborted else "completed",
                "backend": "laya",
                "completed": completed_count,
                "total": total_items,
                "percent": round((completed_count / max(1, total_items)) * 100, 1),
                "node_id": node_config.id,
                "checkpoint_path": checkpoint_path,
                "status_counts": dict(status_counts),
                "error": (str(fatal_error) if fatal_error else None),
                "records": tagged_records,
            })

        # 5. Summary
        latency_summary: Dict[str, Any] = {}
        if latencies_ms:
            ordered = sorted(latencies_ms)
            p50 = ordered[len(ordered) // 2]
            p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
            latency_summary = {
                "p50_ms": round(p50, 1),
                "p95_ms": round(p95, 1),
                "samples": len(ordered),
            }
        throughput = round(completed_count / elapsed_s, 2) if completed_count else 0.0

        summary_payload: Dict[str, Any] = {
            "backend": "laya",
            "endpoint": predict_url,
            "total_items": total_items,
            "processed_items": completed_count,
            "questions": list(payload_questions.keys()),
            "status_counts": dict(status_counts),
            "fields_sent": sorted({f for r in tagged_records for f in (r.get("jev_fields_sent") or [])}),
            "usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": 0,
            },
            "gates": {"confidence": confidence_gate, "top_prob": top_prob_gate, "margin": margin_gate},
            "checkpoint_path": checkpoint_path,
            "aborted": aborted,
            "performance": {
                "elapsed_seconds": round(elapsed_s, 2),
                "rows_per_second": throughput,
                "latency": latency_summary,
                "concurrency": concurrency,
                "service_versions": sorted(service_versions),
            },
            "undescribed_choice_questions": undescribed,
        }
        if fatal_error is not None:
            summary_payload["error"] = {
                "message": str(fatal_error),
                "status_code": fatal_error.status_code,
            }

        breakdowns: Dict[str, Dict[str, int]] = {}
        for q_id, q_cfg in payload_questions.items():
            if str(q_cfg.get("type")).lower() == "choice":
                counts: Dict[str, int] = {}
                for rec in tagged_records:
                    val = str(rec.get(q_id, "(no answer)"))
                    counts[val] = counts.get(val, 0) + 1
                breakdowns[q_id] = dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
        summary_payload["breakdowns"] = breakdowns

        summary_lines = [
            "✅ **Vulnerability Classification Completed**" if not aborted else "⚠️ **Vulnerability Classification Incomplete**",
            f"• **Findings Assessed:** `{completed_count}` of `{total_items}` records classified",
        ]
        for q_id, counts in breakdowns.items():
            dist_str = ", ".join(f"**{k}**: {v}" for k, v in counts.items())
            label = "Category Distribution" if "category" in q_id.lower() else q_id.replace("_", " ").title()
            summary_lines.append(f"• **{label}:** {dist_str}")
        if fatal_error is not None:
            summary_lines.append(f"• **Error:** {fatal_error}")

        summary_msg = "\n".join(summary_lines)

        final_output = tagged_records if is_list else (tagged_records[0] if tagged_records else {})

        # 6. Persist final output atomically
        output_context_val = final_output
        if checkpoint_path:
            _write_checkpoint(checkpoint_path, final_output if is_list else [final_output])
            output_context_val = checkpoint_path

        logger.info(
            f"Laya Node '{node_config.id}' finished: {completed_count}/{total_items} processed, "
            f"status={status_counts}, aborted={aborted}, {throughput} rows/s"
        )

        # 7. Updated context
        new_context = dict(context)
        for stale_key in (
            "context", "data", "msg", "excel_json_data_path", "latest_excel_json",
            "extracted_file_path", "latest_file_path", "vapt_validation_preview",
        ):
            new_context.pop(stale_key, None)
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

        result: Dict[str, Any] = {
            "context": new_context,
            "messages": all_messages,
            "active_agent": node_config.id,
        }
        if fatal_error is not None:
            result["error"] = {
                "message": str(fatal_error),
                "type": "LayaAPIError",
                "status_code": fatal_error.status_code,
            }
            result["error_source"] = node_config.id
        return result

    return laya_node
