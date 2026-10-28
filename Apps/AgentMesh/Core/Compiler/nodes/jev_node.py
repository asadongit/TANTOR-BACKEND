"""
Jev (System One by TypeSafe AI) Node Builder.
Evaluates typed questions (choice, noul, score) against unstructured state or collections of JSON records.
Returns structured, calibrated categorical decisions without generative hallucination.
Supports high-concurrency evaluation and chunk-based streaming.

Reliability contract (Phase 1):
  * ``state`` is sent as a structured JSON object, never a pipe-stringified blob.
  * Field resolution is deterministic: explicit fields first, then a
    scanner-agnostic priority list, then the raw record.  Missing explicit
    fields are reported, never silently substituted.
  * Rows are checkpointed atomically after every chunk, so an interrupted or
    failed run never loses already-paid-for Jev evaluations.
  * Every row ends up with the same shape, discriminated by ``jev_status``:
    ``classified`` | ``needs_review`` | ``insufficient_state`` | ``error``.
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import httpx
from langchain_core.messages import AIMessage
from jinja2.sandbox import SandboxedEnvironment

from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from .vapt_sanitizer import (
    VaptSanitizer,
    RICH_VAPT_QUESTIONS,
    CATEGORY_CANONICAL_MAP,
)

logger = logging.getLogger(__name__)

# Live thread progress registry for external UI / dashboard observers.
_JEV_PROGRESS_REGISTRY: Dict[str, Dict[str, Any]] = {}
_PROGRESS_SUBSCRIBERS: Dict[str, List[asyncio.Queue]] = {}


def get_jev_progress(thread_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve live evaluation progress for an active session thread."""
    if not thread_id:
        return None
    return _JEV_PROGRESS_REGISTRY.get(thread_id)


def subscribe_jev_progress(thread_id: str) -> asyncio.Queue:
    """Subscribe to live progress events for thread_id."""
    q: asyncio.Queue = asyncio.Queue()
    if thread_id not in _PROGRESS_SUBSCRIBERS:
        _PROGRESS_SUBSCRIBERS[thread_id] = []
    _PROGRESS_SUBSCRIBERS[thread_id].append(q)
    return q


def unsubscribe_jev_progress(thread_id: str, q: asyncio.Queue) -> None:
    """Unsubscribe queue from thread_id events."""
    if thread_id in _PROGRESS_SUBSCRIBERS:
        try:
            _PROGRESS_SUBSCRIBERS[thread_id].remove(q)
        except ValueError:
            pass
        if not _PROGRESS_SUBSCRIBERS[thread_id]:
            _PROGRESS_SUBSCRIBERS.pop(thread_id, None)


def set_jev_progress(thread_id: str, data: Dict[str, Any]) -> None:
    """
    Publish live evaluation progress for an active session thread and push to SSE subscribers.
    Automatically generates intermediate dashboard findings if records are supplied.
    """
    if not thread_id:
        return

    records = data.pop("records", None)
    dashboard_data = data.get("dashboard_data")
    if not dashboard_data and records and isinstance(records, list) and len(records) > 0:
        try:
            from Apps.AgentMesh.Utilities.Tools.InfraTool.VAPTtool.vapt_dashboard_api import build_findings_payload
            dashboard_data = build_findings_payload(records, thread_id=thread_id)
            data["dashboard_data"] = dashboard_data
        except Exception as e:
            logger.debug(f"Failed to generate dashboard payload for progress: {e}")

    payload = {
        **data,
        "timestamp": time.time(),
    }
    _JEV_PROGRESS_REGISTRY[thread_id] = payload

    listeners = _PROGRESS_SUBSCRIBERS.get(thread_id, [])
    for q in list(listeners):
        try:
            q.put_nowait(payload)
        except Exception:
            pass


def _write_checkpoint(path: str, records: List[Any]) -> None:
    """Atomically persist tagged records so an interrupted run loses nothing."""
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=str)
        os.replace(tmp_path, path)
    except Exception as exc:
        logger.error(f"Jev Node: failed to write checkpoint '{path}': {exc}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

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


def _is_blank(v: Any) -> bool:
    """True when a cell carries no usable information."""
    if v is None:
        return True
    if isinstance(v, float) and v != v:  # NaN
        return True
    try:
        import math

        if isinstance(v, float) and math.isnan(v):
            return True
    except Exception:
        pass
    s = str(v).strip()
    return s == "" or s.lower() in ("nan", "none", "null", "n/a", "na", "-", "[]", "{}")


# Scanner-agnostic priority attributes.  Order matters: a specific vulnerability
# title must win over a generic `name`/`title` column.  Keys are normalised with
# _norm_key before comparison, so "Vulnerability Name" (Nessus/Excel) and
# "vulnerability_name" (Tenable.io JSON) resolve to the same slot.
PRIORITY_FIELDS: List[str] = [
    # Identity of the finding - most specific first
    "vulnerability_name", "plugin_name", "vulnerability", "issue", "rule_name", "rule",
    "name", "title",
    # What is affected
    "affected_package", "package", "component", "dependency", "software",
    "affected_file", "affected_files", "path", "file", "target", "binary",
    "dns_name", "host_name", "ip_address", "host",
    # Classifiers the scanner already computed
    "plugin_family", "family", "service", "protocol", "port",
    "risk_factor", "severity", "vpr", "vulnerability_priority_rating",
    "cpe", "cpe_id", "cve", "cve_id", "related_cve", "related_cve_alias",
    "cwe", "cwe_id", "cwe_description",
    # Narrative
    "description", "synopsis", "summary", "details",
    "plugin_output", "new_plugin_output", "output",
    "solution", "remediation", "fix", "mitigation", "application_patch",
    "os", "os_flavour", "os_falvour",
]

# Fields whose presence is required before spending a Jev request.  If none of
# these resolve, the state cannot support a classification.
_REQUIRED_ANY: List[str] = [
    "vulnerability_name", "plugin_name", "vulnerability", "issue", "rule_name",
    "rule", "title", "name", "synopsis", "description",
]

# Columns this node (or an earlier pass of it) writes.  These must never be fed
# back into a new classification: a re-run on an already-tagged file would
# otherwise let the previous answer into the state and anchor the next one.
# They are also skipped by the fallback field resolver.
_OUTPUT_COLUMNS: Set[str] = {
    "jev_status", "jev_error", "jev_evaluations", "jev_fields_sent",
    "jev_review_reason", "jev_error_status_code", "confidence",
    "category_confidence", "unclassified", "jev_summary",
    # Default tag key of earlier versions; kept so re-runs on legacy tagged
    # files do not consume their own previous answer.
    "category",
}

_OUTPUT_SUFFIXES: Tuple[str, ...] = (
    "_confidence", "_probabilities", "_margin", "_runner_up",
)


def _is_output_column(name: str) -> bool:
    """True when a column is written by this node rather than sourced from input."""
    n = _norm_key(name)
    if n in {_norm_key(o) for o in _OUTPUT_COLUMNS}:
        return True
    return any(n.endswith(s) for s in _OUTPUT_SUFFIXES)


def _resolve_fields(item: Dict[str, Any], explicit: Optional[List[str]]) -> Tuple[List[str], List[str]]:
    """
    Return (columns_to_send, explicit_columns_that_were_missing).

    Deterministic: an explicit list is honoured exactly.  Missing explicit
    columns are reported back to the caller rather than being silently replaced
    by a different field set.
    """
    present = {_norm_key(k): k for k in item.keys() if not _is_output_column(k)}

    if explicit:
        chosen: List[str] = []
        missing: List[str] = []
        for f in explicit:
            raw = str(f).strip()
            if not raw:
                continue
            actual = present.get(_norm_key(raw))
            if actual is not None:
                if actual not in chosen:
                    chosen.append(actual)
            else:
                missing.append(raw)
        return chosen, missing

    # Tier 2: Priority keys fallback (if any match)
    chosen = []
    seen: Set[str] = set()
    for pk in PRIORITY_FIELDS:
        actual = present.get(_norm_key(pk))
        if actual is not None and actual not in seen and not _is_blank(item.get(actual)):
            chosen.append(actual)
            seen.add(actual)

    if chosen:
        return chosen, []

    # Tier 3: Universal fallback - all non-blank, non-internal record columns
    for k, v in item.items():
        if not _is_output_column(k) and not _is_blank(v):
            if k not in chosen:
                chosen.append(k)

    return chosen, []


def _build_item_state(
    item: Any,
    fields: Optional[List[str]],
    template: Optional[str],
    max_chars: Optional[int] = None,
    tag_target_key: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build the Jev ``state`` for one record.

    Returns ``(state, meta)`` where meta carries diagnostics that are logged and
    surfaced on the run summary, so it is always visible which columns were
    actually sent for a classification.

    ``state`` is a dict (structured JSON) whenever the record is a dict, which is
    the documented preferred shape.  A string is only produced for non-dict rows
    or when an explicit template is configured.
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

    chosen, missing = _resolve_fields(item, fields)
    meta["fields_missing"] = missing

    # Output columns are never sent as state, even when listed explicitly: they
    # are this node's own previous answers, or a tag key it owns. Feeding them
    # back would anchor the next classification on the last one.
    state: Dict[str, Any] = {}
    redirect_cols = {str(tag_target_key).strip()} if tag_target_key else set()
    for col in chosen:
        if _is_output_column(col) or col in redirect_cols:
            continue
        val = item.get(col)
        if _is_blank(val):
            continue
        text = str(val).strip()
        state[col] = text

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
    return state, meta


class JevAPIError(RuntimeError):
    """A non-retryable or exhausted Jev API failure, carrying upstream detail."""

    def __init__(self, message: str, status_code: Optional[int] = None, detail: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail

    @property
    def is_auth_error(self) -> bool:
        return self.status_code in (401, 403)

    @property
    def is_fatal(self) -> bool:
        """Errors that will repeat identically for every remaining row."""
        return self.status_code in (400, 401, 403, 413, 415, 422)


# Transient conditions worth retrying.  httpx places RemoteProtocolError under
# ProtocolError (not NetworkError) and pool starvation under PoolTimeout, so a
# NetworkError-only catch silently leaks both.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504, 529}


def _build_question_payload(questions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Validate and normalise the configured question set."""
    payload_questions: Dict[str, Dict[str, Any]] = {}
    for q_id, q_cfg in questions.items():
        if not isinstance(q_cfg, dict):
            logger.warning(f"Jev Node: Skipping question '{q_id}': configuration is not an object.")
            continue
        q_type = str(q_cfg.get("type", "")).strip().lower()
        if q_type not in VALID_QUESTION_TYPES:
            logger.warning(f"Jev Node: Skipping question '{q_id}' with unsupported type '{q_type}'")
            continue
        qd = dict(q_cfg)
        qd["type"] = q_type
        # `options`/`levels` are canvas-side conveniences; the API only reads
        # `criteria`.  Accept them as aliases so misconfigured nodes still work.
        if not qd.get("criteria"):
            if q_type == "choice" and qd.get("options"):
                qd["criteria"] = qd["options"]
            elif q_type == "score" and qd.get("levels"):
                qd["criteria"] = qd["levels"]
        payload_questions[q_id] = qd

    if not payload_questions:
        raise ValueError("No valid questions (choice, noul, score) provided for Jev evaluation.")
    return payload_questions


def _error_detail(response: "httpx.Response") -> Any:
    """Extract the API's error body, bounded, for logging."""
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


async def _evaluate_state_with_jev(
    client: httpx.AsyncClient,
    api_url: str,
    api_key: str,
    model: str,
    state: Any,
    questions: Dict[str, Dict[str, Any]],
    max_retries: int = 3,
    base_delay: float = 1.0,
    long_context: bool = False,
    preset_questions: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Call the TypeSafe System One endpoint for one state.

    ``state`` may be a string, dict or list; the API accepts all three and
    structured values are strongly preferred over stringified blobs.

    Retries on 429/5xx and on transport-level faults (timeouts, connection
    errors, protocol errors), honouring ``Retry-After`` when the API supplies
    it.  Raises JevAPIError with the upstream detail for anything else.

    Returns the ``answers`` map.  Full response metadata (model, usage) is
    available to callers via ``_last_response_meta``.
    """
    headers = {"Content-Type": "application/json"}
    clean_key = str(api_key or "").strip()
    if clean_key:
        headers["Authorization"] = f"Bearer {clean_key}"
    if long_context:
        headers["TypeSafe-Danger-Zone-Long-Context"] = "true"

    payload_questions = preset_questions or _build_question_payload(questions)
    payload = {
        "model": model,
        "state": state,
        "questions": payload_questions,
    }

    attempts = max(1, int(max_retries))
    last_exc: Optional[BaseException] = None

    for attempt in range(attempts):
        try:
            response = await client.post(api_url, json=payload, headers=headers)

            if response.status_code in _RETRYABLE_STATUS:
                if attempt == attempts - 1:
                    raise JevAPIError(
                        f"Jev API returned {response.status_code} after {attempts} attempts: "
                        f"{_error_detail(response)}",
                        status_code=response.status_code,
                        detail=_error_detail(response),
                    )
                retry_after = response.headers.get("retry-after")
                delay = base_delay * (2 ** attempt)
                try:
                    if retry_after:
                        delay = max(delay, float(retry_after))
                except (ValueError, TypeError):
                    pass
                delay = min(delay, 60.0)
                logger.warning(
                    f"Jev API returned {response.status_code}. Retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{attempts})..."
                )
                await asyncio.sleep(delay)
                continue

            if response.status_code >= 400:
                detail = _error_detail(response)
                raise JevAPIError(
                    f"Jev API rejected the request ({response.status_code}): {detail}",
                    status_code=response.status_code,
                    detail=detail,
                )

            data = response.json()
            return data.get("answers", {}) or {}

        except JevAPIError:
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
                f"Jev API transport error ({type(exc).__name__}: {exc}). "
                f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{attempts})..."
            )
            await asyncio.sleep(delay)

    raise JevAPIError(
        f"Jev API unreachable after {attempts} attempts: {type(last_exc).__name__}: {last_exc}",
        status_code=None,
        detail=str(last_exc),
    )


async def evaluate_state_with_usage(
    client: httpx.AsyncClient,
    api_url: str,
    api_key: str,
    model: str,
    state: Any,
    questions: Dict[str, Dict[str, Any]],
    max_retries: int = 3,
    base_delay: float = 1.0,
    long_context: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate one state and return ``(answers, usage)``.

    ``usage`` is the API-reported token usage so callers can account for cost.
    """
    headers = {"Content-Type": "application/json"}
    clean_key = str(api_key or "").strip()
    if clean_key:
        headers["Authorization"] = f"Bearer {clean_key}"
    if long_context:
        headers["TypeSafe-Danger-Zone-Long-Context"] = "true"

    payload_questions = _build_question_payload(questions)
    payload = {"model": model, "state": state, "questions": payload_questions}

    attempts = max(1, int(max_retries))
    last_exc: Optional[BaseException] = None

    for attempt in range(attempts):
        try:
            response = await client.post(api_url, json=payload, headers=headers)

            if response.status_code in _RETRYABLE_STATUS:
                if attempt == attempts - 1:
                    raise JevAPIError(
                        f"Jev API returned {response.status_code} after {attempts} attempts: "
                        f"{_error_detail(response)}",
                        status_code=response.status_code,
                        detail=_error_detail(response),
                    )
                retry_after = response.headers.get("retry-after")
                delay = base_delay * (2 ** attempt)
                try:
                    if retry_after:
                        delay = max(delay, float(retry_after))
                except (ValueError, TypeError):
                    pass
                await asyncio.sleep(min(delay, 60.0))
                continue

            if response.status_code >= 400:
                detail = _error_detail(response)
                raise JevAPIError(
                    f"Jev API rejected the request ({response.status_code}): {detail}",
                    status_code=response.status_code,
                    detail=detail,
                )

            data = response.json()
            return (data.get("answers", {}) or {}), (data.get("usage", {}) or {})

        except JevAPIError:
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
            await asyncio.sleep(min(base_delay * (2 ** attempt), 60.0))

    raise JevAPIError(
        f"Jev API unreachable after {attempts} attempts: {type(last_exc).__name__}: {last_exc}",
        detail=str(last_exc),
    )


def _as_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (ValueError, TypeError):
        return None
    return f


def _extract_answers_from_response(
    answers: Dict[str, Any],
    questions: Dict[str, Dict[str, Any]],
    tag_target_key: Optional[str] = None,
    confidence_gate: Optional[float] = None,
    margin_gate: Optional[float] = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Extract typed answers into flat row fields keyed by question id.

    Returns ``(extracted, review_reason)``.  ``review_reason`` is non-empty when
    the primary classification is ambiguous and should be routed to review.

    Confidence and margin are different signals:
      * ``confidence`` -- how concentrated the distribution is (display only).
      * ``margin``     -- gap between the top option and the runner-up, i.e. how
                          much the model preferred one category over another.
    Ambiguity is judged on ``margin``; ``confidence`` alone conflates "the model
    is sure" with "the model is sure it is one of two close options".

    ``tag_target_key`` writes the primary choice under a caller-chosen name
    (e.g. ``category``) without hardcoding any question id.
    """
    extracted: Dict[str, Any] = {}
    review_reasons: List[str] = []

    for q_id, q_cfg in questions.items():
        ans = answers.get(q_id)
        if not ans or not isinstance(ans, dict):
            continue

        q_type = str(ans.get("type") or q_cfg.get("type") or "").lower()

        if q_type == "choice":
            choice_val = ans.get("choice")
            if choice_val is not None:
                extracted[q_id] = choice_val

            probs = ans.get("probabilities")
            if isinstance(probs, dict) and probs:
                extracted[f"{q_id}_probabilities"] = {
                    str(k): _as_float(v) for k, v in probs.items() if _as_float(v) is not None
                }
                ranked = sorted(
                    ((k, _as_float(v) or 0.0) for k, v in probs.items()),
                    key=lambda kv: kv[1],
                    reverse=True,
                )
                if ranked:
                    top_name, top_p = ranked[0]
                    runner_p = ranked[1][1] if len(ranked) > 1 else 0.0
                    margin = round(top_p - runner_p, 4)
                    extracted[f"{q_id}_margin"] = margin
                    extracted[f"{q_id}_runner_up"] = ranked[1][0] if len(ranked) > 1 else None
                    if margin_gate is not None and margin < margin_gate:
                        review_reasons.append(
                            f"{q_id}: margin {margin:.2f} < {margin_gate:.2f} "
                            f"({top_name} vs {extracted[f'{q_id}_runner_up']})"
                        )

            conf = _as_float(ans.get("confidence"))
            if conf is not None:
                extracted[f"{q_id}_confidence"] = conf
                if confidence_gate is not None and conf < confidence_gate:
                    review_reasons.append(f"{q_id}: confidence {conf:.2f} < {confidence_gate:.2f}")

            if tag_target_key and choice_val is not None and tag_target_key not in extracted:
                extracted[tag_target_key] = choice_val
                if conf is not None:
                    extracted[f"{tag_target_key}_confidence"] = conf

        elif q_type == "noul":
            val = _as_float(ans.get("noul"))
            extracted[q_id] = val if val is not None else ans.get("noul")

        elif q_type == "score":
            val = _as_float(ans.get("score"))
            extracted[q_id] = val if val is not None else ans.get("score")
            conf = _as_float(ans.get("confidence"))
            if conf is not None:
                extracted[f"{q_id}_confidence"] = conf

    return extracted, "; ".join(review_reasons)


def _source_record(item: Any) -> Dict[str, Any]:
    """Best-effort recovery of the original record, preserving scanner columns."""
    if isinstance(item, dict):
        return dict(item)
    return {"original": item}


def build_jev_node(node_config: NodeConfig) -> Callable:
    """
    Build a Jev (System One by TypeSafe AI) node runner.
    Takes data from context (single item or list of records), evaluates configured questions
    via TypeSafe AI Jev, tags rows with the results, and saves the structured dataset.
    """
    model = getattr(node_config, "jev_model", "jev-latest") or "jev-latest"
    api_key_env = getattr(node_config, "jev_api_key_env", "TYPESAFE_API_KEY") or "TYPESAFE_API_KEY"
    api_key_override = getattr(node_config, "jev_api_key", None) or getattr(node_config, "typesafe_api_key", None)
    raw_api_url = getattr(node_config, "jev_api_url", None)
    if not raw_api_url or "chat/completions" in str(raw_api_url).lower() or ("localhost" in str(raw_api_url).lower() and "8000" in str(raw_api_url)):
        api_url = DEFAULT_TYPESAFE_API_URL
    else:
        api_url = str(raw_api_url).strip()

    input_key = getattr(node_config, "jev_input_key", "excel_json_data") or "excel_json_data"
    output_key = getattr(node_config, "jev_output_key", "excel_json_data") or "excel_json_data"
    item_fields = getattr(node_config, "jev_item_fields", None)
    state_template = getattr(node_config, "jev_state_template", None)
    raw_questions = getattr(node_config, "jev_questions", {}) or {}
    # Use refined 5-option schema if empty or containing legacy insufficient_info choice
    if not raw_questions or "insufficient_info" in (raw_questions.get("vulnerability_category", {}).get("options", [])):
        questions = RICH_VAPT_QUESTIONS
    else:
        questions = raw_questions
    timeout_sec = float(getattr(node_config, "jev_timeout_seconds", 60.0) or 60.0)
    concurrency = int(getattr(node_config, "jev_concurrency", 25) or 25)
    batch_progress_size = int(getattr(node_config, "jev_batch_progress_size", 10) or 10)
    stream_progress = bool(getattr(node_config, "jev_stream_progress", False))

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

    tag_target_key = getattr(node_config, "jev_tag_target_key", None) or None
    tag_key_mode = str(getattr(node_config, "jev_tag_key_mode", "replace") or "replace").strip().lower()
    if tag_key_mode not in ("replace", "preserve"):
        tag_key_mode = "replace"

    max_retries = int(getattr(node_config, "jev_max_retries", 3) or 3)
    retry_base_delay = float(getattr(node_config, "jev_retry_base_delay", 1.0) or 1.0)
    checkpoint_every_chunk = bool(getattr(node_config, "jev_checkpoint_every_chunk", True))
    max_state_chars = getattr(node_config, "jev_max_state_chars", None)
    max_state_chars = int(max_state_chars) if max_state_chars not in (None, "", 0, "0") else None
    long_context = bool(getattr(node_config, "jev_long_context", False))
    abort_on_fatal = bool(getattr(node_config, "jev_abort_on_fatal_error", True))

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
        resolved_api_key = str(api_key_override or "").strip()

        # Fallback: if not configured directly on canvas, check environment variables
        if not resolved_api_key:
            env_val = os.environ.get(api_key_env, "")
            if env_val and str(env_val).strip() and api_key_env != "OPENAI_API_KEY":
                resolved_api_key = str(env_val).strip()
            else:
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

        # 4. Resolve and report the field set once per run, so it is always
        # visible which columns the classification actually saw.
        sample_dict = next(
            (it for it in items_to_process if isinstance(it, dict)), {}
        )
        if sample_dict:
            resolved_cols, missing_cols = _resolve_fields(sample_dict, item_fields)
            if missing_cols:
                logger.warning(
                    f"Jev Node '{node_config.id}': configured jev_item_fields not present in the "
                    f"input and will NOT be sent: {missing_cols}. "
                    f"Resolved columns: {resolved_cols}"
                )
            else:
                logger.info(
                    f"Jev Node '{node_config.id}': state columns for classification: {resolved_cols}"
                )
            if not resolved_cols and not (state_template and str(state_template).strip()):
                logger.error(
                    f"Jev Node '{node_config.id}': none of the configured fields exist in the input. "
                    f"Input columns are: {list(sample_dict.keys())[:25]}"
                )
        else:
            logger.info(f"Jev Node '{node_config.id}': input rows are not dicts; using raw text state.")

        try:
            payload_questions = _build_question_payload(questions)
        except ValueError as exc:
            err = f"Jev Node '{node_config.id}': {exc}"
            logger.error(err)
            return {
                "context": dict(context),
                "messages": [AIMessage(content=f"⚠️ {err}", name=node_config.id)],
                "active_agent": node_config.id,
                "error": {"message": err, "type": "ConfigurationError"},
                "error_source": node_config.id,
            }

        # 5. Processing: concurrency pool, per-chunk checkpoint, circuit breaker
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
        total_output_tokens = 0
        fatal_error: Optional[JevAPIError] = None
        aborted = False

        def _new_row(item: Any) -> Dict[str, Any]:
            """Start a row with the source columns already preserved."""
            return _source_record(item)

        async def process_single_item(item: Any, http_client: httpx.AsyncClient) -> Dict[str, Any]:
            nonlocal total_input_tokens, total_output_tokens

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
                state_value, state_meta = _build_item_state(
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

            if not use_live_api:
                row["jev_status"] = "insufficient_state"
                row["jev_error"] = f"API key not found in {api_key_env}"
                row["jev_evaluations"] = {}
                return row

            answers: Dict[str, Any] = {}
            async with semaphore:
                try:
                    answers, usage = await evaluate_state_with_usage(
                        http_client,
                        effective_api_url,
                        resolved_api_key,
                        model,
                        state_value,
                        payload_questions,
                        max_retries=max_retries,
                        base_delay=retry_base_delay,
                        long_context=long_context,
                    )
                    total_input_tokens += int(usage.get("input_tokens") or 0)
                    total_output_tokens += int(usage.get("output_tokens") or 0)
                except JevAPIError as exc:
                    row["jev_status"] = "error"
                    row["jev_error"] = str(exc)
                    row["jev_error_status_code"] = exc.status_code
                    row["jev_evaluations"] = {}
                    if exc.is_fatal and abort_on_fatal:
                        raise
                    return row
                except Exception as exc:  # defensive: never lose the row
                    logger.exception(f"Jev Node '{node_config.id}': unexpected row failure")
                    row["jev_status"] = "error"
                    row["jev_error"] = f"{type(exc).__name__}: {exc}"
                    row["jev_evaluations"] = {}
                    return row

            # Apply dual OR decision gating (confidence < threshold or top_prob < threshold)
            row = VaptSanitizer.apply_decision_gates(
                row,
                answers,
                confidence_threshold=confidence_gate,
                top_prob_threshold=top_prob_gate,
            )
            row["jev_evaluations"] = answers
            if tag_target_key and tag_target_key != "category":
                row[tag_target_key] = row.get("category")
            return row

        limits = httpx.Limits(max_connections=max(1, concurrency) * 2, max_keepalive_connections=max(1, concurrency))
        chunk_size = max(1, batch_progress_size)

        if thread_id and total_items > 0:
            set_jev_progress(thread_id, {
                "status": "evaluating",
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
                        if isinstance(result, JevAPIError):
                            fatal_error = result
                        else:
                            logger.error(
                                f"Jev Node row processing exception: "
                                f"{type(result).__name__}: {result}"
                            )
                        # Preserve the source record so the row is recoverable.
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

                # Checkpoint paid-for work before anything else can fail.
                if checkpoint_path and checkpoint_every_chunk:
                    _write_checkpoint(checkpoint_path, tagged_records)

                if thread_id:
                    pct = round((completed_count / total_items) * 100, 1)
                    set_jev_progress(thread_id, {
                        "status": "evaluating",
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
                    remaining = total_items - completed_count
                    logger.error(
                        f"Jev Node '{node_config.id}': aborting after fatal API error "
                        f"({fatal_error.status_code}); {completed_count} rows processed, "
                        f"{remaining} rows not attempted. "
                        f"Checkpoint: {checkpoint_path or '(none)'}"
                    )
                    break

        if thread_id and total_items > 0:
            set_jev_progress(thread_id, {
                "status": "aborted" if aborted else "completed",
                "completed": completed_count,
                "total": total_items,
                "percent": round((completed_count / max(1, total_items)) * 100, 1),
                "node_id": node_config.id,
                "checkpoint_path": checkpoint_path,
                "status_counts": dict(status_counts),
                "error": (str(fatal_error) if fatal_error else None),
                "records": tagged_records,
            })

        # 6. Summary statistics
        summary_payload: Dict[str, Any] = {
            "total_items": total_items,
            "processed_items": completed_count,
            "model": model,
            "live_api": use_live_api,
            "questions": list(payload_questions.keys()),
            "status_counts": dict(status_counts),
            "fields_sent": sorted({f for r in tagged_records for f in (r.get("jev_fields_sent") or [])}),
            "usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
            },
            "gates": {"confidence": confidence_gate, "top_prob": top_prob_gate, "margin": margin_gate},
            "checkpoint_path": checkpoint_path,
            "aborted": aborted,
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

        # 7. Persist final output atomically
        output_context_val = final_output
        if checkpoint_path:
            _write_checkpoint(checkpoint_path, final_output if is_list else [final_output])
            logger.info(f"Jev Node '{node_config.id}': persisted tagged findings to '{checkpoint_path}'")
            output_context_val = checkpoint_path

        logger.info(
            f"Jev Node '{node_config.id}' finished: {completed_count}/{total_items} processed, "
            f"status={status_counts}, aborted={aborted}"
        )

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

        result: Dict[str, Any] = {
            "context": new_context,
            "messages": all_messages,
            "active_agent": node_config.id,
        }
        if fatal_error is not None:
            result["error"] = {
                "message": str(fatal_error),
                "type": "JevAPIError",
                "status_code": fatal_error.status_code,
            }
            result["error_source"] = node_config.id
        return result

    return jev_node
