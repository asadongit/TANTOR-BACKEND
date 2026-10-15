import time
import os
import uuid
import json
import logging
import asyncio
import os
import re
import threading
import copy
from datetime import datetime
from typing import Dict, Any, Optional, List
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage, FunctionMessage, ChatMessage

from Database.Database import AsyncSessionLocal
from Database.Models import ExecutionLogModel
from Apps.AgentMesh.Utilities.Security import redact_sensitive_data, sanitize_for_logging

from Apps.AgentMesh.Config import settings
from Apps.AgentMesh.Core.WebSocketManager import ws_manager

try:
    from langfuse import Langfuse, get_client as _langfuse_get_client
    from langfuse.langchain import CallbackHandler as LangfuseHandler
except ImportError:
    Langfuse = None
    LangfuseHandler = None
    _langfuse_get_client = None

logger = logging.getLogger(__name__)

# Global registry for real-time tracking of active nodes across all threads
# Format: { thread_id: { workflow_id: str, active_nodes: [node_id, ...] } }
active_runs_registry: Dict[str, Any] = {}

# Monotonic revision counter bumped on every registry mutation. Lets the
# /active-runs endpoint answer cheap "anything changed since rev N?" catch-ups.
active_runs_revision: int = 0
_revision_lock = threading.Lock()


def _bump_revision() -> int:
    """Thread-safe revision bump. Returns the new revision."""
    global active_runs_revision
    with _revision_lock:
        active_runs_revision += 1
        return active_runs_revision

class AgentMeshObserver(BaseTracer):
    """
    Local observability module for AgentMesh.
    Captures traces and logs them locally/to database without external services.
    """
    
    def __init__(self, workflow_id: str, run_id: Optional[str] = None,
                 progress_url: Optional[str] = None, total_nodes: int = 0,
                 node_meta: Optional[Dict[str, Any]] = None,
                 progress_headers: Optional[Dict[str, str]] = None):
        super().__init__()
        self.workflow_id = workflow_id
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self._seq = 0
        # ── Optional HTTP progress sink ──────────────────────────────────────
        # Backend consumers (another service, not a browser) cannot hold a
        # WebSocket open, so they may ask for progress over HTTP instead.
        #
        # This is opt-in by construction: the URL comes from whoever triggered
        # the run. A workflow nobody asked progress from has progress_url=None
        # and does no work beyond one attribute check per node. Existing
        # workflows are completely unaffected.
        self.progress_url = progress_url or None
        self.total_nodes = int(total_nodes or 0)
        self.progress_timeout = float(os.getenv("AGENTMESH_PROGRESS_TIMEOUT_SEC", "5") or 5)
        # Headers the caller asked us to send with each progress POST. Their
        # endpoint may be authenticated, and the credential belongs to them —
        # supplying it alongside the address means no consumer's secret has to
        # live in mesh configuration.
        self.progress_headers: Dict[str, str] = progress_headers or {}
        if self.progress_url and self.progress_headers:
            from Apps.AgentMesh.Utilities.Security import validate_credential_destination
            validate_credential_destination(self.progress_url)
        # node_id -> {"type", "label"}. LangChain gives the callback only a node
        # name, but a consumer needs the node's TYPE to decide what the step
        # means to it. Resolving it here keeps that decision on the consumer's
        # side while still giving them the fact.
        self._node_meta: Dict[str, Any] = node_meta or {}
        self._completed: set = set()
        try:
            # Capture the loop that created the observer (the main FastAPI loop)
            self.main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self.main_loop = None

    def _emit_run_event(self, thread_id: str, node_id: str, status: str) -> None:
        """
        Fire-and-forget a run_progress push to clients subscribed to this
        thread and workflow. Safe to call from sync LangGraph callbacks because
        the actual send is scheduled on the captured main event loop.
        """
        # Count first, deliver second. Completion is a fact about the run; it
        # stays true whether or not we are able to tell anyone about it. Counting
        # after the delivery guard would silently lose nodes finished while no
        # loop was available, and every later percentage would be too low.
        #
        # A set rather than a counter because a node can report more than once
        # (retries, loops) and progress must never move backwards on a repeat.
        if status in ("done", "error"):
            self._completed.add(node_id)

        if not self.main_loop or not self.main_loop.is_running():
            return

        self._seq += 1
        revision = _bump_revision()

        message = {
            "type": "run_progress",
            "thread_id": thread_id,
            "workflow_id": self.workflow_id,
            "node_id": node_id,
            "status": status,
            "sequence": self._seq,
            "revision": revision,
            "timestamp": datetime.now().isoformat(),
            # Derived from the graph, not hand-maintained: `total` is the node
            # count of the compiled workflow. Adding a node to the canvas changes
            # the denominator by itself, so it cannot drift out of step the way a
            # hand-written stage index does.
            "nodes_done": len(self._completed),
            "nodes_total": self.total_nodes,
            # The node's own type and label, so a consumer can decide what the
            # step means without mesh knowing anything about that consumer's
            # vocabulary.
            "node_type": (self._node_meta.get(node_id) or {}).get("type"),
            "node_label": (self._node_meta.get(node_id) or {}).get("label") or node_id,
        }
        try:
            asyncio.run_coroutine_threadsafe(
                self._async_emit(thread_id, message), self.main_loop
            )
        except Exception as e:
            logger.warning(f"[Run:{self.run_id}] Failed to schedule run_progress push: {e}")

    async def _async_emit(self, thread_id: str, message: Dict[str, Any]) -> None:
        """Deliver the run_progress message to thread- and workflow-subscribers."""
        try:
            await ws_manager.send_to_thread(thread_id, message)
            await ws_manager.send_to_workflow(self.workflow_id, message)
        except Exception as e:
            logger.debug(f"[Run:{self.run_id}] run_progress push error: {e}")

        # Optional HTTP sink, for consumers that cannot hold a socket open.
        # Untouched by the WebSocket delivery above: that is how mesh's own
        # canvas lights up the running node and must keep working regardless.
        if self.progress_url:
            await self._post_progress(message)

    async def _post_progress(self, message: Dict[str, Any]) -> None:
        """POST one progress event. Never raises, never blocks the pipeline.

        Progress is disposable: a credit run must not fail, stall or slow down
        because something downstream is not listening. Hence a short timeout and
        a swallowed exception - no retries, no backoff, no queue.
        """
        try:
            import httpx
            async with httpx.AsyncClient(timeout=self.progress_timeout) as client:
                await client.post(
                    self.progress_url, json=message, headers=self.progress_headers
                )
        except Exception as e:
            logger.debug(f"[Run:{self.run_id}] progress POST failed: {e}")

    def _persist_run(self, run: Run) -> None:
        """
        Custom persistence logic for every 'Run' (Node execution).
        Saves the trace to the database in a thread-safe way.
        """
        # 1. Fire and forget the async save to the main event loop
        if self.main_loop and self.main_loop.is_running():
            asyncio.run_coroutine_threadsafe(self._async_persist(run), self.main_loop)
        else:
            # Fallback if no loop is available (e.g. standalone scripts)
            try:
                # Use a fresh loop if needed, but be careful with shared engines
                asyncio.run(self._async_persist(run))
            except Exception as e:
                logger.error(f"Fallback persistence failed: {e}")

        # 2. Local File Logging (Trace Correlation)
        latency = (run.end_time - run.start_time).total_seconds() if run.end_time else 0
        status = self._get_status(run)
        logger.info(f"[OBSERVER] Trace Captured: {run.name} | Status: {status} | Latency: {latency:.2f}s")

    def _get_output_error(self, run: Run):
        """Extract the swallowed node error from the run's captured outputs.

        `wrap_with_error_handling` catches exceptions and returns them *inside*
        the node's state update (dict with an `error` key) instead of raising, so
        LangGraph records the run as successful and `on_run_error` never fires.
        The error dict is still visible in `run.outputs`, which lets the observer
        report the failure to the WebSocket + DB exactly like a real exception.
        """
        outputs = run.outputs or {}
        if isinstance(outputs, dict):
            err = outputs.get("error")
            if isinstance(err, dict) and err.get("message"):
                return err
        return None

    def _get_status(self, run: Run):
        if run.error: return "error"
        if self._get_output_error(run): return "error"
        if run.end_time: return "success"
        return "running"

    async def _async_persist(self, run: Run):
        """Async database write logic with JSON serialization and metadata extraction"""
        try:
            # 1. Helper to make inputs/outputs JSON serializable
            def _ensure_serializable(data: Any) -> Any:
                if isinstance(data, dict):
                    return {str(k): _ensure_serializable(v) for k, v in data.items()}
                elif isinstance(data, (list, tuple, set)):
                    return [_ensure_serializable(x) for x in data]
                elif hasattr(data, "content"): # For LangChain messages
                    return {
                        "type": type(data).__name__,
                        "content": _ensure_serializable(getattr(data, "content", "")),
                        "metadata": _ensure_serializable(getattr(data, "additional_kwargs", {}) or {})
                    }
                elif isinstance(data, (str, int, float, bool, type(None))):
                    return data
                return str(data)

            def _for_db(data: Any) -> Any:
                """What actually goes into input_data / output_data.

                Redact first, then bound the size. Redaction must run before
                truncation, or a string can be cut mid-secret and leave half
                of it in the log.
                """
                safe = sanitize_for_logging(redact_sensitive_data(_ensure_serializable(data)))
                return _fit_column(_cap_long_strings(
                    _redact_data_uris(safe),
                    _MAX_DB_STRING,
                ))


            # 2. Extract Metadata (Tokens, Model Name)
            tokens = None
            model_name = None
            
            # Scan outputs for token usage (usually in the last message's usage_metadata)
            if run.outputs and "messages" in run.outputs:
                msgs = run.outputs["messages"]
                if msgs:
                    last_msg = msgs[-1]
                    usage = getattr(last_msg, "usage_metadata", None) or {}
                    if usage:
                        tokens = {
                            "input": usage.get("input_tokens"),
                            "output": usage.get("output_tokens"),
                            "total": usage.get("total_tokens")
                        }
                    
                    # Try to get model_name from various LangChain message metadata locations
                    add_kwargs = getattr(last_msg, "additional_kwargs", None) or {}
                    model_name = add_kwargs.get("model_name") or add_kwargs.get("model")
                    
                    if not model_name:
                        resp_meta = getattr(last_msg, "response_metadata", None) or {}
                        model_name = resp_meta.get("model_name") or resp_meta.get("model") or resp_meta.get("openai_model")

            # 3. Calculate Latency (ms)
            duration_ms = 0
            if run.end_time and run.start_time:
                duration_ms = (run.end_time - run.start_time).total_seconds() * 1000

            async with AsyncSessionLocal() as session:
                extra = run.extra or {}
                metadata = extra.get("metadata") or {}
                thread_id = str(metadata.get("thread_id", "anonymous"))

                log = ExecutionLogModel(
                    id=str(run.id),
                    thread_id=thread_id,
                    node_id=run.name,
                    workflow_id=self.workflow_id,
                    input_data=_for_db(run.inputs),
                    output_data=_for_db(run.outputs),
                    latency_ms=duration_ms,
                    tokens_used=tokens,
                    model_name=model_name,
                    status=self._get_status(run),
                    error_message=sanitize_for_logging(str(run.error)) if run.error else sanitize_for_logging((self._get_output_error(run) or {}).get("message"))
                )
                # A node is persisted more than once — first when it starts, then
                # again when it finishes, so the row can move running -> success.
                #
                # session.merge() is not safe for that here: it SELECTs, finds
                # nothing, then INSERTs, and the two calls arrive close enough
                # together in separate sessions that both pass the SELECT and
                # both INSERT. The second violates execution_logs_pkey and the
                # record is lost, leaving the log half-written and noisy.
                #
                # INSERT ... ON CONFLICT DO UPDATE is decided by the database in
                # one statement, so concurrency cannot split it. Whichever call
                # arrives second updates the row instead of failing.
                values = {
                    "id": str(run.id),
                    "thread_id": thread_id,
                    "node_id": run.name,
                    "workflow_id": self.workflow_id,
                    "input_data": _for_db(run.inputs),
                    "output_data": _for_db(run.outputs),
                    "latency_ms": duration_ms,
                    "tokens_used": tokens,
                    "model_name": model_name,
                    "status": self._get_status(run),
                    "error_message": sanitize_for_logging(str(run.error)) if run.error
                    else sanitize_for_logging((self._get_output_error(run) or {}).get("message")),
                }
                try:
                    from sqlalchemy.dialects.postgresql import insert as _pg_insert

                    stmt = _pg_insert(ExecutionLogModel).values(**values)
                    # Never overwrite a finished record with a later-arriving
                    # "running": the outcome is what matters, not the order the
                    # callbacks happened to fire in.
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["id"],
                        set_={k: v for k, v in values.items() if k != "id"},
                        where=ExecutionLogModel.status == "running",
                    )
                    await session.execute(stmt)
                except ImportError:
                    # Non-PostgreSQL backend: fall back to the previous behaviour.
                    await session.merge(log)
                await session.commit()
        except Exception as e:
            logger.error(f"[Run:{self.run_id}] DB Persistence Failed: {e}")

    # ── LangChain tracer hooks ───────────────────────────────────────────────
    # BaseTracer calls _on_run_create / _on_run_update. The three methods below
    # (on_run_start / on_run_end / on_run_error) are NOT hook names LangChain
    # knows, so nothing ever invoked them: active_runs_registry stayed empty,
    # /active-runs always returned nothing, and no run_progress was ever pushed.
    # These two adapters connect the real hooks to the existing logic.

    def _is_graph_node(self, run: Run) -> bool:
        """True if this run is a workflow node rather than an internal sub-run.

        _on_run_create fires for every run in the tree — chains, LLM calls, tool
        calls. Only the nodes the user placed on the canvas are meaningful as
        progress, and counting sub-runs would push nodes_done past nodes_total.
        When the node index is unavailable (some callers construct the observer
        without it) nothing is filtered, preserving previous behaviour.
        """
        if not self._node_meta:
            return True
        return run.name in self._node_meta

    def _on_run_create(self, run: Run) -> None:
        if self._is_graph_node(run):
            self.on_run_start(run)

    def _on_run_update(self, run: Run) -> None:
        if not self._is_graph_node(run):
            return
        if getattr(run, "error", None):
            self.on_run_error(run)
        else:
            self.on_run_end(run)

    def on_run_start(self, run: Run) -> None:
        """Called when a node starts"""
        # Track in active registry
        extra = run.extra or {}
        metadata = extra.get("metadata") or {}
        thread_id = str(metadata.get("thread_id", "anonymous"))

        if thread_id not in active_runs_registry:
            active_runs_registry[thread_id] = {"workflow_id": self.workflow_id, "active_nodes": []}
        
        if run.name not in active_runs_registry[thread_id]["active_nodes"]:
            active_runs_registry[thread_id]["active_nodes"].append(run.name)

        # Push live status to subscribed clients (thread- and workflow-scoped)
        self._emit_run_event(thread_id, run.name, "running")

        # Log to DB as 'running'
        logger.info(f"[OBSERVER] Node Starting: {run.name} (Thread: {thread_id})")
        self._persist_run(run)

    def _remove_from_registry(self, run: Run):
        extra = run.extra or {}
        metadata = extra.get("metadata") or {}
        thread_id = str(metadata.get("thread_id", "anonymous"))

        if thread_id in active_runs_registry:
            if run.name in active_runs_registry[thread_id]["active_nodes"]:
                active_runs_registry[thread_id]["active_nodes"].remove(run.name)
            if not active_runs_registry[thread_id]["active_nodes"]:
                del active_runs_registry[thread_id]
        return thread_id

    def on_run_end(self, run: Run) -> None:
        """Called when a node ends successfully"""
        thread_id = self._remove_from_registry(run)
        # 🛠️ Swallowed failures: wrap_with_error_handling returns the error inside
        # the output dict rather than raising, so detect it here and surface the
        # real error status (WS + DB) instead of a misleading 'done'/'success'.
        output_error = self._get_output_error(run)
        if output_error:
            self._emit_run_event(thread_id, run.name, "error")
        else:
            self._emit_run_event(thread_id, run.name, "done")
        self._persist_run(run)

    def on_run_error(self, run: Run) -> None:
        """Called when a node fails"""
        logger.error(f"[Run:{self.run_id}] Node {run.name} failed: {run.error}")
        thread_id = self._remove_from_registry(run)
        self._emit_run_event(thread_id, run.name, "error")
        self._persist_run(run)

def _langfuse_trace_id(external_id: str) -> str:
    """Derive a valid Langfuse trace id (32 lowercase hex chars) from an
    external id (e.g. thread_id). Deterministic: same seed -> same trace id,
    so HITL pause/resume segments and log_human_decision all land on ONE trace.

    Langfuse rejects ids that aren't 32 hex chars (see SDK `_is_valid_trace_id`),
    so the raw thread_id can't be used as a trace id directly.
    """
    if Langfuse is None:
        return external_id
    return Langfuse.create_trace_id(seed=external_id)


def _set_langfuse_env() -> None:
    """Ensure the SDK can pick up the configured Langfuse credentials from env."""
    from Apps.AgentMesh.Utilities.Security import validate_credential_destination
    validate_credential_destination(settings.LANGFUSE_BASE_URL)
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
    os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
    os.environ["LANGFUSE_BASE_URL"] = settings.LANGFUSE_BASE_URL
    os.environ["LANGFUSE_HOST"] = settings.LANGFUSE_BASE_URL
    os.environ["LANGFUSE_MEDIA_UPLOAD_ENABLED"] = "false"


def _get_langfuse_client():
    """Return the shared Langfuse client used by the LangChain handler.

    The LangfuseHandler resolves its client through `get_client()` internally,
    so we reuse the exact same singleton here. This guarantees `flush_langfuse()`
    and `log_human_decision` write to the SAME client/pipeline as the trace.
    Returns None when Langfuse is unavailable or not configured.
    """
    if not _langfuse_get_client:
        logger.debug("Langfuse SDK not installed. Langfuse client unavailable.")
        return None
    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        logger.debug("Langfuse keys not configured. Langfuse client unavailable.")
        return None
    try:
        _set_langfuse_env()
        return _langfuse_get_client()
    except Exception as e:
        logger.warning(f"Langfuse: Failed to obtain shared client: {e}")
        return None


def flush_langfuse() -> None:
    """Flush any buffered Langfuse spans/events to the server.

    Langfuse batches spans in memory and exports them on a background thread.
    Without an explicit flush at run end, the last nodes (classifier, agents,
    aggregator) and token usage can be lost when the export is read immediately.
    Safe to call anywhere: no-op if Langfuse is unavailable/unconfigured.
    """
    client = _get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as e:
        logger.warning(f"Langfuse: flush failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Trace-input sanitization
#
# OCR / Vision / signature-stamp nodes send base64-encoded images inside the
# LLM prompt, and Langfuse snapshots that ENTIRE prompt into the span `input`.
# With 5+ files per splitter loop, the batched OTLP export easily exceeds the
# Langfuse server's body-size limit (nginx default 1MB) → HTTP 413 → the whole
# batch (including agent/classifier spans) is dropped. We redact base64 image
# payloads (and cap oversized strings) from the *captured* input/output so the
# actual prompts sent to the models are untouched but the spans stay exportable.
# ─────────────────────────────────────────────────────────────────────────────

_DATA_URI_RE = re.compile(r"data:([^;,]+);base64,[A-Za-z0-9+/=\s]*")
_BASE64_BLOB_RE = re.compile(r"^[A-Za-z0-9+/]{2048,}={0,2}$")
_MAX_CAPTURED_STRING = 200_000  # hard cap per string in captured span payload

# The same treatment, applied harder, for the execution_logs table.
#
# input_data and output_data hold a node's whole input and whole output. In a
# workflow that carries a file - a PDF's base64, an OCR dump, a page of
# statement text handed to an agent - that is megabytes per row, written once
# when the node starts and again when it finishes, and again for every page a
# splitter loops over. Measured on this deployment: 182 rows occupying 1.2 GB,
# 6.6 MB a row, after which PostgreSQL stopped accepting connections and the
# application could not load.
#
# The trace sink has been redacted and capped since it was written; only the
# database write went out raw. It gets a tighter limit than the 200k export cap
# because the two sinks answer different questions. A trace is read to see what
# a model was actually sent. A log row is read to see which node ran, whether it
# succeeded, and roughly what passed through it - and the payload itself is
# already durable wherever the workflow chose to write it.
#
# Truncation records the original length, so a reader always knows a value was
# longer than it appears.
_MAX_DB_STRING = int(os.getenv("AGENTMESH_LOG_MAX_STRING", "8000") or 8000)

# A per-string cap does not bound a row. Measured after adding _MAX_DB_STRING:
# rows fell from 1.36 MB to ~800 KB and no further, because a node's state is
# not one big string - it is thousands of small ones. The splitter's input
# carries the accumulated workflow state, and a mapping run holds 282 extracted
# boxes each with a row number, a label and an amount. Nothing in there is
# anywhere near 8,000 characters, so nothing was trimmed, and the row was still
# 818 KB.
#
# So the whole column is bounded too. Past this size the value is replaced by
# its shape - top-level keys and what each one holds - which is what a log
# reader needs from a payload this large anyway: not the contents, but
# confirmation the node received what it should have.
_MAX_DB_COLUMN = int(os.getenv("AGENTMESH_LOG_MAX_COLUMN", "65536") or 65536)


def _shape_of(value: Any) -> Any:
    """A one-line description of a value, used in place of one too big to keep."""
    if isinstance(value, dict):
        return {k: _shape_of(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        inner = _shape_of(value[0]) if value else None
        return f"list[{len(value)}] of {inner}" if inner else f"list[{len(value)}]"
    if isinstance(value, str):
        return f"str[{len(value)} chars]"
    return type(value).__name__


def _fit_column(value: Any) -> Any:
    """Keep the value if it fits in a log row; otherwise keep its shape."""
    try:
        size = len(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return value
    if size <= _MAX_DB_COLUMN:
        return value
    return {
        "_truncated": True,
        "_original_bytes": size,
        "_note": ("too large for the execution log; raised "
                  "AGENTMESH_LOG_MAX_COLUMN to keep more"),
        "_shape": _shape_of(value),
    }


def _redact_data_uris(value: Any) -> Any:
    """Recursively replace base64 data-URI payloads with a small placeholder.

    Covers OpenAI-style content blocks ({'type': 'image_url', 'image_url':
    {'url': 'data:image/png;base64,...'}}) and any other dict/list nesting.
    The length of the redacted payload is kept so traces still show sizes.
    """
    if isinstance(value, BaseMessage):
        clone = copy.deepcopy(value)
        try:
            clone.content = _sanitize_message_content(value.content)
        except Exception:
            pass
        for attribute in ("additional_kwargs", "response_metadata", "usage_metadata"):
            try:
                item = getattr(value, attribute, None)
                if item:
                    setattr(
                        clone,
                        attribute,
                        sanitize_for_logging(redact_sensitive_data(_redact_data_uris(item))),
                    )
            except Exception:
                pass
        return clone
    if isinstance(value, str):
        if ";base64," in value:
            def _repl(match: "re.Match") -> str:
                prefix = match.group(1) if len(match.groups()) else ""
                full = match.group(0)
                return f"data:{prefix};base64,[REDACTED {len(full)} chars]"
            return _DATA_URI_RE.sub(_repl, value)
        return value
    if isinstance(value, dict):
        return {k: _redact_data_uris(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_data_uris(v) for v in value]
    return value


def _cap_long_strings(value: Any, limit: int = _MAX_CAPTURED_STRING) -> Any:
    """Recursively truncate very large strings (e.g. raw base64 blobs, giant
    OCR dumps) so no single captured value can blow up the export payload.

    `limit` lets a caller ask for a tighter cap than the export one. The
    database sink does: a log row is there to show the shape of what a node
    received and returned, not to hold the payload itself.
    """
    if isinstance(value, str):
        if _BASE64_BLOB_RE.match(value):
            return f"[BASE64 {len(value)} chars redacted]"
        if len(value) > limit:
            return value[:limit] + f"... [truncated {len(value)} chars]"
        return value
    if isinstance(value, dict):
        return {k: _cap_long_strings(v, limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cap_long_strings(v, limit) for v in value]
    return value


def _sanitize_capture(value: Any) -> Any:
    """Create an observability-safe copy without changing runtime data."""
    without_binary = _redact_data_uris(value)
    if isinstance(without_binary, BaseMessage):
        return without_binary
    return _cap_long_strings(
        sanitize_for_logging(redact_sensitive_data(without_binary))
    )


def _sanitize_message_content(content: Any) -> Any:
    """Sanitize a message's `content` field (str or list of content blocks)."""
    if isinstance(content, str):
        return _sanitize_capture(content)
    if isinstance(content, list):
        sanitized = []
        for block in content:
            if isinstance(block, dict):
                block = _sanitize_capture(block)
            else:
                block = _sanitize_capture(block)
            sanitized.append(block)
        return sanitized
    return _sanitize_capture(content)


def _sanitize_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Return deep copies of the messages with image payloads redacted.

    Only the observability snapshot is sanitized; the original message objects
    (which the model actually receives) are never mutated.
    """
    sanitized: List[BaseMessage] = []
    for msg in messages:
        try:
            clone = copy.deepcopy(msg)
        except Exception:
            clone = msg
        try:
            new_content = _sanitize_message_content(msg.content)
        except Exception:
            new_content = None
        if new_content is not None and clone is not msg:
            clone.content = new_content
        sanitized.append(clone)
    return sanitized


class SanitizedLangfuseHandler(LangfuseHandler):
    """Langfuse handler that removes secrets and binary payloads from traces.

    Wraps the stock `langfuse.langchain.CallbackHandler` (OTel-backed) so the
    enormous OCR/vision prompts no longer inflate the OTLP export batches past
    the server's request-size limit (413). Model behavior is unaffected — only
    what Langfuse records is shrunk.
    """

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        sanitized = [_sanitize_messages(batch) for batch in messages]
        return super().on_chat_model_start(
            _sanitize_capture(serialized), sanitized, run_id=run_id, parent_run_id=parent_run_id,
            tags=tags, metadata=_sanitize_capture(metadata), **kwargs,
        )

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        sanitized = [_sanitize_capture(p) for p in prompts]
        return super().on_llm_start(
            _sanitize_capture(serialized), sanitized, run_id=run_id, parent_run_id=parent_run_id,
            tags=tags, metadata=_sanitize_capture(metadata), **kwargs,
        )

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        # inputs may be a state dict containing messages / base64 blobs.
        sanitized = _sanitize_capture(inputs)
        return super().on_chain_start(
            _sanitize_capture(serialized), sanitized, run_id=run_id, parent_run_id=parent_run_id,
            tags=tags, metadata=_sanitize_capture(metadata), **kwargs,
        )

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        sanitized = _sanitize_capture(outputs)
        return super().on_chain_end(
            sanitized, run_id=run_id, parent_run_id=parent_run_id, **kwargs
        )

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        sanitized = _sanitize_capture(input_str)
        return super().on_tool_start(
            _sanitize_capture(serialized), sanitized, run_id=run_id, parent_run_id=parent_run_id,
            tags=tags, metadata=_sanitize_capture(metadata), **kwargs,
        )

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        return super().on_tool_end(
            _sanitize_capture(output), run_id=run_id, parent_run_id=parent_run_id, **kwargs
        )

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        safe_response = copy.deepcopy(response)
        for generation_batch in getattr(safe_response, "generations", []) or []:
            for generation in generation_batch:
                if hasattr(generation, "text"):
                    generation.text = _sanitize_capture(generation.text)
                if getattr(generation, "message", None) is not None:
                    generation.message = _sanitize_capture(generation.message)
        if getattr(safe_response, "llm_output", None) is not None:
            safe_response.llm_output = _sanitize_capture(safe_response.llm_output)
        return super().on_llm_end(
            safe_response, run_id=run_id, parent_run_id=parent_run_id, **kwargs
        )


def get_langfuse_handler(trace_id: Optional[str] = None):
    """Factory to get the Langfuse callback handler if configured.

    Pass `trace_id` (e.g. the thread_id) to pin the LangGraph trace to a
    deterministic id, so HITL pause/resume segments and the log_human_decision
    event + score all land on ONE trace per thread instead of sibling traces.
    """
    if not LangfuseHandler:
        logger.warning("Langfuse SDK not installed. Skipping Langfuse tracing.")
        return None
        
    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        logger.debug("Langfuse keys not configured. Skipping Langfuse tracing.")
        return None
        
    # Explicitly set env vars for the SDK to pick up
    _set_langfuse_env()

    # The SDK automatically uses LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST from env
    # which settings already pulls into os.environ via SettingsConfigDict
    if trace_id:
        return SanitizedLangfuseHandler(trace_context={"trace_id": _langfuse_trace_id(trace_id)})
    return SanitizedLangfuseHandler()

def get_langfuse_client():
    """Factory to get a configured Langfuse client if keys are present, else None.

    Reuses the shared client used by the LangChain trace handler so direct
    trace scoring (record_accuracy_score) and event logging share the same
    pipeline as the workflow run.
    """
    return _get_langfuse_client()


import re as _re

def _is_langfuse_trace_id(tid: Any) -> bool:
    """True if the value already looks like a Langfuse trace id (32 lowercase hex chars)."""
    return isinstance(tid, str) and bool(_re.fullmatch(r"[0-9a-f]{32}", tid))


def record_accuracy_score(trace_id: str, accuracy_score: float, evaluator: str = "user_feedback"):
    """Score a Langfuse trace for accuracy (HITL user rating or auto-evaluator).

    `trace_id` may be either the thread_id seed (converted deterministically to the
    workflow trace id) or an already-resolved 32-hex Langfuse trace id. The score
    lands on the SAME trace the workflow run is pinned to.
    """
    lf_client = get_langfuse_client()
    if not lf_client:
        return

    try:
        resolved = trace_id if _is_langfuse_trace_id(trace_id) else _langfuse_trace_id(trace_id)
        # Langfuse v4: score directly by trace_id — the old client.trace(id).score()
        # API was removed in the v4 OTel rewrite.
        lf_client.create_score(
            trace_id=resolved,
            name="accuracy",
            value=float(accuracy_score),
            comment=f"Evaluated by {evaluator}",
        )
        lf_client.flush()
        logger.debug(f"Langfuse: Logged accuracy score {accuracy_score} for trace {resolved}")
    except Exception as e:
        logger.warning(f"Langfuse: Failed to log accuracy score: {e}")


def record_toxicity_score(
    thread_id: str,
    node_id: str,
    node_type: str,
    scores: Dict[str, float],
    threshold: float = 0.5,
) -> None:
    """Log a node's toxicity score into Langfuse (numeric 0..1 score).

    Attaches to the SAME trace the workflow run is pinned to (derived
    deterministically from `thread_id`), so user/session metadata already set
    on the trace via `langfuse_user_id` / `langfuse_session_id` applies to it.

    The score is named `toxicity` so the existing `/metrics/scores` endpoint
    aggregates it (avgValue) and the `/metrics/toxicity` endpoint can break it
    down by user/session. No flush here: the shared client's background
    exporter + the run-end `flush_langfuse()` handle delivery.
    """
    lf_client = _get_langfuse_client()
    if not lf_client:
        return
    overall = _first_score_value(scores.get("toxicity") if isinstance(scores, dict) else scores)
    try:
        trace_id = _langfuse_trace_id(thread_id)
        lf_client.create_score(
            trace_id=trace_id,
            name="toxicity",
            value=round(overall, 4),
            metadata={
                "node_id": node_id,
                "node_type": node_type,
                "threshold": float(threshold),
                "is_toxic": overall >= float(threshold),
                "classes": {k: round(_first_score_value(v), 4) for k, v in (scores or {}).items()},
            },
        )
        logger.debug(f"Langfuse: Logged toxicity {overall} for trace {trace_id} (node {node_id})")
    except Exception as e:
        logger.warning(f"Langfuse: Failed to log toxicity score: {e}")


def _first_score_value(value: Any) -> float:
    """Normalize a scalar-or-single-element-array score to a float."""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else 0.0
    if hasattr(value, "tolist"):
        lst = value.tolist()
        if isinstance(lst, list):
            lst = lst[0] if lst else 0.0
        value = lst
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def mark_langfuse_trace_error(thread_id: str, message: str) -> None:
    """Force an ERROR observation onto the workflow's trace.

    Used when the error router decides to ABORT. LangGraph then ends the run
    "normally" (the node exception was swallowed for error routing), so without
    this the trace would look successful even though the workflow failed. We pin
    the observation to the deterministic trace id derived from `thread_id` (same
    id get_langfuse_handler pins the run trace to) and give it level=ERROR.
    """
    lf_client = _get_langfuse_client()
    if not lf_client:
        return
    try:
        trace_id = _langfuse_trace_id(thread_id)
        lf_client.start_observation(
            trace_context={"trace_id": trace_id},
            name="workflow_error",
            as_type="span",
            level="ERROR",
            status_message=str(message)[:2000],
        ).end()
        lf_client.flush()
        logger.debug(f"Langfuse: Marked trace {trace_id} as ERROR: {message[:200]}")
    except Exception as e:
        logger.warning(f"Langfuse: Failed to mark trace error: {e}")


def get_observer(workflow_id: str, run_id: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None, nodes: Optional[list] = None):
    """Factory to get an observer for a specific workflow run.

    `context` is the run context. If whoever triggered the run supplied a
    `progress_url` in the payload, per-node progress is POSTed there as well as
    pushed to WebSocket subscribers. Absent, nothing extra happens — which is the
    case for every workflow that has not asked for it.

    Taking the address from the caller rather than from node configuration keeps
    the canvas free of environment-specific URLs: the same workflow runs against
    dev, UAT and production unchanged.
    """
    progress_url = None
    progress_headers = {}
    if context:
        payload = context.get("payload") or {}
        if isinstance(payload, dict):
            progress_url = payload.get("progress_url")
            # The caller may protect its endpoint. It supplies the header name
            # and value so mesh can authenticate without holding the secret in
            # its own configuration or knowing what it guards.
            hdr = payload.get("progress_header")
            tok = payload.get("progress_token")
            if hdr and tok:
                progress_headers[str(hdr)] = str(tok)

    node_meta = {}
    for n in (nodes or []):
        try:
            node_meta[n.id] = {"type": n.type, "label": n.label or n.id}
        except AttributeError:
            continue

    return AgentMeshObserver(
        workflow_id=workflow_id,
        run_id=run_id,
        progress_url=progress_url,
        total_nodes=len(nodes or []),
        node_meta=node_meta,
        progress_headers=progress_headers,
    )

def log_human_decision(thread_id: str, action: str, feedback: Optional[str] = None, user_id: str = "anonymous", hitl_level: str = "node"):
    """
    Directly logs a human-in-the-loop decision as an event to Langfuse.
    This ensures decisions like 'Reject' (which stop the graph) are captured.

    The trace id is deliberately derived from `thread_id` via
    Langfuse.create_trace_id(seed=thread_id) — the same id get_langfuse_handler
    pins the run trace to, so the decision event + score attach to the ACTUAL
    workflow trace (works for both node-level and tool-level HITL pauses).
    """
    if not _get_langfuse_client():
        return

    try:
        # Use the SAME singleton client as the LangChain trace handler so the
        # decision event + score share the pipeline and are covered by flush.
        lf_client = _get_langfuse_client()
        
        trace_id = _langfuse_trace_id(thread_id)
        safe_feedback = _sanitize_capture(feedback) if feedback else None
        # Langfuse v4: events are created with a trace_context, not client.trace(id).
        lf_client.create_event(
            trace_context={"trace_id": trace_id},
            name=f"Human Decision: {action.capitalize()}",
            input=safe_feedback or f"User chose to {action}",
            metadata={
                "action": action,
                "thread_id": thread_id,
                "hitl_level": hitl_level,
                "timestamp": datetime.now().isoformat()
            }
        )

        # 📊 NEW (Apr 6, 2026): Automated scoring based on human action
        score_val = 1.0 if action == "approve" else 0.0
        score_comment = safe_feedback if action == "rework" else (f"Human rejected the flow." if action == "reject" else "Human approved output.")
        
        # Langfuse v4: create_score by trace_id (trace.score() removed in v4).
        lf_client.create_score(
            trace_id=trace_id,
            name="human-approval",
            value=score_val,
            comment=score_comment,
        )

        # Flush is async but the client handles threading
        lf_client.flush()
        logger.debug(f"Langfuse: Logged human decision '{action}' for thread {thread_id}")
    except Exception as e:
        logger.warning(f"Langfuse: Failed to log HITL decision: {e}")
