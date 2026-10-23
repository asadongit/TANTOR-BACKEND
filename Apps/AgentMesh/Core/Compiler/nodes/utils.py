"""
Shared utilities for node modules.
Includes common functions used across different node types.
"""

import re
import json
import os
import asyncio
import threading
import uuid
from typing import Dict, Any, List, Optional, Tuple
from pydantic import create_model
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from Apps.AgentMesh.Core.ToolRegistry import registry


import inspect
import logging
from json_repair import loads
from jinja2.sandbox import SandboxedEnvironment

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Rendering environments
#
# Interpolation serves two different jobs, and they need opposite treatment of
# strings:
#
#   TEXT MODE (the default, ~38 call sites) — the result is used literally: a
#     URL, a header value, a file path, an email address, an LLM prompt, a JSON
#     body. A string must render as-is. Quoting it corrupts the result, e.g.
#     "http://host/x" becomes "'http://host/x'", which is not a valid URL.
#
#   EXPRESSION MODE (4 call sites) — the result is fed to SimpleEval to be
#     evaluated, as in the Switch and Condition nodes and Weightage rules. There
#     a bare string is read as an identifier: `approved == 'approved'` raises
#     NameError, while `'approved' == 'approved'` correctly evaluates to True.
#     These call sites must pass quote_strings=True.
#
# Both jobs previously shared one environment that quoted every string, which
# made expression evaluation work but silently corrupted every literal use.
# Text mode is the common case, so it is the default; quoting is opt-in.
# ─────────────────────────────────────────────────────────────────────────────

def _finalize_common(value: Any) -> Any:
    """Shared handling for non-string values."""
    if isinstance(value, (dict, list, bool)):
        return json.dumps(value)
    if value is None:
        return ""
    return value


def agent_mesh_finalize_text(value: Any) -> Any:
    """TEXT mode: render strings literally."""
    return _finalize_common(value)


def agent_mesh_finalize_expr(value: Any) -> Any:
    """EXPRESSION mode: quote strings so SimpleEval treats them as literals."""
    if isinstance(value, str):
        return f"'{value}'"
    return _finalize_common(value)


# Backwards-compatible alias: this name previously meant the quoting behaviour.
agent_mesh_finalize = agent_mesh_finalize_expr


class AgentMeshSandboxedEnvironment(SandboxedEnvironment):
    def getattr(self, obj: Any, attribute: str) -> Any:
        # Prioritize key lookup over methods for dictionaries
        if isinstance(obj, dict) and attribute in obj:
            return obj[attribute]
        return super().getattr(obj, attribute)

_env = AgentMeshSandboxedEnvironment(finalize=agent_mesh_finalize_text)
_env_expr = AgentMeshSandboxedEnvironment(finalize=agent_mesh_finalize_expr)

class SmartInput(dict):
    """
    Production-grade recursive dictionary wrapper.
    Allows dot notation (data.user.id) and safe access at any depth.
    """
    def _wrap(self, val):
        if isinstance(val, dict) and not isinstance(val, SmartInput):
            return SmartInput(val)
        elif isinstance(val, list):
            return [self._wrap(item) for item in val]
        return val

    def __getattr__(self, name):
        return self._wrap(self.get(name))
        
    def __getitem__(self, key):
        return self._wrap(super().get(key))

def resolve_temp_path(path: str) -> Optional[str]:
    """
    Safely resolve a /temp/ URL to a physical disk path under TEMP_MOUNT.
    Protects against path traversal by normalizing and verifying the
    resolved path stays within TEMP_MOUNT.
    """
    if not path or not isinstance(path, str):
        return None
    if os.path.exists(path):
        return path
    if path.startswith("data:"):
        return path

    from Apps.AgentMesh.Config import settings

    clean = path.replace("\\", "/")
    temp_prefix = "/temp/"
    if not clean.startswith(temp_prefix):
        return None

    rel = clean[len(temp_prefix):].lstrip("/")
    temp_mount = os.path.abspath(settings.TEMP_MOUNT).rstrip("/\\")
    resolved = os.path.normpath(os.path.join(temp_mount, rel))

    # Case-safe prefix check (handles Windows drive letter casing)
    if not os.path.normcase(resolved).startswith(os.path.normcase(temp_mount)):
        return None
    if os.path.exists(resolved):
        return resolved
    return None


def make_session_dir(thread_id: str, sub_dir: str = "", base_mount: str = None) -> str:
    """
    Create and return a session-scoped directory under {base_mount}/{thread_id}/{sub_dir}.
    Defaults to TEMP_MOUNT if no base_mount provided.
    """
    safe_thread = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(thread_id)) if thread_id else "default"
    from Apps.AgentMesh.Config import settings
    mount = base_mount or settings.TEMP_MOUNT
    parts = [os.path.abspath(mount), safe_thread]
    if sub_dir:
        parts.append(sub_dir.lstrip("/"))
    session_dir = os.path.join(*parts)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def get_production_data(state: Any) -> Dict[str, Any]:
    """
    Optimized data resolver for AgentMesh workflows.
    Extracts, parses, and prepares all available state into a unified scope.
    Uses json-repair to handle malformed JSON and single quotes common in SetData nodes.
    """
    # Handle both dict and object state
    if isinstance(state, dict):
        context = (state.get("context") or {}).copy()
        messages = state.get("messages", [])
    else:
        context = getattr(state, "context", {}).copy()
        messages = getattr(state, "messages", [])

    # 2. Extract latest message content
    raw_msg = ""
    if messages:
        raw_msg = getattr(messages[-1], "content", str(messages[-1]))
    
    # 3. Smart JSON Extraction (Handles raw JSON, strings, or Markdown blocks)
    extracted_data = {}
    if raw_msg:
        # Try finding JSON block first
        match = re.search(r'```json\s*(.*?)\s*```', raw_msg, re.DOTALL)
        candidate = match.group(1) if match else raw_msg
        
        try:
            # ✅ IMPROVED: use json_repair.loads to handle single quotes
            parsed = loads(candidate)
            if isinstance(parsed, dict):
                extracted_data = parsed
            elif isinstance(parsed, list):
                # Industry Fix (May 2026): Automatically map top-level lists to 'items'
                # This enables easy seeding of data for Splitter nodes.
                extracted_data = {"items": parsed}
        except:
            pass # Not JSON, treat as plain text

    # 4. Prepare Unified Results
    # Extracted data (likely from SetData or LLM) takes priority over existing context
    merged = {**context, **extracted_data}
    
    # Wrap all dictionaries in the scope with SmartInput for safe direct access
    scope = {}
    for k, v in merged.items():
        if isinstance(k, str) and k != "data" and isinstance(v, dict) and not isinstance(v, SmartInput):
            scope[k] = SmartInput(v)
        else:
            scope[k] = v
    
    # Ensure 'msg' and 'results' are always available as reserved keywords
    scope["msg"] = raw_msg
    
    # ✅ FIX: Avoid recursive nesting of the entire scope into 'data'
    # 'data' now just points back to the merged values (without the 'data' key itself)
    scope["data"] = SmartInput({k: v for k, v in scope.items() if k != "data"})
    
    # Expose 'context' as a keyword in the Jinja2 scope for interchangeable access
    scope["context"] = SmartInput({k: v for k, v in scope.items() if k not in ("data", "context")})

    # ✅ DYNAMIC VARIABLE EXPLORER (WORKFLOW VARIABLES) RESOLUTION:
    # Interpolate dynamic template placeholders (e.g. {{nodes.set_data_32602.cust_name}} or cust_name)
    # defined in Variable Explorer (workflow_variables) against the current unified execution scope.
    # Returns the exact native raw object (dict, list, int, string) without extra quote wrapping.
    wf_vars = scope.get("workflow_variables")
    if isinstance(wf_vars, (dict, SmartInput)):
        resolved_wf_vars = dict(wf_vars)
        for var_k, var_v in list(wf_vars.items()):
            if isinstance(var_v, str) and ("{{" in var_v or var_v.startswith("nodes.") or var_v.startswith("context.")):
                try:
                    clean_path = var_v.replace("{{", "").replace("}}", "").strip()
                    native_val = get_nested_val(scope, clean_path)
                    if native_val is not None:
                        resolved_wf_vars[var_k] = native_val
                        scope[var_k] = native_val
                    else:
                        res_val = interpolate(var_v, scope)
                        resolved_wf_vars[var_k] = res_val
                        scope[var_k] = res_val
                except Exception:
                    resolved_wf_vars[var_k] = var_v
            else:
                scope[var_k] = var_v
        scope["workflow_variables"] = SmartInput(resolved_wf_vars)
    
    return scope

def wrap_tool_with_overrides(tool, overrides: Dict[str, Any], alias_override: Optional[str] = None, description_override: Optional[str] = None):
    """
    Wraps a LangChain tool so that specified arguments are pre-filled (frozen)
    and removed from the tool's input schema so the LLM doesn't see them.
    Includes a 'fuzzy' matcher to fix common LLM hallucinations.
    """
    # 🔍 Validation: Ensure tool has a name
    base_name = getattr(tool, "name", None)
    if not base_name:
        # Fallback to function name if it's a raw function decorated with @tool
        base_name = getattr(getattr(tool, "func", None), "__name__", "unnamed_tool")
        
    tool_name = alias_override if alias_override else base_name

    # ⚡ Performance & Groq Schema Compatibility Bypass:
    # If there are no parameter overrides, no alias override, and no description override,
    # return the original BaseTool directly to avoid wrapper overhead and schema mutation.
    if not overrides and not alias_override and not description_override and isinstance(tool, BaseTool):
        return tool

    logger.debug(f"Processing wrapper for tool: {tool_name}")

    base_description = getattr(tool, "description", "")
    if description_override:
        tool_description = f"{base_description} Use this instance specifically for: {description_override}"
    else:
        tool_description = base_description

    # 🔄 Async Support Fix: Check for 'coroutine' first for async tools
    original_func = getattr(tool, "coroutine", None) or getattr(tool, "func", None) or getattr(tool, "_run", None)
    
    # 🔍 Verification: Log what we found
    is_coro = inspect.iscoroutinefunction(original_func)
    logger.debug(f"Tool Wrapper: {tool_name} OriginalFunc found (IsAsync={is_coro})")
    
    original_schema = tool.args_schema

    if not original_schema:
        return tool 

    # 1. Map common hallucinations to real field names
    ALIAS_MAP = {
        "collection": "collection_name",
        "search_query": "query",
        "top_k": "k",
        "limit": "k"
    }

    # 2. Create a new schema without config (LLM won't see it)
    tool_meta = registry.get_meta(getattr(tool, "name", "unnamed_tool"))
    ai_visible = getattr(tool_meta, "ai_visible_fields", None)

    new_fields = {}
    fields = getattr(original_schema, "__fields__", {})
    if not fields:
         fields = getattr(original_schema, "model_fields", {})

    for field_name, field in fields.items():
        # A. If field is explicitly frozen by human overrides, ALWAYS hide from AI
        if field_name in overrides:
            continue
        
        # B. If tool has an explicit 'AI Visibility' list, respect it
        if ai_visible is not None and field_name not in ai_visible:
            continue
            
        # C. Standard inclusion
        if hasattr(field, "outer_type_"): 
            # Pydantic v1
            new_fields[field_name] = (field.outer_type_, field.field_info)
        elif hasattr(field, "annotation"):
            # Pydantic v2
            new_fields[field_name] = (field.annotation, field)
        else:
            new_fields[field_name] = field

    new_schema = create_model(f"{tool_name}Input", **new_fields)

    # 3. Define the wrapped function logic (Shared)
    def _prepare_final_args(kwargs):
        # Extract config if provided (by tool_node.py)
        config = kwargs.pop("config", None)
        
        # A. Start with overrides
        final_args = {**overrides}
        
        # B. Apply incoming kwargs with fuzzy alias correction
        for k, v in kwargs.items():
            corrected_key = ALIAS_MAP.get(k, k)
            final_args[corrected_key] = v
        
        # C. Only pass config if original function expects it (explicit `config`
        # parameter OR **kwargs — many tools like run_subagent receive RunnableConfig
        # via **kwargs, and dropping it there silently loses thread_id).
        try:
            sig = inspect.signature(original_func)
            has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if (("config" in sig.parameters) or has_kwargs) and config is not None:
                final_args["config"] = config
        except Exception:
            pass
        return final_args

    # 4. Create proper Sync or Async wrapper based on original function
    if inspect.iscoroutinefunction(original_func):
        async def wrapped_func_async(**kwargs):
            # Extract config if passed manually or via tool_node
            config = kwargs.pop("config", None)
            kwargs["config"] = config
            final_args = _prepare_final_args(kwargs)
            return await original_func(**final_args)
        
        # 🛠️ Groq/OpenAI compatibility fix: explicitly cast to string and ensure 
        # the name is tied to the coroutine without exposing RunnableConfig to LLM.
        return StructuredTool(
            name=str(tool_name),
            description=str(tool_description),
            coroutine=wrapped_func_async,
            args_schema=new_schema
        )
    else:
        def wrapped_func_sync(**kwargs):
            # Extract config if passed manually or via tool_node
            config = kwargs.pop("config", None)
            kwargs["config"] = config
            final_args = _prepare_final_args(kwargs)
            return original_func(**final_args)
            
        # 🛠️ Groq/OpenAI Compatibility Fix (Apr 17, 2026)
        return StructuredTool(
            name=str(tool_name),
            description=str(tool_description),
            func=wrapped_func_sync,
            args_schema=new_schema
        )


def get_nested_val(data, path):
    """Recursive getter for dictionary and list values using dot notation.
    Includes case-insensitive fallback for dict keys to handle LLM output inconsistencies."""
    parts = path.split('.')
    for p in parts:
        if isinstance(data, dict):
            # Exact match first
            val = data.get(p)
            if val is None and p not in data:
                # Case-insensitive fallback for keys (LLMs return inconsistent casing)
                p_lower = p.lower()
                for k in data:
                    if isinstance(k, str) and k.lower() == p_lower:
                        val = data[k]
                        break
            data = val
        elif isinstance(data, list):
            try:
                idx = int(p)
                data = data[idx]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return data


def interpolate(
    text: str,
    context: Dict[str, Any],
    escape_braces: bool = False,
    quote_strings: bool = False,
) -> str:
    """Interpolates string variables using a secure Sandboxed Jinja2 Environment.

    Args:
        text: template containing {{ }} placeholders.
        context: the run context; its top-level keys are the available variables.
        escape_braces: double any literal braces in the result (for f-string-like
            consumers downstream).
        quote_strings: wrap rendered strings in single quotes. Required ONLY when
            the result is handed to an expression evaluator such as SimpleEval —
            the Switch and Condition nodes and Weightage rules — where a bare
            string would be parsed as an identifier and raise NameError.
            Leave False for every literal use (URLs, headers, paths, email
            addresses, prompts, JSON bodies): quoting corrupts those.
    """
    if not text or not isinstance(text, str):
        return text

    if "{{" not in text:
        return text
        
    # Fix backwards compatibility for legacy SQL queries checking `LIKE '{{%'`
    safe_text = text.replace("{{%", "{{ '{{' }}%")

    # Pre-process Jinja expressions with dot notation containing spaces or ampersands, or prefixed with nodes.node_id.
    def _fix_jinja_expr(match):
        expr = match.group(1).strip()
        # If expression starts with nodes.node_id.key where nodes is not in context, try stripping nodes.node_id.
        if expr.startswith("nodes.") and "nodes" not in context:
            parts = expr.split(".", 2)
            if len(parts) >= 3:
                expr = parts[2]
        
        # Check if dot notation key contains spaces, ampersands, or special characters
        # e.g. aggregated_results.Profit & Loss Statement -> aggregated_results['Profit & Loss Statement']
        dot_idx = expr.rfind(".")
        if dot_idx != -1:
            prefix = expr[:dot_idx]
            key = expr[dot_idx+1:]
            if (" " in key or "&" in key or "-" in key) and not (key.startswith("[") or key.startswith("'") or key.startswith('"')):
                expr = f"{prefix}['{key}']"
                
        return "{{ " + expr + " }}"

    safe_text = re.sub(r"\{\{\s*(.*?)\s*\}\}", _fix_jinja_expr, safe_text)

    try:
        template = (_env_expr if quote_strings else _env).from_string(safe_text)
        rendered = template.render(context)
        if escape_braces:
            rendered = rendered.replace("{", "{{").replace("}", "}}")
        return rendered
    except Exception as e:
        logger.warning(f"Failed to interpolate text with Jinja2: {e}. Falling back to original text.")
        return text


def node_id_to_key(node_id: str) -> str:
    """Converts node ID (e.g. 'node_123') to a state key (e.g. 'node_123_items')."""
    return f"{node_id}_items"


def process_tools_with_overrides(tools: List[Any], tool_configs: Dict[str, Any]) -> List[Any]:
    """
    Apply tool property injection (n8n style) to freeze certain parameters.
    Also ensures all tools have 'config' in their schema for context injection.
    
    Args:
        tools: List of raw LangChain tools
        tool_configs: Dict with tool names as keys and override dicts as values
        
    Returns:
        List of processed tools with overrides applied and config support added
    """
    processed_tools = []
    for t in tools:
        overrides = (tool_configs or {}).get(t.name, {})
        alias_override = overrides.get("alias_override")
        description_override = overrides.get("description_override")
        
        # Filter out internal orchestration/HITL control keys which are not tool args
        ORCHESTRATION_KEYS = {
            "return_direct",
            "interrupt_before",
            "interrupt_after",
            "interrupt",
            "allowed_decisions",
            "allowed_actions",
            "hitl_actions",
            "hitl_feedback",
            "inject_feedback",
            "include_original_output",
            "user_input",
            "decision",
            "feedback",
            "alias_override",
            "description_override",
            "exclude_from_subagent"
        }
        arg_overrides = {k: v for k, v in overrides.items() if k not in ORCHESTRATION_KEYS}
        
        # IMPORTANT: Always wrap tools through wrap_tool_with_overrides to ensure
        # they have 'config' in their schema, even if there are no overrides.
        # This enables context injection in tool_node.py
        processed_tools.append(wrap_tool_with_overrides(t, arg_overrides, alias_override, description_override))
    return processed_tools


def build_subagent_tool_set(processed_tools: List[Any], tool_configs: Optional[Dict[str, Any]] = None) -> List[Any]:
    """
    The tool set a dynamically spawned sub-agent may inherit: the parent's
    processed tools minus `run_subagent` itself and minus any tool flagged
    `exclude_from_subagent: true` in the per-tool config. A sub-agent can never
    access a tool that is not in this returned list (strict boundary).
    """
    tool_configs = tool_configs or {}
    return [
        t for t in processed_tools
        if getattr(t, "name", "") != "run_subagent"
        and not (tool_configs.get(getattr(t, "name", "")) or {}).get("exclude_from_subagent", False)
    ]


def resolve_parent_attr_target(tool: Any) -> Any:
    """Return the object whose parent_* attrs drive the tool's invocation.

    A wrapped tool (StructuredTool from wrap_tool_with_overrides) exposes a fresh
    coroutine that closes over the ORIGINAL tool's bound _arun. The wrapper itself
    has no `parent_tools`/`parent_llm_config` attrs, so injecting onto the wrapper
    silently drops the parent context — inject onto the underlying instance instead.
    Duck-typed: unwrapped tools (no closure) return the tool itself.
    """
    func = getattr(tool, "coroutine", None) or getattr(tool, "func", None)
    for cell in (getattr(func, "__closure__", None) or []):
        obj = cell.cell_contents
        self_ = getattr(obj, "__self__", None)
        if self_ is not None and hasattr(self_, "parent_llm_config"):
            return self_
    return tool


def unwrap_smart_input(val: Any) -> Any:
    """Recursively convert SmartInput/subclasses of dict and list to standard dict/list."""
    if isinstance(val, dict):
        return {k: unwrap_smart_input(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [unwrap_smart_input(item) for item in val]
    return val


def _derive_error_code(exc: BaseException) -> str:
    """Map an exception to a short, stable error code for grouping failures.

    Heuristics scan both the exception class name and the message so provider
    errors (litellm/OpenAI-style) and standard library errors map consistently.
    Falls back to a generic code so every failure still gets one.
    """
    exc_name = type(exc).__name__.lower()
    msg = (str(exc) or "").lower()

    if isinstance(exc, TimeoutError) or "timeout" in msg or "timeout" in exc_name:
        return "TIMEOUT"
    if isinstance(exc, (ConnectionError, ConnectionRefusedError, ConnectionResetError, ConnectionAbortedError)) or "connection" in exc_name:
        return "CONNECTION"
    if any(k in msg for k in ("unauthorized", "authentication", "invalid_api_key", "401")) or "authentication" in exc_name:
        return "AUTH_ERROR"
    if any(k in msg for k in ("permission", "forbidden", "403")) or "permission" in exc_name:
        return "FORBIDDEN"
    if any(k in msg for k in ("not found", "404")) or "notfound" in exc_name:
        return "NOT_FOUND"
    if any(k in msg for k in ("429", "rate limit", "rate_limit", "quota")) or "ratelimit" in exc_name:
        return "RATE_LIMIT"
    if any(k in msg for k in ("500", "502", "503", "504", "service unavailable", "bad gateway", "internal server error")):
        return "UPSTREAM_ERROR"
    if isinstance(exc, KeyError):
        return "MISSING_KEY"
    if isinstance(exc, (ValueError, TypeError)):
        return "VALIDATION"
    if "json" in exc_name or "parse" in exc_name:
        return "JSON_PARSE"
    return "NODE_ERROR"


def _mark_langfuse_span_error(exc: BaseException) -> None:
    """Mark the currently executing node's Langfuse observation span as ERROR.

    Langfuse v4 (OTEL-based) stores the observation level as a span attribute on
    the span it created for the node's LangChain run. That span is the *current*
    OpenTelemetry span while the node function runs. We use the SDK-sanctioned
    ``get_client().update_current_span(level="ERROR", ...)`` to flip it (falling
    back to direct OTel attribute mutation for older SDKs). The node wrapper
    swallows the exception to keep the graph alive for error routing, which would
    otherwise leave the span looking successful. No-op when Langfuse is not
    installed / not tracing.
    """
    message = str(exc) or type(exc).__name__
    try:
        from langfuse import get_client as _lf_get_client

        client = _lf_get_client()
        if client is not None:
            client.update_current_span(
                level="ERROR", status_message=message[:2000]
            )
    except Exception:
        pass  # Fall through to the direct OTel path below.

    # Belt-and-suspenders fallback + exception recording. Safe even when the
    # SDK helper above already ran (set_attributes is idempotent).
    try:
        from opentelemetry import trace as _otel_trace
        from opentelemetry.trace import Status, StatusCode

        span = _otel_trace.get_current_span()
        if span is None or not getattr(span, "is_recording", lambda: False)():
            return

        span.set_attribute("langfuse.observation.level", "ERROR")
        span.set_attribute("langfuse.observation.status_message", message[:2000])
        span.set_status(Status(StatusCode.ERROR, description=message[:1024]))
        try:
            span.record_exception(exc)
        except Exception:
            pass
    except Exception:
        # Observability must never break workflow execution.
        pass


def wrap_with_error_handling(node_id: str, func: Any, label: Optional[str] = None, node_type: str = "unknown") -> Any:
    """
    Wraps a node execution function with a try-except block to capture 
    failures and route them to the error handler.
    """
    from langchain_core.messages import AIMessage
    import logging
    
    logger = logging.getLogger(__name__)

    async def safe_execute(state: Any):
        # 🧪 Langfuse Trace metadata (Apr 2, 2026)
        # We try to inject node metadata into the tracing context
        from Apps.AgentMesh.Utilities.Observer import settings
        try:
             from langfuse import propagate_attributes
        except ImportError:
             propagate_attributes = None

        node_label = label or node_id
        node_metadata = {
            "user_id": state.get("context", {}).get("user_id", "anonymous") if isinstance(state, dict) else "anonymous",
            "session_id": state.get("context", {}).get("thread_id", "anonymous") if isinstance(state, dict) else "anonymous",
            "metadata": {
                "node_id": node_id,
                "node_label": node_label,
                "node_type": node_type,
                "workflow_id": (state.get("context", {}).get("workflow_id") if isinstance(state, dict) else None)
            },
            "tags": [node_type, node_id, node_label]
        }
        
        # Wrap everything in a context to ensure the attributes apply to nested 
        # calls (like LLM/Tool calls within the agent)
        import contextlib
        cm = propagate_attributes(**node_metadata) if propagate_attributes else contextlib.nullcontext()
        
        with cm:
            import time
            from datetime import datetime

            start_dt = datetime.now().astimezone()
            start_perf = time.perf_counter()

            def _get_timing_info():
                end_dt = datetime.now().astimezone()
                elapsed_sec = time.perf_counter() - start_perf
                total_str = f"{elapsed_sec:.3f}s" if elapsed_sec >= 1.0 else f"{elapsed_sec * 1000:.2f}ms"
                return {
                    "start_time": start_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + start_dt.strftime("%z"),
                    "end_time": end_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + end_dt.strftime("%z"),
                    "total_time": total_str
                }

            try:
                # Check if func is a runnable or plain function
                if hasattr(func, "ainvoke"):
                    result = await func.ainvoke(state)
                elif inspect.iscoroutinefunction(func):
                    result = await func(state)
                else:
                    # Regular function - execute and then check if result is awaitable
                    result = func(state)
                    if inspect.iscoroutine(result):
                        result = await result

                timing_info = _get_timing_info()

                # 🛠️ AUTOMATIC PERSISTENCE (Inspired by n8n Group Results)
                # We extract the "Logic" from the result and save to a persistent node history.
                if isinstance(result, dict):
                    # Clear error state upon successful execution
                    result["error"] = None
                    result["error_source"] = None

                    # 1. Start with any context updates the node produced
                    node_output = {**(result.get("context") or {})}
                    
                    # 2. Extract values from the root dictionary (e.g. 'intent', 'routing_key')
                    # These are often used for logic branching and should be visible in debug
                    ROOT_SYSTEM_KEYS = ("messages", "context", "active_agent", "error", "error_source")
                    for k, v in result.items():
                        if k not in ROOT_SYSTEM_KEYS and not str(k).startswith("__"):
                            node_output[k] = v

                    # 3. Extract latest message content (if it's an agent or producer)
                    msgs = result.get("messages", [])
                    if msgs:
                        # Extract parsed JSON/data from the AI response
                        extracted = get_production_data({"messages": msgs, "context": {}})
                        
                        has_structured_data = False
                        # Merge extracted data, but exclude ephemeral/internal keys
                        for k, v in extracted.items():
                            if k not in ("msg", "data", "hitl_action", "hitl_feedback") and not str(k).startswith("__"):
                                node_output[k] = v
                                has_structured_data = True
                        
                        # Fallback: If no JSON structure was found, store content under 'output'
                        # This ensures text-only nodes like 'A1: Name Intake' are visible in debug
                        if not has_structured_data and extracted.get("msg"):
                            node_output["output"] = extracted.get("msg")
                    
                    # 4. Save to persistent node history
                    incoming_ctx = (state.get("context") if isinstance(state, dict) else getattr(state, "context", {})) or {}
                    history = incoming_ctx.get("nodes", {})
                    
                    # Calculate delta: only record keys that were actually added or modified in this node's execution
                    GLOBAL_PERSISTENT_KEYS = ("thread_id", "workflow_id", "workflow_name", "vfs", "hitl_action", "hitl_feedback")
                    INTERNAL_RESERVED = ("nodes", "msg", "context", "data")
                    
                    clean_res = {}
                    for k, v in node_output.items():
                        if k in GLOBAL_PERSISTENT_KEYS or k in INTERNAL_RESERVED or str(k).startswith("__"):
                            continue
                        
                        is_changed = False
                        if node_type in ("form_node", "camera", "stt_trigger"):
                            is_changed = True
                        elif k not in incoming_ctx:
                            is_changed = True
                        else:
                            try:
                                val1 = unwrap_smart_input(incoming_ctx[k])
                                val2 = unwrap_smart_input(v)
                                if val1 != val2:
                                    is_changed = True
                            except Exception:
                                is_changed = True
                        if is_changed:
                            clean_res[k] = v
                    
                    # Always include timing & audit info for standardized node execution metrics
                    clean_res["node_id"] = node_id
                    clean_res["node_label"] = label or node_id
                    clean_res["status"] = "success"
                    clean_res["status_code"] = 200
                    clean_res["start_time"] = timing_info["start_time"]
                    clean_res["end_time"] = timing_info["end_time"]
                    clean_res["total_time"] = timing_info["total_time"]

                    # Even if no other data, record execution with audit & timing info
                    if not clean_res:
                        clean_res = {
                            "node_id": node_id,
                            "node_label": label or node_id,
                            "status": "success",
                            "status_code": 200,
                            **timing_info
                        }

                    # Merge into context. 'nodes' stores history of every node by ID.
                    new_history = {**history, node_id: {**history.get(node_id, {}), **clean_res}}
                    
                    # Limit size of node history to last 50 executed nodes to save DB checkpoint space
                    if len(new_history) > 50:
                        oldest_keys = list(new_history.keys())[:-50]
                        for old_k in oldest_keys:
                            new_history.pop(old_k, None)
                    
                    # Truncate large strings/lists in node history for entries older than the last 10 executed nodes
                    history_keys = list(new_history.keys())
                    for idx, hist_k in enumerate(history_keys):
                        if idx < len(history_keys) - 10:
                            node_hist = new_history[hist_k]
                            if isinstance(node_hist, dict):
                                pruned_node_hist = {}
                                for hk, hv in node_hist.items():
                                    if isinstance(hv, str) and len(hv) > 1000 and "base64" not in hk.lower():
                                        pruned_node_hist[hk] = hv[:1000] + "... [truncated]"
                                    elif isinstance(hv, list) and len(hv) > 100:
                                        pruned_node_hist[hk] = hv[:100] + ["... [truncated]"]
                                    elif isinstance(hv, dict) and len(hv) > 100:
                                        pruned_node_hist[hk] = {dk: dv for i, (dk, dv) in enumerate(hv.items()) if i < 100}
                                    else:
                                        pruned_node_hist[hk] = hv
                                new_history[hist_k] = pruned_node_hist

                    final_ctx = {
                        **incoming_ctx, # ✅ Fix: Preserve all previous variables
                        **node_output,  # ✅ Promotion: Node results are available at root
                        "nodes": new_history,
                        "hitl_action": None, # Resolve HITL
                        "hitl_feedback": None
                    }
                    result["context"] = final_ctx

                # 🧪 Toxicity scoring (fire-and-forget — never blocks the node)
                try:
                    if isinstance(result, dict) and isinstance(result.get("context"), dict):
                        _tox_ctx = result.get("context") or {}
                        _tox_msgs = result.get("messages") or []
                        _tox_text = ""
                        # Score the last PRODUCER (AI) message, skipping trailing
                        # ToolMessage/FunctionMessage/SystemMessage so tool payloads
                        # don't get graded as model output.
                        if _tox_msgs:
                            for _msg in reversed(_tox_msgs):
                                _mtype = getattr(_msg, "type", None)
                                if _mtype is None and isinstance(_msg, dict):
                                    _mtype = _msg.get("type")
                                if _mtype in ("tool", "function", "system"):
                                    continue
                                _content = getattr(_msg, "content", None)
                                if _content is None and isinstance(_msg, dict):
                                    _content = _msg.get("content")
                                if _content:
                                    _tox_text = _content if isinstance(_content, str) else str(_content)
                                    if _tox_text.strip():
                                        break
                        if not isinstance(_tox_text, str):
                            _tox_text = str(_tox_text)
                        if _tox_text.strip():
                            from Apps.AgentMesh.Utilities.toxicity import schedule_toxicity_score

                            schedule_toxicity_score(
                                _tox_ctx.get("thread_id", "anonymous"),
                                node_id,
                                node_type,
                                _tox_text,
                            )
                except Exception as _tox_exc:
                    logger.debug(f"Toxicity scheduling failed for node '{node_id}': {_tox_exc}")

                return result
            except Exception as e:
                # Do not intercept LangGraph control flow interrupts
                from langgraph.errors import GraphInterrupt, Interrupt, NodeInterrupt
                if isinstance(e, (GraphInterrupt, Interrupt, NodeInterrupt)):
                    raise
                
                # Allow explicit UI validation failures to bubble up 
                # to trigger a 500 error for frontend toast interception
                if isinstance(e, ValueError) and "Image validation error" in str(e):
                    raise
                
                display_name = label or node_id
                logger.error(f"Node '{display_name}' ({node_id}) failed: {str(e)}")
                
                # Prepare error state, preserving existing retry metadata
                existing_error = (state.get("error") if isinstance(state, dict) else getattr(state, "error", {})) or {}
                retry_count = existing_error.get("retry_count", 0)
                
                error_msg = f"[SYSTEM ERROR] Node '{node_id}' failed: {str(e)}"

                # Stable error code for grouping failures + mark this node's
                # Langfuse span as ERROR (the exception is swallowed below to keep
                # the graph alive for error routing, so the span would otherwise
                # be recorded as successful).
                error_code = _derive_error_code(e)
                _mark_langfuse_span_error(e)
                
                # Record timing & audit info in context node history even on failure
                timing_info = _get_timing_info()
                audit_error_info = {
                    "node_id": node_id,
                    "node_label": label or node_id,
                    "status": "failed",
                    "status_code": 500,
                    "error_code": error_code,
                    "error": str(e),
                    **timing_info
                }

                incoming_ctx = (state.get("context") if isinstance(state, dict) else getattr(state, "context", {})) or {}
                history = incoming_ctx.get("nodes", {})
                new_history = {**history, node_id: {**history.get(node_id, {}), **audit_error_info}}

                return {
                    "messages": [AIMessage(content=error_msg)],
                    "error": {
                        "message": str(e),
                        "type": type(e).__name__,
                        "error_code": error_code,
                        "retry_count": retry_count, # Persist for the router
                    },
                    "error_source": node_id,
                    "active_agent": node_id,
                    "context": {
                        **incoming_ctx,
                        "nodes": new_history
                    }
                }
            
    # Preserve name for LangGraph debugging and Langfuse tracing
    # Sanitizing label for valid function name if needed, but LangGraph/Langfuse usually handle spaces
    safe_execute.__name__ = label or f"Node_{node_id}"
    return safe_execute


# ═══════════════════════════════════════════════════════════════════════════
# MEMORY MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════
# Fallback token budget when the node's `memory.max_context_tokens` is unset.
# Keeps DeepAgents and windowed agents from ever exceeding the model window.
DEFAULT_MAX_CONTEXT_TOKENS = 32000


def _is_human(msg: Any) -> bool:
    return isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human"


def _is_ai(msg: Any) -> bool:
    return isinstance(msg, AIMessage) or getattr(msg, "type", "") in ("ai", "assistant")


def _is_tool(msg: Any) -> bool:
    return isinstance(msg, ToolMessage) or getattr(msg, "type", "") == "tool"


def _is_system(msg: Any) -> bool:
    return isinstance(msg, SystemMessage) or getattr(msg, "type", "") == "system"


def _content_str(msg: Any) -> str:
    """Normalize a message's content (string, block list, or other) into plain text."""
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("text"):
                    parts.append(block["text"])
        return "\n".join(parts)
    return str(content)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars / 4). Sufficient for context budgeting."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def extract_latest_user_query(messages: List[Any]) -> str:
    """Return the content of the most recent HumanMessage in the list."""
    for msg in reversed(messages):
        if _is_human(msg):
            content = _content_str(msg)
            if content.strip():
                return content
    return ""


DEFAULT_FALLBACK_USER_INPUT = (
    "Please execute your instructions based on the available workflow context "
    "and produce your response."
)


def ensure_non_empty_user_input(messages_for_agent: List[Any]) -> List[Any]:
    """Guarantee the history ends on a NON-EMPTY user turn.

    Some providers (e.g. Groq) reject requests whose only user message is empty,
    and an empty prompt yields empty model responses. Returns a copy of the list
    with an empty trailing user turn replaced by (or a missing one padded with) a
    default instruction.
    """
    msgs = list(messages_for_agent)
    if not msgs:
        return [HumanMessage(content=DEFAULT_FALLBACK_USER_INPUT)]
    last = msgs[-1]
    if _is_human(last):
        if not _content_str(last).strip():
            msgs[-1] = HumanMessage(content=DEFAULT_FALLBACK_USER_INPUT)
    else:
        msgs.append(HumanMessage(content=DEFAULT_FALLBACK_USER_INPUT))
    return msgs


def _trim_to_token_budget(msgs: List[Any], max_tokens: int):
    """Drop the oldest messages until total estimated tokens fit `max_tokens`.

    Only ever removes a PREFIX of the list, so in-list tool-call sequences stay
    intact. Leading ToolMessages orphaned by the cut are dropped as well.
    Returns (kept, dropped).
    """
    if max_tokens <= 0:
        return msgs, []
    total = sum(estimate_tokens(_content_str(m)) for m in msgs)
    if total <= max_tokens:
        return msgs, []
    dropped = []
    idx = 0
    while idx < len(msgs) and total > max_tokens:
        m = msgs[idx]
        total -= estimate_tokens(_content_str(m))
        dropped.append(m)
        idx += 1
    kept = msgs[idx:]
    # Tool-safe boundary: never begin the window on an orphaned ToolMessage
    while kept and _is_tool(kept[0]):
        dropped.append(kept.pop(0))
    return kept, dropped


def _strip_trailing_orphan_tool_call(msgs: List[Any]) -> List[Any]:
    """Remove a trailing AIMessage that still carries unresolved tool_calls.

    A history ending on an unexecuted tool call is invalid for most chat APIs when
    passed to a fresh `prompt | llm.bind_tools()` chain.
    """
    while msgs:
        last = msgs[-1]
        if _is_ai(last) and getattr(last, "tool_calls", None):
            msgs.pop()
        else:
            break
    return msgs


async def _llm_summarize_turns(dropped_msgs: List[Any], node_config: Any, existing_summary: str = ""):
    """Merge newly-dropped turns into a running MEMORY fact-block using the node's LLM.

    The output is formatted as explicit `- <label>: <value>` facts (not a narrative
    recap) because models are far more likely to actually use structured facts when
    answering questions about the user. When `existing_summary` is provided (a prior
    `- ...` fact-block), it is included in the prompt so the model MERGES the new
    turns into the accumulated memory instead of producing a standalone recap.
    Returns None on any failure so the caller can degrade gracefully to the raw-dump
    fallback.
    """
    from Apps.AgentMesh.Utilities.LLM.llm_factory import init_chat_model
    try:
        llm_config = getattr(node_config, "llm", None)
        if not llm_config:
            return None
        summarizer = init_chat_model(llm_config)

        lines = []
        for m in dropped_msgs:
            role = "assistant" if _is_ai(m) else ("user" if _is_human(m) else ("tool" if _is_tool(m) else "system"))
            text = _content_str(m)
            if text.strip():
                lines.append(f"{role}: {text[:500]}")
        if not lines:
            return None

        # Extract the prior fact lines (if any) so the model merges, not replaces.
        prior_facts = []
        for ln in (existing_summary or "").splitlines():
            ln = ln.strip()
            if ln.startswith("-") and len(ln) > 2:
                prior_facts.append(ln)

        facts = []
        CHUNK = 40
        for i in range(0, len(lines), CHUNK):
            chunk_lines = lines[i:i + CHUNK]
            prior_block = "\n".join(prior_facts) if prior_facts else "(none yet)"
            prompt = (
                "You are a conversation-memory engine maintaining a RUNNING fact list "
                "about the USER (name, background, preferences, projects, decisions, "
                "personal details, important context, open questions).\n\n"
                "EXISTING MEMORY FACTS (keep all that are still valid):\n" + prior_block + "\n\n"
                "NEW CONVERSATION TURNS (extract NEW facts; do not repeat facts already "
                "in EXISTING MEMORY FACTS):\n" + "\n".join(chunk_lines) + "\n\n"
                "Output the COMPLETE merged fact list, one fact per line, in the form: "
                "- <label>: <value>. Keep each fact short. Do NOT repeat greeting text or "
                "conversational niceties. If nothing new was revealed, still keep the "
                "existing facts."
            )
            import asyncio as _asyncio
            resp = await _asyncio.wait_for(summarizer.ainvoke(prompt), timeout=60)
            text = resp if isinstance(resp, str) else getattr(resp, "content", "")
            if isinstance(text, str):
                for ln in text.splitlines():
                    ln = ln.strip()
                    ln = ln[1:].strip() if ln.startswith("-") else ln
                    lowered = ln.lower()
                    if ln and not lowered.startswith(("here are", "summary", "the conversation", "facts about", "conversation turns", "existing memory")):
                        facts.append(f"- {ln[:200]}")

        # Fall back to the prior facts if the LLM returned nothing usable.
        if not facts and prior_facts:
            facts = prior_facts

        # Dedup identical facts while preserving order
        seen = set()
        unique_facts = []
        for f in facts:
            key = f.lower()
            if key not in seen:
                seen.add(key)
                unique_facts.append(f)

        if unique_facts:
            return f"[MEMORY OF USER (from previous conversation, {len(dropped_msgs)} new turns)]:\n" + \
                   "\n".join(unique_facts) + "\n---"
    except Exception as e:
        logger.warning(f"LLM conversation summarization failed for {getattr(node_config, 'id', '?')}: {e}")
    return None


async def prepare_agent_memory(messages: List[Any], node_config: Any, state: Optional[Dict[str, Any]] = None) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Filter and prepare memory history based on node configuration.
    Returns a tuple of (messages_for_agent, memory_summary_update).

    Supports:
    - Memory Scope: 'unified' (default) or 'isolated'
    - Memory Type: 'none' (stateless), 'buffer_memory' (message windowing),
      'summary_buffer_memory' (LLM-summarized long-term context), or
      'vector_memory' (retrieval, orchestrated by the caller).
    - Running Summary: Persists accumulated fact summary across turns in `state['memory_summary']`
    - Token budget: `max_context_tokens` (default 32000) always bounds context.
    """
    mem_config = getattr(node_config, "memory", None) or {}
    mem_type = mem_config.get("type_name", "buffer_memory")
    window_size = mem_config.get("window_size", 10)
    mem_scope = mem_config.get("scope", "unified")
    max_tokens = int(mem_config.get("max_context_tokens", 0) or 0) or DEFAULT_MAX_CONTEXT_TOKENS

    messages_for_agent = list(messages)
    node_id = getattr(node_config, "id", "agent")
    memory_summary_state = (state.get("memory_summary") or {}) if isinstance(state, dict) else {}
    node_summary_data = memory_summary_state.get(node_id) or {}
    existing_summary_text = node_summary_data.get("summary", "")
    summary_update = {}

    # 1. Apply Scope Filtering if "isolated"
    if mem_scope == "isolated":
        logger.debug(f"Applying isolated memory scope for agent: {node_config.id}")
        isolated_messages = []
        allowed_tool_ids = set()

        for msg in messages:
            if _is_human(msg):
                isolated_messages.append(msg)
            elif _is_ai(msg):
                metadata = getattr(msg, "response_metadata", {}) or {}
                add_kwargs = getattr(msg, "additional_kwargs", {}) or {}
                active_agent = metadata.get("active_agent") or add_kwargs.get("active_agent")
                if active_agent == node_config.id:
                    isolated_messages.append(msg)
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tc_id = tc.get("id") or getattr(tc, 'id', None)
                            if tc_id:
                                allowed_tool_ids.add(tc_id)
            elif _is_tool(msg):
                tc_id = getattr(msg, "tool_call_id", None)
                if tc_id in allowed_tool_ids:
                    isolated_messages.append(msg)

        messages_for_agent = isolated_messages

    # 2. Apply Memory Type (Stateless vs Windowing vs Summarization)
    if mem_type == "none":
        logger.debug(f"Applying stateless/no-memory for agent: {node_config.id}")
        # Oneshot: keep system instructions + only the latest NON-EMPTY user input.
        system_msgs = [m for m in messages_for_agent if _is_system(m)]
        human_msgs = [m for m in messages_for_agent if _is_human(m) and _content_str(m).strip()]
        messages_for_agent = system_msgs + ([human_msgs[-1]] if human_msgs else [])
    else:

        is_deep_agent = getattr(node_config, "type", "llm_agent") == "llm_deep_agent"
        system_msgs = [m for m in messages_for_agent if _is_system(m)]
        non_system_msgs = [m for m in messages_for_agent if not _is_system(m)]

        dropped_msgs = []
        if (not is_deep_agent) and mem_type in ("buffer_memory", "summary_buffer_memory") and len(non_system_msgs) > window_size:
            logger.debug(f"Applying memory management ({mem_type}): window size {window_size} of total {len(non_system_msgs)}")
            recent_msgs = non_system_msgs[-window_size:]
            # Tool-safe adjustment: never start the window on an orphaned ToolMessage
            while recent_msgs and _is_tool(recent_msgs[0]):
                recent_msgs.pop(0)
            dropped_msgs = non_system_msgs[:-len(recent_msgs)] if recent_msgs else non_system_msgs
        else:
            # DeepAgents keep their planner sequences (no aggressive count window),
            # but they are still bounded by the token budget below.
            recent_msgs = non_system_msgs

        # Token-budget enforcement for ALL memory types (incl. DeepAgents / vector_memory)
        if max_tokens > 0:
            recent_msgs, budget_dropped = _trim_to_token_budget(recent_msgs, max_tokens)
            if budget_dropped:
                dropped_msgs = dropped_msgs + budget_dropped
                logger.debug(f"Token-budget trim for {node_config.id}: dropped {len(budget_dropped)} messages to fit {max_tokens} tokens")

        if mem_type == "summary_buffer_memory" and (dropped_msgs or existing_summary_text):
            # Message-identity tracking: skip turns that were already summarized on prior turns
            already_summarized_ids = set(node_summary_data.get("summarized_ids") or [])
            new_unsummarized_msgs = []
            for m in dropped_msgs:
                msg_id = getattr(m, "id", None) or f"{type(m).__name__}_{hash(_content_str(m))}"
                if msg_id not in already_summarized_ids:
                    new_unsummarized_msgs.append((msg_id, m))

            if new_unsummarized_msgs:
                new_msg_ids = [item[0] for item in new_unsummarized_msgs]
                unsummarized_msgs_list = [item[1] for item in new_unsummarized_msgs]
                summary_notice = await _llm_summarize_turns(unsummarized_msgs_list, node_config, existing_summary=existing_summary_text)
                if summary_notice:
                    system_msgs.append(SystemMessage(content=summary_notice))
                    # Extract pure facts portion to store in memory_summary state
                    fact_lines = [line for line in summary_notice.splitlines() if line.strip().startswith("-")]
                    clean_summary = "\n".join(fact_lines) if fact_lines else summary_notice
                    updated_summarized_ids = list(already_summarized_ids.union(new_msg_ids))
                    summary_update[node_id] = {
                        "summary": clean_summary,
                        "summarized_ids": updated_summarized_ids,
                        "turn_count": len(updated_summarized_ids),
                    }
                    logger.info(f"LLM-summarized running memory buffer for {node_config.id} covering {len(new_msg_ids)} NEW turns (total summarized: {len(updated_summarized_ids)})")
                else:
                    # Fallback: bounded raw dump if LLM summarizer is unavailable
                    summary_lines = []
                    for m in dropped_msgs:
                        role = "user" if _is_human(m) else ("assistant" if _is_ai(m) else ("tool" if _is_tool(m) else "system"))
                        content_str = _content_str(m)
                        if content_str.strip():
                            brief_content = content_str[:250] + "..." if len(content_str) > 250 else content_str
                            summary_lines.append(f"- {role}: {brief_content}")
                    if summary_lines:
                        summary_text = "\n".join(summary_lines)
                        system_msgs.append(SystemMessage(
                            content=f"[MEMORY OF USER (from previous conversation, {len(dropped_msgs)} earlier turns)]:\n{summary_text}\n---"
                        ))
                        logger.info(f"Generated fallback summary buffer for {node_config.id} covering {len(dropped_msgs)} dropped turns")
            elif existing_summary_text:
                # Inject prior accumulated summary without calling LLM (no new unsummarized turns)
                system_msgs.append(SystemMessage(
                    content=f"[MEMORY OF USER (accumulated facts from prior conversation)]:\n{existing_summary_text}\n---"
                ))

        messages_for_agent = system_msgs + recent_msgs
        # Never pass an unresolved tool call as the final message
        messages_for_agent = _strip_trailing_orphan_tool_call(messages_for_agent)

    return messages_for_agent, summary_update


# Per-thread (per-collection) guards for vector-memory operations. The HuggingFace
# embedding model is a process-global singleton (loaded once); running embed+query or
# embed+upsert from multiple worker threads at once can race on first-load, and
# concurrent upsert/prune on the SAME thread-collection can interleave. Keying the lock
# by thread_id keeps unrelated conversations parallel while still serializing all I/O
# on a given collection (mirrors the atomicity the LangGraph checkpointer gives graph
# state, without a global bottleneck).
_VECTOR_MEMORY_LOCK_GUARD = threading.Lock()
_VECTOR_MEMORY_LOCKS: Dict[str, threading.RLock] = {}


def _vector_memory_lock(thread_id: str) -> threading.RLock:
    """Return the per-thread re-entrant lock, creating it on first use."""
    with _VECTOR_MEMORY_LOCK_GUARD:
        lock = _VECTOR_MEMORY_LOCKS.get(thread_id)
        if lock is None:
            lock = threading.RLock()
            _VECTOR_MEMORY_LOCKS[thread_id] = lock
        return lock


def _memory_doc_id(thread_id: str, node_id: str, user_query: str) -> str:
    """Deterministic, backend-portable document id for a conversation turn.

    A valid UUIDv5 (not a raw 40-char sha1 hexdigest) so it works with BOTH
    Chroma (arbitrary string ids) and Qdrant (which requires real UUIDs for
    point ids in local mode).
    """
    key = f"{thread_id}|{node_id}|{user_query}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def retrieve_vector_memory(messages_for_agent: List[Any], node_config: Any, thread_id: str) -> str:
    """Retrieve relevant past turns for THIS node from the thread's vector collection.

    Pure synchronous body. Callers that run inside an event loop should use
    `retrieve_vector_memory_async`, which offloads this (and the shared embedding
    model) to a worker thread under the per-thread lock so it never blocks the loop.
    """
    try:
        from Apps.AgentMesh.Utilities.Tools.RagTools.vector_store_factory import query_vector_store
        from Apps.AgentMesh.Utilities.Tools.RagTools.embedding_factory import init_embedding_model

        mem_config = getattr(node_config, "memory", None) or {}
        query = extract_latest_user_query(messages_for_agent)
        if not query:
            return ""

        collection_name = f"thread_{thread_id}"
        embeddings = init_embedding_model(mem_config.get("embedding_model") or None)
        vector_store_type = mem_config.get("vector_db", "chroma")
        top_k = mem_config.get("top_k", 3)
        search_type = mem_config.get("search_type", "similarity")
        # Hybrid (recency-aware) retrieval: hard raw-similarity floor + recency re-ranking.
        threshold = mem_config.get("retrieval_threshold", 0.2)
        recency_decay = mem_config.get("recency_decay", 0.3)
        decay_halflife_hours = mem_config.get("decay_halflife_hours", 168.0)

        docs = query_vector_store(
            query=query,
            vector_store_type=vector_store_type,
            embedding_model=embeddings,
            collection_name=collection_name,
            k=top_k,
            search_type=search_type,
            where={"node_id": node_config.id},
            threshold=threshold,
            recency_decay=recency_decay,
            decay_halflife_hours=decay_halflife_hours,
        )
        if docs:
            return "\n\n[RELEVANT PAST CONVERSATION MEMORY (RETRIEVED FROM VECTOR STORE)]:\n" + \
                   "\n".join(f"- {d.page_content}" for d in docs) + "\n---"
    except Exception as e:
        logger.error(f"Vector memory retrieval failed for agent {getattr(node_config, 'id', '?')}: {e}")
    return ""


def persist_vector_memory(user_query: str, response_content: Any, node_config: Any, thread_id: str) -> bool:
    """Auto-persist a conversation turn to the thread's vector store.

    Uses a deterministic per-turn id (thread + node + user query) so re-runs and
    retries UPSERT instead of duplicating. Prunes the node's oldest entries to
    `max_memory_entries` (default 200).
    """
    if not user_query or not response_content:
        return False
    try:
        import time
        from langchain_core.documents import Document
        from Apps.AgentMesh.Utilities.Tools.RagTools.vector_store_factory import store_in_vector_store, prune_vector_store
        from Apps.AgentMesh.Utilities.Tools.RagTools.embedding_factory import init_embedding_model

        mem_config = getattr(node_config, "memory", None) or {}
        node_id = getattr(node_config, "id", "")
        content = f"User: {user_query}\nAgent ({node_id}): {str(response_content)[:1000]}"
        # Stable UUIDv5 id => the same user turn always maps to the same document
        # (upsert, no dupes) and is accepted by both Chroma and Qdrant.
        mem_id = _memory_doc_id(thread_id, node_id, user_query)

        doc = Document(
            page_content=content,
            metadata={
                "thread_id": thread_id,
                "node_id": node_id,
                "timestamp": time.time(),
            },
        )
        embeddings = init_embedding_model(mem_config.get("embedding_model") or None)
        vector_store_type = mem_config.get("vector_db", "chroma")
        collection_name = f"thread_{thread_id}"
        store_in_vector_store([doc], vector_store_type, embeddings, collection_name=collection_name, ids=[mem_id])

        cap = int(mem_config.get("max_memory_entries", 200) or 0)
        if cap > 0:
            try:
                pruned = prune_vector_store(
                    vector_store_type, embeddings, collection_name,
                    where={"node_id": node_id}, keep=cap,
                )
                if pruned:
                    logger.info(f"[VECTOR MEMORY] Pruned {pruned} stale entries for {node_id} (cap {cap})")
            except Exception as pe:
                logger.warning(f"[VECTOR MEMORY] Prune failed for {node_id}: {pe}")

        logger.info(f"[VECTOR MEMORY] Auto-persisted conversation turn for agent {node_id} in collection {collection_name}")
        return True
    except Exception as persist_err:
        logger.error(f"[VECTOR MEMORY] Failed to auto-persist turn for agent {getattr(node_config, 'id', '?')}: {persist_err}")
        return False


async def retrieve_vector_memory_async(messages_for_agent: List[Any], node_config: Any, thread_id: str) -> str:
    """Async variant of `retrieve_vector_memory` that never blocks the event loop.

    Offloads embedding + similarity search to a worker thread so the agent loop stays
    responsive while other nodes run. Serialized per-thread by `_vector_memory_lock`
    because the embedding model is a process-global singleton.
    """
    if not _is_async_context():
        return retrieve_vector_memory(messages_for_agent, node_config, thread_id)

    lock = _vector_memory_lock(thread_id)

    def _run():
        with lock:
            return retrieve_vector_memory(messages_for_agent, node_config, thread_id)

    return await asyncio.to_thread(_run)


async def persist_vector_memory_async(user_query: str, response_content: Any, node_config: Any, thread_id: str) -> bool:
    """Async variant of `persist_vector_memory` that never blocks the event loop."""
    if not user_query or not response_content:
        return False
    if not _is_async_context():
        return persist_vector_memory(user_query, response_content, node_config, thread_id)

    lock = _vector_memory_lock(thread_id)

    def _run():
        with lock:
            return persist_vector_memory(user_query, response_content, node_config, thread_id)

    return await asyncio.to_thread(_run)


def _is_async_context() -> bool:
    """Return True when we are inside a running asyncio event loop."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


