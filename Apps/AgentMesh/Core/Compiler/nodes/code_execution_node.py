import ast
import datetime
import decimal
import json
import logging
import math
import re
import statistics
from typing import Callable, Any, Dict, List

from langchain_core.messages import AIMessage
from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState

logger = logging.getLogger(__name__)

_SAFE_BUILTINS = {
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "isinstance": isinstance,
    "True": True,
    "False": False,
    "None": None,
}

_FORBIDDEN_KEYWORDS = [
    "import",
    "exec",
    "eval",
    "open",
    "os.",
    "sys.",
    "subprocess",
    "__",
]

# Standard-library modules the sandbox exposes, pre-imported.
#
# The blanket `import` ban exists to stop code reaching the operating system —
# `import os`, `import subprocess`. That is still enforced: the keyword remains
# forbidden and nothing here can open a file, a socket or a process.
#
# But the ban was written as "no imports at all", which also caught modules that
# are pure computation. Two of those are not optional for financial work:
#
#   decimal   — money must not be held in binary floating point. 0.1 + 0.2 is
#               0.30000000000000004, and across a few hundred line items that
#               error accumulates until a balance sheet fails to reconcile by a
#               fraction of a paisa — reported as an integrity failure against a
#               customer whose figures were fine.
#   datetime  — real date arithmetic. Deriving a daily balance series, or a
#               financial-year boundary, from string slicing works until it does
#               not.
#
# Exposed as ready-made objects rather than allowing the `import` statement, so
# the keyword check stays absolute and there is no path to an arbitrary module.
_SAFE_MODULES = {
    "decimal": decimal,
    "Decimal": decimal.Decimal,
    "datetime": datetime,
    "date": datetime.date,
    "timedelta": datetime.timedelta,
    "math": math,
    "statistics": statistics,
    "json": json,
    "re": re,
}


def _validate_code(code: str):
    code_lower = code.lower()
    for bad in _FORBIDDEN_KEYWORDS:
        if bad in ("__", "os.", "sys."):
            if bad in code_lower:
                raise ValueError(f"Forbidden keyword in code: '{bad}'")
        else:
            if re.search(rf"\b{re.escape(bad)}\b", code_lower):
                raise ValueError(f"Forbidden keyword in code: '{bad}'")
    try:
        ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Invalid Python code syntax: {e}")


def _build_execution_scope(
    input_data: Any, input_items: List[Any], context: Dict[str, Any]
):
    return {
        **_SAFE_BUILTINS,
        **_SAFE_MODULES,
        "input_data": input_data,
        "input_items": input_items,
        "context": context,
    }


def _execute_block(code: str, scope: dict):
    local_vars = {**scope, "result": None, "output_data": None, "output": None}
    exec(code, {"__builtins__": {}}, local_vars)
    return local_vars


def _execute_code(code: str, scope: dict):
    local_vars = _execute_block(code, scope)
    for key in ("result", "output_data", "output"):
        if local_vars.get(key) is not None:
            return local_vars[key]
    return None


def build_code_execution_node(node_config: NodeConfig) -> Callable:
    language = getattr(node_config, "ce_language", "python")
    code = getattr(node_config, "ce_code", "") or ""
    input_key = getattr(node_config, "ce_input_key", None)
    output_key = getattr(node_config, "ce_output_key", "code_result") or "code_result"
    mode = getattr(node_config, "ce_mode", "run_once")
    block_mode = getattr(node_config, "ce_block_mode", "single")
    code_blocks = getattr(node_config, "ce_code_blocks", None) or []

    enabled_blocks = [
        b for b in code_blocks
        if isinstance(b, dict) and b.get("enabled", True) and (b.get("code") or "").strip()
    ]

    async def execute_code_execution(state: AgentState):
        logger.info(
            f"Code Execution Node {node_config.id}: executing {language} code ({mode} mode, {block_mode} block mode)"
        )

        if language != "python":
            raise ValueError(
                f"Unsupported language: '{language}'. Currently only 'python' is supported."
            )

        context: Dict[str, Any] = state.get("context", {}) or {}

        # Resolve input data safely from context or root state
        if input_key and str(input_key).strip():
            input_raw = context.get(input_key, state.get(input_key, {}))
        else:
            input_raw = context.get("latest_api_result", context)

        def _try_parse_json(val):
            if isinstance(val, str):
                s = val.strip()
                if s.startswith("```"):
                    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
                    s = re.sub(r"\s*```$", "", s).strip()
                if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                    try:
                        return json.loads(s)
                    except Exception:
                        pass
            return val

        input_raw = _try_parse_json(input_raw)
        if isinstance(input_raw, list):
            input_raw = [_try_parse_json(x) for x in input_raw]

        input_items = input_raw if isinstance(input_raw, list) else [input_raw]

        def _run_scripts(scripts, names, scope):
            if block_mode == "multi":
                block_results = {}
                last_result = None
                for name, script in zip(names, scripts):
                    res = _execute_code(script, scope)
                    block_results[name] = res
                    if res is not None:
                        last_result = res
                return {"block_results": block_results, "last_result": last_result}
            return _execute_code(scripts[0], scope)

        scripts = [b["code"] for b in enabled_blocks] if block_mode == "multi" else [code]
        names = (
            [b.get("name") or b.get("id") or f"block_{i}" for i, b in enumerate(enabled_blocks)]
            if block_mode == "multi"
            else []
        )
        for script in scripts:
            _validate_code(script)

        scope = _build_execution_scope(input_raw, input_items, context)

        try:
            if mode == "run_for_each":
                results = []
                for idx, item in enumerate(input_items):
                    item_scope = {
                        **scope,
                        "input_data": item,
                        "input_items": input_items,
                        "_index": idx,
                    }
                    res = _run_scripts(scripts, names, item_scope)
                    results.append(res)
                final_result = results
            else:
                final_result = _run_scripts(scripts, names, scope)

            new_context = dict(context)
            new_context[output_key] = final_result
            new_context["latest_code_result"] = final_result

            # Format chat panel bubble according to AgentMesh standard
            try:
                result_str = json.dumps(final_result, indent=2, default=str)
            except Exception:
                result_str = str(final_result)

            chat_message = (
                f"✅ Code Execution complete\n"
                f"Language: `{language}` | Mode: `{mode}` | Blocks: `{block_mode}`\n"
                f"Output: `{output_key}` → {type(final_result).__name__}\n\n"
                f"```python\n{result_str[:2000]}{'...' if len(result_str) > 2000 else ''}\n```"
            )

            logger.info(f"Code Execution Node {node_config.id}: completed -> {output_key}")

            return {
                "messages": [
                    AIMessage(
                        content=chat_message,
                        name=node_config.id,
                        response_metadata={"active_agent": node_config.id},
                    )
                ],
                "context": new_context,
                "active_agent": node_config.id,
            }

        except Exception as e:
            logger.exception(f"Code Execution Node {node_config.id} failed: {e}")
            error_msg = f"❌ Code Execution Error: {str(e)}"
            return {
                "messages": [
                    AIMessage(
                        content=error_msg,
                        name=node_config.id,
                        response_metadata={"active_agent": node_config.id},
                    )
                ],
                "context": {
                    **context,
                    output_key: None,
                    f"{output_key}_error": str(e),
                },
                "active_agent": node_config.id,
            }

    setattr(execute_code_execution, "name", node_config.id)
    return execute_code_execution


