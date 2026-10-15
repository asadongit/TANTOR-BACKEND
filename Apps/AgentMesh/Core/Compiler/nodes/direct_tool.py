"""
Direct Tool Node Builder
Handles direct invocation of tools in the workflow context without LLM intervention.
"""

import logging
import inspect
import re
from typing import Dict, Any, Callable
from langchain_core.messages import AIMessage

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Core.ToolRegistry import registry
from .utils import get_production_data, interpolate, get_nested_val

logger = logging.getLogger(__name__)


def interpolate_value(val: Any, context: Dict[str, Any]) -> Any:
    """Recursively interpolates templates in arguments, preserving type on exact matches."""
    if isinstance(val, str):
        # If the string is exactly a single template variable (e.g. "{{some_key}}"),
        # retrieve the raw value directly from context to preserve its type.
        match = re.match(r"^\{\{([\w\.]+)\}\}$", val)
        if match:
            var_path = match.group(1)
            result = get_nested_val(context, var_path)
            if result is not None:
                return result
        return interpolate(val, context)
    elif isinstance(val, dict):
        return {k: interpolate_value(v, context) for k, v in val.items()}
    elif isinstance(val, list):
        return [interpolate_value(item, context) for item in val]
    return val


def build_direct_tool_node(node_config: NodeConfig) -> Callable:
    """Build a node that executes a tool directly and saves the result to context."""
    tool_name = node_config.direct_tool_name
    if not tool_name:
        raise ValueError(f"Direct Tool node '{node_config.id}' does not have a tool configured.")

    tool = registry.get_tool(tool_name)
    if not tool:
        raise ValueError(f"Tool '{tool_name}' not found in registry.")

    async def execute_direct_tool(state: AgentState) -> dict:
        context = state.get("context", {}) or {}
        thread_id = state.get("thread_id") or context.get("thread_id") or "default_thread"

        # 1. Resolve production variables (context, messages, etc.)
        full_scope = get_production_data(state)

        # 2. Interpolate tool arguments from full scope
        raw_args = node_config.tool_arguments or {}
        tool_args = interpolate_value(raw_args, full_scope)
        logger.info(f"Direct Tool Node {node_config.id}: Executing {tool_name} with args {tool_args}")

        # 3. Build config
        config = {"configurable": {"context": context, "thread_id": thread_id}}

        # 4. Invoke the tool
        try:
            wrapped_func = getattr(tool, 'coroutine', None) or getattr(tool, 'func', None)
            is_async = getattr(tool, 'coroutine', None) is not None

            if wrapped_func:
                final_input = {**tool_args}
                try:
                    sig = inspect.signature(wrapped_func)
                    if 'config' in sig.parameters:
                        final_input["config"] = config
                except Exception:
                    pass

                if is_async or inspect.iscoroutinefunction(wrapped_func):
                    result = await wrapped_func(**final_input)
                else:
                    result = wrapped_func(**final_input)
            else:
                # Fallback to standard invoke
                result = await tool.ainvoke(tool_args)

            logger.info(f"Direct Tool Node {node_config.id}: Executed successfully.")
        except Exception as e:
            logger.exception(f"Direct Tool Node {node_config.id} execution failed")
            # Raise to trigger on_error transitions or global failure handlers
            raise e

        # Try parsing stringified dictionary/list results into actual structured data
        if isinstance(result, str):
            try:
                from json_repair import loads
                parsed = loads(result)
                if isinstance(parsed, (dict, list)):
                    result = parsed
            except Exception:
                try:
                    import ast
                    parsed = ast.literal_eval(result)
                    if isinstance(parsed, (dict, list)):
                        result = parsed
                except Exception:
                    pass

        # 5. Save the result to context
        dest_key = node_config.destination_key or f"tool_result_{node_config.id}"
        
        # Update the context dictionary
        new_context = {**context, dest_key: result}
        clean_context = {k: v for k, v in new_context.items() if k not in ("msg", "data")}

        # Create trace / execution message
        import json
        if isinstance(result, (dict, list)):
            try:
                content = json.dumps(result, ensure_ascii=False)
            except Exception:
                content = str(result)
        else:
            content = str(result)

        return {
            "messages": [AIMessage(content=content, response_metadata={"active_agent": node_config.id})],
            "active_agent": node_config.id,
            "context": clean_context
        }

    return execute_direct_tool
