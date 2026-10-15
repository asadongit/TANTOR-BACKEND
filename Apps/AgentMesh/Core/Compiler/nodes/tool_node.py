"""
Tool Node Builder
Handles execution of tool nodes with context injection.
"""

import logging
import inspect
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Core.ToolRegistry import registry
from .utils import process_tools_with_overrides, build_subagent_tool_set, resolve_parent_attr_target

logger = logging.getLogger(__name__)


def build_tool_node(node_config: NodeConfig):
    """Build a tool node that injects context into tools."""
    raw_tools = registry.get_tools(node_config.tools)
    
    # Apply property injection to tool node
    processed_tools = process_tools_with_overrides(raw_tools, node_config.tool_configs)
    
    # Inject the tool node's tool set into run_subagent so dynamically spawned
    # sub-agents inherit these tools (tool execution happens HERE, not in the agent).
    parent_tools_for_subagent = build_subagent_tool_set(processed_tools, node_config.tool_configs)
    for t in raw_tools:
        if hasattr(t, "parent_tools"):
            t.parent_tools = parent_tools_for_subagent
    for t in processed_tools:
        if hasattr(t, "parent_tools"):
            t.parent_tools = parent_tools_for_subagent
    
    # Build tool map
    tool_map = {t.name: t for t in processed_tools}
    
    # Use async lambda/function for execution
    async def executor_func(state: AgentState):
        return await _execute_tools(state, tool_map, parent_tools_for_subagent)
        
    executor = RunnableLambda(executor_func)
    setattr(executor, "name", node_config.id)
    
    logger.debug(f"Tool node '{node_config.id}' created with {len(processed_tools)} tools (context-aware)")
    return executor


async def _execute_tools(state: AgentState, tool_map: dict, parent_tools_for_subagent=None) -> dict:
    """Execute tools with context injection (Asynchronous)."""
    context = state.get("context", {})
    thread_id = state.get("thread_id") or context.get("thread_id") or "default_thread"
    messages = state.get("messages", [])
    
    if not messages:
        return {"messages": messages, "context": context}
    
    last_message = messages[-1]
    tool_calls = getattr(last_message, 'tool_calls', None) or []
    
    if not tool_calls:
        return {"messages": messages, "context": context}
    
    # Forward the orchestrator agent's LLM config to run_subagent so sub-agents
    # reuse the parent model + API key (stamped on the AIMessage by the agent).
    parent_llm_config = None
    _md = getattr(last_message, 'response_metadata', None) or {}
    _llm_cfg = _md.get("llm_config")
    if _llm_cfg:
        try:
            from Apps.AgentMesh.Schema.WorkflowSchema import LLMConfig
            parent_llm_config = LLMConfig(**_llm_cfg)
        except Exception:
            parent_llm_config = None
    
    tool_results = []
    
    for tool_call in tool_calls:
        tool_name = tool_call.get("name") or getattr(tool_call, 'name', None)
        tool_input = tool_call.get("args", {}) or {}
        tool_id = tool_call.get("id") or getattr(tool_call, 'id', None)
        
        tool = tool_map.get(tool_name)
        if not tool:
            try:
                resolved_tools = registry.get_tools([tool_name])
                if resolved_tools:
                    tool = resolved_tools[0]
                    tool_map[tool_name] = tool
            except Exception as e:
                logger.warning(f"Failed to dynamically resolve tool '{tool_name}' from registry: {e}")
                
        if not tool:
            logger.error(f"Tool '{tool_name}' not found in registry")
            tool_results.append(
                ToolMessage(
                    content=f"Tool '{tool_name}' not found in registry",
                    tool_call_id=tool_id,
                    name=tool_name
                )
            )
            continue
        
        result = await _invoke_tool(tool, tool_name, tool_input, tool_id, context, thread_id, parent_llm_config, parent_tools_for_subagent)
        tool_results.append(result)
    
    return {"messages": tool_results, "context": context}


async def _invoke_tool(tool, tool_name: str, tool_input: dict, tool_id: str, context: dict, thread_id: str, parent_llm_config=None, parent_tools_for_subagent=None) -> ToolMessage:
    """Invoke a single tool with context injection (Asynchronous)."""
    try:
        # Forward the orchestrator agent's LLM config to run_subagent so sub-agents
        # reuse the parent model + API key. Applied at INVOCATION TIME so fresh
        # registry clones (dynamic fallback) get the injection too.
        _target = resolve_parent_attr_target(tool)
        if parent_llm_config is not None and hasattr(_target, "parent_llm_config"):
            _target.parent_llm_config = parent_llm_config
        if parent_tools_for_subagent is not None and hasattr(_target, "parent_tools"):
            _target.parent_tools = parent_tools_for_subagent

        # Build the config object — also carries the parent agent's LLM config and
        # tool set so run_subagent can recover them via the config channel even if
        # instance-attribute injection was bypassed (e.g. wrapped/registry clones).
        config = {"configurable": {"context": context, "thread_id": thread_id}}
        if parent_llm_config is not None:
            try:
                if hasattr(parent_llm_config, "model_dump"):
                    _llm_dict = parent_llm_config.model_dump(exclude_none=True)
                elif hasattr(parent_llm_config, "dict"):
                    _llm_dict = parent_llm_config.dict(exclude_none=True)
                elif isinstance(parent_llm_config, dict):
                    _llm_dict = parent_llm_config
                else:
                    _llm_dict = None
            except Exception:
                _llm_dict = None
            if _llm_dict:
                config["configurable"]["parent_llm_config"] = _llm_dict
        if parent_tools_for_subagent is not None:
            config["configurable"]["parent_tools"] = parent_tools_for_subagent
        
        # CRITICAL: Extract the underlying function (Sync or Async)
        # We check both coroutine (for async tools) and func (for sync tools)
        wrapped_func = getattr(tool, 'coroutine', None) or getattr(tool, 'func', None)
        
        # 🧪 Debug: Log invocation
        is_async = getattr(tool, 'coroutine', None) is not None
        logger.debug(f"Invoking {tool_name} (Async={is_async})")
        
        if wrapped_func:
            # Only inject config if the tool signature accepts it (or **kwargs),
            # otherwise tools like get_current_date_time() crash with
            # TypeError: got an unexpected keyword argument 'config'
            final_input = dict(tool_input)
            sig = inspect.signature(wrapped_func)
            has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if "config" in sig.parameters or has_kwargs:
                final_input["config"] = config
            
            if is_async or inspect.iscoroutinefunction(wrapped_func):
                result = await wrapped_func(**final_input)
            else:
                result = wrapped_func(**final_input)
        else:
            # Fallback to standard dispatch
            result = await tool.ainvoke(tool_input)
        
        # detect human-interaction marker
        if isinstance(result, dict) and result.get("__human_interaction__"):
            return ToolMessage(
                content="__HUMAN_INTERACTION__",
                additional_kwargs={"pause_config": result["pause_config"]},
                tool_call_id=tool_id,
                name=tool_name
            )

        if isinstance(result, ToolMessage):
            result.tool_call_id = tool_id
            result.name = tool_name
            return result
            
        return ToolMessage(content=str(result), tool_call_id=tool_id, name=tool_name)
    except Exception as e:
        logger.error(f"Tool '{tool_name}' execution failed: {e}")
        import traceback
        traceback.print_exc()
        return ToolMessage(content=f"ERROR: {str(e)}", tool_call_id=tool_id, name=tool_name)
