import asyncio
import concurrent.futures
import inspect
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from Apps.AgentMesh.Schema.WorkflowSchema import LLMConfig
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Utilities.LLM.llm_factory import init_chat_model
from Apps.AgentMesh.Core.ToolRegistry import registry

logger = logging.getLogger(__name__)

MAX_CONCURRENT_SUBAGENTS = 10  # Maximum parallel sub-agents allowed
MAX_SUBAGENT_DEPTH = 6         # Recursion safety guard depth


class SubAgentInput(BaseModel):
    task: Optional[str] = Field(
        None,
        description="The primary subtask description for the sub-agent to execute."
    )
    parallel_tasks: Optional[List[Union[str, Dict[str, Any]]]] = Field(
        None,
        description="Optional list of up to 10 subtasks to execute concurrently. Accepts strings or objects like [{'task': '...'}]"
    )
    name: Optional[str] = Field(
        None,
        description="Optional alias or role name for the sub-agent (e.g. 'researcher_agent')."
    )
    arguments: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional nested arguments object for LLM model compatibility."
    )
    model_provider: Optional[str] = Field(
        None, 
        description="Optional LLM provider (e.g., groq, openai, anthropic, ollama). Defaults to system/parent LLM."
    )
    model_name: Optional[str] = Field(
        None, 
        description="Optional model identifier (e.g., gpt-4o, claude-3-5-sonnet-20241022). Defaults to system/parent LLM."
    )
    subagent_tools: Optional[List[str]] = Field(
        None,
        description="Optional whitelist of tool names the sub-agent may use. Strictly intersected with the parent node's tools — the sub-agent can never access tools the parent doesn't have."
    )
    current_depth: Optional[int] = Field(0, description="Internal recursion depth counter.")


def _extract_failed_generation_tool_calls(err_str: str):
    """Recover tool call(s) from a Groq 'tool_use_failed' error payload.

    Open-weights models on Groq (e.g. gpt-oss-20b) sometimes emit a tool call as
    raw JSON text instead of a native function call. Groq rejects the request with
    ``code: tool_use_failed`` and embeds the rejected call in ``failed_generation``
    as a JSON-escaped string. Parse it into a list of {name, args, id} dicts.
    """
    if not err_str:
        return None
    m = re.search(r'"failed_generation"\s*:\s*"((?:[^"\\]|\\.)*)"', err_str)
    if not m:
        return None
    raw = m.group(1)
    try:
        decoded = json.loads('"' + raw + '"')
    except Exception:
        decoded = raw
    if not decoded or not isinstance(decoded, str):
        return None
    try:
        data = json.loads(decoded)
    except Exception:
        from json_repair import loads as _repair_loads
        try:
            data = _repair_loads(decoded)
        except Exception:
            data = None
    if data is None:
        return None
    items = data if isinstance(data, list) else [data]
    calls = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("tool_name") or item.get("function")
        args = item.get("arguments") or item.get("args") or item.get("parameters") or item.get("input") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"input": args}
        if not isinstance(args, dict):
            args = {"input": args}
        if name:
            calls.append({"name": str(name), "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"})
    return calls or None


def _tool_func_accepts_config(func) -> bool:
    try:
        sig = inspect.signature(func)
        return ("config" in sig.parameters) or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
    except Exception:
        return False


async def _execute_recovered_tool_calls(calls: List[Dict[str, Any]], tools: List[Any], thread_id: Optional[str] = None) -> List[str]:
    """Execute recovered text-format tool calls against the sub-agent's tool set.

    Returns a list of formatted result strings (one per recovered call). Calls
    whose tool is not in the sub-agent's inherited set are reported, never executed.
    """
    tool_by_name = {getattr(t, "name", ""): t for t in tools}
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None
    out = []
    for call in calls:
        name = call.get("name")
        args = dict(call.get("args") or {})
        tool = tool_by_name.get(name)
        if tool is None:
            out.append(f"[Tool '{name}' not available to this sub-agent]")
            continue
        try:
            func = getattr(tool, "coroutine", None) or getattr(tool, "func", None)
            is_coro = getattr(tool, "coroutine", None) is not None or inspect.iscoroutinefunction(func)
            if func is not None:
                if config is not None and _tool_func_accepts_config(func):
                    args["config"] = config
                result = await func(**args) if is_coro else func(**args)
            else:
                result = await tool.ainvoke(args, config=config) if config else await tool.ainvoke(args)
            text = result.content if hasattr(result, "content") else str(result)
            out.append(f"### Tool '{name}' result:\n{text}")
        except Exception as te:
            out.append(f"### Tool '{name}' error: {te}")
    return out


async def _run_single_subagent(
    task: str,
    agent_id: int,
    model_provider: Optional[str],
    model_name: Optional[str],
    subagent_tools: Optional[List[str]],
    current_depth: int,
    parent_tool_configs: Optional[Dict[str, Any]],
    parent_llm_config: Optional[LLMConfig],
    parent_tools: Optional[List[BaseTool]],
    max_subagent_depth: Optional[int] = None,
    recursion_limit: Optional[int] = None,
    inherit_parent_tools: Optional[bool] = None,
    thread_id: Optional[str] = None
) -> str:
    """Spawns an individual tool-enabled ReAct sub-agent for a single subtask."""
    eff_max_depth = max_subagent_depth if max_subagent_depth is not None else MAX_SUBAGENT_DEPTH
    if current_depth >= eff_max_depth:
        logger.warning(f"[SubAgent {agent_id}] Recursion depth limit ({eff_max_depth}) reached.")
        return f"[SubAgent #{agent_id} halted: Recursion depth limit reached]"

    try:
        # 1. Initialize LLM (Sub-agents ALWAYS use the parent agent's LLM model)
        if parent_llm_config:
            llm = init_chat_model(parent_llm_config)
        elif model_provider and model_name:
            llm_config = LLMConfig(
                provider=model_provider,
                model=model_name,
                temperature=0.2
            )
            llm = init_chat_model(llm_config)
        else:
            # System default LLM fallback
            llm = init_chat_model(None)

        # 2. STRICT Tool Resolution:
        # A sub-agent may ONLY use its parent node's tools (inherit_parent_tools checkbox,
        # default ON). The LLM-provided subagent_tools whitelist is INTERSECTED with the
        # parent tools so it can never escape the parent's tool boundary. There is NO
        # registry fallback: if inheritance is off (or the parent has no tools), the
        # sub-agent runs as a tool-less (pure LLM) worker.
        eff_inherit_parent = True if inherit_parent_tools is None else bool(inherit_parent_tools)
        processed_tools = []
        if eff_inherit_parent and parent_tools and len(parent_tools) > 0:
            # Strict: child is limited to the parent node's tools only
            processed_tools = [t for t in parent_tools if getattr(t, "name", "") != "run_subagent"]
            if subagent_tools:
                parent_names = {getattr(t, "name", "") for t in processed_tools}
                allowed_names = [n for n in subagent_tools if n in parent_names and n != "run_subagent"]
                processed_tools = [t for t in processed_tools if getattr(t, "name", "") in allowed_names]

        logger.info(
            f"[SubAgent {agent_id}] Inherited tool set: "
            f"{[getattr(t, 'name', '?') for t in processed_tools]} "
            f"(inherit_parent_tools={eff_inherit_parent}, parent_tools={len(parent_tools or [])})"
        )

        if not processed_tools:
            logger.info(f"[SubAgent {agent_id}] Running as tool-less (pure LLM) worker — no inherited tools "
                        f"(inherit_parent_tools={eff_inherit_parent}, parent_tools={len(parent_tools or [])}).")
            resp = await llm.ainvoke([("user", task)])
            content = resp.content if hasattr(resp, "content") else str(resp)
            return f"### Sub-Agent #{agent_id} Output:\n{content}"

        # 3. Compile ReAct Sub-Graph Engine
        inherited_names = ", ".join(getattr(t, "name", "?") for t in processed_tools)
        system_prompt = (
            f"You are Sub-Agent #{agent_id} (Depth {current_depth + 1}/{eff_max_depth}). "
            "Execute your assigned subtask concisely and thoroughly using your available tools. "
            f"Your available tools are: {inherited_names}. "
            "For any request that needs CURRENT or LIVE data (recent news, today's date, "
            "web search, file access) you MUST call the appropriate tool first — never answer "
            "from your training memory alone. "
            "IMPORTANT: Always invoke tools using native function call structures. "
            "Do not output raw XML tags or text wrappers like `<function=...>` in your response. "
            "CRITICAL: After using tools, your final message MUST summarize the results you found — "
            "the parent agent only sees this final message, so it must contain the complete answer "
            "or findings, never a bare statement like 'Done' or 'I used the tools'. "
            "Return a clear, factual synthesis of your findings."
        )

        subagent_graph = create_react_agent(
            model=llm,
            tools=processed_tools,
            prompt=system_prompt
        )

        # 4. Execute Sub-Agent Loop
        inputs = {"messages": [("user", task)]}
        eff_recursion_limit = recursion_limit if recursion_limit is not None else 15
        config = {"recursion_limit": eff_recursion_limit}
        if thread_id:
            config["configurable"] = {"thread_id": thread_id}
        result = await subagent_graph.ainvoke(inputs, config=config)

        final_msg = result["messages"][-1]
        content = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
        logger.info(f"[SubAgent #{agent_id}] Completed subtask successfully ({len(content)} chars) "
                    f"with {len(processed_tools)} tool(s)")
        return f"### Sub-Agent #{agent_id} Output:\n{content}"

    except Exception as e:
        err_str = str(e)
        logger.exception(f"[SubAgent #{agent_id}] Execution failed: {err_str}")
        # Groq open-weights models emit tool calls as raw TEXT → tool_use_failed.
        # Recover the call(s) from failed_generation and EXECUTE them directly
        # instead of degrading to a pure-knowledge answer.
        if "tool_use_failed" in err_str or "failed_generation" in err_str:
            recovered = _extract_failed_generation_tool_calls(err_str)
            if recovered:
                recovered_results = await _execute_recovered_tool_calls(recovered, processed_tools, thread_id)
                if recovered_results:
                    logger.info(f"[SubAgent #{agent_id}] Executed {len(recovered_results)} recovered tool call(s) from failed_generation")
                    return f"### Sub-Agent #{agent_id} Output:\n" + "\n\n".join(recovered_results)
        # Automatic fallback for Groq tool_use_failed / XML syntax errors
        if "tool_use_failed" in err_str or "failed_generation" in err_str or "BadRequestError" in err_str:
            try:
                logger.info(f"[SubAgent #{agent_id}] Attempting direct LLM fallback synthesis...")
                fallback_prompt = (
                    f"Subtask: {task}\n\n"
                    "Provide a direct, factual, and comprehensive answer to this task using your general knowledge."
                )
                fallback_resp = await llm.ainvoke([("user", fallback_prompt)])
                content = fallback_resp.content if hasattr(fallback_resp, "content") else str(fallback_resp)
                return f"### Sub-Agent #{agent_id} Output (Direct Response):\n{content}"
            except Exception as fb_err:
                logger.exception(f"[SubAgent #{agent_id}] Direct fallback failed: {fb_err}")
        return f"[Sub-Agent #{agent_id} Error: {err_str}]"


async def _run_subagent_impl(
    task: Optional[str] = None,
    parallel_tasks: Optional[List[Union[str, Dict[str, Any]]]] = None,
    name: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None,
    model_provider: Optional[str] = None,
    model_name: Optional[str] = None,
    subagent_tools: Optional[List[str]] = None,
    current_depth: int = 0,
    parent_tool_configs: Optional[Dict[str, Any]] = None,
    parent_llm_config: Optional[LLMConfig] = None,
    parent_tools: Optional[List[BaseTool]] = None,
    tool_registry_inst: Any = None,
    max_concurrent_subagents: Optional[int] = None,
    max_subagent_depth: Optional[int] = None,
    recursion_limit: Optional[int] = None,
    inherit_parent_tools: Optional[bool] = None,
    thread_id: Optional[str] = None
) -> str:
    """Orchestrates single or parallel (max 10) sub-agent executions.

    STRICT BOUNDARY: sub-agents may only use the parent node's tools. The LLM's
    subagent_tools whitelist is intersected with parent tools, so a child can
    never access tools the parent doesn't have. No registry fallback.
    """
    reg = tool_registry_inst or registry

    # Resolve effective limits from UI configs, falling back to module defaults
    eff_max_concurrent = max_concurrent_subagents if max_concurrent_subagents is not None else MAX_CONCURRENT_SUBAGENTS

    # Unpack nested 'arguments' wrapper if generated by certain LLM models (e.g. Groq/LiteLLM)
    if arguments and isinstance(arguments, dict):
        if not task and arguments.get("task"):
            task = arguments.get("task")
        if not parallel_tasks and arguments.get("parallel_tasks"):
            parallel_tasks = arguments.get("parallel_tasks")
        if not subagent_tools and arguments.get("subagent_tools"):
            subagent_tools = arguments.get("subagent_tools")

    # Consolidate tasks list cleanly from strings OR dicts
    task_list: List[str] = []
    if parallel_tasks and isinstance(parallel_tasks, list) and len(parallel_tasks) > 0:
        for item in parallel_tasks[:eff_max_concurrent]:
            if isinstance(item, str):
                task_list.append(item)
            elif isinstance(item, dict):
                t_val = item.get("task") or item.get("description") or item.get("subtask") or str(item)
                task_list.append(str(t_val))
            else:
                task_list.append(str(item))
    elif task:
        task_list = [task]
    else:
        return "[SubAgentTool Error: No valid task or parallel_tasks specified]"

    # Enforce MAX concurrent sub-agents limit
    if len(task_list) > eff_max_concurrent:
        task_list = task_list[:eff_max_concurrent]

    logger.info(f"[SubAgentTool] Spawning {len(task_list)} sub-agent(s) (Max allowed: {eff_max_concurrent})")

    # Launch parallel sub-agents via asyncio.gather
    coros = [
        _run_single_subagent(
            task=t,
            agent_id=idx + 1,
            model_provider=model_provider,
            model_name=model_name,
            subagent_tools=subagent_tools,
            current_depth=current_depth,
            parent_tool_configs=parent_tool_configs,
            parent_llm_config=parent_llm_config,
            parent_tools=parent_tools,
            max_subagent_depth=max_subagent_depth,
            recursion_limit=recursion_limit,
            inherit_parent_tools=inherit_parent_tools,
            thread_id=thread_id
        )
        for idx, t in enumerate(task_list)
    ]
    results = await asyncio.gather(*coros, return_exceptions=True)

    spawned = len(task_list)
    completed = sum(1 for r in results if not isinstance(r, Exception))

    formatted_results = []
    for idx, res in enumerate(results):
        if isinstance(res, Exception):
            formatted_results.append(f"### Sub-Agent #{idx + 1} Exception:\n{str(res)}")
        else:
            formatted_results.append(str(res))

    summary = f"**Sub-Agent Summary**: {completed}/{spawned} completed"
    return f"{summary}\n\n" + "\n\n---\n\n".join(formatted_results)


class SubAgentTool(BaseTool):
    name: str = "run_subagent"
    description: str = (
        "Spawn up to 10 autonomous sub-agents in parallel with tool access to execute subtasks independently. "
        "Use this for research, parallel search, codebase investigation, or complex multi-part work."
    )
    args_schema: type = SubAgentInput
    tool_registry_inst: Any = None
    parent_tool_configs: Optional[Dict[str, Any]] = None
    parent_llm_config: Optional[LLMConfig] = None
    parent_tools: Optional[List[BaseTool]] = None
    coroutine: Any = None  # Explicit Pydantic field annotation for tool_node async detection

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.coroutine = self._arun

    async def _arun(
        self,
        task: Optional[str] = None,
        parallel_tasks: Optional[List[Union[str, Dict[str, Any]]]] = None,
        name: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        subagent_tools: Optional[List[str]] = None,
        current_depth: Optional[int] = 0,
        max_concurrent_subagents: Optional[int] = None,
        max_subagent_depth: Optional[int] = None,
        recursion_limit: Optional[int] = None,
        inherit_parent_tools: Optional[bool] = None,
        **kwargs
    ) -> str:
        # Extract thread_id + parent context from the injected runtime config.
        # tool_node injects config={"configurable": {"context": ..., "thread_id": ...,
        # "parent_llm_config": {...}, "parent_tools": [...]}}; LangGraph executors
        # pass configurable.thread_id as well. Instance attrs are the fallback.
        _thread_id = None
        _parent_llm_config = None
        _parent_tools = None
        _runtime_cfg = kwargs.get("config")
        if isinstance(_runtime_cfg, dict):
            _cfg = _runtime_cfg.get("configurable")
            if isinstance(_cfg, dict):
                _thread_id = _cfg.get("thread_id")
                if isinstance(_cfg.get("parent_llm_config"), dict):
                    try:
                        _parent_llm_config = LLMConfig(**_cfg["parent_llm_config"])
                    except Exception:
                        _parent_llm_config = None
                if _cfg.get("parent_tools"):
                    _parent_tools = _cfg["parent_tools"]
        if _parent_llm_config is None:
            _parent_llm_config = self.parent_llm_config
        if _parent_tools is None:
            _parent_tools = self.parent_tools
        return await _run_subagent_impl(
            task=task,
            parallel_tasks=parallel_tasks,
            name=name,
            arguments=arguments,
            model_provider=model_provider,
            model_name=model_name,
            subagent_tools=subagent_tools,
            current_depth=current_depth or 0,
            parent_tool_configs=self.parent_tool_configs or {},
            parent_llm_config=_parent_llm_config,
            parent_tools=_parent_tools,
            tool_registry_inst=self.tool_registry_inst,
            max_concurrent_subagents=max_concurrent_subagents,
            max_subagent_depth=max_subagent_depth,
            recursion_limit=recursion_limit,
            inherit_parent_tools=inherit_parent_tools,
            thread_id=_thread_id
        )

    def _run(self, **kwargs) -> str:
        """Fallback synchronous runner bridging to async _arun."""
        cleaned_kwargs = {k: v for k, v in kwargs.items() if k != "config"}
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(lambda: asyncio.run(self._arun(**cleaned_kwargs))).result()
        else:
            return asyncio.run(self._arun(**cleaned_kwargs))


def register_subagent_tools():
    """Auto-register SubAgentTool into AgentMesh ToolRegistry on module load."""
    meta = ToolMeta(
        name="run_subagent",
        display_name="Run Dynamic Sub-Agent",
        category="agent",
        package="AgentMesh Core",
        version="v1.1",
        allow_duplicate=False,
        description="Spawns up to 10 autonomous sub-agents in parallel to execute tasks independently.",
        ai_visible_fields=["task", "parallel_tasks", "model_provider", "model_name", "subagent_tools"],
        configs=[
            {
                "name": "inherit_parent_tools",
                "type": "boolean",
                "description": "Allow the sub-agent to inherit the parent agent's tools. STRICT: children can only access tools the parent node has. Disable to run sub-agents as tool-less (pure LLM) workers.",
                "required": False,
                "default": True
            },
            {
                "name": "max_concurrent_subagents",
                "type": "number",
                "description": "Maximum number of parallel sub-agents spawned per run_subagent call.",
                "required": False,
                "default": 10
            },
            {
                "name": "max_subagent_depth",
                "type": "number",
                "description": "Maximum recursion depth for nested sub-agent delegation.",
                "required": False,
                "default": 6
            },
            {
                "name": "recursion_limit",
                "type": "number",
                "description": "Maximum steps (LLM turns + tool calls) per sub-agent ReAct loop.",
                "required": False,
                "default": 15
            }
        ]
    )
    subagent_tool_instance = SubAgentTool()
    registry.register(subagent_tool_instance, meta)


# Trigger auto-discovery registration per ToolImplementation.md SOP
register_subagent_tools()
