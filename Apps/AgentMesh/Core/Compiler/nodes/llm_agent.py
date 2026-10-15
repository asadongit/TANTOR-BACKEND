"""
LLM Agent Node Builder
Handles creation of executable LLM-based agents with tool support.
"""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Core.ToolRegistry import registry
from Apps.AgentMesh.Utilities.LLM.llm_factory import init_chat_model
from .utils import (
    process_tools_with_overrides,
    build_subagent_tool_set,
    resolve_parent_attr_target,
    interpolate,
    prepare_agent_memory,
    retrieve_vector_memory_async,
    persist_vector_memory_async,
    extract_latest_user_query,
    ensure_non_empty_user_input,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Batched extraction
#
# A model asked for several hundred fields at once answers worse than the same
# model asked for fifty at a time: entries go missing, and the shape is
# sometimes abandoned part way through. So the list is split and the same prompt
# is run once per part, then the answers are put back together.
#
# Nothing here knows what is being extracted. It divides a list, substitutes one
# part at a time into the prompt, and joins the results - the list itself lives
# in workflow configuration, so adding fields never means editing code.
# ══════════════════════════════════════════════════════════════════════════════

def _dig(root: Any, path: str) -> Any:
    """Follow a dotted path, or nothing if it does not lead anywhere."""
    value = root
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    return value


def _resolve_items(spec: Optional[str], context: dict) -> List[Any]:
    """The list a batching agent is to work through, read from context.

    Accepts `{{ name }}`, `name`, or a dotted path, so a workflow can keep the
    list wherever it already keeps things.

    A list held in the Variable Explorer sits under `workflow_variables`, so
    that is looked in as well - otherwise the obvious `{{ my_fields }}` would
    quietly find nothing and only the longer `{{ workflow_variables.my_fields }}`
    would work, which is a trap rather than a design.

    A list arriving as JSON text - which is how it comes back from some stores -
    is parsed rather than treated as one long item.
    """
    if not spec:
        return []
    path = spec.strip()
    if path.startswith("{{") and path.endswith("}}"):
        path = path[2:-2].strip()

    value = _dig(context, path)
    if value is None:
        value = _dig(context.get("workflow_variables") or {}, path)

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return []
    return list(value) if isinstance(value, (list, tuple)) else []


def _chunk(items: List[Any], size: int) -> List[List[Any]]:
    """The list in consecutive pieces, order preserved."""
    size = max(1, int(size or 1))
    return [items[i:i + size] for i in range(0, len(items), size)]


def _render_items(items: List[Any]) -> str:
    """One batch, as the prompt will see it.

    Plain names are given one per line, which reads unambiguously and costs
    fewer tokens than JSON. Anything structured is given as JSON, because
    flattening it would lose what the prompt needs.
    """
    if items and all(isinstance(i, str) for i in items):
        return "\n".join(items)
    return json.dumps(items, ensure_ascii=False, indent=1)


def _is_blank(v: Any) -> bool:
    """Nothing was said here - as opposed to zero, which is an answer."""
    return v is None or v == "" or v == [] or v == {}


def _join(a: Any, b: Any, key: Optional[str], clashes: List[str],
          where: str = "") -> Any:
    """Two answers to the same question, combined.

    Batches are asked about different fields, so they mostly do not overlap and
    joining is simply filling in. Where they DO overlap and disagree, the first
    answer is kept and the disagreement is recorded rather than silently
    resolved - a figure two batches read differently is exactly the kind of
    thing that must be visible.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = _join(a[k], v, key, clashes, f"{where}.{k}" if where else k) \
                if k in a else v
        return out

    if isinstance(a, list) and isinstance(b, list):
        # Without a key there is no way to tell which entries are the same one,
        # so the lists are laid end to end. With a key, entries carrying the
        # same value are treated as one and merged.
        if not key:
            return a + b
        merged = [dict(x) if isinstance(x, dict) else x for x in a]
        index = {x.get(key): i for i, x in enumerate(merged)
                 if isinstance(x, dict) and key in x}
        for item in b:
            if isinstance(item, dict) and item.get(key) in index:
                at = index[item[key]]
                merged[at] = _join(merged[at], item, key, clashes,
                                   f"{where}[{item[key]}]")
            else:
                merged.append(item)
        return merged

    if _is_blank(a):
        return b
    if _is_blank(b) or a == b:
        return a
    clashes.append(f"{where or '?'}: {a!r} vs {b!r}")
    return a


def _merge_answers(answers: List[Any], how: str,
                   key: Optional[str]) -> tuple:
    """Every batch's answer as one, plus whatever they disagreed about."""
    answers = [a for a in answers if a is not None]
    if not answers:
        return None, []
    if how == "list_concat":
        return answers, []

    clashes: List[str] = []
    merged = answers[0]
    for nxt in answers[1:]:
        if how == "shallow_merge" and isinstance(merged, dict) and isinstance(nxt, dict):
            merged = {**merged, **{k: v for k, v in nxt.items()
                                   if k not in merged or _is_blank(merged[k])}}
        else:
            merged = _join(merged, nxt, key, clashes)
    return merged, clashes


def _answer_is_poor(expression: Optional[str], answer: Any, label: str) -> bool:
    """Does this answer meet the bar the workflow set for it?

    A model sometimes returns a fraction of what it should. The same statement
    that yields two hundred lines comes back with eight - valid JSON, no error,
    nothing to notice. That is worse than a failure, because a failure stops and
    a short answer flows on into the report.

    The test is an expression over the answer itself, so it is arithmetic rather
    than a second model's opinion: it costs nothing, always gives the same
    verdict for the same answer, and can be read by whoever configured it.

    A broken expression returns False - a test that cannot be evaluated must not
    condemn a good answer, and the fault is logged where it can be seen.
    """
    if not expression or answer is None:
        return False
    try:
        from simpleeval import SimpleEval
        import ast
        from .utils import SmartInput

        ev = SimpleEval()
        ev.nodes[ast.List] = lambda node: [ev._eval(e) for e in node.elts]
        wrapped = SmartInput(answer) if isinstance(answer, dict) else answer
        ev.names = {"answer": wrapped, "True": True, "False": False, "None": None}
        ev.functions = {"len": len, "count": len, "int": int, "float": float,
                        "str": str, "bool": bool, "abs": abs, "min": min, "max": max}
        return bool(ev.eval(expression))
    except Exception as err:                       # noqa: BLE001
        logger.warning(f"[AGENT_RETRY] {label}: could not evaluate '{expression}' "
                       f"({err}); the answer is accepted rather than rejected")
        return False


def _extract_failed_generation_tool_calls(err_str: str):
    """Recover tool call(s) from a Groq 'tool_use_failed' error payload.

    Open-weights models on Groq (e.g. gpt-oss-20b) sometimes emit a tool call as
    raw JSON text instead of a native function call. Groq rejects the request with
    ``code: tool_use_failed`` and embeds the rejected call in ``failed_generation``
    as a JSON-escaped string. Parse it into a valid LangChain ``tool_calls`` list so
    the call can be executed normally instead of blind-retrying the same prompt.
    """
    if not err_str:
        return None
    import re as _re
    import uuid as _uuid
    from json_repair import loads as _repair_loads

    m = _re.search(r'"failed_generation"\s*:\s*"((?:[^"\\]|\\.)*)"', err_str)
    if not m:
        return None
    raw = m.group(1)

    # failed_generation is itself a JSON string literal — unescape it
    try:
        decoded = json.loads('"' + raw + '"')
    except Exception:
        decoded = raw
    if not decoded or not isinstance(decoded, str):
        return None

    try:
        data = json.loads(decoded)
    except Exception:
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
            calls.append({
                "name": str(name),
                "args": args,
                "id": f"call_{_uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            })
    return calls or None


def build_llm_agent(node_config: NodeConfig) -> Callable:
    """
    Build an LLM agent node that executes with tool support,
    memory management, and HITL integration.
    """
    llm = init_chat_model(node_config.llm)
    raw_tools = registry.get_tools(node_config.tools)
    
    # Log which model this agent is using
    llm_config = node_config.llm
    _model_info = f"{llm_config.provider}/{llm_config.model}" if llm_config else "default"
    logger.info(f"[AGENT_BUILD] {node_config.id}: LLM={_model_info}, temp={getattr(llm_config, 'temperature', 'N/A')}")
    
    # Apply tool property injection (n8n style)
    processed_tools = process_tools_with_overrides(raw_tools, node_config.tool_configs)
    
    # The effective tool set that the parent agent actually uses (with the user's
    # per-tool overrides applied). Sub-agents inherit EXACTLY these configured
    # tool instances so their tool settings match the parent's.
    parent_tools_for_subagent = build_subagent_tool_set(processed_tools, node_config.tool_configs)
    
    # Inject parent node context into BOTH:
    # 1. raw clones — when a user configures run_subagent settings, the tool is wrapped
    #    into a StructuredTool whose coroutine is this clone's bound _arun; it reads
    #    parent_tools at call time, so it must point at the processed (configured) set.
    # 2. processed tools — covers the unwrapped case directly.
    # NOTE: resolve_parent_attr_target digs through StructuredTool wrappers to the
    # underlying SubAgentTool instance so parent context survives wrapping.
    for t in raw_tools:
        target = resolve_parent_attr_target(t)
        if hasattr(target, "parent_tools"):
            target.parent_tools = parent_tools_for_subagent
            target.parent_llm_config = llm_config
            target.parent_tool_configs = node_config.tool_configs
    
    for t in processed_tools:
        target = resolve_parent_attr_target(t)
        if hasattr(target, "parent_tools"):
            target.parent_tools = parent_tools_for_subagent
            target.parent_llm_config = llm_config
            target.parent_tool_configs = node_config.tool_configs
    
    system_msg = node_config.task or "You are a helpful assistant."
    
    llm_with_tools = llm.bind_tools(processed_tools) if processed_tools else llm
    
    async def run_once(state: AgentState):
        messages = state.get("messages", [])
        thread_id = state.get("thread_id", "default_thread")
        context = state.get("context", {})
        item = context.get("item")
        if item:
            logger.info(f"[AGENT] {node_config.id} processing item: {str(item)[:100]}...")
            
        # ══════════════════════════════════════════════════════════════════════
        
        # CRITICAL FIX: Filter out truly empty AIMessages from history
        filtered_messages = []
        for msg in messages:
            msg_type = type(msg).__name__
            content = getattr(msg, "content", "")
            has_tools = hasattr(msg, "tool_calls") and msg.tool_calls
            
            is_ai = msg_type == "AIMessage"
            has_content = False
            if isinstance(content, str):
                has_content = bool(content.strip())
            elif content:
                has_content = True
            
            if not is_ai or has_content or has_tools:
                filtered_messages.append(msg)
            else:
                logger.debug(f"Filtered out truly empty AIMessage for agent: {node_config.id}")
        
        messages = filtered_messages
        
        logger.debug(f"Agent {node_config.id}: Received {len(messages)} messages, context keys: {list(context.keys())}")
        
        # Handle HITL (Human-In-The-Loop) rework
        hitl_action = context.get("hitl_action")
        hitl_feedback = context.get("hitl_feedback")
        state_updates = []
        final_context = None
        
        if hitl_action == "rework":
            logger.debug(f"HITL Rework triggered for node {node_config.id}")
            include_original = context.get("include_original_output", True)
            inject_feedback = context.get("inject_feedback", True)
            
            if not include_original and messages and isinstance(messages[-1], AIMessage):
                last_msg = messages.pop()
                if hasattr(last_msg, 'id') and last_msg.id:
                    state_updates.append(RemoveMessage(id=last_msg.id))
            
            if hitl_feedback and inject_feedback:
                feedback_msg = HumanMessage(
                    content=f"[HUMAN FEEDBACK]\n{hitl_feedback}\n\nPlease improve your response based on this feedback."
                )
                messages.append(feedback_msg)
                state_updates.append(feedback_msg)
            
            final_context = {**context, "hitl_action": None, "hitl_feedback": None}
        
        # Memory management
        messages_for_agent, memory_summary_update = await prepare_agent_memory(messages, node_config, state=state)
        mem_config = node_config.memory or {}
        mem_type = mem_config.get("type_name", "buffer_memory")
        
        additional_context = ""

        # ══════════════════════════════════════════════════════════════════════
        # CD PIPELINE ISOLATION — Replicate standalone CD's SubAgents.run_agent()
        # ══════════════════════════════════════════════════════════════════════
        # In standalone CD, each specialist agent receives a FRESH, ISOLATED call:
        #   prompt = Consolidation_Prompt.format(aggregated_text=consolidated_text)
        #   response = await llm.ainvoke(prompt)
        #
        # No message history. No accumulated context. Just the prompt with the
        # agent's consolidated text. We detect isolation mode by checking for agent_input_*
        # keys set by routers, and replicate that exact behavior.
        # ══════════════════════════════════════════════════════════════════════
        isolation_mode = False
        
        agent_input_prefix = getattr(node_config, "agent_input_prefix", None) or "agent_input_"
        output_key_prefix = getattr(node_config, "output_key_prefix", None) or "cd_output_"
        
        cd_agent_input_keys = [k for k in context.keys() if k.startswith(agent_input_prefix)]
        referenced_input = None
        
        # ── SUBSCRIPTION ISOLATION MODE ──
        # Use a local copy of system_msg to avoid modifying the closure variable
        local_system_msg = system_msg
        
        if cd_agent_input_keys:
            # ONLY enter CD mode if this agent's system prompt explicitly references
            # an agent_input_* key. This ensures specialist agents get isolated calls
            # while the orchestrator (which doesn't reference these keys) stays in
            # standard mode and sees all specialist outputs in accumulated messages.
            for key in cd_agent_input_keys:
                if key in local_system_msg:  # e.g. system_msg contains "{{agent_input_identity}}"
                    referenced_input = key
                    break
            
            if referenced_input and context.get(referenced_input):
                isolation_mode = True
                input_text = context[referenced_input]
                logger.info(f"[CD_ISOLATION] Agent {node_config.id}: Fresh isolated call "
                            f"(input: {referenced_input}, {len(input_text)} chars) — "
                            f"replicating standalone CD run_agent() behavior")
                
                # Replace accumulated messages with a single HumanMessage
                messages_for_agent = [HumanMessage(
                    content="Analyze the documents provided in your instructions and "
                            "respond with the required JSON format."
                )]
            else:
                logger.debug(f"[CD_ISOLATION] Agent {node_config.id}: agent_input keys found "
                             f"but none referenced or all empty — using standard mode")
        
        # ── ORCHESTRATOR ISOLATION MODE ──
        # When cd_output_* keys exist but this agent didn't enter isolation via
        # agent_input_*, it's the orchestrator agent.
        # Replicate standalone CD's run_orchestration_agent(): give it ONLY the
        # clean specialist summaries (overview/pros/cons/confidence), not the
        # 22k+ char dump of raw OCR text, accumulated messages, and pipeline state.
        _orch_reports_text = None
        if not isolation_mode:
            cd_output_keys = [k for k in context.keys() if k.startswith(output_key_prefix)]
            if cd_output_keys:
                # Build clean reports exactly like standalone CD
                _reports_parts = []
                for _okey in sorted(cd_output_keys):
                    _agent_data = context[_okey]
                    if isinstance(_agent_data, dict):
                        _label = _agent_data.get("agent_name", _okey.replace("cd_output_", ""))
                        _clean = {f: _agent_data[f] for f in
                                  ("overview", "pros", "cons", "missing_fields", "confidence",
                                   "files", "matched_classes")
                                  if f in _agent_data}
                        _reports_parts.append(
                            f"[{_label}]:\n{json.dumps(_clean, indent=2, default=str)}"
                        )
                    else:
                        _reports_parts.append(f"[{_okey}]: {str(_agent_data)}")
                
                # Include skipped agent statuses so orchestrator knows which agents had no data
                for _sk, _sv in context.items():
                    if _sk.startswith("agent_status_") and isinstance(_sv, str) and "skipped" in _sv:
                        _agent_id = _sk.replace("agent_status_", "")
                        _reports_parts.append(
                            f"[{_agent_id}]: No data available (agent skipped — no matching documents)"
                        )
                
                _orch_reports_text = "\n\n".join(_reports_parts)
                isolation_mode = True
                referenced_input = f"{output_key_prefix}orchestrator"
                messages_for_agent = [HumanMessage(
                    content="Review all specialist agent analyses provided in your instructions "
                            "and respond with your credit decision in the required JSON format."
                )]
                logger.info(
                    f"[ORCH_ISOLATION] Agent {node_config.id}: Orchestrator isolation mode — "
                    f"{len(cd_output_keys)} specialist outputs, "
                    f"{len(_orch_reports_text)} chars clean context "
                    f"(vs {len(str(context))} chars full context)"
                )
        
        # Vector memory retrieval (after isolation detection: skipped for fresh isolated calls
        # so the query string is the actual user turn, not the generic isolation prompt)
        if mem_type == "vector_memory" and not isolation_mode and messages_for_agent:
            additional_context = await retrieve_vector_memory_async(messages_for_agent, node_config, thread_id)
        
        # Empty-input guard: providers like Groq reject an empty user message. Guarantee
        # the history ends on a non-empty user turn before it reaches the LLM.
        messages_for_agent = ensure_non_empty_user_input(messages_for_agent)
        
        # --- Final Polish on Interpolation Logic ---
        # 0. Protect LangChain specific variables from Jinja2 evaluation!
        # Jinja2 will evaluate {{context}} as undefined and erase it if we don't protect it.
        base_msg = local_system_msg + (additional_context if additional_context else "")
        
        PROTECTED = {
            "{{input}}": "__JINJA_PROTECT_INPUT__",
            "{{context}}": "__JINJA_PROTECT_CTX__",
            "{{payload}}": "__JINJA_PROTECT_PAYLOAD__",
            "{{threads_id}}": "__JINJA_PROTECT_THREADS_ID__",
            "{input}": "__JINJA_PROTECT_LC_INPUT__",
            "{context}": "__JINJA_PROTECT_LC_CTX__",
            "{payload}": "__JINJA_PROTECT_LC_PAYLOAD__",
            "{threads_id}": "__JINJA_PROTECT_LC_THREADS_ID__"
        }
        for raw, marker in PROTECTED.items():
            base_msg = base_msg.replace(raw, marker)
            
        # 1. First, apply custom {{context.key}} resolution and escape their resulting values
        final_system_msg = interpolate(base_msg, context, escape_braces=True)
        
        # 1.5 Restore the protected LangChain variables
        for raw, marker in PROTECTED.items():
            final_system_msg = final_system_msg.replace(marker, raw)
        
        # Inject the active user query context (skip in isolation mode — not needed)
        if not isolation_mode:
            active_user_query = extract_latest_user_query(messages)
            if active_user_query:
                final_system_msg += f"\n\n[Active User Request / Context]:\n{active_user_query}"
        
        # 2. Defend against LangChain's prompt parser by escaping literal braces
        # We must protect the few variables we actually want LangChain to interpolate
        RESERVED = {
            "{{input}}": "{input}",
            "{{context}}": "{context}",
            "{{payload}}": "{payload}",
            "{{threads_id}}": "{threads_id}" # for edge cases
        }
        
        # Temporary markers to protect intended variables
        for placeholder, marker in {"{input}": "__LC_INPUT__", "{context}": "__LC_CTX__", "{payload}": "__LC_PAYLOAD__"}.items():
            final_system_msg = final_system_msg.replace(placeholder, marker)
            
        # Also convert any remaining {{variable}} that the user might have written for LC
        for raw, lc in RESERVED.items():
            final_system_msg = final_system_msg.replace(raw, lc.replace("{", "__LC_").replace("}", "__"))
            
        # Now escape ALL remaining single braces in the prompt (JSON samples, etc.)
        final_system_msg = final_system_msg.replace("{", "{{").replace("}", "}}")
        
        # Finally, restore the intended LangChain variables to single braces
        final_system_msg = final_system_msg.replace("__LC_INPUT__", "{input}")
        final_system_msg = final_system_msg.replace("__LC_CTX__", "{context}")
        final_system_msg = final_system_msg.replace("__LC_PAYLOAD__", "{payload}")
        # Support the mapping for user-written {{payload}}
        final_system_msg = final_system_msg.replace("__LC_input__", "{input}")
        final_system_msg = final_system_msg.replace("__LC_context__", "{context}")
        final_system_msg = final_system_msg.replace("__LC_payload__", "{payload}")
        
        # Count consecutive tool execution turns to prevent infinite looping
        consecutive_tool_turns = 0
        for msg in reversed(messages_for_agent):
            msg_type = type(msg).__name__
            if msg_type == "ToolMessage" or getattr(msg, "type", "") == "tool":
                consecutive_tool_turns += 1
            elif msg_type == "AIMessage" or getattr(msg, "type", "") == "assistant":
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    continue
            else:
                break
        
        # If the agent has made more than 15 consecutive tool turns, inject a strong warning to stop calling tools
        effective_llm_with_tools = llm_with_tools
        if consecutive_tool_turns >= 15:
            logger.warning(f"Agent {node_config.id} reached max consecutive tool turns ({consecutive_tool_turns}). Instructing agent to stop calling tools and finalize.")
            final_system_msg += (
                "\n\n[SYSTEM WARNING]\n"
                "You have reached the maximum search limit (15 tool turns). "
                "You MUST NOT call any more tools. You must formulate your final answer now using the information already retrieved, "
                "or explain what is missing if you cannot answer."
            )
            
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", final_system_msg),
            ("placeholder", "{messages}"),
        ])
        
        agent_chain = (agent_prompt | effective_llm_with_tools).with_retry(stop_after_attempt=3)
        
        # Prepare state with injected variables
        last_msg_content = messages_for_agent[-1].content if messages_for_agent else ""
        ctx = state.get("context", {})
        
        if isolation_mode:
            # ISOLATION MODE: No context dump, no payload — exactly like standalone routers.
            # The system prompt already has everything via {{agent_input_*}} interpolation.
            # EXCEPTION: Orchestrator isolation passes clean agent reports via {context}
            # so the orchestrator prompt's {{context}} placeholder gets the specialist summaries.
            _cd_context_val = _orch_reports_text if _orch_reports_text else "{}"
            current_state = {
                **state,
                "messages": messages_for_agent,
                "input": last_msg_content,
                "context": _cd_context_val,
                "payload": "{}",
            }
            logger.debug(f"[CD_ISOLATION] Agent {node_config.id}: Sending {len(final_system_msg)} char "
                         f"system prompt + 1 HumanMessage (context={len(_cd_context_val)} chars)")
        else:
            # STANDARD MODE for non-CD-isolation agents.
            # When in a CD pipeline (agent_input_* keys exist), the orchestrator
            # should NOT see raw OCR text — in standalone CD, the orchestrator
            # only receives agent summaries (overview/pros/cons). Filter bulk data.
            # ALWAYS filter bulk data from context dump to prevent token limit crashes
            # Large OCR and aggregated results should only be accessed via specific prompt interpolation, not raw context dumps.
            BULK_KEYS = {
                "latest_ocr_output", "ocr_output", "aggregated_results",
                "agg_aggregator", "current_batch", "loop_metadata",
                "nodes", "headers", "payload", "query_params",
                "classified_pages_grouped", "original_data", "ocr",
                "latest_classification", "items", "item",
                "technical", "transformed_data", "code_result", "latest_code_result",
                "aggregated_classified_documents",
            }
            # The list above is a guess about which keys are too large to be
            # worth sending. It is occasionally wrong: a workflow whose whole
            # purpose is reading a document finds the document quietly withheld
            # from the very node meant to read it, with nothing said. So the
            # workflow can overrule it in either direction.
            _exclude = set(getattr(node_config, "agent_context_exclude", None) or [])
            _include = set(getattr(node_config, "agent_context_include", None) or [])
            if _include:
                logger.info(
                    f"Agent {node_config.id}: keeping {sorted(_include)} in context "
                    f"despite the default exclusions"
                )

            filtered_ctx = {}
            for k, v in ctx.items():
                if k in _include:
                    filtered_ctx[k] = v
                    continue
                if k in _exclude:
                    continue
                if k in BULK_KEYS:
                    continue
                # Encoded binary — a scanned page, a signature crop, a selfie.
                # The node that needs one reads it by name and sends it as an
                # image part, which is the only way a model can use it. Dropped
                # into a text context dump it is noise the model cannot decode,
                # and being large it crowds out the document the agent was
                # actually asked to read. WorkflowService._trim_response_context
                # already drops these on the way out; this is the same rule on
                # the way in. Naming a key in "Always Include" still overrides
                # this, since that check runs first.
                if k == "base64" or k.endswith(("_base64", "_b64")):
                    continue
                if k.startswith(("ocr_output_", "agg_", "split_", "classification_")):
                    continue
                if k.startswith(agent_input_prefix):
                    filtered_ctx[k] = f"[{len(v)} chars of routed text]" if isinstance(v, str) and v else "[empty]"
                    continue
                if k.startswith(output_key_prefix):
                    filtered_ctx[k] = v
                    continue
                
                # Context truncation strategy — per-node config
                ctx_mode = getattr(node_config, "context_mode", "smart") or "smart"
                ctx_max = getattr(node_config, "max_context_chars", None)
                
                if ctx_mode == "passthrough":
                    filtered_ctx[k] = v
                elif ctx_mode == "strict":
                    if isinstance(v, str) and len(v) > 500:
                        filtered_ctx[k] = v[:200] + f"... [{len(v)} chars total]"
                    elif isinstance(v, (list, dict)) and len(str(v)) > 1000:
                        filtered_ctx[k] = f"[{type(v).__name__} with {len(v)} items]"
                    else:
                        filtered_ctx[k] = v
                else:  # "smart" — default
                    limit = ctx_max or 20000
                    if isinstance(v, str) and len(v) > limit:
                        filtered_ctx[k] = v[:limit] + f"... [{len(v)} chars total]"
                    elif isinstance(v, (list, dict)) and len(str(v)) > limit:
                        filtered_ctx[k] = f"[{type(v).__name__} with {len(v)} items]"
                    else:
                        filtered_ctx[k] = v
                
            context_json = json.dumps(filtered_ctx, indent=2, default=str)
            logger.info(f"Agent {node_config.id}: Filtered context from {len(str(ctx))} to {len(context_json)} chars")
            
            current_state = {
                **state,
                "messages": messages_for_agent, 
                "input": last_msg_content,
                "context": context_json,
                "payload": json.dumps(ctx.get("payload", {}), indent=2, default=str) if isinstance(ctx, dict) else "{}"
            }
        
        logger.debug(f"Agent {node_config.id}: Calling LLM with {len(messages_for_agent)} messages")
        
        import asyncio as _asyncio
        
        MAX_EMPTY_RETRIES = 2
        response = None
        last_error = None
        
        for attempt in range(MAX_EMPTY_RETRIES + 1):
            try:
                response = await agent_chain.ainvoke(current_state)
            except Exception as e:
                logger.error(f"LLM call failed for agent {node_config.id} (attempt {attempt+1}): {str(e)}")
                last_error = e

                # Groq open-weights models sometimes emit a tool call as raw TEXT
                # instead of a native function call → Groq rejects with
                # tool_use_failed / failed_generation. Recover the call from the
                # error payload and run it instead of blind-retrying the prompt.
                recovered_calls = None
                _err_str = str(e)
                if ("tool_use_failed" in _err_str) or ("failed_generation" in _err_str):
                    recovered_calls = _extract_failed_generation_tool_calls(_err_str)
                if recovered_calls:
                    logger.info(
                        f"[AGENT] {node_config.id}: Recovered {len(recovered_calls)} "
                        f"tool call(s) from Groq failed_generation — executing instead of retrying"
                    )
                    response = AIMessage(content="", tool_calls=recovered_calls)
                    break

                if attempt < MAX_EMPTY_RETRIES:
                    wait_secs = 2 ** attempt  # 1s, 2s
                    logger.info(f"Retrying agent {node_config.id} in {wait_secs}s...")
                    await _asyncio.sleep(wait_secs)
                    continue
                # All retries exhausted on exception
                error_data = {"message": str(e), "type": type(e).__name__}
                response = AIMessage(content=f"[ERROR] Agent {node_config.id} failed: {str(e)}")
                res_dict = {
                    "messages": state_updates + [response], 
                    "active_agent": node_config.id,
                    "error": error_data,
                    "error_source": node_config.id
                }
                if final_context:
                    res_dict["context"] = final_context
                return res_dict
            
            # Check if response is empty — if so, retry
            _resp_content = getattr(response, 'content', '') or ''
            _resp_tools = getattr(response, 'tool_calls', None) or []
            _has_any = (isinstance(_resp_content, str) and _resp_content.strip()) or _resp_content or _resp_tools
            
            if _has_any:
                break  # Got a real response, proceed
            
            if attempt < MAX_EMPTY_RETRIES:
                wait_secs = 2 ** attempt
                logger.warning(f"Agent {node_config.id} returned empty response (attempt {attempt+1}/{MAX_EMPTY_RETRIES+1}). Retrying in {wait_secs}s...")
                await _asyncio.sleep(wait_secs)
            else:
                logger.error(f"Agent {node_config.id} returned empty response after {MAX_EMPTY_RETRIES+1} attempts. Producing fallback error.")
                response = AIMessage(content=f"[ERROR] Agent {node_config.id} returned empty response after {MAX_EMPTY_RETRIES+1} attempts. The LLM provider may be rate-limiting or the context may be too large.")
        
        if response is None:
            response = AIMessage(content=f"[ERROR] Agent {node_config.id} produced no response.")
            
        # Process response

        response_content = getattr(response, 'content', str(response))
        response_tool_calls = getattr(response, 'tool_calls', None) or []
        
        # 🛠️ SGLang / Text Tool Call Fallback Parser:
        # If response.tool_calls is empty but content contains <tool_call>...</tool_call>
        if not response_tool_calls and isinstance(response_content, str) and "<tool_call>" in response_content:
            import re
            import uuid
            tool_call_matches = re.findall(r'<tool_call>([\s\S]*?)<\/tool_call>', response_content, re.IGNORECASE)
            parsed_tc_list = []
            for match in tool_call_matches:
                try:
                    tc_raw = match.strip()
                    tc_json = json.loads(tc_raw)
                    tc_name = tc_json.get("name")
                    tc_args = tc_json.get("arguments") or tc_json.get("args") or {}
                    if isinstance(tc_args, str):
                        try:
                            tc_args = json.loads(tc_args)
                        except Exception:
                            tc_args = {"query": tc_args}
                    if tc_name:
                        parsed_tc_list.append({
                            "name": tc_name,
                            "args": tc_args if isinstance(tc_args, dict) else {"input": tc_args},
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "tool_call"
                        })
                except Exception as parse_err:
                    logger.warning(f"[AGENT] {node_config.id}: Failed to parse text-based <tool_call>: {parse_err}")
            
            if parsed_tc_list:
                logger.info(f"[AGENT] {node_config.id}: Extracted {len(parsed_tc_list)} <tool_call> text tag(s) from SGLang output: {[t['name'] for t in parsed_tc_list]}")
                response_tool_calls = parsed_tc_list
                if hasattr(response, 'tool_calls'):
                    response.tool_calls = parsed_tc_list
        
        # If we have reached the limit of consecutive turns, strip tool calls to prevent further loop execution
        if consecutive_tool_turns >= 15 and response_tool_calls:
            logger.warning(f"Agent {node_config.id} attempted to call tools after reaching turn limit. Stripping tool calls.")
            if hasattr(response, 'tool_calls'):
                response.tool_calls = []
            if isinstance(getattr(response, 'additional_kwargs', None), dict) and 'tool_calls' in response.additional_kwargs:
                del response.additional_kwargs['tool_calls']
            response_tool_calls = []
            if not response_content or not response_content.strip():
                response.content = "I have reached the limit of database queries and could not find any more information to answer your request."
                response_content = response.content
        
        has_tool_calls = bool(response_tool_calls)
        has_content = bool(response_content) and (not isinstance(response_content, str) or bool(response_content.strip()))
        
        if not has_content and not has_tool_calls:
            logger.warning(f"Agent {node_config.id} returned empty response with no tool calls (post-retry)")
            fallback_msg = AIMessage(content=f"[ERROR] Agent {node_config.id} returned an empty response. The LLM may be overloaded or the input exceeded token limits.")
            res_dict = {
                "messages": state_updates + [fallback_msg],
                "active_agent": node_config.id
            }
            if final_context:
                res_dict["context"] = final_context
            return res_dict
        
        if has_tool_calls:
            logger.debug(f"Tool calls detected: {[t.get('name') for t in response_tool_calls]}")
        
        # Add node ID to metadata
        if hasattr(response, 'response_metadata'):
            _llm_dict = None
            if llm_config:
                if hasattr(llm_config, 'model_dump'):
                    _llm_dict = llm_config.model_dump(exclude_none=True)
                elif hasattr(llm_config, 'dict'):
                    _llm_dict = llm_config.dict(exclude_none=True)
                elif isinstance(llm_config, dict):
                    _llm_dict = llm_config
            response.response_metadata = {
                **(response.response_metadata or {}),
                "active_agent": node_config.id,
                "llm_config": _llm_dict,
            }
        if hasattr(response, 'additional_kwargs'):
            response.additional_kwargs = {**(response.additional_kwargs or {}), "active_agent": node_config.id}
        
        final_messages = state_updates + [response]
        res_dict = {
            "messages": final_messages, 
            "active_agent": node_config.id,
            "agent_name": getattr(node_config, "label", None) or node_config.id
        }
        
        # ══════════════════════════════════════════════════════════════════════
        # OUTPUT PARSING: Parse agent JSON output into context
        # ══════════════════════════════════════════════════════════════════════
        # In standalone CD, run_agent() returns {"overview":"...", "pros":"...", "cons":"..."}
        # as a parsed dict. In mesh, the agent returns an AIMessage with JSON text.
        # We must parse it and inject into context so the postprocessor/payload builder can find it.
        if has_content and response_content:
            try:
                import re as _re
                from json_repair import loads as repair_loads
                parsed = None
                
                # Stage 1: Try targeted regex for simple (non-nested) JSON
                json_match = _re.search(r'\{[^{}]*"overview"[^{}]*\}', response_content, _re.DOTALL)
                if not json_match:
                    json_match = _re.search(r'\{[^{}]*"recommendation"[^{}]*\}', response_content, _re.DOTALL)
                if not json_match:
                    # Stage 2: Greedy match for nested JSON (orchestrator with arrays/objects)
                    json_match = _re.search(r'\{.*\}', response_content, _re.DOTALL)
                
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        # Stage 3: json_repair for malformed/truncated JSON
                        parsed = repair_loads(json_match.group())
                        logger.info(f"[OUTPUT_PARSER] Agent {node_config.id}: json_repair recovered nested JSON")
                
                if not parsed:
                    # Stage 4: Last resort — let json_repair try the entire response
                    repaired = repair_loads(response_content)
                    if isinstance(repaired, dict):
                        parsed = repaired
                        logger.info(f"[OUTPUT_PARSER] Agent {node_config.id}: json_repair recovered from full response")
                
                if parsed and isinstance(parsed, dict):
                    # ── Where the answer goes, when the workflow says so ──────
                    # Below this, an answer is only filed when it happens to
                    # contain particular field names. Those names come from one
                    # earlier use case, so an agent asked for anything else has
                    # its answer produced, logged, and silently dropped — the
                    # node reports success and the next node finds nothing.
                    #
                    # Naming a key here files whatever the agent returned, under
                    # that name, whatever its shape.
                    output_key = (getattr(node_config, "agent_output_key", None) or "").strip()
                    if output_key:
                        if final_context is None:
                            final_context = {**context}
                        final_context[output_key] = parsed
                        logger.info(
                            f"[OUTPUT_PARSER] Agent {node_config.id}: filed answer "
                            f"under '{output_key}' ({len(parsed)} top-level key(s))"
                        )

                    cd_context = {}
                    # Specialist agent fields + Orchestrator decision fields
                    for field in ("overview", "pros", "cons", "missing_fields", "confidence",
                                  "recommendation", "case_type", "reasoning", "key_risks",
                                  "conditions", "credit_memo", "manual_review_required", "priority"):
                        if field in parsed:
                            cd_context[field] = parsed[field]
                    
                    has_specialist = cd_context.get("overview") or cd_context.get("pros") or cd_context.get("cons") or cd_context.get("missing_fields")
                    has_orchestrator = cd_context.get("recommendation") or cd_context.get("case_type")
                    
                    if has_specialist or has_orchestrator:
                        # Use node LABEL (user-visible name) for the key, not random node ID
                        node_label = (getattr(node_config, "label", None) or "").strip()
                        if referenced_input and referenced_input.startswith(agent_input_prefix):
                            agent_key = referenced_input.replace(agent_input_prefix, "")
                        elif node_label:
                            agent_key = _re.sub(r'[^a-z0-9]+', '_', node_label.lower()).strip('_')
                        else:
                            agent_key = node_config.id.replace("llm_agent_", "")
                        unique_key = f"{output_key_prefix}{agent_key}"
                        
                        # Also provide fallback with full node ID
                        unique_key_full = f"{output_key_prefix}{node_config.id}"
                        
                        cd_context["cd_agent_key"] = agent_key
                        cd_context["cd_agent_input_ref"] = referenced_input
                        # Include agent name for webhook grouping
                        cd_context["agent_name"] = node_label or node_config.id
                        
                        if final_context is None:
                            final_context = {**context}
                        
                        final_context[unique_key] = cd_context
                            
                        logger.info(f"[OUTPUT_PARSER] Agent {node_config.id}: Parsed output → "
                                    f"key={unique_key}, label='{node_label}', "
                                    f"specialist={bool(has_specialist)}, orchestrator={bool(has_orchestrator)}")
                    else:
                        logger.warning(f"[OUTPUT_PARSER] Agent {node_config.id}: Parsed JSON but "
                                       f"overview/pros/cons all empty")
                else:
                    logger.debug(f"[OUTPUT_PARSER] Agent {node_config.id}: No JSON dict found in response.")
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"[OUTPUT_PARSER] Agent {node_config.id}: Failed to parse JSON output: {e}")
        

        
        # Auto-persist conversation turn to vector store if vector_memory is enabled
        if mem_type == "vector_memory" and response_content:
            user_query = extract_latest_user_query(messages)
            if user_query:
                await persist_vector_memory_async(user_query, response_content, node_config, thread_id)

        if final_context:
            res_dict["context"] = final_context
        
        if memory_summary_update:
            res_dict["memory_summary"] = memory_summary_update

        return res_dict

    
    async def run_checked(state: AgentState):
        """run_once, asked again if the answer does not meet the stated bar.

        Models are not steady. The same document that returns two hundred lines
        returns eight on one attempt in five - valid JSON, no error, nothing to
        alert anyone. Downstream cannot tell a thin answer from a thin document,
        so the check belongs here, where the input is still in view.

        The first acceptable answer wins. If every attempt is rejected the
        fullest one is kept rather than the last, since a later attempt is not
        automatically better, and the rejection is reported alongside the answer
        so that whoever reads it knows.
        """
        test = getattr(node_config, "agent_retry_when", None)
        if not test:
            return await run_once(state)

        key = (getattr(node_config, "agent_output_key", None) or "").strip()
        attempts = max(0, int(getattr(node_config, "agent_retry_max", 2) or 0)) + 1
        tried = []
        for attempt in range(1, attempts + 1):
            result = await run_once(state)
            answer = ((result or {}).get("context") or {}).get(key) if key else None
            poor = _answer_is_poor(test, answer, node_config.id)
            tried.append((result, answer, poor))
            if not poor:
                if attempt > 1:
                    logger.info(f"[AGENT_RETRY] {node_config.id}: attempt "
                                f"{attempt} of {attempts} was accepted")
                return result
            logger.warning(
                f"[AGENT_RETRY] {node_config.id}: attempt {attempt} of "
                f"{attempts} rejected by '{test}' - asking again"
            )

        # every attempt fell short: keep the fullest, and say so plainly
        best, answer, _ = max(tried, key=lambda t: len(json.dumps(t[1], default=str))
                              if t[1] is not None else 0)
        logger.error(
            f"[AGENT_RETRY] {node_config.id}: all {attempts} attempts were "
            f"rejected by '{test}'. Keeping the fullest of them; the answer "
            f"under '{key}' is likely incomplete."
        )
        res = dict(best or {})
        ctx = {**((best or {}).get("context") or {})}
        ctx[f"{key or node_config.id}_retry"] = {
            "test": test, "attempts": attempts, "all_rejected": True,
        }
        res["context"] = ctx
        return res

    async def execute_agent(state: AgentState):
        """One call, or one call per batch of a configured list.

        With batching off this hands straight to run_checked and the agent
        behaves exactly as it did before. With it on, the same prompt is run
        once per batch - each seeing only its own slice - and the answers are
        combined into the single answer the rest of the workflow expects.
        """
        if not getattr(node_config, "agent_batch_enabled", False):
            return await run_checked(state)

        # Each batch's answer is collected from the key the agent files it
        # under, and an agent files nothing unless it has been told what to call
        # its answer. Without that name there is nowhere to read the batches
        # back from, so batching would make every call and keep none of them -
        # far worse than not batching at all. One call is done instead, and the
        # reason is stated rather than left to be discovered downstream.
        output_key = (getattr(node_config, "agent_output_key", None) or "").strip()
        if not output_key:
            logger.error(
                f"[AGENT_BATCH] {node_config.id}: batching needs 'Save Answer To' "
                f"set - without it each batch's answer is discarded and nothing "
                f"can be combined. Running a single call instead."
            )
            return await run_once(state)

        context = state.get("context", {}) or {}
        spec = getattr(node_config, "agent_batch_items", None)
        items = _resolve_items(spec, context)
        size = getattr(node_config, "agent_batch_size", 50) or 50

        if not items:
            logger.warning(
                f"[AGENT_BATCH] {node_config.id}: batching is on but "
                f"'{spec}' gave no list to work through - running once. Point "
                f"agent_batch_items at a context key holding a list."
            )
            return await run_once(state)

        var = getattr(node_config, "agent_batch_var", None) or "batch_items"
        parts = _chunk(items, size)
        limit = max(1, int(getattr(node_config, "agent_batch_max_parallel", 1) or 1))
        keep_going = (getattr(node_config, "agent_batch_on_error", "continue")
                      != "fail")
        gate = asyncio.Semaphore(limit)

        logger.info(
            f"[AGENT_BATCH] {node_config.id}: {len(items)} items in "
            f"{len(parts)} batches of up to {size}, {limit} at a time, "
            f"as '{{{{ {var} }}}}'"
        )

        async def one_batch(n: int, part: List[Any]):
            async with gate:
                batch_state = {
                    **state,
                    "context": {**context,
                                var: _render_items(part),
                                f"{var}_list": part,
                                f"{var}_index": n,
                                f"{var}_total": len(parts)},
                }
                try:
                    return n, await run_checked(batch_state), None
                except Exception as err:              # noqa: BLE001
                    if not keep_going:
                        raise
                    logger.warning(f"[AGENT_BATCH] {node_config.id}: batch "
                                   f"{n}/{len(parts)} failed ({err}); its "
                                   f"fields are left empty")
                    return n, None, str(err)

        done = await asyncio.gather(*(one_batch(i + 1, p)
                                      for i, p in enumerate(parts)))

        answers, failed, last = [], [], None
        for n, result, err in done:
            if err is not None:
                failed.append(n)
                continue
            last = result or last
            answers.append(((result or {}).get("context") or {}).get(output_key))

        merged, clashes = _merge_answers(
            answers,
            getattr(node_config, "agent_batch_merge", None) or "deep_merge",
            getattr(node_config, "agent_batch_merge_key", None),
        )

        # A batch can come back without failing and still say nothing usable -
        # the call succeeded but the answer would not parse. That is a different
        # thing from a failure and is counted separately, because "13 batches,
        # none failed" alongside an empty answer is otherwise baffling.
        answered = sum(1 for a in answers if a is not None)

        res_dict = dict(last or {})
        final_context = {**context, **(res_dict.get("context") or {})}
        final_context[output_key] = merged
        # Facts about the run, for whoever is configured to act on them. The
        # node states what happened; it does not decide what it means.
        final_context[f"{output_key}_batching"] = {
            "items": len(items),
            "batches": len(parts),
            "batch_size": size,
            "answered": answered,
            "failed_batches": failed,
            "no_usable_answer": len(parts) - answered - len(failed),
            "disagreements": clashes,
            "disagreement_count": len(clashes),
        }
        res_dict["context"] = final_context
        if merged is None:
            # Every call was made and nothing usable came back. Silence here
            # would leave the next node with an empty answer and no reason.
            logger.error(
                f"[AGENT_BATCH] {node_config.id}: all {len(parts)} batches "
                f"produced no usable answer under '{output_key}' "
                f"({len(failed)} failed outright). '{output_key}' is empty."
            )
        else:
            logger.info(
                f"[AGENT_BATCH] {node_config.id}: {answered}/{len(parts)} "
                f"batches answered"
                + (f", {len(failed)} failed {failed}" if failed else "")
                + (f", {len(clashes)} disagreement(s) between batches" if clashes else "")
            )
        return res_dict

    executor = RunnableLambda(execute_agent)
    setattr(executor, "name", node_config.id)
    return executor
