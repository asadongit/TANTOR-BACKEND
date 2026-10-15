"""
DeepAgent Node Builder
Integrates LangChain's DeepAgent with the AgentMesh Core.
Provides advanced planning, autonomous delegation, and persistent context.
"""

import logging
import re
from typing import List, Callable
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, LocalShellBackend, StateBackend, ContextHubBackend
# Removed custom VFS import as we are using the SDK's native tools

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Core.ToolRegistry import registry
from Apps.AgentMesh.Utilities.LLM.llm_factory import init_chat_model
from .utils import (
    interpolate,
    process_tools_with_overrides,
    build_subagent_tool_set,
    resolve_parent_attr_target,
    prepare_agent_memory,
    retrieve_vector_memory_async,
    persist_vector_memory_async,
    extract_latest_user_query,
    ensure_non_empty_user_input,
)

logger = logging.getLogger(__name__)

# Cache compiled DeepAgent runnables to avoid rebuilding planner graph on every single invocation
_agent_cache = {}

def build_llm_deep_agent(node_config: NodeConfig, all_nodes: List[NodeConfig] = None, resolver: Callable = None) -> Callable:
    """
    Build a DeepAgent node with native LangChain support for:
    - write_todos (Internal planning)
    - task (Autonomous sub-delegation)
    - Virtual Filesystem (State management)
    
    Enhanced (Apr 2026): Now resolves local Mesh subagents and supports Skill discovery.
    """
    llm = init_chat_model(node_config.llm)
    
    # 1. Prepare Tools
    raw_tools = registry.get_tools(node_config.tools)
    tools = process_tools_with_overrides(raw_tools, node_config.tool_configs)

    # Inject parent node context into run_subagent so sub-agents inherit the
    # parent's configured tools + LLM config (mirror llm_agent.py). Set on BOTH
    # raw instances (captured by the wrapper closure) and processed instances.
    # resolve_parent_attr_target digs through StructuredTool wrappers so the
    # underlying SubAgentTool instance always receives the parent context.
    parent_tools_for_subagent = build_subagent_tool_set(tools, node_config.tool_configs)
    for t in raw_tools:
        target = resolve_parent_attr_target(t)
        if hasattr(target, "parent_tools"):
            target.parent_tools = parent_tools_for_subagent
            target.parent_llm_config = node_config.llm
            target.parent_tool_configs = node_config.tool_configs
    for t in tools:
        target = resolve_parent_attr_target(t)
        if hasattr(target, "parent_tools"):
            target.parent_tools = parent_tools_for_subagent
            target.parent_llm_config = node_config.llm
            target.parent_tool_configs = node_config.tool_configs

    # SKILL DISCOVERY (2026 Pattern): 
    if node_config.task and "skill:" in node_config.task.lower():
        logger.info(f"DeepAgent {node_config.id}: Prompt-based skill detected.")
    
    if node_config.skills:
        logger.info(f"DeepAgent {node_config.id}: Loading linked skills: {node_config.skills}")
    
    # 2. Configure Backend Persistence (Isolated VFS)
    root_dir = node_config.deep_agent_root_dir or settings.DEEP_AGENTS_MOUNT
    backend_type = (node_config.deep_agent_backend or "sqlite").lower()
    
    # Map durable DB types to StateBackend for LangGraph checkpointing
    if backend_type in ["local", "sandbox"]:
        if backend_type == "sandbox":
            backend = LocalShellBackend(root_dir=root_dir)
        else:
            backend = FilesystemBackend(root_dir=root_dir)
    elif backend_type == "contexthub":
        # ContextHubBackend (May 2026) for versioned skills and memories in LangSmith
        hub_repo = node_config.label or node_config.id
        backend = ContextHubBackend(hub_repo)
    else:
        # Use StateBackend for postgres, sqlite, or durable (standardizes on LangGraph checkpoints)
        # We use a factory lambda because StateBackend now requires a runtime at init.
        backend = lambda runtime: StateBackend(runtime=runtime)

    # 3. Resolve Subagents (A2A / Mesh Delegation)
    resolved_subagents = []
    if node_config.subagents and all_nodes and resolver:
        for sa_id in node_config.subagents:
            sa_config = next((n for n in all_nodes if n.id == sa_id), None)
            if sa_config:
                worker_runnable = resolver(sa_config)
                if not hasattr(worker_runnable, "with_config"):
                    worker_runnable = RunnableLambda(worker_runnable)
                
                # Sanitize name to snake_case to match LLM expectations (e.g., "QA Engineer" -> "qa_engineer")
                raw_name = sa_config.label or sa_config.id
                sa_name = re.sub(r'[^a-zA-Z0-9_]', '_', raw_name.lower().strip())
                sa_name = re.sub(r'_+', '_', sa_name).strip('_')
                
                compiled_sa = {
                    "name": sa_name,
                    "description": sa_config.task or "",
                    "runnable": worker_runnable
                }
                resolved_subagents.append(compiled_sa)

    # 4. Initialize DeepAgent
    # Scoped skills path will be dynamically resolved inside execute_deep_agent based on thread_id
    # Sanitization: Filter out name duplicates from user-provided tools
    combined_tools = {t.name if hasattr(t, 'name') else str(t): t for t in tools}
    sanitized_tools = list(combined_tools.values())

    async def execute_deep_agent(state: AgentState):
        """
        Wrapped execution logic with VFS bridge for persistence.
        """
        # 0. Context Extraction
        messages = state.get("messages", [])
        context = state.get("context", {})
        
        # Enforce delegation depth recursion limit (A2A loop protection)
        subagent_depth = context.get("__subagent_depth", 0)
        if subagent_depth >= 4:
            raise RecursionError("Max DeepAgent delegation recursion depth exceeded to prevent cyclic routing loops.")

        # Memory management using consolidated helper
        messages_for_agent, memory_summary_update = await prepare_agent_memory(messages, node_config, state=state)
        # Empty-input guard: never feed the DeepAgent an empty/blank user turn
        messages_for_agent = ensure_non_empty_user_input(messages_for_agent)
        # Thread ID should try to find the actual active thread from state
        thread_id = state.get("thread_id") or context.get("thread_id") or "default_thread"
        mem_config = getattr(node_config, "memory", None) or {}
        mem_type = mem_config.get("type_name", "buffer_memory")
        
        import os
        from Apps.AgentMesh.Config import settings
        skills_root = os.path.abspath(os.path.join(settings.TEMP_MOUNT, thread_id, "skills", node_config.id))
        skills_paths = [skills_root] if node_config.skills else []
        
        base_prompt = node_config.task or "You are a professional assistant."
        interpolated_prompt = interpolate(base_prompt, context)
        
        # Vector memory retrieval (DeepAgent support for vector_memory)
        additional_context = ""
        if mem_type == "vector_memory" and messages_for_agent:
            additional_context = await retrieve_vector_memory_async(messages_for_agent, node_config, thread_id)
        
        # Final Prompt Calculation (Normal interpolation for task)
        final_prompt = re.sub(r"skill:\s*[\w\.\-/]+", "", interpolated_prompt, flags=re.IGNORECASE).strip()
        if not final_prompt:
             final_prompt = "You are a professional assistant."
        if additional_context:
            final_prompt += additional_context
        
        display_name = node_config.label or node_config.id
        logger.debug(f"DeepAgent {display_name}: invoking thread '{thread_id}' with isolated VFS")
        
        try:
            # Dynamically fetch referenced skills from database and write to disk
            skills_content_map = {}
            if node_config.skills:
                from Database.Database import AsyncSessionLocal
                from Database import AgentMeshCrud
                import os
                import shutil
                
                # Clear previous skill files for this node to avoid stale files
                if os.path.exists(skills_root):
                    try:
                        shutil.rmtree(skills_root)
                    except Exception as ex:
                        logger.warning(f"Failed to clear skills root {skills_root}: {ex}")
                os.makedirs(skills_root, exist_ok=True)
                
                async with AsyncSessionLocal() as db:
                    for skill_ref in node_config.skills:
                        skill = await AgentMeshCrud.get_skill(db, skill_ref)
                        if not skill:
                            skill = await AgentMeshCrud.get_skill_by_name(db, skill_ref)
                        
                        if skill:
                            skill_dir = os.path.join(skills_root, skill.name)
                            os.makedirs(skill_dir, exist_ok=True)
                            skill_file = os.path.join(skill_dir, "SKILL.md")
                            content_str = f"---\nname: {skill.name}\ndescription: {skill.description or 'SOP'}\ntags: {skill.tags or []}\n---\n\n{skill.content}"
                            with open(skill_file, "w", encoding="utf-8") as f:
                                f.write(content_str)
                            skills_content_map[skill.name] = content_str
                            logger.info(f"DeepAgent {node_config.id}: Dynamically loaded database skill: {skill.name}")
            # Optimize context dictionary for serialization to prevent database/checkpointer bloat
            optimized_context = {}
            for k, v in context.items():
                if k in ("vfs", "nodes", "messages"):
                    continue
                if isinstance(v, str) and (len(v) > 2000 or v.startswith("data:") or ";base64," in v):
                    continue
                optimized_context[k] = v
            
            # Propagate the incremented delegation depth to subagents
            optimized_context["__subagent_depth"] = subagent_depth + 1

            # Get compiled agent from cache, or build and cache it
            cache_key = (
                node_config.id,
                final_prompt,
                tuple(sorted(t.name if hasattr(t, "name") else str(t) for t in sanitized_tools)),
                tuple(sorted(skills_paths)),
                tuple(sorted(node_config.subagents or [])),
                node_config.harness_profile,
                node_config.code_interpreter
            )
            
            agent = _agent_cache.get(cache_key)
            if not agent:
                logger.debug(f"Compiling and caching DeepAgent for node: {node_config.id}")
                
                # Dynamic Harness Profile registration to resolve profile under custom keys for custom models
                try:
                    from deepagents.profiles.harness.harness_profiles import _get_harness_profile, register_harness_profile
                    from deepagents._models import get_model_identifier, get_model_provider
                    
                    model_id = get_model_identifier(llm)
                    model_prov = get_model_provider(llm) or "agentmeshlitellm"
                    real_provider = getattr(llm, "provider", None)

                    # 1. Map underlying provider defaults (e.g. 'groq' -> 'agentmeshlitellm')
                    if real_provider:
                        real_profile = _get_harness_profile(real_provider)
                        if real_profile:
                            register_harness_profile(model_prov, real_profile)
                            if model_id:
                                exact_real_profile = _get_harness_profile(f"{real_provider}:{model_id}")
                                if exact_real_profile:
                                    register_harness_profile(f"{model_prov}:{model_id}", exact_real_profile)

                    # 2. Register custom harness profile if explicitly requested in config
                    if node_config.harness_profile:
                        profile = _get_harness_profile(node_config.harness_profile)
                        if profile:
                            if model_id:
                                register_harness_profile(f"{model_prov}:{model_id}", profile)
                            else:
                                register_harness_profile(model_prov, profile)
                except Exception as pe:
                    logger.warning(f"Failed to dynamically register harness profile: {pe}")
                
                agent = create_deep_agent(
                    model=llm,
                    tools=sanitized_tools,
                    skills=skills_paths,
                    system_prompt=final_prompt,
                    backend=backend,
                    subagents=resolved_subagents,
                    context_schema=node_config.context_schema,
                    interrupt_on={t: True for t in node_config.tools} if getattr(node_config, "interrupt_before", False) else None
                )
                _agent_cache[cache_key] = agent

            # 3. Prepare State with Key Bridging (files <-> vfs)
            agent_files = {}
            if "vfs" in state and isinstance(state["vfs"], dict):
                 agent_files = {**state["vfs"].get("files", {})}

            # Inject dynamically loaded skills into VFS for StateBackend compatibility
            if node_config.skills:
                from datetime import datetime
                now_str = datetime.utcnow().isoformat() + "Z"
                for skill_name, skill_md in skills_content_map.items():
                    file_data = {
                        "content": skill_md,
                        "encoding": "utf-8",
                        "created_at": now_str,
                        "modified_at": now_str
                    }
                    # Register multiple path formats to ensure match in state VFS
                    safe_rel = os.path.relpath(skills_root, os.path.abspath(".")).replace("\\", "/")
                    rel_skills_path = f"{safe_rel}/{skill_name}/SKILL.md"
                    agent_files[f"./{rel_skills_path}"] = file_data
                    agent_files[rel_skills_path] = file_data
                    agent_files[f"/{rel_skills_path}"] = file_data

            agent_state = {
                "messages": messages_for_agent,
                "files": agent_files
            }

            agent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "context": optimized_context,
                    "checkpoint_ns": f"deepagent_{node_config.id}",
                    "vfs_root": f"{root_dir}/{thread_id}" 
                },
                # Enforce recursion limit on internal tool calls
                "recursion_limit": getattr(node_config, "max_iterations", 10)
            }
            
            # 2. RUN THE DEEP AGENT with timeout protection
            import asyncio
            timeout = getattr(node_config, "timeout_seconds", 300)
            final_result = await asyncio.wait_for(
                agent.ainvoke(agent_state, config=agent_config),
                timeout=timeout
            )
            
            # 3. Extract results and update state
            new_messages = final_result.get("messages", [])
            for msg in new_messages:
                if type(msg).__name__ == "AIMessage":
                    if hasattr(msg, 'response_metadata'):
                        msg.response_metadata = {**(msg.response_metadata or {}), "active_agent": node_config.id}
                    if hasattr(msg, 'additional_kwargs'):
                        msg.additional_kwargs = {**(msg.additional_kwargs or {}), "active_agent": node_config.id}
            
            res_dict = {
                "messages": new_messages,
                "active_agent": node_config.id
            }
            
            # 4. Extract results and Map Keys back
            # We capture the 'files' channel updated by the SDK and sync it back to 'vfs'
            updated_files = final_result.get("files", {})
            
            res_dict["vfs"] = {
                "files": updated_files,
                "dirs": [] # SDK uses flat paths, dirs are derived on the fly
            }
            

            
            # Clear HITL markers
            res_dict["context"] = {**context, "hitl_action": None, "hitl_feedback": None}

            # Auto-persist conversation turn to vector store if vector_memory is enabled
            if mem_type == "vector_memory" and new_messages:
                user_query = extract_latest_user_query(messages)
                last_ai = next((m for m in reversed(new_messages) if type(m).__name__ == "AIMessage"), None)
                if user_query and last_ai is not None:
                    resp_content = getattr(last_ai, "content", "")
                    await persist_vector_memory_async(user_query, resp_content, node_config, thread_id)

            if memory_summary_update:
                res_dict["memory_summary"] = memory_summary_update

            return res_dict

        except Exception as e:
            # ✅ FIX (Apr 11, 2026): Sanitize for Windows terminal/logs to prevent UnicodeEncodeError
            safe_error = str(e).encode('ascii', 'ignore').decode('ascii')
            display_name = node_config.label or node_config.id
            logger.error(f"DeepAgent {display_name} execution failed: {safe_error}", exc_info=True)
            error_msg = f"[SYSTEM ERROR] DeepAgent '{display_name}' failed: {safe_error}"
            return {
                "messages": [AIMessage(content=error_msg, response_metadata={"active_agent": node_config.id})],
                "active_agent": node_config.id,
                "context": context,
                "error": safe_error
            }

    executor = RunnableLambda(execute_deep_agent)
    setattr(executor, "name", node_config.id)
    return executor
