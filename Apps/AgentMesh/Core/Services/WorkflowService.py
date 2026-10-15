import uuid
import json
import re
from datetime import datetime
from typing import Dict, Any, AsyncGenerator, Optional, Tuple, List

from Apps.AgentMesh.Core.WorkflowRegistry import workflow_registry
from Apps.AgentMesh.Config import settings
from Apps.AgentMesh.Utilities.Memory.checkpointer import get_checkpointer
from Apps.AgentMesh.Utilities.Observer import (
    get_observer,
    get_langfuse_handler,
    log_human_decision,
    flush_langfuse,
)
from Apps.AgentMesh.Core.RedisManager import redis_manager
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Utilities.Serialization.message_helper import serialize_messages
from Apps.AgentMesh.Utilities.Security import redact_sensitive_data, sanitize_for_logging
from langchain_core.messages import HumanMessage, AIMessage
import logging

# Workflow Variables injection
from Database.Database import AsyncSessionLocal
from Database.AgentMeshCrud import list_workflow_variables
from Apps.AgentMesh.Core.Services.WorkflowHistoryService import WorkflowHistoryService

logger = logging.getLogger(__name__)


class WorkflowService:
    """
    Central service for executing and managing workflow runs.
    Decouples execution logic from API endpoints.
    """

    @classmethod
    async def _flush_memory_to_postgres(
        cls, app: Any, config: Dict[str, Any], thread_id: str
    ):
        """No-op: AsyncPostgresSaver automatically persists state directly on every node superstep."""
        pass

    @classmethod
    async def _load_state_from_postgres(
        cls, app: Any, config: Dict[str, Any], thread_id: str
    ) -> bool:
        """No-op: AsyncPostgresSaver reads directly from Postgres checkpoint tables."""
        return True

    @classmethod
    async def _cleanup_completed_thread(cls, thread_id: str) -> bool:
        """Remove terminal LangGraph state and safely release its artifacts.

        Paused and failed runs must not call this helper. Cleanup is idempotent;
        artifact failures remain reference-tracked for the scheduled retry job.
        """
        if not settings.CHECKPOINT_DELETE_ON_COMPLETION:
            return False
        try:
            checkpointer = get_checkpointer(thread_id)
            await checkpointer.adelete_thread(thread_id)
            logger.info(
                "Deleted completed LangGraph checkpoints and released artifacts "
                "for thread %s",
                thread_id,
            )
            return True
        except Exception:
            logger.exception(
                "Completed checkpoint cleanup failed for thread %s; the scheduled "
                "orphan cleanup will retry artifact release",
                thread_id,
            )
            return False

    @classmethod
    async def _finalize_completed_thread(
        cls,
        *,
        thread_id: str,
        workflow_id: str,
        status: str,
        output: str,
        messages: Any,
        context: Any,
        active_agent: Optional[str],
    ) -> bool:
        """Save compact history first; delete checkpoint state only on success."""
        try:
            async with AsyncSessionLocal() as history_db:
                await WorkflowHistoryService.save_completed_history(
                    history_db,
                    thread_id=thread_id,
                    workflow_id=workflow_id,
                    status=status,
                    output=output,
                    messages=messages,
                    context=context,
                    active_agent=active_agent,
                )
        except Exception:
            logger.exception(
                "History persistence failed for completed thread %s; retaining "
                "its checkpoint and artifacts to prevent data loss",
                thread_id,
            )
            return False
        return await cls._cleanup_completed_thread(thread_id)

    @classmethod
    async def _truncate_large_messages_in_checkpoint(
        cls, app: Any, config: Dict[str, Any], workflow_id: str, limit: int = 3000
    ):
        """
        Scans messages stored in the checkpointer state and truncates those exceeding the limit.
        Only truncates messages that triggered return_direct in the current execution turn.
        Uses update_state to overwrite only the oversized messages by their exact ID,
        preserving them in the current run output but keeping memory/checkpoint history lightweight.
        """
        try:
            snapshot = await app.aget_state(config)

            # Clean profiling report HTML from checkpointer context to prevent DB bloat
            context = snapshot.values.get("context", {})
            if context:

                def clean_profiling_data(data):
                    if isinstance(data, str):
                        trimmed = data.strip()
                        if trimmed.startswith("<!doctype html>") and (
                            "ydata-profiling" in trimmed
                            or "pandas-profiling" in trimmed
                            or "Dataset Profile" in trimmed
                            or 'generator" content="YData' in trimmed
                        ):
                            return "[Profiling Report HTML (Pruned for checkpointer performance)]"
                        return data
                    elif isinstance(data, list):
                        return [clean_profiling_data(item) for item in data]
                    elif isinstance(data, dict):
                        return {k: clean_profiling_data(v) for k, v in data.items()}
                    return data

                cleaned_context = clean_profiling_data(context)
                if cleaned_context != context:
                    logger.info(
                        "Oversized profiling HTML detected in context. Pruning for checkpointer performance."
                    )
                    await app.aupdate_state(config, {"context": cleaned_context})
                    snapshot = await app.aget_state(config)

            # If the run is paused (waiting for HITL/human), do not truncate messages yet
            if snapshot.next:
                return

            messages = snapshot.values.get("messages", [])
            if not messages:
                return

            # Find the active/terminal node
            active_node_id = snapshot.values.get("active_agent")
            config_obj = workflow_registry.get_config(workflow_id)

            # Find node configuration
            node_config = None
            if config_obj and active_node_id:
                node_config = next(
                    (n for n in config_obj.nodes if n.id == active_node_id), None
                )

            def is_tool_msg(m):
                if isinstance(m, dict):
                    return (
                        m.get("type") == "tool"
                        or m.get("role") == "tool"
                        or m.get("type") == "ToolMessage"
                    )
                from langchain_core.messages import ToolMessage

                return isinstance(m, ToolMessage) or type(m).__name__ == "ToolMessage"

            def get_t_name(m):
                if isinstance(m, dict):
                    return (
                        m.get("name")
                        or m.get("tool_name")
                        or m.get("additional_kwargs", {}).get("name")
                    )
                return (
                    getattr(m, "name", None)
                    or getattr(m, "tool_name", None)
                    or getattr(m, "additional_kwargs", {}).get("name")
                )

            updates = []

            # Case 1: Terminal message is a ToolMessage (or parallel batch of ToolMessages)
            # This indicates a return_direct execution from a tool node
            if is_tool_msg(messages[-1]):
                # Gather all tool messages in this last turn
                last_turn_tools = []
                for msg in reversed(messages):
                    if is_tool_msg(msg):
                        last_turn_tools.append(msg)
                    else:
                        break

                # Check if return_direct is configured for any of these tools in the active node context
                for msg in last_turn_tools:
                    tool_name = get_t_name(msg)
                    is_rd = False

                    if node_config:
                        # Check agent/node context tool_configs override
                        if node_config.type in ("llm_agent", "tool_node"):
                            t_configs = getattr(node_config, "tool_configs", {}) or {}
                            if t_configs.get(tool_name, {}).get("return_direct"):
                                is_rd = True

                        # Check node-level return_direct
                        if getattr(node_config, "return_direct", False):
                            is_rd = True

                    if is_rd:
                        # Retrieve content safely and check length limit
                        content = (
                            getattr(msg, "content", "")
                            if not isinstance(msg, dict)
                            else msg.get("content", "")
                        )
                        if isinstance(content, str) and len(content) > limit:
                            msg_id = (
                                getattr(msg, "id", None)
                                if not isinstance(msg, dict)
                                else msg.get("id", None)
                            )
                            if msg_id:
                                truncated_content = (
                                    content[:limit]
                                    + "\n\n... [TRUNCATED FOR SYSTEM MEMORY] ..."
                                )
                                if isinstance(msg, dict):
                                    truncated_msg = msg.copy()
                                    truncated_msg["content"] = truncated_content
                                    updates.append(truncated_msg)
                                else:
                                    from langchain_core.messages import ToolMessage

                                    updates.append(
                                        ToolMessage(
                                            id=msg_id,
                                            content=truncated_content,
                                            tool_call_id=getattr(
                                                msg, "tool_call_id", ""
                                            ),
                                            name=getattr(msg, "name", None),
                                        )
                                    )

            # Case 2: Terminal message is an AIMessage from a direct_tool node with return_direct
            elif (
                node_config
                and node_config.type == "direct_tool"
                and getattr(node_config, "return_direct", False)
            ):
                last_msg = messages[-1]
                content = (
                    getattr(last_msg, "content", "")
                    if not isinstance(last_msg, dict)
                    else last_msg.get("content", "")
                )
                if isinstance(content, str) and len(content) > limit:
                    msg_id = (
                        getattr(last_msg, "id", None)
                        if not isinstance(last_msg, dict)
                        else last_msg.get("id", None)
                    )
                    if msg_id:
                        truncated_content = (
                            content[:limit]
                            + "\n\n... [TRUNCATED FOR SYSTEM MEMORY] ..."
                        )
                        if isinstance(last_msg, dict):
                            truncated_msg = last_msg.copy()
                            truncated_msg["content"] = truncated_content
                            updates.append(truncated_msg)
                        else:
                            from langchain_core.messages import AIMessage

                            updates.append(
                                AIMessage(
                                    id=msg_id,
                                    content=truncated_content,
                                    tool_calls=getattr(last_msg, "tool_calls", []),
                                    additional_kwargs=getattr(
                                        last_msg, "additional_kwargs", {}
                                    ),
                                )
                            )

            if updates:
                logger.info(
                    f"Oversized return_direct messages detected at the end of execution. Truncating {len(updates)} messages in checkpointer."
                )
                await app.aupdate_state(config, {"messages": updates})
        except Exception as e:
            logger.error(f"Failed to truncate large messages in checkpointer: {e}")

    @classmethod
    def _redact_messages(cls, serialized: list, config_obj: Any) -> list:
        """Redacts messages generated by nodes configured with mute_chat_bubble=True."""
        if not config_obj or not serialized:
            return serialized
        muted_nodes = {
            n.id for n in config_obj.nodes if getattr(n, "mute_chat_bubble", False)
        }
        if not muted_nodes:
            return serialized
        for msg in serialized:
            node_id = msg.get("metadata", {}).get("active_agent")
            if node_id in muted_nodes:
                msg["content"] = "[Internal Execution Content]"
                if "tool_calls" in msg.get("metadata", {}):
                    msg["metadata"]["tool_calls"] = []
        return serialized

    @classmethod
    async def run(
        cls,
        workflow_id: str,
        input_text: str,
        thread_id: str = "default_thread",
        context: Dict[str, Any] = None,
        target_node_id: str = None,
        db: Any = None,
        trim_context: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Runs a workflow synchronously (waits for final response)."""
        app = await workflow_registry.get_executable_graph(workflow_id, db=db)
        if not app:
            raise ValueError(f"Workflow '{workflow_id}' not found or not compiled.")

        try:
            run_id = str(uuid.uuid4())[:8]
            # Fetched before the observer so its node count can seed progress
            # reporting — the denominator comes from the graph, not from a
            # hand-maintained constant.
            config_obj = workflow_registry.get_config(workflow_id)
            callbacks = [get_observer(
                workflow_id=workflow_id,
                run_id=run_id,
                context=context,
                nodes=config_obj.nodes if config_obj else [],
            )]
            lf_handler = get_langfuse_handler(trace_id=thread_id)
            if lf_handler:
                callbacks.append(lf_handler)

            workflow_name = (
                getattr(config_obj, "name", workflow_id) if config_obj else workflow_id
            )

            config: Dict[str, Any] = {
                "configurable": {"thread_id": thread_id},
                "callbacks": callbacks,
                "run_name": workflow_name,
                "metadata": {
                    # AgentMeshObserver reads metadata["thread_id"] to key the
                    # active-runs registry and its progress events. Without it
                    # every run reported as "anonymous", so concurrent runs
                    # collapsed into one bucket and no consumer could tell which
                    # run an event belonged to.
                    "thread_id": thread_id,
                    "langfuse_session_id": thread_id,
                    "langfuse_user_id": context.get("user_id", "anonymous")
                    if context
                    else "anonymous",
                    "langfuse_tags": [workflow_id, "agentmesh"],
                },
            }

            logger.debug(
                f"Running app {workflow_name} ({workflow_id}) on thread {thread_id}. Has checkpointer: {bool(getattr(app, 'checkpointer', None))}"
            )
            logger.debug(f"Config: {config}")

            # ✅ NEW (Mar 30, 2026): Validate context against context_schema if defined
            if (
                config_obj
                and hasattr(config_obj, "context_schema")
                and config_obj.context_schema
            ):
                cls._validate_context(
                    context or {}, config_obj.context_schema, workflow_id
                )

            # 🕵️ SMART STATE INITIALIZATION (Apr 21, 2026):
            # Fetch existing state to preserve context and active_agent
            state_snapshot = await app.aget_state(config)
            existing_state = state_snapshot.values
            initial_msg_count = len(existing_state.get("messages", []))

            # Prepare initialization state
            # Inject workflow variables from Postgres into the context
            wf_vars = await cls._load_workflow_variables(workflow_id)

            run_state = {
                "active_agent": target_node_id
                if target_node_id
                else existing_state.get("active_agent"),
                "messages": [HumanMessage(content=input_text)],
                "context": {
                    **(existing_state.get("context") or {}),
                    **(context or {}),
                    "thread_id": thread_id,
                    "workflow_id": workflow_id,
                    "workflow_name": workflow_name,
                    "workflow_variables": wf_vars,
                },
            }

            # Optional: Preserving VFS if not provided in new context
            if "vfs" not in run_state:
                run_state["vfs"] = existing_state.get("vfs") or {
                    "files": {},
                    "dirs": [],
                }

            # ── Concurrency guard ─────────────────────────────────────────────────────
            # Set BEFORE astream. Cleared only when workflow is truly done.
            # If paused at HITL, Redis stays set so delete/save/compile remain blocked.
            # resume() or reject() will clear it when the human resolves the gate.
            await redis_manager.set_active(workflow_id)
            try:
                # 🕵️ SMART AUTO-RESUME (Apr 1, 2026):
                # If we hit an interrupt immediately (usually a pre-gate or interrupt_before on the entry node),
                # and no node has processed the input yet, we auto-resume once.
                result = None
                for attempt in range(2):
                    async for chunk in app.astream(
                        run_state, config=config, stream_mode="values"
                    ):
                        result = chunk

                    # Audit for immediate interrupt
                    snapshot = await app.aget_state(config)
                    logger.info(
                        f"[HITL DEBUG] attempt={attempt} | snapshot.next={snapshot.next} | "
                        f"msg_count={len(result.get('messages', []) if result else [])} | "
                        f"initial_msg_count={initial_msg_count}"
                    )
                    if snapshot.next and attempt == 0:
                        # 🛑 STOP: Never auto-resume on a Form Node
                        if cls._is_form_interrupt(snapshot.next, config_obj):
                            logger.info(
                                f"Form Node interrupt detected on {snapshot.next}. Staying paused for user input."
                            )
                            break

                        # Helper to check if a pre-gate belongs to a starting STT/camera node
                        def is_starting_stt_camera_pre_gate(nid):
                            if not nid.startswith("pre_gate_"):
                                return False
                            target_id = nid.replace("pre_gate_", "")
                            node = next((n for n in config_obj.nodes if n.id == target_id), None)
                            return node and node.type in ("camera", "stt_trigger") and cls._is_starting_node(node.id, config_obj)

                        # 🛑 STOP: Never auto-resume on Wire Breakpoints
                        if cls._is_breakpoint_interrupt(snapshot.next, config_obj):
                            logger.info(
                                f"🛑 Wire Breakpoint interrupt detected on {snapshot.next}. Staying paused."
                            )
                            break

                        # 🛑 STOP: Never auto-resume on a HITL pre-gate (e.g. Event Emitter with interrupt_before)
                        # pre_gate_ nodes are injected by the compiler for interrupt_before
                        # nodes. Auto-resuming past them defeats human-in-the-loop verification.
                        # EXCEPT when it's a starting STT or Camera node where input was already gathered.
                        if any(n.startswith("pre_gate_") and not is_starting_stt_camera_pre_gate(n) for n in snapshot.next):
                            logger.info(
                                f"HITL pre-gate interrupt detected on {snapshot.next}. Staying paused for human verification."
                            )
                            break


                        # 🛑 STOP: Never auto-resume on a HITL post-gate (e.g. PII verification)
                        # post_gate_ nodes are injected by the compiler for interrupt_after
                        # nodes. Auto-resuming past them defeats human-in-the-loop verification.
                        if any(n.startswith("post_gate_") for n in snapshot.next):
                            logger.info(
                                f"HITL post-gate interrupt detected on {snapshot.next}. Staying paused for human verification."
                            )
                            break

                        current_msgs = result.get("messages", []) if result else []
                        if len(current_msgs) == initial_msg_count + 1:
                            logger.info(
                                f"Initial turn interrupt detected on {snapshot.next}. Auto-resuming..."
                            )
                            run_state = None  # LangGraph resume signal
                            continue
                    break
            except Exception:
                # On error, always clear the guard (workflow is not running anymore)
                await redis_manager.clear_active(workflow_id)
                raise

            logger.debug(
                f"Stream finished. Current state: {list(result.keys()) if result else 'None'}"
            )

            # 📌 Persist MemorySaver to Postgres on interrupt so HITL survives restarts
            if snapshot and snapshot.next:
                await cls._flush_memory_to_postgres(app, config, thread_id)

            # 🛠️ POST-RUN AUDIT: Get the final state snapshot directly for maximum reliability
            state_snapshot = await cls._propagate_subgraph_pause(
                app, config, workflow_id, db=db
            )
            final_state = state_snapshot.values
            is_paused = len(state_snapshot.next) > 0 or any(t.interrupts for t in state_snapshot.tasks)

            # Extract output: Find the latest non-empty message content
            output = ""
            messages = final_state.get("messages", [])
            for msg in reversed(messages):
                content = msg.content if hasattr(msg, "content") else str(msg)
                if content and isinstance(content, str) and content.strip():
                    output = content
                    break

            # Clear Redis only if workflow is truly done — not paused at a HITL gate.
            # If is_paused=True, the guard must stay active until the human resolves it.
            if not is_paused:
                # The final response is captured below; terminal checkpoint state is
                # removed only after that response has been serialized.
                await cls._flush_memory_to_postgres(app, config, thread_id)
                await redis_manager.clear_active(workflow_id)
                await redis_manager.clear_thread_state_persisted(thread_id)
                workflow_registry.clear_thread_checkpoint(thread_id)
            else:
                # 📌 Persist again in case _propagate_subgraph_pause found new pause info
                await cls._flush_memory_to_postgres(app, config, thread_id)

            final_context = final_state.get("context", {})
            logger.info(f"Final context keys: {list(final_context.keys())}")
            if "latest_audio" in final_context:
                logger.info(
                    f"Found audio metadata in final context: {final_context['latest_audio']}"
                )

            # Resolve pause node metadata for frontend
            is_paused, is_bp, next_node_id, pause_type, pause_config, display_output = (
                cls._snapshot_pause_info(state_snapshot, config_obj, messages=messages)
            )

            if display_output:
                output = display_output

            # Serialize before truncating the DB checkpointer
            serialized_messages = cls._redact_messages(
                serialize_messages(messages), config_obj
            )

            # Truncate large messages in checkpointer memory
            await cls._truncate_large_messages_in_checkpoint(app, config, workflow_id)

            if not is_paused:
                await cls._finalize_completed_thread(
                    thread_id=thread_id,
                    workflow_id=workflow_id,
                    status="success",
                    output=output,
                    messages=serialized_messages,
                    context=final_context,
                    active_agent=final_state.get("active_agent"),
                )

            # 📡 Flush Langfuse so the exported trace includes ALL nodes (classifier,
            # agents, aggregator) with their token usage before the run is read back.
            flush_langfuse()

            return {
                "status": "success" if not is_paused else "paused",
                "output": output,
                "messages": serialized_messages,
                "history_count": len(messages),
                "active_agent": final_state.get("active_agent"),
                "is_paused": is_paused,
                "is_breakpoint": is_bp,
                "next_node": next_node_id,
                "pause_type": "breakpoint" if is_bp else pause_type,
                "pause_config": pause_config,
                "thread_id": thread_id,
                "context": (
                    cls._trim_response_context(final_context)
                    if cls._resolve_trim(trim_context, final_context)
                    else final_context
                ),
            }

        except Exception as e:
            # Safety net: clear the guard on any unhandled exception from this method.
            # Idempotent — if already cleared by the inner handler or is_paused check, this is a no-op.
            await redis_manager.clear_active(workflow_id)
            import traceback

            traceback.print_exc()
            raise e

    @classmethod
    async def stream(
        cls,
        workflow_id: str,
        input_text: str,
        thread_id: str = "default_thread",
        context: Dict[str, Any] = None,
        target_node_id: str = None,
        db: Any = None,
        trim_context: Optional[bool] = None,
    ) -> AsyncGenerator[str, None]:
        """Streams workflow execution events via SSE format."""
        app = await workflow_registry.get_executable_graph(workflow_id, db=db)
        if not app:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Workflow not found'})}\n\n"
            return

        # The compiled graph already has a checkpointer from the WorkflowRegistry.
        # Do NOT create a separate one here — it causes state conflicts.
        run_id = str(uuid.uuid4())[:8]
        _cfg = workflow_registry.get_config(workflow_id)
        callbacks = [get_observer(
            workflow_id=workflow_id,
            run_id=run_id,
            context=context,
            nodes=_cfg.nodes if _cfg else [],
        )]
        lf_handler = get_langfuse_handler(trace_id=thread_id)
        if lf_handler:
            callbacks.append(lf_handler)

        config_obj = workflow_registry.get_config(workflow_id)
        workflow_name = (
            getattr(config_obj, "name", workflow_id) if config_obj else workflow_id
        )

        config: Dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "callbacks": callbacks,
            "run_name": workflow_name,
            "metadata": {
                "langfuse_session_id": thread_id,
                "langfuse_user_id": context.get("user_id", "anonymous")
                if context
                else "anonymous",
                "langfuse_tags": [workflow_id, "agentmesh"],
            },
        }

        # ✅ NEW (Mar 30, 2026): Validate context against context_schema if defined
        if (
            config_obj
            and hasattr(config_obj, "context_schema")
            and config_obj.context_schema
        ):
            cls._validate_context(context or {}, config_obj.context_schema, workflow_id)

        # 🕵️ SMART STATE INITIALIZATION (Apr 21, 2026):
        state_snapshot = await app.aget_state(config)
        existing_state = state_snapshot.values
        initial_msg_count = len(existing_state.get("messages", []))

        # Inject workflow variables from Postgres into the context
        wf_vars = await cls._load_workflow_variables(workflow_id)

        run_state = {
            "active_agent": target_node_id
            if target_node_id
            else existing_state.get("active_agent"),
            "messages": [HumanMessage(content=input_text)],
            "context": {
                **(existing_state.get("context") or {}),
                **(context or {}),
                "workflow_id": workflow_id,
                "workflow_name": workflow_name,
                "thread_id": thread_id,
                "workflow_variables": wf_vars,
            },
        }

        if "vfs" not in run_state:
            run_state["vfs"] = existing_state.get("vfs") or {"files": {}, "dirs": []}

        try:
            # Mark workflow as active for the duration of the stream (mirrors run()).
            await redis_manager.set_active(workflow_id)

            # SSE Heartbeat task could be added here if we had a more complex setup,
            # but for now we'll just yield a comment periodically if possible.
            # However, astream_events is an async iterator. We can wrap it.

            # Fetch recursion limit from config
            config_obj = workflow_registry.get_config(workflow_id)
            recursion_limit = (
                getattr(config_obj, "recursion_limit", 100) if config_obj else 100
            )
            config["recursion_limit"] = recursion_limit

            run_state_arg = run_state
            for attempt in range(2):
                async for event in app.astream_events(
                    run_state_arg, config=config, version="v2"
                ):
                    formatted_event = cls._format_stream_event(event, config_obj)
                    if formatted_event:
                        safe_event = sanitize_for_logging(redact_sensitive_data(formatted_event))
                        yield f"data: {json.dumps(safe_event, default=str)}\n\n"

                # Check for immediate interrupt
                snapshot = await app.aget_state(config)
                if snapshot.next and attempt == 0:
                    # Skip auto-resume for Form Nodes
                    if cls._is_form_interrupt(snapshot.next, config_obj):
                        logger.info(
                            f"Form Node interrupt detected on streaming {snapshot.next}. Staying paused."
                        )
                        break

                    # Helper to check if a pre-gate belongs to a starting STT/camera node
                    def is_starting_stt_camera_pre_gate(nid):
                        if not nid.startswith("pre_gate_"):
                            return False
                        target_id = nid.replace("pre_gate_", "")
                        node = next((n for n in config_obj.nodes if n.id == target_id), None)
                        return node and node.type in ("camera", "stt_trigger") and cls._is_starting_node(node.id, config_obj)

                    # 🛑 STOP: Never auto-resume on Wire Breakpoints
                    if cls._is_breakpoint_interrupt(snapshot.next, config_obj):
                        logger.info(
                            f"🛑 Wire Breakpoint interrupt detected on streaming {snapshot.next}. Staying paused."
                        )
                        break

                    # 🛑 STOP: Never auto-resume on a HITL pre-gate (interrupt_before)
                    # EXCEPT when it's a starting STT or Camera node where input was already gathered.
                    if any(n.startswith("pre_gate_") and not is_starting_stt_camera_pre_gate(n) for n in snapshot.next):
                        logger.info(
                            f"HITL pre-gate interrupt detected on streaming {snapshot.next}. Staying paused."
                        )
                        break


                    # 🛑 STOP: Never auto-resume on a HITL post-gate (interrupt_after)
                    if any(n.startswith("post_gate_") for n in snapshot.next):
                        logger.info(
                            f"HITL post-gate interrupt detected on streaming {snapshot.next}. Staying paused."
                        )
                        break

                    if (
                        len(snapshot.values.get("messages", []))
                        == initial_msg_count + 1
                    ):
                        logger.info(
                            f"Initial turn interrupt detected on streaming {snapshot.next}. Auto-resuming..."
                        )
                        run_state_arg = None
                        continue
                break

            # 📌 Persist MemorySaver to Postgres on interrupt so HITL survives restarts
            if snapshot and snapshot.next:
                await cls._flush_memory_to_postgres(app, config, thread_id)

            # Truncate large messages in checkpointer memory before returning final state snapshot
            await cls._truncate_large_messages_in_checkpoint(app, config, workflow_id)

            # Final Snapshot
            snapshot = (
                await app.aget_state(config)
                if hasattr(app, "aget_state")
                else app.get_state(config)
            )

            # Clear the Redis guard only when the run truly finished (not paused at a HITL gate).
            if not (snapshot.next or any(t.interrupts for t in snapshot.tasks)):
                # Capture the final event before terminal checkpoint cleanup.
                await cls._flush_memory_to_postgres(app, config, thread_id)
                await redis_manager.clear_active(workflow_id)
                await redis_manager.clear_thread_state_persisted(thread_id)

            # 📡 Flush Langfuse before emitting the final snapshot so the exported
            # trace contains every node (classifier, agents, aggregator) + tokens.
            flush_langfuse()

            final_event = cls._get_final_snapshot_data(
                snapshot, thread_id, config_obj, trim_context
            )
            if not (snapshot.next or any(t.interrupts for t in snapshot.tasks)):
                await cls._finalize_completed_thread(
                    thread_id=thread_id,
                    workflow_id=workflow_id,
                    status="success",
                    output=final_event.get("output", ""),
                    messages=final_event.get("messages", []),
                    context=final_event.get("context", {}),
                    active_agent=final_event.get("active_agent"),
                )
            safe_final_event = sanitize_for_logging(redact_sensitive_data(final_event))
            yield f"data: {json.dumps(safe_final_event)}\n\n"

        except Exception as e:
            # Safety net: clear the guard on any unhandled exception from this method.
            await redis_manager.clear_active(workflow_id)
            safe_error = sanitize_for_logging(str(e))
            yield f"data: {json.dumps({'type': 'error', 'message': safe_error})}\n\n"

    @classmethod
    async def resume(
        cls,
        workflow_id: str,
        thread_id: str,
        action: str,
        payload: Any = None,
        db: Any = None,
        trim_context: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Resumes a paused workflow run with human input.

        Actions:
        - 'approve': Continue execution as-is
        - 'rework': Restart from last message with feedback injected
        - 'reject': Stop execution and mark workflow as rejected
        """
        app = await workflow_registry.get_executable_graph(workflow_id, db=db)
        if not app:
            raise ValueError(f"Workflow '{workflow_id}' not found or not compiled.")

        # Validate action
        VALID_ACTIONS = ["approve", "rework", "reject", "submit"]
        if action not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid action '{action}'. Must be one of: {', '.join(VALID_ACTIONS)}"
            )

        # ✅ Check if action is allowed for the paused node
        config_obj = workflow_registry.get_config(workflow_id)
        run_config = {"configurable": {"thread_id": thread_id}}
        state_snapshot = await app.aget_state(run_config)

        # 🏭 MULTI-WORKER RESUME: If MemorySaver is empty, try loading from
        # Postgres. Redis flag is only a hint — always fallback to PG.
        if not state_snapshot.values.get("messages", []):
            loaded = await cls._load_state_from_postgres(app, run_config, thread_id)
            if loaded:
                state_snapshot = await app.aget_state(run_config)

        paused_node = None
        if state_snapshot.next:
            paused_node_id = state_snapshot.next[0]
            real_node_id = paused_node_id
            if real_node_id.startswith("pre_gate_"):
                real_node_id = real_node_id[len("pre_gate_"):]
            elif real_node_id.startswith("post_gate_"):
                real_node_id = real_node_id[len("post_gate_"):]

            if config_obj:
                paused_node = next(
                    (n for n in config_obj.nodes if n.id == real_node_id or n.id == paused_node_id), None
                )
                allowed_actions = list(getattr(
                    paused_node, "hitl_actions", ["approve", "rework", "reject"]
                )) if paused_node else ["approve", "reject"]

                # Form nodes, multi_file_ingestion, human_interaction_node, and interactive trigger nodes support 'submit'
                is_interactive_pause = paused_node_id in ("human_gate", "human_interaction_node") or (
                    paused_node and (
                        paused_node.type in ("form_node", "multi_file_ingestion", "camera", "stt_trigger", "ocr_node", "button_trigger")
                        or getattr(paused_node, "is_interactive", False)
                    )
                )
                if is_interactive_pause and "submit" not in allowed_actions:
                    allowed_actions.append("submit")

                # Tool nodes and global human_gate pauses do not support 'rework'
                if paused_node_id in ("human_gate", "human_interaction_node") or (paused_node and paused_node.type == "tool_node"):
                    allowed_actions = [a for a in allowed_actions if a != "rework"]

                if allowed_actions and action not in allowed_actions:
                    raise ValueError(
                        f"Action '{action}' is not allowed for paused step '{paused_node_id}'. Allowed actions: {', '.join(allowed_actions)}"
                    )

        # ✅ Log the HITL decision for audit trail
        logger.info(
            f"AUDIT [HITL]: Action='{action}', Workflow='{workflow_id}', Thread='{thread_id}'"
        )
        feedback = None
        if payload:
            feedback = payload.get("feedback")
            if feedback:
                logger.info(f"AUDIT [HITL]: User Feedback: {feedback[:100]}...")

        # 🕵️ Langfuse Audit (Apr 3, 2026)
        # We log the decision directly to ensure 'Reject' outcomes are captured
        # even if they don't trigger a graph run. trace id == thread_id matches
        # get_langfuse_handler(trace_id=thread_id) below so the decision + score
        # attach to the SAME trace as the workflow run (node & tool HITL alike).
        hitl_level = "tool" if (paused_node and getattr(paused_node, "type", None) == "tool_node") else "node"
        log_human_decision(
            thread_id=thread_id,
            action=action,
            feedback=feedback,
            user_id=payload.get("user_id", "anonymous")
            if isinstance(payload, dict)
            else "anonymous",
            hitl_level=hitl_level,
        )

        run_id = str(uuid.uuid4())[:8]
        # Resume takes no `context` argument: the paused run's context lives in
        # the checkpoint. Progress on resume therefore arrives with the
        # human-review work, where reading that state is needed anyway. Passing
        # the node total keeps the message shape identical in the meantime.
        _cfg = workflow_registry.get_config(workflow_id)
        callbacks = [get_observer(
            workflow_id=workflow_id,
            run_id=run_id,
            nodes=_cfg.nodes if _cfg else [],
        )]
        lf_handler = get_langfuse_handler(trace_id=thread_id)
        if lf_handler:
            callbacks.append(lf_handler)

        config_obj = workflow_registry.get_config(workflow_id)
        workflow_name = (
            getattr(config_obj, "name", workflow_id) if config_obj else workflow_id
        )

        config: Dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "callbacks": callbacks,
            "run_name": workflow_name,
            "metadata": {
                "langfuse_session_id": thread_id,
                "langfuse_user_id": payload.get("user_id", "anonymous")
                if isinstance(payload, dict)
                else "anonymous",
                "langfuse_tags": [workflow_id, "agentmesh", "resume"],
                "is_resume": "true",
            },
        }

        # 🔴 REJECT: Stop execution immediately — clears the HITL guard
        if action == "reject":
            await app.aupdate_state(
                config,
                {
                    "context": {
                        **(payload if isinstance(payload, dict) else {}),
                        "hitl_action": "reject",
                        "hitl_timestamp": datetime.now().isoformat(),
                        "hitl_feedback": payload.get("feedback")
                        if isinstance(payload, dict)
                        else None,
                    }
                },
            )
            await redis_manager.clear_active(workflow_id)
            # 📌 Persist the rejected run too so its state survives restarts and can
            # be reviewed from the history sidebar.
            await cls._flush_memory_to_postgres(app, config, thread_id)
            await redis_manager.clear_thread_state_persisted(thread_id)
            workflow_registry.clear_thread_checkpoint(thread_id)
            flush_langfuse()
            await cls._finalize_completed_thread(
                thread_id=thread_id,
                workflow_id=workflow_id,
                status="rejected",
                output="Workflow execution rejected by human reviewer.",
                messages=cls._redact_messages(
                    serialize_messages(state_snapshot.values.get("messages", [])),
                    config_obj,
                ),
                context={
                    **(state_snapshot.values.get("context", {}) or {}),
                    **(payload if isinstance(payload, dict) else {}),
                    "hitl_action": "reject",
                },
                active_agent=state_snapshot.values.get("active_agent"),
            )
            return {
                "status": "rejected",
                "output": "Workflow execution rejected by human reviewer.",
                "messages": [],
                "is_paused": False,
                "next_node": None,
                "thread_id": thread_id,
            }

        # Record initial message count to extract ONLY new messages
        initial_msg_count = len(state_snapshot.values.get("messages", []))

        # ── Concurrency guard: mark workflow active for this resume execution ──
        # Cleared explicitly before each return path below.
        # TTL of 3600s acts as safety net if an unexpected exception propagates.
        await redis_manager.set_active(workflow_id)

        # ✅ REWORK ACTION: Update state with feedback, then resume from checkpoint
        if action == "rework":
            logger.debug("REWORK detected - updating state with feedback and resuming")
            current_messages = state_snapshot.values.get("messages", [])

            # Get the user's original prompt (last HumanMessage before the pause)
            from langchain_core.messages import HumanMessage

            original_user_msg = ""
            for msg in reversed(current_messages):
                if isinstance(msg, HumanMessage):
                    original_user_msg = msg.content
                    break

            # Prepare rework context by merging FULL payload with HITL metadata
            # This ensures custom fields like 'decision' are preserved for Graph Routers
            rework_context = {
                **(payload if isinstance(payload, dict) else {}),
                "hitl_action": "rework",
                "hitl_timestamp": datetime.now().isoformat(),
                "original_user_message": original_user_msg,
                "include_original_output": payload.get("include_original_output", True)
                if isinstance(payload, dict)
                else True,
                "inject_feedback": payload.get("inject_feedback", True)
                if isinstance(payload, dict)
                else True,
                "hitl_feedback": feedback,
            }

            # Update state IN-PLACE at the checkpoint, then resume.
            # This triggers the entry router's rework jump back to the active agent.
            await app.aupdate_state(
                config,
                {
                    "context": rework_context,
                    "active_agent": state_snapshot.values.get("active_agent"),
                },
            )

            # Fetch recursion limit from config
            config_obj = workflow_registry.get_config(workflow_id)
            recursion_limit = (
                getattr(config_obj, "recursion_limit", 100) if config_obj else 100
            )
            config["recursion_limit"] = recursion_limit

            result = None
            try:
                async for chunk in app.astream(
                    None, config=config, stream_mode="values"
                ):
                    result = chunk
            except Exception:
                await redis_manager.clear_active(workflow_id)
                raise

            # 🕵️ RE-PAUSE AUDIT (Apr 1, 2026):
            # If the graph pauses again (usually at the same gate it just resumed from),
            # we need to ensure the state snapshot reflects the 'post-rework' state.
            state_snapshot = await cls._propagate_subgraph_pause(
                app, config, workflow_id, db=db
            )
            is_paused = len(state_snapshot.next) > 0 or any(t.interrupts for t in state_snapshot.tasks)
            output = ""

            total_messages = result.get("messages", []) if result else []
            new_messages = total_messages[initial_msg_count:]

            for msg in reversed(new_messages):
                content = getattr(msg, "content", str(msg))
                if content and isinstance(content, str) and content.strip():
                    output = content
                    break

            # Serialize before truncating the DB checkpointer
            serialized_new_messages = cls._redact_messages(
                serialize_messages(new_messages), config_obj
            )

            # Truncate large messages in checkpointer memory
            await cls._truncate_large_messages_in_checkpoint(app, config, workflow_id)

            # Only clear Redis if the workflow is truly done after rework.
            # If it re-paused at the same HITL gate, keep the guard active.
            if not is_paused:
                # Capture response data before terminal checkpoint cleanup.
                await cls._flush_memory_to_postgres(app, config, thread_id)
                await redis_manager.force_clear_active(workflow_id)
                await redis_manager.clear_thread_state_persisted(thread_id)
                workflow_registry.clear_thread_checkpoint(thread_id)

            # 📌 Persist MemorySaver to Postgres on re-pause so HITL survives restarts
            if is_paused:
                await cls._flush_memory_to_postgres(app, config, thread_id)

            # Resolve pause node metadata for frontend
            is_paused, is_bp, next_node_id, pause_type, pause_config, display_output = (
                cls._snapshot_pause_info(state_snapshot, config_obj, messages=total_messages)
            )

            if display_output:
                output = display_output

            flush_langfuse()

            if not is_paused:
                await cls._finalize_completed_thread(
                    thread_id=thread_id,
                    workflow_id=workflow_id,
                    status="success",
                    output=output,
                    messages=cls._redact_messages(
                        serialize_messages(total_messages), config_obj
                    ),
                    context=result.get("context", {}) if result else {},
                    active_agent=result.get("active_agent") if result else None,
                )

            return {
                "status": "success" if not is_paused else "paused",
                "output": output,
                "messages": serialized_new_messages,
                "history_count": len(total_messages),
                "active_agent": result.get("active_agent") if result else None,
                "is_paused": is_paused,
                "is_breakpoint": is_bp,
                "next_node": next_node_id,
                "pause_type": pause_type,
                "pause_config": pause_config,
                "thread_id": thread_id,
                "rework_applied": True,
                "context": (
                    cls._trim_response_context(result.get("context", {}) if result else {})
                    if cls._resolve_trim(trim_context, result.get("context", {}) if result else {})
                    else (result.get("context", {}) if result else {})
                ),
            }


        # APPROVE/SUBMIT: Continue execution from the pause point as-is
        logger.debug(f"{action.upper()} detected - continuing from pause point")
        logger.debug(
            f"Current state has {len(state_snapshot.values.get('messages', []))} messages before resume"
        )
        # Check if we are resuming from an interrupt() call
        has_interrupts = any(t.interrupts for t in state_snapshot.tasks)
        resume_command = None

        selected_option = payload.get("selected_option") if isinstance(payload, dict) else None
        followup_answers = payload.get("followup_answers") if isinstance(payload, dict) else None

        if has_interrupts:
            # We are resuming from an interrupt() call. We must always pass a resume value.
            resume_val = "approve"
            if isinstance(payload, dict):
                if payload.get("skipped"):
                    resume_val = "Skipped"
                elif "followup_answers" in payload:
                    resume_val = payload["followup_answers"]
                elif "selected_option" in payload:
                    resume_val = payload["selected_option"]
                elif "feedback" in payload:
                    resume_val = payload["feedback"]
            elif payload is not None:
                resume_val = payload
                
            from langgraph.types import Command
            resume_command = Command(resume=resume_val)

        # Extract messages from payload if present
        messages_to_inject = []
        if isinstance(payload, dict) and "messages" in payload:
            messages_to_inject = payload.get("messages") or []
            
        # Clean custom keys from context to keep it clean, but preserve other payload fields
        context_payload = {**(payload if isinstance(payload, dict) else {})}
        context_payload.pop("messages", None)
        
        # Explicitly clear followup keys so they don't persist in next turns
        context_payload["followup_active"] = False
        context_payload["followup_options"] = None

        # Build update dictionary
        update_data = {
            "context": {
                **context_payload,
                "hitl_action": action,
                "hitl_timestamp": datetime.now().isoformat(),
            },
            "user_input": payload if isinstance(payload, dict) else {},
        }
        
        # Inject messages if provided
        if messages_to_inject:
            from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
            parsed_messages = []
            for msg in messages_to_inject:
                if isinstance(msg, dict):
                    role = msg.get("role") or msg.get("type")
                    content = msg.get("content", "")
                    if role in ("user", "HumanMessage"):
                        parsed_messages.append(HumanMessage(content=content))
                    elif role in ("ai", "assistant", "AIMessage"):
                        parsed_messages.append(AIMessage(content=content))
                    elif role in ("tool", "ToolMessage"):
                        parsed_messages.append(ToolMessage(content=content, tool_call_id=msg.get("tool_call_id", ""), name=msg.get("name", "")))
                else:
                    parsed_messages.append(msg)
            update_data["messages"] = parsed_messages

        # Merge payload into context to preserve any custom routing keys
        await app.aupdate_state(config, update_data)

        # Fetch recursion limit from config
        config_obj = workflow_registry.get_config(workflow_id)
        recursion_limit = (
            getattr(config_obj, "recursion_limit", 500) if config_obj else 500
        )
        config["recursion_limit"] = recursion_limit

        # 2. Continue from the pause point (the gate or interrupt)
        result = None
        try:
            async for chunk in app.astream(resume_command, config=config, stream_mode="values"):
                result = chunk
        except Exception:
            await redis_manager.clear_active(workflow_id)
            raise

        # 🛠️ POST-RESUME AUDIT: Get final state directly
        state_snapshot = await cls._propagate_subgraph_pause(
            app, config, workflow_id, db=db
        )
        final_state = state_snapshot.values
        is_paused = len(state_snapshot.next) > 0 or any(t.interrupts for t in state_snapshot.tasks)
        final_context = final_state.get("context", {})

        output = ""

        # Extract output from the NEW messages
        total_messages = final_state.get("messages", [])
        new_messages = total_messages[initial_msg_count:]

        logger.debug(
            f"After resume - total messages: {len(total_messages)}, new messages: {len(new_messages)}"
        )
        for i, msg in enumerate(new_messages):
            msg_type = (
                type(msg).__name__ if hasattr(msg, "__class__") else str(type(msg))
            )
            content = getattr(msg, "content", str(msg))
            content_len = (
                len(content) if isinstance(content, str) else len(str(content))
            )
            logger.debug(f"New message {i}: type={msg_type}, content_len={content_len}")

        for msg in reversed(new_messages):
            content = getattr(msg, "content", str(msg))
            if content and isinstance(content, str) and content.strip():
                output = content
                break

        # Serialize before truncating the DB checkpointer
        serialized_new_messages = cls._redact_messages(
            serialize_messages(new_messages), config_obj
        )

        # Truncate large messages in checkpointer memory
        await cls._truncate_large_messages_in_checkpoint(app, config, workflow_id)

        # Only clear Redis if the workflow is truly done — not re-paused at another HITL gate.
        if not is_paused:
            # Capture response data before terminal checkpoint cleanup.
            await cls._flush_memory_to_postgres(app, config, thread_id)
            await redis_manager.force_clear_active(workflow_id)
            await redis_manager.clear_thread_state_persisted(thread_id)
            workflow_registry.clear_thread_checkpoint(thread_id)

        # 📌 Persist MemorySaver to Postgres on re-pause so HITL survives restarts
        if is_paused:
            await cls._flush_memory_to_postgres(app, config, thread_id)

        # Resolve pause node metadata for frontend
        is_paused, is_bp, next_node_id, pause_type, pause_config, display_output = (
            cls._snapshot_pause_info(state_snapshot, config_obj, messages=total_messages)
        )

        if display_output:
            output = display_output

        flush_langfuse()

        if not is_paused:
            await cls._finalize_completed_thread(
                thread_id=thread_id,
                workflow_id=workflow_id,
                status="success",
                output=output,
                messages=cls._redact_messages(
                    serialize_messages(total_messages), config_obj
                ),
                context=final_context,
                active_agent=final_state.get("active_agent"),
            )

        return {
            "status": "success" if not is_paused else "paused",
            "output": output,
            "messages": serialized_new_messages,
            "history_count": len(total_messages),
            "active_agent": final_state.get("active_agent"),
            "is_paused": is_paused,
            "is_breakpoint": is_bp,
            "next_node": next_node_id,
            "pause_type": pause_type,
            "pause_config": pause_config,
            "thread_id": thread_id,
            "context": (
                cls._trim_response_context(final_context)
                if cls._resolve_trim(trim_context, final_context)
                else final_context
            ),
        }


    @classmethod
    async def get_thread_state(cls, thread_id: str, workflow_id: str = None) -> Dict[str, Any]:
        """
        Retrieves state values for a specific thread asynchronously from the checkpointer.
        """
        from Apps.AgentMesh.Utilities.Memory.checkpointer import get_checkpointer

        config = {"configurable": {"thread_id": thread_id}}
        if workflow_id:
            try:
                app = workflow_registry._compiled_graphs.get(workflow_id)
                if app:
                    snapshot = await app.aget_state(config)
                    if snapshot and snapshot.values:
                        return snapshot.values
            except Exception as e:
                logger.warning(f"Error fetching state snapshot for workflow '{workflow_id}', thread '{thread_id}': {e}")

        try:
            cp = get_checkpointer()
            if hasattr(cp, "aget_tuple"):
                tup = await cp.aget_tuple(config)
                if tup and tup.checkpoint and "channel_values" in tup.checkpoint:
                    return tup.checkpoint["channel_values"]
            elif hasattr(cp, "get"):
                chk = cp.get(config)
                if chk and "channel_values" in chk:
                    return chk["channel_values"]
            return {}
        except Exception as e:
            logger.error(f"Error fetching thread state for thread '{thread_id}': {e}")
            return {}

    @classmethod
    def _validate_context(
        cls, context: Dict[str, Any], schema: Dict[str, Any], workflow_id: str
    ) -> None:
        """
        Validate provided context against workflow's context_schema.

        Supports basic type validation:
        - str, int, float, bool, dict, list

        Args:
            context: Provided context dictionary
            schema: Expected schema {key: type_or_typename}
            workflow_id: For error messaging

        Raises:
            ValueError: If context doesn't match schema
        """
        if not schema:
            return

        for key, expected_type in schema.items():
            if key not in context:
                logger.warning(
                    f"Workflow {workflow_id}: Missing context key '{key}' (expected type: {expected_type})"
                )
                # Allow missing keys for backwards compatibility, but log warning
                continue

            value = context[key]

            # Handle string type names
            if isinstance(expected_type, str):
                type_mapping = {
                    "str": str,
                    "string": str,
                    "int": int,
                    "integer": int,
                    "float": float,
                    "bool": bool,
                    "boolean": bool,
                    "dict": dict,
                    "object": dict,
                    "list": list,
                    "array": list,
                }
                expected_type = type_mapping.get(
                    expected_type.lower(), type(expected_type)
                )

            # Type checking
            if not isinstance(value, expected_type):
                raise ValueError(
                    f"Workflow {workflow_id}: context.{key} must be {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )

            logger.debug(
                f"Workflow {workflow_id}: context.{key} validated (type: {expected_type.__name__})"
            )

    @classmethod
    def _is_starting_node(cls, node_id: str, config_obj: Any) -> bool:
        """Helper to check if a node has no incoming edges (is a starting entry node)."""
        if not config_obj or not node_id:
            return False
        for edge in getattr(config_obj, "edges", []):
            if edge.target == node_id and edge.source not in ("START", "start"):
                return False
        return True

    @classmethod
    def _is_form_interrupt(cls, next_nodes: Any, config_obj: Any) -> bool:
        """Helper to determine if an interrupt is triggered by a Form Node."""
        if not next_nodes or not config_obj:
            return False
 
        for nid in next_nodes:
            # Extract actual node ID from gate naming convention (pre_gate_ID or post_gate_ID)
            target_id = nid
            if nid.startswith("pre_gate_"):
                target_id = nid.replace("pre_gate_", "")
            elif nid.startswith("post_gate_"):
                target_id = nid.replace("post_gate_", "")
 
            # Look up node in config
            node = next((n for n in config_obj.nodes if n.id == target_id), None)
            if node and node.type in ("form_node", "multi_file_ingestion", "ocr_node", "camera", "stt_trigger"):
                # If it is a starting STT/camera node, we do NOT treat it as a form interrupt
                # that blocks auto-resume, because the user already spoke/captured input on the frontend.
                if node.type in ("camera", "stt_trigger") and cls._is_starting_node(node.id, config_obj):
                    continue
                return True
        return False

    @classmethod
    def _format_stream_event(
        cls, event: Dict[str, Any], config_obj: Any = None
    ) -> Optional[Dict[str, Any]]:
        """Helper to format various LangGraph events for SSE streaming."""
        kind = event["event"]
        node_id = event.get("metadata", {}).get("langgraph_node")

        # Resolve agent name and metadata from registry
        agent_name = None
        is_muted = False
        if node_id:
            try:
                from Apps.AgentMesh.Core.WorkflowRegistry import workflow_registry

                node_meta = workflow_registry.get_node_metadata(node_id)
                if node_meta:
                    agent_name = node_meta.get("label")
                    is_muted = node_meta.get("mute_chat_bubble", False)
                elif config_obj:
                    n = next(
                        (node for node in config_obj.nodes if node.id == node_id), None
                    )
                    if n:
                        agent_name = getattr(n, "label", None) or getattr(
                            n, "name", None
                        )
                        is_muted = getattr(n, "mute_chat_bubble", False)
            except Exception:
                pass

        # Security check: Redact/mute token streams and tool logs from muted nodes
        if is_muted:
            if kind == "on_chat_model_stream":
                return None
            elif kind == "on_chain_start":
                return {
                    "type": "node_start",
                    "node": node_id,
                    "agent_name": agent_name or node_id,
                    "timestamp": datetime.now().isoformat(),
                    "muted": True,
                }
            elif kind in ("on_tool_start", "on_tool_end"):
                return None
            return None

        # Token Streaming
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                return {
                    "type": "token",
                    "content": content,
                    "node": node_id,
                    "agent_name": agent_name or node_id,
                }

        # Node Transitions
        elif kind == "on_chain_start" and event.get("metadata", {}).get(
            "langgraph_node"
        ):
            return {
                "type": "node_start",
                "node": node_id,
                "agent_name": agent_name or node_id,
                "timestamp": datetime.now().isoformat(),
            }

        # Tool Execution
        elif kind == "on_tool_start":
            return {
                "type": "tool_call",
                "tool": event["name"],
                "input": cls._serializable(event["data"].get("input")),
            }
        elif kind == "on_tool_end":
            return {
                "type": "tool_result",
                "tool": event["name"],
                "output": cls._serializable(event["data"].get("output")),
            }

        return None

    @classmethod
    def _serializable(cls, value: Any) -> Any:
        """Coerce arbitrary stream payloads (ToolMessages, objects, etc.) into
        JSON-serializable primitives so SSE json.dumps never blows up."""
        if isinstance(value, dict):
            return {str(k): cls._serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._serializable(v) for v in value]
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if hasattr(value, "content"):
            return cls._serializable(getattr(value, "content"))
        return str(value)

    @classmethod
    def _get_final_snapshot_data(
        cls, snapshot: Any, thread_id: str, config_obj: Any = None, trim_context: bool = None
    ) -> Dict[str, Any]:
        """
        Helper to generate the final event data from a state snapshot.
        Mirrors the run-mode response payload (status/output/messages/context/active_agent)
        so the frontend can process stream and run responses identically.
        """
        messages = snapshot.values.get("messages", [])
        raw_context = snapshot.values.get("context", {}) or {}
        context = (
            cls._trim_response_context(raw_context)
            if cls._resolve_trim(trim_context, raw_context)
            else raw_context
        )
        serialized_messages = cls._redact_messages(
            serialize_messages(messages), config_obj
        )
        active_agent = snapshot.values.get("active_agent")

        last_msg = messages[-1] if messages else ""
        last_content = (
            last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        )

        has_interrupt = len(snapshot.next) > 0 or any(t.interrupts for t in snapshot.tasks)
        if has_interrupt:
            is_paused, is_bp, next_node_id, pause_type, pause_config, display_output = (
                cls._snapshot_pause_info(snapshot, config_obj, messages=serialized_messages)
            )
            if display_output:
                last_content = display_output
            return {
                "type": "interrupt",
                "node": next_node_id,
                "next_node": next_node_id,
                "session_id": thread_id,
                "status": "paused",
                "is_paused": True,
                "is_breakpoint": is_bp,
                "pause_type": "breakpoint" if is_bp else pause_type,
                "pause_config": pause_config,
                "output": last_content,
                "messages": serialized_messages,
                "context": context,
                "active_agent": active_agent,
            }

        else:
            content = last_content
            if config_obj and active_agent:
                node_cfg = next(
                    (n for n in config_obj.nodes if n.id == active_agent), None
                )
                if node_cfg and getattr(node_cfg, "mute_chat_bubble", False):
                    content = "[Internal Execution Completed]"

            return {
                "type": "final",
                "content": content,
                "output": content,
                "session_id": thread_id,
                "status": "success",
                "is_paused": False,
                "messages": serialized_messages,
                "context": context,
                "active_agent": active_agent,
            }

    @classmethod
    async def _propagate_subgraph_pause(
        cls, app: Any, config: Dict[str, Any], workflow_id: str, db: Any = None
    ) -> Any:
        """
        Detects if the graph is paused at a Subgraph node, and if so, reaches into the
        child subgraph thread to fetch its actual pause metadata and propagates it
        directly to the parent context.
        """
        state_snapshot = await app.aget_state(config)
        if not state_snapshot.next:
            return state_snapshot

        from Apps.AgentMesh.Core.WorkflowRegistry import workflow_registry

        paused_node_id = state_snapshot.next[0]
        config_obj = workflow_registry.get_config(workflow_id)
        paused_node = (
            next((n for n in config_obj.nodes if n.id == paused_node_id), None)
            if config_obj
            else None
        )

        if paused_node and paused_node.type == "subgraph":
            parent_thread_id = config["configurable"].get("thread_id", "default_thread")
            child_thread_id = f"{parent_thread_id}_sub_{paused_node.id}"
            child_config = {"configurable": {"thread_id": child_thread_id}}

            child_app = await workflow_registry.get_executable_graph(
                paused_node.workflow_id, db=db
            )
            if child_app:
                child_snapshot = await child_app.aget_state(child_config)
                if child_snapshot and child_snapshot.next:
                    child_paused_node_id = child_snapshot.next[0]
                    clean_child_node_id = child_paused_node_id.replace(
                        "pre_gate_", ""
                    ).replace("post_gate_", "")

                    child_wf_config = workflow_registry.get_config(
                        paused_node.workflow_id
                    )
                    child_paused_node = (
                        next(
                            (
                                n
                                for n in child_wf_config.nodes
                                if n.id == clean_child_node_id
                            ),
                            None,
                        )
                        if child_wf_config
                        else None
                    )
                    child_node_type = (
                        getattr(child_paused_node, "type", None)
                        if child_paused_node
                        else None
                    )
                    child_node_config = (
                        child_paused_node.model_dump(exclude_unset=True, exclude_none=True)
                        if child_paused_node
                        and hasattr(child_paused_node, "model_dump")
                        else (child_paused_node.__dict__ if child_paused_node else {})
                    )

                    paused_subgraph_buttons = []
                    if child_wf_config:
                        child_edges = child_wf_config.edges or []
                        outgoing_to_buttons = [
                            e for e in child_edges if e.source == clean_child_node_id
                        ]
                        for edge in outgoing_to_buttons:
                            target_node = next(
                                (
                                    n
                                    for n in child_wf_config.nodes
                                    if n.id == edge.target
                                ),
                                None,
                            )
                            if target_node and target_node.type == "button_trigger":
                                paused_subgraph_buttons.append(
                                    {
                                        "id": target_node.id,
                                        "label": getattr(
                                            target_node, "button_label", None
                                        )
                                        or getattr(target_node, "label", None)
                                        or "Proceed",
                                        "button_category": getattr(
                                            target_node, "button_category", None
                                        )
                                        or "primary",
                                    }
                                )

                    parent_context_update = {
                        "paused_subgraph_node_id": clean_child_node_id,
                        "paused_subgraph_node_type": child_node_type,
                        "paused_subgraph_node_config": child_node_config,
                        "paused_subgraph_buttons": paused_subgraph_buttons
                        if paused_subgraph_buttons
                        else None,
                        "paused_subgraph_workflow_id": paused_node.workflow_id,
                    }

                    parent_messages = state_snapshot.values.get("messages", [])
                    child_messages = child_snapshot.values.get("messages", [])
                    new_child_messages = child_messages[len(parent_messages) :]

                    update_payload = {
                        "context": {
                            **state_snapshot.values.get("context", {}),
                            **parent_context_update,
                        }
                    }
                    if new_child_messages:
                        update_payload["messages"] = new_child_messages

                    await app.aupdate_state(config, update_payload)
                    state_snapshot = await app.aget_state(config)
        return state_snapshot

    @classmethod
    def _resolve_pause_node(
        cls, next_node_id: Optional[str], config_obj: Any, messages: List[Any] = None, state_snapshot: Any = None
    ) -> Tuple[Optional[str], Optional[Dict], str]:
        """Returns (pause_type, pause_config, display_output)."""
        if not next_node_id or not config_obj:
            return None, None, ""

        pause_type = None
        pause_config = None

        # 1. New Flow: check interrupt-based pauses from tasks
        if next_node_id in ["human_interaction_node", "human_gate"] and state_snapshot:
            for task in state_snapshot.tasks:
                if task.interrupts:
                    pause_config = task.interrupts[0].value
                    pause_type = pause_config.get("type") if isinstance(pause_config, dict) else None
                    break

        # 2. If not found from tasks, check additional_kwargs in messages
        if not pause_config and messages:
            for msg in reversed(messages):
                if isinstance(msg, dict):
                    kwargs = msg.get("additional_kwargs", {}) or msg.get("metadata", {})
                else:
                    kwargs = getattr(msg, "additional_kwargs", {})
                if kwargs.get("pause_config"):
                    pause_config = kwargs["pause_config"]
                    pause_type = pause_config.get("type")
                    break

        # Build display output for human-interaction tools
        display_output = ""
        if pause_config and pause_type:
            if pause_type == "followup_selection":
                display_output = pause_config.get("question", "")
            elif pause_type == "followup_form":
                display_output = f"Form: {pause_config.get('title', 'Information Required')}"
            return pause_type, pause_config, display_output

        # 3. Canvas Node Gate-level pauses (no display override needed)
        clean_id = re.sub(r"^(gate_|pre_gate_|post_gate_)", "", next_node_id)
        if next_node_id.startswith("bp_gate_"):
            paused_node = next((n for n in config_obj.nodes if next_node_id.endswith(f"_{n.id}") or next_node_id.endswith(n.id)), None)
        else:
            paused_node = next((n for n in config_obj.nodes if n.id == clean_id), None)

        if not paused_node:
            return None, None, ""
        node_type = getattr(paused_node, "type", None)
        node_config = (
            paused_node.model_dump(exclude_unset=True, exclude_none=True)
            if hasattr(paused_node, "model_dump")
            else (paused_node.__dict__ if paused_node else {})
        )
        return node_type, node_config, ""

    @classmethod
    def _snapshot_pause_info(
        cls, state_snapshot: Any, config_obj: Any, messages: Any = None
    ) -> Tuple[bool, bool, Optional[str], Optional[str], Optional[Dict], str]:
        """
        Single source of truth for pause metadata derived from a state snapshot.
        Returns:
            (is_paused, is_bp, next_node, pause_type, pause_config, display_output)

        Used by run/stream/resume/rework and the thread-history endpoint so the
        (is_paused / next_node / pause_type / pause_config) derivation is never
        re-implemented in each caller.
        """
        has_interrupts = any(t.interrupts for t in state_snapshot.tasks)
        is_paused = len(state_snapshot.next) > 0 or has_interrupts
        next_node = state_snapshot.next[0] if state_snapshot.next else None
        if not next_node and has_interrupts:
            next_node = "human_interaction_node"
        is_bp = cls._is_breakpoint_interrupt(state_snapshot.next, config_obj)
        pause_type, pause_config, display_output = cls._resolve_pause_node(
            next_node,
            config_obj,
            messages=messages,
            state_snapshot=state_snapshot,
        )
        return is_paused, is_bp, next_node, pause_type, pause_config, display_output

    @staticmethod
    def _resolve_trim(
        trim_context: Optional[bool], ctx: Optional[Dict[str, Any]]
    ) -> bool:
        """Decide whether to trim the response context.

        Explicit `trim_context` (backward-compat) wins. Otherwise derive from
        the run's `source`: canvas/editor runs keep full context (Variable
        Explorer drag-and-drop); marketplace/webhook/scheduler/API runs trim.
        """
        if trim_context is not None:
            return trim_context
        source = (ctx or {}).get("source") if isinstance(ctx, dict) else None
        return source not in ("canvas", "editor")

    @staticmethod
    def _trim_response_context(ctx: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Strips large flattened node-output duplicates + base64 blobs from the
        run/stream/resume response context. Originals remain available under
        ``ctx['nodes']`` for the frontend variable explorer / HITL UI."""
        if not isinstance(ctx, dict):
            return ctx or {}

        _DROP_EXACT = {
            "vision_result",
            "latest_ocr_output",
            "latest_classification",
            "current_file",
            "current_batch",
            "file_list",
        }
        _DROP_PREFIX = ("ocr_output_", "classification_", "agg_")

        trimmed = {}
        for key, value in ctx.items():
            if key == "latest_file_base64":
                continue
            if key in _DROP_EXACT:
                continue
            if key.startswith(_DROP_PREFIX):
                continue
            if isinstance(value, dict):
                value = WorkflowService._trim_response_context(value)
            trimmed[key] = value
        return trimmed

    @classmethod
    def _is_breakpoint_interrupt(cls, next_nodes: Any, config_obj: Any) -> bool:
        """Checks if snapshot.next contains a target node with an active wire breakpoint."""
        if not next_nodes:
            return False

        # Any bp_gate_ node is by definition a wire breakpoint interrupt
        for n in next_nodes:
            if str(n).startswith("bp_gate_"):
                return True

        if not config_obj or not getattr(config_obj, "edges", None):
            return False

        breakpoint_targets = set()
        for edge in config_obj.edges:
            is_bp = getattr(edge, "is_breakpoint", False) or (
                isinstance(getattr(edge, "data", None), dict) and edge.data.get("isBreakpoint")
            )
            if is_bp and getattr(edge, "target", None):
                breakpoint_targets.add(edge.target)

        for n in next_nodes:
            str_n = str(n)
            clean_n = re.sub(r"^(gate_|pre_gate_|post_gate_)", "", str_n)
            if clean_n in breakpoint_targets or str_n in breakpoint_targets:
                return True
            # Also check matching target node ID inside bp_gate_source_target
            if any(str_n.endswith(f"_{target}") or str_n.endswith(target) for target in breakpoint_targets):
                return True
        return False


    @classmethod
    async def _load_workflow_variables(cls, workflow_id: str) -> Dict[str, Any]:
        """
        Loads workflow variables from Postgres and returns them as a
        flat dict: { variable_key: variable_value }.

        This is injected into state['context']['workflow_variables'] so any node
        can access shared config via: ctx.get('workflow_variables', {}).get('agent_classes')

        Non-blocking: returns empty dict on failure to avoid blocking execution.
        """
        try:
            async with AsyncSessionLocal() as session:
                variables = await list_workflow_variables(
                    session, workflow_id, reveal_secrets=True
                )

            if not variables:
                return {}

            # Transform list of variable records into a flat key-value dict
            return {v["key"]: v["value"] for v in variables}

        except Exception as e:
            logger.warning(
                f"Failed to load workflow variables for {workflow_id}: {e}. Continuing with empty variables."
            )
            return {}
