from typing import Dict, Any, Callable, Optional

# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph, START, END

from Apps.AgentMesh.Schema.WorkflowSchema import WorkflowConfig, NodeConfig, EdgeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Core.Compiler.NodeFactory import NodeFactory
import logging

logger = logging.getLogger(__name__)


def gate_identity_node(state: AgentState):
    return {}


class GraphCompiler:
    """
    Dynamically compiles a WorkflowConfig into an executable LangGraph StateGraph.
    Uses NodeFactory for node-level implementation details.
    """

    @classmethod
    def compile(cls, workflow: WorkflowConfig, checkpointer=None) -> Any:
        # Preprocess / flatten visual Group nodes if any are present
        has_group = any(node.type == "group" for node in workflow.nodes)
        if has_group:
            flat_nodes = []
            flat_edges = []
            group_ids = {node.id for node in workflow.nodes if node.type == "group"}

            # 1. Expand internal nodes of groups
            for node in workflow.nodes:
                if node.type != "group":
                    flat_nodes.append(node)
                else:
                    internal_nodes_raw = node.internal_nodes or []
                    for in_raw in internal_nodes_raw:
                        # Extract nested React Flow data and override the visual type 'agentNode'
                        node_data = in_raw.get("data", {})
                        node_type = node_data.get("type") or in_raw.get("type")

                        flat_dict = {**in_raw, **node_data}
                        flat_dict["type"] = node_type

                        flat_nodes.append(NodeConfig(**flat_dict))

            # 2. Reconstruct edges using the stored original_edges
            for edge in workflow.edges:
                if edge.source not in group_ids and edge.target not in group_ids:
                    flat_edges.append(edge)

            for node in workflow.nodes:
                if node.type == "group":
                    orig_edges_raw = node.original_edges or []
                    for oe_raw in orig_edges_raw:
                        flat_edges.append(EdgeConfig(**oe_raw))

            workflow = workflow.model_copy(
                update={"nodes": flat_nodes, "edges": flat_edges}
            )

        # 1. Initialize StateGraph with Global Defaults (2026 DX Pattern)
        graph = StateGraph(AgentState)

        # Ensure we always have a checkpointer for Persistence features.
        # NOTE: WorkflowRegistry import is intentionally deferred here to avoid a circular import
        # (WorkflowRegistry → GraphCompiler → WorkflowRegistry at module load time).
        if not checkpointer:
            from Apps.AgentMesh.Core.WorkflowRegistry import workflow_registry

            checkpointer = workflow_registry.checkpointer

        # 2. Add all Nodes via NodeFactory FIRST
        iter_redirects = {}
        for node in workflow.nodes:
            node_func = NodeFactory.build(node, all_nodes=workflow.nodes, workflow_id=workflow.id)
            graph.add_node(node.id, node_func)

        # 3. Pre-process for Iterative Loops (Self-Correction)
        from Apps.AgentMesh.Core.Compiler import nodes

        for node in workflow.nodes:
            policy = node.iteration_policy
            if policy and policy.enabled:
                eval_id = f"eval_{node.id}"
                logger.debug(f"Injecting Evaluator for node '{node.id}' -> '{eval_id}'")
                evaluator_func = nodes.build_evaluator_node(node.id, policy)
                graph.add_node(eval_id, evaluator_func)

                # The node always goes to its evaluator first
                graph.add_edge(node.id, eval_id)
                iter_redirects[node.id] = eval_id

        # 4. Pre-process for HITL (Human-in-the-Loop) - BOTH before and after
        pre_gate_redirects = {}
        post_gate_redirects = {}

        def gate_identity_node(state: AgentState):
            return {}

        # 🔄 GLOBAL HUMAN GATE: Always available for Tool calls
        graph.add_node("human_gate", gate_identity_node)

        # 🔄 GLOBAL HUMAN INTERACTION NODE
        from Apps.AgentMesh.Core.Compiler.nodes.human_interaction_node import (
            build_human_interaction_node,
        )

        hi_node_config = NodeConfig(
            id="human_interaction_node",
            type="human_interaction_node",
            name="Human Interaction",
        )
        graph.add_node(
            "human_interaction_node", build_human_interaction_node(hi_node_config)
        )

        # Fix #11: pre-compute once so the inner loop does O(1) lookups instead of O(n·m) scans.
        button_trigger_ids = {
            n.id for n in workflow.nodes if n.type == "button_trigger"
        }
        # Map each source node → number of its edges that target a button_trigger
        button_outgoing_counts: Dict[str, int] = {}
        for _e in workflow.edges:
            if _e.target in button_trigger_ids:
                button_outgoing_counts[_e.source] = (
                    button_outgoing_counts.get(_e.source, 0) + 1
                )

        for node in workflow.nodes:
            has_upstream_node = any(
                edge.target == node.id
                and edge.source not in ("START", "start")
                for edge in workflow.edges
            )

            # An OCR node only needs to collect a file itself when it has no
            # upstream node. If a form/camera/other node already supplied the
            # file, pausing here would ask the user to upload it a second time.
            is_standalone_ocr = node.type == "ocr_node" and not has_upstream_node

            # Check if button_trigger has incoming edges (meaning it's placed mid-stream, not as a start node)
            is_mid_stream_button = False
            if node.type == "button_trigger":
                incoming_edges = [
                    edge
                    for edge in workflow.edges
                    if edge.target == node.id and edge.source not in ("START", "start")
                ]
                if incoming_edges:
                    # Only treat as mid-stream pause if the parent node has a single outgoing edge.
                    # If parent has multiple outgoing edges, the parent's post_gate handles the choice pause.
                    parent_id = incoming_edges[0].source
                    parent_outgoing = [
                        e for e in workflow.edges if e.source == parent_id
                    ]
                    if len(parent_outgoing) <= 1:
                        is_mid_stream_button = True

            # A. interrupt_before -> Inject a gate BEFORE the node
            # Input nodes automatically wait for user submission. OCR does so
            # only when it has no upstream node capable of supplying its file.
            if (
                node.interrupt_before
                or node.type in ("form_node", "camera", "stt_trigger")
                or is_standalone_ocr
                or is_mid_stream_button
            ):
                gate_id = f"pre_gate_{node.id}"
                logger.debug(f"Injecting Pre-Wait Gate: {gate_id} -> '{node.id}'")
                graph.add_node(gate_id, gate_identity_node)
                # Static edge from gate to node
                graph.add_edge(gate_id, node.id)
                pre_gate_redirects[node.id] = gate_id

            # B. interrupt_after -> Inject a gate AFTER the node
            # 🔄 NEW: If a node has multiple outgoing edges to button_triggers, we must pause after it to let the user choose!
            # O(1) lookup using pre-computed map (fix #11)
            has_multiple_button_targets = button_outgoing_counts.get(node.id, 0) > 1

            if node.interrupt_after or has_multiple_button_targets:
                gate_id = f"post_gate_{node.id}"
                logger.debug(f"Injecting Post-Wait Gate: '{node.id}' -> {gate_id}")
                graph.add_node(gate_id, gate_identity_node)

                # If we have an evaluator (iterative), it handles its own Finish-to-Gate transition.
                # Otherwise, we add a static edge from the node to the gate.
                if node.id not in iter_redirects:
                    graph.add_edge(node.id, gate_id)

                post_gate_redirects[node.id] = gate_id

        # 5. Add Edges & Conditional Routing
        cls._apply_graph_edges(
            graph,
            workflow,
            pre_gate_redirects=pre_gate_redirects,
            post_gate_redirects=post_gate_redirects,
            iter_redirects=iter_redirects,
        )

        # 6. Apply Interrupts specifically to Gate nodes and Wire Breakpoints
        interrupt_before_list = ["human_gate"]  # 🔄 Always include human_gate
        for gate_id in pre_gate_redirects.values():
            interrupt_before_list.append(gate_id)
        for gate_id in post_gate_redirects.values():
            interrupt_before_list.append(gate_id)

        # 🛑 WIRE BREAKPOINTS (Alt + Click wire stop functionality)
        for edge in workflow.edges:
            is_bp = getattr(edge, "is_breakpoint", False) or (
                isinstance(getattr(edge, "data", None), dict)
                and edge.data.get("isBreakpoint")
            )
            if is_bp and edge.target:
                bp_gate_id = f"bp_gate_{edge.source}_{edge.target}"
                if bp_gate_id not in interrupt_before_list:
                    interrupt_before_list.append(bp_gate_id)
                    logger.info(
                        f"🛑 Registered Wire Breakpoint interrupt gate: '{bp_gate_id}' (edge {edge.source} -> {edge.target})"
                    )

        logger.info(f"[HITL COMPILE] pre_gate_redirects: {pre_gate_redirects}")
        logger.info(f"[HITL COMPILE] post_gate_redirects: {post_gate_redirects}")
        logger.info(
            f"[HITL COMPILE] Final interrupt_before list: {interrupt_before_list}"
        )

        # 7. Compile to LangGraph application
        return graph.compile(
            checkpointer=checkpointer,
            interrupt_before=interrupt_before_list if interrupt_before_list else None,
        )

    @classmethod
    def _apply_graph_edges(
        cls,
        graph: StateGraph,
        workflow: WorkflowConfig,
        pre_gate_redirects: Dict[str, str] = None,
        post_gate_redirects: Dict[str, str] = None,
        iter_redirects: Dict[str, str] = None,
    ):
        """Processes all edges and implements multi-trigger entry routing."""
        # 1. Identify global configurations
        global_err = workflow.global_error_node
        node_types = {n.id: n.type for n in workflow.nodes}

        # Categorize entry points
        trigger_entry_points = {
            "chat": [],
            "webhook": [],
            "cron_trigger": [],
            "polling": [],
        }

        # First pass: Categorize edges from START or Trigger nodes
        for edge in workflow.edges:
            source_type = node_types.get(edge.source)
            if edge.source in ["START", "start"]:
                is_bp = getattr(edge, "is_breakpoint", False) or (
                    isinstance(getattr(edge, "data", None), dict)
                    and edge.data.get("isBreakpoint")
                )
                target = edge.target
                if is_bp:
                    bp_gate_id = f"bp_gate_{edge.source}_{target}"
                    actual_target = (
                        pre_gate_redirects.get(target, target)
                        if pre_gate_redirects
                        else target
                    )
                    logger.info(
                        f"🛑 Injecting START Breakpoint Gate '{bp_gate_id}': START -> {bp_gate_id} -> {actual_target}"
                    )
                    graph.add_node(bp_gate_id, gate_identity_node)
                    graph.add_edge(bp_gate_id, actual_target)
                    target = bp_gate_id
                trigger_entry_points["chat"].append(target)

            elif source_type == "webhook":
                trigger_entry_points["webhook"].append(edge.source)
            elif source_type == "cron":
                trigger_entry_points["cron_trigger"].append(edge.source)
            elif source_type == "polling":
                trigger_entry_points["polling"].append(edge.source)

        # 2. Implement the Universal Entry Router from LangGraph's START
        dynamic_resumption = workflow.dynamic_resumption

        # Fix #3: defined here (before the closure) so type-checkers can resolve the name
        # and to eliminate the fragile forward-reference that existed when it was below the function.
        all_node_ids = set(node_types.keys())

        def universal_entry_router(state: AgentState):
            context = state.get("context", {})
            # Fix #12: guard against a non-dict value stored under the 'context' key
            if not isinstance(context, dict):
                context = {}
            source = context.get("source", "chat")
            hitl_action = context.get("hitl_action")
            active_agent = state.get("active_agent")

            # ✅ REWORK JUMP: If this is an HITL rework, jump straight back to the active agent
            if hitl_action == "rework" and active_agent:
                # 🔄 REDIRECT: If the agent has a pre-gate (interrupt_before), jump to the gate to allow 'Double Re-edit'
                target_id = active_agent
                if pre_gate_redirects and active_agent in pre_gate_redirects:
                    target_id = pre_gate_redirects[active_agent]

                logger.debug(
                    f"Entry Router - Rework detected, jumping back to agent '{active_agent}' (Target: {target_id})"
                )
                return target_id

            # ✅ RESUMPTION JUMP (OPTIONAL): If we have an active agent and dynamic resumption is enabled,
            # jump straight back to them.
            skip_pre_gate = context.get("skip_pre_gate", False)

            resumable_types = {
                "llm_agent",
                "deep_agent",
                "supervisor",
                "swarm",
                "human_interaction",
                "form_node",
            }

            if (
                dynamic_resumption
                and active_agent
                and active_agent in all_node_ids
                and node_types.get(active_agent) in resumable_types
            ):
                target_id = active_agent
                if (
                    hitl_action == "rework"
                    and not skip_pre_gate
                    and pre_gate_redirects
                    and active_agent in pre_gate_redirects
                ):
                    target_id = pre_gate_redirects[active_agent]

                logger.debug(
                    f"Entry Router - Dynamic Resumption active, jumping back to agent '{active_agent}' (Target: {target_id})"
                )
                return target_id

            # Look up targets for this trigger source
            targets = (
                trigger_entry_points.get(source)
                or trigger_entry_points.get("chat")
                or ["END"]
            )

            # Clean up targets and apply dynamic pre-gate redirects
            final_targets = []
            for t in targets:
                if t in ["END", "end"]:
                    final_targets.append(END)
                else:
                    if (
                        not skip_pre_gate
                        and pre_gate_redirects
                        and t in pre_gate_redirects
                    ):
                        final_targets.append(pre_gate_redirects[t])
                    else:
                        final_targets.append(t)

            # LangGraph supports returning a single string or a list for parallel execution (fan-out)
            return final_targets[0] if len(final_targets) == 1 else final_targets

        # Register the entry router — add gate IDs to path map so the router can reach them
        path_map = {nid: nid for nid in all_node_ids if nid not in ["END", "end"]}
        if pre_gate_redirects:
            for gid in pre_gate_redirects.values():
                path_map[gid] = gid
        if post_gate_redirects:
            for gid in post_gate_redirects.values():
                path_map[gid] = gid

        path_map[END] = END

        graph.add_conditional_edges(START, universal_entry_router, path_map)

        # 3. Group and process all internal edges
        source_groups: Dict[str, list] = {}
        for edge in workflow.edges:
            # We skip 'START' edges here because they are handled by the universal_entry_router
            if edge.source in ["START", "start"]:
                continue

            # 🔄 REDIRECT TARGETS: If a target has a pre-gate, we must route to the pre-gate instead.
            # NOTE: Use model_copy() to avoid mutating the original EdgeConfig — calling compile()
            # twice on the same WorkflowConfig would otherwise stack redirects and corrupt targets.
            actual_target = edge.target
            # Skip pre-gate redirect for handle-based edges (loop-backs, conditional branches).
            # These come from logic nodes (Aggregator, Switch, Condition) and represent
            # routing decisions that should go directly to the target node.
            # Pre-gates are only needed for initial entry from START / dynamic resumption.
            if (
                not edge.source_handle
                and pre_gate_redirects
                and actual_target in pre_gate_redirects
            ):
                actual_target = pre_gate_redirects[actual_target]

            # 🛑 WIRE BREAKPOINT GATES: Inject explicit Breakpoint Gate node before target/pre-gate
            is_bp = getattr(edge, "is_breakpoint", False) or (
                isinstance(getattr(edge, "data", None), dict)
                and edge.data.get("isBreakpoint")
            )
            if is_bp:
                bp_gate_id = f"bp_gate_{edge.source}_{edge.target}"
                logger.info(
                    f"🛑 Injecting Breakpoint Gate '{bp_gate_id}': {edge.source} -> {bp_gate_id} -> {actual_target}"
                )
                graph.add_node(bp_gate_id, gate_identity_node)
                graph.add_edge(bp_gate_id, actual_target)
                actual_target = bp_gate_id

            edge = edge.model_copy(update={"target": actual_target})

            if edge.source not in source_groups:
                source_groups[edge.source] = []
            source_groups[edge.source].append(edge)

        # 4. Process each source group and identify Error Routers
        tool_nodes_found = set()
        error_routers_processed = set()

        # 🔄 ERROR ROUTER LOGIC: MUST process all error routers, even if they have NO outgoing edges in config
        for node in workflow.nodes:
            if node.type == "error_router":
                source_id = node.id
                group = source_groups.get(source_id, [])
                logger.debug(
                    f"Injecting Automated Error Router logic for '{source_id}'"
                )
                cls._apply_error_router_logic(graph, workflow, source_id, group)
                error_routers_processed.add(source_id)

        for source_id, group in source_groups.items():
            if source_id in error_routers_processed:
                continue

            # 🔄 ITERATIVE LOGIC: If this source has an evaluator, we intercept the loop first.
            if iter_redirects and source_id in iter_redirects:
                eval_id = iter_redirects[source_id]
                gate_id = (
                    post_gate_redirects.get(source_id) if post_gate_redirects else None
                )
                cls._apply_iterative_router(
                    graph,
                    workflow,
                    eval_id,
                    source_id,
                    group,
                    tool_nodes_found,
                    gate_id=gate_id,
                    global_err=global_err,
                )

                # 🛠️ FIX (Apr 1, 2026): If an iterative node ALSO has HITL enabled, we must apply the gate router logic
                # to the gate node so it knows how to proceed after the pause.
                if gate_id:
                    cls._apply_gate_router(
                        graph,
                        workflow,
                        gate_id,
                        source_id,
                        group,
                        tool_nodes_found,
                        pre_gate_redirects=pre_gate_redirects,
                        global_err=global_err,
                    )

            # 🔄 HITL LOGIC: If this source has a post-gate, we pause AFTER execution.
            elif post_gate_redirects and source_id in post_gate_redirects:
                gate_id = post_gate_redirects[source_id]
                cls._apply_gate_router(
                    graph,
                    workflow,
                    gate_id,
                    source_id,
                    group,
                    tool_nodes_found,
                    pre_gate_redirects=pre_gate_redirects,
                    global_err=global_err,
                )
            else:
                cls._process_source_group(
                    graph,
                    workflow,
                    source_id,
                    group,
                    tool_nodes_found,
                    pre_gate_redirects=pre_gate_redirects,
                    global_err=global_err,
                )

        # 4. Add Orchestration-specific routing (Supervisor & Swarm)
        cls._apply_orchestration_routing(graph, workflow)

        # 5. Add Global Return Logic for Tool Nodes (Tools -> Caller)
        cls._apply_tool_return_routing(
            graph, workflow, tool_nodes_found, gate_redirects=post_gate_redirects
        )

    @classmethod
    def _apply_error_router_logic(
        cls, graph: StateGraph, workflow: WorkflowConfig, source_id: str, group: list
    ):
        """
        Injects a dynamic router after an Error Router node that either:
        1. Jumps back to the error source node (retry)
        2. Routes to END (abort)
        3. Routes to another target based on manual edges if any
        """
        # Collect nodes that this could potentially jump back to
        all_node_ids = {n.id for n in workflow.nodes}
        path_map = {nid: nid for nid in all_node_ids}
        path_map[END] = END
        path_map["END"] = END

        # Add manual static and conditional edges if the user drew them
        for edge in group:
            tid = edge.target
            if tid in ["END", "end"]:
                path_map[END] = END
            else:
                path_map[tid] = tid

        def error_recovery_router(state: AgentState):
            intent = state.get("intent", "abort")
            error_source = state.get("error_source")

            if intent == "retry" and error_source and error_source in all_node_ids:
                logger.info(
                    f"Error Router [{source_id}] - Performing RETRY jump to source: '{error_source}'"
                )
                return error_source

            if intent == "abort":
                logger.debug(
                    f"Error Router [{source_id}] - ABORT received. Routing to END."
                )
                # 🚨 Mark the Langfuse trace as ERROR so the whole run shows as failed
                # even though LangGraph ends normally (the node exception was swallowed
                # for error routing and is only present in state).
                try:
                    from Apps.AgentMesh.Utilities.Observer import mark_langfuse_trace_error
                    ctx = state.get("context") or {}
                    mark_langfuse_trace_error(
                        ctx.get("thread_id", "anonymous"),
                        f"Workflow aborted at node '{error_source or source_id}' via error router '{source_id}'.",
                    )
                except Exception:
                    pass  # Observability must never break routing.
                return END

            # Fallback to manual edges: find target for this intent in edges
            # (Matches intent to target node ID if possible)
            for edge in group:
                if edge.target == intent:
                    return edge.target

            # Default fallback for abort/unknown
            return END

        error_recovery_router.__name__ = f"recovery_router_{source_id}"
        graph.add_conditional_edges(source_id, error_recovery_router, path_map)

    @classmethod
    def _apply_iterative_router(
        cls,
        graph: StateGraph,
        workflow: WorkflowConfig,
        eval_id: str,
        original_node_id: str,
        group: list,
        tool_nodes: set,
        gate_id: Optional[str] = None,
        global_err: str = None,
    ):
        """
        Injects a dynamic router after an Evaluator that either:
        1. Loops back to the original agent node (LOOP)
        2. Proceeds to the next step (FINISH), possibly through a Gate
        """
        # 1. Identify characteristics of the original node's exit logic
        tool_edges, conditional_edges, other_edges, is_tool, tool_target = (
            cls._classify_edges(workflow, group, tool_nodes)
        )

        # Standard Fallback logic
        standard_fallback = cls._compute_standard_fallback(other_edges, tool_target)

        # 2. Build the combined router factory
        def _make_iterative_router():
            # Instantiate the original logic router for this specific node
            original_logic_router = cls._create_unified_router(
                workflow,
                original_node_id,
                conditional_edges,
                has_tools=is_tool,
                fallback=standard_fallback,
                error_target=global_err,
            )

            def iterative_loop_router(state: AgentState):
                iter_next = state.get("iter_next")

                # LOOP BACK if evaluator says so
                if iter_next == "LOOP":
                    logger.debug(
                        f"Iterative Router [{eval_id}] - Loop back to '{original_node_id}'"
                    )
                    return "loop"

                # PROCEED with original logic if finished
                if gate_id:
                    logger.debug(
                        f"Iterative Router [{eval_id}] - Finished, routing to Gate '{gate_id}'"
                    )
                    return "gate"

                target = original_logic_router(state)
                logger.debug(
                    f"Iterative Router [{eval_id}] - Finished, moving to: '{target}'"
                )
                return target

            iterative_loop_router.__name__ = f"iter_router_{eval_id}"
            return iterative_loop_router

        # 3. Build Path Map
        path_map = {"loop": original_node_id}
        if gate_id:
            path_map["gate"] = gate_id

        # Tools branch
        if is_tool:
            path_map["tools"] = tool_edges[0].target
            path_map["human_gate"] = "human_gate"

        # All static and conditional targets
        for edge in group:
            tid = edge.target
            if tid in ["END", "end"]:
                path_map[END] = END
            elif isinstance(tid, list):
                for sub_t in tid:
                    path_map[sub_t] = END if sub_t in ["END", "end"] else sub_t
            else:
                path_map[tid] = tid

        # Final normalization for LangGraph safety
        for k, v in path_map.items():
            if v in ["END", "end", "END_NODE"]:
                path_map[k] = END

        graph.add_conditional_edges(eval_id, _make_iterative_router(), path_map)

    @classmethod
    def _apply_gate_router(
        cls,
        graph: StateGraph,
        workflow: WorkflowConfig,
        gate_id: str,
        original_node_id: str,
        group: list,
        tool_nodes: set,
        pre_gate_redirects: Dict[str, str] = None,
        global_err: str = None,
    ):
        """
        Injects a dynamic router after an HITL Gate that either:
        1. Jumps back to the original node (Rework)
        2. Evaluates the original node's routing conditions (Approve/Normal)
        """
        # 1. Identify characteristics of the original node's exit logic.
        # NOTE: _apply_gate_router has a 4-way split (tool / conditional / handle / other)
        # so we use _classify_edges only for the tool/conditional parts and compute the rest manually.
        tool_edges, conditional_edges, _, is_tool, tool_target = cls._classify_edges(
            workflow, group, tool_nodes
        )
        # handle_edges MUST be filtered by source_handle explicitly — using _classify_edges' other_edges
        # here caused the bug: other_edges includes non-handle edges with source_handle=None.
        handle_edges = [e for e in group if e.source_handle]
        other_edges = [
            e
            for e in group
            if not e.condition
            and not e.source_handle
            and not cls._is_tool_edge(workflow, e)
        ]

        # Standard Fallback logic
        standard_fallback = cls._compute_standard_fallback(other_edges, tool_target)

        # Build path map for handles (if any)
        handle_path_map = {}
        if handle_edges:
            for e in handle_edges:
                tid = e.target
                actual_target = END if tid in ["END", "end"] else tid
                handle_path_map[str(e.source_handle).lower()] = actual_target

            # Map fallback 'else'
            else_edge = next(
                (
                    e
                    for e in group
                    if not e.source_handle
                    and not e.condition
                    and not cls._is_tool_edge(workflow, e)
                ),
                None,
            )
            if else_edge:
                handle_path_map["else"] = (
                    END if else_edge.target in ["END", "end"] else else_edge.target
                )
            elif "else" not in handle_path_map:
                handle_path_map["else"] = END

        # 2. Build the combined router factory
        def _make_combined_hitl_router():
            # Instantiate the original logic router for this specific node
            if handle_edges:
                # Resolve paths based on handles
                h_map = {
                    e.source_handle.lower(): (
                        END if e.target in ["END", "end"] else e.target
                    )
                    for e in handle_edges
                }
                else_edge = next(
                    (e for e in group if not e.source_handle and not e.condition), None
                )
                fallback_target = (
                    (END if else_edge.target in ["END", "end"] else else_edge.target)
                    if else_edge
                    else END
                )

                def original_logic_router(state: AgentState):
                    intent = str(state.get("intent", "else")).lower()
                    if intent in h_map:
                        return h_map[intent]
                    return fallback_target
            else:
                original_logic_router = cls._create_unified_router(
                    workflow,
                    original_node_id,
                    conditional_edges,
                    has_tools=is_tool,
                    fallback=standard_fallback,
                    error_target=global_err,
                )

            def hitl_gate_router(state: AgentState):
                hitl_action = state.get("context", {}).get("hitl_action")
                messages = state.get("messages", [])

                # 🕵️ CHOICE SELECTION: If the user clicked a specific choice button, route directly to it.
                clicked_button_id = state.get("context", {}).get("clicked_button_id")
                if clicked_button_id:
                    target_id = clicked_button_id
                    if pre_gate_redirects and clicked_button_id in pre_gate_redirects:
                        target_id = pre_gate_redirects[clicked_button_id]
                    logger.debug(
                        f"Gate Router [{gate_id}] - User clicked choice button '{clicked_button_id}'. Routing to '{target_id}'"
                    )
                    return target_id

                # 🕵️ NEW (Apr 1, 2026): Detect if the user sent a new text message while paused.
                is_new_message = False
                if messages:
                    last_msg = messages[-1]
                    msg_type = type(last_msg).__name__
                    if msg_type == "HumanMessage" or (
                        isinstance(last_msg, dict) and last_msg.get("role") == "user"
                    ):
                        is_new_message = True

                # JUMP BACK if human requested a rework OR sent a new message without clicking buttons
                if hitl_action == "rework" or (not hitl_action and is_new_message):
                    logger.debug(
                        f"Gate Router [{gate_id}] - Rework/New Message detected, jumping back to '{original_node_id}'"
                    )
                    return "rework"

                # PROCEED with original logic if approved
                # Handle-based routing logic (Switch/Condition/Action nodes)
                if handle_edges:
                    intent = str(state.get("intent", "else")).lower()
                    if intent in handle_path_map:
                        target = handle_path_map[intent]
                    else:
                        target = handle_path_map.get("else", END)
                    # If the target is found in handle_path_map but we need to return the 'key'
                    # Actually, LangGraph expects the dictionary key OR the direct node name depending on path_map structure.
                    # We return the target node name itself.
                    logger.debug(
                        f"Gate Router [{gate_id}] - Handle match '{intent}', routing to: '{target}'"
                    )
                    return target

                # Conditional / Agent logic
                target = original_logic_router(state)
                logger.debug(
                    f"Gate Router [{gate_id}] - Approved transition to: '{target}'"
                )
                return target

            hitl_gate_router.__name__ = f"hitl_gate_router_{gate_id}"
            return hitl_gate_router

        # 3. Build the Path Map specifically for the combined router
        # Must include 'rework' plus all possible identifiers the original router might return
        # 🔄 REDIRECT REWORK: If the agent has a pre-gate, jump back to the pre-gate for a clean re-entry
        rework_target = original_node_id
        if pre_gate_redirects and original_node_id in pre_gate_redirects:
            rework_target = pre_gate_redirects[original_node_id]

        path_map = {"rework": rework_target}

        # Tools branch
        if is_tool:
            path_map["tools"] = tool_edges[0].target
            path_map["human_gate"] = "human_gate"

        # All static and conditional targets
        for edge in group:
            tid = edge.target
            # tid already points to pre-gates if necessary due to pre-processing in _apply_graph_edges
            if tid in ["END", "end"]:
                path_map[END] = END
            elif isinstance(tid, list):
                for sub_t in tid:
                    path_map[sub_t] = END if sub_t in ["END", "end"] else sub_t
            else:
                path_map[tid] = tid

        # Final normalization for LangGraph safety
        for k, v in path_map.items():
            if v in ["END", "end", "END_NODE"]:
                path_map[k] = END

        graph.add_conditional_edges(gate_id, _make_combined_hitl_router(), path_map)

    @classmethod
    def _process_source_group(
        cls,
        graph: StateGraph,
        workflow: WorkflowConfig,
        source_id: str,
        group: list,
        tool_nodes: set,
        original_source_id: str = None,
        pre_gate_redirects: Dict[str, str] = None,
        global_err: str = None,
    ):
        source = START if source_id in ["START", "start"] else source_id
        # original_source_id is the pre-gate-redirect ID, used for config lookups
        lookup_id = original_source_id or source_id

        # 🔄 ERROR LOGIC: Identify target for failures
        source_config = next((n for n in workflow.nodes if n.id == lookup_id), None)
        error_target = source_config.on_error if source_config else None
        error_target = error_target or global_err

        tool_edges = [e for e in group if cls._is_tool_edge(workflow, e)]
        conditional_edges = [e for e in group if e.condition]

        if tool_edges:
            tool_nodes.add(tool_edges[0].target)

        # Sub-case: Conditional Edges (Manual Rules)
        if conditional_edges:
            # Fallback for Tool Execution
            has_tools = any(n.tools for n in workflow.nodes if n.id == lookup_id)
            tool_target = (
                tool_edges[0].target
                if tool_edges
                else (lookup_id if has_tools else None)
            )

            # Static Fallback: find the first edge in the group that has NO condition
            # If no static edge exists, we default to the tool_target or END.
            other_edges = [
                e
                for e in group
                if not e.condition and not cls._is_tool_edge(workflow, e)
            ]
            static_fallback = (
                other_edges[0].target if other_edges else (tool_target or END)
            )

            cls._apply_conditional_group(
                graph,
                workflow,
                source_id,
                conditional_edges,
                tool_target,
                fallback=static_fallback,
                error_target=error_target,
            )
            return

        # Case: Handle-based Routing (Logic Nodes like Switch/Condition, Splitter)
        handle_edges = [e for e in group if e.source_handle]
        if handle_edges:
            # ── Parallel Fan-Out Detection ──
            # If multiple canvas wires share the same handle (e.g., Splitter DONE → AgentA,
            # Splitter DONE → AgentB), create a synthetic passthrough node that fans out
            # to all targets in parallel. LangGraph merges their state automatically.
            from collections import defaultdict

            handle_groups = defaultdict(list)
            for e in handle_edges:
                handle_groups[e.source_handle.lower()].append(e)

            path_map = {}
            for handle, edges in handle_groups.items():
                if len(edges) > 1:
                    # Multiple targets for same handle → parallel fan-out
                    fan_node_id = f"__fanout_{source_id}_{handle}"

                    async def _fanout_passthrough(state: AgentState, _id=fan_node_id):
                        return state  # Pure passthrough — state flows to all downstream edges

                    _fanout_passthrough.__name__ = fan_node_id

                    from langchain_core.runnables import RunnableLambda

                    graph.add_node(fan_node_id, RunnableLambda(_fanout_passthrough))
                    path_map[handle] = fan_node_id

                    # Create static edges from fan-out node to ALL targets (LangGraph parallel)
                    for e in edges:
                        target = END if e.target in ["END", "end"] else e.target
                        graph.add_edge(fan_node_id, target)
                else:
                    # Single target — normal behavior
                    e = edges[0]
                    target = END if e.target in ["END", "end"] else e.target
                    path_map[handle] = target

            # Support 'else' as a standard fallback in the path map
            # Note: Handlers are case-normalized to ensure matches work with diverse logic node outputs

            # If there's a static edge in the same group, it acts as the 'else' fallback
            else_edge = next(
                (e for e in group if not e.source_handle and not e.condition), None
            )
            if else_edge:
                else_target = (
                    END if else_edge.target in ["END", "end"] else else_edge.target
                )
                if pre_gate_redirects and else_target in pre_gate_redirects:
                    else_target = pre_gate_redirects[else_target]
                path_map["else"] = else_target

            if error_target:
                path_map["error"] = error_target

            # ✅ CRITICAL: Always include END in the map to prevent KeyError: '__end__'
            if END not in path_map:
                path_map[END] = END
            if "else" not in path_map:
                path_map["else"] = path_map.get("else") or END

            def handle_router(state: AgentState):
                if error_target and state.get("error"):
                    return "error"

                # We normalize intent to lowercase for matching handles
                intent = str(state.get("intent", "else")).lower()
                logger.debug(
                    f"[HANDLE ROUTER] node='{source_id}' intent='{intent}' path_map={path_map}"
                )

                if intent in path_map:
                    return intent
                if "else" in path_map:
                    return "else"
                # Fallback to the first available target if no 'else' handle exists
                first_target_key = next(iter(path_map.keys()), END)
                return first_target_key

            handle_router.__name__ = f"handle_router_{source_id}"
            graph.add_conditional_edges(source_id, handle_router, path_map)
            return

        # Sub-case: Agent -> Tools Routing / Error recovery for non-handle nodes
        other_edges = [e for e in group if not cls._is_tool_edge(workflow, e)]

        # Fetch source node configuration for return_direct setting
        is_return_direct = bool(source_config and source_config.return_direct)

        if tool_edges or error_target:
            tool_target = tool_edges[0].target if tool_edges else None
            path_map = {"continue": END}
            if tool_target:
                path_map["tools"] = tool_target
                path_map["human_gate"] = "human_gate"
            if error_target:
                path_map["error"] = error_target

            if other_edges:
                if is_return_direct:
                    path_map["continue"] = END
                else:
                    targets_list = [e.target for e in other_edges]
                    normalized_targets = list(
                        set((END if t in ["END", "end"] else t) for t in targets_list)
                    )
                    path_map["continue"] = (
                        normalized_targets[0] if normalized_targets else END
                    )

            # Use the combined router if we have ANY non-static logic (tools OR errors)
            def combined_logic_router(state: AgentState):
                # Always prioritize Error handling
                if error_target and state.get("error"):
                    return "error"
                # Then Tool calls — Fix #5: guard against empty messages list before indexing
                messages = state.get("messages", [])
                if (
                    tool_target
                    and messages
                    and GraphCompiler._has_tool_calls(messages[-1])
                ):
                    last_msg = messages[-1]
                    tool_calls = getattr(last_msg, "tool_calls", []) or []
                    # Fix #10: delegate to shared helper instead of duplicating the loop
                    if cls._check_tool_interrupt(tool_calls, source_config):
                        logger.info(
                            f"Tool execution requires human approval. Routing to 'human_gate'"
                        )
                        return "human_gate"
                    return "tools"
                return "continue"

            combined_logic_router.__name__ = f"logic_router_{source_id}"
            graph.add_conditional_edges(source_id, combined_logic_router, path_map)
            return

        # Case: Standard Static Edge or Fan-out (Broadcast)
        for edge in group:
            target = (
                END
                if (edge.target in ["END", "end"] or is_return_direct)
                else edge.target
            )
            graph.add_edge(source, target)

    @classmethod
    def _apply_orchestration_routing(cls, graph: StateGraph, workflow: WorkflowConfig):
        """Ensures nodes that set 'next' in state are correctly routed (Supervisor/Swarm)."""
        for node in workflow.nodes:
            if node.type in ["supervisor", "swarm"] and node.member_ids:
                # 1. Map workers to themselves
                path_map = {m: m for m in node.member_ids}

                # 2. Map FINISH to the actual next node in the canvas (if any)
                # Look for edges where this node is the source and the target is NOT a tool node
                next_edge = next(
                    (
                        e
                        for e in workflow.edges
                        if e.source == node.id and not cls._is_tool_edge(workflow, e)
                    ),
                    None,
                )

                target_after_finish = END
                if next_edge:
                    target_after_finish = (
                        END if next_edge.target == "END" else next_edge.target
                    )

                path_map["FINISH"] = target_after_finish
                path_map[END] = END

                # ✅ FIX: Use a factory function to avoid closure variable capture bug.
                # Without this, ALL supervisor/swarm nodes would share the same 'node' reference
                # (the last one in the loop), causing incorrect routing.
                def _make_orch_router(node_id):
                    def orchestration_router(state: AgentState):
                        return state.get("next") or "FINISH"

                    orchestration_router.__name__ = f"orch_router_{node_id}"
                    return orchestration_router

                graph.add_conditional_edges(
                    node.id, _make_orch_router(node.id), path_map
                )

    @classmethod
    def _apply_tool_return_routing(
        cls,
        graph: StateGraph,
        workflow: WorkflowConfig,
        tool_nodes: set,
        gate_redirects: Dict[str, str] = None,
    ):
        """Adds return logic from ToolNodes back to the calling agent."""
        all_agent_ids = [
            n.id
            for n in workflow.nodes
            if n.type in ["llm_agent", "supervisor", "swarm"]
        ]
        path_map = {aid: aid for aid in all_agent_ids}
        path_map[END] = END
        # Also include gate nodes in path_map so tool return can route through gates
        if gate_redirects:
            for gate_id in gate_redirects.values():
                path_map[gate_id] = gate_id

        # 🔄 Include global human_gate and human_interaction_node in path map
        path_map["human_gate"] = "human_gate"
        path_map["human_interaction_node"] = "human_interaction_node"

        for t_node in tool_nodes:
            # Tool nodes themselves don't have gates.
            # The gate is on the AGENT node (interrupt_after on the agent).
            # So tool return routing always starts from the tool node itself.

            def get_tool_return_router(current_t_node):
                def global_tool_return_router(state: AgentState):
                    # Detect Human Interaction Request
                    messages = state.get("messages", [])
                    if messages:
                        last = messages[-1]
                        kwargs = (
                            last.get("additional_kwargs", {})
                            if isinstance(last, dict)
                            else getattr(last, "additional_kwargs", {})
                        )
                        if kwargs.get("pause_config"):
                            logger.info(
                                "Tool Result detected Human Interaction Request. Routing to 'human_interaction_node'"
                            )
                            return "human_interaction_node"

                    t_config = next(
                        (n for n in workflow.nodes if n.id == current_t_node), None
                    )
                    # 1. Node-level return_direct on the ToolNode itself
                    if t_config and t_config.return_direct:
                        return END

                    # 2. Tool-level return_direct checks
                    active_agent_id = state.get("active_agent")
                    agent_config = (
                        next(
                            (n for n in workflow.nodes if n.id == active_agent_id), None
                        )
                        if active_agent_id
                        else None
                    )

                    def is_tool_msg(m):
                        if isinstance(m, dict):
                            return (
                                m.get("type") == "tool"
                                or m.get("role") == "tool"
                                or m.get("type") == "ToolMessage"
                            )
                        from langchain_core.messages import ToolMessage

                        return (
                            isinstance(m, ToolMessage)
                            or type(m).__name__ == "ToolMessage"
                        )

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

                    # Gather all tool messages in the latest turn (supports parallel tool calls)
                    tool_names = []
                    for msg in reversed(messages):
                        if is_tool_msg(msg):
                            t_name = get_t_name(msg)
                            if t_name:
                                tool_names.append(t_name)
                        else:
                            # Stop when we hit a non-tool message
                            break

                    for tool_name in tool_names:
                        # Check agent context tool_configs override
                        if agent_config and agent_config.tool_configs:
                            if agent_config.tool_configs.get(tool_name, {}).get(
                                "return_direct"
                            ):
                                return END

                        # Check local tool node tool_configs override
                        if t_config and t_config.tool_configs:
                            if t_config.tool_configs.get(tool_name, {}).get(
                                "return_direct"
                            ):
                                return END

                    return active_agent_id if active_agent_id else END

                global_tool_return_router.__name__ = f"tool_return_{current_t_node}"
                return global_tool_return_router

            graph.add_conditional_edges(
                t_node, get_tool_return_router(t_node), path_map
            )

        # 🔄 Add universal routing for the human_gate and human_interaction_node
        cls._apply_human_gate_router(graph, workflow)
        cls._apply_human_interaction_router(graph, workflow)

    @classmethod
    def _apply_human_gate_router(cls, graph: StateGraph, workflow: WorkflowConfig):
        """
        Universal router for the 'human_gate'.
        Routes to the tool node if resuming an approved tool execution, otherwise routes back to the active agent.
        """
        all_agent_ids = [n.id for n in workflow.nodes if n.id]
        path_map = {aid: aid for aid in all_agent_ids}
        path_map[END] = END

        def human_gate_exit_router(state: AgentState):
            active_agent = state.get("active_agent")
            messages = state.get("messages", [])
            hitl_action = state.get("context", {}).get("hitl_action")

            if (
                active_agent
                and messages
                and cls._has_tool_calls(messages[-1])
                and hitl_action != "rework"
            ):
                tool_edges = [
                    e
                    for e in workflow.edges
                    if e.source == active_agent and cls._is_tool_edge(workflow, e)
                ]
                if tool_edges:
                    target_tool_node = tool_edges[0].target
                    logger.info(
                        f"Human Gate Exit - Approved tool execution. Routing to tool node '{target_tool_node}'"
                    )
                    return target_tool_node

            logger.debug(f"Human Gate Exit - Resuming back to agent: '{active_agent}'")
            return active_agent if active_agent else END

        graph.add_conditional_edges("human_gate", human_gate_exit_router, path_map)

    @classmethod
    def _apply_human_interaction_router(
        cls, graph: StateGraph, workflow: WorkflowConfig
    ):
        """
        Universal router for the 'human_interaction_node'.
        Routes back to the active agent.
        """
        all_agent_ids = [n.id for n in workflow.nodes if n.id]
        path_map = {aid: aid for aid in all_agent_ids}
        path_map[END] = END

        def human_interaction_exit_router(state: AgentState):
            active_agent = state.get("active_agent")
            logger.debug(
                f"Human Interaction Exit - Resuming back to agent: '{active_agent}'"
            )
            return active_agent if active_agent else END

        graph.add_conditional_edges(
            "human_interaction_node", human_interaction_exit_router, path_map
        )

    @classmethod
    def _is_tool_edge(cls, workflow: WorkflowConfig, edge: Any) -> bool:
        target_node = next((n for n in workflow.nodes if n.id == edge.target), None)
        return target_node and target_node.type == "tool_node"

    @classmethod
    def _classify_edges(
        cls, workflow: WorkflowConfig, group: list, tool_nodes: set = None
    ):
        """Fix #8: shared helper to classify a group of edges into categories.
        Returns: (tool_edges, conditional_edges, other_edges, is_tool, tool_target)
        If tool_nodes set is provided, the tool target is registered into it.
        """
        tool_edges = [e for e in group if cls._is_tool_edge(workflow, e)]
        conditional_edges = [e for e in group if e.condition]
        other_edges = [
            e for e in group if not e.condition and not cls._is_tool_edge(workflow, e)
        ]
        is_tool = bool(tool_edges)
        tool_target = tool_edges[0].target if is_tool else None
        if is_tool and tool_nodes is not None:
            tool_nodes.add(tool_target)
        return tool_edges, conditional_edges, other_edges, is_tool, tool_target

    @staticmethod
    def _compute_standard_fallback(other_edges: list, tool_target) -> Any:
        """Fix #9: shared helper to compute the standard fallback from a list of non-tool, non-conditional edges."""
        if not other_edges:
            return tool_target or END
        if len(other_edges) == 1:
            return other_edges[0].target
        return [e.target for e in other_edges]

    @classmethod
    def _check_tool_interrupt(cls, tool_calls: list, source_config) -> bool:
        """Fix #10: shared helper that checks if any of the pending tool_calls require a human-interrupt
        before execution, based on the source node's tool_configs.
        Returns True if the router should pause at 'human_gate'.
        """
        for tc in tool_calls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name:
                t_conf = {}
                if source_config and source_config.tool_configs:
                    t_conf = source_config.tool_configs.get(name) or {}
                if t_conf.get("interrupt_before") or t_conf.get("interrupt"):
                    return True
        return False

    @classmethod
    def _apply_conditional_group(
        cls,
        graph: StateGraph,
        workflow: WorkflowConfig,
        source_id: str,
        edges: list,
        tool_target: Optional[str] = None,
        fallback: str = END,
        error_target: str = None,
    ):
        path_map = {e.target: (END if e.target == "END" else e.target) for e in edges}
        if END not in path_map:
            path_map[END] = END
        if "END" not in path_map:
            path_map["END"] = END
        if error_target:
            path_map["error"] = error_target

        # Ensure fallback is in path_map
        if fallback and fallback not in path_map:
            path_map[fallback] = END if fallback == "END" else fallback

        if tool_target:
            path_map["tools"] = END if tool_target == "END" else tool_target

        if tool_target:
            path_map["human_gate"] = "human_gate"

        graph.add_conditional_edges(
            source_id,
            cls._create_unified_router(
                workflow,
                source_id,
                edges,
                has_tools=bool(tool_target),
                fallback=fallback,
                error_target=error_target,
            ),
            path_map,
        )

    @staticmethod
    def _has_tool_calls(msg: Any) -> bool:
        """Helper to check if a message has tool calls."""
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            return True
        if isinstance(msg, dict) and msg.get("tool_calls"):
            return True
        content = (
            msg.content
            if hasattr(msg, "content")
            else (msg.get("content") if isinstance(msg, dict) else "")
        )
        if isinstance(content, str) and "<tool_call>" in content:
            return True
        return False

    @staticmethod
    def _extract_content(msg: Any) -> Any:
        """Helper to extract text content from various message formats natively."""
        # pyrefly: ignore [missing-import]
        from langchain_core.messages import BaseMessage

        if isinstance(msg, BaseMessage):
            return msg.content
        if hasattr(msg, "content"):
            return msg.content
        if isinstance(msg, dict) and "content" in msg:
            return msg["content"]
        return str(msg)

    @staticmethod
    def _evaluate_condition(cond, val) -> bool:
        """Helper to evaluate a single edge condition (case-insensitive for strings)."""
        # Convert to strings and lowercase for comparison if they are strings
        s_val = str(val).strip().lower() if val is not None else ""
        s_cond = str(cond.value).strip().lower() if cond.value is not None else ""

        if cond.operator == "==":
            return s_val == s_cond
        if cond.operator == "!=":
            return s_val != s_cond
        if cond.operator == "contains":
            return s_cond in s_val
        if cond.operator == ">":
            try:
                return float(val) > float(cond.value)
            except (ValueError, TypeError):
                return False
        if cond.operator == "<":
            try:
                return float(val) < float(cond.value)
            except (ValueError, TypeError):
                return False
        return False

    @classmethod
    def _create_unified_router(
        cls,
        workflow: WorkflowConfig,
        source_id: str,
        edges_list: list,
        has_tools: bool = False,
        fallback: str = END,
        error_target: str = None,
    ) -> Callable:
        """Helper to create a scoped router function for conditional edges."""

        def unified_router(state: AgentState):
            # 🔄 NEW: Always prioritize system failures first
            if error_target and state.get("error"):
                return "error"

            # 1. ALWAYS prioritize Tool Execution if tool calls are pending AND we have a target
            messages = state.get("messages", [])
            if has_tools and messages and cls._has_tool_calls(messages[-1]):
                last_msg = messages[-1]
                tool_calls = getattr(last_msg, "tool_calls", []) or []
                source_config = next(
                    (n for n in workflow.nodes if n.id == source_id), None
                )
                # Fix #10: delegate to shared helper instead of duplicating the loop
                if cls._check_tool_interrupt(tool_calls, source_config):
                    logger.info(
                        "Tool execution requires human approval. Routing to 'human_gate'"
                    )
                    return "human_gate"
                return "tools"  # Note: In _apply_conditional_group we ensured path_map has targets

            # 2. Evaluate Manual Conditions
            from Apps.AgentMesh.Core.Compiler.nodes.utils import get_nested_val, get_production_data
            data_scope = get_production_data(state)

            for e in edges_list:
                raw_key = e.condition.key or ""
                # Clean up jinja braces if user typed {{path}}
                clean_key = raw_key.replace("{{", "").replace("}}", "").strip()

                # A. Try central get_nested_val helper first
                val = get_nested_val(state, clean_key)

                # B. Try resolving against production data scope (context + state)
                if val is None:
                    val = get_nested_val(data_scope, clean_key)

                # C. Handle Jinja/Nodes prefix if nodes isn't a top-level key (e.g. nodes.llm_agent_sywex.context.msg)
                if val is None and clean_key.startswith("nodes."):
                    parts = clean_key.split(".", 2)
                    if len(parts) >= 3:
                        target_node_id = parts[1]
                        node_state = state.get("nodes", {}).get(target_node_id, {})
                        val = get_nested_val(node_state, parts[2])

                # SMART FALLBACK: If checking for 'intent', 'message', 'messages', or if val is STILL None, try last message content
                if val is None and messages:
                    val = cls._extract_content(messages[-1])

                # Extract content for message-based conditions
                if (
                    clean_key in ("messages", "message")
                    and isinstance(val, (list, tuple))
                    and val
                ):
                    val = cls._extract_content(val[-1])

                match_result = cls._evaluate_condition(e.condition, val)
                logger.info(
                    f"[ROUTER] Node '{source_id}' checking condition key='{raw_key}' (clean='{clean_key}') "
                    f"val='{val}' op='{e.condition.operator}' cond_val='{e.condition.value}' => match={match_result}"
                )

                if match_result:
                    target_id = END if e.target == "END" else e.target
                    logger.info(f"[ROUTER] Node '{source_id}' matched condition! Routing to target='{target_id}'")
                    return target_id

            # 3. Default Fallback (Static edge if any, otherwise END)
            target_id = END if fallback == "END" else fallback
            logger.info(f"[ROUTER] Node '{source_id}' no conditions matched. Routing to fallback='{target_id}'")
            return target_id

        unified_router.__name__ = f"router_{source_id}"
        return unified_router
