"""
Subgraph Node Builder
Handles nested workflow execution.
"""

import logging
from typing import Callable

from langchain_core.runnables import RunnableLambda, RunnableConfig
from langgraph.types import interrupt

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState

logger = logging.getLogger(__name__)


def build_subgraph_node(node_config: NodeConfig) -> Callable:
    """
    Build a subgraph node that executes a nested workflow.
    Prevents message duplication by tracking original count.
    """
    # Late import to avoid circular dependency
    from Apps.AgentMesh.Core.WorkflowRegistry import workflow_registry
    
    async def execute_subgraph(state: AgentState, config: RunnableConfig = None):
        from Database.Database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            app = await workflow_registry.get_executable_graph(node_config.workflow_id, db=db)

        if not app:
            raise ValueError(
                f"Target subgraph workflow_id '{node_config.workflow_id}' not found in registry. "
                "Ensure this workflow is compiled (run it once or set warmup=True) before use as a subgraph."
            )

        # Derive a child thread ID to isolate nested checkpoint persistence
        parent_thread_id = "default_thread"
        if config and config.get("configurable"):
            parent_thread_id = config["configurable"].get("thread_id", "default_thread")
        
        child_thread_id = f"{parent_thread_id}_sub_{node_config.id}"
        
        sub_config = {
            "configurable": {
                "thread_id": child_thread_id
            },
            "callbacks": config.get("callbacks", []) if config else []
        }

        # Check if child workflow is already paused in a checkpoint
        snapshot = await app.aget_state(sub_config)
        child_active_agent = snapshot.values.get("active_agent") if snapshot else None
        
        logger.info(f"[SUBGRAPH DEBUG] snapshot.next for child {node_config.workflow_id} is: {snapshot.next if snapshot else None}")
        
        from langgraph.errors import GraphInterrupt
        
        try:
            if snapshot.next:
                # Child is paused. Resume it by propagating context and user_input changes from parent.
                parent_context = state.get("context", {})
                child_context_update = {}
                for k, v in parent_context.items():
                    if k not in ["workflow_id", "workflow_name", "thread_id"]:
                        child_context_update[k] = v
                
                child_user_input = state.get("user_input") if isinstance(state, dict) else getattr(state, "user_input", {})
                
                logger.info(f"[SUBGRAPH DEBUG] Resuming child with context: {child_context_update}")
                await app.aupdate_state(sub_config, {
                    "context": child_context_update,
                    "user_input": child_user_input
                })
                result_state = await app.ainvoke(None, config=sub_config)
            else:
                # Fresh execution of subgraph
                # Preserve child's own active_agent for dynamic resumption if it exists in checkpoint values
                child_run_state = {
                    **state,
                    "active_agent": child_active_agent if child_active_agent else state.get("active_agent")
                }
                logger.info(f"[SUBGRAPH DEBUG] Starting fresh execution of child with active_agent: {child_run_state.get('active_agent')}")
                result_state = await app.ainvoke(child_run_state, config=sub_config)
        except GraphInterrupt as e:
            logger.info(f"[SUBGRAPH DEBUG] Caught GraphInterrupt during child execution: {str(e)}")
            result_state = {}

        # Check if the child workflow paused during this execution
        new_snapshot = await app.aget_state(sub_config)
        logger.info(f"[SUBGRAPH DEBUG] New snapshot next: {new_snapshot.next if new_snapshot else None}")
        if new_snapshot.next:
            # Child workflow is paused! We must propagate the pause to the parent.
            child_paused_node_id = new_snapshot.next[0]
            clean_child_node_id = child_paused_node_id.replace("pre_gate_", "").replace("post_gate_", "")
            
            # Fetch the child workflow configuration to get the node's type and details
            child_config = workflow_registry.get_config(node_config.workflow_id)
            child_node = next((n for n in child_config.nodes if n.id == clean_child_node_id), None) if child_config else None
            child_node_type = getattr(child_node, "type", None) if child_node else None
            child_node_config = child_node.model_dump() if child_node and hasattr(child_node, "model_dump") else (child_node.__dict__ if child_node else {})
            
            # Identify any outgoing conditional choice buttons in the child graph
            paused_subgraph_buttons = []
            if child_config:
                child_edges = child_config.edges or []
                outgoing_to_buttons = [e for e in child_edges if e.source == clean_child_node_id]
                for edge in outgoing_to_buttons:
                    target_node = next((n for n in child_config.nodes if n.id == edge.target), None)
                    if target_node and target_node.type == "button_trigger":
                        paused_subgraph_buttons.append({
                            "id": target_node.id,
                            "label": getattr(target_node, "button_label", None) or getattr(target_node, "label", None) or "Proceed",
                            "button_category": getattr(target_node, "button_category", None) or "primary"
                        })
            
            interrupt(f"Child subgraph paused at node: {clean_child_node_id}")

        # Prevent message duplication: return only new messages
        original_msg_count = len(state.get("messages", []))
        new_messages = result_state.get("messages", [])[original_msg_count:]

        update = {"active_agent": node_config.id}
        # Clear subgraph pause context from parent context once successfully finished
        parent_context = state.get("context", {})
        if "paused_subgraph_node_id" in parent_context:
            cleaned_context = {k: v for k, v in parent_context.items() if not k.startswith("paused_subgraph_")}
            update["context"] = cleaned_context

        if new_messages:
            update["messages"] = new_messages

        logger.debug(f"Subgraph {node_config.id}: Returned {len(new_messages)} new message(s)")
        return update
    
    return RunnableLambda(execute_subgraph)
