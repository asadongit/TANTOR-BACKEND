"""
Form Submission Node Builder
Handles nodes that request structured user input via a form.
Acts as a pass-through that is usually preceded by a compiler-injected interrupt.
"""

import logging
from typing import Callable
from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState

logger = logging.getLogger(__name__)

def build_form_node(node_config: NodeConfig) -> Callable:
    """
    Build a form submission node.
    This node itself is a pass-through because the actual 'wait' and 'form data injection'
    is handled by the Gate/Interrupt mechanism and the user resuming the workflow.
    """
    async def form_pass(state: AgentState):
        logger.debug(f"Form node {node_config.id} active: Form schema ready for frontend.")
        
        # 📥 DATA CAPTURE: Extract user_input from the state.
        # When a user submits a form, the 'user_input' is injected into the global state.
        # By returning it here, we ensure it's recorded in the 'nodes' history for this specific node.
        user_data = state.get("user_input") if isinstance(state, dict) else getattr(state, "user_input", {})
        
        # We ensure active_agent is set so the frontend knows which node is 'interrupting'
        return {
            "active_agent": node_config.id,
            "context": {**(user_data or {})}
        }
    
    return form_pass
