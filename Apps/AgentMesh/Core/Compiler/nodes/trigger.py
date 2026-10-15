"""
Trigger Node Builder
Handles trigger/entry point nodes (start, webhook, cron, button_trigger, end).
"""

import logging
from typing import Callable

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState

logger = logging.getLogger(__name__)


def build_trigger_node(node_config: NodeConfig) -> Callable:
    """
    Build a passive trigger node that acts as a pass-through entry point.
    Used for start, webhook, cron, button_trigger, and end trigger types.
    """
    async def trigger_pass(state: AgentState):
        logger.debug(f"Trigger node {node_config.id} ({node_config.type}): Pass-through activated")
        
        # Build standard system return
        result = {"active_agent": node_config.id}
        
        # 🕵️ Data Explorer Persistence:
        # If this is a trigger node, capture the seeded input payload from context 
        # so it gets logged under this specific node inside the Data Explorer.
        if node_config.type in ("polling", "webhook", "cron", "button_trigger", "start", "auto_trigger"):
            context = state.get("context", {}) if isinstance(state, dict) else getattr(state, "context", {})
            if isinstance(context, dict):
                # Dynamically scan the context for seeded variables (excluding general metadata/system keys)
                SYSTEM_KEYS = {"thread_id", "workflow_id", "workflow_name", "vfs", "nodes", "source"}
                trigger_data = {
                    k: v for k, v in context.items()
                    if k not in SYSTEM_KEYS and not str(k).startswith("__")
                }
                if trigger_data:
                    # Merge trigger data into result context so wrap_with_error_handling registers it
                    result["context"] = trigger_data
        
        return result
    
    return trigger_pass
