"""
Debug Console Node Builder
Echoes the internal state, context, or specific variables back to the chat for debugging.
"""

import json
import logging
from typing import Callable
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState

from .utils import process_tools_with_overrides, interpolate, get_production_data

logger = logging.getLogger(__name__)

def build_debug_node(node_config: NodeConfig) -> Callable:
    """
    Build a Debug Console node.
    Formats and prints specified parts of the AgentState to the chat history.
    """
    async def execute_debug(state: AgentState):
        messages = state.get("messages", [])
        
        # Determine what to print based on node configuration
        # debug_scope: 'context' (default), 'messages', 'full_state', 'selective', 'evaluation'
        scope = getattr(node_config, "debug_scope", "context")
        target_key = getattr(node_config, "debug_key", None)
        
        data_to_print = {}
        title = "ROOT CONTEXT"

        if scope == "full_state":
            data_to_print = {k: v for k, v in state.items() if k != "messages"}
            title = "FULL AGENT STATE"
        elif scope == "messages" and messages:
            data_to_print = [
                {"role": type(m).__name__, "content": getattr(m, "content", str(m))} 
                for m in messages[-5:] # Show last 5 for brevity
            ]
            title = "LATEST MESSAGES (Last 5)"
        elif scope == "evaluation":
            data_to_print = get_production_data(state)
            title = "EVALUATION SCOPE (Full Data View)"
        elif scope == "selective" and target_key:
            full_scope = get_production_data(state)
            # Support nested keys via dot notation (e.g., nodes.set_data_1.user)
            from .utils import get_nested_val
            data_to_print = get_nested_val(full_scope, target_key)
            if data_to_print is None:
                data_to_print = f"Key '{target_key}' not found in available scope."
            title = f"VARIABLE: {target_key}"
        else:
            # Default: Improved Context (matches what logic nodes see)
            full_scope = get_production_data(state)
            # Filter out internal/system keys and tech-reserved ones for clean view
            data_to_print = {k: v for k, v in full_scope.items() if not str(k).startswith("__") and k not in ("msg", "data")}
            title = "ACTIVE WORKFLOW DATA (Context + Latest Result)"

        # Format as a pretty Markdown block
        try:
            json_str = json.dumps(data_to_print, indent=2, default=str)
        except Exception as e:
            json_str = f"Error serializing state: {str(e)}"

        debug_content = f"### 🛠️ DEBUG CONSOLE: {node_config.label or node_config.id}\n"
        debug_content += f"**{title}**\n\n"
        debug_content += f"```json\n{json_str}\n```"

        logger.info(f"Debug Node {node_config.id}: Outputting state snapshot to chat.")

        return {
            "messages": [AIMessage(content=debug_content, response_metadata={"active_agent": node_config.id})],
            "active_agent": node_config.id
        }

    return RunnableLambda(execute_debug)
