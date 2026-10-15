"""
Router Node Builder
Handles routing based on message content and routing keys.
"""

import logging
from typing import Callable

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState

logger = logging.getLogger(__name__)


from langchain_core.runnables import RunnableLambda

def build_router_node(node_config: NodeConfig) -> Callable:
    """
    Build a router node that extracts routing intent from the last message.
    Supports dynamic routing based on message content.
    """
    async def route_func(state: AgentState):
        update = {"active_agent": node_config.id}
        
        if node_config.routing_key:
            messages = state.get("messages", [])
            if messages:
                last_content = getattr(messages[-1], "content", str(messages[-1]))
                if isinstance(last_content, str):
                    fresh_intent = last_content.strip().lower()
                elif isinstance(last_content, list) and last_content:
                    # Handle multimodal messages
                    first_part = last_content[0]
                    fresh_intent = (
                        first_part.get("text", "") 
                        if isinstance(first_part, dict) 
                        else str(first_part)
                    ).strip().lower()
                else:
                    fresh_intent = str(last_content).strip().lower()
                
                update[node_config.routing_key] = fresh_intent
                logger.debug(f"Router {node_config.id}: Set '{node_config.routing_key}' = '{fresh_intent}'")
        
        return update
    
    return RunnableLambda(route_func)
