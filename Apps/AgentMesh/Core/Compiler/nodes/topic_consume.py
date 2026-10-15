"""
Topic Consume Node
Consumes events from a specified topic in the Event Backbone.
"""
import logging
from typing import Callable

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from .topic_publish import EVENT_BROKER

logger = logging.getLogger(__name__)

def build_topic_consume_node(node_config: NodeConfig) -> Callable:
    """Build a node that consumes payloads from a specific topic."""
    
    async def execute(state: AgentState):
        context = state.get("context", {})
        
        topic_name = node_config.topic_name or "default_topic"
        
        consumed_data = None
        status = "EMPTY_TOPIC"
        
        if topic_name in EVENT_BROKER and len(EVENT_BROKER[topic_name]) > 0:
            # Consume the oldest event (FIFO queue behavior)
            consumed_data = EVENT_BROKER[topic_name].pop(0)
            status = "SUCCESS"
            msg = f"Consumed event from topic '{topic_name}'."
        else:
            msg = f"No events found in topic '{topic_name}'."
            
        logger.info(f"Topic Consume Node {node_config.id}: {msg}")
        
        return {
            "messages": [AIMessage(content=msg, response_metadata={"active_agent": node_config.id})],
            "active_agent": node_config.id,
            "context": {
                **context,
                f"api_result_{node_config.id}": {"status": status, "topic": topic_name, "data": consumed_data}
            }
        }
        
    return RunnableLambda(execute)
