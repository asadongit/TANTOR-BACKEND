"""
Topic Publish Node
Publishes an event payload to a specified topic in the Event Backbone.
"""
import json
import logging
from typing import Callable, Dict, List

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from .utils import interpolate

logger = logging.getLogger(__name__)

# Simulated In-Memory Event Broker (In production, this would be Kafka/Redis)
EVENT_BROKER: Dict[str, List[dict]] = {}

def build_topic_publish_node(node_config: NodeConfig) -> Callable:
    """Build a node that publishes payloads to a specific topic."""
    
    async def execute(state: AgentState):
        context = state.get("context", {})
        
        topic_name = node_config.topic_name or "default_topic"
        raw_payload = node_config.topic_payload or "{}"
        
        # Interpolate the payload using context
        interpolated_payload = interpolate(raw_payload, context)
        
        # Try to parse the payload as JSON if it's a string, otherwise just store it
        payload_data = interpolated_payload
        if isinstance(interpolated_payload, str):
            try:
                payload_data = json.loads(interpolated_payload)
            except json.JSONDecodeError:
                pass # Just store it as a string
                
        # Initialize topic if it doesn't exist
        if topic_name not in EVENT_BROKER:
            EVENT_BROKER[topic_name] = []
            
        EVENT_BROKER[topic_name].append(payload_data)
        
        msg = f"Published to topic '{topic_name}' successfully."
        logger.info(f"Topic Publish Node {node_config.id}: {msg}")
        
        return {
            "messages": [AIMessage(content=msg, response_metadata={"active_agent": node_config.id})],
            "active_agent": node_config.id,
            "context": {
                **context,
                f"api_result_{node_config.id}": {"status": "SUCCESS", "topic": topic_name, "published_data": payload_data}
            }
        }
        
    return RunnableLambda(execute)
