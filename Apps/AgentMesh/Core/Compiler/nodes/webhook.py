import logging
from typing import Dict, Any
from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState

logger = logging.getLogger(__name__)

def build_webhook_response_node(node_config: NodeConfig):
    """
    Builds a node that sets custom webhook response data in the context.
    The universal_webhook endpoint looks for '__webhook_response__' in context.
    """
    
    async def webhook_response_node(state: AgentState) -> Dict[str, Any]:
        logger.debug(f"Executing Webhook Response Node: {node_config.id}")
        
        context = state.get("context", {})
        
        # Determine body
        body = None
        if node_config.response_body_key:
            # Try to get from context first
            body = context.get(node_config.response_body_key)
            if body is None:
                # Fallback to state keys (messages, etc) if not in context
                body = state.get(node_config.response_body_key)
        else:
            # Default to full context if no key specified
            body = context
            
        # Prepare response data
        response_data = {
            "status": node_config.response_status,
            "headers": node_config.response_headers or {},
            "body": body
        }
        
        # Update context with special key
        new_context = {**context, "__webhook_response__": response_data}
        
        return {"context": new_context}
        
    return webhook_response_node
