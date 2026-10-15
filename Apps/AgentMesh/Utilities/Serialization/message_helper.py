from typing import Any, Dict, List
import json
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage

def serialize_message(m: Any) -> Dict[str, Any]:
    """
    Standard serializer for LangChain messages to be sent to the frontend.
    Handles ToolCalls and metadata.
    """
    content = m.content if hasattr(m, 'content') else str(m)
    tool_calls = None
    if isinstance(m, AIMessage) and m.tool_calls:
        tool_calls = [{
            "id": tc.get("id"),
            "function": {"name": tc.get("name"), "arguments": json.dumps(tc.get("args", {}))},
            "type": "function"
        } for tc in m.tool_calls]
    
    # ✅ FIX (Mar 20, 2026): Preserve ALL metadata, not just tool calls
    metadata = {}
    if hasattr(m, 'metadata'):
        metadata.update(m.metadata or {})
    if hasattr(m, 'response_metadata'):
        metadata.update(m.response_metadata or {})
    if hasattr(m, 'additional_kwargs'):
        metadata.update(m.additional_kwargs or {})
    
    # Ensure tool_calls are correctly formatted for the frontend
    if tool_calls:
        metadata["tool_calls"] = tool_calls

    active_agent = metadata.get("active_agent")
    if active_agent:
        try:
            from Apps.AgentMesh.Core.WorkflowRegistry import workflow_registry
            node_meta = workflow_registry.get_node_metadata(active_agent)
            if node_meta:
                metadata["agent_name"] = node_meta.get("label") or active_agent
        except Exception:
            pass
    
    return {
        "type": type(m).__name__,
        "content": content,
        "metadata": metadata
    }

def serialize_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    return [serialize_message(m) for m in messages]
