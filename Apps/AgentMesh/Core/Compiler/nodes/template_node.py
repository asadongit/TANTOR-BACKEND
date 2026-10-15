"""
Template Engine Node Builder
Renders Jinja2 or Markdown templates using variables from the global AgentState.
"""

import logging
import os
from typing import Callable

from langchain_core.messages import AIMessage
from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState

logger = logging.getLogger(__name__)

def build_template_node(node_config: NodeConfig) -> Callable:
    """
    Builds a template engine node.
    """
    # Extract config parameters
    template_content = getattr(node_config, "template_content", None) or ""
    template_file_path = getattr(node_config, "template_file_path", None) or ""
    template_output_key = getattr(node_config, "template_output_key", None) or "rendered_template"
    template_data_key = getattr(node_config, "template_data_key", None) or ""

    async def template_node(state: AgentState):
        logger.info(f"[{node_config.id}] Executing Template Engine")
        print(f"DEBUG TEMPLATE: config={vars(node_config)}")
        
        state_dict = state if isinstance(state, dict) else vars(state)
        
        # Determine the data to pass to the template
        # 1. Start with full context
        data = {}
        if "context" in state_dict and isinstance(state_dict["context"], dict):
            data.update(state_dict["context"])
            
        # Additional global vars if needed
        data["__system__"] = "AgentMesh Template Context"
            
        # Extract the latest user message
        messages = state_dict.get("messages", [])
        latest_message = ""
        for msg in reversed(messages):
            # Handle BaseMessage objects
            if hasattr(msg, "type") and msg.type == "human" or getattr(msg, "__class__", None).__name__ == "HumanMessage":
                latest_message = getattr(msg, "content", "")
                break
            # Handle raw dicts (if deserialized from JSON)
            elif isinstance(msg, dict) and (msg.get("type") == "human" or msg.get("id", "").startswith("human")):
                latest_message = msg.get("content", "")
                break
                
        data["latest_message"] = latest_message
        
        # 3. Handle specific Input Variable Field (template_data_key)
        from Apps.AgentMesh.Core.Compiler.nodes.utils import get_production_data, get_nested_val
        import json
        
        if template_data_key and template_data_key.strip():
            scope = get_production_data(state)
            val = get_nested_val(scope, template_data_key.strip())
            
            # If they pointed to a dictionary, merge it into the root variables!
            if isinstance(val, dict):
                data.update(val)
                data["user_input"] = json.dumps(val, default=str)
            # If they pointed to a string/primitive, make it available as user_input
            elif val is not None:
                data["user_input"] = str(val)
        else:
            data["user_input"] = latest_message

        # Resolve template string
        content_to_render = template_content
        if template_file_path and os.path.exists(template_file_path):
            try:
                with open(template_file_path, "r", encoding="utf-8") as f:
                    content_to_render = f.read()
            except Exception as e:
                logger.error(f"[{node_config.id}] Failed to read template file: {e}")
                content_to_render = f"Error reading template file: {e}"

        # Render using secure Sandboxed interpolate() from utils.py
        try:
            from Apps.AgentMesh.Core.Compiler.nodes.utils import interpolate
            rendered_output = interpolate(content_to_render, data)
        except Exception as e:
            logger.error(f"[{node_config.id}] Template rendering failed: {e}")
            rendered_output = f"Template rendering error: {e}"

        # Save the rendered output into the context
        context = state_dict.get("context", {})
        if not isinstance(context, dict):
            context = {}
            
        context[template_output_key] = rendered_output

        # Also append it as an AIMessage so it can be seen in logs/chat
        new_message = AIMessage(content=rendered_output, name=node_config.id, response_metadata={"active_agent": node_config.id})

        return {
            "messages": [new_message],
            "context": context,
            "active_agent": node_config.id
        }

    return template_node
