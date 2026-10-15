"""
Human Interaction Node Builder
Handles generic pauses/interrupts with structured configurations (e.g., followup forms, selections).
"""

import logging
from typing import Callable
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import ToolMessage
from langgraph.types import interrupt
from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState

logger = logging.getLogger(__name__)


def get_placeholder_msg(messages):
    for msg in reversed(messages):
        kwargs = msg.get("additional_kwargs", {}) if isinstance(msg, dict) else getattr(msg, "additional_kwargs", {})
        if kwargs.get("pause_config"):
            return msg
    return None


async def _execute_human_interaction(state: AgentState):
    messages = state.get("messages", [])
    original_msg = get_placeholder_msg(messages)
            
    if not original_msg:
        logger.warning("Human Interaction Node executed but no pause_config was found in messages.")
        return {}

    kwargs = original_msg.get("additional_kwargs", {}) if isinstance(original_msg, dict) else getattr(original_msg, "additional_kwargs", {})
    config = kwargs.get("pause_config")
    logger.info(f"Human Interaction Node: Interrupting with config: {config}")
    
    # Native interrupt with typed payload
    user_response = interrupt(config)
    
    logger.info(f"Human Interaction Node: Resumed with response: {user_response}")
    
    original_id = original_msg.get("id") if isinstance(original_msg, dict) else getattr(original_msg, "id", None)
    tool_call_id = original_msg.get("tool_call_id") if isinstance(original_msg, dict) else getattr(original_msg, "tool_call_id", "")
    name = original_msg.get("name") if isinstance(original_msg, dict) else getattr(original_msg, "name", "")
    
    new_msg = ToolMessage(
        id=original_id,
        content=str(user_response),
        tool_call_id=tool_call_id,
        name=name
    )
    
    import asyncio
    await asyncio.sleep(0)
    
    return {"messages": [new_msg]}


def build_human_interaction_node(node_config: NodeConfig) -> Callable:
    """Builds a human interaction node that triggers a native interrupt and replaces the placeholder ToolMessage upon resume."""
    logger.debug(f"Building human interaction node '{node_config.id}'")
    return RunnableLambda(_execute_human_interaction)


