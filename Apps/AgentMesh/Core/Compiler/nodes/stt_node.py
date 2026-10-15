"""
STT (Speech-to-Text) Trigger Node Builder
Captures voice input from the user via browser SpeechRecognition API.
Acts as a pass-through similar to form_node.
"""

import logging
from typing import Callable
from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState

logger = logging.getLogger(__name__)


def build_stt_trigger(node_config: NodeConfig) -> Callable:
    """
    Build an STT trigger node.
    Pass-through node — the frontend captures audio via the browser's
    SpeechRecognition API and sends the transcript back on resume.
    """

    async def stt_pass(state: AgentState):
        logger.debug(f"STT node {node_config.id} active: Waiting for voice input.")

        user_data = (
            state.get("user_input")
            if isinstance(state, dict)
            else getattr(state, "user_input", {})
        ) or {}

        # Extract transcription from user_input, or fall back to the latest message
        # (This handles the case where stt_trigger is the starting node of the workflow)
        stt_output = user_data.get("stt_output")
        if not stt_output:
            messages = state.get("messages", [])
            for msg in reversed(messages):
                content = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
                if content and isinstance(content, str) and content.strip():
                    stt_output = content.strip()
                    break

        return {
            "active_agent": node_config.id,
            "stt_output": stt_output,
            "context": {
                "stt_output": stt_output,
                **(user_data or {})
            }
        }

    return stt_pass
