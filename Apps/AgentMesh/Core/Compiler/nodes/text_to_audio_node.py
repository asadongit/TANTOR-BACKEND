"""
Text-to-Audio Node Builder
Handles conversion of text to audio/speech.
"""

import logging
from datetime import datetime
from typing import Callable

from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Utilities.Audio.tts_helper import generate_speech
from .utils import node_id_to_key

logger = logging.getLogger(__name__)


def build_text_to_audio_node(node_config: NodeConfig) -> Callable:
    """
    Build a text-to-audio node that converts message content to speech.
    Supports voice selection and mute option.
    """
    async def execute_tts(state: AgentState):
        messages = state.get("messages", [])
        thread_id = state.get("thread_id", "default_thread")
        
        # Extract text: check if custom audio_input_text is provided, otherwise fall back to the last message
        custom_input = getattr(node_config, "audio_input_text", None)
        if custom_input:
            from .utils import get_production_data, interpolate
            scope = get_production_data(state)
            text = interpolate(custom_input, scope)
            logger.debug(f"TTS node {node_config.id}: Resolved custom text from audio_input_text")
        else:
            if not messages:
                logger.warning(f"TTS node {node_config.id}: No messages in state")
                return {"active_agent": node_config.id}
            last_msg = messages[-1]
            text = getattr(last_msg, "content", str(last_msg))
        
        logger.debug(f"TTS node {node_config.id}: Processing text ({len(text or '')} chars)")
        
        # Check for mute
        if node_config.mute_audio:
            logger.debug(f"TTS node {node_config.id}: Audio is muted")
            return {"active_agent": node_config.id}
        
        # Generate speech
        selected_voice = node_config.audio_voice or "en-US-GuyNeural"
        library = getattr(node_config, "tts_library", "piper")
        model_id = getattr(node_config, "kittentts_model", "KittenML/kitten-tts-micro-0.8")
        speed = getattr(node_config, "audio_speed", 1.0)
        logger.info(f"TTS node {node_config.id}: Generating speech using {library} with voice: {selected_voice} (speed: {speed})")
        
        audio_file = await generate_speech(
            text=text,
            voice=selected_voice,
            thread_id=thread_id,
            node_id=node_config.id,
            tts_library=library,
            kittentts_model=model_id,
            speed=speed
        )
        
        if not audio_file:
            logger.warning(f"TTS node {node_config.id}: Failed to generate speech")
            return {"active_agent": node_config.id}
        
        # Update context with audio metadata
        context = state.get("context", {})
        audio_history = context.get("audio_segments", {})
        audio_history[node_id_to_key(node_config.id)] = {
            "file": audio_file,
            "timestamp": datetime.now().isoformat(),
            "node_label": node_config.label or node_config.id
        }
        
        logger.debug(f"TTS node {node_config.id}: Audio generated: {audio_file}")
        
        return {
            "active_agent": node_config.id,
            "context": {
                **context,
                "audio_segments": audio_history,
                "latest_audio": [{
                    "file": audio_file,
                    "timestamp": datetime.now().isoformat(),
                    "node": node_config.label or node_config.id
                }]
            }
        }
    
    return RunnableLambda(execute_tts)
