import os
import re
import sys
import asyncio
import time
import logging
import subprocess
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

from Apps.AgentMesh.Config import settings

logger = logging.getLogger(__name__)

# Directory where ONNX models are stored
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
AUDIO_MOUNT = settings.AUDIO_MOUNT

# Virtual environment Scripts path for piper.exe (Windows)
def find_piper_path(start_dir):
    """Robustly find piper.exe by searching upwards from the start directory."""
    curr = start_dir
    # Limit search to 10 levels up
    for _ in range(10):
        # Case 1: .venv/Scripts/piper.exe
        potential = os.path.join(curr, ".venv", "Scripts", "piper.exe")
        if os.path.exists(potential):
            return potential
        # Case 2: Scripts/piper.exe (if already in .venv)
        potential = os.path.join(curr, "Scripts", "piper.exe")
        if os.path.exists(potential):
            return potential
        
        parent = os.path.dirname(curr)
        if parent == curr: break
        curr = parent
    return None

PIPER_PATH = find_piper_path(CURRENT_DIR)
if not PIPER_PATH:
    # Absolute fallback just in case the folder name isn't .venv
    # Or if it's on the PATH
    PIPER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR))))), ".venv", "Scripts", "piper.exe")

# Map of simplified voice names to their ONNX model filenames
VOICE_MODEL_MAP = {
    "en_US-amy": "en_US-amy-medium.onnx",
    "en_GB-alba": "en_GB-alba-medium.onnx",
    "zh_CN-huayan": "zh_CN-huayan-medium.onnx",
    "ru_RU-dmitri": "ru_RU-dmitri-medium.onnx",
    "hi_IN-priyamvada": "hi_IN-priyamvada-medium.onnx",
    "hi_IN-rohan": "hi_IN-rohan-medium.onnx",
    # Backward compatibility mappings
    "en-US-GuyNeural": "en_US-amy-medium.onnx",
    "en-US-JennyNeural": "en_US-amy-medium.onnx",
    "en-GB-SoniaNeural": "en_GB-alba-medium.onnx",
    "zh-CN-XiaoxiaoNeural": "zh_CN-huayan-medium.onnx"
}

def sanitize_filename_component(text: str) -> str:
    """Sanitizes strings to prevent path traversal."""
    if not text:
        return "unknown"
    sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(text))
    return sanitized if sanitized.strip('_') else "id"

def cleanup_old_audio(directory: str, max_age_hours: int = 24):
    """Cleanup audio files older than 24h (.wav and .mp3)."""
    if not os.path.exists(directory):
        return
    now = time.time()
    cutoff = now - (max_age_hours * 3600)
    deleted_count = 0
    try:
        for filename in os.listdir(directory):
            if filename.endswith(".wav") or filename.endswith(".mp3"):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.getmtime(file_path) < cutoff:
                        os.remove(file_path)
                        deleted_count += 1
                except Exception: pass
        if deleted_count > 0:
            logger.info(f"[AudioEngine] Storage Cleanup: Removed {deleted_count} expired files.")
    except Exception as e:
        logger.error(f"[AudioEngine] Cleanup failed: {e}")

_kittentts_instances = {}
KITTENTTS_VOICES = {"Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"}

def get_kittentts_model(model_name: str):
    global _kittentts_instances
    if model_name not in _kittentts_instances:
        logger.info(f"[AudioEngine] Loading KittenTTS model: {model_name}")
        from kittentts import KittenTTS
        _kittentts_instances[model_name] = KittenTTS(model_name)
    return _kittentts_instances[model_name]

def clean_text_for_speech(text: str) -> str:
    """Removes special characters for natural TTS output."""
    if not text: return ""
    text = re.sub(r'[*_#`~>\[\]\(\)]', '', text)
    return " ".join(text.split())


async def generate_speech(
    text: str, 
    voice: str = "en_US-amy", 
    thread_id: str = "default", 
    node_id: str = "tts_node",
    tts_library: str = "piper",
    kittentts_model: str = "KittenML/kitten-tts-micro-0.8",
    speed: float = 1.0
) -> Optional[str]:
    """
    Converts text to speech using selected TTS library (Piper or KittenTTS).
    Saves to .wav.
    """
    if not text.strip():
        return None

    s_thread_id = sanitize_filename_component(thread_id)
    session_dir = os.path.abspath(os.path.join(settings.TEMP_MOUNT, s_thread_id))
    os.makedirs(session_dir, exist_ok=True)

    # Prepare Path naming
    s_node_id = sanitize_filename_component(node_id)
    prefix = f"thread_{s_thread_id}_node_{s_node_id}_"
    
    # Delete previous audio files in this session folder to ensure only one audio remains
    try:
        if os.path.exists(session_dir):
            for f in os.listdir(session_dir):
                if f.endswith(".wav") or f.endswith(".mp3"):
                    os.remove(os.path.join(session_dir, f))
    except Exception: pass

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    safe_filename = f"{prefix}{timestamp}.wav"
    full_path = os.path.join(session_dir, safe_filename)

    # Clean text
    if tts_library == "kittentts":
        try:
            from kittentts import normalize_text
            cleaned_content = normalize_text(text)
        except Exception as e:
            logger.warning(f"[AudioEngine] normalize_text failed, using fallback cleaning: {e}")
            cleaned_content = clean_text_for_speech(text)
    else:
        cleaned_content = clean_text_for_speech(text)

    if tts_library == "kittentts":
        try:
            logger.info(f"[AudioEngine] Generating speech using KittenTTS model: {kittentts_model}, voice: {voice}, speed: {speed}")
            # Ensure voice is a valid KittenTTS voice
            valid_voice = voice
            if valid_voice not in KITTENTTS_VOICES:
                # Default to Bella
                valid_voice = "Bella"

            model = get_kittentts_model(kittentts_model)
            
            def run_kittentts():
                if hasattr(model, 'generate_to_file'):
                    model.generate_to_file(cleaned_content, full_path, voice=valid_voice, speed=speed)
                else:
                    import soundfile as sf
                    audio = model.generate(text=cleaned_content, voice=valid_voice, speed=speed)
                    sf.write(full_path, audio, 24000)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, run_kittentts)
            logger.info(f"[AudioEngine] KittenTTS success: {full_path}")
            return f"/temp/{s_thread_id}/{safe_filename}"
        except Exception as e:
            logger.error(f"[AudioEngine] KittenTTS generation failed: {type(e).__name__}: {e}", exc_info=True)
            return None
    else:
        # Default fallback to Piper
        # 1. Resolve Voice Model
        default_model = "en_US-amy-medium.onnx"
        model_filename = VOICE_MODEL_MAP.get(voice, default_model)
        model_path = os.path.join(MODELS_DIR, str(model_filename))
        
        if not os.path.exists(model_path):
            logger.error(f"[AudioEngine] Model not found: {model_path}. Please download voices.")
            # Fallback to English if specifically requested failed
            if voice != "en_US-amy":
                 model_path = os.path.join(MODELS_DIR, default_model)

        # Generate audio via Piper CLI
        try:
            cmd = [
                sys.executable,
                "-m", "piper",
                "-m", model_path,
                "-f", full_path,
                "--",
                cleaned_content
            ]
            
            logger.debug(f"[AudioEngine] Running piper for voice: {voice}")
            
            # Windows compatibility: use executor instead of create_subprocess_exec
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(cmd, capture_output=True)
            )
            
            if result.returncode == 0:
                logger.info(f"[AudioEngine] Local success: {full_path}")
                return f"/temp/{s_thread_id}/{safe_filename}"
            else:
                stderr_msg = result.stderr.decode() if result.stderr else "(no error output)"
                logger.error(f"[AudioEngine] Piper error (code {result.returncode}): {stderr_msg}")
                return None
        except Exception as e:
            logger.error(f"[AudioEngine] TTS generation failed: {type(e).__name__}: {e}", exc_info=True)
            return None
