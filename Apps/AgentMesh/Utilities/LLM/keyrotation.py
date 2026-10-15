"""
API Key Rotation Module.
Loads API keys from environment variables (comma-separated) and provides
round-robin rotation functions.
"""
import os
import random

# Load keys from environment (comma-separated)
def _load_keys(env_var: str) -> list:
    raw = os.getenv(env_var, "")
    return [k.strip() for k in raw.split(",") if k.strip()]

GROQ_API_KEYS = _load_keys("GROQ_API_KEYS")
Mistral_API_KEYS = _load_keys("MISTRAL_API_KEYS")
TOGETHER_API_KEYS = _load_keys("TOGETHER_API_KEYS")

# Round-robin indices
_groq_index = 0
_mistral_index = 0
_together_index = 0

def get_random_groq_api_key():
    global _groq_index
    if not GROQ_API_KEYS:
        return os.getenv("GROQ_API_KEY", "")
    key = GROQ_API_KEYS[_groq_index]
    _groq_index = (_groq_index + 1) % len(GROQ_API_KEYS)
    return key

def get_random_mistral_api_key():
    global _mistral_index
    if not Mistral_API_KEYS:
        return os.getenv("MISTRAL_API_KEY", "")
    key = Mistral_API_KEYS[_mistral_index]
    _mistral_index = (_mistral_index + 1) % len(Mistral_API_KEYS)
    return key

def get_random_together_api_key():
    global _together_index
    if not TOGETHER_API_KEYS:
        return os.getenv("TOGETHER_API_KEY", "")
    key = TOGETHER_API_KEYS[_together_index]
    _together_index = (_together_index + 1) % len(TOGETHER_API_KEYS)
    return key

def GroqVisionModel():
    return get_random_groq_api_key()