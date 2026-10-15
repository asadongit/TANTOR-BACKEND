"""Environment-backed API-key rotation for legacy import paths."""

import os


def _load_keys(env_var: str) -> list[str]:
    raw = os.getenv(env_var, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


GROQ_API_KEYS = _load_keys("GROQ_API_KEYS")
Mistral_API_KEYS = _load_keys("MISTRAL_API_KEYS")
TOGETHER_API_KEYS = _load_keys("TOGETHER_API_KEYS")

_groq_index = 0
_mistral_index = 0
_together_index = 0


def get_random_groq_api_key() -> str:
    global _groq_index
    if not GROQ_API_KEYS:
        return os.getenv("GROQ_API_KEY", "")
    key = GROQ_API_KEYS[_groq_index]
    _groq_index = (_groq_index + 1) % len(GROQ_API_KEYS)
    return key


def get_random_mistral_api_key() -> str:
    global _mistral_index
    if not Mistral_API_KEYS:
        return os.getenv("MISTRAL_API_KEY", "")
    key = Mistral_API_KEYS[_mistral_index]
    _mistral_index = (_mistral_index + 1) % len(Mistral_API_KEYS)
    return key


def get_random_together_api_key() -> str:
    global _together_index
    if not TOGETHER_API_KEYS:
        return os.getenv("TOGETHER_API_KEY", "")
    key = TOGETHER_API_KEYS[_together_index]
    _together_index = (_together_index + 1) % len(TOGETHER_API_KEYS)
    return key


def GroqVisionModel() -> str:
    return get_random_groq_api_key()
