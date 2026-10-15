"""
Legacy global LLM configuration module.
Delegates cleanly to the new unified LiteLLM gateway in Apps/AgentMesh/Utilities/LLM/llm_config.py
to maintain complete backward compatibility and prevent duplicate client initialization.
"""

from Apps.AgentMesh.Utilities.LLM.llm_config import (
    langchain_llm,
    langchainVision_llm,
    langchain_llm_thinking,
    get_langchain_llm_oss
)