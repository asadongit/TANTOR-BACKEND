from langchain_core.language_models.chat_models import BaseChatModel
from Apps.AgentMesh.Schema.WorkflowSchema import LLMConfig
from Apps.AgentMesh.Utilities.LLM.llm_config import get_llm_provider, get_reasoning_llm

def init_chat_model(config: LLMConfig = None) -> BaseChatModel:
    """
    Factory function to initialize a LangChain Chat model based on the UI's NodeConfig.
    Delegates to the existing llm_config.py infrastructure.
    """
    if not config:
        # Fallback to the default environment LLM
        return get_llm_provider()

    # Base parameters
    params = {
        "temperature": config.temperature,
    }

    # Pass custom base_url if provided (for Ollama, SGLang, vLLM, self-hosted, etc.)
    if config.base_url:
        params["base_url"] = config.base_url

    # Check for per-node api_key_override (decrypted securely in AgentMeshCrud)
    if config.api_key_override:
        params["api_key"] = config.api_key_override

    # How much the model may write. Providers cap this by default — Groq allows
    # 2-3k tokens — and a reasoning model spends the same budget thinking, so a
    # node asked for a large structured answer over a dense document can hit the
    # cap mid-thought and return an empty reply with no error. Unset leaves the
    # provider's default alone.
    if config.max_tokens:
        params["max_tokens"] = config.max_tokens

    # Where the model's working goes. Set independently of effort, because the
    # common case is wanting a model to reason as it normally would while
    # keeping that reasoning out of the reply.
    if config.reasoning_format:
        params["reasoning_format"] = config.reasoning_format

    # If it's a reasoning model (o1, etc.), use the reasoning builder
    if config.reasoning_effort:
        params["reasoning_effort"] = config.reasoning_effort
        return get_reasoning_llm(
            provider=config.provider,
            model=config.model,
            **params
        )

    # Otherwise, use the standard chat builder
    return get_llm_provider(
        provider=config.provider,
        model=config.model,
        **params
    )
