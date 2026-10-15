"""
AgentMesh LLM Configuration Module
This module configures LangChain LLM instances using LiteLLM:
- LLM Agnostic: Support multiple providers via LiteLLM completion
- Provider-Agnostic: Switch providers dynamically
- Key Rotation: Supported dynamically and fallbacks configured
- Caching: Integrated in-memory and Redis caching
"""

import os
import time
from typing import Optional, List, Dict, Any
from pydantic import Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
import litellm
from litellm import cache

from Apps.AgentMesh.Config import settings
from .keyrotation import get_random_groq_api_key, get_random_mistral_api_key, get_random_together_api_key, GroqVisionModel

# Suppress verbose LiteLLM debug logs in the terminal
import logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
litellm.set_verbose = False


# ==================== Global Cache Setup ====================
try:
    if os.getenv("REDIS_URL") or os.getenv("REDIS_HOST"):
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        redis_password = os.getenv("REDIS_PASSWORD", None)
        litellm.cache = cache.Cache(
            type="redis",
            host=redis_host,
            port=redis_port,
            password=redis_password,
        )
    else:
        litellm.cache = cache.Cache()  # Default in-memory cache
except Exception:
    pass


# ==================== Custom LiteLLM LangChain Class ====================

class AgentMeshLiteLLM(BaseChatModel):
    model: str
    provider: str
    temperature: float = 0.1
    max_tokens: int = 2000
    api_key: Optional[str] = Field(default=None, repr=False)
    base_url: Optional[str] = None
    reasoning_effort: Optional[str] = None
    # Where a reasoning model's working is sent. Some providers return it inside
    # the answer, which ruins any step expecting the answer to be only what was
    # asked for — a JSON object, a single value. 'hidden' keeps it out.
    reasoning_format: Optional[str] = None

    model_config = {
        "arbitrary_types_allowed": True
    }

    @property
    def _llm_type(self) -> str:
        return "agentmesh-litellm"

    def _convert_messages_to_litellm(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        import json
        from langchain_core.messages import ToolMessage
        litellm_messages = []
        for m in messages:
            if isinstance(m, SystemMessage) or getattr(m, "type", "") == "system":
                litellm_messages.append({"role": "system", "content": m.content})
            elif isinstance(m, AIMessage) or getattr(m, "type", "") == "assistant":
                msg_dict = {"role": "assistant", "content": m.content or ""}
                if hasattr(m, "tool_calls") and m.tool_calls:
                    litellm_tool_calls = []
                    for tc in m.tool_calls:
                        litellm_tool_calls.append({
                            "id": tc.get("id") or "call_default",
                            "type": "function",
                            "function": {
                                "name": tc.get("name"),
                                "arguments": json.dumps(tc.get("args") or {})
                            }
                        })
                    msg_dict["tool_calls"] = litellm_tool_calls
                litellm_messages.append(msg_dict)
            elif isinstance(m, ToolMessage) or getattr(m, "type", "") == "tool":
                tool_name = getattr(m, "name", None) or getattr(m, "tool_name", None)
                if not tool_name:
                    tool_call_id = getattr(m, "tool_call_id", "")
                    tool_name = tool_call_id if tool_call_id else "unnamed_tool"
                litellm_messages.append({
                    "role": "tool",
                    "tool_call_id": getattr(m, "tool_call_id", ""),
                    "name": tool_name,
                    "content": m.content
                })
            else:
                litellm_messages.append({"role": "user", "content": m.content})
        return litellm_messages


    def _build_completion_args(self, messages: List[BaseMessage], **kwargs: Any) -> Dict[str, Any]:
        litellm_messages = self._convert_messages_to_litellm(messages)
        
        model_name = self.model
        provider_normalized = self.provider.lower()
        if provider_normalized == "google_genai":
            provider_normalized = "google"
        
        # Determine appropriate prefix for LiteLLM
        prefix = provider_normalized
        if provider_normalized == "google":
            prefix = "gemini"
        elif provider_normalized == "together":
            prefix = "together_ai"
        elif provider_normalized == "sglang":
            prefix = "openai"
        elif provider_normalized == "vllm":
            prefix = "hosted_vllm"
            if model_name.startswith("vllm/"):
                model_name = model_name[5:]

        # Check if the model needs a provider prefix for LiteLLM (even if it has org slashes like meta-llama/)
        if provider_normalized != "openai":
            possible_prefixes = [f"{provider_normalized}/", f"{prefix}/"]
            if not any(model_name.startswith(p) for p in possible_prefixes):
                model_name = f"{prefix}/{model_name}"
        
        # Log the final model name sent to LiteLLM for debugging
        import logging as _logging
        _logging.getLogger(__name__).debug(
            f"[LiteLLM] provider={self.provider}, raw_model={self.model}, "
            f"final_model={model_name}"
        )

        # Setup completion arguments
        args = {
            "model": model_name,
            "messages": litellm_messages,
            "max_tokens": self.max_tokens,
            **kwargs
        }

        # Omit temperature parameter for OpenAI o1/o3/thinking models to avoid errors
        is_thinking = any(term in model_name.lower() for term in ["o1", "o3", "thinking"])
        if not is_thinking:
            args["temperature"] = self.temperature

        # Handle API keys & overrides
        if self.api_key:
            args["api_key"] = self.api_key
        elif provider_normalized == "groq":
            args["api_key"] = get_random_groq_api_key()
        elif provider_normalized == "mistral":
            args["api_key"] = get_random_mistral_api_key()
        elif provider_normalized == "together":
            args["api_key"] = get_random_together_api_key()
        else:
            # Fallback to standard environment variables
            env_map = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "google": "GOOGLE_API_KEY",
                "huggingface": "HUGGINGFACEHUB_API_TOKEN",
                "deepseek": "DEEPSEEK_API_KEY",
                "cohere": "COHERE_API_KEY",
                "perplexity": "PERPLEXITY_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
                "azure": "AZURE_API_KEY",
                "vllm": "VLLM_API_KEY",
                "sglang": "SGLANG_API_KEY"
            }
            env_var = env_map.get(provider_normalized)
            if env_var:
                # Fall back to the settings object as well as the environment.
                # The .env file is read by pydantic into settings, and separately
                # by load_dotenv into os.environ at start-up; if either path has
                # not run, looking in only one of them finds nothing.
                key = os.getenv(env_var)
                if not key:
                    try:
                        from Apps.AgentMesh.Config import settings as _settings
                        key = getattr(_settings, env_var, None)
                    except Exception:
                        key = None
                if not key:
                    # Sending the word "NONE" as the key produced a bare 401 from
                    # the provider and a forty-line stack trace that named
                    # litellm, httpx and langchain but never the missing
                    # variable. Say what is actually wrong.
                    raise ValueError(
                        f"No API key for provider '{provider_normalized}'. "
                        f"Set {env_var} in the .env file at the Backend root, "
                        f"or enter a key on the node itself. The request was "
                        f"not sent."
                    )
                args["api_key"] = key

        # Handle custom base URL (like Ollama or self-hosted)
        if self.base_url:
            args["api_base"] = self.base_url
        elif provider_normalized == "vllm" and os.getenv("VLLM_API_BASE"):
            args["api_base"] = os.getenv("VLLM_API_BASE")
        elif provider_normalized == "sglang" and (os.getenv("SGLANG_API_BASE") or os.getenv("SGLANG_BASE_URL")):
            args["api_base"] = os.getenv("SGLANG_API_BASE") or os.getenv("SGLANG_BASE_URL")

        # Handle reasoning effort for o1/thinking models
        if self.reasoning_effort:
            args["reasoning_effort"] = self.reasoning_effort

        # A reasoning model narrates unless told otherwise, and that narration
        # arrives in the answer itself. Any step that expects the answer to be
        # exactly what it asked for — a JSON object, a single figure — then gets
        # prose wrapped around it and fails. 'hidden' keeps the working out of
        # the reply without giving up the reasoning that produced it.
        if self.reasoning_format:
            args["reasoning_format"] = self.reasoning_format

        # Normalize tool_choice parameter for compatibility with providers like Groq
        if "tool_choice" in args:
            tool_choice = args["tool_choice"]
            if isinstance(tool_choice, str) and tool_choice not in ["auto", "none"]:
                if tool_choice in ["any", "required"]:
                    # Try to map 'any' or 'required' to the first available tool
                    tools = args.get("tools", [])
                    if tools and isinstance(tools, list):
                        try:
                            first_tool = tools[0]
                            if isinstance(first_tool, dict):
                                tool_name = first_tool.get("function", {}).get("name")
                            else:
                                tool_name = getattr(getattr(first_tool, "function", None), "name", None)
                            
                            if tool_name:
                                args["tool_choice"] = {
                                    "type": "function",
                                    "function": {"name": tool_name}
                                }
                            else:
                                args["tool_choice"] = "auto"
                        except Exception:
                            args["tool_choice"] = "auto"
                    else:
                        args["tool_choice"] = "auto"
                else:
                    # tool_choice is a specific function name as a string
                    args["tool_choice"] = {
                        "type": "function",
                        "function": {"name": tool_choice}
                    }
            elif isinstance(tool_choice, dict):
                # Normalize {"name": "tool_name"} to {"type": "function", "function": {"name": "tool_name"}}
                if "name" in tool_choice and "function" not in tool_choice:
                    args["tool_choice"] = {
                        "type": "function",
                        "function": {"name": tool_choice["name"]}
                    }

        # NOTE: Fallback routing is disabled for now. Will be configured later.

        return args

    def bind_tools(
        self,
        tools: List[Any],
        **kwargs: Any,
    ) -> Any:
        """Bind tools to the model for OpenAI/LiteLLM compatibility."""
        from langchain_core.utils.function_calling import convert_to_openai_tool
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=formatted_tools, **kwargs)

    def _parse_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        import json
        tool_calls = []
        raw_tool_calls = getattr(response.choices[0].message, "tool_calls", None)
        if raw_tool_calls:
            for tc in raw_tool_calls:
                tc_id = getattr(tc, "id", None) or tc.get("id")
                tc_type = getattr(tc, "type", None) or tc.get("type")
                func = getattr(tc, "function", None) or tc.get("function")
                if func:
                    func_name = getattr(func, "name", None) or func.get("name")
                    func_args_str = getattr(func, "arguments", None) or func.get("arguments")
                    try:
                        func_args = json.loads(func_args_str) if isinstance(func_args_str, str) else func_args_str
                    except Exception:
                        func_args = {}
                    tool_calls.append({
                        "name": func_name,
                        "args": func_args or {},
                        "id": tc_id,
                        "type": "tool_call"
                    })
        return tool_calls

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        args = self._build_completion_args(messages, **kwargs)
        if stop:
            args["stop"] = stop
            
        start_time = time.time()
        response = litellm.completion(**args, num_retries=3)
        latency_ms = int((time.time() - start_time) * 1000)

        content = response.choices[0].message.content or ""
        
        # Token and cost calculations
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens
        
        cost = 0.0
        is_local = self.provider in ["ollama"] or (self.base_url and "localhost" in self.base_url)
        if not is_local:
            try:
                cost = litellm.completion_cost(completion_response=response) or 0.0
            except Exception:
                pass

        tool_calls = self._parse_tool_calls(response)
        # Standard OpenAI-style token breakdown. Langfuse reads llm_output["token_usage"]
        # first, and message usage_metadata covers LangChain/Ollama-style handlers plus
        # the local AgentMeshObserver DB persistence path (Observer.py).
        token_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        # NOTE: emit the provider-prefixed model name (e.g. groq/llama-3.3-70b-versatile)
        # — the same string sent to LiteLLM — so Langfuse can match it against its model
        # price registry for automatic cost calculation.
        emitted_model_name = args.get("model", self.model)
        ai_msg = AIMessage(
            content=content,
            tool_calls=tool_calls,
            usage_metadata={
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            response_metadata={
                "latency_ms": latency_ms,
                "tokens_used": total_tokens,
                "cost": cost,
                "model_name": emitted_model_name,
                "token_usage": token_usage,
            }
        )
        return ChatResult(
            generations=[ChatGeneration(message=ai_msg)],
            llm_output={
                "token_usage": token_usage,
                "model_name": emitted_model_name,
            }
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        args = self._build_completion_args(messages, **kwargs)
        if stop:
            args["stop"] = stop
            
        start_time = time.time()
        response = await litellm.acompletion(**args, num_retries=3)
        latency_ms = int((time.time() - start_time) * 1000)

        content = response.choices[0].message.content or ""
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens
        
        cost = 0.0
        is_local = self.provider in ["ollama"] or (self.base_url and "localhost" in self.base_url)
        if not is_local:
            try:
                cost = litellm.completion_cost(completion_response=response) or 0.0
            except Exception:
                pass

        tool_calls = self._parse_tool_calls(response)
        # Standard OpenAI-style token breakdown. Langfuse reads llm_output["token_usage"]
        # first, and message usage_metadata covers LangChain/Ollama-style handlers plus
        # the local AgentMeshObserver DB persistence path (Observer.py).
        token_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        # NOTE: emit the provider-prefixed model name (e.g. groq/llama-3.3-70b-versatile)
        # — the same string sent to LiteLLM — so Langfuse can match it against its model
        # price registry for automatic cost calculation.
        emitted_model_name = args.get("model", self.model)
        ai_msg = AIMessage(
            content=content,
            tool_calls=tool_calls,
            usage_metadata={
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            response_metadata={
                "latency_ms": latency_ms,
                "tokens_used": total_tokens,
                "cost": cost,
                "model_name": emitted_model_name,
                "token_usage": token_usage,
            }
        )
        return ChatResult(
            generations=[ChatGeneration(message=ai_msg)],
            llm_output={
                "token_usage": token_usage,
                "model_name": emitted_model_name,
            }
        )


# ==================== LLM Provider Factory ====================

def get_llm_provider(provider: Optional[str] = None, model: Optional[str] = None, **kwargs):
    """
    Factory function to create LLM instances for any supported provider.
    """
    provider = provider or settings.DEFAULT_LLM_PROVIDER
    provider = provider.lower()
    if provider == 'google_genai':
        provider = 'google'
        
    target_model = model or settings.DEFAULT_LLM_MODEL
    
    # Extract params
    temperature = kwargs.pop('temperature', 0.1)
    max_tokens = kwargs.pop('max_tokens', 2000)
    api_key = kwargs.pop('api_key', None)
    base_url = kwargs.pop('base_url', None)
    
    return AgentMeshLiteLLM(
        model=target_model,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
        **kwargs
    )


# ==================== Vision Model Support ====================

def get_vision_llm(provider: Optional[str] = None, model: Optional[str] = None, **kwargs):
    """
    Create a vision-capable LLM instance.
    """
    provider = provider or os.getenv('LLM_VISION_PROVIDER', 'groq')
    if provider.lower() == 'google_genai':
        provider = 'google'

    if not model:
        if provider.lower() == 'groq':
            # llama-4-scout was decommissioned on 17 July 2026; a node left on
            # the old default failed on every call.
            model = 'meta-llama/llama-4-maverick-17b-128e-instruct'
        elif provider.lower() == 'openai':
            model = 'gpt-4o'
        elif provider.lower() == 'anthropic':
            model = 'claude-3-5-sonnet-20241022'
        elif provider.lower() in ['google', 'google_genai']:
            model = 'gemini-2.0-flash'
        else:
            model = os.getenv('LLM_VISION_MODEL', 'gpt-4o')

    temperature = kwargs.pop('temperature', 0.1)
    max_tokens = kwargs.pop('max_tokens', 6000)
    api_key = kwargs.pop('api_key', None)
    base_url = kwargs.pop('base_url', None)

    return AgentMeshLiteLLM(
        model=model,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
        **kwargs
    )


# ==================== Reasoning Model Support ====================

def get_reasoning_llm(provider: Optional[str] = None, model: Optional[str] = None, **kwargs):
    """
    Create a reasoning-capable LLM instance (with extended reasoning).
    """
    provider = provider or os.getenv('LLM_REASONING_PROVIDER', 'openai')
    
    if not model:
        if provider.lower() == 'openai':
            model = 'o1'
        elif provider.lower() == 'anthropic':
            model = 'claude-3-7-sonnet-20250219'
        else:
            model = 'o1'
            
    # o1 specific handling
    if model and 'o1' in model.lower():
        kwargs.pop('temperature', None)
        if 'reasoning_effort' not in kwargs:
            kwargs['reasoning_effort'] = 'medium'

    temperature = kwargs.pop('temperature', 0.1)
    max_tokens = kwargs.pop('max_tokens', 6000)
    api_key = kwargs.pop('api_key', None)
    base_url = kwargs.pop('base_url', None)
    reasoning_effort = kwargs.pop('reasoning_effort', None)
    reasoning_format = kwargs.pop('reasoning_format', None)

    return AgentMeshLiteLLM(
        model=model,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        reasoning_format=reasoning_format,
        **kwargs
    )


# ==================== Pre-configured Instances ====================

try:
    langchain_llm = get_llm_provider()
except Exception:
    langchain_llm = None

try:
    langchainVision_llm = get_vision_llm()
except Exception:
    langchainVision_llm = None

try:
    langchain_llm_thinking = get_reasoning_llm()
except Exception:
    langchain_llm_thinking = None


# ==================== Convenience Functions ====================

def get_langchain_llm_oss():
    """Get open-source LLM instance."""
    provider = os.getenv('LLM_PROVIDER_OSS', 'groq')
    model = os.getenv('LLM_MODEL_OSS', 'llama-3.3-70b-versatile')
    return get_llm_provider(provider, model, temperature=0.1, max_tokens=6000)


def get_langchain_llm_fast():
    """Get fastest LLM instance for low-latency responses."""
    provider = os.getenv('LLM_PROVIDER_FAST', 'groq')
    model = os.getenv('LLM_MODEL_FAST', 'llama-3.1-8b-instant')
    return get_llm_provider(provider, model, temperature=0.0, max_tokens=4000)


def get_langchain_llm_heavy():
    """Get most capable LLM for complex reasoning tasks."""
    provider = os.getenv('LLM_PROVIDER_HEAVY', 'openai')
    model = os.getenv('LLM_MODEL_HEAVY', 'gpt-4o')
    return get_llm_provider(provider, model, temperature=0.2, max_tokens=8000)


def list_available_providers():
    """Return list of supported LLM providers."""
    return ['groq', 'openai', 'anthropic', 'google', 'ollama', 'together', 'huggingface', 'vllm', 'sglang']
