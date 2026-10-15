from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for an LLM node"""

    provider: str = Field(
        "groq", description="LLM provider, e.g., groq, openai, anthropic"
    )
    model: str = Field(
        "llama-3.3-70b-versatile",
        description="Model name, e.g., llama-3.3-70b-versatile, gpt-4o",
    )
    temperature: float = 0.0
    max_tokens: Optional[int] = Field(
        None,
        description=(
            "Largest answer the model may produce, in tokens. Left unset, the "
            "provider's own default applies — often only a few thousand. A "
            "reasoning model spends this same budget thinking before it writes, "
            "so a long answer over a dense document can exhaust it and return "
            "nothing at all. Raise it when a node must produce a large "
            "structured result."
        ),
    )
    reasoning_effort: Optional[str] = Field(
        None,
        description=(
            "How long a reasoning model thinks before answering: 'none', 'low', "
            "'medium' or 'high'. Which of these a model accepts varies by "
            "provider — one that rejects the value fails the call, so leave "
            "unset unless the model is known to take it. Ignored by models that "
            "do not reason."
        ),
    )
    reasoning_format: Optional[str] = Field(
        None, description="Where a reasoning model's working goes: 'hidden' returns only the answer, 'parsed' keeps it separate, 'raw' includes it in the answer. Use 'hidden' when a later step needs the answer alone."
    )
    api_key_override: Optional[str] = Field(
        None, description="Optional override for API key per node", repr=False
    )
    base_url: Optional[str] = Field(
        None,
        description="Optional base URL for SGLang, Ollama, vLLM or other self-hosted providers (e.g., http://localhost:30000/v1)",
    )


class EdgeCondition(BaseModel):
    """Condition for dynamic routing edges"""

    key: str = Field(..., description="State or context key to evaluate")
    operator: str = Field(
        "==", description="Operator for comparison (==, !=, >, <, contains)"
    )
    value: Any = Field(..., description="Expected value")


class EdgeConfig(BaseModel):
    """Defines a directed edge between two nodes"""

    source: str = Field(..., description="Source Node ID (or 'START')")
    target: str = Field(..., description="Target Node ID (or 'END')")
    condition: Optional[EdgeCondition] = Field(
        None, description="Optional condition for conditional edges"
    )
    # Frontend visual handle metadata
    source_handle: Optional[str] = Field(
        None, description="Source handle ID for visual connections"
    )
    target_handle: Optional[str] = Field(
        None, description="Target handle ID for visual connections"
    )
    is_breakpoint: Optional[bool] = Field(
        False, description="If True, execution pauses/stops before traversing this edge"
    )
    data: Optional[Dict[str, Any]] = Field(
        None, description="Arbitrary edge metadata including visual & breakpoint flags"
    )

