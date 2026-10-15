# AgentMesh LLM Configuration Guide

## Overview

The updated `llm_config.py` module implements AgentMesh's **LLM-agnostic** principle, allowing you to:

- ✅ Switch between multiple LLM providers (Groq, OpenAI, Anthropic, Google, Ollama, Together, HuggingFace)
- ✅ Use different providers for different tasks (fast, heavy reasoning, open-source)
- ✅ Configure vision and reasoning capabilities on a per-provider basis
- ✅ Maintain backward compatibility with existing code
- ✅ Support key rotation for rate limiting (Groq, Mistral)

---

## Supported LLM Providers

| Provider | Best For | API Required | Models |
|----------|----------|--------------|--------|
| **Groq** | Speed, open-source models | Yes (free tier) | Llama 3.3, Llama 4 Scout, GPT-OSS |
| **OpenAI** | State-of-the-art reasoning | Yes (paid) | GPT-4o, O1 |
| **Anthropic** | Safety, long context | Yes (paid) | Claude 3.5 Sonnet |
| **Google** | Multimodal, fast | Yes (free tier) | Gemini 2.0 Flash |
| **Ollama** | Local, no API | No | Mistral, Llama, etc. |
| **Together** | Cost-effective scale | Yes (paid) | Llama 3.1, Qwen |
| **HuggingFace** | Open-source hosting | Yes (free/paid) | Any HF model |

---

## Quick Start

### 1. Setup Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 2. Use Default LLM (Groq)

```python
from Apps.AgentMesh.Utilities.LLM.llm_config import langchain_llm

# Uses Groq with key rotation by default
response = langchain_llm.invoke([{"role": "user", "content": "Hello!"}])
```

### 3. Switch Providers Dynamically

```python
from Apps.AgentMesh.Utilities.LLM.llm_config import get_llm_provider

# Use OpenAI for this task
llm = get_llm_provider("openai", "gpt-4o")

# Use Anthropic
llm = get_llm_provider("anthropic", "claude-3-5-sonnet-20241022")

# Use local Ollama
llm = get_llm_provider("ollama", "mistral")
```

### 4. Use Vision Capabilities

```python
from Apps.AgentMesh.Utilities.LLM.llm_config import get_vision_llm

# Get a vision-capable model
vision_llm = get_vision_llm("openai", "gpt-4o")

# Or use your configured default vision provider
from Apps.AgentMesh.Utilities.LLM.llm_config import langchainVision_llm
```

### 5. Use Reasoning Models

```python
from Apps.AgentMesh.Utilities.LLM.llm_config import get_reasoning_llm

# Get reasoning model (OpenAI o1 or Claude with extended thinking)
reasoning_llm = get_reasoning_llm()
```

### 6. Use Specialized LLMs

```python
from Apps.AgentMesh.Utilities.LLM.llm_config import (
    get_langchain_llm_fast,
    get_langchain_llm_heavy,
    get_langchain_llm_oss
)

# Fast LLM for low-latency responses
fast_llm = get_langchain_llm_fast()

# Heavy LLM for complex reasoning
heavy_llm = get_langchain_llm_heavy()

# Open-source LLM (typically Groq or Ollama)
oss_llm = get_langchain_llm_oss()
```

---

## Integration with LangGraph Agents

### Example: ReAct Agent with Provider Switching

```python
from langchain.agents import create_react_agent
from Apps.AgentMesh.Utilities.LLM.llm_config import get_llm_provider
from langchain_core.tools import tool
from langchain import hub

@tool
def search_web(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

# Choose provider
llm = get_llm_provider("openai", "gpt-4o")

# Create agent
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, [search_web], prompt)

# Run agent...
```

### Example: Tool-Calling Agent (LangChain 1.0)

```python
from langchain.agents import create_tool_calling_agent
from Apps.AgentMesh.Utilities.LLM.llm_config import get_llm_provider
from langgraph.prebuilt import create_react_agent as lg_agent

llm = get_llm_provider("anthropic")  # Use default Anthropic

@tool
def book_flight(origin: str, destination: str, date: str) -> str:
    """Book a flight."""
    return f"Booked: {origin} → {destination} on {date}"

# Create graph-based agent (LangGraph)
app = lg_agent(llm, [book_flight])
result = app.invoke({"messages": [("user", "Book me to Paris tomorrow")]})
```

### Example: Multi-Agent Supervisor with Provider Diversity

```python
from langgraph_supervisor import create_supervisor
from Apps.AgentMesh.Utilities.LLM.llm_config import get_llm_provider
from langgraph.prebuilt import create_react_agent

# Researcher agent uses fast Groq
research_llm = get_llm_provider("groq", "mixtral-8x7b-32768")
research_agent = create_react_agent(research_llm, [search_tool])

# Analyst agent uses powerful OpenAI
analysis_llm = get_llm_provider("openai", "gpt-4o")
analysis_agent = create_react_agent(analysis_llm, [data_tool])

# Supervisor uses balanced Anthropic
supervisor_llm = get_llm_provider("anthropic")

supervisor = create_supervisor(
    agents={"researcher": research_agent, "analyst": analysis_agent},
    llm=supervisor_llm
)
```

---

## Environment Variable Configuration

### Basic Configuration

```bash
# Set default provider (Groq, OpenAI, etc.)
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b

# Vision provider
LLM_VISION_PROVIDER=groq
LLM_VISION_MODEL=llama-3.2-90b-vision-preview
```

### Provider-Specific Configuration

```bash
# OpenAI
LLM_PROVIDER_OPENAI=openai
LLM_MODEL_OPENAI=gpt-4o
OPENAI_API_KEY=sk-...

# Anthropic
LLM_PROVIDER_ANTHROPIC=anthropic
LLM_MODEL_ANTHROPIC=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini
LLM_PROVIDER_GOOGLE=google
LLM_MODEL_GOOGLE=gemini-2.0-flash
GOOGLE_API_KEY=AIza...

# Ollama (local)
LLM_PROVIDER_OLLAMA=ollama
LLM_MODEL_OLLAMA=mistral
OLLAMA_BASE_URL=http://localhost:11434
```

### Task-Specific Configuration

```bash
# Fast responses (Groq)
LLM_PROVIDER_FAST=groq
LLM_MODEL_FAST=llama-3.1-8b-instant

# Complex reasoning (OpenAI)
LLM_PROVIDER_HEAVY=openai
LLM_MODEL_HEAVY=gpt-4o

# Open-source only (Groq or Ollama)
LLM_PROVIDER_OSS=groq
LLM_MODEL_OSS=openai/gpt-oss-120b

# Extended reasoning
LLM_REASONING_PROVIDER=openai
```

---

## Key Rotation (Groq & Mistral)

The configuration maintains support for key rotation via `keyrotation.py`:

```python
from Apps.AgentMesh.Utilities.LLM.llm_config import get_llm_provider

# Automatically rotates through your Groq API keys
llm = get_llm_provider("groq")  # Uses get_random_groq_api_key()

# Each call distributes load across multiple keys
```

Update `keyrotation.py` with your API keys for rate limiting.

---

## Migration from Old Config

### Old Code (Still Works)

```python
from Apps.AgentMesh.Utilities.LLM.llm_config import langchain_llm
llm = langchain_llm  # Defaults to Groq
```

### New Code (Recommended)

```python
from Apps.AgentMesh.Utilities.LLM.llm_config import get_llm_provider

# Explicit and flexible
llm = get_llm_provider("openai", "gpt-4o")
llm = get_llm_provider("anthropic")  # Uses env models
llm = get_llm_provider()  # Uses LLM_PROVIDER env
```

---

## Cost Optimization Strategies

### Strategy 1: Multi-Tier LLM Selection

```python
from Apps.AgentMesh.Utilities.LLM.llm_config import (
    get_langchain_llm_fast,      # Groq: $0 (free tier)
    get_langchain_llm_heavy,     # GPT-4o: Higher cost
)

# Use fast for simple tasks
simple_llm = get_langchain_llm_fast()

# Use heavy only when needed
complex_llm = get_langchain_llm_heavy()
```

### Strategy 2: Local-First (Ollama)

```python
from Apps.AgentMesh.Utilities.LLM.llm_config import get_llm_provider

# Run locally first (free, private)
llm = get_llm_provider("ollama", "mistral")

# Fall back to cloud API only if needed
try:
    response = llm.invoke(...)
except:
    llm = get_llm_provider("openai")
    response = llm.invoke(...)
```

---

## Observability with LangSmith

Pair this with LangSmith for full visibility:

```bash
# Enable in .env
LANGSMITH_API_KEY=your_api_key
LANGSMITH_PROJECT=agentmesh
LANGCHAIN_TRACING_V2=true
```

Then all LLM calls will be traced automatically.

---

## Troubleshooting

### "UnsupportedLLMProvider" Error

```python
# Check available providers
from Apps.AgentMesh.Utilities.LLM.llm_config import list_available_providers
print(list_available_providers())
# Output: ['groq', 'openai', 'anthropic', 'google', 'ollama', 'together', 'huggingface']
```

### Missing API Keys

Ensure all required API keys are in `.env`:

```bash
# For OpenAI
echo "OPENAI_API_KEY=sk-..." >> .env

# For Anthropic
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

### Slow Responses

Use fast provider:

```python
from Apps.AgentMesh.Utilities.LLM.llm_config import get_langchain_llm_fast
llm = get_langchain_llm_fast()  # Defaults to Groq
```

---

## Next Steps

1. ✅ Update `.env` with your API keys
2. ✅ Test provider switching: `python -c "from Apps.AgentMesh.Utilities.LLM.llm_config import get_llm_provider; llm = get_llm_provider('openai'); print(llm)"`
3. ✅ Update your agent code to use `get_llm_provider()` for flexibility
4. ✅ Set up LangSmith for observability
5. ✅ Configure task-specific providers (fast, heavy, oss)

For more info, see [AgentMesh Technical Notes](../Notes/TechincalNote.md).
