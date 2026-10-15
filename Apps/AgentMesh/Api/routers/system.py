from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
import logging
import random
from typing import Dict, Any, Optional, Literal

from Database.Database import get_db
from Database import AgentMeshCrud
from Apps.AgentMesh.Core.ToolRegistry import registry
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from Apps.AgentMesh.Core.WebSocketManager import ws_manager
from Apps.Auth.auth_utils import decode_token

logger = logging.getLogger(__name__)

router = APIRouter(tags=["System"])

@router.get("/notifications")
async def list_notifications(page: int = 1, limit: int = 50, unread_only: bool = False, workflow_id: Optional[str] = None, type: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    skip = (page - 1) * limit
    notifications = await AgentMeshCrud.list_notifications(db, skip=skip, limit=limit, unread_only=unread_only, workflow_id=workflow_id, type=type)
    total = await AgentMeshCrud.count_notifications(db, unread_only=unread_only, workflow_id=workflow_id, type=type)
    return {
        "status": "success", 
        "total": total, 
        "notifications": [
            {
                "id": n.id, 
                "workflow_id": n.workflow_id,
                "thread_id": n.thread_id,
                "title": n.title, 
                "message": n.message, 
                "type": n.type,
                "is_read": n.is_read, 
                "metadata_json": n.metadata_json,
                "created_at": n.created_at,
                "updated_at": n.updated_at
            } for n in notifications
        ]
    }

class NotificationActionRequest(BaseModel):
    action: str
    notification_ids: Optional[list[str]] = None

@router.post("/notifications/action")
async def notification_action(request: NotificationActionRequest, db: AsyncSession = Depends(get_db)):
    if request.action == "mark_all_read":
        await AgentMeshCrud.mark_all_notifications_as_read(db)
        return {"status": "success"}
        
    if not request.notification_ids:
        raise HTTPException(status_code=400, detail="notification_ids required")
        
    count = 0
    if request.action == "mark_read":
        for nid in request.notification_ids:
            if await AgentMeshCrud.mark_notification_as_read(db, nid): 
                count += 1
    elif request.action == "delete":
        for nid in request.notification_ids:
            if await AgentMeshCrud.delete_notification(db, nid): 
                count += 1
    else:
        raise HTTPException(status_code=400, detail="Invalid action")
        
    return {"status": "success", "processed": count}


import os
import requests
import litellm
import time
import hashlib
from Apps.AgentMesh.Utilities.LLM.keyrotation import GROQ_API_KEYS, Mistral_API_KEYS, TOGETHER_API_KEYS

# Keywords that indicate non-chat models (embeddings, speech, image, video, etc.)
_EXCLUDED_KEYWORDS = [
    'whisper', 'embed', 'tts', 'speech', 'audio', 'moderat', 'transcri',
    'imagen', 'dall-e', 'veo', 'generate', 'flux', 'stability', 'recraft',
    'video', 'polly', 'ocr', 'realtime', 'search', 'computer-use',
    'deepgram', 'riva', 'assemblyai', 'jina-reranker',
]

# Curated list of keywords indicating deprecated or inactive models
_DEPRECATED_KEYWORDS = [
    'maverick', 'scout', 'llama-2', 'llama2',
    'gpt-3.5-turbo-0301', 'gpt-3.5-turbo-0613',
    'gpt-4-0314', 'gpt-4-0613', 'gpt-4-32k-0314', 'gpt-4-32k-0613'
]

_LIVE_MODELS_CACHE = {}  # cache_key -> (timestamp, model_list)
_CACHE_TTL = 3600  # 1 hour cache duration

def _is_chat_model(model_name: str) -> bool:
    name_lower = model_name.lower()
    return not any(kw in name_lower for kw in _EXCLUDED_KEYWORDS)

def _is_active_model(model_name: str) -> bool:
    if not _is_chat_model(model_name):
        return False
    name_lower = model_name.lower()
    return not any(kw in name_lower for kw in _DEPRECATED_KEYWORDS)

def _strip_prefix(model_name: str, prefix: str) -> str:
    if model_name.startswith(prefix):
        return model_name[len(prefix):]
    return model_name

def _get_provider_api_key(provider: str) -> Optional[str]:
    if provider == "groq":
        if GROQ_API_KEYS:
            return GROQ_API_KEYS[0]
        return os.getenv("GROQ_API_KEY")
    elif provider == "mistral":
        if Mistral_API_KEYS:
            return Mistral_API_KEYS[0]
        return os.getenv("MISTRAL_API_KEY")
    elif provider == "together":
        if TOGETHER_API_KEYS:
            return TOGETHER_API_KEYS[0]
        return os.getenv("TOGETHER_API_KEY")
    elif provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    elif provider == "openrouter":
        return os.getenv("OPENROUTER_API_KEY")
    elif provider == "deepseek":
        return os.getenv("DEEPSEEK_API_KEY")
    elif provider == "perplexity":
        return os.getenv("PERPLEXITY_API_KEY")
    elif provider == "cohere":
        return os.getenv("COHERE_API_KEY")
    elif provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY")
    elif provider == "google":
        return os.getenv("GOOGLE_API_KEY")
    elif provider == "azure":
        return os.getenv("AZURE_API_KEY")
    elif provider == "bedrock":
        return os.getenv("AWS_SECRET_ACCESS_KEY")
    return None

def _compute_cache_key(provider: str, api_key: str) -> str:
    key_hash = hashlib.sha256(api_key.encode('utf-8')).hexdigest()[:16]
    return f"{provider}:{key_hash}"

def _fetch_live_models(provider: str, url: str, api_key: str) -> Optional[list]:
    cache_key = _compute_cache_key(provider, api_key)
    if cache_key in _LIVE_MODELS_CACHE:
        ts, cached_list = _LIVE_MODELS_CACHE[cache_key]
        if time.time() - ts < _CACHE_TTL:
            logger.debug(f"Serving live models cache for provider: {provider}")
            return cached_list

    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        logger.debug(f"Scanning live models from {url} for provider: {provider}")
        res = requests.get(url, headers=headers, timeout=2.0, allow_redirects=False)
        if res.status_code == 200:
            data = res.json().get("data", [])
            models = []
            for item in data:
                m_id = item.get("id")
                if m_id:
                    models.append(m_id)
            if models:
                _LIVE_MODELS_CACHE[cache_key] = (time.time(), models)
                return models
    except Exception as e:
        logger.error(f"Error fetching live models for {provider}: {e}")
    return None

@router.get("/models/available")
async def list_available_models():
    """
    Dynamically lists available models by scanning:
    1. Active cloud providers (scans live endpoints if API keys are configured, fallback to clean static registries)
    2. Local Ollama server for self-hosted models
    3. Local/Intranet vLLM via open-source OpenAI-compatible scan
    Returns active models with provider labels for the frontend dropdown.
    """
    available = []

    providers_config = [
        {
            "id": "openai",
            "display": "OpenAI",
            "prefix": "",
            "static_list": litellm.open_ai_chat_completion_models,
            "live_url": "https://api.openai.com/v1/models"
        },
        {
            "id": "anthropic",
            "display": "Anthropic",
            "prefix": "",
            "static_list": litellm.anthropic_models,
            "live_url": None
        },
        {
            "id": "google",
            "display": "Google",
            "prefix": "gemini/",
            "static_list": litellm.gemini_models,
            "live_url": None
        },
        {
            "id": "together",
            "display": "Together",
            "prefix": "",
            "static_list": litellm.together_ai_models,
            "live_url": "https://api.together.xyz/v1/models"
        },
        {
            "id": "huggingface",
            "display": "HuggingFace",
            "prefix": "huggingface/",
            "static_list": litellm.huggingface_models,
            "live_url": None
        },
        {
            "id": "deepseek",
            "display": "DeepSeek",
            "prefix": "deepseek/",
            "static_list": litellm.deepseek_models,
            "live_url": "https://api.deepseek.com/v1/models"
        },
        {
            "id": "cohere",
            "display": "Cohere",
            "prefix": "cohere/",
            "static_list": litellm.cohere_models,
            "live_url": "https://api.cohere.com/v1/models"
        },
        {
            "id": "perplexity",
            "display": "Perplexity",
            "prefix": "perplexity/",
            "static_list": litellm.perplexity_models,
            "live_url": "https://api.perplexity.ai/models"
        },
        {
            "id": "openrouter",
            "display": "OpenRouter",
            "prefix": "openrouter/",
            "static_list": litellm.openrouter_models,
            "live_url": "https://openrouter.ai/api/v1/models"
        },
        {
            "id": "azure",
            "display": "Azure",
            "prefix": "azure/",
            "static_list": litellm.azure_models,
            "live_url": None
        },
        {
            "id": "bedrock",
            "display": "Bedrock",
            "prefix": "bedrock/",
            "static_list": litellm.bedrock_models,
            "live_url": None
        },
        {
            "id": "groq",
            "display": "Groq",
            "prefix": "groq/",
            "static_list": litellm.groq_models,
            "live_url": "https://api.groq.com/openai/v1/models"
        },
        {
            "id": "mistral",
            "display": "Mistral",
            "prefix": "mistral/",
            "static_list": litellm.mistral_chat_models,
            "live_url": "https://api.mistral.ai/v1/models"
        }
    ]

    # 1. Standard Cloud & Hosted Providers
    for cfg in providers_config:
        api_key = _get_provider_api_key(cfg["id"])
        models_source = None
        is_live = False

        # Try to scan live models if API key is configured and endpoint is supported
        if api_key and cfg["live_url"]:
            models_source = _fetch_live_models(cfg["id"], cfg["live_url"], api_key)
            if models_source:
                is_live = True

        # Fallback to static list (filtered) if live scan is not supported or failed
        if not models_source:
            models_source = cfg["static_list"]

        for full_model in sorted(models_source):
            # If it is live, we only check _is_chat_model, not _DEPRECATED_KEYWORDS!
            # If it is static fallback, we check _is_active_model (which applies deprecation check)
            should_include = _is_chat_model(full_model) if is_live else _is_active_model(full_model)
            if should_include:
                clean_name = _strip_prefix(full_model, cfg["prefix"])
                available.append({
                    "provider": cfg["id"],
                    "model": clean_name,
                    "label": f"{clean_name} ({cfg['display']})"
                })

    # 2. Local Ollama (dynamic scan - bypass deprecation check since it is a live self-hosted endpoint)
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        res = requests.get(f"{ollama_url}/api/tags", timeout=2.0)
        if res.status_code == 200:
            for m in res.json().get("models", []):
                name = m.get("name", "")
                if name and _is_chat_model(name):
                    available.append({
                        "provider": "ollama",
                        "model": name,
                        "label": f"{name} (Ollama)"
                    })
    except Exception:
        pass

    # 3. Local/Intranet vLLM (dynamic scan - bypass deprecation check since it is a live endpoint)
    vllm_url = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
    vllm_key = os.getenv("VLLM_API_KEY")
    if vllm_url:
        try:
            headers = {}
            if "ngrok" in vllm_url:
                headers["ngrok-skip-browser-warning"] = "69420"
            if vllm_key:
                headers["Authorization"] = f"Bearer {vllm_key}"
            res = requests.get(
                f"{vllm_url}/models",
                headers=headers,
                timeout=2.0,
                allow_redirects=not bool(vllm_key),
            )
            if res.status_code == 200:
                for m in res.json().get("data", []):
                    m_id = m.get("id")
                    if m_id and _is_chat_model(m_id):
                        available.append({
                            "provider": "vllm",
                            "model": m_id,
                            "label": f"{m_id} (vLLM)"
                        })
        except Exception:
            pass

    return {"status": "success", "models": available}

@router.get("/health")
async def health_check():
    return {"status": "online", "engine": "AgentMesh", "version": "1.1.0"}

@router.get("/dashboard/stats")
async def get_dashboard_stats(db: AsyncSession = Depends(get_db)):
    try:
        stats = await AgentMeshCrud.get_dashboard_stats(db)
        all_meta = registry.get_all_meta()
        stats["total_tools"] = len(all_meta)
        return {"status": "success", "data": stats}
    except Exception as e:
        logger.exception("Error fetching dashboard stats")
        raise HTTPException(status_code=500, detail=str(e))


# --- Utilities ---

class UtilsActionRequest(BaseModel):
    action: Literal["fix_grammar", "ml_techniques", "train_model", "dag_create", "dag_status", "explain_workflow", "generate_skill_sop", "chat_architect", "generate_python_rule"]
    payload: dict = {}

import time
import random
dag_store = {}

def _handle_ml_techniques(payload: dict):
    category = payload.get("category", "all")
    techniques = {
        "Supervised": ["Linear Regression", "Decision Trees", "Random Forest", "Support Vector Machines", "Neural Networks"],
        "Unsupervised": ["K-Means Clustering", "PCA", "Association Rules", "Autoencoders"],
        "Reinforcement": ["Q-Learning", "Deep Q-Network", "Policy Gradient", "PPO"],
        "Deep Learning": ["CNN", "RNN", "LSTM", "Transformers", "GANS"]
    }
    if category.capitalize() in techniques:
        return {
            "status": "success",
            "category": category.capitalize(),
            "techniques": techniques[category.capitalize()],
            "count": len(techniques[category.capitalize()])
        }
    return {
        "status": "success",
        "category": "all",
        "data": techniques,
        "total_count": sum(len(v) for v in techniques.values())
    }

def _handle_dag_status(payload: dict):
    dag_id = payload.get("dag_id")
    if dag_id not in dag_store:
        raise HTTPException(status_code=404, detail="DAG ID not found.")
    elapsed = time.time() - dag_store[dag_id]["created_at"]
    if elapsed < 60:
        return {
            "status": "success",
            "dag_id": dag_id,
            "dag_status": "In Progress",
            "elapsed_seconds": int(elapsed),
            "remaining_seconds": int(60 - elapsed)
        }
    return {
        "status": "success",
        "dag_id": dag_id,
        "dag_status": "Completed",
        "completion_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    }

async def _handle_fix_grammar(payload: dict):
    text = payload.get("text", "")
    llm_config = payload.get("llm", {})
    from Apps.AgentMesh.Utilities.ML.workflow_architect import workflow_architect
    try:
        fixed_text = await workflow_architect.repair_text(text, llm_config)
        return {"status": "success", "original": text, "fixed": fixed_text}
    except Exception as e:
        logger.exception("Grammar fix endpoint failed")
        return {"status": "error", "message": str(e), "original": text, "fixed": text}

async def _handle_explain_workflow(payload: dict):
    workflow_data = payload.get("workflow", {})
    llm_config = payload.get("llm", {})
    from Apps.AgentMesh.Utilities.ML.workflow_architect import workflow_architect
    try:
        explanation = await workflow_architect.explain(workflow_data, llm_config)
        return {"status": "success", "explanation": explanation}
    except Exception as e:
        logger.exception("Explain workflow endpoint failed")
        error_msg = str(e)
        if "AuthenticationError" in error_msg or "authentication_error" in error_msg or "x-api-key header is required" in error_msg or "API key is required" in error_msg or "API key" in error_msg:
            provider_name = llm_config.get("provider", "the selected").upper()
            error_msg = f"Authentication Error: API key is missing or invalid for {provider_name}. Please set the environment variable or enter a valid API Key Override in the settings."
        return {"status": "error", "message": error_msg}

async def _handle_chat_architect(payload: dict):
    workflow_data = payload.get("workflow", {})
    context_data = payload.get("context", {})
    chat_history = payload.get("messages", [])
    llm_config = payload.get("llm", {})
    from Apps.AgentMesh.Utilities.ML.workflow_architect import workflow_architect
    try:
        reply = await workflow_architect.chat(workflow_data, context_data, chat_history, llm_config)
        return {"status": "success", "reply": reply}
    except Exception as e:
        logger.exception("Chat architect endpoint failed")
        error_msg = str(e)
        if "AuthenticationError" in error_msg or "authentication_error" in error_msg or "x-api-key header is required" in error_msg or "API key is required" in error_msg or "API key" in error_msg:
            provider_name = llm_config.get("provider", "the selected").upper()
            error_msg = f"Authentication Error: API key is missing or invalid for {provider_name}. Please set the environment variable or enter a valid API Key Override in the settings."
        return {"status": "error", "message": error_msg}

async def _handle_generate_skill_sop(payload: dict):
    skill_name = payload.get("name", "New Skill")
    skill_description = payload.get("description", "")
    skill_tags = payload.get("tags", "")
    llm_config = payload.get("llm", {})
    
    system_instruction = (
        "You are an expert technical writer and AI systems architect.\n"
        "Your task is to write a highly detailed, professional, and clear Standard Operating Procedure (SOP) "
        "document in Markdown format for a custom skill designed for AI agents.\n\n"
        "The SOP must clearly define:\n"
        "1. **Objective**: What the skill accomplishes.\n"
        "2. **Prerequisites**: Any requirements or assumptions.\n"
        "3. **Standard Operating Procedure**: Step-by-step instructions on how the agent should execute the skill.\n"
        "4. **Error Handling**: How the agent should respond to common errors or exceptions.\n\n"
        "Do NOT return any introductory or conversational text. Return ONLY the markdown content itself."
    )
    
    prompt = (
        f"Skill Name: {skill_name}\n"
        f"Skill Description: {skill_description}\n"
        f"Associated Tags: {skill_tags}\n"
    )
    
    from Apps.AgentMesh.Utilities.LLM.llm_config import get_llm_provider
    from langchain_core.messages import SystemMessage, HumanMessage
    
    try:
        llm = get_llm_provider(
            provider=llm_config.get("provider"),
            model=llm_config.get("model"),
            temperature=llm_config.get("temperature", 0.2),
            api_key=llm_config.get("api_key_override")
        )
        
        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=prompt)
        ]
        
        response = await llm.ainvoke(messages)
        return {"status": "success", "sop": response.content}
    except Exception as e:
        logger.exception("Generate skill SOP failed")
        error_msg = str(e)
        if "AuthenticationError" in error_msg or "authentication_error" in error_msg or "x-api-key header is required" in error_msg or "API key is required" in error_msg:
            provider_name = llm_config.get("provider", "the selected").upper()
            error_msg = f"Authentication Error: API key is missing or invalid for {provider_name}. Please set the environment variable or enter a valid API Key Override in the settings."
        return {"status": "error", "message": error_msg}

def _handle_train_model(payload: dict):
    category = payload.get("category", "unknown")
    return {
        "status": "success",
        "message": f"Integration Success! Your {category} data training has started.",
        "estimated_time": "5 minutes",
        "node_id": "api_trainer_v1"
    }

def _handle_dag_create(payload: dict):
    dag_name = payload.get("dag_name", "unnamed")
    dag_id = f"dag_{dag_name.lower().replace(' ', '_')}_{random.randint(1000, 9999)}"
    dag_store[dag_id] = {
        "created_at": time.time(),
        "name": dag_name
    }
    return {
        "status": "success",
        "dag_id": dag_id,
        "message": f"DAG '{dag_name}' initialization started."
    }

async def _handle_generate_python_rule(payload: dict):
    prompt = payload.get("prompt", "")
    schema_str = payload.get("schema", "")
    existing_code = payload.get("existing_code", "")
    llm_config = payload.get("llm", {})

    from Apps.AgentMesh.Utilities.ML.workflow_architect import workflow_architect
    try:
        code = await workflow_architect.generate_rule(prompt, schema_str, existing_code, llm_config)
        return {"status": "success", "code": code}
    except Exception as e:
        logger.exception("Generate Python rule failed")
        return {"status": "error", "message": str(e)}

@router.post("/utils/action")
async def utils_action_dispatcher(request: UtilsActionRequest):
    if request.action == "fix_grammar":
        return await _handle_fix_grammar(request.payload)
    elif request.action == "explain_workflow":
        return await _handle_explain_workflow(request.payload)
    elif request.action == "chat_architect":
        return await _handle_chat_architect(request.payload)
    elif request.action == "generate_skill_sop":
        return await _handle_generate_skill_sop(request.payload)
    elif request.action == "ml_techniques":
        return _handle_ml_techniques(request.payload)
    elif request.action == "train_model":
        return _handle_train_model(request.payload)
    elif request.action == "dag_create":
        return _handle_dag_create(request.payload)
    elif request.action == "dag_status":
        return _handle_dag_status(request.payload)
    elif request.action == "generate_python_rule":
        return await _handle_generate_python_rule(request.payload)

class TestConnectionRequest(BaseModel):
    provider: str
    api_key: Optional[str] = Field(None, repr=False)
    base_url: Optional[str] = None

@router.post("/models/test-connection")
async def test_provider_connection(request: TestConnectionRequest):
    """
    Validates custom provider credentials by hitting their live `/v1/models` endpoint.
    If successful, returns active models and caches them under a credential hash.
    """
    provider = request.provider
    api_key = request.api_key
    base_url = request.base_url

    if not api_key:
        raise HTTPException(status_code=400, detail="API Key is required to test custom connection")

    # Map providers to live standard endpoints
    live_endpoints = {
        "openai": "https://api.openai.com/v1/models",
        "together": "https://api.together.xyz/v1/models",
        "deepseek": "https://api.deepseek.com/v1/models",
        "cohere": "https://api.cohere.com/v1/models",
        "perplexity": "https://api.perplexity.ai/models",
        "openrouter": "https://openrouter.ai/api/v1/models",
        "groq": "https://api.groq.com/openai/v1/models",
        "mistral": "https://api.mistral.ai/v1/models",
    }

    url = base_url if base_url else live_endpoints.get(provider)
    if not url:
        # If provider doesn't support a standard OpenAI-compatible /v1/models endpoint, fallback to success
        return {
            "status": "success",
            "message": f"Connection simulated for {provider.capitalize()} (does not support standard models query)"
        }

    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        logger.debug(f"Testing dynamic custom connection for {provider} at {url}")
        res = requests.get(url, headers=headers, timeout=5.0, allow_redirects=False)
        if res.status_code == 200:
            data = res.json().get("data", [])
            models = []
            for item in data:
                m_id = item.get("id")
                if m_id and _is_chat_model(m_id):
                    # Strip any prefix if returned
                    prefix = f"{provider}/"
                    clean_name = _strip_prefix(m_id, prefix)
                    models.append({
                        "provider": provider,
                        "model": clean_name,
                        "label": f"{clean_name} ({provider.capitalize()})"
                    })
            
            if not models:
                raise HTTPException(status_code=400, detail="Connected successfully, but no chat models were returned.")

            # Update our global memory cache so that the main list endpoint serves these immediately
            cache_key = _compute_cache_key(provider, api_key)
            _LIVE_MODELS_CACHE[cache_key] = (time.time(), [m["model"] for m in models])

            return {
                "status": "success",
                "message": "Connection successful!",
                "models": models
            }
        else:
            detail = f"Error {res.status_code}: {res.text}"
            raise HTTPException(status_code=400, detail=detail)
    except Exception as e:
        logger.error(f"Test connection failed for {provider}: {e}")
        raise HTTPException(status_code=400, detail=f"Connection failed: {str(e)}")


class DbQueryTestRequest(BaseModel):
    db_engine: str
    db_connection_string: Optional[str] = None
    db_database: Optional[str] = None
    db_collection: Optional[str] = None
    db_operation: str
    db_document: Optional[Any] = None
    db_query: Optional[Any] = None
    db_limit: Optional[int] = 100

@router.post("/db/run-query", tags=["Database"])
async def run_db_query(req: DbQueryTestRequest):
    """
    Directly runs/tests a database query for Node Validation.
    """
    import os
    engine = req.db_engine.lower()
    conn_str = req.db_connection_string
    db_name = req.db_database
    coll_name = req.db_collection
    op = req.db_operation
    limit = req.db_limit or 100

    # Fallback to env variables if not set
    if engine == "mongodb":
        if not conn_str:
            conn_str = os.environ.get("MONGODB_URI", "")
        if not db_name:
            db_name = os.environ.get("MONGODB_DATABASE", "")
        if not conn_str or not db_name or not coll_name:
            raise HTTPException(
                status_code=400,
                detail="MongoDB configurations missing: connection string, database, or collection name."
            )
    elif engine in ("postgresql", "postgres"):
        if not conn_str:
            conn_str = os.environ.get("DATABASE_URL", "")
        if not conn_str:
            raise HTTPException(
                status_code=400,
                detail="PostgreSQL connection string missing."
            )
    elif engine == "sqlite":
        if not conn_str:
            raise HTTPException(
                status_code=400,
                detail="SQLite database file path missing."
            )

    try:
        from Apps.AgentMesh.Core.Compiler.nodes.db_node import execute_db_query_core
        
        summary, result_data = await execute_db_query_core(
            db_engine=engine,
            connection_string=conn_str,
            database=db_name,
            collection_name=coll_name,
            operation=op,
            document=req.db_document,
            query=req.db_query,
            limit=limit,
            is_test=True
        )
        return {
            "status": "success",
            "summary": summary,
            "result": result_data
        }

    except Exception as e:
        logger.exception("Test DB Query execution failed")
        raise HTTPException(
            status_code=400,
            detail=f"Query Execution Failed: {str(e)}"
        )


# --- Sensitivity Rules Management ---
import json
import os

RULES_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "Schema", "sensitivity_rules.json")

@router.get("/sensitivity_rules")
async def get_sensitivity_rules():
    """Retrieve global sensitivity rules"""
    try:
        if os.path.exists(RULES_FILE):
            with open(RULES_FILE, "r") as f:
                return json.load(f)
        return {"REGEX_PATTERNS": [], "CRITICALITY_MAP": {}}
    except Exception as e:
        logger.error(f"Failed to read rules: {e}")
        raise HTTPException(status_code=500, detail="Failed to read sensitivity rules")

@router.post("/sensitivity_rules")
async def update_sensitivity_rules(rules: dict):
    """
    Update global sensitivity rules.
    TODO: Add RBAC check here once implemented in the system.
    e.g. if not current_user.is_admin: raise HTTPException(403)
    """
    try:
        # Load existing rules first
        if os.path.exists(RULES_FILE):
            with open(RULES_FILE, "r") as f:
                data = json.load(f)
        else:
            data = {"REGEX_PATTERNS": [], "CRITICALITY_MAP": {}}
        
        # Extract criticality and put it in CRITICALITY_MAP
        criticality = rules.pop("criticality", "HIGH")
        ent_type = rules.get("entity_type", "CUSTOM")
        
        if "CRITICALITY_MAP" not in data:
            data["CRITICALITY_MAP"] = {}
        data["CRITICALITY_MAP"][ent_type] = criticality
        
        # Append regex pattern
        data["REGEX_PATTERNS"].append(rules)
            
        with open(RULES_FILE, "w") as f:
            json.dump(data, f, indent=2)
            
        return {"status": "success", "message": "Sensitivity rules updated globally."}
    except Exception as e:
        logger.error(f"Failed to save rules: {e}")
        raise HTTPException(status_code=500, detail="Failed to save sensitivity rules")

# --- WebSocket Hub ---

@router.websocket("/ws/notifications")
async def notifications_websocket(websocket: WebSocket, client_id: str = "anonymous",
                                  workflow_id: str = None, thread_id: str = None):
    """
    Real-time notification hub.
    Clients connect here to receive push updates when notifications are generated
    and live run-status (run_progress) events for a subscribed workflow/thread.
    """
    # Browsers cannot set an Authorization header during a WebSocket handshake.
    # The frontend therefore offers the Keycloak token as the second WebSocket
    # subprotocol, keeping it out of the URL and normal access logs.
    offered_protocols = [
        value.strip()
        for value in websocket.headers.get("sec-websocket-protocol", "").split(",")
        if value.strip()
    ]
    if len(offered_protocols) != 2 or offered_protocols[0] != "agentmesh":
        await websocket.close(code=4401, reason="Not authenticated")
        return

    try:
        websocket.state.current_user = await run_in_threadpool(
            decode_token,
            offered_protocols[1],
        )
    except HTTPException as error:
        await websocket.close(code=4401, reason=str(error.detail))
        return
    except Exception:
        await websocket.close(code=4401, reason="Invalid authentication token")
        return

    await ws_manager.connect(
        websocket,
        client_id,
        thread_id=thread_id,
        workflow_id=workflow_id,
        subprotocol="agentmesh",
    )
    try:
        # Ack so clients know the subscription is registered before they snapshot
        await websocket.send_json({
            "type": "connected",
            "client_id": client_id,
            "workflow_id": workflow_id,
            "thread_id": thread_id,
        })
        while True:
            # Keep connection alive and wait for client messages (optional)
            data = await websocket.receive_text()
            # Respond to pings if needed
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        ws_manager.disconnect(websocket, client_id)
