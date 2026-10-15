"""
Toxicity scoring for AgentMesh nodes.

Uses Detoxify (Unitary) — a thin wrapper over HuggingFace models distilled for
the Jigsaw toxic-comment task. The model instance is loaded lazily once and
cached, and scoring runs on a worker thread so it never blocks the async event
loop or a workflow node.

Score scale: 0.0 (benign) -> 1.0 (toxic), per class:
    toxicity, severe_toxicity, obscene, threat, insult, identity_attack

Pipeline:
    wrap_with_error_handling()  ->  schedule_toxicity_score()  (fire-and-forget)
        -> score_toxicity_async() -> record_toxicity_score()  (Langfuse score)
"""

import asyncio
import logging
import threading
from typing import Any, Dict, Optional

from Apps.AgentMesh.Config import settings

logger = logging.getLogger(__name__)

_MODEL: Any = None
_MODEL_NAME: Optional[str] = None
_MODEL_LOCK = threading.Lock()

# Strong refs for background tasks so they aren't GC'd mid-flight.
_background_tasks: set = set()


def warm_up(model_name: Optional[str] = None) -> None:
    """Preload the Detoxify model so the first workflow run doesn't hit the
    scoring timeout while the model is still loading (cold start ~5-10s).

    Safe to call at startup from any thread (main thread or lifespan); the
    model is cached globally and guarded by _MODEL_LOCK.
    """
    if not settings.TOXICITY_ENABLED:
        return
    model_name = model_name or settings.TOXICITY_MODEL
    try:
        with _MODEL_LOCK:
            _load_model(model_name)
        logger.info("Toxicity model preloaded at startup.")
    except Exception as e:
        logger.warning(f"Toxicity model warm-up failed: {e}")


def _load_model(model_name: str):
    """Lazily load (and cache) the Detoxify model. Callers hold _MODEL_LOCK."""
    global _MODEL, _MODEL_NAME
    if _MODEL is None or _MODEL_NAME != model_name:
        from detoxify import Detoxify

        _MODEL = Detoxify(model_name)
        _MODEL_NAME = model_name
        logger.info(f"Toxicity model '{model_name}' loaded.")
    return _MODEL


def _score_sync(text: str, model_name: str, max_chars: int) -> Dict[str, float]:
    """Run the Detoxify model synchronously (call inside a worker thread)."""
    # Single source of truth for score normalization lives in Observer
    # (`_first_score_value`); importing it here avoids drift between the two
    # modules that both rounded Detoxify's scalar-or-array outputs.
    from Apps.AgentMesh.Utilities.Observer import _first_score_value

    text = (text or "").strip()
    if not text:
        return {}
    text = text[:max_chars]
    with _MODEL_LOCK:
        model = _load_model(model_name)
        scores = model.predict(text)
    result = {}
    for k, v in (scores or {}).items():
        result[k] = round(_first_score_value(v), 4)
    return result


def score_toxicity(text: str, model_name: Optional[str] = None,
                   max_chars: Optional[int] = None) -> Dict[str, float]:
    """Synchronous convenience scorer (blocking — use async in hot paths)."""
    model_name = model_name or settings.TOXICITY_MODEL
    max_chars = max_chars or settings.TOXICITY_MAX_CHARS
    return _score_sync(text, model_name, max_chars)


async def score_toxicity_async(text: str, model_name: Optional[str] = None,
                               max_chars: Optional[int] = None,
                               timeout: Optional[float] = None) -> Dict[str, float]:
    """Score text on a worker thread, bounded by a timeout. Never raises."""
    text = (text or "").strip()
    if not text:
        return {}
    model_name = model_name or settings.TOXICITY_MODEL
    max_chars = max_chars or settings.TOXICITY_MAX_CHARS
    timeout = settings.TOXICITY_TIMEOUT_SEC if timeout is None else timeout
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _score_sync, text, model_name, max_chars),
            timeout=timeout,
        )
    except Exception as e:
        logger.warning(f"Toxicity scoring failed: {e}")
        return {}


def schedule_toxicity_score(thread_id: str, node_id: str, node_type: str, text: str) -> None:
    """Fire-and-forget toxicity scoring for a node's output.

    Runs the model on a worker thread and logs the score to Langfuse without
    ever blocking the node or the event loop. No-op when toxicity scoring is
    disabled, the text is empty, or no running event loop is available.
    """
    if not settings.TOXICITY_ENABLED or not text or not str(text).strip():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def _score_and_record():
        scores = await score_toxicity_async(str(text))
        if not scores:
            return
        try:
            from Apps.AgentMesh.Utilities.Observer import record_toxicity_score

            record_toxicity_score(thread_id, node_id, node_type, scores, settings.TOXICITY_THRESHOLD)
        except Exception as e:
            logger.debug(f"Toxicity record failed for node '{node_id}': {e}")

    task = loop.create_task(_score_and_record())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
