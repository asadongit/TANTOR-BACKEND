"""
This module provides a singleton BGE reranker class for reranking document chunks based on a query.
It uses the FlagEmbedding library to compute relevance scores and sort chunks accordingly.
"""

import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

try:
    from FlagEmbedding import FlagReranker
    _DEPENDENCIES_INSTALLED = True
except ImportError as e:
    _DEPENDENCIES_INSTALLED = False
    _IMPORT_ERROR = str(e)
except Exception as e:
    # Handle cases where import exists but fails internally (like the transformers version error)
    _DEPENDENCIES_INSTALLED = False
    _IMPORT_ERROR = str(e)

def _get_bool(name: str, default: bool) -> bool:
    """Retrieves a boolean value from environment variables."""
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).lower() in ("1", "true")

class BGEReranker:
    """
    A singleton class to handle the BGE Reranker model.
    This ensures the model is loaded into memory only once.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(BGEReranker, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = None):
        """Initializes the FlagReranker model."""
        if not _DEPENDENCIES_INSTALLED:
            raise ImportError(f"BGE Reranker dependencies missing: {_IMPORT_ERROR}")
            
        if not hasattr(self, 'model'):
            model_name = model_name or os.environ.get("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
            use_fp16 = _get_bool("RERANKER_FP16", False)
            logger.info(f"Initializing BGE Reranker model: {model_name} (fp16={use_fp16})")
            self.model = FlagReranker(model_name, use_fp16=use_fp16) 
            logger.info("BGE Reranker model initialized.")

    def rerank(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reranks a list of chunks based on a query using the BGE model."""
        if not chunks or not query:
            return []
        
        if not hasattr(self, 'model'):
            return chunks

        chunks_copy = [chunk.copy() for chunk in chunks]
        passages = [chunk.get('content', '') for chunk in chunks_copy]
        scores = self.model.compute_score([[query, passage] for passage in passages])

        for i, chunk in enumerate(chunks_copy):
            chunk['rerank_score'] = float(scores[i])

        sorted_chunks = sorted(chunks_copy, key=lambda x: x.get('rerank_score', 0.0), reverse=True)
        return sorted_chunks

# Create a single, global instance of the reranker to be imported elsewhere
bge_reranker_instance = None
if _DEPENDENCIES_INSTALLED:
    try:
        # We don't initialize on import to avoid slow startup during discovery
        # Instead, it will be initialized on first use if needed
        # bge_reranker_instance = BGEReranker() 
        pass
    except Exception as e:
        logger.warning(f"BGE Reranker entry failed: {e}")
else:
    logger.warning(f"BGE Reranker disabled: {_IMPORT_ERROR}")