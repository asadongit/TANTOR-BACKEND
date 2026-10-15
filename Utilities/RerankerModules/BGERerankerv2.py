"""
This module provides a singleton BGE reranker class for reranking document chunks based on a query.
It uses the FlagEmbedding library to compute relevance scores and sort chunks accordingly.
"""

import os
from FlagEmbedding import FlagReranker
from typing import List, Dict, Any

def _get_bool(name: str, default: bool) -> bool:
    """
    Retrieves a boolean value from environment variables.
    
    Args:
        name (str): The environment variable name.
        default (bool): The default value if the variable is not set.
    
    Returns:
        bool: The boolean value based on the environment variable.
    """
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
        """
        Initializes the FlagReranker model.
        The model is loaded only on the first instantiation.
        """
        if not hasattr(self, 'model'):
            model_name = model_name or os.environ.get("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
            # Enable fp16 only if explicitly requested or CUDA is available
            use_fp16 = _get_bool("RERANKER_FP16", False)
            print(f"Initializing BGE Reranker model: {model_name} (fp16={use_fp16})")
            self.model = FlagReranker(model_name, use_fp16=use_fp16) 
            print("BGE Reranker model initialized.")

    def rerank(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reranks a list of chunks based on a query using the BGE model.
        
        Args:
            query (str): The user query.
            chunks (List[Dict[str, Any]]): A list of chunk dictionaries, each must have a 'content' key.

        Returns:
            List[Dict[str, Any]]: The sorted list of chunks with a new 'rerank_score' key.
        """
        if not chunks or not query:
            return []

        # Create a copy of chunks to avoid modifying the input
        chunks_copy = [chunk.copy() for chunk in chunks]

        # Create pairs of [query, document_content] for the model
        passages = [chunk.get('content', '') for chunk in chunks_copy]
        scores = self.model.compute_score([[query, passage] for passage in passages])

        # Add the new rerank score to each chunk dictionary
        for i, chunk in enumerate(chunks_copy):
            chunk['rerank_score'] = float(scores[i])

        # Sort chunks by the new rerank score in descending order
        sorted_chunks = sorted(chunks_copy, key=lambda x: x.get('rerank_score', 0.0), reverse=True)
        
        return sorted_chunks

# Create a single, global instance of the reranker to be imported elsewhere
bge_reranker_instance = BGEReranker()