"""Local Gemma 300m embeddings manager (SentenceTransformers, dim=768)."""

import os
import logging
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Singleton manager for local Gemma 300m embeddings via SentenceTransformers.
    Supports 768-dimensional embeddings for efficient local inference.
    """
    _instance = None
    _embedding_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
        return cls._instance

    def get_embedding_model(self):
        """
        Returns a LangChain-compatible embeddings object with:
          - embed_query(text: str) -> List[float]
          - embed_documents(texts: List[str]) -> List[List[float]]
        Loads the local SentenceTransformers model at models/embeddinggemma (768 dimensions).
        """
        if self._embedding_model is None:
            model_dir = "models/embeddinggemma"
            if not os.path.exists(model_dir):
                logger.error(f"Model directory {model_dir} does not exist. Please download/copy the model first.")
                raise FileNotFoundError(f"{model_dir} not found.")
            logger.info(f"Loading Gemma 300m embeddings locally from {model_dir} (dim=768)")

            self._embedding_model = HuggingFaceEmbeddings(
                model_name=model_dir,
            )
            logger.info("Gemma embeddings loaded successfully.")
        return self._embedding_model

embedding_manager = EmbeddingManager()

