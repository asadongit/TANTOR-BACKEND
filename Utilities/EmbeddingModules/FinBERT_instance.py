import os
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Singleton manager for HuggingFace embeddings using the yiyanghkust/finbert-tone model.
    Supports 768-dimensional embeddings specialized for financial sentiment analysis.
    """
    _instance = None
    _embedding_model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
        return cls._instance
    
    def get_embedding_model(self):
        """
        Returns the HuggingFace embeddings model (yiyanghkust/finbert-tone, 768 dimensions).
        Loads the model once and caches it for subsequent calls.
        """
        if self._embedding_model is None:
            model_dir = "models/finbert"
            model_name = "yiyanghkust/finbert-tone"
            # Check if model is already downloaded
            if not os.path.exists(model_dir):
                logger.info(f"FinBERT model not found in {model_dir}, will download to cache.")
            else:
                logger.info(f"FinBERT model found in {model_dir}, loading from cache.")
            self._embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                cache_folder=model_dir
            )
            logger.info("FinBERT embedding model loaded successfully")
        return self._embedding_model

# Global instance
embedding_manager = EmbeddingManager()
