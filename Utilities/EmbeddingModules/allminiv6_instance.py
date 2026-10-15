import os
from langchain_huggingface import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Singleton manager for HuggingFace embeddings using the all-MiniLM-L6-v2 model.
    Supports 384-dimensional embeddings for efficient semantic similarity tasks.
    """
    _instance = None
    _embedding_model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
        return cls._instance
    
    def get_embedding_model(self):
        """
        Returns the HuggingFace embeddings model (all-MiniLM-L6-v2, 384 dimensions).
        Loads the model once and caches it for subsequent calls.
        """
        if self._embedding_model is None:
            model_dir = "models/all-MiniLM-L6-v2"
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            # Check if model is already downloaded
            if not os.path.exists(model_dir):
                logger.info(f"Model not found in {model_dir}, will download to cache.")
            else:
                logger.info(f"Model found in {model_dir}, loading from cache.")
            self._embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                cache_folder=model_dir
            )
            logger.info("Embedding model loaded successfully")
        return self._embedding_model

# Global instance
embedding_manager = EmbeddingManager()
