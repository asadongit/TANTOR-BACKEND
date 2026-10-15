from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
import os  # Added import for os.path.exists

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Singleton manager for HuggingFace embeddings using the jinaai/jina-embeddings-v2-small-en model.
    Supports 512-dimensional embeddings for lightweight text encoding.
    """
    _instance = None
    _embedding_model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
        return cls._instance
    
    def get_embedding_model(self):
        """
        Returns the HuggingFace embeddings model (jinaai/jina-embeddings-v2-small-en, 512 dimensions).
        Loads the model once and caches it for subsequent calls.
        """
        if self._embedding_model is None:
            model_dir = "models/jina-v2"  # Added model directory for local caching
            model_name = "jinaai/jina-embeddings-v2-small-en"
            # Check if model is already downloaded
            if not os.path.exists(model_dir):
                logger.info(f"Model not found in {model_dir}, will download to cache.")
            else:
                logger.info(f"Model found in {model_dir}, loading from cache.")
            logger.info("Initializing HuggingFace embedding model (one-time setup)")
            self._embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"trust_remote_code": True},  # Added trust_remote_code
                cache_folder=model_dir  # Added cache_folder to download/load from models/jina
            )
            logger.info("Embedding model loaded successfully")
        return self._embedding_model

# Global instance
embedding_manager = EmbeddingManager()