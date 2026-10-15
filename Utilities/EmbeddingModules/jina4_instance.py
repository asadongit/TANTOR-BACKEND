from langchain_community.embeddings import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Singleton manager for HuggingFace embeddings using the jinaai/jina-embeddings-v4 model.
    Supports 512-dimensional embeddings (compressed from 2048 for efficiency).
    """
    _instance = None
    _embedding_model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
        return cls._instance
    
    def get_embedding_model(self):
        """
        Returns the HuggingFace embeddings model (jinaai/jina-embeddings-v4, 512 dimensions).
        Loads the model once with dense encoding and caches it.
        """
        if self._embedding_model is None:
            logger.info("Initializing HuggingFace embedding model (jinaai/jina-embeddings-v4)")
            self._embedding_model = HuggingFaceEmbeddings(
                model_name="jinaai/jina-embeddings-v4",
                model_kwargs={"trust_remote_code": True},
                encode_kwargs={
                    "embedding_type": "dense",  # best for laptop setup
                    "dim": 512                  # compress from 2048 → 512 (MRL)
                }
            )
            logger.info("Embedding model loaded successfully (dense, 512D)")
        return self._embedding_model

# Global instance
embedding_manager = EmbeddingManager()
