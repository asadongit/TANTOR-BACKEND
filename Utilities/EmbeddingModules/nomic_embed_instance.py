from langchain_community.embeddings import OllamaEmbeddings
import logging

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Singleton manager for Ollama embeddings using the nomic-embed-text:latest model.
    Supports 768-dimensional embeddings for general-purpose text encoding.
    """
    _instance = None
    _embedding_model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
        return cls._instance
    
    def get_embedding_model(self):
        """
        Returns the Ollama embeddings model (nomic-embed-text:latest, 768 dimensions).
        Loads the model once and caches it for subsequent calls.
        """
        if self._embedding_model is None:
            logger.info("Initializing Ollama embedding model (one-time setup)")
            self._embedding_model = OllamaEmbeddings(
                model="nomic-embed-text:latest",
                base_url="http://140.238.39.174:31954/"
            )
            logger.info("Ollama Jina embeddings loaded successfully")
        return self._embedding_model

# Global instance
embedding_manager = EmbeddingManager()