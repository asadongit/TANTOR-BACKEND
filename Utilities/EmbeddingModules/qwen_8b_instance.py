import os
import logging
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings  # use the maintained package
from dotenv import load_dotenv

load_dotenv()
OLLAMA_API_H100 = os.getenv("OLLAMA_API_H100")    

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Singleton manager for Ollama embeddings using the dengcao/Qwen3-Embedding-8B:F16 model.
    Supports 768-dimensional embeddings for high-quality text representations.
    """
    _instance = None
    _embedding_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
        return cls._instance

    def get_embedding_model(self):
        """
        Returns a LangChain-compatible embeddings object:
          - embed_query(text: str) -> List[float]
          - embed_documents(texts: List[str]) -> List[List[float]]
        Loads Ollama 'dengcao/Qwen3-Embedding-8B:F16' from base_url (768 dimensions).
        """
        if self._embedding_model is None:
            logger.info("Initializing Ollama embedding model (one-time setup)")
            self._embedding_model = OllamaEmbeddings(
                model="dengcao/Qwen3-Embedding-8B:F16",
                base_url=OLLAMA_API_H100
            )
            logger.info("Ollama dengcao/Qwen3-Embedding-8B:F16 embeddings loaded successfully.")
        return self._embedding_model

embedding_manager = EmbeddingManager()