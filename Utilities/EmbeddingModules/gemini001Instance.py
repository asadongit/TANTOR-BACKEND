import os
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Singleton manager for Google Gemini embeddings using the gemini-embedding-001 model.
    Supports 768-dimensional embeddings for multimodal and text tasks.
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
        Loads the Google Gemini embedding model (gemini-embedding-001, 768 dimensions).
        """
        if self._embedding_model is None:
            logger.info("Initializing Google Gemini embedding model (one-time setup)")
            self._embedding_model = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            logger.info("Google Gemini embeddings loaded successfully.")
        return self._embedding_model

# Global instance
embedding_manager = EmbeddingManager()
