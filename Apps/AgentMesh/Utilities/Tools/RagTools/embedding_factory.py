from langchain_core.embeddings import Embeddings
from typing import Optional
import os
import logging
import warnings
import threading

# Suppress HuggingFace Hub and Transformers warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Disable the "unauthenticated requests" warning specifically from huggingface_hub
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
# Suppress the BertModel load report
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Guards the singleton first-load: the underlying managers set `_embedding_model`
# with a plain `is None` check, which is not atomic under concurrent worker threads.
_EMBEDDING_INIT_LOCK = threading.Lock()


def init_embedding_model(model_name: Optional[str] = "all-MiniLM-L6-v2") -> Embeddings:
    """
    Factory function to initialize a LangChain Embedding model.
    Thread-safe: concurrent first calls serialize on `_EMBEDDING_INIT_LOCK` so the
    heavyweight model is loaded exactly once.
    """
    # Handle explicit None from config
    name = model_name or "all-MiniLM-L6-v2"

    with _EMBEDDING_INIT_LOCK:
        if "jina" in name.lower():
            from Apps.AgentMesh.Utilities.Tools.RagTools.EmbeddingModules.jina_instance import embedding_manager
            return embedding_manager.get_embedding_model()
        else:
            # Default to all-MiniLM-L6-v2
            from Apps.AgentMesh.Utilities.Tools.RagTools.EmbeddingModules.allminiv6_instance import embedding_manager
            return embedding_manager.get_embedding_model()
