import os
import logging
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

def _to_pylist(arr):
    """Convert array-like to list of floats."""
    return arr.tolist() if hasattr(arr, 'tolist') else list(arr)

class NvidiaNVEmbedV2Embeddings:
    """
    LangChain-compatible embeddings wrapper for NVIDIA NV-Embed-v2.
    Supports 2048-dimensional embeddings with query-specific instructions and passage encoding.

    Provides:
      - embed_query(text: str) -> List[float]
      - embed_documents(texts: List[str]) -> List[List[float]]

    Behavior (per model card):
      - Queries use instruction prompt + EOS token.
      - Passages use EOS token only (no instruction).
      - Embeddings are normalized.
    """

    def __init__(
        self,
        model_name: str = None,
        task_instruction: str = None,
        batch_size: int = None,
    ):
        self.model_name = model_name or os.environ.get("NVEMBED_MODEL_NAME", "nvidia/NV-Embed-v2")
        self.task_instruction = task_instruction or os.environ.get(
            "NVEMBED_TASK_INSTRUCT",
            "Given a question, retrieve passages that answer the question"
        )
        self.batch_size = int(os.environ.get("NVEMBED_BATCH_SIZE", str(batch_size or 16)))

        # Device/dtype setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else (
            torch.float16 if self.device == "cuda" else torch.float32
        )

        logger.info(f"Loading SentenceTransformer model '{self.model_name}' on {self.device} ({self.dtype}) ...")
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True, device=self.device)
        # Max seq length and tokenizer settings per model docs
        try:
            self.model.max_seq_length = 32768
        except Exception:
            pass
        try:
            self.model.tokenizer.padding_side = "right"
        except Exception:
            pass

        # EOS token (if missing, treat as empty string)
        try:
            self.eos_token = self.model.tokenizer.eos_token or ""
        except Exception:
            self.eos_token = ""

        # Query prompt prefix
        self.query_prefix = f"Instruct: {self.task_instruction}\nQuery: "

        logger.info("NV-Embed-v2 embeddings initialized.")

    def _add_eos(self, texts: List[str]) -> List[str]:
        if not self.eos_token:
            return texts
        return [t + self.eos_token for t in texts]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Encode passages (no instruction), add EOS, normalize embeddings (2048 dimensions).
        """
        if not texts:
            return []
        inputs = self._add_eos([t if isinstance(t, str) else str(t) for t in texts])
        embs = self.model.encode(
            inputs,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            device=self.device,
            dtype=self.dtype,
        )
        # Ensure list-of-lists of floats
        if isinstance(embs, np.ndarray):
            return [ _to_pylist(row) for row in embs ]
        return [ _to_pylist(row) for row in embs]

    def embed_query(self, text: str) -> List[float]:
        """
        Encode query with instruction + EOS, normalize embeddings (2048 dimensions).
        """
        if not isinstance(text, str):
            text = str(text)
        inputs = self._add_eos([text])
        emb = self.model.encode(
            inputs,
            batch_size=1,
            prompt=self.query_prefix,
            normalize_embeddings=True,
            convert_to_numpy=True,
            device=self.device,
            dtype=self.dtype,
        )
        # encode with a single input returns a 1xD array or vector depending on version
        if isinstance(emb, np.ndarray):
            if emb.ndim == 2 and emb.shape[0] == 1:
                return _to_pylist(emb[0])
            return _to_pylist(emb)
        # Fallback for other types
        if hasattr(emb, "__len__") and len(emb) == 1 and hasattr(emb[0], "__len__"):
            return _to_pylist(emb[0])
        return _to_pylist(emb)

class EmbeddingManager:
    """
    Singleton manager for NVIDIA NV-Embed-v2 embeddings.
    Supports 2048-dimensional embeddings via NvidiaNVEmbedV2Embeddings.
    """
    _instance = None
    _embedding_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
        return cls._instance

    def get_embedding_model(self):
        """
        Returns an object with:
          - embed_query(text: str) -> List[float]
          - embed_documents(texts: List[str]) -> List[List[float]]
        Initializes NV-Embed-v2 (2048 dimensions) once and caches it.
        """
        if self._embedding_model is None:
            logger.info("Initializing NV-Embed-v2 embedding model (one-time setup)")
            self._embedding_model = NvidiaNVEmbedV2Embeddings()
        return self._embedding_model

# Global instance (same name as other models for drop-in replace)
embedding_manager = EmbeddingManager()