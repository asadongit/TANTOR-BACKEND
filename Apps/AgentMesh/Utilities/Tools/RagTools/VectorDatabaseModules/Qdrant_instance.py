import os
import logging
from typing import List, Optional, Dict
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class QdrantManager:
    """
    Singleton manager for Qdrant connections.
    Caches the client and vectorstore instances to prevent redundant connections.
    """

    _instance = None
    _client = None
    _vectorstores: Dict[str, QdrantVectorStore] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QdrantManager, cls).__new__(cls)
        return cls._instance

    def _get_client(self):
        """Initializes and returns the Qdrant client based on environment variables."""
        if self._client is None:
            host = os.environ.get("QDRANT_HOST")
            port = os.environ.get("QDRANT_PORT")

            if host and port:
                logger.info(f"Connecting to remote Qdrant at {host}:{port}")
                self._client = QdrantClient(host=host, port=int(port))
            else:
                path = os.environ.get("QDRANT_PATH", "./VectorDatabaseData/qdrant_db")
                logger.info(f"Using local Qdrant at {path}")
                self._client = QdrantClient(path=path)
        return self._client

    def list_collections(self) -> List[str]:
        """Returns a list of all collection names in Qdrant."""
        client = self._get_client()
        cols = client.get_collections().collections
        return [c.name for c in cols]

    def get_collection_details(
        self, collection_name: str, limit: int = 50, offset: int = 0
    ) -> dict:
        """Returns stats, vector size, distance metric, total points, and scroll-paginated chunks for a Qdrant collection."""
        client = self._get_client()
        if not client.collection_exists(collection_name):
            return {
                "name": collection_name,
                "exists": False,
                "total_chunks": 0,
                "files": [],
                "chunks": [],
            }

        info = client.get_collection(collection_name)
        total_chunks = info.points_count or 0

        vector_size = None
        distance = "COSINE"
        if hasattr(info.config, "params") and hasattr(info.config.params, "vectors"):
            vec_params = info.config.params.vectors
            if hasattr(vec_params, "size"):
                vector_size = vec_params.size
            if hasattr(vec_params, "distance"):
                distance = str(vec_params.distance).upper()

        # Scroll payloads with numeric offset support (same pattern as Chroma: fetch range then slice)
        fetch_size = min(offset + limit, 10000)
        if fetch_size > 0:
            records, _ = client.scroll(
                collection_name=collection_name,
                limit=fetch_size,
                with_payload=True,
                with_vectors=False,
            )
            records = records[offset : offset + limit]
        else:
            records = []

        files_map = {}
        sample_chunks = []
        for r in records:
            p = r.payload or {}
            fname = p.get("filename") or p.get("source") or "Unknown File"
            if fname not in files_map:
                files_map[fname] = {
                    "filename": os.path.basename(fname),
                    "chunks": 0,
                    "source": fname,
                }
            files_map[fname]["chunks"] += 1

            raw_text = p.get("page_content") or p.get("text")
            if not raw_text:
                import json

                try:
                    raw_text = json.dumps(p, indent=2)
                except Exception:
                    raw_text = str(p)

            sample_chunks.append(
                {
                    "id": str(r.id),
                    "filename": os.path.basename(fname),
                    "content": raw_text,
                    "metadata": p,
                }
            )

        return {
            "name": collection_name,
            "exists": True,
            "total_chunks": total_chunks,
            "vector_size": vector_size or 384,
            "distance": distance,
            "unique_files_count": len(files_map),
            "files": list(files_map.values()),
            "chunks": sample_chunks,
        }

    def get_vectorstore(
        self, collection_name: str, embedding_model: Embeddings
    ) -> QdrantVectorStore:
        """Returns a cached or new LangChain Qdrant vectorstore instance, creating the collection if missing."""
        cache_key = f"{collection_name}_{id(embedding_model)}"
        if cache_key not in self._vectorstores:
            client = self._get_client()
            if not client.collection_exists(collection_name):
                # Infer vector size from embedding model if possible (defaulting to 384)
                sample_vec = embedding_model.embed_query("test sample")
                vector_size = len(sample_vec) if sample_vec else 384
                from qdrant_client.models import VectorParams, Distance

                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size, distance=Distance.COSINE
                    ),
                )
                logger.info(
                    f"Auto-created Qdrant collection '{collection_name}' with vector size {vector_size}."
                )
            self._vectorstores[cache_key] = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embedding_model,
            )
        return self._vectorstores[cache_key]


# Singleton instance
qdrant_manager = QdrantManager()


def get_qdrant_vectorstore(
    collection_name: str, embedding_model: Embeddings
) -> QdrantVectorStore:
    """Legacy wrapper for backward compatibility."""
    return qdrant_manager.get_vectorstore(collection_name, embedding_model)


def store_documents(documents: List, collection_name: str, embedding_model: Embeddings, ids: Optional[List[str]] = None):
    """Stores LangChain Document objects in Qdrant. When `ids` are supplied, entries
    with existing ids are upserted (overwritten) rather than duplicated."""
    vectorstore = qdrant_manager.get_vectorstore(collection_name, embedding_model)
    if ids:
        ids = vectorstore.add_documents(documents, ids=ids)
    else:
        ids = vectorstore.add_documents(documents)
    logger.info(
        f"Stored {len(documents)} documents in Qdrant collection '{collection_name}'."
    )
    return ids


# Cache for BM25 retrievers
_bm25_cache = {}


def _build_qdrant_filter(where: Optional[dict]):
    """Convert a metadata `where` dict into a Qdrant Filter.

    langchain-qdrant wraps document metadata under a nested `metadata.*` payload
    key, so filters must reference `metadata.<field>` to actually match.
    """
    if not where:
        return None
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    conditions = [FieldCondition(key=f"metadata.{k}", match=MatchValue(value=v)) for k, v in where.items()]
    return Filter(must=conditions)


def prune_qdrant(collection_name: str, embedding_model: Embeddings, where: Optional[dict] = None, keep: int = 200) -> int:
    """Delete the OLDEST entries (by payload `timestamp`) matching `where`, keeping
    only the `keep` most recent. Returns the number of deleted entries."""
    client = qdrant_manager._get_client()
    qfilter = _build_qdrant_filter(where)
    records, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=qfilter,
        limit=100000,
        with_payload=True,
        with_vectors=False,
    )
    if len(records) <= keep:
        return 0
    records.sort(key=lambda r: ((r.payload or {}).get("metadata") or {}).get("timestamp", 0))
    to_delete = [r.id for r in records[: len(records) - keep]]
    if to_delete:
        client.delete(collection_name=collection_name, points_selector=to_delete)
    return len(to_delete)


def query_qdrant(
    query: str,
    collection_name: str,
    embedding_model: Embeddings,
    k: int = 5,
    threshold: Optional[float] = None,
    search_type: str = "similarity",
    where: Optional[dict] = None,
):
    """Queries the Qdrant vectorstore for relevant documents. `where` restricts
    results to documents matching the given metadata filter."""
    vectorstore = qdrant_manager.get_vectorstore(collection_name, embedding_model)
    qfilter = _build_qdrant_filter(where)

    def _filter_docs(docs, wh):
        if not wh:
            return docs
        return [d for d in docs if all((d.metadata or {}).get(k) == v for k, v in wh.items())]

    if search_type == "similarity":
        return vectorstore.similarity_search(query, k=k, filter=qfilter)

    # For Keyword or Hybrid, we use BM25
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document

    # Simple cache key
    cache_key = f"{collection_name}_bm25"

    if cache_key not in _bm25_cache:
        logger.info(f"Building BM25 index for Qdrant collection '{collection_name}'...")
        # Note: Qdrant doesn't have a simple 'get all' like Chroma in LangChain's wrapper easily
        # but we can try to scroll or just fetch a large number of docs.
        # However, for simplicity and consistency with the Chroma implementation:
        client = qdrant_manager._get_client()
        # Fetch up to 10k docs for BM25 index (limit can be adjusted)
        records, next_page = client.scroll(
            collection_name=collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )

        if not records:
            _bm25_cache[cache_key] = None
        else:
            docs = [
                Document(
                    page_content=r.payload.get("page_content", ""), metadata=r.payload
                )
                for r in records
                if r.payload
            ]
            _bm25_cache[cache_key] = BM25Retriever.from_documents(docs)

    retriever = _bm25_cache[cache_key]

    kw_docs = []
    if retriever:
        retriever.k = k
        kw_docs = retriever.invoke(query)
    kw_docs = _filter_docs(kw_docs, where)

    if search_type == "keyword":
        return kw_docs

    if search_type == "hybrid":
        sim_docs = vectorstore.similarity_search(
            query, k=k * 2, filter=qfilter
        )  # Fetch more for better fusion

        # Merge using Reciprocal Rank Fusion (RRF)
        from Apps.AgentMesh.Utilities.Tools.RagTools.fusion_utils import (
            reciprocal_rank_fusion,
        )

        return reciprocal_rank_fusion([sim_docs, kw_docs], top_n=k)

    return vectorstore.similarity_search(query, k=k, filter=qfilter)


def query_qdrant_with_recency(
    query: str,
    collection_name: str,
    embedding_model: Embeddings,
    k: int = 5,
    threshold: Optional[float] = None,
    where: Optional[dict] = None,
    recency_decay: float = 0.3,
    decay_halflife_hours: float = 168.0,
):
    """Recency-aware (hybrid) retrieval for vector memory. Mirrors
    `query_chroma_with_recency` semantics using Qdrant's native cosine score
    (higher = more similar)."""
    import math
    import time

    vectorstore = qdrant_manager.get_vectorstore(collection_name, embedding_model)
    qfilter = _build_qdrant_filter(where)
    candidate_k = max(k * 4, 20)
    pairs = vectorstore.similarity_search_with_score(query, k=candidate_k, filter=qfilter)

    now = time.time()
    scored = []
    for doc, similarity in pairs:
        sim = float(similarity)  # cosine similarity, higher = better
        if threshold is not None and sim < threshold:
            continue
        meta = doc.metadata or {}
        ts = meta.get("timestamp")
        if isinstance(ts, (int, float)):
            age_h = max(now - float(ts), 0.0) / 3600.0
            recency = math.exp(-age_h / decay_halflife_hours)
        else:
            recency = 0.0
        hybrid = (1.0 - recency_decay) * sim + recency_decay * recency
        scored.append((hybrid, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:k]]
