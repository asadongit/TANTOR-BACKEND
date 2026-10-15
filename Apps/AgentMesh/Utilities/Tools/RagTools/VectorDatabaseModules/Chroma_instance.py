import os
import logging
from typing import List, Optional, Dict
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

class ChromaManager:
    """
    Singleton manager for Chroma connections.
    Caches the client and vectorstore instances to prevent redundant connections.
    """
    _instance = None
    _client = None
    _vectorstores: Dict[str, Chroma] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChromaManager, cls).__new__(cls)
        return cls._instance

    def _get_client(self):
        """Initializes and returns the Chroma client based on environment variables."""
        if self._client is None:
            host = os.environ.get("CHROMA_HOST")
            port = os.environ.get("CHROMA_PORT")
            
            if host and port:
                import chromadb
                logger.info(f"Connecting to remote Chroma at {host}:{port}")
                self._client = chromadb.HttpClient(host=host, port=int(port))
            else:
                import chromadb
                path = os.environ.get("CHROMA_PATH", "./chroma_db")
                logger.info(f"Using local Chroma at {path}")
                self._client = chromadb.PersistentClient(path=path)
        return self._client

    def list_collections(self) -> List[str]:
        """Returns a list of all collection names in Chroma."""
        client = self._get_client()
        cols = client.list_collections()
        return [col.name if hasattr(col, 'name') else str(col) for col in cols]

    def get_collection_details(self, collection_name: str, limit: int = 50, offset: int = 0) -> dict:
        """Returns stats, vector count, and unique ingested files for a Chroma collection."""
        client = self._get_client()
        try:
            col = client.get_collection(collection_name)
        except Exception:
            return {"name": collection_name, "exists": False, "total_chunks": 0, "files": []}

        data = col.get()
        ids = data.get("ids") or []
        total_chunks = len(ids)

        # Derive the real embedding dimension from stored vectors (fallback 384)
        vector_size = 384
        try:
            sample = col.get(include=["embeddings"], limit=1)
            sample_embs = sample.get("embeddings") or []
            if sample_embs and sample_embs[0] is not None:
                vector_size = len(sample_embs[0])
        except Exception:
            pass

        raw_metadatas = data.get("metadatas")
        metadatas = raw_metadatas if raw_metadatas is not None else [None] * total_chunks

        raw_documents = data.get("documents")
        documents = raw_documents if raw_documents is not None else [""] * total_chunks

        files_map = {}
        all_chunks = []
        for idx, (m, doc, c_id) in enumerate(zip(metadatas, documents, ids)):
            m_dict = m or {}
            fname = m_dict.get("filename") or m_dict.get("source") or "Unknown File"
            if fname not in files_map:
                files_map[fname] = {"filename": os.path.basename(fname), "chunks": 0, "source": fname}
            files_map[fname]["chunks"] += 1

            raw_text = doc or m_dict.get("page_content") or m_dict.get("text")
            if not raw_text:
                import json
                try:
                    raw_text = json.dumps(m_dict, indent=2)
                except Exception:
                    raw_text = str(m_dict)

            all_chunks.append({
                "id": str(c_id),
                "filename": os.path.basename(fname),
                "content": raw_text,
                "metadata": m_dict
            })

        paginated_chunks = all_chunks[offset:offset + limit]

        return {
            "name": collection_name,
            "exists": True,
            "total_chunks": total_chunks,
            "vector_size": vector_size,
            "distance": "COSINE",
            "unique_files_count": len(files_map),
            "files": list(files_map.values()),
            "chunks": paginated_chunks
        }

    def get_vectorstore(self, collection_name: str, embedding_model: Embeddings) -> Chroma:
        """Returns a cached or new LangChain Chroma vectorstore instance."""
        cache_key = f"{collection_name}_{id(embedding_model)}"
        if cache_key not in self._vectorstores:
            client = self._get_client()
            self._vectorstores[cache_key] = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=embedding_model
            )
        return self._vectorstores[cache_key]

# Singleton instance
chroma_manager = ChromaManager()

def get_chroma_vectorstore(collection_name: str, embedding_model: Embeddings) -> Chroma:
    """Legacy wrapper for backward compatibility."""
    return chroma_manager.get_vectorstore(collection_name, embedding_model)

def store_documents(documents: List, collection_name: str, embedding_model: Embeddings, ids: Optional[List[str]] = None):
    """Stores LangChain Document objects in Chroma. When `ids` are supplied, entries
    with existing ids are upserted (overwritten) rather than duplicated."""
    vectorstore = chroma_manager.get_vectorstore(collection_name, embedding_model)
    if ids:
        ids = vectorstore.add_documents(documents, ids=ids)
    else:
        ids = vectorstore.add_documents(documents)
    logger.info(f"Stored {len(documents)} documents in Chroma collection '{collection_name}'.")
    return ids

def prune_chroma(collection_name: str, embedding_model: Embeddings, where: Optional[dict] = None, keep: int = 200) -> int:
    """Delete the OLDEST entries (by metadata `timestamp`) matching `where`, keeping
    only the `keep` most recent. Returns the number of deleted entries."""
    vectorstore = chroma_manager.get_vectorstore(collection_name, embedding_model)
    query_kwargs = {"include": ["metadatas"]}
    if where:
        query_kwargs["where"] = where
    result = vectorstore._collection.get(**query_kwargs)
    ids = result.get("ids") or []
    metadatas = result.get("metadatas") or []
    if len(ids) <= keep:
        return 0
    paired = sorted(zip(ids, metadatas), key=lambda x: (x[1] or {}).get("timestamp", 0))
    to_delete = [i for i, _ in paired[: len(ids) - keep]]
    if to_delete:
        vectorstore.delete(ids=to_delete)
    return len(to_delete)

# Cache for BM25 retrievers to avoid rebuilding from all documents on every query
_bm25_cache = {}

def query_chroma(query: str, collection_name: str, embedding_model: Embeddings, k: int = 5, threshold: Optional[float] = None, search_type: str = "similarity", where: Optional[dict] = None):
    """Queries the Chroma vectorstore for relevant documents. `where` restricts
    results to documents matching the given metadata filter."""
    vectorstore = chroma_manager.get_vectorstore(collection_name, embedding_model)
    
    def _filter_docs(docs, wh):
        if not wh:
            return docs
        return [d for d in docs if all((d.metadata or {}).get(k) == v for k, v in wh.items())]
    
    if search_type == "similarity":
        return vectorstore.similarity_search(query, k=k, filter=where)
    
    # For Keyword or Hybrid, we use BM25
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document
    
    # Simple cache key
    cache_key = f"{collection_name}_bm25"
    
    # Note: In a production environment with frequent updates, this cache should be invalidated
    # when new documents are added. For now, we rebuild if not in cache.
    if cache_key not in _bm25_cache:
        logger.info(f"Building BM25 index for collection '{collection_name}'...")
        all_docs_data = vectorstore.get()
        if not all_docs_data['documents']:
            _bm25_cache[cache_key] = None
        else:
            docs = [
                Document(page_content=d, metadata=m or {}) 
                for d, m in zip(all_docs_data['documents'], all_docs_data['metadatas'])
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
        # Perform similarity search
        sim_docs = vectorstore.similarity_search(query, k=k*2, filter=where) # Fetch more for better fusion
        
        # Merge using Reciprocal Rank Fusion (RRF)
        from Apps.AgentMesh.Utilities.Tools.RagTools.fusion_utils import reciprocal_rank_fusion
        return reciprocal_rank_fusion([sim_docs, kw_docs], top_n=k)

    return vectorstore.similarity_search(query, k=k, filter=where)


def query_chroma_with_recency(
    query: str,
    collection_name: str,
    embedding_model: Embeddings,
    k: int = 5,
    threshold: Optional[float] = None,
    where: Optional[dict] = None,
    recency_decay: float = 0.3,
    decay_halflife_hours: float = 168.0,
):
    """Recency-aware (hybrid) retrieval for vector memory.

    Candidates are fetched by pure cosine similarity, hard-filtered by the raw
    similarity `threshold`, then re-ranked by
    `(1 - recency_decay) * sim + recency_decay * recency` so genuinely similar
    memories that are more recent rank higher. Memories without a metadata
    `timestamp` get recency 0 (never favoured by the recency term).
    """
    import math
    import time

    vectorstore = chroma_manager.get_vectorstore(collection_name, embedding_model)
    candidate_k = max(k * 4, 20)
    pairs = vectorstore.similarity_search_with_score(query, k=candidate_k, filter=where)

    now = time.time()
    scored = []
    for doc, distance in pairs:
        # Chroma's cosine distance is `2 * (1 - cosine_similarity)`, so
        # dividing by 2 recovers the true cosine similarity in [0, 1].
        sim = 1.0 - float(distance) / 2.0
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

