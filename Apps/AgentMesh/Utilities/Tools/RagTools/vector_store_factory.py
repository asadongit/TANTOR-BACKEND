from typing import List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

def query_vector_store(
    query: str, 
    vector_store_type: str, 
    embedding_model: Embeddings, 
    collection_name: str = "default", 
    k: int = 5,
    threshold: Optional[float] = None,
    search_type: str = "similarity",
    where: Optional[dict] = None,
    recency_decay: Optional[float] = None,
    decay_halflife_hours: float = 168.0
) -> List[Document]:
    """
    Factory function to query the appropriate vector store.
    `where` restricts results to documents matching the given metadata filter.

    When `recency_decay` is set (used by vector memory), retrieval becomes
    hybrid: candidates are hard-filtered by the raw-similarity `threshold` and
    then re-ranked by `(1 - recency_decay) * sim + recency_decay * recency`
    so more recent memories rank higher among genuinely similar ones.
    """
    if vector_store_type.lower() == "qdrant":
        from Apps.AgentMesh.Utilities.Tools.RagTools.VectorDatabaseModules.Qdrant_instance import query_qdrant, query_qdrant_with_recency
        if recency_decay is not None:
            return query_qdrant_with_recency(query, collection_name, embedding_model, k, threshold, where=where, recency_decay=recency_decay, decay_halflife_hours=decay_halflife_hours)
        return query_qdrant(query, collection_name, embedding_model, k, threshold, search_type, where=where)
    else:
        # Default to Chroma
        from Apps.AgentMesh.Utilities.Tools.RagTools.VectorDatabaseModules.Chroma_instance import query_chroma, query_chroma_with_recency
        if recency_decay is not None:
            return query_chroma_with_recency(query, collection_name, embedding_model, k, threshold, where=where, recency_decay=recency_decay, decay_halflife_hours=decay_halflife_hours)
        return query_chroma(query, collection_name, embedding_model, k, threshold, search_type, where=where)

def store_in_vector_store(
    documents: List[Document], 
    vector_store_type: str, 
    embedding_model: Embeddings, 
    collection_name: str = "default",
    ids: Optional[List[str]] = None
):
    """
    Factory function to store documents in the appropriate vector store.
    When `ids` are supplied, entries with existing ids are upserted (overwritten).
    """
    if vector_store_type.lower() == "qdrant":
        from Apps.AgentMesh.Utilities.Tools.RagTools.VectorDatabaseModules.Qdrant_instance import store_documents
        return store_documents(documents, collection_name, embedding_model, ids=ids)
    else:
        # Default to Chroma
        from Apps.AgentMesh.Utilities.Tools.RagTools.VectorDatabaseModules.Chroma_instance import store_documents
        return store_documents(documents, collection_name, embedding_model, ids=ids)

def prune_vector_store(
    vector_store_type: str,
    embedding_model: Embeddings,
    collection_name: str = "default",
    where: Optional[dict] = None,
    keep: int = 200
) -> int:
    """
    Delete the OLDEST entries (by metadata `timestamp`) matching `where`, keeping
    only the `keep` most recent ones. Returns the number of deleted entries.
    """
    if vector_store_type.lower() == "qdrant":
        from Apps.AgentMesh.Utilities.Tools.RagTools.VectorDatabaseModules.Qdrant_instance import prune_qdrant
        return prune_qdrant(collection_name, embedding_model, where=where, keep=keep)
    else:
        from Apps.AgentMesh.Utilities.Tools.RagTools.VectorDatabaseModules.Chroma_instance import prune_chroma
        return prune_chroma(collection_name, embedding_model, where=where, keep=keep)
