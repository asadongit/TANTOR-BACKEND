from typing import List, Dict
from langchain_core.documents import Document

def reciprocal_rank_fusion(
    results_list: List[List[Document]], 
    k: int = 60, 
    top_n: int = 5
) -> List[Document]:
    """
    Combines multiple lists of documents using Reciprocal Rank Fusion (RRF).
    
    Formula: score = sum(1 / (k + rank))
    
    Args:
        results_list: A list of lists of Documents (e.g., [sim_docs, kw_docs])
        k: Smoothing constant (default 60 as per standard RRF)
        top_n: Number of documents to return
    """
    fused_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}
    
    for docs in results_list:
        for rank, doc in enumerate(docs):
            # Use page_content as unique identifier (or source if available)
            doc_id = doc.page_content
            doc_map[doc_id] = doc
            
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
            
            # Add to score: 1 / (k + rank + 1)
            fused_scores[doc_id] += 1.0 / (k + rank + 1)
            
    # Sort documents by fused scores in descending order
    reranked_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    
    # Return the top N documents
    return [doc_map[doc_id] for doc_id in reranked_ids[:top_n]]
