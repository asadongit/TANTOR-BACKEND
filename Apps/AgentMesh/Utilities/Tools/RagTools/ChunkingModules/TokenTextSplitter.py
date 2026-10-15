"""
This module provides token-based text chunking using LangChain's TokenTextSplitter.
It splits text into fixed-size chunks measured by tokens, with configurable overlap.
"""

import os
try:
    from langchain.text_splitter import TokenTextSplitter
except ImportError:
    from langchain_text_splitters import TokenTextSplitter
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

def chunk_document(text: str, filename: str, source: str, chunk_size: int = 600, chunk_overlap: int = 90):
    """
    Chunks the given text using LangChain's TokenTextSplitter with fixed size (by tokens).
    Returns a list of LangChain Document objects.
    """
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name="cl100k_base"
    )
    
    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{"filename": filename, "source": source}]
    )
    
    logger.info(f"Created {len(chunks)} fixed-size text chunks (by tokens).")
    return chunks