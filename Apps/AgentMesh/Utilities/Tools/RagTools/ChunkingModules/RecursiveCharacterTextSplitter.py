"""
This module provides recursive character-based text chunking using LangChain's RecursiveCharacterTextSplitter.
It splits text hierarchically by characters, starting from larger units like paragraphs.
"""

import os
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

def get_document_splitter(chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Returns a configured LangChain RecursiveCharacterTextSplitter instance.
    """
    separators = ["\n\n", "\n", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    return splitter

def chunk_document(text: str, filename: str, source: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Chunks the given text using LangChain's RecursiveCharacterTextSplitter.
    Returns a list of LangChain Document objects.
    """
    splitter = get_document_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # LangChain Splitter returns Document objects if using create_documents
    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{"filename": filename, "source": source}]
    )
    
    logger.info(f"Created {len(chunks)} recursive character-based text chunks.")
    return chunks