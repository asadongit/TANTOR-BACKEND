"""
This module provides recursive character-based text chunking using LangChain's RecursiveCharacterTextSplitter.
It splits text hierarchically by characters, starting from larger units like paragraphs.
"""

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from haystack import Document # Added import for Haystack Document
import logging # Added import for logging

logger = logging.getLogger(__name__) # Added logger initialization

def get_document_splitter():
    """
    Returns a configured LangChain RecursiveCharacterTextSplitter instance.
    Reads configuration from environment variables.
    """
    chunk_size = int(os.environ.get("CHUNK_SIZE", "1000"))  # ~150 words (adjust based on char count)
    chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", "200"))  # ~30 words overlap
    separators = ["\n\n", "\n", " ", ""]  # Recursive split by paragraphs, lines, spaces, chars

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    return splitter

def chunk_document(text, filename, source):
    """
    Chunks the given text using LangChain's RecursiveCharacterTextSplitter.
    Returns a list of Haystack Document objects.
    """
    splitter = get_document_splitter()
    
    langchain_docs = splitter.create_documents([text])
    
    chunks = []
    for doc in langchain_docs:
        haystack_doc = Document(
            content=doc.page_content,
            meta={"filename": filename, "source": source}
        )
        chunks.append(haystack_doc)
    
    logger.info(f"  Created {len(chunks)} recursive character-based text chunks.")
    return chunks