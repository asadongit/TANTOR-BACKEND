"""
This module provides token-based text chunking using LangChain's TokenTextSplitter.
It splits text into fixed-size chunks measured by tokens, with configurable overlap.
"""

import os
from langchain.text_splitter import TokenTextSplitter  # Changed from RecursiveCharacterTextSplitter
from haystack import Document
import logging

logger = logging.getLogger(__name__)

def chunk_document(text, filename, source):
    """
    Chunks the given text using LangChain's TokenTextSplitter with fixed size (by tokens).
    Returns a list of Haystack Document objects.
    """
    chunk_size = int(os.environ.get("CHUNK_SIZE", "600"))  # Default chunk size in tokens; adjust via env var
    chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", "90"))  # Default overlap in tokens; adjust via env var
    
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name=os.environ.get("ENCODING_NAME", "cl100k_base")  # Configurable encoding; adjust via env var
    )
    
    langchain_docs = splitter.create_documents([text])
    
    chunks = []
    for doc in langchain_docs:
        haystack_doc = Document(
            content=doc.page_content,
            meta={"filename": filename, "source": source}
        )
        chunks.append(haystack_doc)
    
    logger.info(f"  Created {len(chunks)} fixed-size text chunks (by tokens).")
    return chunks