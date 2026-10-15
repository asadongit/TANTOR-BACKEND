"""
This module provides semantic text chunking using LangChain's SemanticChunker.
It splits text based on semantic meaning, using embeddings to determine breakpoints.
"""

import os
import re  # Add this import for regex
from langchain_experimental.text_splitter import SemanticChunker
from haystack import Document
import logging
from FinancialStatementAnalysis.EmbeddingModels.bgeM3_instance import embedding_manager

logger = logging.getLogger(__name__)

def chunk_document(text, filename, source):
    """
    Chunks the given text using SemanticChunker with configured parameters.
    Returns a list of Haystack Document objects where content may include a normalized <table_flat> section.
    """
    # Get embedding model from the manager
    embeddings = embedding_manager.get_embedding_model()
    
    # Configure chunker parameters (tuned for financial data)
    buffer_size = int(os.environ.get("SEMANTIC_BUFFER_SIZE", "1"))
    breakpoint_threshold_type = os.environ.get("SEMANTIC_BREAKPOINT_TYPE", "percentile")
    breakpoint_threshold_amount = float(os.environ.get("SEMANTIC_BREAKPOINT_AMOUNT", "85.0"))  # Lowered for finer chunks
    min_chunk_size = int(os.environ.get("SEMANTIC_MIN_CHUNK_SIZE", "200"))  # Increased for financial coherence
    sentence_split_regex = os.environ.get("SEMANTIC_SENTENCE_REGEX", r'(?<=[.!?])\s+')
    
    chunker = SemanticChunker(
        embeddings=embeddings,
        buffer_size=buffer_size,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
        min_chunk_size=min_chunk_size,
        sentence_split_regex=sentence_split_regex
    )
    
    # Create LangChain documents
    langchain_docs = chunker.create_documents([text])
    
    # Convert to Haystack Documents (no postprocessing)
    chunks = []
    for doc in langchain_docs:
        raw = doc.page_content
        haystack_doc = Document(
            content=raw,
            meta={"filename": filename, "source": source}
        )
        chunks.append(haystack_doc)
    
    logger.info(f"  Created {len(chunks)} semantic text chunks.")
    return chunks
