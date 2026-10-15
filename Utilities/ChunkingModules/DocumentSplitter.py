"""
This module provides word-based text chunking using Haystack's DocumentSplitter.
It splits text by words, with options for overlap and sentence boundary respect.
"""

import os
import re  # Add this import for regex
from haystack.components.preprocessors import DocumentSplitter
from haystack import Document
import logging

logger = logging.getLogger(__name__)

def chunk_document(text, filename, source):
    """
    Chunks the given text using DocumentSplitter with configured parameters.
    """
    CHUNK_SPLIT_LENGTH = int(os.environ.get("CHUNK_SPLIT_LENGTH", "600"))
    CHUNK_SPLIT_OVERLAP = int(os.environ.get("CHUNK_SPLIT_OVERLAP", "120"))

    
    if CHUNK_SPLIT_OVERLAP >= CHUNK_SPLIT_LENGTH:
        logger.warning(f"Overlap ({CHUNK_SPLIT_OVERLAP}) >= Length ({CHUNK_SPLIT_LENGTH}), adjusting overlap to {CHUNK_SPLIT_LENGTH // 2}")
        CHUNK_SPLIT_OVERLAP = CHUNK_SPLIT_LENGTH // 2
    
    splitter = DocumentSplitter(
        split_by=os.environ.get("CHUNK_SPLIT_BY", "word"),
        split_length=CHUNK_SPLIT_LENGTH,
        split_overlap=CHUNK_SPLIT_OVERLAP,
        respect_sentence_boundary=os.environ.get("CHUNK_RESPECT_SENTENCE_BOUNDARY", "True").lower() == "true"
    )
    splitter.warm_up()
    document = Document(content=text, meta={"filename": filename, "source": source})
    result = splitter.run(documents=[document])
    chunks = result["documents"]
    logger.info(f"  Created {len(chunks)} semantic text chunks.")
    return chunks
