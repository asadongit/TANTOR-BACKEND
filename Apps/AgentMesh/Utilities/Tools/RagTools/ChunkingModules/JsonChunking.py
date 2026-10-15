"""
This module provides JSON-based text chunking.
It parses a JSON string, flattens it into key-value pairs, or chunks by top-level objects.
"""

import json
from typing import List
from langchain_core.documents import Document as LangChainDocument
import logging

logger = logging.getLogger(__name__)

def flatten_json(data, prefix=''):
    """
    Flattens a nested JSON dict into a list of key-value strings.
    """
    items = []
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (dict, list)):
                items.extend(flatten_json(value, new_key))
            else:
                items.append(f"{new_key}: {value}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_key = f"{prefix}[{i}]"
            if isinstance(item, (dict, list)):
                items.extend(flatten_json(item, new_key))
            else:
                items.append(f"{new_key}: {item}")
    else:
        items.append(f"{prefix}: {data}")
    return items

def chunk_document(json_text: str, filename: str, source: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[LangChainDocument]:
    """
    Parses the JSON text and creates chunks.
    If it's a list, each item is a chunk.
    If it's an object, it flattens it or keeps it as one chunk depending on size.
    Returns a list of LangChain Document objects.
    """
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        return []
    
    final_docs = []
    
    # Use a secondary splitter for blocks that are still too large
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    secondary_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ", ", " ", ""]
    )

    if isinstance(data, list):
        # If the JSON is a list of objects, each object becomes a candidate chunk
        for i, item in enumerate(data):
            chunk_content = json.dumps(item, indent=2)
            if len(chunk_content) > chunk_size:
                # Sub-split large objects
                sub_docs = secondary_splitter.create_documents(
                    [chunk_content], 
                    metadatas=[{"filename": filename, "source": source, "index": i, "type": "json_chunk_split"}]
                )
                final_docs.extend(sub_docs)
            else:
                doc = LangChainDocument(
                    page_content=chunk_content,
                    metadata={"filename": filename, "source": source, "index": i, "type": "json_chunk"}
                )
                final_docs.append(doc)
    else:
        # For a single object
        chunk_content = json.dumps(data, indent=2)
        if len(chunk_content) > chunk_size:
             sub_docs = secondary_splitter.create_documents(
                [chunk_content], 
                metadatas=[{"filename": filename, "source": source, "type": "json_chunk_large_split"}]
            )
             final_docs.extend(sub_docs)
        else:
            doc = LangChainDocument(
                page_content=chunk_content,
                metadata={"filename": filename, "source": source, "type": "json_chunk"}
            )
            final_docs.append(doc)
    
    logger.info(f"Created {len(final_docs)} JSON-based chunks (sanitized for size).")
    return final_docs