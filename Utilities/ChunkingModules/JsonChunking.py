"""
This module provides JSON-based text chunking.
It parses a JSON string, flattens it into key-value pairs, and chunks the text using a character-based splitter.
"""

import os
import json
from haystack import Document
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

def chunk_document(json_text, filename, source):
    """
    Parses the JSON text and creates chunks where each top-level JSON object
    (if the input is an array) or the entire JSON (if it's a single object)
    forms a complete chunk.
    Returns a list of Haystack Document objects.
    """
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        return []
    
    chunks = []
    if isinstance(data, list):
        # If the JSON is a list of objects, each object becomes a chunk
        for item in data:
            # Convert each item back to a JSON string for the chunk content
            chunk_content = json.dumps(item, indent=2) 
            haystack_doc = Document(
                content=chunk_content,
                meta={"filename": filename, "source": source, "type": "json_chunk"}
            )
            chunks.append(haystack_doc)
    else:
        # If the JSON is a single object or primitive, the entire JSON becomes one chunk
        chunk_content = json.dumps(data, indent=2)
        haystack_doc = Document(
            content=chunk_content,
            meta={"filename": filename, "source": source, "type": "json_chunk"}
        )
        chunks.append(haystack_doc)
    
    logger.info(f"  Created {len(chunks)} JSON-based text chunks (object-level).")
    return chunks