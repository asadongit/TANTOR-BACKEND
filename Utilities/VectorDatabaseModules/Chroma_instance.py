"""
This module provides utilities for interacting with a Chroma vector database.
It supports both local (path-based) and remote (host/port) connections based on environment variables.
"""

import os
import chromadb
from chromadb.config import Settings

def get_chroma_client():
    """
    Creates and returns a Chroma client instance.
    Uses CHROMA_HOST and CHROMA_PORT for remote connection if available,
    otherwise uses CHROMA_PATH for local path-based storage.
    
    Returns:
        chromadb.Client: The configured Chroma client.
    """
    host = os.environ.get("CHROMA_HOST")
    port = os.environ.get("CHROMA_PORT")
    path = os.environ.get("CHROMA_PATH", "./chroma_db")
    if host and port:
        return chromadb.HttpClient(host=host, port=int(port))
    else:
        return chromadb.PersistentClient(path=path)

def store_in_chroma(texts, metadatas, ids, collection_name, embedding_model, batch_size=5000, logger=None):
    """
    Stores texts, metadatas, and embeddings in a Chroma collection.
    
    Args:
        texts (list): List of text strings to embed and store.
        metadatas (list): List of metadata dictionaries for each text.
        ids (list): List of unique IDs for each document.
        collection_name (str): Name of the Chroma collection.
        embedding_model: The embedding model with an embed_query method.
        batch_size (int): Number of items to add in each batch (not directly used in Chroma, but kept for consistency).
        logger: Optional logger for info/error messages.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    client = get_chroma_client()
    try:
        collection = client.get_or_create_collection(name=collection_name)
        embeddings = [embedding_model.embed_query(text) for text in texts]
        collection.add(
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids
        )
        if logger:
            logger.info(f"  Stored {len(texts)} items in Chroma collection '{collection_name}'.")
        return True
    except Exception as e:
        if logger:
            logger.error(f"Failed to store data in Chroma: {e}")
        return False
