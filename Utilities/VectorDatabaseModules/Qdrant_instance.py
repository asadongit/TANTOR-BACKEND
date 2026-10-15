"""
This module provides utilities for interacting with a Qdrant vector database.
It supports both local (path-based) and remote (host/port) connections based on environment variables.
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

def get_qdrant_client():
    """
    Creates and returns a QdrantClient instance.
    Uses QDRANT_HOST and QDRANT_PORT for remote connection if available,
    otherwise uses QDRANT_PATH for local path-based storage.
    
    Returns:
        QdrantClient: The configured Qdrant client.
    """
    host = os.environ.get("QDRANT_HOST")
    port = os.environ.get("QDRANT_PORT")
    path = os.environ.get("QDRANT_PATH", "./VectorDatabaseData/qdrant_db")
    if host and port:
        return QdrantClient(host=host, port=int(port))
    else:
        return QdrantClient(path=path)

def store_in_qdrant(texts, metadatas, ids, collection_name, embedding_model, batch_size=5000, vector_size=384, logger=None):
    """
    Stores texts, metadatas, and embeddings in a Qdrant collection.
    
    Args:
        texts (list): List of text strings to embed and store.
        metadatas (list): List of metadata dictionaries for each text.
        ids (list): List of unique IDs for each point.
        collection_name (str): Name of the Qdrant collection.
        embedding_model: The embedding model with an embed_query method.
        batch_size (int): Number of points to upsert in each batch.
        vector_size (int): Dimensionality of the vectors.
        logger: Optional logger for info/error messages.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    client = get_qdrant_client()
    # Create collection if not exists
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
    if logger:
        logger.info(f"  Storing {len(texts)} items in Qdrant collection '{collection_name}'.")
    try:
        points = []
        for idx, text in enumerate(texts):
            embedding = embedding_model.embed_query(text)
            point = PointStruct(
                id=ids[idx],
                vector=embedding,
                payload=metadatas[idx]
            )
            points.append(point)
            if len(points) == batch_size or idx == len(texts) - 1:
                client.upsert(collection_name=collection_name, points=points)
                if logger:
                    logger.info(f"Stored batch up to index {idx} in Qdrant.")
                points = []
        return True
    except Exception as e:
        if logger:
            logger.error(f"Failed to store data in Qdrant: {e}")
        return False