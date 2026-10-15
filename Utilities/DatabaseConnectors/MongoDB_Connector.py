"""
MongoDB Database Connector.

This module provides a connector for MongoDB databases. It supports connection testing via the
get_connection() function, which pings the admin database and returns a status dictionary.
Uses pymongo library for NoSQL document-based interactions.
"""

import os
from pymongo import MongoClient

# Use environment variables for database connection details
DB_NAME = os.getenv('MONGO_DB_NAME', 'hrdb')
DB_USER = os.getenv('MONGO_DB_USER', 'hrdb')
DB_PASSWORD = os.getenv('MONGO_DB_PASSWORD', 'hrdb@0206')
DB_HOST = os.getenv('MONGO_DB_HOST', 'localhost')
DB_PORT = int(os.getenv('MONGO_DB_PORT', '27017'))

# Setup MongoDB client
conn_string = f"mongodb://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
client = MongoClient(conn_string)

def get_connection():
    """Test the connection and return a status dict."""
    try:
        # Ping the database to test connection
        client.admin.command('ping')
        return {"status": "success", "message": "MongoDB connection successful"}
    except Exception as e:
        return {"status": "failed", "message": str(e)}
