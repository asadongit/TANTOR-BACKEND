"""
SQLite Database Connector.

This module provides a simple connector for SQLite databases. It supports connection testing
via the get_connection() function, which returns a status dictionary indicating success or failure.
Uses sqlite3 library for lightweight, file-based database interactions.
"""

import os
import sqlite3

# Use environment variable for database file path
DB_PATH = os.getenv('SQLITE_DB_PATH', 'hrdb.sqlite')

def get_connection():
    """Test the connection and return a status dict."""
    try:
        conn = sqlite3.connect(DB_PATH)
        # Execute a simple query to test connection
        conn.execute("SELECT 1;")
        conn.close()
        return {"status": "success", "message": "SQLite connection successful"}
    except Exception as e:
        return {"status": "failed", "message": str(e)}
