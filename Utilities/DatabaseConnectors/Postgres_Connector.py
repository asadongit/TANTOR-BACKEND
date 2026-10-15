"""
PostgreSQL Database Connector.

This module provides a connector for PostgreSQL databases using a connection pool for efficiency.
It supports connection testing via the get_connection() function, which sets the schema and returns
a status dictionary. Uses psycopg and psycopg_pool libraries for robust, pooled connections.
"""

import os
import psycopg
from psycopg_pool import ConnectionPool

# Use environment variables for database connection details
DB_NAME = os.getenv('DB_NAME', 'hrdb')
DB_USER = os.getenv('DB_USER', 'hrdb')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'hrdb@0206')
DB_HOST = os.getenv('DB_HOST', '168.138.192.177')
DB_SCHEMA = os.getenv('DB_SCHEMA', 'hrms')  # Default schema

# Setup a connection pool with size limits
conninfo = f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} host={DB_HOST}"
pool = ConnectionPool(conninfo=conninfo, min_size=1, max_size=10, max_idle=300)

def get_connection():
    """Test the connection and return a status dict."""
    try:
        with pool.connection() as conn:
            conn.autocommit = True  # Ensure auto-commit to prevent connection breaks
            conn.execute(f"SET search_path TO {DB_SCHEMA};")
            return {"status": "success", "message": "Database connection successful"}
    except Exception as e:
        return {"status": "failed", "message": str(e)}

