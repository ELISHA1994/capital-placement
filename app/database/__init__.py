"""
Database module for CV Matching Platform.

This module provides database initialization, connection pooling,
and repository implementations for PostgreSQL with pgvector support.
"""

from .initialization import DatabaseManager, get_database_manager, initialize_database, shutdown_database
from .error_handling import (
    DatabaseError,
    ConnectionError,
    MigrationError,
    QueryError,
    TransactionError,
)

__all__ = [
    "DatabaseManager",
    "get_database_manager", 
    "initialize_database",
    "shutdown_database",
    "DatabaseError",
    "ConnectionError",
    "MigrationError",
    "QueryError",
    "TransactionError",
]