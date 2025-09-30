"""
SQLModel-based database initialization and connection management.

This module provides database initialization, connection management,
and startup/shutdown management for PostgreSQL with SQLModel/SQLAlchemy.
"""

from typing import Optional, Dict, Any, List
import structlog
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.sql import text
from sqlalchemy import inspect
import time
import random

from .error_handling import (
    DatabaseError,
    ConnectionError,
    MigrationError,
    handle_database_errors,
    log_database_operation,
)
from .sqlmodel_migration import initialize_sqlmodel_tables
from .sqlmodel_engine import SQLModelDatabaseManager, get_sqlmodel_db_manager, init_sqlmodel_database
from app.core.config import Settings
from app.core.environment import get_current_environment

logger = structlog.get_logger(__name__)


class DatabaseManager:
    """
    SQLModel-based database manager handling initialization, connection management,
    and lifecycle management using SQLModel/SQLAlchemy.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.sqlmodel_manager: Optional[SQLModelDatabaseManager] = None
        self._initialized = False
        
    @property
    def is_initialized(self) -> bool:
        """Check if database manager is initialized."""
        return self._initialized and self.sqlmodel_manager is not None and self.sqlmodel_manager.is_initialized
    
    @property
    def engine(self) -> AsyncEngine:
        """Get the SQLAlchemy async engine."""
        if not self.sqlmodel_manager or not self.sqlmodel_manager.engine:
            raise DatabaseError("Database manager not initialized")
        return self.sqlmodel_manager.engine
    
    @handle_database_errors(context={"operation": "database_initialization"})
    @log_database_operation("initialize_database")
    async def initialize(self, run_migrations: bool = True) -> None:
        """
        Initialize the database manager using SQLModel.
        
        Args:
            run_migrations: Whether to run database migrations
        """
        if self._initialized:
            logger.warning("Database manager already initialized")
            return
        
        logger.info("Initializing database manager")
        
        # Initialize SQLModel database manager
        self.sqlmodel_manager = await init_sqlmodel_database(self.settings)
        
        
        # Verify database connectivity and extensions
        await self._verify_database_setup()
        
        
        # Initialize SQLModel tables (auto-migration in dev, validation in prod)
        try:
            sqlmodel_result = await initialize_sqlmodel_tables()
            logger.info(
                "SQLModel initialization completed", 
                status=sqlmodel_result['status'],
                environment=get_current_environment().value
            )
        except Exception as e:
            logger.error("SQLModel initialization failed", error=str(e))
            if get_current_environment().value == "production":
                # In production, SQLModel errors are more critical
                raise DatabaseError(f"SQLModel initialization failed: {str(e)}") from e
            else:
                # In development, log but continue (might be first run)
                logger.warning("SQLModel auto-migration failed, continuing...")
        
        self._initialized = True
        logger.info("Database manager initialized successfully")
    
    @handle_database_errors(context={"operation": "database_verification"})
    async def _verify_database_setup(self) -> None:
        """
        Verify database setup and extensions using SQLAlchemy.
        """
        async with self.sqlmodel_manager.get_session() as session:
            # Test basic connectivity
            result = await session.execute(text("SELECT 1"))
            test_result = result.scalar()
            if test_result != 1:
                raise DatabaseError("Basic connectivity test failed")
            
            # Check PostgreSQL version
            version_result = await session.execute(text("SELECT version()"))
            version = version_result.scalar()
            logger.info("PostgreSQL version", version=version)
            
            # Verify pgvector extension
            pgvector_result = await session.execute(text(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
            ))
            pgvector_available = pgvector_result.scalar()
            
            if pgvector_available:
                logger.info("pgvector extension is available")
                
                # Test vector operations with unique table name to avoid conflicts
                test_table_name = f"_test_vectors_{int(time.time())}_{random.randint(1000, 9999)}"
                
                try:
                    await session.execute(text(f"CREATE TABLE {test_table_name} (id int, embedding vector(3))"))
                    await session.execute(text(f"INSERT INTO {test_table_name} VALUES (1, '[1,2,3]')"))
                    
                    # Test vector similarity
                    distance_result = await session.execute(text(
                        f"SELECT embedding <-> '[3,2,1]'::vector FROM {test_table_name} WHERE id = 1"
                    ))
                    distance = distance_result.scalar()
                finally:
                    # Always cleanup the test table
                    await session.execute(text(f"DROP TABLE IF EXISTS {test_table_name}"))
                    await session.commit()
                
                logger.info("pgvector functionality verified", test_distance=float(distance))
            else:
                logger.warning("pgvector extension not available - vector operations will not work")
    
    @handle_database_errors(context={"operation": "database_shutdown"})
    @log_database_operation("shutdown_database")
    async def shutdown(self) -> None:
        """
        Gracefully shutdown the database manager.
        """
        if not self._initialized:
            return
        
        logger.info("Shutting down database manager")
        
        if self.sqlmodel_manager:
            await self.sqlmodel_manager.shutdown()
            self.sqlmodel_manager = None
        
        self._initialized = False
        
        logger.info("Database manager shut down successfully")
    
    @asynccontextmanager
    async def get_connection(self):
        """
        Get a database session from SQLModel.
        
        Usage:
            async with db_manager.get_connection() as session:
                result = await session.execute(text("SELECT 1"))
        """
        if not self.sqlmodel_manager:
            raise DatabaseError("Database manager not initialized")
        
        async with self.sqlmodel_manager.get_session() as session:
            yield session
    
    @asynccontextmanager
    async def transaction(self):
        """
        Get a database session with explicit transaction management.
        
        Usage:
            async with db_manager.transaction() as session:
                await session.execute(text("INSERT INTO table VALUES (...)"))
                await session.commit()
        """
        if not self.sqlmodel_manager:
            raise DatabaseError("Database manager not initialized")
        
        async with self.sqlmodel_manager.get_transaction() as session:
            yield session
    
    async def execute_query(
        self, 
        query: str, 
        *args, 
        fetch_mode: str = "none"
    ) -> Any:
        """
        Execute a query with the specified fetch mode using SQLAlchemy.
        
        Args:
            query: SQL query to execute
            *args: Query parameters (will be converted to dict for SQLAlchemy)
            fetch_mode: How to fetch results ("none", "val", "row", "all")
        
        Returns:
            Query result based on fetch_mode
        """
        async with self.get_connection() as session:
            # Convert positional args to dictionary for SQLAlchemy
            params = {}
            if args:
                for i, arg in enumerate(args):
                    params[f'param_{i}'] = arg
                # Replace $1, $2 etc with :param_0, :param_1 for SQLAlchemy
                for i in range(len(args)):
                    query = query.replace(f'${i+1}', f':param_{i}')
            
            result = await session.execute(text(query), params)
            
            if fetch_mode == "val":
                return result.scalar()
            elif fetch_mode == "row":
                return result.fetchone()
            elif fetch_mode == "all":
                return result.fetchall()
            else:  # fetch_mode == "none"
                await session.commit()
                return result.rowcount
    
    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get database health status using SQLAlchemy.
        
        Returns:
            Dict containing health information
        """
        if not self.sqlmodel_manager:
            return {"status": "uninitialized"}
        
        return await self.sqlmodel_manager.health_check()
    
    async def get_detailed_health_status(self) -> Dict[str, Any]:
        """
        Get detailed database health status using SQLAlchemy.
        
        Returns:
            Dict containing detailed health information
        """
        if not self.sqlmodel_manager:
            return {"status": "uninitialized"}
        
        health_status = await self.sqlmodel_manager.health_check()
        
        # Add additional details about SQLModel tables
        try:
            async with self.get_connection() as session:
                # Get table count
                table_count_result = await session.execute(text(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
                ))
                table_count = table_count_result.scalar()
                
                # Get extension info
                extension_result = await session.execute(text(
                    "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'uuid-ossp', 'pg_trgm')"
                ))
                extensions = [row[0] for row in extension_result.fetchall()]
                
                health_status.update({
                    "table_count": table_count,
                    "extensions": extensions,
                    "sqlmodel_initialized": True
                })
        except Exception as e:
            logger.error("Error getting detailed health status", error=str(e))
            health_status.update({
                "detailed_check_error": str(e)
            })
        
        return health_status


# Global database manager instance
_database_manager: Optional[DatabaseManager] = None


async def initialize_database(settings: Settings, run_migrations: bool = True) -> DatabaseManager:
    """
    Initialize the global database manager.
    
    Args:
        settings: Application settings
        run_migrations: Whether to run database migrations
    
    Returns:
        Initialized DatabaseManager instance
    """
    global _database_manager
    
    if _database_manager is not None and _database_manager.is_initialized:
        logger.warning("Database manager already initialized globally")
        return _database_manager
    
    _database_manager = DatabaseManager(settings)
    await _database_manager.initialize(run_migrations=run_migrations)
    
    return _database_manager


def get_database_manager() -> DatabaseManager:
    """
    Get the global database manager instance.
    
    Returns:
        The global DatabaseManager instance
    
    Raises:
        DatabaseError: If database manager not initialized
    """
    if _database_manager is None or not _database_manager.is_initialized:
        raise DatabaseError("Database manager not initialized. Call initialize_database() first.")
    
    return _database_manager


async def shutdown_database() -> None:
    """
    Shutdown the global database manager.
    """
    global _database_manager
    
    if _database_manager:
        await _database_manager.shutdown()
        _database_manager = None


@asynccontextmanager
async def get_db_connection():
    """
    Convenience function to get a database session.
    
    Usage:
        async with get_db_connection() as session:
            result = await session.execute(text("SELECT 1"))
    """
    db_manager = get_database_manager()
    async with db_manager.get_connection() as session:
        yield session


@asynccontextmanager
async def get_db_transaction():
    """
    Convenience function to get a database transaction.
    
    Usage:
        async with get_db_transaction() as session:
            await session.execute(text("INSERT INTO table VALUES (...)"))
            await session.commit()
    """
    db_manager = get_database_manager()
    async with db_manager.transaction() as session:
        yield session