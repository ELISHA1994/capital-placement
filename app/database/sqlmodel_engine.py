"""
SQLModel database engine and session management.

This module provides SQLAlchemy/SQLModel database initialization, connection pooling,
and async session management for PostgreSQL with pgvector support.
"""

from typing import Optional, AsyncGenerator, Dict, Any
from contextlib import asynccontextmanager
import structlog
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
from sqlmodel import SQLModel
from pgvector.sqlalchemy import Vector

from app.core.config import Settings

logger = structlog.get_logger(__name__)


class SQLModelDatabaseManager:
    """
    SQLModel database manager with async session support.
    
    Provides SQLAlchemy engine and session management for the application.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine: Optional[AsyncEngine] = None
        self.async_session_factory: Optional[sessionmaker] = None
        self._initialized = False
        
    @property
    def is_initialized(self) -> bool:
        """Check if SQLModel manager is initialized."""
        return self._initialized and self.engine is not None
    
    def _build_database_url(self) -> str:
        """
        Build SQLAlchemy async database URL from settings.
        
        Converts PostgreSQL URL to SQLAlchemy async format.
        """
        postgres_url = self.settings.get_postgres_url()
        if postgres_url:
            # Convert postgresql:// to postgresql+asyncpg://
            url = str(postgres_url)
            if url.startswith("postgresql://"):
                return url.replace("postgresql://", "postgresql+asyncpg://", 1)
            elif url.startswith("postgres://"):
                return url.replace("postgres://", "postgresql+asyncpg://", 1)
            return url
        
        # Build from individual components
        return (
            f"postgresql+asyncpg://{self.settings.POSTGRES_USER}:{self.settings.POSTGRES_PASSWORD}"
            f"@{self.settings.POSTGRES_HOST}:{self.settings.POSTGRES_PORT}/{self.settings.POSTGRES_DB}"
        )
    
    async def initialize(self) -> None:
        """
        Initialize SQLModel engine and session factory.
        
        Sets up async engine with optimized configuration for PostgreSQL
        and pgvector extension support.
        """
        if self._initialized:
            logger.warning("SQLModel database manager already initialized")
            return
        
        try:
            database_url = self._build_database_url()
            
            # Create async engine with optimized settings
            self.engine = create_async_engine(
                database_url,
                # Connection pool settings
                pool_size=20,  # Optimized for concurrent connections
                max_overflow=40,
                pool_timeout=30,
                pool_recycle=3600,  # Recycle connections every hour
                pool_pre_ping=True,  # Validate connections
                
                # SQLAlchemy settings
                echo=False,  # Set to True for SQL logging in development
                future=True,
                connect_args={
                    "server_settings": {
                        "application_name": "cv-matching-platform-sqlmodel",
                    }
                }
            )
            
            # Create async session factory
            self.async_session_factory = sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,  # Keep objects usable after commit
                autoflush=True,
                autocommit=False
            )
            
            # Test connection and register pgvector
            async with self.engine.begin() as conn:
                # Register pgvector extension
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await conn.commit()
            
            self._initialized = True
            logger.info(
                "SQLModel database manager initialized successfully",
                database_url=database_url.split("@")[0] + "@***"  # Hide credentials
            )
            
        except Exception as e:
            logger.error("Failed to initialize SQLModel database manager", error=str(e))
            raise
    
    async def create_tables(self) -> None:
        """
        Create all SQLModel tables.
        
        This is used for development and testing. In production,
        use Alembic migrations instead.
        """
        if not self.engine:
            raise RuntimeError("Database manager not initialized")
        
        try:
            async with self.engine.begin() as conn:
                # Import all models to ensure they're registered
                from app.models.embedding import EmbeddingTable
                from app.models.auth import UserTable
                from app.models.profile import ProfileTable
                
                # Create all tables
                await conn.run_sync(SQLModel.metadata.create_all)
                await conn.commit()
                
            logger.info("SQLModel tables created successfully")
            
        except Exception as e:
            logger.error("Failed to create SQLModel tables", error=str(e))
            raise
    
    async def drop_tables(self) -> None:
        """
        Drop all SQLModel tables.
        
        WARNING: This will delete all data! Use only for development/testing.
        """
        if not self.engine:
            raise RuntimeError("Database manager not initialized")
        
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(SQLModel.metadata.drop_all)
                await conn.commit()
                
            logger.warning("SQLModel tables dropped successfully")
            
        except Exception as e:
            logger.error("Failed to drop SQLModel tables", error=str(e))
            raise
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get async database session with automatic cleanup.
        
        Usage:
            async with db_manager.get_session() as session:
                # Use session for database operations
                result = await session.execute(select(UserTable))
        """
        if not self.async_session_factory:
            raise RuntimeError("Database manager not initialized")
        
        session = self.async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    @asynccontextmanager
    async def get_transaction(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get async database session with explicit transaction control.
        
        The session will not auto-commit. You must call session.commit()
        explicitly or the transaction will be rolled back.
        
        Usage:
            async with db_manager.get_transaction() as session:
                # Perform operations
                user = UserTable(...)
                session.add(user)
                await session.commit()  # Explicit commit required
        """
        if not self.async_session_factory:
            raise RuntimeError("Database manager not initialized")
        
        session = self.async_session_factory()
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on SQLModel database connection.
        
        Returns health status information for monitoring.
        """
        if not self.engine:
            return {
                "status": "unhealthy",
                "error": "Database manager not initialized"
            }
        
        try:
            async with self.engine.begin() as conn:
                # Simple health check query
                result = await conn.execute(text("SELECT 1 as health_check"))
                await conn.commit()
                
                # Get connection pool stats
                pool = self.engine.pool
                
                return {
                    "status": "healthy",
                    "pool_size": pool.size(),
                    "checked_in": pool.checkedin(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow()
                }
                
        except Exception as e:
            logger.error("SQLModel database health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def shutdown(self) -> None:
        """
        Shutdown SQLModel database manager and close connections.
        """
        if self.engine:
            try:
                await self.engine.dispose()
                logger.info("SQLModel database manager shut down successfully")
            except Exception as e:
                logger.error("Error during SQLModel database shutdown", error=str(e))
            finally:
                self.engine = None
                self.async_session_factory = None
                self._initialized = False


# Global SQLModel database manager instance
_sqlmodel_db_manager: Optional[SQLModelDatabaseManager] = None


def get_sqlmodel_db_manager(settings: Optional[Settings] = None) -> SQLModelDatabaseManager:
    """
    Get global SQLModel database manager instance.
    
    Creates the instance on first call with provided settings.
    Subsequent calls return the existing instance.
    """
    global _sqlmodel_db_manager
    
    if _sqlmodel_db_manager is None:
        if settings is None:
            from app.core.config import get_settings
            settings = get_settings()
        _sqlmodel_db_manager = SQLModelDatabaseManager(settings)
    
    return _sqlmodel_db_manager


async def init_sqlmodel_database(settings: Optional[Settings] = None) -> SQLModelDatabaseManager:
    """
    Initialize global SQLModel database manager.
    
    Call this during application startup to set up the database connection.
    """
    db_manager = get_sqlmodel_db_manager(settings)
    await db_manager.initialize()
    return db_manager


async def shutdown_sqlmodel_database() -> None:
    """
    Shutdown global SQLModel database manager.
    
    Call this during application shutdown to clean up connections.
    """
    global _sqlmodel_db_manager
    if _sqlmodel_db_manager:
        await _sqlmodel_db_manager.shutdown()
        _sqlmodel_db_manager = None


# Dependency injection helpers for FastAPI
async def get_sqlmodel_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for getting SQLModel database session.
    
    Usage:
        @app.get("/users")
        async def get_users(session: AsyncSession = Depends(get_sqlmodel_session)):
            result = await session.execute(select(UserTable))
            return result.scalars().all()
    """
    db_manager = get_sqlmodel_db_manager()
    if not db_manager.is_initialized:
        raise RuntimeError("SQLModel database manager not initialized")
    
    async with db_manager.get_session() as session:
        yield session


async def get_sqlmodel_transaction() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for getting SQLModel database transaction.
    
    Usage:
        @app.post("/users")
        async def create_user(
            user_data: UserCreate,
            session: AsyncSession = Depends(get_sqlmodel_transaction)
        ):
            user = UserTable(**user_data.dict())
            session.add(user)
            await session.commit()
            return user
    """
    db_manager = get_sqlmodel_db_manager()
    if not db_manager.is_initialized:
        raise RuntimeError("SQLModel database manager not initialized")
    
    async with db_manager.get_transaction() as session:
        yield session