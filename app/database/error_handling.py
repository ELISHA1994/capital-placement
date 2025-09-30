"""
Centralized error handling for SQLAlchemy-based database operations.

This module provides custom exception classes and error handling utilities
for database operations with proper logging and recovery strategies using SQLAlchemy.
"""

import logging
from typing import Any, Dict, Optional, Type
import asyncio
from functools import wraps
from sqlalchemy.exc import (
    SQLAlchemyError,
    DatabaseError as SQLAlchemyDatabaseError,
    IntegrityError as SQLAlchemyIntegrityError,
    DataError as SQLAlchemyDataError,
    OperationalError,
    ProgrammingError,
    InvalidRequestError,
    DisconnectionError,
    TimeoutError as SQLAlchemyTimeoutError,
    ArgumentError,
    CompileError
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text
import structlog

logger = structlog.get_logger(__name__)


class DatabaseError(Exception):
    """Base exception for all database-related errors."""
    
    def __init__(
        self, 
        message: str, 
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.original_error = original_error
        self.context = context or {}
        
        # Log the error with context
        logger.error(
            "Database error occurred",
            error_type=self.__class__.__name__,
            message=message,
            original_error=str(original_error) if original_error else None,
            context=self.context
        )


class TransactionError(DatabaseError):
    """Exception for transaction-related errors."""
    
    def __init__(
        self, 
        message: str, 
        original_error: Optional[Exception] = None,
        transaction_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        context = context or {}
        if transaction_id:
            context['transaction_id'] = transaction_id
        super().__init__(message, original_error, context)
        self.transaction_id = transaction_id


class ConnectionError(DatabaseError):
    """Raised when database connection issues occur."""
    pass


class MigrationError(DatabaseError):
    """Raised when database migration issues occur."""
    pass


class QueryError(DatabaseError):
    """Raised when query execution issues occur."""
    pass


class IntegrityViolationError(DatabaseError):
    """Raised when database integrity constraints are violated."""
    pass


class ConfigurationError(DatabaseError):
    """Raised when database configuration issues occur."""
    pass


def map_sqlalchemy_error(error: SQLAlchemyError) -> Type[DatabaseError]:
    """Map SQLAlchemy errors to custom exception types."""
    error_mapping = {
        DisconnectionError: ConnectionError,
        OperationalError: ConnectionError,
        SQLAlchemyTimeoutError: ConnectionError,
        SQLAlchemyDataError: QueryError,
        SQLAlchemyIntegrityError: IntegrityViolationError,
        ProgrammingError: QueryError,
        CompileError: QueryError,
        InvalidRequestError: QueryError,
        ArgumentError: ConfigurationError,
    }
    
    return error_mapping.get(type(error), DatabaseError)


async def retry_on_connection_error(
    func,
    *args,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_multiplier: float = 2.0,
    **kwargs
):
    """
    Retry database operations on connection errors with exponential backoff.
    
    Args:
        func: The async function to retry
        *args: Positional arguments for the function
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff_multiplier: Multiplier for exponential backoff
        **kwargs: Keyword arguments for the function
    
    Returns:
        The result of the successful function call
    
    Raises:
        ConnectionError: If all retries fail
    """
    last_error = None
    delay = initial_delay
    
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except (DisconnectionError, OperationalError, SQLAlchemyTimeoutError) as e:
            last_error = e
            
            if attempt == max_retries:
                logger.error(
                    "Max retries exceeded for database operation",
                    function=func.__name__,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e)
                )
                raise ConnectionError(
                    f"Database operation failed after {max_retries} retries: {str(e)}",
                    original_error=e,
                    context={"function": func.__name__, "attempts": attempt + 1}
                )
            
            logger.warning(
                "Database operation failed, retrying",
                function=func.__name__,
                attempt=attempt + 1,
                max_retries=max_retries,
                delay=delay,
                error=str(e)
            )
            
            await asyncio.sleep(delay)
            delay *= backoff_multiplier
    
    # This should never be reached, but just in case
    raise ConnectionError(
        "Unexpected error in retry logic",
        original_error=last_error,
        context={"function": func.__name__}
    )


def handle_database_errors(
    reraise_as: Optional[Type[DatabaseError]] = None,
    context: Optional[Dict[str, Any]] = None
):
    """
    Decorator to handle database errors and convert them to custom exceptions.
    
    Args:
        reraise_as: Custom exception type to raise instead of mapped type
        context: Additional context to include in the exception
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except DatabaseError:
                # Re-raise our custom exceptions as-is
                raise
            except SQLAlchemyError as e:
                # Map SQLAlchemy errors to custom exceptions
                exception_class = reraise_as or map_sqlalchemy_error(e)
                raise exception_class(
                    f"Database operation failed in {func.__name__}: {str(e)}",
                    original_error=e,
                    context=context or {}
                )
            except Exception as e:
                # Handle unexpected errors
                logger.error(
                    "Unexpected error in database operation",
                    function=func.__name__,
                    error=str(e),
                    error_type=type(e).__name__,
                    context=context or {}
                )
                raise DatabaseError(
                    f"Unexpected error in {func.__name__}: {str(e)}",
                    original_error=e,
                    context=context or {}
                )
        
        return wrapper
    return decorator


class DatabaseHealthChecker:
    """Health checker for SQLAlchemy-based database connections."""
    
    def __init__(self, engine_or_session_factory):
        self.engine_or_session_factory = engine_or_session_factory
        self.logger = structlog.get_logger(__name__)
    
    @handle_database_errors(context={"operation": "health_check"})
    async def check_connection(self) -> Dict[str, Any]:
        """
        Check database connection health using SQLAlchemy.
        
        Returns:
            Dict containing health status and metrics
        """
        try:
            # Import here to avoid circular imports
            from app.database.sqlmodel_engine import get_sqlmodel_db_manager
            
            db_manager = get_sqlmodel_db_manager()
            
            async with db_manager.get_session() as session:
                # Basic connectivity test
                result = await session.execute(text("SELECT 1"))
                test_result = result.scalar()
                
                # Check pgvector extension
                pgvector_result = await session.execute(text(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                ))
                pgvector_available = pgvector_result.scalar()
                
                # Get basic database info
                version_result = await session.execute(text("SELECT version()"))
                db_version = version_result.scalar()
                
                # Get engine pool statistics
                pool_stats = {}
                if hasattr(db_manager.engine, 'pool'):
                    pool = db_manager.engine.pool
                    pool_stats = {
                        "size": pool.size(),
                        "checked_in": pool.checkedin(),
                        "checked_out": pool.checkedout(),
                        "overflow": pool.overflow()
                    }
                
                return {
                    "status": "healthy",
                    "connection_test": test_result == 1,
                    "pgvector_available": pgvector_available,
                    "database_version": db_version,
                    "pool_stats": pool_stats,
                }
                
        except Exception as e:
            self.logger.error("Database health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "connection_test": False,
                "pgvector_available": False,
            }
    
    @handle_database_errors(context={"operation": "detailed_health_check"})
    async def detailed_health_check(self) -> Dict[str, Any]:
        """
        Perform a more detailed health check including performance metrics.
        
        Returns:
            Dict containing detailed health information
        """
        basic_health = await self.check_connection()
        
        if basic_health["status"] != "healthy":
            return basic_health
        
        try:
            # Import here to avoid circular imports
            from app.database.sqlmodel_engine import get_sqlmodel_db_manager
            
            db_manager = get_sqlmodel_db_manager()
            
            async with db_manager.get_session() as session:
                # Check active connections
                active_conn_result = await session.execute(text(
                    "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
                ))
                active_connections = active_conn_result.scalar()
                
                # Check database size
                db_size_result = await session.execute(text(
                    "SELECT pg_database_size(current_database())"
                ))
                db_size = db_size_result.scalar()
                
                # Check for long-running queries
                long_queries_result = await session.execute(text(
                    """
                    SELECT count(*) FROM pg_stat_activity 
                    WHERE state = 'active' 
                    AND query_start < now() - interval '1 minute'
                    """
                ))
                long_queries = long_queries_result.scalar()
                
                basic_health.update({
                    "active_connections": active_connections,
                    "database_size_bytes": db_size,
                    "long_running_queries": long_queries,
                })
                
        except Exception as e:
            self.logger.warning("Detailed health check partially failed", error=str(e))
            basic_health["detailed_check_error"] = str(e)
        
        return basic_health


def log_database_operation(operation: str, **context):
    """
    Log database operations with structured logging.
    
    Args:
        operation: Name of the database operation
        **context: Additional context to log
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = asyncio.get_event_loop().time()
            
            logger.info(
                "Database operation started",
                operation=operation,
                function=func.__name__,
                **context
            )
            
            try:
                result = await func(*args, **kwargs)
                
                duration = asyncio.get_event_loop().time() - start_time
                logger.info(
                    "Database operation completed successfully",
                    operation=operation,
                    function=func.__name__,
                    duration=duration,
                    **context
                )
                
                return result
                
            except Exception as e:
                duration = asyncio.get_event_loop().time() - start_time
                logger.error(
                    "Database operation failed",
                    operation=operation,
                    function=func.__name__,
                    duration=duration,
                    error=str(e),
                    **context
                )
                raise
        
        return wrapper
    return decorator


# Utility functions for SQLAlchemy-specific error handling
def is_connection_error(error: Exception) -> bool:
    """Check if an error is a connection-related error."""
    return isinstance(error, (DisconnectionError, OperationalError, SQLAlchemyTimeoutError))


def is_integrity_error(error: Exception) -> bool:
    """Check if an error is an integrity constraint violation."""
    return isinstance(error, SQLAlchemyIntegrityError)


def is_data_error(error: Exception) -> bool:
    """Check if an error is a data-related error."""
    return isinstance(error, SQLAlchemyDataError)


def get_error_code(error: SQLAlchemyError) -> Optional[str]:
    """Extract error code from SQLAlchemy error if available."""
    if hasattr(error, 'orig') and hasattr(error.orig, 'pgcode'):
        return error.orig.pgcode
    elif hasattr(error, 'code'):
        return error.code
    return None


def get_error_details(error: SQLAlchemyError) -> Dict[str, Any]:
    """Extract detailed error information from SQLAlchemy error."""
    details = {
        "error_type": type(error).__name__,
        "message": str(error),
    }
    
    # Add error code if available
    error_code = get_error_code(error)
    if error_code:
        details["error_code"] = error_code
    
    # Add original exception details if available
    if hasattr(error, 'orig'):
        details["original_error"] = {
            "type": type(error.orig).__name__,
            "message": str(error.orig)
        }
    
    # Add statement details if available
    if hasattr(error, 'statement'):
        details["statement"] = str(error.statement)[:500]  # Truncate long statements
    
    # Add parameters if available
    if hasattr(error, 'params'):
        details["params"] = str(error.params)[:200]  # Truncate long parameters
    
    return details


# Context manager for error handling in database operations
class database_operation:
    """Context manager for database operations with automatic error handling."""
    
    def __init__(self, operation_name: str, context: Optional[Dict[str, Any]] = None):
        self.operation_name = operation_name
        self.context = context or {}
        self.start_time = None
    
    async def __aenter__(self):
        self.start_time = asyncio.get_event_loop().time()
        logger.info(
            "Database operation started",
            operation=self.operation_name,
            **self.context
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = asyncio.get_event_loop().time() - self.start_time
        
        if exc_type is None:
            logger.info(
                "Database operation completed successfully",
                operation=self.operation_name,
                duration=duration,
                **self.context
            )
        else:
            logger.error(
                "Database operation failed",
                operation=self.operation_name,
                duration=duration,
                error=str(exc_val),
                error_type=exc_type.__name__,
                **self.context
            )
            
            # Convert SQLAlchemy errors to custom exceptions
            if isinstance(exc_val, SQLAlchemyError):
                custom_exception = map_sqlalchemy_error(exc_val)
                raise custom_exception(
                    f"Database operation '{self.operation_name}' failed: {str(exc_val)}",
                    original_error=exc_val,
                    context=self.context
                )
        
        return False  # Don't suppress exceptions