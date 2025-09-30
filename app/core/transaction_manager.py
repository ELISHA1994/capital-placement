"""
SQLModel Transaction Management System for Multi-Tenant FastAPI Application

This module provides a comprehensive transaction management system using SQLAlchemy with:
- Service-level transaction boundaries
- Async context managers and decorators
- Nested transaction support with savepoints
- Error handling and rollback strategies
- Integration with SQLModel repository patterns
- Multi-tenant session scoping
- Performance monitoring hooks
"""

import asyncio
import functools
import inspect
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, TypeVar, Union
from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, DataError, OperationalError
from sqlalchemy.sql import text

from app.core.config import get_settings
from app.database.error_handling import (
    DatabaseError,
    TransactionError,
    handle_database_errors,
    log_database_operation,
)
from app.database.sqlmodel_engine import get_sqlmodel_db_manager

logger = structlog.get_logger(__name__)

# Type definitions
F = TypeVar('F', bound=Callable[..., Any])
ReturnType = TypeVar('ReturnType')


class TransactionContext:
    """
    SQLAlchemy-based transaction context for tracking nested transactions and savepoints.
    """
    
    def __init__(self, session: AsyncSession, transaction_id: str):
        self.session = session
        self.transaction_id = transaction_id
        self.savepoint_stack: List[str] = []
        self.is_nested = False
        self.committed = False
        self.rolled_back = False
    
    @property
    def connection(self):
        """Provide access to the underlying connection for compatibility."""
        return self.session
        
    async def create_savepoint(self, name: Optional[str] = None) -> str:
        """Create a savepoint for nested transaction support."""
        if name is None:
            name = f"sp_{len(self.savepoint_stack)}_{int(time.time())}"
        
        await self.session.execute(text(f"SAVEPOINT {name}"))
        self.savepoint_stack.append(name)
        logger.debug("Savepoint created", savepoint=name, transaction_id=self.transaction_id)
        return name
        
    async def rollback_to_savepoint(self, name: str) -> None:
        """Rollback to a specific savepoint."""
        if name not in self.savepoint_stack:
            raise TransactionError(f"Savepoint {name} not found in transaction {self.transaction_id}")
        
        await self.session.execute(text(f"ROLLBACK TO SAVEPOINT {name}"))
        
        # Remove savepoints after the rolled back one
        while self.savepoint_stack and self.savepoint_stack[-1] != name:
            self.savepoint_stack.pop()
        
        logger.info("Rolled back to savepoint", savepoint=name, transaction_id=self.transaction_id)
        
    async def release_savepoint(self, name: str) -> None:
        """Release a savepoint."""
        if name not in self.savepoint_stack:
            raise TransactionError(f"Savepoint {name} not found in transaction {self.transaction_id}")
        
        await self.session.execute(text(f"RELEASE SAVEPOINT {name}"))
        self.savepoint_stack.remove(name)
        logger.debug("Savepoint released", savepoint=name, transaction_id=self.transaction_id)


class SQLModelTransactionManager:
    """
    SQLAlchemy-based transaction manager for multi-tenant applications.
    """
    
    def __init__(self):
        self.active_transactions: Dict[str, TransactionContext] = {}
        self.transaction_stats = {
            "total_transactions": 0,
            "successful_commits": 0,
            "rollbacks": 0,
            "nested_transactions": 0,
            "avg_transaction_time": 0.0
        }
        
    @property
    def is_initialized(self) -> bool:
        """Check if transaction manager can operate."""
        try:
            db_manager = get_sqlmodel_db_manager()
            return db_manager.is_initialized
        except Exception:
            return False
    
    def _generate_transaction_id(self) -> str:
        """Generate unique transaction ID."""
        return f"txn_{uuid.uuid4().hex[:12]}"
    
    @handle_database_errors(context={"operation": "begin_transaction"})
    async def begin_transaction(
        self, 
        tenant_id: Optional[str] = None,
        transaction_id: Optional[str] = None,
        isolation_level: Optional[str] = None
    ) -> TransactionContext:
        """
        Begin a new transaction using SQLAlchemy session.
        
        Args:
            tenant_id: Optional tenant identifier for multi-tenant isolation
            transaction_id: Optional custom transaction ID
            isolation_level: Optional isolation level override
            
        Returns:
            TransactionContext for managing the transaction
        """
        if transaction_id is None:
            transaction_id = self._generate_transaction_id()
        
        if transaction_id in self.active_transactions:
            # Return existing transaction for nested transaction support
            existing_ctx = self.active_transactions[transaction_id]
            existing_ctx.is_nested = True
            logger.debug("Reusing existing transaction context", transaction_id=transaction_id)
            return existing_ctx
        
        db_manager = get_sqlmodel_db_manager()
        session = db_manager.async_session_factory()
        
        # Set isolation level if specified
        if isolation_level:
            await session.execute(text(f"SET TRANSACTION ISOLATION LEVEL {isolation_level}"))
        
        # Set tenant context if provided
        if tenant_id:
            # Validate tenant_id is a valid UUID format to prevent SQL injection
            try:
                from uuid import UUID
                UUID(tenant_id)
                await session.execute(text(f"SET LOCAL app.current_tenant_id = '{tenant_id}'"))
            except ValueError:
                raise TransactionError(f"Invalid tenant_id format: {tenant_id}")
        
        transaction_ctx = TransactionContext(session, transaction_id)
        self.active_transactions[transaction_id] = transaction_ctx
        
        self.transaction_stats["total_transactions"] += 1
        
        logger.info(
            "Transaction begun",
            transaction_id=transaction_id,
            tenant_id=tenant_id,
            isolation_level=isolation_level
        )
        
        return transaction_ctx
    
    @handle_database_errors(context={"operation": "commit_transaction"})
    async def commit_transaction(self, transaction_ctx: TransactionContext) -> None:
        """
        Commit a transaction.
        
        Args:
            transaction_ctx: Transaction context to commit
        """
        if transaction_ctx.committed:
            logger.warning("Transaction already committed", transaction_id=transaction_ctx.transaction_id)
            return
        
        if transaction_ctx.rolled_back:
            raise TransactionError(f"Cannot commit rolled back transaction {transaction_ctx.transaction_id}")
        
        try:
            await transaction_ctx.session.commit()
            transaction_ctx.committed = True
            
            self.transaction_stats["successful_commits"] += 1
            
            logger.info("Transaction committed successfully", transaction_id=transaction_ctx.transaction_id)
            
        except SQLAlchemyError as e:
            logger.error("Transaction commit failed", 
                        transaction_id=transaction_ctx.transaction_id, 
                        error=str(e))
            await self._rollback_transaction_internal(transaction_ctx)
            raise TransactionError(f"Commit failed: {str(e)}") from e
        finally:
            await self._cleanup_transaction(transaction_ctx)
    
    @handle_database_errors(context={"operation": "rollback_transaction"})
    async def rollback_transaction(self, transaction_ctx: TransactionContext) -> None:
        """
        Rollback a transaction.
        
        Args:
            transaction_ctx: Transaction context to rollback
        """
        await self._rollback_transaction_internal(transaction_ctx)
        await self._cleanup_transaction(transaction_ctx)
    
    async def _rollback_transaction_internal(self, transaction_ctx: TransactionContext) -> None:
        """Internal rollback implementation."""
        if transaction_ctx.rolled_back:
            logger.warning("Transaction already rolled back", transaction_id=transaction_ctx.transaction_id)
            return
        
        try:
            await transaction_ctx.session.rollback()
            transaction_ctx.rolled_back = True
            
            self.transaction_stats["rollbacks"] += 1
            
            logger.info("Transaction rolled back", transaction_id=transaction_ctx.transaction_id)
            
        except SQLAlchemyError as e:
            logger.error("Transaction rollback failed", 
                        transaction_id=transaction_ctx.transaction_id, 
                        error=str(e))
            # Even if rollback fails, mark as rolled back to prevent further operations
            transaction_ctx.rolled_back = True
    
    async def _cleanup_transaction(self, transaction_ctx: TransactionContext) -> None:
        """Clean up transaction resources."""
        try:
            await transaction_ctx.session.close()
        except Exception as e:
            logger.warning("Error closing session", 
                          transaction_id=transaction_ctx.transaction_id, 
                          error=str(e))
        
        # Remove from active transactions
        if transaction_ctx.transaction_id in self.active_transactions:
            del self.active_transactions[transaction_ctx.transaction_id]
    
    @asynccontextmanager
    async def transaction(
        self,
        tenant_id: Optional[str] = None,
        isolation_level: Optional[str] = None,
        auto_commit: bool = True
    ) -> AsyncGenerator[AsyncSession, None]:
        """
        Context manager for transaction handling.
        
        Args:
            tenant_id: Optional tenant identifier
            isolation_level: Optional isolation level
            auto_commit: Whether to auto-commit on success
            
        Usage:
            async with transaction_manager.transaction() as session:
                user = UserTable(name="test")
                session.add(user)
                # Auto-committed on success, auto-rolled back on error
        """
        transaction_ctx = await self.begin_transaction(
            tenant_id=tenant_id,
            isolation_level=isolation_level
        )
        
        try:
            yield transaction_ctx.session
            
            if auto_commit and not transaction_ctx.committed and not transaction_ctx.rolled_back:
                await self.commit_transaction(transaction_ctx)
                
        except Exception as e:
            if not transaction_ctx.rolled_back:
                await self.rollback_transaction(transaction_ctx)
            raise e
        finally:
            # Ensure cleanup if not already done
            if transaction_ctx.transaction_id in self.active_transactions:
                await self._cleanup_transaction(transaction_ctx)
    
    @asynccontextmanager
    async def nested_transaction(
        self,
        parent_ctx: TransactionContext,
        savepoint_name: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Create a nested transaction using savepoints.
        
        Args:
            parent_ctx: Parent transaction context
            savepoint_name: Optional savepoint name
            
        Usage:
            async with transaction_manager.nested_transaction(ctx) as savepoint:
                # Nested operations
                pass
        """
        if not savepoint_name:
            savepoint_name = f"nested_{int(time.time())}_{len(parent_ctx.savepoint_stack)}"
        
        await parent_ctx.create_savepoint(savepoint_name)
        self.transaction_stats["nested_transactions"] += 1
        
        try:
            yield savepoint_name
            await parent_ctx.release_savepoint(savepoint_name)
            
        except Exception as e:
            await parent_ctx.rollback_to_savepoint(savepoint_name)
            raise e
    
    def get_transaction_stats(self) -> Dict[str, Any]:
        """Get transaction statistics."""
        return {
            **self.transaction_stats,
            "active_transactions": len(self.active_transactions),
            "active_transaction_ids": list(self.active_transactions.keys())
        }


# Decorators for transaction management
def transactional(
    tenant_id: Optional[str] = None,
    isolation_level: Optional[str] = None,
    auto_commit: bool = True
):
    """
    Decorator for automatic transaction management.
    
    Args:
        tenant_id: Optional tenant identifier
        isolation_level: Optional isolation level
        auto_commit: Whether to auto-commit on success
    """
    def decorator(func: F) -> F:
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                tx_manager = get_transaction_manager()
                
                # Get transaction context for injection
                transaction_ctx = await tx_manager.begin_transaction(
                    tenant_id=tenant_id,
                    isolation_level=isolation_level
                )
                
                try:
                    # Inject parameters if function expects them
                    sig = inspect.signature(func)
                    if 'session' in sig.parameters:
                        kwargs['session'] = transaction_ctx.session
                    if '_transaction_context' in sig.parameters:
                        kwargs['_transaction_context'] = transaction_ctx
                    
                    result = await func(*args, **kwargs)
                    
                    if auto_commit and not transaction_ctx.committed and not transaction_ctx.rolled_back:
                        await tx_manager.commit_transaction(transaction_ctx)
                    
                    return result
                    
                except Exception as e:
                    if not transaction_ctx.rolled_back:
                        await tx_manager.rollback_transaction(transaction_ctx)
                    raise e
                finally:
                    # Ensure cleanup if not already done
                    if transaction_ctx.transaction_id in tx_manager.active_transactions:
                        await tx_manager._cleanup_transaction(transaction_ctx)
            return async_wrapper
        else:
            raise ValueError("@transactional decorator can only be used with async functions")
    
    return decorator


def requires_transaction(func: F) -> F:
    """
    Decorator to ensure a function runs within a transaction context.
    """
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Check if we're already in a transaction
            tx_manager = get_transaction_manager()
            if not tx_manager.active_transactions:
                raise TransactionError(f"Function {func.__name__} requires an active transaction")
            
            return await func(*args, **kwargs)
        return async_wrapper
    else:
        raise ValueError("@requires_transaction decorator can only be used with async functions")


# Global transaction manager instance
_transaction_manager: Optional[SQLModelTransactionManager] = None


def get_transaction_manager() -> SQLModelTransactionManager:
    """
    Get the global transaction manager instance.
    
    Returns:
        The global SQLModelTransactionManager instance
    """
    global _transaction_manager
    
    if _transaction_manager is None:
        _transaction_manager = SQLModelTransactionManager()
    
    return _transaction_manager


def initialize_transaction_manager(database_manager=None) -> SQLModelTransactionManager:
    """
    Initialize the global transaction manager.
    
    Args:
        database_manager: Database manager (for compatibility, not used in SQLModel version)
    
    Returns:
        Initialized SQLModelTransactionManager instance
    """
    global _transaction_manager
    
    if _transaction_manager is None:
        _transaction_manager = SQLModelTransactionManager()
    
    return _transaction_manager


async def execute_in_transaction(
    operation: Callable,
    *args,
    tenant_id: Optional[str] = None,
    isolation_level: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Execute an operation within a transaction.
    
    Args:
        operation: Function to execute
        *args: Positional arguments for the operation
        tenant_id: Optional tenant identifier
        isolation_level: Optional isolation level
        **kwargs: Keyword arguments for the operation
    
    Returns:
        Result of the operation
    """
    tx_manager = get_transaction_manager()
    
    async with tx_manager.transaction(
        tenant_id=tenant_id,
        isolation_level=isolation_level
    ) as session:
        # Add session to kwargs if the operation expects it
        sig = inspect.signature(operation)
        if 'session' in sig.parameters:
            kwargs['session'] = session
        
        if inspect.iscoroutinefunction(operation):
            return await operation(*args, **kwargs)
        else:
            return operation(*args, **kwargs)


# Transaction monitoring utilities
async def get_transaction_health() -> Dict[str, Any]:
    """Get transaction manager health status."""
    tx_manager = get_transaction_manager()
    
    return {
        "status": "healthy" if tx_manager.is_initialized else "uninitialized",
        "initialized": tx_manager.is_initialized,
        "stats": tx_manager.get_transaction_stats()
    }