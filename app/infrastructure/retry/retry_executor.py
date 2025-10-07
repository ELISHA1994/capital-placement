"""Retry operation executor for integrating retry logic with operations."""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple
from uuid import uuid4

import structlog

from app.domain.interfaces import IHealthCheck
from app.domain.retry import (
    ErrorCategory, IRetryOperationExecutor, IRetryService, 
    RetryPolicy, RetryResult
)
from app.infrastructure.retry.error_classifier import DefaultErrorClassifier


logger = structlog.get_logger(__name__)


class RetryOperationExecutor(IRetryOperationExecutor, IHealthCheck):
    """Executor for operations with integrated retry logic."""
    
    def __init__(
        self, 
        retry_service: IRetryService,
        error_classifier: Optional[DefaultErrorClassifier] = None
    ):
        self._retry_service = retry_service
        self._error_classifier = error_classifier or DefaultErrorClassifier()
        self._logger = structlog.get_logger(__name__)
        
        # Track execution statistics
        self._execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "retried_executions": 0,
            "dead_lettered_executions": 0
        }
    
    async def execute_with_retry(
        self,
        operation_func: Callable[..., Any],
        operation_id: str,
        operation_type: str,
        tenant_id: str,
        *,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        policy: Optional[RetryPolicy] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute an operation with retry logic."""
        
        kwargs = kwargs or {}
        context = context or {}
        execution_start = datetime.utcnow()
        
        # Add operation metadata to context
        execution_context = {
            **context,
            "operation_func": operation_func.__name__,
            "operation_module": operation_func.__module__,
            "execution_start": execution_start.isoformat(),
            "args_count": len(args),
            "kwargs_keys": list(kwargs.keys())
        }
        
        self._execution_stats["total_executions"] += 1
        
        self._logger.info(
            "Starting operation execution with retry",
            operation_id=operation_id,
            operation_type=operation_type,
            operation_func=operation_func.__name__,
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        # Create retry state
        retry_id = await self._retry_service.create_retry_state(
            operation_id=operation_id,
            operation_type=operation_type,
            tenant_id=tenant_id,
            user_id=user_id,
            policy=policy,
            context=execution_context
        )
        
        last_error = None
        attempt_count = 0
        
        while True:
            attempt_count += 1
            attempt_start = datetime.utcnow()
            
            try:
                self._logger.debug(
                    "Executing operation attempt",
                    retry_id=retry_id,
                    operation_id=operation_id,
                    attempt=attempt_count
                )
                
                # Execute the operation
                if inspect.iscoroutinefunction(operation_func):
                    result = await operation_func(*args, **kwargs)
                else:
                    result = operation_func(*args, **kwargs)
                
                # Calculate operation duration
                operation_duration = int((datetime.utcnow() - attempt_start).total_seconds() * 1000)
                
                # Record successful attempt
                await self._retry_service.record_attempt(
                    retry_id=retry_id,
                    success=True,
                    operation_duration_ms=operation_duration,
                    context={
                        "attempt_number": attempt_count,
                        "result_type": type(result).__name__,
                        "execution_duration_ms": int((datetime.utcnow() - execution_start).total_seconds() * 1000)
                    }
                )
                
                self._execution_stats["successful_executions"] += 1
                
                self._logger.info(
                    "Operation executed successfully",
                    retry_id=retry_id,
                    operation_id=operation_id,
                    attempt=attempt_count,
                    operation_duration_ms=operation_duration
                )
                
                return result
            
            except Exception as error:
                operation_duration = int((datetime.utcnow() - attempt_start).total_seconds() * 1000)
                last_error = error
                
                # Enhance context with attempt information
                attempt_context = {
                    "attempt_number": attempt_count,
                    "operation_type": operation_type,
                    "operation_func": operation_func.__name__,
                    "error_type": type(error).__name__,
                    "execution_duration_ms": int((datetime.utcnow() - execution_start).total_seconds() * 1000)
                }
                
                # Record failed attempt
                retry_attempt = await self._retry_service.record_attempt(
                    retry_id=retry_id,
                    error=error,
                    success=False,
                    operation_duration_ms=operation_duration,
                    context=attempt_context
                )
                
                self._logger.warning(
                    "Operation attempt failed",
                    retry_id=retry_id,
                    operation_id=operation_id,
                    attempt=attempt_count,
                    error=str(error)[:200],
                    error_category=retry_attempt.error_category.value if retry_attempt.error_category else None,
                    operation_duration_ms=operation_duration
                )
                
                # Get updated retry state to check if we should continue
                retry_state = await self._retry_service.get_retry_state(retry_id)
                
                if not retry_state:
                    self._logger.error("Lost retry state during execution", retry_id=retry_id)
                    self._execution_stats["failed_executions"] += 1
                    raise error
                
                # Check if we should continue retrying
                if retry_state.status == RetryResult.SUCCESS:
                    # This shouldn't happen, but handle gracefully
                    return None
                
                elif retry_state.status in [
                    RetryResult.MAX_ATTEMPTS_EXCEEDED,
                    RetryResult.NON_RETRYABLE_ERROR,
                    RetryResult.CANCELLED
                ]:
                    # No more retries
                    self._logger.error(
                        "Operation failed permanently",
                        retry_id=retry_id,
                        operation_id=operation_id,
                        status=retry_state.status.value,
                        total_attempts=retry_state.total_attempts,
                        final_error=str(error)[:200]
                    )
                    
                    if retry_state.policy.dead_letter_enabled:
                        # Move to dead letter queue
                        try:
                            dead_letter_id = await self._retry_service.move_to_dead_letter(
                                retry_id=retry_id,
                                reason=f"Operation failed: {retry_state.status.value}"
                            )
                            self._execution_stats["dead_lettered_executions"] += 1
                            self._logger.info(
                                "Operation moved to dead letter queue",
                                retry_id=retry_id,
                                dead_letter_id=dead_letter_id
                            )
                        except Exception as dl_error:
                            self._logger.error(
                                "Failed to move operation to dead letter queue",
                                retry_id=retry_id,
                                error=str(dl_error)
                            )
                    
                    self._execution_stats["failed_executions"] += 1
                    raise error
                
                elif retry_state.status == RetryResult.FAILED and retry_state.next_attempt_at:
                    # Wait for next retry
                    self._execution_stats["retried_executions"] += 1
                    
                    if retry_state.next_attempt_at > datetime.utcnow():
                        delay = (retry_state.next_attempt_at - datetime.utcnow()).total_seconds()
                        
                        self._logger.info(
                            "Waiting before next retry attempt",
                            retry_id=retry_id,
                            operation_id=operation_id,
                            delay_seconds=delay,
                            next_attempt=attempt_count + 1
                        )
                        
                        await asyncio.sleep(delay)
                    
                    # Continue to next iteration for retry
                    continue
                
                else:
                    # Unexpected state
                    self._logger.error(
                        "Unexpected retry state during execution",
                        retry_id=retry_id,
                        status=retry_state.status.value
                    )
                    self._execution_stats["failed_executions"] += 1
                    raise error
    
    async def execute_with_manual_retry_state(
        self,
        operation_func: Callable[..., Any],
        retry_id: str,
        *,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, bool]:
        """
        Execute an operation using an existing retry state.
        
        Returns:
            Tuple of (result, success) where success indicates if operation completed successfully
        """
        
        kwargs = kwargs or {}
        context = context or {}
        
        retry_state = await self._retry_service.get_retry_state(retry_id)
        if not retry_state:
            raise ValueError(f"Retry state {retry_id} not found")
        
        if retry_state.status != RetryResult.FAILED:
            raise ValueError(f"Retry state {retry_id} is not in failed state (current: {retry_state.status})")
        
        attempt_start = datetime.utcnow()
        
        try:
            self._logger.debug(
                "Executing operation with existing retry state",
                retry_id=retry_id,
                operation_id=retry_state.operation_id,
                current_attempt=retry_state.current_attempt
            )
            
            # Execute the operation
            if inspect.iscoroutinefunction(operation_func):
                result = await operation_func(*args, **kwargs)
            else:
                result = operation_func(*args, **kwargs)
            
            operation_duration = int((datetime.utcnow() - attempt_start).total_seconds() * 1000)
            
            # Record successful attempt
            await self._retry_service.record_attempt(
                retry_id=retry_id,
                success=True,
                operation_duration_ms=operation_duration,
                context={
                    **context,
                    "manual_retry": True,
                    "result_type": type(result).__name__
                }
            )
            
            self._logger.info(
                "Manual retry operation executed successfully",
                retry_id=retry_id,
                operation_id=retry_state.operation_id,
                operation_duration_ms=operation_duration
            )
            
            return result, True
        
        except Exception as error:
            operation_duration = int((datetime.utcnow() - attempt_start).total_seconds() * 1000)
            
            # Record failed attempt
            await self._retry_service.record_attempt(
                retry_id=retry_id,
                error=error,
                success=False,
                operation_duration_ms=operation_duration,
                context={
                    **context,
                    "manual_retry": True,
                    "error_type": type(error).__name__
                }
            )
            
            self._logger.warning(
                "Manual retry operation failed",
                retry_id=retry_id,
                operation_id=retry_state.operation_id,
                error=str(error)[:200],
                operation_duration_ms=operation_duration
            )
            
            return None, False
    
    async def execute_batch_with_retry(
        self,
        operations: List[Dict[str, Any]],
        operation_type: str,
        tenant_id: str,
        *,
        user_id: Optional[str] = None,
        policy: Optional[RetryPolicy] = None,
        max_concurrent: int = 10,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a batch of operations with retry logic.
        
        Args:
            operations: List of operation definitions with keys:
                - operation_id: Unique identifier for the operation
                - operation_func: Function to execute
                - args: Arguments for the function (optional)
                - kwargs: Keyword arguments for the function (optional)
                - context: Operation-specific context (optional)
            operation_type: Type of operations being executed
            tenant_id: Tenant identifier
            user_id: User identifier (optional)
            policy: Retry policy to use (optional)
            max_concurrent: Maximum concurrent operations
            context: Global context for all operations
            
        Returns:
            Dictionary with batch execution results
        """
        
        batch_id = str(uuid4())
        batch_start = datetime.utcnow()
        
        self._logger.info(
            "Starting batch operation execution with retry",
            batch_id=batch_id,
            operation_type=operation_type,
            operation_count=len(operations),
            tenant_id=tenant_id,
            max_concurrent=max_concurrent
        )
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_single_operation(op_def: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                operation_id = op_def["operation_id"]
                operation_func = op_def["operation_func"]
                args = op_def.get("args", ())
                kwargs = op_def.get("kwargs", {})
                op_context = {**(context or {}), **(op_def.get("context", {}))}
                op_context["batch_id"] = batch_id
                
                try:
                    result = await self.execute_with_retry(
                        operation_func=operation_func,
                        operation_id=operation_id,
                        operation_type=operation_type,
                        tenant_id=tenant_id,
                        args=args,
                        kwargs=kwargs,
                        user_id=user_id,
                        policy=policy,
                        context=op_context
                    )
                    
                    return {
                        "operation_id": operation_id,
                        "status": "success",
                        "result": result,
                        "error": None
                    }
                
                except Exception as error:
                    return {
                        "operation_id": operation_id,
                        "status": "failed",
                        "result": None,
                        "error": str(error)
                    }
        
        # Execute all operations concurrently
        tasks = [execute_single_operation(op_def) for op_def in operations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_ops = []
        failed_ops = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Gather call itself failed
                failed_ops.append({
                    "operation_id": operations[i]["operation_id"],
                    "status": "failed",
                    "result": None,
                    "error": str(result)
                })
            elif result["status"] == "success":
                successful_ops.append(result)
            else:
                failed_ops.append(result)
        
        batch_duration = int((datetime.utcnow() - batch_start).total_seconds() * 1000)
        
        batch_result = {
            "batch_id": batch_id,
            "operation_type": operation_type,
            "total_operations": len(operations),
            "successful_operations": len(successful_ops),
            "failed_operations": len(failed_ops),
            "success_rate": len(successful_ops) / len(operations) if operations else 0.0,
            "batch_duration_ms": batch_duration,
            "successful_results": successful_ops,
            "failed_results": failed_ops,
            "started_at": batch_start.isoformat(),
            "completed_at": datetime.utcnow().isoformat()
        }
        
        self._logger.info(
            "Batch operation execution completed",
            batch_id=batch_id,
            total_operations=len(operations),
            successful_operations=len(successful_ops),
            failed_operations=len(failed_ops),
            success_rate=batch_result["success_rate"],
            batch_duration_ms=batch_duration
        )
        
        return batch_result
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics for monitoring."""
        total = self._execution_stats["total_executions"]
        
        return {
            **self._execution_stats,
            "success_rate": self._execution_stats["successful_executions"] / total if total > 0 else 0.0,
            "retry_rate": self._execution_stats["retried_executions"] / total if total > 0 else 0.0,
            "dead_letter_rate": self._execution_stats["dead_lettered_executions"] / total if total > 0 else 0.0
        }
    
    def reset_statistics(self) -> None:
        """Reset execution statistics."""
        self._execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "retried_executions": 0,
            "dead_lettered_executions": 0
        }
    
    async def check_health(self) -> Dict[str, Any]:
        """Check health of the retry executor."""
        stats = self.get_execution_statistics()
        
        # Check retry service health
        retry_service_health = await self._retry_service.check_health()
        
        return {
            "status": "healthy" if retry_service_health.get("status") == "healthy" else "degraded",
            "execution_statistics": stats,
            "retry_service_health": retry_service_health,
            "error_classifier_health": await self._error_classifier.check_health()
        }


__all__ = ["RetryOperationExecutor"]