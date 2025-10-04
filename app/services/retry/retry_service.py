"""Retry service implementation with exponential backoff and configurable policies."""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog
from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.error_handling import handle_database_errors
from app.database.sqlmodel_engine import get_sqlmodel_db_manager
from app.domain.interfaces import IHealthCheck
from app.domain.retry import (
    BackoffStrategy, ErrorCategory, IRetryService, RetryAttempt, 
    RetryPolicy, RetryResult, RetryState
)
from app.models.retry_models import (
    RetryAttemptModel, RetryPolicyTemplate, RetryStateModel, RetryStatistics
)
from app.services.retry.error_classifier import DefaultErrorClassifier


logger = structlog.get_logger(__name__)


class RetryService(IRetryService, IHealthCheck):
    """Service for managing retry operations with exponential backoff."""
    
    def __init__(self, error_classifier: Optional[DefaultErrorClassifier] = None):
        self._error_classifier = error_classifier or DefaultErrorClassifier()
        self._logger = structlog.get_logger(__name__)
        self._db_manager = get_sqlmodel_db_manager()
        
        # Default policies by operation type
        self._default_policies = {
            "document_processing": RetryPolicy(
                max_attempts=3,
                base_delay_seconds=2.0,
                max_delay_seconds=300.0,
                backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
            ),
            "webhook_delivery": RetryPolicy(
                max_attempts=5,
                base_delay_seconds=1.0,
                max_delay_seconds=600.0,
                backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
                jitter_factor=0.2,
            ),
            "ai_processing": RetryPolicy(
                max_attempts=2,
                base_delay_seconds=5.0,
                max_delay_seconds=180.0,
                backoff_strategy=BackoffStrategy.EXPONENTIAL,
                jitter_factor=0.0,
            ),
            "batch_processing": RetryPolicy(
                max_attempts=10,
                base_delay_seconds=0.5,
                max_delay_seconds=120.0,
                backoff_strategy=BackoffStrategy.LINEAR,
                circuit_breaker_enabled=False,
            ),
        }
    
    @handle_database_errors(context={"operation": "create_retry_state"})
    async def create_retry_state(
        self,
        operation_id: str,
        operation_type: str,
        tenant_id: str,
        *,
        user_id: Optional[str] = None,
        policy: Optional[RetryPolicy] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new retry state for an operation."""
        
        retry_id = str(uuid4())
        
        # Get policy from template or use provided/default
        if policy is None:
            policy = await self._get_policy_for_operation(operation_type, tenant_id)
        
        # Create retry state model
        retry_state = RetryStateModel(
            id=retry_id,
            operation_id=operation_id,
            operation_type=operation_type,
            tenant_id=tenant_id,
            user_id=user_id,
            policy_config=self._serialize_policy(policy),
            operation_context=context or {},
            metadata={
                "created_by": "retry_service",
                "policy_source": "template" if policy != policy else "custom"
            }
        )
        
        async with self._db_manager.get_session() as session:
            session.add(retry_state)
            await session.commit()
        
        self._logger.info(
            "Created retry state",
            retry_id=retry_id,
            operation_id=operation_id,
            operation_type=operation_type,
            tenant_id=tenant_id,
            max_attempts=policy.max_attempts
        )
        
        return retry_id
    
    @handle_database_errors(context={"operation": "get_retry_state"})
    async def get_retry_state(self, retry_id: str) -> Optional[RetryState]:
        """Get retry state by ID."""
        
        async with self._db_manager.get_session() as session:
            # Get retry state with attempts
            stmt = select(RetryStateModel).where(RetryStateModel.id == retry_id)
            result = await session.execute(stmt)
            retry_model = result.scalar_one_or_none()
            
            if not retry_model:
                return None
            
            # Get attempts
            attempts_stmt = select(RetryAttemptModel).where(
                RetryAttemptModel.retry_state_id == retry_id
            ).order_by(RetryAttemptModel.attempt_number)
            attempts_result = await session.execute(attempts_stmt)
            attempt_models = attempts_result.scalars().all()
        
        # Convert to domain objects
        attempts = [self._convert_attempt_model(attempt) for attempt in attempt_models]
        policy = self._deserialize_policy(retry_model.policy_config)
        
        retry_state = RetryState(
            retry_id=retry_model.id,
            operation_id=retry_model.operation_id,
            operation_type=retry_model.operation_type,
            tenant_id=retry_model.tenant_id,
            user_id=retry_model.user_id,
            policy=policy,
            current_attempt=retry_model.current_attempt,
            total_attempts=retry_model.total_attempts,
            status=RetryResult(retry_model.status),
            created_at=retry_model.created_at,
            updated_at=retry_model.updated_at,
            next_attempt_at=retry_model.next_attempt_at,
            completed_at=retry_model.completed_at,
            attempts=attempts,
            first_error=retry_model.first_error,
            last_error=retry_model.last_error,
            last_error_category=ErrorCategory(retry_model.last_error_category) if retry_model.last_error_category else None,
            operation_context=retry_model.operation_context,
            metadata=retry_model.metadata
        )
        
        return retry_state
    
    @handle_database_errors(context={"operation": "record_attempt"})
    async def record_attempt(
        self,
        retry_id: str,
        error: Optional[Exception] = None,
        success: bool = False,
        operation_duration_ms: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RetryAttempt:
        """Record a retry attempt."""
        
        attempt_started = datetime.utcnow()
        error_message = str(error) if error else None
        error_category = None
        
        if error:
            # Use context from operation for better classification
            classification_context = context or {}
            classification_context.update({
                "retry_id": retry_id,
                "attempt_started": attempt_started
            })
            error_category = self._error_classifier.classify_error(error, classification_context)
        
        async with self._db_manager.get_session() as session:
            # Get current retry state
            retry_stmt = select(RetryStateModel).where(RetryStateModel.id == retry_id)
            retry_result = await session.execute(retry_stmt)
            retry_model = retry_result.scalar_one_or_none()
            
            if not retry_model:
                raise ValueError(f"Retry state {retry_id} not found")
            
            # Calculate attempt number
            attempt_number = retry_model.current_attempt + 1
            
            # Create attempt record
            attempt = RetryAttemptModel(
                retry_state_id=retry_id,
                attempt_number=attempt_number,
                started_at=attempt_started,
                completed_at=datetime.utcnow(),
                operation_duration_ms=operation_duration_ms,
                success=success,
                error_message=error_message,
                error_category=error_category.value if error_category else None,
                attempt_context=context or {}
            )
            
            # Update retry state
            retry_model.current_attempt = attempt_number
            retry_model.total_attempts = attempt_number
            retry_model.updated_at = datetime.utcnow()
            
            if error_message:
                if retry_model.first_error is None:
                    retry_model.first_error = error_message
                retry_model.last_error = error_message
                retry_model.last_error_category = error_category.value if error_category else None
            
            if success:
                retry_model.status = RetryResult.SUCCESS.value
                retry_model.completed_at = datetime.utcnow()
                retry_model.next_attempt_at = None
            else:
                # Determine if we should continue retrying
                policy = self._deserialize_policy(retry_model.policy_config)
                
                if attempt_number >= policy.max_attempts:
                    retry_model.status = RetryResult.MAX_ATTEMPTS_EXCEEDED.value
                    retry_model.completed_at = datetime.utcnow()
                    retry_model.next_attempt_at = None
                elif error_category and not policy.is_retryable(error_category):
                    retry_model.status = RetryResult.NON_RETRYABLE_ERROR.value
                    retry_model.completed_at = datetime.utcnow()
                    retry_model.next_attempt_at = None
                else:
                    # Schedule next attempt
                    delay = self._calculate_delay(policy, attempt_number, error)
                    retry_model.next_attempt_at = datetime.utcnow() + timedelta(seconds=delay)
                    attempt.delay_before_attempt = delay
            
            session.add(attempt)
            await session.commit()
            
            # Refresh to get updated values
            await session.refresh(attempt)
        
        # Convert to domain object
        domain_attempt = self._convert_attempt_model(attempt)
        
        self._logger.info(
            "Recorded retry attempt",
            retry_id=retry_id,
            attempt_number=attempt_number,
            success=success,
            error_category=error_category.value if error_category else None,
            next_attempt_at=retry_model.next_attempt_at,
            status=retry_model.status
        )
        
        return domain_attempt
    
    @handle_database_errors(context={"operation": "schedule_next_attempt"})
    async def schedule_next_attempt(
        self,
        retry_id: str,
        delay_override: Optional[float] = None
    ) -> Optional[datetime]:
        """Schedule the next retry attempt."""
        
        async with self._db_manager.get_session() as session:
            retry_stmt = select(RetryStateModel).where(RetryStateModel.id == retry_id)
            retry_result = await session.execute(retry_stmt)
            retry_model = retry_result.scalar_one_or_none()
            
            if not retry_model:
                return None
            
            if retry_model.status != RetryResult.FAILED.value:
                return None
            
            policy = self._deserialize_policy(retry_model.policy_config)
            
            # Check if we can retry
            if retry_model.current_attempt >= policy.max_attempts:
                return None
            
            # Calculate delay
            if delay_override is not None:
                delay = delay_override
            else:
                delay = self._calculate_delay(policy, retry_model.current_attempt + 1)
            
            next_attempt_at = datetime.utcnow() + timedelta(seconds=delay)
            retry_model.next_attempt_at = next_attempt_at
            retry_model.updated_at = datetime.utcnow()
            
            await session.commit()
        
        self._logger.debug(
            "Scheduled next retry attempt",
            retry_id=retry_id,
            next_attempt_at=next_attempt_at,
            delay_seconds=delay
        )
        
        return next_attempt_at
    
    @handle_database_errors(context={"operation": "get_ready_retries"})
    async def get_ready_retries(
        self,
        operation_type: Optional[str] = None,
        limit: int = 100
    ) -> List[RetryState]:
        """Get retry operations that are ready to be attempted."""
        
        now = datetime.utcnow()
        
        async with self._db_manager.get_session() as session:
            stmt = select(RetryStateModel).where(
                and_(
                    RetryStateModel.status == RetryResult.FAILED.value,
                    RetryStateModel.next_attempt_at.isnot(None),
                    RetryStateModel.next_attempt_at <= now
                )
            )
            
            if operation_type:
                stmt = stmt.where(RetryStateModel.operation_type == operation_type)
            
            stmt = stmt.order_by(RetryStateModel.next_attempt_at).limit(limit)
            
            result = await session.execute(stmt)
            retry_models = result.scalars().all()
        
        # Convert to domain objects
        retry_states = []
        for retry_model in retry_models:
            # Get attempts for each retry state
            async with self._db_manager.get_session() as session:
                attempts_stmt = select(RetryAttemptModel).where(
                    RetryAttemptModel.retry_state_id == retry_model.id
                ).order_by(RetryAttemptModel.attempt_number)
                attempts_result = await session.execute(attempts_stmt)
                attempt_models = attempts_result.scalars().all()
            
            attempts = [self._convert_attempt_model(attempt) for attempt in attempt_models]
            policy = self._deserialize_policy(retry_model.policy_config)
            
            retry_state = RetryState(
                retry_id=retry_model.id,
                operation_id=retry_model.operation_id,
                operation_type=retry_model.operation_type,
                tenant_id=retry_model.tenant_id,
                user_id=retry_model.user_id,
                policy=policy,
                current_attempt=retry_model.current_attempt,
                total_attempts=retry_model.total_attempts,
                status=RetryResult(retry_model.status),
                created_at=retry_model.created_at,
                updated_at=retry_model.updated_at,
                next_attempt_at=retry_model.next_attempt_at,
                completed_at=retry_model.completed_at,
                attempts=attempts,
                first_error=retry_model.first_error,
                last_error=retry_model.last_error,
                last_error_category=ErrorCategory(retry_model.last_error_category) if retry_model.last_error_category else None,
                operation_context=retry_model.operation_context,
                metadata=retry_model.metadata
            )
            
            retry_states.append(retry_state)
        
        return retry_states
    
    @handle_database_errors(context={"operation": "cancel_retry"})
    async def cancel_retry(
        self,
        retry_id: str,
        reason: str = "Cancelled by user"
    ) -> bool:
        """Cancel a retry operation."""
        
        async with self._db_manager.get_session() as session:
            retry_stmt = select(RetryStateModel).where(RetryStateModel.id == retry_id)
            retry_result = await session.execute(retry_stmt)
            retry_model = retry_result.scalar_one_or_none()
            
            if not retry_model:
                return False
            
            if retry_model.status not in [RetryResult.FAILED.value, RetryResult.SUCCESS.value]:
                retry_model.status = RetryResult.CANCELLED.value
                retry_model.completed_at = datetime.utcnow()
                retry_model.next_attempt_at = None
                retry_model.updated_at = datetime.utcnow()
                retry_model.last_error = reason
                
                await session.commit()
                
                self._logger.info(
                    "Cancelled retry operation",
                    retry_id=retry_id,
                    reason=reason
                )
                
                return True
        
        return False
    
    @handle_database_errors(context={"operation": "move_to_dead_letter"})
    async def move_to_dead_letter(
        self,
        retry_id: str,
        reason: str = "Max attempts exceeded"
    ) -> str:
        """Move a failed retry to dead letter queue."""
        
        # Import here to avoid circular dependency
        from app.infrastructure.providers.retry_provider import get_dead_letter_service
        
        async with self._db_manager.get_session() as session:
            retry_stmt = select(RetryStateModel).where(RetryStateModel.id == retry_id)
            retry_result = await session.execute(retry_stmt)
            retry_model = retry_result.scalar_one_or_none()
            
            if not retry_model:
                raise ValueError(f"Retry state {retry_id} not found")
            
            # Update retry state status
            retry_model.status = RetryResult.DEAD_LETTERED.value
            retry_model.completed_at = datetime.utcnow()
            retry_model.updated_at = datetime.utcnow()
            
            await session.commit()
        
        # Add to dead letter queue
        dead_letter_service = await get_dead_letter_service()
        dead_letter_id = await dead_letter_service.add_to_dead_letter(
            operation_id=retry_model.operation_id,
            operation_type=retry_model.operation_type,
            tenant_id=retry_model.tenant_id,
            retry_id=retry_id,
            final_error=retry_model.last_error,
            error_category=ErrorCategory(retry_model.last_error_category) if retry_model.last_error_category else None,
            retry_attempts=retry_model.total_attempts,
            context=retry_model.operation_context,
            metadata={"moved_reason": reason}
        )
        
        self._logger.info(
            "Moved retry to dead letter queue",
            retry_id=retry_id,
            dead_letter_id=dead_letter_id,
            reason=reason
        )
        
        return dead_letter_id
    
    @handle_database_errors(context={"operation": "cleanup_expired_retries"})
    async def cleanup_expired_retries(
        self,
        max_age_hours: int = 24,
        batch_size: int = 100
    ) -> int:
        """Clean up expired retry states."""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        async with self._db_manager.get_session() as session:
            # Find expired retry states
            stmt = select(RetryStateModel.id).where(
                and_(
                    RetryStateModel.created_at < cutoff_time,
                    RetryStateModel.status.in_([
                        RetryResult.SUCCESS.value,
                        RetryResult.CANCELLED.value,
                        RetryResult.MAX_ATTEMPTS_EXCEEDED.value,
                        RetryResult.NON_RETRYABLE_ERROR.value,
                        RetryResult.DEAD_LETTERED.value
                    ])
                )
            ).limit(batch_size)
            
            result = await session.execute(stmt)
            expired_ids = [row[0] for row in result.fetchall()]
            
            if expired_ids:
                # Delete retry attempts first (cascade should handle this, but being explicit)
                delete_attempts_stmt = select(RetryAttemptModel).where(
                    RetryAttemptModel.retry_state_id.in_(expired_ids)
                )
                await session.execute(delete_attempts_stmt)
                
                # Delete retry states
                delete_states_stmt = select(RetryStateModel).where(
                    RetryStateModel.id.in_(expired_ids)
                )
                await session.execute(delete_states_stmt)
                
                await session.commit()
                cleaned_count = len(expired_ids)
        
        if cleaned_count > 0:
            self._logger.info(
                "Cleaned up expired retry states",
                count=cleaned_count,
                max_age_hours=max_age_hours
            )
        
        return cleaned_count
    
    @handle_database_errors(context={"operation": "get_retry_statistics"})
    async def get_retry_statistics(
        self,
        tenant_id: Optional[str] = None,
        operation_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get retry statistics."""
        
        async with self._db_manager.get_session() as session:
            # Base query
            base_stmt = select(RetryStateModel)
            
            if tenant_id:
                base_stmt = base_stmt.where(RetryStateModel.tenant_id == tenant_id)
            if operation_type:
                base_stmt = base_stmt.where(RetryStateModel.operation_type == operation_type)
            if start_time:
                base_stmt = base_stmt.where(RetryStateModel.created_at >= start_time)
            if end_time:
                base_stmt = base_stmt.where(RetryStateModel.created_at <= end_time)
            
            # Total counts
            total_stmt = select(func.count()).select_from(base_stmt.subquery())
            total_result = await session.execute(total_stmt)
            total_retry_states = total_result.scalar() or 0
            
            # Count by status
            status_stmt = select(
                RetryStateModel.status,
                func.count().label('count')
            ).group_by(RetryStateModel.status)
            
            # Apply same filters
            if tenant_id:
                status_stmt = status_stmt.where(RetryStateModel.tenant_id == tenant_id)
            if operation_type:
                status_stmt = status_stmt.where(RetryStateModel.operation_type == operation_type)
            if start_time:
                status_stmt = status_stmt.where(RetryStateModel.created_at >= start_time)
            if end_time:
                status_stmt = status_stmt.where(RetryStateModel.created_at <= end_time)
            
            status_result = await session.execute(status_stmt)
            by_status = {row.status: row.count for row in status_result.fetchall()}
            
            # Count by operation type
            op_type_stmt = select(
                RetryStateModel.operation_type,
                func.count().label('count')
            ).group_by(RetryStateModel.operation_type)
            
            # Apply filters except operation_type
            if tenant_id:
                op_type_stmt = op_type_stmt.where(RetryStateModel.tenant_id == tenant_id)
            if start_time:
                op_type_stmt = op_type_stmt.where(RetryStateModel.created_at >= start_time)
            if end_time:
                op_type_stmt = op_type_stmt.where(RetryStateModel.created_at <= end_time)
            
            op_type_result = await session.execute(op_type_stmt)
            by_operation_type = {row.operation_type: row.count for row in op_type_result.fetchall()}
            
            # Average attempts
            avg_stmt = select(func.avg(RetryStateModel.total_attempts))
            if tenant_id:
                avg_stmt = avg_stmt.where(RetryStateModel.tenant_id == tenant_id)
            if operation_type:
                avg_stmt = avg_stmt.where(RetryStateModel.operation_type == operation_type)
            if start_time:
                avg_stmt = avg_stmt.where(RetryStateModel.created_at >= start_time)
            if end_time:
                avg_stmt = avg_stmt.where(RetryStateModel.created_at <= end_time)
            
            avg_result = await session.execute(avg_stmt)
            average_attempts = float(avg_result.scalar() or 0.0)
            
            # Success rate
            success_count = by_status.get(RetryResult.SUCCESS.value, 0)
            success_rate = success_count / total_retry_states if total_retry_states > 0 else 0.0
        
        return {
            "total_retry_states": total_retry_states,
            "active_retries": by_status.get(RetryResult.FAILED.value, 0),
            "completed_retries": success_count,
            "failed_retries": by_status.get(RetryResult.MAX_ATTEMPTS_EXCEEDED.value, 0) + by_status.get(RetryResult.NON_RETRYABLE_ERROR.value, 0),
            "cancelled_retries": by_status.get(RetryResult.CANCELLED.value, 0),
            "by_operation_type": by_operation_type,
            "by_status": by_status,
            "average_attempts": average_attempts,
            "success_rate": success_rate,
            "time_period": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None
            }
        }
    
    async def _get_policy_for_operation(self, operation_type: str, tenant_id: str) -> RetryPolicy:
        """Get retry policy for an operation type."""
        
        try:
            async with self._db_manager.get_session() as session:
                # Try to get tenant-specific template first
                stmt = select(RetryPolicyTemplate).where(
                    and_(
                        RetryPolicyTemplate.operation_type == operation_type,
                        RetryPolicyTemplate.tenant_id == tenant_id,
                        RetryPolicyTemplate.is_active == True
                    )
                ).order_by(desc(RetryPolicyTemplate.is_default))
                
                result = await session.execute(stmt)
                template = result.scalar_one_or_none()
                
                # Fall back to global template
                if not template:
                    stmt = select(RetryPolicyTemplate).where(
                        and_(
                            RetryPolicyTemplate.operation_type == operation_type,
                            RetryPolicyTemplate.tenant_id.is_(None),
                            RetryPolicyTemplate.is_active == True
                        )
                    ).order_by(desc(RetryPolicyTemplate.is_default))
                    
                    result = await session.execute(stmt)
                    template = result.scalar_one_or_none()
                
                if template:
                    return self._deserialize_policy(template.policy_config)
        
        except Exception as e:
            self._logger.warning(
                "Failed to load retry policy template",
                operation_type=operation_type,
                tenant_id=tenant_id,
                error=str(e)
            )
        
        # Fall back to default policy
        return self._default_policies.get(operation_type, RetryPolicy())
    
    def _serialize_policy(self, policy: RetryPolicy) -> Dict[str, Any]:
        """Serialize retry policy to JSON-compatible dict."""
        return {
            "max_attempts": policy.max_attempts,
            "base_delay_seconds": policy.base_delay_seconds,
            "max_delay_seconds": policy.max_delay_seconds,
            "backoff_strategy": policy.backoff_strategy.value,
            "jitter_factor": policy.jitter_factor,
            "retryable_errors": [error.value for error in policy.retryable_errors],
            "non_retryable_errors": [error.value for error in policy.non_retryable_errors],
            "rate_limit_retry_after": policy.rate_limit_retry_after,
            "circuit_breaker_enabled": policy.circuit_breaker_enabled,
            "dead_letter_enabled": policy.dead_letter_enabled,
            "operation_timeout_seconds": policy.operation_timeout_seconds,
            "total_timeout_seconds": policy.total_timeout_seconds,
            "custom_error_classifier": policy.custom_error_classifier,
            "custom_backoff_calculator": policy.custom_backoff_calculator,
        }
    
    def _deserialize_policy(self, config: Dict[str, Any]) -> RetryPolicy:
        """Deserialize retry policy from JSON dict."""
        return RetryPolicy(
            max_attempts=config.get("max_attempts", 3),
            base_delay_seconds=config.get("base_delay_seconds", 1.0),
            max_delay_seconds=config.get("max_delay_seconds", 300.0),
            backoff_strategy=BackoffStrategy(config.get("backoff_strategy", "exponential_jitter")),
            jitter_factor=config.get("jitter_factor", 0.1),
            retryable_errors=[ErrorCategory(e) for e in config.get("retryable_errors", [])],
            non_retryable_errors=[ErrorCategory(e) for e in config.get("non_retryable_errors", [])],
            rate_limit_retry_after=config.get("rate_limit_retry_after", True),
            circuit_breaker_enabled=config.get("circuit_breaker_enabled", True),
            dead_letter_enabled=config.get("dead_letter_enabled", True),
            operation_timeout_seconds=config.get("operation_timeout_seconds"),
            total_timeout_seconds=config.get("total_timeout_seconds"),
            custom_error_classifier=config.get("custom_error_classifier"),
            custom_backoff_calculator=config.get("custom_backoff_calculator"),
        )
    
    def _convert_attempt_model(self, attempt_model: RetryAttemptModel) -> RetryAttempt:
        """Convert SQLModel attempt to domain object."""
        return RetryAttempt(
            attempt_number=attempt_model.attempt_number,
            started_at=attempt_model.started_at,
            completed_at=attempt_model.completed_at,
            error_message=attempt_model.error_message,
            error_category=ErrorCategory(attempt_model.error_category) if attempt_model.error_category else None,
            delay_before_attempt=attempt_model.delay_before_attempt,
            operation_duration_ms=attempt_model.operation_duration_ms,
            success=attempt_model.success
        )
    
    def _calculate_delay(
        self, 
        policy: RetryPolicy, 
        attempt: int, 
        last_error: Optional[Exception] = None
    ) -> float:
        """Calculate delay before next retry attempt."""
        
        # Check for retry-after header first
        if last_error and policy.rate_limit_retry_after:
            retry_after = self._error_classifier.extract_retry_after(last_error)
            if retry_after:
                return min(retry_after, policy.max_delay_seconds)
        
        # Use policy's calculation method
        delay = policy.calculate_delay(attempt, last_error)
        
        # Add additional jitter for exponential_jitter strategy
        if policy.backoff_strategy == BackoffStrategy.EXPONENTIAL_JITTER:
            jitter = delay * policy.jitter_factor * (random.random() * 2 - 1)
            delay = max(0, delay + jitter)
        
        return min(delay, policy.max_delay_seconds)
    
    async def check_health(self) -> Dict[str, Any]:
        """Check health of the retry service."""
        try:
            # Check database connectivity
            async with self._db_manager.get_session() as session:
                stmt = select(func.count()).select_from(RetryStateModel)
                result = await session.execute(stmt)
                total_retries = result.scalar() or 0
                
                # Check for ready retries
                ready_retries = await self.get_ready_retries(limit=10)
            
            return {
                "status": "healthy",
                "total_retry_states": total_retries,
                "ready_retries": len(ready_retries),
                "error_classifier": await self._error_classifier.check_health(),
                "default_policies": list(self._default_policies.keys())
            }
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


__all__ = ["RetryService"]