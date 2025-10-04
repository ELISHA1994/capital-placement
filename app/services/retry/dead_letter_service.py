"""Dead letter queue service for permanently failed tasks."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog
from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.error_handling import handle_database_errors
from app.database.sqlmodel_engine import get_sqlmodel_db_manager
from app.domain.interfaces import IHealthCheck
from app.domain.retry import ErrorCategory, IDeadLetterService, RetryPolicy
from app.models.retry_models import DeadLetterModel, DeadLetterStatistics, RetryStateModel


logger = structlog.get_logger(__name__)


class DeadLetterService(IDeadLetterService, IHealthCheck):
    """Service for managing dead letter queue for permanently failed operations."""
    
    def __init__(self):
        self._logger = structlog.get_logger(__name__)
        self._db_manager = get_sqlmodel_db_manager()
    
    @handle_database_errors(context={"operation": "add_to_dead_letter"})
    async def add_to_dead_letter(
        self,
        operation_id: str,
        operation_type: str,
        tenant_id: str,
        *,
        retry_id: Optional[str] = None,
        final_error: Optional[str] = None,
        error_category: Optional[ErrorCategory] = None,
        retry_attempts: int = 0,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a failed operation to the dead letter queue."""
        
        dead_letter_id = str(uuid4())
        
        dead_letter_entry = DeadLetterModel(
            id=dead_letter_id,
            operation_id=operation_id,
            operation_type=operation_type,
            tenant_id=tenant_id,
            retry_state_id=retry_id,
            final_error=final_error,
            error_category=error_category.value if error_category else None,
            retry_attempts=retry_attempts,
            operation_context=context or {},
            metadata=metadata or {}
        )
        
        async with self._db_manager.get_session() as session:
            session.add(dead_letter_entry)
            await session.commit()
        
        self._logger.warning(
            "Operation added to dead letter queue",
            dead_letter_id=dead_letter_id,
            operation_id=operation_id,
            operation_type=operation_type,
            tenant_id=tenant_id,
            error_category=error_category.value if error_category else None,
            retry_attempts=retry_attempts,
            final_error=final_error[:200] if final_error else None
        )
        
        return dead_letter_id
    
    @handle_database_errors(context={"operation": "get_dead_letter_entry"})
    async def get_dead_letter_entry(self, dead_letter_id: str) -> Optional[Dict[str, Any]]:
        """Get a dead letter entry by ID."""
        
        async with self._db_manager.get_session() as session:
            stmt = select(DeadLetterModel).where(DeadLetterModel.id == dead_letter_id)
            result = await session.execute(stmt)
            entry = result.scalar_one_or_none()
            
            if not entry:
                return None
            
            return {
                "id": entry.id,
                "operation_id": entry.operation_id,
                "operation_type": entry.operation_type,
                "tenant_id": entry.tenant_id,
                "retry_state_id": entry.retry_state_id,
                "final_error": entry.final_error,
                "error_category": entry.error_category,
                "retry_attempts": entry.retry_attempts,
                "created_at": entry.created_at,
                "resolved_at": entry.resolved_at,
                "is_resolved": entry.is_resolved,
                "resolution_action": entry.resolution_action,
                "resolved_by": entry.resolved_by,
                "resolution_notes": entry.resolution_notes,
                "requeued_count": entry.requeued_count,
                "last_requeued_at": entry.last_requeued_at,
                "operation_context": entry.operation_context,
                "metadata": entry.metadata
            }
    
    @handle_database_errors(context={"operation": "requeue_from_dead_letter"})
    async def requeue_from_dead_letter(
        self,
        dead_letter_id: str,
        *,
        admin_user_id: str,
        notes: Optional[str] = None,
        new_policy: Optional[RetryPolicy] = None
    ) -> str:
        """Requeue an operation from the dead letter queue."""
        
        async with self._db_manager.get_session() as session:
            # Get dead letter entry
            stmt = select(DeadLetterModel).where(DeadLetterModel.id == dead_letter_id)
            result = await session.execute(stmt)
            dead_letter_entry = result.scalar_one_or_none()
            
            if not dead_letter_entry:
                raise ValueError(f"Dead letter entry {dead_letter_id} not found")
            
            if dead_letter_entry.is_resolved:
                raise ValueError(f"Dead letter entry {dead_letter_id} is already resolved")
            
            # Update requeue tracking
            dead_letter_entry.requeued_count += 1
            dead_letter_entry.last_requeued_at = datetime.utcnow()
            
            # Add requeue metadata
            requeue_metadata = dead_letter_entry.metadata.copy()
            requeue_metadata.update({
                "requeued_by": admin_user_id,
                "requeue_notes": notes,
                "requeue_timestamp": datetime.utcnow().isoformat(),
                "requeue_count": dead_letter_entry.requeued_count
            })
            dead_letter_entry.metadata = requeue_metadata
            
            await session.commit()
        
        # Create new retry state for the requeued operation
        from app.infrastructure.providers.retry_provider import get_retry_service
        
        retry_service = await get_retry_service()
        
        # Prepare context for new retry
        new_context = dead_letter_entry.operation_context.copy()
        new_context.update({
            "requeued_from_dead_letter": dead_letter_id,
            "original_retry_attempts": dead_letter_entry.retry_attempts,
            "requeue_admin": admin_user_id,
            "requeue_notes": notes
        })
        
        new_retry_id = await retry_service.create_retry_state(
            operation_id=dead_letter_entry.operation_id,
            operation_type=dead_letter_entry.operation_type,
            tenant_id=dead_letter_entry.tenant_id,
            policy=new_policy,
            context=new_context
        )
        
        self._logger.info(
            "Operation requeued from dead letter queue",
            dead_letter_id=dead_letter_id,
            new_retry_id=new_retry_id,
            operation_id=dead_letter_entry.operation_id,
            admin_user_id=admin_user_id,
            requeue_count=dead_letter_entry.requeued_count
        )
        
        return new_retry_id
    
    @handle_database_errors(context={"operation": "resolve_dead_letter"})
    async def resolve_dead_letter(
        self,
        dead_letter_id: str,
        *,
        resolution_action: str,
        admin_user_id: str,
        notes: Optional[str] = None
    ) -> bool:
        """Mark a dead letter entry as resolved."""
        
        async with self._db_manager.get_session() as session:
            stmt = select(DeadLetterModel).where(DeadLetterModel.id == dead_letter_id)
            result = await session.execute(stmt)
            dead_letter_entry = result.scalar_one_or_none()
            
            if not dead_letter_entry:
                return False
            
            if dead_letter_entry.is_resolved:
                self._logger.warning(
                    "Dead letter entry already resolved",
                    dead_letter_id=dead_letter_id,
                    resolved_by=dead_letter_entry.resolved_by,
                    resolved_at=dead_letter_entry.resolved_at
                )
                return False
            
            # Mark as resolved
            dead_letter_entry.is_resolved = True
            dead_letter_entry.resolved_at = datetime.utcnow()
            dead_letter_entry.resolution_action = resolution_action
            dead_letter_entry.resolved_by = admin_user_id
            dead_letter_entry.resolution_notes = notes
            
            # Add resolution metadata
            resolution_metadata = dead_letter_entry.metadata.copy()
            resolution_metadata.update({
                "resolution_timestamp": datetime.utcnow().isoformat(),
                "resolution_action": resolution_action,
                "resolved_by": admin_user_id,
                "resolution_notes": notes
            })
            dead_letter_entry.metadata = resolution_metadata
            
            await session.commit()
        
        self._logger.info(
            "Dead letter entry resolved",
            dead_letter_id=dead_letter_id,
            resolution_action=resolution_action,
            admin_user_id=admin_user_id,
            operation_id=dead_letter_entry.operation_id
        )
        
        return True
    
    @handle_database_errors(context={"operation": "get_dead_letter_queue"})
    async def get_dead_letter_queue(
        self,
        *,
        tenant_id: Optional[str] = None,
        operation_type: Optional[str] = None,
        error_category: Optional[ErrorCategory] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        resolved: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get dead letter queue entries with filtering."""
        
        async with self._db_manager.get_session() as session:
            # Build base query
            stmt = select(DeadLetterModel)
            count_stmt = select(func.count()).select_from(DeadLetterModel)
            
            # Apply filters
            conditions = []
            
            if tenant_id:
                conditions.append(DeadLetterModel.tenant_id == tenant_id)
            
            if operation_type:
                conditions.append(DeadLetterModel.operation_type == operation_type)
            
            if error_category:
                conditions.append(DeadLetterModel.error_category == error_category.value)
            
            if start_date:
                conditions.append(DeadLetterModel.created_at >= start_date)
            
            if end_date:
                conditions.append(DeadLetterModel.created_at <= end_date)
            
            if resolved is not None:
                conditions.append(DeadLetterModel.is_resolved == resolved)
            
            if conditions:
                stmt = stmt.where(and_(*conditions))
                count_stmt = count_stmt.where(and_(*conditions))
            
            # Get total count
            count_result = await session.execute(count_stmt)
            total_count = count_result.scalar() or 0
            
            # Apply ordering, limit, and offset
            stmt = stmt.order_by(desc(DeadLetterModel.created_at)).limit(limit).offset(offset)
            
            # Execute query
            result = await session.execute(stmt)
            entries = result.scalars().all()
        
        # Convert to response format
        entry_list = []
        for entry in entries:
            entry_dict = {
                "id": entry.id,
                "operation_id": entry.operation_id,
                "operation_type": entry.operation_type,
                "tenant_id": entry.tenant_id,
                "retry_state_id": entry.retry_state_id,
                "final_error": entry.final_error,
                "error_category": entry.error_category,
                "retry_attempts": entry.retry_attempts,
                "created_at": entry.created_at,
                "resolved_at": entry.resolved_at,
                "is_resolved": entry.is_resolved,
                "resolution_action": entry.resolution_action,
                "resolved_by": entry.resolved_by,
                "resolution_notes": entry.resolution_notes,
                "requeued_count": entry.requeued_count,
                "last_requeued_at": entry.last_requeued_at,
                "operation_context": entry.operation_context,
                "metadata": entry.metadata
            }
            entry_list.append(entry_dict)
        
        return {
            "entries": entry_list,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + len(entry_list)) < total_count,
            "filters": {
                "tenant_id": tenant_id,
                "operation_type": operation_type,
                "error_category": error_category.value if error_category else None,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "resolved": resolved
            }
        }
    
    @handle_database_errors(context={"operation": "cleanup_old_dead_letters"})
    async def cleanup_old_dead_letters(
        self,
        older_than_days: int = 30,
        batch_size: int = 100
    ) -> int:
        """Clean up old dead letter entries."""
        
        cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
        cleaned_count = 0
        
        async with self._db_manager.get_session() as session:
            # Only clean up resolved entries that are old enough
            stmt = select(DeadLetterModel.id).where(
                and_(
                    DeadLetterModel.is_resolved == True,
                    DeadLetterModel.resolved_at < cutoff_date
                )
            ).limit(batch_size)
            
            result = await session.execute(stmt)
            old_entry_ids = [row[0] for row in result.fetchall()]
            
            if old_entry_ids:
                # Delete old entries
                delete_stmt = select(DeadLetterModel).where(
                    DeadLetterModel.id.in_(old_entry_ids)
                )
                delete_result = await session.execute(delete_stmt)
                
                await session.commit()
                cleaned_count = len(old_entry_ids)
        
        if cleaned_count > 0:
            self._logger.info(
                "Cleaned up old dead letter entries",
                count=cleaned_count,
                older_than_days=older_than_days
            )
        
        return cleaned_count
    
    @handle_database_errors(context={"operation": "get_dead_letter_statistics"})
    async def get_dead_letter_statistics(
        self,
        *,
        tenant_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> DeadLetterStatistics:
        """Get dead letter queue statistics."""
        
        async with self._db_manager.get_session() as session:
            # Base query
            base_stmt = select(DeadLetterModel)
            
            if tenant_id:
                base_stmt = base_stmt.where(DeadLetterModel.tenant_id == tenant_id)
            if start_time:
                base_stmt = base_stmt.where(DeadLetterModel.created_at >= start_time)
            if end_time:
                base_stmt = base_stmt.where(DeadLetterModel.created_at <= end_time)
            
            # Total counts
            total_stmt = select(func.count()).select_from(base_stmt.subquery())
            total_result = await session.execute(total_stmt)
            total_entries = total_result.scalar() or 0
            
            # Resolved vs unresolved
            resolved_stmt = select(func.count()).select_from(base_stmt.subquery()).where(
                DeadLetterModel.is_resolved == True
            )
            if tenant_id:
                resolved_stmt = resolved_stmt.where(DeadLetterModel.tenant_id == tenant_id)
            if start_time:
                resolved_stmt = resolved_stmt.where(DeadLetterModel.created_at >= start_time)
            if end_time:
                resolved_stmt = resolved_stmt.where(DeadLetterModel.created_at <= end_time)
            
            resolved_result = await session.execute(resolved_stmt)
            resolved_entries = resolved_result.scalar() or 0
            
            unresolved_entries = total_entries - resolved_entries
            
            # Count by operation type
            op_type_stmt = select(
                DeadLetterModel.operation_type,
                func.count().label('count')
            ).group_by(DeadLetterModel.operation_type)
            
            if tenant_id:
                op_type_stmt = op_type_stmt.where(DeadLetterModel.tenant_id == tenant_id)
            if start_time:
                op_type_stmt = op_type_stmt.where(DeadLetterModel.created_at >= start_time)
            if end_time:
                op_type_stmt = op_type_stmt.where(DeadLetterModel.created_at <= end_time)
            
            op_type_result = await session.execute(op_type_stmt)
            by_operation_type = {row.operation_type: row.count for row in op_type_result.fetchall()}
            
            # Count by error category
            error_cat_stmt = select(
                DeadLetterModel.error_category,
                func.count().label('count')
            ).group_by(DeadLetterModel.error_category)
            
            if tenant_id:
                error_cat_stmt = error_cat_stmt.where(DeadLetterModel.tenant_id == tenant_id)
            if start_time:
                error_cat_stmt = error_cat_stmt.where(DeadLetterModel.created_at >= start_time)
            if end_time:
                error_cat_stmt = error_cat_stmt.where(DeadLetterModel.created_at <= end_time)
            
            error_cat_result = await session.execute(error_cat_stmt)
            by_error_category = {
                row.error_category or "unknown": row.count 
                for row in error_cat_result.fetchall()
            }
            
            # Count by tenant (if not filtered by tenant)
            by_tenant = {}
            if not tenant_id:
                tenant_stmt = select(
                    DeadLetterModel.tenant_id,
                    func.count().label('count')
                ).group_by(DeadLetterModel.tenant_id)
                
                if start_time:
                    tenant_stmt = tenant_stmt.where(DeadLetterModel.created_at >= start_time)
                if end_time:
                    tenant_stmt = tenant_stmt.where(DeadLetterModel.created_at <= end_time)
                
                tenant_result = await session.execute(tenant_stmt)
                by_tenant = {row.tenant_id: row.count for row in tenant_result.fetchall()}
            
            # Count requeued entries
            requeued_stmt = select(func.count()).select_from(base_stmt.subquery()).where(
                DeadLetterModel.requeued_count > 0
            )
            if tenant_id:
                requeued_stmt = requeued_stmt.where(DeadLetterModel.tenant_id == tenant_id)
            if start_time:
                requeued_stmt = requeued_stmt.where(DeadLetterModel.created_at >= start_time)
            if end_time:
                requeued_stmt = requeued_stmt.where(DeadLetterModel.created_at <= end_time)
            
            requeued_result = await session.execute(requeued_stmt)
            requeued_entries = requeued_result.scalar() or 0
            
            # Oldest unresolved entry
            oldest_stmt = select(func.min(DeadLetterModel.created_at)).where(
                DeadLetterModel.is_resolved == False
            )
            if tenant_id:
                oldest_stmt = oldest_stmt.where(DeadLetterModel.tenant_id == tenant_id)
            
            oldest_result = await session.execute(oldest_stmt)
            oldest_unresolved = oldest_result.scalar()
            
            # Resolution rate
            resolution_rate = resolved_entries / total_entries if total_entries > 0 else 0.0
        
        return DeadLetterStatistics(
            total_entries=total_entries,
            unresolved_entries=unresolved_entries,
            resolved_entries=resolved_entries,
            requeued_entries=requeued_entries,
            by_operation_type=by_operation_type,
            by_error_category=by_error_category,
            by_tenant=by_tenant,
            oldest_unresolved=oldest_unresolved,
            resolution_rate=resolution_rate,
            time_period={
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None
            }
        )
    
    async def get_actionable_dead_letters(
        self,
        *,
        tenant_id: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get dead letter entries that might be actionable for resolution."""
        
        async with self._db_manager.get_session() as session:
            # Look for unresolved entries with specific error categories that might be resolvable
            actionable_categories = [
                ErrorCategory.CONFIGURATION_ERROR.value,
                ErrorCategory.AUTHENTICATION_ERROR.value,
                ErrorCategory.AUTHORIZATION_ERROR.value,
                ErrorCategory.QUOTA_EXCEEDED.value,
                ErrorCategory.RATE_LIMITED.value,
                ErrorCategory.VALIDATION_ERROR.value
            ]
            
            stmt = select(DeadLetterModel).where(
                and_(
                    DeadLetterModel.is_resolved == False,
                    DeadLetterModel.error_category.in_(actionable_categories)
                )
            )
            
            if tenant_id:
                stmt = stmt.where(DeadLetterModel.tenant_id == tenant_id)
            
            # Order by creation date (oldest first) and requeue count (fewer retries first)
            stmt = stmt.order_by(
                DeadLetterModel.requeued_count.asc(),
                DeadLetterModel.created_at.asc()
            ).limit(limit)
            
            result = await session.execute(stmt)
            entries = result.scalars().all()
        
        actionable_entries = []
        for entry in entries:
            suggestions = self._get_resolution_suggestions(entry)
            
            actionable_entries.append({
                "id": entry.id,
                "operation_id": entry.operation_id,
                "operation_type": entry.operation_type,
                "tenant_id": entry.tenant_id,
                "error_category": entry.error_category,
                "final_error": entry.final_error,
                "retry_attempts": entry.retry_attempts,
                "requeued_count": entry.requeued_count,
                "created_at": entry.created_at,
                "age_hours": (datetime.utcnow() - entry.created_at).total_seconds() / 3600,
                "resolution_suggestions": suggestions,
                "operation_context": entry.operation_context
            })
        
        return actionable_entries
    
    def _get_resolution_suggestions(self, entry: DeadLetterModel) -> List[str]:
        """Get suggested resolution actions for a dead letter entry."""
        suggestions = []
        
        error_category = entry.error_category
        error_message = (entry.final_error or "").lower()
        operation_type = entry.operation_type
        
        if error_category == ErrorCategory.CONFIGURATION_ERROR.value:
            suggestions.extend([
                "Check service configuration settings",
                "Verify environment variables and secrets",
                "Review API endpoint configurations"
            ])
        
        elif error_category == ErrorCategory.AUTHENTICATION_ERROR.value:
            suggestions.extend([
                "Verify API keys and credentials",
                "Check token expiration",
                "Review authentication configuration"
            ])
        
        elif error_category == ErrorCategory.AUTHORIZATION_ERROR.value:
            suggestions.extend([
                "Check user permissions and roles",
                "Verify tenant access rights",
                "Review API scope permissions"
            ])
        
        elif error_category == ErrorCategory.QUOTA_EXCEEDED.value:
            suggestions.extend([
                "Check API quota limits",
                "Review usage patterns",
                "Consider upgrading service plan",
                "Wait for quota reset"
            ])
        
        elif error_category == ErrorCategory.RATE_LIMITED.value:
            suggestions.extend([
                "Wait for rate limit reset",
                "Review request frequency",
                "Implement exponential backoff",
                "Consider distributing load"
            ])
        
        elif error_category == ErrorCategory.VALIDATION_ERROR.value:
            suggestions.extend([
                "Check input data format",
                "Verify required fields",
                "Review data validation rules"
            ])
        
        # Add operation-specific suggestions
        if operation_type == "webhook_delivery":
            suggestions.append("Verify webhook endpoint is accessible")
            suggestions.append("Check webhook URL configuration")
        
        elif operation_type == "document_processing":
            suggestions.append("Verify document format is supported")
            suggestions.append("Check file size limits")
        
        elif operation_type == "ai_processing":
            suggestions.append("Check AI service availability")
            suggestions.append("Verify model configuration")
        
        # Add requeue suggestion if appropriate
        if entry.requeued_count == 0:
            suggestions.append("Consider requeuing with updated configuration")
        elif entry.requeued_count < 3:
            suggestions.append("Consider requeuing with different retry policy")
        
        return suggestions
    
    async def check_health(self) -> Dict[str, Any]:
        """Check health of the dead letter service."""
        try:
            async with self._db_manager.get_session() as session:
                # Count total entries
                total_stmt = select(func.count()).select_from(DeadLetterModel)
                total_result = await session.execute(total_stmt)
                total_entries = total_result.scalar() or 0
                
                # Count unresolved entries
                unresolved_stmt = select(func.count()).select_from(DeadLetterModel).where(
                    DeadLetterModel.is_resolved == False
                )
                unresolved_result = await session.execute(unresolved_stmt)
                unresolved_entries = unresolved_result.scalar() or 0
                
                # Get actionable entries count
                actionable_entries = await self.get_actionable_dead_letters(limit=100)
            
            return {
                "status": "healthy",
                "total_entries": total_entries,
                "unresolved_entries": unresolved_entries,
                "actionable_entries": len(actionable_entries),
                "resolution_rate": (total_entries - unresolved_entries) / total_entries if total_entries > 0 else 0.0
            }
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


__all__ = ["DeadLetterService"]