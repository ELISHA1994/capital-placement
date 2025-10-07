"""
Dead letter queue service for failed webhook deliveries.

This service manages webhooks that have failed after all retry attempts,
providing functionality to manually retry, resolve, and manage dead letters.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog

from app.domain.interfaces import IWebhookDeadLetterService, IDatabase
from app.api.schemas.webhook_schemas import WebhookDeliveryStatus, WebhookFailureReason

logger = structlog.get_logger(__name__)


class WebhookDeadLetterService(IWebhookDeadLetterService):
    """Service for managing failed webhook deliveries in dead letter queue."""

    def __init__(self, database: IDatabase):
        """
        Initialize dead letter service.
        
        Args:
            database: Database interface for persistence
        """
        self.database = database
        
    async def check_health(self) -> Dict[str, Any]:
        """Return dead letter service health."""
        try:
            # Get dead letter statistics
            total_dead_letters = await self._get_dead_letter_count()
            unresolved_count = await self._get_unresolved_dead_letter_count()
            
            return {
                "status": "healthy",
                "service": "WebhookDeadLetterService",
                "total_dead_letters": total_dead_letters,
                "unresolved_dead_letters": unresolved_count,
                "resolution_rate": (
                    ((total_dead_letters - unresolved_count) / total_dead_letters * 100)
                    if total_dead_letters > 0 else 100.0
                )
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": "WebhookDeadLetterService",
                "error": str(e)
            }
    
    async def move_to_dead_letter(
        self,
        delivery_id: str,
        *,
        final_failure_reason: str,
        final_error_message: Optional[str] = None,
        moved_by: str = "system"
    ) -> str:
        """
        Move a failed delivery to dead letter queue.
        
        Args:
            delivery_id: Original delivery ID
            final_failure_reason: Final reason for failure
            final_error_message: Final error message
            moved_by: Who moved it to dead letter
            
        Returns:
            Dead letter ID
        """
        dead_letter_id = str(uuid4())
        
        try:
            # Get original delivery record
            delivery = await self.database.get_item(
                "webhook_deliveries",
                delivery_id,
                delivery_id
            )
            
            if not delivery:
                raise ValueError(f"Delivery record not found: {delivery_id}")
            
            # Create dead letter record
            dead_letter_record = {
                "id": dead_letter_id,
                "delivery_id": delivery_id,
                "tenant_id": delivery["tenant_id"],
                "endpoint_id": delivery["endpoint_id"],
                "event_type": delivery["event_type"],
                "event_id": delivery["event_id"],
                "payload": delivery["payload"],
                "total_attempts": delivery.get("attempt_number", 1),
                "final_failure_reason": final_failure_reason,
                "final_error_message": final_error_message,
                "first_attempted_at": delivery.get("first_attempted_at", delivery.get("scheduled_at")),
                "last_attempted_at": delivery.get("last_attempted_at", datetime.utcnow()),
                "dead_lettered_at": datetime.utcnow(),
                "dead_lettered_by": moved_by,
                "can_retry": True,
                "retry_count": 0,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Store dead letter record
            await self.database.create_item("webhook_dead_letters", dead_letter_record)
            
            # Update original delivery status
            await self.database.update_item(
                "webhook_deliveries",
                delivery_id,
                {
                    "status": WebhookDeliveryStatus.DEAD_LETTER.value,
                    "dead_letter_id": dead_letter_id,
                    "dead_lettered_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
            )
            
            logger.warning(
                "Webhook delivery moved to dead letter queue",
                delivery_id=delivery_id,
                dead_letter_id=dead_letter_id,
                endpoint_id=delivery["endpoint_id"],
                event_type=delivery["event_type"],
                final_failure_reason=final_failure_reason,
                total_attempts=delivery.get("attempt_number", 1),
                moved_by=moved_by
            )
            
            return dead_letter_id
            
        except Exception as e:
            logger.error(
                "Failed to move delivery to dead letter queue",
                delivery_id=delivery_id,
                final_failure_reason=final_failure_reason,
                moved_by=moved_by,
                error=str(e)
            )
            raise
    
    async def retry_dead_letter(
        self,
        dead_letter_id: str,
        *,
        admin_user_id: str,
        notes: Optional[str] = None
    ) -> str:
        """
        Retry a dead letter delivery.
        
        Args:
            dead_letter_id: Dead letter ID to retry
            admin_user_id: Admin performing the retry
            notes: Retry notes
            
        Returns:
            New delivery ID for retry
        """
        try:
            # Get dead letter record
            dead_letter = await self.database.get_item(
                "webhook_dead_letters",
                dead_letter_id,
                dead_letter_id
            )
            
            if not dead_letter:
                raise ValueError(f"Dead letter record not found: {dead_letter_id}")
            
            if not dead_letter.get("can_retry", True):
                raise ValueError(f"Dead letter cannot be retried: {dead_letter_id}")
            
            # Create new delivery record
            new_delivery_id = str(uuid4())
            
            delivery_record = {
                "id": new_delivery_id,
                "tenant_id": dead_letter["tenant_id"],
                "endpoint_id": dead_letter["endpoint_id"],
                "event_type": dead_letter["event_type"],
                "event_id": dead_letter["event_id"],
                "payload": dead_letter["payload"],
                "status": WebhookDeliveryStatus.PENDING.value,
                "attempt_number": 1,
                "max_attempts": 5,  # Reset max attempts for manual retry
                "scheduled_at": datetime.utcnow(),
                "correlation_id": f"dead_letter_retry_{dead_letter_id}",
                "priority": 10,  # Higher priority for manual retries
                "manual_retry": True,
                "retry_admin_user_id": admin_user_id,
                "retry_notes": notes,
                "original_dead_letter_id": dead_letter_id,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Store new delivery record
            await self.database.create_item("webhook_deliveries", delivery_record)
            
            # Update dead letter record
            retry_count = dead_letter.get("retry_count", 0) + 1
            await self.database.update_item(
                "webhook_dead_letters",
                dead_letter_id,
                {
                    "retry_count": retry_count,
                    "last_retry_at": datetime.utcnow(),
                    "last_retry_delivery_id": new_delivery_id,
                    "last_retry_admin_user_id": admin_user_id,
                    "last_retry_notes": notes,
                    "updated_at": datetime.utcnow()
                }
            )
            
            logger.info(
                "Dead letter retry scheduled",
                dead_letter_id=dead_letter_id,
                new_delivery_id=new_delivery_id,
                admin_user_id=admin_user_id,
                retry_count=retry_count,
                notes=notes
            )
            
            return new_delivery_id
            
        except Exception as e:
            logger.error(
                "Failed to retry dead letter",
                dead_letter_id=dead_letter_id,
                admin_user_id=admin_user_id,
                error=str(e)
            )
            raise
    
    async def resolve_dead_letter(
        self,
        dead_letter_id: str,
        *,
        resolution_action: str,
        admin_user_id: str,
        notes: Optional[str] = None
    ) -> bool:
        """
        Mark a dead letter as resolved.
        
        Args:
            dead_letter_id: Dead letter ID to resolve
            resolution_action: Action taken to resolve
            admin_user_id: Admin resolving the issue
            notes: Resolution notes
            
        Returns:
            True if resolution was successful
        """
        try:
            # Get dead letter record
            dead_letter = await self.database.get_item(
                "webhook_dead_letters",
                dead_letter_id,
                dead_letter_id
            )
            
            if not dead_letter:
                logger.warning("Dead letter not found for resolution", dead_letter_id=dead_letter_id)
                return False
            
            # Update dead letter record
            updates = {
                "reviewed_at": datetime.utcnow(),
                "reviewed_by": admin_user_id,
                "resolution_action": resolution_action,
                "resolution_notes": notes,
                "can_retry": False,  # Disable further retries once resolved
                "updated_at": datetime.utcnow()
            }
            
            await self.database.update_item("webhook_dead_letters", dead_letter_id, updates)
            
            logger.info(
                "Dead letter resolved",
                dead_letter_id=dead_letter_id,
                resolution_action=resolution_action,
                admin_user_id=admin_user_id,
                notes=notes
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to resolve dead letter",
                dead_letter_id=dead_letter_id,
                resolution_action=resolution_action,
                admin_user_id=admin_user_id,
                error=str(e)
            )
            return False
    
    async def get_dead_letters(
        self,
        *,
        tenant_id: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get dead letter queue entries.
        
        Args:
            tenant_id: Filter by tenant
            endpoint_id: Filter by endpoint
            event_type: Filter by event type
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum entries to return
            offset: Offset for pagination
            
        Returns:
            Dead letter entries with pagination info
        """
        try:
            # Build query conditions
            conditions = []
            parameters = []
            
            if tenant_id:
                conditions.append("tenant_id = %s")
                parameters.append({"name": "@tenant_id", "value": tenant_id})
            
            if endpoint_id:
                conditions.append("endpoint_id = %s")
                parameters.append({"name": "@endpoint_id", "value": endpoint_id})
            
            if event_type:
                conditions.append("event_type = %s")
                parameters.append({"name": "@event_type", "value": event_type})
            
            if start_date:
                conditions.append("dead_lettered_at >= %s")
                parameters.append({"name": "@start_date", "value": start_date})
            
            if end_date:
                conditions.append("dead_lettered_at <= %s")
                parameters.append({"name": "@end_date", "value": end_date})
            
            # Build query
            where_clause = ""
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)
            
            # Get total count
            count_query = f"SELECT COUNT(*) as total FROM webhook_dead_letters {where_clause}"
            count_result = await self.database.query_items(
                "webhook_dead_letters",
                count_query,
                parameters
            )
            total_count = count_result[0]["total"] if count_result else 0
            
            # Get dead letters
            data_query = f"""
                SELECT * FROM webhook_dead_letters {where_clause}
                ORDER BY dead_lettered_at DESC 
                OFFSET %s LIMIT %s
            """
            
            data_parameters = parameters + [
                {"name": "@offset", "value": offset},
                {"name": "@limit", "value": limit}
            ]
            
            dead_letters = await self.database.query_items(
                "webhook_dead_letters",
                data_query,
                data_parameters
            )
            
            # Calculate pagination info
            has_more = (offset + len(dead_letters)) < total_count
            next_offset = offset + len(dead_letters) if has_more else None
            
            return {
                "dead_letters": dead_letters,
                "pagination": {
                    "total_count": total_count,
                    "limit": limit,
                    "offset": offset,
                    "has_more": has_more,
                    "next_offset": next_offset,
                    "returned_count": len(dead_letters)
                }
            }
            
        except Exception as e:
            logger.error(
                "Failed to get dead letters",
                tenant_id=tenant_id,
                endpoint_id=endpoint_id,
                event_type=event_type,
                error=str(e)
            )
            
            return {
                "dead_letters": [],
                "pagination": {
                    "total_count": 0,
                    "limit": limit,
                    "offset": offset,
                    "has_more": False,
                    "next_offset": None,
                    "returned_count": 0
                },
                "error": str(e)
            }
    
    async def cleanup_old_dead_letters(
        self,
        *,
        older_than_days: int = 30,
        batch_size: int = 100
    ) -> int:
        """
        Clean up old dead letter entries.
        
        Args:
            older_than_days: Remove entries older than this
            batch_size: Batch size for cleanup
            
        Returns:
            Number of entries cleaned up
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
            total_cleaned = 0
            
            while True:
                # Get batch of old dead letters
                query = """
                    SELECT id FROM webhook_dead_letters 
                    WHERE dead_lettered_at < %s 
                    AND (reviewed_at IS NOT NULL OR can_retry = false)
                    ORDER BY dead_lettered_at ASC 
                    LIMIT %s
                """
                
                old_dead_letters = await self.database.query_items(
                    "webhook_dead_letters",
                    query,
                    [
                        {"name": "@cutoff_date", "value": cutoff_date},
                        {"name": "@batch_size", "value": batch_size}
                    ]
                )
                
                if not old_dead_letters:
                    break
                
                # Delete batch
                for dead_letter in old_dead_letters:
                    try:
                        await self.database.delete_item(
                            "webhook_dead_letters",
                            dead_letter["id"],
                            dead_letter["id"]
                        )
                        total_cleaned += 1
                        
                    except Exception as e:
                        logger.warning(
                            "Failed to delete old dead letter",
                            dead_letter_id=dead_letter["id"],
                            error=str(e)
                        )
                
                # Break if we got less than batch size (no more entries)
                if len(old_dead_letters) < batch_size:
                    break
            
            if total_cleaned > 0:
                logger.info(
                    "Cleaned up old dead letters",
                    total_cleaned=total_cleaned,
                    older_than_days=older_than_days,
                    cutoff_date=cutoff_date
                )
            
            return total_cleaned
            
        except Exception as e:
            logger.error(
                "Failed to cleanup old dead letters",
                older_than_days=older_than_days,
                batch_size=batch_size,
                error=str(e)
            )
            return 0
    
    async def get_dead_letter_summary(
        self,
        *,
        tenant_id: Optional[str] = None,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """
        Get summary statistics for dead letters.
        
        Args:
            tenant_id: Filter by tenant
            days_back: Number of days to look back
            
        Returns:
            Summary statistics
        """
        try:
            start_date = datetime.utcnow() - timedelta(days=days_back)
            
            # Build query conditions
            conditions = ["dead_lettered_at >= %s"]
            parameters = [{"name": "@start_date", "value": start_date}]
            
            if tenant_id:
                conditions.append("tenant_id = %s")
                parameters.append({"name": "@tenant_id", "value": tenant_id})
            
            where_clause = "WHERE " + " AND ".join(conditions)
            
            # Get summary statistics
            summary_query = f"""
                SELECT 
                    COUNT(*) as total_dead_letters,
                    COUNT(CASE WHEN reviewed_at IS NOT NULL THEN 1 END) as resolved_count,
                    COUNT(CASE WHEN can_retry = true THEN 1 END) as retryable_count,
                    COUNT(CASE WHEN retry_count > 0 THEN 1 END) as retry_attempted_count,
                    AVG(total_attempts) as avg_attempts_before_dead_letter,
                    MAX(total_attempts) as max_attempts_before_dead_letter
                FROM webhook_dead_letters {where_clause}
            """
            
            summary_result = await self.database.query_items(
                "webhook_dead_letters",
                summary_query,
                parameters
            )
            
            summary = summary_result[0] if summary_result else {}
            
            # Get breakdown by failure reason
            reason_query = f"""
                SELECT 
                    final_failure_reason,
                    COUNT(*) as count
                FROM webhook_dead_letters {where_clause}
                GROUP BY final_failure_reason
                ORDER BY count DESC
            """
            
            reason_breakdown = await self.database.query_items(
                "webhook_dead_letters",
                reason_query,
                parameters
            )
            
            # Get breakdown by endpoint
            endpoint_query = f"""
                SELECT 
                    endpoint_id,
                    COUNT(*) as count
                FROM webhook_dead_letters {where_clause}
                GROUP BY endpoint_id
                ORDER BY count DESC
                LIMIT 10
            """
            
            endpoint_breakdown = await self.database.query_items(
                "webhook_dead_letters",
                endpoint_query,
                parameters
            )
            
            return {
                "period_days": days_back,
                "start_date": start_date,
                "summary": {
                    "total_dead_letters": summary.get("total_dead_letters", 0),
                    "resolved_count": summary.get("resolved_count", 0),
                    "retryable_count": summary.get("retryable_count", 0),
                    "retry_attempted_count": summary.get("retry_attempted_count", 0),
                    "avg_attempts_before_dead_letter": summary.get("avg_attempts_before_dead_letter", 0),
                    "max_attempts_before_dead_letter": summary.get("max_attempts_before_dead_letter", 0)
                },
                "failure_reason_breakdown": reason_breakdown,
                "endpoint_breakdown": endpoint_breakdown
            }
            
        except Exception as e:
            logger.error(
                "Failed to get dead letter summary",
                tenant_id=tenant_id,
                days_back=days_back,
                error=str(e)
            )
            
            return {
                "period_days": days_back,
                "summary": {},
                "failure_reason_breakdown": [],
                "endpoint_breakdown": [],
                "error": str(e)
            }
    
    async def _get_dead_letter_count(self) -> int:
        """Get total count of dead letters."""
        try:
            query = "SELECT COUNT(*) as count FROM webhook_dead_letters"
            result = await self.database.query_items("webhook_dead_letters", query)
            return result[0]["count"] if result else 0
        except Exception:
            return 0
    
    async def _get_unresolved_dead_letter_count(self) -> int:
        """Get count of unresolved dead letters."""
        try:
            query = "SELECT COUNT(*) as count FROM webhook_dead_letters WHERE reviewed_at IS NULL"
            result = await self.database.query_items("webhook_dead_letters", query)
            return result[0]["count"] if result else 0
        except Exception:
            return 0


__all__ = ["WebhookDeadLetterService"]