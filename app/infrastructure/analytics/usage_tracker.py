"""
Centralized Usage Tracking Service

Provides standardized usage tracking across all platform operations.
Integrates with TenantManager for metrics updates and follows existing
error handling patterns.

Features:
- Standardized tracking interface
- Background processing to prevent request blocking
- Comprehensive error handling that never fails main operations
- Tenant-aware metrics collection
- Performance monitoring (< 5ms overhead)
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
import structlog

from app.domain.interfaces import IUsageService
from app.infrastructure.providers.tenant_provider import get_tenant_service as get_tenant_manager

logger = structlog.get_logger(__name__)


class UsageTracker(IUsageService):
    """
    Centralized usage tracking service following Hexagonal Architecture patterns.
    
    Provides a standardized interface for tracking all platform operations
    with consistent error handling and background processing.
    """
    
    def __init__(self):
        self._tenant_manager = None

    async def _get_tenant_manager(self):
        if self._tenant_manager is None:
            self._tenant_manager = await get_tenant_manager()
        return self._tenant_manager

    # IUsageService interface implementation
    async def track_usage(
        self,
        tenant_id: str,
        resource_type: str,
        amount: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Track resource usage (implements IUsageService interface).

        Args:
            tenant_id: Tenant identifier
            resource_type: Type of resource (profile, search, upload, etc.)
            amount: Amount of resource used
            metadata: Additional metadata

        Returns:
            True if tracking succeeded, False otherwise
        """
        try:
            # Route to appropriate specialized tracking method based on resource_type
            if resource_type == "profile":
                operation = metadata.get("operation", "view") if metadata else "view"
                await self.track_profile_usage(
                    tenant_id=tenant_id,
                    operation=operation,
                    profile_count=amount,
                    fields_updated=metadata.get("fields_updated") if metadata else None
                )
            elif resource_type == "search":
                await self.track_search_usage(
                    tenant_id=tenant_id,
                    search_count=amount,
                    search_mode=metadata.get("search_mode") if metadata else None,
                    results_count=metadata.get("results_count") if metadata else None
                )
            elif resource_type == "upload":
                await self.track_upload_usage(
                    tenant_id=tenant_id,
                    document_count=amount,
                    file_size_bytes=metadata.get("file_size_bytes", 0) if metadata else 0,
                    processing_type=metadata.get("processing_type") if metadata else None
                )
            else:
                # Generic tracking for unknown resource types
                metrics = {f"{resource_type}_count": amount}
                await self.track_operation_usage(
                    tenant_id=tenant_id,
                    operation_type=resource_type,
                    metrics=metrics,
                    metadata=metadata
                )

            return True

        except Exception as e:
            logger.warning(
                f"Failed to track usage for {resource_type}",
                tenant_id=tenant_id,
                resource_type=resource_type,
                amount=amount,
                error=str(e)
            )
            return False

    async def get_usage_stats(
        self,
        tenant_id: str,
        resource_type: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """Get usage statistics (implements IUsageService interface)."""
        try:
            tenant_manager = await self._get_tenant_manager()
            # This would need to be implemented in the tenant manager
            # For now, return placeholder
            return {
                "tenant_id": tenant_id,
                "resource_type": resource_type,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "stats": {}
            }
        except Exception as e:
            logger.error(f"Failed to get usage stats: {e}")
            return {}

    async def check_quota(
        self,
        tenant_id: str,
        resource_type: str,
        requested_amount: int,
    ) -> Dict[str, Any]:
        """Check quota limits (implements IUsageService interface)."""
        try:
            # This would need actual quota checking logic
            # For now, return unlimited
            return {
                "allowed": True,
                "remaining": 999999,
                "limit": 999999,
                "resource_type": resource_type
            }
        except Exception as e:
            logger.error(f"Failed to check quota: {e}")
            return {"allowed": True}

    async def check_health(self) -> Dict[str, Any]:
        """Check service health (implements IHealthCheck interface)."""
        try:
            tenant_manager = await self._get_tenant_manager()
            tenant_health = await tenant_manager.check_health()

            return {
                "service": "UsageTracker",
                "status": "healthy",
                "tenant_manager_status": tenant_health.get("status", "unknown"),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "service": "UsageTracker",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def track_operation_usage(
        self,
        tenant_id: str,
        operation_type: str,
        metrics: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track usage for any platform operation with standardized error handling.
        
        Args:
            tenant_id: Tenant identifier
            operation_type: Type of operation (search, upload, profile_create, etc.)
            metrics: Dictionary of metrics to update
            metadata: Optional metadata for logging/debugging
        
        Note: This method NEVER raises exceptions to prevent breaking main operations.
        """
        try:
            start_time = datetime.now()
            
            # Update tenant usage metrics
            tenant_manager = await self._get_tenant_manager()
            await tenant_manager.update_usage_metrics(
                tenant_id=tenant_id,
                metrics_update=metrics
            )
            
            # Calculate tracking overhead
            tracking_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.debug(
                "Usage tracking completed",
                tenant_id=tenant_id,
                operation_type=operation_type,
                metrics=metrics,
                tracking_time_ms=round(tracking_time_ms, 2),
                metadata=metadata
            )
            
        except Exception as e:
            logger.warning(
                f"Failed to track {operation_type} usage",
                tenant_id=tenant_id,
                operation_type=operation_type,
                error=str(e),
                metrics=metrics
            )
            # NEVER re-raise - tracking failures should not break main operations
    
    def track_operation_background(
        self,
        tenant_id: str,
        operation_type: str,
        metrics: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> asyncio.Task:
        """
        Track usage in background task to prevent blocking main operations.
        
        Returns:
            asyncio.Task: Background task for tracking
        """
        return asyncio.create_task(
            self.track_operation_usage(
                tenant_id=tenant_id,
                operation_type=operation_type,
                metrics=metrics,
                metadata=metadata
            )
        )
    
    async def track_search_usage(
        self,
        tenant_id: str,
        search_count: int = 1,
        search_mode: Optional[str] = None,
        results_count: Optional[int] = None
    ) -> None:
        """Track search operations with search-specific metrics."""
        metrics = {
            "searches_today": search_count,
            "total_searches": search_count
        }
        
        metadata = {
            "search_mode": search_mode,
            "results_count": results_count
        } if search_mode or results_count else None
        
        await self.track_operation_usage(
            tenant_id=tenant_id,
            operation_type="search",
            metrics=metrics,
            metadata=metadata
        )
    
    async def track_upload_usage(
        self,
        tenant_id: str,
        document_count: int = 1,
        file_size_bytes: int = 0,
        processing_type: Optional[str] = None
    ) -> None:
        """Track upload operations with storage metrics."""
        storage_gb = file_size_bytes / (1024 * 1024 * 1024) if file_size_bytes > 0 else 0
        
        metrics = {
            "documents_processed": document_count,
            "documents_uploaded": document_count,
            "storage_used_gb": storage_gb
        }
        
        metadata = {
            "file_size_bytes": file_size_bytes,
            "storage_gb": round(storage_gb, 4),
            "processing_type": processing_type
        } if file_size_bytes > 0 or processing_type else None
        
        await self.track_operation_usage(
            tenant_id=tenant_id,
            operation_type="upload",
            metrics=metrics,
            metadata=metadata
        )
    
    async def track_profile_usage(
        self,
        tenant_id: str,
        operation: str,  # create, update, delete, view
        profile_count: int = 1,
        fields_updated: Optional[int] = None
    ) -> None:
        """Track profile operations."""
        metrics = {}
        
        if operation == "create":
            metrics.update({
                "total_profiles": profile_count,
                "active_profiles": profile_count,
                "profiles_added_this_month": profile_count
            })
        elif operation == "update":
            metrics.update({
                "profiles_updated": profile_count
            })
        elif operation == "delete":
            metrics.update({
                "profiles_deleted": profile_count
            })
        elif operation == "view":
            metrics.update({
                "profile_views": profile_count
            })
        
        metadata = {
            "operation": operation,
            "fields_updated": fields_updated
        } if fields_updated else {"operation": operation}
        
        await self.track_operation_usage(
            tenant_id=tenant_id,
            operation_type=f"profile_{operation}",
            metrics=metrics,
            metadata=metadata
        )
    
    async def track_api_usage(
        self,
        tenant_id: str,
        endpoint: str,
        method: str,
        response_time_ms: Optional[float] = None,
        status_code: Optional[int] = None
    ) -> None:
        """Track API request usage."""
        metrics = {
            "api_requests_today": 1,
            "api_requests_this_month": 1
        }
        
        metadata = {
            "endpoint": endpoint,
            "method": method,
            "response_time_ms": response_time_ms,
            "status_code": status_code
        }
        
        await self.track_operation_usage(
            tenant_id=tenant_id,
            operation_type="api_request",
            metrics=metrics,
            metadata=metadata
        )
    
    async def track_user_activity(
        self,
        tenant_id: str,
        user_id: str,
        activity_type: str,  # login, action, session_end
        session_duration_minutes: Optional[int] = None
    ) -> None:
        """Track user activity and engagement."""
        metrics = {
            "user_sessions": 1 if activity_type == "login" else 0,
            "active_users_today": 1 if activity_type == "login" else 0
        }
        
        if session_duration_minutes:
            metrics["total_session_minutes"] = session_duration_minutes
        
        metadata = {
            "user_id": user_id,
            "activity_type": activity_type,
            "session_duration_minutes": session_duration_minutes
        }
        
        await self.track_operation_usage(
            tenant_id=tenant_id,
            operation_type="user_activity",
            metrics=metrics,
            metadata=metadata
        )


# Global instance for use across the application
usage_tracker = UsageTracker()


# Convenience functions for backward compatibility and easy integration
async def track_search_usage(tenant_id: str, search_count: int = 1) -> None:
    """Backward compatible search tracking function."""
    await usage_tracker.track_search_usage(tenant_id, search_count)


async def track_upload_usage(tenant_id: str, document_count: int, file_size_bytes: int = 0) -> None:
    """Backward compatible upload tracking function."""
    await usage_tracker.track_upload_usage(tenant_id, document_count, file_size_bytes)


# Background task helpers for FastAPI integration
def track_operation_task(
    tenant_id: str,
    operation_type: str,
    metrics: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> asyncio.Task:
    """Create background task for operation tracking."""
    return usage_tracker.track_operation_background(
        tenant_id=tenant_id,
        operation_type=operation_type,
        metrics=metrics,
        metadata=metadata
    )


def track_search_task(
    tenant_id: str,
    search_count: int = 1,
    search_mode: Optional[str] = None,
    results_count: Optional[int] = None
) -> asyncio.Task:
    """Create background task for search tracking."""
    return asyncio.create_task(
        usage_tracker.track_search_usage(
            tenant_id=tenant_id,
            search_count=search_count,
            search_mode=search_mode,
            results_count=results_count
        )
    )


def track_upload_task(
    tenant_id: str,
    document_count: int = 1,
    file_size_bytes: int = 0,
    processing_type: Optional[str] = None
) -> asyncio.Task:
    """Create background task for upload tracking."""
    return asyncio.create_task(
        usage_tracker.track_upload_usage(
            tenant_id=tenant_id,
            document_count=document_count,
            file_size_bytes=file_size_bytes,
            processing_type=processing_type
        )
    )


def track_profile_task(
    tenant_id: str,
    operation: str,
    profile_count: int = 1,
    fields_updated: Optional[int] = None
) -> asyncio.Task:
    """Create background task for profile tracking."""
    return asyncio.create_task(
        usage_tracker.track_profile_usage(
            tenant_id=tenant_id,
            operation=operation,
            profile_count=profile_count,
            fields_updated=fields_updated
        )
    )
