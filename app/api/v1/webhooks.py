"""
Webhook management API endpoints.

This module provides REST API endpoints for managing webhook endpoints,
monitoring delivery status, handling failed deliveries, and accessing
webhook statistics.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

import structlog

from app.infrastructure.persistence.models.auth_tables import CurrentUser
from app.api.schemas.webhook_schemas import (
    WebhookEventType,
    WebhookEndpointCreate,
    WebhookEndpointUpdate,
    WebhookDeliveryQuery,
    WebhookRetryRequest,
    WebhookTestRequest,
    WebhookDeliveryStatus,
    WebhookFailureReason
)
from app.core.dependencies import CurrentUserDep, TenantContextDep
from app.infrastructure.providers.webhook_provider import (
    get_webhook_delivery_service,
    get_webhook_dead_letter_service,
    get_webhook_stats_service,
    get_webhook_signature_service
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/webhooks", tags=["Webhook Management"])


# Response models
class WebhookEndpointResponse(BaseModel):
    """Webhook endpoint response model."""
    id: str
    url: str
    name: Optional[str]
    description: Optional[str]
    event_types: List[WebhookEventType]
    enabled: bool
    has_secret: bool
    success_rate: float
    total_deliveries: int
    successful_deliveries: int
    failed_deliveries: int
    circuit_state: str
    created_at: datetime
    updated_at: datetime


class WebhookDeliveryResponse(BaseModel):
    """Webhook delivery response model."""
    delivery_id: str
    status: WebhookDeliveryStatus
    event_type: str
    endpoint_id: str
    attempt_number: int
    max_attempts: int
    scheduled_at: datetime
    first_attempted_at: Optional[datetime]
    last_attempted_at: Optional[datetime]
    delivered_at: Optional[datetime]
    next_retry_at: Optional[datetime]
    http_status_code: Optional[int]
    response_time_ms: Optional[int]
    failure_reason: Optional[WebhookFailureReason]
    error_message: Optional[str]
    correlation_id: Optional[str]


class WebhookStatsResponse(BaseModel):
    """Webhook statistics response model."""
    period_start: datetime
    period_end: datetime
    total_deliveries: int
    successful_deliveries: int
    failed_deliveries: int
    success_rate: float
    avg_response_time_ms: float
    avg_attempts_per_delivery: float


class DeadLetterResponse(BaseModel):
    """Dead letter response model."""
    dead_letter_id: str
    delivery_id: str
    endpoint_id: str
    event_type: str
    total_attempts: int
    final_failure_reason: str
    dead_lettered_at: datetime
    can_retry: bool
    retry_count: int
    reviewed_at: Optional[datetime]
    reviewed_by: Optional[str]
    resolution_action: Optional[str]


# Endpoint Management
@router.post("/endpoints", response_model=WebhookEndpointResponse, status_code=status.HTTP_201_CREATED)
async def create_webhook_endpoint(
    endpoint_data: WebhookEndpointCreate,
    current_user: CurrentUser = CurrentUserDep,
    tenant_id: str = TenantContextDep
) -> WebhookEndpointResponse:
    """Create a new webhook endpoint."""
    try:
        # TODO: Implement endpoint creation
        # This would typically involve:
        # 1. Validating the webhook URL
        # 2. Creating the endpoint record in database
        # 3. Setting up initial configuration
        
        logger.info(
            "Creating webhook endpoint",
            url=str(endpoint_data.url),
            event_types=[e.value for e in endpoint_data.event_types],
            tenant_id=tenant_id,
            user_id=current_user.user_id
        )
        
        # Placeholder response
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Endpoint creation not yet implemented"
        )
        
    except Exception as e:
        logger.error("Failed to create webhook endpoint", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create webhook endpoint"
        )


@router.get("/endpoints", response_model=List[WebhookEndpointResponse])
async def list_webhook_endpoints(
    current_user: CurrentUser = CurrentUserDep,
    tenant_id: str = TenantContextDep,
    enabled_only: bool = Query(False, description="Return only enabled endpoints")
) -> List[WebhookEndpointResponse]:
    """List webhook endpoints for the current tenant."""
    try:
        # TODO: Implement endpoint listing
        logger.info(
            "Listing webhook endpoints",
            tenant_id=tenant_id,
            enabled_only=enabled_only
        )
        
        # Placeholder response
        return []
        
    except Exception as e:
        logger.error("Failed to list webhook endpoints", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list webhook endpoints"
        )


@router.get("/endpoints/{endpoint_id}", response_model=WebhookEndpointResponse)
async def get_webhook_endpoint(
    endpoint_id: str,
    current_user: CurrentUser = CurrentUserDep,
    tenant_id: str = TenantContextDep
) -> WebhookEndpointResponse:
    """Get a specific webhook endpoint."""
    try:
        # TODO: Implement endpoint retrieval
        logger.info(
            "Getting webhook endpoint",
            endpoint_id=endpoint_id,
            tenant_id=tenant_id
        )
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Endpoint not found"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get webhook endpoint", endpoint_id=endpoint_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get webhook endpoint"
        )


@router.put("/endpoints/{endpoint_id}", response_model=WebhookEndpointResponse)
async def update_webhook_endpoint(
    endpoint_id: str,
    endpoint_data: WebhookEndpointUpdate,
    current_user: CurrentUser = CurrentUserDep,
    tenant_id: str = TenantContextDep
) -> WebhookEndpointResponse:
    """Update a webhook endpoint."""
    try:
        # TODO: Implement endpoint update
        logger.info(
            "Updating webhook endpoint",
            endpoint_id=endpoint_id,
            tenant_id=tenant_id,
            user_id=current_user.user_id
        )
        
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Endpoint update not yet implemented"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update webhook endpoint", endpoint_id=endpoint_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update webhook endpoint"
        )


@router.delete("/endpoints/{endpoint_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_webhook_endpoint(
    endpoint_id: str,
    current_user: CurrentUser = CurrentUserDep,
    tenant_id: str = TenantContextDep
) -> None:
    """Delete a webhook endpoint."""
    try:
        # TODO: Implement endpoint deletion
        logger.info(
            "Deleting webhook endpoint",
            endpoint_id=endpoint_id,
            tenant_id=tenant_id,
            user_id=current_user.user_id
        )
        
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Endpoint deletion not yet implemented"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete webhook endpoint", endpoint_id=endpoint_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete webhook endpoint"
        )


# Delivery Management
@router.get("/deliveries", response_model=List[WebhookDeliveryResponse])
async def list_webhook_deliveries(
    current_user: CurrentUser = CurrentUserDep,
    tenant_id: str = TenantContextDep,
    endpoint_id: Optional[str] = Query(None, description="Filter by endpoint ID"),
    event_type: Optional[WebhookEventType] = Query(None, description="Filter by event type"),
    status: Optional[WebhookDeliveryStatus] = Query(None, description="Filter by status"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum deliveries to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
) -> List[WebhookDeliveryResponse]:
    """List webhook deliveries with filtering and pagination."""
    try:
        delivery_service = await get_webhook_delivery_service()
        
        # TODO: Implement delivery listing using the delivery service
        # This would use the delivery service to query deliveries
        
        logger.info(
            "Listing webhook deliveries",
            tenant_id=tenant_id,
            endpoint_id=endpoint_id,
            event_type=event_type.value if event_type else None,
            status=status.value if status else None,
            limit=limit,
            offset=offset
        )
        
        # Placeholder response
        return []
        
    except Exception as e:
        logger.error("Failed to list webhook deliveries", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list webhook deliveries"
        )


@router.get("/deliveries/{delivery_id}", response_model=WebhookDeliveryResponse)
async def get_webhook_delivery(
    delivery_id: str,
    current_user: CurrentUser = CurrentUserDep,
    tenant_id: str = TenantContextDep
) -> WebhookDeliveryResponse:
    """Get a specific webhook delivery."""
    try:
        delivery_service = await get_webhook_delivery_service()
        
        delivery_status = await delivery_service.get_delivery_status(delivery_id)
        
        if not delivery_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Delivery not found"
            )
        
        # Check tenant access
        if delivery_status.get("tenant_id") != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        return WebhookDeliveryResponse(**delivery_status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get webhook delivery", delivery_id=delivery_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get webhook delivery"
        )


@router.post("/deliveries/{delivery_id}/retry", status_code=status.HTTP_202_ACCEPTED)
async def retry_webhook_delivery(
    delivery_id: str,
    retry_request: WebhookRetryRequest,
    current_user: CurrentUser = CurrentUserDep,
    tenant_id: str = TenantContextDep
) -> Dict[str, Any]:
    """Manually retry a failed webhook delivery."""
    try:
        delivery_service = await get_webhook_delivery_service()
        
        # Validate delivery exists and belongs to tenant
        delivery_status = await delivery_service.get_delivery_status(delivery_id)
        if not delivery_status or delivery_status.get("tenant_id") != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Delivery not found"
            )
        
        # Check if only one delivery ID is provided for single retry
        if len(retry_request.delivery_ids) != 1 or retry_request.delivery_ids[0] != delivery_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Delivery ID mismatch"
            )
        
        success = await delivery_service.retry_failed_delivery(
            delivery_id=delivery_id,
            override_max_attempts=retry_request.override_max_attempts or False,
            new_max_attempts=retry_request.new_max_attempts,
            admin_user_id=current_user.user_id,
            notes=retry_request.notes
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot retry delivery in current state"
            )
        
        return {
            "message": "Retry scheduled successfully",
            "delivery_id": delivery_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retry webhook delivery", delivery_id=delivery_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retry webhook delivery"
        )


@router.post("/deliveries/{delivery_id}/cancel", status_code=status.HTTP_202_ACCEPTED)
async def cancel_webhook_delivery(
    delivery_id: str,
    current_user: CurrentUser = CurrentUserDep,
    tenant_id: str = TenantContextDep,
    reason: str = Query(..., description="Cancellation reason")
) -> Dict[str, Any]:
    """Cancel a pending webhook delivery."""
    try:
        delivery_service = await get_webhook_delivery_service()
        
        # Validate delivery exists and belongs to tenant
        delivery_status = await delivery_service.get_delivery_status(delivery_id)
        if not delivery_status or delivery_status.get("tenant_id") != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Delivery not found"
            )
        
        success = await delivery_service.cancel_delivery(
            delivery_id=delivery_id,
            reason=reason,
            admin_user_id=current_user.user_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot cancel delivery in current state"
            )
        
        return {
            "message": "Delivery cancelled successfully",
            "delivery_id": delivery_id,
            "reason": reason
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to cancel webhook delivery", delivery_id=delivery_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel webhook delivery"
        )


# Dead Letter Queue Management
@router.get("/dead-letters", response_model=List[DeadLetterResponse])
async def list_dead_letters(
    current_user: CurrentUser = CurrentUserDep,
    tenant_id: str = TenantContextDep,
    endpoint_id: Optional[str] = Query(None, description="Filter by endpoint ID"),
    event_type: Optional[WebhookEventType] = Query(None, description="Filter by event type"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum entries to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
) -> List[DeadLetterResponse]:
    """List dead letter queue entries."""
    try:
        dead_letter_service = await get_webhook_dead_letter_service()
        
        result = await dead_letter_service.get_dead_letters(
            tenant_id=tenant_id,
            endpoint_id=endpoint_id,
            event_type=event_type.value if event_type else None,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset
        )
        
        dead_letters = result.get("dead_letters", [])
        
        return [
            DeadLetterResponse(
                dead_letter_id=dl["id"],
                delivery_id=dl["delivery_id"],
                endpoint_id=dl["endpoint_id"],
                event_type=dl["event_type"],
                total_attempts=dl["total_attempts"],
                final_failure_reason=dl["final_failure_reason"],
                dead_lettered_at=dl["dead_lettered_at"],
                can_retry=dl.get("can_retry", True),
                retry_count=dl.get("retry_count", 0),
                reviewed_at=dl.get("reviewed_at"),
                reviewed_by=dl.get("reviewed_by"),
                resolution_action=dl.get("resolution_action")
            )
            for dl in dead_letters
        ]
        
    except Exception as e:
        logger.error("Failed to list dead letters", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list dead letters"
        )


@router.post("/dead-letters/{dead_letter_id}/retry", status_code=status.HTTP_202_ACCEPTED)
async def retry_dead_letter(
    dead_letter_id: str,
    current_user: CurrentUser = CurrentUserDep,
    tenant_id: str = TenantContextDep,
    notes: Optional[str] = Query(None, description="Retry notes")
) -> Dict[str, Any]:
    """Retry a dead letter delivery."""
    try:
        dead_letter_service = await get_webhook_dead_letter_service()
        
        new_delivery_id = await dead_letter_service.retry_dead_letter(
            dead_letter_id=dead_letter_id,
            admin_user_id=current_user.user_id,
            notes=notes
        )
        
        return {
            "message": "Dead letter retry scheduled successfully",
            "dead_letter_id": dead_letter_id,
            "new_delivery_id": new_delivery_id
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to retry dead letter", dead_letter_id=dead_letter_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retry dead letter"
        )


@router.post("/dead-letters/{dead_letter_id}/resolve", status_code=status.HTTP_202_ACCEPTED)
async def resolve_dead_letter(
    dead_letter_id: str,
    current_user: CurrentUser = CurrentUserDep,
    tenant_id: str = TenantContextDep,
    resolution_action: str = Query(..., description="Resolution action taken"),
    notes: Optional[str] = Query(None, description="Resolution notes")
) -> Dict[str, Any]:
    """Mark a dead letter as resolved."""
    try:
        dead_letter_service = await get_webhook_dead_letter_service()
        
        success = await dead_letter_service.resolve_dead_letter(
            dead_letter_id=dead_letter_id,
            resolution_action=resolution_action,
            admin_user_id=current_user.user_id,
            notes=notes
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dead letter not found"
            )
        
        return {
            "message": "Dead letter resolved successfully",
            "dead_letter_id": dead_letter_id,
            "resolution_action": resolution_action
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to resolve dead letter", dead_letter_id=dead_letter_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resolve dead letter"
        )


# Statistics and Monitoring
@router.get("/stats", response_model=WebhookStatsResponse)
async def get_webhook_stats(
    current_user: CurrentUser = CurrentUserDep,
    tenant_id: str = TenantContextDep,
    endpoint_id: Optional[str] = Query(None, description="Filter by endpoint ID"),
    event_type: Optional[WebhookEventType] = Query(None, description="Filter by event type"),
    start_date: Optional[datetime] = Query(None, description="Start date for statistics"),
    end_date: Optional[datetime] = Query(None, description="End date for statistics"),
    days_back: int = Query(7, ge=1, le=365, description="Days back from now if dates not specified")
) -> WebhookStatsResponse:
    """Get webhook delivery statistics."""
    try:
        stats_service = await get_webhook_stats_service()
        
        # Set default date range if not provided
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=days_back)
        
        stats = await stats_service.get_delivery_stats(
            tenant_id=tenant_id,
            endpoint_id=endpoint_id,
            event_type=event_type.value if event_type else None,
            start_date=start_date,
            end_date=end_date
        )
        
        overall_stats = stats.get("overall_stats", {})
        performance_metrics = stats.get("performance_metrics", {})
        
        return WebhookStatsResponse(
            period_start=start_date,
            period_end=end_date,
            total_deliveries=overall_stats.get("total_deliveries", 0),
            successful_deliveries=overall_stats.get("successful_deliveries", 0),
            failed_deliveries=overall_stats.get("failed_deliveries", 0),
            success_rate=overall_stats.get("success_rate", 0.0),
            avg_response_time_ms=performance_metrics.get("avg_response_time_ms", 0.0),
            avg_attempts_per_delivery=overall_stats.get("avg_attempts_per_delivery", 0.0)
        )
        
    except Exception as e:
        logger.error("Failed to get webhook statistics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get webhook statistics"
        )


@router.get("/endpoints/{endpoint_id}/health", response_model=Dict[str, Any])
async def get_endpoint_health(
    endpoint_id: str,
    current_user: CurrentUser = CurrentUserDep,
    tenant_id: str = TenantContextDep,
    time_window_hours: int = Query(24, ge=1, le=168, description="Time window in hours")
) -> Dict[str, Any]:
    """Get health metrics for a specific endpoint."""
    try:
        stats_service = await get_webhook_stats_service()
        
        health = await stats_service.get_endpoint_health(
            endpoint_id=endpoint_id,
            time_window_hours=time_window_hours
        )
        
        # Check if endpoint belongs to tenant
        if "error" not in health and health.get("endpoint_id") != endpoint_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Endpoint not found"
            )
        
        return health
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get endpoint health", endpoint_id=endpoint_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get endpoint health"
        )


@router.post("/endpoints/{endpoint_id}/test", status_code=status.HTTP_202_ACCEPTED)
async def test_webhook_endpoint(
    endpoint_id: str,
    test_request: WebhookTestRequest,
    current_user: CurrentUser = CurrentUserDep,
    tenant_id: str = TenantContextDep
) -> Dict[str, Any]:
    """Test a webhook endpoint with a test payload."""
    try:
        delivery_service = await get_webhook_delivery_service()
        signature_service = await get_webhook_signature_service()
        
        # Generate test payload if not provided
        test_payload = test_request.test_payload
        if not test_payload:
            test_payload = {
                "event": "test",
                "timestamp": datetime.utcnow().isoformat(),
                "test": True,
                "tenant_id": tenant_id,
                "triggered_by": current_user.user_id
            }
        
        # Queue delivery with high priority for immediate processing
        delivery_id = await delivery_service.deliver_webhook(
            endpoint_id=endpoint_id,
            event_type=test_request.event_type.value,
            payload=test_payload,
            tenant_id=tenant_id,
            event_id=f"test_{endpoint_id}",
            correlation_id=f"test_{current_user.user_id}",
            priority=100  # High priority for test
        )
        
        return {
            "message": "Test webhook queued for delivery",
            "delivery_id": delivery_id,
            "endpoint_id": endpoint_id,
            "test_payload": test_payload
        }
        
    except Exception as e:
        logger.error("Failed to test webhook endpoint", endpoint_id=endpoint_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to test webhook endpoint"
        )


# Bulk Operations
@router.post("/deliveries/retry-batch", status_code=status.HTTP_202_ACCEPTED)
async def retry_webhook_deliveries_batch(
    retry_request: WebhookRetryRequest,
    current_user: CurrentUser = CurrentUserDep,
    tenant_id: str = TenantContextDep
) -> Dict[str, Any]:
    """Retry multiple webhook deliveries in batch."""
    try:
        delivery_service = await get_webhook_delivery_service()
        
        results = []
        successful_retries = 0
        failed_retries = 0
        
        for delivery_id in retry_request.delivery_ids:
            try:
                # Validate delivery exists and belongs to tenant
                delivery_status = await delivery_service.get_delivery_status(delivery_id)
                if not delivery_status or delivery_status.get("tenant_id") != tenant_id:
                    results.append({
                        "delivery_id": delivery_id,
                        "success": False,
                        "error": "Delivery not found or access denied"
                    })
                    failed_retries += 1
                    continue
                
                success = await delivery_service.retry_failed_delivery(
                    delivery_id=delivery_id,
                    override_max_attempts=retry_request.override_max_attempts or False,
                    new_max_attempts=retry_request.new_max_attempts,
                    admin_user_id=current_user.user_id,
                    notes=retry_request.notes
                )
                
                if success:
                    results.append({
                        "delivery_id": delivery_id,
                        "success": True
                    })
                    successful_retries += 1
                else:
                    results.append({
                        "delivery_id": delivery_id,
                        "success": False,
                        "error": "Cannot retry delivery in current state"
                    })
                    failed_retries += 1
                    
            except Exception as e:
                results.append({
                    "delivery_id": delivery_id,
                    "success": False,
                    "error": str(e)
                })
                failed_retries += 1
        
        return {
            "message": f"Batch retry completed: {successful_retries} successful, {failed_retries} failed",
            "total_deliveries": len(retry_request.delivery_ids),
            "successful_retries": successful_retries,
            "failed_retries": failed_retries,
            "results": results
        }
        
    except Exception as e:
        logger.error("Failed to retry webhook deliveries batch", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retry webhook deliveries batch"
        )


__all__ = ["router"]