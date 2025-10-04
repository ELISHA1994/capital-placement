"""
API endpoints for audit log management and compliance reporting.

This module provides secure endpoints for querying audit logs, generating compliance 
reports, and performing integrity verification following security best practices.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.dependencies import CurrentUserDep, TenantContextDep
from app.domain.interfaces import IAuditService
from app.infrastructure.providers.audit_provider import get_audit_service
from app.models.audit import (
    AuditEventType,
    AuditRiskLevel,
    AuditLogQuery,
    AuditLogResponse,
    AuditLogStats,
)

router = APIRouter(prefix="/audit", tags=["audit"])


class AuditLogQueryRequest(BaseModel):
    """Request model for audit log queries."""
    
    user_id: Optional[str] = Field(None, description="Filter by user ID")
    event_types: Optional[List[AuditEventType]] = Field(None, description="Filter by event types")
    resource_type: Optional[str] = Field(None, description="Filter by resource type")
    resource_id: Optional[str] = Field(None, description="Filter by resource ID")
    risk_level: Optional[AuditRiskLevel] = Field(None, description="Filter by risk level")
    suspicious_only: bool = Field(False, description="Show only suspicious events")
    start_time: Optional[datetime] = Field(None, description="Start time filter (ISO format)")
    end_time: Optional[datetime] = Field(None, description="End time filter (ISO format)")
    correlation_id: Optional[str] = Field(None, description="Filter by correlation ID")
    batch_id: Optional[str] = Field(None, description="Filter by batch ID")
    ip_address: Optional[str] = Field(None, description="Filter by IP address")
    page: int = Field(1, ge=1, le=1000, description="Page number")
    size: int = Field(50, ge=1, le=1000, description="Page size")


class AuditLogQueryResponse(BaseModel):
    """Response model for audit log queries."""
    
    audit_logs: List[AuditLogResponse] = Field(..., description="List of audit log entries")
    pagination: Dict[str, Any] = Field(..., description="Pagination metadata")
    query_timestamp: str = Field(..., description="Query execution timestamp")
    total_events: int = Field(..., description="Total events matching filters")


class AuditStatsRequest(BaseModel):
    """Request model for audit statistics."""
    
    start_time: Optional[datetime] = Field(None, description="Start time for statistics")
    end_time: Optional[datetime] = Field(None, description="End time for statistics")


class AuditIntegrityRequest(BaseModel):
    """Request model for audit log integrity verification."""
    
    log_id: str = Field(..., description="Audit log entry ID to verify")


class AuditExportRequest(BaseModel):
    """Request model for audit log export."""
    
    format: str = Field("json", pattern="^(json|csv|xml)$", description="Export format")
    start_time: Optional[datetime] = Field(None, description="Start time for export")
    end_time: Optional[datetime] = Field(None, description="End time for export")
    event_types: Optional[List[AuditEventType]] = Field(None, description="Filter by event types")


async def get_audit_service_dep() -> IAuditService:
    """Dependency to get audit service."""
    return await get_audit_service()


def get_client_info(request: Request) -> Dict[str, str]:
    """Extract client information from request."""
    return {
        "ip_address": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", "unknown"),
    }


@router.get(
    "/logs",
    response_model=AuditLogQueryResponse,
    summary="Query audit logs with filtering and pagination",
    description="""
    Retrieve audit logs with comprehensive filtering options for compliance reporting.
    
    This endpoint supports:
    - Filtering by event types, risk levels, and time ranges
    - User and resource-specific queries
    - Suspicious activity detection
    - Pagination for large result sets
    - Correlation tracking for related events
    """,
    responses={
        200: {"description": "Audit logs retrieved successfully"},
        400: {"description": "Invalid query parameters"},
        403: {"description": "Access denied - insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def query_audit_logs(
    request: Request,
    current_user: CurrentUserDep,
    tenant_context: TenantContextDep,
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    event_types: Optional[List[AuditEventType]] = Query(None, description="Filter by event types"),
    resource_type: Optional[str] = Query(None, description="Filter by resource type"),
    resource_id: Optional[str] = Query(None, description="Filter by resource ID"),
    risk_level: Optional[AuditRiskLevel] = Query(None, description="Filter by risk level"),
    suspicious_only: bool = Query(False, description="Show only suspicious events"),
    start_time: Optional[datetime] = Query(None, description="Start time filter (ISO format)"),
    end_time: Optional[datetime] = Query(None, description="End time filter (ISO format)"),
    correlation_id: Optional[str] = Query(None, description="Filter by correlation ID"),
    batch_id: Optional[str] = Query(None, description="Filter by batch ID"),
    ip_address: Optional[str] = Query(None, description="Filter by IP address"),
    page: int = Query(1, ge=1, le=1000, description="Page number"),
    size: int = Query(50, ge=1, le=1000, description="Page size"),
    audit_service: IAuditService = Depends(get_audit_service_dep),
) -> AuditLogQueryResponse:
    """Query audit logs with filtering and pagination."""
    
    # Log the audit query request itself
    client_info = get_client_info(request)
    try:
        await audit_service.log_event(
            event_type=AuditEventType.DATA_EXPORT.value,
            tenant_id=current_user.tenant_id,
            action="audit_logs_queried",
            resource_type="audit_logs",
            user_id=current_user.user_id,
            user_email=current_user.email,
            details={
                "query_filters": {
                    "user_id": user_id,
                    "event_types": [et.value for et in event_types] if event_types else None,
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    "risk_level": risk_level.value if risk_level else None,
                    "suspicious_only": suspicious_only,
                    "start_time": start_time.isoformat() if start_time else None,
                    "end_time": end_time.isoformat() if end_time else None,
                    "correlation_id": correlation_id,
                    "batch_id": batch_id,
                    "ip_address": ip_address,
                    "page": page,
                    "size": size,
                }
            },
            ip_address=client_info["ip_address"],
            user_agent=client_info["user_agent"],
            risk_level="medium",  # Audit queries are medium risk
        )
    except Exception:
        # Don't fail the request if audit logging fails
        pass
    
    try:
        # Validate time range
        if start_time and end_time and start_time >= end_time:
            raise HTTPException(
                status_code=400,
                detail="start_time must be before end_time"
            )
        
        # Convert event types to strings if provided
        event_type_strings = None
        if event_types:
            event_type_strings = [et.value for et in event_types]
        
        # Query audit logs
        result = await audit_service.query_audit_logs(
            tenant_id=current_user.tenant_id,
            user_id=user_id,
            event_types=event_type_strings,
            resource_type=resource_type,
            resource_id=resource_id,
            risk_level=risk_level.value if risk_level else None,
            suspicious_only=suspicious_only,
            start_time=start_time,
            end_time=end_time,
            correlation_id=correlation_id,
            batch_id=batch_id,
            ip_address=ip_address,
            page=page,
            size=size,
        )
        
        return AuditLogQueryResponse(
            audit_logs=[AuditLogResponse(**log) for log in result["audit_logs"]],
            pagination=result["pagination"],
            query_timestamp=datetime.now(timezone.utc).isoformat(),
            total_events=result["pagination"]["total"],
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query audit logs: {str(e)}"
        )


@router.get(
    "/statistics",
    response_model=AuditLogStats,
    summary="Get audit log statistics for compliance reporting",
    description="""
    Retrieve aggregated statistics for audit logs within a specified time range.
    
    Provides insights including:
    - Total event counts by type and risk level
    - Suspicious activity summary
    - User activity patterns
    - IP address diversity
    - Recent activity trends
    """,
    responses={
        200: {"description": "Statistics retrieved successfully"},
        400: {"description": "Invalid time range parameters"},
        403: {"description": "Access denied - insufficient permissions"},
    },
)
async def get_audit_statistics(
    request: Request,
    current_user: CurrentUserDep,
    tenant_context: TenantContextDep,
    start_time: Optional[datetime] = Query(None, description="Start time for statistics"),
    end_time: Optional[datetime] = Query(None, description="End time for statistics"),
    audit_service: IAuditService = Depends(get_audit_service_dep),
) -> AuditLogStats:
    """Get audit log statistics for compliance reporting."""
    
    # Log the statistics request
    client_info = get_client_info(request)
    try:
        await audit_service.log_event(
            event_type=AuditEventType.DATA_EXPORT.value,
            tenant_id=current_user.tenant_id,
            action="audit_statistics_requested",
            resource_type="audit_statistics",
            user_id=current_user.user_id,
            user_email=current_user.email,
            details={
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None,
            },
            ip_address=client_info["ip_address"],
            user_agent=client_info["user_agent"],
            risk_level="low",
        )
    except Exception:
        pass
    
    try:
        # Validate time range
        if start_time and end_time and start_time >= end_time:
            raise HTTPException(
                status_code=400,
                detail="start_time must be before end_time"
            )
        
        # Get statistics
        stats = await audit_service.get_audit_statistics(
            tenant_id=current_user.tenant_id,
            start_time=start_time,
            end_time=end_time,
        )
        
        return AuditLogStats(**stats)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get audit statistics: {str(e)}"
        )


@router.get(
    "/integrity/{log_id}",
    summary="Verify audit log integrity",
    description="""
    Verify the integrity of a specific audit log entry using cryptographic hashing.
    
    This endpoint:
    - Recalculates the expected hash for the log entry
    - Compares it with the stored hash
    - Detects tampering or corruption
    - Provides detailed integrity verification results
    """,
    responses={
        200: {"description": "Integrity verification completed"},
        404: {"description": "Audit log entry not found"},
        403: {"description": "Access denied - insufficient permissions"},
    },
)
async def verify_log_integrity(
    log_id: str,
    request: Request,
    current_user: CurrentUserDep,
    tenant_context: TenantContextDep,
    audit_service: IAuditService = Depends(get_audit_service_dep),
) -> Dict[str, Any]:
    """Verify the integrity of a specific audit log entry."""
    
    # Log the integrity verification request
    client_info = get_client_info(request)
    try:
        await audit_service.log_event(
            event_type=AuditEventType.DATA_EXPORT.value,
            tenant_id=current_user.tenant_id,
            action="audit_integrity_verification",
            resource_type="audit_log",
            resource_id=log_id,
            user_id=current_user.user_id,
            user_email=current_user.email,
            details={"log_id": log_id},
            ip_address=client_info["ip_address"],
            user_agent=client_info["user_agent"],
            risk_level="medium",
        )
    except Exception:
        pass
    
    try:
        # Verify integrity
        result = await audit_service.verify_log_integrity(
            tenant_id=current_user.tenant_id,
            log_id=log_id,
        )
        
        if "error" in result:
            raise HTTPException(
                status_code=404,
                detail=result["error"]
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to verify log integrity: {str(e)}"
        )


@router.post(
    "/export",
    summary="Export audit logs for compliance",
    description="""
    Export audit logs in various formats for compliance reporting and external analysis.
    
    Supported formats:
    - JSON: Structured data for programmatic processing
    - CSV: Tabular format for spreadsheet analysis
    - XML: Standard format for regulatory compliance
    
    The export includes all audit log fields with proper formatting and metadata.
    """,
    responses={
        200: {"description": "Audit logs exported successfully"},
        400: {"description": "Invalid export parameters"},
        403: {"description": "Access denied - insufficient permissions"},
        429: {"description": "Export rate limit exceeded"},
    },
)
async def export_audit_logs(
    export_request: AuditExportRequest,
    request: Request,
    current_user: CurrentUserDep,
    tenant_context: TenantContextDep,
    audit_service: IAuditService = Depends(get_audit_service_dep),
) -> Response:
    """Export audit logs for compliance reporting."""
    
    # Log the export request
    client_info = get_client_info(request)
    try:
        await audit_service.log_event(
            event_type=AuditEventType.DATA_EXPORT.value,
            tenant_id=current_user.tenant_id,
            action="audit_logs_exported",
            resource_type="audit_logs",
            user_id=current_user.user_id,
            user_email=current_user.email,
            details={
                "export_format": export_request.format,
                "start_time": export_request.start_time.isoformat() if export_request.start_time else None,
                "end_time": export_request.end_time.isoformat() if export_request.end_time else None,
                "event_types": [et.value for et in export_request.event_types] if export_request.event_types else None,
            },
            ip_address=client_info["ip_address"],
            user_agent=client_info["user_agent"],
            risk_level="high",  # Export operations are high risk
        )
    except Exception:
        pass
    
    try:
        # Validate time range
        if (export_request.start_time and export_request.end_time and 
            export_request.start_time >= export_request.end_time):
            raise HTTPException(
                status_code=400,
                detail="start_time must be before end_time"
            )
        
        # Convert event types to strings if provided
        event_type_strings = None
        if export_request.event_types:
            event_type_strings = [et.value for et in export_request.event_types]
        
        # Export audit logs
        exported_data = await audit_service.export_audit_logs(
            tenant_id=current_user.tenant_id,
            format=export_request.format,
            start_time=export_request.start_time,
            end_time=export_request.end_time,
            event_types=event_type_strings,
        )
        
        # Determine content type and filename
        content_type_mapping = {
            "json": "application/json",
            "csv": "text/csv",
            "xml": "application/xml",
        }
        
        file_extension_mapping = {
            "json": "json",
            "csv": "csv", 
            "xml": "xml",
        }
        
        content_type = content_type_mapping.get(export_request.format, "application/octet-stream")
        file_extension = file_extension_mapping.get(export_request.format, "txt")
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"audit_logs_export_{timestamp}.{file_extension}"
        
        return Response(
            content=exported_data,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(exported_data)),
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export audit logs: {str(e)}"
        )


@router.get(
    "/events/types",
    summary="Get available audit event types",
    description="""
    Retrieve all available audit event types for filtering and documentation purposes.
    
    Returns a structured list of event types categorized by:
    - Authentication events
    - File upload and processing events
    - Security events
    - Administrative events
    - Data access events
    """,
    response_model=Dict[str, List[str]],
)
async def get_audit_event_types() -> Dict[str, List[str]]:
    """Get all available audit event types categorized."""
    
    categories = {
        "authentication": [
            AuditEventType.LOGIN_SUCCESS.value,
            AuditEventType.LOGIN_FAILED.value,
            AuditEventType.LOGOUT.value,
            AuditEventType.PASSWORD_CHANGED.value,
            AuditEventType.PASSWORD_RESET_REQUESTED.value,
            AuditEventType.PASSWORD_RESET_COMPLETED.value,
            AuditEventType.SESSION_EXPIRED.value,
            AuditEventType.ACCOUNT_LOCKED.value,
            AuditEventType.ACCOUNT_UNLOCKED.value,
        ],
        "authorization": [
            AuditEventType.ACCESS_GRANTED.value,
            AuditEventType.ACCESS_DENIED.value,
            AuditEventType.PERMISSION_ESCALATION.value,
        ],
        "file_operations": [
            AuditEventType.FILE_UPLOAD_STARTED.value,
            AuditEventType.FILE_UPLOAD_SUCCESS.value,
            AuditEventType.FILE_UPLOAD_FAILED.value,
            AuditEventType.FILE_VALIDATION_FAILED.value,
            AuditEventType.FILE_SECURITY_WARNING.value,
            AuditEventType.FILE_REJECTED_SECURITY.value,
            AuditEventType.DOCUMENT_PROCESSING_STARTED.value,
            AuditEventType.DOCUMENT_PROCESSING_SUCCESS.value,
            AuditEventType.DOCUMENT_PROCESSING_FAILED.value,
            AuditEventType.BATCH_UPLOAD_STARTED.value,
            AuditEventType.BATCH_UPLOAD_COMPLETED.value,
        ],
        "data_access": [
            AuditEventType.DATA_EXPORT.value,
            AuditEventType.DATA_IMPORT.value,
            AuditEventType.SEARCH_PERFORMED.value,
            AuditEventType.PROFILE_ACCESSED.value,
            AuditEventType.PROFILE_MODIFIED.value,
            AuditEventType.PROFILE_DELETED.value,
        ],
        "administrative": [
            AuditEventType.USER_CREATED.value,
            AuditEventType.USER_MODIFIED.value,
            AuditEventType.USER_DELETED.value,
            AuditEventType.TENANT_CREATED.value,
            AuditEventType.TENANT_MODIFIED.value,
            AuditEventType.SYSTEM_CONFIG_CHANGED.value,
            AuditEventType.API_KEY_CREATED.value,
            AuditEventType.API_KEY_USED.value,
            AuditEventType.API_KEY_REVOKED.value,
        ],
        "security": [
            AuditEventType.SUSPICIOUS_ACTIVITY.value,
            AuditEventType.RATE_LIMIT_EXCEEDED.value,
            AuditEventType.MALICIOUS_FILE_DETECTED.value,
            AuditEventType.UNAUTHORIZED_ACCESS_ATTEMPT.value,
            AuditEventType.SECURITY_SCAN_FAILED.value,
            AuditEventType.WEBHOOK_VALIDATION_FAILED.value,
        ],
    }
    
    return categories


@router.get(
    "/health",
    summary="Check audit service health",
    description="Verify that the audit logging service is operational and database connectivity is working.",
)
async def check_audit_health(
    audit_service: IAuditService = Depends(get_audit_service_dep),
) -> Dict[str, Any]:
    """Check audit service health status."""
    
    try:
        health_status = await audit_service.check_health()
        return health_status
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Audit service health check failed: {str(e)}"
        )


__all__ = ["router"]