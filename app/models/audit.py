"""
Audit logging SQLModel models for security compliance and tamper-resistant audit trails.

This module provides comprehensive audit logging capabilities following the existing
hexagonal architecture patterns while ensuring data integrity and compliance requirements.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from enum import Enum

from sqlalchemy import Boolean, Column, String, DateTime, Integer, Text, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID, JSONB, INET
from sqlalchemy.sql import func
from sqlmodel import Field, SQLModel, Relationship

from .base import BaseModel, TenantModel, create_tenant_id_column


class AuditEventType(str, Enum):
    """Audit event types for categorization and filtering."""
    
    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    PASSWORD_CHANGED = "password_changed"
    PASSWORD_RESET_REQUESTED = "password_reset_requested"
    PASSWORD_RESET_COMPLETED = "password_reset_completed"
    SESSION_EXPIRED = "session_expired"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_UNLOCKED = "account_unlocked"
    
    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_ESCALATION = "permission_escalation"
    
    # File upload and processing events
    FILE_UPLOAD_STARTED = "file_upload_started"
    FILE_UPLOAD_SUCCESS = "file_upload_success"
    FILE_UPLOAD_FAILED = "file_upload_failed"
    FILE_VALIDATION_FAILED = "file_validation_failed"
    FILE_SECURITY_WARNING = "file_security_warning"
    FILE_REJECTED_SECURITY = "file_rejected_security"
    DOCUMENT_PROCESSING_STARTED = "document_processing_started"
    DOCUMENT_PROCESSING_SUCCESS = "document_processing_success"
    DOCUMENT_PROCESSING_FAILED = "document_processing_failed"
    PROCESSING_CANCELLED = "processing_cancelled"
    BATCH_UPLOAD_STARTED = "batch_upload_started"
    BATCH_UPLOAD_COMPLETED = "batch_upload_completed"
    
    # Data access events
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    SEARCH_PERFORMED = "search_performed"
    PROFILE_ACCESSED = "profile_accessed"
    PROFILE_MODIFIED = "profile_modified"
    PROFILE_DELETED = "profile_deleted"
    
    # Administrative events
    USER_CREATED = "user_created"
    USER_MODIFIED = "user_modified"
    USER_DELETED = "user_deleted"
    TENANT_CREATED = "tenant_created"
    TENANT_MODIFIED = "tenant_modified"
    SYSTEM_CONFIG_CHANGED = "system_config_changed"
    API_KEY_CREATED = "api_key_created"
    API_KEY_USED = "api_key_used"
    API_KEY_REVOKED = "api_key_revoked"
    
    # Security events
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    MALICIOUS_FILE_DETECTED = "malicious_file_detected"
    UNAUTHORIZED_ACCESS_ATTEMPT = "unauthorized_access_attempt"
    SECURITY_SCAN_FAILED = "security_scan_failed"
    WEBHOOK_VALIDATION_FAILED = "webhook_validation_failed"


class AuditRiskLevel(str, Enum):
    """Risk levels for audit events."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditLogTable(TenantModel, table=True):
    """
    Tamper-resistant audit log table for security compliance.
    
    This table stores comprehensive audit logs for all system activities,
    designed for compliance with security standards and tamper resistance.
    """
    __tablename__ = "audit_logs"
    
    # Tenant isolation field (required for SQLModel to create foreign key constraints)
    tenant_id: UUID = Field(
        sa_column=create_tenant_id_column(),
        description="Tenant identifier for multi-tenant isolation"
    )
    
    # Relationships
    tenant: Optional["TenantTable"] = Relationship(back_populates="audit_logs")
    
    # Core audit fields
    event_type: AuditEventType = Field(
        sa_column=Column(String(100), nullable=False, index=True),
        description="Type of audit event"
    )
    
    # User identification (optional for system events)
    user_id: Optional[UUID] = Field(
        default=None,
        sa_column=Column(PostgreSQLUUID(as_uuid=True), nullable=True, index=True),
        description="User who performed the action (null for system events)"
    )
    
    user_email: Optional[str] = Field(
        default=None,
        sa_column=Column(String(255), nullable=True),
        description="Email of user who performed the action"
    )
    
    # Session and authentication context
    session_id: Optional[str] = Field(
        default=None,
        sa_column=Column(String(255), nullable=True, index=True),
        description="Session identifier"
    )
    
    api_key_id: Optional[UUID] = Field(
        default=None,
        sa_column=Column(PostgreSQLUUID(as_uuid=True), nullable=True),
        description="API key used for the action (if applicable)"
    )
    
    # Resource identification
    resource_type: str = Field(
        sa_column=Column(String(100), nullable=False, index=True),
        description="Type of resource affected (user, file, profile, etc.)"
    )
    
    resource_id: Optional[str] = Field(
        default=None,
        sa_column=Column(String(255), nullable=True, index=True),
        description="Identifier of the affected resource"
    )
    
    # Action details
    action: str = Field(
        sa_column=Column(String(255), nullable=False),
        description="Specific action performed"
    )
    
    details: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default={}),
        description="Detailed information about the action"
    )
    
    # Network and client information
    ip_address: str = Field(
        sa_column=Column(INET, nullable=False),
        description="Client IP address"
    )
    
    user_agent: str = Field(
        sa_column=Column(Text, nullable=False),
        description="Client user agent string"
    )
    
    # Risk assessment
    risk_level: AuditRiskLevel = Field(
        default=AuditRiskLevel.LOW,
        sa_column=Column(String(20), nullable=False, index=True),
        description="Risk level of the event"
    )
    
    suspicious: bool = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False, index=True),
        description="Whether the event is flagged as suspicious"
    )
    
    # Compliance and integrity fields
    event_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            DateTime(timezone=True), 
            nullable=False, 
            default=func.now(),
            index=True
        ),
        description="Exact timestamp when the event occurred"
    )
    
    logged_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            DateTime(timezone=True), 
            nullable=False, 
            default=func.now(),
            index=True
        ),
        description="Timestamp when the log entry was created"
    )
    
    # Immutability and tamper resistance
    log_hash: Optional[str] = Field(
        default=None,
        sa_column=Column(String(64), nullable=True),
        description="SHA-256 hash of log entry for tamper detection"
    )
    
    sequence_number: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, nullable=True),
        description="Sequence number for ordering and gap detection"
    )
    
    # Additional context
    correlation_id: Optional[str] = Field(
        default=None,
        sa_column=Column(String(255), nullable=True, index=True),
        description="Correlation ID for tracing related events"
    )
    
    batch_id: Optional[str] = Field(
        default=None,
        sa_column=Column(String(255), nullable=True, index=True),
        description="Batch identifier for grouped operations"
    )
    
    # Error information (for failed operations)
    error_code: Optional[str] = Field(
        default=None,
        sa_column=Column(String(100), nullable=True),
        description="Error code if the operation failed"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
        description="Error message if the operation failed"
    )
    
    # Database indexes for performance and compliance queries
    __table_args__ = (
        Index("ix_audit_logs_tenant_event_time", "tenant_id", "event_type", "event_timestamp"),
        Index("ix_audit_logs_user_time", "user_id", "event_timestamp"),
        Index("ix_audit_logs_resource", "resource_type", "resource_id"),
        Index("ix_audit_logs_risk_time", "risk_level", "event_timestamp"),
        Index("ix_audit_logs_suspicious", "suspicious", "event_timestamp"),
        Index("ix_audit_logs_correlation", "correlation_id"),
        Index("ix_audit_logs_sequence", "tenant_id", "sequence_number"),
    )
    
    def __repr__(self):
        return (
            f"AuditLogTable(id={self.id}, tenant_id={self.tenant_id}, "
            f"event_type={self.event_type}, user_id={self.user_id}, "
            f"resource_type={self.resource_type}, action={self.action})"
        )


# Request/Response models for API operations
class AuditLogCreate(BaseModel):
    """Model for creating audit log entries."""
    
    event_type: AuditEventType = Field(..., description="Type of audit event")
    user_id: Optional[str] = Field(None, description="User ID (if applicable)")
    user_email: Optional[str] = Field(None, description="User email (if applicable)")
    session_id: Optional[str] = Field(None, description="Session ID")
    api_key_id: Optional[str] = Field(None, description="API key ID")
    resource_type: str = Field(..., description="Type of resource affected")
    resource_id: Optional[str] = Field(None, description="Resource identifier")
    action: str = Field(..., description="Action performed")
    details: Dict[str, Any] = Field(default_factory=dict, description="Action details")
    ip_address: str = Field(..., description="Client IP address")
    user_agent: str = Field(..., description="Client user agent")
    risk_level: AuditRiskLevel = Field(default=AuditRiskLevel.LOW, description="Risk level")
    suspicious: bool = Field(default=False, description="Suspicious activity flag")
    correlation_id: Optional[str] = Field(None, description="Correlation ID")
    batch_id: Optional[str] = Field(None, description="Batch ID")
    error_code: Optional[str] = Field(None, description="Error code")
    error_message: Optional[str] = Field(None, description="Error message")


class AuditLogResponse(BaseModel):
    """Model for audit log API responses."""
    
    id: str = Field(..., description="Audit log ID")
    tenant_id: str = Field(..., description="Tenant ID")
    event_type: str = Field(..., description="Event type")
    user_id: Optional[str] = Field(None, description="User ID")
    user_email: Optional[str] = Field(None, description="User email")
    resource_type: str = Field(..., description="Resource type")
    resource_id: Optional[str] = Field(None, description="Resource ID")
    action: str = Field(..., description="Action")
    details: Dict[str, Any] = Field(..., description="Details")
    ip_address: str = Field(..., description="IP address")
    risk_level: str = Field(..., description="Risk level")
    suspicious: bool = Field(..., description="Suspicious flag")
    event_timestamp: str = Field(..., description="Event timestamp")
    correlation_id: Optional[str] = Field(None, description="Correlation ID")
    
    @classmethod
    def from_audit_log_table(cls, audit_log: AuditLogTable) -> "AuditLogResponse":
        """Create AuditLogResponse from AuditLogTable instance."""
        return cls(
            id=str(audit_log.id),
            tenant_id=str(audit_log.tenant_id),
            event_type=audit_log.event_type.value,
            user_id=str(audit_log.user_id) if audit_log.user_id else None,
            user_email=audit_log.user_email,
            resource_type=audit_log.resource_type,
            resource_id=audit_log.resource_id,
            action=audit_log.action,
            details=audit_log.details,
            ip_address=str(audit_log.ip_address),
            risk_level=audit_log.risk_level.value,
            suspicious=audit_log.suspicious,
            event_timestamp=audit_log.event_timestamp.isoformat(),
            correlation_id=audit_log.correlation_id
        )


class AuditLogQuery(BaseModel):
    """Model for querying audit logs."""
    
    tenant_id: Optional[str] = Field(None, description="Filter by tenant ID")
    user_id: Optional[str] = Field(None, description="Filter by user ID")
    event_types: Optional[List[AuditEventType]] = Field(None, description="Filter by event types")
    resource_type: Optional[str] = Field(None, description="Filter by resource type")
    resource_id: Optional[str] = Field(None, description="Filter by resource ID")
    risk_level: Optional[AuditRiskLevel] = Field(None, description="Filter by risk level")
    suspicious_only: bool = Field(False, description="Show only suspicious events")
    start_time: Optional[datetime] = Field(None, description="Start time filter")
    end_time: Optional[datetime] = Field(None, description="End time filter")
    correlation_id: Optional[str] = Field(None, description="Filter by correlation ID")
    batch_id: Optional[str] = Field(None, description="Filter by batch ID")
    ip_address: Optional[str] = Field(None, description="Filter by IP address")
    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(50, ge=1, le=1000, description="Page size")


class AuditLogStats(BaseModel):
    """Model for audit log statistics."""
    
    total_events: int = Field(..., description="Total number of events")
    events_by_type: Dict[str, int] = Field(..., description="Events grouped by type")
    events_by_risk_level: Dict[str, int] = Field(..., description="Events grouped by risk level")
    suspicious_events: int = Field(..., description="Number of suspicious events")
    recent_events: int = Field(..., description="Recent events (last 24 hours)")
    unique_users: int = Field(..., description="Number of unique users")
    unique_ip_addresses: int = Field(..., description="Number of unique IP addresses")


# Backward compatibility
AuditLog = AuditLogTable  # For backward compatibility during migration


__all__ = [
    "AuditEventType",
    "AuditRiskLevel", 
    "AuditLogTable",
    "AuditLogCreate",
    "AuditLogResponse",
    "AuditLogQuery",
    "AuditLogStats",
    "AuditLog",  # Backward compatibility
]