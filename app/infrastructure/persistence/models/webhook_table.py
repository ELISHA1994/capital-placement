"""SQLModel table definitions for webhook reliability system."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlmodel import Column, DateTime, Field, JSON, Relationship, SQLModel, String, Text, ARRAY, Integer, Boolean

from app.infrastructure.persistence.models.base import TenantModel, create_tenant_id_column


class WebhookEndpointTable(TenantModel, table=True):
    """SQLModel for webhook endpoint configuration with circuit breaker state."""

    __tablename__ = "webhook_endpoints"

    # Tenant isolation field (required for SQLModel to create foreign key constraints)
    tenant_id: UUID = Field(
        sa_column=create_tenant_id_column(),
        description="Tenant identifier for multi-tenant isolation"
    )

    # Primary identification
    url: str = Field(..., max_length=2048, description="Webhook URL")
    name: Optional[str] = Field(None, max_length=100, description="Endpoint name")
    description: Optional[str] = Field(None, max_length=500, description="Endpoint description")

    # Security
    secret: Optional[str] = Field(None, max_length=255, description="Secret for signature generation")
    signature_header: str = Field(default="X-Webhook-Signature", max_length=100, description="Signature header name")
    signature_algorithm: str = Field(default="sha256", max_length=50, description="Signature algorithm")

    # Configuration
    enabled: bool = Field(default=True, description="Endpoint is enabled")
    event_types: Optional[str] = Field(
        default=None,
        sa_column=Column(ARRAY(String(50))),
        description="Subscribed event types"
    )
    retry_policy: Optional[str] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Retry policy configuration"
    )
    timeout_seconds: int = Field(default=30, description="Request timeout")

    # Circuit breaker state
    circuit_state: str = Field(default="closed", max_length=20, description="Circuit breaker state")
    failure_count: int = Field(default=0, description="Consecutive failure count")
    last_failure_at: Optional[datetime] = Field(
        None,
        sa_column=Column(DateTime(timezone=True)),
        description="Last failure timestamp"
    )
    circuit_opened_at: Optional[datetime] = Field(
        None,
        sa_column=Column(DateTime(timezone=True)),
        description="Circuit opened timestamp"
    )
    half_open_attempts: int = Field(default=0, description="Half-open state attempt count")

    # Statistics
    total_deliveries: int = Field(default=0, description="Total delivery attempts")
    successful_deliveries: int = Field(default=0, description="Successful deliveries")
    failed_deliveries: int = Field(default=0, description="Failed deliveries")
    avg_response_time_ms: Optional[float] = Field(None, description="Average response time")
    last_successful_delivery: Optional[datetime] = Field(
        None,
        sa_column=Column(DateTime(timezone=True)),
        description="Last successful delivery"
    )

    # Timestamps (created_at and updated_at inherited from TenantModel)

    # Relationships
    deliveries: List["WebhookDeliveryTable"] = Relationship(
        back_populates="endpoint",
        cascade_delete=True
    )
    dead_letters: List["WebhookDeadLetterTable"] = Relationship(
        back_populates="endpoint",
        cascade_delete=True
    )


class WebhookDeliveryTable(TenantModel, table=True):
    """SQLModel for webhook delivery attempt record."""

    __tablename__ = "webhook_deliveries"

    # Tenant isolation field (required for SQLModel to create foreign key constraints)
    tenant_id: UUID = Field(
        sa_column=create_tenant_id_column(),
        description="Tenant identifier for multi-tenant isolation"
    )

    # Primary identification
    endpoint_id: UUID = Field(..., foreign_key="webhook_endpoints.id", index=True, description="Webhook endpoint ID")

    # Event details
    event_type: str = Field(..., max_length=50, sa_column=Column(String(50), index=True), description="Event type")
    event_id: str = Field(..., max_length=255, description="Unique event identifier")
    payload: Optional[str] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Webhook payload"
    )

    # Delivery tracking
    status: str = Field(default="pending", max_length=20, sa_column=Column(String(20), index=True), description="Delivery status")
    attempt_number: int = Field(default=1, description="Current attempt number")
    max_attempts: int = Field(default=5, description="Maximum attempts allowed")

    # Timing
    scheduled_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), index=True),
        description="Scheduled delivery time"
    )
    first_attempted_at: Optional[datetime] = Field(
        None,
        sa_column=Column(DateTime(timezone=True)),
        description="First attempt timestamp"
    )
    last_attempted_at: Optional[datetime] = Field(
        None,
        sa_column=Column(DateTime(timezone=True)),
        description="Last attempt timestamp"
    )
    delivered_at: Optional[datetime] = Field(
        None,
        sa_column=Column(DateTime(timezone=True)),
        description="Successful delivery timestamp"
    )
    next_retry_at: Optional[datetime] = Field(
        None,
        sa_column=Column(DateTime(timezone=True), index=True),
        description="Next retry timestamp"
    )

    # Response details
    http_status_code: Optional[int] = Field(None, description="HTTP response status code")
    response_body: Optional[str] = Field(
        None,
        sa_column=Column(Text),
        description="Response body (truncated)"
    )
    response_headers: Optional[str] = Field(
        None,
        sa_column=Column(JSON),
        description="Response headers"
    )
    response_time_ms: Optional[int] = Field(None, description="Response time in milliseconds")

    # Error tracking
    failure_reason: Optional[str] = Field(None, max_length=50, description="Failure reason")
    error_message: Optional[str] = Field(None, max_length=1000, description="Error message")
    error_details: Optional[str] = Field(
        None,
        sa_column=Column(JSON),
        description="Additional error details"
    )

    # Security
    signature_generated: Optional[str] = Field(None, max_length=255, description="Generated signature")
    signature_verified: Optional[bool] = Field(None, description="Signature verification result")

    # Metadata
    user_agent: str = Field(
        default="CapitalPlacement-Webhook/1.0",
        max_length=255,
        description="User agent for requests"
    )
    request_headers: Optional[str] = Field(
        None,
        sa_column=Column(JSON),
        description="Request headers sent"
    )
    correlation_id: Optional[str] = Field(
        None,
        max_length=255,
        sa_column=Column(String(255), index=True),
        description="Correlation ID for tracking"
    )
    priority: int = Field(default=0, sa_column=Column(Integer, index=True), description="Delivery priority")

    # Manual operations
    manual_retry: bool = Field(default=False, description="Manual retry flag")
    retry_admin_user_id: Optional[str] = Field(None, max_length=255, description="Admin user ID for retry")
    retry_notes: Optional[str] = Field(None, max_length=500, description="Retry notes")
    max_attempts_overridden: bool = Field(default=False, description="Max attempts overridden flag")

    # Dead letter tracking
    dead_letter_id: Optional[UUID] = Field(None, description="Associated dead letter ID")
    dead_lettered_at: Optional[datetime] = Field(
        None,
        sa_column=Column(DateTime(timezone=True)),
        description="Dead letter timestamp"
    )

    # Timestamps (created_at and updated_at inherited from TenantModel)

    # Relationships
    endpoint: WebhookEndpointTable = Relationship(back_populates="deliveries")
    dead_letter: Optional["WebhookDeadLetterTable"] = Relationship(
        back_populates="delivery",
        cascade_delete=True
    )


class WebhookDeadLetterTable(TenantModel, table=True):
    """SQLModel for dead letter queue for failed webhook deliveries."""

    __tablename__ = "webhook_dead_letters"

    # Tenant isolation field (required for SQLModel to create foreign key constraints)
    tenant_id: UUID = Field(
        sa_column=create_tenant_id_column(),
        description="Tenant identifier for multi-tenant isolation"
    )

    # Primary identification
    delivery_id: UUID = Field(..., foreign_key="webhook_deliveries.id", index=True, description="Original delivery ID")
    endpoint_id: UUID = Field(..., foreign_key="webhook_endpoints.id", index=True, description="Webhook endpoint ID")

    # Original delivery details
    event_type: str = Field(..., max_length=50, sa_column=Column(String(50), index=True), description="Event type")
    event_id: str = Field(..., max_length=255, description="Event identifier")
    payload: Optional[str] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Original payload"
    )

    # Failure summary
    total_attempts: int = Field(..., description="Total delivery attempts made")
    final_failure_reason: str = Field(..., max_length=50, description="Final failure reason")
    final_error_message: Optional[str] = Field(None, max_length=1000, description="Final error message")
    first_attempted_at: datetime = Field(
        ...,
        sa_column=Column(DateTime(timezone=True)),
        description="First delivery attempt"
    )
    last_attempted_at: datetime = Field(
        ...,
        sa_column=Column(DateTime(timezone=True)),
        description="Last delivery attempt"
    )

    # Dead letter metadata
    dead_lettered_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), index=True),
        description="Dead letter timestamp"
    )
    dead_lettered_by: str = Field(default="system", max_length=255, description="Who moved to dead letter")

    # Admin actions
    reviewed_at: Optional[datetime] = Field(
        None,
        sa_column=Column(DateTime(timezone=True), index=True),
        description="Admin review timestamp"
    )
    reviewed_by: Optional[str] = Field(None, max_length=255, description="Admin who reviewed")
    resolution_action: Optional[str] = Field(None, max_length=255, description="Resolution action taken")
    resolution_notes: Optional[str] = Field(
        None,
        max_length=1000,
        description="Resolution notes"
    )

    # Retry controls
    can_retry: bool = Field(default=True, sa_column=Column(Boolean, index=True), description="Can be retried manually")
    retry_count: int = Field(default=0, description="Manual retry attempts")
    last_retry_at: Optional[datetime] = Field(
        None,
        sa_column=Column(DateTime(timezone=True)),
        description="Last manual retry"
    )
    last_retry_delivery_id: Optional[UUID] = Field(None, description="Last retry delivery ID")
    last_retry_admin_user_id: Optional[str] = Field(None, max_length=255, description="Last retry admin user ID")
    last_retry_notes: Optional[str] = Field(None, max_length=500, description="Last retry notes")

    # Timestamps (created_at and updated_at inherited from TenantModel)

    # Relationships
    delivery: WebhookDeliveryTable = Relationship(back_populates="dead_letter")
    endpoint: WebhookEndpointTable = Relationship(back_populates="dead_letters")


__all__ = [
    "WebhookEndpointTable",
    "WebhookDeliveryTable",
    "WebhookDeadLetterTable"
]