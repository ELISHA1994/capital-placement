"""
Webhook API Schemas - DTOs for webhook reliability system endpoints.

This module contains all request/response models for webhook API layer,
separated from domain entities and persistence tables following hexagonal architecture.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


class WebhookEventType(str, Enum):
    """Webhook event type enumeration."""
    # Document events
    DOCUMENT_UPLOADED = "document.uploaded"
    DOCUMENT_PROCESSED = "document.processed"
    DOCUMENT_PROCESSING_FAILED = "document.processing.failed"
    DOCUMENT_DELETED = "document.deleted"

    # Profile events
    PROFILE_CREATED = "profile.created"
    PROFILE_UPDATED = "profile.updated"
    PROFILE_DELETED = "profile.deleted"

    # Search events
    SEARCH_COMPLETED = "search.completed"
    SEARCH_FAILED = "search.failed"

    # Job events
    JOB_CREATED = "job.created"
    JOB_UPDATED = "job.updated"
    JOB_DELETED = "job.deleted"
    JOB_MATCH_FOUND = "job.match.found"

    # User events
    USER_REGISTERED = "user.registered"
    USER_ACTIVATED = "user.activated"
    USER_DEACTIVATED = "user.deactivated"

    # System events
    SYSTEM_ERROR = "system.error"
    QUOTA_EXCEEDED = "quota.exceeded"
    TENANT_CREATED = "tenant.created"
    TENANT_UPDATED = "tenant.updated"


class WebhookDeliveryStatus(str, Enum):
    """Webhook delivery status enumeration."""
    PENDING = "pending"
    SENDING = "sending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"
    CANCELLED = "cancelled"


class WebhookFailureReason(str, Enum):
    """Webhook failure reason enumeration."""
    # Network errors
    CONNECTION_ERROR = "connection_error"
    TIMEOUT = "timeout"
    DNS_ERROR = "dns_error"
    SSL_ERROR = "ssl_error"

    # HTTP errors
    HTTP_ERROR = "http_error"
    BAD_REQUEST = "bad_request"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    SERVER_ERROR = "server_error"

    # Circuit breaker
    CIRCUIT_OPEN = "circuit_open"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    ENDPOINT_DISABLED = "endpoint_disabled"

    # Payload errors
    PAYLOAD_TOO_LARGE = "payload_too_large"
    INVALID_PAYLOAD = "invalid_payload"

    # Configuration errors
    INVALID_URL = "invalid_url"
    SIGNATURE_VERIFICATION_FAILED = "signature_verification_failed"

    # Administrative
    CANCELLED_BY_ADMIN = "cancelled_by_admin"

    # Other
    RATE_LIMITED = "rate_limited"
    UNKNOWN_ERROR = "unknown_error"


class CircuitBreakerState(str, Enum):
    """Circuit breaker state enumeration."""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Circuit tripped, rejecting requests
    HALF_OPEN = "half_open"    # Testing if service recovered


class RetryPolicy(BaseModel):
    """Retry policy configuration for webhook delivery."""

    max_attempts: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of delivery attempts"
    )
    base_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Base delay between retries in seconds"
    )
    max_delay_seconds: float = Field(
        default=300.0,
        ge=1.0,
        le=3600.0,
        description="Maximum delay between retries in seconds"
    )
    backoff_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="Exponential backoff multiplier"
    )
    timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Request timeout in seconds"
    )
    retry_on_status_codes: List[int] = Field(
        default_factory=lambda: [408, 429, 500, 502, 503, 504],
        description="HTTP status codes that trigger retry"
    )


class WebhookEndpointCreate(BaseModel):
    """Request model for creating a webhook endpoint."""

    url: HttpUrl = Field(..., description="Webhook endpoint URL")
    name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=100,
        description="Endpoint name"
    )
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Endpoint description"
    )
    event_types: List[WebhookEventType] = Field(
        ...,
        min_length=1,
        description="Event types to subscribe to"
    )
    secret: Optional[str] = Field(
        None,
        min_length=16,
        max_length=255,
        description="Secret for signature generation"
    )
    enabled: bool = Field(
        default=True,
        description="Endpoint is enabled"
    )
    retry_policy: Optional[RetryPolicy] = Field(
        None,
        description="Custom retry policy"
    )
    timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Request timeout in seconds"
    )
    signature_header: str = Field(
        default="X-Webhook-Signature",
        max_length=100,
        description="Header name for signature"
    )
    signature_algorithm: str = Field(
        default="sha256",
        pattern="^(sha256|sha512)$",
        description="Signature algorithm"
    )


class WebhookEndpointUpdate(BaseModel):
    """Request model for updating a webhook endpoint."""

    url: Optional[HttpUrl] = Field(None, description="Webhook endpoint URL")
    name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=100,
        description="Endpoint name"
    )
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Endpoint description"
    )
    event_types: Optional[List[WebhookEventType]] = Field(
        None,
        description="Event types to subscribe to"
    )
    secret: Optional[str] = Field(
        None,
        min_length=16,
        max_length=255,
        description="Secret for signature generation"
    )
    enabled: Optional[bool] = Field(
        None,
        description="Endpoint is enabled"
    )
    retry_policy: Optional[RetryPolicy] = Field(
        None,
        description="Custom retry policy"
    )
    timeout_seconds: Optional[int] = Field(
        None,
        ge=5,
        le=300,
        description="Request timeout in seconds"
    )


class WebhookEndpointResponse(BaseModel):
    """Response model for webhook endpoint."""

    id: UUID = Field(..., description="Endpoint ID")
    url: str = Field(..., description="Webhook URL")
    name: Optional[str] = Field(None, description="Endpoint name")
    description: Optional[str] = Field(None, description="Endpoint description")
    event_types: List[WebhookEventType] = Field(..., description="Subscribed event types")
    enabled: bool = Field(..., description="Endpoint is enabled")

    # Circuit breaker state
    circuit_state: CircuitBreakerState = Field(..., description="Circuit breaker state")
    failure_count: int = Field(..., description="Consecutive failure count")
    last_failure_at: Optional[datetime] = Field(None, description="Last failure timestamp")

    # Statistics
    total_deliveries: int = Field(..., description="Total delivery attempts")
    successful_deliveries: int = Field(..., description="Successful deliveries")
    failed_deliveries: int = Field(..., description="Failed deliveries")
    avg_response_time_ms: Optional[float] = Field(None, description="Average response time")
    last_successful_delivery: Optional[datetime] = Field(
        None,
        description="Last successful delivery"
    )

    # Metadata
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    tenant_id: UUID = Field(..., description="Tenant ID")


class WebhookDeliveryResponse(BaseModel):
    """Response model for webhook delivery."""

    id: UUID = Field(..., description="Delivery ID")
    endpoint_id: UUID = Field(..., description="Endpoint ID")
    event_type: WebhookEventType = Field(..., description="Event type")
    event_id: str = Field(..., description="Event identifier")
    status: WebhookDeliveryStatus = Field(..., description="Delivery status")

    # Attempt tracking
    attempt_number: int = Field(..., description="Current attempt number")
    max_attempts: int = Field(..., description="Maximum attempts allowed")

    # Timing
    scheduled_at: datetime = Field(..., description="Scheduled delivery time")
    first_attempted_at: Optional[datetime] = Field(None, description="First attempt")
    last_attempted_at: Optional[datetime] = Field(None, description="Last attempt")
    delivered_at: Optional[datetime] = Field(None, description="Successful delivery timestamp")
    next_retry_at: Optional[datetime] = Field(None, description="Next retry timestamp")

    # Response details
    http_status_code: Optional[int] = Field(None, description="HTTP status code")
    response_time_ms: Optional[int] = Field(None, description="Response time")

    # Error details
    failure_reason: Optional[WebhookFailureReason] = Field(None, description="Failure reason")
    error_message: Optional[str] = Field(None, description="Error message")

    # Metadata
    correlation_id: Optional[str] = Field(None, description="Correlation ID")
    priority: int = Field(..., description="Delivery priority")
    created_at: datetime = Field(..., description="Creation timestamp")
    tenant_id: UUID = Field(..., description="Tenant ID")


class WebhookDeliveryQuery(BaseModel):
    """Query parameters for filtering webhook deliveries."""

    endpoint_id: Optional[UUID] = Field(None, description="Filter by endpoint")
    event_type: Optional[WebhookEventType] = Field(None, description="Filter by event type")
    status: Optional[WebhookDeliveryStatus] = Field(None, description="Filter by status")
    start_date: Optional[datetime] = Field(None, description="Filter by start date")
    end_date: Optional[datetime] = Field(None, description="Filter by end date")
    limit: int = Field(default=50, ge=1, le=500, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Result offset for pagination")


class WebhookRetryRequest(BaseModel):
    """Request model for manually retrying a webhook delivery."""

    delivery_id: UUID = Field(..., description="Delivery ID to retry")
    override_circuit_breaker: bool = Field(
        default=False,
        description="Override circuit breaker state"
    )
    notes: Optional[str] = Field(
        None,
        max_length=500,
        description="Admin notes for retry"
    )


class WebhookTestRequest(BaseModel):
    """Request model for testing a webhook endpoint."""

    endpoint_id: UUID = Field(..., description="Endpoint ID to test")
    event_type: WebhookEventType = Field(
        default=WebhookEventType.SYSTEM_ERROR,
        description="Event type for test"
    )
    test_payload: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom test payload"
    )


class WebhookDeadLetterResponse(BaseModel):
    """Response model for webhook dead letter entry."""

    id: UUID = Field(..., description="Dead letter ID")
    delivery_id: UUID = Field(..., description="Original delivery ID")
    endpoint_id: UUID = Field(..., description="Endpoint ID")
    event_type: WebhookEventType = Field(..., description="Event type")
    event_id: str = Field(..., description="Event identifier")

    # Failure summary
    total_attempts: int = Field(..., description="Total delivery attempts")
    final_failure_reason: WebhookFailureReason = Field(..., description="Final failure reason")
    final_error_message: Optional[str] = Field(None, description="Final error message")
    first_attempted_at: datetime = Field(..., description="First attempt")
    last_attempted_at: datetime = Field(..., description="Last attempt")

    # Dead letter metadata
    dead_lettered_at: datetime = Field(..., description="Dead letter timestamp")
    dead_lettered_by: str = Field(..., description="Who moved to dead letter")

    # Admin actions
    reviewed_at: Optional[datetime] = Field(None, description="Review timestamp")
    reviewed_by: Optional[str] = Field(None, description="Who reviewed")
    resolution_action: Optional[str] = Field(None, description="Resolution action")
    resolution_notes: Optional[str] = Field(None, description="Resolution notes")

    # Retry controls
    can_retry: bool = Field(..., description="Can be retried")
    retry_count: int = Field(..., description="Manual retry count")
    last_retry_at: Optional[datetime] = Field(None, description="Last retry timestamp")

    # Metadata
    created_at: datetime = Field(..., description="Creation timestamp")
    tenant_id: UUID = Field(..., description="Tenant ID")


class WebhookStatsResponse(BaseModel):
    """Response model for webhook statistics."""

    # Overall statistics
    total_endpoints: int = Field(..., description="Total webhook endpoints")
    active_endpoints: int = Field(..., description="Active endpoints")
    disabled_endpoints: int = Field(..., description="Disabled endpoints")

    # Delivery statistics
    total_deliveries: int = Field(..., description="Total deliveries")
    successful_deliveries: int = Field(..., description="Successful deliveries")
    failed_deliveries: int = Field(..., description="Failed deliveries")
    pending_deliveries: int = Field(..., description="Pending deliveries")
    dead_letter_count: int = Field(..., description="Dead letter entries")

    # Performance metrics
    avg_response_time_ms: Optional[float] = Field(None, description="Average response time")
    success_rate: float = Field(..., description="Success rate (0.0 to 1.0)")

    # Circuit breaker statistics
    circuits_open: int = Field(..., description="Circuits in open state")
    circuits_half_open: int = Field(..., description="Circuits in half-open state")
    circuits_closed: int = Field(..., description="Circuits in closed state")

    # Time period
    period_start: Optional[datetime] = Field(None, description="Statistics period start")
    period_end: Optional[datetime] = Field(None, description="Statistics period end")


class WebhookEventPayload(BaseModel):
    """Base webhook event payload structure."""

    event_id: str = Field(..., description="Unique event identifier")
    event_type: WebhookEventType = Field(..., description="Event type")
    timestamp: datetime = Field(..., description="Event timestamp")
    tenant_id: UUID = Field(..., description="Tenant ID")
    data: Dict[str, Any] = Field(..., description="Event data")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


__all__ = [
    # Enums
    "WebhookEventType",
    "WebhookDeliveryStatus",
    "WebhookFailureReason",
    "CircuitBreakerState",

    # Request/Response models
    "RetryPolicy",
    "WebhookEndpointCreate",
    "WebhookEndpointUpdate",
    "WebhookEndpointResponse",
    "WebhookDeliveryResponse",
    "WebhookDeliveryQuery",
    "WebhookRetryRequest",
    "WebhookTestRequest",
    "WebhookDeadLetterResponse",
    "WebhookStatsResponse",
    "WebhookEventPayload",
]