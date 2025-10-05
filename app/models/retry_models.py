"""SQLModel models for retry mechanism and error recovery system."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import validator
from sqlmodel import Column, DateTime, Enum as SQLEnum, Field, JSON, Relationship, SQLModel, Text

from app.domain.retry import ErrorCategory, RetryResult, BackoffStrategy
from app.models.base import BaseModel, TenantModel


class RetryStateModel(TenantModel, table=True):
    """SQLModel for retry state persistence."""
    
    __tablename__ = "retry_states"
    
    # Primary identification (id inherited from TenantModel)
    operation_id: str = Field(index=True, description="Original operation identifier")
    operation_type: str = Field(index=True, description="Type of operation being retried")
    user_id: Optional[str] = Field(default=None, index=True, description="User identifier")
    
    # Policy configuration (stored as JSON)
    policy_config: Optional[str] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Retry policy configuration"
    )
    
    # State tracking
    current_attempt: int = Field(default=0, description="Current attempt number")
    total_attempts: int = Field(default=0, description="Total attempts made")
    status: str = Field(
        default="failed",
        description="Current retry status"
    )
    
    # Timing (created_at and updated_at inherited from TenantModel)
    next_attempt_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), index=True),
        description="When next attempt should be made"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True)),
        description="When retry completed (success or failure)"
    )
    
    # Error tracking
    first_error: Optional[str] = Field(
        default=None,
        sa_column=Column(Text),
        description="First error message"
    )
    last_error: Optional[str] = Field(
        default=None,
        sa_column=Column(Text),
        description="Last error message"
    )
    last_error_category: Optional[str] = Field(
        default=None,
        description="Category of last error"
    )
    
    # Context and metadata
    operation_context: Optional[str] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Operation context data"
    )
    metadata_: Optional[str] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Additional metadata"
    )
    
    # Relationships
    attempts: List["RetryAttemptModel"] = Relationship(
        back_populates="retry_state",
        cascade_delete=True
    )
    dead_letter_entry: Optional["DeadLetterModel"] = Relationship(
        back_populates="retry_state",
        cascade_delete=True
    )


class RetryAttemptModel(TenantModel, table=True):
    """SQLModel for individual retry attempts."""
    
    __tablename__ = "retry_attempts"
    
    # Primary identification (id inherited from TenantModel)
    retry_state_id: UUID = Field(foreign_key="retry_states.id", index=True)
    attempt_number: int = Field(description="Attempt sequence number")
    
    # Timing
    started_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True)),
        description="When attempt started"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True)),
        description="When attempt completed"
    )
    operation_duration_ms: Optional[int] = Field(
        default=None,
        description="Operation duration in milliseconds"
    )
    delay_before_attempt: Optional[float] = Field(
        default=None,
        description="Delay before this attempt in seconds"
    )
    
    # Result tracking
    success: bool = Field(default=False, description="Whether attempt succeeded")
    error_message: Optional[str] = Field(
        default=None,
        sa_column=Column(Text),
        description="Error message if failed"
    )
    error_category: Optional[str] = Field(
        default=None,
        description="Category of error"
    )
    
    # Context
    attempt_context: Optional[str] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Attempt-specific context"
    )
    
    # Relationships
    retry_state: RetryStateModel = Relationship(back_populates="attempts")


class DeadLetterModel(TenantModel, table=True):
    """SQLModel for dead letter queue entries."""
    
    __tablename__ = "dead_letter_queue"
    
    # Primary identification (id inherited from TenantModel)
    operation_id: str = Field(index=True, description="Original operation identifier")
    operation_type: str = Field(index=True, description="Type of operation")
    retry_state_id: Optional[UUID] = Field(
        default=None,
        foreign_key="retry_states.id",
        index=True,
        description="Associated retry state ID"
    )
    
    # Error information
    final_error: Optional[str] = Field(
        default=None,
        sa_column=Column(Text),
        description="Final error message"
    )
    error_category: Optional[str] = Field(
        default=None,
        description="Category of final error"
    )
    retry_attempts: int = Field(default=0, description="Number of retry attempts made")
    
    # Timing (created_at inherited from TenantModel)
    resolved_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True)),
        description="When resolved (if applicable)"
    )
    
    # Resolution tracking
    is_resolved: bool = Field(default=False, index=True, description="Whether entry is resolved")
    resolution_action: Optional[str] = Field(
        default=None,
        description="Action taken to resolve"
    )
    resolved_by: Optional[str] = Field(
        default=None,
        description="Admin user who resolved"
    )
    resolution_notes: Optional[str] = Field(
        default=None,
        sa_column=Column(Text),
        description="Notes about resolution"
    )
    
    # Requeue tracking
    requeued_count: int = Field(default=0, description="Number of times requeued")
    last_requeued_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True)),
        description="Last requeue timestamp"
    )
    
    # Context and metadata
    operation_context: Optional[str] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Original operation context"
    )
    metadata_: Optional[str] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Additional metadata"
    )
    
    # Relationships
    retry_state: Optional[RetryStateModel] = Relationship(back_populates="dead_letter_entry")


class RetryPolicyTemplate(TenantModel, table=True):
    """SQLModel for storing reusable retry policy templates."""
    
    __tablename__ = "retry_policy_templates"
    
    # Primary identification (id inherited from TenantModel)
    name: str = Field(index=True, description="Template name")
    operation_type: str = Field(index=True, description="Operation type this applies to")
    is_global: bool = Field(default=False, description="Global template (applies to all tenants)")
    
    # Policy configuration
    policy_config: Optional[str] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Retry policy configuration"
    )
    
    # Metadata
    description: Optional[str] = Field(
        default=None,
        description="Template description"
    )
    is_active: bool = Field(default=True, index=True, description="Whether template is active")
    is_default: bool = Field(default=False, index=True, description="Whether this is the default template")
    
    # Timing (created_at and updated_at inherited from TenantModel)
    created_by: Optional[str] = Field(default=None, description="User who created template")
    updated_by: Optional[str] = Field(default=None, description="User who last updated template")


# Pydantic models for API responses
class RetryStateResponse(BaseModel):
    """Response model for retry state information."""
    
    id: UUID
    operation_id: str
    operation_type: str
    tenant_id: str
    user_id: Optional[str]
    
    current_attempt: int
    total_attempts: int
    status: RetryResult
    
    created_at: datetime
    updated_at: datetime
    next_attempt_at: Optional[datetime]
    completed_at: Optional[datetime]
    
    first_error: Optional[str]
    last_error: Optional[str]
    last_error_category: Optional[ErrorCategory]
    
    policy_config: Dict[str, Any]
    operation_context: Dict[str, Any]
    metadata_: Dict[str, Any]
    
    @classmethod
    def from_model(cls, model: RetryStateModel) -> "RetryStateResponse":
        """Create response from SQLModel."""
        return cls(
            id=model.id,
            operation_id=model.operation_id,
            operation_type=model.operation_type,
            tenant_id=model.tenant_id,
            user_id=model.user_id,
            current_attempt=model.current_attempt,
            total_attempts=model.total_attempts,
            status=RetryResult(model.status),
            created_at=model.created_at,
            updated_at=model.updated_at,
            next_attempt_at=model.next_attempt_at,
            completed_at=model.completed_at,
            first_error=model.first_error,
            last_error=model.last_error,
            last_error_category=ErrorCategory(model.last_error_category) if model.last_error_category else None,
            policy_config=model.policy_config,
            operation_context=model.operation_context,
            metadata_=model.metadata_
        )


class RetryAttemptResponse(BaseModel):
    """Response model for retry attempt information."""
    
    id: UUID
    retry_state_id: UUID
    attempt_number: int
    
    started_at: datetime
    completed_at: Optional[datetime]
    operation_duration_ms: Optional[int]
    delay_before_attempt: Optional[float]
    
    success: bool
    error_message: Optional[str]
    error_category: Optional[ErrorCategory]
    
    attempt_context: Dict[str, Any]
    
    @classmethod
    def from_model(cls, model: RetryAttemptModel) -> "RetryAttemptResponse":
        """Create response from SQLModel."""
        return cls(
            id=model.id,
            retry_state_id=model.retry_state_id,
            attempt_number=model.attempt_number,
            started_at=model.started_at,
            completed_at=model.completed_at,
            operation_duration_ms=model.operation_duration_ms,
            delay_before_attempt=model.delay_before_attempt,
            success=model.success,
            error_message=model.error_message,
            error_category=ErrorCategory(model.error_category) if model.error_category else None,
            attempt_context=model.attempt_context
        )


class DeadLetterResponse(BaseModel):
    """Response model for dead letter queue entries."""
    
    id: UUID
    operation_id: str
    operation_type: str
    tenant_id: str
    retry_state_id: Optional[UUID]
    
    final_error: Optional[str]
    error_category: Optional[ErrorCategory]
    retry_attempts: int
    
    created_at: datetime
    resolved_at: Optional[datetime]
    
    is_resolved: bool
    resolution_action: Optional[str]
    resolved_by: Optional[str]
    resolution_notes: Optional[str]
    
    requeued_count: int
    last_requeued_at: Optional[datetime]
    
    operation_context: Dict[str, Any]
    metadata_: Dict[str, Any]
    
    @classmethod
    def from_model(cls, model: DeadLetterModel) -> "DeadLetterResponse":
        """Create response from SQLModel."""
        return cls(
            id=model.id,
            operation_id=model.operation_id,
            operation_type=model.operation_type,
            tenant_id=model.tenant_id,
            retry_state_id=model.retry_state_id,
            final_error=model.final_error,
            error_category=ErrorCategory(model.error_category) if model.error_category else None,
            retry_attempts=model.retry_attempts,
            created_at=model.created_at,
            resolved_at=model.resolved_at,
            is_resolved=model.is_resolved,
            resolution_action=model.resolution_action,
            resolved_by=model.resolved_by,
            resolution_notes=model.resolution_notes,
            requeued_count=model.requeued_count,
            last_requeued_at=model.last_requeued_at,
            operation_context=model.operation_context,
            metadata_=model.metadata_
        )


class RetryPolicyCreate(BaseModel):
    """Model for creating retry policy templates."""
    
    name: str = Field(..., min_length=1, max_length=100)
    operation_type: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = Field(None, max_length=500)
    
    # Policy configuration
    max_attempts: int = Field(default=3, ge=1, le=20)
    base_delay_seconds: float = Field(default=1.0, ge=0.1, le=3600)
    max_delay_seconds: float = Field(default=300.0, ge=1.0, le=7200)
    backoff_strategy: BackoffStrategy = Field(default=BackoffStrategy.EXPONENTIAL_JITTER)
    jitter_factor: float = Field(default=0.1, ge=0.0, le=1.0)
    
    retryable_errors: List[ErrorCategory] = Field(default_factory=list)
    non_retryable_errors: List[ErrorCategory] = Field(default_factory=list)
    
    rate_limit_retry_after: bool = Field(default=True)
    circuit_breaker_enabled: bool = Field(default=True)
    dead_letter_enabled: bool = Field(default=True)
    
    operation_timeout_seconds: Optional[float] = Field(None, ge=1.0, le=3600)
    total_timeout_seconds: Optional[float] = Field(None, ge=1.0, le=86400)
    
    is_default: bool = Field(default=False)
    
    @validator('max_delay_seconds')
    def validate_max_delay(cls, v, values):
        """Ensure max delay is greater than base delay."""
        if 'base_delay_seconds' in values and v < values['base_delay_seconds']:
            raise ValueError('max_delay_seconds must be greater than base_delay_seconds')
        return v


class RetryStatistics(BaseModel):
    """Statistics about retry operations."""
    
    total_retry_states: int = Field(default=0)
    active_retries: int = Field(default=0)
    completed_retries: int = Field(default=0)
    failed_retries: int = Field(default=0)
    cancelled_retries: int = Field(default=0)
    
    by_operation_type: Dict[str, int] = Field(default_factory=dict)
    by_error_category: Dict[str, int] = Field(default_factory=dict)
    by_status: Dict[str, int] = Field(default_factory=dict)
    
    average_attempts: float = Field(default=0.0)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    
    dead_letter_count: int = Field(default=0)
    resolved_dead_letters: int = Field(default=0)
    
    time_period: Dict[str, datetime] = Field(default_factory=dict)


class DeadLetterStatistics(BaseModel):
    """Statistics about dead letter queue."""
    
    total_entries: int = Field(default=0)
    unresolved_entries: int = Field(default=0)
    resolved_entries: int = Field(default=0)
    requeued_entries: int = Field(default=0)
    
    by_operation_type: Dict[str, int] = Field(default_factory=dict)
    by_error_category: Dict[str, int] = Field(default_factory=dict)
    by_tenant: Dict[str, int] = Field(default_factory=dict)
    
    oldest_unresolved: Optional[datetime] = Field(default=None)
    resolution_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    
    time_period: Dict[str, datetime] = Field(default_factory=dict)


__all__ = [
    "RetryStateModel",
    "RetryAttemptModel", 
    "DeadLetterModel",
    "RetryPolicyTemplate",
    "RetryStateResponse",
    "RetryAttemptResponse",
    "DeadLetterResponse",
    "RetryPolicyCreate",
    "RetryStatistics",
    "DeadLetterStatistics"
]