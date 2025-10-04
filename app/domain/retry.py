"""Domain interfaces and models for retry mechanisms and error recovery."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union, Callable
import asyncio


class ErrorCategory(str, Enum):
    """Categories of errors for retry classification."""
    
    # Infrastructure errors (usually retryable)
    NETWORK_ERROR = "network_error"
    CONNECTION_ERROR = "connection_error"
    TIMEOUT_ERROR = "timeout_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    RATE_LIMITED = "rate_limited"
    TEMPORARY_FAILURE = "temporary_failure"
    
    # Application errors (may be retryable)
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    RESOURCE_NOT_FOUND = "resource_not_found"
    CONFLICT_ERROR = "conflict_error"
    
    # System errors (usually not retryable)
    CONFIGURATION_ERROR = "configuration_error"
    PERMISSION_ERROR = "permission_error"
    QUOTA_EXCEEDED = "quota_exceeded"
    MALFORMED_REQUEST = "malformed_request"
    UNSUPPORTED_OPERATION = "unsupported_operation"
    
    # Business logic errors (usually not retryable)
    DATA_INTEGRITY_ERROR = "data_integrity_error"
    BUSINESS_RULE_VIOLATION = "business_rule_violation"
    INVALID_STATE = "invalid_state"
    
    # Unknown/uncategorized
    UNKNOWN_ERROR = "unknown_error"


class RetryResult(str, Enum):
    """Possible results of a retry operation."""
    
    SUCCESS = "success"
    FAILED = "failed"
    MAX_ATTEMPTS_EXCEEDED = "max_attempts_exceeded"
    NON_RETRYABLE_ERROR = "non_retryable_error"
    CANCELLED = "cancelled"
    DEAD_LETTERED = "dead_lettered"


class BackoffStrategy(str, Enum):
    """Backoff strategies for retry delays."""
    
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"
    FIBONACCI = "fibonacci"
    CUSTOM = "custom"


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    
    # Basic retry settings
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 300.0  # 5 minutes
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER
    jitter_factor: float = 0.1
    
    # Error classification
    retryable_errors: List[ErrorCategory] = field(default_factory=lambda: [
        ErrorCategory.NETWORK_ERROR,
        ErrorCategory.CONNECTION_ERROR,
        ErrorCategory.TIMEOUT_ERROR,
        ErrorCategory.SERVICE_UNAVAILABLE,
        ErrorCategory.RATE_LIMITED,
        ErrorCategory.TEMPORARY_FAILURE,
    ])
    
    non_retryable_errors: List[ErrorCategory] = field(default_factory=lambda: [
        ErrorCategory.CONFIGURATION_ERROR,
        ErrorCategory.PERMISSION_ERROR,
        ErrorCategory.MALFORMED_REQUEST,
        ErrorCategory.DATA_INTEGRITY_ERROR,
        ErrorCategory.BUSINESS_RULE_VIOLATION,
        ErrorCategory.INVALID_STATE,
    ])
    
    # Special handling
    rate_limit_retry_after: bool = True  # Respect Retry-After headers
    circuit_breaker_enabled: bool = True
    dead_letter_enabled: bool = True
    
    # Timeout settings
    operation_timeout_seconds: Optional[float] = None
    total_timeout_seconds: Optional[float] = None
    
    # Custom behaviors
    custom_error_classifier: Optional[str] = None  # Function name for custom classification
    custom_backoff_calculator: Optional[str] = None  # Function name for custom backoff
    
    def is_retryable(self, error_category: ErrorCategory) -> bool:
        """Check if an error category is retryable according to this policy."""
        if error_category in self.non_retryable_errors:
            return False
        return error_category in self.retryable_errors
    
    def calculate_delay(self, attempt: int, last_error: Optional[Exception] = None) -> float:
        """Calculate the delay before the next retry attempt."""
        if self.backoff_strategy == BackoffStrategy.FIXED:
            delay = self.base_delay_seconds
        elif self.backoff_strategy == BackoffStrategy.LINEAR:
            delay = self.base_delay_seconds * attempt
        elif self.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.base_delay_seconds * (2 ** (attempt - 1))
        elif self.backoff_strategy == BackoffStrategy.EXPONENTIAL_JITTER:
            base_delay = self.base_delay_seconds * (2 ** (attempt - 1))
            jitter = base_delay * self.jitter_factor * (2 * hash(str(datetime.now())) % 1000 / 1000 - 1)
            delay = base_delay + jitter
        elif self.backoff_strategy == BackoffStrategy.FIBONACCI:
            if attempt <= 2:
                delay = self.base_delay_seconds
            else:
                # Simple fibonacci approximation
                fib = ((1.618 ** attempt) / (5 ** 0.5))
                delay = self.base_delay_seconds * fib
        else:
            delay = self.base_delay_seconds
        
        return min(delay, self.max_delay_seconds)


@dataclass
class RetryAttempt:
    """Details about a single retry attempt."""
    
    attempt_number: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    error_category: Optional[ErrorCategory] = None
    delay_before_attempt: Optional[float] = None
    operation_duration_ms: Optional[int] = None
    success: bool = False
    
    @property
    def duration_ms(self) -> Optional[int]:
        """Get attempt duration in milliseconds."""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None


@dataclass
class RetryState:
    """Current state of a retry operation."""
    
    # Identification
    retry_id: str
    operation_id: str  # upload_id, task_id, etc.
    operation_type: str  # document_processing, webhook_delivery, etc.
    tenant_id: str
    user_id: Optional[str] = None
    
    # Configuration
    policy: RetryPolicy = field(default_factory=RetryPolicy)
    
    # State tracking
    current_attempt: int = 0
    total_attempts: int = 0
    status: RetryResult = RetryResult.FAILED
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    next_attempt_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Attempt history
    attempts: List[RetryAttempt] = field(default_factory=list)
    
    # Error tracking
    first_error: Optional[str] = None
    last_error: Optional[str] = None
    last_error_category: Optional[ErrorCategory] = None
    
    # Context
    operation_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_attempt(self, attempt: RetryAttempt) -> None:
        """Add a retry attempt to the history."""
        self.attempts.append(attempt)
        self.total_attempts = len(self.attempts)
        self.current_attempt = attempt.attempt_number
        self.updated_at = datetime.utcnow()
        
        if not attempt.success:
            if self.first_error is None and attempt.error_message:
                self.first_error = attempt.error_message
            self.last_error = attempt.error_message
            self.last_error_category = attempt.error_category
    
    def should_retry(self) -> bool:
        """Check if operation should be retried based on current state."""
        if self.status != RetryResult.FAILED:
            return False
        
        if self.current_attempt >= self.policy.max_attempts:
            return False
        
        if self.last_error_category and not self.policy.is_retryable(self.last_error_category):
            return False
        
        return True
    
    def calculate_next_attempt_time(self) -> datetime:
        """Calculate when the next attempt should be made."""
        delay = self.policy.calculate_delay(self.current_attempt + 1)
        return datetime.utcnow() + timedelta(seconds=delay)
    
    def is_expired(self, max_age_hours: int = 24) -> bool:
        """Check if retry state has expired."""
        age = datetime.utcnow() - self.created_at
        return age.total_seconds() > (max_age_hours * 3600)


# Error classifier functions type
ErrorClassifierFunc = Callable[[Exception, Dict[str, Any]], ErrorCategory]
BackoffCalculatorFunc = Callable[[int, float, Optional[Exception]], float]


class IErrorClassifier(ABC):
    """Interface for classifying errors into retry categories."""
    
    @abstractmethod
    def classify_error(
        self, 
        error: Exception, 
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorCategory:
        """
        Classify an error into a retry category.
        
        Args:
            error: The exception that occurred
            context: Additional context about the operation
            
        Returns:
            ErrorCategory for the error
        """
        pass
    
    @abstractmethod
    def is_transient(self, error: Exception) -> bool:
        """Check if an error is likely transient and retryable."""
        pass
    
    @abstractmethod
    def extract_retry_after(self, error: Exception) -> Optional[float]:
        """Extract retry-after delay from error if available."""
        pass


class IRetryService(ABC):
    """Service interface for managing retry operations."""
    
    @abstractmethod
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
        """
        Create a new retry state for an operation.
        
        Args:
            operation_id: Unique identifier for the operation
            operation_type: Type of operation being retried
            tenant_id: Tenant identifier
            user_id: User identifier (optional)
            policy: Retry policy to use (optional, uses default if not provided)
            context: Additional context for the operation
            
        Returns:
            Retry ID for tracking
        """
        pass
    
    @abstractmethod
    async def get_retry_state(self, retry_id: str) -> Optional[RetryState]:
        """Get retry state by ID."""
        pass
    
    @abstractmethod
    async def record_attempt(
        self,
        retry_id: str,
        error: Optional[Exception] = None,
        success: bool = False,
        operation_duration_ms: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RetryAttempt:
        """
        Record a retry attempt.
        
        Args:
            retry_id: Retry state identifier
            error: Error that occurred (if any)
            success: Whether the attempt was successful
            operation_duration_ms: How long the operation took
            context: Additional context
            
        Returns:
            RetryAttempt record
        """
        pass
    
    @abstractmethod
    async def schedule_next_attempt(
        self,
        retry_id: str,
        delay_override: Optional[float] = None
    ) -> Optional[datetime]:
        """
        Schedule the next retry attempt.
        
        Args:
            retry_id: Retry state identifier
            delay_override: Override calculated delay
            
        Returns:
            When the next attempt is scheduled, or None if no more attempts
        """
        pass
    
    @abstractmethod
    async def get_ready_retries(
        self,
        operation_type: Optional[str] = None,
        limit: int = 100
    ) -> List[RetryState]:
        """
        Get retry operations that are ready to be attempted.
        
        Args:
            operation_type: Filter by operation type
            limit: Maximum number of retries to return
            
        Returns:
            List of ready retry states
        """
        pass
    
    @abstractmethod
    async def cancel_retry(
        self,
        retry_id: str,
        reason: str = "Cancelled by user"
    ) -> bool:
        """
        Cancel a retry operation.
        
        Args:
            retry_id: Retry state identifier
            reason: Cancellation reason
            
        Returns:
            True if cancellation was successful
        """
        pass
    
    @abstractmethod
    async def move_to_dead_letter(
        self,
        retry_id: str,
        reason: str = "Max attempts exceeded"
    ) -> str:
        """
        Move a failed retry to dead letter queue.
        
        Args:
            retry_id: Retry state identifier
            reason: Reason for moving to dead letter
            
        Returns:
            Dead letter ID
        """
        pass
    
    @abstractmethod
    async def cleanup_expired_retries(
        self,
        max_age_hours: int = 24,
        batch_size: int = 100
    ) -> int:
        """
        Clean up expired retry states.
        
        Args:
            max_age_hours: Age threshold for cleanup
            batch_size: Maximum number to clean in one batch
            
        Returns:
            Number of retry states cleaned up
        """
        pass
    
    @abstractmethod
    async def get_retry_statistics(
        self,
        tenant_id: Optional[str] = None,
        operation_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get retry statistics.
        
        Args:
            tenant_id: Filter by tenant
            operation_type: Filter by operation type
            start_time: Start time for statistics
            end_time: End time for statistics
            
        Returns:
            Statistics dictionary
        """
        pass


class IDeadLetterService(ABC):
    """Service interface for managing dead letter queue."""
    
    @abstractmethod
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
        """
        Add a failed operation to the dead letter queue.
        
        Args:
            operation_id: Original operation identifier
            operation_type: Type of operation
            tenant_id: Tenant identifier
            retry_id: Associated retry ID (if any)
            final_error: Final error message
            error_category: Category of the final error
            retry_attempts: Number of retry attempts made
            context: Operation context
            metadata: Additional metadata
            
        Returns:
            Dead letter ID
        """
        pass
    
    @abstractmethod
    async def get_dead_letter_entry(self, dead_letter_id: str) -> Optional[Dict[str, Any]]:
        """Get a dead letter entry by ID."""
        pass
    
    @abstractmethod
    async def requeue_from_dead_letter(
        self,
        dead_letter_id: str,
        *,
        admin_user_id: str,
        notes: Optional[str] = None,
        new_policy: Optional[RetryPolicy] = None
    ) -> str:
        """
        Requeue an operation from the dead letter queue.
        
        Args:
            dead_letter_id: Dead letter entry ID
            admin_user_id: Admin user performing the requeue
            notes: Notes about the requeue
            new_policy: New retry policy to use
            
        Returns:
            New retry ID
        """
        pass
    
    @abstractmethod
    async def resolve_dead_letter(
        self,
        dead_letter_id: str,
        *,
        resolution_action: str,
        admin_user_id: str,
        notes: Optional[str] = None
    ) -> bool:
        """
        Mark a dead letter entry as resolved.
        
        Args:
            dead_letter_id: Dead letter entry ID
            resolution_action: Action taken to resolve
            admin_user_id: Admin user resolving
            notes: Resolution notes
            
        Returns:
            True if resolution was successful
        """
        pass
    
    @abstractmethod
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
        """
        Get dead letter queue entries with filtering.
        
        Args:
            tenant_id: Filter by tenant
            operation_type: Filter by operation type
            error_category: Filter by error category
            start_date: Start date filter
            end_date: End date filter
            resolved: Filter by resolution status
            limit: Maximum entries to return
            offset: Offset for pagination
            
        Returns:
            Dead letter entries with pagination info
        """
        pass
    
    @abstractmethod
    async def cleanup_old_dead_letters(
        self,
        older_than_days: int = 30,
        batch_size: int = 100
    ) -> int:
        """
        Clean up old dead letter entries.
        
        Args:
            older_than_days: Age threshold for cleanup
            batch_size: Batch size for cleanup
            
        Returns:
            Number of entries cleaned up
        """
        pass


class IRetryOperationExecutor(ABC):
    """Interface for executing retry operations."""
    
    @abstractmethod
    async def execute_with_retry(
        self,
        operation_func: Callable[..., Any],
        operation_id: str,
        operation_type: str,
        tenant_id: str,
        *,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        policy: Optional[RetryPolicy] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute an operation with retry logic.
        
        Args:
            operation_func: Function to execute
            operation_id: Unique operation identifier
            operation_type: Type of operation
            tenant_id: Tenant identifier
            args: Function arguments
            kwargs: Function keyword arguments
            user_id: User identifier
            policy: Retry policy
            context: Operation context
            
        Returns:
            Result of successful operation
            
        Raises:
            Exception: If operation fails after all retries
        """
        pass


__all__ = [
    "ErrorCategory",
    "RetryResult", 
    "BackoffStrategy",
    "RetryPolicy",
    "RetryAttempt",
    "RetryState",
    "IErrorClassifier",
    "IRetryService",
    "IDeadLetterService",
    "IRetryOperationExecutor",
    "ErrorClassifierFunc",
    "BackoffCalculatorFunc"
]