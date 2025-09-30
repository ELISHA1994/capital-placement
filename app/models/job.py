"""
Job models for background task processing.

This module defines models for background jobs, job queues,
and task execution tracking.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import Field, validator

from .base import BaseModel, TimestampedModel


class JobStatus(str, Enum):
    """Job execution status enumeration."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobPriority(str, Enum):
    """Job priority enumeration."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class JobType(str, Enum):
    """Job type enumeration."""
    DOCUMENT_PROCESSING = "document_processing"
    PROFILE_INDEXING = "profile_indexing"
    SEARCH_INDEXING = "search_indexing"
    EMAIL_NOTIFICATION = "email_notification"
    BULK_IMPORT = "bulk_import"
    BULK_EXPORT = "bulk_export"
    ANALYTICS_AGGREGATION = "analytics_aggregation"
    CLEANUP_TASK = "cleanup_task"
    WEBHOOK_DELIVERY = "webhook_delivery"
    REPORT_GENERATION = "report_generation"


class RetryStrategy(BaseModel):
    """Job retry strategy configuration."""
    
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    initial_delay: int = Field(default=60, ge=1, description="Initial retry delay (seconds)")
    backoff_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Backoff multiplier for retry delays"
    )
    max_delay: int = Field(
        default=3600,
        ge=1,
        description="Maximum retry delay (seconds)"
    )
    jitter: bool = Field(default=True, description="Add random jitter to delays")
    
    def calculate_delay(self, attempt: int) -> int:
        """Calculate retry delay for given attempt."""
        delay = self.initial_delay * (self.backoff_multiplier ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)  # 50-100% of calculated delay
        
        return int(delay)


class JobProgress(BaseModel):
    """Job progress tracking model."""
    
    current_step: int = Field(default=0, ge=0, description="Current step number")
    total_steps: int = Field(default=1, ge=1, description="Total number of steps")
    percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="Progress percentage")
    message: Optional[str] = Field(None, description="Progress message")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional progress details"
    )
    
    def update_progress(self, current: int, total: int, message: Optional[str] = None) -> None:
        """Update progress information."""
        self.current_step = current
        self.total_steps = total
        self.percentage = (current / total) * 100 if total > 0 else 0.0
        if message:
            self.message = message


class JobResult(BaseModel):
    """Job execution result model."""
    
    success: bool = Field(..., description="Whether job completed successfully")
    result_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Job result data"
    )
    output_files: List[str] = Field(
        default_factory=list,
        description="Output file paths/URLs"
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Job execution metrics"
    )
    summary: Optional[str] = Field(None, description="Result summary")
    
    # Error information
    error_code: Optional[str] = Field(None, description="Error code if failed")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    error_details: Optional[Dict[str, Any]] = Field(
        None,
        description="Detailed error information"
    )
    stack_trace: Optional[str] = Field(None, description="Stack trace if failed")


class Job(TimestampedModel):
    """Background job model."""
    
    # Job identification
    id: UUID = Field(default_factory=uuid4, description="Job unique identifier")
    job_type: JobType = Field(..., description="Type of job")
    name: str = Field(..., min_length=1, max_length=200, description="Job name")
    description: Optional[str] = Field(None, max_length=1000, description="Job description")
    
    # Context
    tenant_id: UUID = Field(..., description="Tenant ID")
    user_id: Optional[UUID] = Field(None, description="User who initiated the job")
    
    # Job configuration
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Job input parameters"
    )
    priority: JobPriority = Field(default=JobPriority.NORMAL, description="Job priority")
    queue_name: str = Field(default="default", description="Queue name")
    
    # Scheduling
    scheduled_for: Optional[datetime] = Field(
        None,
        description="Scheduled execution time"
    )
    expires_at: Optional[datetime] = Field(
        None,
        description="Job expiration time"
    )
    
    # Retry configuration
    retry_strategy: RetryStrategy = Field(
        default_factory=RetryStrategy,
        description="Retry strategy"
    )
    
    # Execution tracking
    status: JobStatus = Field(default=JobStatus.PENDING, description="Current job status")
    started_at: Optional[datetime] = Field(None, description="Execution start time")
    completed_at: Optional[datetime] = Field(None, description="Execution completion time")
    cancelled_at: Optional[datetime] = Field(None, description="Cancellation time")
    
    # Progress tracking
    progress: JobProgress = Field(
        default_factory=JobProgress,
        description="Job progress information"
    )
    
    # Retry tracking
    attempt_count: int = Field(default=0, description="Number of execution attempts")
    last_attempt_at: Optional[datetime] = Field(None, description="Last attempt timestamp")
    next_retry_at: Optional[datetime] = Field(None, description="Next retry timestamp")
    
    # Results
    result: Optional[JobResult] = Field(None, description="Job execution result")
    
    # Dependencies
    depends_on: List[UUID] = Field(
        default_factory=list,
        description="Job IDs this job depends on"
    )
    blocks: List[UUID] = Field(
        default_factory=list,
        description="Job IDs blocked by this job"
    )
    
    # Worker information
    worker_id: Optional[str] = Field(None, description="Worker that processed the job")
    worker_hostname: Optional[str] = Field(None, description="Worker hostname")
    
    # Tags and metadata
    tags: List[str] = Field(default_factory=list, description="Job tags")
    job_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional job metadata"
    )
    
    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
    
    @property
    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.status == JobStatus.RUNNING
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get job execution duration."""
        if not self.started_at:
            return None
        
        end_time = self.completed_at or datetime.utcnow()
        return end_time - self.started_at
    
    @property
    def can_retry(self) -> bool:
        """Check if job can be retried."""
        return (
            self.status == JobStatus.FAILED and
            self.attempt_count < self.retry_strategy.max_retries
        )
    
    @property
    def is_expired(self) -> bool:
        """Check if job is expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    def start_execution(self, worker_id: str, worker_hostname: str) -> None:
        """Mark job as started."""
        self.status = JobStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.worker_id = worker_id
        self.worker_hostname = worker_hostname
        self.attempt_count += 1
        self.last_attempt_at = self.started_at
        self.update_timestamp()
    
    def complete_success(self, result: JobResult) -> None:
        """Mark job as successfully completed."""
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.result = result
        self.progress.percentage = 100.0
        self.update_timestamp()
    
    def complete_failure(self, result: JobResult) -> None:
        """Mark job as failed."""
        self.status = JobStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.result = result
        
        # Schedule retry if applicable
        if self.can_retry:
            delay = self.retry_strategy.calculate_delay(self.attempt_count)
            self.next_retry_at = datetime.utcnow() + timedelta(seconds=delay)
            self.status = JobStatus.RETRYING
        
        self.update_timestamp()
    
    def cancel(self, reason: Optional[str] = None) -> None:
        """Cancel the job."""
        self.status = JobStatus.CANCELLED
        self.cancelled_at = datetime.utcnow()
        if reason:
            self.job_metadata["cancellation_reason"] = reason
        self.update_timestamp()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the job."""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the job."""
        if tag in self.tags:
            self.tags.remove(tag)


class JobCreate(BaseModel):
    """Model for job creation."""
    
    job_type: JobType = Field(..., description="Type of job")
    name: str = Field(..., min_length=1, max_length=200, description="Job name")
    description: Optional[str] = Field(None, max_length=1000)
    
    # Configuration
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: JobPriority = Field(default=JobPriority.NORMAL)
    queue_name: str = Field(default="default")
    
    # Scheduling
    scheduled_for: Optional[datetime] = None
    expires_in_hours: Optional[int] = Field(None, ge=1, le=168, description="Expiry in hours")
    
    # Retry configuration
    max_retries: Optional[int] = Field(None, ge=0, le=10)
    
    # Dependencies
    depends_on: List[UUID] = Field(default_factory=list)
    
    # Tags
    tags: List[str] = Field(default_factory=list)
    job_metadata: Dict[str, Any] = Field(default_factory=dict)


class JobUpdate(BaseModel):
    """Model for job updates."""
    
    priority: Optional[JobPriority] = None
    scheduled_for: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    tags: Optional[List[str]] = None
    job_metadata: Optional[Dict[str, Any]] = None


class JobQueue(BaseModel):
    """Job queue model."""
    
    name: str = Field(..., min_length=1, max_length=50, description="Queue name")
    description: Optional[str] = Field(None, max_length=500)
    
    # Queue configuration
    max_workers: int = Field(default=5, ge=1, le=100, description="Maximum concurrent workers")
    max_jobs_per_worker: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Maximum jobs per worker"
    )
    default_priority: JobPriority = Field(
        default=JobPriority.NORMAL,
        description="Default job priority"
    )
    
    # Retry configuration
    default_retry_strategy: RetryStrategy = Field(
        default_factory=RetryStrategy,
        description="Default retry strategy"
    )
    
    # Queue limits
    max_queue_size: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum queue size (None = unlimited)"
    )
    job_timeout: int = Field(
        default=3600,
        ge=1,
        description="Job execution timeout (seconds)"
    )
    
    # Status
    is_active: bool = Field(default=True, description="Queue is active")
    is_paused: bool = Field(default=False, description="Queue is paused")
    
    # Statistics
    total_jobs: int = Field(default=0, description="Total jobs processed")
    pending_jobs: int = Field(default=0, description="Jobs pending execution")
    running_jobs: int = Field(default=0, description="Jobs currently running")
    completed_jobs: int = Field(default=0, description="Successfully completed jobs")
    failed_jobs: int = Field(default=0, description="Failed jobs")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class JobBatch(BaseModel):
    """Batch job processing model."""
    
    id: UUID = Field(default_factory=uuid4, description="Batch ID")
    name: str = Field(..., min_length=1, max_length=200, description="Batch name")
    description: Optional[str] = Field(None, max_length=1000)
    
    # Context
    tenant_id: UUID = Field(..., description="Tenant ID")
    user_id: Optional[UUID] = Field(None, description="User who created the batch")
    
    # Jobs
    job_ids: List[UUID] = Field(..., description="Job IDs in the batch")
    
    # Configuration
    run_in_parallel: bool = Field(default=True, description="Run jobs in parallel")
    stop_on_failure: bool = Field(default=False, description="Stop batch on first failure")
    
    # Status tracking
    status: JobStatus = Field(default=JobStatus.PENDING, description="Overall batch status")
    started_at: Optional[datetime] = Field(None, description="Batch start time")
    completed_at: Optional[datetime] = Field(None, description="Batch completion time")
    
    # Progress
    completed_jobs: int = Field(default=0, description="Number of completed jobs")
    failed_jobs: int = Field(default=0, description="Number of failed jobs")
    
    # Results
    results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Batch execution results"
    )
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def total_jobs(self) -> int:
        """Get total number of jobs in batch."""
        return len(self.job_ids)
    
    @property
    def progress_percentage(self) -> float:
        """Calculate batch progress percentage."""
        if self.total_jobs == 0:
            return 0.0
        return (self.completed_jobs / self.total_jobs) * 100
    
    @property
    def is_complete(self) -> bool:
        """Check if all jobs in batch are complete."""
        return self.completed_jobs + self.failed_jobs == self.total_jobs


class JobStats(BaseModel):
    """Job execution statistics."""
    
    # Overall statistics
    total_jobs: int = Field(default=0, description="Total number of jobs")
    pending_jobs: int = Field(default=0, description="Pending jobs")
    running_jobs: int = Field(default=0, description="Currently running jobs")
    completed_jobs: int = Field(default=0, description="Successfully completed jobs")
    failed_jobs: int = Field(default=0, description="Failed jobs")
    cancelled_jobs: int = Field(default=0, description="Cancelled jobs")
    
    # Success metrics
    success_rate: float = Field(default=0.0, description="Overall success rate (%)")
    avg_execution_time: float = Field(default=0.0, description="Average execution time (seconds)")
    
    # Queue statistics
    by_queue: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Statistics by queue"
    )
    by_type: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Statistics by job type"
    )
    by_priority: Dict[str, int] = Field(
        default_factory=dict,
        description="Jobs by priority"
    )
    
    # Time-based statistics
    jobs_last_hour: int = Field(default=0, description="Jobs completed in last hour")
    jobs_last_day: int = Field(default=0, description="Jobs completed in last day")
    jobs_last_week: int = Field(default=0, description="Jobs completed in last week")
    
    # Error analysis
    common_errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Most common error types"
    )
    
    # Performance metrics
    throughput_per_hour: float = Field(
        default=0.0,
        description="Jobs processed per hour"
    )
    avg_queue_time: float = Field(
        default=0.0,
        description="Average time jobs spend in queue (seconds)"
    )