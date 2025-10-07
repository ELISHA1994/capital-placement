"""Pure domain representation of document processing operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID


class ProcessingType(str, Enum):
    """Types of document processing operations."""

    AI_ANALYSIS = "ai_analysis"
    OCR = "ocr"
    EMBEDDING_GENERATION = "embedding_generation"
    QUALITY_CHECK = "quality_check"
    TEXT_EXTRACTION = "text_extraction"
    METADATA_EXTRACTION = "metadata_extraction"


class ProcessingStatus(str, Enum):
    """Document processing lifecycle statuses."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


@dataclass
class ProcessingTiming:
    """Timing information for document processing operations."""

    started_at: datetime
    completed_at: Optional[datetime] = None
    processing_duration_ms: Optional[int] = None

    def mark_completed(self) -> None:
        """Mark processing as completed and calculate duration."""
        self.completed_at = datetime.now(timezone.utc)
        if self.started_at:
            duration = (self.completed_at - self.started_at).total_seconds() * 1000
            self.processing_duration_ms = int(duration)

    def is_completed(self) -> bool:
        """Check if processing has completed."""
        return self.completed_at is not None

    def duration_seconds(self) -> Optional[float]:
        """Get duration in seconds."""
        if self.processing_duration_ms is not None:
            return self.processing_duration_ms / 1000.0
        return None


@dataclass
class QualityMetrics:
    """Quality assessment metrics for processed documents."""

    quality_score: Optional[float] = None
    confidence_level: Optional[float] = None
    completeness_score: Optional[float] = None
    readability_score: Optional[float] = None
    extraction_accuracy: Optional[float] = None

    def is_high_quality(self, threshold: float = 0.7) -> bool:
        """Check if quality meets the threshold."""
        return self.quality_score is not None and self.quality_score >= threshold

    def has_metrics(self) -> bool:
        """Check if any metrics are available."""
        return any([
            self.quality_score,
            self.confidence_level,
            self.completeness_score,
            self.readability_score,
            self.extraction_accuracy
        ])


@dataclass
class ProcessingError:
    """Structured error information for failed processing operations."""

    error_code: str
    error_message: str
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    retry_count: int = 0
    recoverable: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def increment_retry(self) -> None:
        """Increment retry count."""
        self.retry_count += 1

    def can_retry(self, max_retries: int = 3) -> bool:
        """Check if processing can be retried."""
        return self.recoverable and self.retry_count < max_retries

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_code": self.error_code,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "stack_trace": self.stack_trace,
            "retry_count": self.retry_count,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class DocumentProcessing:
    """
    Aggregate root representing a document processing operation.

    This entity tracks the lifecycle of document processing operations including
    AI analysis, OCR, embedding generation, and quality checks. It provides
    business logic for managing processing state, quality metrics, and error handling.
    """

    id: UUID
    tenant_id: UUID
    document_id: UUID
    processing_type: ProcessingType
    status: ProcessingStatus
    input_metadata: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)
    error: Optional[ProcessingError] = None
    timing: ProcessingTiming = field(default_factory=lambda: ProcessingTiming(started_at=datetime.now(timezone.utc)))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Initialize computed fields after creation."""
        # Ensure timing is initialized if not provided
        if not hasattr(self, 'timing') or self.timing is None:
            self.timing = ProcessingTiming(started_at=self.created_at)

    def start_processing(self) -> None:
        """Mark processing as started."""
        if self.status != ProcessingStatus.PENDING:
            raise ValueError(f"Cannot start processing from status: {self.status}")

        self.status = ProcessingStatus.PROCESSING
        self.timing = ProcessingTiming(started_at=datetime.now(timezone.utc))
        self.updated_at = datetime.now(timezone.utc)

    def complete_processing(
        self,
        output_data: Dict[str, Any],
        quality_score: Optional[float] = None
    ) -> None:
        """Mark processing as completed successfully."""
        if self.status != ProcessingStatus.PROCESSING:
            raise ValueError(f"Cannot complete processing from status: {self.status}")

        self.status = ProcessingStatus.COMPLETED
        self.output_data = output_data
        self.timing.mark_completed()

        if quality_score is not None:
            self.quality_metrics.quality_score = quality_score

        self.updated_at = datetime.now(timezone.utc)

    def fail_processing(
        self,
        error_code: str,
        error_message: str,
        error_type: Optional[str] = None,
        recoverable: bool = False
    ) -> None:
        """Mark processing as failed with error details."""
        self.status = ProcessingStatus.FAILED
        self.error = ProcessingError(
            error_code=error_code,
            error_message=error_message,
            error_type=error_type,
            recoverable=recoverable
        )
        self.timing.mark_completed()
        self.updated_at = datetime.now(timezone.utc)

    def mark_partial_completion(
        self,
        partial_output: Dict[str, Any],
        reason: str
    ) -> None:
        """Mark processing as partially completed."""
        self.status = ProcessingStatus.PARTIAL
        self.output_data = partial_output
        self.output_data["partial_reason"] = reason
        self.timing.mark_completed()
        self.updated_at = datetime.now(timezone.utc)

    def cancel_processing(self, reason: Optional[str] = None) -> None:
        """Cancel the processing operation."""
        if self.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
            raise ValueError(f"Cannot cancel processing in status: {self.status}")

        self.status = ProcessingStatus.CANCELLED
        if reason:
            self.output_data = {"cancellation_reason": reason}
        self.timing.mark_completed()
        self.updated_at = datetime.now(timezone.utc)

    def retry_processing(self) -> None:
        """Retry failed processing operation."""
        if self.status != ProcessingStatus.FAILED:
            raise ValueError("Can only retry failed processing")

        if not self.error or not self.error.can_retry():
            raise ValueError("Processing cannot be retried")

        self.error.increment_retry()
        self.status = ProcessingStatus.PENDING
        self.timing = ProcessingTiming(started_at=datetime.now(timezone.utc))
        self.updated_at = datetime.now(timezone.utc)

    def set_quality_metrics(
        self,
        quality_score: Optional[float] = None,
        confidence_level: Optional[float] = None,
        completeness_score: Optional[float] = None
    ) -> None:
        """Update quality metrics for the processing operation."""
        if quality_score is not None:
            self.quality_metrics.quality_score = quality_score
        if confidence_level is not None:
            self.quality_metrics.confidence_level = confidence_level
        if completeness_score is not None:
            self.quality_metrics.completeness_score = completeness_score

        self.updated_at = datetime.now(timezone.utc)

    def add_output_data(self, key: str, value: Any) -> None:
        """Add or update a key in the output data."""
        if self.output_data is None:
            self.output_data = {}
        self.output_data[key] = value
        self.updated_at = datetime.now(timezone.utc)

    def add_input_metadata(self, key: str, value: Any) -> None:
        """Add or update a key in the input metadata."""
        if self.input_metadata is None:
            self.input_metadata = {}
        self.input_metadata[key] = value
        self.updated_at = datetime.now(timezone.utc)

    def is_completed(self) -> bool:
        """Check if processing has completed successfully."""
        return self.status == ProcessingStatus.COMPLETED

    def is_failed(self) -> bool:
        """Check if processing has failed."""
        return self.status == ProcessingStatus.FAILED

    def is_in_progress(self) -> bool:
        """Check if processing is currently in progress."""
        return self.status == ProcessingStatus.PROCESSING

    def can_be_retried(self) -> bool:
        """Check if the processing operation can be retried."""
        return (
            self.status == ProcessingStatus.FAILED and
            self.error is not None and
            self.error.can_retry()
        )

    def get_duration_ms(self) -> Optional[int]:
        """Get processing duration in milliseconds."""
        return self.timing.processing_duration_ms

    def get_duration_seconds(self) -> Optional[float]:
        """Get processing duration in seconds."""
        return self.timing.duration_seconds()

    def is_high_quality(self, threshold: float = 0.7) -> bool:
        """Check if processing result meets quality threshold."""
        return self.quality_metrics.is_high_quality(threshold)

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get a summary of the processing operation."""
        return {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "processing_type": self.processing_type.value,
            "status": self.status.value,
            "quality_score": self.quality_metrics.quality_score,
            "duration_ms": self.get_duration_ms(),
            "started_at": self.timing.started_at.isoformat(),
            "completed_at": self.timing.completed_at.isoformat() if self.timing.completed_at else None,
            "has_error": self.error is not None,
            "error_details": self.error.to_dict() if self.error else None
        }

    def validate(self) -> list[str]:
        """Validate the entity and return any validation errors."""
        errors = []

        if not self.tenant_id:
            errors.append("tenant_id is required")

        if not self.document_id:
            errors.append("document_id is required")

        if not self.processing_type:
            errors.append("processing_type is required")

        if self.status == ProcessingStatus.COMPLETED and not self.output_data:
            errors.append("output_data is required for completed processing")

        if self.status == ProcessingStatus.FAILED and not self.error:
            errors.append("error details are required for failed processing")

        if self.quality_metrics.quality_score is not None:
            if not 0.0 <= self.quality_metrics.quality_score <= 1.0:
                errors.append("quality_score must be between 0 and 1")

        return errors


__all__ = [
    "DocumentProcessing",
    "ProcessingType",
    "ProcessingStatus",
    "ProcessingTiming",
    "QualityMetrics",
    "ProcessingError",
]