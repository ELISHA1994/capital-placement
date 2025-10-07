"""
Pure domain tests for DocumentProcessing entity.

These tests validate the business logic of the DocumentProcessing aggregate root
and its value objects WITHOUT any infrastructure dependencies (no database, no mappers).

Test Coverage:
- ProcessingTiming value object behavior
- QualityMetrics value object behavior
- ProcessingError value object behavior
- DocumentProcessing entity creation
- State transition business rules
- Quality metrics management
- Error handling and retry logic
- Timing and duration calculations
- Validation rules
- Processing summaries
"""

import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from typing import Dict, Any

from app.domain.entities.document_processing import (
    DocumentProcessing,
    ProcessingType,
    ProcessingStatus,
    ProcessingTiming,
    QualityMetrics,
    ProcessingError,
)


# ============================================================================
# ProcessingTiming Tests
# ============================================================================


class TestProcessingTiming:
    """Tests for ProcessingTiming value object."""

    def test_timing_initialization(self):
        """ProcessingTiming should initialize with started_at."""
        started = datetime.now(timezone.utc)
        timing = ProcessingTiming(started_at=started)

        assert timing.started_at == started
        assert timing.completed_at is None
        assert timing.processing_duration_ms is None

    def test_mark_completed_sets_timestamp(self):
        """mark_completed should set completed_at timestamp."""
        started = datetime.now(timezone.utc)
        timing = ProcessingTiming(started_at=started)

        timing.mark_completed()

        assert timing.completed_at is not None
        assert timing.completed_at > started

    def test_mark_completed_calculates_duration(self):
        """mark_completed should calculate processing duration in milliseconds."""
        started = datetime.now(timezone.utc) - timedelta(seconds=2)
        timing = ProcessingTiming(started_at=started)

        timing.mark_completed()

        assert timing.processing_duration_ms is not None
        assert timing.processing_duration_ms >= 2000  # At least 2 seconds

    def test_is_completed_returns_false_initially(self):
        """is_completed should return False before completion."""
        timing = ProcessingTiming(started_at=datetime.now(timezone.utc))

        assert timing.is_completed() is False

    def test_is_completed_returns_true_after_marking(self):
        """is_completed should return True after mark_completed."""
        timing = ProcessingTiming(started_at=datetime.now(timezone.utc))
        timing.mark_completed()

        assert timing.is_completed() is True

    def test_duration_seconds_returns_none_initially(self):
        """duration_seconds should return None before completion."""
        timing = ProcessingTiming(started_at=datetime.now(timezone.utc))

        assert timing.duration_seconds() is None

    def test_duration_seconds_converts_from_milliseconds(self):
        """duration_seconds should convert milliseconds to seconds."""
        timing = ProcessingTiming(started_at=datetime.now(timezone.utc))
        timing.processing_duration_ms = 5000

        assert timing.duration_seconds() == 5.0


# ============================================================================
# QualityMetrics Tests
# ============================================================================


class TestQualityMetrics:
    """Tests for QualityMetrics value object."""

    def test_quality_metrics_initialization(self):
        """QualityMetrics should initialize with all None values."""
        metrics = QualityMetrics()

        assert metrics.quality_score is None
        assert metrics.confidence_level is None
        assert metrics.completeness_score is None
        assert metrics.readability_score is None
        assert metrics.extraction_accuracy is None

    def test_is_high_quality_returns_false_for_none(self):
        """is_high_quality should return False when quality_score is None."""
        metrics = QualityMetrics()

        assert metrics.is_high_quality() is False

    def test_is_high_quality_uses_default_threshold(self):
        """is_high_quality should use 0.7 as default threshold."""
        metrics_high = QualityMetrics(quality_score=0.8)
        metrics_low = QualityMetrics(quality_score=0.6)

        assert metrics_high.is_high_quality() is True
        assert metrics_low.is_high_quality() is False

    def test_is_high_quality_uses_custom_threshold(self):
        """is_high_quality should accept custom threshold."""
        metrics = QualityMetrics(quality_score=0.75)

        assert metrics.is_high_quality(threshold=0.7) is True
        assert metrics.is_high_quality(threshold=0.8) is False

    def test_is_high_quality_boundary_condition(self):
        """is_high_quality should return True when score equals threshold."""
        metrics = QualityMetrics(quality_score=0.7)

        assert metrics.is_high_quality(threshold=0.7) is True

    def test_has_metrics_returns_false_for_all_none(self):
        """has_metrics should return False when all metrics are None."""
        metrics = QualityMetrics()

        assert metrics.has_metrics() is False

    def test_has_metrics_returns_true_for_any_metric(self):
        """has_metrics should return True if any metric is set."""
        metrics1 = QualityMetrics(quality_score=0.8)
        metrics2 = QualityMetrics(confidence_level=0.9)
        metrics3 = QualityMetrics(readability_score=0.7)

        assert metrics1.has_metrics() is True
        assert metrics2.has_metrics() is True
        assert metrics3.has_metrics() is True


# ============================================================================
# ProcessingError Tests
# ============================================================================


class TestProcessingError:
    """Tests for ProcessingError value object."""

    def test_processing_error_initialization(self):
        """ProcessingError should initialize with required fields."""
        error = ProcessingError(
            error_code="ERR_001",
            error_message="Processing failed"
        )

        assert error.error_code == "ERR_001"
        assert error.error_message == "Processing failed"
        assert error.error_type is None
        assert error.retry_count == 0
        assert error.recoverable is False

    def test_processing_error_with_optional_fields(self):
        """ProcessingError should accept optional fields."""
        error = ProcessingError(
            error_code="ERR_002",
            error_message="Timeout",
            error_type="TimeoutError",
            recoverable=True,
            stack_trace="line 1\nline 2"
        )

        assert error.error_type == "TimeoutError"
        assert error.recoverable is True
        assert error.stack_trace == "line 1\nline 2"

    def test_increment_retry_increases_count(self):
        """increment_retry should increase retry_count by 1."""
        error = ProcessingError(error_code="ERR", error_message="Test")

        assert error.retry_count == 0
        error.increment_retry()
        assert error.retry_count == 1
        error.increment_retry()
        assert error.retry_count == 2

    def test_can_retry_returns_false_for_non_recoverable(self):
        """can_retry should return False when error is not recoverable."""
        error = ProcessingError(
            error_code="ERR",
            error_message="Test",
            recoverable=False
        )

        assert error.can_retry() is False

    def test_can_retry_returns_true_within_limit(self):
        """can_retry should return True for recoverable errors within retry limit."""
        error = ProcessingError(
            error_code="ERR",
            error_message="Test",
            recoverable=True
        )
        error.retry_count = 2

        assert error.can_retry(max_retries=3) is True

    def test_can_retry_returns_false_at_limit(self):
        """can_retry should return False when retry limit is reached."""
        error = ProcessingError(
            error_code="ERR",
            error_message="Test",
            recoverable=True
        )
        error.retry_count = 3

        assert error.can_retry(max_retries=3) is False

    def test_to_dict_returns_complete_dictionary(self):
        """to_dict should return all error fields as dictionary."""
        timestamp = datetime.now(timezone.utc)
        error = ProcessingError(
            error_code="ERR_001",
            error_message="Test error",
            error_type="ValueError",
            stack_trace="stack",
            recoverable=True
        )
        error.retry_count = 2
        error.timestamp = timestamp

        result = error.to_dict()

        assert result["error_code"] == "ERR_001"
        assert result["error_message"] == "Test error"
        assert result["error_type"] == "ValueError"
        assert result["stack_trace"] == "stack"
        assert result["retry_count"] == 2
        assert result["recoverable"] is True
        assert result["timestamp"] == timestamp.isoformat()


# ============================================================================
# DocumentProcessing Entity Creation Tests
# ============================================================================


class TestDocumentProcessingCreation:
    """Tests for DocumentProcessing entity creation."""

    def test_minimal_entity_creation(self):
        """DocumentProcessing should be created with minimal required fields."""
        doc_id = uuid4()
        tenant_id = uuid4()
        document_id = uuid4()

        processing = DocumentProcessing(
            id=doc_id,
            tenant_id=tenant_id,
            document_id=document_id,
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.PENDING
        )

        assert processing.id == doc_id
        assert processing.tenant_id == tenant_id
        assert processing.document_id == document_id
        assert processing.processing_type == ProcessingType.AI_ANALYSIS
        assert processing.status == ProcessingStatus.PENDING

    def test_entity_initializes_default_values(self):
        """DocumentProcessing should initialize default values."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.OCR,
            status=ProcessingStatus.PENDING
        )

        assert isinstance(processing.quality_metrics, QualityMetrics)
        assert isinstance(processing.timing, ProcessingTiming)
        assert processing.input_metadata is None
        assert processing.output_data is None
        assert processing.error is None
        assert processing.created_at is not None
        assert processing.updated_at is not None

    def test_entity_with_all_fields(self):
        """DocumentProcessing should accept all fields."""
        created = datetime.now(timezone.utc)
        timing = ProcessingTiming(started_at=created)
        metrics = QualityMetrics(quality_score=0.9)

        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.EMBEDDING_GENERATION,
            status=ProcessingStatus.COMPLETED,
            input_metadata={"key": "value"},
            output_data={"result": "data"},
            quality_metrics=metrics,
            timing=timing,
            created_at=created,
            updated_at=created
        )

        assert processing.input_metadata == {"key": "value"}
        assert processing.output_data == {"result": "data"}
        assert processing.quality_metrics == metrics
        assert processing.timing == timing


# ============================================================================
# State Transition Tests
# ============================================================================


class TestDocumentProcessingStateTransitions:
    """Tests for DocumentProcessing state transition business rules."""

    def test_start_processing_from_pending(self):
        """start_processing should transition from PENDING to PROCESSING."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.PENDING
        )

        processing.start_processing()

        assert processing.status == ProcessingStatus.PROCESSING
        assert processing.timing.started_at is not None

    def test_start_processing_raises_from_non_pending(self):
        """start_processing should raise ValueError from non-PENDING status."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.PROCESSING
        )

        with pytest.raises(ValueError, match="Cannot start processing from status"):
            processing.start_processing()

    def test_complete_processing_from_processing(self):
        """complete_processing should transition from PROCESSING to COMPLETED."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.PROCESSING
        )

        output = {"analysis": "results"}
        processing.complete_processing(output_data=output, quality_score=0.85)

        assert processing.status == ProcessingStatus.COMPLETED
        assert processing.output_data == output
        assert processing.quality_metrics.quality_score == 0.85
        assert processing.timing.is_completed() is True

    def test_complete_processing_raises_from_non_processing(self):
        """complete_processing should raise ValueError from non-PROCESSING status."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.PENDING
        )

        with pytest.raises(ValueError, match="Cannot complete processing from status"):
            processing.complete_processing(output_data={"test": "data"})

    def test_fail_processing_sets_error(self):
        """fail_processing should transition to FAILED and set error details."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.OCR,
            status=ProcessingStatus.PROCESSING
        )

        processing.fail_processing(
            error_code="ERR_OCR_001",
            error_message="OCR extraction failed",
            error_type="OCRError",
            recoverable=True
        )

        assert processing.status == ProcessingStatus.FAILED
        assert processing.error is not None
        assert processing.error.error_code == "ERR_OCR_001"
        assert processing.error.error_message == "OCR extraction failed"
        assert processing.error.error_type == "OCRError"
        assert processing.error.recoverable is True
        assert processing.timing.is_completed() is True

    def test_mark_partial_completion(self):
        """mark_partial_completion should set status to PARTIAL with reason."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.TEXT_EXTRACTION,
            status=ProcessingStatus.PROCESSING
        )

        partial_output = {"extracted_pages": [1, 2, 3]}
        processing.mark_partial_completion(
            partial_output=partial_output,
            reason="Pages 4-5 corrupted"
        )

        assert processing.status == ProcessingStatus.PARTIAL
        assert processing.output_data["extracted_pages"] == [1, 2, 3]
        assert processing.output_data["partial_reason"] == "Pages 4-5 corrupted"
        assert processing.timing.is_completed() is True

    def test_cancel_processing_from_pending(self):
        """cancel_processing should transition to CANCELLED from PENDING."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.PENDING
        )

        processing.cancel_processing(reason="User requested cancellation")

        assert processing.status == ProcessingStatus.CANCELLED
        assert processing.output_data["cancellation_reason"] == "User requested cancellation"
        assert processing.timing.is_completed() is True

    def test_cancel_processing_raises_from_completed(self):
        """cancel_processing should raise ValueError from COMPLETED status."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.COMPLETED
        )

        with pytest.raises(ValueError, match="Cannot cancel processing in status"):
            processing.cancel_processing()

    def test_cancel_processing_raises_from_failed(self):
        """cancel_processing should raise ValueError from FAILED status."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.FAILED
        )

        with pytest.raises(ValueError, match="Cannot cancel processing in status"):
            processing.cancel_processing()

    def test_retry_processing_from_failed_recoverable(self):
        """retry_processing should transition from FAILED to PENDING if recoverable."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.EMBEDDING_GENERATION,
            status=ProcessingStatus.FAILED
        )
        processing.error = ProcessingError(
            error_code="ERR_TIMEOUT",
            error_message="Request timeout",
            recoverable=True
        )

        processing.retry_processing()

        assert processing.status == ProcessingStatus.PENDING
        assert processing.error.retry_count == 1

    def test_retry_processing_raises_from_non_failed(self):
        """retry_processing should raise ValueError from non-FAILED status."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.PROCESSING
        )

        with pytest.raises(ValueError, match="Can only retry failed processing"):
            processing.retry_processing()

    def test_retry_processing_raises_when_not_recoverable(self):
        """retry_processing should raise ValueError when error is not recoverable."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.FAILED
        )
        processing.error = ProcessingError(
            error_code="ERR_FATAL",
            error_message="Fatal error",
            recoverable=False
        )

        with pytest.raises(ValueError, match="Processing cannot be retried"):
            processing.retry_processing()

    def test_retry_processing_respects_retry_limit(self):
        """retry_processing should raise ValueError when retry limit exceeded."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.FAILED
        )
        processing.error = ProcessingError(
            error_code="ERR_TIMEOUT",
            error_message="Timeout",
            recoverable=True
        )
        processing.error.retry_count = 3  # At max retries

        with pytest.raises(ValueError, match="Processing cannot be retried"):
            processing.retry_processing()


# ============================================================================
# Quality Metrics Management Tests
# ============================================================================


class TestDocumentProcessingQualityMetrics:
    """Tests for quality metrics management."""

    def test_set_quality_metrics_updates_quality_score(self):
        """set_quality_metrics should update quality_score."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.QUALITY_CHECK,
            status=ProcessingStatus.PROCESSING
        )

        processing.set_quality_metrics(quality_score=0.92)

        assert processing.quality_metrics.quality_score == 0.92

    def test_set_quality_metrics_updates_confidence_level(self):
        """set_quality_metrics should update confidence_level."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.PROCESSING
        )

        processing.set_quality_metrics(confidence_level=0.88)

        assert processing.quality_metrics.confidence_level == 0.88

    def test_set_quality_metrics_updates_multiple_scores(self):
        """set_quality_metrics should update multiple metrics at once."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.PROCESSING
        )

        processing.set_quality_metrics(
            quality_score=0.85,
            confidence_level=0.90,
            completeness_score=0.95
        )

        assert processing.quality_metrics.quality_score == 0.85
        assert processing.quality_metrics.confidence_level == 0.90
        assert processing.quality_metrics.completeness_score == 0.95

    def test_is_high_quality_delegates_to_quality_metrics(self):
        """is_high_quality should delegate to QualityMetrics.is_high_quality."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.COMPLETED
        )
        processing.quality_metrics.quality_score = 0.85

        assert processing.is_high_quality(threshold=0.7) is True
        assert processing.is_high_quality(threshold=0.9) is False


# ============================================================================
# Data Management Tests
# ============================================================================


class TestDocumentProcessingDataManagement:
    """Tests for input/output data management."""

    def test_add_output_data_creates_dict_if_none(self):
        """add_output_data should create output_data dict if None."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.PROCESSING
        )

        processing.add_output_data("key1", "value1")

        assert processing.output_data == {"key1": "value1"}

    def test_add_output_data_appends_to_existing_dict(self):
        """add_output_data should add to existing output_data."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.PROCESSING
        )
        processing.output_data = {"existing": "data"}

        processing.add_output_data("new_key", "new_value")

        assert processing.output_data == {
            "existing": "data",
            "new_key": "new_value"
        }

    def test_add_input_metadata_creates_dict_if_none(self):
        """add_input_metadata should create input_metadata dict if None."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.OCR,
            status=ProcessingStatus.PENDING
        )

        processing.add_input_metadata("file_size", 1024)

        assert processing.input_metadata == {"file_size": 1024}

    def test_add_input_metadata_appends_to_existing_dict(self):
        """add_input_metadata should add to existing input_metadata."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.OCR,
            status=ProcessingStatus.PENDING
        )
        processing.input_metadata = {"file_type": "pdf"}

        processing.add_input_metadata("pages", 10)

        assert processing.input_metadata == {
            "file_type": "pdf",
            "pages": 10
        }


# ============================================================================
# Status Query Tests
# ============================================================================


class TestDocumentProcessingStatusQueries:
    """Tests for status query methods."""

    def test_is_completed_returns_true_for_completed(self):
        """is_completed should return True for COMPLETED status."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.COMPLETED
        )

        assert processing.is_completed() is True

    def test_is_completed_returns_false_for_other_statuses(self):
        """is_completed should return False for non-COMPLETED statuses."""
        statuses = [
            ProcessingStatus.PENDING,
            ProcessingStatus.PROCESSING,
            ProcessingStatus.FAILED,
            ProcessingStatus.PARTIAL,
            ProcessingStatus.CANCELLED
        ]

        for status in statuses:
            processing = DocumentProcessing(
                id=uuid4(),
                tenant_id=uuid4(),
                document_id=uuid4(),
                processing_type=ProcessingType.AI_ANALYSIS,
                status=status
            )
            assert processing.is_completed() is False

    def test_is_failed_returns_true_for_failed(self):
        """is_failed should return True for FAILED status."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.FAILED
        )

        assert processing.is_failed() is True

    def test_is_in_progress_returns_true_for_processing(self):
        """is_in_progress should return True for PROCESSING status."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.PROCESSING
        )

        assert processing.is_in_progress() is True

    def test_can_be_retried_returns_true_for_recoverable_error(self):
        """can_be_retried should return True for failed recoverable operations."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.FAILED
        )
        processing.error = ProcessingError(
            error_code="ERR",
            error_message="Test",
            recoverable=True
        )

        assert processing.can_be_retried() is True

    def test_can_be_retried_returns_false_for_non_failed(self):
        """can_be_retried should return False for non-FAILED status."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.PROCESSING
        )

        assert processing.can_be_retried() is False


# ============================================================================
# Duration and Timing Tests
# ============================================================================


class TestDocumentProcessingDuration:
    """Tests for duration calculation methods."""

    def test_get_duration_ms_returns_none_initially(self):
        """get_duration_ms should return None before completion."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.PROCESSING
        )

        assert processing.get_duration_ms() is None

    def test_get_duration_ms_returns_value_after_completion(self):
        """get_duration_ms should return duration after completion."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.PROCESSING
        )
        processing.complete_processing(output_data={"test": "data"})

        duration = processing.get_duration_ms()
        assert duration is not None
        assert duration >= 0

    def test_get_duration_seconds_converts_from_milliseconds(self):
        """get_duration_seconds should convert milliseconds to seconds."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.COMPLETED
        )
        processing.timing.processing_duration_ms = 3500

        assert processing.get_duration_seconds() == 3.5


# ============================================================================
# Processing Summary Tests
# ============================================================================


class TestDocumentProcessingSummary:
    """Tests for get_processing_summary method."""

    def test_get_processing_summary_includes_all_fields(self):
        """get_processing_summary should include all required fields."""
        doc_id = uuid4()
        document_id = uuid4()

        processing = DocumentProcessing(
            id=doc_id,
            tenant_id=uuid4(),
            document_id=document_id,
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.COMPLETED
        )
        processing.quality_metrics.quality_score = 0.9
        processing.timing.processing_duration_ms = 2500
        processing.timing.completed_at = datetime.now(timezone.utc)

        summary = processing.get_processing_summary()

        assert summary["id"] == str(doc_id)
        assert summary["document_id"] == str(document_id)
        assert summary["processing_type"] == "ai_analysis"
        assert summary["status"] == "completed"
        assert summary["quality_score"] == 0.9
        assert summary["duration_ms"] == 2500
        assert summary["started_at"] is not None
        assert summary["completed_at"] is not None
        assert summary["has_error"] is False
        assert summary["error_details"] is None

    def test_get_processing_summary_includes_error_details(self):
        """get_processing_summary should include error details when present."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.OCR,
            status=ProcessingStatus.FAILED
        )
        processing.fail_processing(
            error_code="ERR_001",
            error_message="Processing failed",
            recoverable=True
        )

        summary = processing.get_processing_summary()

        assert summary["has_error"] is True
        assert summary["error_details"] is not None
        assert summary["error_details"]["error_code"] == "ERR_001"


# ============================================================================
# Validation Tests
# ============================================================================


class TestDocumentProcessingValidation:
    """Tests for entity validation rules."""

    def test_validate_succeeds_for_valid_entity(self):
        """validate should return empty list for valid entity."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.PENDING
        )

        errors = processing.validate()

        assert errors == []

    def test_validate_detects_missing_tenant_id(self):
        """validate should detect missing tenant_id."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=None,  # type: ignore
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.PENDING
        )

        errors = processing.validate()

        assert "tenant_id is required" in errors

    def test_validate_detects_missing_document_id(self):
        """validate should detect missing document_id."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=None,  # type: ignore
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.PENDING
        )

        errors = processing.validate()

        assert "document_id is required" in errors

    def test_validate_detects_completed_without_output(self):
        """validate should detect COMPLETED status without output_data."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.COMPLETED
        )
        processing.output_data = None

        errors = processing.validate()

        assert "output_data is required for completed processing" in errors

    def test_validate_detects_failed_without_error(self):
        """validate should detect FAILED status without error details."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.FAILED
        )
        processing.error = None

        errors = processing.validate()

        assert "error details are required for failed processing" in errors

    def test_validate_detects_invalid_quality_score(self):
        """validate should detect quality_score outside 0-1 range."""
        processing1 = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.COMPLETED
        )
        processing1.quality_metrics.quality_score = 1.5

        processing2 = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.COMPLETED
        )
        processing2.quality_metrics.quality_score = -0.1

        errors1 = processing1.validate()
        errors2 = processing2.validate()

        assert "quality_score must be between 0 and 1" in errors1
        assert "quality_score must be between 0 and 1" in errors2

    def test_validate_returns_multiple_errors(self):
        """validate should return all validation errors."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=None,  # type: ignore
            document_id=None,  # type: ignore
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.COMPLETED
        )
        processing.output_data = None
        processing.quality_metrics.quality_score = 2.0

        errors = processing.validate()

        assert len(errors) >= 3
        assert "tenant_id is required" in errors
        assert "document_id is required" in errors


# ============================================================================
# Edge Cases and Business Rules
# ============================================================================


class TestDocumentProcessingEdgeCases:
    """Tests for edge cases and complex business rules."""

    def test_multiple_retry_attempts(self):
        """Should handle multiple retry attempts correctly."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.EMBEDDING_GENERATION,
            status=ProcessingStatus.FAILED
        )
        processing.error = ProcessingError(
            error_code="ERR_TIMEOUT",
            error_message="Timeout",
            recoverable=True
        )

        # First retry
        processing.retry_processing()
        assert processing.status == ProcessingStatus.PENDING
        assert processing.error.retry_count == 1

        # Simulate failure again
        processing.status = ProcessingStatus.FAILED

        # Second retry
        processing.retry_processing()
        assert processing.error.retry_count == 2

        # Simulate failure again
        processing.status = ProcessingStatus.FAILED

        # Third retry (last one)
        processing.retry_processing()
        assert processing.error.retry_count == 3

        # Simulate failure again - should not be retryable
        processing.status = ProcessingStatus.FAILED
        assert processing.can_be_retried() is False

    def test_cancel_processing_without_reason(self):
        """cancel_processing should work without reason."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.PENDING
        )

        processing.cancel_processing()

        assert processing.status == ProcessingStatus.CANCELLED
        # output_data should remain None when no reason provided
        assert processing.output_data is None

    def test_complete_processing_without_quality_score(self):
        """complete_processing should work without quality_score."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.TEXT_EXTRACTION,
            status=ProcessingStatus.PROCESSING
        )

        processing.complete_processing(output_data={"text": "extracted"})

        assert processing.status == ProcessingStatus.COMPLETED
        assert processing.quality_metrics.quality_score is None

    def test_quality_score_boundary_conditions(self):
        """Quality scores at boundaries (0.0 and 1.0) should be valid."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.QUALITY_CHECK,
            status=ProcessingStatus.COMPLETED
        )

        # Test 0.0
        processing.quality_metrics.quality_score = 0.0
        errors = processing.validate()
        assert "quality_score must be between 0 and 1" not in errors

        # Test 1.0
        processing.quality_metrics.quality_score = 1.0
        errors = processing.validate()
        assert "quality_score must be between 0 and 1" not in errors

    def test_processing_type_enum_values(self):
        """All ProcessingType enum values should be valid."""
        processing_types = [
            ProcessingType.AI_ANALYSIS,
            ProcessingType.OCR,
            ProcessingType.EMBEDDING_GENERATION,
            ProcessingType.QUALITY_CHECK,
            ProcessingType.TEXT_EXTRACTION,
            ProcessingType.METADATA_EXTRACTION
        ]

        for proc_type in processing_types:
            processing = DocumentProcessing(
                id=uuid4(),
                tenant_id=uuid4(),
                document_id=uuid4(),
                processing_type=proc_type,
                status=ProcessingStatus.PENDING
            )
            assert processing.processing_type == proc_type

    def test_updated_at_changes_on_operations(self):
        """updated_at should be refreshed on all state-changing operations."""
        processing = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.PENDING
        )
        initial_updated = processing.updated_at

        # Small delay to ensure timestamp difference
        import time
        time.sleep(0.01)

        processing.start_processing()
        assert processing.updated_at > initial_updated