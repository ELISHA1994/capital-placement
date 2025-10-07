"""
Comprehensive test suite for DocumentProcessingMapper bidirectional conversions.

This test suite ensures complete coverage of DocumentProcessingMapper functionality including:
- Basic entity <-> table conversions
- Complex nested structure handling (QualityMetrics, ProcessingError, ProcessingTiming)
- Value object conversions (ProcessingId, TenantId, DocumentId)
- Enum conversions (ProcessingType, ProcessingStatus)
- JSONB serialization/deserialization
- Optional/null field handling
- Update operations
- Business logic state transitions
- Edge cases and error conditions
- Property-based testing for roundtrip conversions
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from uuid import uuid4
from typing import Optional, Dict, Any

import pytest
from hypothesis import given, strategies as st, assume

from app.domain.entities.document_processing import (
    DocumentProcessing,
    ProcessingType,
    ProcessingStatus,
    ProcessingTiming,
    QualityMetrics,
    ProcessingError,
)
from app.infrastructure.persistence.models.document_processing_table import DocumentProcessingTable
from app.infrastructure.persistence.mappers.document_processing_mapper import DocumentProcessingMapper


# ========================================================================
# Test Fixtures and Factories
# ========================================================================

@pytest.fixture
def sample_processing_id():
    """Create a sample processing ID."""
    return uuid4()


@pytest.fixture
def sample_tenant_id():
    """Create a sample tenant ID."""
    return uuid4()


@pytest.fixture
def sample_document_id():
    """Create a sample document ID."""
    return uuid4()


@pytest.fixture
def sample_quality_metrics() -> QualityMetrics:
    """Create comprehensive QualityMetrics."""
    return QualityMetrics(
        quality_score=0.85,
        confidence_level=0.92,
        completeness_score=0.88,
        readability_score=0.90,
        extraction_accuracy=0.87
    )


@pytest.fixture
def sample_processing_error() -> ProcessingError:
    """Create a sample ProcessingError."""
    return ProcessingError(
        error_code="OCR_FAILURE",
        error_message="Failed to extract text from page 3",
        error_type="OCRException",
        stack_trace="Traceback (most recent call last)...",
        retry_count=1,
        recoverable=True,
        timestamp=datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    )


@pytest.fixture
def sample_processing_timing() -> ProcessingTiming:
    """Create sample ProcessingTiming."""
    return ProcessingTiming(
        started_at=datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        completed_at=datetime(2025, 1, 15, 10, 5, 0, tzinfo=timezone.utc),
        processing_duration_ms=300000  # 5 minutes
    )


@pytest.fixture
def sample_input_metadata() -> Dict[str, Any]:
    """Create sample input metadata."""
    return {
        "filename": "resume.pdf",
        "file_size": 1024000,
        "file_type": "application/pdf",
        "priority": "high",
        "source": "upload_api",
        "user_id": str(uuid4())
    }


@pytest.fixture
def sample_output_data() -> Dict[str, Any]:
    """Create sample output data."""
    return {
        "extracted_text": "John Doe\nSoftware Engineer\n...",
        "page_count": 3,
        "quality_metrics": {
            "confidence_level": 0.92,
            "completeness_score": 0.88,
            "readability_score": 0.90,
            "extraction_accuracy": 0.87
        },
        "metadata": {
            "parser_version": "2.1.0",
            "model_version": "gpt-4"
        }
    }


@pytest.fixture
def sample_entity(
    sample_processing_id,
    sample_tenant_id,
    sample_document_id,
    sample_quality_metrics,
    sample_processing_error,
    sample_processing_timing,
    sample_input_metadata,
    sample_output_data
) -> DocumentProcessing:
    """Create a complete DocumentProcessing entity."""
    return DocumentProcessing(
        id=sample_processing_id,
        tenant_id=sample_tenant_id,
        document_id=sample_document_id,
        processing_type=ProcessingType.AI_ANALYSIS,
        status=ProcessingStatus.COMPLETED,
        input_metadata=sample_input_metadata,
        output_data=sample_output_data,
        quality_metrics=sample_quality_metrics,
        error=None,  # No error for completed processing
        timing=sample_processing_timing,
        created_at=datetime(2025, 1, 15, 9, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2025, 1, 15, 10, 5, 0, tzinfo=timezone.utc)
    )


@pytest.fixture
def sample_table(
    sample_processing_id,
    sample_tenant_id,
    sample_document_id,
    sample_input_metadata,
    sample_output_data
) -> DocumentProcessingTable:
    """Create a sample DocumentProcessingTable."""
    return DocumentProcessingTable(
        id=sample_processing_id,
        tenant_id=sample_tenant_id,
        document_id=sample_document_id,
        processing_type="ai_analysis",
        status="completed",
        input_metadata=sample_input_metadata,
        output_data=sample_output_data,
        quality_score=0.85,
        error_details=None,
        started_at=datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        completed_at=datetime(2025, 1, 15, 10, 5, 0, tzinfo=timezone.utc),
        processing_duration_ms=300000,
        created_at=datetime(2025, 1, 15, 9, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2025, 1, 15, 10, 5, 0, tzinfo=timezone.utc)
    )


@pytest.fixture
def failed_entity(
    sample_processing_id,
    sample_tenant_id,
    sample_document_id,
    sample_processing_error
) -> DocumentProcessing:
    """Create a failed DocumentProcessing entity."""
    timing = ProcessingTiming(
        started_at=datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        completed_at=datetime(2025, 1, 15, 10, 2, 0, tzinfo=timezone.utc),
        processing_duration_ms=120000
    )

    return DocumentProcessing(
        id=sample_processing_id,
        tenant_id=sample_tenant_id,
        document_id=sample_document_id,
        processing_type=ProcessingType.OCR,
        status=ProcessingStatus.FAILED,
        input_metadata={"filename": "corrupted.pdf"},
        output_data=None,
        quality_metrics=QualityMetrics(),
        error=sample_processing_error,
        timing=timing,
        created_at=datetime(2025, 1, 15, 9, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2025, 1, 15, 10, 2, 0, tzinfo=timezone.utc)
    )


# ========================================================================
# A. Basic Conversions
# ========================================================================

class TestBasicConversions:
    """Test basic DocumentProcessingMapper conversions between domain and table models."""

    def test_to_domain_basic(self, sample_table: DocumentProcessingTable):
        """Test basic conversion from DocumentProcessingTable to DocumentProcessing domain entity."""
        # Act
        entity = DocumentProcessingMapper.to_domain(sample_table)

        # Assert
        assert isinstance(entity, DocumentProcessing)
        assert entity.id == sample_table.id
        assert entity.tenant_id == sample_table.tenant_id
        assert entity.document_id == sample_table.document_id
        assert entity.processing_type == ProcessingType.AI_ANALYSIS
        assert entity.status == ProcessingStatus.COMPLETED

    def test_to_table_basic(self, sample_entity: DocumentProcessing):
        """Test basic conversion from DocumentProcessing domain entity to DocumentProcessingTable."""
        # Act
        table = DocumentProcessingMapper.to_table(sample_entity)

        # Assert
        assert isinstance(table, DocumentProcessingTable)
        assert table.id == sample_entity.id
        assert table.tenant_id == sample_entity.tenant_id
        assert table.document_id == sample_entity.document_id
        assert table.processing_type == sample_entity.processing_type.value
        assert table.status == sample_entity.status.value

    def test_roundtrip_conversion(self, sample_entity: DocumentProcessing):
        """Test that DocumentProcessing -> Table -> DocumentProcessing preserves all data."""
        # Act
        table = DocumentProcessingMapper.to_table(sample_entity)
        result = DocumentProcessingMapper.to_domain(table)

        # Assert - Core identifiers
        assert result.id == sample_entity.id
        assert result.tenant_id == sample_entity.tenant_id
        assert result.document_id == sample_entity.document_id
        assert result.processing_type == sample_entity.processing_type
        assert result.status == sample_entity.status

        # Assert - Metadata
        assert result.input_metadata == sample_entity.input_metadata
        assert result.output_data == sample_entity.output_data

        # Assert - Quality metrics
        assert result.quality_metrics.quality_score == sample_entity.quality_metrics.quality_score

        # Assert - Timing
        assert result.timing.started_at == sample_entity.timing.started_at
        assert result.timing.completed_at == sample_entity.timing.completed_at
        assert result.timing.processing_duration_ms == sample_entity.timing.processing_duration_ms

        # Assert - Timestamps
        assert result.created_at == sample_entity.created_at
        assert result.updated_at == sample_entity.updated_at


# ========================================================================
# B. Value Object Conversions
# ========================================================================

class TestValueObjectConversions:
    """Test conversion of value objects (IDs, enums)."""

    def test_processing_type_enum_conversion(self):
        """Test all ProcessingType enum values convert correctly."""
        for processing_type in ProcessingType:
            # Arrange
            table = DocumentProcessingTable(
                id=uuid4(),
                tenant_id=uuid4(),
                document_id=uuid4(),
                processing_type=processing_type.value,
                status="pending",
                started_at=datetime.now(timezone.utc),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )

            # Act
            entity = DocumentProcessingMapper.to_domain(table)
            result_table = DocumentProcessingMapper.to_table(entity)

            # Assert
            assert entity.processing_type == processing_type
            assert result_table.processing_type == processing_type.value

    def test_processing_status_enum_conversion(self):
        """Test all ProcessingStatus enum values convert correctly."""
        for status in ProcessingStatus:
            # Arrange
            table = DocumentProcessingTable(
                id=uuid4(),
                tenant_id=uuid4(),
                document_id=uuid4(),
                processing_type="ai_analysis",
                status=status.value,
                started_at=datetime.now(timezone.utc),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )

            # Act
            entity = DocumentProcessingMapper.to_domain(table)
            result_table = DocumentProcessingMapper.to_table(entity)

            # Assert
            assert entity.status == status
            assert result_table.status == status.value

    def test_uuid_conversions(self):
        """Test UUID value objects convert correctly."""
        # Arrange
        processing_id = uuid4()
        tenant_id = uuid4()
        document_id = uuid4()

        table = DocumentProcessingTable(
            id=processing_id,
            tenant_id=tenant_id,
            document_id=document_id,
            processing_type="embedding_generation",
            status="processing",
            started_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )

        # Act
        entity = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert entity.id == processing_id
        assert entity.tenant_id == tenant_id
        assert entity.document_id == document_id


# ========================================================================
# C. Complex Object Conversions
# ========================================================================

class TestComplexObjectConversions:
    """Test conversion of complex nested structures."""

    def test_quality_metrics_conversion_full(self, sample_quality_metrics: QualityMetrics):
        """Test QualityMetrics conversion with all fields populated via output_data."""
        # Arrange - Quality metrics are stored in output_data for full preservation
        output_data = {
            "result": "success",
            "quality_metrics": {
                "confidence_level": sample_quality_metrics.confidence_level,
                "completeness_score": sample_quality_metrics.completeness_score,
                "readability_score": sample_quality_metrics.readability_score,
                "extraction_accuracy": sample_quality_metrics.extraction_accuracy
            }
        }

        entity = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.QUALITY_CHECK,
            status=ProcessingStatus.COMPLETED,
            quality_metrics=sample_quality_metrics,
            output_data=output_data,
            timing=ProcessingTiming(started_at=datetime.now(timezone.utc))
        )

        # Act
        table = DocumentProcessingMapper.to_table(entity)
        result = DocumentProcessingMapper.to_domain(table)

        # Assert - Main quality_score from table field
        assert result.quality_metrics.quality_score == sample_quality_metrics.quality_score
        # Assert - Additional metrics from output_data
        assert result.quality_metrics.confidence_level == sample_quality_metrics.confidence_level
        assert result.quality_metrics.completeness_score == sample_quality_metrics.completeness_score
        assert result.quality_metrics.readability_score == sample_quality_metrics.readability_score
        assert result.quality_metrics.extraction_accuracy == sample_quality_metrics.extraction_accuracy

    def test_quality_metrics_conversion_minimal(self):
        """Test QualityMetrics conversion with minimal fields."""
        # Arrange
        entity = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.COMPLETED,
            quality_metrics=QualityMetrics(quality_score=0.75),
            timing=ProcessingTiming(started_at=datetime.now(timezone.utc))
        )

        # Act
        table = DocumentProcessingMapper.to_table(entity)
        result = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert result.quality_metrics.quality_score == 0.75
        assert result.quality_metrics.confidence_level is None
        assert result.quality_metrics.completeness_score is None
        assert result.quality_metrics.readability_score is None
        assert result.quality_metrics.extraction_accuracy is None

    def test_processing_error_conversion_full(self, sample_processing_error: ProcessingError):
        """Test ProcessingError conversion with all fields."""
        # Arrange
        entity = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.OCR,
            status=ProcessingStatus.FAILED,
            error=sample_processing_error,
            timing=ProcessingTiming(started_at=datetime.now(timezone.utc))
        )

        # Act
        table = DocumentProcessingMapper.to_table(entity)
        result = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert result.error is not None
        assert result.error.error_code == sample_processing_error.error_code
        assert result.error.error_message == sample_processing_error.error_message
        assert result.error.error_type == sample_processing_error.error_type
        assert result.error.stack_trace == sample_processing_error.stack_trace
        assert result.error.retry_count == sample_processing_error.retry_count
        assert result.error.recoverable == sample_processing_error.recoverable

    def test_processing_error_timestamp_as_string(self):
        """Test ProcessingError conversion with timestamp as ISO string."""
        # Arrange
        error_details = {
            "error_code": "TIMEOUT",
            "error_message": "Processing timed out",
            "timestamp": "2025-01-15T10:30:00+00:00",
            "retry_count": 2,
            "recoverable": True
        }

        table = DocumentProcessingTable(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type="ai_analysis",
            status="failed",
            error_details=error_details,
            started_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )

        # Act
        entity = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert entity.error is not None
        assert entity.error.error_code == "TIMEOUT"
        assert entity.error.timestamp.year == 2025
        assert entity.error.timestamp.month == 1
        assert entity.error.timestamp.day == 15

    def test_processing_error_timestamp_as_datetime(self):
        """Test ProcessingError conversion with timestamp as datetime object."""
        # Arrange
        timestamp = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        error_details = {
            "error_code": "VALIDATION_ERROR",
            "error_message": "Invalid input format",
            "timestamp": timestamp,
            "retry_count": 0,
            "recoverable": False
        }

        table = DocumentProcessingTable(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type="text_extraction",
            status="failed",
            error_details=error_details,
            started_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )

        # Act
        entity = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert entity.error is not None
        assert entity.error.error_code == "VALIDATION_ERROR"
        assert entity.error.timestamp == timestamp

    def test_processing_timing_conversion(self, sample_processing_timing: ProcessingTiming):
        """Test ProcessingTiming conversion."""
        # Arrange
        entity = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.EMBEDDING_GENERATION,
            status=ProcessingStatus.COMPLETED,
            timing=sample_processing_timing
        )

        # Act
        table = DocumentProcessingMapper.to_table(entity)
        result = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert result.timing.started_at == sample_processing_timing.started_at
        assert result.timing.completed_at == sample_processing_timing.completed_at
        assert result.timing.processing_duration_ms == sample_processing_timing.processing_duration_ms


# ========================================================================
# D. JSONB Conversions
# ========================================================================

class TestJSONBConversions:
    """Test JSONB field serialization and deserialization."""

    def test_input_metadata_serialization(self, sample_input_metadata: Dict[str, Any]):
        """Test input_metadata JSONB serialization."""
        # Arrange
        entity = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.PENDING,
            input_metadata=sample_input_metadata,
            timing=ProcessingTiming(started_at=datetime.now(timezone.utc))
        )

        # Act
        table = DocumentProcessingMapper.to_table(entity)
        result = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert result.input_metadata == sample_input_metadata
        assert result.input_metadata["filename"] == "resume.pdf"
        assert result.input_metadata["file_size"] == 1024000

    def test_output_data_serialization(self, sample_output_data: Dict[str, Any]):
        """Test output_data JSONB serialization."""
        # Arrange
        entity = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.TEXT_EXTRACTION,
            status=ProcessingStatus.COMPLETED,
            output_data=sample_output_data,
            timing=ProcessingTiming(started_at=datetime.now(timezone.utc))
        )

        # Act
        table = DocumentProcessingMapper.to_table(entity)
        result = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert result.output_data == sample_output_data
        assert result.output_data["page_count"] == 3
        assert "quality_metrics" in result.output_data

    def test_error_details_serialization(self, sample_processing_error: ProcessingError):
        """Test error_details JSONB serialization."""
        # Arrange
        entity = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.OCR,
            status=ProcessingStatus.FAILED,
            error=sample_processing_error,
            timing=ProcessingTiming(started_at=datetime.now(timezone.utc))
        )

        # Act
        table = DocumentProcessingMapper.to_table(entity)

        # Assert
        assert table.error_details is not None
        assert table.error_details["error_code"] == "OCR_FAILURE"
        assert table.error_details["error_message"] == "Failed to extract text from page 3"
        assert table.error_details["retry_count"] == 1
        assert "timestamp" in table.error_details

    def test_large_jsonb_data(self):
        """Test handling of large JSONB data (1MB+)."""
        # Arrange
        large_text = "x" * 1_000_000  # 1MB of text
        large_output = {
            "extracted_text": large_text,
            "metadata": {"size": len(large_text)}
        }

        entity = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.TEXT_EXTRACTION,
            status=ProcessingStatus.COMPLETED,
            output_data=large_output,
            timing=ProcessingTiming(started_at=datetime.now(timezone.utc))
        )

        # Act
        table = DocumentProcessingMapper.to_table(entity)
        result = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert result.output_data["extracted_text"] == large_text
        assert len(result.output_data["extracted_text"]) == 1_000_000

    def test_jsonb_with_special_characters(self):
        """Test JSONB handling with special characters."""
        # Arrange
        special_data = {
            "text": "Line 1\nLine 2\tTab\r\nWindows",
            "unicode": "Hello 世界 🌍",
            "quotes": 'He said "hello"',
            "paths": "C:\\Users\\test\\file.pdf"
        }

        entity = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.METADATA_EXTRACTION,
            status=ProcessingStatus.COMPLETED,
            output_data=special_data,
            timing=ProcessingTiming(started_at=datetime.now(timezone.utc))
        )

        # Act
        table = DocumentProcessingMapper.to_table(entity)
        result = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert result.output_data == special_data
        assert "世界" in result.output_data["unicode"]
        assert "🌍" in result.output_data["unicode"]


# ========================================================================
# E. Optional/Null Field Handling
# ========================================================================

class TestOptionalNullHandling:
    """Test handling of optional and null fields."""

    def test_minimal_required_fields_only(self):
        """Test entity with only required fields."""
        # Arrange
        entity = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.PENDING
        )

        # Act
        table = DocumentProcessingMapper.to_table(entity)
        result = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert result.id == entity.id
        assert result.tenant_id == entity.tenant_id
        assert result.document_id == entity.document_id
        assert result.input_metadata is None
        assert result.output_data is None
        assert result.error is None

    def test_null_quality_score(self):
        """Test handling of null quality_score."""
        # Arrange
        table = DocumentProcessingTable(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type="ai_analysis",
            status="processing",
            quality_score=None,
            started_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )

        # Act
        entity = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert entity.quality_metrics.quality_score is None
        assert entity.quality_metrics.has_metrics() is False

    def test_null_error_details(self):
        """Test handling of null error_details."""
        # Arrange
        table = DocumentProcessingTable(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type="ocr",
            status="completed",
            error_details=None,
            started_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )

        # Act
        entity = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert entity.error is None

    def test_null_timestamps(self):
        """Test handling of null completed_at timestamp."""
        # Arrange
        table = DocumentProcessingTable(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type="embedding_generation",
            status="processing",
            started_at=datetime.now(timezone.utc),
            completed_at=None,
            processing_duration_ms=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )

        # Act
        entity = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert entity.timing.completed_at is None
        assert entity.timing.processing_duration_ms is None
        assert entity.timing.is_completed() is False

    def test_all_optional_fields_populated(self, sample_entity: DocumentProcessing):
        """Test entity with all optional fields populated."""
        # Act
        table = DocumentProcessingMapper.to_table(sample_entity)
        result = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert result.input_metadata is not None
        assert result.output_data is not None
        assert result.quality_metrics.quality_score is not None
        assert result.timing.completed_at is not None
        assert result.timing.processing_duration_ms is not None


# ========================================================================
# F. Update Operations
# ========================================================================

class TestUpdateOperations:
    """Test update_table_from_domain functionality."""

    def test_update_table_basic_fields(self):
        """Test updating basic fields on existing table."""
        # Arrange
        original_table = DocumentProcessingTable(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type="ai_analysis",
            status="pending",
            started_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )

        updated_entity = DocumentProcessing(
            id=original_table.id,
            tenant_id=original_table.tenant_id,
            document_id=uuid4(),  # Changed
            processing_type=ProcessingType.OCR,  # Changed
            status=ProcessingStatus.COMPLETED,  # Changed
            timing=ProcessingTiming(started_at=datetime.now(timezone.utc))
        )

        # Act
        result = DocumentProcessingMapper.update_table_from_domain(original_table, updated_entity)

        # Assert
        assert result.document_id == updated_entity.document_id
        assert result.processing_type == updated_entity.processing_type.value
        assert result.status == updated_entity.status.value

    def test_update_table_preserves_immutable_fields(self):
        """Test that update preserves immutable fields (id, tenant_id, created_at)."""
        # Arrange
        original_id = uuid4()
        original_tenant_id = uuid4()
        original_created_at = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        original_table = DocumentProcessingTable(
            id=original_id,
            tenant_id=original_tenant_id,
            document_id=uuid4(),
            processing_type="ai_analysis",
            status="pending",
            started_at=datetime.now(timezone.utc),
            created_at=original_created_at,
            updated_at=datetime.now(timezone.utc)
        )

        # Entity with same IDs
        updated_entity = DocumentProcessing(
            id=original_id,
            tenant_id=original_tenant_id,
            document_id=uuid4(),
            processing_type=ProcessingType.EMBEDDING_GENERATION,
            status=ProcessingStatus.COMPLETED,
            timing=ProcessingTiming(started_at=datetime.now(timezone.utc)),
            created_at=original_created_at,
            updated_at=datetime.now(timezone.utc)
        )

        # Act
        result = DocumentProcessingMapper.update_table_from_domain(original_table, updated_entity)

        # Assert - Immutable fields preserved
        assert result.id == original_id
        assert result.tenant_id == original_tenant_id

    def test_update_table_quality_metrics(self):
        """Test updating quality metrics."""
        # Arrange
        original_table = DocumentProcessingTable(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type="quality_check",
            status="processing",
            quality_score=None,
            started_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )

        updated_entity = DocumentProcessing(
            id=original_table.id,
            tenant_id=original_table.tenant_id,
            document_id=original_table.document_id,
            processing_type=ProcessingType.QUALITY_CHECK,
            status=ProcessingStatus.COMPLETED,
            quality_metrics=QualityMetrics(quality_score=0.89),
            timing=ProcessingTiming(started_at=datetime.now(timezone.utc))
        )

        # Act
        result = DocumentProcessingMapper.update_table_from_domain(original_table, updated_entity)

        # Assert
        assert result.quality_score == 0.89

    def test_update_table_error_details(self, sample_processing_error: ProcessingError):
        """Test updating error details."""
        # Arrange
        original_table = DocumentProcessingTable(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type="ocr",
            status="processing",
            error_details=None,
            started_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )

        updated_entity = DocumentProcessing(
            id=original_table.id,
            tenant_id=original_table.tenant_id,
            document_id=original_table.document_id,
            processing_type=ProcessingType.OCR,
            status=ProcessingStatus.FAILED,
            error=sample_processing_error,
            timing=ProcessingTiming(started_at=datetime.now(timezone.utc))
        )

        # Act
        result = DocumentProcessingMapper.update_table_from_domain(original_table, updated_entity)

        # Assert
        assert result.error_details is not None
        assert result.error_details["error_code"] == "OCR_FAILURE"

    def test_update_table_timestamps(self):
        """Test updating timing fields."""
        # Arrange
        original_table = DocumentProcessingTable(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type="ai_analysis",
            status="processing",
            started_at=datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            completed_at=None,
            processing_duration_ms=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )

        completed_at = datetime(2025, 1, 15, 10, 5, 0, tzinfo=timezone.utc)
        updated_entity = DocumentProcessing(
            id=original_table.id,
            tenant_id=original_table.tenant_id,
            document_id=original_table.document_id,
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.COMPLETED,
            timing=ProcessingTiming(
                started_at=original_table.started_at,
                completed_at=completed_at,
                processing_duration_ms=300000
            ),
            updated_at=completed_at
        )

        # Act
        result = DocumentProcessingMapper.update_table_from_domain(original_table, updated_entity)

        # Assert
        assert result.completed_at == completed_at
        assert result.processing_duration_ms == 300000
        assert result.updated_at == completed_at


# ========================================================================
# G. Business Logic Methods
# ========================================================================

class TestBusinessLogicIntegration:
    """Test that business logic methods work correctly with mapper."""

    def test_entity_start_processing_flow(self):
        """Test start_processing() business logic through mapper."""
        # Arrange
        entity = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.PENDING
        )

        # Act
        entity.start_processing()
        table = DocumentProcessingMapper.to_table(entity)
        result = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert result.status == ProcessingStatus.PROCESSING
        assert result.timing.started_at is not None

    def test_entity_complete_processing_flow(self):
        """Test complete_processing() business logic through mapper."""
        # Arrange
        entity = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.EMBEDDING_GENERATION,
            status=ProcessingStatus.PENDING
        )

        # Act
        entity.start_processing()
        output_data = {"embeddings": [0.1, 0.2, 0.3]}
        entity.complete_processing(output_data, quality_score=0.95)

        table = DocumentProcessingMapper.to_table(entity)
        result = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert result.status == ProcessingStatus.COMPLETED
        assert result.output_data == output_data
        assert result.quality_metrics.quality_score == 0.95
        assert result.timing.completed_at is not None

    def test_entity_fail_processing_flow(self):
        """Test fail_processing() business logic through mapper."""
        # Arrange
        entity = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.OCR,
            status=ProcessingStatus.PENDING
        )

        # Act
        entity.start_processing()
        entity.fail_processing(
            error_code="OCR_ERROR",
            error_message="Failed to process page",
            recoverable=True
        )

        table = DocumentProcessingMapper.to_table(entity)
        result = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert result.status == ProcessingStatus.FAILED
        assert result.error is not None
        assert result.error.error_code == "OCR_ERROR"
        assert result.error.recoverable is True

    def test_entity_retry_processing_flow(self):
        """Test retry_processing() business logic through mapper."""
        # Arrange
        entity = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.TEXT_EXTRACTION,
            status=ProcessingStatus.PENDING
        )

        # Act - Initial failure
        entity.start_processing()
        entity.fail_processing(
            error_code="TIMEOUT",
            error_message="Processing timed out",
            recoverable=True
        )

        # Act - Retry
        entity.retry_processing()

        table = DocumentProcessingMapper.to_table(entity)
        result = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert result.status == ProcessingStatus.PENDING
        assert result.error.retry_count == 1

    def test_entity_cancel_processing_flow(self):
        """Test cancel_processing() business logic through mapper."""
        # Arrange
        entity = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.METADATA_EXTRACTION,
            status=ProcessingStatus.PENDING
        )

        # Act
        entity.start_processing()
        entity.cancel_processing(reason="User requested cancellation")

        table = DocumentProcessingMapper.to_table(entity)
        result = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert result.status == ProcessingStatus.CANCELLED
        assert result.output_data is not None
        assert result.output_data["cancellation_reason"] == "User requested cancellation"

    def test_entity_set_quality_metrics_flow(self):
        """Test set_quality_metrics() business logic through mapper.

        Note: Only quality_score is preserved automatically. Additional metrics
        must be stored in output_data.quality_metrics to survive roundtrip.
        """
        # Arrange
        entity = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.QUALITY_CHECK,
            status=ProcessingStatus.COMPLETED
        )

        # Act
        entity.set_quality_metrics(
            quality_score=0.92,
            confidence_level=0.88,
            completeness_score=0.95
        )

        table = DocumentProcessingMapper.to_table(entity)
        result = DocumentProcessingMapper.to_domain(table)

        # Assert - Only quality_score is preserved automatically
        assert result.quality_metrics.quality_score == 0.92
        # Additional metrics would need to be in output_data to be preserved
        # This is by design - only the main quality_score has a dedicated column

    def test_entity_validation_through_mapper(self):
        """Test that validation works correctly through mapper."""
        # Arrange - Invalid entity (completed without output)
        entity = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.COMPLETED,
            output_data=None  # Invalid for completed status
        )

        # Act
        validation_errors = entity.validate()

        # Assert
        assert len(validation_errors) > 0
        assert any("output_data" in error for error in validation_errors)


# ========================================================================
# H. Edge Cases
# ========================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_long_error_message(self):
        """Test handling of very long error messages."""
        # Arrange
        long_message = "Error: " + ("x" * 10_000)
        error = ProcessingError(
            error_code="LONG_ERROR",
            error_message=long_message,
            recoverable=False
        )

        entity = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.FAILED,
            error=error,
            timing=ProcessingTiming(started_at=datetime.now(timezone.utc))
        )

        # Act
        table = DocumentProcessingMapper.to_table(entity)
        result = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert len(result.error.error_message) > 10_000
        assert result.error.error_message == long_message

    def test_quality_score_boundary_values(self):
        """Test quality score at boundary values (0.0, 1.0)."""
        for score in [0.0, 0.5, 1.0]:
            # Arrange
            entity = DocumentProcessing(
                id=uuid4(),
                tenant_id=uuid4(),
                document_id=uuid4(),
                processing_type=ProcessingType.QUALITY_CHECK,
                status=ProcessingStatus.COMPLETED,
                quality_metrics=QualityMetrics(quality_score=score),
                timing=ProcessingTiming(started_at=datetime.now(timezone.utc))
            )

            # Act
            table = DocumentProcessingMapper.to_table(entity)
            result = DocumentProcessingMapper.to_domain(table)

            # Assert
            assert result.quality_metrics.quality_score == score

    def test_processing_duration_edge_cases(self):
        """Test processing duration edge cases (0ms, very long)."""
        for duration_ms in [0, 1, 86400000]:  # 0ms, 1ms, 24 hours
            # Arrange
            entity = DocumentProcessing(
                id=uuid4(),
                tenant_id=uuid4(),
                document_id=uuid4(),
                processing_type=ProcessingType.TEXT_EXTRACTION,
                status=ProcessingStatus.COMPLETED,
                timing=ProcessingTiming(
                    started_at=datetime.now(timezone.utc),
                    processing_duration_ms=duration_ms
                )
            )

            # Act
            table = DocumentProcessingMapper.to_table(entity)
            result = DocumentProcessingMapper.to_domain(table)

            # Assert
            assert result.timing.processing_duration_ms == duration_ms

    def test_empty_jsonb_fields(self):
        """Test handling of empty JSONB objects."""
        # Arrange
        entity = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.METADATA_EXTRACTION,
            status=ProcessingStatus.COMPLETED,
            input_metadata={},
            output_data={},
            timing=ProcessingTiming(started_at=datetime.now(timezone.utc))
        )

        # Act
        table = DocumentProcessingMapper.to_table(entity)
        result = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert result.input_metadata == {}
        assert result.output_data == {}

    def test_max_retry_count(self):
        """Test handling of maximum retry count."""
        # Arrange
        error = ProcessingError(
            error_code="MAX_RETRIES",
            error_message="Exceeded retry limit",
            retry_count=10,  # High retry count
            recoverable=True
        )

        entity = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.OCR,
            status=ProcessingStatus.FAILED,
            error=error,
            timing=ProcessingTiming(started_at=datetime.now(timezone.utc))
        )

        # Act
        table = DocumentProcessingMapper.to_table(entity)
        result = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert result.error.retry_count == 10
        assert result.error.can_retry(max_retries=3) is False


# ========================================================================
# I. Property-Based Testing
# ========================================================================

class TestPropertyBasedRoundtrip:
    """Property-based tests using Hypothesis for roundtrip conversions."""

    @given(
        processing_id=st.uuids(),
        tenant_id=st.uuids(),
        document_id=st.uuids(),
        processing_type=st.sampled_from(list(ProcessingType)),
        status=st.sampled_from(list(ProcessingStatus)),
        quality_score=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0))
    )
    def test_roundtrip_property(
        self,
        processing_id,
        tenant_id,
        document_id,
        processing_type,
        status,
        quality_score
    ):
        """Property test: any valid entity should survive roundtrip conversion."""
        # Arrange
        entity = DocumentProcessing(
            id=processing_id,
            tenant_id=tenant_id,
            document_id=document_id,
            processing_type=processing_type,
            status=status,
            quality_metrics=QualityMetrics(quality_score=quality_score),
            timing=ProcessingTiming(started_at=datetime.now(timezone.utc))
        )

        # Act
        table = DocumentProcessingMapper.to_table(entity)
        result = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert result.id == entity.id
        assert result.tenant_id == entity.tenant_id
        assert result.document_id == entity.document_id
        assert result.processing_type == entity.processing_type
        assert result.status == entity.status
        assert result.quality_metrics.quality_score == entity.quality_metrics.quality_score

    @given(processing_type=st.sampled_from(list(ProcessingType)))
    def test_processing_type_bijection(self, processing_type):
        """Property test: ProcessingType enum bijection."""
        # Arrange
        entity = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=processing_type,
            status=ProcessingStatus.PENDING
        )

        # Act
        table = DocumentProcessingMapper.to_table(entity)
        result = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert result.processing_type == processing_type
        assert table.processing_type == processing_type.value

    @given(status=st.sampled_from(list(ProcessingStatus)))
    def test_processing_status_bijection(self, status):
        """Property test: ProcessingStatus enum bijection."""
        # Arrange
        entity = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=status
        )

        # Act
        table = DocumentProcessingMapper.to_table(entity)
        result = DocumentProcessingMapper.to_domain(table)

        # Assert
        assert result.status == status
        assert table.status == status.value


# ========================================================================
# J. Integration Tests
# ========================================================================

class TestMapperIntegration:
    """Integration tests combining multiple mapper operations."""

    def test_complete_workflow_through_mapper(self):
        """Test complete processing workflow through mapper conversions."""
        # Arrange - Create pending processing
        entity = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.AI_ANALYSIS,
            status=ProcessingStatus.PENDING,
            input_metadata={"filename": "test.pdf"}
        )

        # Act - Start processing
        entity.start_processing()
        table1 = DocumentProcessingMapper.to_table(entity)
        entity1 = DocumentProcessingMapper.to_domain(table1)

        # Act - Complete processing
        entity1.complete_processing(
            output_data={"result": "success"},
            quality_score=0.92
        )
        table2 = DocumentProcessingMapper.to_table(entity1)
        entity2 = DocumentProcessingMapper.to_domain(table2)

        # Assert final state
        assert entity2.status == ProcessingStatus.COMPLETED
        assert entity2.output_data["result"] == "success"
        assert entity2.quality_metrics.quality_score == 0.92
        assert entity2.timing.completed_at is not None

    def test_failed_retry_workflow_through_mapper(self):
        """Test failure and retry workflow through mapper."""
        # Arrange
        entity = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.OCR,
            status=ProcessingStatus.PENDING
        )

        # Act - Start and fail
        entity.start_processing()
        entity.fail_processing(
            error_code="TIMEOUT",
            error_message="Processing timed out",
            recoverable=True
        )

        table1 = DocumentProcessingMapper.to_table(entity)
        entity1 = DocumentProcessingMapper.to_domain(table1)

        # Act - Retry
        entity1.retry_processing()
        table2 = DocumentProcessingMapper.to_table(entity1)
        entity2 = DocumentProcessingMapper.to_domain(table2)

        # Assert
        assert entity2.status == ProcessingStatus.PENDING
        assert entity2.error.retry_count == 1
        # After retry, can_be_retried() should still be True if under max retries
        # This checks if can_retry() with default max_retries=3
        assert entity2.error.can_retry(max_retries=3) is True

    def test_multiple_status_transitions(self):
        """Test multiple status transitions preserve data integrity."""
        # Arrange
        entity = DocumentProcessing(
            id=uuid4(),
            tenant_id=uuid4(),
            document_id=uuid4(),
            processing_type=ProcessingType.EMBEDDING_GENERATION,
            status=ProcessingStatus.PENDING
        )

        # Act - PENDING -> PROCESSING
        entity.start_processing()
        table1 = DocumentProcessingMapper.to_table(entity)
        entity = DocumentProcessingMapper.to_domain(table1)

        # Act - PROCESSING -> CANCELLED
        entity.cancel_processing(reason="Test cancellation")
        table2 = DocumentProcessingMapper.to_table(entity)
        result = DocumentProcessingMapper.to_domain(table2)

        # Assert
        assert result.status == ProcessingStatus.CANCELLED
        assert result.output_data["cancellation_reason"] == "Test cancellation"