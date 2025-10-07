"""
Mapper between DocumentProcessing domain entities and DocumentProcessingTable persistence models.

This module provides bidirectional mapping between the pure domain representation
of document processing operations and the SQLModel persistence layer, following
hexagonal architecture principles.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from app.domain.entities.document_processing import (
    DocumentProcessing,
    ProcessingType,
    ProcessingStatus,
    ProcessingTiming,
    QualityMetrics,
    ProcessingError,
)
from app.infrastructure.persistence.models.document_processing_table import DocumentProcessingTable


class DocumentProcessingMapper:
    """
    Maps between DocumentProcessing domain entities and DocumentProcessingTable persistence models.

    This mapper handles the transformation of domain entities to database models and vice versa,
    including proper handling of value objects, enums, and nested structures.
    """

    @staticmethod
    def to_domain(table: DocumentProcessingTable) -> DocumentProcessing:
        """
        Convert DocumentProcessingTable (persistence) to DocumentProcessing (domain).

        Args:
            table: The persistence model to convert

        Returns:
            DocumentProcessing domain entity with all value objects and business logic

        Example:
            >>> table = DocumentProcessingTable(...)
            >>> entity = DocumentProcessingMapper.to_domain(table)
            >>> entity.is_completed()
            True
        """
        # Map quality metrics from flat fields and JSONB
        quality_metrics = DocumentProcessingMapper._map_quality_metrics_to_domain(
            table.quality_score,
            table.output_data
        )

        # Map error details from JSONB
        error = DocumentProcessingMapper._map_error_to_domain(table.error_details)

        # Map timing information
        timing = ProcessingTiming(
            started_at=table.started_at,
            completed_at=table.completed_at,
            processing_duration_ms=table.processing_duration_ms
        )

        # Create domain entity
        return DocumentProcessing(
            id=table.id,
            tenant_id=table.tenant_id,
            document_id=table.document_id,
            processing_type=ProcessingType(table.processing_type),
            status=ProcessingStatus(table.status),
            input_metadata=table.input_metadata,
            output_data=table.output_data,
            quality_metrics=quality_metrics,
            error=error,
            timing=timing,
            created_at=table.created_at,
            updated_at=table.updated_at
        )

    @staticmethod
    def to_table(entity: DocumentProcessing) -> DocumentProcessingTable:
        """
        Convert DocumentProcessing (domain) to DocumentProcessingTable (persistence).

        Args:
            entity: The domain entity to convert

        Returns:
            DocumentProcessingTable persistence model ready for database operations

        Example:
            >>> entity = DocumentProcessing(...)
            >>> table = DocumentProcessingMapper.to_table(entity)
            >>> # Ready to save to database
        """
        # Map error details to JSONB
        error_details = None
        if entity.error:
            error_details = entity.error.to_dict()

        # Create persistence model
        return DocumentProcessingTable(
            id=entity.id,
            tenant_id=entity.tenant_id,
            document_id=entity.document_id,
            processing_type=entity.processing_type.value,
            status=entity.status.value,
            input_metadata=entity.input_metadata,
            output_data=entity.output_data,
            quality_score=entity.quality_metrics.quality_score,
            error_details=error_details,
            started_at=entity.timing.started_at,
            completed_at=entity.timing.completed_at,
            processing_duration_ms=entity.timing.processing_duration_ms,
            created_at=entity.created_at,
            updated_at=entity.updated_at
        )

    @staticmethod
    def update_table_from_domain(
        table: DocumentProcessingTable,
        entity: DocumentProcessing
    ) -> DocumentProcessingTable:
        """
        Update existing DocumentProcessingTable with data from DocumentProcessing domain entity.

        This method is useful for updating existing database records without creating new instances.

        Args:
            table: The existing persistence model to update
            entity: The domain entity containing updated data

        Returns:
            Updated DocumentProcessingTable instance

        Example:
            >>> table = session.get(DocumentProcessingTable, entity_id)
            >>> updated_table = DocumentProcessingMapper.update_table_from_domain(table, entity)
            >>> session.add(updated_table)
            >>> session.commit()
        """
        # Update basic fields
        table.document_id = entity.document_id
        table.processing_type = entity.processing_type.value
        table.status = entity.status.value
        table.input_metadata = entity.input_metadata
        table.output_data = entity.output_data

        # Update quality metrics
        table.quality_score = entity.quality_metrics.quality_score

        # Update error details
        table.error_details = entity.error.to_dict() if entity.error else None

        # Update timing information
        table.started_at = entity.timing.started_at
        table.completed_at = entity.timing.completed_at
        table.processing_duration_ms = entity.timing.processing_duration_ms

        # Update timestamps
        table.updated_at = entity.updated_at

        return table

    @staticmethod
    def _map_quality_metrics_to_domain(
        quality_score: Optional[float],
        output_data: Optional[Dict[str, Any]]
    ) -> QualityMetrics:
        """
        Map quality metrics from table fields to domain QualityMetrics value object.

        Quality metrics can be stored in multiple places:
        - quality_score: Direct field on the table
        - output_data: JSONB field that may contain additional metrics

        Args:
            quality_score: The primary quality score from the table
            output_data: JSONB output data that may contain additional metrics

        Returns:
            QualityMetrics value object with all available metrics
        """
        # Extract additional metrics from output_data if available
        confidence_level = None
        completeness_score = None
        readability_score = None
        extraction_accuracy = None

        if output_data and isinstance(output_data, dict):
            quality_data = output_data.get('quality_metrics', {})
            if isinstance(quality_data, dict):
                confidence_level = quality_data.get('confidence_level')
                completeness_score = quality_data.get('completeness_score')
                readability_score = quality_data.get('readability_score')
                extraction_accuracy = quality_data.get('extraction_accuracy')

        return QualityMetrics(
            quality_score=quality_score,
            confidence_level=confidence_level,
            completeness_score=completeness_score,
            readability_score=readability_score,
            extraction_accuracy=extraction_accuracy
        )

    @staticmethod
    def _map_error_to_domain(
        error_details: Optional[Dict[str, Any]]
    ) -> Optional[ProcessingError]:
        """
        Map error details from JSONB to domain ProcessingError value object.

        Args:
            error_details: JSONB error details from the table

        Returns:
            ProcessingError value object if error details exist, None otherwise
        """
        if not error_details:
            return None

        # Parse timestamp if present
        timestamp = None
        if 'timestamp' in error_details:
            timestamp_str = error_details['timestamp']
            if isinstance(timestamp_str, str):
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            elif isinstance(timestamp_str, datetime):
                timestamp = timestamp_str

        return ProcessingError(
            error_code=error_details.get('error_code', 'UNKNOWN'),
            error_message=error_details.get('error_message', 'Unknown error'),
            error_type=error_details.get('error_type'),
            stack_trace=error_details.get('stack_trace'),
            retry_count=error_details.get('retry_count', 0),
            recoverable=error_details.get('recoverable', False),
            timestamp=timestamp if timestamp else datetime.now()
        )


__all__ = ["DocumentProcessingMapper"]