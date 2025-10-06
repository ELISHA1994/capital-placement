"""
Document processing tracking SQLModel for monitoring and analytics.

This module provides comprehensive tracking of document processing operations
following the hexagonal architecture patterns with proper tenant isolation.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy import Column, String, DateTime, Integer, Float, Index
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID, JSONB
from sqlalchemy.sql import func
from sqlmodel import Field, SQLModel, Relationship

from .base import TenantModel, create_tenant_id_column


class DocumentProcessingTable(TenantModel, table=True):
    """
    Document processing tracking table for monitoring and analytics.

    This table stores comprehensive tracking data for all document processing
    operations, designed for monitoring, debugging, and analytics purposes.
    """
    __tablename__ = "document_processing"

    # Tenant isolation field (required for SQLModel to create foreign key constraints)
    tenant_id: UUID = Field(
        sa_column=create_tenant_id_column(),
        description="Tenant identifier for multi-tenant isolation"
    )

    # Relationships
    tenant: Optional["TenantTable"] = Relationship(back_populates="document_processing")

    # Document reference
    document_id: UUID = Field(
        sa_column=Column(PostgreSQLUUID(as_uuid=True), nullable=False),
        description="Reference to the document being processed"
    )

    # Processing details
    processing_type: str = Field(
        sa_column=Column(String(100), nullable=False),
        description="Type of processing (e.g., 'ai_analysis', 'ocr', 'embedding_generation')"
    )

    status: str = Field(
        sa_column=Column(String(50), nullable=False, index=True),
        description="Current processing status (e.g., 'processing', 'completed', 'failed')"
    )

    # Input and output data
    input_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSONB, nullable=True),
        description="Input metadata including filename, file size, priority, etc."
    )

    output_data: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSONB, nullable=True),
        description="Output data from processing operation"
    )

    # Quality metrics
    quality_score: Optional[float] = Field(
        default=None,
        sa_column=Column(Float, nullable=True),
        description="Quality score of the processing result (0-1)"
    )

    # Error handling
    error_details: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSONB, nullable=True),
        description="Error details if processing failed"
    )

    # Timing information
    started_at: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            index=True
        ),
        description="Timestamp when processing started"
    )

    completed_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            nullable=True,
            index=True
        ),
        description="Timestamp when processing completed"
    )

    processing_duration_ms: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, nullable=True),
        description="Processing duration in milliseconds"
    )

    # Database indexes for performance
    __table_args__ = (
        Index("ix_document_processing_tenant_status", "tenant_id", "status"),
        Index("ix_document_processing_tenant_type", "tenant_id", "processing_type"),
        Index("ix_document_processing_document_id", "document_id"),
        Index("ix_document_processing_started_at", "started_at"),
        Index("ix_document_processing_completed_at", "completed_at"),
    )

    def __repr__(self):
        return (
            f"DocumentProcessingTable(id={self.id}, tenant_id={self.tenant_id}, "
            f"document_id={self.document_id}, processing_type={self.processing_type}, "
            f"status={self.status})"
        )


# Backward compatibility
DocumentProcessing = DocumentProcessingTable


__all__ = [
    "DocumentProcessingTable",
    "DocumentProcessing",
]
