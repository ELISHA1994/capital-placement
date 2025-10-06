"""Pydantic models for upload responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models.profile import ProcessingStatus


class UploadResponse(BaseModel):
    """Response model for file upload operations."""

    upload_id: str = Field(..., description="Unique upload identifier")
    profile_id: str = Field(..., description="Generated profile identifier")
    filename: str = Field(..., description="Original filename")
    status: ProcessingStatus = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")
    estimated_processing_time_seconds: Optional[int] = Field(
        None,
        description="Estimated processing time",
    )
    webhook_url: Optional[str] = Field(
        None,
        description="Webhook URL for status updates",
    )


class BatchUploadResponse(BaseModel):
    """Response model for batch upload operations."""

    batch_id: str = Field(..., description="Unique batch identifier")
    total_files: int = Field(..., description="Total files in batch")
    accepted_files: int = Field(..., description="Files accepted for processing")
    rejected_files: int = Field(..., description="Files rejected")
    uploads: List[UploadResponse] = Field(..., description="Individual upload responses")
    rejected_reasons: Dict[str, str] = Field(
        default_factory=dict,
        description="Rejection reasons",
    )


class ProcessingStatusResponse(BaseModel):
    """Response model for processing status queries."""

    upload_id: str = Field(..., description="Upload identifier")
    profile_id: str = Field(..., description="Profile identifier")
    status: ProcessingStatus = Field(..., description="Current processing status")
    progress_percentage: int = Field(..., ge=0, le=100, description="Processing progress")
    processing_duration_seconds: Optional[float] = Field(
        None,
        description="Processing duration",
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if failed",
    )
    quality_score: Optional[float] = Field(
        None,
        description="Extraction quality score",
    )
    extracted_data_preview: Optional[Dict[str, Any]] = Field(
        None,
        description="Preview of extracted data",
    )


class UploadHistoryItem(BaseModel):
    """Individual upload history record for document listing."""

    upload_id: str = Field(..., description="Upload identifier")
    profile_id: str = Field(..., description="Profile identifier")
    filename: str = Field(..., description="Original filename")
    status: ProcessingStatus = Field(..., description="Processing status")
    created_at: datetime = Field(..., description="Upload timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    processing_duration_seconds: Optional[float] = Field(
        None,
        description="Processing duration in seconds",
    )
    quality_score: Optional[float] = Field(
        None,
        description="Document quality score",
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if processing failed",
    )
    file_size_bytes: Optional[int] = Field(
        None,
        description="File size in bytes",
    )


__all__ = [
    "UploadResponse",
    "BatchUploadResponse",
    "ProcessingStatusResponse",
    "UploadHistoryItem",
]

