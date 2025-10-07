"""
Document API Schemas - DTOs for document upload and processing endpoints.

This module contains all request/response models for document management API layer,
separated from domain entities and persistence tables following hexagonal architecture.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, HttpUrl


class DocumentType(str, Enum):
    """Document type enumeration."""
    CV = "cv"
    RESUME = "resume"
    COVER_LETTER = "cover_letter"
    PORTFOLIO = "portfolio"
    TRANSCRIPT = "transcript"
    CERTIFICATE = "certificate"
    OTHER = "other"


class DocumentStatus(str, Enum):
    """Document processing status enumeration."""
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"


class DocumentFormat(str, Enum):
    """Supported document formats."""
    PDF = "pdf"
    DOC = "doc"
    DOCX = "docx"
    TXT = "txt"
    RTF = "rtf"
    HTML = "html"
    ODT = "odt"


class ExtractionMethod(str, Enum):
    """Document content extraction methods."""
    TEXT = "text"
    OCR = "ocr"
    HYBRID = "hybrid"
    AI_PARSING = "ai_parsing"


class DocumentContent(BaseModel):
    """Extracted document content model."""

    raw_text: str = Field(..., description="Raw extracted text")
    formatted_text: Optional[str] = Field(
        None,
        description="Formatted text with structure preserved"
    )
    html_content: Optional[str] = Field(
        None,
        description="HTML representation of document content"
    )

    # Structured extraction
    sections: Dict[str, str] = Field(
        default_factory=dict,
        description="Document sections (e.g., 'summary', 'experience', 'education')"
    )

    # Metadata extraction
    detected_language: Optional[str] = Field(None, description="Detected content language")
    word_count: int = Field(default=0, description="Total word count")
    page_count: int = Field(default=1, description="Number of pages")

    # AI extraction results
    extracted_entities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Named entities extracted from content"
    )
    extracted_skills: List[str] = Field(
        default_factory=list,
        description="Skills extracted from content"
    )
    extracted_experience: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Work experience entries extracted"
    )
    extracted_education: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Education entries extracted"
    )
    extracted_certifications: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Certifications extracted"
    )
    extracted_contact: Dict[str, str] = Field(
        default_factory=dict,
        description="Contact information extracted"
    )

    # Quality metrics
    extraction_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score for extraction quality"
    )
    ocr_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="OCR confidence score if applicable"
    )


class DocumentProcessingError(BaseModel):
    """Document processing error details."""

    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Error message")
    error_details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )
    occurred_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )
    retry_count: int = Field(default=0, description="Number of retry attempts")
    is_recoverable: bool = Field(default=True, description="Whether error is recoverable")


class Document(BaseModel):
    """Document model for file management and processing."""

    # IDs
    id: UUID = Field(..., description="Document ID")
    tenant_id: UUID = Field(..., description="Tenant ID")

    # Basic Information
    filename: str = Field(..., min_length=1, max_length=255, description="Original filename")
    original_filename: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="User-provided filename"
    )
    file_size: int = Field(..., ge=0, description="File size in bytes")
    mime_type: str = Field(..., description="MIME type of the file")
    file_format: DocumentFormat = Field(..., description="Document format")
    document_type: DocumentType = Field(..., description="Document type/category")

    # Storage Information
    blob_url: str = Field(..., description="Azure Blob Storage URL")
    blob_container: str = Field(..., description="Blob storage container name")
    blob_path: str = Field(..., description="Blob storage path")

    # Processing Status
    status: DocumentStatus = Field(
        default=DocumentStatus.UPLOADED,
        description="Processing status"
    )
    processing_started_at: Optional[datetime] = Field(
        None,
        description="Processing start timestamp"
    )
    processing_completed_at: Optional[datetime] = Field(
        None,
        description="Processing completion timestamp"
    )
    extraction_method: Optional[ExtractionMethod] = Field(
        None,
        description="Content extraction method used"
    )

    # Content
    content: Optional[DocumentContent] = Field(
        None,
        description="Extracted content and metadata"
    )

    # Processing Results
    processing_errors: List[DocumentProcessingError] = Field(
        default_factory=list,
        description="Processing errors encountered"
    )
    processing_warnings: List[str] = Field(
        default_factory=list,
        description="Processing warnings"
    )

    # Security and Validation
    virus_scan_status: Optional[str] = Field(
        None,
        description="Virus scan status (clean/infected/pending)"
    )
    checksum: Optional[str] = Field(None, description="File checksum for integrity")

    # Associations
    profile_id: Optional[UUID] = Field(
        None,
        description="Associated profile ID"
    )
    uploaded_by: UUID = Field(..., description="User who uploaded the document")

    # Search and Indexing
    is_indexed: bool = Field(default=False, description="Whether document is indexed for search")
    index_updated_at: Optional[datetime] = Field(
        None,
        description="Last index update timestamp"
    )
    embedding_vector: Optional[List[float]] = Field(
        None,
        description="Document embedding vector for similarity search"
    )

    # Access Control
    is_public: bool = Field(default=False, description="Whether document is publicly accessible")
    access_level: str = Field(default="private", description="Access level (private/tenant/public)")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    tags: List[str] = Field(default_factory=list, description="Document tags")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def file_size_mb(self) -> float:
        """Get file size in megabytes."""
        return round(self.file_size / (1024 * 1024), 2)

    @property
    def processing_duration(self) -> Optional[int]:
        """Get processing duration in seconds."""
        if not self.processing_started_at or not self.processing_completed_at:
            return None
        delta = self.processing_completed_at - self.processing_started_at
        return int(delta.total_seconds())

    @property
    def has_errors(self) -> bool:
        """Check if document has processing errors."""
        return len(self.processing_errors) > 0

    @property
    def is_processed(self) -> bool:
        """Check if document processing is completed."""
        return self.status == DocumentStatus.COMPLETED

    @property
    def file_extension(self) -> str:
        """Get file extension from filename."""
        return self.filename.split('.')[-1].lower() if '.' in self.filename else ''


class DocumentCreate(BaseModel):
    """Model for document creation/upload."""

    filename: str = Field(..., min_length=1, max_length=255)
    file_size: int = Field(..., ge=0)
    mime_type: str = Field(...)
    document_type: DocumentType = Field(...)
    profile_id: Optional[UUID] = None

    # Optional metadata
    description: Optional[str] = Field(None, max_length=1000)
    is_public: bool = Field(default=False)

    @field_validator('filename')
    @classmethod
    def validate_filename(cls, v):
        """Validate filename format."""
        if not '.' in v:
            raise ValueError('Filename must have an extension')

        extension = v.split('.')[-1].lower()
        allowed_extensions = ['pdf', 'doc', 'docx', 'txt', 'rtf', 'html', 'odt']
        if extension not in allowed_extensions:
            raise ValueError(f'File extension {extension} is not supported')

        return v


class DocumentUpdate(BaseModel):
    """Model for document updates."""

    document_type: Optional[DocumentType] = None
    profile_id: Optional[UUID] = None
    is_public: Optional[bool] = None
    access_level: Optional[str] = None
    description: Optional[str] = Field(None, max_length=1000)


class DocumentResponse(BaseModel):
    """Document response model."""

    id: UUID
    filename: str
    original_filename: str
    file_size: int
    file_size_mb: float
    mime_type: str
    file_format: DocumentFormat
    document_type: DocumentType
    status: DocumentStatus
    processing_duration: Optional[int]
    has_errors: bool
    is_processed: bool
    profile_id: Optional[UUID]
    uploaded_by: UUID
    is_indexed: bool
    is_public: bool
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_document(cls, document: Document) -> "DocumentResponse":
        """Create DocumentResponse from Document model."""
        return cls(
            id=document.id,
            filename=document.filename,
            original_filename=document.original_filename,
            file_size=document.file_size,
            file_size_mb=document.file_size_mb,
            mime_type=document.mime_type,
            file_format=document.file_format,
            document_type=document.document_type,
            status=document.status,
            processing_duration=document.processing_duration,
            has_errors=document.has_errors,
            is_processed=document.is_processed,
            profile_id=document.profile_id,
            uploaded_by=document.uploaded_by,
            is_indexed=document.is_indexed,
            is_public=document.is_public,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )


class BulkUploadRequest(BaseModel):
    """Model for bulk document upload requests."""

    documents: List[DocumentCreate] = Field(..., max_length=100, description="Documents to upload")
    profile_id: Optional[UUID] = Field(None, description="Associate all documents with this profile")
    process_async: bool = Field(default=True, description="Process documents asynchronously")
    notify_on_completion: bool = Field(default=True, description="Send notification when complete")

    @field_validator('documents')
    @classmethod
    def validate_documents(cls, v):
        """Validate document list."""
        if len(v) == 0:
            raise ValueError('At least one document must be provided')
        return v


class BulkUploadResponse(BaseModel):
    """Response for bulk upload operations."""

    batch_id: UUID = Field(..., description="Batch processing ID")
    total_documents: int = Field(..., description="Total number of documents")
    accepted_documents: int = Field(..., description="Number of accepted documents")
    rejected_documents: int = Field(..., description="Number of rejected documents")
    rejected_reasons: List[str] = Field(
        default_factory=list,
        description="Reasons for document rejection"
    )
    estimated_completion: Optional[datetime] = Field(
        None,
        description="Estimated completion time"
    )


class DocumentSearchFilters(BaseModel):
    """Filters for document search."""

    # Basic filters
    keywords: Optional[str] = Field(None, description="Search keywords")
    document_types: Optional[List[DocumentType]] = Field(None, description="Document types")
    file_formats: Optional[List[DocumentFormat]] = Field(None, description="File formats")
    statuses: Optional[List[DocumentStatus]] = Field(None, description="Processing statuses")

    # Size filters
    min_file_size: Optional[int] = Field(None, ge=0, description="Minimum file size in bytes")
    max_file_size: Optional[int] = Field(None, ge=0, description="Maximum file size in bytes")

    # Content filters
    has_content: Optional[bool] = Field(None, description="Has extracted content")
    min_word_count: Optional[int] = Field(None, ge=0, description="Minimum word count")
    languages: Optional[List[str]] = Field(None, description="Content languages")

    # Association filters
    profile_ids: Optional[List[UUID]] = Field(None, description="Associated profile IDs")
    uploaded_by: Optional[List[UUID]] = Field(None, description="Uploader user IDs")

    # Date filters
    uploaded_after: Optional[datetime] = Field(None, description="Uploaded after date")
    uploaded_before: Optional[datetime] = Field(None, description="Uploaded before date")
    processed_after: Optional[datetime] = Field(None, description="Processed after date")

    # Access filters
    is_public: Optional[bool] = Field(None, description="Public documents only")
    is_indexed: Optional[bool] = Field(None, description="Indexed documents only")


class DocumentStats(BaseModel):
    """Document statistics model."""

    total_documents: int = Field(default=0, description="Total document count")
    by_status: Dict[str, int] = Field(
        default_factory=dict,
        description="Documents by processing status"
    )
    by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Documents by document type"
    )
    by_format: Dict[str, int] = Field(
        default_factory=dict,
        description="Documents by file format"
    )
    total_size_mb: float = Field(default=0.0, description="Total size in MB")
    average_processing_time: Optional[float] = Field(
        None,
        description="Average processing time in seconds"
    )
    processing_success_rate: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Processing success rate (0-1)"
    )


class DocumentAnalytics(BaseModel):
    """Document analytics model."""

    upload_trends: Dict[str, int] = Field(
        default_factory=dict,
        description="Upload trends by date"
    )
    processing_trends: Dict[str, int] = Field(
        default_factory=dict,
        description="Processing completion trends"
    )
    error_analysis: Dict[str, int] = Field(
        default_factory=dict,
        description="Error frequency by type"
    )
    popular_file_types: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Most popular file types"
    )
    extraction_quality: Dict[str, float] = Field(
        default_factory=dict,
        description="Content extraction quality metrics"
    )


__all__ = [
    "DocumentType",
    "DocumentStatus",
    "DocumentFormat",
    "ExtractionMethod",
    "DocumentContent",
    "DocumentProcessingError",
    "Document",
    "DocumentCreate",
    "DocumentUpdate",
    "DocumentResponse",
    "BulkUploadRequest",
    "BulkUploadResponse",
    "DocumentSearchFilters",
    "DocumentStats",
    "DocumentAnalytics",
]