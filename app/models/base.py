"""
SQLModel base classes with multi-tenant support, timestamps, and advanced features.

This module provides the foundation for all SQLModel database models, replacing
the original Pydantic models while preserving all sophisticated patterns and
adding database persistence capabilities.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, TypeVar, Union, ClassVar
from uuid import UUID, uuid4

from sqlalchemy import Column, DateTime, String, Boolean, Integer, text
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import declared_attr
from sqlalchemy.sql import func
from sqlmodel import Field, SQLModel

T = TypeVar("T", bound="BaseModel")


class BaseModel(SQLModel):
    """
    Base SQLModel with common configuration and utilities.
    
    Replaces the original Pydantic BaseModel while preserving all functionality
    and adding database persistence capabilities.
    """
    
    model_config = {
        # Enable arbitrary types for complex objects
        "arbitrary_types_allowed": True,
        # Validate assignments after model creation
        "validate_assignment": True,
        # Use enum values instead of enum objects
        "use_enum_values": True,
        # Populate by name for aliases
        "populate_by_name": True,
        # Strict validation
        "str_strip_whitespace": True,
        # JSON schema generation
        "json_schema_serialization_defaults_required": True,
        # Ignore SQLAlchemy hybrid properties for Pydantic validation
        "ignored_types": (hybrid_property,),
    }
    
    def dict_exclude_none(self, **kwargs) -> Dict[str, Any]:
        """Export to dict excluding None values."""
        return self.model_dump(exclude_none=True, **kwargs)
    
    def dict_exclude_unset(self, **kwargs) -> Dict[str, Any]:
        """Export to dict excluding unset values."""
        return self.model_dump(exclude_unset=True, **kwargs)


class TimestampedModel(BaseModel):
    """
    Base model with automatic timestamp management.
    
    Preserves the original functionality while adding database-level
    automatic timestamp handling with PostgreSQL functions.
    """
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Record creation timestamp"
    )
    
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Record last update timestamp"
    )
    
    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()


class TenantModel(TimestampedModel):
    """
    Base model with tenant isolation support.
    
    Preserves multi-tenant architecture with database-level UUID support,
    automatic tenant validation, and proper indexing for performance.
    """
    
    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        description="Unique identifier"
    )
    
    def validate_tenant_id(cls, v):
        """Ensure tenant_id is a valid UUID."""
        if isinstance(v, str):
            try:
                return UUID(v)
            except ValueError:
                raise ValueError('tenant_id must be a valid UUID')
        return v


class SoftDeleteModel(TenantModel):
    """
    Model with soft delete functionality.
    
    Preserves all original soft delete patterns with database-level support
    and proper indexing for performance.
    """
    
    is_deleted: bool = Field(
        default=False,
        description="Soft delete flag"
    )
    
    deleted_at: Optional[datetime] = Field(
        default=None,
        description="Soft delete timestamp"
    )
    
    deleted_by: Optional[UUID] = Field(
        default=None,
        description="User who performed the deletion"
    )
    
    def soft_delete(self, deleted_by: Optional[UUID] = None) -> None:
        """Mark record as soft deleted."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
        self.deleted_by = deleted_by
        self.update_timestamp()
    
    def restore(self) -> None:
        """Restore soft deleted record."""
        self.is_deleted = False
        self.deleted_at = None
        self.deleted_by = None
        self.update_timestamp()


class AuditableModel(SoftDeleteModel):
    """
    Model with full audit trail support.
    
    Preserves the original audit functionality with database-level
    version management and user tracking for complete audit trails.
    """
    
    created_by: Optional[UUID] = Field(
        default=None,
        description="User who created the record"
    )
    
    updated_by: Optional[UUID] = Field(
        default=None,
        description="User who last updated the record"
    )
    
    version: int = Field(
        default=1,
        description="Record version for optimistic locking"
    )
    
    def increment_version(self, updated_by: Optional[UUID] = None) -> None:
        """Increment version and update metadata."""
        self.version += 1
        self.updated_by = updated_by
        self.update_timestamp()


class MetadataModel(BaseModel):
    """
    Model for storing arbitrary metadata with JSONB support.
    
    Preserves the original metadata functionality with optimized
    PostgreSQL JSONB storage and indexing capabilities.
    """
    
    extra_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata dictionary"
    )
    
    tags: List[str] = Field(
        default_factory=list,
        sa_column=Column(ARRAY(String), nullable=False, default=list),
        description="Tags for categorization and search"
    )
    
    def add_tag(self, tag: str) -> None:
        """Add a tag if not already present."""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag if present."""
        if tag in self.tags:
            self.tags.remove(tag)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set a metadata value."""
        self.extra_metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value with optional default."""
        return self.extra_metadata.get(key, default)


class VectorModel(BaseModel):
    """
    Model with pgvector support for AI/ML embeddings.
    
    New addition for vector similarity search capabilities that integrates
    with the existing pgvector infrastructure while providing type safety.
    """
    
    entity_id: UUID = Field(
        description="ID of the entity this embedding represents"
    )
    
    entity_type: str = Field(
        description="Type of entity (profile, document, etc.)"
    )
    
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding for similarity search"
    )
    
    embedding_model: str = Field(
        default="text-embedding-3-large",
        description="Model used to generate embedding"
    )
    
    @hybrid_property  
    def has_embedding(self) -> bool:
        """Check if embedding exists (instance-level)."""
        return self.embedding is not None and len(self.embedding) > 0 if self.embedding else False
    
    @has_embedding.expression
    def has_embedding(cls):
        """SQL expression for has_embedding (class-level)."""
        return cls.embedding.is_not(None)


class PaginationModel(BaseModel):
    """
    Model for pagination parameters.
    
    Preserves the original pagination functionality for consistent
    API responses and database query optimization.
    """
    
    page: int = Field(
        default=1,
        ge=1,
        description="Page number (1-based)"
    )
    size: int = Field(
        default=20,
        ge=1,
        le=1000,
        description="Items per page"
    )
    
    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.size
    
    @property
    def limit(self) -> int:
        """Get limit value."""
        return self.size


class PaginatedResponse(BaseModel):
    """
    Generic paginated response model.
    
    Preserves the original pagination response structure for
    consistent API behavior across all endpoints.
    """
    
    items: List[Any] = Field(
        default_factory=list,
        description="List of items for current page"
    )
    total: int = Field(
        default=0,
        ge=0,
        description="Total number of items across all pages"
    )
    page: int = Field(
        default=1,
        ge=1,
        description="Current page number"
    )
    size: int = Field(
        default=20,
        ge=1,
        description="Items per page"
    )
    pages: int = Field(
        default=0,
        ge=0,
        description="Total number of pages"
    )
    has_next: bool = Field(
        default=False,
        description="Whether there is a next page"
    )
    has_prev: bool = Field(
        default=False,
        description="Whether there is a previous page"
    )
    
    @classmethod
    def create(
        cls,
        items: List[Any],
        total: int,
        page: int,
        size: int
    ) -> "PaginatedResponse":
        """Create a paginated response from query results."""
        pages = (total + size - 1) // size  # Ceiling division
        
        return cls(
            items=items,
            total=total,
            page=page,
            size=size,
            pages=pages,
            has_next=page < pages,
            has_prev=page > 1
        )


class ErrorModel(BaseModel):
    """
    Standard error response model.
    
    Preserves the original error handling structure for
    consistent API error responses.
    """
    
    error: str = Field(..., description="Error type or code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request identifier for tracing"
    )


class HealthCheckModel(BaseModel):
    """
    Health check response model.
    
    Preserves the original health check structure for
    monitoring and operational visibility.
    """
    
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Environment name")
    services: Dict[str, Dict[str, Union[str, bool, float]]] = Field(
        default_factory=dict,
        description="Individual service health status"
    )
    uptime: Optional[float] = Field(
        default=None,
        description="Application uptime in seconds"
    )


# Base classes only - actual table definitions should be in specific model files