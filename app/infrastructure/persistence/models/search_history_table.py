"""
SQLModel SearchHistory table definition with JSONB storage for complex data.

This module provides the database persistence layer for search history tracking
with comprehensive indexing for analytics queries.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from uuid import UUID

from sqlalchemy import Column, String, Integer, DateTime, Text, Index, Float, Boolean
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID, JSONB, ARRAY
from sqlmodel import Field

from app.infrastructure.persistence.models.base import TenantModel, create_tenant_id_column


class SearchHistoryTable(TenantModel, table=True):
    """
    Complete search history model with database persistence.

    Stores search execution details, parameters, results, and engagement
    metrics with optimized indexing for analytics queries.
    """
    __tablename__ = "search_history"

    # Tenant isolation field (each table model must define its own)
    tenant_id: UUID = Field(
        sa_column=create_tenant_id_column(),
        description="Tenant identifier for multi-tenant isolation"
    )

    # Override id column from TenantModel with explicit sa_column
    id: UUID = Field(
        sa_column=Column(
            PostgreSQLUUID(as_uuid=True),
            primary_key=True,
            nullable=False
        ),
        description="Unique search history identifier"
    )

    # Override created_at and updated_at from TimestampedModel
    created_at: datetime = Field(
        sa_column=Column(
            DateTime,
            nullable=False,
            default=datetime.utcnow,
            index=True
        ),
        description="Record creation timestamp"
    )

    updated_at: datetime = Field(
        sa_column=Column(
            DateTime,
            nullable=False,
            default=datetime.utcnow,
            onupdate=datetime.utcnow
        ),
        description="Record last update timestamp"
    )

    # Composite indexes for performance
    __table_args__ = (
        Index("idx_search_history_tenant_user", "tenant_id", "user_id"),
        Index("idx_search_history_tenant_executed", "tenant_id", "executed_at"),
        Index("idx_search_history_user_executed", "user_id", "executed_at"),
        Index("idx_search_history_outcome", "search_outcome"),
        Index("idx_search_history_query_text", "query_text"),  # For suggestions
        Index("idx_search_history_analytics", "tenant_id", "executed_at", "search_outcome"),
    )

    # User attribution
    user_id: UUID = Field(
        sa_column=Column(PostgreSQLUUID(as_uuid=True), nullable=False, index=True),
        description="User who performed the search"
    )

    # Search parameters (denormalized for quick filtering)
    query_text: str = Field(
        sa_column=Column(Text, nullable=False, index=True),
        description="Original search query text"
    )
    search_mode: str = Field(
        sa_column=Column(String(50), nullable=False, index=True),
        description="Search mode used (keyword, vector, hybrid, etc.)"
    )
    search_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default={}),
        description="Complete search parameters and filters"
    )

    # Results summary (denormalized for analytics)
    total_results: int = Field(
        default=0,
        sa_column=Column(Integer, nullable=False, default=0, index=True),
        description="Total number of matching results"
    )
    returned_results: int = Field(
        default=0,
        sa_column=Column(Integer, nullable=False, default=0),
        description="Number of results returned to user"
    )
    results_summary: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default={}),
        description="Summary of search results"
    )

    # Performance metrics (denormalized for analytics)
    search_duration_ms: int = Field(
        default=0,
        sa_column=Column(Integer, nullable=False, default=0, index=True),
        description="Total search execution time in milliseconds"
    )
    cache_hit: bool = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False, default=False, index=True),
        description="Whether search results came from cache"
    )
    performance_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default={}),
        description="Detailed performance metrics"
    )

    # Search outcome
    search_outcome: str = Field(
        default="success",
        sa_column=Column(String(20), nullable=False, default="success", index=True),
        description="Search outcome (success, no_results, error, abandoned, timeout)"
    )

    # User engagement (updated over time)
    results_clicked: List[str] = Field(
        default_factory=list,
        sa_column=Column(ARRAY(String), nullable=False, default=list),
        description="Profile IDs that were clicked"
    )
    profiles_contacted: List[str] = Field(
        default_factory=list,
        sa_column=Column(ARRAY(String), nullable=False, default=list),
        description="Profile IDs that were contacted"
    )
    engagement_data: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default={}),
        description="Complete user engagement data"
    )
    engagement_score: float = Field(
        default=0.0,
        sa_column=Column(Float, nullable=False, default=0.0, index=True),
        description="Calculated engagement score (0.0-1.0)"
    )

    # Context and attribution
    search_context: Optional[str] = Field(
        default=None,
        sa_column=Column(String(200), nullable=True),
        description="Search context identifier (job_post_id, etc.)"
    )
    source: str = Field(
        default="web_ui",
        sa_column=Column(String(50), nullable=False, default="web_ui"),
        description="Source of search (web_ui, api, automation, etc.)"
    )
    user_agent: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
        description="User agent string"
    )
    ip_address: Optional[str] = Field(
        default=None,
        sa_column=Column(String(45), nullable=True),
        description="IP address (IPv4 or IPv6)"
    )

    # Timestamps
    executed_at: datetime = Field(
        sa_column=Column(DateTime, nullable=False, index=True),
        description="When search was executed"
    )
    last_interaction_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime, nullable=True),
        description="Last user interaction with results"
    )

    # Additional metadata
    extra_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default={}),
        description="Additional metadata and tags"
    )


__all__ = ["SearchHistoryTable"]
