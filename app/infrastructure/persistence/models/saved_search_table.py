"""
SQLModel SavedSearch table definition with JSONB configuration storage.

This module provides the database persistence layer for saved search management
with comprehensive indexing for performance.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import Boolean, Column, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID as PostgreSQLUUID
from sqlmodel import Field

from app.infrastructure.persistence.models.base import AuditableModel, create_tenant_id_column


class SavedSearchTable(AuditableModel, table=True):
    """
    Complete saved search model with database persistence.

    Stores search configurations as JSONB for flexibility while maintaining
    performance through strategic denormalization and indexing.
    """
    __tablename__ = "saved_searches"

    # Tenant isolation field (each table model must define its own)
    tenant_id: UUID = Field(
        sa_column=create_tenant_id_column(),
        description="Tenant identifier for multi-tenant isolation"
    )

    # Composite indexes for performance
    __table_args__ = (
        Index("idx_saved_searches_tenant_created_by", "tenant_id", "created_by"),
        Index("idx_saved_searches_tenant_status", "tenant_id", "status"),
        Index("idx_saved_searches_alert_scheduling", "is_alert", "next_alert_at"),
        Index("idx_saved_searches_name_tenant", "tenant_id", "name"),
    )

    # Core identification (denormalized for query performance)
    name: str = Field(
        sa_column=Column(String(200), nullable=False, index=True),
        description="Saved search name"
    )
    description: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
        description="Saved search description"
    )

    # Search configuration (stored as JSONB for flexibility)
    configuration: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default={}),
        description="Complete search configuration structure"
    )

    # Alert configuration
    is_alert: bool = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False, default=False, index=True),
        description="Whether this saved search is configured as an alert"
    )
    alert_frequency: str = Field(
        default="never",
        sa_column=Column(String(20), nullable=False, default="never"),
        description="Alert frequency (daily, weekly, monthly, never)"
    )
    next_alert_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime, nullable=True, index=True),
        description="Next scheduled alert execution time"
    )

    # Sharing configuration
    is_shared: bool = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False, default=False),
        description="Whether this saved search is shared with other users"
    )
    shared_with_users: List[str] = Field(
        default_factory=list,
        sa_column=Column(ARRAY(String), nullable=False, default=list),
        description="List of user IDs this search is shared with"
    )

    # Status
    status: str = Field(
        default="active",
        sa_column=Column(String(20), nullable=False, default="active", index=True),
        description="Saved search status (active, paused, archived, deleted)"
    )

    # Execution statistics
    total_executions: int = Field(
        default=0,
        sa_column=Column(Integer, nullable=False, default=0),
        description="Total number of times this search has been executed"
    )
    last_run: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime, nullable=True),
        description="Last execution time"
    )
    last_result_count: int = Field(
        default=0,
        sa_column=Column(Integer, nullable=False, default=0),
        description="Result count from last execution"
    )
    new_results_since_last_run: int = Field(
        default=0,
        sa_column=Column(Integer, nullable=False, default=0),
        description="New results since last execution"
    )
    average_result_count: float = Field(
        default=0.0,
        sa_column=Column(Float, nullable=False, default=0.0),
        description="Average result count across all executions"
    )
    average_execution_time_ms: int = Field(
        default=0,
        sa_column=Column(Integer, nullable=False, default=0),
        description="Average execution time in milliseconds"
    )

    # Additional metadata
    extra_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default={}),
        description="Additional metadata and audit information"
    )


__all__ = ["SavedSearchTable"]