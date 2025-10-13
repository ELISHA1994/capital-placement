"""
Time-series optimized SQLModel table for search click events.

This model uses PostgreSQL partitioning for high-volume time-series data
with optimized indexes for both writes and analytics queries.
"""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy import Column, DateTime, Float, Index, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID as PostgreSQLUUID
from sqlalchemy.sql import func
from sqlmodel import Field

from app.infrastructure.persistence.models.base import TenantModel, create_tenant_id_column


class SearchClickTable(TenantModel, table=True):
    """
    High-performance search click event table with time-series optimization.

    Architecture:
    - Time-based partitioning (monthly) for scalability
    - Write-optimized with minimal indexes
    - Read-optimized aggregation queries
    - BRIN indexes for time-series data
    - Automatic partition management
    """
    __tablename__ = "search_clicks"

    # Primary key (inherited from TenantModel but defined here for sa_column)
    id: UUID = Field(
        sa_column=Column(
            PostgreSQLUUID(as_uuid=True),
            primary_key=True,
            nullable=False
        ),
        description="Unique click event identifier"
    )

    # Tenant isolation field
    tenant_id: UUID = Field(
        sa_column=create_tenant_id_column(),
        description="Tenant identifier for multi-tenant isolation"
    )

    # Timestamps (inherited from TenantModel but defined here for sa_column)
    created_at: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            default=text("NOW()"),
            server_default=func.now()
        ),
        description="Record creation timestamp"
    )

    updated_at: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            default=text("NOW()"),
            server_default=func.now(),
            onupdate=func.now()
        ),
        description="Record last update timestamp"
    )

    # Core event fields
    user_id: UUID = Field(
        sa_column=Column(PostgreSQLUUID(as_uuid=True), nullable=False, index=True),
        description="User who performed the click"
    )

    search_id: str = Field(
        sa_column=Column(String(255), nullable=False, index=True),
        description="Reference to the search execution"
    )

    profile_id: UUID = Field(
        sa_column=Column(PostgreSQLUUID(as_uuid=True), nullable=False, index=True),
        description="Profile that was clicked"
    )

    position: int = Field(
        sa_column=Column(Integer, nullable=False),
        description="Position in search results (0-based)"
    )

    # Temporal field (critical for partitioning)
    clicked_at: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            default=text("NOW()"),
            index=True
        ),
        description="When the click occurred (partition key)"
    )

    # Derived analytics fields (denormalized for query performance)
    relevance_score: Optional[float] = Field(
        default=None,
        sa_column=Column(Float, nullable=True),
        description="Relevance score of the clicked result"
    )

    rank_quality: Optional[float] = Field(
        default=None,
        sa_column=Column(Float, nullable=True),
        description="Position quality metric (1.0 for top result)"
    )

    engagement_signal: str = Field(
        default="medium",
        sa_column=Column(String(20), nullable=False, index=True),
        description="strong, medium, or weak engagement classification"
    )

    # Context data (stored as JSONB for flexibility)
    context: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default={}),
        description="Click context (session, device, timing, etc.)"
    )

    # Session tracking
    session_id: Optional[str] = Field(
        default=None,
        sa_column=Column(String(255), nullable=True, index=True),
        description="User session identifier for session analytics"
    )

    # Network information (for fraud detection)
    ip_address: Optional[str] = Field(
        default=None,
        sa_column=Column(INET, nullable=True),
        description="Client IP address (privacy considerations)"
    )

    # Additional metadata
    extra_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default={}),
        description="Additional event metadata"
    )

    # Performance optimization: Composite indexes
    __table_args__ = (
        # Time-series queries (BRIN index for large time ranges)
        Index(
            "ix_search_clicks_tenant_time_brin",
            "tenant_id",
            "clicked_at",
            postgresql_using="brin"
        ),

        # Search analytics (CTR calculations)
        Index(
            "ix_search_clicks_search_tenant",
            "search_id",
            "tenant_id",
            "clicked_at"
        ),

        # Profile analytics (profile popularity)
        Index(
            "ix_search_clicks_profile_tenant_time",
            "profile_id",
            "tenant_id",
            "clicked_at"
        ),

        # User behavior analytics
        Index(
            "ix_search_clicks_user_tenant_time",
            "user_id",
            "tenant_id",
            "clicked_at"
        ),

        # Session analytics
        Index(
            "ix_search_clicks_session",
            "session_id",
            "clicked_at",
            postgresql_where=text("session_id IS NOT NULL")
        ),

        # Position analytics (partial index for common positions)
        Index(
            "ix_search_clicks_position_tenant",
            "position",
            "tenant_id",
            "clicked_at",
            postgresql_where=text("position < 100")
        ),

        # Engagement analytics
        Index(
            "ix_search_clicks_engagement",
            "engagement_signal",
            "tenant_id",
            "clicked_at"
        ),

        # Partitioning configuration
        {
            "postgresql_partition_by": "RANGE (clicked_at)",
            "info": {
                "partition_type": "range",
                "partition_key": "clicked_at",
                "partition_interval": "1 month",
                "retention_months": 3,
            }
        }
    )


__all__ = ["SearchClickTable"]
