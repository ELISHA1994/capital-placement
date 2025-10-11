"""
Query expansion caching table with tenant isolation and lifecycle tracking.

This table stores generated query expansion metadata so that the search layer
can reuse prior expansions without invoking the AI provider on every request.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TCH003
from uuid import UUID  # noqa: TCH003

from sqlalchemy import Column, DateTime, Float, Index, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field

from app.infrastructure.persistence.models.base import (
    TenantModel,
    create_tenant_id_column,
)


class QueryExpansionTable(TenantModel, table=True):
    """Tenant-scoped cache of AI-generated query expansions."""

    __tablename__ = "query_expansions"

    tenant_id: UUID = Field(
        sa_column=create_tenant_id_column(),
        description="Tenant identifier for multi-tenant isolation",
    )

    original_query: str = Field(
        sa_column=Column(String(1000), nullable=False, index=True),
        description="Original user query text",
    )
    query_hash: str = Field(
        sa_column=Column(String(128), nullable=False),
        description="Deterministic hash of the query for cache lookups",
    )
    expanded_terms: list[str] = Field(
        default_factory=list,
        sa_column=Column(JSONB, nullable=False, default=list),
        description="Expanded terms generated for the query",
    )
    primary_skills: list[str] = Field(
        default_factory=list,
        sa_column=Column(JSONB, nullable=False, default=list),
        description="Detected primary skills for the query",
    )
    job_roles: list[str] = Field(
        default_factory=list,
        sa_column=Column(JSONB, nullable=False, default=list),
        description="Relevant job roles inferred from the query",
    )
    experience_level: str | None = Field(
        default=None,
        sa_column=Column(String(100), nullable=True),
        description="Experience level inferred from the query",
    )
    industry: str | None = Field(
        default=None,
        sa_column=Column(String(100), nullable=True),
        description="Industry inferred from the query",
    )
    confidence_score: float | None = Field(
        default=None,
        sa_column=Column(Float, nullable=True),
        description="Confidence score for the expansion",
    )
    ai_model_used: str | None = Field(
        default=None,
        sa_column=Column(String(100), nullable=True),
        description="AI model used to generate the expansion",
    )
    expires_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
        description="When this expansion should expire",
    )
    usage_count: int = Field(
        default=0,
        sa_column=Column(Integer, nullable=False, default=0),
        description="Number of times this expansion has been used",
    )
    last_used_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
        description="Most recent usage timestamp",
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "query_hash",
            name="uq_query_expansions_tenant_hash",
        ),
        Index("ix_query_expansions_tenant_expires_at", "tenant_id", "expires_at"),
        Index("ix_query_expansions_query_hash", "query_hash"),
    )


__all__ = ["QueryExpansionTable"]
