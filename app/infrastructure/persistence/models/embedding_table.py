"""
Vector embedding models for AI/ML functionality.

This module contains SQLModel table definitions for storing and managing
vector embeddings used in semantic search and similarity operations.
"""

from typing import List, Optional
from uuid import UUID
from sqlalchemy import Column, ForeignKey, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlmodel import Field, Relationship
from pgvector.sqlalchemy import Vector

from app.infrastructure.persistence.models.base import VectorModel, TenantModel, create_tenant_id_column



class EmbeddingTable(VectorModel, TenantModel, table=True):
    """Dedicated table for vector embeddings and similarity search operations"""
    __tablename__ = "embeddings"

    # Tenant isolation field (each table model must define its own)
    # Using function call to ensure each model gets its own Column instance
    tenant_id: UUID = Field(
        sa_column=create_tenant_id_column(),
        description="Tenant identifier for multi-tenant isolation"
    )

    # Override embedding field with proper Vector column
    # Using 3072 dimensions for text-embedding-3-large model
    embedding: Optional[List[float]] = Field(
        default=None,
        sa_column=Column(Vector(3072), nullable=True),
        description="Vector embedding for similarity search"
    )

    # Relationship to tenant (required for SQLModel to create foreign key constraints)
    tenant: Optional["TenantTable"] = Relationship(back_populates="embeddings")

    # Table constraints and indexes
    __table_args__ = (
        UniqueConstraint("entity_id", "entity_type", "tenant_id", name="uq_embeddings_entity_tenant"),
        Index("ix_embeddings_tenant_entity", "tenant_id", "entity_id", "entity_type"),
    )


__all__ = [
    "EmbeddingTable",
]