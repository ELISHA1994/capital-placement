"""
Vector embedding models for AI/ML functionality.

This module contains SQLModel table definitions for storing and managing
vector embeddings used in semantic search and similarity operations.
"""

from typing import List, Optional
from uuid import UUID
from sqlalchemy import Column, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlmodel import Field, Relationship
from pgvector.sqlalchemy import Vector

from app.models.base import VectorModel, TenantModel, create_tenant_id_column



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
    embedding: Optional[List[float]] = Field(
        default=None,
        sa_column=Column(Vector(1536), nullable=True),
        description="Vector embedding for similarity search"
    )
    
    # Relationship to tenant (required for SQLModel to create foreign key constraints)
    tenant: Optional["TenantTable"] = Relationship(back_populates="embeddings")
    
    # Indexes will be created via migrations for optimal performance:
    # CREATE INDEX embeddings_tenant_entity_idx ON embeddings (tenant_id, entity_id, entity_type);
    # CREATE INDEX embeddings_embedding_hnsw_idx ON embeddings USING hnsw (embedding vector_cosine_ops);