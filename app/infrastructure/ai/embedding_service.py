"""
Embedding Service with pgvector Integration

Advanced vector operations for semantic search and similarity matching:
- Efficient embedding generation and storage
- pgvector integration for database operations
- Semantic similarity search with configurable thresholds
- Batch processing and optimization
- Vector indexing and performance tuning
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import structlog
from sqlalchemy import text, select, func
from sqlalchemy.ext.asyncio import AsyncConnection
import pgvector
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import get_settings
from app.infrastructure.ai.openai_service import OpenAIService
from app.infrastructure.adapters.postgres_adapter import PostgresAdapter

logger = structlog.get_logger(__name__)


class EmbeddingService:
    """
    Comprehensive embedding service with pgvector integration.
    
    Features:
    - High-performance embedding generation
    - Optimized pgvector database operations  
    - Semantic similarity search with ranking
    - Batch processing for large datasets
    - Vector indexing and query optimization
    - Comprehensive metrics and monitoring
    """
    
    def __init__(
        self,
        openai_service: OpenAIService,
        db_adapter: PostgresAdapter,
        cache_service=None
    ):
        self.settings = get_settings()
        self.openai_service = openai_service
        self.db_adapter = db_adapter
        self.cache_service = cache_service
        self._metrics = {
            "embeddings_generated": 0,
            "similarity_searches": 0,
            "cache_hits": 0,
            "db_operations": 0
        }

    async def generate_embedding(self, text: str, **kwargs) -> List[float]:
        """
        Generate embedding vector for text without storing.

        Args:
            text: Text content to embed
            **kwargs: Additional arguments (model, etc.)

        Returns:
            Generated embedding vector
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        model = kwargs.get("model", self.settings.OPENAI_EMBEDDING_MODEL)

        try:
            embedding = await self.openai_service.generate_embedding(text, model)
            self._metrics["embeddings_generated"] += 1

            logger.debug(
                "Generated embedding",
                dimensions=len(embedding),
                model=model
            )

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    async def generate_and_store_embedding(
        self,
        entity_id: str,
        entity_type: str,
        content: str,
        model: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> List[float]:
        """
        Generate embedding and store in database.
        
        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of entity (profile, job, document, etc.)
            content: Text content to embed
            model: Embedding model to use
            tenant_id: Tenant ID for multi-tenancy
            
        Returns:
            Generated embedding vector
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")
        
        try:
            # Generate embedding
            embedding = await self.openai_service.generate_embedding(content, model)
            
            # Store in database
            await self._store_embedding(
                entity_id=entity_id,
                entity_type=entity_type,
                embedding=embedding,
                model=model or self.settings.OPENAI_EMBEDDING_MODEL,
                tenant_id=tenant_id
            )
            
            self._metrics["embeddings_generated"] += 1
            
            logger.debug(
                "Generated and stored embedding",
                entity_id=entity_id,
                entity_type=entity_type,
                dimensions=len(embedding),
                tenant_id=tenant_id
            )
            
            return embedding
            
        except Exception as e:
            logger.error(
                f"Failed to generate and store embedding: {e}",
                entity_id=entity_id,
                entity_type=entity_type
            )
            raise
    
    async def generate_and_store_batch(
        self,
        entities: List[Dict[str, Any]],
        model: Optional[str] = None
    ) -> List[Tuple[str, List[float]]]:
        """
        Generate and store embeddings for multiple entities efficiently.
        
        Args:
            entities: List of entity dictionaries with keys:
                     - entity_id: str
                     - entity_type: str
                     - content: str
                     - tenant_id: Optional[str]
            model: Embedding model to use
            
        Returns:
            List of (entity_id, embedding) tuples
        """
        if not entities:
            return []
        
        try:
            # Extract content for batch processing
            contents = [entity["content"] for entity in entities]
            
            # Generate embeddings in batch
            embeddings = await self.openai_service.generate_embeddings_batch(contents, model)
            
            # Store embeddings in batch
            await self._store_embeddings_batch(entities, embeddings, model)
            
            self._metrics["embeddings_generated"] += len(entities)
            
            result = [(entity["entity_id"], embedding) for entity, embedding in zip(entities, embeddings)]
            
            logger.info(
                "Generated and stored batch embeddings",
                count=len(entities),
                model=model or self.settings.OPENAI_EMBEDDING_MODEL
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate and store batch embeddings: {e}")
            raise
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        entity_type: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 20,
        threshold: float = 0.7,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search using pgvector.
        
        Args:
            query_embedding: Query vector for similarity search
            entity_type: Filter by entity type
            tenant_id: Filter by tenant ID
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            include_metadata: Include entity metadata in results
            
        Returns:
            List of similar entities with similarity scores
        """
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")
        
        try:
            # Build query with filters
            query_parts = [
                "SELECT entity_id, entity_type, embedding_vector,",
                "1 - (embedding_vector <=> %s::vector) as similarity",
                "FROM embeddings",
                "WHERE 1 - (embedding_vector <=> %s::vector) >= %s"
            ]
            params = [query_embedding, query_embedding, threshold]
            
            if entity_type:
                query_parts.append("AND entity_type = %s")
                params.append(entity_type)
            
            if tenant_id:
                query_parts.append("AND tenant_id = %s")
                params.append(tenant_id)
            
            query_parts.extend([
                "ORDER BY embedding_vector <=> %s::vector",
                "LIMIT %s"
            ])
            params.extend([query_embedding, limit])
            
            query_sql = " ".join(query_parts)
            
            # Execute similarity search
            async with self.db_adapter.get_connection() as conn:
                result = await conn.execute(text(query_sql), params)
                rows = result.fetchall()
            
            # Process results
            similar_entities = []
            for row in rows:
                entity_data = {
                    "entity_id": row.entity_id,
                    "entity_type": row.entity_type,
                    "similarity": float(row.similarity),
                }
                
                # Add metadata if requested
                if include_metadata:
                    metadata = await self._get_entity_metadata(
                        row.entity_id,
                        row.entity_type,
                        tenant_id
                    )
                    entity_data["metadata"] = metadata
                
                similar_entities.append(entity_data)
            
            self._metrics["similarity_searches"] += 1
            self._metrics["db_operations"] += 1
            
            logger.debug(
                "Similarity search completed",
                query_dimensions=len(query_embedding),
                entity_type=entity_type,
                results_count=len(similar_entities),
                threshold=threshold
            )
            
            return similar_entities
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            raise
    
    async def find_similar_text(
        self,
        query_text: str,
        entity_type: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 20,
        threshold: float = 0.7,
        model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find similar entities using text query.
        
        Args:
            query_text: Text to search for similar content
            entity_type: Filter by entity type
            tenant_id: Filter by tenant ID
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            model: Embedding model to use
            
        Returns:
            List of similar entities with similarity scores
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")
        
        # Generate query embedding
        query_embedding = await self.openai_service.generate_embedding(query_text, model)
        
        # Perform similarity search
        return await self.similarity_search(
            query_embedding=query_embedding,
            entity_type=entity_type,
            tenant_id=tenant_id,
            limit=limit,
            threshold=threshold
        )
    
    async def get_entity_embedding(
        self,
        entity_id: str,
        entity_type: str,
        tenant_id: Optional[str] = None
    ) -> Optional[List[float]]:
        """
        Retrieve stored embedding for an entity.
        
        Args:
            entity_id: Entity identifier
            entity_type: Entity type
            tenant_id: Tenant ID
            
        Returns:
            Embedding vector or None if not found
        """
        try:
            query_parts = [
                "SELECT embedding_vector FROM embeddings",
                "WHERE entity_id = %s AND entity_type = %s"
            ]
            params = [entity_id, entity_type]
            
            if tenant_id:
                query_parts.append("AND tenant_id = %s")
                params.append(tenant_id)
            
            query_sql = " ".join(query_parts)
            
            async with self.db_adapter.get_connection() as conn:
                result = await conn.execute(text(query_sql), params)
                row = result.fetchone()
            
            if row:
                return list(row.embedding_vector)
            return None
            
        except Exception as e:
            logger.error(
                f"Failed to retrieve embedding: {e}",
                entity_id=entity_id,
                entity_type=entity_type
            )
            return None
    
    async def update_entity_embedding(
        self,
        entity_id: str,
        entity_type: str,
        new_content: str,
        tenant_id: Optional[str] = None,
        model: Optional[str] = None
    ) -> List[float]:
        """
        Update existing entity embedding with new content.
        
        Args:
            entity_id: Entity identifier
            entity_type: Entity type
            new_content: New content to embed
            tenant_id: Tenant ID
            model: Embedding model to use
            
        Returns:
            New embedding vector
        """
        if not new_content or not new_content.strip():
            raise ValueError("Content cannot be empty")
        
        try:
            # Generate new embedding
            new_embedding = await self.openai_service.generate_embedding(new_content, model)
            
            # Update in database
            query_parts = [
                "UPDATE embeddings SET",
                "embedding_vector = %s::vector,",
                "embedding_model = %s,",
                "updated_at = NOW()",
                "WHERE entity_id = %s AND entity_type = %s"
            ]
            params = [
                new_embedding,
                model or self.settings.OPENAI_EMBEDDING_MODEL,
                entity_id,
                entity_type
            ]
            
            if tenant_id:
                query_parts.insert(-1, "AND tenant_id = %s")
                params.insert(-2, tenant_id)
            
            query_sql = " ".join(query_parts)
            
            async with self.db_adapter.get_connection() as conn:
                result = await conn.execute(text(query_sql), params)
            
            if result.rowcount == 0:
                # Entity doesn't exist, create new
                await self._store_embedding(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    embedding=new_embedding,
                    model=model or self.settings.OPENAI_EMBEDDING_MODEL,
                    tenant_id=tenant_id
                )
            
            logger.debug(
                "Updated entity embedding",
                entity_id=entity_id,
                entity_type=entity_type,
                dimensions=len(new_embedding)
            )
            
            return new_embedding
            
        except Exception as e:
            logger.error(
                f"Failed to update embedding: {e}",
                entity_id=entity_id,
                entity_type=entity_type
            )
            raise
    
    async def delete_entity_embedding(
        self,
        entity_id: str,
        entity_type: str,
        tenant_id: Optional[str] = None
    ) -> bool:
        """
        Delete stored embedding for an entity.
        
        Args:
            entity_id: Entity identifier
            entity_type: Entity type
            tenant_id: Tenant ID
            
        Returns:
            True if deleted, False if not found
        """
        try:
            query_parts = [
                "DELETE FROM embeddings",
                "WHERE entity_id = %s AND entity_type = %s"
            ]
            params = [entity_id, entity_type]
            
            if tenant_id:
                query_parts.append("AND tenant_id = %s")
                params.append(tenant_id)
            
            query_sql = " ".join(query_parts)
            
            async with self.db_adapter.get_connection() as conn:
                result = await conn.execute(text(query_sql), params)
            
            deleted = result.rowcount > 0
            
            if deleted:
                logger.debug(
                    "Deleted entity embedding",
                    entity_id=entity_id,
                    entity_type=entity_type
                )
            
            return deleted
            
        except Exception as e:
            logger.error(
                f"Failed to delete embedding: {e}",
                entity_id=entity_id,
                entity_type=entity_type
            )
            return False
    
    async def calculate_similarity_matrix(
        self,
        embeddings: List[List[float]]
    ) -> np.ndarray:
        """
        Calculate similarity matrix between multiple embeddings.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Similarity matrix as numpy array
        """
        if not embeddings:
            return np.array([])
        
        embeddings_array = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings_array)
        
        return similarity_matrix
    
    async def _store_embedding(
        self,
        entity_id: str,
        entity_type: str,
        embedding: List[float],
        model: str,
        tenant_id: Optional[str] = None
    ) -> None:
        """Store single embedding in database"""
        query_sql = """
            INSERT INTO embeddings (
                id, entity_id, entity_type, embedding_model, 
                embedding_vector, tenant_id, created_at, updated_at
            ) VALUES (
                gen_random_uuid(), %s, %s, %s, %s::vector, %s, NOW(), NOW()
            )
            ON CONFLICT (entity_id, entity_type, tenant_id)
            DO UPDATE SET
                embedding_vector = EXCLUDED.embedding_vector,
                embedding_model = EXCLUDED.embedding_model,
                updated_at = NOW()
        """
        
        params = [entity_id, entity_type, model, embedding, tenant_id]
        
        async with self.db_adapter.get_connection() as conn:
            await conn.execute(text(query_sql), params)
        
        self._metrics["db_operations"] += 1
    
    async def _store_embeddings_batch(
        self,
        entities: List[Dict[str, Any]],
        embeddings: List[List[float]],
        model: Optional[str] = None
    ) -> None:
        """Store multiple embeddings efficiently"""
        if len(entities) != len(embeddings):
            raise ValueError("Entities and embeddings count mismatch")
        
        model = model or self.settings.OPENAI_EMBEDDING_MODEL
        
        # Prepare batch insert data
        values = []
        for entity, embedding in zip(entities, embeddings):
            values.extend([
                entity["entity_id"],
                entity["entity_type"], 
                model,
                embedding,
                entity.get("tenant_id")
            ])
        
        # Build batch insert query
        placeholders = ", ".join([
            "(%s, %s, %s, %s::vector, %s)" 
            for _ in range(len(entities))
        ])
        
        query_sql = f"""
            INSERT INTO embeddings (
                entity_id, entity_type, embedding_model, 
                embedding_vector, tenant_id
            ) VALUES {placeholders}
            ON CONFLICT (entity_id, entity_type, tenant_id)
            DO UPDATE SET
                embedding_vector = EXCLUDED.embedding_vector,
                embedding_model = EXCLUDED.embedding_model,
                updated_at = NOW()
        """
        
        async with self.db_adapter.get_connection() as conn:
            await conn.execute(text(query_sql), values)
        
        self._metrics["db_operations"] += 1
    
    async def _get_entity_metadata(
        self,
        entity_id: str,
        entity_type: str,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get entity metadata from appropriate table"""
        # This would typically join with the main entity tables
        # For now, return basic metadata
        return {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "tenant_id": tenant_id
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        # Get database statistics
        try:
            async with self.db_adapter.get_connection() as conn:
                result = await conn.execute(text(
                    "SELECT COUNT(*) as total_embeddings FROM embeddings"
                ))
                total_embeddings = result.fetchone().total_embeddings
        except Exception:
            total_embeddings = 0
        
        return {
            "operations": self._metrics.copy(),
            "database": {
                "total_embeddings": total_embeddings
            },
            "configuration": {
                "embedding_model": self.settings.OPENAI_EMBEDDING_MODEL,
                "embedding_dimension": self.settings.EMBEDDING_DIMENSION,
                "similarity_threshold": self.settings.SEARCH_SIMILARITY_THRESHOLD
            }
        }
    
    async def check_health(self) -> Dict[str, Any]:
        """Check service health"""
        try:
            # Test embedding generation
            test_embedding = await self.openai_service.generate_embedding("health check")
            
            # Test database connection
            async with self.db_adapter.get_connection() as conn:
                await conn.execute(text("SELECT 1"))
            
            return {
                "status": "healthy",
                "embedding_service": "operational",
                "database": "connected",
                "embedding_dimension": len(test_embedding),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
