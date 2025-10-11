"""
Vector Search Service with pgvector Integration

High-performance semantic similarity search using pgvector for PostgreSQL:
- Fast cosine similarity and L2 distance searches
- Efficient vector indexing with IVFFlat algorithm
- Multi-tenant data isolation and security
- Intelligent caching for frequent queries
- Batch search operations
- Performance monitoring and analytics
- Configurable similarity thresholds
"""

import json
import hashlib
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import structlog
import numpy as np
from uuid import UUID, uuid4

from app.core.config import get_settings
from app.domain.interfaces import IHealthCheck
from app.infrastructure.ai.embedding_service import EmbeddingService
from app.infrastructure.ai.cache_manager import CacheManager
from app.infrastructure.adapters.postgres_adapter import PostgresAdapter

logger = structlog.get_logger(__name__)


@dataclass
class VectorSearchResult:
    """Individual vector search result with metadata"""
    entity_id: str
    entity_type: str
    similarity_score: float
    distance: float
    metadata: Dict[str, Any]
    content_preview: Optional[str] = None
    tenant_id: Optional[str] = None
    embedding_model: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class VectorSearchResponse:
    """Complete vector search response with analytics"""
    results: List[VectorSearchResult]
    query_id: str
    total_candidates: int
    search_time_ms: int
    similarity_threshold: float
    search_metadata: Dict[str, Any]
    cache_hit: bool = False


@dataclass
class SearchFilter:
    """Search filters for vector queries"""
    entity_types: Optional[List[str]] = None
    tenant_ids: Optional[List[str]] = None
    embedding_models: Optional[List[str]] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    metadata_filters: Optional[Dict[str, Any]] = None
    exclude_entity_ids: Optional[List[str]] = None


class VectorSearchService(IHealthCheck):
    """
    Advanced vector search service with pgvector integration.
    
    Features:
    - High-performance similarity search with pgvector indexes
    - Multiple similarity metrics (cosine, L2 distance)
    - Intelligent query caching and optimization
    - Multi-tenant isolation and security
    - Batch search operations for efficiency
    - Real-time performance monitoring
    - Configurable similarity thresholds
    - Advanced filtering and metadata search
    """
    
    def __init__(
        self,
        db_adapter: PostgresAdapter,
        embedding_service: EmbeddingService,
        cache_manager: Optional[CacheManager] = None
    ):
        self.settings = get_settings()
        self.db_adapter = db_adapter
        self.embedding_service = embedding_service
        self.cache_manager = cache_manager
        
        # Performance and analytics tracking
        self._stats = {
            "searches_performed": 0,
            "cache_hits": 0,
            "total_search_time_ms": 0,
            "average_search_time_ms": 0,
            "results_returned": 0,
            "embedding_generations": 0,
            "errors": 0
        }
        
        # Search configuration
        self.default_similarity_threshold = 0.7
        self.max_results_per_search = 1000
        self.default_search_limit = 20
        
        # Cache configuration
        self.vector_cache_ttl = 3600  # 1 hour
        self.enable_search_caching = True
        
    async def similarity_search(
        self,
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        tenant_id: Optional[str] = None,
        limit: int = 20,
        similarity_threshold: float = 0.7,
        similarity_metric: str = "cosine",  # cosine, l2
        search_filter: Optional[SearchFilter] = None,
        use_cache: bool = True,
        include_metadata: bool = True
    ) -> VectorSearchResponse:
        """
        Perform semantic similarity search using vector embeddings.
        
        Args:
            query_text: Text to convert to embedding for search
            query_embedding: Pre-computed embedding vector
            tenant_id: Tenant ID for data isolation
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0.0-1.0)
            similarity_metric: Similarity calculation method
            search_filter: Additional filters for search
            use_cache: Enable result caching
            include_metadata: Include entity metadata in results
            
        Returns:
            VectorSearchResponse with ranked results and analytics
        """
        start_time = datetime.now()
        query_id = str(uuid4())
        
        try:
            # Validate inputs
            if not query_text and not query_embedding:
                raise ValueError("Either query_text or query_embedding must be provided")
            
            if limit > self.max_results_per_search:
                limit = self.max_results_per_search
                logger.warning(f"Search limit capped at {self.max_results_per_search}")
            
            # Generate embedding if needed
            if query_text and not query_embedding:
                query_embedding = await self._get_or_generate_embedding(query_text, use_cache)
            
            # Check cache for similar searches
            cached_results = None
            if use_cache and self.enable_search_caching:
                cached_results = await self._get_cached_search_results(
                    query_embedding, tenant_id, similarity_threshold, search_filter
                )
                if cached_results:
                    self._stats["cache_hits"] += 1
                    return cached_results
            
            # Perform vector search
            raw_results = await self._perform_vector_search(
                query_embedding=query_embedding,
                tenant_id=tenant_id,
                limit=limit,
                similarity_threshold=similarity_threshold,
                similarity_metric=similarity_metric,
                search_filter=search_filter,
                include_metadata=include_metadata
            )
            
            # Calculate search time
            search_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Create response
            response = VectorSearchResponse(
                results=raw_results,
                query_id=query_id,
                total_candidates=len(raw_results),
                search_time_ms=search_time_ms,
                similarity_threshold=similarity_threshold,
                search_metadata={
                    "similarity_metric": similarity_metric,
                    "query_text": query_text[:100] if query_text else None,
                    "tenant_id": tenant_id,
                    "filters_applied": search_filter is not None,
                    "embedding_model": self.embedding_service.model_name if hasattr(self.embedding_service, 'model_name') else "unknown"
                },
                cache_hit=False
            )
            
            # Cache results if enabled
            if use_cache and self.enable_search_caching and len(raw_results) > 0:
                await self._cache_search_results(response, query_embedding, tenant_id)
            
            # Update statistics
            self._update_search_stats(search_time_ms, len(raw_results))
            
            logger.info(
                "Vector search completed",
                query_id=query_id,
                results_count=len(raw_results),
                search_time_ms=search_time_ms,
                cache_hit=False,
                tenant_id=tenant_id
            )
            
            return response
            
        except Exception as e:
            self._stats["errors"] += 1
            search_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            logger.error(
                "Vector search failed",
                query_id=query_id,
                error=str(e),
                search_time_ms=search_time_ms,
                tenant_id=tenant_id
            )
            
            # Return empty response on error
            return VectorSearchResponse(
                results=[],
                query_id=query_id,
                total_candidates=0,
                search_time_ms=search_time_ms,
                similarity_threshold=similarity_threshold,
                search_metadata={"error": str(e)},
                cache_hit=False
            )
    
    async def batch_similarity_search(
        self,
        queries: List[Dict[str, Any]],
        tenant_id: Optional[str] = None,
        use_cache: bool = True
    ) -> List[VectorSearchResponse]:
        """
        Perform multiple vector searches in batch for efficiency.
        
        Args:
            queries: List of query dictionaries with search parameters
            tenant_id: Tenant ID for data isolation
            use_cache: Enable result caching
            
        Returns:
            List of VectorSearchResponse objects
        """
        start_time = datetime.now()
        
        try:
            # Process queries concurrently
            search_tasks = []
            for query in queries:
                task = self.similarity_search(
                    query_text=query.get("query_text"),
                    query_embedding=query.get("query_embedding"),
                    tenant_id=tenant_id or query.get("tenant_id"),
                    limit=query.get("limit", self.default_search_limit),
                    similarity_threshold=query.get("similarity_threshold", self.default_similarity_threshold),
                    similarity_metric=query.get("similarity_metric", "cosine"),
                    search_filter=query.get("search_filter"),
                    use_cache=use_cache,
                    include_metadata=query.get("include_metadata", True)
                )
                search_tasks.append(task)
            
            # Execute all searches concurrently
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Handle any exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Batch search query {i} failed: {result}")
                    # Create empty response for failed queries
                    processed_results.append(VectorSearchResponse(
                        results=[],
                        query_id=str(uuid4()),
                        total_candidates=0,
                        search_time_ms=0,
                        similarity_threshold=0.0,
                        search_metadata={"error": str(result)},
                        cache_hit=False
                    ))
                else:
                    processed_results.append(result)
            
            batch_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.info(
                "Batch vector search completed",
                queries_count=len(queries),
                successful_queries=len([r for r in results if not isinstance(r, Exception)]),
                total_time_ms=batch_time_ms,
                tenant_id=tenant_id
            )
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Batch similarity search failed: {e}")
            # Return empty responses for all queries
            return [
                VectorSearchResponse(
                    results=[],
                    query_id=str(uuid4()),
                    total_candidates=0,
                    search_time_ms=0,
                    similarity_threshold=0.0,
                    search_metadata={"batch_error": str(e)},
                    cache_hit=False
                ) for _ in queries
            ]
    
    async def find_similar_entities(
        self,
        entity_id: str,
        entity_type: str,
        tenant_id: Optional[str] = None,
        limit: int = 20,
        similarity_threshold: float = 0.8,
        exclude_self: bool = True
    ) -> VectorSearchResponse:
        """
        Find entities similar to a specific entity using its stored embedding.
        
        Args:
            entity_id: ID of the reference entity
            entity_type: Type of the reference entity
            tenant_id: Tenant ID for data isolation
            limit: Maximum number of similar entities to return
            similarity_threshold: Minimum similarity score
            exclude_self: Whether to exclude the reference entity from results
            
        Returns:
            VectorSearchResponse with similar entities
        """
        try:
            # Get the reference entity's embedding
            async with self.db_adapter.get_connection() as conn:
                embedding_row = await conn.fetchrow("""
                    SELECT embedding, embedding_model
                    FROM embeddings
                    WHERE entity_id = $1
                    AND entity_type = $2
                    AND (tenant_id = $3 OR tenant_id IS NULL)
                    ORDER BY created_at DESC
                    LIMIT 1
                """, entity_id, entity_type, tenant_id)

                if not embedding_row:
                    logger.warning(f"No embedding found for entity {entity_id}")
                    return VectorSearchResponse(
                        results=[],
                        query_id=str(uuid4()),
                        total_candidates=0,
                        search_time_ms=0,
                        similarity_threshold=similarity_threshold,
                        search_metadata={"error": "Reference entity embedding not found"},
                        cache_hit=False
                    )

            # Create search filter to exclude self if needed
            search_filter = SearchFilter(
                entity_types=[entity_type],
                exclude_entity_ids=[entity_id] if exclude_self else None
            )

            # Perform similarity search using the reference embedding
            return await self.similarity_search(
                query_embedding=embedding_row['embedding'],
                tenant_id=tenant_id,
                limit=limit,
                similarity_threshold=similarity_threshold,
                search_filter=search_filter,
                use_cache=True,
                include_metadata=True
            )
            
        except Exception as e:
            logger.error(f"Find similar entities failed: {e}")
            return VectorSearchResponse(
                results=[],
                query_id=str(uuid4()),
                total_candidates=0,
                search_time_ms=0,
                similarity_threshold=similarity_threshold,
                search_metadata={"error": str(e)},
                cache_hit=False
            )
    
    async def _perform_vector_search(
        self,
        query_embedding: List[float],
        tenant_id: Optional[str],
        limit: int,
        similarity_threshold: float,
        similarity_metric: str,
        search_filter: Optional[SearchFilter],
        include_metadata: bool
    ) -> List[VectorSearchResult]:
        """Perform the actual vector search in the database"""
        
        try:
            # Build the SQL query based on similarity metric
            if similarity_metric == "cosine":
                distance_op = "<=>"
                # For cosine similarity, smaller distances mean higher similarity
                # Convert threshold to distance (1 - similarity)
                distance_threshold = 1 - similarity_threshold
            elif similarity_metric == "l2":
                distance_op = "<->"
                # For L2 distance, we'll use a reasonable distance threshold
                distance_threshold = 2.0 * (1 - similarity_threshold)
            else:
                raise ValueError(f"Unsupported similarity metric: {similarity_metric}")
            
            # Build base query
            where_conditions = []
            # Convert embedding list to JSON string for pgvector compatibility
            embedding_str = json.dumps(query_embedding)
            params = [embedding_str, limit]
            param_count = 2
            
            # Add tenant filtering
            if tenant_id:
                param_count += 1
                where_conditions.append(f"e.tenant_id = ${param_count}")
                params.append(tenant_id)
            
            # Apply search filters
            if search_filter:
                if search_filter.entity_types:
                    param_count += 1
                    where_conditions.append(f"e.entity_type = ANY(${param_count})")
                    params.append(search_filter.entity_types)

                if search_filter.embedding_models:
                    param_count += 1
                    where_conditions.append(f"e.embedding_model = ANY(${param_count})")
                    params.append(search_filter.embedding_models)
                
                if search_filter.created_after:
                    param_count += 1
                    where_conditions.append(f"e.created_at >= ${param_count}")
                    params.append(search_filter.created_after)

                if search_filter.created_before:
                    param_count += 1
                    where_conditions.append(f"e.created_at <= ${param_count}")
                    params.append(search_filter.created_before)
                
                if search_filter.exclude_entity_ids:
                    param_count += 1
                    where_conditions.append(f"e.entity_id != ALL(${param_count})")
                    params.append(search_filter.exclude_entity_ids)
                
                if include_metadata and search_filter.metadata_filters:
                    metadata_column_map = {
                        "status": "p.status",
                        "experience_level": "p.experience_level",
                        "location_city": "p.location_city",
                        "location_state": "p.location_state",
                        "location_country": "p.location_country",
                        "email": "p.email",
                    }
                    for key, value in search_filter.metadata_filters.items():
                        column_ref = metadata_column_map.get(key)
                        if not column_ref:
                            continue
                        param_count += 1
                        where_conditions.append(f"{column_ref} = ${param_count}")
                        params.append(str(value))
            
            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""

            # Construct the full query with JOIN to profiles table for metadata
            if include_metadata:
                # Query with LEFT JOIN to fetch profile data
                query = f"""
                    SELECT
                        e.entity_id,
                        e.entity_type,
                        e.embedding {distance_op} $1 AS distance,
                        e.tenant_id,
                        e.embedding_model,
                        e.created_at,
                        -- Profile metadata from profiles table
                        p.name,
                        p.email,
                        p.phone,
                        p.location_city,
                        p.location_state,
                        p.location_country,
                        p.normalized_skills,
                        p.searchable_text,
                        p.status,
                        p.experience_level,
                        p.profile_data
                    FROM embeddings e
                    LEFT JOIN profiles p ON e.entity_id::uuid = p.id AND e.entity_type = 'profile'
                    {where_clause}
                    ORDER BY e.embedding {distance_op} $1
                    LIMIT $2
                """
            else:
                # Minimal query without metadata
                query = f"""
                    SELECT
                        entity_id,
                        entity_type,
                        embedding {distance_op} $1 AS distance,
                        tenant_id,
                        embedding_model,
                        created_at
                    FROM embeddings
                    {where_clause}
                    ORDER BY embedding {distance_op} $1
                    LIMIT $2
                """

            async with self.db_adapter.get_connection() as conn:
                rows = await conn.fetch(query, *params)

                # Convert results to VectorSearchResult objects
                results = []
                for row in rows:
                    # Calculate similarity from distance based on metric
                    if similarity_metric == "cosine":
                        similarity_score = 1 - row['distance']
                    else:  # L2
                        # Convert L2 distance to similarity (0-1 scale)
                        similarity_score = max(0, 1 - (row['distance'] / 4.0))

                    # Apply similarity threshold
                    if similarity_score >= similarity_threshold:
                        # Build metadata from joined profile data
                        metadata = {}
                        content_preview = None

                        if include_metadata and row.get('name'):
                            # Extract profile data from joined columns
                            metadata = {
                                "name": row['name'],
                                "email": row['email'],
                                "phone": row.get('phone'),
                                "location": f"{row['location_city']}, {row['location_country']}" if row.get('location_city') and row.get('location_country') else row.get('location_city') or row.get('location_country'),
                                "location_city": row.get('location_city'),
                                "location_state": row.get('location_state'),
                                "location_country": row.get('location_country'),
                                "skills": row.get('normalized_skills') or [],
                                "status": row.get('status'),
                                "experience_level": row.get('experience_level')
                            }

                            # Extract title and summary from profile_data JSONB
                            if row.get('profile_data'):
                                profile_data = row['profile_data']
                                if isinstance(profile_data, dict):
                                    metadata["title"] = profile_data.get("headline") or profile_data.get("title")
                                    metadata["summary"] = profile_data.get("summary")
                                    if profile_data.get("highest_degree"):
                                        metadata["highest_degree"] = profile_data.get("highest_degree")
                                    education = profile_data.get("education")
                                    if education:
                                        metadata["education"] = education
                                    compensation = profile_data.get("compensation")
                                    if isinstance(compensation, dict):
                                        metadata["expected_salary"] = compensation.get("expected_salary")
                                    total_exp = profile_data.get("total_experience_years")
                                    if total_exp is not None:
                                        metadata["total_experience_years"] = total_exp

                            # Set content preview from searchable_text
                            if row.get('searchable_text'):
                                content_preview = row['searchable_text'][:500]

                        result = VectorSearchResult(
                            entity_id=row['entity_id'],
                            entity_type=row['entity_type'],
                            similarity_score=float(similarity_score),
                            distance=float(row['distance']),
                            metadata=metadata,
                            content_preview=content_preview,
                            tenant_id=str(row['tenant_id']) if row['tenant_id'] else None,
                            embedding_model=row.get('embedding_model'),
                            created_at=row.get('created_at')
                        )
                        results.append(result)

                return results
                
        except Exception as e:
            logger.error(f"Vector search database query failed: {e}")
            raise
    
    async def _get_or_generate_embedding(
        self,
        text: str,
        use_cache: bool = True
    ) -> List[float]:
        """Get or generate embedding for text with caching"""
        
        try:
            # Check cache first
            if use_cache and self.cache_manager:
                cache_key = f"embedding:{hashlib.sha256(text.encode()).hexdigest()[:16]}"
                cached_embedding = await self.cache_manager.get(
                    cache_key,
                    content_type="embedding"
                )
                if cached_embedding and isinstance(cached_embedding, list):
                    logger.debug("Retrieved embedding from cache")
                    return cached_embedding
            
            # Generate new embedding
            embedding = await self.embedding_service.generate_embedding(text)
            self._stats["embedding_generations"] += 1
            
            # Cache the embedding if enabled
            if use_cache and self.cache_manager:
                await self.cache_manager.set(
                    key=cache_key,
                    value=embedding,
                    ttl=self.vector_cache_ttl,
                    content_type="embedding"
                )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    async def _get_cached_search_results(
        self,
        query_embedding: List[float],
        tenant_id: Optional[str],
        similarity_threshold: float,
        search_filter: Optional[SearchFilter]
    ) -> Optional[VectorSearchResponse]:
        """Retrieve cached search results for similar queries"""
        
        if not self.cache_manager:
            return None
        
        try:
            # Create cache key based on query parameters
            cache_data = {
                "embedding_hash": hashlib.sha256(
                    json.dumps(query_embedding, sort_keys=True).encode()
                ).hexdigest()[:16],
                "tenant_id": tenant_id,
                "threshold": similarity_threshold,
                "filters": asdict(search_filter) if search_filter else None
            }
            
            cache_key = f"vector_search:{hashlib.sha256(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()[:16]}"
            
            cached_response = await self.cache_manager.get(
                cache_key,
                content_type="vector_search_results"
            )
            
            if cached_response and isinstance(cached_response, dict):
                logger.debug("Retrieved vector search results from cache")
                # Reconstruct VectorSearchResponse from cached data
                cached_response["cache_hit"] = True
                return VectorSearchResponse(**cached_response)
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cached search results: {e}")
            return None
    
    async def _cache_search_results(
        self,
        response: VectorSearchResponse,
        query_embedding: List[float],
        tenant_id: Optional[str]
    ) -> None:
        """Cache search results for future queries"""
        
        if not self.cache_manager:
            return
        
        try:
            # Create cache key
            cache_data = {
                "embedding_hash": hashlib.sha256(
                    json.dumps(query_embedding, sort_keys=True).encode()
                ).hexdigest()[:16],
                "tenant_id": tenant_id,
                "threshold": response.similarity_threshold
            }
            
            cache_key = f"vector_search:{hashlib.sha256(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()[:16]}"
            
            # Convert response to cacheable format
            cacheable_response = asdict(response)
            cacheable_response["cache_hit"] = False  # Reset for caching
            
            await self.cache_manager.set(
                key=cache_key,
                value=cacheable_response,
                ttl=self.vector_cache_ttl,
                content_type="vector_search_results"
            )
            
            logger.debug("Cached vector search results")
            
        except Exception as e:
            logger.warning(f"Failed to cache search results: {e}")
    
    def _update_search_stats(self, search_time_ms: int, results_count: int) -> None:
        """Update internal search statistics"""
        self._stats["searches_performed"] += 1
        self._stats["total_search_time_ms"] += search_time_ms
        self._stats["results_returned"] += results_count
        
        if self._stats["searches_performed"] > 0:
            self._stats["average_search_time_ms"] = (
                self._stats["total_search_time_ms"] / self._stats["searches_performed"]
            )
    
    async def get_search_analytics(
        self,
        tenant_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get search analytics and performance metrics"""
        
        try:
            analytics = {
                "service_stats": self._stats.copy(),
                "configuration": {
                    "default_similarity_threshold": self.default_similarity_threshold,
                    "max_results_per_search": self.max_results_per_search,
                    "cache_enabled": self.cache_manager is not None,
                    "cache_ttl": self.vector_cache_ttl
                }
            }
            
            # Get database-level analytics if available
            if tenant_id:
                async with self.db_adapter.get_connection() as conn:
                    # Embedding statistics
                    embedding_stats = await conn.fetchrow("""
                        SELECT 
                            COUNT(*) as total_embeddings,
                            COUNT(DISTINCT entity_type) as entity_types,
                            COUNT(DISTINCT embedding_model) as models_used,
                            AVG(CASE WHEN content_hash IS NOT NULL THEN 1.0 ELSE 0.0 END) as dedup_rate
                        FROM embeddings
                        WHERE tenant_id = $1 
                        OR tenant_id IS NULL
                    """, tenant_id)
                    
                    analytics["embedding_stats"] = dict(embedding_stats) if embedding_stats else {}
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get search analytics: {e}")
            return {"error": str(e), "service_stats": self._stats.copy()}
    
    async def optimize_search_performance(self) -> Dict[str, Any]:
        """Optimize search performance by updating vector indexes"""
        
        try:
            optimization_results = {}
            
            async with self.db_adapter.get_connection() as conn:
                # Update index statistics
                await conn.execute("ANALYZE embeddings")
                
                # Get current index usage statistics
                index_stats = await conn.fetch("""
                    SELECT 
                        indexname,
                        idx_tup_read,
                        idx_tup_fetch
                    FROM pg_stat_user_indexes 
                    WHERE relname = 'embeddings'
                """)
                
                optimization_results["index_stats"] = [dict(row) for row in index_stats]
                optimization_results["optimization_completed"] = datetime.now().isoformat()
                
            logger.info("Vector search performance optimization completed")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Search performance optimization failed: {e}")
            return {"error": str(e)}
    
    async def check_health(self) -> Dict[str, Any]:
        """Check vector search service health"""
        
        try:
            start_time = datetime.now()
            
            # Test database connectivity
            async with self.db_adapter.get_connection() as conn:
                # Test vector extension
                await conn.fetchval("SELECT 1")
                
                # Test embedding table access
                embedding_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM embeddings LIMIT 1"
                )
                
            # Test embedding service
            embedding_service_health = "unknown"
            try:
                if hasattr(self.embedding_service, 'check_health'):
                    embedding_health = await self.embedding_service.check_health()
                    embedding_service_health = embedding_health.get('status', 'unknown')
                else:
                    embedding_service_health = "available"
            except Exception:
                embedding_service_health = "error"
            
            # Test cache service
            cache_service_health = "disabled"
            if self.cache_manager:
                try:
                    cache_health = await self.cache_manager.check_health()
                    cache_service_health = cache_health.get('status', 'unknown')
                except Exception:
                    cache_service_health = "error"
            
            health_check_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return {
                "status": "healthy",
                "vector_search_service": "operational",
                "database_connectivity": "operational",
                "embedding_service": embedding_service_health,
                "cache_service": cache_service_health,
                "embeddings_available": embedding_count > 0 if embedding_count else False,
                "stats": self._stats.copy(),
                "health_check_time_ms": health_check_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
