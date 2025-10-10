"""
Hybrid Search Service with Intelligent Result Fusion

Advanced search combining multiple search modalities:
- Traditional text search using PostgreSQL full-text search
- Semantic vector search using pgvector embeddings
- Intelligent result fusion with configurable weights
- Query expansion and enhancement
- Multi-stage search optimization
- Performance monitoring and caching
- Advanced filtering and ranking
"""

from __future__ import annotations

import json
import hashlib
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set, TYPE_CHECKING
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from uuid import uuid4

from app.core.config import get_settings
from app.domain.interfaces import IHealthCheck
from app.infrastructure.ai.cache_manager import UUIDEncoder

# Import infrastructure types only for type checking
if TYPE_CHECKING:
    from app.infrastructure.search.vector_search import VectorSearchService, VectorSearchResult, SearchFilter
    from app.infrastructure.search.query_processor import QueryProcessor, ProcessedQuery
    from app.infrastructure.ai.cache_manager import CacheManager
    from app.infrastructure.adapters.postgres_adapter import PostgresAdapter

logger = structlog.get_logger(__name__)


class SearchMode(Enum):
    """Search mode configuration"""
    TEXT_ONLY = "text_only"
    VECTOR_ONLY = "vector_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"  # Automatically choose best approach


class FusionMethod(Enum):
    """Result fusion methods"""
    WEIGHTED_AVERAGE = "weighted_average"
    RANK_FUSION = "rank_fusion"
    RECIPROCAL_RANK = "reciprocal_rank"
    BAYESIAN_FUSION = "bayesian_fusion"


@dataclass
class TextSearchResult:
    """Individual text search result"""
    entity_id: str
    entity_type: str
    relevance_score: float
    rank: int
    highlighted_content: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class HybridSearchResult:
    """Unified search result from hybrid search"""
    entity_id: str
    entity_type: str
    final_score: float
    text_score: Optional[float] = None
    vector_score: Optional[float] = None
    rank: int = 0
    source_methods: List[str] = None  # Which search methods contributed
    content_preview: Optional[str] = None
    metadata: Dict[str, Any] = None
    explanation: Optional[Dict[str, Any]] = None  # Scoring explanation


@dataclass
class HybridSearchResponse:
    """Complete hybrid search response"""
    results: List[HybridSearchResult]
    query_id: str
    search_mode: SearchMode
    fusion_method: FusionMethod
    total_candidates: int
    text_results_count: int
    vector_results_count: int
    search_time_ms: int
    fusion_time_ms: int
    cache_hit: bool = False
    search_metadata: Dict[str, Any] = None


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search behavior"""
    text_weight: float = 0.4
    vector_weight: float = 0.6
    fusion_method: FusionMethod = FusionMethod.WEIGHTED_AVERAGE
    text_similarity_threshold: float = 0.1
    vector_similarity_threshold: float = 0.3  # Lowered from 0.7 to allow more results (actual similarities ~0.47)
    max_text_results: int = 100
    max_vector_results: int = 100
    enable_query_expansion: bool = True
    enable_result_diversification: bool = True
    enable_caching: bool = True


class HybridSearchService(IHealthCheck):
    """
    Advanced hybrid search service combining text and vector search.
    
    Features:
    - Multi-modal search with intelligent result fusion
    - Configurable search weights and thresholds
    - Query expansion and enhancement
    - Result deduplication and diversification  
    - Performance optimization and caching
    - Adaptive search mode selection
    - Comprehensive analytics and monitoring
    - Multi-tenant isolation and security
    """
    
    def __init__(
        self,
        db_adapter: PostgresAdapter,
        vector_search_service: VectorSearchService,
        query_processor: QueryProcessor,
        cache_manager: Optional[CacheManager] = None,
        default_config: Optional[HybridSearchConfig] = None
    ):
        self.settings = get_settings()
        self.db_adapter = db_adapter
        self.vector_search_service = vector_search_service
        self.query_processor = query_processor
        self.cache_manager = cache_manager
        
        # Default configuration
        self.config = default_config or HybridSearchConfig()
        
        # Performance tracking
        self._stats = {
            "searches_performed": 0,
            "text_searches": 0,
            "vector_searches": 0,
            "hybrid_searches": 0,
            "cache_hits": 0,
            "total_search_time_ms": 0,
            "average_search_time_ms": 0,
            "fusion_operations": 0,
            "query_expansions": 0,
            "errors": 0
        }
        
        # Caching configuration
        self.search_cache_ttl = 1800  # 30 minutes
        
    async def hybrid_search(
        self,
        query: str,
        tenant_id: Optional[str] = None,
        limit: int = 20,
        search_mode: SearchMode = SearchMode.HYBRID,
        config: Optional[HybridSearchConfig] = None,
        search_filter: Optional[SearchFilter] = None,
        use_cache: bool = True,
        include_explanations: bool = False
    ) -> HybridSearchResponse:
        """
        Perform comprehensive hybrid search combining text and vector search.
        
        Args:
            query: Search query text
            tenant_id: Tenant ID for data isolation
            limit: Maximum number of results to return
            search_mode: Search mode configuration
            config: Custom search configuration
            search_filter: Additional search filters
            use_cache: Enable result caching
            include_explanations: Include scoring explanations
            
        Returns:
            HybridSearchResponse with unified results and analytics
        """
        start_time = datetime.now()
        query_id = str(uuid4())
        
        try:
            # Use provided config or default
            search_config = config or self.config
            
            # Check cache first
            if use_cache:
                cached_response = await self._get_cached_search_results(
                    query, tenant_id, search_mode, search_config, search_filter
                )
                if cached_response:
                    self._stats["cache_hits"] += 1
                    return cached_response
            
            # Process and expand query
            processed_query = None
            if search_config.enable_query_expansion:
                processed_query = await self.query_processor.process_query(
                    query=query,
                    tenant_id=tenant_id,
                    expand_query=True,
                    cache_results=use_cache
                )
                self._stats["query_expansions"] += 1
            
            # Determine optimal search strategy
            if search_mode == SearchMode.ADAPTIVE:
                search_mode = await self._determine_optimal_search_mode(
                    query, processed_query, tenant_id
                )
            
            # Execute searches based on mode
            text_results = []
            vector_results = []
            
            search_tasks = []
            
            if search_mode in [SearchMode.TEXT_ONLY, SearchMode.HYBRID]:
                search_tasks.append(
                    self._perform_text_search(
                        query, processed_query, tenant_id, search_config, search_filter
                    )
                )
            
            if search_mode in [SearchMode.VECTOR_ONLY, SearchMode.HYBRID]:
                search_tasks.append(
                    self._perform_vector_search(
                        query, processed_query, tenant_id, search_config, search_filter
                    )
                )
            
            # Execute searches concurrently
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Process search results
            for i, result in enumerate(search_results):
                if isinstance(result, Exception):
                    logger.error(f"Search task {i} failed: {result}")
                    continue
                
                if search_mode in [SearchMode.TEXT_ONLY, SearchMode.HYBRID] and i == 0:
                    text_results = result
                elif search_mode in [SearchMode.VECTOR_ONLY, SearchMode.HYBRID]:
                    if (search_mode == SearchMode.VECTOR_ONLY) or (search_mode == SearchMode.HYBRID and i == 1):
                        vector_results = result
            
            # Fusion start time
            fusion_start = datetime.now()
            
            # Fuse results based on search mode
            if search_mode == SearchMode.TEXT_ONLY:
                fused_results = await self._convert_text_results(text_results, include_explanations)
            elif search_mode == SearchMode.VECTOR_ONLY:
                fused_results = await self._convert_vector_results(vector_results, include_explanations)
            else:  # HYBRID
                fused_results = await self._fuse_search_results(
                    text_results, vector_results, search_config, include_explanations
                )
                self._stats["fusion_operations"] += 1
            
            fusion_time_ms = int((datetime.now() - fusion_start).total_seconds() * 1000)
            
            # Apply result diversification if enabled
            if search_config.enable_result_diversification:
                fused_results = await self._diversify_results(fused_results, tenant_id)
            
            # Limit and rank results
            final_results = sorted(fused_results, key=lambda x: x.final_score, reverse=True)[:limit]
            
            # Add ranking information
            for i, result in enumerate(final_results):
                result.rank = i + 1
            
            # Calculate total search time
            total_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Create response
            response = HybridSearchResponse(
                results=final_results,
                query_id=query_id,
                search_mode=search_mode,
                fusion_method=search_config.fusion_method,
                total_candidates=len(fused_results),
                text_results_count=len(text_results),
                vector_results_count=len(vector_results),
                search_time_ms=total_time_ms,
                fusion_time_ms=fusion_time_ms,
                cache_hit=False,
                search_metadata={
                    "original_query": query,
                    "processed_query": processed_query.normalized_query if processed_query else query,
                    "query_expansion_used": processed_query is not None,
                    "tenant_id": tenant_id,
                    "config": asdict(search_config)
                }
            )
            
            # Cache results if enabled
            if use_cache and len(final_results) > 0:
                await self._cache_search_results(
                    response, query, tenant_id, search_mode, search_config
                )
            
            # Update statistics
            self._update_search_stats(search_mode, total_time_ms, len(final_results))
            
            logger.info(
                "Hybrid search completed",
                query_id=query_id,
                search_mode=search_mode.value,
                results_count=len(final_results),
                total_time_ms=total_time_ms,
                fusion_time_ms=fusion_time_ms,
                tenant_id=tenant_id
            )
            
            return response
            
        except Exception as e:
            self._stats["errors"] += 1
            search_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            logger.error(
                "Hybrid search failed",
                query_id=query_id,
                error=str(e),
                search_time_ms=search_time_ms,
                tenant_id=tenant_id
            )
            
            # Return empty response on error
            return HybridSearchResponse(
                results=[],
                query_id=query_id,
                search_mode=search_mode,
                fusion_method=self.config.fusion_method,
                total_candidates=0,
                text_results_count=0,
                vector_results_count=0,
                search_time_ms=search_time_ms,
                fusion_time_ms=0,
                cache_hit=False,
                search_metadata={"error": str(e)}
            )
    
    async def _perform_text_search(
        self,
        query: str,
        processed_query: Optional[ProcessedQuery],
        tenant_id: Optional[str],
        config: HybridSearchConfig,
        search_filter: Optional[SearchFilter]
    ) -> List[TextSearchResult]:
        """Perform full-text search using PostgreSQL text search capabilities"""
        
        try:
            self._stats["text_searches"] += 1
            
            # Use processed query if available
            search_text = processed_query.normalized_query if processed_query else query
            
            # Build text search query with expanded terms
            if processed_query and processed_query.expansion.expanded_terms:
                # Combine original query with expanded terms
                expanded_terms = " | ".join(processed_query.expansion.expanded_terms[:10])
                search_text = f"{search_text} | {expanded_terms}"
            
            # Build filter conditions
            where_conditions = []
            params = [search_text]
            param_count = 1
            
            # Add tenant filtering
            if tenant_id:
                param_count += 1
                where_conditions.append(f"tenant_id = ${param_count}")
                params.append(tenant_id)
            
            # Apply search filters
            if search_filter:
                if search_filter.entity_types:
                    param_count += 1
                    where_conditions.append(f"entity_type = ANY(${param_count})")
                    params.append(search_filter.entity_types)
                
                if search_filter.created_after:
                    param_count += 1
                    where_conditions.append(f"created_at >= ${param_count}")
                    params.append(search_filter.created_after)
                
                if search_filter.created_before:
                    param_count += 1
                    where_conditions.append(f"created_at <= ${param_count}")
                    params.append(search_filter.created_before)
            
            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
            
            # Construct text search query
            # Note: This assumes you have a text search setup on your entities
            # You might need to adjust table/column names based on your schema
            query_sql = f"""
                SELECT
                    id::text as entity_id,
                    'profile' as entity_type,
                    ts_rank_cd(search_vector, plainto_tsquery('english', $1)) as relevance_score,
                    ts_headline('english', content, plainto_tsquery('english', $1)) as highlighted_content,
                    metadata
                FROM (
                    SELECT
                        id,
                        to_tsvector('english',
                            COALESCE(searchable_text, '') || ' ' ||
                            COALESCE(name, '') || ' ' ||
                            COALESCE(array_to_string(normalized_skills, ' '), '')
                        ) as search_vector,
                        COALESCE(searchable_text, name) as content,
                        jsonb_build_object(
                            'name', name,
                            'email', email,
                            'skills', normalized_skills,
                            'location_city', location_city,
                            'location_country', location_country
                        ) as metadata,
                        tenant_id,
                        created_at
                    FROM profiles
                    WHERE (searchable_text IS NOT NULL OR name IS NOT NULL)
                ) searchable_profiles
                {where_clause}
                AND plainto_tsquery('english', $1) @@ search_vector
                ORDER BY relevance_score DESC
                LIMIT {config.max_text_results}
            """
            
            async with self.db_adapter.get_connection() as conn:
                rows = await conn.fetch(query_sql, *params)
                
                # Convert to TextSearchResult objects
                results = []
                for i, row in enumerate(rows):
                    if row['relevance_score'] >= config.text_similarity_threshold:
                        result = TextSearchResult(
                            entity_id=row['entity_id'],
                            entity_type=row['entity_type'],
                            relevance_score=float(row['relevance_score']),
                            rank=i + 1,
                            highlighted_content=row.get('highlighted_content'),
                            metadata=row.get('metadata', {})
                        )
                        results.append(result)
                
                logger.debug(f"Text search returned {len(results)} results")
                return results
                
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []
    
    async def _perform_vector_search(
        self,
        query: str,
        processed_query: Optional[ProcessedQuery],
        tenant_id: Optional[str],
        config: HybridSearchConfig,
        search_filter: Optional[SearchFilter]
    ) -> List[VectorSearchResult]:
        """Perform vector similarity search"""
        
        try:
            self._stats["vector_searches"] += 1
            
            # Use expanded query if available
            search_text = query
            if processed_query and processed_query.expansion.expanded_terms:
                # Combine with expanded terms for better semantic understanding
                expanded_text = " ".join(processed_query.expansion.expanded_terms[:5])
                search_text = f"{query} {expanded_text}"
            
            # Perform vector search
            vector_response = await self.vector_search_service.similarity_search(
                query_text=search_text,
                tenant_id=tenant_id,
                limit=config.max_vector_results,
                similarity_threshold=config.vector_similarity_threshold,
                search_filter=search_filter,
                use_cache=config.enable_caching,
                include_metadata=True
            )
            
            logger.debug(f"Vector search returned {len(vector_response.results)} results")
            return vector_response.results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def _fuse_search_results(
        self,
        text_results: List[TextSearchResult],
        vector_results: List[VectorSearchResult],
        config: HybridSearchConfig,
        include_explanations: bool
    ) -> List[HybridSearchResult]:
        """Intelligently fuse text and vector search results"""
        
        try:
            # Create entity maps for efficient lookup
            text_entities = {result.entity_id: result for result in text_results}
            vector_entities = {result.entity_id: result for result in vector_results}
            
            # Get all unique entities
            all_entity_ids = set(text_entities.keys()) | set(vector_entities.keys())
            
            fused_results = []
            
            for entity_id in all_entity_ids:
                text_result = text_entities.get(entity_id)
                vector_result = vector_entities.get(entity_id)
                
                # Calculate fused score based on fusion method
                final_score = 0.0
                text_score = text_result.relevance_score if text_result else 0.0
                vector_score = vector_result.similarity_score if vector_result else 0.0
                source_methods = []
                explanation = {}
                
                if text_result:
                    source_methods.append("text_search")
                if vector_result:
                    source_methods.append("vector_search")
                
                if config.fusion_method == FusionMethod.WEIGHTED_AVERAGE:
                    # Weighted average of scores
                    if text_result and vector_result:
                        final_score = (text_score * config.text_weight + 
                                     vector_score * config.vector_weight)
                        if include_explanations:
                            explanation = {
                                "method": "weighted_average",
                                "text_weight": config.text_weight,
                                "vector_weight": config.vector_weight,
                                "text_contribution": text_score * config.text_weight,
                                "vector_contribution": vector_score * config.vector_weight
                            }
                    elif text_result:
                        final_score = text_score * config.text_weight
                    else:
                        final_score = vector_score * config.vector_weight
                
                elif config.fusion_method == FusionMethod.RECIPROCAL_RANK:
                    # Reciprocal rank fusion
                    rr_score = 0.0
                    if text_result:
                        rr_score += 1.0 / (text_result.rank + 60)  # k=60 is common
                    if vector_result:
                        # Create rank from similarity score
                        vector_rank = 1 / (vector_result.similarity_score + 0.001)
                        rr_score += 1.0 / (vector_rank + 60)
                    final_score = rr_score
                    
                    if include_explanations:
                        explanation = {
                            "method": "reciprocal_rank",
                            "text_rank": text_result.rank if text_result else None,
                            "vector_pseudo_rank": vector_rank if vector_result else None,
                            "combined_rr_score": rr_score
                        }
                
                elif config.fusion_method == FusionMethod.RANK_FUSION:
                    # Simple rank fusion
                    rank_score = 0.0
                    if text_result:
                        rank_score += (1.0 - (text_result.rank / len(text_results)))
                    if vector_result:
                        # Convert similarity to rank-like score
                        rank_score += vector_result.similarity_score
                    final_score = rank_score / (2 if text_result and vector_result else 1)
                
                # Get entity metadata (prefer vector result for richer metadata)
                entity_type = (vector_result.entity_type if vector_result 
                             else text_result.entity_type)
                metadata = {}
                content_preview = None
                
                if vector_result:
                    metadata.update(vector_result.metadata)
                    content_preview = vector_result.content_preview
                if text_result:
                    if text_result.metadata:
                        metadata.update(text_result.metadata)
                    if text_result.highlighted_content:
                        content_preview = text_result.highlighted_content
                
                # Create fused result
                fused_result = HybridSearchResult(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    final_score=final_score,
                    text_score=text_score if text_result else None,
                    vector_score=vector_score if vector_result else None,
                    source_methods=source_methods,
                    content_preview=content_preview,
                    metadata=metadata,
                    explanation=explanation if include_explanations else None
                )
                
                fused_results.append(fused_result)
            
            logger.debug(f"Fused {len(fused_results)} unique entities from search results")
            return fused_results
            
        except Exception as e:
            logger.error(f"Result fusion failed: {e}")
            return []
    
    async def _convert_text_results(
        self,
        text_results: List[TextSearchResult],
        include_explanations: bool
    ) -> List[HybridSearchResult]:
        """Convert text search results to hybrid format"""
        
        hybrid_results = []
        for result in text_results:
            hybrid_result = HybridSearchResult(
                entity_id=result.entity_id,
                entity_type=result.entity_type,
                final_score=result.relevance_score,
                text_score=result.relevance_score,
                vector_score=None,
                source_methods=["text_search"],
                content_preview=result.highlighted_content,
                metadata=result.metadata,
                explanation={"method": "text_only"} if include_explanations else None
            )
            hybrid_results.append(hybrid_result)
        
        return hybrid_results
    
    async def _convert_vector_results(
        self,
        vector_results: List[VectorSearchResult],
        include_explanations: bool
    ) -> List[HybridSearchResult]:
        """Convert vector search results to hybrid format"""
        
        hybrid_results = []
        for result in vector_results:
            hybrid_result = HybridSearchResult(
                entity_id=result.entity_id,
                entity_type=result.entity_type,
                final_score=result.similarity_score,
                text_score=None,
                vector_score=result.similarity_score,
                source_methods=["vector_search"],
                content_preview=result.content_preview,
                metadata=result.metadata,
                explanation={"method": "vector_only"} if include_explanations else None
            )
            hybrid_results.append(hybrid_result)
        
        return hybrid_results
    
    async def _diversify_results(
        self,
        results: List[HybridSearchResult],
        tenant_id: Optional[str]
    ) -> List[HybridSearchResult]:
        """Apply result diversification to avoid over-representation of similar entities"""
        
        try:
            if len(results) <= 10:
                return results  # No diversification needed for small result sets
            
            # Group by entity type
            type_groups = {}
            for result in results:
                if result.entity_type not in type_groups:
                    type_groups[result.entity_type] = []
                type_groups[result.entity_type].append(result)
            
            # Apply diversification within each group
            diversified_results = []
            for entity_type, group_results in type_groups.items():
                # Sort by score and take top results from each group
                group_results.sort(key=lambda x: x.final_score, reverse=True)
                
                # Take proportional results from each type (minimum 2 per type)
                group_limit = max(2, len(results) // (len(type_groups) * 2))
                diversified_results.extend(group_results[:group_limit])
            
            # Sort final results by score
            diversified_results.sort(key=lambda x: x.final_score, reverse=True)
            
            logger.debug(f"Applied diversification: {len(results)} -> {len(diversified_results)}")
            return diversified_results
            
        except Exception as e:
            logger.warning(f"Result diversification failed: {e}")
            return results
    
    async def _determine_optimal_search_mode(
        self,
        query: str,
        processed_query: Optional[ProcessedQuery],
        tenant_id: Optional[str]
    ) -> SearchMode:
        """Determine optimal search mode based on query characteristics"""
        
        try:
            # Simple heuristics for search mode selection
            query_length = len(query.split())
            
            # Very short queries often work better with vector search
            if query_length <= 2:
                return SearchMode.VECTOR_ONLY
            
            # Long, descriptive queries benefit from text search
            if query_length > 10:
                return SearchMode.TEXT_ONLY
            
            # Check if query contains specific technical terms
            if processed_query and processed_query.expansion.primary_skills:
                # Skill-based queries work well with hybrid approach
                return SearchMode.HYBRID
            
            # Default to hybrid for balanced coverage
            return SearchMode.HYBRID
            
        except Exception as e:
            logger.warning(f"Failed to determine optimal search mode: {e}")
            return SearchMode.HYBRID
    
    async def _get_cached_search_results(
        self,
        query: str,
        tenant_id: Optional[str],
        search_mode: SearchMode,
        config: HybridSearchConfig,
        search_filter: Optional[SearchFilter]
    ) -> Optional[HybridSearchResponse]:
        """Retrieve cached search results"""
        
        if not self.cache_manager:
            return None
        
        try:
            # Create cache key
            cache_data = {
                "query": query.lower().strip(),
                "tenant_id": tenant_id,
                "search_mode": search_mode.value,
                "config": {
                    "text_weight": config.text_weight,
                    "vector_weight": config.vector_weight,
                    "fusion_method": config.fusion_method.value,
                    "text_threshold": config.text_similarity_threshold,
                    "vector_threshold": config.vector_similarity_threshold
                },
                "filters": asdict(search_filter) if search_filter else None
            }
            
            cache_key = f"hybrid_search:{hashlib.sha256(json.dumps(cache_data, sort_keys=True, cls=UUIDEncoder).encode()).hexdigest()[:16]}"

            cached_response = await self.cache_manager.get(
                cache_key,
                content_type="hybrid_search_results"
            )
            
            if cached_response and isinstance(cached_response, dict):
                logger.debug("Retrieved hybrid search results from cache")
                cached_response["cache_hit"] = True
                return HybridSearchResponse(**cached_response)
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cached search results: {e}")
            return None
    
    async def _cache_search_results(
        self,
        response: HybridSearchResponse,
        query: str,
        tenant_id: Optional[str],
        search_mode: SearchMode,
        config: HybridSearchConfig
    ) -> None:
        """Cache search results for future queries"""
        
        if not self.cache_manager:
            return
        
        try:
            # Create cache key
            cache_data = {
                "query": query.lower().strip(),
                "tenant_id": tenant_id,
                "search_mode": search_mode.value,
                "config": {
                    "text_weight": config.text_weight,
                    "vector_weight": config.vector_weight,
                    "fusion_method": config.fusion_method.value,
                    "text_threshold": config.text_similarity_threshold,
                    "vector_threshold": config.vector_similarity_threshold
                }
            }
            
            cache_key = f"hybrid_search:{hashlib.sha256(json.dumps(cache_data, sort_keys=True, cls=UUIDEncoder).encode()).hexdigest()[:16]}"

            # Convert response to cacheable format
            cacheable_response = asdict(response)
            cacheable_response["cache_hit"] = False  # Reset for caching
            
            await self.cache_manager.set(
                key=cache_key,
                value=cacheable_response,
                ttl=self.search_cache_ttl,
                content_type="hybrid_search_results"
            )
            
            logger.debug("Cached hybrid search results")
            
        except Exception as e:
            logger.warning(f"Failed to cache search results: {e}")
    
    def _update_search_stats(
        self,
        search_mode: SearchMode,
        search_time_ms: int,
        results_count: int
    ) -> None:
        """Update search statistics"""
        self._stats["searches_performed"] += 1
        
        if search_mode == SearchMode.HYBRID:
            self._stats["hybrid_searches"] += 1
        
        self._stats["total_search_time_ms"] += search_time_ms
        
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
        """Get comprehensive search analytics"""
        
        try:
            analytics = {
                "service_stats": self._stats.copy(),
                "configuration": {
                    "default_config": asdict(self.config),
                    "cache_enabled": self.cache_manager is not None,
                    "cache_ttl": self.search_cache_ttl
                },
                "performance_metrics": {
                    "average_search_time_ms": self._stats["average_search_time_ms"],
                    "cache_hit_rate": (
                        self._stats["cache_hits"] / max(1, self._stats["searches_performed"])
                    ),
                    "fusion_efficiency": (
                        self._stats["fusion_operations"] / max(1, self._stats["hybrid_searches"])
                    )
                }
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get search analytics: {e}")
            return {"error": str(e), "service_stats": self._stats.copy()}
    
    async def check_health(self) -> Dict[str, Any]:
        """Check hybrid search service health"""
        
        try:
            start_time = datetime.now()
            
            # Test vector search service
            vector_health = await self.vector_search_service.check_health()
            
            # Test query processor
            query_health = "unknown"
            try:
                if hasattr(self.query_processor, 'check_health'):
                    qp_health = await self.query_processor.check_health()
                    query_health = qp_health.get('status', 'unknown')
                else:
                    query_health = "available"
            except Exception:
                query_health = "error"
            
            # Test database connectivity
            async with self.db_adapter.get_connection() as conn:
                await conn.fetchval("SELECT 1")
            
            # Test cache service
            cache_health = "disabled"
            if self.cache_manager:
                try:
                    cache_status = await self.cache_manager.check_health()
                    cache_health = cache_status.get('status', 'unknown')
                except Exception:
                    cache_health = "error"
            
            health_check_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return {
                "status": "healthy",
                "hybrid_search_service": "operational",
                "vector_search_service": vector_health.get('status', 'unknown'),
                "query_processor": query_health,
                "database_connectivity": "operational",
                "cache_service": cache_health,
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
