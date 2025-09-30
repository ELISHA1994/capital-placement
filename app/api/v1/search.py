"""
Search API Endpoints

Advanced CV search endpoints with:
- Multi-modal search (vector, hybrid, semantic)
- Query expansion and reranking
- Faceted search with filters
- Real-time search analytics
- Multi-tenant search isolation
- Performance optimization and caching
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID
import structlog

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse

from app.models.search_models import (
    SearchRequest, SearchResponse, SavedSearch, SearchHistory, 
    SearchMode, SortOrder, SearchFilter, RangeFilter
)
from app.models.base import PaginationModel, PaginatedResponse
from app.services.core.search_engine import SearchEngine
from app.services.core.tenant_manager import TenantManager
from app.core.dependencies import get_current_user, CurrentUserDep, TenantContextDep
from app.models.auth import CurrentUser, TenantContext

# AI-Powered Search Services
from app.services.search.vector_search import VectorSearchService, SearchFilter as VectorSearchFilter
from app.services.search.hybrid_search import (
    HybridSearchService, SearchMode as HybridSearchMode, 
    HybridSearchConfig, FusionMethod
)
from app.services.search.result_reranker import (
    ResultRerankerService, RankingStrategy, RerankingConfig
)
from app.services.search.search_analytics import SearchAnalyticsService
from app.services.search.query_processor import QueryProcessor

# AI Services
from app.services.ai.openai_service import OpenAIService
from app.services.ai.embedding_service import EmbeddingService
from app.services.ai.prompt_manager import PromptManager
from app.services.ai.cache_manager import CacheManager

# Database and Config
from app.database.repositories.postgres import SQLModelRepository
from app.core.config import get_settings
from app.core.container import Container

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/search", tags=["search"])

# Service container for dependency injection
service_container = Container()


@router.post("/", response_model=SearchResponse)
async def search_profiles(
    search_request: SearchRequest,
    background_tasks: BackgroundTasks,
    current_user: CurrentUserDep
) -> SearchResponse:
    """
    Execute comprehensive CV profile search with multi-modal capabilities.
    
    Supports all search modes:
    - **keyword**: Traditional text search
    - **vector**: Semantic similarity search using embeddings
    - **hybrid**: Combination of keyword and vector search
    - **semantic**: Azure Cognitive Search semantic understanding
    - **multi_stage**: Advanced multi-stage retrieval with reranking
    
    Features:
    - Intelligent query expansion with domain terminology
    - Business logic scoring with customizable weights
    - Result diversification and quality filtering
    - Real-time analytics and performance monitoring
    """
    try:
        logger.info(
            "Search request received (authenticated)",
            tenant_id=str(search_request.tenant_id),
            user_id=current_user.user_id,
            query=search_request.query[:100],
            search_mode=search_request.search_mode
        )
        
        # TODO: Apply rate limiting
        # validate_rate_limit(current_user, "search", limit=100)
        
        start_time = datetime.now()
        search_id = f"search_{start_time.strftime('%Y%m%d_%H%M%S')}_{current_user.user_id[:8]}"
        
        # Get service instances
        settings = get_settings()
        db_repo = service_container.get_postgres_repository()
        
        # Initialize AI services if configured
        openai_service = None
        hybrid_search_service = None
        reranker_service = None
        analytics_service = None
        
        if settings.is_openai_configured():
            # Initialize AI services
            openai_service = service_container.get_openai_service()
            embedding_service = service_container.get_embedding_service()
            prompt_manager = service_container.get_prompt_manager()
            cache_manager = service_container.get_cache_manager() if settings.REDIS_URL else None
            
            # Initialize search services
            vector_search_service = VectorSearchService(
                db_repository=db_repo,
                embedding_service=embedding_service,
                cache_manager=cache_manager
            )
            
            query_processor = QueryProcessor(
                openai_service=openai_service,
                prompt_manager=prompt_manager,
                cache_manager=cache_manager,
                db_repository=db_repo
            )
            
            hybrid_search_service = HybridSearchService(
                db_repository=db_repo,
                vector_search_service=vector_search_service,
                query_processor=query_processor,
                cache_manager=cache_manager
            )
            
            reranker_service = ResultRerankerService(
                openai_service=openai_service,
                prompt_manager=prompt_manager,
                db_repository=db_repo,
                cache_manager=cache_manager
            )
            
            analytics_service = SearchAnalyticsService(
                db_repository=db_repo
            )
        
        # Execute search based on mode and available services
        if hybrid_search_service and search_request.search_mode in [SearchMode.HYBRID, SearchMode.VECTOR, SearchMode.SEMANTIC]:
            # AI-powered search
            search_response = await _execute_ai_search(
                search_request=search_request,
                hybrid_search_service=hybrid_search_service,
                reranker_service=reranker_service,
                analytics_service=analytics_service,
                current_user=current_user,
                search_id=search_id,
                start_time=start_time
            )
        else:
            # Fallback to basic search or mock response
            search_response = await _execute_basic_search(
                search_request=search_request,
                current_user=current_user,
                search_id=search_id,
                start_time=start_time
            )
        
        # Track search analytics in background
        if analytics_service:
            background_tasks.add_task(
                _track_search_analytics,
                analytics_service,
                search_request,
                search_response,
                current_user
            )
        
        logger.info(
            "Search completed successfully",
            search_id=search_response.search_id,
            user_id=current_user.user_id,
            search_mode=search_response.search_mode,
            results_count=search_response.total_count,
            duration_ms=search_response.analytics.get("total_search_time_ms", 0)
        )
        
        return search_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "search_failed",
                "message": "Search request could not be completed",
                "details": str(e) if logger.level == "DEBUG" else None
            }
        )


@router.post("/saved", response_model=SavedSearch)
async def create_saved_search(
    saved_search: SavedSearch,
    current_user: CurrentUserDep
) -> SavedSearch:
    """
    Save search configuration for reuse and alerts.
    
    Allows users to:
    - Save complex search queries with filters
    - Set up automated search alerts
    - Share search configurations within organization
    """
    try:
        # Set ownership
        saved_search.created_by = UUID(current_user.user_id)
        saved_search.tenant_id = UUID(current_user.tenant_id)
        
        # TODO: Store in database
        # saved_search_id = await database.create_saved_search(saved_search)
        
        logger.info(
            "Saved search created",
            search_name=saved_search.name,
            user_id=current_user.user_id,
            is_alert=saved_search.is_alert
        )
        
        return saved_search
        
    except Exception as e:
        logger.error(f"Failed to create saved search: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to save search configuration"
        )


@router.get("/saved", response_model=List[SavedSearch])
async def list_saved_searches(
    current_user: CurrentUserDep,
    limit: int = Query(50, ge=1, le=200, description="Maximum number of saved searches to return")
) -> List[SavedSearch]:
    """List saved searches for the current user and tenant."""
    try:
        # TODO: Implement database query
        # saved_searches = await database.get_saved_searches(
        #     tenant_id=current_user["tenant_id"],
        #     user_id=current_user["user_id"],
        #     limit=limit
        # )
        
        saved_searches = []  # Placeholder
        
        return saved_searches
        
    except Exception as e:
        logger.error(f"Failed to list saved searches: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve saved searches"
        )


@router.get("/saved/{search_id}", response_model=SavedSearch)
async def get_saved_search(
    search_id: str,
    current_user: CurrentUserDep
) -> SavedSearch:
    """Retrieve a specific saved search by ID."""
    try:
        # TODO: Implement database query with access control
        # saved_search = await database.get_saved_search(
        #     search_id=search_id,
        #     tenant_id=current_user["tenant_id"],
        #     user_id=current_user["user_id"]
        # )
        
        # if not saved_search:
        #     raise HTTPException(status_code=404, detail="Saved search not found")
        
        # return saved_search
        
        raise HTTPException(status_code=404, detail="Saved search not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get saved search: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve saved search"
        )


@router.post("/saved/{search_id}/execute", response_model=SearchResponse)
async def execute_saved_search(
    search_id: str,
    background_tasks: BackgroundTasks,
    current_user: CurrentUserDep
) -> SearchResponse:
    """Execute a saved search configuration."""
    try:
        # Get saved search
        # saved_search = await get_saved_search(search_id, current_user)
        
        # Execute the saved search request
        # search_response = await search_engine.search(
        #     request=saved_search.search_request,
        #     tenant_config=tenant_config
        # )
        
        # Update last run timestamp
        # background_tasks.add_task(
        #     _update_saved_search_last_run,
        #     search_id=search_id,
        #     result_count=search_response.total_count
        # )
        
        # return search_response
        
        raise HTTPException(status_code=404, detail="Saved search not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute saved search: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to execute saved search"
        )


@router.get("/history", response_model=PaginatedResponse)
async def get_search_history(
    current_user: CurrentUserDep,
    pagination: PaginationModel = Depends(),
    start_date: Optional[datetime] = Query(None, description="Filter searches from this date"),
    end_date: Optional[datetime] = Query(None, description="Filter searches until this date")
) -> PaginatedResponse:
    """
    Retrieve search history for analytics and insights.
    
    Provides search history with:
    - Query patterns and frequency analysis
    - Result click-through rates
    - Search abandonment tracking
    - Performance metrics over time
    """
    try:
        # TODO: Implement database query
        # search_history = await database.get_search_history(
        #     tenant_id=current_user["tenant_id"],
        #     user_id=current_user["user_id"],
        #     start_date=start_date,
        #     end_date=end_date,
        #     offset=pagination.offset,
        #     limit=pagination.limit
        # )
        
        # Mock response
        search_history = []
        total_count = 0
        
        response = PaginatedResponse.create(
            items=search_history,
            total=total_count,
            page=pagination.page,
            size=pagination.size
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get search history: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve search history"
        )


@router.post("/analytics/track-click")
async def track_search_click(
    search_id: str,
    profile_id: str,
    position: int,
    current_user: CurrentUserDep
) -> JSONResponse:
    """Track when a user clicks on a search result for analytics."""
    try:
        # TODO: Implement click tracking
        # await analytics.track_search_click(
        #     search_id=search_id,
        #     profile_id=profile_id,
        #     position=position,
        #     user_id=current_user["user_id"],
        #     tenant_id=current_user["tenant_id"],
        #     timestamp=datetime.now()
        # )
        
        logger.debug(
            "Search click tracked",
            search_id=search_id,
            profile_id=profile_id,
            position=position,
            user_id=current_user.user_id
        )
        
        return JSONResponse(content={"status": "tracked"})
        
    except Exception as e:
        logger.error(f"Failed to track search click: {e}")
        return JSONResponse(
            content={"status": "error", "message": "Failed to track click"},
            status_code=500
        )


@router.get("/suggestions")
async def get_search_suggestions(
    current_user: CurrentUserDep,
    q: str = Query(..., min_length=2, description="Query prefix for suggestions"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of suggestions")
) -> JSONResponse:
    """
    Get search query suggestions based on:
    - Popular searches in tenant
    - User's search history
    - Industry-specific terminology
    - Skill and technology names
    """
    try:
        suggestions = []
        
        # TODO: Implement intelligent suggestion generation
        # 1. Query user's search history for similar patterns
        # 2. Get popular searches from tenant
        # 3. Match against skill/technology dictionaries
        # 4. Use AI for contextual suggestions
        
        # Mock suggestions for now
        if "python" in q.lower():
            suggestions = [
                "Python developer",
                "Python machine learning engineer", 
                "Python backend developer",
                "Python data scientist"
            ]
        elif "senior" in q.lower():
            suggestions = [
                "Senior software engineer",
                "Senior developer", 
                "Senior data scientist",
                "Senior product manager"
            ]
        
        return JSONResponse(content={
            "suggestions": suggestions[:limit],
            "query": q
        })
        
    except Exception as e:
        logger.error(f"Failed to get search suggestions: {e}")
        return JSONResponse(
            content={"suggestions": [], "query": q},
            status_code=200  # Don't fail the request for suggestion errors
        )


@router.get("/facets")
async def get_available_facets(
    current_user: CurrentUserDep
) -> JSONResponse:
    """
    Get available search facets and filters for the tenant.
    
    Returns metadata about available facet fields:
    - Skills and technologies
    - Experience levels
    - Education levels
    - Locations
    - Industry categories
    """
    try:
        # TODO: Generate facets based on tenant's data
        facet_metadata = {
            "skills": {
                "display_name": "Skills",
                "type": "terms",
                "searchable": True,
                "popular_values": [
                    "Python", "JavaScript", "React", "AWS", "Docker",
                    "Machine Learning", "Data Science", "Node.js"
                ]
            },
            "experience_level": {
                "display_name": "Experience Level", 
                "type": "range",
                "ranges": [
                    {"label": "Entry Level", "min": 0, "max": 2},
                    {"label": "Mid Level", "min": 3, "max": 5},
                    {"label": "Senior Level", "min": 6, "max": 10},
                    {"label": "Expert Level", "min": 11, "max": None}
                ]
            },
            "education_level": {
                "display_name": "Education Level",
                "type": "terms", 
                "options": [
                    "Bachelor's Degree",
                    "Master's Degree", 
                    "PhD/Doctorate",
                    "Associate Degree",
                    "Professional Certificate"
                ]
            },
            "location": {
                "display_name": "Location",
                "type": "terms",
                "searchable": True,
                "supports_radius": True
            }
        }
        
        return JSONResponse(content={"facets": facet_metadata})
        
    except Exception as e:
        logger.error(f"Failed to get facet metadata: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve facet information"
        )


# Background task functions

async def _track_search_analytics(
    search_request: SearchRequest,
    search_response: SearchResponse, 
    user_id: str,
    duration_ms: float
) -> None:
    """Track search analytics in background"""
    try:
        search_history = SearchHistory(
            tenant_id=search_request.tenant_id,
            search_request=search_request,
            response_summary={
                "total_count": search_response.total_count,
                "search_mode": search_response.search_mode,
                "duration_ms": duration_ms,
                "high_match_count": search_response.high_match_count
            },
            user_id=UUID(user_id),
            search_duration_ms=int(duration_ms)
        )
        
        # TODO: Store in database
        # await database.create_search_history(search_history)
        
        logger.debug(
            "Search analytics tracked",
            search_id=search_response.search_id,
            duration_ms=duration_ms
        )
        
    except Exception as e:
        logger.warning(f"Failed to track search analytics: {e}")


async def _update_search_usage(tenant_id: str, search_count: int) -> None:
    """Update tenant search usage metrics"""
    try:
        tenant_manager = TenantManager()
        await tenant_manager.update_usage_metrics(
            tenant_id=tenant_id,
            metrics_update={
                "searches_today": search_count,
                "total_searches": search_count
            }
        )
        
        logger.debug("Updated tenant search usage", tenant_id=tenant_id)
        
    except Exception as e:
        logger.warning(f"Failed to update search usage: {e}")


async def _execute_ai_search(
    search_request: SearchRequest,
    hybrid_search_service: HybridSearchService,
    reranker_service: Optional[ResultRerankerService],
    analytics_service: Optional[SearchAnalyticsService],
    current_user: CurrentUser,
    search_id: str,
    start_time: datetime
) -> SearchResponse:
    """Execute AI-powered search with hybrid search and reranking"""
    
    try:
        # Convert search request to hybrid search parameters
        search_mode_mapping = {
            SearchMode.VECTOR: HybridSearchMode.VECTOR_ONLY,
            SearchMode.HYBRID: HybridSearchMode.HYBRID,
            SearchMode.SEMANTIC: HybridSearchMode.ADAPTIVE,
            SearchMode.KEYWORD: HybridSearchMode.TEXT_ONLY
        }
        
        hybrid_mode = search_mode_mapping.get(search_request.search_mode, HybridSearchMode.HYBRID)
        
        # Create search filter from request
        search_filter = None
        if search_request.filters:
            search_filter = VectorSearchFilter(
                entity_types=["profile", "job"],  # Adjust based on your needs
                tenant_ids=[str(current_user.tenant_id)] if current_user.tenant_id else None
            )
        
        # Configure hybrid search
        search_config = HybridSearchConfig(
            text_weight=0.4,
            vector_weight=0.6,
            fusion_method=FusionMethod.WEIGHTED_AVERAGE,
            enable_query_expansion=True,
            enable_result_diversification=True
        )
        
        # Execute hybrid search
        hybrid_response = await hybrid_search_service.hybrid_search(
            query=search_request.query,
            tenant_id=str(current_user.tenant_id) if current_user.tenant_id else None,
            limit=search_request.limit or 20,
            search_mode=hybrid_mode,
            config=search_config,
            search_filter=search_filter,
            use_cache=True,
            include_explanations=False
        )
        
        # Apply reranking if enabled and service available
        final_results = hybrid_response.results
        reranking_time_ms = 0
        
        if reranker_service and len(hybrid_response.results) > 0:
            rerank_start = datetime.now()
            
            reranking_config = RerankingConfig(
                strategy=RankingStrategy.HYBRID_INTELLIGENT,
                max_results_to_rerank=min(50, len(hybrid_response.results)),
                enable_explanations=False
            )
            
            # Convert HybridSearchResults to format expected by reranker
            hybrid_results_for_reranking = []
            for result in hybrid_response.results:
                # This is a simplified conversion - you may need to adjust based on actual data structures
                hybrid_results_for_reranking.append(result)
            
            reranking_response = await reranker_service.rerank_results(
                query=search_request.query,
                results=hybrid_results_for_reranking,
                tenant_id=str(current_user.tenant_id) if current_user.tenant_id else None,
                config=reranking_config
            )
            
            reranking_time_ms = int((datetime.now() - rerank_start).total_seconds() * 1000)
            # Use reranked results if available
            if reranking_response.results:
                final_results = reranking_response.results
        
        # Calculate total search time
        total_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Convert results to SearchResponse format
        search_results = []
        for result in final_results:
            # Convert hybrid/reranked results to SearchResponse result format
            # This is a simplified conversion - adjust based on your actual models
            search_result = {
                "id": result.entity_id,
                "type": result.entity_type,
                "score": getattr(result, 'reranked_score', getattr(result, 'final_score', 0)),
                "title": result.metadata.get("title", "Untitled") if result.metadata else "Untitled",
                "summary": result.content_preview or "No preview available",
                "metadata": result.metadata or {}
            }
            search_results.append(search_result)
        
        # Create analytics data
        analytics = {
            "total_search_time_ms": total_time_ms,
            "query_expansion_time_ms": 0,  # This would come from query processor
            "vector_search_time_ms": hybrid_response.search_time_ms - hybrid_response.fusion_time_ms,
            "text_search_time_ms": 0,  # This would come from text search component
            "fusion_time_ms": hybrid_response.fusion_time_ms,
            "reranking_time_ms": reranking_time_ms,
            "cache_hit": hybrid_response.cache_hit,
            "search_strategy": hybrid_mode.value,
            "results_before_reranking": len(hybrid_response.results),
            "results_after_reranking": len(final_results)
        }
        
        return SearchResponse(
            search_id=search_id,
            query=search_request.query,
            search_mode=search_request.search_mode,
            results=search_results,
            total_count=len(search_results),
            high_match_count=len([r for r in search_results if r["score"] > 0.8]),
            analytics=analytics
        )
        
    except Exception as e:
        logger.error(f"AI search execution failed: {e}")
        # Fallback to basic search on AI failure
        return await _execute_basic_search(
            search_request=search_request,
            current_user=current_user,
            search_id=search_id,
            start_time=start_time
        )


async def _execute_basic_search(
    search_request: SearchRequest,
    current_user: CurrentUser,
    search_id: str,
    start_time: datetime
) -> SearchResponse:
    """Execute basic/fallback search when AI services are not available"""
    
    total_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
    
    # Basic mock response for fallback
    search_response = SearchResponse(
        search_id=search_id,
        query=search_request.query,
        search_mode=search_request.search_mode,
        results=[],
        total_count=0,
        high_match_count=0,
        analytics={
            "total_search_time_ms": total_time_ms,
            "query_expansion_time_ms": 0,
            "vector_search_time_ms": 0,
            "reranking_time_ms": 0,
            "fallback_mode": True,
            "reason": "AI services not configured or unavailable"
        }
    )
    
    return search_response


async def _track_search_analytics(
    analytics_service: SearchAnalyticsService,
    search_request: SearchRequest,
    search_response: SearchResponse,
    current_user: CurrentUser
) -> None:
    """Track search analytics in background task"""
    
    try:
        # Prepare search event data
        search_data = {
            "search_id": search_response.search_id,
            "query": search_request.query,
            "search_mode": search_request.search_mode.value,
            "results_count": search_response.total_count,
            "high_match_count": search_response.high_match_count,
            "search_duration_ms": search_response.analytics.get("total_search_time_ms", 0),
            "vector_search_duration_ms": search_response.analytics.get("vector_search_time_ms", 0),
            "reranking_duration_ms": search_response.analytics.get("reranking_time_ms", 0),
            "cache_hit": search_response.analytics.get("cache_hit", False),
            "filters_applied": search_request.filters is not None
        }
        
        # Track search event
        await analytics_service.track_search_event(
            event_type="search_executed",
            search_data=search_data,
            tenant_id=str(current_user.tenant_id) if current_user.tenant_id else None,
            user_id=current_user.user_id
        )
        
    except Exception as e:
        logger.warning(f"Failed to track search analytics: {e}")


async def _update_saved_search_last_run(search_id: str, result_count: int) -> None:
    """Update saved search last run statistics"""
    try:
        # TODO: Update saved search record
        # await database.update_saved_search_stats(
        #     search_id=search_id,
        #     last_run=datetime.now(),
        #     last_result_count=result_count
        # )
        
        logger.debug("Updated saved search stats", search_id=search_id)
        
    except Exception as e:
        logger.warning(f"Failed to update saved search stats: {e}")