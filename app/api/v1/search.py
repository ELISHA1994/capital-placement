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
from typing import Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from app.api.converters import build_search_request_from_config
from app.api.dependencies import (
    ClickTrackingServiceDep,
    SavedSearchServiceDep,
    SearchServiceDep,
    map_domain_exception_to_http,
)
from app.api.schemas.base import PaginatedResponse
from app.api.schemas.facet_schemas import (
    FacetMetadataResponse,
    FacetFieldResponse,
    FacetValueResponse,
    RangeBucketResponse,
)
from app.api.schemas.search_schemas import (
    EducationRequirement,
    ExperienceRequirement,
    LocationFilter,
    RangeFilter,
    SalaryFilter,
    SavedSearch,
    SavedSearchCreate,
    SearchFilter,
    SearchMode,
    SearchRequest,
    SearchResponse,
    SkillRequirement,
)
from app.application.facet_application_service import FacetApplicationService
from app.core.config import get_settings
from app.core.dependencies import CurrentUserDep
from app.domain.entities.saved_search import SearchConfiguration
from app.domain.exceptions import DomainException
from app.domain.value_objects import TenantId
from app.infrastructure.persistence.models.base import PaginationModel
from app.infrastructure.providers.facet_provider import get_facet_application_service

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/search", tags=["search"])
settings = get_settings()


@router.post("/", response_model=SearchResponse)
async def search_profiles(
    search_request: SearchRequest,
    background_tasks: BackgroundTasks,
    current_user: CurrentUserDep,
    search_service: SearchServiceDep,
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
        return await search_service.execute_search(
            search_request=search_request,
            current_user=current_user,
            schedule_task=background_tasks,
        )
    except DomainException as domain_exc:
        # Map domain exceptions to appropriate HTTP responses
        raise map_domain_exception_to_http(domain_exc)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - unexpected errors bubble to API layer
        logger.error("Search request failed", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "search_failed",
                "message": "Search request could not be completed",
                "details": str(exc)
                if settings.ENVIRONMENT in ("local", "development")
                else None,
            },
        )


@router.post("/saved", response_model=SavedSearch)
async def create_saved_search(
    saved_search_request: SavedSearchCreate,
    current_user: CurrentUserDep,
    saved_search_service: SavedSearchServiceDep,
) -> SavedSearch:
    """
    Save search configuration for reuse and alerts.

    Allows users to:
    - Save complex search queries with filters
    - Set up automated search alerts
    - Share search configurations within organization
    """
    try:
        # Create saved search via application service
        saved_search_entity = await saved_search_service.create_saved_search(
            name=saved_search_request.name,
            search_request=saved_search_request.search_request,
            tenant_id=str(current_user.tenant_id),
            user_id=str(current_user.user_id),
            description=saved_search_request.description,
            is_alert=saved_search_request.is_alert,
            alert_frequency=saved_search_request.alert_frequency,
        )

        # Convert domain entity to API response
        return SavedSearch(
            id=str(saved_search_entity.id),
            name=saved_search_entity.name,
            description=saved_search_entity.description,
            search_request=saved_search_request.search_request,
            tenant_id=saved_search_entity.tenant_id.value,
            is_alert=saved_search_entity.is_alert,
            alert_frequency=saved_search_entity.alert_frequency.value,
            last_run=saved_search_entity.statistics.last_run,
            last_result_count=saved_search_entity.statistics.last_result_count,
            new_results_since_last_run=saved_search_entity.statistics.new_results_since_last_run,
            created_at=saved_search_entity.created_at,
            updated_at=saved_search_entity.updated_at,
        )

    except DomainException as domain_exc:
        raise map_domain_exception_to_http(domain_exc)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to create saved search", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "saved_search_creation_failed",
                "message": "Failed to save search configuration",
                "details": str(exc)
                if settings.ENVIRONMENT in ("local", "development")
                else None,
            },
        )


@router.get("/saved", response_model=list[SavedSearch])
async def list_saved_searches(
    current_user: CurrentUserDep,
    saved_search_service: SavedSearchServiceDep,
    limit: int = Query(
        50, ge=1, le=200, description="Maximum number of saved searches to return"
    ),
    status: str | None = Query(
        None, description="Filter by status (active, paused, archived)"
    ),
) -> list[SavedSearch]:
    """
    List saved searches for the current user and tenant.

    Returns searches that the user:
    - Created (owned searches)
    - Has been granted access to (shared searches)

    Results are automatically filtered by tenant for multi-tenant isolation.
    """
    try:
        # Call application service (already handles access control)
        saved_search_entities = await saved_search_service.list_user_saved_searches(
            user_id=str(current_user.user_id),
            tenant_id=str(current_user.tenant_id),
            status=status,
            limit=limit,
            offset=0,  # No pagination in V1 - future enhancement
        )

        # Convert domain entities to API responses
        return [
            SavedSearch(
                id=str(entity.id),
                name=entity.name,
                description=entity.description,
                search_request=build_search_request_from_config(
                    config=entity.configuration, tenant_id=entity.tenant_id.value
                ),
                tenant_id=entity.tenant_id.value,
                is_alert=entity.is_alert,
                alert_frequency=entity.alert_frequency.value,
                last_run=entity.statistics.last_run,
                last_result_count=entity.statistics.last_result_count,
                new_results_since_last_run=entity.statistics.new_results_since_last_run,
                created_at=entity.created_at,
                updated_at=entity.updated_at,
            )
            for entity in saved_search_entities
        ]

    except DomainException as domain_exc:
        # Map domain exceptions to HTTP responses
        raise map_domain_exception_to_http(domain_exc)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to list saved searches",
            error=str(exc),
            user_id=current_user.user_id,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "saved_search_list_failed",
                "message": "Failed to retrieve saved searches",
                "details": str(exc)
                if settings.ENVIRONMENT in ("local", "development")
                else None,
            },
        )


@router.get("/saved/{search_id}", response_model=SavedSearch)
async def get_saved_search(
    search_id: str,
    current_user: CurrentUserDep,
    saved_search_service: SavedSearchServiceDep,
) -> SavedSearch:
    """
    Retrieve a specific saved search by ID.

    Returns the complete saved search configuration including:
    - Search parameters and filters
    - Alert settings and execution statistics
    - Metadata and timestamps

    Access control:
    - User must be the creator OR the search must be shared with them
    - Multi-tenant isolation enforced automatically
    """
    try:
        # Retrieve saved search via application service (handles access control)
        saved_search_entity = await saved_search_service.get_saved_search(
            saved_search_id=search_id,
            tenant_id=str(current_user.tenant_id),
            user_id=str(current_user.user_id),
        )

        # Reconstruct SearchRequest from domain configuration
        search_request = build_search_request_from_config(
            config=saved_search_entity.configuration,
            tenant_id=saved_search_entity.tenant_id.value,
        )

        # Map domain entity to API response
        return SavedSearch(
            id=str(saved_search_entity.id),
            name=saved_search_entity.name,
            description=saved_search_entity.description,
            search_request=search_request,
            tenant_id=saved_search_entity.tenant_id.value,
            is_alert=saved_search_entity.is_alert,
            alert_frequency=saved_search_entity.alert_frequency.value,
            last_run=saved_search_entity.statistics.last_run,
            last_result_count=saved_search_entity.statistics.last_result_count,
            new_results_since_last_run=saved_search_entity.statistics.new_results_since_last_run,
            created_at=saved_search_entity.created_at,
            updated_at=saved_search_entity.updated_at,
        )

    except DomainException as domain_exc:
        raise map_domain_exception_to_http(domain_exc)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to retrieve saved search", error=str(exc), search_id=search_id
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "saved_search_retrieval_failed",
                "message": "Failed to retrieve saved search",
                "details": str(exc)
                if settings.ENVIRONMENT in ("local", "development")
                else None,
            },
        )


@router.post("/saved/{search_id}/execute", response_model=SearchResponse)
async def execute_saved_search(
    search_id: str,
    background_tasks: BackgroundTasks,
    current_user: CurrentUserDep,
    saved_search_service: SavedSearchServiceDep,
    search_service: SearchServiceDep,
) -> SearchResponse:
    """
    Execute a saved search configuration.

    Retrieves a saved search and executes it with the current search engine,
    then updates execution statistics in the background.

    Args:
        search_id: ID of the saved search to execute
        background_tasks: FastAPI background tasks for async statistics update
        current_user: Current authenticated user
        saved_search_service: SavedSearchApplicationService dependency
        search_service: SearchApplicationService dependency

    Returns:
        SearchResponse with results from executing the saved search

    Raises:
        404: Saved search not found
        403: User doesn't have access to saved search
        500: Search execution failed
    """
    try:
        # 1. Fetch saved search (includes access control)
        saved_search_entity = await saved_search_service.get_saved_search(
            saved_search_id=search_id,
            tenant_id=str(current_user.tenant_id),
            user_id=str(current_user.user_id),
        )

        # 2. Reconstruct SearchRequest from domain configuration
        search_request = build_search_request_from_config(
            saved_search_entity.configuration,
            current_user.tenant_id,
        )

        # 3. Execute search via existing service
        search_response = await search_service.execute_search(
            search_request=search_request,
            current_user=current_user,
            schedule_task=background_tasks,
        )

        # 4. Schedule statistics update in background (non-blocking)
        background_tasks.add_task(
            _update_saved_search_statistics,
            saved_search_service=saved_search_service,
            saved_search_id=search_id,
            tenant_id=str(current_user.tenant_id),
            user_id=str(current_user.user_id),
            result_count=search_response.total_count,
            execution_time_ms=search_response.analytics.total_search_time_ms,
        )

        return search_response

    except DomainException as domain_exc:
        # Map domain exceptions to appropriate HTTP responses
        # Handles: saved_search_not_found (404), saved_search_access_denied (403)
        raise map_domain_exception_to_http(domain_exc)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to execute saved search", error=str(exc), search_id=search_id
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "saved_search_execution_failed",
                "message": "Failed to execute saved search",
                "details": str(exc)
                if settings.ENVIRONMENT in ("local", "development")
                else None,
            },
        )


@router.get("/history", response_model=PaginatedResponse)
async def get_search_history(
    current_user: CurrentUserDep,
    pagination: PaginationModel = Depends(),
    start_date: datetime | None = Query(
        None, description="Filter searches from this date"
    ),
    end_date: datetime | None = Query(
        None, description="Filter searches until this date"
    ),
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
            size=pagination.size,
        )

        return response

    except Exception as e:
        logger.error(f"Failed to get search history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve search history")


@router.post("/analytics/track-click")
async def track_search_click(
    search_id: str,
    profile_id: str,
    position: int,
    current_user: CurrentUserDep,
    click_tracking_service: ClickTrackingServiceDep,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    Track when a user clicks on a search result for analytics.

    This is a high-performance endpoint optimized for minimal latency.
    Processing happens in the background to avoid blocking the user.

    Performance: Sub-50ms response time guaranteed.
    """
    try:
        # Quick validation
        if position < 0:
            raise HTTPException(status_code=400, detail="Position must be non-negative")

        # Track click in background task for speed (non-blocking)
        background_tasks.add_task(
            _track_click_in_background,
            click_tracking_service=click_tracking_service,
            search_id=search_id,
            profile_id=profile_id,
            position=position,
            tenant_id=str(current_user.tenant_id),
            user_id=str(current_user.user_id),
        )

        # Return immediately (sub-50ms)
        return JSONResponse(
            content={"status": "tracked", "search_id": search_id},
            status_code=202  # Accepted (async processing)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Click tracking endpoint failed", error=str(e))
        # Don't fail user request for analytics failure
        return JSONResponse(
            content={"status": "error", "message": "Failed to track click"},
            status_code=500
        )


@router.get("/suggestions")
async def get_search_suggestions(
    current_user: CurrentUserDep,
    q: str = Query(..., min_length=2, description="Query prefix for suggestions"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of suggestions"),
) -> JSONResponse:
    """
    Get intelligent search query suggestions.

    Provides autocomplete suggestions from multiple sources:
    - **User's search history**: Personalized recent searches
    - **Popular tenant searches**: Frequently used queries in organization
    - **Skill dictionaries**: Technology skills, certifications
    - **Job title variations**: Standard job titles and roles

    Performance: <100ms response time (p95)
    """
    try:
        # Get suggestion service
        from app.infrastructure.providers.suggestion_provider import get_suggestion_service
        suggestion_service = await get_suggestion_service()

        # Generate suggestions
        tenant_id = TenantId(str(current_user.tenant_id))
        suggestions = await suggestion_service.get_suggestions(
            query_prefix=q,
            tenant_id=tenant_id,
            user_id=str(current_user.user_id),
            limit=limit
        )

        # Convert to response format
        suggestion_responses = [
            {
                "text": s.text,
                "source": s.source.value,
                "score": s.score,
                "metadata": s.metadata
            }
            for s in suggestions
        ]

        return JSONResponse(content={
            "suggestions": suggestion_responses,
            "query": q,
            "count": len(suggestion_responses),
            "cached": False
        })

    except Exception as e:
        logger.error("Failed to get search suggestions", error=str(e))
        # Graceful degradation - return empty suggestions
        return JSONResponse(
            content={"suggestions": [], "query": q, "count": 0, "cached": False},
            status_code=200  # Don't fail the request for suggestion errors
        )


@router.get("/facets", response_model=FacetMetadataResponse)
async def get_available_facets(
    current_user: CurrentUserDep,
    facet_service: Annotated[
        FacetApplicationService,
        Depends(get_facet_application_service)
    ],
    force_refresh: bool = Query(False, description="Force regeneration of facets")
) -> FacetMetadataResponse:
    """
    Get available search facets and filters for the tenant.

    Returns dynamic facet metadata based on actual tenant profile data:
    - Skills and technologies (most common)
    - Experience levels and years
    - Education levels
    - Locations (country, city)
    - Previous companies
    - Languages

    Facets are cached for 1 hour and automatically invalidated on bulk changes.
    """
    try:
        # Get tenant ID from current user
        tenant_id = TenantId(current_user.tenant_id)

        # Get facets (with caching)
        facet_metadata = await facet_service.get_facets(
            tenant_id=tenant_id,
            force_refresh=force_refresh
        )

        # Check if from cache
        cache_hit = not force_refresh and not facet_metadata.is_stale()

        # Convert to API response
        return FacetMetadataResponse(
            facets=[
                FacetFieldResponse(
                    field_name=f.field_name.value,
                    facet_type=f.facet_type.value,
                    display_name=f.display_name,
                    description=f.description,
                    values=[
                        FacetValueResponse(
                            value=v.value,
                            count=v.count,
                            display_name=v.display_name,
                            percentage=v.percentage
                        )
                        for v in f.values
                    ],
                    buckets=[
                        RangeBucketResponse(
                            label=b.label,
                            min_value=b.min_value,
                            max_value=b.max_value,
                            count=b.count,
                            percentage=b.percentage
                        )
                        for b in f.buckets
                    ],
                    min_value=f.min_value,
                    max_value=f.max_value,
                    searchable=f.searchable,
                    multi_select=f.multi_select,
                    total_count=f.total_count,
                    unique_count=f.unique_count
                )
                for f in facet_metadata.facet_fields
            ],
            total_profiles=facet_metadata.total_profiles,
            active_profiles=facet_metadata.active_profiles,
            generated_at=facet_metadata.generated_at.isoformat(),
            cache_hit=cache_hit
        )

    except DomainException as domain_exc:
        raise map_domain_exception_to_http(domain_exc)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get facets", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "facet_generation_failed",
                "message": "Failed to generate facet metadata",
                "details": str(exc) if settings.ENVIRONMENT in ("local", "development") else None
            }
        )


async def _track_click_in_background(
    click_tracking_service,
    search_id: str,
    profile_id: str,
    position: int,
    tenant_id: str,
    user_id: str,
) -> None:
    """
    Track click event in background task (non-blocking).

    This function runs as a background task to persist click tracking
    without blocking the API response.

    Args:
        click_tracking_service: ClickTrackingApplicationService instance
        search_id: Search execution identifier
        profile_id: Profile that was clicked
        position: Position in results
        tenant_id: Tenant identifier
        user_id: User identifier
    """
    try:
        await click_tracking_service.track_click(
            search_id=search_id,
            profile_id=profile_id,
            position=position,
            tenant_id=tenant_id,
            user_id=user_id,
        )

        logger.debug(
            "Click tracked in background",
            search_id=search_id,
            profile_id=profile_id,
            position=position,
        )
    except Exception as exc:
        # Log but don't fail the request (this is background processing)
        logger.error(
            "Failed to track click in background",
            search_id=search_id,
            profile_id=profile_id,
            error=str(exc),
        )


async def _update_saved_search_statistics(
    saved_search_service,
    saved_search_id: str,
    tenant_id: str,
    user_id: str,
    result_count: int,
    execution_time_ms: int,
) -> None:
    """
    Update saved search execution statistics in background.

    This function runs as a background task to update the saved search's
    execution statistics without blocking the search response.

    Args:
        saved_search_service: SavedSearchApplicationService instance
        saved_search_id: ID of the saved search to update
        tenant_id: Tenant ID for access control
        user_id: User ID for access control
        result_count: Number of results from the search
        execution_time_ms: Search execution time in milliseconds
    """
    try:
        # Retrieve saved search (with access control)
        saved_search = await saved_search_service.get_saved_search(
            saved_search_id=saved_search_id,
            tenant_id=tenant_id,
            user_id=user_id,
        )

        # Record execution (domain method)
        saved_search.record_execution(
            result_count=result_count,
            execution_time_ms=execution_time_ms,
        )

        # Persist changes
        await saved_search_service._deps.saved_search_repository.save(saved_search)

        logger.info(
            "Saved search statistics updated",
            saved_search_id=saved_search_id,
            result_count=result_count,
            execution_time_ms=execution_time_ms,
        )
    except Exception as exc:
        # Log but don't fail the search response (this is background processing)
        logger.warning(
            "Failed to update saved search statistics",
            saved_search_id=saved_search_id,
            error=str(exc),
        )
