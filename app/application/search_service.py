"""Application layer orchestrator for search workflows following hexagonal architecture."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog

from app.core.config import get_settings
from app.api.schemas.search_schemas import SearchMode, SearchRequest, SearchResponse
from app.application.search.hybrid_search import (
    FusionMethod,
    HybridSearchConfig,
)
from app.application.search.hybrid_search import (
    SearchMode as HybridSearchMode,
)
from app.application.search.result_reranker import (
    RankingStrategy,
    RerankingConfig,
)
if TYPE_CHECKING:
    from app.application.dependencies.search_dependencies import SearchDependencies


# NOTE: CurrentUser is an API-layer DTO passed from FastAPI dependencies.
# Application layer accepts it as Any to avoid depending on infrastructure/API layers.
# The application layer extracts only the data it needs (user_id, tenant_id).
class SearchApplicationService:
    """Coordinates search execution across domain services and infrastructure.

    This application service follows hexagonal architecture principles by:
    - Using dependency injection via constructor
    - Depending only on domain interfaces (ports)
    - Orchestrating workflow without implementing business logic
    - Maintaining separation between domain and infrastructure concerns
    """

    def __init__(self, dependencies: SearchDependencies) -> None:
        """Initialize with injected dependencies.

        Args:
            dependencies: All required services and repositories
        """
        self._deps = dependencies
        self._logger = structlog.get_logger(__name__)

    async def execute_search(
        self,
        search_request: SearchRequest,
        current_user: Any,  # API-layer DTO, not imported to maintain hexagonal boundaries
        schedule_task: Any | None = None,
    ) -> SearchResponse:
        """Perform a search and dispatch background side-effects.

        Args:
            search_request: The search parameters and filters
            current_user: Current authenticated user context
            schedule_task: Optional task scheduler for background operations

        Returns:
            SearchResponse with results and analytics
        """
        self._logger.info(
            "Search request received (application layer)",
            tenant_id=str(search_request.tenant_id),
            user_id=current_user.user_id,
            search_mode=search_request.search_mode,
        )

        settings = get_settings()
        start_time = datetime.now()
        search_id = f"search_{start_time.strftime('%Y%m%d_%H%M%S')}_{current_user.user_id[:8]}"

        # Determine search strategy based on configuration and mode
        if settings.is_openai_configured() and search_request.search_mode in {
            SearchMode.HYBRID,
            SearchMode.VECTOR,
            SearchMode.SEMANTIC,
        }:
            search_response = await self._execute_ai_search(
                search_request=search_request,
                current_user=current_user,
                search_id=search_id,
                start_time=start_time,
            )
        else:
            search_response = await self._execute_basic_search(
                search_request=search_request,
                current_user=current_user,
                search_id=search_id,
                start_time=start_time,
            )

        # Schedule background tasks
        await self._schedule_analytics_tracking(
            schedule_task, search_request, search_response, current_user
        )
        await self._schedule_usage_tracking(
            schedule_task, str(search_request.tenant_id), 1
        )

        self._logger.info(
            "Search completed",
            search_id=search_response.search_id,
            user_id=current_user.user_id,
            search_mode=search_response.search_mode,
            results_count=search_response.total_count,
            duration_ms=search_response.analytics.get("total_search_time_ms", 0),
        )

        return search_response

    async def _execute_ai_search(
        self,
        search_request: SearchRequest,
        current_user: Any,  # API-layer DTO
        search_id: str,
        start_time: datetime,
    ) -> SearchResponse:
        """Execute AI-powered hybrid search with optional reranking.

        Args:
            search_request: Search parameters
            current_user: Current user context
            search_id: Unique search identifier
            start_time: Search start timestamp

        Returns:
            SearchResponse with AI-enhanced results
        """
        try:
            # Map search modes to hybrid search modes
            search_mode_mapping = {
                SearchMode.VECTOR: HybridSearchMode.VECTOR_ONLY,
                SearchMode.HYBRID: HybridSearchMode.HYBRID,
                SearchMode.SEMANTIC: HybridSearchMode.ADAPTIVE,
                SearchMode.KEYWORD: HybridSearchMode.TEXT_ONLY,
            }
            hybrid_mode = search_mode_mapping.get(search_request.search_mode, HybridSearchMode.HYBRID)

            # Configure search filters
            search_filter = self._create_search_filter(search_request, current_user)

            # Configure search parameters
            search_config = self._create_search_config()

            # Execute hybrid search
            hybrid_response = await self._deps.search_service.hybrid_search(
                query=search_request.query,
                tenant_id=str(current_user.tenant_id) if current_user.tenant_id else None,
                limit=search_request.limit or 20,
                search_mode=hybrid_mode,
                config=search_config,
                search_filter=search_filter,
                use_cache=True,
                include_explanations=False,
            )

            # Apply reranking if available
            final_results, reranking_time_ms = await self._apply_reranking(
                search_request, hybrid_response, current_user
            )

            # Build response
            return self._build_search_response(
                search_id=search_id,
                search_request=search_request,
                hybrid_response=hybrid_response,
                final_results=final_results,
                reranking_time_ms=reranking_time_ms,
                hybrid_mode=hybrid_mode,
                start_time=start_time,
            )

        except Exception as e:
            self._logger.error("AI search failed, falling back to basic search", error=str(e))
            return await self._execute_basic_search(
                search_request, current_user, search_id, start_time
            )

    async def _execute_basic_search(
        self,
        search_request: SearchRequest,
        current_user: Any,
        search_id: str,
        start_time: datetime,
    ) -> SearchResponse:
        """Execute basic search when AI search is unavailable.

        Attempts repository-based text search as fallback.

        Args:
            search_request: Search parameters
            current_user: Current user context
            search_id: Unique search identifier
            start_time: Search start timestamp

        Returns:
            SearchResponse with basic search results or empty response
        """
        try:
            # Use repository for basic text-based search
            if current_user.tenant_id:
                from app.domain.value_objects import TenantId
                tenant_id = TenantId(current_user.tenant_id)
                profiles = await self._deps.profile_repository.search_by_text(
                    tenant_id=tenant_id,
                    query=search_request.query,
                    limit=search_request.limit or 20
                )

                search_results = []
                for profile in profiles:
                    search_results.append({
                        "id": str(profile.id.value),
                        "type": "profile",
                        "score": 0.5,  # Default score for basic search
                        "title": profile.profile_data.name,
                        "summary": profile.profile_data.summary or "No summary available",
                        "metadata": {
                            "email": str(profile.profile_data.email),
                            "location": (
                                profile.profile_data.location.city
                                if profile.profile_data.location
                                else None
                            ),
                        },
                    })

                total_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

                return SearchResponse(
                    search_id=search_id,
                    query=search_request.query,
                    search_mode=search_request.search_mode,
                    results=search_results,
                    total_count=len(search_results),
                    high_match_count=0,
                    analytics={
                        "total_search_time_ms": total_time_ms,
                        "query_expansion_time_ms": 0,
                        "vector_search_time_ms": 0,
                        "reranking_time_ms": 0,
                        "fallback_mode": True,
                        "reason": "Using repository-based search",
                    },
                )

        except Exception as e:
            self._logger.error("Basic search failed", error=str(e))

        # Final fallback - empty results
        total_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        return SearchResponse(
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
                "reason": "All search methods failed",
            },
        )

    def _create_search_filter(
        self, search_request: SearchRequest, current_user: Any
    ) -> Any | None:
        """Create search filter based on request parameters.

        Args:
            search_request: Search parameters
            current_user: Current user context

        Returns:
            Configured search filter or None
        """
        # TODO: SearchFilter should be defined in domain layer or passed via dependency
        # For now, return None to avoid importing from infrastructure
        if not search_request.filters:
            return None

        # Filters will be applied by the search service based on tenant_id passed separately
        return None

    def _create_search_config(self) -> HybridSearchConfig:
        """Create search configuration with optimized parameters.

        Returns:
            Configured search parameters
        """
        return HybridSearchConfig(
            text_weight=0.4,
            vector_weight=0.6,
            fusion_method=FusionMethod.WEIGHTED_AVERAGE,
            enable_query_expansion=True,
            enable_result_diversification=True,
        )

    async def _apply_reranking(
        self,
        search_request: SearchRequest,
        hybrid_response: Any,
        current_user: Any,
    ) -> tuple[list[Any], int]:
        """Apply reranking to search results if available.

        Args:
            search_request: Original search request
            hybrid_response: Response from hybrid search
            current_user: Current user context

        Returns:
            Tuple of (final_results, reranking_time_ms)
        """
        final_results = hybrid_response.results
        reranking_time_ms = 0

        if self._deps.reranker_service and hybrid_response.results:
            rerank_start = datetime.now()
            reranking_config = RerankingConfig(
                strategy=RankingStrategy.HYBRID_INTELLIGENT,
                max_results_to_rerank=min(50, len(hybrid_response.results)),
                enable_explanations=False,
            )

            reranking_response = await self._deps.reranker_service.rerank_results(
                query=search_request.query,
                results=list(hybrid_response.results),
                tenant_id=str(current_user.tenant_id) if current_user.tenant_id else None,
                config=reranking_config,
            )

            reranking_time_ms = int((datetime.now() - rerank_start).total_seconds() * 1000)
            if reranking_response.results:
                final_results = reranking_response.results

        return final_results, reranking_time_ms

    def _build_search_response(
        self,
        search_id: str,
        search_request: SearchRequest,
        hybrid_response: Any,
        final_results: list[Any],
        reranking_time_ms: int,
        hybrid_mode: Any,
        start_time: datetime,
    ) -> SearchResponse:
        """Build standardized search response.

        Args:
            search_id: Unique search identifier
            search_request: Original search request
            hybrid_response: Response from hybrid search
            final_results: Final processed results
            reranking_time_ms: Time spent on reranking
            hybrid_mode: Search mode used
            start_time: Search start timestamp

        Returns:
            Formatted search response
        """
        total_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        search_results = []
        for result in final_results:
            score = getattr(result, "reranked_score", getattr(result, "final_score", 0))
            metadata = getattr(result, "metadata", {}) or {}
            search_results.append(
                {
                    "id": result.entity_id,
                    "type": result.entity_type,
                    "score": score,
                    "title": metadata.get("title", "Untitled"),
                    "summary": getattr(result, "content_preview", "No preview available"),
                    "metadata": metadata,
                }
            )

        analytics = {
            "total_search_time_ms": total_time_ms,
            "query_expansion_time_ms": 0,
            "vector_search_time_ms": hybrid_response.search_time_ms - hybrid_response.fusion_time_ms,
            "text_search_time_ms": 0,
            "fusion_time_ms": hybrid_response.fusion_time_ms,
            "reranking_time_ms": reranking_time_ms,
            "cache_hit": hybrid_response.cache_hit,
            "search_strategy": hybrid_mode.value,
            "results_before_reranking": len(hybrid_response.results),
            "results_after_reranking": len(final_results),
        }

        return SearchResponse(
            search_id=search_id,
            query=search_request.query,
            search_mode=search_request.search_mode,
            results=search_results,
            total_count=len(search_results),
            high_match_count=len([r for r in search_results if r["score"] > 0.8]),
            analytics=analytics,
        )

    async def _schedule(self, schedule_task: Any | None, func: Any, *args: Any) -> None:
        """Run coroutine immediately or schedule via background tasks.

        Args:
            schedule_task: Optional task scheduler
            func: Function to execute
            *args: Function arguments
        """
        if schedule_task is None:
            await func(*args)
            return

        add_task = getattr(schedule_task, "add_task", None)
        if callable(add_task):
            add_task(func, *args)
            return

        if callable(schedule_task):
            schedule_task(func, *args)
            return

        raise AttributeError("Provided scheduler does not support task scheduling")

    async def _schedule_analytics_tracking(
        self,
        schedule_task: Any | None,
        search_request: SearchRequest,
        search_response: SearchResponse,
        current_user: Any,
    ) -> None:
        """Schedule analytics tracking in background.

        Args:
            schedule_task: Optional task scheduler
            search_request: Original search request
            search_response: Search response with results
            current_user: Current user context
        """
        if self._deps.analytics_service:
            await self._schedule(
                schedule_task,
                self._track_search_analytics,
                search_request,
                search_response,
                current_user,
            )

    async def _schedule_usage_tracking(
        self,
        schedule_task: Any | None,
        tenant_id: str,
        search_count: int,
    ) -> None:
        """Schedule usage metrics tracking in background.

        Args:
            schedule_task: Optional task scheduler
            tenant_id: Tenant identifier
            search_count: Number of searches to track
        """
        await self._schedule(
            schedule_task,
            self._update_search_usage,
            tenant_id,
            search_count,
        )

    async def _track_search_analytics(
        self,
        search_request: SearchRequest,
        search_response: SearchResponse,
        current_user: Any,
    ) -> None:
        """Track search analytics in background.

        Args:
            search_request: Original search request
            search_response: Search response with results
            current_user: Current user context
        """
        try:
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
                "filters_applied": search_request.filters is not None,
            }

            if self._deps.analytics_service:
                await self._deps.analytics_service.track_search_event(
                    event_type="search_executed",
                    search_data=search_data,
                    tenant_id=str(current_user.tenant_id) if current_user.tenant_id else None,
                    user_id=current_user.user_id,
                )
        except Exception as exc:  # pragma: no cover - analytics failures shouldn't break flow
            self._logger.warning("Failed to track search analytics", error=str(exc))

    async def _update_search_usage(self, tenant_id: str, search_count: int) -> None:
        """Update tenant search usage metrics.

        Args:
            tenant_id: Tenant identifier
            search_count: Number of searches performed
        """
        try:
            await self._deps.tenant_manager.update_usage_metrics(
                tenant_id=tenant_id,
                metrics_update={"searches_performed": search_count},
            )
        except Exception as exc:  # pragma: no cover - best effort
            self._logger.warning("Failed to update search usage", error=str(exc))

    async def update_saved_search_last_run(self, search_id: str, result_count: int) -> None:
        """Update statistics for saved search execution.

        Args:
            search_id: Saved search identifier
            result_count: Number of results returned
        """
        self._logger.debug(
            "Saved search stats update scheduled",
            search_id=search_id,
            result_count=result_count,
            tenant_manager=self._deps.tenant_manager.__class__.__name__,
        )


__all__ = ["SearchApplicationService"]

