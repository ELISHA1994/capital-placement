"""Application layer orchestrator for search workflows following hexagonal architecture."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Dict, List

import structlog

from app.api.schemas.search_schemas import (
    MatchScore,
    SearchAnalytics,
    SearchMode,
    SearchRequest,
    SearchResponse,
)
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
from app.core.config import get_settings

if TYPE_CHECKING:
    from app.application.dependencies.search_dependencies import SearchDependencies


logger = structlog.get_logger(__name__)


@dataclass
class HybridSearchFilter:
    """Search filter payload passed to the hybrid search service."""

    entity_types: list[str] | None = None
    tenant_ids: list[str] | None = None
    embedding_models: list[str] | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    metadata_filters: dict[str, Any] | None = None
    exclude_entity_ids: list[str] | None = None


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
        # Track search appearances for profiles in results
        await self._schedule_search_appearance_tracking(
            schedule_task,
            search_response,
            str(search_request.tenant_id),
        )

        self._logger.info(
            "Search completed",
            search_id=search_response.search_id,
            user_id=current_user.user_id,
            search_mode=search_response.search_mode,
            results_count=search_response.total_count,
            duration_ms=search_response.analytics.total_search_time_ms,
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
                limit=search_request.max_results or 100,
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
                    limit=search_request.max_results or 100
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
                    results=[],  # Basic search returns simple dicts, not SearchResult models
                    total_count=len(search_results),
                    page=search_request.pagination.page if hasattr(search_request, 'pagination') else 1,
                    page_size=search_request.pagination.page_size if hasattr(search_request, 'pagination') else 20,
                    total_pages=1,
                    has_next_page=False,
                    has_prev_page=False,
                    analytics=SearchAnalytics(
                        total_search_time_ms=total_time_ms,
                        vector_search_time_ms=0,
                        keyword_search_time_ms=0,
                        reranking_time_ms=0,
                        total_candidates=0,
                        candidates_after_filters=len(search_results),
                        candidates_reranked=0,
                        query_expanded=False,
                    ),
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
            page=search_request.pagination.page if hasattr(search_request, 'pagination') else 1,
            page_size=search_request.pagination.page_size if hasattr(search_request, 'pagination') else 20,
            total_pages=0,
            has_next_page=False,
            has_prev_page=False,
            analytics=SearchAnalytics(
                total_search_time_ms=total_time_ms,
                vector_search_time_ms=0,
                keyword_search_time_ms=0,
                reranking_time_ms=0,
                total_candidates=0,
                candidates_after_filters=0,
                candidates_reranked=0,
                query_expanded=False,
            ),
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
        tenant_id = self._resolve_tenant_id(search_request, current_user)
        metadata_filters = self._build_metadata_filters(search_request)
        created_after, created_before = self._extract_date_bounds(
            search_request.range_filters
        )
        exclude_ids = self._extract_excluded_profile_ids(search_request)

        return HybridSearchFilter(
            entity_types=["profile"],
            tenant_ids=[tenant_id] if tenant_id else None,
            created_after=created_after,
            created_before=created_before,
            metadata_filters=metadata_filters or None,
            exclude_entity_ids=exclude_ids or None,
        )

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

        # Import SearchResult model for proper result construction
        from uuid import UUID

        from app.api.schemas.search_schemas import SearchResult

        # Convert hybrid search results to API SearchResult models
        search_results = []
        for result in final_results:
            score = getattr(result, "reranked_score", getattr(result, "final_score", 0))
            metadata = getattr(result, "metadata", {}) or {}

            # Build match score from result data
            match_score = MatchScore(
                overall_score=float(score),
                relevance_score=float(getattr(result, "final_score", score)),
                skill_match_score=0.0,
                experience_match_score=0.0,
                education_match_score=0.0,
                location_match_score=0.0,
                salary_match_score=0.0,
                vector_similarity=float(getattr(result, "vector_score", 0)) if hasattr(result, "vector_score") else None,
                keyword_relevance=float(getattr(result, "text_score", 0)) if hasattr(result, "text_score") else None,
                reranker_score=float(score) if reranking_time_ms > 0 else None,
            )

            match_score = self._enrich_match_score(
                match_score,
                metadata,
                search_request,
            )

            # Convert result to SearchResult model
            search_result = SearchResult(
                profile_id=str(result.entity_id),  # ✅ FIX P0-A: Convert UUID to string
                email=metadata.get("email", ""),
                tenant_id=UUID(str(search_request.tenant_id)),
                full_name=self._resolve_full_name(metadata),
                title=self._resolve_title(metadata),
                summary=getattr(result, "content_preview", metadata.get("summary", "No preview available")),
                current_company=metadata.get("current_company"),
                current_location=metadata.get("location"),
                total_experience_years=metadata.get("total_experience_years"),
                top_skills=metadata.get("skills", [])[:5] if isinstance(metadata.get("skills"), list) else [],
                key_achievements=[],
                highest_degree=metadata.get("education"),
                match_score=match_score,
                search_highlights={},
                last_updated=datetime.utcnow(),
                profile_completeness=0.8,
                availability_status=metadata.get("availability_status"),
            )

            search_results.append(search_result)

        analytics = SearchAnalytics(
            total_search_time_ms=total_time_ms,
            vector_search_time_ms=self._calculate_vector_search_time(hybrid_response),
            keyword_search_time_ms=0,
            reranking_time_ms=reranking_time_ms,
            total_candidates=len(hybrid_response.results) if hybrid_response.results else 0,
            candidates_after_filters=len(final_results),
            candidates_reranked=len(final_results) if reranking_time_ms > 0 else 0,
            query_expanded=False,
        )

        return SearchResponse(
            search_id=search_id,
            query=search_request.query,
            search_mode=search_request.search_mode,
            results=search_results,  # Now passing actual SearchResult model instances
            total_count=len(search_results),
            page=search_request.pagination.page if hasattr(search_request, 'pagination') else 1,
            page_size=search_request.pagination.page_size if hasattr(search_request, 'pagination') else 20,
            total_pages=(len(search_results) + 19) // 20 if search_results else 0,  # Calculate pages assuming 20 per page
            has_next_page=len(search_results) > (search_request.pagination.page * search_request.pagination.page_size) if hasattr(search_request, 'pagination') else False,
            has_prev_page=search_request.pagination.page > 1 if hasattr(search_request, 'pagination') else False,
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

    async def _schedule_search_appearance_tracking(
        self,
        schedule_task: Any | None,
        search_response: SearchResponse,
        tenant_id: str,
    ) -> None:
        """Schedule search appearance tracking for all profiles in results.

        Args:
            schedule_task: Task scheduler
            search_response: Search response with results
            tenant_id: Tenant identifier
        """
        if not search_response.results:
            return

        # Extract profile IDs from SearchResult models
        profile_ids = [result.profile_id for result in search_response.results]

        await self._schedule(
            schedule_task,
            self._track_search_appearances,
            profile_ids,
            tenant_id,
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
                "search_duration_ms": search_response.analytics.total_search_time_ms,
                "vector_search_duration_ms": search_response.analytics.vector_search_time_ms or 0,
                "reranking_duration_ms": search_response.analytics.reranking_time_ms or 0,
                "cache_hit": search_response.analytics.cache_hit_rate is not None,
                "filters_applied": search_request.has_filters,
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

    async def _track_search_appearances(
        self,
        profile_ids: list[str],
        tenant_id: str,
    ) -> None:
        """Track search appearances for multiple profiles.

        Args:
            profile_ids: List of profile IDs that appeared in search
            tenant_id: Tenant identifier
        """
        try:
            from app.domain.value_objects import ProfileId, TenantId

            tenant = TenantId(tenant_id)
            tracked_count = 0

            for profile_id_str in profile_ids:
                try:
                    profile = await self._deps.profile_repository.get_by_id(
                        ProfileId(profile_id_str),
                        tenant,
                    )

                    if profile:
                        profile.record_search_appearance()
                        await self._deps.profile_repository.save(profile)
                        tracked_count += 1
                except Exception as profile_exc:
                    self._logger.warning(
                        "Failed to track appearance for profile",
                        profile_id=profile_id_str,
                        error=str(profile_exc),
                    )
                    continue

            self._logger.debug(
                "Search appearances tracked",
                tenant_id=tenant_id,
                total_profiles=len(profile_ids),
                tracked_count=tracked_count,
            )
        except Exception as exc:
            # Analytics tracking failures should not break the search flow
            self._logger.warning(
                "Failed to track search appearances",
                tenant_id=tenant_id,
                profile_count=len(profile_ids) if profile_ids else 0,
                error=str(exc),
            )

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


    # === STATIC HELPER METHODS ===

    def _enrich_match_score(
        self,
        match_score: MatchScore,
        metadata: dict[str, Any],
        search_request: SearchRequest,
    ) -> MatchScore:
        """Enrich base match score with request-specific scoring."""

        if not search_request.skill_requirements:
            return match_score

        candidate_skills = self._collect_candidate_skills(metadata)
        candidate_skills_lower = [skill.lower() for skill in candidate_skills]

        matched_skills: List[str] = []
        missing_skills: List[str] = []
        skill_gaps: Dict[str, str] = {}
        total_weight = 0.0
        matched_weight = 0.0

        for requirement in search_request.skill_requirements:
            weight = requirement.weight or 1.0
            total_weight += weight

            skill_name = requirement.name.lower()
            alternatives = [alt.lower() for alt in requirement.alternatives]

            found = any(
                self._skill_matches(candidate_skill, skill_name, alternatives)
                for candidate_skill in candidate_skills_lower
            )

            if found:
                matched_skills.append(requirement.name)
                matched_weight += weight
            else:
                missing_skills.append(requirement.name)
                if requirement.required:
                    skill_gaps[requirement.name] = "Required skill missing"

        if total_weight > 0:
            skill_score = matched_weight / total_weight
        else:
            skill_score = 0.0

        match_score.skill_match_score = skill_score
        match_score.matched_skills = matched_skills
        match_score.missing_skills = missing_skills
        match_score.skill_gaps = skill_gaps

        return match_score

    @staticmethod
    def _resolve_tenant_id(
        search_request: SearchRequest,
        current_user: Any,
    ) -> str | None:
        """Resolve an authoritative tenant ID for downstream queries."""
        request_tenant = (
            str(search_request.tenant_id) if getattr(search_request, "tenant_id", None) else None
        )
        user_tenant = getattr(current_user, "tenant_id", None)

        if user_tenant and request_tenant and str(user_tenant) != request_tenant:
            logger.warning(
                "Search request tenant mismatch detected",
                request_tenant=request_tenant,
                user_tenant=str(user_tenant),
            )

        if user_tenant:
            return str(user_tenant)
        return request_tenant

    @staticmethod
    def _build_metadata_filters(search_request: SearchRequest) -> dict[str, str]:
        """Map API search filters to metadata filters used by hybrid search."""
        metadata_filters: dict[str, str] = {}

        for basic_filter in getattr(search_request, "basic_filters", []) or []:
            mapped_key = SearchApplicationService._map_basic_filter_field(
                getattr(basic_filter, "field", "")
            )
            if not mapped_key:
                continue

            value = getattr(basic_filter, "value", None)
            if value is None:
                continue

            if isinstance(value, list):
                if not value:
                    continue
                metadata_filters[mapped_key] = str(value[0])
            else:
                metadata_filters[mapped_key] = str(value)

        if not getattr(search_request, "include_inactive", False):
            metadata_filters.setdefault("status", "active")

        location_filter = getattr(search_request, "location_filter", None)
        if location_filter:
            preferred_locations = getattr(location_filter, "preferred_locations", None) or []
            center_location = getattr(location_filter, "center_location", None)
            location_value = preferred_locations[0] if preferred_locations else center_location
            if location_value:
                metadata_filters.setdefault("location_city", str(location_value))

        return metadata_filters

    @staticmethod
    def _extract_date_bounds(
        range_filters: list[Any] | None,
    ) -> tuple[datetime | None, datetime | None]:
        """Extract created/updated bounds from range filters."""
        created_after: datetime | None = None
        created_before: datetime | None = None

        for range_filter in range_filters or []:
            field_name = getattr(range_filter, "field", "")
            if field_name not in {"created_at", "profile_created_at", "last_updated"}:
                continue

            min_value = getattr(range_filter, "min_value", None)
            max_value = getattr(range_filter, "max_value", None)

            if min_value and not created_after:
                created_after = SearchApplicationService._coerce_datetime(min_value)
            if max_value and not created_before:
                created_before = SearchApplicationService._coerce_datetime(max_value)

        return created_after, created_before

    @staticmethod
    def _extract_excluded_profile_ids(search_request: SearchRequest) -> list[str]:
        """Extract profile IDs that should be excluded from results."""
        preferences = getattr(search_request, "user_preferences", {}) or {}
        raw_ids = preferences.get("exclude_profile_ids", [])
        if not isinstance(raw_ids, list):
            return []

        return [str(profile_id) for profile_id in raw_ids if profile_id]

    @staticmethod
    def _map_basic_filter_field(field_name: str) -> str | None:
        """Resolve metadata key for a given basic filter field."""
        mapping = {
            "status": "status",
            "experience_level": "experience_level",
            "current_location": "location_city",
            "location_country": "location_country",
            "email": "email",
        }
        return mapping.get(field_name)

    @staticmethod
    def _coerce_datetime(value: Any) -> datetime | None:
        """Convert date-like values into datetime."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time())
        return None

    @staticmethod
    def _calculate_vector_search_time(hybrid_response: Any) -> int:
        """Calculate vector search time from hybrid response metrics."""
        search_time = getattr(hybrid_response, "search_time_ms", None)
        if search_time is None:
            return 0

        fusion_time = getattr(hybrid_response, "fusion_time_ms", 0) or 0
        return max(int(search_time) - int(fusion_time), 0)

    @staticmethod
    def _resolve_full_name(metadata: dict[str, Any]) -> str:
        """Determine best full name for result."""
        if not metadata:
            return "Unknown"
        name = metadata.get("name")
        if name:
            return str(name)
        title = metadata.get("title")
        if title:
            return str(title)
        return "Unknown"

    @staticmethod
    def _resolve_title(metadata: dict[str, Any]) -> str:
        """Determine best title/headline for result."""
        if not metadata:
            return ""
        if metadata.get("title"):
            return str(metadata["title"])
        headline = metadata.get("headline")
        if headline:
            return str(headline)
        # As a final fallback, avoid duplicating "Unknown" if name exists
        name = metadata.get("name")
        return str(name) if name else ""

    @staticmethod
    def _collect_candidate_skills(metadata: dict[str, Any]) -> List[str]:
        """Gather candidate skills from metadata payload."""

        candidate_skills: List[str] = []

        if not metadata:
            return candidate_skills

        for key in ("skills", "top_skills", "matched_skills"):
            value = metadata.get(key)
            if not value:
                continue

            if isinstance(value, str):
                candidate_skills.extend(
                    [item.strip() for item in value.split(",") if item and item.strip()]
                )
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        candidate_skills.append(item.strip())
                    elif isinstance(item, dict) and item.get("name"):
                        name = str(item["name"]).strip()
                        if name:
                            candidate_skills.append(name)

        # Remove duplicates preserving order
        seen = set()
        unique_skills: List[str] = []
        for skill in candidate_skills:
            skill_lower = skill.lower()
            if skill_lower not in seen:
                seen.add(skill_lower)
                unique_skills.append(skill)

        return unique_skills

    @staticmethod
    def _skill_matches(candidate_skill: str, target_skill: str, alternatives: List[str]) -> bool:
        """Check if candidate skill satisfies requirement."""

        if not candidate_skill:
            return False

        if target_skill in candidate_skill or candidate_skill in target_skill:
            return True

        for alternative in alternatives:
            if alternative in candidate_skill or candidate_skill in alternative:
                return True

        return False


__all__ = ["SearchApplicationService"]
