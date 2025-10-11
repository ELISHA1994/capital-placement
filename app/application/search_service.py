"""Application layer orchestrator for search workflows following hexagonal architecture."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

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
            vector_attr = getattr(result, "vector_score", None)
            text_attr = getattr(result, "text_score", None)

            match_score = MatchScore(
                overall_score=float(score),
                relevance_score=float(getattr(result, "final_score", score)),
                skill_match_score=0.0,
                experience_match_score=0.0,
                education_match_score=0.0,
                location_match_score=0.0,
                salary_match_score=0.0,
                vector_similarity=float(vector_attr) if vector_attr is not None else None,
                keyword_relevance=float(text_attr) if text_attr is not None else None,
                reranker_score=float(score) if reranking_time_ms > 0 else None,
            )

            match_score = self._enrich_match_score(
                match_score,
                metadata,
                search_request,
                vector_score=getattr(result, "vector_score", None),
                text_score=getattr(result, "text_score", None),
                semantic_score=getattr(result, "semantic_score", None),
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
                top_skills=self._resolve_top_skills(metadata),
                key_achievements=[],
                highest_degree=self._resolve_highest_degree(metadata),
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
        vector_score: Optional[float] = None,
        text_score: Optional[float] = None,
        semantic_score: Optional[float] = None,
    ) -> MatchScore:
        """Enrich base match score with request-specific scoring."""

        if search_request.skill_requirements:
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

        if search_request.experience_requirements:
            experience_years = self._extract_experience_years(metadata)
            match_score.experience_match_score = self._score_experience_years(
                experience_years,
                search_request.experience_requirements,
            )

        if search_request.education_requirements:
            match_score.education_match_score = self._score_education(
                metadata,
                search_request.education_requirements,
            )

        if search_request.location_filter:
            match_score.location_match_score = self._score_location(
                metadata,
                search_request.location_filter,
            )

        if search_request.salary_filter:
            match_score.salary_match_score = self._score_salary(
                metadata,
                search_request.salary_filter,
            )

        if vector_score is not None:
            match_score.vector_similarity = vector_score
        if text_score is not None:
            match_score.keyword_relevance = text_score
        if semantic_score is not None:
            match_score.semantic_relevance = semantic_score

        weights = search_request.custom_scoring or {
            "relevance": 0.3,
            "skills": 0.4,
            "experience": 0.2,
            "education": 0.05,
            "location": 0.03,
            "salary": 0.02,
        }

        contributions: Dict[str, float] = {}
        explanations: List[str] = []

        def record_contribution(label: str, weight_key: str, score: Optional[float]) -> None:
            if score is None:
                return
            weight = weights.get(weight_key, 0.0)
            contribution = round(weight * score, 4)
            contributions[weight_key] = contribution
            explanations.append(
                f"{label}: score {score:.2f} × weight {weight:.2f} = {contribution:.2f}"
            )

        record_contribution("Relevance", "relevance", match_score.relevance_score)
        record_contribution("Skills", "skills", match_score.skill_match_score)
        record_contribution("Experience", "experience", match_score.experience_match_score)
        record_contribution("Education", "education", match_score.education_match_score)
        record_contribution("Location", "location", match_score.location_match_score)
        record_contribution("Salary", "salary", match_score.salary_match_score)

        if contributions:
            match_score.score_breakdown = contributions
        if explanations:
            match_score.match_explanation = explanations

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
    def _resolve_top_skills(metadata: dict[str, Any]) -> List[str]:
        """Extract top skills suitable for response schema."""

        raw_skills = metadata.get("skills")

        if isinstance(raw_skills, list):
            cleaned: List[str] = []
            for item in raw_skills:
                if isinstance(item, str) and item.strip():
                    cleaned.append(item.strip())
                elif isinstance(item, dict):
                    name = item.get("name") or item.get("skill") or item.get("value")
                    if isinstance(name, str) and name.strip():
                        cleaned.append(name.strip())
            return cleaned[:5]

        if isinstance(raw_skills, str) and raw_skills.strip():
            return [part.strip() for part in raw_skills.split(",") if part.strip()][:5]

        return []

    @staticmethod
    def _resolve_highest_degree(metadata: dict[str, Any]) -> Optional[str]:
        """Return a string suitable for the highest_degree field."""

        def extract_degree(value: Any) -> Optional[str]:
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, dict):
                for key in ("degree", "name", "title", "level"):
                    candidate = value.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate.strip()
            if isinstance(value, list):
                for item in value:
                    extracted = extract_degree(item)
                    if extracted:
                        return extracted
            return None

        degree = extract_degree(metadata.get("highest_degree"))
        if degree:
            return degree

        degree = extract_degree(metadata.get("education"))
        if degree:
            return degree

        return None

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

    @staticmethod
    def _extract_experience_years(metadata: dict[str, Any]) -> float:
        """Infer total experience in years from metadata payload."""

        if not metadata:
            return 0.0

        raw_value = metadata.get("total_experience_years")
        if isinstance(raw_value, (int, float)):
            return float(raw_value)
        if isinstance(raw_value, str):
            try:
                return float(raw_value)
            except ValueError:
                pass

        level = metadata.get("experience_level")
        if isinstance(level, str) and level.strip():
            return SearchApplicationService._map_experience_level_to_years(level)
        if isinstance(level, list):
            for item in level:
                if isinstance(item, str) and item.strip():
                    return SearchApplicationService._map_experience_level_to_years(item)
                if isinstance(item, dict):
                    label = item.get("level") or item.get("name") or item.get("title")
                    if isinstance(label, str) and label.strip():
                        return SearchApplicationService._map_experience_level_to_years(label)
        if isinstance(level, dict):
            label = (
                level.get("level")
                or level.get("name")
                or level.get("title")
            )
            if isinstance(label, str) and label.strip():
                return SearchApplicationService._map_experience_level_to_years(label)

        return 0.0

    @staticmethod
    def _map_experience_level_to_years(level: str) -> float:
        """Convert experience level label to approximate experience years."""

        mapping = {
            "entry": 0.5,
            "intern": 0.5,
            "junior": 2.0,
            "associate": 3.0,
            "mid": 5.0,
            "intermediate": 5.0,
            "senior": 8.0,
            "lead": 10.0,
            "principal": 12.0,
            "staff": 12.0,
            "executive": 15.0,
            "manager": 9.0,
        }

        if not isinstance(level, str):
            return 0.0

        return mapping.get(level.lower(), 0.0)

    @staticmethod
    def _score_experience_years(experience_years: float, requirements: Any) -> float:
        """Score experience years against requirement thresholds."""

        score = 1.0
        min_years = getattr(requirements, "min_total_years", None)
        max_years = getattr(requirements, "max_total_years", None)

        if min_years is not None:
            if experience_years >= min_years:
                excess = experience_years - min_years
                score = min(1.0, 0.8 + (excess * 0.02))
            else:
                shortfall = min_years - experience_years
                score = max(0.0, 0.8 - (shortfall * 0.1))

        if max_years is not None and experience_years > max_years:
            over_qualified = experience_years - max_years
            if over_qualified > 5:
                score *= 0.9

        return max(0.0, min(score, 1.0))

    @staticmethod
    def _score_education(metadata: dict[str, Any], requirements: Any) -> float:
        """Score candidate education against requirements."""

        metadata_safe = metadata or {}
        candidate_degrees: List[str] = []

        def collect(value: Any) -> None:
            if isinstance(value, str) and value.strip():
                candidate_degrees.append(value)
            elif isinstance(value, dict):
                for key in ("degree", "name", "title", "level"):
                    label = value.get(key)
                    if isinstance(label, str) and label.strip():
                        candidate_degrees.append(label)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(metadata_safe.get("highest_degree"))
        collect(metadata_safe.get("education"))

        if not candidate_degrees:
            return 0.0

        required_levels = getattr(requirements, "required_degree_levels", None) or []
        if not required_levels:
            return 1.0

        candidate_level = max(
            SearchApplicationService._map_degree_level(degree)
            for degree in candidate_degrees
        )
        required_level = max(
            SearchApplicationService._map_degree_level(level)
            for level in required_levels
        )

        if candidate_level >= required_level:
            return 1.0
        if candidate_level == 0:
            return 0.0

        return max(0.0, candidate_level / required_level)

    @staticmethod
    def _map_degree_level(degree: str) -> int:
        """Map degree labels to hierarchy values."""

        hierarchy = {
            "doctoral": 6,
            "phd": 6,
            "doctorate": 6,
            "master": 5,
            "masters": 5,
            "msc": 5,
            "mba": 5,
            "bachelor": 4,
            "bachelors": 4,
            "bsc": 4,
            "ba": 4,
            "associate": 3,
            "diploma": 2,
            "certificate": 1,
        }

        if not isinstance(degree, str):
            return 0

        degree_lower = degree.lower()
        for label, level in hierarchy.items():
            if label in degree_lower:
                return level
        return 0

    @staticmethod
    def _score_location(metadata: dict[str, Any], location_filter: Any) -> float:
        """Score candidate location against location filter."""

        metadata_safe = metadata or {}
        candidate_location = metadata_safe.get("location")
        candidate_city = metadata_safe.get("location_city")
        candidate_country = metadata_safe.get("location_country")

        preferred_locations = getattr(location_filter, "preferred_locations", None) or []
        center_location = getattr(location_filter, "center_location", None)

        def normalize(value: Optional[str]) -> str:
            return value.lower().strip() if isinstance(value, str) else ""

        candidate_tokens: set[str] = set()

        def add_token(value: Any) -> None:
            if isinstance(value, str):
                token = normalize(value)
                if token:
                    candidate_tokens.add(token)
            elif isinstance(value, list):
                for item in value:
                    add_token(item)
            elif isinstance(value, dict):
                for key in ("city", "country", "state", "name"):
                    if key in value:
                        add_token(value[key])

        add_token(candidate_location)
        add_token(candidate_city)
        add_token(candidate_country)
        add_token(metadata_safe.get("location_state"))

        preferred_tokens = {normalize(loc) for loc in preferred_locations}
        preferred_tokens.discard("")

        center_token = normalize(center_location)

        if preferred_tokens and candidate_tokens & preferred_tokens:
            return 1.0

        if center_token and center_token in candidate_tokens:
            return 0.9

        return 0.0

    @staticmethod
    def _score_salary(metadata: dict[str, Any], salary_filter: Any) -> float:
        """Score salary alignment."""

        metadata_safe = metadata or {}
        expected_salary = metadata_safe.get("expected_salary")
        if expected_salary is None:
            return 0.0

        def parse_salary(value: Any) -> Optional[float]:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                digits = "".join(
                    ch for ch in value if ch.isdigit() or ch in {".", "-", "+"}
                )
                if not digits:
                    return None
                try:
                    return float(digits)
                except ValueError:
                    return None
            if isinstance(value, dict):
                for key in ("amount", "value", "expected_salary", "min", "max"):
                    parsed = parse_salary(value.get(key))
                    if parsed is not None:
                        return parsed
            if isinstance(value, list):
                for item in value:
                    parsed = parse_salary(item)
                    if parsed is not None:
                        return parsed
            return None

        expected_value = parse_salary(expected_salary)
        if expected_value is None:
            return 0.0

        min_salary = getattr(salary_filter, "min_salary", None)
        max_salary = getattr(salary_filter, "max_salary", None)

        if min_salary is not None and expected_value < float(min_salary):
            return 0.0

        if max_salary is not None and expected_value > float(max_salary):
            return 0.0

        return 1.0


__all__ = ["SearchApplicationService"]
