"""
Suggestion Application Service

Application service for generating search suggestions with multi-source aggregation.
"""

from typing import List, Optional
import asyncio
import math
import structlog

from app.domain.interfaces import ISuggestionSourceRepository, ISuggestionCache
from app.domain.value_objects import SearchSuggestion, TenantId, SuggestionSource
from app.domain.exceptions import ValidationException

logger = structlog.get_logger(__name__)


class SuggestionApplicationService:
    """
    Application service for generating search suggestions.

    Orchestrates multiple suggestion sources, ranking, merging,
    and caching to provide fast, relevant suggestions.

    Key responsibilities:
    - Query multiple suggestion sources in parallel
    - Merge and deduplicate suggestions
    - Rank by relevance and personalization
    - Manage caching for <100ms performance
    """

    def __init__(
        self,
        popular_search_repo: ISuggestionSourceRepository,
        user_history_repo: ISuggestionSourceRepository,
        dictionary_repo: ISuggestionSourceRepository,
        cache: ISuggestionCache
    ):
        self.popular_search_repo = popular_search_repo
        self.user_history_repo = user_history_repo
        self.dictionary_repo = dictionary_repo
        self.cache = cache

        # Configuration
        self.min_query_length = 2
        self.max_suggestions = 50
        self.cache_ttl = 3600  # 1 hour

        # Source weights for ranking
        self.source_weights = {
            SuggestionSource.USER_HISTORY: 1.2,      # Highest priority
            SuggestionSource.TENANT_POPULAR: 1.0,
            SuggestionSource.SKILL_DICTIONARY: 0.8,
            SuggestionSource.JOB_TITLE: 0.8,
            SuggestionSource.INDUSTRY_TERM: 0.7,
        }

    async def get_suggestions(
        self,
        query_prefix: str,
        tenant_id: TenantId,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[SearchSuggestion]:
        """
        Get search suggestions for query prefix.

        Args:
            query_prefix: Query prefix (min 2 chars)
            tenant_id: Tenant context
            user_id: Optional user for personalization
            limit: Maximum suggestions to return (1-50)

        Returns:
            Ranked list of SearchSuggestion objects

        Raises:
            ValidationException: If query_prefix is invalid
        """
        # Validate input
        if len(query_prefix) < self.min_query_length:
            raise ValidationException(
                f"Query prefix must be at least {self.min_query_length} characters"
            )

        if not 1 <= limit <= self.max_suggestions:
            raise ValidationException(
                f"Limit must be between 1 and {self.max_suggestions}"
            )

        try:
            # Check cache first
            cache_key = self._make_cache_key(query_prefix, tenant_id, user_id)
            cached_suggestions = await self.cache.get_suggestions(cache_key, tenant_id)

            if cached_suggestions:
                logger.debug(
                    "Suggestions served from cache",
                    prefix=query_prefix,
                    tenant_id=str(tenant_id.value),
                    cached_count=len(cached_suggestions)
                )
                return cached_suggestions[:limit]

            # Cache miss - gather from all sources in parallel
            suggestions = await self._gather_suggestions(
                query_prefix,
                tenant_id,
                user_id,
                limit * 3  # Fetch more for better ranking
            )

            # Merge and deduplicate
            merged_suggestions = self._merge_suggestions(suggestions)

            # Rank by relevance
            ranked_suggestions = self._rank_suggestions(
                merged_suggestions,
                user_id is not None
            )

            # Take top results
            final_suggestions = ranked_suggestions[:limit]

            # Cache for next time
            await self.cache.set_suggestions(
                cache_key,
                final_suggestions,
                tenant_id,
                self.cache_ttl
            )

            logger.info(
                "Suggestions generated",
                prefix=query_prefix,
                tenant_id=str(tenant_id.value),
                user_id=user_id,
                total_found=len(merged_suggestions),
                returned=len(final_suggestions)
            )

            return final_suggestions

        except ValidationException:
            raise
        except Exception as e:
            logger.error(
                "Failed to generate suggestions",
                error=str(e),
                prefix=query_prefix
            )
            # Return empty list on error (graceful degradation)
            return []

    async def _gather_suggestions(
        self,
        prefix: str,
        tenant_id: TenantId,
        user_id: Optional[str],
        limit: int
    ) -> dict:
        """
        Gather suggestions from all sources in parallel for speed.

        Args:
            prefix: Query prefix
            tenant_id: Tenant context
            user_id: User context (optional)
            limit: Total limit to fetch

        Returns:
            Dictionary with suggestions from each source
        """
        # Query all sources concurrently for speed
        user_history_task = self.user_history_repo.get_suggestions(
            prefix, tenant_id, user_id, limit // 2
        )

        popular_search_task = self.popular_search_repo.get_suggestions(
            prefix, tenant_id, user_id, limit // 2
        )

        dictionary_task = self.dictionary_repo.get_suggestions(
            prefix, tenant_id, user_id, limit
        )

        # Wait for all sources
        user_history, popular_searches, dictionary = await asyncio.gather(
            user_history_task,
            popular_search_task,
            dictionary_task,
            return_exceptions=True
        )

        # Handle any errors gracefully
        return {
            'user_history': user_history if isinstance(user_history, list) else [],
            'popular': popular_searches if isinstance(popular_searches, list) else [],
            'dictionary': dictionary if isinstance(dictionary, list) else []
        }

    def _merge_suggestions(
        self,
        suggestions_by_source: dict
    ) -> List[SearchSuggestion]:
        """
        Merge suggestions from multiple sources, deduplicating by text.

        For duplicates, keeps the one with highest score and combines metadata.

        Args:
            suggestions_by_source: Dict of source -> suggestions list

        Returns:
            Merged and deduplicated list of suggestions
        """
        # Use dict to deduplicate by text (case-insensitive)
        merged = {}

        for source_name, suggestions in suggestions_by_source.items():
            for suggestion in suggestions:
                key = suggestion.text.lower()

                if key not in merged:
                    # First occurrence
                    merged[key] = suggestion
                else:
                    # Duplicate - keep higher score, merge metadata
                    existing = merged[key]

                    if suggestion.score > existing.score:
                        merged[key] = suggestion

                    # Combine frequencies
                    combined_freq = merged[key].frequency + suggestion.frequency

                    # Merge metadata
                    existing_meta = merged[key].metadata or {}
                    new_meta = suggestion.metadata or {}

                    # Track which sources provided this suggestion
                    sources = existing_meta.get('sources', [merged[key].source.value])
                    if suggestion.source.value not in sources:
                        sources.append(suggestion.source.value)

                    merged_metadata = {
                        **existing_meta,
                        **new_meta,
                        'sources': sources
                    }

                    # Create new suggestion with merged data
                    merged[key] = SearchSuggestion(
                        text=merged[key].text,
                        source=merged[key].source,
                        score=max(merged[key].score, suggestion.score),
                        frequency=combined_freq,
                        last_used=merged[key].last_used or suggestion.last_used,
                        metadata=merged_metadata
                    )

        return list(merged.values())

    def _rank_suggestions(
        self,
        suggestions: List[SearchSuggestion],
        is_personalized: bool
    ) -> List[SearchSuggestion]:
        """
        Rank suggestions by relevance.

        Ranking factors:
        1. Base score from source
        2. Source weight (user history > popular > dictionary)
        3. Frequency (usage count)
        4. Personalization boost

        Args:
            suggestions: List of suggestions to rank
            is_personalized: Whether user context is available

        Returns:
            Sorted list by rank score (descending)
        """
        def calculate_rank_score(suggestion: SearchSuggestion) -> float:
            # Base score from source
            base_score = suggestion.score

            # Apply source weight
            source_weight = self.source_weights.get(suggestion.source, 1.0)

            # Boost user history if personalized
            if is_personalized and suggestion.source == SuggestionSource.USER_HISTORY:
                source_weight *= 1.3

            # Frequency bonus (logarithmic to avoid domination)
            frequency_bonus = math.log(suggestion.frequency + 1) / 10.0

            # Combine factors
            final_score = (base_score * source_weight) + frequency_bonus

            return min(final_score, 2.0)  # Cap at 2.0

        # Sort by calculated rank score
        ranked = sorted(
            suggestions,
            key=calculate_rank_score,
            reverse=True
        )

        return ranked

    async def invalidate_tenant_cache(self, tenant_id: TenantId) -> bool:
        """
        Invalidate all cached suggestions for a tenant.

        Call this when:
        - New search patterns emerge
        - Dictionary updates occur
        - Manual cache refresh needed

        Args:
            tenant_id: Tenant to invalidate cache for

        Returns:
            True if invalidation succeeded
        """
        try:
            success = await self.cache.invalidate_tenant(tenant_id)

            logger.info(
                "Tenant suggestion cache invalidated",
                tenant_id=str(tenant_id.value),
                success=success
            )

            return success

        except Exception as e:
            logger.error(
                "Failed to invalidate tenant cache",
                tenant_id=str(tenant_id.value),
                error=str(e)
            )
            return False

    @staticmethod
    def _make_cache_key(
        prefix: str,
        tenant_id: TenantId,
        user_id: Optional[str]
    ) -> str:
        """
        Generate cache key for suggestions.

        Args:
            prefix: Query prefix
            tenant_id: Tenant context
            user_id: User context (optional)

        Returns:
            Cache key string
        """
        prefix_normalized = prefix.lower().strip()

        if user_id:
            return f"suggestions:tenant:{tenant_id.value}:user:{user_id}:{prefix_normalized}"
        else:
            return f"suggestions:tenant:{tenant_id.value}:{prefix_normalized}"
