"""
Suggestion Cache Adapter

Redis-based cache for search suggestions with tenant isolation.
"""

from typing import List, Optional
import structlog

from app.domain.interfaces import ISuggestionCache
from app.domain.value_objects import SearchSuggestion, TenantId, SuggestionSource
from app.infrastructure.adapters.redis_cache_adapter import RedisCacheService

logger = structlog.get_logger(__name__)


class SuggestionCacheAdapter(ISuggestionCache):
    """
    Redis-based cache for search suggestions.

    Provides fast (sub-millisecond) retrieval of cached suggestions
    with tenant isolation and TTL management.
    """

    def __init__(self, redis_cache: RedisCacheService):
        self.cache = redis_cache
        self.default_ttl = 3600  # 1 hour
        self.key_prefix = "suggestions"

    def _make_cache_key(
        self,
        prefix: str,
        tenant_id: TenantId,
        user_id: Optional[str] = None
    ) -> str:
        """
        Generate cache key with tenant isolation.

        Args:
            prefix: Query prefix
            tenant_id: Tenant context
            user_id: Optional user for personalized caching

        Returns:
            Cache key string
        """
        prefix_normalized = prefix.lower().strip()

        if user_id:
            # Personalized cache key
            return f"{self.key_prefix}:tenant:{tenant_id.value}:user:{user_id}:{prefix_normalized}"
        else:
            # Tenant-wide cache key
            return f"{self.key_prefix}:tenant:{tenant_id.value}:{prefix_normalized}"

    async def get_suggestions(
        self,
        cache_key: str,
        tenant_id: TenantId
    ) -> Optional[List[SearchSuggestion]]:
        """
        Get cached suggestions.

        Args:
            cache_key: Cache key to retrieve
            tenant_id: Tenant context for validation

        Returns:
            List of SearchSuggestion objects or None if not cached
        """
        try:
            cached_data = await self.cache.get(cache_key)

            if not cached_data:
                return None

            # Deserialize suggestions
            suggestions = []
            for item in cached_data:
                suggestions.append(SearchSuggestion(
                    text=item['text'],
                    source=SuggestionSource(item['source']),
                    score=item['score'],
                    frequency=item.get('frequency', 0),
                    last_used=item.get('last_used'),
                    metadata=item.get('metadata')
                ))

            logger.debug(
                "Suggestions cache hit",
                cache_key=cache_key,
                count=len(suggestions)
            )

            return suggestions

        except Exception as e:
            logger.error(
                "Failed to get cached suggestions",
                error=str(e),
                cache_key=cache_key
            )
            return None

    async def set_suggestions(
        self,
        cache_key: str,
        suggestions: List[SearchSuggestion],
        tenant_id: TenantId,
        ttl_seconds: int = 3600
    ) -> bool:
        """
        Cache suggestions with TTL.

        Args:
            cache_key: Cache key to store under
            suggestions: List of suggestions to cache
            tenant_id: Tenant context for isolation
            ttl_seconds: Time-to-live in seconds

        Returns:
            True if caching succeeded
        """
        try:
            # Serialize suggestions
            cache_data = []
            for suggestion in suggestions:
                cache_data.append({
                    'text': suggestion.text,
                    'source': suggestion.source.value,
                    'score': suggestion.score,
                    'frequency': suggestion.frequency,
                    'last_used': suggestion.last_used,
                    'metadata': suggestion.metadata
                })

            success = await self.cache.set(cache_key, cache_data, ttl=ttl_seconds)

            if success:
                logger.debug(
                    "Suggestions cached",
                    cache_key=cache_key,
                    count=len(suggestions),
                    ttl=ttl_seconds
                )

            return success

        except Exception as e:
            logger.error(
                "Failed to cache suggestions",
                error=str(e),
                cache_key=cache_key
            )
            return False

    async def invalidate_tenant(self, tenant_id: TenantId) -> bool:
        """
        Invalidate all cached suggestions for tenant.

        Args:
            tenant_id: Tenant to invalidate cache for

        Returns:
            True if invalidation succeeded
        """
        try:
            pattern = f"{self.key_prefix}:tenant:{tenant_id.value}:*"
            deleted_count = await self.cache.clear(pattern)

            logger.info(
                "Tenant suggestions cache invalidated",
                tenant_id=str(tenant_id.value),
                deleted_count=deleted_count
            )

            return deleted_count > 0

        except Exception as e:
            logger.error(
                "Failed to invalidate tenant cache",
                error=str(e),
                tenant_id=str(tenant_id.value)
            )
            return False