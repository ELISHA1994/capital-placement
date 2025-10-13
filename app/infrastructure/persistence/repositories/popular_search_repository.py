"""
Popular Search Repository

Repository for retrieving popular tenant searches from materialized view.
"""

from typing import List, Optional
from uuid import UUID
import structlog

from app.domain.interfaces import ISuggestionSourceRepository
from app.domain.value_objects import SearchSuggestion, SuggestionSource, TenantId
from app.infrastructure.adapters.postgres_adapter import PostgresAdapter

logger = structlog.get_logger(__name__)


class PopularSearchRepository(ISuggestionSourceRepository):
    """
    Repository for retrieving popular tenant searches.

    Uses materialized view for performance, refreshed hourly.
    Provides suggestions based on frequently searched queries within a tenant.
    """

    def __init__(self, db_adapter: PostgresAdapter):
        self.db = db_adapter

    async def get_suggestions(
        self,
        prefix: str,
        tenant_id: TenantId,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[SearchSuggestion]:
        """
        Get popular searches matching prefix for tenant.

        Args:
            prefix: Query prefix to match (min 2 chars)
            tenant_id: Tenant context for isolation
            user_id: Not used for popular searches
            limit: Maximum suggestions to return

        Returns:
            List of SearchSuggestion objects ranked by frequency
        """
        try:
            # Use ILIKE for case-insensitive prefix matching
            query = """
                SELECT
                    query,
                    frequency,
                    last_used,
                    success_rate
                FROM tenant_popular_searches
                WHERE tenant_id = $1
                    AND query ILIKE $2 || '%'
                ORDER BY frequency DESC, success_rate DESC
                LIMIT $3
            """

            async with self.db.get_connection() as conn:
                rows = await conn.fetch(
                    query,
                    UUID(str(tenant_id.value)),
                    prefix.lower(),
                    limit
                )

            suggestions = []
            for row in rows:
                # Score based on frequency and success rate
                # Normalize frequency to 0-1 range (cap at 100 searches)
                freq_score = min(row['frequency'] / 100.0, 1.0)
                success_score = float(row['success_rate']) if row['success_rate'] else 0.0

                # Weighted combination: 70% frequency, 30% success rate
                score = min(0.7 * freq_score + 0.3 * success_score, 1.0)

                suggestions.append(SearchSuggestion(
                    text=row['query'],
                    source=SuggestionSource.TENANT_POPULAR,
                    score=score,
                    frequency=row['frequency'],
                    last_used=row['last_used'].isoformat() if row['last_used'] else None,
                    metadata={'success_rate': float(success_score)}
                ))

            logger.debug(
                "Popular search suggestions retrieved",
                prefix=prefix,
                tenant_id=str(tenant_id.value),
                count=len(suggestions)
            )

            return suggestions

        except Exception as e:
            logger.error(
                "Failed to get popular search suggestions",
                error=str(e),
                prefix=prefix,
                tenant_id=str(tenant_id.value)
            )
            # Return empty list on error (graceful degradation)
            return []