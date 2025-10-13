"""
User History Repository

Repository for retrieving user's personal search history for personalized suggestions.
"""

from typing import List, Optional
from datetime import datetime, timedelta
from uuid import UUID
import structlog

from app.domain.interfaces import ISuggestionSourceRepository
from app.domain.value_objects import SearchSuggestion, SuggestionSource, TenantId
from app.infrastructure.adapters.postgres_adapter import PostgresAdapter

logger = structlog.get_logger(__name__)


class UserHistoryRepository(ISuggestionSourceRepository):
    """
    Repository for retrieving user's personal search history.

    Provides personalized suggestions based on user's past queries.
    Only considers searches from the last 30 days for freshness.
    """

    def __init__(self, db_adapter: PostgresAdapter):
        self.db = db_adapter
        self.lookback_days = 30  # Only consider last 30 days

    async def get_suggestions(
        self,
        prefix: str,
        tenant_id: TenantId,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[SearchSuggestion]:
        """
        Get user's recent searches matching prefix.

        Args:
            prefix: Query prefix to match
            tenant_id: Tenant context for isolation
            user_id: User to get history for (required)
            limit: Maximum suggestions to return

        Returns:
            List of SearchSuggestion objects ranked by recency and frequency
        """
        if not user_id:
            # No user context, skip personalization
            return []

        try:
            cutoff_date = datetime.now() - timedelta(days=self.lookback_days)

            query = """
                SELECT
                    query,
                    COUNT(*) as frequency,
                    MAX(timestamp) as last_used,
                    SUM(CASE WHEN clicked THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as click_rate
                FROM user_search_history
                WHERE tenant_id = $1
                    AND user_id = $2
                    AND query ILIKE $3 || '%'
                    AND timestamp >= $4
                GROUP BY query
                ORDER BY
                    MAX(timestamp) DESC,  -- Recency first
                    COUNT(*) DESC         -- Then frequency
                LIMIT $5
            """

            async with self.db.get_connection() as conn:
                rows = await conn.fetch(
                    query,
                    UUID(str(tenant_id.value)),
                    UUID(user_id),
                    prefix.lower(),
                    cutoff_date,
                    limit
                )

            suggestions = []
            for row in rows:
                # Score based on recency and click rate
                days_ago = (datetime.now() - row['last_used'].replace(tzinfo=None)).days
                recency_score = max(0.0, 1.0 - (days_ago / self.lookback_days))

                click_rate = float(row['click_rate']) if row['click_rate'] else 0.0

                # Weighted combination: 60% recency, 40% click rate
                score = min(0.6 * recency_score + 0.4 * click_rate, 1.0)

                suggestions.append(SearchSuggestion(
                    text=row['query'],
                    source=SuggestionSource.USER_HISTORY,
                    score=score,
                    frequency=row['frequency'],
                    last_used=row['last_used'].isoformat(),
                    metadata={'click_rate': float(click_rate)}
                ))

            logger.debug(
                "User history suggestions retrieved",
                prefix=prefix,
                user_id=user_id,
                count=len(suggestions)
            )

            return suggestions

        except Exception as e:
            logger.error(
                "Failed to get user history suggestions",
                error=str(e),
                prefix=prefix,
                user_id=user_id
            )
            # Return empty list on error (graceful degradation)
            return []