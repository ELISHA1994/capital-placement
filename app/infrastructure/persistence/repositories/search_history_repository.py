"""PostgreSQL implementation of ISearchHistoryRepository using SearchHistoryMapper."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from sqlalchemy import desc, func, and_, or_
from sqlmodel import select

from app.domain.entities.search_history import SearchHistory, SearchOutcome
from app.domain.repositories.search_history_repository import ISearchHistoryRepository
from app.domain.value_objects import SearchHistoryId, TenantId, UserId
from app.infrastructure.persistence.mappers.search_history_mapper import SearchHistoryMapper
from app.infrastructure.persistence.models.search_history_table import SearchHistoryTable
from app.infrastructure.providers.postgres_provider import get_postgres_adapter


class PostgresSearchHistoryRepository(ISearchHistoryRepository):
    """PostgreSQL adapter implementation of ISearchHistoryRepository."""

    def __init__(self):
        self._adapter = None

    async def _get_adapter(self):
        """Get database adapter (lazy initialization)."""
        if self._adapter is None:
            self._adapter = await get_postgres_adapter()
        return self._adapter

    async def save(self, search_history: SearchHistory) -> SearchHistory:
        """Save search history to database and return updated domain entity."""
        adapter = await self._get_adapter()

        try:
            search_history_table = SearchHistoryMapper.to_table(search_history)
            db_manager = adapter.db_manager

            async with db_manager.get_session() as session:
                existing_row = await session.get(SearchHistoryTable, search_history_table.id)

                if existing_row:
                    SearchHistoryMapper.update_table_from_domain(existing_row, search_history)
                else:
                    session.add(search_history_table)

            return search_history

        except Exception as e:
            raise Exception(f"Failed to save search history: {str(e)}")

    async def get_by_id(
        self,
        search_history_id: SearchHistoryId,
        tenant_id: TenantId
    ) -> Optional[SearchHistory]:
        """Load a search history record by identifier within tenant scope."""
        adapter = await self._get_adapter()

        try:
            db_manager = adapter.db_manager
            async with db_manager.get_session() as session:
                stmt = select(SearchHistoryTable).where(
                    SearchHistoryTable.id == search_history_id.value,
                    SearchHistoryTable.tenant_id == tenant_id.value,
                )

                result = await session.execute(stmt)
                table_obj = result.scalars().first()

                if not table_obj:
                    return None

                return SearchHistoryMapper.to_domain(table_obj)

        except Exception as e:
            raise Exception(f"Failed to get search history by ID: {str(e)}")

    async def list_by_user(
        self,
        user_id: UserId,
        tenant_id: TenantId,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[SearchHistory]:
        """List search history for a user with optional date filtering."""
        adapter = await self._get_adapter()

        try:
            db_manager = adapter.db_manager
            async with db_manager.get_session() as session:
                stmt = select(SearchHistoryTable).where(
                    SearchHistoryTable.tenant_id == tenant_id.value,
                    SearchHistoryTable.user_id == user_id.value,
                )

                if start_date:
                    stmt = stmt.where(SearchHistoryTable.executed_at >= start_date)
                if end_date:
                    stmt = stmt.where(SearchHistoryTable.executed_at <= end_date)

                stmt = (
                    stmt.order_by(desc(SearchHistoryTable.executed_at))
                    .offset(offset)
                    .limit(limit)
                )

                result = await session.execute(stmt)
                rows = result.scalars().all()

            return [SearchHistoryMapper.to_domain(row) for row in rows]

        except Exception as e:
            raise Exception(f"Failed to list search history by user: {str(e)}")

    async def list_by_tenant(
        self,
        tenant_id: TenantId,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        outcome: Optional[SearchOutcome] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[SearchHistory]:
        """List search history for a tenant with filters."""
        adapter = await self._get_adapter()

        try:
            db_manager = adapter.db_manager
            async with db_manager.get_session() as session:
                stmt = select(SearchHistoryTable).where(
                    SearchHistoryTable.tenant_id == tenant_id.value
                )

                if start_date:
                    stmt = stmt.where(SearchHistoryTable.executed_at >= start_date)
                if end_date:
                    stmt = stmt.where(SearchHistoryTable.executed_at <= end_date)
                if outcome:
                    stmt = stmt.where(SearchHistoryTable.search_outcome == outcome.value)

                stmt = (
                    stmt.order_by(desc(SearchHistoryTable.executed_at))
                    .offset(offset)
                    .limit(limit)
                )

                result = await session.execute(stmt)
                rows = result.scalars().all()

            return [SearchHistoryMapper.to_domain(row) for row in rows]

        except Exception as e:
            raise Exception(f"Failed to list search history by tenant: {str(e)}")

    async def count_by_user(
        self,
        user_id: UserId,
        tenant_id: TenantId,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> int:
        """Count search history records for a user."""
        adapter = await self._get_adapter()

        try:
            conditions = [
                "tenant_id = $1",
                "user_id = $2"
            ]
            params = [tenant_id.value, user_id.value]
            param_count = 2

            if start_date:
                param_count += 1
                conditions.append(f"executed_at >= ${param_count}")
                params.append(start_date)

            if end_date:
                param_count += 1
                conditions.append(f"executed_at <= ${param_count}")
                params.append(end_date)

            where_clause = " AND ".join(conditions)

            record = await adapter.fetch_one(
                f"SELECT COUNT(*) as count FROM search_history WHERE {where_clause}",
                *params
            )

            return record["count"] if record else 0

        except Exception as e:
            raise Exception(f"Failed to count search history: {str(e)}")

    async def get_popular_queries(
        self,
        tenant_id: TenantId,
        limit: int = 10,
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """Get most popular search queries in tenant."""
        adapter = await self._get_adapter()

        try:
            start_date = datetime.utcnow() - timedelta(days=days_back)

            records = await adapter.fetch_all(
                """
                SELECT
                    query_text,
                    COUNT(*) as search_count,
                    AVG(total_results) as avg_results,
                    AVG(engagement_score) as avg_engagement
                FROM search_history
                WHERE tenant_id = $1
                AND executed_at >= $2
                AND search_outcome = 'success'
                GROUP BY query_text
                ORDER BY search_count DESC
                LIMIT $3
                """,
                tenant_id.value,
                start_date,
                limit
            )

            return [
                {
                    "query": record["query_text"],
                    "count": record["search_count"],
                    "avg_results": float(record["avg_results"]) if record["avg_results"] else 0,
                    "avg_engagement": float(record["avg_engagement"]) if record["avg_engagement"] else 0,
                }
                for record in records
            ]

        except Exception as e:
            raise Exception(f"Failed to get popular queries: {str(e)}")

    async def get_user_query_suggestions(
        self,
        user_id: UserId,
        tenant_id: TenantId,
        query_prefix: str,
        limit: int = 5
    ) -> List[str]:
        """Get search suggestions based on user's history."""
        adapter = await self._get_adapter()

        try:
            records = await adapter.fetch_all(
                """
                SELECT DISTINCT query_text, MAX(executed_at) as last_used
                FROM search_history
                WHERE tenant_id = $1
                AND user_id = $2
                AND query_text ILIKE $3
                AND search_outcome = 'success'
                GROUP BY query_text
                ORDER BY last_used DESC
                LIMIT $4
                """,
                tenant_id.value,
                user_id.value,
                f"{query_prefix}%",
                limit
            )

            return [record["query_text"] for record in records]

        except Exception as e:
            raise Exception(f"Failed to get query suggestions: {str(e)}")

    async def get_analytics_summary(
        self,
        tenant_id: TenantId,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get aggregated analytics for a time period."""
        adapter = await self._get_adapter()

        try:
            # Get overall statistics
            stats = await adapter.fetch_one(
                """
                SELECT
                    COUNT(*) as total_searches,
                    COUNT(DISTINCT user_id) as unique_users,
                    AVG(total_results) as avg_results,
                    AVG(search_duration_ms) as avg_duration_ms,
                    AVG(engagement_score) as avg_engagement,
                    SUM(CASE WHEN total_results = 0 THEN 1 ELSE 0 END) as zero_result_searches,
                    SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) as cache_hits
                FROM search_history
                WHERE tenant_id = $1
                AND executed_at >= $2
                AND executed_at <= $3
                """,
                tenant_id.value,
                start_date,
                end_date
            )

            # Get outcome distribution
            outcomes = await adapter.fetch_all(
                """
                SELECT search_outcome, COUNT(*) as count
                FROM search_history
                WHERE tenant_id = $1
                AND executed_at >= $2
                AND executed_at <= $3
                GROUP BY search_outcome
                """,
                tenant_id.value,
                start_date,
                end_date
            )

            return {
                "total_searches": stats["total_searches"],
                "unique_users": stats["unique_users"],
                "avg_results": float(stats["avg_results"]) if stats["avg_results"] else 0,
                "avg_duration_ms": float(stats["avg_duration_ms"]) if stats["avg_duration_ms"] else 0,
                "avg_engagement": float(stats["avg_engagement"]) if stats["avg_engagement"] else 0,
                "zero_result_rate": float(stats["zero_result_searches"]) / stats["total_searches"] if stats["total_searches"] > 0 else 0,
                "cache_hit_rate": float(stats["cache_hits"]) / stats["total_searches"] if stats["total_searches"] > 0 else 0,
                "outcome_distribution": {
                    record["search_outcome"]: record["count"]
                    for record in outcomes
                },
            }

        except Exception as e:
            raise Exception(f"Failed to get analytics summary: {str(e)}")

    async def delete_old_records(
        self,
        tenant_id: TenantId,
        before_date: datetime
    ) -> int:
        """Delete search history records older than specified date (data retention)."""
        adapter = await self._get_adapter()

        try:
            result = await adapter.execute(
                "DELETE FROM search_history WHERE tenant_id = $1 AND executed_at < $2",
                tenant_id.value,
                before_date
            )

            # Extract count from result string like "DELETE 123"
            if result:
                count = result.split()[-1]
                return int(count) if count.isdigit() else 0
            return 0

        except Exception as e:
            raise Exception(f"Failed to delete old records: {str(e)}")


__all__ = ["PostgresSearchHistoryRepository"]
