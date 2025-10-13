"""PostgreSQL implementation of ISearchClickRepository with time-series optimizations."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy import desc, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlmodel import select as sqlmodel_select

from app.domain.entities.search_click import SearchClick
from app.domain.repositories.search_click_repository import ISearchClickRepository
from app.domain.value_objects import SearchClickId, TenantId, UserId, ProfileId
from app.infrastructure.persistence.mappers.search_click_mapper import SearchClickMapper
from app.infrastructure.persistence.models.search_click_table import SearchClickTable
from app.infrastructure.providers.postgres_provider import get_postgres_adapter


class PostgresSearchClickRepository(ISearchClickRepository):
    """
    High-performance PostgreSQL implementation for search click events.

    Optimizations:
    - Batch inserts for high throughput
    - Partition-aware queries
    - BRIN index utilization for time-series
    - Prepared statements for common queries
    - Connection pooling
    """

    def __init__(self):
        self._adapter = None

    async def _get_adapter(self):
        """Get database adapter (lazy initialization)."""
        if self._adapter is None:
            self._adapter = await get_postgres_adapter()
        return self._adapter

    async def save(self, click: SearchClick) -> SearchClick:
        """
        Save single click event with upsert semantics (idempotency).

        Uses INSERT ... ON CONFLICT DO NOTHING for idempotency.
        """
        adapter = await self._get_adapter()

        try:
            click_table = SearchClickMapper.to_table(click)
            db_manager = adapter.db_manager

            async with db_manager.get_session() as session:
                # Use INSERT ... ON CONFLICT for idempotency
                stmt = insert(SearchClickTable).values(**click_table.model_dump())
                stmt = stmt.on_conflict_do_nothing(index_elements=["id"])
                await session.execute(stmt)

            return click

        except Exception as e:
            raise Exception(f"Failed to save click event: {str(e)}")

    async def save_batch(self, clicks: List[SearchClick]) -> int:
        """
        High-performance batch insert for multiple click events.

        Uses COPY or bulk INSERT for maximum throughput.
        """
        if not clicks:
            return 0

        adapter = await self._get_adapter()

        try:
            tables = SearchClickMapper.to_table_batch(clicks)
            db_manager = adapter.db_manager

            async with db_manager.get_session() as session:
                # Batch insert with conflict handling
                values = [t.model_dump() for t in tables]
                stmt = insert(SearchClickTable).values(values)
                stmt = stmt.on_conflict_do_nothing(index_elements=["id"])
                await session.execute(stmt)

            return len(clicks)

        except Exception as e:
            raise Exception(f"Failed to batch save click events: {str(e)}")

    async def get_by_id(
        self,
        click_id: SearchClickId,
        tenant_id: TenantId
    ) -> Optional[SearchClick]:
        """Load a specific click event (rarely used)."""
        adapter = await self._get_adapter()

        try:
            db_manager = adapter.db_manager
            async with db_manager.get_session() as session:
                stmt = select(SearchClickTable).where(
                    SearchClickTable.id == click_id.value,
                    SearchClickTable.tenant_id == tenant_id.value,
                )

                result = await session.execute(stmt)
                table_obj = result.scalars().first()

                if not table_obj:
                    return None

                return SearchClickMapper.to_domain(table_obj)

        except Exception as e:
            raise Exception(f"Failed to get click by ID: {str(e)}")

    async def get_clicks_for_search(
        self,
        search_id: str,
        tenant_id: TenantId,
        limit: int = 100
    ) -> List[SearchClick]:
        """Get all clicks for a specific search (CTR analysis)."""
        adapter = await self._get_adapter()

        try:
            db_manager = adapter.db_manager
            async with db_manager.get_session() as session:
                stmt = (
                    select(SearchClickTable)
                    .where(
                        SearchClickTable.search_id == search_id,
                        SearchClickTable.tenant_id == tenant_id.value,
                    )
                    .order_by(SearchClickTable.clicked_at)
                    .limit(limit)
                )

                result = await session.execute(stmt)
                rows = result.scalars().all()

            return [SearchClickMapper.to_domain(row) for row in rows]

        except Exception as e:
            raise Exception(f"Failed to get clicks for search: {str(e)}")

    async def get_clicks_for_profile(
        self,
        profile_id: ProfileId,
        tenant_id: TenantId,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[SearchClick]:
        """Get clicks for a profile (popularity tracking)."""
        adapter = await self._get_adapter()

        try:
            db_manager = adapter.db_manager
            async with db_manager.get_session() as session:
                stmt = select(SearchClickTable).where(
                    SearchClickTable.profile_id == profile_id.value,
                    SearchClickTable.tenant_id == tenant_id.value,
                )

                if start_date:
                    stmt = stmt.where(SearchClickTable.clicked_at >= start_date)
                if end_date:
                    stmt = stmt.where(SearchClickTable.clicked_at <= end_date)

                stmt = stmt.order_by(desc(SearchClickTable.clicked_at)).limit(limit)

                result = await session.execute(stmt)
                rows = result.scalars().all()

            return [SearchClickMapper.to_domain(row) for row in rows]

        except Exception as e:
            raise Exception(f"Failed to get clicks for profile: {str(e)}")

    async def count_clicks_by_search(
        self,
        search_id: str,
        tenant_id: TenantId
    ) -> int:
        """Count total clicks for a search."""
        adapter = await self._get_adapter()

        try:
            db_manager = adapter.db_manager
            async with db_manager.get_session() as session:
                stmt = (
                    select(func.count())
                    .select_from(SearchClickTable)
                    .where(
                        SearchClickTable.search_id == search_id,
                        SearchClickTable.tenant_id == tenant_id.value,
                    )
                )

                result = await session.execute(stmt)
                count = result.scalar_one()

            return count or 0

        except Exception as e:
            raise Exception(f"Failed to count clicks: {str(e)}")

    async def get_click_through_rate(
        self,
        tenant_id: TenantId,
        start_date: datetime,
        end_date: datetime,
        group_by: str = "day"
    ) -> List[Dict[str, Any]]:
        """
        Calculate CTR with time-series grouping.

        Optimized query using date_trunc for grouping.
        """
        adapter = await self._get_adapter()

        try:
            # Determine truncation level
            trunc_level = {
                "hour": "hour",
                "day": "day",
                "week": "week",
                "month": "month"
            }.get(group_by, "day")

            query = f"""
                SELECT
                    DATE_TRUNC('{trunc_level}', clicked_at) as time_bucket,
                    COUNT(*) as click_count,
                    COUNT(DISTINCT search_id) as search_count,
                    COUNT(DISTINCT user_id) as user_count,
                    AVG(position) as avg_position,
                    SUM(CASE WHEN position < 3 THEN 1 ELSE 0 END) as top_3_clicks,
                    SUM(CASE WHEN engagement_signal = 'strong' THEN 1 ELSE 0 END) as strong_engagement
                FROM search_clicks
                WHERE tenant_id = $1
                AND clicked_at BETWEEN $2 AND $3
                GROUP BY time_bucket
                ORDER BY time_bucket
            """

            records = await adapter.fetch_all(
                query,
                tenant_id.value,
                start_date,
                end_date
            )

            return [
                {
                    "time_bucket": rec["time_bucket"],
                    "click_count": rec["click_count"],
                    "search_count": rec["search_count"],
                    "ctr": rec["click_count"] / max(rec["search_count"], 1),
                    "user_count": rec["user_count"],
                    "avg_position": float(rec["avg_position"]) if rec["avg_position"] else 0.0,
                    "top_3_clicks": rec["top_3_clicks"],
                    "strong_engagement": rec["strong_engagement"],
                }
                for rec in records
            ]

        except Exception as e:
            raise Exception(f"Failed to calculate CTR: {str(e)}")

    async def get_position_analytics(
        self,
        tenant_id: TenantId,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[int, Dict[str, Any]]:
        """Get click statistics by result position."""
        adapter = await self._get_adapter()

        try:
            query = """
                SELECT
                    position,
                    COUNT(*) as click_count,
                    AVG((context->>'time_to_click_ms')::INTEGER) as avg_time_to_click,
                    AVG(rank_quality) as avg_rank_quality,
                    COUNT(DISTINCT search_id) as search_count
                FROM search_clicks
                WHERE tenant_id = $1
                AND clicked_at BETWEEN $2 AND $3
                AND position < 100  -- Focus on first 100 positions
                GROUP BY position
                ORDER BY position
            """

            records = await adapter.fetch_all(
                query,
                tenant_id.value,
                start_date,
                end_date
            )

            return {
                rec["position"]: {
                    "click_count": rec["click_count"],
                    "avg_time_to_click_ms": float(rec["avg_time_to_click"]) if rec["avg_time_to_click"] else None,
                    "avg_rank_quality": float(rec["avg_rank_quality"]) if rec["avg_rank_quality"] else 0.0,
                    "search_count": rec["search_count"],
                    "ctr": rec["click_count"] / max(rec["search_count"], 1),
                }
                for rec in records
            }

        except Exception as e:
            raise Exception(f"Failed to get position analytics: {str(e)}")

    async def delete_old_events(
        self,
        before_date: datetime,
        tenant_id: Optional[TenantId] = None
    ) -> int:
        """
        Delete old events for data retention (GDPR compliance).

        For partitioned tables, prefer DROP PARTITION for performance.
        """
        adapter = await self._get_adapter()

        try:
            if tenant_id:
                query = """
                    DELETE FROM search_clicks
                    WHERE clicked_at < $1
                    AND tenant_id = $2
                """
                result = await adapter.execute(query, before_date, tenant_id.value)
            else:
                query = """
                    DELETE FROM search_clicks
                    WHERE clicked_at < $1
                """
                result = await adapter.execute(query, before_date)

            # Parse delete count from result
            deleted = int(result.split()[-1]) if result else 0
            return deleted

        except Exception as e:
            raise Exception(f"Failed to delete old events: {str(e)}")


__all__ = ["PostgresSearchClickRepository"]
