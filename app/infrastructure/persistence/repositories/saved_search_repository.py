"""PostgreSQL implementation of ISavedSearchRepository using SavedSearchMapper."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import desc
from sqlmodel import select

from app.domain.entities.saved_search import SavedSearch, SavedSearchStatus
from app.domain.repositories.saved_search_repository import ISavedSearchRepository
from app.domain.value_objects import SavedSearchId, TenantId, UserId
from app.infrastructure.persistence.mappers.saved_search_mapper import SavedSearchMapper
from app.infrastructure.persistence.models.saved_search_table import SavedSearchTable
from app.infrastructure.providers.postgres_provider import get_postgres_adapter


class PostgresSavedSearchRepository(ISavedSearchRepository):
    """PostgreSQL adapter implementation of ISavedSearchRepository."""

    def __init__(self):
        self._adapter = None

    async def _get_adapter(self):
        """Get database adapter (lazy initialization)."""
        if self._adapter is None:
            self._adapter = await get_postgres_adapter()
        return self._adapter

    async def save(self, saved_search: SavedSearch) -> SavedSearch:
        """Save saved search to database and return updated domain entity."""
        adapter = await self._get_adapter()

        try:
            saved_search_table = SavedSearchMapper.to_table(saved_search)
            db_manager = adapter.db_manager

            async with db_manager.get_session() as session:
                existing_row = await session.get(SavedSearchTable, saved_search_table.id)

                if existing_row:
                    SavedSearchMapper.update_table_from_domain(existing_row, saved_search)
                else:
                    session.add(saved_search_table)

            return saved_search

        except Exception as e:
            raise Exception(f"Failed to save saved search: {str(e)}")

    async def get_by_id(
        self,
        saved_search_id: SavedSearchId,
        tenant_id: TenantId
    ) -> Optional[SavedSearch]:
        """Load a saved search by identifier within tenant scope."""
        adapter = await self._get_adapter()

        try:
            db_manager = adapter.db_manager
            async with db_manager.get_session() as session:
                stmt = select(SavedSearchTable).where(
                    SavedSearchTable.id == saved_search_id.value,
                    SavedSearchTable.tenant_id == tenant_id.value,
                )

                result = await session.execute(stmt)
                table_obj = result.scalars().first()

                if not table_obj:
                    return None

                return SavedSearchMapper.to_domain(table_obj)

        except Exception as e:
            raise Exception(f"Failed to get saved search by ID: {str(e)}")

    async def list_by_user(
        self,
        user_id: UserId,
        tenant_id: TenantId,
        status: Optional[SavedSearchStatus] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[SavedSearch]:
        """List saved searches for a user (owned or shared)."""
        adapter = await self._get_adapter()

        try:
            db_manager = adapter.db_manager
            async with db_manager.get_session() as session:
                # User can see searches they created or that are shared with them
                stmt = select(SavedSearchTable).where(
                    SavedSearchTable.tenant_id == tenant_id.value,
                    (
                        (SavedSearchTable.created_by == user_id.value) |
                        (SavedSearchTable.shared_with_users.contains([str(user_id.value)]))
                    )
                )

                if status:
                    stmt = stmt.where(SavedSearchTable.status == status.value)

                stmt = (
                    stmt.order_by(desc(SavedSearchTable.updated_at))
                    .offset(offset)
                    .limit(limit)
                )

                result = await session.execute(stmt)
                rows = result.scalars().all()

            return [SavedSearchMapper.to_domain(row) for row in rows]

        except Exception as e:
            raise Exception(f"Failed to list saved searches by user: {str(e)}")

    async def list_by_tenant(
        self,
        tenant_id: TenantId,
        status: Optional[SavedSearchStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[SavedSearch]:
        """List all saved searches for a tenant."""
        adapter = await self._get_adapter()

        try:
            db_manager = adapter.db_manager
            async with db_manager.get_session() as session:
                stmt = select(SavedSearchTable).where(
                    SavedSearchTable.tenant_id == tenant_id.value
                )

                if status:
                    stmt = stmt.where(SavedSearchTable.status == status.value)

                stmt = (
                    stmt.order_by(desc(SavedSearchTable.updated_at))
                    .offset(offset)
                    .limit(limit)
                )

                result = await session.execute(stmt)
                rows = result.scalars().all()

            return [SavedSearchMapper.to_domain(row) for row in rows]

        except Exception as e:
            raise Exception(f"Failed to list saved searches by tenant: {str(e)}")

    async def list_pending_alerts(
        self,
        before_time: datetime,
        limit: int = 100
    ) -> List[SavedSearch]:
        """List saved searches with alerts due before specified time."""
        adapter = await self._get_adapter()

        try:
            db_manager = adapter.db_manager
            async with db_manager.get_session() as session:
                stmt = (
                    select(SavedSearchTable)
                    .where(
                        SavedSearchTable.is_alert == True,
                        SavedSearchTable.status == SavedSearchStatus.ACTIVE.value,
                        SavedSearchTable.next_alert_at <= before_time,
                    )
                    .order_by(SavedSearchTable.next_alert_at)
                    .limit(limit)
                )

                result = await session.execute(stmt)
                rows = result.scalars().all()

            return [SavedSearchMapper.to_domain(row) for row in rows]

        except Exception as e:
            raise Exception(f"Failed to list pending alerts: {str(e)}")

    async def delete(
        self,
        saved_search_id: SavedSearchId,
        tenant_id: TenantId
    ) -> bool:
        """Delete a saved search (hard delete)."""
        adapter = await self._get_adapter()

        try:
            result = await adapter.execute(
                "DELETE FROM saved_searches WHERE id = $1 AND tenant_id = $2",
                saved_search_id.value,
                tenant_id.value,
            )

            return result and result.split()[-1] != "0"

        except Exception as e:
            raise Exception(f"Failed to delete saved search: {str(e)}")

    async def count_by_user(
        self,
        user_id: UserId,
        tenant_id: TenantId,
        status: Optional[SavedSearchStatus] = None
    ) -> int:
        """Count saved searches for a user."""
        adapter = await self._get_adapter()

        try:
            if status:
                record = await adapter.fetch_one(
                    """
                    SELECT COUNT(*) as count FROM saved_searches
                    WHERE tenant_id = $1
                    AND (created_by = $2 OR $2 = ANY(shared_with_users))
                    AND status = $3
                    """,
                    tenant_id.value,
                    str(user_id.value),
                    status.value,
                )
            else:
                record = await adapter.fetch_one(
                    """
                    SELECT COUNT(*) as count FROM saved_searches
                    WHERE tenant_id = $1
                    AND (created_by = $2 OR $2 = ANY(shared_with_users))
                    """,
                    tenant_id.value,
                    str(user_id.value),
                )

            return record["count"] if record else 0

        except Exception as e:
            raise Exception(f"Failed to count saved searches: {str(e)}")

    async def get_by_name(
        self,
        name: str,
        user_id: UserId,
        tenant_id: TenantId
    ) -> Optional[SavedSearch]:
        """Get a saved search by name for a specific user."""
        adapter = await self._get_adapter()

        try:
            db_manager = adapter.db_manager
            async with db_manager.get_session() as session:
                stmt = select(SavedSearchTable).where(
                    SavedSearchTable.tenant_id == tenant_id.value,
                    SavedSearchTable.created_by == user_id.value,
                    SavedSearchTable.name == name,
                )

                result = await session.execute(stmt)
                table_obj = result.scalars().first()

                if not table_obj:
                    return None

                return SavedSearchMapper.to_domain(table_obj)

        except Exception as e:
            raise Exception(f"Failed to get saved search by name: {str(e)}")


__all__ = ["PostgresSavedSearchRepository"]