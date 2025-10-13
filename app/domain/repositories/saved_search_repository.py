"""Domain repository contracts for saved search aggregates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from app.domain.entities.saved_search import SavedSearch, SavedSearchStatus
from app.domain.value_objects import SavedSearchId, TenantId, UserId


class ISavedSearchRepository(ABC):
    """Domain-facing abstraction for saved search persistence operations."""

    @abstractmethod
    async def get_by_id(
        self,
        saved_search_id: SavedSearchId,
        tenant_id: TenantId
    ) -> Optional[SavedSearch]:
        """Load a saved search by identifier within tenant scope."""
        raise NotImplementedError

    @abstractmethod
    async def save(self, saved_search: SavedSearch) -> SavedSearch:
        """Persist changes to a saved search aggregate."""
        raise NotImplementedError

    @abstractmethod
    async def list_by_user(
        self,
        user_id: UserId,
        tenant_id: TenantId,
        status: Optional[SavedSearchStatus] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[SavedSearch]:
        """List saved searches for a user (owned or shared)."""
        raise NotImplementedError

    @abstractmethod
    async def list_by_tenant(
        self,
        tenant_id: TenantId,
        status: Optional[SavedSearchStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[SavedSearch]:
        """List all saved searches for a tenant."""
        raise NotImplementedError

    @abstractmethod
    async def list_pending_alerts(
        self,
        before_time: datetime,
        limit: int = 100
    ) -> List[SavedSearch]:
        """List saved searches with alerts due before specified time."""
        raise NotImplementedError

    @abstractmethod
    async def delete(
        self,
        saved_search_id: SavedSearchId,
        tenant_id: TenantId
    ) -> bool:
        """Delete a saved search (hard delete)."""
        raise NotImplementedError

    @abstractmethod
    async def count_by_user(
        self,
        user_id: UserId,
        tenant_id: TenantId,
        status: Optional[SavedSearchStatus] = None
    ) -> int:
        """Count saved searches for a user."""
        raise NotImplementedError

    @abstractmethod
    async def get_by_name(
        self,
        name: str,
        user_id: UserId,
        tenant_id: TenantId
    ) -> Optional[SavedSearch]:
        """Get a saved search by name for a specific user."""
        raise NotImplementedError


__all__ = ["ISavedSearchRepository"]