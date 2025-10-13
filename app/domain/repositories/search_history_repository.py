"""Domain repository contracts for search history aggregates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any

from app.domain.entities.search_history import SearchHistory, SearchOutcome
from app.domain.value_objects import SearchHistoryId, TenantId, UserId


class ISearchHistoryRepository(ABC):
    """Domain-facing abstraction for search history persistence operations."""

    @abstractmethod
    async def save(self, search_history: SearchHistory) -> SearchHistory:
        """Persist a search history record."""
        raise NotImplementedError

    @abstractmethod
    async def get_by_id(
        self,
        search_history_id: SearchHistoryId,
        tenant_id: TenantId
    ) -> Optional[SearchHistory]:
        """Load a search history record by identifier within tenant scope."""
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    async def count_by_user(
        self,
        user_id: UserId,
        tenant_id: TenantId,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> int:
        """Count search history records for a user."""
        raise NotImplementedError

    @abstractmethod
    async def get_popular_queries(
        self,
        tenant_id: TenantId,
        limit: int = 10,
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """Get most popular search queries in tenant."""
        raise NotImplementedError

    @abstractmethod
    async def get_user_query_suggestions(
        self,
        user_id: UserId,
        tenant_id: TenantId,
        query_prefix: str,
        limit: int = 5
    ) -> List[str]:
        """Get search suggestions based on user's history."""
        raise NotImplementedError

    @abstractmethod
    async def get_analytics_summary(
        self,
        tenant_id: TenantId,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get aggregated analytics for a time period."""
        raise NotImplementedError

    @abstractmethod
    async def delete_old_records(
        self,
        tenant_id: TenantId,
        before_date: datetime
    ) -> int:
        """Delete search history records older than specified date (data retention)."""
        raise NotImplementedError


__all__ = ["ISearchHistoryRepository"]