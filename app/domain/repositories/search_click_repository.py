"""Domain repository contracts for search click event persistence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any

from app.domain.entities.search_click import SearchClick
from app.domain.value_objects import SearchClickId, TenantId, UserId, ProfileId


class ISearchClickRepository(ABC):
    """Domain-facing abstraction for search click event persistence."""

    @abstractmethod
    async def save(self, click: SearchClick) -> SearchClick:
        """
        Persist a search click event.

        Optimized for high-throughput writes with minimal blocking.
        """
        raise NotImplementedError

    @abstractmethod
    async def save_batch(self, clicks: List[SearchClick]) -> int:
        """
        Persist multiple click events in a single transaction.

        Returns number of events successfully saved.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_by_id(
        self,
        click_id: SearchClickId,
        tenant_id: TenantId
    ) -> Optional[SearchClick]:
        """Load a specific click event (rarely used, mainly for auditing)."""
        raise NotImplementedError

    @abstractmethod
    async def get_clicks_for_search(
        self,
        search_id: str,
        tenant_id: TenantId,
        limit: int = 100
    ) -> List[SearchClick]:
        """Get all clicks for a specific search execution."""
        raise NotImplementedError

    @abstractmethod
    async def get_clicks_for_profile(
        self,
        profile_id: ProfileId,
        tenant_id: TenantId,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[SearchClick]:
        """Get all clicks for a specific profile (analytics)."""
        raise NotImplementedError

    @abstractmethod
    async def count_clicks_by_search(
        self,
        search_id: str,
        tenant_id: TenantId
    ) -> int:
        """Count total clicks for a search (CTR calculation)."""
        raise NotImplementedError

    @abstractmethod
    async def get_click_through_rate(
        self,
        tenant_id: TenantId,
        start_date: datetime,
        end_date: datetime,
        group_by: str = "day"  # day, hour, search
    ) -> List[Dict[str, Any]]:
        """
        Calculate click-through rates for analytics.

        Returns aggregated CTR data grouped by time period.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_position_analytics(
        self,
        tenant_id: TenantId,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[int, Dict[str, Any]]:
        """
        Get click statistics by result position.

        Returns: {position: {click_count, ctr, avg_time_to_click}}
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_old_events(
        self,
        before_date: datetime,
        tenant_id: Optional[TenantId] = None
    ) -> int:
        """
        Delete old click events for data retention compliance.

        Returns number of events deleted.
        """
        raise NotImplementedError


__all__ = ["ISearchClickRepository"]
