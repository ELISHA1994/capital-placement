"""Domain repository contracts for profile aggregates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from app.domain.entities.profile import Profile, ProfileStatus, ExperienceLevel
from app.domain.value_objects import ProfileId, TenantId, MatchScore


class IProfileRepository(ABC):
    """Domain-facing abstraction for profile persistence operations."""

    @abstractmethod
    async def get_by_id(self, profile_id: ProfileId, tenant_id: TenantId) -> Optional[Profile]:
        """Load a profile aggregate by identifier within tenant scope."""
        raise NotImplementedError

    @abstractmethod
    async def save(self, profile: Profile) -> Profile:
        """Persist changes to a profile aggregate."""
        raise NotImplementedError

    @abstractmethod
    async def get_by_email(self, email: str, tenant_id: TenantId) -> Optional[Profile]:
        """Get a profile by email within tenant scope."""
        raise NotImplementedError

    @abstractmethod
    async def list_by_tenant(
        self,
        tenant_id: TenantId,
        status: Optional[ProfileStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Profile]:
        """List profiles for a tenant with optional filtering."""
        raise NotImplementedError

    @abstractmethod
    async def search_by_text(
        self,
        tenant_id: TenantId,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> List[Profile]:
        """Search profiles by text query."""
        raise NotImplementedError

    @abstractmethod
    async def search_by_vector(
        self,
        tenant_id: TenantId,
        query_vector: List[float],
        *,
        limit: int = 20,
        threshold: float = 0.7,
    ) -> List[tuple[Profile, MatchScore]]:
        """Run vector similarity search for candidate discovery."""
        raise NotImplementedError

    @abstractmethod
    async def search_by_skills(
        self,
        tenant_id: TenantId,
        skills: List[str],
        experience_level: Optional[ExperienceLevel] = None,
        limit: int = 50
    ) -> List[tuple[Profile, MatchScore]]:
        """Search profiles by required skills."""
        raise NotImplementedError

    @abstractmethod
    async def delete(self, profile_id: ProfileId, tenant_id: TenantId) -> bool:
        """Delete a profile (hard delete)."""
        raise NotImplementedError

    @abstractmethod
    async def count_by_tenant(
        self,
        tenant_id: TenantId,
        status: Optional[ProfileStatus] = None
    ) -> int:
        """Count profiles for a tenant."""
        raise NotImplementedError

    @abstractmethod
    async def get_by_ids(
        self,
        profile_ids: List[ProfileId],
        tenant_id: TenantId
    ) -> List[Profile]:
        """Get multiple profiles by IDs."""
        raise NotImplementedError

    @abstractmethod
    async def update_analytics(
        self,
        profile_id: ProfileId,
        tenant_id: TenantId,
        view_increment: int = 0,
        search_appearance_increment: int = 0
    ) -> bool:
        """Update profile analytics counters."""
        raise NotImplementedError

    @abstractmethod
    async def list_pending_processing(
        self,
        tenant_id: Optional[TenantId] = None,
        limit: int = 100
    ) -> List[Profile]:
        """List profiles pending processing."""
        raise NotImplementedError

    @abstractmethod
    async def list_for_archival(
        self,
        tenant_id: TenantId,
        days_inactive: int = 90
    ) -> List[Profile]:
        """List profiles eligible for archival."""
        raise NotImplementedError

    @abstractmethod
    async def get_statistics(
        self,
        tenant_id: TenantId
    ) -> Dict[str, Any]:
        """Get profile statistics for a tenant."""
        raise NotImplementedError


__all__ = ["IProfileRepository"]