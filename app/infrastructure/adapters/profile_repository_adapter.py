"""Adapter that exposes the domain profile repository interface over SQLModel storage."""

from __future__ import annotations

from typing import List, Optional

from app.domain.entities.profile import Profile
from app.domain.repositories import IProfileRepository
from app.domain.value_objects import ProfileId, TenantId
from app.database.repositories.postgres import ProfileRepository


class ProfileRepositoryAdapter(IProfileRepository):
    """Hexagonal adapter turning SQLModel repositories into domain abstractions."""

    def __init__(self, repository: Optional[ProfileRepository] = None):
        self._repository = repository or ProfileRepository()
        
    async def get_by_id(self, profile_id: ProfileId) -> Optional[Profile]:
        return await self._repository.get_profile(profile_id)

    async def save(self, profile: Profile) -> Profile:
        return await self._repository.save_profile(profile)

    async def search_by_vector(
        self,
        tenant_id: TenantId,
        query_vector: List[float],
        *,
        limit: int = 20,
        threshold: float = 0.7,
    ) -> List[Profile]:
        results = await self._repository.search_profiles_by_vector(
            query_vector=query_vector,
            tenant_id=tenant_id.value,
            limit=limit,
            threshold=threshold,
            as_domain=True,
        )
        # search_profiles_by_vector returns either a list of dicts or domain entities; in this
        # configuration we force domain results but normalise typing for mypy friendliness.
        return list(results)  # type: ignore[return-value]


__all__ = ["ProfileRepositoryAdapter"]
