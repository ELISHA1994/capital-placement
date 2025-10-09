"""Application service orchestrating profile listing use cases."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional
from uuid import UUID

import structlog

from app.application.dependencies.profile_dependencies import ProfileDependencies
from app.domain.entities.profile import Profile
from app.domain.entities.profile import ProcessingStatus as DomainProcessingStatus
from app.domain.value_objects import ProfileId, TenantId

logger = structlog.get_logger(__name__)


@dataclass
class ProfileListItem:
    """Lightweight profile summary returned to the API layer."""

    profile_id: str
    email: str
    full_name: Optional[str]
    title: Optional[str]
    current_company: Optional[str]
    total_experience_years: Optional[int]
    top_skills: List[str]
    last_updated: datetime
    processing_status: DomainProcessingStatus
    quality_score: Optional[float]


@dataclass
class ProfileListResult:
    """Paginated result returned by the profile service."""

    items: List[ProfileListItem]
    total: int


class ProfileApplicationService:
    """Provide profile list operations for the API layer."""

    def __init__(self, dependencies: ProfileDependencies) -> None:
        self._deps = dependencies

    async def list_profiles(
        self,
        *,
        tenant_id: UUID,
        pagination_offset: int,
        pagination_limit: int,
        status_filter: Optional[DomainProcessingStatus],
        include_incomplete: bool,
        quality_min: Optional[float],
        experience_min: Optional[int],
        experience_max: Optional[int],
        skill_filter: Optional[str],
        sort_by: str,
        sort_order: str,
    ) -> ProfileListResult:
        """Return paginated profile summaries for a tenant."""

        tenant = TenantId(tenant_id)
        repository = self._deps.profile_repository

        # Accumulate filtered summaries; fetch data in batches to honour pagination.
        filtered_summaries: List[ProfileListItem] = []
        fetch_offset = 0
        batch_size = max(pagination_limit, 50)
        normalized_skill_filter = (skill_filter or "").strip().lower()

        while True:
            profiles = await repository.list_by_tenant(
                tenant_id=tenant,
                status=status_filter,
                limit=batch_size,
                offset=fetch_offset,
            )

            if not profiles:
                break

            summaries = self._build_summaries(
                profiles,
                quality_min=quality_min,
                experience_min=experience_min,
                experience_max=experience_max,
                skill_filter=normalized_skill_filter,
                status_filter=status_filter,
                include_incomplete=include_incomplete,
            )

            filtered_summaries.extend(summaries)

            fetch_offset += batch_size

        # Ensure full ordering according to requested sort fields
        reverse = sort_order.lower() == "desc"
        sort_key = self._resolve_sort_key(sort_by)
        filtered_summaries.sort(key=sort_key, reverse=reverse)

        total_count = len(filtered_summaries)
        start_index = min(pagination_offset, total_count)
        end_index = min(start_index + pagination_limit, total_count)
        page_items = filtered_summaries[start_index:end_index]

        return ProfileListResult(items=page_items, total=total_count)

    async def get_profile(
        self,
        *,
        tenant_id: TenantId,
        profile_id: ProfileId,
    ) -> Optional[Profile]:
        """Load a profile aggregate for a tenant."""

        repository = self._deps.profile_repository
        return await repository.get_by_id(profile_id, tenant_id)

    def _build_summaries(
        self,
        profiles: Iterable[Profile],
        *,
        quality_min: Optional[float],
        experience_min: Optional[int],
        experience_max: Optional[int],
        skill_filter: str,
        status_filter: Optional[DomainProcessingStatus],
        include_incomplete: bool,
    ) -> List[ProfileListItem]:
        summaries: List[ProfileListItem] = []

        for profile in profiles:
            if status_filter and profile.processing.status != status_filter:
                continue

            if not include_incomplete and profile.processing.status != DomainProcessingStatus.COMPLETED:
                continue

            quality_score = profile.processing.quality_score
            if quality_min is not None and (quality_score or 0.0) < quality_min:
                continue

            experience_years = profile.profile_data.total_experience_years()
            if experience_min is not None and experience_years < experience_min:
                continue
            if experience_max is not None and experience_years > experience_max:
                continue

            normalized_skills = [str(skill) for skill in (profile.normalized_skills or [])]
            if skill_filter and not any(skill_filter in skill for skill in normalized_skills):
                continue

            top_skills = [skill.name.value for skill in profile.profile_data.skills[:5]]

            current_company = None
            for experience in profile.profile_data.experience:
                if experience.is_current_role():
                    current_company = experience.company
                    break
            if current_company is None and profile.profile_data.experience:
                current_company = profile.profile_data.experience[0].company

            summaries.append(
                ProfileListItem(
                    profile_id=str(profile.id.value),
                    email=str(profile.profile_data.email),
                    full_name=profile.profile_data.name,
                    title=profile.profile_data.headline,
                    current_company=current_company,
                    total_experience_years=int(round(experience_years)) if experience_years else None,
                    top_skills=top_skills,
                    last_updated=profile.updated_at,
                    processing_status=profile.processing.status,
                    quality_score=quality_score,
                )
            )

        return summaries

    def _resolve_sort_key(self, sort_by: str):
        mapping = {
            "quality_score": lambda item: item.quality_score or 0.0,
            "experience": lambda item: item.total_experience_years or 0,
        }
        return mapping.get(sort_by, lambda item: item.last_updated)


__all__ = [
    "ProfileApplicationService",
    "ProfileListItem",
    "ProfileListResult",
]
