"""Application service orchestrating profile listing use cases."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

import structlog

from app.application.dependencies.profile_dependencies import ProfileDependencies
from app.domain.entities.profile import (
    Education,
    Experience,
    Location,
    Profile,
    ProfileData,
    Skill,
)
from app.domain.entities.profile import ProcessingStatus as DomainProcessingStatus
from app.domain.value_objects import (
    EmailAddress,
    PhoneNumber,
    ProfileId,
    SkillName,
    TenantId,
)

logger = structlog.get_logger(__name__)


@dataclass
class ProfileListItem:
    """Lightweight profile summary returned to the API layer."""

    profile_id: str
    email: str
    full_name: str | None
    title: str | None
    current_company: str | None
    total_experience_years: int | None
    top_skills: list[str]
    last_updated: datetime
    processing_status: DomainProcessingStatus
    quality_score: float | None


@dataclass
class ProfileListResult:
    """Paginated result returned by the profile service."""

    items: list[ProfileListItem]
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
        status_filter: DomainProcessingStatus | None,
        include_incomplete: bool,
        quality_min: float | None,
        experience_min: int | None,
        experience_max: int | None,
        skill_filter: str | None,
        sort_by: str,
        sort_order: str,
    ) -> ProfileListResult:
        """Return paginated profile summaries for a tenant."""

        tenant = TenantId(tenant_id)
        repository = self._deps.profile_repository

        # Accumulate filtered summaries; fetch data in batches to honour pagination.
        filtered_summaries: list[ProfileListItem] = []
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
    ) -> Profile | None:
        """Load a profile aggregate for a tenant."""

        repository = self._deps.profile_repository
        return await repository.get_by_id(profile_id, tenant_id)

    async def update_profile(
        self,
        *,
        tenant_id: TenantId,
        profile_id: ProfileId,
        update_data: dict[str, Any],
        user_id: str,
        regenerate_embeddings: bool = False,
        schedule_task: Any | None = None,
    ) -> Profile | None:
        """Update an existing profile with partial data.

        Orchestrates the complete profile update workflow including:
        - Data validation and merging
        - Domain entity updates
        - Background task scheduling (embeddings, search index)
        - Usage tracking and audit logging

        Args:
            tenant_id: Tenant identifier for isolation
            profile_id: Profile identifier
            update_data: Dictionary with optional fields to update:
                - title: maps to ProfileData.headline
                - summary: maps to ProfileData.summary
                - skills: List[Dict] that needs conversion to List[Skill]
                - experience_entries: List[Dict] that needs conversion to List[Experience]
                - education_entries: List[Dict] that needs conversion to List[Education]
                - contact_info: Dict with email, phone fields
                - location: Dict with city, state, country, coordinates fields
                - tags: List[str] (stored in metadata)
            user_id: User performing the update (for audit trail)
            regenerate_embeddings: Whether to regenerate embeddings after update
            schedule_task: Task scheduler for background operations

        Returns:
            Updated Profile entity, or None if profile not found

        Raises:
            ValueError: If update_data contains invalid field values
        """
        logger.info(
            "update_profile_requested",
            tenant_id=str(tenant_id),
            profile_id=str(profile_id),
            user_id=user_id,
            update_fields=list(update_data.keys()),
            regenerate_embeddings=regenerate_embeddings,
        )

        # Fetch existing profile
        repository = self._deps.profile_repository
        existing_profile = await repository.get_by_id(profile_id, tenant_id)

        if not existing_profile:
            logger.warning(
                "update_profile_not_found",
                tenant_id=str(tenant_id),
                profile_id=str(profile_id),
            )
            return None

        # Get existing profile data
        current_data = existing_profile.profile_data

        try:
            # Build updated ProfileData by merging update_data with existing fields
            updated_profile_data = self._merge_profile_data(
                current_data=current_data,
                update_data=update_data,
            )

            # Call domain entity's update method to recompute derived fields
            existing_profile.update_profile_data(updated_profile_data)

            # Handle metadata updates (tags)
            if "tags" in update_data:
                tags = update_data["tags"]
                if isinstance(tags, list):
                    existing_profile.metadata["tags"] = tags
                    logger.debug("updated_profile_tags", tags_count=len(tags))

            # Save updated profile via repository
            updated_profile = await repository.save(existing_profile)

            # Calculate completeness for logging and analytics
            completeness_score = updated_profile.calculate_completeness_score()

            # Schedule background tasks for post-update processing
            if regenerate_embeddings and self._deps.embedding_service:
                await self._enqueue_background(
                    schedule_task,
                    self._regenerate_profile_embeddings,
                    str(profile_id.value),
                    str(tenant_id.value),
                    user_id,
                )

            # Always update search index after profile update
            if self._deps.search_index_service:
                await self._enqueue_background(
                    schedule_task,
                    self._update_search_index,
                    str(profile_id.value),
                    str(tenant_id.value),
                    updated_profile,
                )

            # Track usage in background
            await self._enqueue_background(
                schedule_task,
                self._track_profile_update_usage,
                str(tenant_id.value),
                user_id,
                len(update_data),
            )

            # Log audit event in background
            if self._deps.audit_service:
                await self._enqueue_background(
                    schedule_task,
                    self._log_profile_update_audit,
                    str(tenant_id.value),
                    user_id,
                    str(profile_id.value),
                    list(update_data.keys()),
                    completeness_score,
                )

            logger.info(
                "update_profile_completed",
                tenant_id=str(tenant_id),
                profile_id=str(profile_id),
                user_id=user_id,
                completeness_score=completeness_score,
                fields_updated=len(update_data),
            )

            return updated_profile

        except (ValueError, TypeError) as e:
            logger.error(
                "update_profile_validation_failed",
                tenant_id=str(tenant_id),
                profile_id=str(profile_id),
                user_id=user_id,
                error=str(e),
            )
            raise

    def _merge_profile_data(
        self,
        *,
        current_data: ProfileData,
        update_data: dict[str, Any],
    ) -> ProfileData:
        """Merge update_data with existing ProfileData.

        Only updates provided fields, keeps existing values for unprovided fields.
        Handles type conversions from API DTOs to domain value objects.

        Args:
            current_data: Existing ProfileData
            update_data: Dictionary with optional update fields

        Returns:
            New ProfileData instance with merged values

        Raises:
            ValueError: If field values fail validation
        """
        # Start with existing values
        name = current_data.name
        email = current_data.email
        phone = current_data.phone
        location = current_data.location
        summary = current_data.summary
        headline = current_data.headline
        experience = current_data.experience
        education = current_data.education
        skills = current_data.skills
        languages = current_data.languages

        # Update title (maps to headline)
        if "title" in update_data and update_data["title"] is not None:
            headline = str(update_data["title"]).strip() or None

        # Update summary
        if "summary" in update_data and update_data["summary"] is not None:
            summary = str(update_data["summary"]).strip() or None

        # Update contact info
        if "contact_info" in update_data and isinstance(update_data["contact_info"], dict):
            contact_info = update_data["contact_info"]

            # Update email if provided
            if "email" in contact_info and contact_info["email"]:
                try:
                    email = EmailAddress(contact_info["email"])
                except ValueError as e:
                    logger.warning("invalid_email_address", error=str(e))
                    raise

            # Update phone if provided
            if "phone" in contact_info:
                phone_value = contact_info["phone"]
                if phone_value:
                    try:
                        phone = PhoneNumber(phone_value)
                    except ValueError as e:
                        logger.warning("invalid_phone_number", error=str(e))
                        raise
                else:
                    phone = None

        # Update location
        if "location" in update_data and isinstance(update_data["location"], dict):
            location = self._convert_location_from_dict(update_data["location"])

        # Update skills
        if "skills" in update_data and isinstance(update_data["skills"], list):
            skills = self._convert_skills_from_dict(update_data["skills"])

        # Update experience entries
        if "experience_entries" in update_data and isinstance(update_data["experience_entries"], list):
            experience = self._convert_experience_from_dict(update_data["experience_entries"])

        # Update education entries
        if "education_entries" in update_data and isinstance(update_data["education_entries"], list):
            education = self._convert_education_from_dict(update_data["education_entries"])

        # Build new ProfileData with merged values
        return ProfileData(
            name=name,
            email=email,
            phone=phone,
            location=location,
            summary=summary,
            headline=headline,
            experience=experience,
            education=education,
            skills=skills,
            languages=languages,
        )

    def _convert_skills_from_dict(self, skills_data: list[dict[str, Any]]) -> list[Skill]:
        """Convert skill dictionaries to Skill domain objects.

        Args:
            skills_data: List of skill dictionaries with name, category, etc.

        Returns:
            List of Skill domain objects
        """
        skills: list[Skill] = []

        for skill_dict in skills_data:
            if not isinstance(skill_dict, dict):
                continue

            # Name is required
            name = skill_dict.get("name")
            if not name:
                continue

            try:
                skill = Skill(
                    name=SkillName(name),
                    category=skill_dict.get("category", "technical"),
                    proficiency=skill_dict.get("proficiency"),
                    years_of_experience=skill_dict.get("years_of_experience"),
                    endorsed=skill_dict.get("endorsed", False),
                    last_used=skill_dict.get("last_used"),
                )
                skills.append(skill)
            except ValueError as e:
                logger.warning("invalid_skill_data", skill_name=name, error=str(e))
                continue

        return skills

    def _convert_experience_from_dict(self, experience_data: list[dict[str, Any]]) -> list[Experience]:
        """Convert experience dictionaries to Experience domain objects.

        Args:
            experience_data: List of experience dictionaries

        Returns:
            List of Experience domain objects
        """
        experience_entries: list[Experience] = []

        for exp_dict in experience_data:
            if not isinstance(exp_dict, dict):
                continue

            # Required fields: title, company, start_date
            title = exp_dict.get("title")
            company = exp_dict.get("company")
            start_date = exp_dict.get("start_date")

            if not (title and company and start_date):
                logger.warning("incomplete_experience_entry", exp_dict=exp_dict)
                continue

            # Convert skills if present
            exp_skills: list[SkillName] = []
            if "skills" in exp_dict and isinstance(exp_dict["skills"], list):
                for skill_name in exp_dict["skills"]:
                    try:
                        exp_skills.append(SkillName(skill_name))
                    except ValueError:
                        continue

            try:
                experience = Experience(
                    title=title,
                    company=company,
                    start_date=start_date,
                    description=exp_dict.get("description", ""),
                    end_date=exp_dict.get("end_date"),
                    current=exp_dict.get("current", False),
                    location=exp_dict.get("location"),
                    achievements=exp_dict.get("achievements", []),
                    skills=exp_skills,
                )
                experience_entries.append(experience)
            except (ValueError, TypeError) as e:
                logger.warning("invalid_experience_entry", error=str(e))
                continue

        return experience_entries

    def _convert_education_from_dict(self, education_data: list[dict[str, Any]]) -> list[Education]:
        """Convert education dictionaries to Education domain objects.

        Args:
            education_data: List of education dictionaries

        Returns:
            List of Education domain objects
        """
        education_entries: list[Education] = []

        for edu_dict in education_data:
            if not isinstance(edu_dict, dict):
                continue

            # Required fields: institution, degree, field
            institution = edu_dict.get("institution")
            degree = edu_dict.get("degree")
            field = edu_dict.get("field")

            if not (institution and degree and field):
                logger.warning("incomplete_education_entry", edu_dict=edu_dict)
                continue

            try:
                education = Education(
                    institution=institution,
                    degree=degree,
                    field=field,
                    start_date=edu_dict.get("start_date"),
                    end_date=edu_dict.get("end_date"),
                    gpa=edu_dict.get("gpa"),
                    achievements=edu_dict.get("achievements", []),
                )
                education_entries.append(education)
            except (ValueError, TypeError) as e:
                logger.warning("invalid_education_entry", error=str(e))
                continue

        return education_entries

    def _convert_location_from_dict(self, location_data: dict[str, Any]) -> Location | None:
        """Convert location dict to Location domain object.

        Args:
            location_data: Dict with city, state, country, coordinates

        Returns:
            Location domain object or None if all fields are empty
        """
        city = location_data.get("city")
        state = location_data.get("state")
        country = location_data.get("country")
        coordinates_data = location_data.get("coordinates")

        # Parse coordinates if provided
        coordinates = None
        if coordinates_data:
            if isinstance(coordinates_data, dict):
                lat = coordinates_data.get("lat")
                lng = coordinates_data.get("lng") or coordinates_data.get("lon")
                if lat is not None and lng is not None:
                    try:
                        coordinates = (float(lat), float(lng))
                    except (TypeError, ValueError):
                        logger.warning("invalid_coordinates", coordinates=coordinates_data)
            elif isinstance(coordinates_data, (list, tuple)) and len(coordinates_data) == 2:
                try:
                    coordinates = (float(coordinates_data[0]), float(coordinates_data[1]))
                except (TypeError, ValueError):
                    logger.warning("invalid_coordinates", coordinates=coordinates_data)

        # Only create Location if at least one field has a value
        if city or state or country or coordinates:
            return Location(
                city=city,
                state=state,
                country=country,
                coordinates=coordinates
            )
        return None

    def _build_summaries(
        self,
        profiles: Iterable[Profile],
        *,
        quality_min: float | None,
        experience_min: int | None,
        experience_max: int | None,
        skill_filter: str,
        status_filter: DomainProcessingStatus | None,
        include_incomplete: bool,
    ) -> list[ProfileListItem]:
        summaries: list[ProfileListItem] = []

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

    # Background task handlers following upload_service pattern

    async def _enqueue_background(self, scheduler: Any | None, func, *args, **kwargs) -> None:
        """Enqueue a background task for async execution.

        Similar to upload_service pattern - handles both FastAPI BackgroundTasks
        and task manager scheduling.

        Args:
            scheduler: Task scheduler (FastAPI BackgroundTasks or TaskManager)
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
        """
        import asyncio

        if scheduler is None:
            # No scheduler provided, run as fire-and-forget task
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)
        else:
            # Use the provided scheduler (FastAPI BackgroundTasks)
            if hasattr(scheduler, 'add_task'):
                scheduler.add_task(func, *args, **kwargs)
            else:
                # Fallback to creating task
                result = func(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)

    async def _track_profile_update_usage(
        self,
        tenant_id: str,
        user_id: str,
        fields_updated_count: int,
    ) -> None:
        """Track profile update usage metrics.

        Args:
            tenant_id: Tenant identifier
            user_id: User who performed the update
            fields_updated_count: Number of fields updated
        """
        try:
            if self._deps.usage_service:
                await self._deps.usage_service.track_usage(
                    tenant_id=tenant_id,
                    resource_type="profile",
                    amount=1,
                    metadata={
                        "operation": "update",
                        "fields_updated": fields_updated_count,
                        "user_id": user_id,
                    },
                )
                logger.debug(
                    "Profile update usage tracked",
                    tenant_id=tenant_id,
                    user_id=user_id,
                    fields_updated=fields_updated_count,
                )
        except Exception as exc:
            # Usage tracking failures should not break the main flow
            logger.warning(
                "Failed to track profile update usage",
                tenant_id=tenant_id,
                error=str(exc),
            )

    async def _regenerate_profile_embeddings(
        self,
        profile_id: str,
        tenant_id: str,
        user_id: str,
    ) -> None:
        """Regenerate embeddings for a profile in background.

        Args:
            profile_id: Profile identifier
            tenant_id: Tenant identifier
            user_id: User who requested regeneration
        """
        try:
            if not self._deps.embedding_service:
                logger.warning("Embedding service not available", profile_id=profile_id)
                return

            logger.info(
                "Regenerating profile embeddings",
                profile_id=profile_id,
                tenant_id=tenant_id,
            )

            # Load profile
            profile = await self._deps.profile_repository.get_by_id(
                ProfileId(profile_id),
                TenantId(tenant_id),
            )

            if not profile:
                logger.warning("Profile not found for embedding regeneration", profile_id=profile_id)
                return

            # Generate new embeddings from searchable text
            searchable_text = profile.searchable_text
            if searchable_text:
                embedding_vector = await self._deps.embedding_service.generate_embedding(
                    text=searchable_text,
                    model="text-embedding-3-large",
                )

                # Update profile with new embeddings
                from app.domain.entities.profile import ProfileEmbeddings
                from app.domain.value_objects import EmbeddingVector

                profile.embeddings = ProfileEmbeddings(
                    overall=EmbeddingVector(dimensions=len(embedding_vector), values=embedding_vector)
                )

                # Save updated profile
                await self._deps.profile_repository.save(profile)

                logger.info(
                    "Profile embeddings regenerated successfully",
                    profile_id=profile_id,
                    embedding_dimensions=len(embedding_vector),
                )
        except Exception as exc:
            logger.error(
                "Failed to regenerate profile embeddings",
                profile_id=profile_id,
                error=str(exc),
            )

    async def _update_search_index(
        self,
        profile_id: str,
        tenant_id: str,
        profile: Profile,
    ) -> None:
        """Update search index for a profile in background.

        Args:
            profile_id: Profile identifier
            tenant_id: Tenant identifier
            profile: Updated profile entity
        """
        try:
            if not self._deps.search_index_service:
                logger.debug("Search index service not available", profile_id=profile_id)
                return

            logger.info(
                "Updating profile search index",
                profile_id=profile_id,
                tenant_id=tenant_id,
            )

            # Convert profile to indexable format
            profile_data = {
                "profile_id": profile_id,
                "tenant_id": tenant_id,
                "name": profile.profile_data.name,
                "email": str(profile.profile_data.email),
                "headline": profile.profile_data.headline,
                "summary": profile.profile_data.summary,
                "skills": profile.normalized_skills,
                "searchable_text": profile.searchable_text,
                "experience_level": profile.experience_level.value if profile.experience_level else None,
                "status": profile.status.value,
                "updated_at": profile.updated_at.isoformat(),
            }

            await self._deps.search_index_service.update_profile_index(
                profile_id=profile_id,
                profile_data=profile_data,
            )

            logger.info(
                "Profile search index updated successfully",
                profile_id=profile_id,
            )
        except Exception as exc:
            logger.warning(
                "Failed to update profile search index",
                profile_id=profile_id,
                error=str(exc),
            )

    async def _log_profile_update_audit(
        self,
        tenant_id: str,
        user_id: str,
        profile_id: str,
        fields_updated: list[str],
        completeness_score: float,
    ) -> None:
        """Log profile update audit event in background.

        Args:
            tenant_id: Tenant identifier
            user_id: User who performed the update
            profile_id: Profile identifier
            fields_updated: List of fields that were updated
            completeness_score: Profile completeness score after update
        """
        try:
            if not self._deps.audit_service:
                return

            await self._deps.audit_service.log_event(
                event_type="profile_updated",
                tenant_id=tenant_id,
                action="update",  # Required positional parameter
                resource_type="profile",
                user_id=user_id,
                resource_id=profile_id,
                details={
                    "fields_updated": fields_updated,
                    "completeness_score": completeness_score,
                },
            )

            logger.debug(
                "Profile update audit logged",
                tenant_id=tenant_id,
                profile_id=profile_id,
            )
        except Exception as exc:
            logger.warning(
                "Failed to log profile update audit",
                profile_id=profile_id,
                error=str(exc),
            )


__all__ = [
    "ProfileApplicationService",
    "ProfileListItem",
    "ProfileListResult",
]
