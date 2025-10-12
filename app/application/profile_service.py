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
    ProfileEmbeddings,
    Skill,
)
from app.domain.entities.profile import ProcessingStatus as DomainProcessingStatus
from app.domain.value_objects import (
    EmailAddress,
    EmbeddingVector,
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


@dataclass
class ProfileDeletionResult:
    """Result of profile deletion operation returned to the API layer."""

    success: bool
    deletion_type: str  # "soft_delete" or "permanent_delete"
    profile_id: str
    message: str
    can_restore: bool


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
        schedule_task: Any | None = None,
    ) -> Profile | None:
        """Load a profile aggregate for a tenant.

        Args:
            tenant_id: Tenant identifier for multi-tenant isolation
            profile_id: Profile identifier
            schedule_task: Optional task scheduler for background operations

        Returns:
            Profile entity if found, None otherwise

        Note:
            Profile views are tracked asynchronously to avoid latency impact.
        """
        repository = self._deps.profile_repository
        profile = await repository.get_by_id(profile_id, tenant_id)

        # Track view in background (non-blocking)
        if profile:
            await self._enqueue_background(
                schedule_task,
                self._record_profile_view,
                str(profile_id.value),
                str(tenant_id.value),
            )

        return profile

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

    async def delete_profile(
        self,
        *,
        tenant_id: TenantId,
        profile_id: ProfileId,
        user_id: str,
        permanent: bool = False,
        reason: str | None = None,
        schedule_task: Any | None = None,
    ) -> ProfileDeletionResult:
        """Delete a profile (soft or permanent based on business logic).

        This method encapsulates the business logic decision of which deletion type to use.
        The API layer should call this method and map the result to HTTP responses.

        Args:
            tenant_id: Tenant identifier
            profile_id: Profile to delete
            user_id: User performing deletion
            permanent: If True, permanently deletes. If False (default), soft deletes.
            reason: Optional deletion reason (used for soft delete audit trail)
            schedule_task: Task scheduler for background operations

        Returns:
            ProfileDeletionResult with deletion status and details

        Raises:
            ValueError: If deletion validation fails
        """
        logger.info(
            "delete_profile_requested",
            tenant_id=str(tenant_id),
            profile_id=str(profile_id),
            user_id=user_id,
            permanent=permanent,
            reason=reason,
        )

        # Business logic: Determine deletion type
        if permanent:
            # Permanent deletion workflow
            success = await self.permanently_delete_profile(
                tenant_id=tenant_id,
                profile_id=profile_id,
                user_id=user_id,
                schedule_task=schedule_task,
            )

            if not success:
                # Profile not found
                return ProfileDeletionResult(
                    success=False,
                    deletion_type="permanent_delete",
                    profile_id=str(profile_id.value),
                    message="Profile not found",
                    can_restore=False,
                )

            return ProfileDeletionResult(
                success=True,
                deletion_type="permanent_delete",
                profile_id=str(profile_id.value),
                message="Profile permanently deleted",
                can_restore=False,
            )
        else:
            # Soft deletion workflow
            deleted_profile = await self.soft_delete_profile(
                tenant_id=tenant_id,
                profile_id=profile_id,
                user_id=user_id,
                reason=reason,
                schedule_task=schedule_task,
            )

            if not deleted_profile:
                # Profile not found
                return ProfileDeletionResult(
                    success=False,
                    deletion_type="soft_delete",
                    profile_id=str(profile_id.value),
                    message="Profile not found",
                    can_restore=False,
                )

            return ProfileDeletionResult(
                success=True,
                deletion_type="soft_delete",
                profile_id=str(profile_id.value),
                message="Profile deleted (can be restored)",
                can_restore=True,
            )

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
        compensation = getattr(current_data, "compensation", None)
        total_experience_override = getattr(current_data, "total_experience_years_override", None)

        # Update title (maps to headline)
        if "title" in update_data and update_data["title"] is not None:
            headline = str(update_data["title"]).strip() or None

        # Update summary
        if "summary" in update_data and update_data["summary"] is not None:
            summary = str(update_data["summary"]).strip() or None

        # Update contact info
        if "contact_info" in update_data and isinstance(
            update_data["contact_info"], dict
        ):
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
        if "experience_entries" in update_data and isinstance(
            update_data["experience_entries"], list
        ):
            experience = self._convert_experience_from_dict(
                update_data["experience_entries"]
            )

        # Update education entries
        if "education_entries" in update_data and isinstance(
            update_data["education_entries"], list
        ):
            education = self._convert_education_from_dict(
                update_data["education_entries"]
            )

        if "compensation" in update_data:
            compensation_value = update_data["compensation"]
            if compensation_value is None or isinstance(compensation_value, dict):
                compensation = compensation_value
            else:
                raise ValueError("compensation must be a dictionary or null")

        job_preferences = update_data.get("job_preferences")
        if isinstance(job_preferences, dict) and not compensation:
            salary_expectation = job_preferences.get("salary_expectation")
            if isinstance(salary_expectation, dict):
                compensation = salary_expectation

        if "total_experience_years" in update_data:
            exp_value = update_data["total_experience_years"]
            if exp_value is None:
                total_experience_override = None
            else:
                try:
                    total_experience_override = float(exp_value)
                except (TypeError, ValueError):
                    raise ValueError("total_experience_years must be a number")

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
            compensation=compensation,
            total_experience_years_override=total_experience_override,
        )

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

        Generates all four embedding types (overall, skills, experience, summary)
        using batch processing for efficiency. Uses a single API call to generate
        all embeddings, reducing latency and maintaining data consistency.

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
                logger.warning(
                    "Profile not found for embedding regeneration",
                    profile_id=profile_id,
                )
                return

            # Extract text for all embedding types
            embedding_texts = self._extract_embedding_texts(profile)

            # Check if we have at least the overall text
            if not embedding_texts["overall"]:
                logger.warning(
                    "No content available for embedding generation",
                    profile_id=profile_id,
                )
                return

            # Generate all embeddings in batch (single API call)
            embedding_vectors = await self._generate_profile_embeddings_batch(
                embedding_texts,
                model="text-embedding-3-large",
            )

            # Create complete ProfileEmbeddings with all fields
            profile.embeddings = ProfileEmbeddings(
                overall=EmbeddingVector(
                    dimensions=len(embedding_vectors["overall"]),
                    values=embedding_vectors["overall"],
                )
                if embedding_vectors["overall"]
                else None,
                skills=EmbeddingVector(
                    dimensions=len(embedding_vectors["skills"]),
                    values=embedding_vectors["skills"],
                )
                if embedding_vectors["skills"]
                else None,
                experience=EmbeddingVector(
                    dimensions=len(embedding_vectors["experience"]),
                    values=embedding_vectors["experience"],
                )
                if embedding_vectors["experience"]
                else None,
                summary=EmbeddingVector(
                    dimensions=len(embedding_vectors["summary"]),
                    values=embedding_vectors["summary"],
                )
                if embedding_vectors["summary"]
                else None,
            )

            # Save updated profile
            await self._deps.profile_repository.save(profile)

            logger.info(
                "Profile embeddings regenerated successfully",
                profile_id=profile_id,
                embeddings_generated={
                    "overall": embedding_vectors["overall"] is not None,
                    "skills": embedding_vectors["skills"] is not None,
                    "experience": embedding_vectors["experience"] is not None,
                    "summary": embedding_vectors["summary"] is not None,
                },
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
                logger.debug(
                    "Search index service not available", profile_id=profile_id
                )
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
                "experience_level": profile.experience_level.value
                if profile.experience_level
                else None,
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

    async def soft_delete_profile(
        self,
        *,
        tenant_id: TenantId,
        profile_id: ProfileId,
        user_id: str,
        reason: str | None = None,
        schedule_task: Any | None = None,
    ) -> Profile | None:
        """Soft delete a profile (marks as deleted, preserves data).

        Orchestrates soft deletion workflow:
        - Validate deletion is allowed
        - Mark profile as deleted in domain
        - Save to repository
        - Schedule background cleanup tasks (search index removal)
        - Track usage and audit logging

        Args:
            tenant_id: Tenant identifier
            profile_id: Profile to delete
            user_id: User performing deletion
            reason: Optional deletion reason
            schedule_task: Task scheduler for background operations

        Returns:
            Deleted Profile entity, or None if not found

        Raises:
            ValueError: If deletion validation fails
        """
        logger.info(
            "soft_delete_profile_requested",
            tenant_id=str(tenant_id),
            profile_id=str(profile_id),
            user_id=user_id,
            reason=reason,
        )

        # Fetch profile
        repository = self._deps.profile_repository
        profile = await repository.get_by_id(profile_id, tenant_id)

        if not profile:
            logger.warning(
                "soft_delete_profile_not_found",
                tenant_id=str(tenant_id),
                profile_id=str(profile_id),
            )
            return None

        # Validate deletion is allowed
        validation_issues = profile.validate_can_delete()
        if validation_issues:
            error_msg = f"Cannot delete profile: {', '.join(validation_issues)}"
            logger.error(
                "soft_delete_profile_validation_failed",
                tenant_id=str(tenant_id),
                profile_id=str(profile_id),
                issues=validation_issues,
            )
            raise ValueError(error_msg)

        # Perform soft delete at domain level
        profile.soft_delete(reason=reason)

        # Save deleted profile
        deleted_profile = await repository.save(profile)

        # Schedule background cleanup tasks
        # Remove from search index
        if self._deps.search_index_service:
            await self._enqueue_background(
                schedule_task,
                self._remove_from_search_index,
                str(profile_id.value),
                str(tenant_id.value),
            )

        # Track usage
        await self._enqueue_background(
            schedule_task,
            self._track_profile_deletion_usage,
            str(tenant_id.value),
            user_id,
            "soft_delete",
        )

        # Log audit event
        if self._deps.audit_service:
            await self._enqueue_background(
                schedule_task,
                self._log_profile_deletion_audit,
                str(tenant_id.value),
                user_id,
                str(profile_id.value),
                "soft_delete",
                reason,
            )

        logger.info(
            "soft_delete_profile_completed",
            tenant_id=str(tenant_id),
            profile_id=str(profile_id),
            user_id=user_id,
        )

        return deleted_profile

    async def restore_profile(
        self,
        *,
        tenant_id: TenantId,
        profile_id: ProfileId,
        user_id: str,
        schedule_task: Any | None = None,
    ) -> Profile | None:
        """Restore a soft-deleted profile.

        Orchestrates profile restoration workflow:
        - Validate restoration is allowed
        - Restore profile in domain
        - Save to repository
        - Schedule background tasks (regenerate embeddings, update search index)
        - Track usage and audit logging

        Args:
            tenant_id: Tenant identifier
            profile_id: Profile to restore
            user_id: User performing restoration
            schedule_task: Task scheduler for background operations

        Returns:
            Restored Profile entity, or None if not found

        Raises:
            ValueError: If restoration validation fails
        """
        logger.info(
            "restore_profile_requested",
            tenant_id=str(tenant_id),
            profile_id=str(profile_id),
            user_id=user_id,
        )

        # Fetch profile
        repository = self._deps.profile_repository
        profile = await repository.get_by_id(profile_id, tenant_id)

        if not profile:
            logger.warning(
                "restore_profile_not_found",
                tenant_id=str(tenant_id),
                profile_id=str(profile_id),
            )
            return None

        # Validate restoration is allowed
        validation_issues = profile.validate_can_restore()
        if validation_issues:
            error_msg = f"Cannot restore profile: {', '.join(validation_issues)}"
            logger.error(
                "restore_profile_validation_failed",
                tenant_id=str(tenant_id),
                profile_id=str(profile_id),
                issues=validation_issues,
            )
            raise ValueError(error_msg)

        # Perform restoration at domain level
        profile.restore()

        # Save restored profile
        restored_profile = await repository.save(profile)

        # Schedule background tasks
        # Regenerate embeddings for restored profile
        if self._deps.embedding_service:
            await self._enqueue_background(
                schedule_task,
                self._regenerate_profile_embeddings,
                str(profile_id.value),
                str(tenant_id.value),
                user_id,
            )

        # Update search index to make profile searchable again
        if self._deps.search_index_service:
            await self._enqueue_background(
                schedule_task,
                self._update_search_index,
                str(profile_id.value),
                str(tenant_id.value),
                restored_profile,
            )

        # Track usage
        await self._enqueue_background(
            schedule_task,
            self._track_profile_restoration_usage,
            str(tenant_id.value),
            user_id,
        )

        # Log audit event
        if self._deps.audit_service:
            await self._enqueue_background(
                schedule_task,
                self._log_profile_restore_audit,
                str(tenant_id.value),
                user_id,
                str(profile_id.value),
            )

        logger.info(
            "restore_profile_completed",
            tenant_id=str(tenant_id),
            profile_id=str(profile_id),
            user_id=user_id,
        )

        return restored_profile

    async def get_profile_analytics(
        self,
        *,
        profile_id: ProfileId,
        tenant_id: TenantId,
        time_range_days: int = 30,
    ) -> dict[str, Any] | None:
        """
        Retrieve comprehensive analytics for a profile.

        Args:
            profile_id: The profile identifier
            tenant_id: The tenant identifier for multi-tenant isolation
            time_range_days: Analytics time range in days (default: 30)

        Returns:
            Dictionary containing profile analytics data, or None if profile not found

        Note:
            MVP implementation returns basic analytics (views, searches, completeness).
            Advanced metrics (match score distribution, popular searches, skill demand)
            require additional infrastructure and return placeholder values.
        """
        logger.info(
            "Profile analytics requested",
            profile_id=str(profile_id.value),
            tenant_id=str(tenant_id.value),
            time_range_days=time_range_days,
        )

        try:
            # Fetch the profile
            repository = self._deps.profile_repository
            profile = await repository.get_by_id(profile_id, tenant_id)

            if not profile:
                logger.warning(
                    "get_profile_analytics_not_found",
                    profile_id=str(profile_id.value),
                    tenant_id=str(tenant_id.value),
                )
                return None

            # Calculate profile completeness
            completeness_score = profile.calculate_completeness_score()

            # Extract analytics data from profile entity
            analytics_data = {
                "profile_id": str(profile_id.value),
                "view_count": profile.analytics.view_count,
                "search_appearances": profile.analytics.search_appearances,
                "last_viewed": profile.analytics.last_viewed_at,
                "profile_completeness": completeness_score,
                # MVP: Return empty/None for advanced metrics that require additional infrastructure
                "match_score_distribution": {},
                "popular_searches": [],
                "skill_demand_score": None,
            }

            logger.info(
                "Profile analytics retrieved successfully",
                profile_id=str(profile_id.value),
                tenant_id=str(tenant_id.value),
                view_count=profile.analytics.view_count,
                search_appearances=profile.analytics.search_appearances,
                completeness=completeness_score,
            )

            return analytics_data

        except Exception as exc:
            logger.error(
                "Failed to retrieve profile analytics",
                profile_id=str(profile_id.value),
                tenant_id=str(tenant_id.value),
                error=str(exc),
            )
            raise

    async def find_similar_profiles(
        self,
        *,
        profile_id: ProfileId,
        tenant_id: TenantId,
        similarity_threshold: float = 0.7,
        max_results: int = 10,
        schedule_task: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Find profiles similar to the target profile using vector similarity.

        Args:
            profile_id: Target profile to find similarities for
            tenant_id: Tenant identifier for multi-tenant isolation
            similarity_threshold: Minimum similarity score (0.0-1.0)
            max_results: Maximum number of results to return
            schedule_task: Optional task scheduler for analytics tracking

        Returns:
            List of similar profile dictionaries with similarity scores

        Raises:
            ValueError: If target profile not found
        """
        logger.info(
            "find_similar_profiles_requested",
            profile_id=str(profile_id),
            tenant_id=str(tenant_id),
            threshold=similarity_threshold,
            max_results=max_results,
        )

        # 1. Fetch target profile
        repository = self._deps.profile_repository
        target_profile = await repository.get_by_id(profile_id, tenant_id)

        if not target_profile:
            logger.warning(
                "find_similar_profiles_target_not_found",
                profile_id=str(profile_id),
                tenant_id=str(tenant_id),
            )
            raise ValueError(f"Target profile {profile_id} not found")

        # 2. Get target profile's embedding vector
        query_vector = None
        if target_profile.embeddings and target_profile.embeddings.overall:
            query_vector = target_profile.embeddings.overall.values

        # If no pre-computed embedding, generate on-the-fly
        if not query_vector and self._deps.embedding_service:
            searchable_text = target_profile.searchable_text
            if searchable_text:
                logger.info(
                    "Generating embedding for target profile",
                    profile_id=str(profile_id),
                )
                query_vector = await self._deps.embedding_service.generate_embedding(
                    text=searchable_text,
                    model="text-embedding-3-large",
                )

        if not query_vector:
            logger.warning(
                "find_similar_profiles_no_embedding",
                profile_id=str(profile_id),
            )
            return []

        # 3. Perform vector similarity search
        similar_profiles = await repository.search_by_vector(
            tenant_id=tenant_id,
            query_vector=query_vector,
            limit=max_results + 1,  # +1 to account for filtering out target profile
            threshold=similarity_threshold,
        )

        # 4. Filter out the target profile from results
        filtered_results = [
            (profile, score)
            for profile, score in similar_profiles
            if profile.id != profile_id
        ][:max_results]

        # 5. Build response data
        results = []
        for profile, match_score in filtered_results:
            # Get current company
            current_company = None
            for exp in profile.profile_data.experience:
                if exp.is_current_role():
                    current_company = exp.company
                    break
            if not current_company and profile.profile_data.experience:
                current_company = profile.profile_data.experience[0].company

            # Get top skills
            top_skills = [skill.name.value for skill in profile.profile_data.skills[:5]]

            # Build match explanation (simple version)
            match_explanation = self._build_match_explanation(
                target_skills=target_profile.normalized_skills or [],
                similar_skills=profile.normalized_skills or [],
                similarity_score=match_score.value,
            )

            results.append(
                {
                    "profile_id": str(profile.id.value),
                    "full_name": profile.profile_data.name,
                    "title": profile.profile_data.headline,
                    "current_company": current_company,
                    "top_skills": top_skills,
                    "similarity_score": match_score.value,
                    "match_explanation": match_explanation,
                }
            )

        # 6. Track analytics in background (non-blocking)
        if results:
            await self._enqueue_background(
                schedule_task,
                self._track_similarity_search_usage,
                str(tenant_id.value),
                str(profile_id.value),
                len(results),
            )

        logger.info(
            "find_similar_profiles_completed",
            profile_id=str(profile_id),
            results_count=len(results),
            threshold=similarity_threshold,
        )

        return results

    async def permanently_delete_profile(
        self,
        *,
        tenant_id: TenantId,
        profile_id: ProfileId,
        user_id: str,
        schedule_task: Any | None = None,
    ) -> bool:
        """Permanently delete a profile (removes all data).

        Orchestrates permanent deletion workflow:
        - Validate profile exists
        - Schedule background cleanup (search index, documents)
        - Track usage and audit logging BEFORE deletion
        - Delete from repository (hard delete)

        Args:
            tenant_id: Tenant identifier
            profile_id: Profile to delete permanently
            user_id: User performing deletion
            schedule_task: Task scheduler for background operations

        Returns:
            True if deleted, False if not found

        Raises:
            ValueError: If deletion fails
        """
        logger.info(
            "permanently_delete_profile_requested",
            tenant_id=str(tenant_id),
            profile_id=str(profile_id),
            user_id=user_id,
        )

        # Fetch profile first (need data for cleanup tasks)
        repository = self._deps.profile_repository
        profile = await repository.get_by_id(profile_id, tenant_id)

        if not profile:
            logger.warning(
                "permanently_delete_profile_not_found",
                tenant_id=str(tenant_id),
                profile_id=str(profile_id),
            )
            return False

        # Schedule cleanup tasks BEFORE deletion (while we still have profile data)
        # Remove from search index
        if self._deps.search_index_service:
            await self._enqueue_background(
                schedule_task,
                self._remove_from_search_index,
                str(profile_id.value),
                str(tenant_id.value),
            )

        # Cleanup document storage
        await self._enqueue_background(
            schedule_task,
            self._cleanup_profile_documents,
            str(profile_id.value),
            str(tenant_id.value),
            profile.metadata.get("original_filename"),
        )

        # Track usage BEFORE deletion
        await self._enqueue_background(
            schedule_task,
            self._track_profile_deletion_usage,
            str(tenant_id.value),
            user_id,
            "permanent_delete",
        )

        # Log audit event BEFORE deletion
        if self._deps.audit_service:
            await self._enqueue_background(
                schedule_task,
                self._log_profile_deletion_audit,
                str(tenant_id.value),
                user_id,
                str(profile_id.value),
                "permanent_delete",
                None,
            )

        # Perform hard delete via repository
        deleted = await repository.delete(profile_id, tenant_id)

        if deleted:
            logger.info(
                "permanently_delete_profile_completed",
                tenant_id=str(tenant_id),
                profile_id=str(profile_id),
                user_id=user_id,
            )
        else:
            logger.error(
                "permanently_delete_profile_failed",
                tenant_id=str(tenant_id),
                profile_id=str(profile_id),
            )

        return deleted

    async def _track_profile_deletion_usage(
        self,
        tenant_id: str,
        user_id: str,
        deletion_type: str,
    ) -> None:
        """Track profile deletion usage metrics."""
        try:
            if self._deps.usage_service:
                await self._deps.usage_service.track_usage(
                    tenant_id=tenant_id,
                    resource_type="profile",
                    amount=1,
                    metadata={
                        "operation": "delete",
                        "deletion_type": deletion_type,
                        "user_id": user_id,
                    },
                )
                logger.debug(
                    "Profile deletion usage tracked",
                    tenant_id=tenant_id,
                    user_id=user_id,
                    deletion_type=deletion_type,
                )
        except Exception as exc:
            logger.warning(
                "Failed to track profile deletion usage",
                tenant_id=tenant_id,
                error=str(exc),
            )

    async def _log_profile_deletion_audit(
        self,
        tenant_id: str,
        user_id: str,
        profile_id: str,
        deletion_type: str,
        reason: str | None,
    ) -> None:
        """Log profile deletion audit event."""
        try:
            if not self._deps.audit_service:
                return

            await self._deps.audit_service.log_event(
                event_type="profile_deleted",
                tenant_id=tenant_id,
                action="delete",
                resource_type="profile",
                user_id=user_id,
                resource_id=profile_id,
                details={
                    "deletion_type": deletion_type,
                    "reason": reason,
                },
            )

            logger.debug(
                "Profile deletion audit logged",
                tenant_id=tenant_id,
                profile_id=profile_id,
                deletion_type=deletion_type,
            )
        except Exception as exc:
            logger.warning(
                "Failed to log profile deletion audit",
                profile_id=profile_id,
                error=str(exc),
            )

    async def _track_profile_restoration_usage(
        self,
        tenant_id: str,
        user_id: str,
    ) -> None:
        """Track profile restoration usage metrics."""
        try:
            if self._deps.usage_service:
                await self._deps.usage_service.track_usage(
                    tenant_id=tenant_id,
                    resource_type="profile",
                    amount=1,
                    metadata={
                        "operation": "restore",
                        "user_id": user_id,
                    },
                )
                logger.debug(
                    "Profile restoration usage tracked",
                    tenant_id=tenant_id,
                    user_id=user_id,
                )
        except Exception as exc:
            logger.warning(
                "Failed to track profile restoration usage",
                tenant_id=tenant_id,
                error=str(exc),
            )

    async def _log_profile_restore_audit(
        self,
        tenant_id: str,
        user_id: str,
        profile_id: str,
    ) -> None:
        """Log profile restoration audit event."""
        try:
            if not self._deps.audit_service:
                return

            await self._deps.audit_service.log_event(
                event_type="profile_restored",
                tenant_id=tenant_id,
                action="restore",
                resource_type="profile",
                user_id=user_id,
                resource_id=profile_id,
                details={
                    "operation": "restore_deleted_profile",
                },
            )

            logger.debug(
                "Profile restoration audit logged",
                tenant_id=tenant_id,
                profile_id=profile_id,
            )
        except Exception as exc:
            logger.warning(
                "Failed to log profile restoration audit",
                profile_id=profile_id,
                error=str(exc),
            )

    async def _record_profile_view(
        self,
        profile_id: str,
        tenant_id: str,
    ) -> None:
        """Record profile view in background.

        Args:
            profile_id: Profile identifier string
            tenant_id: Tenant identifier string
        """
        try:
            # Re-fetch profile for update
            profile = await self._deps.profile_repository.get_by_id(
                ProfileId(profile_id),
                TenantId(tenant_id),
            )

            if profile:
                # Update analytics
                profile.record_view()

                # Persist updated analytics
                await self._deps.profile_repository.save(profile)

                logger.debug(
                    "Profile view recorded",
                    profile_id=profile_id,
                    view_count=profile.analytics.view_count,
                    last_viewed_at=profile.analytics.last_viewed_at.isoformat()
                    if profile.analytics.last_viewed_at
                    else None,
                )
        except Exception as exc:
            logger.warning(
                "Failed to record profile view",
                profile_id=profile_id,
                tenant_id=tenant_id,
                error=str(exc),
            )

    async def _remove_from_search_index(
        self,
        profile_id: str,
        tenant_id: str,
    ) -> None:
        """Remove profile from search index."""
        try:
            if not self._deps.search_index_service:
                logger.debug(
                    "Search index service not available", profile_id=profile_id
                )
                return

            logger.info(
                "Removing profile from search index",
                profile_id=profile_id,
                tenant_id=tenant_id,
            )

            await self._deps.search_index_service.remove_profile_index(
                profile_id=profile_id,
            )

            logger.info(
                "Profile removed from search index",
                profile_id=profile_id,
            )
        except Exception as exc:
            logger.warning(
                "Failed to remove profile from search index",
                profile_id=profile_id,
                error=str(exc),
            )

    async def _track_similarity_search_usage(
        self,
        tenant_id: str,
        profile_id: str,
        results_count: int,
    ) -> None:
        """Track similarity search usage metrics."""
        try:
            if self._deps.usage_service:
                await self._deps.usage_service.track_usage(
                    tenant_id=tenant_id,
                    resource_type="profile",
                    amount=1,
                    metadata={
                        "operation": "similarity_search",
                        "profile_id": profile_id,
                        "results_count": results_count,
                    },
                )
                logger.debug(
                    "Similarity search usage tracked",
                    tenant_id=tenant_id,
                    profile_id=profile_id,
                    results_count=results_count,
                )
        except Exception as exc:
            logger.warning(
                "Failed to track similarity search usage",
                tenant_id=tenant_id,
                error=str(exc),
            )

    @staticmethod
    def _build_match_explanation(
        target_skills: list[str],
        similar_skills: list[str],
        similarity_score: float,
    ) -> str:
        """Build a brief explanation for why profiles match.

        Args:
            target_skills: Normalized skills from target profile
            similar_skills: Normalized skills from similar profile
            similarity_score: Similarity score (0.0-1.0)

        Returns:
            Human-readable match explanation
        """
        # Find overlapping skills
        target_set = set(s.lower() for s in target_skills)
        similar_set = set(s.lower() for s in similar_skills)
        overlap = target_set & similar_set

        if not overlap:
            return f"{int(similarity_score * 100)}% overall similarity"

        overlap_count = len(overlap)
        if overlap_count == 1:
            skill_name = list(overlap)[0]
            return f"Shared expertise in {skill_name}"
        elif overlap_count <= 3:
            skills_str = ", ".join(sorted(overlap)[:3])
            return f"Shared expertise in {skills_str}"
        else:
            return f"{overlap_count} shared skills ({int(similarity_score * 100)}% similarity)"

    @staticmethod
    def _convert_skills_from_dict(skills_data: list[dict[str, Any]]) -> list[Skill]:
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

    @staticmethod
    def _convert_experience_from_dict(
        experience_data: list[dict[str, Any]],
    ) -> list[Experience]:
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

    @staticmethod
    def _convert_education_from_dict(
        education_data: list[dict[str, Any]],
    ) -> list[Education]:
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

    @staticmethod
    def _convert_location_from_dict(location_data: dict[str, Any]) -> Location | None:
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
                        logger.warning(
                            "invalid_coordinates", coordinates=coordinates_data
                        )
            elif (
                isinstance(coordinates_data, (list, tuple))
                and len(coordinates_data) == 2
            ):
                try:
                    coordinates = (
                        float(coordinates_data[0]),
                        float(coordinates_data[1]),
                    )
                except (TypeError, ValueError):
                    logger.warning("invalid_coordinates", coordinates=coordinates_data)

        # Only create Location if at least one field has a value
        if city or state or country or coordinates:
            return Location(
                city=city, state=state, country=country, coordinates=coordinates
            )
        return None

    @staticmethod
    def _build_summaries(
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

            if (
                not include_incomplete
                and profile.processing.status != DomainProcessingStatus.COMPLETED
            ):
                continue

            quality_score = profile.processing.quality_score
            if quality_min is not None and (quality_score or 0.0) < quality_min:
                continue

            experience_years = profile.profile_data.total_experience_years()
            if experience_min is not None and experience_years < experience_min:
                continue
            if experience_max is not None and experience_years > experience_max:
                continue

            normalized_skills = [
                str(skill) for skill in (profile.normalized_skills or [])
            ]
            if skill_filter and not any(
                skill_filter in skill for skill in normalized_skills
            ):
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
                    total_experience_years=int(round(experience_years))
                    if experience_years
                    else None,
                    top_skills=top_skills,
                    last_updated=profile.updated_at,
                    processing_status=profile.processing.status,
                    quality_score=quality_score,
                )
            )

        return summaries

    @staticmethod
    def _resolve_sort_key(sort_by: str):
        mapping = {
            "quality_score": lambda item: item.quality_score or 0.0,
            "experience": lambda item: item.total_experience_years or 0,
        }
        return mapping.get(sort_by, lambda item: item.last_updated)

    @staticmethod
    async def _enqueue_background(scheduler: Any | None, func, *args, **kwargs) -> None:
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
            if hasattr(scheduler, "add_task"):
                scheduler.add_task(func, *args, **kwargs)
            else:
                # Fallback to creating task
                result = func(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)

    async def _generate_profile_embeddings_batch(
        self,
        embedding_texts: dict[str, str | None],
        model: str,
    ) -> dict[str, list[float] | None]:
        """Generate embeddings for multiple text sections using batch processing.

        This method uses the OpenAI batch API to generate all embeddings in a single
        API call, improving efficiency and reducing latency.

        Args:
            embedding_texts: Dictionary mapping embedding type to text content
            model: OpenAI embedding model to use

        Returns:
            Dictionary mapping embedding type to embedding vector (or None if no text)

        Raises:
            RuntimeError: If embedding generation fails
        """
        # Prepare texts for batch processing
        # Order matters - we'll use indices to map back to types
        text_types = []  # Track which type each text belongs to
        texts_to_embed = []  # Only non-None texts

        for embedding_type in ["overall", "skills", "experience", "summary"]:
            text = embedding_texts.get(embedding_type)
            if text and text.strip():  # Only include non-empty texts
                text_types.append(embedding_type)
                texts_to_embed.append(text)

        # If no texts to embed, return all None
        if not texts_to_embed:
            logger.warning("No texts available for embedding generation")
            return {
                "overall": None,
                "skills": None,
                "experience": None,
                "summary": None,
            }

        try:
            # Generate all embeddings in batch (single API call)
            # This uses OpenAIService.generate_embeddings_batch() which is optimized for batch processing
            embedding_vectors = await self._deps.embedding_service.openai_service.generate_embeddings_batch(
                texts=texts_to_embed,
                model=model,
            )

            # Map results back to embedding types
            results = {
                "overall": None,
                "skills": None,
                "experience": None,
                "summary": None,
            }

            for i, embedding_type in enumerate(text_types):
                results[embedding_type] = embedding_vectors[i]

            logger.debug(
                "Batch embeddings generated",
                count=len(embedding_vectors),
                types=text_types,
                dimensions=len(embedding_vectors[0]) if embedding_vectors else 0,
            )

            return results

        except Exception as exc:
            logger.error(
                "Failed to generate batch embeddings",
                error=str(exc),
                texts_count=len(texts_to_embed),
            )
            raise

    @staticmethod
    def _extract_embedding_texts(profile: Profile) -> dict[str, str | None]:
        """Extract text content for each embedding type.

        Args:
            profile: Profile domain entity

        Returns:
            Dictionary mapping embedding type to extracted text

        Example:
            {
                "overall": "John Doe john@example.com Python Developer...",
                "skills": "Python, JavaScript, React, Node.js, Docker...",
                "experience": "Senior Engineer at TechCorp (2020-2024)...",
                "summary": "Experienced software engineer with 10+ years..."
            }
        """
        # Overall: Use existing searchable_text (combines all content)
        overall_text = profile.searchable_text if profile.searchable_text else None

        # Skills: Extract from profile_data.skills
        skills_text = None
        if profile.profile_data.skills:
            skill_parts = []
            for skill in profile.profile_data.skills:
                # Build skill description with context
                skill_desc = skill.name.value

                # Add proficiency context if available
                if skill.proficiency:
                    proficiency_labels = {
                        1: "beginner",
                        2: "intermediate",
                        3: "proficient",
                        4: "advanced",
                        5: "expert",
                    }
                    proficiency_label = proficiency_labels.get(
                        skill.proficiency, "skilled"
                    )
                    skill_desc = f"{proficiency_label} in {skill_desc}"

                # Add experience context if available
                if skill.years_of_experience:
                    skill_desc += f" ({skill.years_of_experience} years)"

                # Add category context
                if skill.category:
                    skill_desc += f" [{skill.category}]"

                # Add last used if available
                if skill.last_used:
                    skill_desc += f" (last used: {skill.last_used})"

                skill_parts.append(skill_desc)

            if skill_parts:
                skills_text = " | ".join(skill_parts)

        # Experience: Extract from profile_data.experience
        experience_text = None
        if profile.profile_data.experience:
            experience_parts = []
            for exp in profile.profile_data.experience:
                # Build experience description
                exp_desc_parts = [
                    f"{exp.title} at {exp.company}",
                    f"({exp.start_date} - {exp.end_date or 'Present'})",
                ]

                # Add location if available
                if exp.location:
                    exp_desc_parts.append(f"Location: {exp.location}")

                # Add description
                if exp.description:
                    exp_desc_parts.append(exp.description)

                # Add achievements
                if exp.achievements:
                    achievements_text = "; ".join(exp.achievements)
                    exp_desc_parts.append(f"Achievements: {achievements_text}")

                # Add skills used in this role
                if exp.skills:
                    skills_used = ", ".join(skill.value for skill in exp.skills)
                    exp_desc_parts.append(f"Skills: {skills_used}")

                experience_parts.append(" | ".join(exp_desc_parts))

            if experience_parts:
                experience_text = "\n\n".join(experience_parts)

        # Summary: Extract from profile_data.summary and headline
        summary_text = None
        summary_parts = []

        # Add headline if available
        if profile.profile_data.headline:
            summary_parts.append(profile.profile_data.headline)

        # Add summary if available
        if profile.profile_data.summary:
            summary_parts.append(profile.profile_data.summary)

        # Add experience level context
        if profile.experience_level:
            summary_parts.append(f"Experience Level: {profile.experience_level.value}")

        # Add total years of experience
        total_years = profile.profile_data.total_experience_years()
        if total_years > 0:
            summary_parts.append(f"Total Experience: {int(total_years)} years")

        if summary_parts:
            summary_text = "\n".join(summary_parts)

        return {
            "overall": overall_text,
            "skills": skills_text,
            "experience": experience_text,
            "summary": summary_text,
        }

    @staticmethod
    async def _cleanup_profile_documents(
        profile_id: str,
        tenant_id: str,
        original_filename: str | None,
    ) -> None:
        """Cleanup profile document storage."""
        try:
            logger.info(
                "Cleaning up profile documents",
                profile_id=profile_id,
                tenant_id=tenant_id,
                filename=original_filename,
            )

            # TODO: Implement document storage cleanup
            # This would interface with storage service (S3, local filesystem, etc.)
            # Example:
            # if self._deps.storage_service:
            #     await self._deps.storage_service.delete_profile_documents(
            #         tenant_id=tenant_id,
            #         profile_id=profile_id,
            #     )

            logger.info(
                "Profile documents cleanup completed",
                profile_id=profile_id,
            )
        except Exception as exc:
            logger.warning(
                "Failed to cleanup profile documents",
                profile_id=profile_id,
                error=str(exc),
            )


__all__ = [
    "ProfileApplicationService",
    "ProfileDeletionResult",
    "ProfileListItem",
    "ProfileListResult",
]
