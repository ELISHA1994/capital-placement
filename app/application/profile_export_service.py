"""Application service for profile CSV export functionality.

Provides memory-efficient streaming CSV export with configurable privacy settings
and filtering capabilities following pure hexagonal architecture principles.
"""

from __future__ import annotations

import csv
from collections.abc import Generator
from io import StringIO

import structlog

from app.application.dependencies.profile_dependencies import ProfileDependencies
from app.domain.entities.profile import ProcessingStatus, Profile
from app.domain.value_objects import TenantId

logger = structlog.get_logger(__name__)


class ProfileExportService:
    """Application service orchestrating profile CSV export operations.

    Implements memory-efficient streaming export using generators to handle
    large datasets (100k+ profiles) without memory bloat. Supports configurable
    field selection for privacy compliance and filtering by status and skills.

    Architecture:
        - Works with domain entities (Profile), never persistence tables
        - Batch-processes profiles from repository (100 at a time)
        - Yields CSV rows incrementally for streaming response
        - Uses Python's csv module for proper CSV formatting
        - Respects privacy settings via conditional field inclusion
    """

    def __init__(self, dependencies: ProfileDependencies) -> None:
        """Initialize with injected dependencies.

        Args:
            dependencies: Profile domain dependencies including repository
        """
        self._deps = dependencies

    def generate_csv_export(
        self,
        *,
        tenant_id: TenantId,
        status_filter: ProcessingStatus | None = None,
        skill_filter: str | None = None,
        include_contact_info: bool = False,
        include_full_text: bool = False,
    ) -> Generator[str, None, None]:
        """Generate CSV export as streaming generator.

        This is a synchronous generator (NOT async) because:
        1. CSV generation is CPU-bound, not I/O-bound
        2. FastAPI StreamingResponse expects sync generators
        3. File I/O operations (StringIO) are synchronous

        The generator yields CSV data row by row, processing profiles in
        batches to maintain memory efficiency for large datasets.

        Args:
            tenant_id: Tenant identifier for data isolation
            status_filter: Optional filter by processing status
            skill_filter: Optional filter by skill (partial match)
            include_contact_info: Include email, phone, location fields
            include_full_text: Include summary and searchable text

        Yields:
            CSV row strings (header first, then data rows)

        Note:
            This method is synchronous (def, not async def) to work with
            FastAPI's StreamingResponse. Repository calls are made synchronous
            using asyncio.run() within the generator.
        """
        logger.info(
            "csv_export_generation_started",
            tenant_id=str(tenant_id.value),
            status_filter=status_filter.value if status_filter else None,
            skill_filter=skill_filter,
            include_contact_info=include_contact_info,
            include_full_text=include_full_text,
        )

        # Normalize skill filter for case-insensitive matching
        normalized_skill_filter = (skill_filter or "").strip().lower() if skill_filter else None

        # 1. Yield CSV header row first
        header_row = self._build_csv_header(
            include_contact_info=include_contact_info,
            include_full_text=include_full_text,
        )

        # Convert header list to CSV string
        header_io = StringIO()
        header_writer = csv.writer(header_io)
        header_writer.writerow(header_row)
        yield header_io.getvalue()

        # 2. Stream profile data rows in batches
        batch_size = 100
        offset = 0
        total_exported = 0
        total_filtered = 0

        while True:
            # Fetch batch of profiles from repository
            # Note: We use asyncio.run() to make async repository call synchronous
            # This is acceptable because we're in a generator context
            import asyncio

            try:
                profiles = asyncio.run(
                    self._deps.profile_repository.list_by_tenant(
                        tenant_id=tenant_id,
                        status=status_filter,
                        limit=batch_size,
                        offset=offset,
                    )
                )
            except RuntimeError:
                # If there's already an event loop running, use run_coroutine_threadsafe
                # This can happen in certain FastAPI contexts
                loop = asyncio.get_event_loop()
                future = asyncio.ensure_future(
                    self._deps.profile_repository.list_by_tenant(
                        tenant_id=tenant_id,
                        status=status_filter,
                        limit=batch_size,
                        offset=offset,
                    )
                )
                profiles = loop.run_until_complete(future)

            # If no more profiles, we're done
            if not profiles:
                logger.info(
                    "csv_export_generation_completed",
                    tenant_id=str(tenant_id.value),
                    total_exported=total_exported,
                    total_filtered=total_filtered,
                )
                break

            # 3. Filter and format each profile in the batch
            for profile in profiles:
                total_filtered += 1

                # Apply skill filter if provided
                if normalized_skill_filter:
                    profile_skills = [s.lower() for s in (profile.normalized_skills or [])]
                    if not any(normalized_skill_filter in skill for skill in profile_skills):
                        continue  # Skip profiles that don't match skill filter

                # Format profile as CSV row
                row_data = self._format_csv_row(
                    profile=profile,
                    include_contact_info=include_contact_info,
                    include_full_text=include_full_text,
                )

                # Convert row dict to CSV string
                row_io = StringIO()
                row_writer = csv.DictWriter(row_io, fieldnames=header_row)
                row_writer.writerow(row_data)
                yield row_io.getvalue()

                total_exported += 1

            # Move to next batch
            offset += batch_size

            # Log progress every 1000 profiles
            if total_exported > 0 and total_exported % 1000 == 0:
                logger.debug(
                    "csv_export_progress",
                    tenant_id=str(tenant_id.value),
                    exported=total_exported,
                    filtered=total_filtered,
                )

    # === STATIC HELPER METHODS ===

    @staticmethod
    def _build_csv_header(
        include_contact_info: bool,
        include_full_text: bool,
    ) -> list[str]:
        """Build CSV header row based on field selection.

        Args:
            include_contact_info: Include contact fields
            include_full_text: Include full text fields

        Returns:
            List of header field names
        """
        # Base fields (always included)
        header = [
            "profile_id",
            "full_name",
            "title",
            "current_company",
            "total_experience_years",
            "top_skills",
            "processing_status",
            "quality_score",
        ]

        # Conditional contact fields
        if include_contact_info:
            header.extend(["email", "phone", "location"])

        # Conditional full text fields
        if include_full_text:
            header.extend(["summary", "searchable_text"])

        return header

    @staticmethod
    def _format_csv_row(
        profile: Profile,
        include_contact_info: bool,
        include_full_text: bool,
    ) -> dict[str, str]:
        """Format profile entity as CSV row dictionary.

        Extracts relevant fields from domain entity and formats them
        for CSV export. Handles None values gracefully by converting
        to empty strings.

        Args:
            profile: Domain profile entity
            include_contact_info: Include contact information
            include_full_text: Include full text fields

        Returns:
            Dictionary mapping header fields to formatted values
        """
        # Base fields
        row = {
            "profile_id": str(profile.id.value),
            "full_name": profile.profile_data.name or "",
            "title": profile.profile_data.headline or "",
            "current_company": ProfileExportService._get_current_company(profile) or "",
            "total_experience_years": str(int(profile.profile_data.total_experience_years())) or "0",
            "top_skills": ProfileExportService._get_top_skills(profile, max_skills=5),
            "processing_status": profile.processing.status.value,
            "quality_score": f"{profile.processing.quality_score:.2f}" if profile.processing.quality_score else "",
        }

        # Conditional contact fields
        if include_contact_info:
            row["email"] = str(profile.profile_data.email) if profile.profile_data.email else ""
            row["phone"] = str(profile.profile_data.phone) if profile.profile_data.phone else ""
            row["location"] = ProfileExportService._format_location(profile) or ""

        # Conditional full text fields
        if include_full_text:
            row["summary"] = profile.profile_data.summary or ""
            # Truncate searchable text to 500 chars to keep CSV manageable
            searchable_text = profile.searchable_text or ""
            row["searchable_text"] = searchable_text[:500] if searchable_text else ""

        return row

    @staticmethod
    def _get_current_company(profile: Profile) -> str | None:
        """Extract current company from profile experience.

        Args:
            profile: Domain profile entity

        Returns:
            Current company name, or most recent company, or None
        """
        # Try to find current role
        for exp in profile.profile_data.experience:
            if exp.is_current_role():
                return exp.company

        # Fall back to most recent experience (first in list)
        if profile.profile_data.experience:
            return profile.profile_data.experience[0].company

        return None

    @staticmethod
    def _get_top_skills(profile: Profile, max_skills: int = 5) -> str:
        """Extract top skills as comma-separated string.

        Args:
            profile: Domain profile entity
            max_skills: Maximum number of skills to include

        Returns:
            Comma-separated skill names
        """
        if not profile.profile_data.skills:
            return ""

        top_skills = [
            skill.name.value
            for skill in profile.profile_data.skills[:max_skills]
        ]

        return ", ".join(top_skills)

    @staticmethod
    def _format_location(profile: Profile) -> str | None:
        """Format location as readable string.

        Args:
            profile: Domain profile entity

        Returns:
            Formatted location string (e.g., "San Francisco, CA, USA")
        """
        location = profile.profile_data.location
        if not location:
            return None

        parts = []
        if location.city:
            parts.append(location.city)
        if location.state:
            parts.append(location.state)
        if location.country:
            parts.append(location.country)

        return ", ".join(parts) if parts else None


__all__ = ["ProfileExportService"]
