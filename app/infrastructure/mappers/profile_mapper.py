"""Mapper translating between persistence models and domain profile entities."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from app.domain.entities.profile import (
    ExperienceLevel,
    Location,
    ProcessingMetadata as DomainProcessingMetadata,
    ProcessingStatus,
    PrivacySettings as DomainPrivacySettings,
    Profile,
    ProfileAnalytics as DomainProfileAnalytics,
    ProfileStatus,
)
from app.domain.value_objects import ProfileId, TenantId
from app.infrastructure.persistence.models.profile_table import (
    ProcessingMetadata,
    PrivacySettings,
    ProfileEmbeddings,
    ProfileTable,
    ProfileData,
)


def _parse_datetime(value: Optional[str | datetime]) -> Optional[datetime]:
    """Normalize ISO strings coming from Pydantic models to datetime objects."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class ProfileMapper:
    """Bi-directional mapper between SQLModel persistence and domain entities."""

    def to_domain(self, model: ProfileTable) -> Profile:
        profile_data = model.get_profile_data().model_dump(exclude_none=True)
        embeddings = model.get_embeddings()
        processing = model.get_processing_metadata()
        privacy = model.get_privacy_settings()
        analytics = model.get_analytics()

        location: Optional[Location] = None
        if any([model.location_city, model.location_state, model.location_country]):
            location = Location(
                city=model.location_city,
                state=model.location_state,
                country=model.location_country,
            )

        processing_metadata = DomainProcessingMetadata(
            status=ProcessingStatus(processing.processing_status.value if hasattr(processing.processing_status, "value") else processing.processing_status),
            version=getattr(processing, "processing_version", "1.0"),
            time_ms=getattr(processing, "processing_time", None),
            last_processed=_parse_datetime(getattr(processing, "last_processed", None)),
            error_message=getattr(processing, "error_message", None),
            additional={
                "extraction_method": getattr(processing, "extraction_method", None),
                "processing_started_at": getattr(processing, "processing_started_at", None),
                "processing_completed_at": getattr(processing, "processing_completed_at", None),
            },
        )

        privacy_settings = DomainPrivacySettings(
            consent_given=privacy.consent_given,
            consent_date=_parse_datetime(privacy.consent_date),
            data_retention_date=_parse_datetime(privacy.data_retention_date),
            gdpr_export_requested=privacy.gdpr_export_requested,
            deletion_requested=privacy.deletion_requested,
        )

        analytics_settings = DomainProfileAnalytics(
            view_count=model.view_count,
            search_appearances=model.search_appearances,
            last_viewed_at=_parse_datetime(analytics.last_viewed_at),
            match_score=analytics.match_score,
        )

        return Profile(
            id=ProfileId(model.id),
            tenant_id=TenantId(model.tenant_id),
            status=ProfileStatus(model.status.value if hasattr(model.status, "value") else model.status),
            name=model.name,
            email=model.email,
            phone=model.phone,
            searchable_text=model.searchable_text,
            profile_data=profile_data,
            normalized_skills=list(model.normalized_skills or []),
            keywords=list(model.keywords or []),
            experience_level=ExperienceLevel(model.experience_level) if model.experience_level else None,
            location=location,
            embeddings={
                "overall": embeddings.overall or [],
                "skills": embeddings.skills or [],
                "experience": embeddings.experience or [],
                "summary": embeddings.summary or [],
            },
            metadata={
                "source": model.source,
                "original_filename": model.original_filename,
                "file_size": model.file_size,
                "university_id": model.university_id,
                "corporate_id": model.corporate_id,
            },
            processing=processing_metadata,
            privacy=privacy_settings,
            analytics=analytics_settings,
            created_at=model.created_at,
            updated_at=model.updated_at,
            last_activity_at=model.last_activity_at,
        )

    def to_model(self, profile: Profile, *, existing: Optional[ProfileTable] = None) -> ProfileTable:
        model = existing or ProfileTable(
            id=profile.id.value,
            tenant_id=profile.tenant_id.value,
        )

        model.status = profile.status.value if isinstance(profile.status, ProfileStatus) else profile.status
        model.name = profile.name
        model.email = profile.email
        model.phone = profile.phone
        model.searchable_text = profile.searchable_text
        model.keywords = list(profile.keywords)
        model.normalized_skills = list(profile.normalized_skills)
        model.experience_level = profile.experience_level.value if profile.experience_level else None

        if profile.location:
            model.location_city = profile.location.city
            model.location_state = profile.location.state
            model.location_country = profile.location.country

        embeddings = ProfileEmbeddings(
            overall=profile.embeddings.get("overall", []),
            skills=profile.embeddings.get("skills"),
            experience=profile.embeddings.get("experience"),
            summary=profile.embeddings.get("summary"),
        )
        model.set_embeddings(embeddings)

        profile_data = ProfileData.model_validate(profile.profile_data)
        model.set_profile_data(profile_data)

        processing = ProcessingMetadata(
            processing_status=ProcessingStatus(profile.processing.status),
            processing_time=profile.processing.time_ms,
            processing_version=profile.processing.version,
            last_processed=profile.processing.last_processed.isoformat() if profile.processing.last_processed else None,
            error_message=profile.processing.error_message,
            extraction_method=profile.processing.additional.get("extraction_method"),
            processing_started_at=profile.processing.additional.get("processing_started_at"),
            processing_completed_at=profile.processing.additional.get("processing_completed_at"),
        )
        model.set_processing_metadata(processing)

        privacy = PrivacySettings(
            consent_given=profile.privacy.consent_given,
            consent_date=profile.privacy.consent_date.isoformat() if profile.privacy.consent_date else None,
            data_retention_date=profile.privacy.data_retention_date.isoformat() if profile.privacy.data_retention_date else None,
            gdpr_export_requested=profile.privacy.gdpr_export_requested,
            deletion_requested=profile.privacy.deletion_requested,
        )
        model.set_privacy_settings(privacy)

        model.view_count = profile.analytics.view_count
        model.search_appearances = profile.analytics.search_appearances
        model.last_viewed_at = profile.analytics.last_viewed_at
        model.last_activity_at = profile.last_activity_at

        metadata: Dict[str, Any] = profile.metadata
        model.source = metadata.get("source", model.source)
        model.original_filename = metadata.get("original_filename")
        model.file_size = metadata.get("file_size")
        model.university_id = metadata.get("university_id")
        model.corporate_id = metadata.get("corporate_id")

        model.created_at = profile.created_at
        model.updated_at = profile.updated_at

        return model
