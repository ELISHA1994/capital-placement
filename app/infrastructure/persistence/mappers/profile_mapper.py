"""
Comprehensive mapper between Profile domain entities and ProfileTable persistence models.

This mapper handles bidirectional conversion between the rich domain model (Profile)
and the database persistence model (ProfileTable), managing:
- Value object conversions (ProfileId, TenantId, EmailAddress, etc.)
- JSONB serialization for complex nested structures
- Vector embedding transformations
- Datetime timezone handling
- Enum conversions
- Denormalized field synchronization
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.domain.entities.profile import (
    Education,
    Experience,
    ExperienceLevel,
    Location,
    PrivacySettings,
    ProcessingMetadata,
    ProcessingStatus,
    Profile,
    ProfileAnalytics,
    ProfileData,
    ProfileEmbeddings,
    ProfileStatus,
    Skill,
)
from app.domain.value_objects import (
    EmailAddress,
    EmbeddingVector,
    PhoneNumber,
    ProfileId,
    SkillName,
    TenantId,
)
from app.infrastructure.persistence.models.profile_table import ProfileTable


class ProfileMapper:
    """
    Maps between Profile domain entities and ProfileTable persistence models.

    This mapper ensures complete bidirectional conversion while handling:
    - Domain value objects ↔ Primitive types
    - Nested structures ↔ JSONB fields
    - Vector embeddings ↔ Database vector columns
    - Denormalized fields for query performance
    """

    @staticmethod
    def to_domain(table: ProfileTable) -> Profile:
        """
        Convert ProfileTable (persistence) to Profile (domain entity).

        Args:
            table: ProfileTable instance from database

        Returns:
            Profile domain entity with all value objects properly constructed

        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Map profile data from JSONB
        profile_data_dict = table.profile_data or {}
        profile_data = ProfileMapper._map_profile_data_to_domain(profile_data_dict)

        # Map embeddings from vector columns
        embeddings = ProfileMapper._map_embeddings_to_domain(
            overall=table.overall_embedding,
            skills=table.skills_embedding,
            experience=table.experience_embedding,
            education=getattr(table, 'education_embedding', None),
            summary=table.summary_embedding
        )

        # Map processing metadata from JSONB
        processing = ProfileMapper._map_processing_metadata_to_domain(
            metadata_dict=table.processing_metadata or {},
            status=table.processing_status
        )

        # Map privacy settings from JSONB and denormalized fields
        privacy = ProfileMapper._map_privacy_settings_to_domain(
            privacy_dict=table.privacy_settings or {},
            consent_given=table.consent_given,
            consent_date=table.consent_date
        )

        # Map analytics from denormalized fields
        analytics = ProfileAnalytics(
            view_count=table.view_count,
            search_appearances=table.search_appearances,
            last_viewed_at=table.last_viewed_at,
            match_score=None  # Calculated dynamically during search
        )

        # Create domain entity
        return Profile(
            id=ProfileId(table.id),
            tenant_id=TenantId(table.tenant_id),
            status=ProfileStatus(table.status),
            profile_data=profile_data,
            searchable_text=table.searchable_text or "",
            normalized_skills=table.normalized_skills or [],
            keywords=table.keywords or [],
            experience_level=ExperienceLevel(table.experience_level) if table.experience_level else None,
            embeddings=embeddings,
            metadata={},  # Can be extended to map additional metadata
            processing=processing,
            privacy=privacy,
            analytics=analytics,
            created_at=table.created_at,
            updated_at=table.updated_at,
            last_activity_at=table.last_activity_at
        )

    @staticmethod
    def to_table(entity: Profile) -> ProfileTable:
        """
        Convert Profile (domain entity) to ProfileTable (persistence).

        Args:
            entity: Profile domain entity

        Returns:
            ProfileTable ready for database persistence
        """
        # Map profile data to JSONB
        profile_data_dict = ProfileMapper._map_profile_data_to_persistence(entity.profile_data)

        # Map processing metadata to JSONB
        processing_dict = ProfileMapper._map_processing_metadata_to_persistence(entity.processing)

        # Map privacy settings to JSONB
        privacy_dict = ProfileMapper._map_privacy_settings_to_persistence(entity.privacy)

        # Create persistence model
        table = ProfileTable(
            id=entity.id.value,
            tenant_id=entity.tenant_id.value,

            # Status fields
            status=entity.status.value,
            experience_level=entity.experience_level.value if entity.experience_level else None,

            # Profile data (JSONB)
            profile_data=profile_data_dict,

            # Search fields
            searchable_text=entity.searchable_text,
            keywords=entity.keywords,
            normalized_skills=entity.normalized_skills,

            # Denormalized contact fields for performance
            name=entity.profile_data.name,
            email=str(entity.profile_data.email),
            phone=str(entity.profile_data.phone) if entity.profile_data.phone else None,

            # Denormalized location fields for geo queries
            location_city=entity.profile_data.location.city if entity.profile_data.location else None,
            location_state=entity.profile_data.location.state if entity.profile_data.location else None,
            location_country=entity.profile_data.location.country if entity.profile_data.location else None,

            # Vector embeddings
            overall_embedding=entity.embeddings.overall.values if entity.embeddings.overall else None,
            skills_embedding=entity.embeddings.skills.values if entity.embeddings.skills else None,
            experience_embedding=entity.embeddings.experience.values if entity.embeddings.experience else None,
            summary_embedding=entity.embeddings.summary.values if entity.embeddings.summary else None,

            # Processing metadata
            processing_status=entity.processing.status.value,
            processing_metadata=processing_dict,
            quality_score=entity.processing.quality_score,

            # Privacy settings
            privacy_settings=privacy_dict,
            consent_given=entity.privacy.consent_given,
            consent_date=entity.privacy.consent_date,

            # Analytics
            view_count=entity.analytics.view_count,
            search_appearances=entity.analytics.search_appearances,
            last_viewed_at=entity.analytics.last_viewed_at,

            # Timestamps
            last_activity_at=entity.last_activity_at,
            created_at=entity.created_at,
            updated_at=entity.updated_at
        )

        return table

    @staticmethod
    def update_table_from_domain(table: ProfileTable, entity: Profile) -> ProfileTable:
        """
        Update existing ProfileTable with data from Profile domain entity.

        This method preserves the table's database identity while syncing all
        fields from the domain entity. Useful for updates without creating new instances.

        Args:
            table: Existing ProfileTable instance to update
            entity: Profile domain entity with updated data

        Returns:
            Updated ProfileTable (same instance, modified in place)
        """
        # Update status fields
        table.status = entity.status.value
        table.experience_level = entity.experience_level.value if entity.experience_level else None

        # Update search fields
        table.searchable_text = entity.searchable_text
        table.keywords = entity.keywords
        table.normalized_skills = entity.normalized_skills

        # Update denormalized contact fields
        table.name = entity.profile_data.name
        table.email = str(entity.profile_data.email)
        table.phone = str(entity.profile_data.phone) if entity.profile_data.phone else None

        # Update denormalized location fields
        if entity.profile_data.location:
            table.location_city = entity.profile_data.location.city
            table.location_state = entity.profile_data.location.state
            table.location_country = entity.profile_data.location.country
        else:
            table.location_city = None
            table.location_state = None
            table.location_country = None

        # Update JSONB fields
        table.profile_data = ProfileMapper._map_profile_data_to_persistence(entity.profile_data)
        table.processing_metadata = ProfileMapper._map_processing_metadata_to_persistence(entity.processing)
        table.privacy_settings = ProfileMapper._map_privacy_settings_to_persistence(entity.privacy)

        # Update vector embeddings
        table.overall_embedding = entity.embeddings.overall.values if entity.embeddings.overall else None
        table.skills_embedding = entity.embeddings.skills.values if entity.embeddings.skills else None
        table.experience_embedding = entity.embeddings.experience.values if entity.embeddings.experience else None
        table.summary_embedding = entity.embeddings.summary.values if entity.embeddings.summary else None

        # Update processing metadata
        table.processing_status = entity.processing.status.value
        table.quality_score = entity.processing.quality_score

        # Update privacy settings
        table.consent_given = entity.privacy.consent_given
        table.consent_date = entity.privacy.consent_date

        # Update analytics
        table.view_count = entity.analytics.view_count
        table.search_appearances = entity.analytics.search_appearances
        table.last_viewed_at = entity.analytics.last_viewed_at

        # Update timestamps
        table.updated_at = entity.updated_at
        table.last_activity_at = entity.last_activity_at

        return table

    # ========================================================================
    # Private helper methods for complex nested structures
    # ========================================================================

    @staticmethod
    def _map_profile_data_to_domain(data: Dict[str, Any]) -> ProfileData:
        """
        Map JSONB profile data dictionary to domain ProfileData.

        Args:
            data: Dictionary from profile_data JSONB field

        Returns:
            ProfileData with all nested value objects constructed
        """
        # Map location
        location = None
        location_data = data.get('location')
        if location_data:
            # Handle both tuple and dict formats for coordinates
            coords = location_data.get('coordinates')
            if coords:
                if isinstance(coords, dict):
                    coords = (coords.get('lat'), coords.get('lng'))
                elif isinstance(coords, list):
                    coords = tuple(coords)

            location = Location(
                city=location_data.get('city'),
                state=location_data.get('state'),
                country=location_data.get('country'),
                coordinates=coords
            )

        # Map experience entries
        experience = []
        for exp_data in data.get('experience', []):
            # Convert skill strings to SkillName value objects
            skills = [SkillName(skill) for skill in exp_data.get('skills', [])]

            experience.append(Experience(
                title=exp_data['title'],
                company=exp_data['company'],
                start_date=exp_data['start_date'],
                description=exp_data.get('description', ''),
                end_date=exp_data.get('end_date'),
                current=exp_data.get('current', False),
                location=exp_data.get('location'),
                achievements=exp_data.get('achievements', []),
                skills=skills
            ))

        # Map education entries
        education = []
        for edu_data in data.get('education', []):
            education.append(Education(
                institution=edu_data['institution'],
                degree=edu_data['degree'],
                field=edu_data['field'],
                start_date=edu_data.get('start_date'),
                end_date=edu_data.get('end_date'),
                gpa=edu_data.get('gpa'),
                achievements=edu_data.get('achievements', [])
            ))

        # Map skills with SkillName value objects
        skills = []
        for skill_data in data.get('skills', []):
            skills.append(Skill(
                name=SkillName(skill_data['name']),
                category=skill_data.get('category', 'technical'),
                proficiency=skill_data.get('proficiency'),
                years_of_experience=skill_data.get('years_of_experience'),
                endorsed=skill_data.get('endorsed', False),
                last_used=skill_data.get('last_used')
            ))

        # Construct ProfileData with value objects
        return ProfileData(
            name=data['name'],
            email=EmailAddress(data['email']),
            phone=PhoneNumber(data['phone']) if data.get('phone') else None,
            location=location,
            summary=data.get('summary'),
            headline=data.get('headline'),
            experience=experience,
            education=education,
            skills=skills,
            languages=data.get('languages', [])
        )

    @staticmethod
    def _map_profile_data_to_persistence(profile_data: ProfileData) -> Dict[str, Any]:
        """
        Map domain ProfileData to JSONB dictionary.

        Args:
            profile_data: ProfileData domain object

        Returns:
            Dictionary suitable for JSONB storage
        """
        # Map location
        location_dict = None
        if profile_data.location:
            location_dict = {
                'city': profile_data.location.city,
                'state': profile_data.location.state,
                'country': profile_data.location.country,
                'coordinates': list(profile_data.location.coordinates) if profile_data.location.coordinates else None
            }

        # Map experience entries
        experience_list = []
        for exp in profile_data.experience:
            experience_list.append({
                'title': exp.title,
                'company': exp.company,
                'start_date': exp.start_date,
                'description': exp.description,
                'end_date': exp.end_date,
                'current': exp.current,
                'location': exp.location,
                'achievements': exp.achievements,
                'skills': [str(skill) for skill in exp.skills]  # Convert SkillName to string
            })

        # Map education entries
        education_list = []
        for edu in profile_data.education:
            education_list.append({
                'institution': edu.institution,
                'degree': edu.degree,
                'field': edu.field,
                'start_date': edu.start_date,
                'end_date': edu.end_date,
                'gpa': edu.gpa,
                'achievements': edu.achievements
            })

        # Map skills
        skills_list = []
        for skill in profile_data.skills:
            skills_list.append({
                'name': str(skill.name),  # Convert SkillName to string
                'category': skill.category,
                'proficiency': skill.proficiency,
                'years_of_experience': skill.years_of_experience,
                'endorsed': skill.endorsed,
                'last_used': skill.last_used
            })

        return {
            'name': profile_data.name,
            'email': str(profile_data.email),  # Convert EmailAddress to string
            'phone': str(profile_data.phone) if profile_data.phone else None,  # Convert PhoneNumber to string
            'location': location_dict,
            'summary': profile_data.summary,
            'headline': profile_data.headline,
            'experience': experience_list,
            'education': education_list,
            'skills': skills_list,
            'languages': profile_data.languages
        }

    @staticmethod
    def _map_embeddings_to_domain(
        overall: Optional[List[float]],
        skills: Optional[List[float]],
        experience: Optional[List[float]],
        education: Optional[List[float]],
        summary: Optional[List[float]]
    ) -> ProfileEmbeddings:
        """
        Map embedding vector lists to domain ProfileEmbeddings.

        Args:
            overall: Overall profile embedding vector
            skills: Skills-specific embedding vector
            experience: Experience-specific embedding vector
            education: Education-specific embedding vector
            summary: Summary-specific embedding vector

        Returns:
            ProfileEmbeddings with EmbeddingVector value objects
        """
        def _as_embedding_vector(values: Optional[List[float]]) -> Optional[EmbeddingVector]:
            if not values:
                return None
            length = len(values)
            return EmbeddingVector(dimensions=length, values=list(values))

        return ProfileEmbeddings(
            overall=_as_embedding_vector(overall),
            skills=_as_embedding_vector(skills),
            experience=_as_embedding_vector(experience),
            education=_as_embedding_vector(education),
            summary=_as_embedding_vector(summary),
        )

    @staticmethod
    def _map_processing_metadata_to_domain(
        metadata_dict: Dict[str, Any],
        status: str
    ) -> ProcessingMetadata:
        """
        Map processing metadata dictionary to domain ProcessingMetadata.

        Args:
            metadata_dict: Dictionary from processing_metadata JSONB field
            status: Processing status string

        Returns:
            ProcessingMetadata domain object
        """
        # Parse datetime from ISO string if present
        last_processed = None
        if metadata_dict.get('last_processed'):
            try:
                last_processed = datetime.fromisoformat(
                    metadata_dict['last_processed'].replace('Z', '+00:00')
                )
            except (ValueError, AttributeError):
                pass

        return ProcessingMetadata(
            status=ProcessingStatus(status),
            version=metadata_dict.get('version', '1.0'),
            time_ms=metadata_dict.get('time_ms'),
            last_processed=last_processed,
            error_message=metadata_dict.get('error_message'),
            extraction_method=metadata_dict.get('extraction_method'),
            quality_score=metadata_dict.get('quality_score'),
            confidence_score=metadata_dict.get('confidence_score'),
            pages_processed=metadata_dict.get('pages_processed'),
            additional=metadata_dict.get('additional', {})
        )

    @staticmethod
    def _map_processing_metadata_to_persistence(processing: ProcessingMetadata) -> Dict[str, Any]:
        """
        Map domain ProcessingMetadata to JSONB dictionary.

        Args:
            processing: ProcessingMetadata domain object

        Returns:
            Dictionary suitable for JSONB storage
        """
        return {
            'version': processing.version,
            'time_ms': processing.time_ms,
            'last_processed': processing.last_processed.isoformat() if processing.last_processed else None,
            'error_message': processing.error_message,
            'extraction_method': processing.extraction_method,
            'quality_score': processing.quality_score,
            'confidence_score': processing.confidence_score,
            'pages_processed': processing.pages_processed,
            'additional': processing.additional
        }

    @staticmethod
    def _map_privacy_settings_to_domain(
        privacy_dict: Dict[str, Any],
        consent_given: bool,
        consent_date: Optional[datetime]
    ) -> PrivacySettings:
        """
        Map privacy settings dictionary to domain PrivacySettings.

        Args:
            privacy_dict: Dictionary from privacy_settings JSONB field
            consent_given: Consent flag from denormalized field
            consent_date: Consent date from denormalized field

        Returns:
            PrivacySettings domain object
        """
        # Parse datetime fields
        data_retention_date = None
        if privacy_dict.get('data_retention_date'):
            try:
                data_retention_date = datetime.fromisoformat(
                    privacy_dict['data_retention_date'].replace('Z', '+00:00')
                )
            except (ValueError, AttributeError):
                pass

        return PrivacySettings(
            consent_given=consent_given,
            consent_date=consent_date,
            data_retention_date=data_retention_date,
            gdpr_export_requested=privacy_dict.get('gdpr_export_requested', False),
            deletion_requested=privacy_dict.get('deletion_requested', False)
        )

    @staticmethod
    def _map_privacy_settings_to_persistence(privacy: PrivacySettings) -> Dict[str, Any]:
        """
        Map domain PrivacySettings to JSONB dictionary.

        Args:
            privacy: PrivacySettings domain object

        Returns:
            Dictionary suitable for JSONB storage
        """
        return {
            'data_retention_date': privacy.data_retention_date.isoformat() if privacy.data_retention_date else None,
            'gdpr_export_requested': privacy.gdpr_export_requested,
            'deletion_requested': privacy.deletion_requested
        }


__all__ = ["ProfileMapper"]
