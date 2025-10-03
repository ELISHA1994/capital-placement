"""Mapper between Profile domain entities and ProfileTable persistence models."""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from uuid import UUID

from app.domain.entities.profile import (
    Profile, 
    ProfileData, 
    ProfileEmbeddings, 
    ProcessingMetadata,
    PrivacySettings,
    ProfileAnalytics,
    Location,
    Experience,
    Education,
    Skill,
    ProfileStatus,
    ProcessingStatus,
    ExperienceLevel
)
from app.domain.value_objects import (
    ProfileId, 
    TenantId, 
    EmbeddingVector, 
    SkillName,
    EmailAddress,
    PhoneNumber
)
from app.models.profile import ProfileTable  # SQLModel persistence model


class ProfileMapper:
    """Maps between Profile domain entities and ProfileTable persistence models."""

    @staticmethod
    def to_domain(profile_table: ProfileTable) -> Profile:
        """Convert ProfileTable (persistence) to Profile (domain)."""
        # Map profile data
        profile_data_dict = profile_table.profile_data or {}
        profile_data = ProfileMapper._map_profile_data_to_domain(profile_data_dict)
        
        # Map embeddings
        embeddings = ProfileMapper._map_embeddings_to_domain(
            profile_table.overall_embedding,
            profile_table.skills_embedding,
            profile_table.experience_embedding,
            profile_table.summary_embedding
        )
        
        # Map processing metadata
        processing = ProfileMapper._map_processing_metadata_to_domain(
            profile_table.processing_metadata or {},
            profile_table.processing_status
        )
        
        # Map privacy settings
        privacy = ProfileMapper._map_privacy_settings_to_domain(
            profile_table.privacy_settings or {},
            profile_table.consent_given,
            profile_table.consent_date
        )
        
        # Map analytics
        analytics = ProfileAnalytics(
            view_count=profile_table.view_count,
            search_appearances=profile_table.search_appearances,
            last_viewed_at=profile_table.last_viewed_at,
            match_score=None  # Calculated dynamically
        )
        
        # Create domain entity
        return Profile(
            id=ProfileId(profile_table.id),
            tenant_id=TenantId(profile_table.tenant_id),
            status=ProfileStatus(profile_table.status),
            profile_data=profile_data,
            searchable_text=profile_table.searchable_text,
            normalized_skills=profile_table.normalized_skills or [],
            keywords=profile_table.keywords or [],
            experience_level=ExperienceLevel(profile_table.experience_level) if profile_table.experience_level else None,
            embeddings=embeddings,
            metadata=profile_table.metadata if hasattr(profile_table, 'metadata') else {},
            processing=processing,
            privacy=privacy,
            analytics=analytics,
            created_at=profile_table.created_at,
            updated_at=profile_table.updated_at,
            last_activity_at=profile_table.last_activity_at
        )

    @staticmethod
    def to_persistence(profile: Profile) -> ProfileTable:
        """Convert Profile (domain) to ProfileTable (persistence)."""
        # Map profile data to JSONB
        profile_data_dict = ProfileMapper._map_profile_data_to_persistence(profile.profile_data)
        
        # Map processing metadata to JSONB
        processing_dict = ProfileMapper._map_processing_metadata_to_persistence(profile.processing)
        
        # Map privacy settings to JSONB
        privacy_dict = ProfileMapper._map_privacy_settings_to_persistence(profile.privacy)
        
        # Create persistence model
        profile_table = ProfileTable(
            id=profile.id.value,
            tenant_id=profile.tenant_id.value,
            status=profile.status.value,
            experience_level=profile.experience_level.value if profile.experience_level else None,
            profile_data=profile_data_dict,
            searchable_text=profile.searchable_text,
            keywords=profile.keywords,
            normalized_skills=profile.normalized_skills,
            name=profile.profile_data.name,
            email=str(profile.profile_data.email),
            phone=str(profile.profile_data.phone) if profile.profile_data.phone else None,
            location_city=profile.profile_data.location.city if profile.profile_data.location else None,
            location_state=profile.profile_data.location.state if profile.profile_data.location else None,
            location_country=profile.profile_data.location.country if profile.profile_data.location else None,
            overall_embedding=profile.embeddings.overall.values if profile.embeddings.overall else None,
            skills_embedding=profile.embeddings.skills.values if profile.embeddings.skills else None,
            experience_embedding=profile.embeddings.experience.values if profile.embeddings.experience else None,
            summary_embedding=profile.embeddings.summary.values if profile.embeddings.summary else None,
            processing_status=profile.processing.status.value,
            processing_metadata=processing_dict,
            quality_score=profile.processing.quality_score,
            privacy_settings=privacy_dict,
            consent_given=profile.privacy.consent_given,
            consent_date=profile.privacy.consent_date,
            view_count=profile.analytics.view_count,
            search_appearances=profile.analytics.search_appearances,
            last_viewed_at=profile.analytics.last_viewed_at,
            last_activity_at=profile.last_activity_at,
            created_at=profile.created_at,
            updated_at=profile.updated_at
        )
        
        return profile_table

    @staticmethod
    def _map_profile_data_to_domain(data: Dict[str, Any]) -> ProfileData:
        """Map JSONB profile data to domain ProfileData."""
        # Map location
        location_data = data.get('location')
        location = None
        if location_data:
            location = Location(
                city=location_data.get('city'),
                state=location_data.get('state'),
                country=location_data.get('country'),
                coordinates=tuple(location_data['coordinates']) if location_data.get('coordinates') else None
            )
        
        # Map experience
        experience = []
        for exp_data in data.get('experience', []):
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
        
        # Map education
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
        
        # Map skills
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
        """Map domain ProfileData to JSONB dict."""
        # Map location
        location_dict = None
        if profile_data.location:
            location_dict = {
                'city': profile_data.location.city,
                'state': profile_data.location.state,
                'country': profile_data.location.country,
                'coordinates': list(profile_data.location.coordinates) if profile_data.location.coordinates else None
            }
        
        # Map experience
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
                'skills': [str(skill) for skill in exp.skills]
            })
        
        # Map education
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
                'name': str(skill.name),
                'category': skill.category,
                'proficiency': skill.proficiency,
                'years_of_experience': skill.years_of_experience,
                'endorsed': skill.endorsed,
                'last_used': skill.last_used
            })
        
        return {
            'name': profile_data.name,
            'email': str(profile_data.email),
            'phone': str(profile_data.phone) if profile_data.phone else None,
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
        summary: Optional[List[float]]
    ) -> ProfileEmbeddings:
        """Map embedding vectors to domain ProfileEmbeddings."""
        return ProfileEmbeddings(
            overall=EmbeddingVector(1536, overall) if overall else None,
            skills=EmbeddingVector(1536, skills) if skills else None,
            experience=EmbeddingVector(1536, experience) if experience else None,
            summary=EmbeddingVector(1536, summary) if summary else None
        )

    @staticmethod
    def _map_processing_metadata_to_domain(
        metadata_dict: Dict[str, Any],
        status: str
    ) -> ProcessingMetadata:
        """Map processing metadata to domain ProcessingMetadata."""
        return ProcessingMetadata(
            status=ProcessingStatus(status),
            version=metadata_dict.get('version', '1.0'),
            time_ms=metadata_dict.get('time_ms'),
            last_processed=metadata_dict.get('last_processed'),
            error_message=metadata_dict.get('error_message'),
            extraction_method=metadata_dict.get('extraction_method'),
            quality_score=metadata_dict.get('quality_score'),
            confidence_score=metadata_dict.get('confidence_score'),
            pages_processed=metadata_dict.get('pages_processed'),
            additional=metadata_dict.get('additional', {})
        )

    @staticmethod
    def _map_processing_metadata_to_persistence(processing: ProcessingMetadata) -> Dict[str, Any]:
        """Map domain ProcessingMetadata to JSONB dict."""
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
        consent_date: Optional[Any]
    ) -> PrivacySettings:
        """Map privacy settings to domain PrivacySettings."""
        return PrivacySettings(
            consent_given=consent_given,
            consent_date=consent_date,
            data_retention_date=privacy_dict.get('data_retention_date'),
            gdpr_export_requested=privacy_dict.get('gdpr_export_requested', False),
            deletion_requested=privacy_dict.get('deletion_requested', False)
        )

    @staticmethod
    def _map_privacy_settings_to_persistence(privacy: PrivacySettings) -> Dict[str, Any]:
        """Map domain PrivacySettings to JSONB dict."""
        return {
            'data_retention_date': privacy.data_retention_date.isoformat() if privacy.data_retention_date else None,
            'gdpr_export_requested': privacy.gdpr_export_requested,
            'deletion_requested': privacy.deletion_requested
        }

    @staticmethod
    def update_persistence_from_domain(profile_table: ProfileTable, profile: Profile) -> ProfileTable:
        """Update existing ProfileTable with data from Profile domain entity."""
        # Update denormalized fields
        profile_table.status = profile.status.value
        profile_table.experience_level = profile.experience_level.value if profile.experience_level else None
        profile_table.searchable_text = profile.searchable_text
        profile_table.keywords = profile.keywords
        profile_table.normalized_skills = profile.normalized_skills
        profile_table.name = profile.profile_data.name
        profile_table.email = str(profile.profile_data.email)
        profile_table.phone = str(profile.profile_data.phone) if profile.profile_data.phone else None
        
        if profile.profile_data.location:
            profile_table.location_city = profile.profile_data.location.city
            profile_table.location_state = profile.profile_data.location.state
            profile_table.location_country = profile.profile_data.location.country
        
        # Update JSONB fields
        profile_table.profile_data = ProfileMapper._map_profile_data_to_persistence(profile.profile_data)
        profile_table.processing_metadata = ProfileMapper._map_processing_metadata_to_persistence(profile.processing)
        profile_table.privacy_settings = ProfileMapper._map_privacy_settings_to_persistence(profile.privacy)
        
        # Update embeddings
        profile_table.overall_embedding = profile.embeddings.overall.values if profile.embeddings.overall else None
        profile_table.skills_embedding = profile.embeddings.skills.values if profile.embeddings.skills else None
        profile_table.experience_embedding = profile.embeddings.experience.values if profile.embeddings.experience else None
        profile_table.summary_embedding = profile.embeddings.summary.values if profile.embeddings.summary else None
        
        # Update status and metadata
        profile_table.processing_status = profile.processing.status.value
        profile_table.quality_score = profile.processing.quality_score
        profile_table.consent_given = profile.privacy.consent_given
        profile_table.consent_date = profile.privacy.consent_date
        
        # Update analytics
        profile_table.view_count = profile.analytics.view_count
        profile_table.search_appearances = profile.analytics.search_appearances
        profile_table.last_viewed_at = profile.analytics.last_viewed_at
        
        # Update timestamps
        profile_table.updated_at = profile.updated_at
        profile_table.last_activity_at = profile.last_activity_at
        
        return profile_table


__all__ = ["ProfileMapper"]