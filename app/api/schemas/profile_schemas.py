"""
Profile API schemas and DTOs.

This module contains Pydantic models for Profile API requests and responses.
Following hexagonal architecture, these are pure DTOs in the API layer.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from pydantic import Field

from app.infrastructure.persistence.models.base import BaseModel
from app.infrastructure.persistence.models.profile_table import (
    ProfileData,
    ProfileEmbeddings,
    ProfileQuality,
    ProfileMetadata,
    PrivacySettings,
    ProfileAnalytics,
    ProfileTable,
    ProcessingMetadata,
)


class Profile(BaseModel):
    """Backward compatibility wrapper for ProfileTable."""
    # System fields
    id: str = Field(default_factory=lambda: str(uuid4()))
    tenant_id: str = Field(..., description="Tenant identifier")
    university_id: Optional[str] = Field(None, description="University identifier")
    corporate_id: Optional[str] = Field(None, description="Corporate identifier")

    # Profile data
    profile: ProfileData

    # Embeddings and search optimization
    embeddings: ProfileEmbeddings
    searchable_text: str = Field(..., description="Concatenated searchable content")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    normalized_skills: List[str] = Field(default_factory=list, description="Standardized skills")

    # Metadata
    profile_metadata: ProfileMetadata = Field(default_factory=ProfileMetadata)
    processing: ProcessingMetadata = Field(default_factory=ProcessingMetadata)

    # Timestamps
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    last_activity_at: Optional[str] = Field(None, description="Last activity timestamp")

    # Privacy and compliance
    privacy: PrivacySettings = Field(default_factory=PrivacySettings)

    # Analytics
    analytics: ProfileAnalytics = Field(default_factory=ProfileAnalytics)

    @classmethod
    def from_table(cls, profile_table: ProfileTable) -> "Profile":
        """Convert ProfileTable to Profile model for backward compatibility."""
        return cls(
            id=str(profile_table.id),
            tenant_id=str(profile_table.tenant_id),
            university_id=str(profile_table.university_id) if profile_table.university_id else None,
            corporate_id=str(profile_table.corporate_id) if profile_table.corporate_id else None,
            profile=profile_table.get_profile_data(),
            embeddings=profile_table.get_embeddings(),
            searchable_text=profile_table.searchable_text,
            keywords=profile_table.keywords,
            normalized_skills=profile_table.normalized_skills,
            profile_metadata=ProfileMetadata(
                source=profile_table.source,
                original_filename=profile_table.original_filename,
                file_size=profile_table.file_size,
                processing_version=profile_table.processing_metadata.get("processing_version", "1.0"),
                processing_time=profile_table.processing_metadata.get("processing_time"),
                last_updated=profile_table.updated_at.isoformat(),
                update_count=profile_table.version,
                quality=ProfileQuality(
                    score=int(profile_table.quality_score or 0),
                    missing_fields=[],
                    warnings=[]
                )
            ),
            processing=profile_table.get_processing_metadata(),
            created_at=profile_table.created_at.isoformat(),
            updated_at=profile_table.updated_at.isoformat(),
            last_activity_at=profile_table.last_activity_at.isoformat() if profile_table.last_activity_at else None,
            privacy=profile_table.get_privacy_settings(),
            analytics=profile_table.get_analytics()
        )

    def to_table(self, tenant_id: UUID) -> ProfileTable:
        """Convert Profile to ProfileTable for database storage."""
        profile_table = ProfileTable(
            id=UUID(self.id) if isinstance(self.id, str) else self.id,
            tenant_id=tenant_id,
            university_id=UUID(self.university_id) if self.university_id else None,
            corporate_id=UUID(self.corporate_id) if self.corporate_id else None,
            name=self.profile.name,
            email=self.profile.email,
            phone=self.profile.phone,
            searchable_text=self.searchable_text,
            keywords=self.keywords,
            normalized_skills=self.normalized_skills,
            source=self.profile_metadata.source,
            original_filename=self.profile_metadata.original_filename,
            file_size=self.profile_metadata.file_size
        )

        # Set complex data
        profile_table.set_profile_data(self.profile)
        profile_table.set_embeddings(self.embeddings)
        profile_table.set_processing_metadata(self.processing)
        profile_table.set_privacy_settings(self.privacy)

        return profile_table


class ProfileCreate(BaseModel):
    """Profile creation request"""
    tenant_id: str
    university_id: Optional[str] = None
    corporate_id: Optional[str] = None
    profile: ProfileData
    privacy: Optional[PrivacySettings] = None


class ProfileUpdate(BaseModel):
    """Profile update request"""
    profile: Optional[ProfileData] = None
    privacy: Optional[PrivacySettings] = None
    profile_metadata: Optional[Dict[str, Any]] = None


class ProfileResponse(BaseModel):
    """Profile response for API"""
    id: str
    tenant_id: str
    profile: ProfileData
    profile_metadata: ProfileMetadata
    created_at: str
    updated_at: str
    privacy: PrivacySettings
    analytics: Optional[ProfileAnalytics] = None  # Only for authorized users

    @classmethod
    def from_profile(cls, profile: Profile, include_analytics: bool = False) -> "ProfileResponse":
        """Create response from profile model"""
        return cls(
            id=profile.id,
            tenant_id=profile.tenant_id,
            profile=profile.profile,
            profile_metadata=profile.profile_metadata,
            created_at=profile.created_at,
            updated_at=profile.updated_at,
            privacy=profile.privacy,
            analytics=profile.analytics if include_analytics else None
        )

    @classmethod
    def from_table(cls, profile_table: ProfileTable, include_analytics: bool = False) -> "ProfileResponse":
        """Create response from ProfileTable"""
        return cls(
            id=str(profile_table.id),
            tenant_id=str(profile_table.tenant_id),
            profile=profile_table.get_profile_data(),
            profile_metadata=ProfileMetadata(
                source=profile_table.source,
                original_filename=profile_table.original_filename,
                file_size=profile_table.file_size,
                processing_version=profile_table.processing_metadata.get("processing_version", "1.0"),
                processing_time=profile_table.processing_metadata.get("processing_time"),
                last_updated=profile_table.updated_at.isoformat(),
                update_count=profile_table.version,
                quality=ProfileQuality(
                    score=int(profile_table.quality_score or 0),
                    missing_fields=[],
                    warnings=[]
                )
            ),
            created_at=profile_table.created_at.isoformat(),
            updated_at=profile_table.updated_at.isoformat(),
            privacy=profile_table.get_privacy_settings(),
            analytics=profile_table.get_analytics() if include_analytics else None
        )


class ProfileListResponse(BaseModel):
    """Paginated profile list response"""
    profiles: List[ProfileResponse]
    total_count: int
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)
    has_next: bool = False


class ProfileSearchFilters(BaseModel):
    """Profile search filters"""
    skills: Optional[List[str]] = None
    location: Optional[str] = None
    years_experience: Optional[int] = Field(None, ge=0)
    education_level: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None
    languages: Optional[List[str]] = None
    certifications: Optional[List[str]] = None
    availability: Optional[str] = None  # "available", "employed", "any"


# Backward compatibility aliases
CVProfile = Profile