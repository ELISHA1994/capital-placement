"""
SQLModel Profile models with comprehensive data structures and vector support.

This module provides complete profile management with database persistence,
vector embeddings for AI/ML similarity search, and comprehensive validation
while preserving all original Pydantic functionality.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from enum import Enum

from pydantic import field_validator, model_validator
from sqlalchemy import Column, String, Boolean, Integer, Float, Text, Index, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID, JSONB, ARRAY
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql import func
from sqlmodel import Field, SQLModel, Relationship
from pgvector.sqlalchemy import Vector

from .base import AuditableModel, VectorModel, MetadataModel, BaseModel


def create_tenant_id_column():
    """Create a unique tenant_id Column instance for each table."""
    return Column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )


class ProfileStatus(str, Enum):
    """Profile status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DRAFT = "draft"
    ARCHIVED = "archived"
    DELETED = "deleted"


class ExperienceLevel(str, Enum):
    """Experience level enumeration"""
    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"
    EXECUTIVE = "executive"


class ProcessingStatus(str, Enum):
    """Processing status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class ProcessingMetadata(BaseModel):
    """Processing metadata for profile"""
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    processing_time: Optional[int] = Field(None, description="Processing time in milliseconds")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    processing_version: str = Field("1.0", description="Processing version")
    last_processed: Optional[str] = Field(None, description="Last processed timestamp")
    
    # Additional fields used by document_processor
    extraction_method: Optional[str] = Field(None, description="Extraction method used")
    processing_started_at: Optional[datetime] = Field(None, description="Processing start time")
    processing_completed_at: Optional[datetime] = Field(None, description="Processing completion time")
    processing_duration_seconds: Optional[float] = Field(None, description="Processing duration in seconds")
    quality_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Quality assessment score")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Processing confidence score")
    pages_processed: Optional[int] = Field(None, ge=0, description="Number of pages processed")
    document_language: Optional[str] = Field("en", description="Detected document language")


class ContactInfo(BaseModel):
    """Contact information model"""
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number") 
    address: Optional[str] = Field(None, description="Full address")
    city: Optional[str] = Field(None, description="City")
    state: Optional[str] = Field(None, description="State/Province")
    country: Optional[str] = Field(None, description="Country")
    linkedin: Optional[str] = Field(None, description="LinkedIn profile URL")
    website: Optional[str] = Field(None, description="Personal website")


class Location(BaseModel):
    """Geographic location information"""
    city: Optional[str] = None
    state: Optional[str] = None
    country: str
    coordinates: Optional[Dict[str, float]] = None  # {"lat": 0.0, "lng": 0.0}


class Experience(BaseModel):
    """Professional experience entry"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    location: Optional[str] = None
    start_date: str = Field(..., description="Start date in ISO format")
    end_date: Optional[str] = Field(None, description="End date in ISO format, null if current")
    current: bool = False
    description: str = Field("", description="Job description")
    achievements: List[str] = Field(default_factory=list, description="Key achievements")
    skills: List[str] = Field(default_factory=list, description="Skills used in this role")
    normalized_title: Optional[str] = Field(None, description="Standardized job title")
    
    @field_validator("start_date", "end_date")
    @classmethod
    def validate_dates(cls, v):
        if v:
            try:
                datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                raise ValueError("Date must be in ISO format")
        return v
    
    @model_validator(mode="after")
    def validate_current_job(self):
        if self.current and self.end_date:
            raise ValueError("Current job cannot have end date")
        return self


class Education(BaseModel):
    """Education entry"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    institution: str = Field(..., description="Educational institution")
    degree: str = Field(..., description="Degree or qualification")
    field: str = Field(..., description="Field of study")
    start_date: Optional[str] = Field(None, description="Start date in ISO format")
    end_date: Optional[str] = Field(None, description="End date in ISO format")
    gpa: Optional[float] = Field(None, ge=0.0, le=4.0, description="GPA score")
    achievements: List[str] = Field(default_factory=list, description="Academic achievements")


class Skill(BaseModel):
    """Skill information"""
    name: str = Field(..., description="Skill name")
    category: str = Field("technical", description="Skill category")
    proficiency: Optional[int] = Field(None, ge=1, le=5, description="Proficiency level 1-5")
    years_of_experience: Optional[int] = Field(None, ge=0, description="Years of experience")
    endorsed: bool = Field(False, description="Whether skill is endorsed")
    last_used: Optional[str] = Field(None, description="Last used date in ISO format")


class Certification(BaseModel):
    """Professional certification"""
    name: str = Field(..., description="Certification name")
    issuer: str = Field(..., description="Issuing organization")
    issue_date: str = Field(..., description="Issue date in ISO format")
    expiry_date: Optional[str] = Field(None, description="Expiry date in ISO format")
    credential_id: Optional[str] = None
    url: Optional[str] = Field(None, description="Verification URL")


class Language(BaseModel):
    """Language proficiency"""
    name: str = Field(..., description="Language name")
    proficiency: str = Field(..., description="Proficiency level")
    
    @field_validator("proficiency")
    @classmethod
    def validate_proficiency(cls, v):
        allowed_levels = ["native", "fluent", "professional", "conversational", "basic"]
        if v.lower() not in allowed_levels:
            raise ValueError(f"Proficiency must be one of: {allowed_levels}")
        return v.lower()


class Publication(BaseModel):
    """Publication or paper"""
    title: str = Field(..., description="Publication title")
    publisher: Optional[str] = None
    date: Optional[str] = Field(None, description="Publication date in ISO format")
    url: Optional[str] = Field(None, description="Publication URL")
    description: Optional[str] = None


class Project(BaseModel):
    """Personal or professional project"""
    name: str = Field(..., description="Project name")
    description: str = Field(..., description="Project description")
    role: Optional[str] = Field(None, description="Role in project")
    start_date: Optional[str] = Field(None, description="Start date in ISO format")
    end_date: Optional[str] = Field(None, description="End date in ISO format")
    skills: List[str] = Field(default_factory=list, description="Skills used")
    url: Optional[str] = Field(None, description="Project URL")


class Award(BaseModel):
    """Award or recognition"""
    title: str = Field(..., description="Award title")
    issuer: Optional[str] = Field(None, description="Issuing organization")
    date: Optional[str] = Field(None, description="Award date in ISO format")
    description: Optional[str] = None


class ProfileData(BaseModel):
    """Core profile information"""
    # Personal Information
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    phone: Optional[str] = None
    location: Optional[Location] = None
    
    # Professional Summary
    summary: Optional[str] = Field(None, description="Professional summary")
    headline: Optional[str] = Field(None, description="Professional headline")
    
    # Professional Information
    experience: List[Experience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    skills: List[Skill] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)
    languages: List[Language] = Field(default_factory=list)
    
    # Additional Information
    publications: List[Publication] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    awards: List[Award] = Field(default_factory=list)


class ProfileEmbeddings(BaseModel):
    """Vector embeddings for profile sections"""
    overall: List[float] = Field(..., description="Overall profile embedding")
    skills: Optional[List[float]] = Field(None, description="Skills embedding")
    experience: Optional[List[float]] = Field(None, description="Experience embedding")
    education: Optional[List[float]] = Field(None, description="Education embedding")
    summary: Optional[List[float]] = Field(None, description="Summary embedding")


class ProfileQuality(BaseModel):
    """Profile quality metrics"""
    score: int = Field(..., ge=0, le=100, description="Quality score 0-100")
    missing_fields: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ProfileMetadata(BaseModel):
    """Profile metadata"""
    source: str = Field("upload", description="Profile source")
    original_filename: Optional[str] = None
    file_size: Optional[int] = None
    processing_version: str = Field("1.0", description="Processing version")
    processing_time: Optional[int] = Field(None, description="Processing time in milliseconds")
    last_updated: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    update_count: int = Field(0, description="Number of updates")
    quality: ProfileQuality = Field(default_factory=lambda: ProfileQuality(score=0))


class PrivacySettings(BaseModel):
    """Profile privacy settings"""
    consent_given: bool = Field(False, description="Data processing consent")
    consent_date: Optional[str] = Field(None, description="Consent date in ISO format")
    data_retention_date: Optional[str] = Field(None, description="Data retention expiry")
    gdpr_export_requested: bool = Field(False, description="GDPR export requested")
    deletion_requested: bool = Field(False, description="Deletion requested")


class ProfileAnalytics(BaseModel):
    """Profile analytics data"""
    view_count: int = Field(0, description="Number of views")
    search_appearances: int = Field(0, description="Search result appearances")
    last_viewed_at: Optional[str] = Field(None, description="Last viewed timestamp")
    match_score: Optional[float] = Field(None, description="Average match score")


class ProfileTable(AuditableModel, table=True):
    """
    Complete profile model with database persistence and vector support.
    
    Combines profile data, vector embeddings, and comprehensive metadata
    while maintaining all original validation and business logic.
    """
    __tablename__ = "profiles"
    
    # Tenant isolation field (each table model must define its own)
    tenant_id: UUID = Field(
        sa_column=create_tenant_id_column(),
        description="Tenant identifier for multi-tenant isolation"
    )
    
    # Additional indexes for performance
    __table_args__ = (
        Index("idx_profiles_tenant_status", "tenant_id", "status"),
        Index("idx_profiles_searchable_text", "searchable_text"),
        Index("idx_profiles_skills", "normalized_skills"),
        Index("idx_profiles_location", "location_city", "location_country"),
        Index("idx_profiles_experience_level", "experience_level"),
        Index("idx_profiles_processing_status", "processing_status"),
    )
    
    # Core profile identifiers
    university_id: Optional[UUID] = Field(
        default=None,
        sa_column=Column(PostgreSQLUUID(as_uuid=True), nullable=True, index=True),
        description="University identifier for academic profiles"
    )
    corporate_id: Optional[UUID] = Field(
        default=None,
        sa_column=Column(PostgreSQLUUID(as_uuid=True), nullable=True, index=True),
        description="Corporate identifier for company profiles"
    )
    
    # Profile status and classification
    status: ProfileStatus = Field(
        default=ProfileStatus.ACTIVE,
        sa_column=Column(String(20), nullable=False, default=ProfileStatus.ACTIVE, index=True),
        description="Profile status"
    )
    experience_level: Optional[ExperienceLevel] = Field(
        default=None,
        sa_column=Column(String(20), nullable=True, index=True),
        description="Overall experience level"
    )
    
    # Core profile data (stored as JSONB for flexibility)
    profile_data: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default={}),
        description="Complete profile data structure"
    )
    
    # Search and indexing fields
    searchable_text: str = Field(
        sa_column=Column(Text, nullable=False, default=""),
        description="Concatenated searchable content for full-text search"
    )
    keywords: List[str] = Field(
        default_factory=list,
        sa_column=Column(ARRAY(String), nullable=False, default=list),
        description="Extracted keywords for search optimization"
    )
    normalized_skills: List[str] = Field(
        default_factory=list,
        sa_column=Column(ARRAY(String), nullable=False, default=list),
        description="Standardized skills for matching"
    )
    
    # Contact information (denormalized for performance)
    name: str = Field(
        sa_column=Column(String(255), nullable=False, index=True),
        description="Full name"
    )
    email: str = Field(
        sa_column=Column(String(255), nullable=False, index=True),
        description="Primary email address"
    )
    phone: Optional[str] = Field(
        default=None,
        sa_column=Column(String(50), nullable=True),
        description="Primary phone number"
    )
    
    # Location fields (denormalized for geo search)
    location_city: Optional[str] = Field(
        default=None,
        sa_column=Column(String(100), nullable=True, index=True),
        description="City location"
    )
    location_state: Optional[str] = Field(
        default=None,
        sa_column=Column(String(100), nullable=True),
        description="State/province location"
    )
    location_country: Optional[str] = Field(
        default=None,
        sa_column=Column(String(100), nullable=True, index=True),
        description="Country location"
    )
    
    # Vector embeddings for different sections
    overall_embedding: Optional[List[float]] = Field(
        default=None,
        sa_column=Column(Vector(1536), nullable=True),
        description="Overall profile vector embedding"
    )
    skills_embedding: Optional[List[float]] = Field(
        default=None,
        sa_column=Column(Vector(1536), nullable=True),
        description="Skills-specific vector embedding"
    )
    experience_embedding: Optional[List[float]] = Field(
        default=None,
        sa_column=Column(Vector(1536), nullable=True),
        description="Experience-specific vector embedding"
    )
    summary_embedding: Optional[List[float]] = Field(
        default=None,
        sa_column=Column(Vector(1536), nullable=True),
        description="Summary-specific vector embedding"
    )
    
    # Processing and quality metadata
    processing_status: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING,
        sa_column=Column(String(20), nullable=False, default=ProcessingStatus.PENDING, index=True),
        description="Processing status"
    )
    processing_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default={}),
        description="Processing metadata and diagnostics"
    )
    quality_score: Optional[float] = Field(
        default=None,
        sa_column=Column(Float, nullable=True),
        description="Overall profile quality score (0-100)"
    )
    
    # Privacy and compliance
    privacy_settings: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default={}),
        description="Privacy and GDPR compliance settings"
    )
    consent_given: bool = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False, default=False),
        description="Data processing consent flag"
    )
    consent_date: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime, nullable=True),
        description="Date consent was given"
    )
    
    # Analytics and usage tracking
    view_count: int = Field(
        default=0,
        sa_column=Column(Integer, nullable=False, default=0),
        description="Number of profile views"
    )
    search_appearances: int = Field(
        default=0,
        sa_column=Column(Integer, nullable=False, default=0),
        description="Number of search result appearances"
    )
    last_viewed_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime, nullable=True),
        description="Last time profile was viewed"
    )
    last_activity_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime, nullable=True),
        description="Last profile activity timestamp"
    )
    
    # File and source metadata
    original_filename: Optional[str] = Field(
        default=None,
        sa_column=Column(String(255), nullable=True),
        description="Original uploaded filename"
    )
    file_size: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, nullable=True),
        description="Original file size in bytes"
    )
    source: str = Field(
        default="upload",
        sa_column=Column(String(50), nullable=False, default="upload"),
        description="Profile data source"
    )
    
    # Business logic methods
    @hybrid_property
    def is_active(self) -> bool:
        """Check if profile is active."""
        return self.status == ProfileStatus.ACTIVE and not self.is_deleted
    
    @hybrid_property
    def has_vector_embeddings(self) -> bool:
        """Check if profile has vector embeddings."""
        return self.overall_embedding is not None and len(self.overall_embedding) > 0
    
    def get_profile_data(self) -> ProfileData:
        """Convert JSONB profile data to ProfileData model."""
        return ProfileData.model_validate(self.profile_data)
    
    def set_profile_data(self, profile_data: ProfileData) -> None:
        """Set profile data from ProfileData model."""
        self.profile_data = profile_data.model_dump(exclude_none=True)
        
        # Update denormalized fields for performance
        self.name = profile_data.name
        self.email = profile_data.email
        self.phone = profile_data.phone
        
        if profile_data.location:
            self.location_city = profile_data.location.city
            self.location_state = profile_data.location.state
            self.location_country = profile_data.location.country
    
    def get_embeddings(self) -> ProfileEmbeddings:
        """Get embeddings as ProfileEmbeddings model."""
        return ProfileEmbeddings(
            overall=self.overall_embedding or [],
            skills=self.skills_embedding,
            experience=self.experience_embedding,
            summary=self.summary_embedding
        )
    
    def set_embeddings(self, embeddings: ProfileEmbeddings) -> None:
        """Set embeddings from ProfileEmbeddings model."""
        self.overall_embedding = embeddings.overall
        self.skills_embedding = embeddings.skills
        self.experience_embedding = embeddings.experience
        self.summary_embedding = embeddings.summary
    
    def get_processing_metadata(self) -> ProcessingMetadata:
        """Get processing metadata as ProcessingMetadata model."""
        return ProcessingMetadata.model_validate(self.processing_metadata)
    
    def set_processing_metadata(self, metadata: ProcessingMetadata) -> None:
        """Set processing metadata from ProcessingMetadata model."""
        self.processing_metadata = metadata.model_dump(exclude_none=True)
        self.processing_status = metadata.processing_status
        if metadata.quality_score is not None:
            self.quality_score = metadata.quality_score
    
    def get_privacy_settings(self) -> PrivacySettings:
        """Get privacy settings as PrivacySettings model."""
        return PrivacySettings.model_validate(self.privacy_settings)
    
    def set_privacy_settings(self, privacy: PrivacySettings) -> None:
        """Set privacy settings from PrivacySettings model."""
        self.privacy_settings = privacy.model_dump(exclude_none=True)
        self.consent_given = privacy.consent_given
        if privacy.consent_date:
            try:
                self.consent_date = datetime.fromisoformat(privacy.consent_date.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass
    
    def get_analytics(self) -> ProfileAnalytics:
        """Get analytics data as ProfileAnalytics model."""
        return ProfileAnalytics(
            view_count=self.view_count,
            search_appearances=self.search_appearances,
            last_viewed_at=self.last_viewed_at.isoformat() if self.last_viewed_at else None,
            match_score=None  # Calculated dynamically during search
        )
    
    def increment_view_count(self) -> None:
        """Increment profile view count and update last viewed timestamp."""
        self.view_count += 1
        self.last_viewed_at = datetime.utcnow()
        self.update_timestamp()
    
    def increment_search_appearances(self) -> None:
        """Increment search appearances count."""
        self.search_appearances += 1
        self.update_timestamp()
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity_at = datetime.utcnow()
        self.update_timestamp()
    
    def calculate_completeness_score(self) -> float:
        """Calculate profile completeness score based on available data."""
        if not self.profile_data:
            return 0.0
        
        try:
            profile_data = self.get_profile_data()
            score = 0.0
            total_fields = 10  # Number of important fields
            
            # Core fields (40% weight)
            if profile_data.name: score += 4
            if profile_data.email: score += 4
            if profile_data.summary: score += 4
            if profile_data.headline: score += 4
            
            # Experience (25% weight)
            if profile_data.experience: score += 25
            
            # Education (15% weight)
            if profile_data.education: score += 15
            
            # Skills (10% weight)
            if profile_data.skills: score += 10
            
            # Contact and location (10% weight)
            if profile_data.location: score += 5
            if profile_data.phone: score += 5
            
            return min(score, 100.0)
            
        except Exception:
            return 0.0
    
    # Relationship to tenant (required for SQLModel to create foreign key constraints)
    tenant: Optional["TenantTable"] = Relationship(back_populates="profiles")


# Keep original Profile model as alias for backward compatibility
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
ProfileModel = ProfileTable  # For SQLModel repositories