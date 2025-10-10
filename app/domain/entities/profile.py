"""Pure domain representation of profile aggregates."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from app.domain.value_objects import (
    ProfileId, 
    TenantId, 
    EmbeddingVector, 
    MatchScore, 
    SkillName,
    EmailAddress,
    PhoneNumber
)


class ProfileStatus(str, Enum):
    """Lifecycle status for a profile."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    DRAFT = "draft"
    ARCHIVED = "archived"
    DELETED = "deleted"


class ExperienceLevel(str, Enum):
    """Experience seniority classifications used for matching."""

    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"
    EXECUTIVE = "executive"


class ProcessingStatus(str, Enum):
    """Processing workflow statuses for ingestion pipelines."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


@dataclass
class ProcessingMetadata:
    """Domain view of ingestion pipeline telemetry."""

    status: ProcessingStatus = ProcessingStatus.PENDING
    version: str = "1.0"
    time_ms: Optional[int] = None
    last_processed: Optional[datetime] = None
    error_message: Optional[str] = None
    extraction_method: Optional[str] = None
    quality_score: Optional[float] = None
    confidence_score: Optional[float] = None
    pages_processed: Optional[int] = None
    additional: Dict[str, Any] = field(default_factory=dict)

    def mark_processing_started(self) -> None:
        """Mark processing as started."""
        self.status = ProcessingStatus.PROCESSING
        self.last_processed = datetime.utcnow()

    def mark_processing_completed(self, quality_score: Optional[float] = None) -> None:
        """Mark processing as completed successfully."""
        self.status = ProcessingStatus.COMPLETED
        self.last_processed = datetime.utcnow()
        if quality_score is not None:
            self.quality_score = quality_score

    def mark_processing_failed(self, error_message: str) -> None:
        """Mark processing as failed with error details."""
        self.status = ProcessingStatus.FAILED
        self.error_message = error_message
        self.last_processed = datetime.utcnow()

    def mark_processing_cancelled(self, reason: Optional[str] = None) -> None:
        """Mark processing as cancelled."""
        self.status = ProcessingStatus.CANCELLED
        self.error_message = reason or "Processing cancelled by user"
        self.last_processed = datetime.utcnow()


@dataclass
class PrivacySettings:
    """Consent and privacy preferences captured for a profile."""

    consent_given: bool = False
    consent_date: Optional[datetime] = None
    data_retention_date: Optional[datetime] = None
    gdpr_export_requested: bool = False
    deletion_requested: bool = False

    def give_consent(self) -> None:
        """Give data processing consent."""
        self.consent_given = True
        self.consent_date = datetime.utcnow()

    def request_deletion(self) -> None:
        """Request account deletion."""
        self.deletion_requested = True

    def request_gdpr_export(self) -> None:
        """Request GDPR data export."""
        self.gdpr_export_requested = True


@dataclass
class Location:
    """Location metadata for a profile."""

    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    coordinates: Optional[tuple[float, float]] = None

    def is_complete(self) -> bool:
        """Check if location has sufficient information."""
        return bool(self.city and self.country)


@dataclass
class ProfileAnalytics:
    """Observation metrics for profile engagement."""

    view_count: int = 0
    search_appearances: int = 0
    last_viewed_at: Optional[datetime] = None
    match_score: Optional[float] = None

    def record_view(self) -> None:
        """Record a profile view."""
        self.view_count += 1
        self.last_viewed_at = datetime.utcnow()

    def record_search_appearance(self) -> None:
        """Record appearance in search results."""
        self.search_appearances += 1


@dataclass
class Experience:
    """Professional experience entry."""
    
    title: str
    company: str
    start_date: str
    description: str = ""
    end_date: Optional[str] = None
    current: bool = False
    location: Optional[str] = None
    achievements: List[str] = field(default_factory=list)
    skills: List[SkillName] = field(default_factory=list)

    def is_current_role(self) -> bool:
        """Check if this is a current role."""
        return self.current and self.end_date is None

    def duration_years(self) -> Optional[float]:
        """Calculate duration in years if possible."""
        try:
            start = datetime.fromisoformat(self.start_date.replace("Z", "+00:00"))
            end = datetime.fromisoformat(self.end_date.replace("Z", "+00:00")) if self.end_date else datetime.utcnow()
            return (end - start).days / 365.25
        except (ValueError, AttributeError):
            return None


@dataclass
class Education:
    """Education entry."""
    
    institution: str
    degree: str
    field: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    gpa: Optional[float] = None
    achievements: List[str] = field(default_factory=list)

    def is_completed(self) -> bool:
        """Check if education is completed."""
        return self.end_date is not None


@dataclass
class Skill:
    """Skill information with proficiency and experience."""
    
    name: SkillName
    category: str = "technical"
    proficiency: Optional[int] = None  # 1-5 scale
    years_of_experience: Optional[int] = None
    endorsed: bool = False
    last_used: Optional[str] = None

    def is_expert_level(self) -> bool:
        """Check if skill is at expert level."""
        return (self.proficiency and self.proficiency >= 4) or (self.years_of_experience and self.years_of_experience >= 5)


@dataclass
class ProfileEmbeddings:
    """Vector embeddings for profile sections."""
    
    overall: Optional[EmbeddingVector] = None
    skills: Optional[EmbeddingVector] = None
    experience: Optional[EmbeddingVector] = None
    education: Optional[EmbeddingVector] = None
    summary: Optional[EmbeddingVector] = None

    def has_complete_embeddings(self) -> bool:
        """Check if all essential embeddings are present."""
        return self.overall is not None and self.skills is not None


@dataclass
class ProfileData:
    """Core profile information structured for domain logic."""
    
    name: str
    email: EmailAddress
    phone: Optional[PhoneNumber] = None
    location: Optional[Location] = None
    summary: Optional[str] = None
    headline: Optional[str] = None
    experience: List[Experience] = field(default_factory=list)
    education: List[Education] = field(default_factory=list)
    skills: List[Skill] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)

    def total_experience_years(self) -> float:
        """Calculate total years of professional experience."""
        total = 0.0
        for exp in self.experience:
            duration = exp.duration_years()
            if duration:
                total += duration
        return total

    def get_skill_names(self) -> List[str]:
        """Extract skill names for indexing."""
        return [skill.name.normalized for skill in self.skills]

    def get_companies(self) -> List[str]:
        """Extract company names from experience."""
        return [exp.company for exp in self.experience]

    def has_required_fields(self) -> bool:
        """Check if profile has minimum required fields."""
        return bool(self.name and self.email and (self.summary or self.experience))


@dataclass
class Profile:
    """Aggregate root representing a candidate profile."""

    id: ProfileId
    tenant_id: TenantId
    status: ProfileStatus
    profile_data: ProfileData
    searchable_text: str = ""
    normalized_skills: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    experience_level: Optional[ExperienceLevel] = None
    embeddings: ProfileEmbeddings = field(default_factory=ProfileEmbeddings)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing: ProcessingMetadata = field(default_factory=ProcessingMetadata)
    privacy: PrivacySettings = field(default_factory=PrivacySettings)
    analytics: ProfileAnalytics = field(default_factory=ProfileAnalytics)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_activity_at: Optional[datetime] = None

    def __post_init__(self):
        """Initialize computed fields after creation."""
        self._update_computed_fields()

    def _update_computed_fields(self) -> None:
        """Update computed fields based on profile data."""
        self.normalized_skills = self.profile_data.get_skill_names()
        self.experience_level = self._calculate_experience_level()
        self.searchable_text = self._build_searchable_text()

    def _calculate_experience_level(self) -> ExperienceLevel:
        """Calculate experience level based on years of experience."""
        total_years = self.profile_data.total_experience_years()
        
        if total_years >= 15:
            return ExperienceLevel.EXECUTIVE
        elif total_years >= 10:
            return ExperienceLevel.PRINCIPAL
        elif total_years >= 7:
            return ExperienceLevel.LEAD
        elif total_years >= 4:
            return ExperienceLevel.SENIOR
        elif total_years >= 2:
            return ExperienceLevel.MID
        elif total_years >= 1:
            return ExperienceLevel.JUNIOR
        else:
            return ExperienceLevel.ENTRY

    def _build_searchable_text(self) -> str:
        """Build searchable text from all profile content."""
        parts = [
            self.profile_data.name,
            str(self.profile_data.email),
            self.profile_data.headline or "",
            self.profile_data.summary or "",
        ]
        
        # Add experience content
        for exp in self.profile_data.experience:
            parts.extend([exp.title, exp.company, exp.description])
            parts.extend(exp.achievements)
        
        # Add education content
        for edu in self.profile_data.education:
            parts.extend([edu.institution, edu.degree, edu.field])
        
        # Add skills
        parts.extend(skill.name.value for skill in self.profile_data.skills)
        
        return " ".join(filter(None, parts))

    def calculate_match_score(self, job_requirements: List[str]) -> MatchScore:
        """Calculate match score against job requirements."""
        if not job_requirements or not self.normalized_skills:
            return MatchScore(0.0)
        
        job_skills_lower = [req.lower() for req in job_requirements]
        matching_skills = sum(1 for skill in self.normalized_skills if skill in job_skills_lower)
        
        # Simple skill-based matching (can be enhanced with ML models)
        score = min(matching_skills / len(job_requirements), 1.0)
        return MatchScore(score)

    def mark_deleted(self) -> None:
        """Soft-delete the profile in the domain model."""
        self.status = ProfileStatus.DELETED
        self.updated_at = datetime.utcnow()

    def request_deletion(self) -> None:
        """Request profile deletion (GDPR compliance)."""
        self.privacy.request_deletion()
        self.updated_at = datetime.utcnow()

    def soft_delete(self, reason: str | None = None) -> None:
        """Soft delete profile - marks as deleted but preserves data.

        Args:
            reason: Optional reason for deletion (stored in metadata)
        """
        if self.status == ProfileStatus.DELETED:
            raise ValueError("Profile is already deleted")

        self.mark_deleted()
        self.privacy.request_deletion()

        if reason:
            self.metadata["deletion_reason"] = reason
            self.metadata["deleted_at"] = datetime.utcnow().isoformat()

    def validate_can_delete(self) -> list[str]:
        """Validate if profile can be deleted, return list of issues.

        Returns:
            List of validation issues preventing deletion. Empty list means deletion is allowed.
        """
        issues = []

        # Check if already deleted
        if self.status == ProfileStatus.DELETED:
            issues.append("Profile is already deleted")

        # Check if processing is in progress
        if self.processing.status == ProcessingStatus.PROCESSING:
            issues.append("Cannot delete profile while processing is in progress")

        return issues

    def activate(self) -> None:
        """Activate the profile."""
        if self.status == ProfileStatus.DELETED:
            raise ValueError("Cannot activate deleted profile")
        
        self.status = ProfileStatus.ACTIVE
        self.updated_at = datetime.utcnow()

    def archive(self) -> None:
        """Archive the profile."""
        self.status = ProfileStatus.ARCHIVED
        self.updated_at = datetime.utcnow()

    def update_profile_data(self, new_data: ProfileData) -> None:
        """Update profile data and recompute derived fields."""
        self.profile_data = new_data
        self.updated_at = datetime.utcnow()
        self.last_activity_at = datetime.utcnow()
        self._update_computed_fields()

    def set_embeddings(self, embeddings: ProfileEmbeddings) -> None:
        """Set vector embeddings for the profile."""
        self.embeddings = embeddings
        self.updated_at = datetime.utcnow()

    def record_view(self) -> None:
        """Record a profile view."""
        self.analytics.record_view()
        self.last_activity_at = datetime.utcnow()

    def record_search_appearance(self) -> None:
        """Record appearance in search results."""
        self.analytics.record_search_appearance()

    def is_searchable(self) -> bool:
        """Check if profile is in a searchable state."""
        return (
            self.status == ProfileStatus.ACTIVE and
            self.processing.status == ProcessingStatus.COMPLETED and
            self.privacy.consent_given and
            not self.privacy.deletion_requested
        )

    def calculate_completeness_score(self) -> float:
        """Calculate profile completeness score (0-100)."""
        score = 0.0
        
        # Essential fields (50%)
        if self.profile_data.name:
            score += 10
        if self.profile_data.email:
            score += 10
        if self.profile_data.summary:
            score += 15
        if self.profile_data.experience:
            score += 15
        
        # Important fields (30%)
        if self.profile_data.skills:
            score += 10
        if self.profile_data.education:
            score += 10
        if self.profile_data.location and self.profile_data.location.is_complete():
            score += 10
        
        # Additional fields (20%)
        if self.profile_data.phone:
            score += 5
        if self.profile_data.headline:
            score += 5
        if self.profile_data.languages:
            score += 5
        if self.embeddings.has_complete_embeddings():
            score += 5
        
        return min(score, 100.0)

    def get_quality_issues(self) -> List[str]:
        """Get list of quality issues that should be addressed."""
        issues = []
        
        if not self.profile_data.has_required_fields():
            issues.append("Missing required fields (name, email, summary/experience)")
        
        if not self.profile_data.skills:
            issues.append("No skills listed")
        
        if not self.profile_data.experience:
            issues.append("No professional experience")
        
        if not self.embeddings.has_complete_embeddings():
            issues.append("Missing vector embeddings")
        
        if self.processing.status != ProcessingStatus.COMPLETED:
            issues.append("Processing not completed")
        
        return issues


__all__ = [
    "ExperienceLevel",
    "ProcessingMetadata",
    "ProcessingStatus",
    "PrivacySettings",
    "Profile",
    "ProfileData",
    "ProfileEmbeddings",
    "ProfileAnalytics",
    "ProfileStatus",
    "Location",
    "Experience",
    "Education",
    "Skill",
]