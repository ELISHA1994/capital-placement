"""Profile-related domain events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import DomainEvent
from app.domain.value_objects import ProfileId, TenantId, UserId


@dataclass
class ProfileCreatedEvent(DomainEvent):
    """Event fired when a new profile is created."""
    
    profile_id: ProfileId
    created_by_user_id: UserId
    source: str  # "upload", "manual", "api", etc.
    original_filename: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "profile_id": str(self.profile_id),
            "created_by_user_id": str(self.created_by_user_id),
            "source": self.source,
            "original_filename": self.original_filename,
        })
        return base


@dataclass
class ProfileUpdatedEvent(DomainEvent):
    """Event fired when a profile is updated."""
    
    profile_id: ProfileId
    updated_by_user_id: UserId
    changed_fields: List[str]
    previous_version: int
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "profile_id": str(self.profile_id),
            "updated_by_user_id": str(self.updated_by_user_id),
            "changed_fields": self.changed_fields,
            "previous_version": self.previous_version,
        })
        return base


@dataclass
class ProfileDeletedEvent(DomainEvent):
    """Event fired when a profile is deleted."""
    
    profile_id: ProfileId
    deleted_by_user_id: UserId
    deletion_reason: Optional[str] = None
    soft_delete: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "profile_id": str(self.profile_id),
            "deleted_by_user_id": str(self.deleted_by_user_id),
            "deletion_reason": self.deletion_reason,
            "soft_delete": self.soft_delete,
        })
        return base


@dataclass
class ProfileViewedEvent(DomainEvent):
    """Event fired when a profile is viewed."""
    
    profile_id: ProfileId
    viewed_by_user_id: UserId
    view_context: str  # "search_result", "direct_link", "recommendation", etc.
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "profile_id": str(self.profile_id),
            "viewed_by_user_id": str(self.viewed_by_user_id),
            "view_context": self.view_context,
        })
        return base


@dataclass
class ProfileSearchedEvent(DomainEvent):
    """Event fired when profiles are searched."""
    
    searched_by_user_id: UserId
    search_query: str
    search_type: str  # "text", "vector", "hybrid", "skills"
    filters_applied: Dict[str, Any]
    results_count: int
    search_duration_ms: int
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "searched_by_user_id": str(self.searched_by_user_id),
            "search_query": self.search_query,
            "search_type": self.search_type,
            "filters_applied": self.filters_applied,
            "results_count": self.results_count,
            "search_duration_ms": self.search_duration_ms,
        })
        return base


@dataclass
class ProfileProcessingCompletedEvent(DomainEvent):
    """Event fired when profile processing is completed successfully."""
    
    profile_id: ProfileId
    processing_duration_ms: int
    quality_score: Optional[float] = None
    embeddings_generated: bool = False
    skills_extracted: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "profile_id": str(self.profile_id),
            "processing_duration_ms": self.processing_duration_ms,
            "quality_score": self.quality_score,
            "embeddings_generated": self.embeddings_generated,
            "skills_extracted": self.skills_extracted,
        })
        return base


@dataclass
class ProfileProcessingFailedEvent(DomainEvent):
    """Event fired when profile processing fails."""
    
    profile_id: ProfileId
    error_message: str
    error_code: Optional[str] = None
    processing_duration_ms: Optional[int] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "profile_id": str(self.profile_id),
            "error_message": self.error_message,
            "error_code": self.error_code,
            "processing_duration_ms": self.processing_duration_ms,
            "retry_count": self.retry_count,
        })
        return base


@dataclass
class ProfileMatchedEvent(DomainEvent):
    """Event fired when a profile is matched against job requirements."""
    
    profile_id: ProfileId
    matched_by_user_id: UserId
    match_score: float
    job_requirements: Dict[str, Any]
    matching_skills: List[str]
    missing_skills: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "profile_id": str(self.profile_id),
            "matched_by_user_id": str(self.matched_by_user_id),
            "match_score": self.match_score,
            "job_requirements": self.job_requirements,
            "matching_skills": self.matching_skills,
            "missing_skills": self.missing_skills,
        })
        return base


@dataclass
class ProfileEmbeddingsGeneratedEvent(DomainEvent):
    """Event fired when AI embeddings are generated for a profile."""
    
    profile_id: ProfileId
    embedding_model: str
    embedding_dimensions: int
    processing_time_ms: int
    sections_embedded: List[str]  # "overall", "skills", "experience", etc.
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "profile_id": str(self.profile_id),
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "processing_time_ms": self.processing_time_ms,
            "sections_embedded": self.sections_embedded,
        })
        return base


__all__ = [
    "ProfileCreatedEvent",
    "ProfileUpdatedEvent",
    "ProfileDeletedEvent",
    "ProfileViewedEvent",
    "ProfileSearchedEvent",
    "ProfileProcessingCompletedEvent",
    "ProfileProcessingFailedEvent",
    "ProfileMatchedEvent",
    "ProfileEmbeddingsGeneratedEvent",
]