"""Shared schemas for document processing infrastructure."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


@dataclass
class ExtractedSection:
    """Represents an extracted document section."""

    section_type: str
    title: str
    content: str
    confidence: float
    metadata: Dict[str, Any]
    start_position: int
    end_position: int


@dataclass
class StructuredContent:
    """Represents structured content extracted from a document."""

    document_type: str
    sections: List[ExtractedSection]
    summary: str
    key_information: Dict[str, Any]
    quality_assessment: Dict[str, Any]
    processing_metadata: Dict[str, Any]


class CVLocationModel(BaseModel):
    """Structured location details for personal information."""

    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None


class CVSkillModel(BaseModel):
    """Structured representation of a candidate skill."""

    name: str
    category: Optional[str] = None
    proficiency: Optional[str] = None
    years_of_experience: Optional[float] = None
    endorsed: Optional[bool] = None
    last_used: Optional[str] = None


class CVExperienceModel(BaseModel):
    """Structured experience entry."""

    title: str
    company: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    current: Optional[bool] = None
    location: Optional[str] = None
    description: Optional[str] = None
    achievements: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)


class CVEducationModel(BaseModel):
    """Structured education entry."""

    institution: Optional[str] = None
    degree: Optional[str] = None
    field: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    gpa: Optional[Union[str, float]] = None
    achievements: List[str] = Field(default_factory=list)


class CVPersonalInfoModel(BaseModel):
    """Structured personal information."""

    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[Union[str, CVLocationModel]] = None
    headline: Optional[str] = None
    linkedin: Optional[str] = None
    website: Optional[str] = None


class CVAnalysisModel(BaseModel):
    """Pydantic schema describing expected CV analysis output."""

    summary: Optional[str] = None
    personal_info: Optional[CVPersonalInfoModel] = None
    experience: List[CVExperienceModel] = Field(default_factory=list)
    education: List[CVEducationModel] = Field(default_factory=list)
    skills: List[Union[CVSkillModel, str]] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    seniority_level: Optional[str] = None
    industries: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    contact: Optional[Dict[str, Any]] = None
    structured_data: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


@dataclass
class DocumentEmbedding:
    """Represents a document-level embedding."""

    document_id: str
    document_type: str
    embedding_vector: List[float]
    content_hash: str
    metadata: Dict[str, Any]
    created_at: datetime


@dataclass
class SectionEmbedding:
    """Represents a section-level embedding."""

    section_id: str
    document_id: str
    section_type: str
    title: str
    embedding_vector: List[float]
    content_hash: str
    metadata: Dict[str, Any]
    created_at: datetime


@dataclass
class EmbeddingResult:
    """Complete embedding generation result."""

    document_embedding: DocumentEmbedding
    section_embeddings: List[SectionEmbedding]
    processing_info: Dict[str, Any]
    semantic_relationships: List[Dict[str, Any]]


__all__ = [
    "ExtractedSection",
    "StructuredContent",
    "CVLocationModel",
    "CVSkillModel",
    "CVExperienceModel",
    "CVEducationModel",
    "CVPersonalInfoModel",
    "CVAnalysisModel",
    "DocumentEmbedding",
    "SectionEmbedding",
    "EmbeddingResult",
]
