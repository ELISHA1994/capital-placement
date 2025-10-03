"""Domain value objects used across aggregates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID


def _coerce_uuid(value: Any, *, field_name: str) -> UUID:
    """Convert strings to UUID instances while validating type."""
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        return UUID(value)
    raise TypeError(f"{field_name} must be a UUID-compatible value")


@dataclass(frozen=True)
class TenantId:
    """Strongly-typed tenant identifier used for multi-tenant isolation."""

    value: UUID

    def __init__(self, value: Any):
        object.__setattr__(self, "value", _coerce_uuid(value, field_name="tenant_id"))

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class ProfileId:
    """Aggregate identifier for Profile domain entities."""

    value: UUID

    def __init__(self, value: Any):
        object.__setattr__(self, "value", _coerce_uuid(value, field_name="profile_id"))

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class JobId:
    """Aggregate identifier for Job domain entities."""

    value: UUID

    def __init__(self, value: Any):
        object.__setattr__(self, "value", _coerce_uuid(value, field_name="job_id"))

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class UserId:
    """Aggregate identifier for User domain entities."""

    value: UUID

    def __init__(self, value: Any):
        object.__setattr__(self, "value", _coerce_uuid(value, field_name="user_id"))

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class DocumentId:
    """Aggregate identifier for Document domain entities."""

    value: UUID

    def __init__(self, value: Any):
        object.__setattr__(self, "value", _coerce_uuid(value, field_name="document_id"))

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class MatchId:
    """Aggregate identifier for Match domain entities."""

    value: UUID

    def __init__(self, value: Any):
        object.__setattr__(self, "value", _coerce_uuid(value, field_name="match_id"))

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class EmbeddingVector:
    """Value object representing a vector embedding."""

    dimensions: int
    values: list[float]

    def __post_init__(self):
        if len(self.values) != self.dimensions:
            raise ValueError(f"Vector must have exactly {self.dimensions} dimensions")
        if not all(isinstance(v, (int, float)) for v in self.values):
            raise ValueError("All vector values must be numeric")


@dataclass(frozen=True)
class MatchScore:
    """Value object representing a match score between entities."""

    value: float
    
    def __post_init__(self):
        if not 0.0 <= self.value <= 1.0:
            raise ValueError("Match score must be between 0.0 and 1.0")


@dataclass(frozen=True)
class SkillName:
    """Value object for skill names with normalization."""

    value: str
    normalized: str

    def __init__(self, value: str):
        if not value or not value.strip():
            raise ValueError("Skill name cannot be empty")
        
        normalized = value.strip().lower()
        object.__setattr__(self, "value", value.strip())
        object.__setattr__(self, "normalized", normalized)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class EmailAddress:
    """Value object for email addresses with validation."""

    value: str

    def __init__(self, value: str):
        if not value or "@" not in value:
            raise ValueError("Invalid email address format")
        
        object.__setattr__(self, "value", value.lower().strip())

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class PhoneNumber:
    """Value object for phone numbers with basic validation."""

    value: str

    def __init__(self, value: str):
        if not value or len(value.strip()) < 10:
            raise ValueError("Phone number must be at least 10 characters")
        
        object.__setattr__(self, "value", value.strip())

    def __str__(self) -> str:
        return self.value


__all__ = [
    "TenantId", 
    "ProfileId", 
    "JobId", 
    "UserId", 
    "DocumentId", 
    "MatchId",
    "EmbeddingVector",
    "MatchScore",
    "SkillName",
    "EmailAddress",
    "PhoneNumber"
]