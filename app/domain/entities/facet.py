"""
Domain entities for facet metadata and filtering.

This module defines the core domain model for the dynamic faceting system,
which analyzes tenant profile data to provide rich filtering capabilities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

from app.domain.value_objects import TenantId


class FacetType(str, Enum):
    """Types of facets supported in the system."""

    TERMS = "terms"  # Categorical: skills, locations, etc.
    RANGE = "range"  # Numerical: experience years, salary
    DATE_RANGE = "date_range"  # Date ranges: last active, created date
    HIERARCHICAL = "hierarchical"  # Nested: location > city
    BOOLEAN = "boolean"  # Binary: remote_willing, has_degree


class FacetFieldName(str, Enum):
    """Standardized facet field names mapped to profile schema."""

    # Skills and Technologies
    SKILLS = "skills"
    SKILL_CATEGORIES = "skill_categories"

    # Experience
    EXPERIENCE_LEVEL = "experience_level"
    TOTAL_EXPERIENCE_YEARS = "total_experience_years"
    COMPANIES = "companies"
    JOB_TITLES = "job_titles"
    INDUSTRIES = "industries"

    # Education
    EDUCATION_LEVEL = "education_level"
    DEGREE_TYPES = "degree_types"
    INSTITUTIONS = "institutions"
    FIELDS_OF_STUDY = "fields_of_study"

    # Location
    LOCATION_COUNTRY = "location_country"
    LOCATION_STATE = "location_state"
    LOCATION_CITY = "location_city"

    # Languages
    LANGUAGES = "languages"

    # Certifications
    CERTIFICATIONS = "certifications"

    # Availability
    STATUS = "status"
    LAST_ACTIVITY = "last_activity"


@dataclass
class FacetValue:
    """Individual facet value with count and metadata."""

    value: str
    count: int
    display_name: Optional[str] = None
    percentage: Optional[float] = None  # % of total candidates
    metadata: Dict[str, any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure display name defaults to value."""
        if self.display_name is None:
            self.display_name = self.value


@dataclass
class RangeBucket:
    """Range bucket for numerical facets."""

    label: str
    min_value: Optional[Union[int, float]]
    max_value: Optional[Union[int, float]]
    count: int
    percentage: Optional[float] = None

    def contains(self, value: Union[int, float]) -> bool:
        """Check if value falls within this bucket."""
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True


@dataclass
class FacetField:
    """Complete facet field configuration with values."""

    field_name: FacetFieldName
    facet_type: FacetType
    display_name: str
    description: Optional[str] = None

    # For TERMS facets
    values: List[FacetValue] = field(default_factory=list)

    # For RANGE facets
    buckets: List[RangeBucket] = field(default_factory=list)
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None

    # Facet behavior
    searchable: bool = False  # Can users search within facet values?
    multi_select: bool = True  # Can select multiple values?
    sortable: bool = True

    # Metadata
    total_count: int = 0  # Total candidates with this field populated
    unique_count: int = 0  # Number of unique values

    def get_top_values(self, limit: int = 10) -> List[FacetValue]:
        """Get top N most common values."""
        return sorted(self.values, key=lambda v: v.count, reverse=True)[:limit]

    def get_value_by_name(self, value: str) -> Optional[FacetValue]:
        """Find specific facet value by name."""
        for v in self.values:
            if v.value == value:
                return v
        return None


@dataclass
class FacetMetadata:
    """
    Aggregate root for facet metadata.

    Represents all available facets for a tenant's candidate pool.
    """

    tenant_id: TenantId
    facet_fields: List[FacetField]

    total_profiles: int
    active_profiles: int

    generated_at: datetime = field(default_factory=datetime.utcnow)
    cache_key: Optional[str] = None
    metadata: Dict[str, any] = field(default_factory=dict)

    def get_facet(self, field_name: FacetFieldName) -> Optional[FacetField]:
        """Get specific facet field by name."""
        for facet in self.facet_fields:
            if facet.field_name == field_name:
                return facet
        return None

    def get_facets_by_type(self, facet_type: FacetType) -> List[FacetField]:
        """Get all facets of a specific type."""
        return [f for f in self.facet_fields if f.facet_type == facet_type]

    def is_stale(self, max_age_seconds: int = 3600) -> bool:
        """Check if facet metadata needs refresh."""
        age = (datetime.utcnow() - self.generated_at).total_seconds()
        return age > max_age_seconds