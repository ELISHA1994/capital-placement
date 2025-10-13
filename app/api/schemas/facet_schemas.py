"""API response schemas for facet metadata."""

from typing import List, Optional
from pydantic import BaseModel, Field


class FacetValueResponse(BaseModel):
    """Individual facet value in API response."""

    value: str = Field(..., description="Facet value")
    count: int = Field(..., ge=0, description="Number of candidates")
    display_name: str = Field(..., description="Human-readable name")
    percentage: Optional[float] = Field(None, description="Percentage of total candidates")


class RangeBucketResponse(BaseModel):
    """Range bucket for numerical facets."""

    label: str = Field(..., description="Bucket label")
    min_value: Optional[float] = Field(None, description="Minimum value (inclusive)")
    max_value: Optional[float] = Field(None, description="Maximum value (inclusive)")
    count: int = Field(..., ge=0, description="Number of candidates in bucket")
    percentage: Optional[float] = Field(None, description="Percentage of total")


class FacetFieldResponse(BaseModel):
    """Complete facet field configuration."""

    field_name: str = Field(..., description="Field identifier")
    facet_type: str = Field(..., description="Facet type (terms, range, etc.)")
    display_name: str = Field(..., description="Display name")
    description: Optional[str] = Field(None, description="Field description")

    # For term facets
    values: List[FacetValueResponse] = Field(
        default_factory=list,
        description="Available values (for term facets)"
    )

    # For range facets
    buckets: List[RangeBucketResponse] = Field(
        default_factory=list,
        description="Range buckets (for range facets)"
    )
    min_value: Optional[float] = Field(None, description="Minimum value")
    max_value: Optional[float] = Field(None, description="Maximum value")

    # Behavior
    searchable: bool = Field(False, description="Can users search within values")
    multi_select: bool = Field(True, description="Can select multiple values")

    # Metadata
    total_count: int = Field(0, description="Total candidates with this field")
    unique_count: int = Field(0, description="Number of unique values")


class FacetMetadataResponse(BaseModel):
    """Complete facet metadata response."""

    facets: List[FacetFieldResponse] = Field(..., description="Available facet fields")
    total_profiles: int = Field(..., description="Total profiles in tenant")
    active_profiles: int = Field(..., description="Active profiles in tenant")
    generated_at: str = Field(..., description="When facets were generated")
    cache_hit: bool = Field(False, description="Whether result was cached")

    class Config:
        json_schema_extra = {
            "example": {
                "facets": [
                    {
                        "field_name": "skills",
                        "facet_type": "terms",
                        "display_name": "Skills & Technologies",
                        "values": [
                            {"value": "Python", "count": 1250, "display_name": "Python", "percentage": 62.5},
                            {"value": "JavaScript", "count": 980, "display_name": "JavaScript", "percentage": 49.0}
                        ],
                        "searchable": True,
                        "multi_select": True,
                        "total_count": 2000,
                        "unique_count": 250
                    }
                ],
                "total_profiles": 2000,
                "active_profiles": 1850,
                "generated_at": "2025-10-13T10:30:00Z",
                "cache_hit": True
            }
        }
