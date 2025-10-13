"""
Search Suggestion Schemas

API response models for search suggestion endpoint.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SuggestionResponse(BaseModel):
    """Single search suggestion"""

    text: str = Field(..., description="Suggestion text")
    source: str = Field(..., description="Source of suggestion")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    metadata: Optional[dict] = Field(None, description="Additional metadata")


class SuggestionsResponse(BaseModel):
    """Response for suggestion endpoint"""

    suggestions: List[SuggestionResponse] = Field(
        default_factory=list,
        description="List of suggestions"
    )
    query: str = Field(..., description="Query prefix")
    count: int = Field(..., description="Number of suggestions returned")
    cached: bool = Field(False, description="Whether result was cached")