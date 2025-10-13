"""
Search API Schemas

Comprehensive search request/response models for the CV matching system:
- Multi-stage search with vector, hybrid, and reranking capabilities
- Query expansion and international terminology handling
- Faceted search with filters and aggregations
- Search analytics and performance tracking
- Multi-tenant search isolation
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from uuid import UUID

from pydantic import Field, field_validator, computed_field, ConfigDict, BaseModel


class SearchMode(str, Enum):
    """Search execution modes"""
    KEYWORD = "keyword"          # Traditional keyword/text search
    VECTOR = "vector"            # Pure vector similarity search
    HYBRID = "hybrid"            # Combination of keyword + vector
    SEMANTIC = "semantic"        # Semantic understanding with Azure Cognitive Search
    MULTI_STAGE = "multi_stage"  # Multi-stage: vector → rerank → business logic


class SortOrder(str, Enum):
    """Sort order options"""
    RELEVANCE = "relevance"      # Search relevance score
    DATE_DESC = "date_desc"      # Most recent first
    DATE_ASC = "date_asc"        # Oldest first
    EXPERIENCE_DESC = "experience_desc"  # Most experienced first
    EXPERIENCE_ASC = "experience_asc"    # Least experienced first
    NAME_ASC = "name_asc"        # Alphabetical by name
    SALARY_DESC = "salary_desc"  # Highest salary expectation first
    SALARY_ASC = "salary_asc"    # Lowest salary expectation first


class FilterOperator(str, Enum):
    """Filter operation types"""
    EQUALS = "eq"                # Exact match
    NOT_EQUALS = "ne"            # Not equal
    GREATER_THAN = "gt"          # Greater than
    GREATER_EQUAL = "ge"         # Greater than or equal
    LESS_THAN = "lt"             # Less than
    LESS_EQUAL = "le"            # Less than or equal
    IN = "in"                    # Value in list
    NOT_IN = "nin"               # Value not in list
    CONTAINS = "contains"        # String contains
    STARTS_WITH = "starts_with"  # String starts with
    REGEX = "regex"              # Regular expression match


class PaginationModel(BaseModel):
    """Pagination parameters"""
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")


class SearchFilter(BaseModel):
    """Individual search filter with operator and value"""

    field: str = Field(..., description="Field name to filter on")
    operator: FilterOperator = Field(default=FilterOperator.EQUALS, description="Filter operator")
    value: Union[str, int, float, bool, List[Any]] = Field(..., description="Filter value(s)")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"field": "skills", "operator": "contains", "value": "Python"},
                {"field": "total_experience_years", "operator": "ge", "value": 5},
                {"field": "current_location", "operator": "in", "value": ["San Francisco", "New York"]}
            ]
        }
    )


class RangeFilter(BaseModel):
    """Range filter for numeric and date values"""

    field: str = Field(..., description="Field name to filter on")
    min_value: Optional[Union[int, float, date, datetime]] = Field(None, description="Minimum value (inclusive)")
    max_value: Optional[Union[int, float, date, datetime]] = Field(None, description="Maximum value (inclusive)")

    @field_validator('max_value')
    @classmethod
    def validate_range(cls, v, info):
        """Ensure max_value >= min_value"""
        if v is not None and 'min_value' in info.data:
            min_val = info.data['min_value']
            if min_val is not None and v < min_val:
                raise ValueError('max_value must be >= min_value')
        return v


class LocationFilter(BaseModel):
    """Geographic location filter with radius search"""

    center_location: str = Field(..., description="Center location (city, country)")
    radius_km: Optional[float] = Field(None, ge=0, description="Search radius in kilometers")
    include_remote: bool = Field(default=True, description="Include remote-capable candidates")
    preferred_locations: List[str] = Field(default_factory=list, description="Preferred specific locations")


class SalaryFilter(BaseModel):
    """Salary expectation filter"""

    min_salary: Optional[Decimal] = Field(None, ge=0, description="Minimum salary requirement")
    max_salary: Optional[Decimal] = Field(None, ge=0, description="Maximum salary budget")
    currency: str = Field(default="USD", description="Currency code")
    period: str = Field(default="annual", description="Salary period (annual, monthly, hourly)")
    include_negotiable: bool = Field(default=True, description="Include candidates with negotiable salary")


class SkillRequirement(BaseModel):
    """Skill requirement with importance weighting"""

    name: str = Field(..., description="Skill name", min_length=1)
    required: bool = Field(default=False, description="Whether skill is mandatory")
    min_years: Optional[int] = Field(None, ge=0, description="Minimum years of experience")
    min_level: Optional[str] = Field(None, description="Minimum proficiency level")
    weight: float = Field(default=1.0, ge=0, le=10, description="Importance weight for scoring")
    alternatives: List[str] = Field(default_factory=list, description="Alternative/equivalent skills")


class ExperienceRequirement(BaseModel):
    """Experience requirements"""

    min_total_years: Optional[int] = Field(None, ge=0, description="Minimum total experience")
    max_total_years: Optional[int] = Field(None, ge=0, description="Maximum total experience")
    required_titles: List[str] = Field(default_factory=list, description="Required job titles")
    preferred_companies: List[str] = Field(default_factory=list, description="Preferred companies")
    required_industries: List[str] = Field(default_factory=list, description="Required industries")
    min_company_size: Optional[str] = Field(None, description="Minimum company size")

    @field_validator('max_total_years')
    @classmethod
    def validate_experience_range(cls, v, info):
        """Ensure max >= min for experience years"""
        if v is not None and 'min_total_years' in info.data:
            min_years = info.data['min_total_years']
            if min_years is not None and v < min_years:
                raise ValueError('max_total_years must be >= min_total_years')
        return v


class EducationRequirement(BaseModel):
    """Education requirements"""

    required_degree_levels: List[str] = Field(default_factory=list, description="Required degree levels")
    preferred_institutions: List[str] = Field(default_factory=list, description="Preferred institutions")
    required_majors: List[str] = Field(default_factory=list, description="Required majors/fields")
    min_gpa: Optional[Decimal] = Field(None, ge=0, le=4, description="Minimum GPA requirement")


class SearchRequest(BaseModel):
    """
    Comprehensive search request with multi-modal search capabilities.

    Supports all search modes from simple keyword to advanced multi-stage
    retrieval with business logic scoring and reranking.
    """

    # Core Search Parameters
    query: str = Field(..., description="Primary search query", min_length=1, max_length=1000)
    search_mode: SearchMode = Field(default=SearchMode.HYBRID, description="Search execution mode")
    tenant_id: UUID = Field(..., description="Tenant identifier")

    # Pagination and Limits
    pagination: PaginationModel = Field(default_factory=PaginationModel, description="Pagination parameters")
    max_results: int = Field(default=100, ge=1, le=1000, description="Maximum results to return")

    # Sorting and Ranking
    sort_order: SortOrder = Field(default=SortOrder.RELEVANCE, description="Result sort order")
    custom_scoring: Optional[Dict[str, float]] = Field(None, description="Custom field weights for scoring")

    # Filtering
    basic_filters: List[SearchFilter] = Field(default_factory=list, description="Basic field filters")
    range_filters: List[RangeFilter] = Field(default_factory=list, description="Range filters")
    location_filter: Optional[LocationFilter] = Field(None, description="Location-based filtering")
    salary_filter: Optional[SalaryFilter] = Field(None, description="Salary expectation filtering")

    # Requirements
    skill_requirements: List[SkillRequirement] = Field(default_factory=list, description="Skill requirements")
    experience_requirements: Optional[ExperienceRequirement] = Field(None, description="Experience requirements")
    education_requirements: Optional[EducationRequirement] = Field(None, description="Education requirements")

    # Search Behavior
    include_inactive: bool = Field(default=False, description="Include inactive profiles")
    min_match_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum match score threshold")
    enable_query_expansion: bool = Field(default=True, description="Enable automatic query expansion")
    include_synonyms: bool = Field(default=True, description="Include synonym matching")
    boost_recent_activity: bool = Field(default=True, description="Boost recently active profiles")

    # Multi-stage Search Configuration
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Vector search weight in hybrid mode")
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Keyword search weight in hybrid mode")
    rerank_top_k: int = Field(default=50, ge=1, le=200, description="Number of results to rerank")

    # Analytics and Tracking
    track_search: bool = Field(default=True, description="Track this search for analytics")
    search_context: Optional[str] = Field(None, description="Search context (job_post_id, etc.)")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User-specific preferences")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Senior Python developer with machine learning experience",
                "search_mode": "hybrid",
                "skill_requirements": [
                    {"name": "Python", "required": True, "min_years": 3, "weight": 3.0},
                    {"name": "Machine Learning", "required": True, "min_years": 2, "weight": 2.5}
                ],
                "experience_requirements": {
                    "min_total_years": 5,
                    "required_titles": ["Senior Developer", "Lead Engineer", "Principal Engineer"]
                },
                "location_filter": {
                    "center_location": "San Francisco, CA",
                    "radius_km": 50,
                    "include_remote": True
                }
            }
        }
    )

    @field_validator('vector_weight', 'keyword_weight')
    @classmethod
    def validate_weights_sum(cls, v, info):
        """Ensure vector_weight + keyword_weight <= 1.0"""
        if 'vector_weight' in info.data:
            vector_weight = info.data.get('vector_weight', 0.0)
            if info.field_name == 'keyword_weight':
                if vector_weight + v > 1.0:
                    raise ValueError('vector_weight + keyword_weight must not exceed 1.0')
        return v

    @computed_field
    @property
    def has_filters(self) -> bool:
        """Check if any filters are applied"""
        return bool(
            self.basic_filters or
            self.range_filters or
            self.location_filter or
            self.salary_filter or
            self.skill_requirements or
            self.experience_requirements or
            self.education_requirements
        )

    @computed_field
    @property
    def required_skills(self) -> List[str]:
        """Get list of required skill names"""
        return [skill.name for skill in self.skill_requirements if skill.required]

    @computed_field
    @property
    def preferred_skills(self) -> List[str]:
        """Get list of preferred (non-required) skill names"""
        return [skill.name for skill in self.skill_requirements if not skill.required]


class MatchScore(BaseModel):
    """Individual match scoring details"""

    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall match score")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Search relevance score")
    skill_match_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Skill matching score")
    experience_match_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Experience matching score")
    education_match_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Education matching score")
    location_match_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Location matching score")
    salary_match_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Salary compatibility score")

    # Detailed breakdown
    matched_skills: List[str] = Field(default_factory=list, description="Skills that matched")
    missing_skills: List[str] = Field(default_factory=list, description="Required skills not found")
    skill_gaps: Dict[str, str] = Field(default_factory=dict, description="Skill level gaps")

    # Search engine scores
    vector_similarity: Optional[float] = Field(None, description="Vector similarity score")
    keyword_relevance: Optional[float] = Field(None, description="Keyword relevance score")
    semantic_relevance: Optional[float] = Field(None, description="Semantic relevance score")
    reranker_score: Optional[float] = Field(None, description="Cross-encoder reranker score")

    # Explanation
    match_explanation: List[str] = Field(default_factory=list, description="Human-readable match reasons")
    score_breakdown: Dict[str, float] = Field(default_factory=dict, description="Detailed score components")


class SearchResult(BaseModel):
    """Individual search result with profile data and match scoring"""

    # Profile Identity
    profile_id: str = Field(..., description="Profile identifier")
    email: str = Field(..., description="Profile email")
    tenant_id: UUID = Field(..., description="Tenant identifier")

    # Core Profile Data (subset for search results)
    full_name: Optional[str] = Field(None, description="Full name")
    title: Optional[str] = Field(None, description="Professional title")
    summary: Optional[str] = Field(None, description="Professional summary")
    current_company: Optional[str] = Field(None, description="Current employer")
    current_location: Optional[str] = Field(None, description="Current location")
    total_experience_years: Optional[int] = Field(None, description="Years of experience")

    # Key Skills and Qualifications
    top_skills: List[str] = Field(default_factory=list, description="Top matching skills")
    key_achievements: List[str] = Field(default_factory=list, description="Key achievements")
    highest_degree: Optional[str] = Field(None, description="Highest education level")

    # Match Information
    match_score: MatchScore = Field(..., description="Detailed match scoring")
    search_highlights: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Highlighted text snippets"
    )

    # Metadata
    last_updated: datetime = Field(..., description="Profile last update time")
    profile_completeness: float = Field(default=0.0, ge=0.0, le=1.0, description="Profile completeness score")
    availability_status: Optional[str] = Field(None, description="Availability for new opportunities")

    @computed_field
    @property
    def match_percentage(self) -> int:
        """Get match score as percentage"""
        return int(self.match_score.overall_score * 100)

    @computed_field
    @property
    def is_high_match(self) -> bool:
        """Check if this is a high-quality match (>80%)"""
        return self.match_score.overall_score >= 0.8


class FacetValue(BaseModel):
    """Individual facet value with count"""

    value: str = Field(..., description="Facet value")
    count: int = Field(..., ge=0, description="Number of results with this value")
    display_name: Optional[str] = Field(None, description="Human-readable display name")


class SearchFacet(BaseModel):
    """Search facet with possible values and counts"""

    field: str = Field(..., description="Field name")
    display_name: str = Field(..., description="Human-readable facet name")
    values: List[FacetValue] = Field(..., description="Facet values with counts")
    facet_type: str = Field(default="terms", description="Facet type (terms, range, date)")


class SearchAnalytics(BaseModel):
    """Search analytics and performance metrics"""

    total_search_time_ms: int = Field(..., ge=0, description="Total search time in milliseconds")
    vector_search_time_ms: Optional[int] = Field(None, description="Vector search time")
    keyword_search_time_ms: Optional[int] = Field(None, description="Keyword search time")
    reranking_time_ms: Optional[int] = Field(None, description="Reranking time")

    # Index statistics
    total_candidates: int = Field(..., ge=0, description="Total candidates in index")
    candidates_after_filters: int = Field(..., ge=0, description="Candidates after filtering")
    candidates_reranked: int = Field(default=0, description="Number of candidates reranked")

    # Query information
    query_expanded: bool = Field(default=False, description="Whether query was expanded")
    expanded_terms: List[str] = Field(default_factory=list, description="Added expansion terms")
    synonyms_used: List[str] = Field(default_factory=list, description="Synonyms applied")

    # Resource usage
    ru_consumed: Optional[float] = Field(None, description="Request units consumed (Cosmos DB)")
    cache_hit_rate: Optional[float] = Field(None, description="Cache hit rate")


class SearchResponse(BaseModel):
    """
    Complete search response with results, facets, and analytics.

    Provides comprehensive search results with:
    - Ranked and scored candidate profiles
    - Search facets for refinement
    - Performance analytics and metrics
    - Pagination information
    """

    # Results
    results: List[SearchResult] = Field(..., description="Search results")
    total_count: int = Field(..., ge=0, description="Total matching candidates")

    # Pagination
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, description="Results per page")
    total_pages: int = Field(..., ge=0, description="Total pages available")
    has_next_page: bool = Field(..., description="Whether next page exists")
    has_prev_page: bool = Field(..., description="Whether previous page exists")

    # Search metadata
    search_id: str = Field(..., description="Unique search identifier for tracking")
    search_mode: SearchMode = Field(..., description="Search mode used")
    query: str = Field(..., description="Original search query")

    # Facets for refinement
    facets: List[SearchFacet] = Field(default_factory=list, description="Available search facets")

    # Analytics and performance
    analytics: SearchAnalytics = Field(..., description="Search performance metrics")

    # Timestamps
    search_timestamp: datetime = Field(default_factory=datetime.utcnow, description="When search was executed")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "profile_id": "12345",
                        "email": "john.doe@example.com",
                        "full_name": "John Doe",
                        "title": "Senior Python Developer",
                        "match_score": {
                            "overall_score": 0.85,
                            "skill_match_score": 0.90,
                            "experience_match_score": 0.80
                        }
                    }
                ],
                "total_count": 156,
                "page": 1,
                "page_size": 20,
                "analytics": {
                    "total_search_time_ms": 245,
                    "total_candidates": 50000,
                    "candidates_after_filters": 156
                }
            }
        }
    )

    @computed_field
    @property
    def has_results(self) -> bool:
        """Check if search returned any results"""
        return len(self.results) > 0

    @computed_field
    @property
    def high_match_count(self) -> int:
        """Count of high-quality matches (>80% score)"""
        return sum(1 for result in self.results if result.is_high_match)

    @computed_field
    @property
    def average_match_score(self) -> float:
        """Average match score across all results"""
        if not self.results:
            return 0.0
        return sum(result.match_score.overall_score for result in self.results) / len(self.results)


class SavedSearchCreate(BaseModel):
    """Request schema for creating a saved search."""

    name: str = Field(..., description="Saved search name", min_length=1, max_length=200)
    description: Optional[str] = Field(None, description="Optional description")
    search_request: SearchRequest = Field(..., description="Complete search configuration")
    is_alert: bool = Field(default=False, description="Enable as alert")
    alert_frequency: Optional[str] = Field(None, description="Alert frequency (daily, weekly, monthly)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Senior Python Developers in SF",
                "description": "Looking for senior Python developers with ML experience",
                "search_request": {
                    "query": "Senior Python developer with machine learning",
                    "search_mode": "hybrid",
                    "skill_requirements": [
                        {"name": "Python", "required": True, "min_years": 5}
                    ]
                },
                "is_alert": True,
                "alert_frequency": "daily"
            }
        }
    )


class SavedSearch(BaseModel):
    """Complete saved search response with metadata."""

    id: Optional[str] = Field(None, description="Saved search ID")
    name: str = Field(..., description="Search name", min_length=1, max_length=200)
    description: Optional[str] = Field(None, description="Search description")
    search_request: SearchRequest = Field(..., description="Saved search configuration")
    tenant_id: UUID = Field(..., description="Tenant identifier")

    # Automation
    is_alert: bool = Field(default=False, description="Whether to run as alert")
    alert_frequency: Optional[str] = Field(None, description="Alert frequency")
    last_run: Optional[datetime] = Field(None, description="Last execution time")

    # Results tracking
    last_result_count: int = Field(default=0, description="Results from last run")
    new_results_since_last_run: int = Field(default=0, description="New results since last check")

    # Metadata
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class SearchHistory(BaseModel):
    """Search history for analytics and recommendations"""

    search_request: SearchRequest = Field(..., description="Original search request")
    response_summary: Dict[str, Any] = Field(..., description="Search response summary")
    user_id: Optional[UUID] = Field(None, description="User who performed search")
    tenant_id: UUID = Field(..., description="Tenant identifier")

    # User behavior
    results_clicked: List[str] = Field(default_factory=list, description="Profile IDs clicked")
    results_contacted: List[str] = Field(default_factory=list, description="Profile IDs contacted")
    search_abandoned: bool = Field(default=False, description="Whether search was abandoned")

    # Performance
    search_duration_ms: int = Field(..., description="Search execution time")
    satisfaction_rating: Optional[int] = Field(None, ge=1, le=5, description="User satisfaction rating")


class SearchHistoryItem(BaseModel):
    """Individual search history item for list responses."""

    search_id: str = Field(..., description="Search history identifier")
    query: str = Field(..., description="Search query")
    search_mode: str = Field(..., description="Search mode used")
    total_results: int = Field(..., description="Total results found")
    search_outcome: str = Field(..., description="Search outcome")
    duration_ms: int = Field(..., description="Search duration in milliseconds")
    engagement_score: float = Field(..., description="User engagement score (0.0-1.0)")
    results_clicked_count: int = Field(..., description="Number of results clicked")
    executed_at: datetime = Field(..., description="When search was executed")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "search_id": "12345678-1234-5678-1234-567812345678",
                "query": "Senior Python developer",
                "search_mode": "hybrid",
                "total_results": 45,
                "search_outcome": "success",
                "duration_ms": 423,
                "engagement_score": 0.7,
                "results_clicked_count": 3,
                "executed_at": "2025-10-13T10:30:00Z"
            }
        }
    )
