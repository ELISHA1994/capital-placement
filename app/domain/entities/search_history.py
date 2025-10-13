"""Pure domain representation of search history records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.domain.value_objects import SearchHistoryId, TenantId, UserId


class SearchOutcome(str, Enum):
    """Outcome of a search operation."""

    SUCCESS = "success"  # Search completed successfully with results
    NO_RESULTS = "no_results"  # Search completed but returned zero results
    ERROR = "error"  # Search failed with an error
    ABANDONED = "abandoned"  # User left before viewing results
    TIMEOUT = "timeout"  # Search exceeded timeout threshold


class InteractionType(str, Enum):
    """Types of user interactions with search results."""

    RESULT_VIEWED = "result_viewed"  # User viewed result details
    RESULT_CLICKED = "result_clicked"  # User clicked on a result
    PROFILE_CONTACTED = "profile_contacted"  # User contacted candidate
    PROFILE_SHORTLISTED = "profile_shortlisted"  # User added to shortlist
    SEARCH_REFINED = "search_refined"  # User modified search parameters
    SEARCH_ABANDONED = "search_abandoned"  # User left without interaction


@dataclass
class SearchParameters:
    """Domain representation of search configuration used."""

    # Core search parameters
    query: str
    search_mode: str  # keyword, vector, hybrid, semantic, multi_stage
    max_results: int = 100

    # Filters applied (stored as dicts for flexibility)
    basic_filters: List[Dict[str, Any]] = field(default_factory=list)
    range_filters: List[Dict[str, Any]] = field(default_factory=list)
    location_filter: Optional[Dict[str, Any]] = None
    salary_filter: Optional[Dict[str, Any]] = None

    # Requirements
    skill_requirements: List[Dict[str, Any]] = field(default_factory=list)
    experience_requirements: Optional[Dict[str, Any]] = None
    education_requirements: Optional[Dict[str, Any]] = None

    # Search behavior
    include_inactive: bool = False
    min_match_score: float = 0.0
    enable_query_expansion: bool = True

    # Weights
    vector_weight: float = 0.7
    keyword_weight: float = 0.3

    def has_filters(self) -> bool:
        """Check if any filters were applied."""
        return bool(
            self.basic_filters or
            self.range_filters or
            self.location_filter or
            self.salary_filter or
            self.skill_requirements or
            self.experience_requirements or
            self.education_requirements
        )

    def get_filter_count(self) -> int:
        """Count total number of filters applied."""
        count = len(self.basic_filters) + len(self.range_filters)
        if self.location_filter:
            count += 1
        if self.salary_filter:
            count += 1
        count += len(self.skill_requirements)
        if self.experience_requirements:
            count += 1
        if self.education_requirements:
            count += 1
        return count


@dataclass
class SearchResultsSummary:
    """Summary of search results returned."""

    total_count: int  # Total matching profiles
    returned_count: int  # Number of results actually returned
    has_more: bool  # Whether there are more results available

    # Match quality
    average_match_score: float = 0.0
    high_match_count: int = 0  # Results with score >= 80%

    # Result distribution
    top_skills_found: List[str] = field(default_factory=list)
    location_distribution: Dict[str, int] = field(default_factory=dict)
    experience_range: Dict[str, int] = field(default_factory=dict)  # min, max, avg

    # Search engine details
    cache_hit: bool = False
    vector_search_used: bool = False
    query_expanded: bool = False
    expanded_terms: List[str] = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    """Performance metrics for search execution."""

    # Timing breakdown (all in milliseconds)
    total_duration_ms: int
    vector_search_duration_ms: Optional[int] = None
    text_search_duration_ms: Optional[int] = None
    reranking_duration_ms: Optional[int] = None
    post_processing_duration_ms: Optional[int] = None

    # Resource usage
    cache_hit: bool = False
    cache_hit_rate: Optional[float] = None
    database_queries: int = 0

    # AI operations
    ai_tokens_used: Optional[int] = None
    ai_model_used: Optional[str] = None
    ai_cost_estimate: Optional[float] = None

    def is_slow_search(self, threshold_ms: int = 2000) -> bool:
        """Check if search exceeded performance threshold."""
        return self.total_duration_ms > threshold_ms

    def is_cache_effective(self, min_rate: float = 0.3) -> bool:
        """Check if cache hit rate is above minimum threshold."""
        return self.cache_hit_rate is not None and self.cache_hit_rate >= min_rate


@dataclass
class UserEngagement:
    """Track user engagement with search results."""

    # Interaction tracking
    results_viewed: List[str] = field(default_factory=list)  # Profile IDs
    results_clicked: List[str] = field(default_factory=list)  # Profile IDs with timestamps
    profiles_contacted: List[str] = field(default_factory=list)  # Profile IDs
    profiles_shortlisted: List[str] = field(default_factory=list)  # Profile IDs

    # Engagement metrics
    time_on_results_seconds: int = 0  # Time spent viewing results
    search_refined: bool = False  # Whether user refined the search
    session_abandoned: bool = False  # Whether user left without interaction

    # Satisfaction indicators
    satisfaction_rating: Optional[int] = None  # 1-5 scale if collected
    feedback_provided: Optional[str] = None  # User feedback text

    def get_click_through_rate(self, total_results: int) -> float:
        """Calculate CTR as percentage of results clicked."""
        if total_results == 0:
            return 0.0
        return (len(self.results_clicked) / total_results) * 100

    def get_engagement_score(self) -> float:
        """Calculate engagement score from 0.0 to 1.0."""
        score = 0.0

        # Viewing results: 0.2
        if len(self.results_viewed) > 0:
            score += 0.2

        # Clicking results: 0.3
        if len(self.results_clicked) > 0:
            score += 0.3

        # Contacting or shortlisting: 0.3
        if len(self.profiles_contacted) > 0 or len(self.profiles_shortlisted) > 0:
            score += 0.3

        # Time spent (1 min+ = 0.2)
        if self.time_on_results_seconds >= 60:
            score += 0.2

        return min(score, 1.0)

    def is_engaged(self) -> bool:
        """Check if user showed meaningful engagement."""
        return (
            len(self.results_clicked) > 0 or
            len(self.profiles_contacted) > 0 or
            len(self.profiles_shortlisted) > 0 or
            self.time_on_results_seconds >= 30
        )


@dataclass
class SearchHistory:
    """
    Aggregate root for search history records.

    Represents a complete search execution with all context, results,
    performance metrics, and user engagement data.
    """

    id: SearchHistoryId
    tenant_id: TenantId
    user_id: UserId

    # Search context
    search_parameters: SearchParameters
    results_summary: SearchResultsSummary

    # Performance and outcome
    performance_metrics: PerformanceMetrics
    search_outcome: SearchOutcome

    # User engagement
    engagement: UserEngagement = field(default_factory=UserEngagement)

    # Context and attribution
    search_context: Optional[str] = None  # job_post_id, campaign_id, etc.
    source: str = "web_ui"  # web_ui, api, automation, saved_search_alert
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None

    # Timestamps
    executed_at: datetime = field(default_factory=datetime.utcnow)
    last_interaction_at: Optional[datetime] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate search history after creation."""
        self._validate()

    def _validate(self) -> None:
        """Validate search history data."""
        if not self.search_parameters.query:
            raise ValueError("Search history must have a query")

        if self.results_summary.total_count < 0:
            raise ValueError("Total count cannot be negative")

        if self.performance_metrics.total_duration_ms < 0:
            raise ValueError("Duration cannot be negative")

    def record_interaction(
        self,
        interaction_type: InteractionType,
        profile_id: Optional[str] = None,
        position: Optional[int] = None
    ) -> None:
        """Record a user interaction with search results."""
        self.last_interaction_at = datetime.utcnow()

        if interaction_type == InteractionType.RESULT_CLICKED and profile_id:
            if profile_id not in self.engagement.results_clicked:
                self.engagement.results_clicked.append(profile_id)

        elif interaction_type == InteractionType.PROFILE_CONTACTED and profile_id:
            if profile_id not in self.engagement.profiles_contacted:
                self.engagement.profiles_contacted.append(profile_id)

        elif interaction_type == InteractionType.PROFILE_SHORTLISTED and profile_id:
            if profile_id not in self.engagement.profiles_shortlisted:
                self.engagement.profiles_shortlisted.append(profile_id)

        elif interaction_type == InteractionType.SEARCH_REFINED:
            self.engagement.search_refined = True

        elif interaction_type == InteractionType.SEARCH_ABANDONED:
            self.engagement.session_abandoned = True

    def mark_as_abandoned(self) -> None:
        """Mark this search as abandoned by the user."""
        self.search_outcome = SearchOutcome.ABANDONED
        self.engagement.session_abandoned = True
        self.last_interaction_at = datetime.utcnow()

    def update_engagement_time(self, seconds: int) -> None:
        """Update time spent viewing results."""
        self.engagement.time_on_results_seconds += seconds
        self.last_interaction_at = datetime.utcnow()

    def set_satisfaction_rating(self, rating: int, feedback: Optional[str] = None) -> None:
        """Record user satisfaction rating."""
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")

        self.engagement.satisfaction_rating = rating
        if feedback:
            self.engagement.feedback_provided = feedback
        self.last_interaction_at = datetime.utcnow()

    def is_successful_search(self) -> bool:
        """Check if search was successful based on outcome and engagement."""
        return (
            self.search_outcome == SearchOutcome.SUCCESS and
            self.results_summary.total_count > 0 and
            self.engagement.is_engaged()
        )

    def needs_optimization(self) -> bool:
        """Check if this search pattern suggests need for optimization."""
        return (
            self.search_outcome == SearchOutcome.NO_RESULTS or
            self.performance_metrics.is_slow_search() or
            (self.search_outcome == SearchOutcome.SUCCESS and not self.engagement.is_engaged())
        )

    def get_analytics_tags(self) -> List[str]:
        """Generate tags for analytics categorization."""
        tags = []

        # Outcome tags
        tags.append(f"outcome:{self.search_outcome.value}")

        # Performance tags
        if self.performance_metrics.is_slow_search():
            tags.append("performance:slow")
        if self.performance_metrics.cache_hit:
            tags.append("cache:hit")

        # Engagement tags
        if self.engagement.is_engaged():
            tags.append("engagement:high")
        else:
            tags.append("engagement:low")

        # Results tags
        if self.results_summary.total_count == 0:
            tags.append("results:zero")
        elif self.results_summary.total_count > 100:
            tags.append("results:many")

        # Filter tags
        if self.search_parameters.has_filters():
            tags.append(f"filters:{self.search_parameters.get_filter_count()}")

        return tags


__all__ = [
    "SearchHistory",
    "SearchParameters",
    "SearchResultsSummary",
    "PerformanceMetrics",
    "UserEngagement",
    "SearchOutcome",
    "InteractionType",
]