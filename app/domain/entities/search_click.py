"""Pure domain representation of search click events."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID

from app.domain.value_objects import SearchClickId, TenantId, UserId, ProfileId


class ClickSource(str, Enum):
    """Source context where the click occurred."""

    SEARCH_RESULTS = "search_results"  # Main search results page
    SAVED_SEARCH = "saved_search"  # From saved search execution
    SEARCH_HISTORY = "search_history"  # Re-running historical search
    RECOMMENDATION = "recommendation"  # From recommendation engine
    EMAIL_ALERT = "email_alert"  # From email alert link
    API = "api"  # Direct API call
    UNKNOWN = "unknown"


class ClickDevice(str, Enum):
    """Device type for click event."""

    DESKTOP = "desktop"
    MOBILE = "mobile"
    TABLET = "tablet"
    API_CLIENT = "api_client"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClickContext:
    """Immutable context information for a click event."""

    # Session information
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_type: ClickDevice = ClickDevice.UNKNOWN

    # Behavioral context
    time_to_click_ms: Optional[int] = None  # Time from search to click
    scroll_position: Optional[int] = None  # Scroll depth when clicked
    viewport_height: Optional[int] = None
    previous_clicks: int = 0  # Number of results clicked before this one

    # Search context
    query_length: int = 0
    results_shown: int = 0
    filter_count: int = 0

    # Source tracking
    source: ClickSource = ClickSource.SEARCH_RESULTS
    referrer_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "device_type": self.device_type.value,
            "time_to_click_ms": self.time_to_click_ms,
            "scroll_position": self.scroll_position,
            "viewport_height": self.viewport_height,
            "previous_clicks": self.previous_clicks,
            "query_length": self.query_length,
            "results_shown": self.results_shown,
            "filter_count": self.filter_count,
            "source": self.source.value,
            "referrer_url": self.referrer_url,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> ClickContext:
        """Reconstruct from dictionary."""
        return ClickContext(
            session_id=data.get("session_id"),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            device_type=ClickDevice(data.get("device_type", "unknown")),
            time_to_click_ms=data.get("time_to_click_ms"),
            scroll_position=data.get("scroll_position"),
            viewport_height=data.get("viewport_height"),
            previous_clicks=data.get("previous_clicks", 0),
            query_length=data.get("query_length", 0),
            results_shown=data.get("results_shown", 0),
            filter_count=data.get("filter_count", 0),
            source=ClickSource(data.get("source", "search_results")),
            referrer_url=data.get("referrer_url"),
        )


@dataclass
class SearchClick:
    """
    Domain aggregate for search result click events.

    Represents a single user interaction with a search result.
    Optimized for high-volume writes and time-series analytics.
    """

    # Identity
    id: SearchClickId
    tenant_id: TenantId
    user_id: UserId

    # Core event data
    search_id: str  # Reference to the search that produced results
    profile_id: ProfileId  # Profile that was clicked
    position: int  # Position in search results (0-based)

    # Temporal data
    clicked_at: datetime = field(default_factory=datetime.utcnow)

    # Context (immutable)
    context: ClickContext = field(default_factory=ClickContext)

    # Derived metrics (computed at creation)
    relevance_score: Optional[float] = None  # Score of clicked result
    rank_quality: Optional[float] = None  # Position quality metric

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate click event after creation."""
        self._validate()
        self._compute_derived_metrics()

    def _validate(self) -> None:
        """Validate click event invariants."""
        if self.position < 0:
            raise ValueError("Click position cannot be negative")

        if self.position > 10000:
            raise ValueError("Click position exceeds reasonable bounds")

        if not self.search_id:
            raise ValueError("Search ID is required")

        # Validate temporal ordering
        if self.clicked_at > datetime.utcnow():
            raise ValueError("Click timestamp cannot be in the future")

    def _compute_derived_metrics(self) -> None:
        """Compute derived analytics metrics."""
        # Rank quality: Higher positions get lower quality scores
        # Top result (position 0) gets 1.0, decreases exponentially
        if self.position == 0:
            self.rank_quality = 1.0
        else:
            # Exponential decay: 0.9^position
            self.rank_quality = 0.9 ** self.position

    def is_top_result(self) -> bool:
        """Check if this was a click on the top result."""
        return self.position == 0

    def is_first_page(self) -> bool:
        """Check if click was on first page (top 20 results)."""
        return self.position < 20

    def is_quick_click(self) -> bool:
        """Check if this was a quick click (< 5 seconds from search)."""
        return (
            self.context.time_to_click_ms is not None and
            self.context.time_to_click_ms < 5000
        )

    def get_engagement_signal(self) -> str:
        """
        Classify click as strong, medium, or weak engagement signal.

        Strong: Top 3 results, quick clicks
        Medium: First page, reasonable time
        Weak: Deep results, slow clicks
        """
        if self.position < 3 and self.is_quick_click():
            return "strong"
        elif self.position < 20:
            return "medium"
        else:
            return "weak"

    def to_event_data(self) -> Dict[str, Any]:
        """Convert to event data for analytics processing."""
        return {
            "event_id": str(self.id.value),
            "tenant_id": str(self.tenant_id.value),
            "user_id": str(self.user_id.value),
            "search_id": self.search_id,
            "profile_id": str(self.profile_id.value),
            "position": self.position,
            "clicked_at": self.clicked_at.isoformat(),
            "relevance_score": self.relevance_score,
            "rank_quality": self.rank_quality,
            "engagement_signal": self.get_engagement_signal(),
            "is_top_result": self.is_top_result(),
            "is_first_page": self.is_first_page(),
            "context": self.context.to_dict(),
            "metadata": self.metadata,
        }


__all__ = [
    "SearchClick",
    "ClickContext",
    "ClickSource",
    "ClickDevice",
]