"""Pure domain representation of saved search configurations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from app.domain.value_objects import SavedSearchId, TenantId, UserId


class AlertFrequency(str, Enum):
    """Frequency options for search alerts."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    NEVER = "never"


class SavedSearchStatus(str, Enum):
    """Lifecycle status for saved searches."""

    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"
    DELETED = "deleted"


@dataclass
class SearchConfiguration:
    """Search configuration parameters stored in domain."""

    # Core search parameters
    query: str
    search_mode: str
    max_results: int = 100

    # Filters (stored as dictionaries for flexibility)
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
        """Check if any filters are applied."""
        return bool(
            self.basic_filters or
            self.range_filters or
            self.location_filter or
            self.salary_filter or
            self.skill_requirements or
            self.experience_requirements or
            self.education_requirements
        )


@dataclass
class ExecutionStatistics:
    """Statistics about saved search execution history."""

    total_executions: int = 0
    last_run: Optional[datetime] = None
    last_result_count: int = 0
    new_results_since_last_run: int = 0
    average_result_count: float = 0.0
    average_execution_time_ms: int = 0

    def record_execution(
        self,
        result_count: int,
        execution_time_ms: int
    ) -> None:
        """Record a search execution."""
        self.last_run = datetime.utcnow()
        self.last_result_count = result_count
        self.total_executions += 1

        # Update rolling average
        if self.total_executions == 1:
            self.average_result_count = float(result_count)
            self.average_execution_time_ms = execution_time_ms
        else:
            self.average_result_count = (
                (self.average_result_count * (self.total_executions - 1) + result_count)
                / self.total_executions
            )
            self.average_execution_time_ms = int(
                (self.average_execution_time_ms * (self.total_executions - 1) + execution_time_ms)
                / self.total_executions
            )


@dataclass
class SavedSearch:
    """Aggregate root for saved search configurations."""

    id: SavedSearchId
    tenant_id: TenantId
    created_by: UserId
    name: str
    description: Optional[str]

    # Search configuration
    configuration: SearchConfiguration

    # Alert settings
    is_alert: bool = False
    alert_frequency: AlertFrequency = AlertFrequency.NEVER
    next_alert_at: Optional[datetime] = None

    # Sharing and permissions
    is_shared: bool = False
    shared_with_users: List[UserId] = field(default_factory=list)

    # Status and metadata
    status: SavedSearchStatus = SavedSearchStatus.ACTIVE
    statistics: ExecutionStatistics = field(default_factory=ExecutionStatistics)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    updated_by: Optional[UserId] = None

    # Audit trail
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate saved search after creation."""
        self._validate()

    def _validate(self) -> None:
        """Validate saved search configuration."""
        if not self.name or len(self.name.strip()) == 0:
            raise ValueError("Saved search must have a name")

        if len(self.name) > 200:
            raise ValueError("Saved search name must be 200 characters or less")

        if not self.configuration.query:
            raise ValueError("Search configuration must have a query")

        if self.is_alert and self.alert_frequency == AlertFrequency.NEVER:
            raise ValueError("Alert frequency must be set when is_alert is True")

    def activate(self) -> None:
        """Activate the saved search."""
        if self.status == SavedSearchStatus.DELETED:
            raise ValueError("Cannot activate deleted saved search")
        self.status = SavedSearchStatus.ACTIVE
        self.updated_at = datetime.utcnow()

    def pause(self) -> None:
        """Pause the saved search and its alerts."""
        self.status = SavedSearchStatus.PAUSED
        self.updated_at = datetime.utcnow()

    def archive(self) -> None:
        """Archive the saved search."""
        self.status = SavedSearchStatus.ARCHIVED
        self.updated_at = datetime.utcnow()

    def delete(self) -> None:
        """Soft-delete the saved search."""
        self.status = SavedSearchStatus.DELETED
        self.updated_at = datetime.utcnow()

    def update_configuration(
        self,
        configuration: SearchConfiguration,
        updated_by: UserId
    ) -> None:
        """Update search configuration."""
        self.configuration = configuration
        self.updated_by = updated_by
        self.updated_at = datetime.utcnow()

    def enable_alert(
        self,
        frequency: AlertFrequency,
        updated_by: UserId
    ) -> None:
        """Enable alert with specified frequency."""
        if frequency == AlertFrequency.NEVER:
            raise ValueError("Cannot enable alert with NEVER frequency")

        self.is_alert = True
        self.alert_frequency = frequency
        self.next_alert_at = self._calculate_next_alert_time(frequency)
        self.updated_by = updated_by
        self.updated_at = datetime.utcnow()

    def disable_alert(self, updated_by: UserId) -> None:
        """Disable alert."""
        self.is_alert = False
        self.alert_frequency = AlertFrequency.NEVER
        self.next_alert_at = None
        self.updated_by = updated_by
        self.updated_at = datetime.utcnow()

    def record_execution(
        self,
        result_count: int,
        execution_time_ms: int
    ) -> None:
        """Record a search execution."""
        self.statistics.record_execution(result_count, execution_time_ms)

        # Update next alert time if this is an alert
        if self.is_alert and self.alert_frequency != AlertFrequency.NEVER:
            self.next_alert_at = self._calculate_next_alert_time(self.alert_frequency)

        self.updated_at = datetime.utcnow()

    def share_with_user(self, user_id: UserId) -> None:
        """Share saved search with another user."""
        if user_id not in self.shared_with_users:
            self.shared_with_users.append(user_id)
            self.is_shared = True
            self.updated_at = datetime.utcnow()

    def unshare_with_user(self, user_id: UserId) -> None:
        """Remove sharing with a user."""
        if user_id in self.shared_with_users:
            self.shared_with_users.remove(user_id)
            self.is_shared = len(self.shared_with_users) > 0
            self.updated_at = datetime.utcnow()

    def can_be_accessed_by(self, user_id: UserId) -> bool:
        """Check if user can access this saved search."""
        return (
            self.created_by == user_id or
            user_id in self.shared_with_users
        )

    @staticmethod
    def _calculate_next_alert_time(frequency: AlertFrequency) -> datetime:
        """Calculate next alert execution time."""
        now = datetime.utcnow()

        if frequency == AlertFrequency.DAILY:
            return now + timedelta(days=1)
        elif frequency == AlertFrequency.WEEKLY:
            return now + timedelta(weeks=1)
        elif frequency == AlertFrequency.MONTHLY:
            # Approximate month as 30 days
            return now + timedelta(days=30)
        else:
            return now


__all__ = [
    "SavedSearch",
    "SearchConfiguration",
    "ExecutionStatistics",
    "AlertFrequency",
    "SavedSearchStatus",
]