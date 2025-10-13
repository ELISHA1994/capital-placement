"""
Analytics API Schemas - DTOs for analytics and reporting endpoints.

This module contains all request/response models for analytics API layer,
separated from domain entities and persistence tables following hexagonal architecture.
"""

from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class MetricType(str, Enum):
    """Metric type enumeration."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AggregationType(str, Enum):
    """Aggregation type enumeration."""
    SUM = "sum"
    COUNT = "count"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    PERCENTILE = "percentile"


class TimeInterval(str, Enum):
    """Time interval enumeration."""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class UsageEvent(BaseModel):
    """Individual usage event model."""

    # Event identification
    event_id: str = Field(..., description="Unique event identifier")
    event_type: str = Field(..., description="Event type/category")
    event_name: str = Field(..., description="Specific event name")

    # Context
    tenant_id: UUID = Field(..., description="Tenant ID")
    user_id: Optional[UUID] = Field(None, description="User ID (if applicable)")
    session_id: Optional[str] = Field(None, description="Session ID")

    # Timing
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Event timestamp"
    )
    duration_ms: Optional[int] = Field(None, description="Event duration in milliseconds")

    # Event data
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific properties"
    )
    metrics: Dict[str, Union[int, float]] = Field(
        default_factory=dict,
        description="Numeric metrics associated with event"
    )

    # Request context
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")
    referer: Optional[str] = Field(None, description="HTTP referer")

    # Status
    success: bool = Field(default=True, description="Whether event was successful")
    error_code: Optional[str] = Field(None, description="Error code if failed")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class Usage(BaseModel):
    """Aggregated usage statistics model."""

    id: UUID = Field(..., description="Usage record ID")
    tenant_id: UUID = Field(..., description="Tenant ID")

    # Time period
    period_start: datetime = Field(..., description="Period start timestamp")
    period_end: datetime = Field(..., description="Period end timestamp")
    interval: TimeInterval = Field(..., description="Time interval")

    # API Usage
    api_requests_total: int = Field(default=0, description="Total API requests")
    api_requests_successful: int = Field(default=0, description="Successful API requests")
    api_requests_failed: int = Field(default=0, description="Failed API requests")
    api_response_time_avg: float = Field(default=0.0, description="Average response time (ms)")
    api_rate_limit_exceeded: int = Field(default=0, description="Rate limit violations")

    # Search Usage
    searches_total: int = Field(default=0, description="Total searches performed")
    searches_keyword: int = Field(default=0, description="Keyword searches")
    searches_semantic: int = Field(default=0, description="Semantic searches")
    search_response_time_avg: float = Field(default=0.0, description="Average search time (ms)")
    search_results_avg: float = Field(default=0.0, description="Average results per search")

    # Document Usage
    documents_uploaded: int = Field(default=0, description="Documents uploaded")
    documents_processed: int = Field(default=0, description="Documents processed")
    documents_failed: int = Field(default=0, description="Processing failures")
    storage_used_bytes: int = Field(default=0, description="Storage used in bytes")
    processing_time_total: int = Field(default=0, description="Total processing time (ms)")

    # User Activity
    active_users: int = Field(default=0, description="Number of active users")
    new_users: int = Field(default=0, description="Number of new users")
    user_sessions: int = Field(default=0, description="Number of user sessions")
    session_duration_avg: float = Field(default=0.0, description="Average session duration (minutes)")

    # Profile Activity
    profiles_created: int = Field(default=0, description="Profiles created")
    profiles_updated: int = Field(default=0, description="Profiles updated")
    profile_matches: int = Field(default=0, description="Profile matches generated")
    profile_views: int = Field(default=0, description="Profile views")

    # Notification Usage
    notifications_sent: int = Field(default=0, description="Notifications sent")
    notifications_delivered: int = Field(default=0, description="Notifications delivered")
    notifications_opened: int = Field(default=0, description="Notifications opened")
    notifications_clicked: int = Field(default=0, description="Notifications clicked")

    # Performance Metrics
    cpu_usage_avg: Optional[float] = Field(None, description="Average CPU usage (%)")
    memory_usage_avg: Optional[float] = Field(None, description="Average memory usage (%)")
    disk_usage_bytes: Optional[int] = Field(None, description="Disk usage in bytes")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def api_success_rate(self) -> float:
        """Calculate API success rate."""
        if self.api_requests_total == 0:
            return 0.0
        return (self.api_requests_successful / self.api_requests_total) * 100

    @property
    def document_processing_success_rate(self) -> float:
        """Calculate document processing success rate."""
        total_processed = self.documents_processed + self.documents_failed
        if total_processed == 0:
            return 0.0
        return (self.documents_processed / total_processed) * 100

    @property
    def notification_delivery_rate(self) -> float:
        """Calculate notification delivery rate."""
        if self.notifications_sent == 0:
            return 0.0
        return (self.notifications_delivered / self.notifications_sent) * 100

    @property
    def storage_used_mb(self) -> float:
        """Get storage used in megabytes."""
        return round(self.storage_used_bytes / (1024 * 1024), 2)


class Metrics(BaseModel):
    """System metrics model."""

    # Metric identification
    name: str = Field(..., description="Metric name")
    type: MetricType = Field(..., description="Metric type")
    unit: Optional[str] = Field(None, description="Metric unit")

    # Values
    value: Union[int, float] = Field(..., description="Metric value")
    values: Optional[List[Union[int, float]]] = Field(
        None,
        description="Multiple values for histograms"
    )

    # Metadata
    labels: Dict[str, str] = Field(
        default_factory=dict,
        description="Metric labels/tags"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Metric timestamp"
    )

    # Context
    tenant_id: Optional[UUID] = Field(None, description="Tenant ID")
    user_id: Optional[UUID] = Field(None, description="User ID")

    # Aggregation info
    aggregation_period: Optional[str] = Field(None, description="Aggregation period")
    sample_count: Optional[int] = Field(None, description="Number of samples")


class AnalyticsData(BaseModel):
    """Comprehensive analytics data model."""

    # Time period
    period_start: datetime = Field(..., description="Analysis period start")
    period_end: datetime = Field(..., description="Analysis period end")
    interval: TimeInterval = Field(..., description="Data interval")

    # Usage summary
    total_usage: Usage = Field(..., description="Aggregated usage statistics")

    # Time series data
    usage_over_time: List[Usage] = Field(
        default_factory=list,
        description="Usage data points over time"
    )

    # Top-level metrics
    key_metrics: Dict[str, Union[int, float, str]] = Field(
        default_factory=dict,
        description="Key performance indicators"
    )

    # Trends and insights
    trends: Dict[str, Dict[str, Union[float, str]]] = Field(
        default_factory=dict,
        description="Trend analysis (growth rates, patterns)"
    )

    # Comparative data
    previous_period_comparison: Optional[Dict[str, float]] = Field(
        None,
        description="Comparison with previous period"
    )

    # Forecasting
    forecasts: Optional[Dict[str, List[Dict[str, Any]]]] = Field(
        None,
        description="Usage forecasts"
    )

    # Alerts and anomalies
    alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Usage alerts and anomalies"
    )


class TenantAnalytics(BaseModel):
    """Tenant-specific analytics model."""

    tenant_id: UUID = Field(..., description="Tenant ID")
    tenant_name: str = Field(..., description="Tenant name")

    # Current period analytics
    current_analytics: AnalyticsData = Field(..., description="Current period data")

    # Historical data
    historical_usage: List[Usage] = Field(
        default_factory=list,
        description="Historical usage data"
    )

    # Tenant-specific insights
    plan_utilization: Dict[str, float] = Field(
        default_factory=dict,
        description="Plan limit utilization percentages"
    )
    feature_adoption: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Feature adoption metrics"
    )
    user_engagement: Dict[str, Any] = Field(
        default_factory=dict,
        description="User engagement metrics"
    )

    # Cost and billing insights
    estimated_costs: Optional[Dict[str, float]] = Field(
        None,
        description="Estimated costs by service"
    )

    # Recommendations
    optimization_recommendations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Usage optimization recommendations"
    )


class SystemAnalytics(BaseModel):
    """System-wide analytics model."""

    # Global metrics
    total_tenants: int = Field(..., description="Total number of tenants")
    active_tenants: int = Field(..., description="Active tenants in period")
    total_users: int = Field(..., description="Total number of users")
    active_users: int = Field(..., description="Active users in period")

    # Aggregated usage
    system_usage: Usage = Field(..., description="System-wide usage")

    # Performance metrics
    average_response_time: float = Field(..., description="Average response time (ms)")
    system_availability: float = Field(..., description="System availability (%)")
    error_rate: float = Field(..., description="System error rate (%)")

    # Capacity metrics
    cpu_utilization: float = Field(..., description="Average CPU utilization (%)")
    memory_utilization: float = Field(..., description="Average memory utilization (%)")
    storage_utilization: float = Field(..., description="Storage utilization (%)")

    # Growth metrics
    tenant_growth_rate: float = Field(..., description="Tenant growth rate (%)")
    user_growth_rate: float = Field(..., description="User growth rate (%)")
    usage_growth_rate: float = Field(..., description="Usage growth rate (%)")

    # Top performers
    top_tenants_by_usage: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top tenants by usage"
    )

    # System health
    health_indicators: Dict[str, Any] = Field(
        default_factory=dict,
        description="System health indicators"
    )


class ReportRequest(BaseModel):
    """Analytics report request model."""

    # Report parameters
    report_type: str = Field(..., description="Type of report to generate")
    period_start: datetime = Field(..., description="Report period start")
    period_end: datetime = Field(..., description="Report period end")
    interval: TimeInterval = Field(default=TimeInterval.DAY, description="Data interval")

    # Scope
    tenant_ids: Optional[List[UUID]] = Field(None, description="Specific tenants to include")
    user_ids: Optional[List[UUID]] = Field(None, description="Specific users to include")

    # Filters
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional filters"
    )

    # Output options
    include_charts: bool = Field(default=True, description="Include chart data")
    include_raw_data: bool = Field(default=False, description="Include raw data")
    format: str = Field(default="json", pattern="^(json|csv|excel|pdf)$", description="Output format")

    # Aggregation options
    aggregations: List[str] = Field(
        default_factory=list,
        description="Metrics to aggregate"
    )
    group_by: List[str] = Field(
        default_factory=list,
        description="Fields to group by"
    )

    @field_validator('period_end')
    @classmethod
    def validate_period(cls, v, info):
        """Validate that period_end is after period_start."""
        if 'period_start' in info.data and v <= info.data['period_start']:
            raise ValueError('period_end must be after period_start')
        return v


class ReportResponse(BaseModel):
    """Analytics report response model."""

    # Report metadata
    report_id: UUID = Field(..., description="Report ID")
    report_type: str = Field(..., description="Report type")
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Report generation timestamp"
    )

    # Request parameters
    request: ReportRequest = Field(..., description="Original request parameters")

    # Report data
    data: AnalyticsData = Field(..., description="Analytics data")

    # Additional insights
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Executive summary"
    )
    insights: List[str] = Field(
        default_factory=list,
        description="Key insights and observations"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations"
    )

    # Export information
    download_url: Optional[str] = Field(None, description="Report download URL")
    expires_at: Optional[datetime] = Field(None, description="Download link expiration")


class Dashboard(BaseModel):
    """Dashboard configuration model."""

    id: UUID = Field(..., description="Dashboard ID")
    name: str = Field(..., min_length=1, max_length=100, description="Dashboard name")
    description: Optional[str] = Field(None, max_length=500, description="Dashboard description")

    # Configuration
    layout: Dict[str, Any] = Field(..., description="Dashboard layout configuration")
    widgets: List[Dict[str, Any]] = Field(..., description="Dashboard widgets")
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default dashboard filters"
    )

    # Access control
    tenant_id: Optional[UUID] = Field(None, description="Tenant ID (if tenant-specific)")
    owner_id: UUID = Field(..., description="Dashboard owner")
    is_public: bool = Field(default=False, description="Public dashboard")
    shared_with: List[UUID] = Field(
        default_factory=list,
        description="Users with access"
    )

    # Settings
    auto_refresh: bool = Field(default=True, description="Auto-refresh enabled")
    refresh_interval: int = Field(default=300, description="Refresh interval in seconds")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_viewed: Optional[datetime] = Field(None, description="Last view timestamp")


class AlertRule(BaseModel):
    """Analytics alert rule model."""

    id: UUID = Field(..., description="Alert rule ID")
    name: str = Field(..., min_length=1, max_length=100, description="Rule name")
    description: Optional[str] = Field(None, max_length=500, description="Rule description")

    # Rule configuration
    metric: str = Field(..., description="Metric to monitor")
    condition: str = Field(..., description="Alert condition (>, <, =, etc.)")
    threshold: Union[int, float] = Field(..., description="Alert threshold value")

    # Evaluation
    evaluation_window: int = Field(
        default=300,
        description="Evaluation window in seconds"
    )
    evaluation_frequency: int = Field(
        default=60,
        description="How often to evaluate (seconds)"
    )

    # Thresholds
    warning_threshold: Optional[Union[int, float]] = Field(
        None,
        description="Warning threshold"
    )
    critical_threshold: Optional[Union[int, float]] = Field(
        None,
        description="Critical threshold"
    )

    # Scope
    tenant_id: Optional[UUID] = Field(None, description="Tenant-specific rule")
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional filters"
    )

    # Notification
    notification_channels: List[str] = Field(
        default_factory=list,
        description="Notification channels"
    )
    cooldown_period: int = Field(
        default=300,
        description="Cooldown period between alerts (seconds)"
    )

    # State
    is_active: bool = Field(default=True, description="Rule is active")
    last_triggered: Optional[datetime] = Field(None, description="Last trigger timestamp")
    trigger_count: int = Field(default=0, description="Number of times triggered")

    # Metadata
    created_by: UUID = Field(..., description="User who created the rule")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ClickTrackRequest(BaseModel):
    """Request to track a search result click."""

    search_id: str = Field(..., description="Search execution identifier")
    profile_id: str = Field(..., description="Profile that was clicked")
    position: int = Field(..., ge=0, description="Position in results (0-based)")

    # Optional enrichment
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance score")
    context: Optional[Dict[str, Any]] = Field(None, description="Click context data")

    class Config:
        json_schema_extra = {
            "example": {
                "search_id": "search_20251013_142530_abc123de",
                "profile_id": "550e8400-e29b-41d4-a716-446655440000",
                "position": 2,
                "relevance_score": 0.87,
                "context": {
                    "session_id": "sess_xyz789",
                    "device_type": "desktop",
                    "time_to_click_ms": 3500,
                    "previous_clicks": 1
                }
            }
        }


class ClickAnalyticsResponse(BaseModel):
    """Analytics summary for clicks."""

    search_id: str
    total_clicks: int
    unique_profiles: int
    avg_position: float
    top_3_clicks: int
    position_distribution: Dict[int, int]
    engagement_signals: Dict[str, int]


class CTRReportItem(BaseModel):
    """Single time bucket in CTR report."""

    time_bucket: datetime
    click_count: int
    search_count: int
    ctr: float
    user_count: int
    avg_position: float


class PositionPerformance(BaseModel):
    """Click performance for a specific position."""

    position: int
    click_count: int
    avg_time_to_click_ms: Optional[float]
    avg_rank_quality: float
    ctr: float


__all__ = [
    "MetricType",
    "AggregationType",
    "TimeInterval",
    "UsageEvent",
    "Usage",
    "Metrics",
    "AnalyticsData",
    "TenantAnalytics",
    "SystemAnalytics",
    "ReportRequest",
    "ReportResponse",
    "Dashboard",
    "AlertRule",
    "ClickTrackRequest",
    "ClickAnalyticsResponse",
    "CTRReportItem",
    "PositionPerformance",
]