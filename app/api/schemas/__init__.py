"""
API Schemas - DTOs for REST API following hexagonal architecture.

This module contains all request/response models for the API layer.
These are separated from domain entities and persistence tables.

All schemas use Pydantic BaseModel and are organized by feature area:
- analytics_schemas: Analytics and reporting DTOs
- document_schemas: Document processing DTOs
- job_schemas: Background job processing DTOs
- notification_schemas: Notification and messaging DTOs
- profile_schemas: Profile management DTOs
- search_schemas: Search and matching DTOs
- upload_schemas: File upload DTOs
"""

# Analytics schemas
from app.api.schemas.analytics_schemas import (
    AggregationType,
    AlertRule,
    AnalyticsData,
    Dashboard,
    Metrics,
    MetricType,
    ReportRequest,
    ReportResponse,
    SystemAnalytics,
    TenantAnalytics,
    TimeInterval,
    Usage,
    UsageEvent,
)

# Document schemas
from app.api.schemas.document_schemas import (
    BulkUploadRequest,
    BulkUploadResponse,
    Document,
    DocumentAnalytics,
    DocumentContent,
    DocumentCreate,
    DocumentFormat,
    DocumentProcessingError,
    DocumentResponse,
    DocumentSearchFilters,
    DocumentStats,
    DocumentStatus,
    DocumentType,
    DocumentUpdate,
    ExtractionMethod,
)

# Job schemas
from app.api.schemas.job_schemas import (
    Job,
    JobBatch,
    JobCreate,
    JobPriority,
    JobProgress,
    JobQueue,
    JobResult,
    JobStats,
    JobStatus,
    JobType,
    JobUpdate,
    RetryStrategy,
)

# Notification schemas
from app.api.schemas.notification_schemas import (
    BulkNotificationRequest,
    Notification,
    NotificationAnalytics,
    NotificationChannel,
    NotificationCreate,
    NotificationDelivery,
    NotificationDigest,
    NotificationPreferences,
    NotificationPriority,
    NotificationStats,
    NotificationStatus,
    NotificationTemplate,
    NotificationType,
    NotificationUpdate,
    WebhookNotification,
)

# Profile schemas
from app.api.schemas.profile_schemas import (
    BulkOperationRequest,
    ProfileAnalyticsSummary,
    ProfileCreate,
    ProfileListResponse,
    ProfileResponse,
    ProfileSearchFilters,
    ProfileSummary,
    ProfileUpdate,
)

# Search schemas
from app.api.schemas.search_schemas import (
    EducationRequirement,
    ExperienceRequirement,
    FacetValue,
    FilterOperator,
    LocationFilter,
    MatchScore,
    RangeFilter,
    SalaryFilter,
    SavedSearch,
    SearchAnalytics,
    SearchFacet,
    SearchFilter,
    SearchHistory,
    SearchMode,
    SearchRequest,
    SearchResult,
    SearchResponse,
    SkillRequirement,
    SortOrder,
)

# Upload schemas
from app.api.schemas.upload_schemas import (
    BatchUploadResponse,
    ProcessingStatusResponse,
    UploadHistoryItem,
    UploadResponse,
)

# Webhook schemas
from app.api.schemas.webhook_schemas import (
    CircuitBreakerState,
    RetryPolicy,
    WebhookDeadLetterResponse,
    WebhookDeliveryQuery,
    WebhookDeliveryResponse,
    WebhookDeliveryStatus,
    WebhookEndpointCreate,
    WebhookEndpointResponse,
    WebhookEndpointUpdate,
    WebhookEventPayload,
    WebhookEventType,
    WebhookFailureReason,
    WebhookRetryRequest,
    WebhookStatsResponse,
    WebhookTestRequest,
)


__all__ = [
    # Analytics
    "AggregationType",
    "AlertRule",
    "AnalyticsData",
    "Dashboard",
    "Metrics",
    "MetricType",
    "ReportRequest",
    "ReportResponse",
    "SystemAnalytics",
    "TenantAnalytics",
    "TimeInterval",
    "Usage",
    "UsageEvent",
    # Documents
    "BulkUploadRequest",
    "BulkUploadResponse",
    "Document",
    "DocumentAnalytics",
    "DocumentContent",
    "DocumentCreate",
    "DocumentFormat",
    "DocumentProcessingError",
    "DocumentResponse",
    "DocumentSearchFilters",
    "DocumentStats",
    "DocumentStatus",
    "DocumentType",
    "DocumentUpdate",
    "ExtractionMethod",
    # Jobs
    "Job",
    "JobBatch",
    "JobCreate",
    "JobPriority",
    "JobProgress",
    "JobQueue",
    "JobResult",
    "JobStats",
    "JobStatus",
    "JobType",
    "JobUpdate",
    "RetryStrategy",
    # Notifications
    "BulkNotificationRequest",
    "Notification",
    "NotificationAnalytics",
    "NotificationChannel",
    "NotificationCreate",
    "NotificationDelivery",
    "NotificationDigest",
    "NotificationPreferences",
    "NotificationPriority",
    "NotificationStats",
    "NotificationStatus",
    "NotificationTemplate",
    "NotificationType",
    "NotificationUpdate",
    "WebhookNotification",
    # Profiles
    "ProfileCreate",
    "BulkOperationRequest",
    "ProfileListResponse",
    "ProfileResponse",
    "ProfileSearchFilters",
    "ProfileSummary",
    "ProfileUpdate",
    "ProfileAnalyticsSummary",
    # Search
    "EducationRequirement",
    "ExperienceRequirement",
    "FacetValue",
    "FilterOperator",
    "LocationFilter",
    "MatchScore",
    "RangeFilter",
    "SalaryFilter",
    "SavedSearch",
    "SearchAnalytics",
    "SearchFacet",
    "SearchFilter",
    "SearchHistory",
    "SearchMode",
    "SearchRequest",
    "SearchResult",
    "SearchResponse",
    "SkillRequirement",
    "SortOrder",
    # Upload
    "BatchUploadResponse",
    "ProcessingStatusResponse",
    "UploadHistoryItem",
    "UploadResponse",
    # Webhooks
    "CircuitBreakerState",
    "RetryPolicy",
    "WebhookDeadLetterResponse",
    "WebhookDeliveryQuery",
    "WebhookDeliveryResponse",
    "WebhookDeliveryStatus",
    "WebhookEndpointCreate",
    "WebhookEndpointResponse",
    "WebhookEndpointUpdate",
    "WebhookEventPayload",
    "WebhookEventType",
    "WebhookFailureReason",
    "WebhookRetryRequest",
    "WebhookStatsResponse",
    "WebhookTestRequest",
]
