"""
Infrastructure persistence models module.

This module contains database table definitions following hexagonal architecture,
separated from domain models and business logic.
"""

from app.infrastructure.persistence.models.audit_table import (
    AuditEventType,
    AuditLog,
    AuditLogTable,
)
from app.infrastructure.persistence.models.auth_tables import (
    APIKeyTable,
    User,
    UserRole,
    UserSessionTable,
    UserTable,
)
from app.infrastructure.persistence.models.document_processing_table import (
    DocumentProcessing,
    DocumentProcessingTable,
)
from app.infrastructure.persistence.models.embedding_table import (
    EmbeddingTable,
)
from app.infrastructure.persistence.models.profile_table import (
    Award,
    Certification,
    ContactInfo,
    Education,
    Experience,
    Language,
    Location,
    ProcessingMetadata,
    ProfileAnalytics,
    ProfileData,
    ProfileEmbeddings,
    ProfileMetadata,
    ProfileQuality,
    ProfileTable,
    Project,
    Publication,
    PrivacySettings,
    Skill,
)
from app.infrastructure.persistence.models.retry_table import (
    DeadLetterModel,
    RetryAttemptModel,
    RetryStateModel,
)
from app.infrastructure.persistence.models.tenant_table import (
    QuotaLimits,
    SubscriptionTier,
    Tenant,
    TenantTable,
)
from app.infrastructure.persistence.models.webhook_table import (
    WebhookDeliveryTable,
    WebhookEndpointTable,
)
from app.infrastructure.persistence.models.query_expansion_table import (
    QueryExpansionTable,
)

__all__ = [
    # Audit
    "AuditEventType",
    "AuditLog",
    "AuditLogTable",
    # Auth
    "APIKeyTable",
    "User",
    "UserRole",
    "UserSessionTable",
    "UserTable",
    # Document Processing
    "DocumentProcessing",
    "DocumentProcessingTable",
    # Embedding
    "EmbeddingTable",
    # Profile and related models
    "Award",
    "Certification",
    "ContactInfo",
    "Education",
    "Experience",
    "Language",
    "Location",
    "ProcessingMetadata",
    "ProfileAnalytics",
    "ProfileData",
    "ProfileEmbeddings",
    "ProfileMetadata",
    "ProfileQuality",
    "ProfileTable",
    "PrivacySettings",
    "Project",
    "Publication",
    "Skill",
    # Retry
    "DeadLetterModel",
    "RetryAttemptModel",
    "RetryStateModel",
    # Tenant
    "QuotaLimits",
    "SubscriptionTier",
    "Tenant",
    "TenantTable",
    # Webhook
    "WebhookDeliveryTable",
    "WebhookEndpointTable",
    # Query expansions
    "QueryExpansionTable",
]
