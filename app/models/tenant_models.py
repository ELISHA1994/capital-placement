"""
SQLModel Tenant Models with comprehensive multi-tenant configuration.

Multi-tenant configuration and management models:
- Tenant configuration with search settings and quotas
- Subscription tiers with feature access control
- Index strategies (shared vs dedicated)
- Performance monitoring and resource usage
- Billing and usage tracking
- Full database persistence with SQLModel
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from uuid import UUID, uuid4

from sqlalchemy import Column, String, Boolean, Integer, Float, Date, Numeric, Text, Index
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID, JSONB
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql import func
from sqlmodel import Field, SQLModel, Relationship
from pydantic import field_validator, computed_field, ConfigDict

from .base import AuditableModel, BaseModel, TimestampedModel


class SubscriptionTier(str, Enum):
    """Subscription tier levels with different feature access"""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class IndexStrategy(str, Enum):
    """Multi-tenant index isolation strategies"""
    SHARED = "shared"        # Shared index with tenant filtering
    DEDICATED = "dedicated"  # Dedicated index per tenant
    HYBRID = "hybrid"        # Mix of shared and dedicated based on usage


class ProcessingPriority(str, Enum):
    """Document processing priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class SearchConfiguration(BaseModel):
    """Tenant-specific search configuration"""
    
    # Search behavior settings
    default_search_mode: str = Field(default="hybrid", description="Default search mode")
    enable_vector_search: bool = Field(default=True, description="Enable vector search")
    enable_semantic_search: bool = Field(default=True, description="Enable semantic search")
    enable_query_expansion: bool = Field(default=True, description="Enable query expansion")
    
    # Result preferences
    default_page_size: int = Field(default=20, ge=1, le=100, description="Default results per page")
    max_results_per_search: int = Field(default=1000, ge=1, le=10000, description="Maximum results per search")
    min_match_score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum match score")
    
    # Performance settings
    search_timeout_seconds: int = Field(default=30, ge=5, le=300, description="Search timeout")
    enable_result_caching: bool = Field(default=True, description="Enable search result caching")
    cache_ttl_minutes: int = Field(default=60, ge=1, le=1440, description="Cache TTL in minutes")
    
    # Scoring weights
    skill_match_weight: float = Field(default=0.4, ge=0.0, le=1.0, description="Skill matching weight")
    experience_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Experience weight")
    education_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="Education weight")
    location_weight: float = Field(default=0.1, ge=0.0, le=1.0, description="Location weight")
    
    # Advanced features
    enable_reranking: bool = Field(default=True, description="Enable cross-encoder reranking")
    rerank_top_k: int = Field(default=50, ge=10, le=200, description="Number of results to rerank")
    enable_diversity: bool = Field(default=True, description="Enable result diversity")
    diversity_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Diversity threshold")


class FileTypeValidationConfig(BaseModel):
    """Configuration for file type validation and security policies"""
    
    # File type restrictions
    allowed_file_extensions: List[str] = Field(
        default_factory=lambda: [".pdf", ".doc", ".docx", ".txt"],
        description="Allowed file extensions"
    )
    
    # Size limits per file type (in MB)
    file_type_limits: Dict[str, int] = Field(
        default_factory=lambda: {
            ".pdf": 25,
            ".doc": 10,
            ".docx": 10,
            ".txt": 1
        },
        description="Maximum file size per type in MB"
    )
    
    # Content validation settings
    require_mime_validation: bool = Field(default=True, description="Require MIME type validation")
    require_signature_validation: bool = Field(default=True, description="Require file signature validation")
    enable_content_scanning: bool = Field(default=True, description="Enable security content scanning")
    
    # Security policies
    block_executable_content: bool = Field(default=True, description="Block files with executable content")
    block_macro_documents: bool = Field(default=True, description="Block documents with macros")
    block_script_content: bool = Field(default=True, description="Block files with script content")
    
    # Validation strictness
    validation_mode: str = Field(
        default="strict",
        description="Validation mode: strict, standard, or permissive"
    )
    min_confidence_score: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for validation"
    )
    
    # Error handling
    reject_on_validation_errors: bool = Field(default=True, description="Reject files with validation errors")
    reject_on_security_warnings: bool = Field(default=False, description="Reject files with security warnings")
    log_validation_details: bool = Field(default=True, description="Log detailed validation information")


class RateLimitConfiguration(BaseModel):
    """Advanced rate limiting configuration"""
    
    # Rate limiting behavior
    enable_rate_limiting: bool = Field(default=True, description="Enable rate limiting")
    enable_ip_rate_limiting: bool = Field(default=True, description="Enable IP-based rate limiting")
    enable_user_rate_limiting: bool = Field(default=True, description="Enable user-based rate limiting")
    enable_endpoint_rate_limiting: bool = Field(default=True, description="Enable endpoint-specific rate limiting")
    
    # DDoS protection
    enable_ddos_protection: bool = Field(default=True, description="Enable DDoS protection features")
    burst_limit_multiplier: float = Field(default=1.5, ge=1.0, le=5.0, description="Burst limit multiplier")
    
    # Admin and whitelist settings
    bypass_for_admins: bool = Field(default=True, description="Bypass rate limits for admin users")
    enable_ip_whitelist: bool = Field(default=True, description="Enable IP whitelist functionality")
    enable_user_whitelist: bool = Field(default=True, description="Enable user whitelist functionality")
    
    # Rate limiting strategy
    fail_open_on_errors: bool = Field(default=True, description="Allow requests if rate limiting fails")
    use_sliding_window: bool = Field(default=True, description="Use sliding window instead of fixed window")
    
    # Custom rate limit rules for specific endpoints
    custom_endpoint_limits: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Custom rate limits for specific endpoints"
    )
    
    # Monitoring and alerting
    enable_rate_limit_logging: bool = Field(default=True, description="Enable rate limit violation logging")
    enable_rate_limit_alerts: bool = Field(default=False, description="Enable rate limit alerts")
    alert_threshold_percentage: float = Field(default=80.0, ge=50.0, le=100.0, description="Alert when usage exceeds this percentage")


class ProcessingConfiguration(BaseModel):
    """Tenant-specific document processing configuration"""
    
    # Processing behavior
    auto_process_uploads: bool = Field(default=True, description="Auto-process uploaded documents")
    processing_priority: ProcessingPriority = Field(default=ProcessingPriority.NORMAL, description="Processing priority")
    enable_ocr: bool = Field(default=True, description="Enable OCR for scanned documents")
    enable_table_extraction: bool = Field(default=True, description="Enable table extraction")
    
    # Quality settings
    min_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum extraction confidence")
    require_manual_review: bool = Field(default=False, description="Require manual review for low confidence")
    enable_duplicate_detection: bool = Field(default=True, description="Enable duplicate CV detection")
    
    # Language settings
    supported_languages: List[str] = Field(
        default_factory=lambda: ["en", "es", "fr", "de"],
        description="Supported document languages"
    )
    auto_detect_language: bool = Field(default=True, description="Auto-detect document language")
    
    # Storage settings
    retain_original_documents: bool = Field(default=True, description="Keep original documents")
    document_retention_days: int = Field(default=2555, ge=30, description="Document retention period")
    compress_processed_content: bool = Field(default=True, description="Compress processed content")
    
    # File validation settings
    file_validation: FileTypeValidationConfig = Field(
        default_factory=FileTypeValidationConfig,
        description="File type validation configuration"
    )
    
    # Rate limiting settings
    rate_limit_config: RateLimitConfiguration = Field(
        default_factory=RateLimitConfiguration,
        description="Rate limiting configuration"
    )


class QuotaLimits(BaseModel):
    """Resource usage quotas and limits"""
    
    # Storage quotas
    max_profiles: Optional[int] = Field(None, ge=1, description="Maximum CV profiles")
    max_storage_gb: Optional[Decimal] = Field(None, ge=0, description="Maximum storage in GB")
    max_document_size_mb: int = Field(default=10, ge=1, le=100, description="Max document size in MB")
    
    # Processing quotas
    max_documents_per_day: Optional[int] = Field(None, ge=1, description="Max documents processed per day")
    max_documents_per_month: Optional[int] = Field(None, ge=1, description="Max documents per month")
    processing_timeout_minutes: int = Field(default=10, ge=1, le=60, description="Processing timeout")
    
    # Search quotas
    max_searches_per_day: Optional[int] = Field(None, ge=1, description="Max searches per day")
    max_searches_per_month: Optional[int] = Field(None, ge=1, description="Max searches per month")
    max_concurrent_searches: int = Field(default=10, ge=1, le=100, description="Max concurrent searches")
    
    # API quotas
    max_api_requests_per_minute: int = Field(default=100, ge=1, description="API rate limit per minute")
    max_api_requests_per_hour: int = Field(default=1000, ge=1, description="API rate limit per hour")
    max_api_requests_per_day: Optional[int] = Field(None, ge=1, description="API rate limit per day")
    
    # Upload-specific rate limits
    max_upload_requests_per_minute: int = Field(default=10, ge=1, description="Upload requests per minute")
    max_upload_requests_per_hour: int = Field(default=100, ge=1, description="Upload requests per hour")
    max_upload_requests_per_day: Optional[int] = Field(None, ge=1, description="Upload requests per day")
    
    # IP-based rate limits (for DDoS protection)
    max_requests_per_ip_per_minute: int = Field(default=60, ge=1, description="Requests per IP per minute")
    max_requests_per_ip_per_hour: int = Field(default=1000, ge=1, description="Requests per IP per hour")
    
    # User-specific rate limits
    max_requests_per_user_per_minute: int = Field(default=30, ge=1, description="Requests per user per minute")
    max_requests_per_user_per_hour: int = Field(default=500, ge=1, description="Requests per user per hour")
    
    # User quotas
    max_users: Optional[int] = Field(None, ge=1, description="Maximum number of users")
    max_admin_users: int = Field(default=5, ge=1, description="Maximum admin users")


class FeatureFlags(BaseModel):
    """Feature access control flags"""
    
    # Core features
    enable_advanced_search: bool = Field(default=True, description="Enable advanced search features")
    enable_bulk_operations: bool = Field(default=True, description="Enable bulk upload/processing")
    enable_export: bool = Field(default=True, description="Enable data export")
    enable_webhooks: bool = Field(default=False, description="Enable webhook notifications")
    
    # AI features
    enable_ai_recommendations: bool = Field(default=False, description="Enable AI-powered recommendations")
    enable_skill_extraction: bool = Field(default=True, description="Enable AI skill extraction")
    enable_sentiment_analysis: bool = Field(default=False, description="Enable sentiment analysis")
    enable_candidate_scoring: bool = Field(default=True, description="Enable AI candidate scoring")
    
    # Analytics features
    enable_analytics_dashboard: bool = Field(default=True, description="Enable analytics dashboard")
    enable_custom_reports: bool = Field(default=False, description="Enable custom reporting")
    enable_data_insights: bool = Field(default=False, description="Enable data insights")
    
    # Integration features
    enable_ats_integration: bool = Field(default=False, description="Enable ATS integrations")
    enable_crm_integration: bool = Field(default=False, description="Enable CRM integrations")
    enable_api_access: bool = Field(default=True, description="Enable API access")
    enable_sso: bool = Field(default=False, description="Enable SSO authentication")


class BillingConfiguration(BaseModel):
    """Billing and payment configuration"""
    
    billing_cycle: str = Field(default="monthly", description="Billing cycle (monthly, annual)")
    currency: str = Field(default="USD", description="Billing currency")
    
    # Pricing
    base_price: Decimal = Field(default=Decimal("0"), ge=0, description="Base subscription price")
    price_per_profile: Optional[Decimal] = Field(None, ge=0, description="Price per CV profile")
    price_per_search: Optional[Decimal] = Field(None, ge=0, description="Price per search")
    price_per_gb_storage: Optional[Decimal] = Field(None, ge=0, description="Price per GB storage")
    
    # Usage tracking
    track_profile_usage: bool = Field(default=True, description="Track profile usage for billing")
    track_search_usage: bool = Field(default=True, description="Track search usage")
    track_storage_usage: bool = Field(default=True, description="Track storage usage")
    track_api_usage: bool = Field(default=True, description="Track API usage")


class UsageMetrics(BaseModel):
    """Current usage metrics and statistics"""
    
    # Profile metrics
    total_profiles: int = Field(default=0, ge=0, description="Total CV profiles")
    active_profiles: int = Field(default=0, ge=0, description="Active profiles")
    profiles_added_this_month: int = Field(default=0, ge=0, description="Profiles added this month")
    
    # Search metrics
    total_searches: int = Field(default=0, ge=0, description="Total searches performed")
    searches_this_month: int = Field(default=0, ge=0, description="Searches this month")
    searches_today: int = Field(default=0, ge=0, description="Searches today")
    average_search_time_ms: Optional[float] = Field(None, description="Average search time")
    
    # Storage metrics
    storage_used_gb: Decimal = Field(default=Decimal("0"), ge=0, description="Storage used in GB")
    documents_processed: int = Field(default=0, ge=0, description="Total documents processed")
    documents_pending: int = Field(default=0, ge=0, description="Documents pending processing")
    
    # API metrics
    api_requests_today: int = Field(default=0, ge=0, description="API requests today")
    api_requests_this_month: int = Field(default=0, ge=0, description="API requests this month")
    
    # Performance metrics
    average_processing_time_seconds: Optional[float] = Field(None, description="Average processing time")
    success_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Processing success rate")
    
    # Last updated
    metrics_updated_at: datetime = Field(default_factory=datetime.utcnow, description="When metrics were last updated")


class TenantTable(TimestampedModel, table=True):
    """
    Core tenant model for basic tenant information.
    
    This is the primary tenant table that other tables reference with foreign keys.
    Contains essential tenant data used by the bootstrap service and authentication system.
    """
    __tablename__ = "tenants"
    
    # Primary key
    id: UUID = Field(
        default_factory=uuid4,
        sa_column=Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4),
        description="Unique tenant identifier"
    )
    
    # Basic tenant information - matching bootstrap service expectations
    name: str = Field(
        sa_column=Column(String(200), nullable=False, unique=True, index=True),
        description="Unique tenant name",
        min_length=1,
        max_length=200
    )
    slug: str = Field(
        sa_column=Column(String(200), nullable=False, unique=True, index=True),
        description="URL-friendly tenant slug",
        min_length=1,
        max_length=200
    )
    display_name: str = Field(
        sa_column=Column(String(200), nullable=False),
        description="Display name for UI",
        min_length=1,
        max_length=200
    )
    description: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
        description="Tenant description"
    )
    
    # Subscription and status
    subscription_tier: str = Field(
        default="free",
        sa_column=Column(String(20), nullable=False, default="free"),
        description="Subscription tier"
    )
    is_active: bool = Field(
        default=True,
        sa_column=Column(Boolean, nullable=False, default=True, index=True),
        description="Whether tenant is active"
    )
    is_system_tenant: bool = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False, default=False),
        description="Whether this is the system tenant for super admins"
    )
    
    # Contact information
    primary_contact_email: str = Field(
        sa_column=Column(String(255), nullable=False, index=True),
        description="Primary contact email"
    )
    
    # Settings - matching bootstrap service defaults
    data_region: str = Field(
        default="us-central",
        sa_column=Column(String(50), nullable=False, default="us-central"),
        description="Data residency region"
    )
    timezone: str = Field(
        default="UTC",
        sa_column=Column(String(50), nullable=False, default="UTC"),
        description="Tenant timezone"
    )
    locale: str = Field(
        default="en-US",
        sa_column=Column(String(10), nullable=False, default="en-US"),
        description="Tenant locale"
    )
    date_format: str = Field(
        default="YYYY-MM-DD",
        sa_column=Column(String(20), nullable=False, default="YYYY-MM-DD"),
        description="Preferred date format"
    )
    
    # Additional indexes for performance
    __table_args__ = (
        Index("idx_tenants_name", "name"),
        Index("idx_tenants_active", "is_active"),
        Index("idx_tenants_system", "is_system_tenant"),
        Index("idx_tenants_contact", "primary_contact_email"),
    )
    
    # Relationships (required for SQLModel to create foreign key constraints)
    users: List["UserTable"] = Relationship(back_populates="tenant")
    user_sessions: List["UserSessionTable"] = Relationship(back_populates="tenant")
    api_keys: List["APIKeyTable"] = Relationship(back_populates="tenant")
    profiles: List["ProfileTable"] = Relationship(back_populates="tenant")
    embeddings: List["EmbeddingTable"] = Relationship(back_populates="tenant")
    audit_logs: List["AuditLogTable"] = Relationship(back_populates="tenant")


class TenantConfigurationTable(TimestampedModel, table=True):
    """
    Complete tenant configuration and management with database persistence.
    
    Manages all aspects of a tenant's configuration including:
    - Subscription tier and feature access
    - Search and processing configurations  
    - Resource quotas and usage tracking
    - Index strategy and performance settings
    - Billing and payment information
    """
    __tablename__ = "tenant_configurations"
    
    # Primary key
    id: UUID = Field(
        default_factory=uuid4,
        sa_column=Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4),
        description="Unique tenant identifier"
    )
    
    # Additional indexes for performance
    __table_args__ = (
        Index("idx_tenant_configurations_name", "name"),
        Index("idx_tenant_configurations_subscription_tier", "subscription_tier"),
        Index("idx_tenant_configurations_active", "is_active"),
        Index("idx_tenant_configurations_status", "is_active", "is_suspended"),
        Index("idx_tenant_configurations_subscription_dates", "subscription_start_date", "subscription_end_date"),
        Index("idx_tenant_configurations_contact", "primary_contact_email"),
    )
    
    # Basic tenant information
    name: str = Field(
        sa_column=Column(String(200), nullable=False, unique=True, index=True),
        description="Unique tenant name",
        min_length=1,
        max_length=200
    )
    display_name: str = Field(
        sa_column=Column(String(200), nullable=False),
        description="Display name for UI",
        min_length=1,
        max_length=200
    )
    description: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
        description="Tenant description"
    )
    
    # Subscription and billing
    subscription_tier: SubscriptionTier = Field(
        default=SubscriptionTier.FREE,
        sa_column=Column(String(20), nullable=False, default=SubscriptionTier.FREE, index=True),
        description="Subscription tier"
    )
    subscription_start_date: date = Field(
        default_factory=date.today,
        sa_column=Column(Date, nullable=False, default=func.current_date()),
        description="Subscription start date"
    )
    subscription_end_date: Optional[date] = Field(
        default=None,
        sa_column=Column(Date, nullable=True),
        description="Subscription end date"
    )
    
    # Configuration stored as JSONB for flexibility
    billing_configuration: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default={}),
        description="Billing configuration as JSONB"
    )
    search_configuration: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default={}),
        description="Search configuration as JSONB"
    )
    processing_configuration: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default={}),
        description="Processing configuration as JSONB"
    )
    quota_limits: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default={}),
        description="Quota limits as JSONB"
    )
    feature_flags: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default={}),
        description="Feature flags as JSONB"
    )
    
    # Infrastructure settings
    index_strategy: IndexStrategy = Field(
        default=IndexStrategy.SHARED,
        sa_column=Column(String(20), nullable=False, default=IndexStrategy.SHARED),
        description="Index isolation strategy"
    )
    dedicated_search_index: Optional[str] = Field(
        default=None,
        sa_column=Column(String(100), nullable=True),
        description="Dedicated search index name"
    )
    data_region: str = Field(
        default="us-central",
        sa_column=Column(String(50), nullable=False, default="us-central"),
        description="Data residency region"
    )
    
    # Status and health
    is_active: bool = Field(
        default=True,
        sa_column=Column(Boolean, nullable=False, default=True, index=True),
        description="Whether tenant is active"
    )
    is_suspended: bool = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False, default=False, index=True),
        description="Whether tenant is suspended"
    )
    suspension_reason: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
        description="Reason for suspension"
    )
    is_system_tenant: bool = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False, default=False),
        description="Whether this is the system tenant for super admins"
    )
    
    # Usage tracking stored as JSONB
    current_usage: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default={}),
        description="Current usage metrics as JSONB"
    )
    
    # Contact information
    primary_contact_email: str = Field(
        sa_column=Column(String(255), nullable=False, index=True),
        description="Primary contact email"
    )
    billing_contact_email: Optional[str] = Field(
        default=None,
        sa_column=Column(String(255), nullable=True),
        description="Billing contact email"
    )
    technical_contact_email: Optional[str] = Field(
        default=None,
        sa_column=Column(String(255), nullable=True),
        description="Technical contact email"
    )
    
    # Settings
    timezone: str = Field(
        default="UTC",
        sa_column=Column(String(50), nullable=False, default="UTC"),
        description="Tenant timezone"
    )
    locale: str = Field(
        default="en-US",
        sa_column=Column(String(10), nullable=False, default="en-US"),
        description="Tenant locale"
    )
    date_format: str = Field(
        default="YYYY-MM-DD",
        sa_column=Column(String(20), nullable=False, default="YYYY-MM-DD"),
        description="Preferred date format"
    )
    
    @field_validator('subscription_end_date')
    @classmethod
    def validate_subscription_dates(cls, v, info):
        """Ensure end_date is after start_date"""
        if v is not None and 'subscription_start_date' in info.data:
            start_date = info.data['subscription_start_date']
            if v < start_date:
                raise ValueError('Subscription end date must be after start date')
        return v
    
    @hybrid_property
    def is_subscription_active(self) -> bool:
        """Check if subscription is currently active"""
        if not self.is_active or self.is_suspended:
            return False
        
        today = date.today()
        if self.subscription_end_date and today > self.subscription_end_date:
            return False
        
        return True
    
    @hybrid_property
    def days_until_expiry(self) -> Optional[int]:
        """Days until subscription expires"""
        if not self.subscription_end_date:
            return None
        
        delta = self.subscription_end_date - date.today()
        return delta.days
    
    @hybrid_property
    def subscription_status(self) -> str:
        """Human-readable subscription status"""
        if not self.is_active:
            return "inactive"
        elif self.is_suspended:
            return "suspended"
        elif not self.is_subscription_active:
            return "expired"
        elif self.days_until_expiry is not None and self.days_until_expiry <= 7:
            return "expiring_soon"
        else:
            return "active"
    
    # Business logic methods for configuration access
    def get_search_configuration(self) -> SearchConfiguration:
        """Get search configuration as SearchConfiguration model."""
        default_config = SearchConfiguration()
        if self.search_configuration:
            return SearchConfiguration.model_validate({**default_config.model_dump(), **self.search_configuration})
        return default_config
    
    def set_search_configuration(self, config: SearchConfiguration) -> None:
        """Set search configuration from SearchConfiguration model."""
        self.search_configuration = config.model_dump(exclude_none=True)
        self.update_timestamp()
    
    def get_processing_configuration(self) -> ProcessingConfiguration:
        """Get processing configuration as ProcessingConfiguration model."""
        default_config = ProcessingConfiguration()
        if self.processing_configuration:
            return ProcessingConfiguration.model_validate({**default_config.model_dump(), **self.processing_configuration})
        return default_config
    
    def set_processing_configuration(self, config: ProcessingConfiguration) -> None:
        """Set processing configuration from ProcessingConfiguration model."""
        self.processing_configuration = config.model_dump(exclude_none=True)
        self.update_timestamp()
    
    def get_quota_limits(self) -> QuotaLimits:
        """Get quota limits as QuotaLimits model."""
        default_limits = QuotaLimits()
        if self.quota_limits:
            return QuotaLimits.model_validate({**default_limits.model_dump(), **self.quota_limits})
        return default_limits
    
    def set_quota_limits(self, limits: QuotaLimits) -> None:
        """Set quota limits from QuotaLimits model."""
        self.quota_limits = limits.model_dump(exclude_none=True)
        self.update_timestamp()
    
    def get_feature_flags(self) -> FeatureFlags:
        """Get feature flags as FeatureFlags model."""
        default_flags = FeatureFlags()
        if self.feature_flags:
            return FeatureFlags.model_validate({**default_flags.model_dump(), **self.feature_flags})
        return default_flags
    
    def set_feature_flags(self, flags: FeatureFlags) -> None:
        """Set feature flags from FeatureFlags model."""
        self.feature_flags = flags.model_dump(exclude_none=True)
        self.update_timestamp()
    
    def get_billing_configuration(self) -> BillingConfiguration:
        """Get billing configuration as BillingConfiguration model."""
        default_config = BillingConfiguration()
        if self.billing_configuration:
            return BillingConfiguration.model_validate({**default_config.model_dump(), **self.billing_configuration})
        return default_config
    
    def set_billing_configuration(self, config: BillingConfiguration) -> None:
        """Set billing configuration from BillingConfiguration model."""
        self.billing_configuration = config.model_dump(exclude_none=True)
        self.update_timestamp()
    
    def get_usage_metrics(self) -> UsageMetrics:
        """Get usage metrics as UsageMetrics model."""
        default_metrics = UsageMetrics()
        if self.current_usage:
            return UsageMetrics.model_validate({**default_metrics.model_dump(), **self.current_usage})
        return default_metrics
    
    def set_usage_metrics(self, metrics: UsageMetrics) -> None:
        """Set usage metrics from UsageMetrics model."""
        self.current_usage = metrics.model_dump(exclude_none=True)
        self.update_timestamp()
    
    def check_quota_limit(self, resource_type: str, current_usage: int) -> bool:
        """Check if current usage is within quota limits"""
        limits = self.get_quota_limits()
        
        quota_map = {
            "profiles": limits.max_profiles,
            "searches_per_day": limits.max_searches_per_day,
            "searches_per_month": limits.max_searches_per_month,
            "documents_per_day": limits.max_documents_per_day,
            "documents_per_month": limits.max_documents_per_month,
            "api_requests_per_minute": limits.max_api_requests_per_minute,
            "api_requests_per_hour": limits.max_api_requests_per_hour,
            "api_requests_per_day": limits.max_api_requests_per_day,
            "users": limits.max_users
        }
        
        quota = quota_map.get(resource_type)
        if quota is None:
            return True  # No limit set
        
        return current_usage < quota
    
    def has_feature(self, feature_name: str) -> bool:
        """Check if tenant has access to a specific feature"""
        flags = self.get_feature_flags()
        return getattr(flags, f"enable_{feature_name}", False)
    
    def update_usage_metrics(self, metrics_update: Dict[str, Any]) -> None:
        """Update usage metrics with new values"""
        current_metrics = self.get_usage_metrics()
        
        # Update metrics
        for key, value in metrics_update.items():
            if hasattr(current_metrics, key):
                setattr(current_metrics, key, value)
        
        # Update timestamp
        current_metrics.metrics_updated_at = datetime.utcnow()
        
        # Save back to database
        self.set_usage_metrics(current_metrics)
    
    def get_effective_limits(self) -> Dict[str, Any]:
        """Get effective limits based on subscription tier"""
        # Base limits by tier
        tier_limits = {
            SubscriptionTier.FREE: {
                "max_profiles": 100,
                "max_searches_per_day": 50,
                "max_storage_gb": 1,
                "max_users": 1
            },
            SubscriptionTier.BASIC: {
                "max_profiles": 1000,
                "max_searches_per_day": 500,
                "max_storage_gb": 10,
                "max_users": 5
            },
            SubscriptionTier.PROFESSIONAL: {
                "max_profiles": 10000,
                "max_searches_per_day": 2000,
                "max_storage_gb": 100,
                "max_users": 25
            },
            SubscriptionTier.ENTERPRISE: {
                "max_profiles": None,  # Unlimited
                "max_searches_per_day": None,
                "max_storage_gb": None,
                "max_users": None
            }
        }
        
        base_limits = tier_limits.get(self.subscription_tier, tier_limits[SubscriptionTier.FREE])
        
        # Override with custom quota limits if set
        custom_limits = self.get_quota_limits().model_dump(exclude_none=True)
        
        return {**base_limits, **custom_limits}


# Keep original TenantConfiguration model for backward compatibility
class TenantConfiguration(TimestampedModel):
    """
    Backward compatibility wrapper for TenantConfigurationTable.
    
    Manages all aspects of a tenant's configuration including:
    - Subscription tier and feature access
    - Search and processing configurations  
    - Resource quotas and usage tracking
    - Index strategy and performance settings
    - Billing and payment information
    """
    
    # Unique identifier
    id: UUID = Field(
        default_factory=lambda: uuid4(),
        description="Unique tenant identifier"
    )
    
    # Basic tenant information
    name: str = Field(..., description="Tenant name", min_length=1, max_length=200)
    display_name: str = Field(..., description="Display name for UI", min_length=1, max_length=200)
    description: Optional[str] = Field(None, description="Tenant description")
    
    # Subscription and billing
    subscription_tier: SubscriptionTier = Field(default=SubscriptionTier.FREE, description="Subscription tier")
    subscription_start_date: date = Field(default_factory=date.today, description="Subscription start date")
    subscription_end_date: Optional[date] = Field(None, description="Subscription end date")
    billing_configuration: BillingConfiguration = Field(default_factory=BillingConfiguration)
    
    # Technical configuration
    search_configuration: SearchConfiguration = Field(default_factory=SearchConfiguration)
    processing_configuration: ProcessingConfiguration = Field(default_factory=ProcessingConfiguration)
    quota_limits: QuotaLimits = Field(default_factory=QuotaLimits)
    feature_flags: FeatureFlags = Field(default_factory=FeatureFlags)
    
    # Infrastructure settings
    index_strategy: IndexStrategy = Field(default=IndexStrategy.SHARED, description="Index isolation strategy")
    dedicated_search_index: Optional[str] = Field(None, description="Dedicated search index name")
    data_region: str = Field(default="us-central", description="Data residency region")
    
    # Status and health
    is_active: bool = Field(default=True, description="Whether tenant is active")
    is_suspended: bool = Field(default=False, description="Whether tenant is suspended")
    suspension_reason: Optional[str] = Field(None, description="Reason for suspension")
    is_system_tenant: bool = Field(default=False, description="Whether this is the system tenant for super admins")
    
    # Usage tracking
    current_usage: UsageMetrics = Field(default_factory=UsageMetrics, description="Current usage metrics")
    
    # Contact information
    primary_contact_email: str = Field(..., description="Primary contact email")
    billing_contact_email: Optional[str] = Field(None, description="Billing contact email")
    technical_contact_email: Optional[str] = Field(None, description="Technical contact email")
    
    # Settings
    timezone: str = Field(default="UTC", description="Tenant timezone")
    locale: str = Field(default="en-US", description="Tenant locale")
    date_format: str = Field(default="YYYY-MM-DD", description="Preferred date format")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "acme-corp",
                "display_name": "ACME Corporation", 
                "subscription_tier": "professional",
                "primary_contact_email": "admin@acme.com",
                "search_configuration": {
                    "default_search_mode": "hybrid",
                    "max_results_per_search": 500
                },
                "quota_limits": {
                    "max_profiles": 10000,
                    "max_searches_per_day": 1000
                }
            }
        }
    )
    
    @field_validator('subscription_end_date')
    @classmethod
    def validate_subscription_dates(cls, v, info):
        """Ensure end_date is after start_date"""
        if v is not None and 'subscription_start_date' in info.data:
            start_date = info.data['subscription_start_date']
            if v < start_date:
                raise ValueError('Subscription end date must be after start date')
        return v
    
    @computed_field
    @property
    def is_subscription_active(self) -> bool:
        """Check if subscription is currently active"""
        if not self.is_active or self.is_suspended:
            return False
        
        today = date.today()
        if self.subscription_end_date and today > self.subscription_end_date:
            return False
        
        return True
    
    @computed_field
    @property
    def days_until_expiry(self) -> Optional[int]:
        """Days until subscription expires"""
        if not self.subscription_end_date:
            return None
        
        delta = self.subscription_end_date - date.today()
        return delta.days
    
    @computed_field
    @property
    def subscription_status(self) -> str:
        """Human-readable subscription status"""
        if not self.is_active:
            return "inactive"
        elif self.is_suspended:
            return "suspended"
        elif not self.is_subscription_active:
            return "expired"
        elif self.days_until_expiry is not None and self.days_until_expiry <= 7:
            return "expiring_soon"
        else:
            return "active"
    
    def check_quota_limit(self, resource_type: str, current_usage: int) -> bool:
        """Check if current usage is within quota limits"""
        limits = self.quota_limits
        
        quota_map = {
            "profiles": limits.max_profiles,
            "searches_per_day": limits.max_searches_per_day,
            "searches_per_month": limits.max_searches_per_month,
            "documents_per_day": limits.max_documents_per_day,
            "documents_per_month": limits.max_documents_per_month,
            "api_requests_per_minute": limits.max_api_requests_per_minute,
            "api_requests_per_hour": limits.max_api_requests_per_hour,
            "api_requests_per_day": limits.max_api_requests_per_day,
            "users": limits.max_users
        }
        
        quota = quota_map.get(resource_type)
        if quota is None:
            return True  # No limit set
        
        return current_usage < quota
    
    def has_feature(self, feature_name: str) -> bool:
        """Check if tenant has access to a specific feature"""
        return getattr(self.feature_flags, f"enable_{feature_name}", False)
    
    def update_usage_metrics(self, metrics_update: Dict[str, Any]) -> None:
        """Update usage metrics with new values"""
        for key, value in metrics_update.items():
            if hasattr(self.current_usage, key):
                setattr(self.current_usage, key, value)
        
        self.current_usage.metrics_updated_at = datetime.utcnow()
        self.update_timestamp()
    
    def get_effective_limits(self) -> Dict[str, Any]:
        """Get effective limits based on subscription tier"""
        # Base limits by tier
        tier_limits = {
            SubscriptionTier.FREE: {
                "max_profiles": 100,
                "max_searches_per_day": 50,
                "max_storage_gb": 1,
                "max_users": 1
            },
            SubscriptionTier.BASIC: {
                "max_profiles": 1000,
                "max_searches_per_day": 500,
                "max_storage_gb": 10,
                "max_users": 5
            },
            SubscriptionTier.PROFESSIONAL: {
                "max_profiles": 10000,
                "max_searches_per_day": 2000,
                "max_storage_gb": 100,
                "max_users": 25
            },
            SubscriptionTier.ENTERPRISE: {
                "max_profiles": None,  # Unlimited
                "max_searches_per_day": None,
                "max_storage_gb": None,
                "max_users": None
            }
        }
        
        base_limits = tier_limits.get(self.subscription_tier, tier_limits[SubscriptionTier.FREE])
        
        # Override with custom quota limits if set
        custom_limits = self.quota_limits.model_dump(exclude_none=True)
        
        return {**base_limits, **custom_limits}
    
    @classmethod
    def from_table(cls, tenant_table: TenantConfigurationTable) -> "TenantConfiguration":
        """Convert TenantConfigurationTable to TenantConfiguration for backward compatibility."""
        return cls(
            id=tenant_table.id,
            name=tenant_table.name,
            display_name=tenant_table.display_name,
            description=tenant_table.description,
            subscription_tier=tenant_table.subscription_tier,
            subscription_start_date=tenant_table.subscription_start_date,
            subscription_end_date=tenant_table.subscription_end_date,
            billing_configuration=tenant_table.get_billing_configuration(),
            search_configuration=tenant_table.get_search_configuration(),
            processing_configuration=tenant_table.get_processing_configuration(),
            quota_limits=tenant_table.get_quota_limits(),
            feature_flags=tenant_table.get_feature_flags(),
            index_strategy=tenant_table.index_strategy,
            dedicated_search_index=tenant_table.dedicated_search_index,
            data_region=tenant_table.data_region,
            is_active=tenant_table.is_active,
            is_suspended=tenant_table.is_suspended,
            suspension_reason=tenant_table.suspension_reason,
            is_system_tenant=tenant_table.is_system_tenant,
            current_usage=tenant_table.get_usage_metrics(),
            primary_contact_email=tenant_table.primary_contact_email,
            billing_contact_email=tenant_table.billing_contact_email,
            technical_contact_email=tenant_table.technical_contact_email,
            timezone=tenant_table.timezone,
            locale=tenant_table.locale,
            date_format=tenant_table.date_format,
            created_at=tenant_table.created_at,
            updated_at=tenant_table.updated_at
        )
    
    def to_table(self) -> TenantConfigurationTable:
        """Convert TenantConfiguration to TenantConfigurationTable for database storage."""
        tenant_table = TenantConfigurationTable(
            id=self.id,
            name=self.name,
            display_name=self.display_name,
            description=self.description,
            subscription_tier=self.subscription_tier,
            subscription_start_date=self.subscription_start_date,
            subscription_end_date=self.subscription_end_date,
            index_strategy=self.index_strategy,
            dedicated_search_index=self.dedicated_search_index,
            data_region=self.data_region,
            is_active=self.is_active,
            is_suspended=self.is_suspended,
            suspension_reason=self.suspension_reason,
            is_system_tenant=self.is_system_tenant,
            primary_contact_email=self.primary_contact_email,
            billing_contact_email=self.billing_contact_email,
            technical_contact_email=self.technical_contact_email,
            timezone=self.timezone,
            locale=self.locale,
            date_format=self.date_format
        )
        
        # Set complex configurations
        tenant_table.set_billing_configuration(self.billing_configuration)
        tenant_table.set_search_configuration(self.search_configuration)
        tenant_table.set_processing_configuration(self.processing_configuration)
        tenant_table.set_quota_limits(self.quota_limits)
        tenant_table.set_feature_flags(self.feature_flags)
        tenant_table.set_usage_metrics(self.current_usage)
        
        return tenant_table


# Backward compatibility aliases
TenantConfigModel = TenantConfigurationTable  # For SQLModel repositories
Tenant = TenantConfiguration  # Legacy alias