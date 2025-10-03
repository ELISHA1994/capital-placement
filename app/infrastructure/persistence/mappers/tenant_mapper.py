"""Mapper between Tenant domain entities and TenantTable persistence models."""

from __future__ import annotations

from typing import Dict, Any, Optional
from uuid import UUID
from datetime import datetime, date

from app.domain.entities.tenant import (
    Tenant,
    TenantType,
    TenantStatus,
    TenantLimits,
    TenantUsage,
    TenantSettings,
    TenantSubscription,
    SubscriptionTier
)
from app.domain.value_objects import TenantId
from app.models.tenant_models import TenantTable, TenantConfigurationTable  # SQLModel persistence models


class TenantMapper:
    """Maps between Tenant domain entities and TenantTable/TenantConfigurationTable persistence models."""

    @staticmethod
    def to_domain(tenant_table: TenantTable, config_table: Optional[TenantConfigurationTable] = None) -> Tenant:
        """Convert TenantTable (with optional config) to Tenant domain entity."""
        # Map tenant type
        tenant_type = TenantType.ENTERPRISE
        if tenant_table.is_system_tenant:
            tenant_type = TenantType.SYSTEM
        
        # Map tenant status
        tenant_status = TenantStatus.ACTIVE if tenant_table.is_active else TenantStatus.INACTIVE
        
        # Map limits from config or use defaults
        limits = TenantMapper._map_limits_to_domain(config_table)
        
        # Map usage from config or use defaults
        usage = TenantMapper._map_usage_to_domain(config_table)
        
        # Map settings from config or use defaults
        settings = TenantMapper._map_settings_to_domain(config_table)
        
        # Map subscription from config or use defaults
        subscription = TenantMapper._map_subscription_to_domain(tenant_table, config_table)
        
        # Create domain entity
        return Tenant(
            id=TenantId(tenant_table.id),
            name=tenant_table.name,
            type=tenant_type,
            status=tenant_status,
            limits=limits,
            usage=usage,
            settings=settings,
            subscription=subscription,
            metadata={},  # Could be expanded
            created_at=tenant_table.created_at,
            updated_at=tenant_table.updated_at
        )

    @staticmethod
    def to_persistence(tenant: Tenant) -> tuple[TenantTable, TenantConfigurationTable]:
        """Convert Tenant domain entity to persistence models."""
        # Create basic tenant table
        tenant_table = TenantTable(
            id=tenant.id.value,
            name=tenant.name,
            slug=tenant.name.lower().replace(' ', '-'),  # Simple slug generation
            display_name=tenant.name,
            description=f"Tenant organization: {tenant.name}",
            subscription_tier=tenant.subscription.tier.value,
            is_active=tenant.status == TenantStatus.ACTIVE,
            is_system_tenant=tenant.type == TenantType.SYSTEM,
            primary_contact_email="admin@" + tenant.name.lower().replace(' ', '') + ".com",  # Placeholder
            data_region=tenant.settings.default_timezone if tenant.settings.default_timezone != "UTC" else "us-central",
            timezone=tenant.settings.default_timezone,
            locale=tenant.settings.default_language + "-US",
            date_format="YYYY-MM-DD",
            created_at=tenant.created_at,
            updated_at=tenant.updated_at
        )
        
        # Create configuration table
        config_table = TenantMapper._create_config_table(tenant)
        
        return tenant_table, config_table

    @staticmethod
    def _create_config_table(tenant: Tenant) -> TenantConfigurationTable:
        """Create TenantConfigurationTable from domain entity."""
        config_table = TenantConfigurationTable(
            id=tenant.id.value,
            name=tenant.name,
            display_name=tenant.name,
            description=f"Configuration for {tenant.name}",
            subscription_tier=tenant.subscription.tier,
            subscription_start_date=tenant.subscription.started_at.date(),
            subscription_end_date=tenant.subscription.expires_at.date() if tenant.subscription.expires_at else None,
            is_active=tenant.status == TenantStatus.ACTIVE,
            is_suspended=tenant.status == TenantStatus.SUSPENDED,
            is_system_tenant=tenant.type == TenantType.SYSTEM,
            primary_contact_email="admin@" + tenant.name.lower().replace(' ', '') + ".com",  # Placeholder
            timezone=tenant.settings.default_timezone,
            locale=tenant.settings.default_language + "-US",
            date_format="YYYY-MM-DD"
        )
        
        # Set complex configurations using the table's methods
        config_table.set_quota_limits(TenantMapper._map_limits_to_persistence(tenant.limits))
        config_table.set_usage_metrics(TenantMapper._map_usage_to_persistence(tenant.usage))
        config_table.set_feature_flags(TenantMapper._map_settings_to_feature_flags(tenant.settings))
        
        # Set billing configuration
        from app.models.tenant_models import BillingConfiguration
        billing_config = BillingConfiguration(
            billing_cycle="monthly",
            currency="USD",
            base_price=0 if tenant.subscription.tier == SubscriptionTier.FREE else 99
        )
        config_table.set_billing_configuration(billing_config)
        
        # Set search and processing configurations with defaults
        from app.models.tenant_models import SearchConfiguration, ProcessingConfiguration
        search_config = SearchConfiguration()
        processing_config = ProcessingConfiguration()
        config_table.set_search_configuration(search_config)
        config_table.set_processing_configuration(processing_config)
        
        return config_table

    @staticmethod
    def _map_limits_to_domain(config_table: Optional[TenantConfigurationTable]) -> TenantLimits:
        """Map persistence limits to domain TenantLimits."""
        if not config_table:
            return TenantLimits()
        
        quota_limits = config_table.get_quota_limits()
        
        return TenantLimits(
            max_users=quota_limits.max_users or 10,
            max_profiles=quota_limits.max_profiles or 1000,
            max_storage_gb=int(quota_limits.max_storage_gb or 5),
            max_searches_per_month=quota_limits.max_searches_per_month or 1000,
            max_api_calls_per_day=quota_limits.max_api_requests_per_day or 10000,
            vector_search_enabled=True,  # Default enabled
            ai_features_enabled=True,  # Default enabled
            advanced_analytics=config_table.subscription_tier != "free",
            custom_branding=config_table.subscription_tier in ["professional", "enterprise"],
            sso_enabled=config_table.subscription_tier == "enterprise"
        )

    @staticmethod
    def _map_limits_to_persistence(limits: TenantLimits):
        """Map domain TenantLimits to persistence QuotaLimits."""
        from app.models.tenant_models import QuotaLimits
        from decimal import Decimal
        
        return QuotaLimits(
            max_profiles=limits.max_profiles,
            max_storage_gb=Decimal(str(limits.max_storage_gb)),
            max_documents_per_day=100,  # Default
            max_documents_per_month=1000,  # Default
            max_searches_per_day=limits.max_searches_per_month // 30,
            max_searches_per_month=limits.max_searches_per_month,
            max_api_requests_per_day=limits.max_api_calls_per_day,
            max_users=limits.max_users
        )

    @staticmethod
    def _map_usage_to_domain(config_table: Optional[TenantConfigurationTable]) -> TenantUsage:
        """Map persistence usage to domain TenantUsage."""
        if not config_table:
            return TenantUsage()
        
        usage_metrics = config_table.get_usage_metrics()
        
        return TenantUsage(
            user_count=0,  # Would need to be calculated
            profile_count=usage_metrics.total_profiles,
            storage_used_gb=float(usage_metrics.storage_used_gb),
            searches_this_month=usage_metrics.searches_this_month,
            api_calls_today=usage_metrics.api_requests_today,
            last_activity_at=usage_metrics.metrics_updated_at if hasattr(usage_metrics, 'metrics_updated_at') else None,
            total_searches=usage_metrics.total_searches,
            total_uploads=usage_metrics.documents_processed
        )

    @staticmethod
    def _map_usage_to_persistence(usage: TenantUsage):
        """Map domain TenantUsage to persistence UsageMetrics."""
        from app.models.tenant_models import UsageMetrics
        from decimal import Decimal
        
        return UsageMetrics(
            total_profiles=usage.profile_count,
            active_profiles=usage.profile_count,  # Approximation
            profiles_added_this_month=0,  # Not tracked
            total_searches=usage.total_searches,
            searches_this_month=usage.searches_this_month,
            searches_today=0,  # Not tracked
            storage_used_gb=Decimal(str(usage.storage_used_gb)),
            documents_processed=usage.total_uploads,
            documents_pending=0,  # Not tracked
            api_requests_today=usage.api_calls_today,
            api_requests_this_month=0,  # Not tracked
            metrics_updated_at=usage.last_activity_at or datetime.utcnow()
        )

    @staticmethod
    def _map_settings_to_domain(config_table: Optional[TenantConfigurationTable]) -> TenantSettings:
        """Map persistence settings to domain TenantSettings."""
        if not config_table:
            return TenantSettings()
        
        return TenantSettings(
            default_language=config_table.locale.split('-')[0] if config_table.locale else "en",
            default_timezone=config_table.timezone,
            allow_public_profiles=False,  # Default
            require_profile_approval=False,  # Default
            enable_analytics=True,  # Default
            enable_notifications=True,  # Default
            data_retention_days=365,  # Default
            profile_auto_archive_days=90,  # Default
            custom_logo_url=None,
            custom_colors={},
            email_domains=[],
            sso_configuration={}
        )

    @staticmethod
    def _map_settings_to_feature_flags(settings: TenantSettings):
        """Map domain TenantSettings to persistence FeatureFlags."""
        from app.models.tenant_models import FeatureFlags
        
        return FeatureFlags(
            enable_advanced_search=True,
            enable_bulk_operations=True,
            enable_export=True,
            enable_webhooks=settings.enable_notifications,
            enable_ai_recommendations=False,
            enable_skill_extraction=True,
            enable_sentiment_analysis=False,
            enable_candidate_scoring=True,
            enable_analytics_dashboard=settings.enable_analytics,
            enable_custom_reports=False,
            enable_data_insights=False,
            enable_ats_integration=False,
            enable_crm_integration=False,
            enable_api_access=True,
            enable_sso=bool(settings.sso_configuration)
        )

    @staticmethod
    def _map_subscription_to_domain(
        tenant_table: TenantTable,
        config_table: Optional[TenantConfigurationTable]
    ) -> TenantSubscription:
        """Map persistence subscription to domain TenantSubscription."""
        tier_str = tenant_table.subscription_tier
        tier = SubscriptionTier.FREE
        try:
            tier = SubscriptionTier(tier_str)
        except ValueError:
            tier = SubscriptionTier.FREE
        
        started_at = tenant_table.created_at
        expires_at = None
        
        if config_table and config_table.subscription_end_date:
            expires_at = datetime.combine(config_table.subscription_end_date, datetime.min.time())
        
        return TenantSubscription(
            tier=tier,
            started_at=started_at,
            expires_at=expires_at,
            auto_renew=False,  # Default
            payment_method_id=None,
            last_payment_at=None,
            next_payment_at=None,
            trial_ends_at=None,
            billing_email=tenant_table.primary_contact_email
        )

    @staticmethod
    def update_persistence_from_domain(
        tenant_table: TenantTable,
        config_table: TenantConfigurationTable,
        tenant: Tenant
    ) -> tuple[TenantTable, TenantConfigurationTable]:
        """Update existing persistence models with data from Tenant domain entity."""
        # Update tenant table
        tenant_table.name = tenant.name
        tenant_table.display_name = tenant.name
        tenant_table.subscription_tier = tenant.subscription.tier.value
        tenant_table.is_active = tenant.status == TenantStatus.ACTIVE
        tenant_table.is_system_tenant = tenant.type == TenantType.SYSTEM
        tenant_table.timezone = tenant.settings.default_timezone
        tenant_table.updated_at = tenant.updated_at
        
        # Update config table
        config_table.name = tenant.name
        config_table.display_name = tenant.name
        config_table.subscription_tier = tenant.subscription.tier
        config_table.is_active = tenant.status == TenantStatus.ACTIVE
        config_table.is_suspended = tenant.status == TenantStatus.SUSPENDED
        config_table.is_system_tenant = tenant.type == TenantType.SYSTEM
        config_table.timezone = tenant.settings.default_timezone
        
        # Update subscription dates
        if tenant.subscription.expires_at:
            config_table.subscription_end_date = tenant.subscription.expires_at.date()
        
        # Update complex configurations
        config_table.set_quota_limits(TenantMapper._map_limits_to_persistence(tenant.limits))
        config_table.set_usage_metrics(TenantMapper._map_usage_to_persistence(tenant.usage))
        config_table.set_feature_flags(TenantMapper._map_settings_to_feature_flags(tenant.settings))
        
        return tenant_table, config_table


__all__ = ["TenantMapper"]