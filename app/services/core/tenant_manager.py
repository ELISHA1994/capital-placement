"""
Tenant Manager Service

Multi-tenant data and configuration management service with:
- Tenant configuration and settings management
- Resource quota enforcement and monitoring
- Data isolation and access control
- Feature flag management per tenant
- Usage tracking and billing support
- Performance optimization per tenant
"""

import asyncio
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
import structlog

from app.infrastructure.persistence.models.tenant_table import (
    TenantConfiguration, SubscriptionTier, IndexStrategy,
    UsageMetrics, QuotaLimits, FeatureFlags, BillingConfiguration
)
from app.domain.interfaces import IDatabase, ICacheService

logger = structlog.get_logger(__name__)


class TenantManager:
    """
    Production-ready tenant management service.
    
    Manages all aspects of multi-tenant operations:
    - Tenant configuration and settings
    - Resource quotas and usage monitoring
    - Feature access control and billing
    - Data isolation and security
    - Performance optimization per tenant
    - Usage analytics and reporting
    """
    
    def __init__(
        self,
        database: Optional[IDatabase] = None,
        cache_service: Optional[ICacheService] = None
    ):
        self.database = database
        self.cache_service = cache_service
        
        # Tenant configuration cache
        self._tenant_cache: Dict[str, Tuple[TenantConfiguration, datetime]] = {}
        self._cache_ttl = timedelta(minutes=15)
        
        # Usage tracking
        self._usage_stats = {
            "tenants_active": 0,
            "configurations_cached": 0,
            "quota_violations": 0,
            "feature_checks": 0,
            "usage_updates": 0
        }
        
        # Default configurations by tier
        self._tier_defaults = {
            SubscriptionTier.FREE: {
                "search_config": {
                    "default_search_mode": "keyword",
                    "enable_vector_search": False,
                    "max_results_per_search": 50,
                    "enable_reranking": False
                },
                "quota_limits": {
                    "max_profiles": 100,
                    "max_searches_per_day": 50,
                    "max_storage_gb": Decimal("1.0"),
                    "max_documents_per_day": 10
                },
                "feature_flags": {
                    "enable_advanced_search": False,
                    "enable_ai_recommendations": False,
                    "enable_bulk_operations": False,
                    "enable_analytics_dashboard": False
                }
            },
            SubscriptionTier.BASIC: {
                "search_config": {
                    "default_search_mode": "hybrid",
                    "enable_vector_search": True,
                    "max_results_per_search": 200,
                    "enable_reranking": True
                },
                "quota_limits": {
                    "max_profiles": 1000,
                    "max_searches_per_day": 500,
                    "max_storage_gb": Decimal("10.0"),
                    "max_documents_per_day": 50
                },
                "feature_flags": {
                    "enable_advanced_search": True,
                    "enable_ai_recommendations": False,
                    "enable_bulk_operations": True,
                    "enable_analytics_dashboard": True
                }
            },
            SubscriptionTier.PROFESSIONAL: {
                "search_config": {
                    "default_search_mode": "multi_stage",
                    "enable_vector_search": True,
                    "max_results_per_search": 1000,
                    "enable_reranking": True
                },
                "quota_limits": {
                    "max_profiles": 10000,
                    "max_searches_per_day": 2000,
                    "max_storage_gb": Decimal("100.0"),
                    "max_documents_per_day": 200
                },
                "feature_flags": {
                    "enable_advanced_search": True,
                    "enable_ai_recommendations": True,
                    "enable_bulk_operations": True,
                    "enable_analytics_dashboard": True,
                    "enable_custom_reports": True
                }
            },
            SubscriptionTier.ENTERPRISE: {
                "search_config": {
                    "default_search_mode": "multi_stage",
                    "enable_vector_search": True,
                    "max_results_per_search": 5000,
                    "enable_reranking": True,
                    "enable_diversity": True
                },
                "quota_limits": {
                    "max_profiles": None,  # Unlimited
                    "max_searches_per_day": None,
                    "max_storage_gb": None,
                    "max_documents_per_day": None
                },
                "feature_flags": {
                    "enable_advanced_search": True,
                    "enable_ai_recommendations": True,
                    "enable_bulk_operations": True,
                    "enable_analytics_dashboard": True,
                    "enable_custom_reports": True,
                    "enable_api_access": True,
                    "enable_webhooks": True,
                    "enable_sso": True
                }
            }
        }
    
    async def get_tenant_configuration(
        self,
        tenant_id: str,
        use_cache: bool = True
    ) -> Optional[TenantConfiguration]:
        """
        Get tenant configuration with caching support.
        
        Args:
            tenant_id: Tenant identifier
            use_cache: Whether to use cached configuration
            
        Returns:
            Tenant configuration or None if not found
        """
        try:
            # Check cache first
            if use_cache:
                cached_config = self._get_cached_tenant_config(tenant_id)
                if cached_config:
                    return cached_config
            
            # Load from database
            if not self.database:
                logger.warning("No database service configured for tenant manager")
                return None
            
            # Query tenant configuration
            tenant_data = await self.database.get_item(
                container="tenant-config",
                item_id=tenant_id,
                partition_key=tenant_id
            )
            
            if not tenant_data:
                logger.warning(f"Tenant configuration not found: {tenant_id}")
                return None
            
            # Convert to TenantConfiguration object
            config = TenantConfiguration(**tenant_data)
            
            # Cache the configuration
            if use_cache:
                self._cache_tenant_config(tenant_id, config)
            
            logger.debug(f"Loaded tenant configuration: {tenant_id}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to get tenant configuration: {e}", tenant_id=tenant_id)
            return None
    
    async def create_tenant_configuration(
        self,
        tenant_id: str,
        tenant_name: str,
        subscription_tier: SubscriptionTier,
        primary_contact_email: str,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> TenantConfiguration:
        """
        Create new tenant configuration with tier-based defaults.
        
        Args:
            tenant_id: Unique tenant identifier
            tenant_name: Display name for tenant
            subscription_tier: Subscription tier level
            primary_contact_email: Primary contact email
            custom_config: Optional custom configuration overrides
            
        Returns:
            Created tenant configuration
        """
        try:
            # Build configuration from tier defaults
            tier_config = self._tier_defaults.get(subscription_tier, self._tier_defaults[SubscriptionTier.FREE])
            
            # Create tenant configuration
            config = TenantConfiguration(
                tenant_id=tenant_id,
                name=tenant_name,
                display_name=tenant_name,
                subscription_tier=subscription_tier,
                primary_contact_email=primary_contact_email,
                **tier_config
            )
            
            # Apply custom configuration if provided
            if custom_config:
                for key, value in custom_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            
            # Store in database
            if self.database:
                tenant_data = config.dict()
                await self.database.create_item(
                    container="tenant-config",
                    item=tenant_data
                )
            
            # Cache the configuration
            self._cache_tenant_config(tenant_id, config)
            
            logger.info(
                "Created tenant configuration",
                tenant_id=tenant_id,
                tier=subscription_tier,
                name=tenant_name
            )
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to create tenant configuration: {e}")
            raise RuntimeError(f"Tenant configuration creation failed: {e}")
    
    async def update_tenant_configuration(
        self,
        tenant_id: str,
        updates: Dict[str, Any],
        updated_by: Optional[str] = None
    ) -> TenantConfiguration:
        """
        Update tenant configuration with validation.
        
        Args:
            tenant_id: Tenant identifier
            updates: Configuration updates to apply
            updated_by: User making the update
            
        Returns:
            Updated tenant configuration
        """
        try:
            # Get current configuration
            current_config = await self.get_tenant_configuration(tenant_id, use_cache=False)
            if not current_config:
                raise ValueError(f"Tenant configuration not found: {tenant_id}")
            
            # Apply updates
            updated_data = current_config.dict()
            updated_data.update(updates)
            
            # Validate updated configuration
            updated_config = TenantConfiguration(**updated_data)
            
            # Update version and metadata
            if updated_by:
                updated_config.updated_by = updated_by
            updated_config.increment_version()
            
            # Store updated configuration
            if self.database:
                await self.database.update_item(
                    container="tenant-config",
                    item_id=tenant_id,
                    item=updated_config.dict()
                )
            
            # Update cache
            self._cache_tenant_config(tenant_id, updated_config)
            
            logger.info(
                "Updated tenant configuration",
                tenant_id=tenant_id,
                updates=list(updates.keys()),
                version=updated_config.version
            )
            
            return updated_config
            
        except Exception as e:
            logger.error(f"Failed to update tenant configuration: {e}")
            raise RuntimeError(f"Tenant configuration update failed: {e}")
    
    async def check_quota_limit(
        self,
        tenant_id: str,
        resource_type: str,
        current_usage: int,
        increment: int = 1
    ) -> Dict[str, Any]:
        """
        Check if resource usage is within quota limits.
        
        Args:
            tenant_id: Tenant identifier
            resource_type: Type of resource (profiles, searches_per_day, etc.)
            current_usage: Current usage count
            increment: Amount to increment (for pre-check)
            
        Returns:
            Quota check result with allowed status and details
        """
        try:
            config = await self.get_tenant_configuration(tenant_id)
            if not config:
                return {"allowed": False, "reason": "Tenant configuration not found"}
            
            # Check if tenant is active
            if not config.is_subscription_active:
                return {"allowed": False, "reason": f"Subscription {config.subscription_status}"}
            
            # Check specific quota limit
            new_usage = current_usage + increment
            within_limit = config.check_quota_limit(resource_type, new_usage)
            
            if not within_limit:
                # Get the actual limit for detailed response
                effective_limits = config.get_effective_limits()
                limit = effective_limits.get(resource_type)
                
                self._usage_stats["quota_violations"] += 1
                
                return {
                    "allowed": False,
                    "reason": "Quota limit exceeded",
                    "current_usage": current_usage,
                    "limit": limit,
                    "resource_type": resource_type
                }
            
            return {
                "allowed": True,
                "current_usage": current_usage,
                "new_usage": new_usage,
                "resource_type": resource_type
            }
            
        except Exception as e:
            logger.error(f"Failed to check quota limit: {e}")
            return {"allowed": False, "reason": "Quota check failed"}
    
    async def check_feature_access(
        self,
        tenant_id: str,
        feature_name: str
    ) -> bool:
        """
        Check if tenant has access to a specific feature.
        
        Args:
            tenant_id: Tenant identifier
            feature_name: Feature name to check
            
        Returns:
            True if tenant has access to feature
        """
        try:
            config = await self.get_tenant_configuration(tenant_id)
            if not config:
                return False
            
            # Check subscription status
            if not config.is_subscription_active:
                return False
            
            # Check feature access
            has_access = config.has_feature(feature_name)
            
            self._usage_stats["feature_checks"] += 1
            
            logger.debug(
                "Feature access check",
                tenant_id=tenant_id,
                feature=feature_name,
                access=has_access
            )
            
            return has_access
            
        except Exception as e:
            logger.error(f"Failed to check feature access: {e}")
            return False
    
    async def update_usage_metrics(
        self,
        tenant_id: str,
        metrics_update: Dict[str, Any]
    ) -> bool:
        """
        Update tenant usage metrics.
        
        Args:
            tenant_id: Tenant identifier
            metrics_update: Usage metrics to update
            
        Returns:
            True if update successful
        """
        try:
            config = await self.get_tenant_configuration(tenant_id)
            if not config:
                logger.warning(f"Cannot update metrics for unknown tenant: {tenant_id}")
                return False
            
            # Update metrics
            config.update_usage_metrics(metrics_update)
            
            # Store updated configuration
            if self.database:
                await self.database.update_item(
                    container="tenant-config",
                    item_id=tenant_id,
                    item=config.dict()
                )
            
            # Update cache
            self._cache_tenant_config(tenant_id, config)
            
            self._usage_stats["usage_updates"] += 1
            
            logger.debug(
                "Updated usage metrics",
                tenant_id=tenant_id,
                metrics=list(metrics_update.keys())
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update usage metrics: {e}")
            return False
    
    async def get_tenant_usage_summary(
        self,
        tenant_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive usage summary for tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Usage summary with quotas, limits, and current usage
        """
        try:
            config = await self.get_tenant_configuration(tenant_id)
            if not config:
                return None
            
            effective_limits = config.get_effective_limits()
            current_usage = config.current_usage.dict()
            
            # Calculate usage percentages
            usage_percentages = {}
            for resource_type, limit in effective_limits.items():
                if limit is not None:
                    usage_key = {
                        "max_profiles": "total_profiles",
                        "max_searches_per_day": "searches_today",
                        "max_searches_per_month": "searches_this_month",
                        "max_storage_gb": "storage_used_gb",
                        "max_documents_per_day": "documents_processed",  # Simplified mapping
                    }.get(resource_type)
                    
                    if usage_key and usage_key in current_usage:
                        usage_value = current_usage[usage_key]
                        if isinstance(usage_value, (int, float, Decimal)) and usage_value > 0:
                            percentage = min(100, (float(usage_value) / float(limit)) * 100)
                            usage_percentages[resource_type] = percentage
            
            return {
                "tenant_id": tenant_id,
                "subscription_tier": config.subscription_tier,
                "subscription_status": config.subscription_status,
                "is_active": config.is_subscription_active,
                "days_until_expiry": config.days_until_expiry,
                "current_usage": current_usage,
                "quota_limits": effective_limits,
                "usage_percentages": usage_percentages,
                "feature_access": {
                    "advanced_search": config.feature_flags.enable_advanced_search,
                    "ai_recommendations": config.feature_flags.enable_ai_recommendations,
                    "bulk_operations": config.feature_flags.enable_bulk_operations,
                    "analytics": config.feature_flags.enable_analytics_dashboard,
                    "api_access": config.feature_flags.enable_api_access
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get tenant usage summary: {e}")
            return None
    
    async def list_tenants(
        self,
        active_only: bool = True,
        subscription_tier: Optional[SubscriptionTier] = None,
        limit: int = 100
    ) -> List[TenantConfiguration]:
        """
        List tenants with optional filtering.
        
        Args:
            active_only: Only return active tenants
            subscription_tier: Filter by subscription tier
            limit: Maximum number of tenants to return
            
        Returns:
            List of tenant configurations
        """
        try:
            if not self.database:
                return []
            
            # Build query
            query_parts = ["SELECT * FROM c"]
            parameters = []
            
            conditions = []
            if active_only:
                conditions.append("c.is_active = true AND c.is_suspended = false")
            
            if subscription_tier:
                conditions.append("c.subscription_tier = @tier")
                parameters.append({"name": "@tier", "value": subscription_tier.value})
            
            if conditions:
                query_parts.append("WHERE " + " AND ".join(conditions))
            
            query_parts.append("ORDER BY c.created_at DESC")
            
            query = " ".join(query_parts)
            
            # Execute query
            results = await self.database.query_items(
                container="tenant-config",
                query=query,
                parameters=parameters,
                max_items=limit
            )
            
            # Convert to TenantConfiguration objects
            tenants = []
            for result in results:
                try:
                    tenant_config = TenantConfiguration(**result)
                    tenants.append(tenant_config)
                except Exception as e:
                    logger.warning(f"Failed to parse tenant configuration: {e}")
                    continue
            
            return tenants
            
        except Exception as e:
            logger.error(f"Failed to list tenants: {e}")
            return []
    
    async def get_tenant_statistics(self) -> Dict[str, Any]:
        """Get overall tenant statistics and metrics"""
        try:
            # Get active tenant count
            active_tenants = await self.list_tenants(active_only=True)
            
            # Count by subscription tier
            tier_counts = {tier: 0 for tier in SubscriptionTier}
            total_usage = {
                "total_profiles": 0,
                "total_searches_today": 0,
                "total_storage_gb": Decimal("0.0")
            }
            
            for tenant in active_tenants:
                tier_counts[tenant.subscription_tier] += 1
                
                # Aggregate usage
                usage = tenant.current_usage
                total_usage["total_profiles"] += usage.total_profiles
                total_usage["total_searches_today"] += usage.searches_today
                total_usage["total_storage_gb"] += usage.storage_used_gb
            
            return {
                "active_tenants": len(active_tenants),
                "tier_distribution": {tier.value: count for tier, count in tier_counts.items()},
                "aggregate_usage": total_usage,
                "service_stats": self._usage_stats,
                "cache_size": len(self._tenant_cache),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get tenant statistics: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def _get_cached_tenant_config(self, tenant_id: str) -> Optional[TenantConfiguration]:
        """Get tenant configuration from cache"""
        if tenant_id in self._tenant_cache:
            config, cached_at = self._tenant_cache[tenant_id]
            if datetime.now() - cached_at < self._cache_ttl:
                self._usage_stats["configurations_cached"] += 1
                return config
            else:
                # Expired, remove from cache
                del self._tenant_cache[tenant_id]
        
        return None
    
    def _cache_tenant_config(self, tenant_id: str, config: TenantConfiguration) -> None:
        """Cache tenant configuration with expiration"""
        # Clean expired entries first
        now = datetime.now()
        expired_keys = [
            key for key, (_, cached_at) in self._tenant_cache.items()
            if now - cached_at >= self._cache_ttl
        ]
        
        for key in expired_keys:
            del self._tenant_cache[key]
        
        # Limit cache size (keep most recent 1000 tenants)
        if len(self._tenant_cache) >= 1000:
            # Remove 20% of oldest entries
            sorted_items = sorted(
                self._tenant_cache.items(),
                key=lambda x: x[1][1]  # Sort by cached_at timestamp
            )
            
            remove_count = len(self._tenant_cache) // 5
            for key, _ in sorted_items[:remove_count]:
                del self._tenant_cache[key]
        
        # Cache the configuration
        self._tenant_cache[tenant_id] = (config, now)
    
    async def cleanup_expired_tenants(self) -> Dict[str, int]:
        """Clean up expired or inactive tenant configurations"""
        try:
            cleanup_stats = {
                "expired_tenants": 0,
                "suspended_tenants": 0,
                "deleted_configs": 0
            }
            
            # Get all tenants (including inactive)
            all_tenants = await self.list_tenants(active_only=False, limit=10000)
            
            for tenant in all_tenants:
                # Check if tenant subscription has expired
                if not tenant.is_subscription_active and tenant.subscription_end_date:
                    days_expired = (date.today() - tenant.subscription_end_date).days
                    
                    if days_expired > 30:  # Grace period of 30 days
                        # Suspend tenant
                        if not tenant.is_suspended:
                            await self.update_tenant_configuration(
                                tenant.tenant_id,
                                {
                                    "is_suspended": True,
                                    "suspension_reason": f"Subscription expired {days_expired} days ago"
                                }
                            )
                            cleanup_stats["suspended_tenants"] += 1
                        
                        cleanup_stats["expired_tenants"] += 1
                        
                        # Archive tenant data after 90 days
                        if days_expired > 90:
                            # In a real implementation, you would move data to archive storage
                            # For now, just log the action
                            logger.info(
                                "Tenant eligible for archival",
                                tenant_id=tenant.tenant_id,
                                days_expired=days_expired
                            )
            
            logger.info("Tenant cleanup completed", **cleanup_stats)
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Tenant cleanup failed: {e}")
            return {"error": str(e)}
    
    def get_manager_statistics(self) -> Dict[str, Any]:
        """Get tenant manager performance statistics"""
        return {
            **self._usage_stats,
            "cache_size": len(self._tenant_cache),
            "cache_ttl_minutes": self._cache_ttl.total_seconds() / 60,
            "supported_tiers": [tier.value for tier in SubscriptionTier],
            "tier_defaults_configured": len(self._tier_defaults)
        }
