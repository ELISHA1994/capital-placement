"""Pure domain representation of tenant aggregates."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from app.domain.value_objects import TenantId


class TenantType(str, Enum):
    """Types of tenant organizations."""

    ENTERPRISE = "enterprise"
    SMALL_BUSINESS = "small_business"
    STARTUP = "startup"
    EDUCATIONAL = "educational"
    NON_PROFIT = "non_profit"
    GOVERNMENT = "government"
    SYSTEM = "system"  # Special system tenant


class TenantStatus(str, Enum):
    """Tenant account status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    EXPIRED = "expired"
    DELETED = "deleted"


class SubscriptionTier(str, Enum):
    """Subscription tier levels."""

    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    UNLIMITED = "unlimited"


@dataclass
class TenantLimits:
    """Resource limits for a tenant."""

    max_users: int = 10
    max_profiles: int = 1000
    max_storage_gb: int = 5
    max_searches_per_month: int = 1000
    max_api_calls_per_day: int = 10000
    vector_search_enabled: bool = True
    ai_features_enabled: bool = True
    advanced_analytics: bool = False
    custom_branding: bool = False
    sso_enabled: bool = False

    def can_add_user(self, current_user_count: int) -> bool:
        """Check if tenant can add another user."""
        return current_user_count < self.max_users

    def can_add_profile(self, current_profile_count: int) -> bool:
        """Check if tenant can add another profile."""
        return current_profile_count < self.max_profiles

    def can_perform_search(self, searches_this_month: int) -> bool:
        """Check if tenant can perform another search."""
        return searches_this_month < self.max_searches_per_month

    def can_make_api_call(self, api_calls_today: int) -> bool:
        """Check if tenant can make another API call."""
        return api_calls_today < self.max_api_calls_per_day


@dataclass
class TenantUsage:
    """Current usage statistics for a tenant."""

    user_count: int = 0
    profile_count: int = 0
    storage_used_gb: float = 0.0
    searches_this_month: int = 0
    api_calls_today: int = 0
    last_activity_at: Optional[datetime] = None
    total_searches: int = 0
    total_uploads: int = 0

    def record_search(self) -> None:
        """Record a search operation."""
        self.searches_this_month += 1
        self.total_searches += 1
        self.last_activity_at = datetime.utcnow()

    def record_api_call(self) -> None:
        """Record an API call."""
        self.api_calls_today += 1
        self.last_activity_at = datetime.utcnow()

    def record_upload(self, size_gb: float) -> None:
        """Record a profile upload."""
        self.profile_count += 1
        self.total_uploads += 1
        self.storage_used_gb += size_gb
        self.last_activity_at = datetime.utcnow()

    def remove_profile(self, size_gb: float) -> None:
        """Record profile removal."""
        self.profile_count = max(0, self.profile_count - 1)
        self.storage_used_gb = max(0.0, self.storage_used_gb - size_gb)

    def add_user(self) -> None:
        """Record user addition."""
        self.user_count += 1

    def remove_user(self) -> None:
        """Record user removal."""
        self.user_count = max(0, self.user_count - 1)

    def reset_monthly_counters(self) -> None:
        """Reset monthly usage counters."""
        self.searches_this_month = 0

    def reset_daily_counters(self) -> None:
        """Reset daily usage counters."""
        self.api_calls_today = 0


@dataclass
class TenantSettings:
    """Tenant configuration and preferences."""

    default_language: str = "en"
    default_timezone: str = "UTC"
    allow_public_profiles: bool = False
    require_profile_approval: bool = False
    enable_analytics: bool = True
    enable_notifications: bool = True
    data_retention_days: int = 365
    profile_auto_archive_days: int = 90
    custom_logo_url: Optional[str] = None
    custom_colors: Dict[str, str] = field(default_factory=dict)
    email_domains: List[str] = field(default_factory=list)  # Allowed email domains
    sso_configuration: Dict[str, Any] = field(default_factory=dict)

    def add_allowed_email_domain(self, domain: str) -> None:
        """Add an allowed email domain."""
        domain = domain.lower().strip()
        if domain not in self.email_domains:
            self.email_domains.append(domain)

    def remove_allowed_email_domain(self, domain: str) -> None:
        """Remove an allowed email domain."""
        domain = domain.lower().strip()
        if domain in self.email_domains:
            self.email_domains.remove(domain)

    def is_email_domain_allowed(self, email: str) -> bool:
        """Check if email domain is allowed."""
        if not self.email_domains:
            return True  # No restrictions
        
        domain = email.split("@")[-1].lower() if "@" in email else ""
        return domain in self.email_domains


@dataclass
class TenantSubscription:
    """Tenant subscription information."""

    tier: SubscriptionTier = SubscriptionTier.FREE
    started_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    auto_renew: bool = False
    payment_method_id: Optional[str] = None
    last_payment_at: Optional[datetime] = None
    next_payment_at: Optional[datetime] = None
    trial_ends_at: Optional[datetime] = None
    billing_email: Optional[str] = None

    def is_active(self) -> bool:
        """Check if subscription is currently active."""
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True

    def is_trial(self) -> bool:
        """Check if subscription is in trial period."""
        return (
            self.tier == SubscriptionTier.FREE and
            self.trial_ends_at and
            datetime.utcnow() < self.trial_ends_at
        )

    def days_until_expiry(self) -> Optional[int]:
        """Calculate days until subscription expires."""
        if not self.expires_at:
            return None
        
        delta = self.expires_at - datetime.utcnow()
        return max(0, delta.days)

    def upgrade_tier(self, new_tier: SubscriptionTier) -> None:
        """Upgrade subscription tier."""
        self.tier = new_tier
        if new_tier != SubscriptionTier.FREE:
            # Set expiry to one month from now for paid tiers
            self.expires_at = datetime.utcnow().replace(month=datetime.utcnow().month + 1)

    def renew_subscription(self, duration_days: int = 30) -> None:
        """Renew subscription for specified duration."""
        now = datetime.utcnow()
        if self.expires_at and self.expires_at > now:
            # Extend existing subscription
            self.expires_at = self.expires_at.replace(day=self.expires_at.day + duration_days)
        else:
            # Start new subscription period
            self.expires_at = now.replace(day=now.day + duration_days)
        
        self.last_payment_at = now


@dataclass
class Tenant:
    """Aggregate root representing a tenant organization."""

    id: TenantId
    name: str
    type: TenantType
    status: TenantStatus
    limits: TenantLimits
    usage: TenantUsage = field(default_factory=TenantUsage)
    settings: TenantSettings = field(default_factory=TenantSettings)
    subscription: TenantSubscription = field(default_factory=TenantSubscription)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def activate(self) -> None:
        """Activate tenant."""
        if self.status == TenantStatus.DELETED:
            raise ValueError("Cannot activate deleted tenant")
        
        self.status = TenantStatus.ACTIVE
        self.updated_at = datetime.utcnow()

    def deactivate(self) -> None:
        """Deactivate tenant."""
        self.status = TenantStatus.INACTIVE
        self.updated_at = datetime.utcnow()

    def suspend(self, reason: Optional[str] = None) -> None:
        """Suspend tenant."""
        self.status = TenantStatus.SUSPENDED
        if reason:
            self.metadata["suspension_reason"] = reason
        self.metadata["suspended_at"] = datetime.utcnow().isoformat()
        self.updated_at = datetime.utcnow()

    def mark_deleted(self) -> None:
        """Soft delete tenant."""
        self.status = TenantStatus.DELETED
        self.updated_at = datetime.utcnow()

    def update_name(self, new_name: str) -> None:
        """Update tenant name."""
        self.name = new_name
        self.updated_at = datetime.utcnow()

    def can_add_user(self) -> bool:
        """Check if tenant can add another user."""
        return (
            self.status == TenantStatus.ACTIVE and
            self.subscription.is_active() and
            self.limits.can_add_user(self.usage.user_count)
        )

    def can_add_profile(self) -> bool:
        """Check if tenant can add another profile."""
        return (
            self.status == TenantStatus.ACTIVE and
            self.subscription.is_active() and
            self.limits.can_add_profile(self.usage.profile_count)
        )

    def can_perform_search(self) -> bool:
        """Check if tenant can perform a search."""
        return (
            self.status == TenantStatus.ACTIVE and
            self.subscription.is_active() and
            self.limits.can_perform_search(self.usage.searches_this_month)
        )

    def can_use_ai_features(self) -> bool:
        """Check if tenant can use AI features."""
        return (
            self.status == TenantStatus.ACTIVE and
            self.subscription.is_active() and
            self.limits.ai_features_enabled
        )

    def can_use_vector_search(self) -> bool:
        """Check if tenant can use vector search."""
        return (
            self.status == TenantStatus.ACTIVE and
            self.subscription.is_active() and
            self.limits.vector_search_enabled
        )

    def record_search(self) -> None:
        """Record a search operation."""
        if not self.can_perform_search():
            raise ValueError("Search limit exceeded or tenant not active")
        
        self.usage.record_search()
        self.updated_at = datetime.utcnow()

    def record_profile_upload(self, size_gb: float) -> None:
        """Record a profile upload."""
        if not self.can_add_profile():
            raise ValueError("Profile limit exceeded or tenant not active")
        
        self.usage.record_upload(size_gb)
        self.updated_at = datetime.utcnow()

    def add_user(self) -> None:
        """Add a user to the tenant."""
        if not self.can_add_user():
            raise ValueError("User limit exceeded or tenant not active")
        
        self.usage.add_user()
        self.updated_at = datetime.utcnow()

    def remove_user(self) -> None:
        """Remove a user from the tenant."""
        self.usage.remove_user()
        self.updated_at = datetime.utcnow()

    def upgrade_subscription(self, new_tier: SubscriptionTier) -> None:
        """Upgrade subscription tier and update limits."""
        self.subscription.upgrade_tier(new_tier)
        self._update_limits_for_tier(new_tier)
        self.updated_at = datetime.utcnow()

    def _update_limits_for_tier(self, tier: SubscriptionTier) -> None:
        """Update limits based on subscription tier."""
        if tier == SubscriptionTier.FREE:
            self.limits = TenantLimits(
                max_users=3,
                max_profiles=100,
                max_storage_gb=1,
                max_searches_per_month=100,
                max_api_calls_per_day=1000,
                vector_search_enabled=False,
                ai_features_enabled=False,
                advanced_analytics=False,
                custom_branding=False,
                sso_enabled=False
            )
        elif tier == SubscriptionTier.BASIC:
            self.limits = TenantLimits(
                max_users=10,
                max_profiles=1000,
                max_storage_gb=5,
                max_searches_per_month=1000,
                max_api_calls_per_day=10000,
                vector_search_enabled=True,
                ai_features_enabled=True,
                advanced_analytics=False,
                custom_branding=False,
                sso_enabled=False
            )
        elif tier == SubscriptionTier.PROFESSIONAL:
            self.limits = TenantLimits(
                max_users=50,
                max_profiles=10000,
                max_storage_gb=50,
                max_searches_per_month=10000,
                max_api_calls_per_day=100000,
                vector_search_enabled=True,
                ai_features_enabled=True,
                advanced_analytics=True,
                custom_branding=True,
                sso_enabled=False
            )
        elif tier == SubscriptionTier.ENTERPRISE:
            self.limits = TenantLimits(
                max_users=500,
                max_profiles=100000,
                max_storage_gb=500,
                max_searches_per_month=100000,
                max_api_calls_per_day=1000000,
                vector_search_enabled=True,
                ai_features_enabled=True,
                advanced_analytics=True,
                custom_branding=True,
                sso_enabled=True
            )
        elif tier == SubscriptionTier.UNLIMITED:
            self.limits = TenantLimits(
                max_users=999999,
                max_profiles=999999,
                max_storage_gb=999999,
                max_searches_per_month=999999,
                max_api_calls_per_day=999999,
                vector_search_enabled=True,
                ai_features_enabled=True,
                advanced_analytics=True,
                custom_branding=True,
                sso_enabled=True
            )

    def is_system_tenant(self) -> bool:
        """Check if this is the system tenant."""
        return self.type == TenantType.SYSTEM

    def is_active(self) -> bool:
        """Check if tenant is active."""
        return (
            self.status == TenantStatus.ACTIVE and
            self.subscription.is_active()
        )

    def get_usage_percentage(self) -> Dict[str, float]:
        """Get usage as percentage of limits."""
        return {
            "users": (self.usage.user_count / self.limits.max_users) * 100,
            "profiles": (self.usage.profile_count / self.limits.max_profiles) * 100,
            "storage": (self.usage.storage_used_gb / self.limits.max_storage_gb) * 100,
            "searches": (self.usage.searches_this_month / self.limits.max_searches_per_month) * 100,
            "api_calls": (self.usage.api_calls_today / self.limits.max_api_calls_per_day) * 100,
        }


__all__ = [
    "Tenant",
    "TenantType",
    "TenantStatus",
    "TenantLimits",
    "TenantUsage",
    "TenantSettings",
    "TenantSubscription",
    "SubscriptionTier",
]