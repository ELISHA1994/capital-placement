"""
Test fixtures for tenant-related testing.

Provides reusable test data builders, fixtures, and utilities
for tenant management testing scenarios.
"""

import pytest
from typing import Dict, List, Any, Optional
from uuid import uuid4
from datetime import datetime, timezone

from app.models.tenant_models import (
    SubscriptionTier, QuotaLimits, FeatureFlags, 
    UsageMetrics, BillingConfiguration
)


class TenantTestBuilder:
    """Builder pattern for creating test tenant data."""
    
    def __init__(self):
        """Initialize with default tenant data."""
        self.tenant_data = {
            "name": "test-tenant",
            "display_name": "Test Tenant",
            "primary_contact_email": "admin@test.com",
            "subscription_tier": SubscriptionTier.FREE,
            "is_active": True,
            "is_suspended": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
    
    def with_name(self, name: str) -> "TenantTestBuilder":
        """Set tenant name."""
        self.tenant_data["name"] = name
        return self
    
    def with_display_name(self, display_name: str) -> "TenantTestBuilder":
        """Set tenant display name."""
        self.tenant_data["display_name"] = display_name
        return self
    
    def with_email(self, email: str) -> "TenantTestBuilder":
        """Set primary contact email."""
        self.tenant_data["primary_contact_email"] = email
        return self
    
    def with_subscription_tier(self, tier: SubscriptionTier) -> "TenantTestBuilder":
        """Set subscription tier."""
        self.tenant_data["subscription_tier"] = tier
        return self
    
    def with_admin_user(self, email: str, password: str, full_name: str = "Test Admin") -> "TenantTestBuilder":
        """Add admin user data for tenant creation."""
        self.tenant_data["admin_user_data"] = {
            "email": email,
            "password": password,
            "full_name": full_name
        }
        return self
    
    def as_suspended(self, reason: str = "Testing") -> "TenantTestBuilder":
        """Mark tenant as suspended."""
        self.tenant_data["is_suspended"] = True
        self.tenant_data["suspension_reason"] = reason
        self.tenant_data["suspended_at"] = datetime.now(timezone.utc).isoformat()
        return self
    
    def as_inactive(self) -> "TenantTestBuilder":
        """Mark tenant as inactive."""
        self.tenant_data["is_active"] = False
        self.tenant_data["deleted_at"] = datetime.now(timezone.utc).isoformat()
        return self
    
    def with_id(self, tenant_id: str) -> "TenantTestBuilder":
        """Set specific tenant ID."""
        self.tenant_data["id"] = tenant_id
        return self
    
    def with_quota_limits(self, **limits) -> "TenantTestBuilder":
        """Set custom quota limits."""
        quota_data = QuotaLimits().model_dump()
        quota_data.update(limits)
        self.tenant_data["quota_limits"] = quota_data
        return self
    
    def with_usage_metrics(self, **metrics) -> "TenantTestBuilder":
        """Set usage metrics."""
        usage_data = UsageMetrics().model_dump()
        usage_data.update(metrics)
        self.tenant_data["usage_metrics"] = usage_data
        return self
    
    def with_feature_flags(self, **flags) -> "TenantTestBuilder":
        """Set feature flags."""
        feature_data = FeatureFlags().model_dump()
        feature_data.update(flags)
        self.tenant_data["feature_flags"] = feature_data
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the tenant data."""
        # Ensure we have an ID
        if "id" not in self.tenant_data:
            self.tenant_data["id"] = str(uuid4())
        
        return self.tenant_data.copy()


class TenantScenarioBuilder:
    """Builder for creating multi-tenant testing scenarios."""
    
    def __init__(self):
        self.tenants: List[Dict[str, Any]] = []
        self.users: List[Dict[str, Any]] = []
        self.relationships: List[Dict[str, Any]] = []
    
    def add_tenant(self, builder: TenantTestBuilder) -> "TenantScenarioBuilder":
        """Add a tenant to the scenario."""
        tenant_data = builder.build()
        self.tenants.append(tenant_data)
        return self
    
    def add_user_to_last_tenant(self, email: str, roles: List[str] = None) -> "TenantScenarioBuilder":
        """Add a user to the last added tenant."""
        if not self.tenants:
            raise ValueError("No tenants added yet")
        
        last_tenant = self.tenants[-1]
        user_data = {
            "id": str(uuid4()),
            "tenant_id": last_tenant["id"],
            "email": email,
            "full_name": f"User for {email}",
            "roles": roles or ["user"],
            "is_active": True,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        self.users.append(user_data)
        return self
    
    def build_scenario(self) -> Dict[str, Any]:
        """Build the complete scenario."""
        return {
            "tenants": self.tenants,
            "users": self.users,
            "relationships": self.relationships
        }


# Pytest fixtures

@pytest.fixture
def sample_tenant():
    """Basic tenant for testing."""
    return TenantTestBuilder().build()


@pytest.fixture
def tenant_with_admin():
    """Tenant with admin user data for creation testing."""
    return (TenantTestBuilder()
            .with_admin_user("admin@test.com", "SecurePassword123!")
            .build())


@pytest.fixture
def suspended_tenant():
    """Suspended tenant for testing."""
    return (TenantTestBuilder()
            .as_suspended("Account violation")
            .build())


@pytest.fixture
def inactive_tenant():
    """Inactive tenant for testing."""
    return (TenantTestBuilder()
            .as_inactive()
            .build())


@pytest.fixture
def enterprise_tenant():
    """Enterprise tier tenant with custom settings."""
    return (TenantTestBuilder()
            .with_subscription_tier(SubscriptionTier.ENTERPRISE)
            .with_name("enterprise-corp")
            .with_display_name("Enterprise Corporation")
            .with_email("admin@enterprise.com")
            .with_quota_limits(
                max_profiles=None,  # Unlimited
                max_searches_per_month=None,  # Unlimited
                max_storage_gb=None,  # Unlimited
                max_users=None  # Unlimited
            )
            .with_feature_flags(
                advanced_analytics=True,
                api_access=True,
                custom_branding=True,
                webhook_integrations=True,
                white_label=True
            )
            .build())


@pytest.fixture
def free_tier_tenant():
    """Free tier tenant with restrictions."""
    return (TenantTestBuilder()
            .with_subscription_tier(SubscriptionTier.FREE)
            .with_quota_limits(
                max_profiles=100,
                max_searches_per_month=50,
                max_storage_gb=1,
                max_users=2
            )
            .with_feature_flags(
                advanced_analytics=False,
                api_access=False,
                custom_branding=False
            )
            .build())


@pytest.fixture
def multiple_tenants():
    """Multiple tenants for isolation testing."""
    return [
        TenantTestBuilder().with_name(f"tenant-{i}").build()
        for i in range(3)
    ]


@pytest.fixture
def tenant_with_usage_near_limits():
    """Tenant with usage near quota limits."""
    return (TenantTestBuilder()
            .with_subscription_tier(SubscriptionTier.BASIC)
            .with_quota_limits(
                max_profiles=1000,
                max_searches_per_month=500,
                max_storage_gb=10
            )
            .with_usage_metrics(
                profiles_count=950,  # 95% of limit
                searches_this_month=480,  # 96% of limit
                storage_used_gb=9.5  # 95% of limit
            )
            .build())


@pytest.fixture
def tenant_over_quota():
    """Tenant that has exceeded quota limits."""
    return (TenantTestBuilder()
            .with_subscription_tier(SubscriptionTier.BASIC)
            .with_quota_limits(
                max_profiles=1000,
                max_searches_per_month=500
            )
            .with_usage_metrics(
                profiles_count=1050,  # Over limit
                searches_this_month=520  # Over limit
            )
            .build())


@pytest.fixture
def multi_tenant_scenario():
    """Complex multi-tenant scenario for comprehensive testing."""
    scenario = TenantScenarioBuilder()
    
    # Add enterprise tenant with multiple users
    scenario.add_tenant(
        TenantTestBuilder()
        .with_name("enterprise-corp")
        .with_subscription_tier(SubscriptionTier.ENTERPRISE)
        .with_admin_user("admin@enterprise.com", "EnterprisePassword123!")
    )
    scenario.add_user_to_last_tenant("user1@enterprise.com", ["user"])
    scenario.add_user_to_last_tenant("manager1@enterprise.com", ["user", "manager"])
    
    # Add basic tenant
    scenario.add_tenant(
        TenantTestBuilder()
        .with_name("small-business")
        .with_subscription_tier(SubscriptionTier.BASIC)
        .with_admin_user("admin@smallbiz.com", "SmallBizPassword123!")
    )
    scenario.add_user_to_last_tenant("employee@smallbiz.com", ["user"])
    
    # Add free tenant
    scenario.add_tenant(
        TenantTestBuilder()
        .with_name("startup")
        .with_subscription_tier(SubscriptionTier.FREE)
        .with_admin_user("founder@startup.com", "StartupPassword123!")
    )
    
    return scenario.build_scenario()


@pytest.fixture
def system_tenant():
    """System tenant for super admin operations."""
    return (TenantTestBuilder()
            .with_id("00000000-0000-0000-0000-000000000000")
            .with_name("system")
            .with_display_name("System Tenant")
            .with_email("system@platform.com")
            .with_subscription_tier(SubscriptionTier.ENTERPRISE)
            .build())


# Utility functions for tests

def create_tenant_data(**overrides) -> Dict[str, Any]:
    """Create tenant data with optional overrides."""
    builder = TenantTestBuilder()
    for key, value in overrides.items():
        if hasattr(builder, f"with_{key}"):
            getattr(builder, f"with_{key}")(value)
    return builder.build()


def create_tenant_with_users(
    tenant_name: str,
    user_emails: List[str],
    subscription_tier: SubscriptionTier = SubscriptionTier.FREE
) -> Dict[str, Any]:
    """Create a tenant with multiple users."""
    tenant = (TenantTestBuilder()
              .with_name(tenant_name)
              .with_subscription_tier(subscription_tier)
              .build())
    
    users = []
    for email in user_emails:
        users.append({
            "id": str(uuid4()),
            "tenant_id": tenant["id"],
            "email": email,
            "full_name": f"User {email}",
            "roles": ["user"],
            "is_active": True
        })
    
    return {
        "tenant": tenant,
        "users": users
    }


def assert_tenant_isolation(tenant1_data: Dict, tenant2_data: Dict):
    """Assert that two tenants are properly isolated."""
    assert tenant1_data["id"] != tenant2_data["id"]
    assert tenant1_data["name"] != tenant2_data["name"]
    # Add more isolation assertions as needed


def assert_tenant_quota_valid(tenant_data: Dict, subscription_tier: SubscriptionTier):
    """Assert that tenant quota limits match subscription tier."""
    quota_limits = tenant_data.get("quota_limits", {})
    
    if subscription_tier == SubscriptionTier.FREE:
        assert quota_limits.get("max_profiles") == 100
        assert quota_limits.get("max_users") == 2
    elif subscription_tier == SubscriptionTier.ENTERPRISE:
        assert quota_limits.get("max_profiles") is None  # Unlimited
        assert quota_limits.get("max_users") is None  # Unlimited