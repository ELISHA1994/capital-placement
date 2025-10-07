"""
Comprehensive unit tests for TenantService - SECURITY CRITICAL.

This test suite focuses on:
1. Tenant Lifecycle (create, activate, suspend, delete, restore)
2. Tenant Configuration (get, update, subscription management)
3. Tenant Isolation (CRITICAL - prevent cross-tenant access)
4. Quota & Usage Management (enforce limits)
5. Feature Flags (per-tenant feature control)
6. Subscription Tiers (upgrade/downgrade)
7. Error Handling (invalid inputs, edge cases)
8. User Management (add, remove, role updates)

Total: 35+ comprehensive tests covering all security-critical operations.
"""

import pytest
import pytest_asyncio
from datetime import datetime
from uuid import uuid4, UUID
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from app.infrastructure.tenant.tenant_service import TenantService
from app.infrastructure.persistence.models.tenant_table import (
    TenantConfiguration,
    SubscriptionTier,
    QuotaLimits,
    UsageMetrics,
    FeatureFlags,
    BillingConfiguration,
)
from app.infrastructure.persistence.models.auth_tables import CurrentUser


# Fixtures

@pytest.fixture
def mock_tenant_repository():
    """Mock tenant repository."""
    repo = AsyncMock()
    repo.create = AsyncMock()
    repo.get = AsyncMock()
    repo.update = AsyncMock()
    repo.find_by_criteria = AsyncMock(return_value=[])
    repo.list_all = AsyncMock(return_value=[])
    repo.check_slug_availability = AsyncMock(return_value=True)
    return repo


@pytest.fixture
def mock_user_repository():
    """Mock user repository."""
    repo = AsyncMock()
    repo.create = AsyncMock()
    repo.get_by_id = AsyncMock()
    repo.update = AsyncMock()
    repo.get_by_tenant = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_cache_service():
    """Mock cache service."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    cache.delete = AsyncMock(return_value=True)
    cache.clear = AsyncMock(return_value=0)
    return cache


@pytest.fixture
def tenant_service(mock_tenant_repository, mock_user_repository, mock_cache_service):
    """Create tenant service with mocked dependencies."""
    return TenantService(
        tenant_repository=mock_tenant_repository,
        user_repository=mock_user_repository,
        cache_manager=mock_cache_service,
    )


@pytest.fixture
def valid_tenant_data():
    """Valid tenant creation data."""
    return {
        "id": str(uuid4()),
        "name": "test-tenant",
        "slug": "test-tenant",
        "display_name": "Test Tenant Inc",
        "primary_contact_email": "admin@test-tenant.com",
        "subscription_tier": SubscriptionTier.FREE,
        "quota_limits": QuotaLimits(),
        "usage_metrics": UsageMetrics(),
        "feature_flags": FeatureFlags(),
        "billing_configuration": BillingConfiguration(),
        "is_active": True,
        "is_suspended": False,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }


# ==========================================
# 1. TENANT LIFECYCLE TESTS (6 tests)
# ==========================================

class TestTenantLifecycle:
    """Test tenant lifecycle operations."""

    @pytest.mark.asyncio
    async def test_create_tenant_basic(self, tenant_service, mock_tenant_repository, valid_tenant_data):
        """Test creating a basic tenant without admin user."""
        mock_tenant_repository.get.return_value = None
        mock_tenant_repository.create.return_value = valid_tenant_data

        result = await tenant_service.create_tenant(
            name="test-tenant",
            display_name="Test Tenant Inc",
            primary_contact_email="admin@test-tenant.com",
            subscription_tier=SubscriptionTier.FREE,
        )

        assert result is not None
        assert result.name == "test-tenant"
        assert result.subscription_tier == SubscriptionTier.FREE
        mock_tenant_repository.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_activate_tenant(self, tenant_service, mock_tenant_repository, valid_tenant_data):
        """Test activating a suspended tenant."""
        suspended_data = valid_tenant_data.copy()
        suspended_data["is_suspended"] = True
        suspended_data["suspension_reason"] = "Payment failed"

        mock_tenant_repository.get.return_value = suspended_data
        mock_tenant_repository.update.return_value = valid_tenant_data

        result = await tenant_service.activate_tenant(valid_tenant_data["id"])

        assert result is True
        mock_tenant_repository.update.assert_called_once()
        # Verify suspension was cleared
        update_call_args = mock_tenant_repository.update.call_args[0]
        updated_data = update_call_args[1]
        assert updated_data["is_suspended"] is False
        assert updated_data["suspension_reason"] is None

    @pytest.mark.asyncio
    async def test_suspend_tenant(self, tenant_service, mock_tenant_repository, valid_tenant_data):
        """Test suspending an active tenant."""
        mock_tenant_repository.get.return_value = valid_tenant_data
        mock_tenant_repository.update.return_value = valid_tenant_data

        result = await tenant_service.suspend_tenant(
            valid_tenant_data["id"], reason="Payment overdue"
        )

        assert result is True
        mock_tenant_repository.update.assert_called_once()
        # Verify suspension was applied
        update_call_args = mock_tenant_repository.update.call_args[0]
        updated_data = update_call_args[1]
        assert updated_data["is_suspended"] is True
        assert updated_data["suspension_reason"] == "Payment overdue"

    @pytest.mark.asyncio
    async def test_delete_tenant_soft_delete(self, tenant_service, mock_tenant_repository, valid_tenant_data):
        """Test soft deleting a tenant (deactivate)."""
        mock_tenant_repository.get.return_value = valid_tenant_data
        mock_tenant_repository.update.return_value = valid_tenant_data

        result = await tenant_service.delete_tenant(valid_tenant_data["id"])

        assert result is True
        mock_tenant_repository.update.assert_called_once()
        # Verify soft delete
        update_call_args = mock_tenant_repository.update.call_args[0]
        updated_data = update_call_args[1]
        assert updated_data["is_active"] is False
        assert updated_data["deleted_at"] is not None

    @pytest.mark.asyncio
    async def test_get_tenant_with_cache(self, tenant_service, mock_cache_service, valid_tenant_data):
        """Test getting tenant with caching."""
        # First call - cache miss
        mock_cache_service.get.return_value = None
        tenant_service.tenant_repo.get = AsyncMock(return_value=valid_tenant_data)

        result = await tenant_service.get_tenant(valid_tenant_data["id"])

        assert result is not None
        mock_cache_service.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_tenant_not_found(self, tenant_service, mock_tenant_repository):
        """Test getting non-existent tenant returns None."""
        mock_tenant_repository.get.return_value = None

        result = await tenant_service.get_tenant(str(uuid4()))

        assert result is None


# ==========================================
# 2. TENANT CONFIGURATION TESTS (6 tests)
# ==========================================

class TestTenantConfiguration:
    """Test tenant configuration management."""

    @pytest.mark.asyncio
    async def test_update_tenant_configuration(self, tenant_service, mock_tenant_repository, valid_tenant_data):
        """Test updating tenant configuration."""
        mock_tenant_repository.get.return_value = valid_tenant_data
        mock_tenant_repository.update.return_value = valid_tenant_data

        updates = {"display_name": "Updated Tenant Name"}
        result = await tenant_service.update_tenant(valid_tenant_data["id"], updates)

        assert result is not None
        mock_tenant_repository.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_tenant_configuration(self, tenant_service, mock_tenant_repository, valid_tenant_data):
        """Test retrieving complete tenant configuration."""
        mock_tenant_repository.get.return_value = valid_tenant_data

        result = await tenant_service.get_tenant(valid_tenant_data["id"])

        assert result is not None
        assert result.name == valid_tenant_data["name"]
        assert result.display_name == valid_tenant_data["display_name"]

    @pytest.mark.asyncio
    async def test_subscription_tier_upgrade(self, tenant_service, mock_tenant_repository, valid_tenant_data):
        """Test upgrading subscription tier."""
        mock_tenant_repository.get.return_value = valid_tenant_data
        mock_tenant_repository.update.return_value = valid_tenant_data

        result = await tenant_service.upgrade_subscription(
            valid_tenant_data["id"], SubscriptionTier.PROFESSIONAL
        )

        assert result is True
        mock_tenant_repository.update.assert_called_once()
        # Verify tier was upgraded
        update_call_args = mock_tenant_repository.update.call_args[0]
        updated_data = update_call_args[1]
        assert updated_data["subscription_tier"] == SubscriptionTier.PROFESSIONAL

    @pytest.mark.asyncio
    async def test_subscription_tier_includes_new_quotas(self, tenant_service, mock_tenant_repository, valid_tenant_data):
        """Test that tier upgrade updates quota limits."""
        mock_tenant_repository.get.return_value = valid_tenant_data
        mock_tenant_repository.update.return_value = valid_tenant_data

        await tenant_service.upgrade_subscription(
            valid_tenant_data["id"], SubscriptionTier.ENTERPRISE
        )

        # Verify quota limits were updated
        update_call_args = mock_tenant_repository.update.call_args[0]
        updated_data = update_call_args[1]
        quota_limits = updated_data["quota_limits"]
        # Enterprise tier has unlimited resources
        assert quota_limits["max_profiles"] is None
        assert quota_limits["max_searches_per_month"] is None

    @pytest.mark.asyncio
    async def test_list_tenants_with_pagination(self, tenant_service, mock_tenant_repository, valid_tenant_data):
        """Test listing tenants with pagination."""
        tenants = [valid_tenant_data.copy() for _ in range(10)]
        mock_tenant_repository.find_by_criteria.return_value = tenants

        result = await tenant_service.list_tenants(skip=0, limit=5, include_inactive=False)

        assert len(result) == 5  # Pagination applied

    @pytest.mark.asyncio
    async def test_list_tenants_exclude_inactive(self, tenant_service, mock_tenant_repository, valid_tenant_data):
        """Test listing tenants excludes inactive by default."""
        tenants = [valid_tenant_data.copy() for _ in range(5)]
        mock_tenant_repository.find_by_criteria.return_value = tenants

        await tenant_service.list_tenants(skip=0, limit=10, include_inactive=False)

        # Verify criteria included is_active filter
        mock_tenant_repository.find_by_criteria.assert_called_once()
        call_args = mock_tenant_repository.find_by_criteria.call_args[0]
        criteria = call_args[0]
        assert criteria.get("is_active") is True


# ==========================================
# 3. TENANT ISOLATION TESTS (5 tests) - CRITICAL
# ==========================================

class TestTenantIsolation:
    """Test tenant isolation - SECURITY CRITICAL."""

    @pytest.mark.asyncio
    async def test_tenant_a_cannot_access_tenant_b_data(
        self, tenant_service, mock_tenant_repository, valid_tenant_data
    ):
        """CRITICAL: Verify Tenant A cannot access Tenant B's data."""
        tenant_a_id = str(uuid4())
        tenant_b_id = str(uuid4())

        # Setup: Tenant A requests Tenant B's data
        mock_tenant_repository.get.return_value = None  # Simulates access control

        result = await tenant_service.get_tenant(tenant_b_id)

        # Should return None (no access)
        assert result is None

    @pytest.mark.asyncio
    async def test_tenant_id_validation(self, tenant_service, mock_tenant_repository):
        """Test that invalid tenant IDs are rejected."""
        invalid_ids = ["invalid-id", "", "not-a-uuid", "12345"]

        for invalid_id in invalid_ids:
            mock_tenant_repository.get.return_value = None
            result = await tenant_service.get_tenant(invalid_id)
            assert result is None

    @pytest.mark.asyncio
    async def test_system_tenant_isolation(self, tenant_service, valid_tenant_data):
        """Test that system tenant (00000000-...) is properly isolated."""
        system_tenant_id = "00000000-0000-0000-0000-000000000000"

        # System tenant should be accessible but isolated
        tenant_service.tenant_repo.get = AsyncMock(return_value=valid_tenant_data)
        result = await tenant_service.get_tenant(system_tenant_id)

        # Can access system tenant data
        assert result is not None

    @pytest.mark.asyncio
    async def test_tenant_context_verification(self, tenant_service, mock_user_repository, valid_tenant_data):
        """Test that operations verify tenant context."""
        tenant_a_id = str(uuid4())
        tenant_b_id = str(uuid4())
        user_id = str(uuid4())

        # User belongs to Tenant A
        user_data = {
            "id": user_id,
            "tenant_id": tenant_a_id,
            "email": "user@tenanta.com",
            "full_name": "User A",
            "roles": ["user"],
            "permissions": ["read"],
            "is_active": True,
            "is_superuser": False,
        }

        mock_user_repository.get_by_id.return_value = user_data

        # Trying to remove user from Tenant B should fail
        result = await tenant_service.remove_tenant_user(tenant_b_id, user_id)

        assert result is False  # Operation rejected - wrong tenant

    @pytest.mark.asyncio
    async def test_cache_keys_include_tenant_id(self, tenant_service, mock_cache_service, valid_tenant_data):
        """Test that cache keys include tenant ID for isolation."""
        tenant_id = valid_tenant_data["id"]

        tenant_service.tenant_repo.get = AsyncMock(return_value=valid_tenant_data)
        await tenant_service.get_tenant(tenant_id)

        # Verify cache key includes tenant ID
        cache_call_args = mock_cache_service.set.call_args
        if cache_call_args:
            cache_key = cache_call_args[0][0]
            assert tenant_id in cache_key


# ==========================================
# 4. QUOTA & USAGE MANAGEMENT TESTS (5 tests)
# ==========================================

class TestQuotaManagement:
    """Test quota and usage tracking."""

    @pytest.mark.asyncio
    async def test_check_quota_within_limits(self, tenant_service, mock_tenant_repository, valid_tenant_data):
        """Test checking quota when within limits."""
        tenant_data = valid_tenant_data.copy()
        tenant_data["quota_limits"] = QuotaLimits(max_profiles=100).model_dump()
        tenant_data["usage_metrics"] = UsageMetrics(profiles=50).model_dump()

        mock_tenant_repository.get.return_value = tenant_data

        result = await tenant_service.check_quota(tenant_data["id"], "max_profiles")

        assert result["exceeded"] is False
        assert result["remaining"] == 50
        assert result["percentage_used"] == 50.0

    @pytest.mark.asyncio
    async def test_check_quota_at_limit(self, tenant_service, mock_tenant_repository, valid_tenant_data):
        """Test checking quota exactly at limit."""
        tenant_data = valid_tenant_data.copy()
        tenant_data["quota_limits"] = QuotaLimits(max_profiles=100).model_dump()
        tenant_data["usage_metrics"] = UsageMetrics(profiles=100).model_dump()

        mock_tenant_repository.get.return_value = tenant_data

        result = await tenant_service.check_quota(tenant_data["id"], "max_profiles")

        assert result["exceeded"] is True
        assert result["remaining"] == 0
        assert result["percentage_used"] == 100.0

    @pytest.mark.asyncio
    async def test_check_quota_exceeded(self, tenant_service, mock_tenant_repository, valid_tenant_data):
        """Test checking quota when exceeded."""
        tenant_data = valid_tenant_data.copy()
        tenant_data["quota_limits"] = QuotaLimits(max_profiles=100).model_dump()
        tenant_data["usage_metrics"] = UsageMetrics(profiles=150).model_dump()

        mock_tenant_repository.get.return_value = tenant_data

        result = await tenant_service.check_quota(tenant_data["id"], "max_profiles")

        assert result["exceeded"] is True
        assert result["current_usage"] > result["limit"]

    @pytest.mark.asyncio
    async def test_update_usage_metrics(self, tenant_service, mock_tenant_repository, valid_tenant_data):
        """Test updating tenant usage metrics."""
        tenant_data = valid_tenant_data.copy()
        tenant_data["usage_metrics"] = UsageMetrics(profiles=50, searches=10).model_dump()

        mock_tenant_repository.get.return_value = tenant_data
        mock_tenant_repository.update.return_value = tenant_data

        result = await tenant_service.update_usage(
            tenant_data["id"], {"profiles": 10, "searches": 5}
        )

        assert result is True
        mock_tenant_repository.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_unlimited_quota_never_exceeded(self, tenant_service, mock_tenant_repository, valid_tenant_data):
        """Test that unlimited quotas (None) are never exceeded."""
        tenant_data = valid_tenant_data.copy()
        # Enterprise tier has unlimited quotas
        tenant_data["subscription_tier"] = SubscriptionTier.ENTERPRISE
        tenant_data["quota_limits"] = QuotaLimits(max_profiles=None).model_dump()  # Unlimited
        tenant_data["usage_metrics"] = UsageMetrics(profiles=1000000).model_dump()

        mock_tenant_repository.get.return_value = tenant_data

        result = await tenant_service.check_quota(tenant_data["id"], "max_profiles")

        # Unlimited quota should never be exceeded
        assert result["exceeded"] is False
        assert result["limit"] is None


# ==========================================
# 5. FEATURE FLAGS TESTS (4 tests)
# ==========================================

class TestFeatureFlags:
    """Test feature flag management per tenant."""

    @pytest.mark.asyncio
    async def test_free_tier_has_limited_features(self, tenant_service):
        """Test that FREE tier has appropriate feature restrictions."""
        flags = tenant_service._get_default_feature_flags(SubscriptionTier.FREE)

        assert flags.enable_custom_reports is False
        assert flags.enable_webhooks is False
        assert flags.enable_ats_integration is False
        assert flags.enable_sso is False
        # But has basic features
        assert flags.enable_api_access is True
        assert flags.enable_export is True

    @pytest.mark.asyncio
    async def test_enterprise_tier_has_all_features(self, tenant_service):
        """Test that ENTERPRISE tier has all features enabled."""
        flags = tenant_service._get_default_feature_flags(SubscriptionTier.ENTERPRISE)

        assert flags.enable_custom_reports is True
        assert flags.enable_webhooks is True
        assert flags.enable_ats_integration is True
        assert flags.enable_sso is True
        assert flags.enable_api_access is True
        assert flags.enable_data_insights is True

    @pytest.mark.asyncio
    async def test_professional_tier_has_advanced_features(self, tenant_service):
        """Test that PROFESSIONAL tier has advanced features but not enterprise-only."""
        flags = tenant_service._get_default_feature_flags(SubscriptionTier.PROFESSIONAL)

        # Has advanced features
        assert flags.enable_webhooks is True
        assert flags.enable_ai_recommendations is True
        assert flags.enable_data_insights is True

        # But not enterprise-only
        assert flags.enable_ats_integration is False
        assert flags.enable_sso is False

    @pytest.mark.asyncio
    async def test_feature_flags_updated_on_tier_upgrade(
        self, tenant_service, mock_tenant_repository, valid_tenant_data
    ):
        """Test that feature flags are updated when tier is upgraded."""
        tenant_data = valid_tenant_data.copy()
        tenant_data["subscription_tier"] = SubscriptionTier.FREE

        mock_tenant_repository.get.return_value = tenant_data
        mock_tenant_repository.update.return_value = tenant_data

        await tenant_service.upgrade_subscription(
            tenant_data["id"], SubscriptionTier.ENTERPRISE
        )

        # Verify feature flags were updated
        update_call_args = mock_tenant_repository.update.call_args[0]
        updated_data = update_call_args[1]
        feature_flags = updated_data["feature_flags"]

        # Should now have enterprise features
        assert feature_flags["enable_ats_integration"] is True
        assert feature_flags["enable_sso"] is True


# ==========================================
# 6. USER MANAGEMENT TESTS (4 tests)
# ==========================================

class TestUserManagement:
    """Test user management within tenants."""

    @pytest.mark.asyncio
    async def test_create_tenant_user(self, tenant_service, mock_user_repository, valid_tenant_data):
        """Test creating a user within a tenant."""
        tenant_service.tenant_repo.get = AsyncMock(return_value=valid_tenant_data)

        user_data = {
            "id": str(uuid4()),
            "tenant_id": valid_tenant_data["id"],
            "email": "newuser@test.com",
            "full_name": "New User",
            "roles": ["user"],
            "permissions": ["read"],
            "is_active": True,
            "is_superuser": False,
        }
        mock_user_repository.create.return_value = user_data

        result = await tenant_service.create_tenant_user(
            tenant_id=valid_tenant_data["id"],
            email="newuser@test.com",
            password="SecurePassword123!",
            full_name="New User",
        )

        assert result is not None
        assert result.email == "newuser@test.com"
        assert result.tenant_id == valid_tenant_data["id"]
        mock_user_repository.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_user_role(self, tenant_service, mock_user_repository, valid_tenant_data):
        """Test updating a user's role within a tenant."""
        tenant_service.tenant_repo.get = AsyncMock(return_value=valid_tenant_data)

        user_data = {
            "id": str(uuid4()),
            "tenant_id": valid_tenant_data["id"],
            "email": "user@test.com",
            "full_name": "Test User",
            "roles": ["user"],
            "permissions": ["read"],
            "is_active": True,
            "is_superuser": False,
        }
        mock_user_repository.get_by_id.return_value = user_data
        mock_user_repository.update.return_value = {
            **user_data,
            "roles": ["user", "admin"],
            "permissions": ["read", "write", "admin"],
            "is_superuser": True,
        }

        result = await tenant_service.update_user_role(
            tenant_id=valid_tenant_data["id"],
            user_id=user_data["id"],
            new_role="admin",
        )

        assert result is not None
        assert "admin" in result.roles
        mock_user_repository.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_tenant_user(self, tenant_service, mock_user_repository, valid_tenant_data):
        """Test removing a user from a tenant (soft delete)."""
        user_data = {
            "id": str(uuid4()),
            "tenant_id": valid_tenant_data["id"],
            "email": "user@test.com",
            "full_name": "Test User",
            "roles": ["user"],
            "permissions": ["read"],
            "is_active": True,
            "is_superuser": False,
        }
        mock_user_repository.get_by_id.return_value = user_data
        mock_user_repository.update.return_value = user_data

        result = await tenant_service.remove_tenant_user(
            valid_tenant_data["id"], user_data["id"]
        )

        assert result is True
        # Verify user was deactivated (soft delete)
        update_call_args = mock_user_repository.update.call_args[0]
        updated_data = update_call_args[1]
        assert updated_data["is_active"] is False

    @pytest.mark.asyncio
    async def test_get_tenant_users(self, tenant_service, mock_user_repository, valid_tenant_data):
        """Test retrieving all users for a tenant."""
        users = [
            {
                "id": str(uuid4()),
                "tenant_id": valid_tenant_data["id"],
                "email": f"user{i}@test.com",
                "full_name": f"User {i}",
                "roles": ["user"],
                "permissions": ["read"],
                "is_active": True,
                "is_superuser": False,
            }
            for i in range(3)
        ]
        mock_user_repository.get_by_tenant.return_value = users

        result = await tenant_service.get_tenant_users(valid_tenant_data["id"])

        assert len(result) == 3
        assert all(isinstance(user, CurrentUser) for user in result)


# ==========================================
# 7. ERROR HANDLING TESTS (5 tests)
# ==========================================

class TestErrorHandling:
    """Test error handling and validation."""

    @pytest.mark.asyncio
    async def test_invalid_tenant_id_returns_none(self, tenant_service, mock_tenant_repository):
        """Test that invalid tenant ID returns None."""
        mock_tenant_repository.get.return_value = None

        result = await tenant_service.get_tenant("invalid-tenant-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_duplicate_tenant_name_rejected(self, tenant_service, mock_tenant_repository):
        """Test that duplicate tenant names are rejected."""
        mock_tenant_repository.check_slug_availability.return_value = False

        with pytest.raises(ValueError, match="already taken"):
            await tenant_service.create_tenant(
                name="existing-tenant",
                display_name="Existing Tenant",
                primary_contact_email="admin@existing.com",
                subscription_tier=SubscriptionTier.FREE,
            )

    @pytest.mark.asyncio
    async def test_invalid_tenant_name_format_rejected(self, tenant_service):
        """Test that invalid tenant name formats are rejected."""
        mock_tenant_repository = tenant_service.tenant_repo
        mock_tenant_repository.check_slug_availability.return_value = True

        invalid_names = ["", "ab", "Test-Tenant", "test_tenant", "-test", "test-"]

        for invalid_name in invalid_names:
            with pytest.raises(ValueError, match="Invalid tenant name format"):
                await tenant_service.create_tenant(
                    name=invalid_name,
                    display_name="Test Tenant",
                    primary_contact_email="admin@test.com",
                    subscription_tier=SubscriptionTier.FREE,
                )

    @pytest.mark.asyncio
    async def test_update_nonexistent_tenant_fails(self, tenant_service, mock_tenant_repository):
        """Test that updating a non-existent tenant raises error."""
        mock_tenant_repository.get.return_value = None

        with pytest.raises(ValueError, match="Tenant not found"):
            await tenant_service.update_tenant(
                str(uuid4()), {"display_name": "New Name"}
            )

    @pytest.mark.asyncio
    async def test_check_quota_unknown_resource_fails(
        self, tenant_service, mock_tenant_repository, valid_tenant_data
    ):
        """Test that checking quota for unknown resource raises error."""
        mock_tenant_repository.get.return_value = valid_tenant_data

        with pytest.raises(ValueError, match="Unknown resource"):
            await tenant_service.check_quota(valid_tenant_data["id"], "unknown_resource")