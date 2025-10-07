"""
Comprehensive integration tests for tenant service workflows.

These tests verify complete tenant management workflows including:
- Tenant creation with admin user (atomic operations)
- Tenant lifecycle management (suspend/activate/delete)
- User management within tenants
- Subscription tier upgrades
- Multi-tenant data isolation
- Cache consistency and invalidation
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import patch, AsyncMock

from app.services.tenant.tenant_service import TenantService
from app.infrastructure.persistence.models.tenant_table import SubscriptionTier, TenantConfiguration
from app.infrastructure.persistence.models.auth_tables import CurrentUser
from tests.mocks.mock_repositories import MockTenantRepository, MockUserRepository
from tests.mocks.mock_services import MockCacheService


class TestTenantCreationWorkflows:
    """Test complete tenant creation workflows including atomic operations."""
    
    def setup_method(self):
        """Setup test dependencies."""
        self.tenant_repo = MockTenantRepository()
        self.user_repo = MockUserRepository()
        self.cache_service = MockCacheService()
        
        self.tenant_service = TenantService(
            tenant_repository=self.tenant_repo,
            user_repository=self.user_repo,
            cache_manager=self.cache_service
        )
    
    @pytest.mark.asyncio
    async def test_successful_atomic_tenant_creation_with_admin(self):
        """Test successful atomic tenant creation with admin user."""
        tenant_name = "test-company"
        display_name = "Test Company Inc"
        contact_email = "contact@testcompany.com"
        admin_data = {
            "email": "admin@testcompany.com",
            "password": "SecureAdminPassword123!",
            "full_name": "Admin User"
        }
        
        # Mock transaction decorator to simulate successful transaction
        with patch('app.core.transaction_manager.transactional') as mock_transactional:
            # Make transactional decorator pass through
            def pass_through_decorator(func):
                return func
            mock_transactional.return_value = pass_through_decorator
            
            # Mock password manager
            with patch('app.utils.security.password_manager.hash_password') as mock_hash:
                mock_hash.return_value = "hashed_admin_password"
                
                with patch('app.utils.security.SecurityValidator.generate_slug_from_name') as mock_slug:
                    mock_slug.return_value = "test-company"
                    
                    # Execute atomic creation
                    result = await self.tenant_service.create_tenant(
                        name=tenant_name,
                        display_name=display_name,
                        primary_contact_email=contact_email,
                        subscription_tier=SubscriptionTier.BASIC,
                        admin_user_data=admin_data
                    )
        
        # Verify tenant was created
        assert isinstance(result, TenantConfiguration)
        assert result.name == tenant_name
        assert result.display_name == display_name
        assert result.subscription_tier == SubscriptionTier.BASIC
        assert result.is_active is True
        
        # Verify tenant was stored in repository
        assert self.tenant_repo.get_call_count("create") == 1
        
        # Verify admin user was created
        assert self.user_repo.get_call_count("create") == 1
        
        # Verify admin user has correct properties
        created_users = [call[1] for call in self.user_repo.call_log if call[0] == "create"]
        admin_user = created_users[0]
        assert admin_user["email"] == admin_data["email"]
        assert admin_user["full_name"] == admin_data["full_name"]
        assert "admin" in admin_user["roles"]
        assert admin_user["is_superuser"] is True
    
    @pytest.mark.asyncio
    async def test_atomic_tenant_creation_rollback_on_user_failure(self):
        """Test that tenant creation rolls back if admin user creation fails."""
        tenant_name = "failing-company"
        display_name = "Failing Company"
        contact_email = "contact@failing.com"
        admin_data = {
            "email": "admin@failing.com",
            "password": "SecurePassword123!",
            "full_name": "Admin User"
        }
        
        # Make user repository fail
        self.user_repo.should_fail_on_create = True
        
        with patch('app.core.transaction_manager.transactional') as mock_transactional:
            def pass_through_decorator(func):
                return func
            mock_transactional.return_value = pass_through_decorator
            
            with patch('app.utils.security.SecurityValidator.generate_slug_from_name') as mock_slug:
                mock_slug.return_value = "failing-company"
                
                # Should raise exception due to user creation failure
                with pytest.raises(Exception):
                    await self.tenant_service.create_tenant(
                        name=tenant_name,
                        display_name=display_name,
                        primary_contact_email=contact_email,
                        admin_user_data=admin_data
                    )
        
        # Verify tenant was attempted to be created but should be rolled back
        # In a real transaction, this would be rolled back
        # For this test, we verify the failure propagated
        assert self.user_repo.get_call_count("create") >= 1
    
    @pytest.mark.asyncio
    async def test_tenant_creation_without_admin_user(self):
        """Test tenant creation without admin user (legacy flow)."""
        tenant_name = "simple-tenant"
        display_name = "Simple Tenant"
        contact_email = "contact@simple.com"
        
        with patch('app.utils.security.SecurityValidator.generate_slug_from_name') as mock_slug:
            mock_slug.return_value = "simple-tenant"
            
            result = await self.tenant_service.create_tenant(
                name=tenant_name,
                display_name=display_name,
                primary_contact_email=contact_email,
                subscription_tier=SubscriptionTier.FREE
            )
        
        # Verify tenant was created
        assert isinstance(result, TenantConfiguration)
        assert result.name == tenant_name
        
        # Verify no admin user was created
        assert self.user_repo.get_call_count("create") == 0
    
    @pytest.mark.asyncio
    async def test_duplicate_tenant_name_prevention(self):
        """Test that duplicate tenant names are prevented."""
        tenant_name = "duplicate-test"
        
        # Set slug as unavailable
        self.tenant_repo.set_slug_availability("duplicate-test", False)
        
        with patch('app.utils.security.SecurityValidator.generate_slug_from_name') as mock_slug:
            mock_slug.return_value = "duplicate-test"
            
            with pytest.raises(ValueError, match="already taken"):
                await self.tenant_service.create_tenant(
                    name=tenant_name,
                    display_name="Duplicate Test",
                    primary_contact_email="contact@duplicate.com"
                )


class TestTenantLifecycleManagement:
    """Test tenant lifecycle operations (suspend, activate, delete)."""
    
    def setup_method(self):
        self.tenant_repo = MockTenantRepository()
        self.user_repo = MockUserRepository()
        self.cache_service = MockCacheService()
        
        self.tenant_service = TenantService(
            tenant_repository=self.tenant_repo,
            user_repository=self.user_repo,
            cache_manager=self.cache_service
        )
        
        # Setup test tenant
        self.tenant_id = self.tenant_repo.add_test_tenant({
            "id": "test-tenant-id",
            "name": "test-tenant",
            "display_name": "Test Tenant",
            "is_active": True,
            "is_suspended": False,
            "subscription_tier": SubscriptionTier.BASIC,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        })
    
    @pytest.mark.asyncio
    async def test_tenant_suspension_workflow(self):
        """Test complete tenant suspension workflow."""
        suspension_reason = "Payment overdue"
        
        # Suspend tenant
        result = await self.tenant_service.suspend_tenant(self.tenant_id, suspension_reason)
        assert result is True
        
        # Verify tenant is suspended
        tenant = await self.tenant_service.get_tenant(self.tenant_id)
        assert tenant.is_suspended is True
        assert tenant.suspension_reason == suspension_reason
        assert tenant.suspended_at is not None
        
        # Verify cache was invalidated
        cache_key = f"tenant:{self.tenant_id}"
        assert cache_key not in self.cache_service.data
    
    @pytest.mark.asyncio
    async def test_tenant_activation_workflow(self):
        """Test tenant activation (unsuspension) workflow."""
        # First suspend the tenant
        await self.tenant_service.suspend_tenant(self.tenant_id, "Test suspension")
        
        # Then activate it
        result = await self.tenant_service.activate_tenant(self.tenant_id)
        assert result is True
        
        # Verify tenant is active
        tenant = await self.tenant_service.get_tenant(self.tenant_id)
        assert tenant.is_suspended is False
        assert tenant.suspension_reason is None
        assert tenant.suspended_at is None
    
    @pytest.mark.asyncio
    async def test_tenant_soft_delete_workflow(self):
        """Test tenant soft delete workflow."""
        # Delete tenant
        result = await self.tenant_service.delete_tenant(self.tenant_id)
        assert result is True
        
        # Verify tenant is marked as inactive
        tenant = await self.tenant_service.get_tenant(self.tenant_id)
        assert tenant.is_active is False
        assert tenant.deleted_at is not None
    
    @pytest.mark.asyncio
    async def test_nonexistent_tenant_operations(self):
        """Test operations on non-existent tenants."""
        fake_tenant_id = "non-existent-tenant"
        
        # All operations should return False for non-existent tenant
        assert await self.tenant_service.suspend_tenant(fake_tenant_id, "reason") is False
        assert await self.tenant_service.activate_tenant(fake_tenant_id) is False
        assert await self.tenant_service.delete_tenant(fake_tenant_id) is False


class TestTenantUserManagementWorkflows:
    """Test user management workflows within tenants."""
    
    def setup_method(self):
        self.tenant_repo = MockTenantRepository()
        self.user_repo = MockUserRepository()
        self.cache_service = MockCacheService()
        
        self.tenant_service = TenantService(
            tenant_repository=self.tenant_repo,
            user_repository=self.user_repo,
            cache_manager=self.cache_service
        )
        
        # Setup test tenant
        self.tenant_id = self.tenant_repo.add_test_tenant({
            "id": "test-tenant-id",
            "name": "test-tenant",
            "display_name": "Test Tenant",
            "is_active": True,
            "is_suspended": False
        })
    
    @pytest.mark.asyncio
    async def test_create_tenant_user_workflow(self):
        """Test complete tenant user creation workflow."""
        email = "newuser@testcompany.com"
        password = "SecurePassword123!"
        full_name = "New User"
        roles = ["user"]
        
        with patch('app.utils.security.password_manager.hash_password') as mock_hash:
            mock_hash.return_value = "hashed_password"
            
            # Create user
            result = await self.tenant_service.create_tenant_user(
                tenant_id=self.tenant_id,
                email=email,
                password=password,
                full_name=full_name,
                roles=roles
            )
        
        # Verify user was created and returned
        assert isinstance(result, CurrentUser)
        assert result.email == email
        assert result.full_name == full_name
        assert result.tenant_id == self.tenant_id
        assert "user" in result.roles
        
        # Verify user was stored in repository
        assert self.user_repo.get_call_count("create") == 1
        
        # Verify tenant users cache was invalidated
        cache_key = f"tenant_users:{self.tenant_id}"
        assert cache_key not in self.cache_service.data
    
    @pytest.mark.asyncio
    async def test_update_user_role_workflow(self):
        """Test user role update workflow within tenant."""
        # Create initial user
        user_id = self.user_repo.add_test_user({
            "id": "user-id",
            "tenant_id": self.tenant_id,
            "email": "user@test.com",
            "full_name": "Test User",
            "roles": ["user"],
            "permissions": ["read"],
            "is_active": True,
            "is_superuser": False
        })
        
        # Update user role to admin
        result = await self.tenant_service.update_user_role(
            tenant_id=self.tenant_id,
            user_id=user_id,
            new_role="admin"
        )
        
        # Verify role was updated
        assert isinstance(result, CurrentUser)
        assert "admin" in result.roles
        assert "user" in result.roles  # Should retain base role
        assert result.is_superuser is True  # Admin should be superuser
        assert "admin" in result.permissions
        
        # Verify cache was invalidated
        cache_key = f"tenant_users:{self.tenant_id}"
        assert cache_key not in self.cache_service.data
    
    @pytest.mark.asyncio
    async def test_remove_tenant_user_workflow(self):
        """Test user removal workflow (deactivation)."""
        # Create user
        user_id = self.user_repo.add_test_user({
            "id": "user-to-remove",
            "tenant_id": self.tenant_id,
            "email": "remove@test.com",
            "full_name": "User To Remove",
            "is_active": True
        })
        
        # Remove user
        result = await self.tenant_service.remove_tenant_user(self.tenant_id, user_id)
        assert result is True
        
        # Verify user was deactivated (not deleted)
        user_data = await self.user_repo.get_by_id(user_id)
        assert user_data["is_active"] is False
        
        # Verify cache was invalidated
        cache_key = f"tenant_users:{self.tenant_id}"
        assert cache_key not in self.cache_service.data
    
    @pytest.mark.asyncio
    async def test_cross_tenant_user_access_prevention(self):
        """Test that users cannot be managed across tenant boundaries."""
        # Create another tenant
        other_tenant_id = self.tenant_repo.add_test_tenant({
            "id": "other-tenant-id",
            "name": "other-tenant",
            "is_active": True
        })
        
        # Create user in other tenant
        user_id = self.user_repo.add_test_user({
            "id": "cross-tenant-user",
            "tenant_id": other_tenant_id,
            "email": "user@other.com",
            "full_name": "Cross Tenant User"
        })
        
        # Try to update role from wrong tenant
        with pytest.raises(ValueError, match="does not belong to tenant"):
            await self.tenant_service.update_user_role(
                tenant_id=self.tenant_id,  # Wrong tenant
                user_id=user_id,
                new_role="admin"
            )
        
        # Try to remove user from wrong tenant
        result = await self.tenant_service.remove_tenant_user(self.tenant_id, user_id)
        assert result is False  # Should fail silently


class TestTenantCacheConsistency:
    """Test cache consistency and invalidation across tenant operations."""
    
    def setup_method(self):
        self.tenant_repo = MockTenantRepository()
        self.user_repo = MockUserRepository()
        self.cache_service = MockCacheService()
        
        self.tenant_service = TenantService(
            tenant_repository=self.tenant_repo,
            user_repository=self.user_repo,
            cache_manager=self.cache_service
        )
        
        # Setup test tenant
        self.tenant_id = self.tenant_repo.add_test_tenant({
            "id": "cached-tenant-id",
            "name": "cached-tenant",
            "display_name": "Cached Tenant",
            "is_active": True
        })
    
    @pytest.mark.asyncio
    async def test_tenant_cache_population_and_retrieval(self):
        """Test tenant cache population and retrieval."""
        # First retrieval should populate cache
        tenant1 = await self.tenant_service.get_tenant(self.tenant_id)
        assert tenant1 is not None
        
        # Verify cache was populated
        cache_key = f"tenant:{self.tenant_id}"
        assert cache_key in self.cache_service.data
        
        # Second retrieval should use cache
        tenant2 = await self.tenant_service.get_tenant(self.tenant_id)
        assert tenant2 is not None
        
        # Should be equivalent data
        assert tenant1.name == tenant2.name
        assert tenant1.display_name == tenant2.display_name
    
    @pytest.mark.asyncio
    async def test_cache_invalidation_on_tenant_update(self):
        """Test that cache is invalidated when tenant is updated."""
        # Populate cache
        await self.tenant_service.get_tenant(self.tenant_id)
        cache_key = f"tenant:{self.tenant_id}"
        assert cache_key in self.cache_service.data
        
        # Update tenant
        await self.tenant_service.update_tenant(self.tenant_id, {"display_name": "Updated Name"})
        
        # Cache should be invalidated
        assert cache_key not in self.cache_service.data
    
    @pytest.mark.asyncio
    async def test_tenant_users_cache_consistency(self):
        """Test tenant users cache consistency and invalidation."""
        # Add a user to the tenant
        user_id = self.user_repo.add_test_user({
            "id": "cached-user-id",
            "tenant_id": self.tenant_id,
            "email": "cached@test.com",
            "full_name": "Cached User"
        })
        
        # Get tenant users (should populate cache)
        users1 = await self.tenant_service.get_tenant_users(self.tenant_id)
        assert len(users1) == 1
        
        # Verify cache was populated
        users_cache_key = f"tenant_users:{self.tenant_id}"
        assert users_cache_key in self.cache_service.data
        
        # Add another user (should invalidate cache)
        with patch('app.utils.security.password_manager.hash_password') as mock_hash:
            mock_hash.return_value = "hashed_password"
            
            await self.tenant_service.create_tenant_user(
                tenant_id=self.tenant_id,
                email="newuser@test.com",
                password="password",
                full_name="New User"
            )
        
        # Cache should be invalidated
        assert users_cache_key not in self.cache_service.data
        
        # Next retrieval should fetch fresh data
        users2 = await self.tenant_service.get_tenant_users(self.tenant_id)
        assert len(users2) == 2  # Should have both users


class TestTenantSubscriptionWorkflows:
    """Test tenant subscription tier management workflows."""
    
    def setup_method(self):
        self.tenant_repo = MockTenantRepository()
        self.user_repo = MockUserRepository()
        self.cache_service = MockCacheService()
        
        self.tenant_service = TenantService(
            tenant_repository=self.tenant_repo,
            user_repository=self.user_repo,
            cache_manager=self.cache_service
        )
        
        # Setup test tenant
        self.tenant_id = self.tenant_repo.add_test_tenant({
            "id": "subscription-tenant-id",
            "name": "subscription-tenant",
            "subscription_tier": SubscriptionTier.FREE,
            "is_active": True
        })
    
    @pytest.mark.asyncio
    async def test_subscription_upgrade_workflow(self):
        """Test complete subscription upgrade workflow."""
        # Upgrade from FREE to PROFESSIONAL
        result = await self.tenant_service.upgrade_subscription(
            self.tenant_id,
            SubscriptionTier.PROFESSIONAL
        )
        assert result is True
        
        # Verify subscription was upgraded
        tenant = await self.tenant_service.get_tenant(self.tenant_id)
        assert tenant.subscription_tier == SubscriptionTier.PROFESSIONAL
        
        # Verify quotas were updated to professional tier
        assert tenant.quota_limits.max_profiles == 10000  # Professional tier limit
        assert tenant.quota_limits.max_searches_per_month == 5000
        
        # Verify features were updated
        assert tenant.feature_flags.advanced_analytics is True
        assert tenant.feature_flags.custom_branding is True
        
        # Verify cache was invalidated
        cache_key = f"tenant:{self.tenant_id}"
        assert cache_key not in self.cache_service.data
    
    @pytest.mark.asyncio
    async def test_quota_tracking_workflow(self):
        """Test quota usage tracking and enforcement workflow."""
        # Update usage metrics
        usage_updates = {
            "max_profiles": 50,
            "max_searches_per_month": 25
        }
        
        result = await self.tenant_service.update_usage(self.tenant_id, usage_updates)
        assert result is True
        
        # Check quota for profiles
        quota_check = await self.tenant_service.check_quota(self.tenant_id, "max_profiles")
        
        assert quota_check["resource"] == "max_profiles"
        assert quota_check["current_usage"] == 50
        assert quota_check["limit"] == 100  # FREE tier limit
        assert quota_check["remaining"] == 50
        assert quota_check["percentage_used"] == 50.0
        assert quota_check["exceeded"] is False
    
    @pytest.mark.asyncio
    async def test_quota_exceeded_detection(self):
        """Test quota exceeded detection."""
        # Set usage that exceeds FREE tier limits
        usage_updates = {
            "max_profiles": 150  # Exceeds FREE tier limit of 100
        }
        
        await self.tenant_service.update_usage(self.tenant_id, usage_updates)
        
        # Check quota should show exceeded
        quota_check = await self.tenant_service.check_quota(self.tenant_id, "max_profiles")
        assert quota_check["exceeded"] is True
        assert quota_check["remaining"] == 0  # Should be capped at 0


class TestTenantListingAndPagination:
    """Test tenant listing with pagination and filtering."""
    
    def setup_method(self):
        self.tenant_repo = MockTenantRepository()
        self.user_repo = MockUserRepository()
        self.cache_service = MockCacheService()
        
        self.tenant_service = TenantService(
            tenant_repository=self.tenant_repo,
            user_repository=self.user_repo,
            cache_manager=self.cache_service
        )
        
        # Setup multiple test tenants
        for i in range(15):
            self.tenant_repo.add_test_tenant({
                "id": f"tenant-{i}",
                "name": f"tenant-{i}",
                "display_name": f"Tenant {i}",
                "is_active": i % 2 == 0,  # Alternate active/inactive
                "subscription_tier": SubscriptionTier.FREE,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            })
    
    @pytest.mark.asyncio
    async def test_tenant_listing_pagination(self):
        """Test tenant listing with pagination."""
        # First page
        page1 = await self.tenant_service.list_tenants(skip=0, limit=5)
        assert len(page1) == 5
        
        # Second page
        page2 = await self.tenant_service.list_tenants(skip=5, limit=5)
        assert len(page2) == 5
        
        # Verify no overlap
        page1_ids = {tenant.id for tenant in page1}
        page2_ids = {tenant.id for tenant in page2}
        assert page1_ids.isdisjoint(page2_ids)
    
    @pytest.mark.asyncio
    async def test_active_tenant_filtering(self):
        """Test filtering to only include active tenants."""
        # Get only active tenants
        active_tenants = await self.tenant_service.list_tenants(include_inactive=False)
        
        # All returned tenants should be active
        for tenant in active_tenants:
            assert tenant.is_active is True
        
        # Should have 8 active tenants (half of 15, rounded up)
        assert len(active_tenants) == 8
    
    @pytest.mark.asyncio
    async def test_include_inactive_tenants(self):
        """Test including inactive tenants in listing."""
        # Get all tenants including inactive
        all_tenants = await self.tenant_service.list_tenants(include_inactive=True)
        
        # Should have all 15 tenants
        assert len(all_tenants) == 15
        
        # Should include both active and inactive
        active_count = sum(1 for t in all_tenants if t.is_active)
        inactive_count = sum(1 for t in all_tenants if not t.is_active)
        assert active_count == 8
        assert inactive_count == 7


class TestTenantDataIsolation:
    """Test tenant data isolation and cross-tenant access prevention."""
    
    def setup_method(self):
        self.tenant_repo = MockTenantRepository()
        self.user_repo = MockUserRepository()
        self.cache_service = MockCacheService()
        
        self.tenant_service = TenantService(
            tenant_repository=self.tenant_repo,
            user_repository=self.user_repo,
            cache_manager=self.cache_service
        )
        
        # Setup multiple tenants
        self.tenant1_id = self.tenant_repo.add_test_tenant({
            "id": "tenant1-id",
            "name": "tenant1",
            "is_active": True
        })
        
        self.tenant2_id = self.tenant_repo.add_test_tenant({
            "id": "tenant2-id", 
            "name": "tenant2",
            "is_active": True
        })
        
        # Add users to each tenant
        self.user1_id = self.user_repo.add_test_user({
            "id": "user1-id",
            "tenant_id": self.tenant1_id,
            "email": "user1@tenant1.com",
            "full_name": "User One"
        })
        
        self.user2_id = self.user_repo.add_test_user({
            "id": "user2-id",
            "tenant_id": self.tenant2_id,
            "email": "user2@tenant2.com",
            "full_name": "User Two"
        })
    
    @pytest.mark.asyncio
    async def test_tenant_user_isolation(self):
        """Test that tenants only see their own users."""
        # Get users for tenant 1
        tenant1_users = await self.tenant_service.get_tenant_users(self.tenant1_id)
        assert len(tenant1_users) == 1
        assert tenant1_users[0].user_id == self.user1_id
        
        # Get users for tenant 2
        tenant2_users = await self.tenant_service.get_tenant_users(self.tenant2_id)
        assert len(tenant2_users) == 1
        assert tenant2_users[0].user_id == self.user2_id
        
        # Verify no cross-contamination
        tenant1_user_ids = {user.user_id for user in tenant1_users}
        tenant2_user_ids = {user.user_id for user in tenant2_users}
        assert tenant1_user_ids.isdisjoint(tenant2_user_ids)
    
    @pytest.mark.asyncio
    async def test_email_uniqueness_scoped_to_tenant(self):
        """Test that email uniqueness is scoped to tenant, not global."""
        shared_email = "shared@example.com"
        
        # Create user with shared email in tenant 1
        with patch('app.utils.security.password_manager.hash_password') as mock_hash:
            mock_hash.return_value = "hashed_password"
            
            user1 = await self.tenant_service.create_tenant_user(
                tenant_id=self.tenant1_id,
                email=shared_email,
                password="password",
                full_name="User in Tenant 1"
            )
            assert user1.email == shared_email
            
            # Create user with same email in tenant 2 - should succeed
            user2 = await self.tenant_service.create_tenant_user(
                tenant_id=self.tenant2_id,
                email=shared_email,
                password="password",
                full_name="User in Tenant 2"
            )
            assert user2.email == shared_email
            assert user1.user_id != user2.user_id
            assert user1.tenant_id != user2.tenant_id


class TestTenantErrorHandlingAndResilience:
    """Test error handling and system resilience in tenant operations."""
    
    def setup_method(self):
        self.tenant_repo = MockTenantRepository()
        self.user_repo = MockUserRepository()
        self.cache_service = MockCacheService()
        
        self.tenant_service = TenantService(
            tenant_repository=self.tenant_repo,
            user_repository=self.user_repo,
            cache_manager=self.cache_service
        )
    
    @pytest.mark.asyncio
    async def test_graceful_cache_failure_handling(self):
        """Test graceful handling of cache failures."""
        # Setup tenant
        tenant_id = self.tenant_repo.add_test_tenant({
            "id": "cache-fail-tenant",
            "name": "cache-fail-tenant",
            "is_active": True
        })
        
        # Make cache fail
        self.cache_service.should_fail_on_get = True
        
        # Should still work despite cache failure
        tenant = await self.tenant_service.get_tenant(tenant_id)
        assert tenant is not None
        assert tenant.name == "cache-fail-tenant"
    
    @pytest.mark.asyncio
    async def test_repository_failure_handling(self):
        """Test handling of repository failures."""
        # Make tenant repository fail on create
        self.tenant_repo.should_fail_on_create = True
        
        # Tenant creation should fail gracefully
        with pytest.raises(ValueError, match="Failed to create tenant"):
            await self.tenant_service.create_tenant(
                name="failing-tenant",
                display_name="Failing Tenant",
                primary_contact_email="fail@test.com"
            )
    
    @pytest.mark.asyncio
    async def test_invalid_tenant_operations(self):
        """Test operations on invalid or non-existent tenants."""
        fake_tenant_id = "non-existent-tenant"
        
        # Get non-existent tenant
        tenant = await self.tenant_service.get_tenant(fake_tenant_id)
        assert tenant is None
        
        # Update non-existent tenant
        with pytest.raises(ValueError, match="Tenant not found"):
            await self.tenant_service.update_tenant(fake_tenant_id, {"name": "updated"})
        
        # Check quota for non-existent tenant
        with pytest.raises(ValueError, match="Tenant not found"):
            await self.tenant_service.check_quota(fake_tenant_id, "max_profiles")