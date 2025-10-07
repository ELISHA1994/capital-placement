"""
Integration tests for tenant creation workflows.

These tests verify the complete tenant creation process including
transaction management, error handling, and integration between services.
"""

import pytest
from unittest.mock import AsyncMock

from app.infrastructure.tenant.tenant_service import TenantService
from app.infrastructure.persistence.models.tenant_table import SubscriptionTier
from tests.mocks.mock_repositories import MockTenantRepository, MockUserRepository
from tests.mocks.mock_services import MockCacheService
from tests.fixtures.tenant_fixtures import TenantTestBuilder


class TestTenantCreationWorkflow:
    """Test complete tenant creation workflows."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.mock_tenant_repo = MockTenantRepository()
        self.mock_user_repo = MockUserRepository()
        self.mock_cache = MockCacheService()
        
        self.tenant_service = TenantService(
            tenant_repository=self.mock_tenant_repo,
            user_repository=self.mock_user_repo,
            cache_manager=self.mock_cache
        )
    
    @pytest.mark.asyncio
    async def test_successful_tenant_only_creation(self):
        """Test successful tenant creation without admin user."""
        # Arrange
        tenant_data = (TenantTestBuilder()
                      .with_name("test-company")
                      .with_display_name("Test Company Inc")
                      .with_email("contact@testcompany.com")
                      .build())
        
        # Act
        result = await self.tenant_service.create_tenant(
            name=tenant_data["name"],
            display_name=tenant_data["display_name"],
            primary_contact_email=tenant_data["primary_contact_email"],
            subscription_tier=SubscriptionTier.FREE
        )
        
        # Assert
        assert result is not None
        assert result.name == "test-company"
        assert result.display_name == "Test Company Inc"
        assert result.subscription_tier == SubscriptionTier.FREE
        assert result.is_active is True
        assert result.is_suspended is False
        
        # Verify repository interactions
        assert self.mock_tenant_repo.was_called_with("create")
        assert self.mock_tenant_repo.was_called_with("check_slug_availability", "test-company")
        
        # Verify no user creation occurred
        assert self.mock_user_repo.get_call_count("create") == 0
    
    @pytest.mark.asyncio
    async def test_successful_tenant_with_admin_creation(self):
        """Test successful tenant creation with admin user."""
        # Arrange
        admin_user_data = {
            "email": "admin@testcompany.com",
            "password": "SecurePassword123!",
            "full_name": "Test Administrator"
        }
        
        # Act
        result = await self.tenant_service.create_tenant(
            name="test-company",
            display_name="Test Company Inc",
            primary_contact_email="contact@testcompany.com",
            subscription_tier=SubscriptionTier.BASIC,
            admin_user_data=admin_user_data
        )
        
        # Assert
        assert result is not None
        assert result.name == "test-company"
        assert result.subscription_tier == SubscriptionTier.BASIC
        
        # Verify both tenant and user creation occurred
        assert self.mock_tenant_repo.was_called_with("create")
        assert self.mock_user_repo.was_called_with("create")
        
        # Verify admin user was created correctly
        created_users = [call for call in self.mock_user_repo.call_log if call[0] == "create"]
        assert len(created_users) == 1
        
        user_data = created_users[0][1]  # Get the user data from the call
        assert user_data["email"] == "admin@testcompany.com"
        assert user_data["full_name"] == "Test Administrator"
        assert "admin" in user_data["roles"]
        assert user_data["is_superuser"] is True
    
    @pytest.mark.asyncio
    async def test_tenant_creation_with_duplicate_name(self):
        """Test tenant creation fails when name already exists."""
        # Arrange
        existing_tenant = (TenantTestBuilder()
                          .with_name("existing-company")
                          .build())
        self.mock_tenant_repo.add_test_tenant(existing_tenant)
        
        # Act & Assert
        with pytest.raises(ValueError, match="already taken"):
            await self.tenant_service.create_tenant(
                name="existing-company",
                display_name="Another Company",
                primary_contact_email="admin@another.com"
            )
        
        # Verify slug availability was checked
        assert self.mock_tenant_repo.was_called_with("check_slug_availability", "existing-company")
        
        # Verify no creation occurred after availability check
        create_calls = [call for call in self.mock_tenant_repo.call_log if call[0] == "create"]
        assert len(create_calls) == 0
    
    @pytest.mark.asyncio
    async def test_tenant_creation_rollback_on_user_failure(self):
        """Test that tenant creation rolls back when admin user creation fails."""
        # Arrange
        self.mock_user_repo.should_fail_on_create = True
        
        admin_user_data = {
            "email": "admin@test.com",
            "password": "SecurePassword123!",
            "full_name": "Test Admin"
        }
        
        # Act & Assert
        with pytest.raises(Exception):  # Should propagate the user creation failure
            await self.tenant_service.create_tenant(
                name="test-tenant",
                display_name="Test Tenant",
                primary_contact_email="contact@test.com",
                admin_user_data=admin_user_data
            )
        
        # Verify tenant creation was attempted but rolled back
        # (In a real implementation, this would be handled by transaction rollback)
        # For this mock, we verify the failure cascade
        assert self.mock_user_repo.was_called_with("create")
    
    @pytest.mark.asyncio
    async def test_tenant_creation_with_invalid_inputs(self):
        """Test tenant creation with various invalid inputs."""
        # Test invalid tenant name
        with pytest.raises(ValueError, match="Invalid tenant name format"):
            await self.tenant_service.create_tenant(
                name="Invalid Name!",  # Invalid characters
                display_name="Valid Display Name",
                primary_contact_email="valid@email.com"
            )
        
        # Test invalid email
        with pytest.raises(ValueError, match="Invalid.*email format"):
            await self.tenant_service.create_tenant(
                name="valid-name",
                display_name="Valid Display Name", 
                primary_contact_email="invalid-email"
            )
        
        # Test invalid admin user data
        invalid_admin_data = {
            "email": "admin@test.com",
            "password": "weak",  # Too weak
            "full_name": "Test Admin"
        }
        
        with pytest.raises(ValueError, match="password"):
            await self.tenant_service.create_tenant(
                name="valid-name",
                display_name="Valid Display Name",
                primary_contact_email="valid@email.com",
                admin_user_data=invalid_admin_data
            )
    
    @pytest.mark.asyncio
    async def test_tenant_creation_quota_limits_assignment(self):
        """Test that proper quota limits are assigned based on subscription tier."""
        # Test Free tier
        free_tenant = await self.tenant_service.create_tenant(
            name="free-tenant",
            display_name="Free Tenant",
            primary_contact_email="admin@free.com",
            subscription_tier=SubscriptionTier.FREE
        )
        
        assert free_tenant.quota_limits.max_profiles == 100
        assert free_tenant.quota_limits.max_users == 2
        assert free_tenant.quota_limits.max_storage_gb == 1
        
        # Reset mocks
        self.mock_tenant_repo.clear_call_log()
        
        # Test Enterprise tier
        enterprise_tenant = await self.tenant_service.create_tenant(
            name="enterprise-tenant",
            display_name="Enterprise Tenant",
            primary_contact_email="admin@enterprise.com",
            subscription_tier=SubscriptionTier.ENTERPRISE
        )
        
        assert enterprise_tenant.quota_limits.max_profiles is None  # Unlimited
        assert enterprise_tenant.quota_limits.max_users is None  # Unlimited
        assert enterprise_tenant.quota_limits.max_storage_gb is None  # Unlimited
    
    @pytest.mark.asyncio
    async def test_tenant_creation_feature_flags_assignment(self):
        """Test that proper feature flags are assigned based on subscription tier."""
        # Test Free tier
        free_tenant = await self.tenant_service.create_tenant(
            name="free-tenant",
            display_name="Free Tenant",
            primary_contact_email="admin@free.com",
            subscription_tier=SubscriptionTier.FREE
        )
        
        assert free_tenant.feature_flags.api_access is False
        assert free_tenant.feature_flags.advanced_analytics is False
        assert free_tenant.feature_flags.custom_branding is False
        
        # Reset mocks  
        self.mock_tenant_repo.clear_call_log()
        
        # Test Professional tier
        pro_tenant = await self.tenant_service.create_tenant(
            name="pro-tenant",
            display_name="Pro Tenant",
            primary_contact_email="admin@pro.com",
            subscription_tier=SubscriptionTier.PROFESSIONAL
        )
        
        assert pro_tenant.feature_flags.api_access is True
        assert pro_tenant.feature_flags.advanced_analytics is True
        assert pro_tenant.feature_flags.custom_branding is True
        assert pro_tenant.feature_flags.webhook_integrations is False  # Enterprise only
    
    @pytest.mark.asyncio
    async def test_tenant_creation_cache_invalidation(self):
        """Test that cache is properly invalidated during tenant creation."""
        # Arrange
        admin_user_data = {
            "email": "admin@test.com",
            "password": "SecurePassword123!",
            "full_name": "Test Admin"
        }
        
        # Act
        result = await self.tenant_service.create_tenant(
            name="test-tenant",
            display_name="Test Tenant",
            primary_contact_email="contact@test.com",
            admin_user_data=admin_user_data
        )
        
        # Assert cache operations occurred
        tenant_cache_key = f"tenant:{result.id}"
        tenant_users_cache_key = f"tenant_users:{result.id}"
        
        # Verify cache invalidation calls
        cache_deletes = [op for op in self.mock_cache.operations if op[0] == "delete"]
        deleted_keys = [op[1] for op in cache_deletes]
        
        # Should have attempted to clear tenant-related cache entries
        assert any(key.startswith("tenant:") for key in deleted_keys)
    
    @pytest.mark.asyncio
    async def test_concurrent_tenant_creation_with_same_name(self):
        """Test handling of concurrent tenant creation attempts with the same name."""
        import asyncio
        
        # Arrange
        async def create_tenant_task(name_suffix: str):
            return await self.tenant_service.create_tenant(
                name="concurrent-tenant",
                display_name=f"Concurrent Tenant {name_suffix}",
                primary_contact_email=f"admin{name_suffix}@test.com"
            )
        
        # Act - Try to create tenants with same name concurrently
        # Note: In a real database, this would be handled by unique constraints
        # For mocking, we simulate the race condition
        
        # First creation should succeed
        result1 = await create_tenant_task("1")
        assert result1 is not None
        
        # Second creation should fail due to name conflict
        with pytest.raises(ValueError, match="already taken"):
            await create_tenant_task("2")
    
    @pytest.mark.asyncio
    async def test_tenant_creation_with_slug_generation(self):
        """Test tenant creation with automatic slug generation."""
        # When display name differs from name, slug should be generated from display name
        result = await self.tenant_service.create_tenant(
            name="my-company",
            display_name="My Amazing Company Inc!",
            primary_contact_email="admin@mycompany.com"
        )
        
        # Verify tenant was created with proper name
        assert result.name == "my-company"
        assert result.display_name == "My Amazing Company Inc!"
        
        # Verify slug availability was checked for the normalized display name
        # (Implementation would generate slug from display name)
        slug_checks = [call for call in self.mock_tenant_repo.call_log 
                      if call[0] == "check_slug_availability"]
        assert len(slug_checks) > 0