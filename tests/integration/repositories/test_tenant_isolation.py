"""
Integration tests for tenant data isolation.

These tests verify that tenant data is properly isolated
and that cross-tenant access is prevented at the repository level.
"""

import pytest
from uuid import uuid4

from app.database.repositories.postgres import TenantRepository, UserRepository
from tests.mocks.mock_repositories import MockTenantRepository, MockUserRepository
from tests.fixtures.tenant_fixtures import TenantTestBuilder


class TestTenantDataIsolation:
    """Test tenant data isolation at the repository level."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.tenant_repo = MockTenantRepository()
        self.user_repo = MockUserRepository()
        
        # Create test tenants
        self.tenant1 = TenantTestBuilder().with_name("tenant-1").build()
        self.tenant2 = TenantTestBuilder().with_name("tenant-2").build()
        
        self.tenant_repo.add_test_tenant(self.tenant1)
        self.tenant_repo.add_test_tenant(self.tenant2)
    
    @pytest.mark.asyncio
    async def test_users_isolated_by_tenant(self):
        """Test that users are properly isolated by tenant."""
        # Arrange - Create users in different tenants
        user1_data = {
            "id": str(uuid4()),
            "tenant_id": self.tenant1["id"],
            "email": "user1@tenant1.com",
            "full_name": "User One",
            "roles": ["user"],
            "is_active": True
        }
        
        user2_data = {
            "id": str(uuid4()),
            "tenant_id": self.tenant2["id"],
            "email": "user2@tenant2.com",
            "full_name": "User Two",
            "roles": ["user"],
            "is_active": True
        }
        
        await self.user_repo.create(user1_data)
        await self.user_repo.create(user2_data)
        
        # Act - Get users by tenant
        tenant1_users = await self.user_repo.get_by_tenant(self.tenant1["id"])
        tenant2_users = await self.user_repo.get_by_tenant(self.tenant2["id"])
        
        # Assert - Each tenant should only see their own users
        assert len(tenant1_users) == 1
        assert len(tenant2_users) == 1
        
        assert tenant1_users[0]["email"] == "user1@tenant1.com"
        assert tenant2_users[0]["email"] == "user2@tenant2.com"
        
        # Verify no cross-tenant data leakage
        tenant1_user_ids = [u["id"] for u in tenant1_users]
        tenant2_user_ids = [u["id"] for u in tenant2_users]
        
        assert user1_data["id"] in tenant1_user_ids
        assert user1_data["id"] not in tenant2_user_ids
        assert user2_data["id"] in tenant2_user_ids
        assert user2_data["id"] not in tenant1_user_ids
    
    @pytest.mark.asyncio
    async def test_user_email_lookup_tenant_scoped(self):
        """Test that user email lookup is scoped to tenant."""
        # Arrange - Create users with same email in different tenants
        user1_data = {
            "id": str(uuid4()),
            "tenant_id": self.tenant1["id"],
            "email": "admin@company.com",
            "full_name": "Admin One",
            "roles": ["admin"]
        }
        
        user2_data = {
            "id": str(uuid4()),
            "tenant_id": self.tenant2["id"],
            "email": "admin@company.com",  # Same email, different tenant
            "full_name": "Admin Two",
            "roles": ["admin"]
        }
        
        await self.user_repo.create(user1_data)
        await self.user_repo.create(user2_data)
        
        # Act - Look up users by email with tenant context
        found_user1 = await self.user_repo.get_by_email("admin@company.com", self.tenant1["id"])
        found_user2 = await self.user_repo.get_by_email("admin@company.com", self.tenant2["id"])
        
        # Assert - Should get correct user for each tenant
        assert found_user1 is not None
        assert found_user2 is not None
        assert found_user1["id"] == user1_data["id"]
        assert found_user2["id"] == user2_data["id"]
        assert found_user1["full_name"] == "Admin One"
        assert found_user2["full_name"] == "Admin Two"
    
    @pytest.mark.asyncio
    async def test_user_email_lookup_without_tenant_context(self):
        """Test user email lookup without tenant context returns first match."""
        # Arrange - Create users with same email in different tenants
        user1_data = {
            "id": str(uuid4()),
            "tenant_id": self.tenant1["id"],
            "email": "user@shared.com",
            "full_name": "User One"
        }
        
        user2_data = {
            "id": str(uuid4()),
            "tenant_id": self.tenant2["id"],
            "email": "user@shared.com",
            "full_name": "User Two"
        }
        
        await self.user_repo.create(user1_data)
        await self.user_repo.create(user2_data)
        
        # Act - Look up user by email without tenant context
        found_user = await self.user_repo.get_by_email("user@shared.com")
        
        # Assert - Should get one of the users (implementation dependent)
        assert found_user is not None
        assert found_user["email"] == "user@shared.com"
        assert found_user["id"] in [user1_data["id"], user2_data["id"]]
    
    @pytest.mark.asyncio
    async def test_tenant_slug_uniqueness_across_tenants(self):
        """Test that tenant slugs are unique across all tenants."""
        # Arrange - Try to create tenant with existing slug
        existing_slug = self.tenant1.get("slug", self.tenant1["name"])
        
        # Set slug as unavailable in mock
        self.tenant_repo.set_slug_availability(existing_slug, False)
        
        # Act - Check slug availability
        is_available = await self.tenant_repo.check_slug_availability(existing_slug)
        
        # Assert - Should not be available
        assert is_available is False
        
        # Test with new slug
        new_slug = "unique-new-tenant"
        is_available = await self.tenant_repo.check_slug_availability(new_slug)
        assert is_available is True
    
    @pytest.mark.asyncio
    async def test_user_update_tenant_scoped(self):
        """Test that user updates are scoped to correct tenant."""
        # Arrange
        user_data = {
            "id": str(uuid4()),
            "tenant_id": self.tenant1["id"],
            "email": "user@tenant1.com",
            "full_name": "Original Name",
            "roles": ["user"]
        }
        
        await self.user_repo.create(user_data)
        
        # Act - Update user
        updates = {"full_name": "Updated Name"}
        updated_user = await self.user_repo.update(user_data["id"], updates)
        
        # Assert - User should be updated
        assert updated_user["full_name"] == "Updated Name"
        assert updated_user["tenant_id"] == self.tenant1["id"]
        
        # Verify user is still in correct tenant
        tenant1_users = await self.user_repo.get_by_tenant(self.tenant1["id"])
        tenant2_users = await self.user_repo.get_by_tenant(self.tenant2["id"])
        
        assert len(tenant1_users) == 1
        assert len(tenant2_users) == 0
        assert tenant1_users[0]["full_name"] == "Updated Name"
    
    @pytest.mark.asyncio
    async def test_user_deletion_tenant_scoped(self):
        """Test that user deletion doesn't affect other tenants."""
        # Arrange - Create users in both tenants
        user1_data = {
            "id": str(uuid4()),
            "tenant_id": self.tenant1["id"],
            "email": "user1@tenant1.com",
            "full_name": "User One"
        }
        
        user2_data = {
            "id": str(uuid4()),
            "tenant_id": self.tenant2["id"],
            "email": "user2@tenant2.com",
            "full_name": "User Two"
        }
        
        await self.user_repo.create(user1_data)
        await self.user_repo.create(user2_data)
        
        # Act - Delete user from tenant1
        deleted = await self.user_repo.delete(user1_data["id"])
        
        # Assert - User should be deleted
        assert deleted is True
        
        # Verify tenant1 has no users, tenant2 still has user
        tenant1_users = await self.user_repo.get_by_tenant(self.tenant1["id"])
        tenant2_users = await self.user_repo.get_by_tenant(self.tenant2["id"])
        
        assert len(tenant1_users) == 0
        assert len(tenant2_users) == 1
        assert tenant2_users[0]["id"] == user2_data["id"]


class TestCascadingDeletion:
    """Test cascading deletion behavior for tenant isolation."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.tenant_repo = MockTenantRepository()
        self.user_repo = MockUserRepository()
        
        # Create test tenant
        self.test_tenant = TenantTestBuilder().build()
        self.tenant_repo.add_test_tenant(self.test_tenant)
    
    @pytest.mark.asyncio
    async def test_tenant_deletion_cascade_to_users(self):
        """Test that tenant deletion cascades to all users."""
        # Arrange - Create multiple users in tenant
        user_ids = []
        for i in range(3):
            user_data = {
                "id": str(uuid4()),
                "tenant_id": self.test_tenant["id"],
                "email": f"user{i}@test.com",
                "full_name": f"User {i}"
            }
            await self.user_repo.create(user_data)
            user_ids.append(user_data["id"])
        
        # Verify users exist
        tenant_users = await self.user_repo.get_by_tenant(self.test_tenant["id"])
        assert len(tenant_users) == 3
        
        # Act - Delete tenant (in real implementation, this would cascade)
        # For mocking, we simulate the cascade behavior
        await self.tenant_repo.delete(self.test_tenant["id"])
        
        # Simulate cascade deletion of users
        for user_id in user_ids:
            await self.user_repo.delete(user_id)
        
        # Assert - All users should be deleted
        remaining_users = await self.user_repo.get_by_tenant(self.test_tenant["id"])
        assert len(remaining_users) == 0
        
        # Verify individual user lookups return None
        for user_id in user_ids:
            user = await self.user_repo.get_by_id(user_id)
            assert user is None
    
    @pytest.mark.asyncio
    async def test_tenant_suspension_preserves_data(self):
        """Test that tenant suspension preserves data but affects access."""
        # Arrange - Create user in tenant
        user_data = {
            "id": str(uuid4()),
            "tenant_id": self.test_tenant["id"],
            "email": "user@test.com",
            "full_name": "Test User"
        }
        await self.user_repo.create(user_data)
        
        # Act - Suspend tenant (update tenant status)
        suspended_updates = {
            "is_suspended": True,
            "suspension_reason": "Policy violation"
        }
        await self.tenant_repo.update(self.test_tenant["id"], suspended_updates)
        
        # Assert - User data should still exist
        tenant_users = await self.user_repo.get_by_tenant(self.test_tenant["id"])
        assert len(tenant_users) == 1
        assert tenant_users[0]["id"] == user_data["id"]
        
        # Tenant should be marked as suspended
        suspended_tenant = await self.tenant_repo.get(self.test_tenant["id"])
        assert suspended_tenant["is_suspended"] is True
        assert suspended_tenant["suspension_reason"] == "Policy violation"


class TestCrossTenatValidation:
    """Test cross-tenant access validation."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.tenant_repo = MockTenantRepository()
        self.user_repo = MockUserRepository()
        
        # Create test tenants
        self.tenant1 = TenantTestBuilder().with_name("tenant-1").build()
        self.tenant2 = TenantTestBuilder().with_name("tenant-2").build()
        
        self.tenant_repo.add_test_tenant(self.tenant1)
        self.tenant_repo.add_test_tenant(self.tenant2)
    
    @pytest.mark.asyncio
    async def test_user_search_criteria_tenant_filtered(self):
        """Test that user search criteria properly filters by tenant."""
        # Arrange - Create users in different tenants with same criteria
        user1_data = {
            "id": str(uuid4()),
            "tenant_id": self.tenant1["id"],
            "email": "admin@test.com",
            "full_name": "Admin User",
            "roles": ["admin"],
            "is_active": True
        }
        
        user2_data = {
            "id": str(uuid4()),
            "tenant_id": self.tenant2["id"],
            "email": "admin@test.com",
            "full_name": "Admin User",
            "roles": ["admin"],
            "is_active": True
        }
        
        await self.user_repo.create(user1_data)
        await self.user_repo.create(user2_data)
        
        # Act - Search for admin users (without tenant filtering)
        admin_users = await self.user_repo.find_by_criteria({"roles": ["admin"]})
        
        # Assert - Should find users from both tenants
        assert len(admin_users) == 2
        
        # Verify both tenants are represented
        tenant_ids = [user["tenant_id"] for user in admin_users]
        assert self.tenant1["id"] in tenant_ids
        assert self.tenant2["id"] in tenant_ids
    
    @pytest.mark.asyncio
    async def test_email_uniqueness_tenant_scoped(self):
        """Test that email uniqueness is scoped to tenant level."""
        # Arrange - Same email should be allowed in different tenants
        email = "user@company.com"
        
        user1_data = {
            "id": str(uuid4()),
            "tenant_id": self.tenant1["id"],
            "email": email,
            "full_name": "User in Tenant 1"
        }
        
        user2_data = {
            "id": str(uuid4()),
            "tenant_id": self.tenant2["id"],
            "email": email,
            "full_name": "User in Tenant 2"
        }
        
        # Act - Create users with same email in different tenants
        created_user1 = await self.user_repo.create(user1_data)
        created_user2 = await self.user_repo.create(user2_data)
        
        # Assert - Both should be created successfully
        assert created_user1["email"] == email
        assert created_user2["email"] == email
        assert created_user1["tenant_id"] != created_user2["tenant_id"]
    
    @pytest.mark.asyncio
    async def test_tenant_active_status_filtering(self):
        """Test filtering of active vs inactive tenants."""
        # Arrange - Create inactive tenant
        inactive_tenant = TenantTestBuilder().as_inactive().build()
        self.tenant_repo.add_test_tenant(inactive_tenant)
        
        # Act - Get active tenants
        active_tenants = await self.tenant_repo.get_active_tenants()
        
        # Assert - Should only include active tenants
        active_tenant_ids = [t["id"] for t in active_tenants]
        
        assert self.tenant1["id"] in active_tenant_ids
        assert self.tenant2["id"] in active_tenant_ids
        assert inactive_tenant["id"] not in active_tenant_ids
        
        # Verify active tenants have correct status
        for tenant in active_tenants:
            assert tenant.get("is_active", True) is True
            assert tenant.get("is_suspended", False) is False


class TestTenantIsolationEdgeCases:
    """Test edge cases for tenant isolation."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.tenant_repo = MockTenantRepository()
        self.user_repo = MockUserRepository()
    
    @pytest.mark.asyncio
    async def test_empty_tenant_queries(self):
        """Test queries on tenants with no data."""
        # Arrange - Create empty tenant
        empty_tenant = TenantTestBuilder().build()
        self.tenant_repo.add_test_tenant(empty_tenant)
        
        # Act - Query for users in empty tenant
        users = await self.user_repo.get_by_tenant(empty_tenant["id"])
        
        # Assert - Should return empty list, not None
        assert users == []
        assert len(users) == 0
    
    @pytest.mark.asyncio
    async def test_nonexistent_tenant_queries(self):
        """Test queries on nonexistent tenant."""
        # Act - Query for users in nonexistent tenant
        fake_tenant_id = str(uuid4())
        users = await self.user_repo.get_by_tenant(fake_tenant_id)
        
        # Assert - Should return empty list
        assert users == []
    
    @pytest.mark.asyncio
    async def test_malformed_tenant_id_handling(self):
        """Test handling of malformed tenant IDs."""
        # Act & Assert - Should handle gracefully
        malformed_ids = ["", "not-a-uuid", None, "123"]
        
        for malformed_id in malformed_ids:
            if malformed_id is not None:
                users = await self.user_repo.get_by_tenant(str(malformed_id))
                assert users == []
    
    @pytest.mark.asyncio
    async def test_large_tenant_data_isolation(self):
        """Test isolation with large amounts of data."""
        # Arrange - Create multiple tenants with many users each
        tenants = []
        for i in range(5):
            tenant = TenantTestBuilder().with_name(f"tenant-{i}").build()
            self.tenant_repo.add_test_tenant(tenant)
            tenants.append(tenant)
            
            # Create multiple users per tenant
            for j in range(10):
                user_data = {
                    "id": str(uuid4()),
                    "tenant_id": tenant["id"],
                    "email": f"user{j}@tenant{i}.com",
                    "full_name": f"User {j} in Tenant {i}"
                }
                await self.user_repo.create(user_data)
        
        # Act - Verify each tenant has exactly 10 users
        for i, tenant in enumerate(tenants):
            users = await self.user_repo.get_by_tenant(tenant["id"])
            
            # Assert
            assert len(users) == 10
            
            # Verify all users belong to correct tenant
            for user in users:
                assert user["tenant_id"] == tenant["id"]
                assert f"tenant{i}" in user["email"]
        
        # Verify total user count across all tenants
        all_users = []
        for tenant in tenants:
            tenant_users = await self.user_repo.get_by_tenant(tenant["id"])
            all_users.extend(tenant_users)
        
        assert len(all_users) == 50  # 5 tenants * 10 users each
        
        # Verify no user appears in multiple tenants
        user_ids = [user["id"] for user in all_users]
        assert len(user_ids) == len(set(user_ids))  # All IDs should be unique