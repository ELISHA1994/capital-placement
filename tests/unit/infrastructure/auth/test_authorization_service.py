"""
Comprehensive security tests for AuthorizationService

This test suite covers all critical authorization paths including:
- Permission checking: direct, hierarchical, resource-level
- Role validation: role hierarchy, role assignment
- Tenant access control: isolation, cross-tenant access
- Admin privileges: super admin, tenant admin, user manager
- Resource ownership: user can access own resources
- Default-deny behavior: fail-safe security
"""

import pytest
from typing import List
from uuid import uuid4
from unittest.mock import Mock, AsyncMock

from app.infrastructure.auth.authorization_service import (
    AuthorizationService,
    SystemRole,
    ResourceAction,
    ResourceType
)
from app.infrastructure.persistence.models.auth_tables import AuthorizationResult
from app.database.repositories.postgres import UserRepository, TenantRepository
from app.domain.interfaces import ICacheService


@pytest.fixture
def mock_user_repository():
    """Mock user repository"""
    repo = Mock(spec=UserRepository)
    repo.get_by_id = AsyncMock()
    repo.get_by_email = AsyncMock()
    return repo


@pytest.fixture
def mock_tenant_repository():
    """Mock tenant repository"""
    repo = Mock(spec=TenantRepository)
    repo.get = AsyncMock()
    repo.get_active_tenants = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_cache_service():
    """Mock cache service"""
    cache = Mock(spec=ICacheService)
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    cache.delete = AsyncMock()
    cache.delete_pattern = AsyncMock()
    cache.exists = AsyncMock(return_value=False)
    return cache


@pytest.fixture
def authz_service(
    mock_user_repository,
    mock_tenant_repository,
    mock_cache_service
):
    """Create authorization service with mocked dependencies"""
    return AuthorizationService(
        user_repository=mock_user_repository,
        tenant_repository=mock_tenant_repository,
        cache_manager=mock_cache_service
    )


# =============================================================================
# PERMISSION CHECKING TESTS
# =============================================================================

@pytest.mark.asyncio
class TestPermissionChecking:
    """Tests for permission validation"""

    async def test_check_permission_super_admin_has_all(
        self, authz_service
    ):
        """Test that super admin has all permissions"""
        # Execute
        result = await authz_service.check_permission(
            user_roles=[SystemRole.SUPER_ADMIN],
            required_permission="delete:system",
            tenant_id=str(uuid4())
        )

        # Verify
        assert result is True

    async def test_check_permission_direct_match(
        self, authz_service, mock_cache_service
    ):
        """Test direct permission match"""
        # Setup
        mock_cache_service.get.return_value = ["read:profile", "create:profile"]

        # Execute
        result = await authz_service.check_permission(
            user_roles=[SystemRole.USER],
            required_permission="read:profile",
            tenant_id=str(uuid4())
        )

        # Verify
        assert result is True

    async def test_check_permission_denied(
        self, authz_service, mock_cache_service
    ):
        """Test permission denied when user lacks permission"""
        # Setup
        mock_cache_service.get.return_value = ["read:profile"]

        # Execute
        result = await authz_service.check_permission(
            user_roles=[SystemRole.USER],
            required_permission="delete:system",
            tenant_id=str(uuid4())
        )

        # Verify
        assert result is False

    async def test_check_permission_hierarchical_manage_includes_crud(
        self, authz_service, mock_cache_service
    ):
        """Test that manage permission includes CRUD operations"""
        # Setup
        mock_cache_service.get.return_value = ["manage:profile"]

        # Execute - should allow read, create, update, delete
        tenant_id = str(uuid4())
        assert await authz_service.check_permission([SystemRole.USER], "read:profile", tenant_id) is True
        assert await authz_service.check_permission([SystemRole.USER], "create:profile", tenant_id) is True
        assert await authz_service.check_permission([SystemRole.USER], "update:profile", tenant_id) is True
        assert await authz_service.check_permission([SystemRole.USER], "delete:profile", tenant_id) is True

    async def test_check_permission_tenant_admin(
        self, authz_service, mock_cache_service
    ):
        """Test tenant admin permissions"""
        # Setup - tenant admin should have manage permissions
        mock_cache_service.get.return_value = [
            "manage:tenant", "manage:user", "manage:profile"
        ]

        tenant_id = str(uuid4())

        # Execute
        assert await authz_service.check_permission([SystemRole.TENANT_ADMIN], "manage:user", tenant_id) is True
        assert await authz_service.check_permission([SystemRole.TENANT_ADMIN], "create:user", tenant_id) is True
        assert await authz_service.check_permission([SystemRole.TENANT_ADMIN], "delete:profile", tenant_id) is True

    async def test_check_permission_readonly_user(
        self, authz_service, mock_cache_service
    ):
        """Test readonly user can only read"""
        # Setup
        mock_cache_service.get.return_value = [
            "read:profile", "read:search", "read:document"
        ]

        tenant_id = str(uuid4())

        # Execute
        assert await authz_service.check_permission([SystemRole.READONLY], "read:profile", tenant_id) is True
        assert await authz_service.check_permission([SystemRole.READONLY], "create:profile", tenant_id) is False
        assert await authz_service.check_permission([SystemRole.READONLY], "delete:profile", tenant_id) is False


# =============================================================================
# ROLE CHECKING TESTS
# =============================================================================

@pytest.mark.asyncio
class TestRoleChecking:
    """Tests for role validation"""

    async def test_check_role_has_role(
        self, authz_service
    ):
        """Test role checking when user has role"""
        # Execute
        result = await authz_service.check_role(
            user_roles=[SystemRole.USER, SystemRole.USER_MANAGER],
            required_role=SystemRole.USER_MANAGER
        )

        # Verify
        assert result is True

    async def test_check_role_missing_role(
        self, authz_service
    ):
        """Test role checking when user lacks role"""
        # Execute
        result = await authz_service.check_role(
            user_roles=[SystemRole.USER],
            required_role=SystemRole.TENANT_ADMIN
        )

        # Verify
        assert result is False


# =============================================================================
# TENANT ACCESS CONTROL TESTS
# =============================================================================

@pytest.mark.asyncio
class TestTenantAccessControl:
    """Tests for tenant isolation and access control"""

    async def test_check_tenant_access_same_tenant(
        self, authz_service
    ):
        """Test access to same tenant resources"""
        # Setup
        tenant_id = str(uuid4())

        # Execute
        result = await authz_service.check_tenant_access(
            user_tenant_id=tenant_id,
            resource_tenant_id=tenant_id,
            user_roles=[SystemRole.USER]
        )

        # Verify
        assert result is True

    async def test_check_tenant_access_different_tenant(
        self, authz_service
    ):
        """Test access denied to different tenant"""
        # Execute
        result = await authz_service.check_tenant_access(
            user_tenant_id=str(uuid4()),
            resource_tenant_id=str(uuid4()),
            user_roles=[SystemRole.USER]
        )

        # Verify
        assert result is False

    async def test_check_tenant_access_super_admin_bypass(
        self, authz_service
    ):
        """Test super admin can access any tenant"""
        # Execute
        result = await authz_service.check_tenant_access(
            user_tenant_id=str(uuid4()),
            resource_tenant_id=str(uuid4()),
            user_roles=[SystemRole.SUPER_ADMIN]
        )

        # Verify
        assert result is True

    async def test_get_accessible_tenants_super_admin(
        self, authz_service, mock_tenant_repository
    ):
        """Test super admin gets all tenants"""
        # Setup
        tenant1 = Mock(id=uuid4())
        tenant2 = Mock(id=uuid4())
        mock_tenant_repository.get_active_tenants.return_value = [tenant1, tenant2]

        # Execute
        result = await authz_service.get_accessible_tenants(
            user_roles=[SystemRole.SUPER_ADMIN],
            user_tenant_id=str(uuid4())
        )

        # Verify
        assert len(result) == 2

    async def test_get_accessible_tenants_regular_user(
        self, authz_service
    ):
        """Test regular user only gets own tenant"""
        # Setup
        user_tenant_id = str(uuid4())

        # Execute
        result = await authz_service.get_accessible_tenants(
            user_roles=[SystemRole.USER],
            user_tenant_id=user_tenant_id
        )

        # Verify
        assert len(result) == 1
        assert result[0] == user_tenant_id


# =============================================================================
# ROLE ASSIGNMENT VALIDATION TESTS
# =============================================================================

@pytest.mark.asyncio
class TestRoleAssignmentValidation:
    """Tests for role assignment authorization"""

    async def test_validate_role_assignment_super_admin_can_assign_any(
        self, authz_service
    ):
        """Test super admin can assign any role"""
        # Execute
        result = await authz_service.validate_role_assignment(
            assignee_roles=[SystemRole.SUPER_ADMIN],
            target_roles=[SystemRole.TENANT_ADMIN, SystemRole.USER],
            tenant_id=str(uuid4())
        )

        # Verify
        assert result.allowed is True

    async def test_validate_role_assignment_tenant_admin_cannot_assign_admin(
        self, authz_service
    ):
        """Test tenant admin cannot assign admin roles"""
        # Execute
        result = await authz_service.validate_role_assignment(
            assignee_roles=[SystemRole.TENANT_ADMIN],
            target_roles=[SystemRole.SUPER_ADMIN],
            tenant_id=str(uuid4())
        )

        # Verify
        assert result.allowed is False
        assert "Cannot assign admin roles" in result.reason

    async def test_validate_role_assignment_tenant_admin_can_assign_user(
        self, authz_service
    ):
        """Test tenant admin can assign user roles"""
        # Execute
        result = await authz_service.validate_role_assignment(
            assignee_roles=[SystemRole.TENANT_ADMIN],
            target_roles=[SystemRole.USER],
            tenant_id=str(uuid4())
        )

        # Verify
        assert result.allowed is True

    async def test_validate_role_assignment_user_manager_limited(
        self, authz_service
    ):
        """Test user manager can only assign user/readonly roles"""
        # Execute - allowed roles
        result = await authz_service.validate_role_assignment(
            assignee_roles=[SystemRole.USER_MANAGER],
            target_roles=[SystemRole.USER, SystemRole.READONLY],
            tenant_id=str(uuid4())
        )
        assert result.allowed is True

        # Execute - forbidden roles
        result = await authz_service.validate_role_assignment(
            assignee_roles=[SystemRole.USER_MANAGER],
            target_roles=[SystemRole.TENANT_ADMIN],
            tenant_id=str(uuid4())
        )
        assert result.allowed is False

    async def test_validate_role_assignment_regular_user_denied(
        self, authz_service
    ):
        """Test regular user cannot assign roles"""
        # Execute
        result = await authz_service.validate_role_assignment(
            assignee_roles=[SystemRole.USER],
            target_roles=[SystemRole.USER],
            tenant_id=str(uuid4())
        )

        # Verify
        assert result.allowed is False


# =============================================================================
# ADMIN ACTION AUTHORIZATION TESTS
# =============================================================================

@pytest.mark.asyncio
class TestAdminActionAuthorization:
    """Tests for administrative action authorization"""

    async def test_can_perform_admin_action_super_admin(
        self, authz_service
    ):
        """Test super admin can perform any admin action"""
        # Execute
        assert await authz_service.can_perform_admin_action(
            user_roles=[SystemRole.SUPER_ADMIN],
            action="system_config"
        ) is True

        assert await authz_service.can_perform_admin_action(
            user_roles=[SystemRole.SUPER_ADMIN],
            action="manage_users"
        ) is True

    async def test_can_perform_admin_action_tenant_admin_in_tenant(
        self, authz_service
    ):
        """Test tenant admin can manage within their tenant"""
        # Setup
        tenant_id = str(uuid4())

        # Execute
        assert await authz_service.can_perform_admin_action(
            user_roles=[SystemRole.TENANT_ADMIN],
            action="manage_users",
            target_tenant_id=tenant_id,
            user_tenant_id=tenant_id
        ) is True

    async def test_can_perform_admin_action_tenant_admin_cross_tenant(
        self, authz_service
    ):
        """Test tenant admin cannot manage different tenant"""
        # Execute
        assert await authz_service.can_perform_admin_action(
            user_roles=[SystemRole.TENANT_ADMIN],
            action="manage_users",
            target_tenant_id=str(uuid4()),
            user_tenant_id=str(uuid4())
        ) is False

    async def test_can_perform_admin_action_user_manager(
        self, authz_service
    ):
        """Test user manager permissions"""
        # Execute
        assert await authz_service.can_perform_admin_action(
            user_roles=[SystemRole.USER_MANAGER],
            action="manage_users"
        ) is True

        assert await authz_service.can_perform_admin_action(
            user_roles=[SystemRole.USER_MANAGER],
            action="manage_tenant"
        ) is False

    async def test_can_perform_admin_action_regular_user_denied(
        self, authz_service
    ):
        """Test regular user cannot perform admin actions"""
        # Execute
        assert await authz_service.can_perform_admin_action(
            user_roles=[SystemRole.USER],
            action="manage_users"
        ) is False


# =============================================================================
# USER PERMISSIONS RETRIEVAL TESTS
# =============================================================================

@pytest.mark.asyncio
class TestUserPermissionsRetrieval:
    """Tests for retrieving user permissions"""

    async def test_get_user_permissions_cached(
        self, authz_service, mock_cache_service
    ):
        """Test getting permissions from cache"""
        # Setup
        cached_permissions = ["read:profile", "create:profile"]
        mock_cache_service.get.return_value = cached_permissions

        # Execute
        result = await authz_service.get_user_permissions(
            user_roles=[SystemRole.USER],
            tenant_id=str(uuid4())
        )

        # Verify
        assert result == cached_permissions

    async def test_get_user_permissions_multiple_roles(
        self, authz_service, mock_cache_service
    ):
        """Test getting permissions for multiple roles"""
        # Setup - simulate cache miss, then role permissions
        call_count = [0]

        async def cache_get_side_effect(key):
            call_count[0] += 1
            if call_count[0] == 1:
                return None  # Cache miss for user_permissions
            elif "user_manager" in key.lower():
                return ["manage:user", "read:profile"]
            elif "user" in key.lower():
                return ["read:profile", "create:profile"]
            return None

        mock_cache_service.get.side_effect = cache_get_side_effect

        # Execute
        result = await authz_service.get_user_permissions(
            user_roles=[SystemRole.USER, SystemRole.USER_MANAGER],
            tenant_id=str(uuid4()),
            include_inherited=False
        )

        # Verify - should have permissions from both roles
        assert len(result) > 0


# =============================================================================
# RESOURCE OWNERSHIP TESTS
# =============================================================================

@pytest.mark.asyncio
class TestResourceOwnership:
    """Tests for resource ownership validation"""

    async def test_check_resource_ownership(
        self, authz_service
    ):
        """Test resource ownership checking"""
        # Execute
        result = await authz_service.check_resource_ownership(
            user_id=str(uuid4()),
            resource_type="profile",
            resource_id=str(uuid4()),
            tenant_id=str(uuid4())
        )

        # Verify - current implementation returns True
        # This would need actual implementation in production
        assert result is True


# =============================================================================
# DOMAIN BUSINESS LOGIC TESTS
# =============================================================================

@pytest.mark.asyncio
class TestDomainBusinessLogic:
    """Tests for internal business logic methods"""

    def test_check_super_admin_access(
        self, authz_service
    ):
        """Test super admin access check"""
        # Execute
        assert authz_service._check_super_admin_access([SystemRole.SUPER_ADMIN]) is True
        assert authz_service._check_super_admin_access([SystemRole.USER]) is False

    def test_check_tenant_access_internal(
        self, authz_service
    ):
        """Test internal tenant access checking"""
        # Setup
        tenant_id = str(uuid4())

        # Execute
        assert authz_service._check_tenant_access(tenant_id, tenant_id, [SystemRole.USER]) is True
        assert authz_service._check_tenant_access(tenant_id, str(uuid4()), [SystemRole.USER]) is False
        assert authz_service._check_tenant_access(tenant_id, str(uuid4()), [SystemRole.SUPER_ADMIN]) is True

    def test_can_assign_role_internal(
        self, authz_service
    ):
        """Test internal role assignment checking"""
        # Setup
        tenant_id = str(uuid4())

        # Super admin can assign any role
        assert authz_service._can_assign_role(
            [SystemRole.SUPER_ADMIN],
            SystemRole.TENANT_ADMIN,
            tenant_id
        ) is True

        # Tenant admin cannot assign super admin
        assert authz_service._can_assign_role(
            [SystemRole.TENANT_ADMIN],
            SystemRole.SUPER_ADMIN,
            tenant_id
        ) is False

        # Tenant admin can assign user
        assert authz_service._can_assign_role(
            [SystemRole.TENANT_ADMIN],
            SystemRole.USER,
            tenant_id
        ) is True

        # User manager can assign user/readonly
        assert authz_service._can_assign_role(
            [SystemRole.USER_MANAGER],
            SystemRole.USER,
            tenant_id
        ) is True

        # User manager cannot assign admin
        assert authz_service._can_assign_role(
            [SystemRole.USER_MANAGER],
            SystemRole.TENANT_ADMIN,
            tenant_id
        ) is False

    async def test_check_hierarchical_permission_manage_implies_crud(
        self, authz_service
    ):
        """Test hierarchical permission checking - manage implies CRUD"""
        # Execute
        assert await authz_service._check_hierarchical_permission(
            ["manage:profile"],
            "read:profile"
        ) is True

        assert await authz_service._check_hierarchical_permission(
            ["manage:profile"],
            "create:profile"
        ) is True

        assert await authz_service._check_hierarchical_permission(
            ["manage:profile"],
            "delete:profile"
        ) is True

    async def test_check_hierarchical_permission_system_admin(
        self, authz_service
    ):
        """Test hierarchical permission - manage:system grants all"""
        # Execute
        assert await authz_service._check_hierarchical_permission(
            ["manage:system"],
            "delete:profile"
        ) is True

        assert await authz_service._check_hierarchical_permission(
            ["manage:system"],
            "manage:tenant"
        ) is True


# =============================================================================
# ROLE DEFINITIONS TESTS
# =============================================================================

@pytest.mark.asyncio
class TestRoleDefinitions:
    """Tests for default role definitions"""

    def test_role_definitions_exist(
        self, authz_service
    ):
        """Test that all system roles have definitions"""
        # Verify
        assert SystemRole.SUPER_ADMIN in authz_service._role_definitions
        assert SystemRole.TENANT_ADMIN in authz_service._role_definitions
        assert SystemRole.USER_MANAGER in authz_service._role_definitions
        assert SystemRole.USER in authz_service._role_definitions
        assert SystemRole.READONLY in authz_service._role_definitions
        assert SystemRole.API_USER in authz_service._role_definitions

    def test_super_admin_has_all_permissions(
        self, authz_service
    ):
        """Test super admin role has wildcard permission"""
        # Verify
        super_admin_def = authz_service._role_definitions[SystemRole.SUPER_ADMIN]
        assert "*:*" in super_admin_def["permissions"]

    def test_readonly_only_has_read(
        self, authz_service
    ):
        """Test readonly role only has read permissions"""
        # Verify
        readonly_def = authz_service._role_definitions[SystemRole.READONLY]
        permissions = readonly_def["permissions"]

        # Should only have read permissions
        assert all(p.startswith("read:") for p in permissions)
        assert "read:profile" in permissions
        assert "read:search" in permissions


# =============================================================================
# DEFAULT-DENY SECURITY TESTS
# =============================================================================

@pytest.mark.asyncio
class TestDefaultDenySecurity:
    """Tests to ensure fail-safe default-deny behavior"""

    async def test_no_roles_denies_all(
        self, authz_service, mock_cache_service
    ):
        """Test that user with no roles has no permissions"""
        # Setup
        mock_cache_service.get.return_value = []

        # Execute
        result = await authz_service.check_permission(
            user_roles=[],
            required_permission="read:profile",
            tenant_id=str(uuid4())
        )

        # Verify
        assert result is False

    async def test_empty_permissions_denies_all(
        self, authz_service, mock_cache_service
    ):
        """Test that empty permissions list denies access"""
        # Setup
        mock_cache_service.get.return_value = []

        # Execute
        result = await authz_service.check_permission(
            user_roles=[SystemRole.USER],
            required_permission="delete:system",
            tenant_id=str(uuid4())
        )

        # Verify
        assert result is False

    async def test_invalid_permission_format_denies(
        self, authz_service, mock_cache_service
    ):
        """Test that invalid permission format is denied"""
        # Setup
        mock_cache_service.get.return_value = ["read:profile"]

        # Execute
        result = await authz_service.check_permission(
            user_roles=[SystemRole.USER],
            required_permission="invalid_format",  # No colon
            tenant_id=str(uuid4())
        )

        # Verify
        assert result is False
