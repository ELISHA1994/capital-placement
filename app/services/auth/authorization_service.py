"""
Authorization Service Implementation

Provides role-based access control (RBAC) and permission management:
- Role and permission validation
- Tenant-level access control
- Resource-level authorization
- Admin privileges management
- Permission inheritance and hierarchies
"""

from typing import Dict, Any, List, Optional, Set
from enum import Enum

import structlog
from app.core.config import get_settings
from app.models.auth import CurrentUser, AuthorizationResult, Permission, UserRole
from app.database.repositories.postgres import UserRepository, TenantRepository
# Note: RoleRepository and PermissionRepository not yet implemented in PostgreSQL
from app.services.adapters.memory_cache_adapter import MemoryCacheService

logger = structlog.get_logger(__name__)


class ResourceAction(str, Enum):
    """Standard resource actions"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LIST = "list"
    EXECUTE = "execute"
    MANAGE = "manage"


class ResourceType(str, Enum):
    """Standard resource types"""
    PROFILE = "profile"
    SEARCH = "search"
    DOCUMENT = "document"
    USER = "user"
    TENANT = "tenant"
    API_KEY = "api_key"
    SYSTEM = "system"


class SystemRole(str, Enum):
    """Predefined system roles"""
    SUPER_ADMIN = "super_admin"
    TENANT_ADMIN = "tenant_admin"
    USER_MANAGER = "user_manager"
    USER = "user"
    READONLY = "readonly"
    API_USER = "api_user"


class AuthorizationService:
    """Core authorization service implementation"""
    
    def __init__(
        self,
        user_repository: UserRepository,
        tenant_repository: TenantRepository,
        cache_manager: MemoryCacheService,
        role_repository=None,  # Not yet implemented
        permission_repository=None  # Not yet implemented
    ):
        self.user_repo = user_repository
        self.role_repo = role_repository  # Will be None for now
        self.permission_repo = permission_repository  # Will be None for now
        self.tenant_repo = tenant_repository
        self.cache = cache_manager
        self.settings = get_settings()
        
        # Cache keys
        self.ROLE_CACHE_PREFIX = "role:"
        self.PERMISSION_CACHE_PREFIX = "permission:"
        self.USER_PERMISSIONS_CACHE_PREFIX = "user_permissions:"
        
        # Initialize role definitions
        self._role_definitions = self._get_default_role_definitions()
    
    async def check_permission(
        self,
        user_roles: List[str],
        required_permission: str,
        tenant_id: str,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """Check if user has required permission"""
        
        try:
            # Super admin has all permissions
            if SystemRole.SUPER_ADMIN in user_roles:
                return True
            
            # Get all permissions for user roles
            user_permissions = await self._get_permissions_for_roles(user_roles, tenant_id)
            
            # Check direct permission match
            if required_permission in user_permissions:
                return True
            
            # Check hierarchical permissions
            if await self._check_hierarchical_permission(required_permission, user_permissions):
                return True
            
            # Check resource-level permissions
            if resource_id and user_id:
                if await self._check_resource_permission(
                    required_permission, 
                    resource_id, 
                    user_id, 
                    tenant_id
                ):
                    return True
            
            return False
            
        except Exception as e:
            logger.error("Permission check failed", error=str(e), permission=required_permission)
            return False
    
    async def check_role(self, user_roles: List[str], required_role: str) -> bool:
        """Check if user has required role"""
        return required_role in user_roles
    
    async def check_tenant_access(
        self, 
        user_tenant_id: str, 
        resource_tenant_id: str,
        user_roles: Optional[List[str]] = None
    ) -> bool:
        """Check if user has access to tenant resources"""
        
        # Super admin can access all tenants
        if user_roles and SystemRole.SUPER_ADMIN in user_roles:
            return True
        
        # Users can only access resources in their own tenant
        return user_tenant_id == resource_tenant_id
    
    async def check_resource_ownership(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        tenant_id: str
    ) -> bool:
        """Check if user owns the resource"""
        
        try:
            # This would need to be implemented based on resource type
            # For now, we'll return True for simplicity
            return True
            
        except Exception as e:
            logger.error("Resource ownership check failed", error=str(e))
            return False
    
    async def get_user_permissions(
        self, 
        user_roles: List[str], 
        tenant_id: str,
        include_inherited: bool = True
    ) -> List[str]:
        """Get all permissions for user"""
        
        try:
            # Check cache first
            cache_key = f"{self.USER_PERMISSIONS_CACHE_PREFIX}{tenant_id}:{':'.join(sorted(user_roles))}"
            cached_permissions = await self.cache.get(cache_key)
            
            if cached_permissions:
                return cached_permissions
            
            # Get permissions for all roles
            all_permissions = set()
            
            for role in user_roles:
                role_permissions = await self._get_role_permissions(role, tenant_id)
                all_permissions.update(role_permissions)
            
            # Add inherited permissions if requested
            if include_inherited:
                inherited = await self._get_inherited_permissions(list(all_permissions))
                all_permissions.update(inherited)
            
            permissions_list = list(all_permissions)
            
            # Cache for 10 minutes
            await self.cache.set(cache_key, permissions_list, ttl=600)
            
            return permissions_list
            
        except Exception as e:
            logger.error("Failed to get user permissions", error=str(e))
            return []
    
    async def validate_role_assignment(
        self,
        assignee_roles: List[str],
        target_roles: List[str],
        tenant_id: str
    ) -> AuthorizationResult:
        """Validate if user can assign roles to another user"""
        
        try:
            # Super admin can assign any role
            if SystemRole.SUPER_ADMIN in assignee_roles:
                return AuthorizationResult(allowed=True)
            
            # Tenant admin can assign non-admin roles within their tenant
            if SystemRole.TENANT_ADMIN in assignee_roles:
                forbidden_roles = [SystemRole.SUPER_ADMIN, SystemRole.TENANT_ADMIN]
                if any(role in target_roles for role in forbidden_roles):
                    return AuthorizationResult(
                        allowed=False,
                        reason="Cannot assign admin roles",
                        required_permissions=["manage:admin_roles"]
                    )
                return AuthorizationResult(allowed=True)
            
            # User manager can assign user roles
            if SystemRole.USER_MANAGER in assignee_roles:
                allowed_roles = [SystemRole.USER, SystemRole.READONLY]
                if all(role in allowed_roles for role in target_roles):
                    return AuthorizationResult(allowed=True)
                else:
                    return AuthorizationResult(
                        allowed=False,
                        reason="Can only assign user and readonly roles",
                        required_permissions=["manage:users"]
                    )
            
            return AuthorizationResult(
                allowed=False,
                reason="Insufficient privileges for role assignment",
                required_permissions=["manage:users"]
            )
            
        except Exception as e:
            logger.error("Role assignment validation failed", error=str(e))
            return AuthorizationResult(
                allowed=False,
                reason="Authorization check failed"
            )
    
    async def get_accessible_tenants(self, user_roles: List[str], user_tenant_id: str) -> List[str]:
        """Get list of tenants user can access"""
        
        # Super admin can access all tenants
        if SystemRole.SUPER_ADMIN in user_roles:
            tenants = await self.tenant_repo.get_active_tenants()
            return [str(tenant.id) for tenant in tenants]
        
        # Regular users can only access their own tenant
        return [user_tenant_id]
    
    async def can_perform_admin_action(
        self,
        user_roles: List[str],
        action: str,
        target_tenant_id: Optional[str] = None,
        user_tenant_id: Optional[str] = None
    ) -> bool:
        """Check if user can perform administrative actions"""
        
        # Super admin can perform any admin action
        if SystemRole.SUPER_ADMIN in user_roles:
            return True
        
        # Tenant admin can perform actions within their tenant
        if SystemRole.TENANT_ADMIN in user_roles:
            if target_tenant_id and user_tenant_id:
                return target_tenant_id == user_tenant_id
            return True
        
        # Specific admin actions
        admin_actions = {
            "manage_users": [SystemRole.USER_MANAGER, SystemRole.TENANT_ADMIN],
            "manage_tenant": [SystemRole.TENANT_ADMIN],
            "view_audit_logs": [SystemRole.TENANT_ADMIN],
            "manage_api_keys": [SystemRole.TENANT_ADMIN],
            "system_config": [SystemRole.SUPER_ADMIN]
        }
        
        allowed_roles = admin_actions.get(action, [])
        return any(role in user_roles for role in allowed_roles)
    
    # Private helper methods
    
    async def _get_permissions_for_roles(self, roles: List[str], tenant_id: str) -> Set[str]:
        """Get all permissions for a list of roles"""
        
        all_permissions = set()
        
        for role in roles:
            role_permissions = await self._get_role_permissions(role, tenant_id)
            all_permissions.update(role_permissions)
        
        return all_permissions
    
    async def _get_role_permissions(self, role: str, tenant_id: str) -> List[str]:
        """Get permissions for a specific role"""
        
        try:
            # Check cache first
            cache_key = f"{self.ROLE_CACHE_PREFIX}{tenant_id}:{role}"
            cached_permissions = await self.cache.get(cache_key)
            
            if cached_permissions:
                return cached_permissions
            
            # Get from default role definitions
            if role in self._role_definitions:
                permissions = self._role_definitions[role]["permissions"]
            else:
                # Get custom role from database
                role_data = await self.role_repo.get_by_name(role, tenant_id)
                permissions = role_data.permissions if role_data else []
            
            # Cache for 30 minutes
            await self.cache.set(cache_key, permissions, ttl=1800)
            
            return permissions
            
        except Exception as e:
            logger.error("Failed to get role permissions", role=role, error=str(e))
            return []
    
    async def _check_hierarchical_permission(
        self, 
        required_permission: str, 
        user_permissions: Set[str]
    ) -> bool:
        """Check permission hierarchies (e.g., manage:* includes read:*, write:*)"""
        
        try:
            # Parse required permission
            parts = required_permission.split(":")
            if len(parts) != 2:
                return False
            
            action, resource = parts
            
            # Check for wildcard permissions
            wildcard_permissions = [
                f"*:*",  # Full admin
                f"*:{resource}",  # Full access to resource
                f"{action}:*",  # Action on all resources
                f"manage:{resource}",  # Manage includes all actions
            ]
            
            # For read actions, also check list permissions
            if action == "read":
                wildcard_permissions.append(f"list:{resource}")
            
            return any(perm in user_permissions for perm in wildcard_permissions)
            
        except Exception:
            return False
    
    async def _check_resource_permission(
        self,
        required_permission: str,
        resource_id: str,
        user_id: str,
        tenant_id: str
    ) -> bool:
        """Check resource-level permissions (e.g., can user access their own profile)"""
        
        try:
            # Parse permission
            parts = required_permission.split(":")
            if len(parts) != 2:
                return False
            
            action, resource_type = parts
            
            # For profile resources, users can access their own profiles
            if resource_type == "profile":
                # This would need actual implementation to check ownership
                return True
            
            return False
            
        except Exception:
            return False
    
    async def _get_inherited_permissions(self, base_permissions: List[str]) -> Set[str]:
        """Get inherited permissions based on permission hierarchy"""
        
        inherited = set()
        
        for permission in base_permissions:
            # If user has manage permission, they also get read/write/update/delete
            if permission.startswith("manage:"):
                resource = permission.split(":", 1)[1]
                inherited.update([
                    f"create:{resource}",
                    f"read:{resource}",
                    f"update:{resource}",
                    f"delete:{resource}",
                    f"list:{resource}"
                ])
            
            # If user has write permission, they also get read
            elif permission.startswith("write:"):
                resource = permission.split(":", 1)[1]
                inherited.add(f"read:{resource}")
            
            # If user has update permission, they also get read
            elif permission.startswith("update:"):
                resource = permission.split(":", 1)[1]
                inherited.add(f"read:{resource}")
        
        return inherited
    
    def _get_default_role_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Get default system role definitions"""
        
        return {
            SystemRole.SUPER_ADMIN: {
                "name": "Super Administrator",
                "description": "Full system access",
                "permissions": ["*:*"]
            },
            SystemRole.TENANT_ADMIN: {
                "name": "Tenant Administrator",
                "description": "Full access within tenant",
                "permissions": [
                    "manage:tenant",
                    "manage:user",
                    "manage:profile",
                    "manage:search",
                    "manage:document",
                    "manage:api_key",
                    "read:audit_log"
                ]
            },
            SystemRole.USER_MANAGER: {
                "name": "User Manager",
                "description": "Can manage users within tenant",
                "permissions": [
                    "manage:user",
                    "read:profile",
                    "read:search",
                    "read:document"
                ]
            },
            SystemRole.USER: {
                "name": "User",
                "description": "Standard user access",
                "permissions": [
                    "create:profile",
                    "read:profile",
                    "update:profile",
                    "delete:profile",
                    "create:search",
                    "read:search",
                    "create:document",
                    "read:document",
                    "update:document",
                    "delete:document"
                ]
            },
            SystemRole.READONLY: {
                "name": "Read Only",
                "description": "Read-only access",
                "permissions": [
                    "read:profile",
                    "read:search",
                    "read:document"
                ]
            },
            SystemRole.API_USER: {
                "name": "API User",
                "description": "API access only",
                "permissions": [
                    "create:profile",
                    "read:profile",
                    "update:profile",
                    "create:search",
                    "read:search"
                ]
            }
        }
    
    async def create_custom_role(
        self,
        role_name: str,
        permissions: List[str],
        tenant_id: str,
        description: Optional[str] = None
    ) -> UserRole:
        """Create a custom role for a tenant"""
        
        custom_role = UserRole(
            name=role_name,
            permissions=permissions,
            description=description
        )
        
        # Save to database
        created_role = await self.role_repo.create(custom_role)
        
        # Invalidate cache
        cache_key = f"{self.ROLE_CACHE_PREFIX}{tenant_id}:{role_name}"
        await self.cache.delete(cache_key)
        
        logger.info("Custom role created", role=role_name, tenant_id=tenant_id)
        
        return created_role
    
    async def update_role_permissions(
        self,
        role_name: str,
        permissions: List[str],
        tenant_id: str
    ) -> bool:
        """Update permissions for a role"""
        
        try:
            # Update role in database
            role = await self.role_repo.get_by_name(role_name, tenant_id)
            if role:
                role.permissions = permissions
                await self.role_repo.update(role)
            
            # Invalidate caches
            role_cache_key = f"{self.ROLE_CACHE_PREFIX}{tenant_id}:{role_name}"
            await self.cache.delete(role_cache_key)
            
            # Invalidate user permissions cache for this tenant
            await self.cache.delete_pattern(f"{self.USER_PERMISSIONS_CACHE_PREFIX}{tenant_id}:*")
            
            logger.info("Role permissions updated", role=role_name, tenant_id=tenant_id)
            
            return True
            
        except Exception as e:
            logger.error("Failed to update role permissions", error=str(e))
            return False