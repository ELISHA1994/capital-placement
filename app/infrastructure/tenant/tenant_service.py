"""
Tenant Service Implementation

Provides comprehensive tenant management functionality:
- Tenant CRUD operations
- Subscription management
- Usage tracking and quota enforcement
- User assignment and role management
- Tenant lifecycle management
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
from uuid import uuid4
import re

import structlog
from app.core.config import get_settings
from app.utils.security import SecurityValidator
from app.infrastructure.persistence.models.tenant_table import (
    TenantConfiguration, SubscriptionTier, QuotaLimits, 
    UsageMetrics, FeatureFlags, BillingConfiguration
)
from app.infrastructure.persistence.models.auth_tables import User, CurrentUser
from app.database.repositories.postgres import TenantRepository, UserRepository
from app.domain.interfaces import ICacheService
from app.core.transaction_manager import transactional, get_transaction_manager
from app.database.error_handling import TransactionError, DatabaseError

logger = structlog.get_logger(__name__)


class TenantService:
    """Comprehensive tenant management service"""
    
    def __init__(
        self,
        tenant_repository: TenantRepository,
        user_repository: UserRepository,
        cache_manager: ICacheService
    ):
        self.tenant_repo = tenant_repository
        self.user_repo = user_repository
        self.cache = cache_manager
        self.settings = get_settings()
        
        # SQLModel repositories already handle transactions through session management
        # No need for separate transaction-aware wrappers
        
        # Cache keys
        self.TENANT_CACHE_PREFIX = "tenant:"
        self.TENANT_USERS_PREFIX = "tenant_users:"
        
        # Performance tracking
        self._operation_metrics = {
            "tenant_creation_count": 0,
            "tenant_creation_errors": 0,
            "average_creation_time_ms": 0.0
        }
        
    async def create_tenant(
        self, 
        name: str,
        display_name: str,
        primary_contact_email: str,
        subscription_tier: SubscriptionTier = SubscriptionTier.FREE,
        admin_user_data: Optional[Dict[str, Any]] = None
    ) -> TenantConfiguration:
        """
        Create a new tenant with optional admin user creation.
        
        When admin_user_data is provided, this method ensures atomic creation:
        either both tenant and admin user are created successfully, or neither is created.
        
        Args:
            name: Tenant slug/identifier
            display_name: Human-readable tenant name
            primary_contact_email: Primary contact email
            subscription_tier: Initial subscription tier
            admin_user_data: Optional admin user creation data (email, password, full_name)
        
        Returns:
            TenantConfiguration: Complete tenant configuration
            
        Raises:
            TransactionError: If atomic operation fails
            ValueError: If validation fails
        """
        
        if admin_user_data:
            # Use atomic creation when admin user is provided
            return await self._create_tenant_with_admin_atomic(
                name, display_name, primary_contact_email, 
                subscription_tier, admin_user_data
            )
        else:
            # Original behavior - tenant only
            return await self._create_tenant_only(
                name, display_name, primary_contact_email, subscription_tier
            )
    
    @transactional(isolation_level="READ COMMITTED")
    async def _create_tenant_with_admin_atomic(
        self,
        name: str,
        display_name: str,
        primary_contact_email: str,
        subscription_tier: SubscriptionTier,
        admin_user_data: Dict[str, Any],
        _transaction_context=None  # Injected by @transactional decorator
    ) -> TenantConfiguration:
        """
        Create tenant and admin user atomically in a single transaction.
        
        Either both succeed or both fail - no orphaned data.
        """
        start_time = datetime.now(timezone.utc)
        operation_id = str(uuid4())
        
        logger.info(
            "Starting atomic tenant creation",
            operation_id=operation_id,
            tenant_name=name,
            display_name=display_name,
            subscription_tier=subscription_tier.value,
            transaction_id=_transaction_context.transaction_id
        )
        
        try:
            # Step 1: Validate inputs
            await self._validate_tenant_creation_inputs(
                name, display_name, primary_contact_email, admin_user_data
            )
            
            # Step 2: Generate the slug that will be used and check its availability
            slug = SecurityValidator.generate_slug_from_name(display_name) if name != display_name else name
            logger.debug(f"Checking availability for slug: '{slug}' (generated from name: '{name}', display_name: '{display_name}')")
            is_available = await self.tenant_repo.check_slug_availability(slug)
            logger.debug(f"Slug '{slug}' availability check result: {is_available}")
            if not is_available:
                raise ValueError(f"Tenant name '{name}' is already taken")
            
            # Step 3: Create tenant record
            tenant_data = await self._prepare_tenant_data(
                name, display_name, primary_contact_email, subscription_tier
            )
            
            created_tenant = await self.tenant_repo.create(tenant_data)
            
            logger.info(
                "Tenant record created successfully",
                operation_id=operation_id,
                tenant_id=created_tenant["id"],
                transaction_id=_transaction_context.transaction_id
            )
            
            # Step 4: Create admin user
            admin_user = await self._create_tenant_admin_user_transactional(
                str(created_tenant["id"]),
                admin_user_data,
                _transaction_context.connection
            )
            
            logger.info(
                "Admin user created successfully",
                operation_id=operation_id,
                tenant_id=created_tenant["id"],
                admin_user_id=admin_user["id"],
                transaction_id=_transaction_context.transaction_id
            )
            
            # Step 5: Initialize tenant defaults
            await self._initialize_tenant_defaults_transactional(
                created_tenant["id"], _transaction_context.connection
            )
            
            # Step 6: Clear any cached data
            await self._invalidate_tenant_cache(created_tenant["id"])
            
            # Calculate operation time
            end_time = datetime.now(timezone.utc)
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            # Update metrics
            await self._update_operation_metrics("create_tenant_success", duration_ms)
            
            logger.info(
                "Atomic tenant creation completed successfully",
                operation_id=operation_id,
                tenant_id=created_tenant["id"],
                admin_user_id=admin_user["id"],
                duration_ms=duration_ms,
                transaction_id=_transaction_context.transaction_id
            )
            
            # Return complete tenant configuration
            return self._build_tenant_configuration(
                created_tenant, subscription_tier, primary_contact_email
            )
            
        except Exception as e:
            # Update error metrics
            await self._update_operation_metrics("create_tenant_error", 0)
            
            logger.error(
                "Atomic tenant creation failed",
                operation_id=operation_id,
                tenant_name=name,
                error=str(e),
                error_type=type(e).__name__,
                transaction_id=_transaction_context.transaction_id
            )
            
            # Re-raise as TransactionError for better handling
            if isinstance(e, (ValueError, DatabaseError)):
                raise e
            else:
                raise TransactionError(
                    f"Tenant creation failed: {str(e)}",
                    original_error=e,
                    transaction_id=_transaction_context.transaction_id
                )
    
    async def _create_tenant_only(
        self,
        name: str,
        display_name: str,
        primary_contact_email: str,
        subscription_tier: SubscriptionTier
    ) -> TenantConfiguration:
        """Create tenant without admin user (original behavior)."""
        
        # Validate tenant name format
        if not self._validate_tenant_name(name):
            raise ValueError("Invalid tenant name format")
        
        # Generate the slug that will be used and check its availability  
        slug = SecurityValidator.generate_slug_from_name(display_name) if name != display_name else name
        if not await self.tenant_repo.check_slug_availability(slug):
            raise ValueError(f"Tenant name '{name}' is already taken")
        
        # Prepare tenant data
        tenant_data = await self._prepare_tenant_data(
            name, display_name, primary_contact_email, subscription_tier
        )
        
        try:
            # Create tenant in database
            created_tenant_data = await self.tenant_repo.create(tenant_data)
            
            # Initialize tenant defaults
            await self._initialize_tenant_defaults(created_tenant_data["id"])
            
            # Clear tenant cache
            await self._invalidate_tenant_cache(created_tenant_data["id"])
            
            logger.info(
                "Tenant created successfully (no admin user)",
                tenant_id=created_tenant_data["id"],
                tenant_name=name,
                display_name=display_name,
                subscription_tier=subscription_tier.value
            )
            
            return self._build_tenant_configuration(
                created_tenant_data, subscription_tier, primary_contact_email
            )
            
        except Exception as e:
            logger.error(
                "Failed to create tenant",
                error=str(e),
                tenant_name=name,
                display_name=display_name
            )
            raise ValueError(f"Failed to create tenant: {str(e)}")
    
    async def get_tenant(self, tenant_id: str) -> Optional[TenantConfiguration]:
        """Get tenant by ID with caching"""
        
        # Try cache first
        cache_key = f"{self.TENANT_CACHE_PREFIX}{tenant_id}"
        cached_tenant = await self.cache.get(cache_key)
        
        if cached_tenant:
            return TenantConfiguration(**cached_tenant)
        
        # Get from repository
        tenant_data = await self.tenant_repo.get(tenant_id)
        
        if tenant_data:
            # Deserialize JSON fields before creating TenantConfiguration object
            processed_data = self._deserialize_tenant_json_fields(tenant_data)
            tenant = TenantConfiguration(**processed_data)
            # Cache for 1 hour
            await self.cache.set(cache_key, tenant.dict(), ttl=3600)
            return tenant
        
        return None

    async def get_tenant_configuration(self, tenant_id: str) -> Dict[str, Any]:
        """
        Get tenant configuration as dictionary.

        This is a convenience method that wraps get_tenant() and returns
        the configuration as a dictionary for compatibility with services
        that expect dict-based configuration.

        Args:
            tenant_id: Tenant ID

        Returns:
            Tenant configuration dictionary or empty dict if not found
        """
        tenant = await self.get_tenant(tenant_id)
        if tenant:
            return tenant.dict()
        return {}

    async def check_quota_limit(
        self,
        tenant_id: str,
        resource_type: str,
        current_usage: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Check if tenant has reached quota limits for a specific resource.

        Args:
            tenant_id: Tenant ID
            resource_type: Type of resource to check (e.g., 'documents_per_day', 'api_requests', 'storage')
            current_usage: Current usage count for the resource
            **kwargs: Additional parameters for quota checking

        Returns:
            Dictionary with quota check results:
            - allowed: bool - Whether the action is allowed
            - remaining: int - Remaining quota
            - limit: int - Total quota limit
            - exceeded: bool - Whether quota is exceeded
        """
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            # If tenant not found, allow but log warning
            logger.warning(f"Tenant not found for quota check: {tenant_id}")
            return {
                "allowed": True,
                "remaining": 999999,
                "limit": 999999,
                "exceeded": False
            }

        # Get quota limits from tenant configuration
        quota_limits = tenant.quota_limits

        # Map resource types to quota limit fields
        quota_field_map = {
            "documents_per_day": "max_documents_per_day",
            "documents_per_month": "max_documents_per_month",
            "api_requests_per_minute": "max_api_requests_per_minute",
            "api_requests_per_hour": "max_api_requests_per_hour",
            "api_requests_per_day": "max_api_requests_per_day",
            "storage": "max_storage_gb",
            "searches_per_day": "max_searches_per_day",
            "searches_per_month": "max_searches_per_month",
        }

        # Get the limit for this resource type
        quota_field = quota_field_map.get(resource_type)
        if not quota_field:
            logger.warning(f"Unknown resource type for quota check: {resource_type}")
            return {"allowed": True, "remaining": 999999, "limit": 999999, "exceeded": False}

        # Get the actual limit value
        limit = getattr(quota_limits, quota_field, None)
        if limit is None:
            # No limit set, allow
            return {"allowed": True, "remaining": 999999, "limit": 999999, "exceeded": False}

        # Calculate remaining quota
        remaining = max(0, limit - current_usage)
        exceeded = current_usage >= limit
        allowed = not exceeded

        return {
            "allowed": allowed,
            "remaining": remaining,
            "limit": limit,
            "exceeded": exceeded,
            "current_usage": current_usage
        }

    async def update_usage_metrics(
        self,
        tenant_id: str,
        metrics_update: Dict[str, Any]
    ) -> None:
        """
        Update tenant usage metrics.

        Args:
            tenant_id: Tenant ID
            metrics_update: Dictionary of metrics to update
        """
        tenant_data = await self.tenant_repo.get(tenant_id)
        if not tenant_data:
            logger.warning(f"Tenant not found for usage metrics update: {tenant_id}")
            return

        # Update metrics in tenant data
        # This is a simplified implementation - in production you might want to
        # track usage metrics in a separate table or time-series database
        logger.info(
            "Tenant usage metrics updated",
            tenant_id=tenant_id,
            metrics=metrics_update
        )

    async def update_tenant(
        self, 
        tenant_id: str, 
        updates: Dict[str, Any]
    ) -> TenantConfiguration:
        """Update tenant configuration"""
        
        tenant_data = await self.tenant_repo.get(tenant_id)
        if not tenant_data:
            raise ValueError("Tenant not found")
        
        # Ensure we're working with a dictionary
        if hasattr(tenant_data, 'model_dump'):
            # Convert Pydantic model to dict
            tenant_dict = tenant_data.model_dump()
        else:
            # Already a dict or dict-like object
            tenant_dict = dict(tenant_data) if not isinstance(tenant_data, dict) else tenant_data
        
        # Apply updates - only update fields that exist in the tenant
        for field, value in updates.items():
            if field in tenant_dict:
                tenant_dict[field] = value
        
        # Update the timestamp
        tenant_dict["updated_at"] = datetime.utcnow()
        
        # Save updates
        updated_tenant_data = await self.tenant_repo.update(tenant_id, tenant_dict)
        
        # Clear cache
        await self._invalidate_tenant_cache(tenant_id)
        
        logger.info(
            "Tenant updated",
            tenant_id=tenant_id,
            updates=updates
        )
        
        # Convert to TenantConfiguration for return
        processed_data = self._deserialize_tenant_json_fields(updated_tenant_data)
        return TenantConfiguration(**processed_data)
    
    async def suspend_tenant(self, tenant_id: str, reason: str) -> bool:
        """Suspend tenant and prevent access"""
        
        tenant_data = await self.tenant_repo.get(tenant_id)
        if not tenant_data:
            return False
        
        # Ensure we're working with a dictionary
        if hasattr(tenant_data, 'model_dump'):
            tenant_dict = tenant_data.model_dump()
        else:
            tenant_dict = dict(tenant_data) if not isinstance(tenant_data, dict) else tenant_data
        
        tenant_dict["is_suspended"] = True
        tenant_dict["suspension_reason"] = reason
        tenant_dict["suspended_at"] = datetime.utcnow()
        
        await self.tenant_repo.update(tenant_id, tenant_dict)
        await self._invalidate_tenant_cache(tenant_id)
        
        logger.warning(
            "Tenant suspended",
            tenant_id=tenant_id,
            reason=reason
        )
        
        return True
    
    async def activate_tenant(self, tenant_id: str) -> bool:
        """Activate suspended tenant"""
        
        tenant_data = await self.tenant_repo.get(tenant_id)
        if not tenant_data:
            return False
        
        # Ensure we're working with a dictionary
        if hasattr(tenant_data, 'model_dump'):
            tenant_dict = tenant_data.model_dump()
        else:
            tenant_dict = dict(tenant_data) if not isinstance(tenant_data, dict) else tenant_data
        
        tenant_dict["is_suspended"] = False
        tenant_dict["suspension_reason"] = None
        tenant_dict["suspended_at"] = None
        
        await self.tenant_repo.update(tenant_id, tenant_dict)
        await self._invalidate_tenant_cache(tenant_id)
        
        logger.info(
            "Tenant activated",
            tenant_id=tenant_id
        )
        
        return True
    
    async def delete_tenant(self, tenant_id: str) -> bool:
        """Soft delete tenant (deactivate)"""
        
        tenant_data = await self.tenant_repo.get(tenant_id)
        if not tenant_data:
            return False
        
        # Ensure we're working with a dictionary
        if hasattr(tenant_data, 'model_dump'):
            tenant_dict = tenant_data.model_dump()
        else:
            tenant_dict = dict(tenant_data) if not isinstance(tenant_data, dict) else tenant_data
        
        tenant_dict["is_active"] = False
        tenant_dict["deleted_at"] = datetime.utcnow()
        
        await self.tenant_repo.update(tenant_id, tenant_dict)
        await self._invalidate_tenant_cache(tenant_id)
        
        logger.info(
            "Tenant deleted",
            tenant_id=tenant_id
        )
        
        return True
    
    async def check_quota(self, tenant_id: str, resource: str) -> Dict[str, Any]:
        """Check quota usage for a specific resource"""
        
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise ValueError("Tenant not found")
        
        quota_limits = tenant.quota_limits
        usage_metrics = tenant.usage_metrics
        
        if not hasattr(quota_limits, resource) or not hasattr(usage_metrics, resource):
            raise ValueError(f"Unknown resource: {resource}")
        
        limit = getattr(quota_limits, resource)
        current_usage = getattr(usage_metrics, resource)
        
        return {
            "resource": resource,
            "limit": limit,
            "current_usage": current_usage,
            "remaining": max(0, limit - current_usage),
            "percentage_used": (current_usage / limit * 100) if limit > 0 else 0,
            "exceeded": current_usage >= limit
        }
    
    async def update_usage(self, tenant_id: str, metrics: Dict[str, int]) -> bool:
        """Update tenant usage metrics"""
        
        tenant_data = await self.tenant_repo.get(tenant_id)
        if not tenant_data:
            return False
        
        # Ensure we're working with a dictionary
        if hasattr(tenant_data, 'model_dump'):
            tenant_dict = tenant_data.model_dump()
        else:
            tenant_dict = dict(tenant_data) if not isinstance(tenant_data, dict) else tenant_data
        
        # Update usage metrics
        usage_metrics = tenant_dict.get("usage_metrics", {})
        for metric, value in metrics.items():
            if metric in usage_metrics:
                current_value = usage_metrics.get(metric, 0)
                usage_metrics[metric] = current_value + value
        
        tenant_dict["usage_metrics"] = usage_metrics
        tenant_dict["updated_at"] = datetime.utcnow()
        
        await self.tenant_repo.update(tenant_id, tenant_dict)
        await self._invalidate_tenant_cache(tenant_id)
        
        return True
    
    async def get_tenant_users(self, tenant_id: str) -> List[CurrentUser]:
        """Get all users belonging to a tenant"""
        
        # Try cache first
        cache_key = f"{self.TENANT_USERS_PREFIX}{tenant_id}"
        cached_users = await self.cache.get(cache_key)
        
        if cached_users:
            return [CurrentUser(**user) for user in cached_users]
        
        # Get users from repository (returns List[Dict[str, Any]])
        users = await self.user_repo.get_by_tenant(tenant_id)
        
        # Convert dictionary data to CurrentUser objects
        current_users = []
        for user_dict in users:
            # Handle both 'full_name' field and 'first_name'/'last_name' combination
            if 'full_name' in user_dict and user_dict['full_name']:
                full_name = user_dict['full_name']
            else:
                first_name = user_dict.get('first_name', '')
                last_name = user_dict.get('last_name', '')
                full_name = f"{first_name} {last_name}".strip()
            
            current_user = CurrentUser(
                user_id=str(user_dict.get('id', user_dict.get('user_id', ''))),
                email=user_dict.get('email', ''),
                full_name=full_name,
                tenant_id=str(user_dict.get('tenant_id', '')),
                roles=user_dict.get('roles', []),
                permissions=user_dict.get('permissions', []),
                is_active=user_dict.get('is_active', True),
                is_superuser=user_dict.get('is_superuser', False)
            )
            current_users.append(current_user)
        
        # Cache for 30 minutes
        await self.cache.set(
            cache_key, 
            [user.dict() for user in current_users], 
            ttl=1800
        )
        
        return current_users
    
    async def create_tenant_user(
        self,
        tenant_id: str,
        email: str,
        password: str,
        full_name: str,
        roles: List[str] = None
    ) -> CurrentUser:
        """
        Create a new user and add them to a tenant.
        
        This is the proper service layer method for tenant user creation that:
        - Validates tenant exists and is active
        - Creates the user with proper password hashing
        - Associates the user with the tenant
        - Returns the created user as CurrentUser object
        
        Args:
            tenant_id: The tenant to add the user to
            email: User's email address
            password: User's password (will be hashed)
            full_name: User's full name
            roles: List of roles to assign (defaults to ["user"])
            
        Returns:
            CurrentUser object representing the created user
            
        Raises:
            ValueError: If tenant not found or user creation fails
        """
        # Verify tenant exists and is active
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        # Check if tenant is active
        is_active = tenant.is_active if hasattr(tenant, 'is_active') else tenant.get('is_active', True)
        if not is_active:
            raise ValueError(f"Tenant {tenant_id} is not active")
        
        # Default roles if not provided
        if roles is None:
            roles = ["user"]
        
        # Hash the password
        from app.utils.security import password_manager
        hashed_password = password_manager.hash_password(password)

        # Prepare user data
        now = datetime.utcnow()
        user_data = {
            "id": str(uuid4()),
            "tenant_id": tenant_id,
            "email": email,
            "hashed_password": hashed_password,
            "full_name": full_name,
            "first_name": full_name.split()[0] if full_name else "",
            "last_name": " ".join(full_name.split()[1:]) if len(full_name.split()) > 1 else "",
            "roles": roles,
            "permissions": ["read", "write"] if "admin" in roles else ["read"],
            "is_active": True,
            "is_verified": False,  # Users should verify email
            "is_superuser": "admin" in roles,
            "created_at": now,
            "updated_at": now
        }
        
        # Create user in database
        created_user = await self.user_repo.create(user_data)
        
        # Clear cache
        await self._invalidate_tenant_users_cache(tenant_id)
        
        logger.info(
            "User created and added to tenant",
            tenant_id=tenant_id,
            user_id=created_user["id"],
            email=email,
            roles=roles
        )
        
        # Convert to CurrentUser for return
        return CurrentUser(
            user_id=str(created_user["id"]),
            email=created_user["email"],
            full_name=created_user["full_name"],
            tenant_id=str(created_user["tenant_id"]),
            roles=created_user["roles"],
            permissions=created_user["permissions"],
            is_active=created_user["is_active"],
            is_superuser=created_user.get("is_superuser", False)
        )
    
    async def add_tenant_user(
        self, 
        tenant_id: str, 
        user_id: str, 
        role: str = "user"
    ) -> bool:
        """Add existing user to tenant with specific role"""
        
        # Verify tenant exists
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise ValueError("Tenant not found")
        
        # Get user
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        
        # Update user tenant and role
        user.tenant_id = tenant_id
        if role not in user.roles:
            user.roles.append(role)
        
        await self.user_repo.update(user)
        
        # Clear cache
        await self._invalidate_tenant_users_cache(tenant_id)
        
        logger.info(
            "User added to tenant",
            tenant_id=tenant_id,
            user_id=user_id,
            role=role
        )
        
        return True
    
    async def update_user_role(
        self,
        tenant_id: str,
        user_id: str,
        new_role: str
    ) -> CurrentUser:
        """
        Update a user's role within a tenant.
        
        Args:
            tenant_id: The tenant ID
            user_id: The user ID to update
            new_role: The new role to assign to the user
            
        Returns:
            CurrentUser object representing the updated user
            
        Raises:
            ValueError: If tenant not found, user not found, or user not in tenant
        """
        # Verify tenant exists
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        # Get user from repository
        user_data = await self.user_repo.get_by_id(user_id)
        if not user_data:
            raise ValueError(f"User {user_id} not found")
        
        # Handle both dict and object formats for user data
        if hasattr(user_data, 'model_dump'):
            user_dict = user_data.model_dump()
        elif hasattr(user_data, '__dict__'):
            user_dict = user_data.__dict__.copy()
        else:
            user_dict = dict(user_data) if not isinstance(user_data, dict) else user_data.copy()
        
        # Verify user belongs to the tenant
        if str(user_dict.get('tenant_id')) != str(tenant_id):
            raise ValueError(f"User {user_id} does not belong to tenant {tenant_id}")
        
        # Validate the new role
        valid_roles = ["user", "admin", "super_admin"]
        if new_role not in valid_roles:
            raise ValueError(f"Invalid role '{new_role}'. Valid roles are: {', '.join(valid_roles)}")
        
        # Update user roles - replace existing roles with the new role
        # Keep "user" as base role and add the new role if it's not "user"
        if new_role == "user":
            updated_roles = ["user"]
        elif new_role == "admin":
            updated_roles = ["user", "admin"]
        elif new_role == "super_admin":
            updated_roles = ["user", "admin", "super_admin"]
        else:
            updated_roles = ["user", new_role]
        
        # Update permissions based on role
        if "admin" in updated_roles:
            permissions = ["read", "write", "admin"]
        elif "super_admin" in updated_roles:
            permissions = ["read", "write", "admin", "super_admin"]
        else:
            permissions = ["read"]
        
        # Prepare updated user data
        user_dict["roles"] = updated_roles
        user_dict["permissions"] = permissions
        user_dict["is_superuser"] = "admin" in updated_roles or "super_admin" in updated_roles
        user_dict["updated_at"] = datetime.utcnow()
        
        # Update user in database
        updated_user_data = await self.user_repo.update(user_id, user_dict)
        
        # Clear cache
        await self._invalidate_tenant_users_cache(tenant_id)
        
        logger.info(
            "User role updated in tenant",
            tenant_id=tenant_id,
            user_id=user_id,
            old_roles=user_data.get("roles", []) if isinstance(user_data, dict) else getattr(user_data, "roles", []),
            new_roles=updated_roles,
            new_role_primary=new_role
        )
        
        # Convert to CurrentUser for return
        return CurrentUser(
            user_id=str(updated_user_data.get("id", user_id)),
            email=updated_user_data.get("email", ""),
            full_name=updated_user_data.get("full_name", ""),
            tenant_id=str(updated_user_data.get("tenant_id", tenant_id)),
            roles=updated_user_data.get("roles", updated_roles),
            permissions=updated_user_data.get("permissions", permissions),
            is_active=updated_user_data.get("is_active", True),
            is_superuser=updated_user_data.get("is_superuser", False)
        )
    
    async def remove_tenant_user(self, tenant_id: str, user_id: str) -> bool:
        """Remove user from tenant"""
        
        # Get user from repository (returns Dict[str, Any])
        user_data = await self.user_repo.get_by_id(user_id)
        if not user_data:
            return False
        
        # Handle both dict and object formats for user data
        if hasattr(user_data, 'model_dump'):
            user_dict = user_data.model_dump()
        elif hasattr(user_data, '__dict__'):
            user_dict = user_data.__dict__.copy()
        else:
            user_dict = dict(user_data) if not isinstance(user_data, dict) else user_data.copy()
        
        # Verify user belongs to the tenant
        if str(user_dict.get('tenant_id')) != str(tenant_id):
            return False
        
        # Deactivate user instead of deleting - update dictionary fields
        user_dict['is_active'] = False
        user_dict['updated_at'] = datetime.utcnow()
        
        # Update user in database with proper parameters (user_id, user_dict)
        await self.user_repo.update(user_id, user_dict)
        
        # Clear cache
        await self._invalidate_tenant_users_cache(tenant_id)
        
        logger.info(
            "User removed from tenant",
            tenant_id=tenant_id,
            user_id=user_id,
            user_email=user_dict.get('email', 'unknown')
        )
        
        return True
    
    async def upgrade_subscription(
        self, 
        tenant_id: str, 
        tier: SubscriptionTier
    ) -> bool:
        """Upgrade tenant subscription tier"""
        
        tenant_data = await self.tenant_repo.get(tenant_id)
        if not tenant_data:
            return False
        
        # Ensure we're working with a dictionary
        if hasattr(tenant_data, 'model_dump'):
            tenant_dict = tenant_data.model_dump()
        else:
            tenant_dict = dict(tenant_data) if not isinstance(tenant_data, dict) else tenant_data
        
        old_tier = tenant_dict.get("subscription_tier")
        tenant_dict["subscription_tier"] = tier
        tenant_dict["quota_limits"] = self._get_default_quota_limits(tier).model_dump()
        tenant_dict["feature_flags"] = self._get_default_feature_flags(tier).model_dump()
        tenant_dict["updated_at"] = datetime.utcnow()
        
        await self.tenant_repo.update(tenant_id, tenant_dict)
        await self._invalidate_tenant_cache(tenant_id)
        
        logger.info(
            "Tenant subscription upgraded",
            tenant_id=tenant_id,
            old_tier=old_tier.value if hasattr(old_tier, 'value') else str(old_tier),
            new_tier=tier.value
        )
        
        return True
    
    async def list_tenants(
        self, 
        skip: int = 0, 
        limit: int = 100, 
        include_inactive: bool = False
    ) -> List[TenantConfiguration]:
        """
        List all tenants with pagination.
        
        Args:
            skip: Number of tenants to skip for pagination
            limit: Maximum number of tenants to return
            include_inactive: Whether to include inactive/deleted tenants
            
        Returns:
            List of TenantConfiguration objects
        """
        
        try:
            # Build filter criteria
            criteria = {}
            if not include_inactive:
                criteria["is_active"] = True
            
            # Get tenants from repository with pagination
            if criteria:
                # Use find_by_criteria with pagination logic
                all_tenants = await self.tenant_repo.find_by_criteria(criteria)
            else:
                # Get all tenants
                all_tenants = await self.tenant_repo.list_all()
            
            # Apply manual pagination (since the repository doesn't support it natively)
            total_tenants = len(all_tenants)
            paginated_tenants = all_tenants[skip:skip + limit]
            
            # Convert to TenantConfiguration objects
            tenant_configs = []
            for tenant_data in paginated_tenants:
                # Deserialize JSON fields and convert to TenantConfiguration
                processed_data = self._deserialize_tenant_json_fields(tenant_data)
                tenant_config = TenantConfiguration(**processed_data)
                tenant_configs.append(tenant_config)
            
            logger.info(
                "Listed tenants successfully",
                total_tenants=total_tenants,
                returned_count=len(tenant_configs),
                skip=skip,
                limit=limit,
                include_inactive=include_inactive
            )
            
            return tenant_configs
            
        except Exception as e:
            logger.error(
                "Failed to list tenants",
                error=str(e),
                error_type=type(e).__name__,
                skip=skip,
                limit=limit,
                include_inactive=include_inactive
            )
            raise ValueError(f"Failed to list tenants: {str(e)}")
    
    # Private helper methods
    
    def _validate_tenant_name(self, name: str) -> bool:
        """Validate tenant name format"""
        if not name or len(name) < 3 or len(name) > 50:
            return False
        
        # Cannot start or end with hyphen
        if name.startswith('-') or name.endswith('-'):
            return False
        
        # Cannot have double hyphens
        if '--' in name:
            return False
        
        # Cannot be numbers only
        if name.isdigit():
            return False
        
        # Only allow lowercase letters, numbers, and hyphens (no uppercase, underscores, spaces, special chars)
        if not re.match(r'^[a-z0-9-]+$', name):
            return False
        
        # Check for reserved words
        reserved_words = {
            'api', 'www', 'admin', 'app', 'mail', 'ftp', 'blog', 'dev', 'test', 
            'staging', 'support', 'help', 'docs', 'status', 'login', 'signup',
            'dashboard', 'settings', 'profile', 'account', 'billing', 'security'
        }
        
        return name not in reserved_words
    
    def _get_default_quota_limits(self, tier: SubscriptionTier) -> QuotaLimits:
        """Get default quota limits for subscription tier"""
        
        quota_mapping = {
            SubscriptionTier.FREE: QuotaLimits(
                max_profiles=100,
                max_searches_per_month=50,
                max_storage_gb=1,  # 1 GB
                max_api_requests_per_minute=10,
                max_users=2
            ),
            SubscriptionTier.BASIC: QuotaLimits(
                max_profiles=1000,
                max_searches_per_month=500,
                max_storage_gb=10,  # 10 GB
                max_api_requests_per_minute=60,
                max_users=10
            ),
            SubscriptionTier.PROFESSIONAL: QuotaLimits(
                max_profiles=10000,
                max_searches_per_month=5000,
                max_storage_gb=100,  # 100 GB
                max_api_requests_per_minute=300,
                max_users=50
            ),
            SubscriptionTier.ENTERPRISE: QuotaLimits(
                max_profiles=None,  # Unlimited
                max_searches_per_month=None,  # Unlimited
                max_storage_gb=None,  # Unlimited
                max_api_requests_per_minute=1000,
                max_users=None  # Unlimited
            )
        }
        
        return quota_mapping.get(tier, quota_mapping[SubscriptionTier.FREE])
    
    def _get_default_feature_flags(self, tier: SubscriptionTier) -> FeatureFlags:
        """Get default feature flags for subscription tier"""
        
        return FeatureFlags(
            # Core features - enabled for all tiers by default
            enable_advanced_search=True,
            enable_bulk_operations=True,
            enable_export=True,
            enable_webhooks=tier in [SubscriptionTier.PROFESSIONAL, SubscriptionTier.ENTERPRISE],
            
            # AI features - basic enabled, advanced for higher tiers
            enable_ai_recommendations=tier in [SubscriptionTier.PROFESSIONAL, SubscriptionTier.ENTERPRISE],
            enable_skill_extraction=True,
            enable_sentiment_analysis=tier == SubscriptionTier.ENTERPRISE,
            enable_candidate_scoring=True,
            
            # Analytics features - progressive enablement
            enable_analytics_dashboard=True,
            enable_custom_reports=tier in [SubscriptionTier.BASIC, SubscriptionTier.PROFESSIONAL, SubscriptionTier.ENTERPRISE],
            enable_data_insights=tier in [SubscriptionTier.PROFESSIONAL, SubscriptionTier.ENTERPRISE],
            
            # Integration features - higher tiers only
            enable_ats_integration=tier == SubscriptionTier.ENTERPRISE,
            enable_crm_integration=tier == SubscriptionTier.ENTERPRISE,
            enable_api_access=True,  # API access for all tiers
            enable_sso=tier == SubscriptionTier.ENTERPRISE
        )
    
    async def _create_tenant_admin_user(
        self, 
        tenant_id: str, 
        admin_data: Dict[str, Any]
    ) -> User:
        """Create admin user for new tenant"""
        
        from app.utils.security import password_manager
        
        hashed_password = password_manager.hash_password(admin_data["password"])
        
        admin_user = User(
            tenant_id=tenant_id,
            email=admin_data["email"],
            hashed_password=hashed_password,
            full_name=admin_data["full_name"],
            roles=["admin", "user"],
            is_active=True,
            is_verified=False,
            is_superuser=True  # Admin users should be superusers
        )
        
        # UserRepository now handles schema mapping automatically
        user_data = admin_user.model_dump()
        return await self.user_repo.create(user_data)
    
    async def _initialize_tenant_defaults(self, tenant_id: str):
        """Initialize default tenant configuration"""
        # Add any default initialization logic here
        pass
    
    async def _invalidate_tenant_cache(self, tenant_id: str):
        """Invalidate all tenant-related cache entries"""
        cache_keys = [
            f"{self.TENANT_CACHE_PREFIX}{tenant_id}",
            f"{self.TENANT_USERS_PREFIX}{tenant_id}"
        ]
        
        for cache_key in cache_keys:
            await self.cache.delete(cache_key)
    
    async def _invalidate_tenant_users_cache(self, tenant_id: str):
        """Invalidate tenant users cache"""
        cache_key = f"{self.TENANT_USERS_PREFIX}{tenant_id}"
        await self.cache.delete(cache_key)
    
    # Helper methods for atomic operations
    
    async def _validate_tenant_creation_inputs(
        self,
        name: str,
        display_name: str,
        primary_contact_email: str,
        admin_user_data: Dict[str, Any]
    ) -> None:
        """Validate all inputs for tenant creation."""
        
        # Validate tenant name format
        if not self._validate_tenant_name(name):
            raise ValueError("Invalid tenant name format")
        
        # Validate display name
        if not display_name or len(display_name.strip()) < 2:
            raise ValueError("Display name must be at least 2 characters")
        
        # Validate email format
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, primary_contact_email):
            raise ValueError("Invalid primary contact email format")
        
        # Validate admin user data
        if not admin_user_data:
            raise ValueError("Admin user data is required")
        
        required_fields = ["email", "password", "full_name"]
        for field in required_fields:
            if field not in admin_user_data or not admin_user_data[field]:
                raise ValueError(f"Admin user {field} is required")
        
        # Validate admin email format
        if not re.match(email_pattern, admin_user_data["email"]):
            raise ValueError("Invalid admin user email format")
        
        # Check password strength
        password = admin_user_data["password"]
        if len(password) < 8:
            raise ValueError("Admin password must be at least 8 characters")
        
        # Additional password complexity checks
        if password.isdigit():
            raise ValueError("Admin password cannot be only numbers")
        
        # Check for common weak passwords
        common_weak_passwords = {'password', 'password123', 'admin', 'admin123', '12345678'}
        if password.lower() in common_weak_passwords:
            raise ValueError("Admin password is too weak - use a stronger password")
        
        logger.debug(
            "Tenant creation inputs validated successfully",
            tenant_name=name,
            display_name=display_name,
            admin_email=admin_user_data["email"]
        )
    
    async def _prepare_tenant_data(
        self,
        name: str,
        display_name: str,
        primary_contact_email: str,
        subscription_tier: SubscriptionTier
    ) -> Dict[str, Any]:
        """Prepare tenant data for creation."""

        now = datetime.utcnow()

        # Generate slug from display name if name is different, otherwise use name as slug
        slug = SecurityValidator.generate_slug_from_name(display_name) if name != display_name else name

        return {
            "id": str(uuid4()),
            "name": name,
            "slug": slug,
            "display_name": display_name,
            "primary_contact_email": primary_contact_email,
            "subscription_tier": subscription_tier,
            "quota_limits": self._get_default_quota_limits(subscription_tier),
            "usage_metrics": UsageMetrics(),
            "feature_flags": self._get_default_feature_flags(subscription_tier),
            "billing_configuration": BillingConfiguration(),
            "is_active": True,
            "is_suspended": False,
            "created_at": now,
            "updated_at": now
        }
    
    async def _create_tenant_admin_user_transactional(
        self,
        tenant_id: str,
        admin_data: Dict[str, Any],
        connection
    ) -> Dict[str, Any]:
        """Create admin user within an existing transaction."""

        from app.utils.security import password_manager

        # Hash password
        hashed_password = password_manager.hash_password(admin_data["password"])

        # Prepare user data
        now = datetime.utcnow()
        user_data = {
            "id": str(uuid4()),
            "tenant_id": tenant_id,
            "email": admin_data["email"],
            "hashed_password": hashed_password,
            "full_name": admin_data["full_name"],
            "first_name": admin_data.get("first_name", admin_data["full_name"].split()[0]),
            "last_name": admin_data.get("last_name", " ".join(admin_data["full_name"].split()[1:])),
            "roles": ["admin", "user"],
            "permissions": ["read", "write", "admin"],
            "is_active": True,
            "is_verified": True,  # Auto-verify admin users
            "is_superuser": True,
            "created_at": now,
            "updated_at": now
        }
        
        # Create user using SQLModel repository
        created_user = await self.user_repo.create(user_data)
        
        logger.info(
            "Admin user created in transaction",
            tenant_id=tenant_id,
            user_id=created_user["id"],
            email=admin_data["email"]
        )
        
        return created_user
    
    async def _initialize_tenant_defaults_transactional(
        self,
        tenant_id: str,
        connection
    ) -> None:
        """Initialize tenant defaults within an existing transaction."""
        
        # For now, this is a placeholder for future tenant initialization logic
        # such as creating default folders, settings, etc.
        
        logger.debug(
            "Tenant defaults initialized in transaction",
            tenant_id=tenant_id
        )
    
    def _deserialize_tenant_json_fields(self, tenant_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize JSON fields from database data for Pydantic model creation."""
        import json
        from app.infrastructure.persistence.models.tenant_table import (
            SearchConfiguration, ProcessingConfiguration, ProcessingPriority,
            UsageMetrics, QuotaLimits, FeatureFlags, BillingConfiguration
        )
        
        # Create a copy to avoid modifying the original
        processed_data = tenant_data.copy()
        
        # Complete list of JSON fields that need deserialization
        json_fields = [
            "quota_limits", "usage_metrics", "feature_flags", "billing_configuration",
            "search_configuration", "processing_configuration", "current_usage"
        ]
        
        # Default instances for each field type
        field_defaults = {
            "quota_limits": lambda: QuotaLimits(),
            "usage_metrics": lambda: UsageMetrics(),
            "feature_flags": lambda: FeatureFlags(),
            "billing_configuration": lambda: BillingConfiguration(),
            "search_configuration": lambda: SearchConfiguration(),
            "processing_configuration": lambda: ProcessingConfiguration(),
            "current_usage": lambda: UsageMetrics()
        }
        
        for field in json_fields:
            if field in processed_data:
                value = processed_data[field]
                
                # Handle None values
                if value is None:
                    processed_data[field] = field_defaults[field]().model_dump()
                    continue
                
                # Handle string values (JSON)
                if isinstance(value, str):
                    # Handle empty JSON objects or invalid JSON
                    if value.strip() in ['{}', '']:
                        processed_data[field] = field_defaults[field]().model_dump()
                        continue
                    
                    try:
                        parsed_value = json.loads(value)
                        # If parsed result is empty dict, use defaults
                        if not parsed_value or parsed_value == {}:
                            processed_data[field] = field_defaults[field]().model_dump()
                        else:
                            processed_data[field] = parsed_value
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(
                            f"Failed to deserialize {field}, using default",
                            field=field,
                            value=str(value)[:100],  # Truncate long values
                            error=str(e)
                        )
                        processed_data[field] = field_defaults[field]().model_dump()
                
                # If already a dict, check if it's empty
                elif isinstance(value, dict) and not value:
                    processed_data[field] = field_defaults[field]().model_dump()
            
            else:
                # Field is missing entirely, add defaults
                processed_data[field] = field_defaults[field]().model_dump()
        
        # Handle primary_contact_email specifically
        if not processed_data.get("primary_contact_email"):
            processed_data["primary_contact_email"] = "admin@tenant.local"  # Fallback
        
        # Ensure required string fields are not None
        string_fields = ["name", "display_name"]
        for field in string_fields:
            if not processed_data.get(field):
                processed_data[field] = f"Unknown {field.replace('_', ' ').title()}"
        
        return processed_data
    
    def _build_tenant_configuration(
        self,
        tenant_data: Dict[str, Any],
        subscription_tier: SubscriptionTier,
        primary_contact_email: str
    ) -> TenantConfiguration:
        """Build complete tenant configuration object."""
        
        # Deserialize JSON fields before creating TenantConfiguration
        processed_data = self._deserialize_tenant_json_fields(tenant_data)
        
        return TenantConfiguration(
            id=processed_data["id"],
            name=processed_data["name"],
            display_name=processed_data["display_name"],
            primary_contact_email=primary_contact_email,
            subscription_tier=subscription_tier,
            quota_limits=processed_data.get("quota_limits") or self._get_default_quota_limits(subscription_tier),
            usage_metrics=processed_data.get("usage_metrics") or UsageMetrics(),
            feature_flags=processed_data.get("feature_flags") or self._get_default_feature_flags(subscription_tier),
            billing_configuration=processed_data.get("billing_configuration") or BillingConfiguration(),
            is_active=processed_data.get("is_active", True),
            is_suspended=processed_data.get("is_suspended", False),
            suspension_reason=processed_data.get("suspension_reason"),
            created_at=processed_data["created_at"],
            updated_at=processed_data["updated_at"],
            suspended_at=processed_data.get("suspended_at"),
            deleted_at=processed_data.get("deleted_at")
        )
    
    async def _update_operation_metrics(
        self,
        operation_type: str,
        duration_ms: float
    ) -> None:
        """Update operation performance metrics."""
        
        if operation_type == "create_tenant_success":
            self._operation_metrics["tenant_creation_count"] += 1
            
            # Update rolling average
            current_avg = self._operation_metrics["average_creation_time_ms"]
            count = self._operation_metrics["tenant_creation_count"]
            
            new_avg = ((current_avg * (count - 1)) + duration_ms) / count
            self._operation_metrics["average_creation_time_ms"] = new_avg
            
        elif operation_type == "create_tenant_error":
            self._operation_metrics["tenant_creation_errors"] += 1
        
        logger.debug(
            "Operation metrics updated",
            operation_type=operation_type,
            duration_ms=duration_ms,
            metrics=self._operation_metrics
        )
    
    async def get_operation_metrics(self) -> Dict[str, Any]:
        """Get current operation performance metrics."""
        
        total_operations = (
            self._operation_metrics["tenant_creation_count"] + 
            self._operation_metrics["tenant_creation_errors"]
        )
        
        success_rate = 0.0
        if total_operations > 0:
            success_rate = (
                self._operation_metrics["tenant_creation_count"] / total_operations
            ) * 100
        
        return {
            "tenant_creation": {
                "total_attempts": total_operations,
                "successful_creations": self._operation_metrics["tenant_creation_count"],
                "failed_creations": self._operation_metrics["tenant_creation_errors"],
                "success_rate_percentage": round(success_rate, 2),
                "average_creation_time_ms": round(
                    self._operation_metrics["average_creation_time_ms"], 2
                )
            },
            "transaction_manager_stats": await get_transaction_manager().get_transaction_stats()
        }
