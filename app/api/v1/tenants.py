"""
Tenant Management API Endpoints

Provides tenant management functionality according to the implementation plan:
- Super Admin: Create, list, update, delete tenants
- Tenant Admin: Manage users within their tenant
- Usage and billing management
- Subscription management
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse

from app.models.tenant_models import (
    TenantConfiguration, SubscriptionTier, QuotaLimits, 
    UsageMetrics, FeatureFlags
)
from app.models.auth import CurrentUser, User
from app.core.dependencies import (
    get_current_user,
    CurrentUserDep, TenantServiceDep
)
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/tenants", tags=["tenants"])



# Request/Response Models
class TenantCreateRequest(BaseModel):
    """Request model for creating a new tenant"""
    name: str = Field(..., min_length=2, max_length=50, description="Tenant slug/identifier")
    display_name: str = Field(..., min_length=2, max_length=255, description="Human-readable tenant name")
    primary_contact_email: str = Field(..., description="Primary contact email")
    subscription_tier: SubscriptionTier = Field(SubscriptionTier.FREE, description="Initial subscription tier")
    
    # Optional admin user creation
    admin_user: Optional[Dict[str, str]] = Field(None, description="Admin user data {email, password, full_name}")


class TenantUpdateRequest(BaseModel):
    """Request model for updating tenant"""
    display_name: Optional[str] = Field(None, min_length=2, max_length=255)
    primary_contact_email: Optional[str] = None
    is_active: Optional[bool] = None


class TenantResponse(BaseModel):
    """Response model for tenant data"""
    id: str
    name: str
    display_name: str
    subscription_tier: SubscriptionTier
    primary_contact_email: str
    is_active: bool
    is_suspended: bool
    quota_limits: QuotaLimits
    feature_flags: FeatureFlags
    usage_metrics: UsageMetrics
    created_at: str
    updated_at: str


class TenantUserRequest(BaseModel):
    """Request model for adding user to tenant"""
    email: str = Field(..., description="User email")
    password: str = Field(..., min_length=8, description="User password")
    full_name: str = Field(..., description="User full name")
    role: str = Field("user", description="User role in tenant")


class UserRoleUpdateRequest(BaseModel):
    """Request model for updating user role"""
    role: str = Field(..., description="New role for user")


class SubscriptionUpdateRequest(BaseModel):
    """Request model for subscription updates"""
    tier: SubscriptionTier = Field(..., description="New subscription tier")


# Helper function for TenantConfiguration to TenantResponse conversion
def _convert_tenant_config_to_response(tenant) -> TenantResponse:
    """Convert TenantConfiguration or dict to TenantResponse with proper field mapping"""
    
    # Handle both TenantConfiguration objects and dicts
    if hasattr(tenant, 'model_dump'):
        # It's a Pydantic model
        tenant_dict = tenant.model_dump()
    else:
        # It's already a dict or dict-like object
        tenant_dict = dict(tenant) if not isinstance(tenant, dict) else tenant
    
    # Handle datetime conversions more robustly
    created_at = tenant_dict.get("created_at")
    updated_at = tenant_dict.get("updated_at")
    
    # Convert datetime objects to strings if they exist and aren't already strings
    if created_at:
        if hasattr(created_at, 'isoformat'):
            created_at = created_at.isoformat()
        else:
            created_at = str(created_at)  # Already a string or can be converted
    else:
        created_at = ""
        
    if updated_at:
        if hasattr(updated_at, 'isoformat'):
            updated_at = updated_at.isoformat()
        else:
            updated_at = str(updated_at)  # Already a string or can be converted
    else:
        updated_at = ""
    
    return TenantResponse(
        id=str(tenant_dict["id"]),  # Convert UUID to string
        name=tenant_dict["name"],
        display_name=tenant_dict["display_name"],
        subscription_tier=tenant_dict["subscription_tier"],
        primary_contact_email=tenant_dict["primary_contact_email"],
        is_active=tenant_dict["is_active"],
        is_suspended=tenant_dict["is_suspended"],
        quota_limits=tenant_dict["quota_limits"],
        feature_flags=tenant_dict["feature_flags"],
        usage_metrics=tenant_dict.get("current_usage", tenant_dict.get("usage_metrics", {})),  # Map current_usage to usage_metrics
        created_at=created_at,
        updated_at=updated_at
    )


# Super Admin Endpoints (Tenant Management)

@router.post(
    "/",
    response_model=TenantResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create new tenant",
    description="Create a new tenant organization. Super admin only."
)
async def create_tenant(
    tenant_data: TenantCreateRequest,
    current_user: CurrentUserDep,
    tenant_service: TenantServiceDep
):
    """Create a new tenant (Super Admin only)"""

    # Check if user has super_admin role
    if "super_admin" not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super admin role required"
        )

    try:
        # Create tenant with optional admin user
        created_tenant = await tenant_service.create_tenant(
            name=tenant_data.name,
            display_name=tenant_data.display_name,
            primary_contact_email=tenant_data.primary_contact_email,
            subscription_tier=tenant_data.subscription_tier,
            admin_user_data=tenant_data.admin_user
        )

        logger.info(
            "Tenant created by super admin",
            tenant_id=created_tenant.id,
            tenant_name=created_tenant.name,
            created_by=current_user.user_id
        )

        # Convert TenantConfiguration to TenantResponse
        return _convert_tenant_config_to_response(created_tenant)

    except ValueError as e:
        logger.warning(
            "Tenant creation failed",
            error=str(e),
            tenant_name=tenant_data.name,
            created_by=current_user.user_id
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Tenant creation system error",
            error=str(e),
            tenant_name=tenant_data.name,
            created_by=current_user.user_id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create tenant"
        )


@router.get(
    "/list",
    response_model=List[TenantResponse],
    summary="List all tenants",
    description="List all tenants in the system. Super admin only."
)
async def list_tenants(
    current_user: CurrentUserDep,
    tenant_service: TenantServiceDep,
    skip: int = Query(0, ge=0, description="Number of tenants to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of tenants to return")
):
    """List all tenants (Super Admin only)"""
    
    # Check if user has super_admin role
    if "super_admin" not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super admin role required"
        )
    
    try:
        # Get tenants using the implemented service method
        tenants = await tenant_service.list_tenants(skip=skip, limit=limit)
        
        # Convert TenantConfiguration objects to TenantResponse format
        return [_convert_tenant_config_to_response(tenant) for tenant in tenants]
        
    except Exception as e:
        logger.error("Failed to list tenants", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list tenants"
        )


@router.get(
    "/{tenant_id}",
    response_model=TenantResponse,
    summary="Get tenant details",
    description="Get detailed information about a specific tenant. Super admin can access any tenant, regular users can only access their own tenant."
)
async def get_tenant(
    tenant_id: str,
    current_user: CurrentUserDep,
    tenant_service: TenantServiceDep
):
    """Get tenant details (Super Admin can access any tenant, regular users can access their own tenant only)"""
    
    # Check authorization: super_admin can access any tenant, regular users only their own
    if "super_admin" not in current_user.roles:
        # Regular users can only access their own tenant
        if tenant_id != current_user.tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. You can only view your own tenant information."
            )
    
    try:
        tenant = await tenant_service.get_tenant(tenant_id)
        
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        # Convert TenantConfiguration to TenantResponse
        return _convert_tenant_config_to_response(tenant)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get tenant", error=str(e), tenant_id=tenant_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get tenant"
        )


@router.put(
    "/{tenant_id}",
    response_model=TenantResponse,
    summary="Update tenant",
    description="Update tenant configuration. Super admin can update any tenant, tenant admins can update their own tenant."
)
async def update_tenant(
    tenant_id: str,
    updates: TenantUpdateRequest,
    current_user: CurrentUserDep,
    tenant_service: TenantServiceDep
):
    """Update tenant (Super Admin can update any tenant, tenant admins can update their own tenant)"""
    
    # Check authorization: super_admin can update any tenant, tenant admins only their own
    if "super_admin" not in current_user.roles:
        # Check if user has admin role
        if "admin" not in current_user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin or Super Admin role required"
            )
        # Tenant admins can only update their own tenant
        if tenant_id != current_user.tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. Tenant admins can only update their own tenant."
            )
    
    try:
        updated_tenant = await tenant_service.update_tenant(
            tenant_id, 
            updates.model_dump(exclude_unset=True)
        )
        
        logger.info(
            "Tenant updated",
            tenant_id=tenant_id,
            updates=updates.model_dump(exclude_unset=True),
            updated_by=current_user.user_id,
            updated_by_role="super_admin" if "super_admin" in current_user.roles else "admin"
        )
        
        # Convert TenantConfiguration to TenantResponse
        return _convert_tenant_config_to_response(updated_tenant)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to update tenant", error=str(e), tenant_id=tenant_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update tenant"
        )


@router.delete(
    "/{tenant_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete tenant",
    description="Soft delete a tenant (deactivate). Super admin only."
)
async def delete_tenant(
    tenant_id: str,
    current_user: CurrentUserDep,
    tenant_service: TenantServiceDep
):
    """Delete tenant (Super Admin only)"""
    
    # Check if user has super_admin role
    if "super_admin" not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super admin role required"
        )
    
    try:
        success = await tenant_service.delete_tenant(tenant_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        logger.info(
            "Tenant deleted by super admin",
            tenant_id=tenant_id,
            deleted_by=current_user.user_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete tenant", error=str(e), tenant_id=tenant_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete tenant"
        )


@router.post(
    "/{tenant_id}/suspend",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Suspend tenant",
    description="Suspend a tenant and prevent access. Super admin only."
)
async def suspend_tenant(
    tenant_id: str,
    current_user: CurrentUserDep,
    tenant_service: TenantServiceDep,
    reason: str = Query(..., description="Reason for suspension")
):
    """Suspend tenant (Super Admin only)"""
    
    # Check if user has super_admin role
    if "super_admin" not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super admin role required"
        )
    
    try:
        success = await tenant_service.suspend_tenant(tenant_id, reason)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        logger.warning(
            "Tenant suspended by super admin",
            tenant_id=tenant_id,
            reason=reason,
            suspended_by=current_user.user_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to suspend tenant", error=str(e), tenant_id=tenant_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to suspend tenant"
        )


@router.post(
    "/{tenant_id}/activate",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Activate tenant",
    description="Activate a suspended tenant. Super admin only."
)
async def activate_tenant(
    tenant_id: str,
    current_user: CurrentUserDep,
    tenant_service: TenantServiceDep
):
    """Activate tenant (Super Admin only)"""
    
    # Check if user has super_admin role
    if "super_admin" not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super admin role required"
        )
    
    try:
        success = await tenant_service.activate_tenant(tenant_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        logger.info(
            "Tenant activated by super admin",
            tenant_id=tenant_id,
            activated_by=current_user.user_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to activate tenant", error=str(e), tenant_id=tenant_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to activate tenant"
        )


# Tenant Admin Endpoints (User Management within tenant)

@router.get(
    "/{tenant_id}/users",
    response_model=List[CurrentUser],
    summary="List tenant users",
    description="List all users in the tenant. Tenant admin only."
)
async def list_tenant_users(
    tenant_id: str,
    current_user: CurrentUserDep,
    tenant_service: TenantServiceDep
):
    """List tenant users (Tenant Admin only)"""
    
    # Check if user has admin role
    if not any(role in current_user.roles for role in ["admin", "super_admin"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or Super Admin role required"
        )
    
    # Verify user is admin of the requested tenant (super_admin can access any tenant)
    if "super_admin" not in current_user.roles and current_user.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Admins can only manage users in their own tenant"
        )
    
    try:
        users = await tenant_service.get_tenant_users(tenant_id)
        return users
        
    except Exception as e:
        logger.error("Failed to list tenant users", error=str(e), tenant_id=tenant_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list tenant users"
        )


@router.post(
    "/{tenant_id}/users",
    response_model=CurrentUser,
    status_code=status.HTTP_201_CREATED,
    summary="Add user to tenant",
    description="Add a new user to the tenant. Tenant admin only."
)
async def add_tenant_user(
    tenant_id: str,
    user_data: TenantUserRequest,
    current_user: CurrentUserDep,
    tenant_service: TenantServiceDep
):
    """Add user to tenant (Tenant Admin only)"""
    
    # Check if a user has an admin role
    if not any(role in current_user.roles for role in ["admin", "super_admin"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or Super Admin role required"
        )
    
    # Verify the user is admin of the requested tenant (super_admin can access any tenant)
    if "super_admin" not in current_user.roles and current_user.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Admins can only manage users in their own tenant"
        )
    
    try:
        # Use tenant service to create and add user - proper service layer abstraction
        created_user = await tenant_service.create_tenant_user(
            tenant_id=tenant_id,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name,
            roles=[user_data.role]
        )
        
        logger.info(
            "User added to tenant by admin",
            tenant_id=tenant_id,
            user_id=created_user.user_id,
            role=user_data.role,
            added_by=current_user.user_id
        )
        
        return created_user
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to add user to tenant", error=str(e), tenant_id=tenant_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add user to tenant"
        )


@router.put(
    "/{tenant_id}/users/{user_id}",
    response_model=CurrentUser,
    summary="Update user role",
    description="Update user role within the tenant. Tenant admin only."
)
async def update_user_role(
    tenant_id: str,
    user_id: str,
    role_update: UserRoleUpdateRequest,
    current_user: CurrentUserDep,
    tenant_service: TenantServiceDep
):
    """Update user role (Tenant Admin only)"""
    
    # Check if user has an admin role
    if not any(role in current_user.roles for role in ["admin", "super_admin"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or Super Admin role required"
        )
    
    # Verify the user is admin of the requested tenant (super_admin can access any tenant)
    if "super_admin" not in current_user.roles and current_user.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Admins can only manage users in their own tenant"
        )
    
    try:
        # Use tenant service to update user role - proper service layer abstraction
        updated_user = await tenant_service.update_user_role(
            tenant_id=tenant_id,
            user_id=user_id,
            new_role=role_update.role
        )
        
        logger.info(
            "User role updated by admin",
            tenant_id=tenant_id,
            user_id=user_id,
            new_role=role_update.role,
            updated_by=current_user.user_id
        )
        
        return updated_user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update user role", error=str(e), tenant_id=tenant_id, user_id=user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user role"
        )


@router.delete(
    "/{tenant_id}/users/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove user from tenant",
    description="Remove user from the tenant. Tenant admin only."
)
async def remove_tenant_user(
    tenant_id: str,
    user_id: str,
    current_user: CurrentUserDep,
    tenant_service: TenantServiceDep
):
    """Remove user from tenant (Tenant Admin only)"""
    
    # Check if user has admin role
    if not any(role in current_user.roles for role in ["admin", "super_admin"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or Super Admin role required"
        )
    
    # Verify user is admin of the requested tenant (super_admin can access any tenant)
    if "super_admin" not in current_user.roles and current_user.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Admins can only manage users in their own tenant"
        )
    
    # Prevent admin from removing themselves
    if user_id == current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot remove yourself from the tenant"
        )
    
    try:
        success = await tenant_service.remove_tenant_user(tenant_id, user_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found in tenant"
            )
        
        logger.info(
            "User removed from tenant by admin",
            tenant_id=tenant_id,
            user_id=user_id,
            removed_by=current_user.user_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to remove user from tenant", error=str(e), tenant_id=tenant_id, user_id=user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove user from tenant"
        )


# Usage and Billing Endpoints

@router.get(
    "/{tenant_id}/usage",
    response_model=UsageMetrics,
    summary="Get usage metrics",
    description="Get current usage metrics for the tenant. Tenant admin only."
)
async def get_tenant_usage(
    tenant_id: str,
    current_user: CurrentUserDep,
    tenant_service: TenantServiceDep
):
    """Get tenant usage metrics (Tenant Admin only)"""
    
    # Check if user has admin role
    if not any(role in current_user.roles for role in ["admin", "super_admin"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or Super Admin role required"
        )
    
    # Verify user is admin of the requested tenant (super_admin can access any tenant)
    if "super_admin" not in current_user.roles and current_user.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Admins can only view usage for their own tenant"
        )
    
    try:
        tenant = await tenant_service.get_tenant(tenant_id)
        
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        # Handle both TenantConfiguration objects and dicts  
        if hasattr(tenant, 'current_usage'):
            # It's a TenantConfiguration object
            return tenant.current_usage
        elif isinstance(tenant, dict):
            # It's a dict, check for current_usage or usage_metrics
            return tenant.get("current_usage", tenant.get("usage_metrics", UsageMetrics()))
        else:
            # Fallback - return default UsageMetrics
            return UsageMetrics()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get tenant usage", error=str(e), tenant_id=tenant_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get tenant usage"
        )


@router.put(
    "/{tenant_id}/subscription",
    response_model=TenantResponse,
    summary="Update subscription",
    description="Update tenant subscription tier. Tenant admin only."
)
async def update_subscription(
    tenant_id: str,
    subscription_update: SubscriptionUpdateRequest,
    current_user: CurrentUserDep,
    tenant_service: TenantServiceDep
):
    """Update subscription tier (Tenant Admin only)"""
    
    # Check if user has admin role
    if not any(role in current_user.roles for role in ["admin", "super_admin"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or Super Admin role required"
        )
    
    # Verify the user is admin of the requested tenant (super_admin can access any tenant)
    if "super_admin" not in current_user.roles and current_user.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Admins can only manage subscription for their own tenant"
        )
    
    try:
        success = await tenant_service.upgrade_subscription(tenant_id, subscription_update.tier)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        # Get updated tenant
        updated_tenant = await tenant_service.get_tenant(tenant_id)
        
        logger.info(
            "Subscription updated by tenant admin",
            tenant_id=tenant_id,
            new_tier=subscription_update.tier.value,
            updated_by=current_user.user_id
        )
        
        # Convert TenantConfiguration to TenantResponse
        return _convert_tenant_config_to_response(updated_tenant)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update subscription", error=str(e), tenant_id=tenant_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update subscription"
        )