"""
Authentication API Endpoints

Provides authentication and user management endpoints:
- User registration and login
- Token refresh and logout
- Password management
- Profile management
- Session management
"""

from datetime import datetime
from typing import List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials

from app.core.dependencies import (
    get_async_auth_service, get_async_authz_service, get_current_user, 
    get_current_user_optional, security
)
from app.models.auth import (
    UserCreate, UserLogin, UserUpdate, TokenResponse, RefreshTokenRequest,
    CurrentUser, PasswordChangeRequest, PasswordResetRequest, PasswordResetConfirm,
    APIKeyCreate, APIKeyResponse, APIKeyInfo, SessionInfo
)
from app.services.auth import AuthenticationService, AuthorizationService
from app.utils.security import security_validator

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post(
    "/register",
    response_model=CurrentUser,
    status_code=status.HTTP_201_CREATED,
    summary="Register new user",
    description="Register a new user account with tenant association"
)
async def register_user(
    user_data: UserCreate,
    request: Request,
    auth_service: AuthenticationService = Depends(get_async_auth_service)
):
    """Register a new user account"""
    
    try:
        # Additional email validation
        if not security_validator.validate_email(user_data.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid email format"
            )
        
        # Register user
        created_user = await auth_service.register_user(user_data)
        
        logger.info(
            "User registration completed",
            user_id=created_user.id,
            email=created_user.email,
            tenant_id=created_user.tenant_id,
            ip_address=request.client.host if request.client else "unknown"
        )
        
        # Convert to CurrentUser for response
        return CurrentUser(
            user_id=str(created_user.id),
            email=created_user.email,
            full_name=created_user.full_name,
            tenant_id=str(created_user.tenant_id),
            roles=created_user.roles,
            permissions=created_user.permissions,
            is_active=created_user.is_active,
            is_superuser=created_user.is_superuser
        )
        
    except ValueError as e:
        # Business logic errors (validation, conflicts, etc.)
        logger.warning(
            "Registration validation failed",
            error=str(e),
            email=user_data.email,
            tenant_id=user_data.tenant_id,
            ip_address=request.client.host if request.client else "unknown"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Unexpected system errors
        logger.error(
            "User registration system failure",
            error=str(e),
            email=user_data.email,
            tenant_id=user_data.tenant_id,
            ip_address=request.client.host if request.client else "unknown"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed due to system error"
        )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="User login",
    description="Authenticate user and return access/refresh tokens"
)
async def login(
    credentials: UserLogin,
    request: Request,
    auth_service: AuthenticationService = Depends(get_async_auth_service)
):
    """Authenticate user and return tokens"""
    
    try:
        # Add request context to credentials if needed
        # credentials.ip_address = request.client.host if request.client else "unknown"
        # credentials.user_agent = request.headers.get("user-agent", "unknown")
        
        # Authenticate user
        result = await auth_service.authenticate(credentials)
        
        if not result.success:
            logger.warning(
                "Login attempt failed",
                email=credentials.email,
                tenant_id=credentials.tenant_id,
                error=result.error,
                ip_address=request.client.host if request.client else "unknown"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=result.error or "Authentication failed"
            )
        
        logger.info(
            "User logged in successfully",
            user_id=result.user.user_id if result.user else "unknown",
            email=credentials.email,
            tenant_id=credentials.tenant_id,
            ip_address=request.client.host if request.client else "unknown"
        )
        
        return result.tokens
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh tokens",
    description="Refresh access token using refresh token"
)
async def refresh_tokens(
    request_data: RefreshTokenRequest,
    auth_service: AuthenticationService = Depends(get_async_auth_service)
):
    """Refresh access token using refresh token"""
    
    try:
        tokens = await auth_service.refresh_tokens(request_data)
        
        if not tokens:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token"
            )
        
        logger.info("Tokens refreshed successfully")
        
        return tokens
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Token refresh failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="User logout",
    description="Logout user and revoke tokens"
)
async def logout(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    current_user: Optional[CurrentUser] = Depends(get_current_user_optional),
    auth_service: AuthenticationService = Depends(get_async_auth_service)
):
    """Logout user and revoke current token"""
    
    try:
        if credentials:
            # Revoke the current token
            await auth_service.revoke_token(credentials.credentials)
            
            if current_user:
                logger.info(
                    "User logged out successfully",
                    user_id=current_user.user_id,
                    tenant_id=current_user.tenant_id
                )
        
        return Response(status_code=status.HTTP_204_NO_CONTENT)
        
    except Exception as e:
        logger.error("Logout failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get(
    "/me",
    response_model=CurrentUser,
    summary="Get current user",
    description="Get current authenticated user information"
)
async def get_current_user_info(
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get current authenticated user information"""
    return current_user


@router.put(
    "/me",
    response_model=CurrentUser,
    summary="Update user profile",
    description="Update current user's profile information"
)
async def update_user_profile(
    user_data: UserUpdate,
    current_user: CurrentUser = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_async_auth_service)
):
    """Update current user's profile"""
    
    try:
        logger.info(
            "User profile update requested",
            user_id=current_user.user_id,
            updates=user_data.dict(exclude_unset=True)
        )
        
        updated_user = await auth_service.update_user_profile(
            current_user=current_user,
            update_request=user_data
        )

        logger.info(
            "User profile updated successfully",
            user_id=updated_user.user_id,
            tenant_id=updated_user.tenant_id
        )

        return updated_user
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Profile update failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile update failed"
        )


@router.post(
    "/change-password",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Change password",
    description="Change user password"
)
async def change_password(
    request_data: PasswordChangeRequest,
    current_user: CurrentUser = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_async_auth_service)
):
    """Change user password"""
    
    try:
        success = await auth_service.change_password(current_user.user_id, request_data)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        logger.info("Password changed successfully", user_id=current_user.user_id)
        
        return Response(status_code=status.HTTP_204_NO_CONTENT)
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Password change failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )


@router.post(
    "/forgot-password",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Request password reset",
    description="Request password reset email"
)
async def forgot_password(
    request_data: PasswordResetRequest,
    auth_service: AuthenticationService = Depends(get_async_auth_service)
):
    """Request password reset"""
    
    try:
        # TODO: Implement password reset functionality
        logger.info(
            "Password reset requested",
            email=request_data.email,
            tenant_id=request_data.tenant_id
        )
        
        # Always return success to prevent email enumeration
        return Response(status_code=status.HTTP_204_NO_CONTENT)
        
    except Exception as e:
        logger.error("Password reset request failed", error=str(e))
        # Still return success to prevent information disclosure
        return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/reset-password",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Reset password",
    description="Reset password using reset token"
)
async def reset_password(
    request_data: PasswordResetConfirm,
    auth_service: AuthenticationService = Depends(get_async_auth_service)
):
    """Reset password using reset token"""
    
    try:
        # TODO: Implement password reset confirmation
        logger.info("Password reset attempted with token")
        
        return Response(status_code=status.HTTP_204_NO_CONTENT)
        
    except Exception as e:
        logger.error("Password reset failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )


@router.post(
    "/api-keys",
    response_model=APIKeyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create API key",
    description="Create a new API key for programmatic access"
)
async def create_api_key(
    request_data: APIKeyCreate,
    current_user: CurrentUser = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_async_auth_service),
    authz_service: AuthorizationService = Depends(get_async_authz_service)
):
    """Create a new API key"""
    
    try:
        # Check permission to create API keys
        has_permission = await authz_service.check_permission(
            user_roles=current_user.roles,
            required_permission="create:api_key",
            tenant_id=current_user.tenant_id
        )
        
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to create API keys"
            )
        
        # Set tenant ID from current user
        request_data.tenant_id = current_user.tenant_id
        
        # Create API key
        api_key_response = await auth_service.create_api_key(request_data)
        
        logger.info(
            "API key created",
            user_id=current_user.user_id,
            key_id=api_key_response.key_id,
            name=request_data.name
        )
        
        return api_key_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("API key creation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key creation failed"
        )


@router.get(
    "/api-keys",
    response_model=List[APIKeyInfo],
    summary="List API keys",
    description="List user's API keys (without the key values)"
)
async def list_api_keys(
    current_user: CurrentUser = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_async_auth_service),
    authz_service: AuthorizationService = Depends(get_async_authz_service)
):
    """List user's API keys"""
    
    try:
        # Check permission to list API keys
        has_permission = await authz_service.check_permission(
            user_roles=current_user.roles,
            required_permission="read:api_key",
            tenant_id=current_user.tenant_id
        )
        
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to list API keys"
            )
        
        # TODO: Implement list_api_keys in AuthenticationService
        return []
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("API key listing failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key listing failed"
        )


@router.delete(
    "/api-keys/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke API key",
    description="Revoke an API key"
)
async def revoke_api_key(
    key_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_async_auth_service),
    authz_service: AuthorizationService = Depends(get_async_authz_service)
):
    """Revoke an API key"""
    
    try:
        # Check permission to revoke API keys
        has_permission = await authz_service.check_permission(
            user_roles=current_user.roles,
            required_permission="delete:api_key",
            tenant_id=current_user.tenant_id
        )
        
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to revoke API keys"
            )
        
        # TODO: Implement revoke_api_key in AuthenticationService
        logger.info(
            "API key revoked",
            user_id=current_user.user_id,
            key_id=key_id
        )
        
        return Response(status_code=status.HTTP_204_NO_CONTENT)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("API key revocation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key revocation failed"
        )


@router.get(
    "/sessions",
    response_model=List[SessionInfo],
    summary="List active sessions",
    description="List user's active sessions"
)
async def list_sessions(
    current_user: CurrentUser = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_async_auth_service)
):
    """List user's active sessions"""
    
    try:
        # TODO: Implement list_sessions in AuthenticationService
        sessions = []  # await auth_service.list_sessions(current_user.user_id)
        
        return sessions
        
    except Exception as e:
        logger.error("Session listing failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Session listing failed"
        )


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Terminate session",
    description="Terminate a specific session"
)
async def terminate_session(
    session_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_async_auth_service)
):
    """Terminate a specific session"""
    
    try:
        # TODO: Implement terminate_session in AuthenticationService
        # success = await auth_service.terminate_session(session_id, current_user.user_id)
        
        logger.info(
            "Session terminated",
            user_id=current_user.user_id,
            session_id=session_id
        )
        
        return Response(status_code=status.HTTP_204_NO_CONTENT)
        
    except Exception as e:
        logger.error("Session termination failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Session termination failed"
        )

