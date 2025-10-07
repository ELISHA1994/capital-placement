"""
FastAPI Dependencies
"""

from typing import Annotated, Dict, Optional, TYPE_CHECKING

import structlog
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import Settings, get_settings
from app.infrastructure.persistence.models.auth_tables import CurrentUser, TenantContext
from app.domain.interfaces import ICacheService
from app.infrastructure.providers.auth_provider import (
    get_authentication_service as resolve_authentication_service,
    get_authorization_service as resolve_authorization_service,
    get_tenant_service as resolve_tenant_service,
)
from app.infrastructure.providers.bootstrap_provider import (
    get_bootstrap_service as resolve_bootstrap_service,
)
from app.infrastructure.providers.cache_provider import (
    get_cache_service as resolve_cache_service,
)
from app.services.auth.authentication_service import AuthenticationService
from app.services.auth.authorization_service import AuthorizationService
from app.services.bootstrap_service import BootstrapService
# from app.services.core.document_processor import DocumentProcessor  # Temporarily disabled until CV models are created
# from app.services.core.search_engine import SearchEngine  # Temporarily disabled until CV models are created
# NotificationService and ProfileService - to be implemented later
# from app.services.notification import NotificationService  
# from app.services.profile import ProfileService

# Database imports
from app.database import get_database_manager, DatabaseManager
from app.database.repositories.postgres import (
    TenantRepository,
    UserRepository,
    JobRepository,
    CandidateRepository,
    MatchRepository,
)

# Transaction manager imports
from app.core.transaction_manager import SQLModelTransactionManager, get_transaction_manager

logger = structlog.get_logger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)


# Settings dependency
def get_settings_dependency() -> Settings:
    """Get application settings"""
    return get_settings()


# Provider-backed service dependencies
async def get_async_auth_service() -> AuthenticationService:
    """Resolve the authentication service from infrastructure providers."""
    return await resolve_authentication_service()


async def get_async_authz_service() -> AuthorizationService:
    """Resolve the authorization service from infrastructure providers."""
    return await resolve_authorization_service()


async def get_async_bootstrap_service() -> BootstrapService:
    """Resolve the bootstrap service from infrastructure providers."""
    return await resolve_bootstrap_service()


async def get_async_tenant_service() -> "TenantService":
    """Resolve the tenant service from infrastructure providers."""
    return await resolve_tenant_service()


# TODO: Add provider-backed profile/search/document services when implementations are available.

# Compatibility stubs (to be removed once all modules drop legacy imports)
async def get_async_container():
    """Legacy shim retained for backward compatibility with provider migration."""
    raise RuntimeError("Async container has been removed; rely on provider helpers instead.")

async def get_cache_service_dependency() -> ICacheService:
    """Resolve the cache service from infrastructure providers."""
    return await resolve_cache_service()


# Database dependencies
def get_database_manager_dependency() -> DatabaseManager:
    """Get database manager"""
    return get_database_manager()


def get_transaction_manager_dependency() -> SQLModelTransactionManager:
    """Get transaction manager"""
    return get_transaction_manager()


def get_tenant_repository() -> TenantRepository:
    """Get tenant repository"""
    return TenantRepository()


def get_user_repository() -> UserRepository:
    """Get user repository"""
    return UserRepository()


def get_job_repository() -> JobRepository:
    """Get job repository"""
    return JobRepository()


def get_candidate_repository() -> CandidateRepository:
    """Get candidate repository"""
    return CandidateRepository()


def get_match_repository() -> MatchRepository:
    """Get match repository"""
    return MatchRepository()


# Authentication dependencies
async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth_service: AuthenticationService = Depends(get_async_auth_service),
) -> Optional[CurrentUser]:
    """Get current user from token (optional)"""
    if not credentials:
        return None
    
    try:
        user_data = await auth_service.verify_token(credentials.credentials)
        if not user_data:
            return None
        
        return CurrentUser(**user_data)
    except Exception as e:
        logger.warning("Failed to verify token", error=str(e))
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthenticationService = Depends(get_async_auth_service),
) -> CurrentUser:
    """Get current authenticated user"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        user_data = await auth_service.verify_token(credentials.credentials)
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return CurrentUser(**user_data)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Token verification failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_tenant_context(
    request: Request,
    current_user: Optional[CurrentUser] = Depends(get_current_user_optional),
) -> TenantContext:
    """Get tenant context from request"""
    
    # Try to get tenant ID from various sources
    tenant_id = None
    
    # 1. From authenticated user
    if current_user:
        tenant_id = current_user.tenant_id
    
    # 2. From X-Tenant-ID header
    if not tenant_id:
        tenant_id = request.headers.get("X-Tenant-ID")
    
    # 3. From query parameter
    if not tenant_id:
        tenant_id = request.query_params.get("tenant_id")
    
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant ID required"
        )
    
    # Get tenant configuration
    try:
        # Get tenant service
        tenant_service = await get_async_tenant_service()
        tenant_config = await tenant_service.get_tenant(tenant_id)
        
        if not tenant_config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        # Check if tenant is active - handle both dict and object formats
        is_active = tenant_config.get("is_active") if isinstance(tenant_config, dict) else tenant_config.is_active
        if not is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tenant is not active"
            )
        
        return TenantContext(
            tenant_id=tenant_id,
            tenant_type="organization",  # Default type
            configuration={},  # Default empty config
            is_active=is_active
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get tenant context", tenant_id=tenant_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resolve tenant context"
        )


# Authorization dependencies
def require_permission(permission: str):
    """Dependency factory for permission-based authorization"""
    
    async def permission_checker(
        current_user: CurrentUser = Depends(get_current_user),
        tenant_context: TenantContext = Depends(get_tenant_context),
        authz_service: AuthorizationService = Depends(get_async_authz_service),
    ) -> CurrentUser:
        """Check if user has required permission"""
        
        has_permission = await authz_service.check_permission(
            user_roles=current_user.roles,
            required_permission=permission,
            tenant_id=tenant_context.tenant_id
        )
        
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {permission}"
            )
        
        return current_user
    
    return permission_checker


def require_tenant_access():
    """Dependency for tenant-level access control"""
    
    async def tenant_access_checker(
        current_user: CurrentUser = Depends(get_current_user),
        tenant_context: TenantContext = Depends(get_tenant_context),
        authz_service: AuthorizationService = Depends(get_async_authz_service),
    ) -> CurrentUser:
        """Check if user has access to tenant resources"""
        
        has_access = await authz_service.check_tenant_access(
            user_tenant_id=current_user.tenant_id,
            resource_tenant_id=tenant_context.tenant_id
        )
        
        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )
        
        return current_user
    
    return tenant_access_checker


def require_role(role: str):
    """Dependency factory for role-based authorization"""
    
    async def role_checker(
        current_user: CurrentUser = Depends(get_current_user),
    ) -> CurrentUser:
        """Check if user has required role"""
        
        if role not in current_user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' required"
            )
        
        return current_user
    
    return role_checker


# API Key dependencies
async def validate_api_key(
    request: Request,
    auth_service: AuthenticationService = Depends(get_async_auth_service),
) -> Dict[str, str]:
    """Validate API key from header"""
    
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    try:
        key_info = await auth_service.validate_api_key(api_key)
        if not key_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired API key"
            )
        
        return key_info
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("API key validation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key validation failed"
        )


# Combined auth dependencies
async def get_auth_context(
    current_user: Optional[CurrentUser] = Depends(get_current_user_optional),
    api_key_info: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Get authentication context (user or API key)"""
    
    if current_user:
        return {
            "type": "user",
            "user_id": current_user.user_id,
            "tenant_id": current_user.tenant_id,
            "roles": current_user.roles,
            "permissions": current_user.permissions,
        }
    
    if api_key_info:
        return {
            "type": "api_key",
            "key_id": api_key_info["id"],
            "tenant_id": api_key_info["tenant_id"],
            "permissions": api_key_info["permissions"],
        }
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required"
    )


# Type aliases for cleaner code
AuthService = Annotated[AuthenticationService, Depends(get_async_auth_service)]
AuthzService = Annotated[AuthorizationService, Depends(get_async_authz_service)]
# ProfileService = Annotated[ProfileService, Depends(get_profile_service)]  # Not implemented yet
# SearchService = Annotated[SearchEngine, Depends(get_search_service)]  # Temporarily disabled
# DocumentService = Annotated[DocumentProcessor, Depends(get_document_service)]  # Temporarily disabled
# Import TenantService for type annotation
if TYPE_CHECKING:
    from app.services.tenant.tenant_service import TenantService

TenantServiceDep = Annotated["TenantService", Depends(get_async_tenant_service)]
# NotificationService = Annotated[NotificationService, Depends(get_notification_service)]  # Not implemented yet
CacheServiceDep = Annotated[ICacheService, Depends(get_cache_service_dependency)]
CurrentUserDep = Annotated[CurrentUser, Depends(get_current_user)]
OptionalCurrentUser = Annotated[Optional[CurrentUser], Depends(get_current_user_optional)]
TenantContextDep = Annotated[TenantContext, Depends(get_tenant_context)]

# Database type aliases
DatabaseManagerDep = Annotated[DatabaseManager, Depends(get_database_manager_dependency)]
TransactionManagerDep = Annotated[SQLModelTransactionManager, Depends(get_transaction_manager_dependency)]
TenantRepositoryDep = Annotated[TenantRepository, Depends(get_tenant_repository)]
UserRepositoryDep = Annotated[UserRepository, Depends(get_user_repository)]
JobRepositoryDep = Annotated[JobRepository, Depends(get_job_repository)]
CandidateRepositoryDep = Annotated[CandidateRepository, Depends(get_candidate_repository)]
MatchRepositoryDep = Annotated[MatchRepository, Depends(get_match_repository)]
