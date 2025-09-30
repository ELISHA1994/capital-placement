"""
Dependency Injection Container

Provides a lightweight dependency injection container for managing service instances.
This container works alongside the service_factory to provide access to auth services
and other core services through FastAPI dependencies.
"""

import asyncio
from typing import Dict, Type, TypeVar, Optional, Any
from datetime import datetime
import time
import structlog

from app.core.service_factory import get_service_factory
from app.services.auth import AuthenticationService, AuthorizationService
from app.services.bootstrap_service import BootstrapService
from app.services.adapters.memory_cache_adapter import MemoryCacheService
from app.services.tenant.tenant_service import TenantService

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class Container:
    """Simple dependency injection container"""
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._factory = get_service_factory()
    
    def register_service(self, service_type: Type[T], instance: T) -> None:
        """Register a service instance"""
        self._services[service_type] = instance
    
    def get_service(self, service_type: Type[T]) -> T:
        """Get a service instance"""
        if service_type in self._services:
            return self._services[service_type]
        
        # For core services, delegate to factory
        service = self._create_service(service_type)
        if service:
            self._services[service_type] = service
            return service
        
        raise ValueError(f"Service {service_type.__name__} not registered and cannot be created")
    
    def _create_service(self, service_type: Type[T]) -> Optional[T]:
        """Create a service if possible"""
        
        # Handle auth services specially since they're not in the factory yet
        if service_type == AuthenticationService:
            # We'll return a mock for now - this will be replaced with proper implementation
            return self._create_auth_service()
        
        if service_type == AuthorizationService:
            return self._create_authz_service()
        
        if service_type == BootstrapService:
            return self._create_bootstrap_service()
        
        # For other services, try to use factory methods
        # This is a temporary solution until we fully integrate auth services
        return None
    
    def _create_auth_service(self) -> AuthenticationService:
        """Create authentication service with dependencies"""
        try:
            # Create real implementations for repositories 
            from app.database.repositories.postgres import UserRepository, TenantRepository
            from app.database import get_database_manager
            
            # Get database manager
            try:
                db_manager = get_database_manager()
            except Exception:
                # Fallback to None if not initialized yet
                db_manager = None
            
            auth_repo = UserRepository()
            tenant_repo = TenantRepository()
            
            # Use mock cache for now - will be replaced by AsyncContainer
            cache_manager = MockCacheManager()
            
            return AuthenticationService(auth_repo, tenant_repo, cache_manager)
        except Exception as e:
            logger.error("Failed to create auth service", error=str(e))
            raise
    
    def _create_authz_service(self) -> AuthorizationService:
        """Create authorization service with dependencies"""
        try:
            # Create real implementations for repositories - using PostgreSQL implementations
            from app.database.repositories.postgres import UserRepository, TenantRepository
            from app.database import get_database_manager
            
            # Get database manager
            try:
                db_manager = get_database_manager()
            except Exception:
                # Fallback to None if not initialized yet
                db_manager = None
            
            user_repo = UserRepository() 
            tenant_repo = TenantRepository()
            
            # Use mock cache for now
            cache_manager = MockCacheManager()
            
            # For now, use UserRepository for all auth-related repos until we can align interfaces
            return AuthorizationService(user_repo, user_repo, user_repo, tenant_repo, cache_manager)
        except Exception as e:
            logger.error("Failed to create authz service", error=str(e))
            raise
    
    def _create_bootstrap_service(self) -> BootstrapService:
        """Create bootstrap service with dependencies"""
        try:
            # Create real implementations for repositories
            from app.database.repositories.postgres import TenantRepository, UserRepository
            from app.database import get_database_manager
            
            # Get database manager
            try:
                db_manager = get_database_manager()
            except Exception:
                # Fallback to None if not initialized yet
                db_manager = None
            
            tenant_repo = TenantRepository()
            user_repo = UserRepository()
            
            # Get auth service
            auth_service = self._create_auth_service()
            
            return BootstrapService(tenant_repo, user_repo, auth_service)
        except Exception as e:
            logger.error("Failed to create bootstrap service", error=str(e))
            raise


# Mock implementations for development
class MockAuthRepository:
    """Mock authentication repository"""
    
    async def get_user_by_email(self, email: str, tenant_id: str):
        return None
    
    async def get_by_email(self, email: str, tenant_id: str):
        return None
    
    async def create_user(self, user):
        user.id = "mock-user-id"
        return user
    
    async def create(self, user):
        user.id = "mock-user-id" 
        return user
    
    async def get_user(self, user_id: str):
        return None
    
    async def update_user_last_login(self, user_id: str):
        pass
    
    async def update_user_password(self, user_id: str, hashed_password: str):
        pass
    
    async def create_api_key(self, api_key):
        api_key.id = "mock-key-id"
        return api_key
    
    async def get_active_api_keys(self):
        return []
    
    async def update_api_key_usage(self, key_id: str):
        pass
    
    async def create_session(self, session_info, refresh_token: str):
        pass
    
    async def update_session_token(self, user_id: str, refresh_token: str):
        pass
    
    async def revoke_user_sessions(self, user_id: str):
        pass
    
    async def create_audit_log(self, audit_log):
        pass
    
    async def get_role(self, role_name: str, tenant_id: str):
        return None
    
    async def create_role(self, role, tenant_id: str):
        return role
    
    async def update_role_permissions(self, role_name: str, permissions: list, tenant_id: str):
        pass


class MockTenantRepository:
    """Mock tenant repository"""
    
    def __init__(self):
        self._tenants = {}
    
    async def get_tenant(self, tenant_id: str):
        # Return a basic tenant structure
        class MockTenant:
            id = tenant_id
            is_active = True
            is_suspended = False
        return MockTenant()
    
    async def get(self, tenant_id: str):
        # Alias for get_tenant to match expected interface
        return await self.get_tenant(tenant_id)
    
    async def get_tenant_config(self, tenant_id: str):
        # Return basic tenant config
        class MockTenantConfig:
            tenant_id = tenant_id
            tenant_type = "standard"
            configuration = {}
            is_active = True
        return MockTenantConfig()
    
    async def list_all_tenants(self):
        return []
    
    async def list_all(self):
        """Get all tenants (alias for list_all_tenants)"""
        return await self.list_all_tenants()
    
    async def get_active_tenants(self):
        """Get all active tenants"""
        return list(self._tenants.values())
    
    async def check_slug_availability(self, slug: str, exclude_id: Optional[str] = None) -> bool:
        """Check if tenant slug is available"""
        # For mock purposes, let's allow all slugs except some reserved ones
        reserved_slugs = {"admin", "api", "www", "test"}
        
        if slug in reserved_slugs:
            return False
        
        # Check existing tenants
        for tenant in self._tenants.values():
            if hasattr(tenant, 'name') and tenant.name == slug:
                if exclude_id is None or tenant.id != exclude_id:
                    return False
        
        return True
    
    async def create(self, tenant_config):
        """Create a new tenant"""
        # Assign a mock ID
        tenant_config.id = f"tenant-{len(self._tenants) + 1}"
        
        # Store in mock storage
        self._tenants[tenant_config.id] = tenant_config
        
        return tenant_config
    
    async def get_by_slug(self, slug: str):
        """Get tenant by slug"""
        for tenant in self._tenants.values():
            if hasattr(tenant, 'name') and tenant.name == slug:
                return tenant
        return None


class MockCacheManager:
    """Mock cache manager"""
    
    def __init__(self):
        self._cache = {}
    
    async def get(self, key: str):
        return self._cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        self._cache[key] = value
    
    async def delete(self, key: str):
        self._cache.pop(key, None)
    
    async def exists(self, key: str) -> bool:
        return key in self._cache
    
    async def delete_pattern(self, pattern: str):
        # Simple pattern matching for mock
        keys_to_delete = [k for k in self._cache.keys() if pattern.replace("*", "") in k]
        for key in keys_to_delete:
            del self._cache[key]


class AsyncContainer:
    """Async-first dependency injection container"""
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._service_locks: Dict[Type, asyncio.Lock] = {}
        self._factory = get_service_factory()
        self._initialized = False
    
    async def initialize(self):
        """Initialize container and critical services"""
        if self._initialized:
            return
        
        try:
            # Pre-initialize critical async services
            cache_service = await self._factory.create_cache_service()
            self._services[MemoryCacheService] = cache_service
            
            logger.info("AsyncContainer initialized with services")
            self._initialized = True
            
        except Exception as e:
            logger.error("AsyncContainer initialization failed", error=str(e))
            raise
    
    async def get_service(self, service_type: Type[T]) -> T:
        """Get service instance (async)"""
        if not self._initialized:
            await self.initialize()
        
        if service_type in self._services:
            return self._services[service_type]
        
        # Use locks to prevent concurrent service creation
        if service_type not in self._service_locks:
            self._service_locks[service_type] = asyncio.Lock()
        
        async with self._service_locks[service_type]:
            # Double-check pattern
            if service_type in self._services:
                return self._services[service_type]
            
            service = await self._create_service(service_type)
            if service:
                self._services[service_type] = service
                return service
        
        raise ValueError(f"Service {service_type.__name__} not registered and cannot be created")
    
    async def _create_service(self, service_type: Type[T]) -> Optional[T]:
        """Create service asynchronously"""
        
        if service_type == AuthenticationService:
            return await self._create_auth_service()
        
        if service_type == AuthorizationService:
            return await self._create_authz_service()
        
        if service_type == BootstrapService:
            return await self._create_bootstrap_service()
        
        if service_type == TenantService:
            return await self._create_tenant_service()
        
        # Handle other services...
        return None
    
    async def _create_auth_service(self) -> AuthenticationService:
        """Create authentication service with proper async initialization"""
        try:
            # Create real repositories
            from app.database.repositories.postgres import UserRepository, TenantRepository
            from app.database import get_database_manager
            
            # Get database manager
            try:
                db_manager = get_database_manager()
            except Exception:
                # Fallback to None if not initialized yet
                db_manager = None
            
            auth_repo = UserRepository()
            tenant_repo = TenantRepository()
            
            # Get cache service from container (already initialized)
            cache_service = self._services.get(MemoryCacheService)
            if not cache_service:
                cache_service = await self._factory.create_cache_service()
                self._services[MemoryCacheService] = cache_service
            
            auth_service = AuthenticationService(auth_repo, tenant_repo, cache_service)
            
            logger.info("AuthenticationService created successfully")
            return auth_service
            
        except Exception as e:
            logger.error("Failed to create auth service", error=str(e))
            raise
    
    async def _create_authz_service(self) -> AuthorizationService:
        """Create authorization service with proper async initialization"""
        try:
            # Create real repositories for AuthorizationService
            from app.database.repositories.postgres import UserRepository, TenantRepository
            from app.database import get_database_manager
            
            # Get database manager
            try:
                db_manager = get_database_manager()
            except Exception:
                # Fallback to None if not initialized yet
                db_manager = None
            
            user_repo = UserRepository()
            tenant_repo = TenantRepository()
            
            # Get cache service from container
            cache_service = self._services.get(MemoryCacheService)
            if not cache_service:
                cache_service = await self._factory.create_cache_service()
                self._services[MemoryCacheService] = cache_service
            
            # For now, use UserRepository for all auth-related repos until we can align interfaces
            authz_service = AuthorizationService(
                user_repo, user_repo, user_repo, tenant_repo, cache_service
            )
            
            logger.info("AuthorizationService created successfully")
            return authz_service
            
        except Exception as e:
            logger.error("Failed to create authz service", error=str(e))
            raise
    
    async def _create_bootstrap_service(self) -> BootstrapService:
        """Create bootstrap service with proper async initialization"""
        try:
            # Create real repositories
            from app.database.repositories.postgres import TenantRepository, UserRepository
            from app.database import get_database_manager
            
            # Get database manager
            try:
                db_manager = get_database_manager()
            except Exception:
                # Fallback to None if not initialized yet
                db_manager = None
            
            tenant_repo = TenantRepository()
            user_repo = UserRepository()
            
            # Get auth service
            auth_service = await self._create_auth_service()
            
            bootstrap_service = BootstrapService(tenant_repo, user_repo, auth_service)
            
            logger.info("BootstrapService created successfully")
            return bootstrap_service
            
        except Exception as e:
            logger.error("Failed to create bootstrap service", error=str(e))
            raise
    
    async def _create_tenant_service(self) -> TenantService:
        """Create tenant service with proper async initialization"""
        try:
            # Create real repositories
            from app.database.repositories.postgres import TenantRepository, UserRepository
            from app.database import get_database_manager
            
            # Get database manager
            try:
                db_manager = get_database_manager()
            except Exception:
                # Fallback to None if not initialized yet
                db_manager = None
            
            tenant_repo = TenantRepository()
            user_repo = UserRepository()
            
            # Get cache service from container (already initialized)
            cache_service = self._services.get(MemoryCacheService)
            if not cache_service:
                cache_service = await self._factory.create_cache_service()
                self._services[MemoryCacheService] = cache_service
            
            tenant_service = TenantService(
                tenant_repository=tenant_repo,
                user_repository=user_repo,
                cache_manager=cache_service
            )
            
            logger.info("TenantService created successfully")
            return tenant_service
            
        except Exception as e:
            logger.error("Failed to create tenant service", error=str(e))
            raise
    
    async def cleanup(self):
        """Cleanup async services"""
        try:
            # Close any connections, clear caches, etc.
            if MemoryCacheService in self._services:
                cache_service = self._services[MemoryCacheService]
                # Perform cleanup if needed
            
            self._services.clear()
            self._service_locks.clear()
            self._initialized = False
            
            logger.info("AsyncContainer cleanup completed")
            
        except Exception as e:
            logger.error("AsyncContainer cleanup error", error=str(e))


# Global container instances
_container: Optional[Container] = None


def get_container() -> Container:
    """Get or create the global container instance"""
    global _container
    if _container is None:
        _container = Container()
        logger.info("Container initialized")
    return _container


def reset_container() -> None:
    """Reset the global container (mainly for testing)"""
    global _container
    _container = None


# Global async container instance
_async_container: Optional[AsyncContainer] = None


async def get_async_container() -> AsyncContainer:
    """Get or create the global async container instance"""
    global _async_container
    if _async_container is None:
        _async_container = AsyncContainer()
        await _async_container.initialize()
        logger.info("AsyncContainer initialized")
    return _async_container


async def reset_async_container() -> None:
    """Reset the global async container (mainly for testing)"""
    global _async_container
    if _async_container:
        await _async_container.cleanup()
    _async_container = None