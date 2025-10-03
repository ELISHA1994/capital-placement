"""Authentication/authorization provider utilities."""

from __future__ import annotations

import asyncio
from typing import Optional

from app.services.auth.authentication_service import AuthenticationService
from app.services.auth.authorization_service import AuthorizationService
from app.services.bootstrap_service import BootstrapService
from app.services.tenant.tenant_service import TenantService
from app.database.repositories.postgres import (
    UserRepository,
    TenantRepository,
    UserSessionRepository,
)
from app.infrastructure.providers.cache_provider import get_cache_service
from app.infrastructure.providers.notification_provider import get_notification_service

_authentication_service: Optional[AuthenticationService] = None
_authorization_service: Optional[AuthorizationService] = None
_bootstrap_service: Optional[BootstrapService] = None
_tenant_service: Optional[TenantService] = None

_auth_lock = asyncio.Lock()
_authz_lock = asyncio.Lock()
_bootstrap_lock = asyncio.Lock()
_tenant_lock = asyncio.Lock()


async def get_authentication_service() -> AuthenticationService:
    """Return singleton authentication service configured with repositories."""
    global _authentication_service

    if _authentication_service is not None:
        return _authentication_service

    async with _auth_lock:
        if _authentication_service is not None:
            return _authentication_service

        cache_service = await get_cache_service()
        notification_service = await get_notification_service()

        _authentication_service = AuthenticationService(
            user_repository=UserRepository(),
            tenant_repository=TenantRepository(),
            cache_manager=cache_service,
            notification_service=notification_service,
            session_repository=UserSessionRepository(),
        )
        return _authentication_service


async def reset_authentication_service() -> None:
    global _authentication_service
    async with _auth_lock:
        _authentication_service = None


async def get_authorization_service() -> AuthorizationService:
    """Return singleton authorization service."""
    global _authorization_service

    if _authorization_service is not None:
        return _authorization_service

    async with _authz_lock:
        if _authorization_service is not None:
            return _authorization_service

        cache_service = await get_cache_service()
        _authorization_service = AuthorizationService(
            user_repository=UserRepository(),
            tenant_repository=TenantRepository(),
            cache_manager=cache_service,
        )
        return _authorization_service


async def reset_authorization_service() -> None:
    global _authorization_service
    async with _authz_lock:
        _authorization_service = None


async def get_bootstrap_service() -> BootstrapService:
    """Return singleton bootstrap service."""
    global _bootstrap_service

    if _bootstrap_service is not None:
        return _bootstrap_service

    async with _bootstrap_lock:
        if _bootstrap_service is not None:
            return _bootstrap_service

        auth_service = await get_authentication_service()
        _bootstrap_service = BootstrapService(
            tenant_repository=TenantRepository(),
            user_repository=UserRepository(),
            auth_service=auth_service,
        )
        return _bootstrap_service


async def reset_bootstrap_service() -> None:
    global _bootstrap_service
    async with _bootstrap_lock:
        _bootstrap_service = None


async def get_tenant_service() -> TenantService:
    """Return singleton tenant service."""
    global _tenant_service

    if _tenant_service is not None:
        return _tenant_service

    async with _tenant_lock:
        if _tenant_service is not None:
            return _tenant_service

        cache_service = await get_cache_service()
        _tenant_service = TenantService(
            tenant_repository=TenantRepository(),
            user_repository=UserRepository(),
            cache_manager=cache_service,
        )
        return _tenant_service


async def reset_tenant_service() -> None:
    global _tenant_service
    async with _tenant_lock:
        _tenant_service = None


__all__ = [
    "get_authentication_service",
    "reset_authentication_service",
    "get_authorization_service",
    "reset_authorization_service",
    "get_bootstrap_service",
    "reset_bootstrap_service",
    "get_tenant_service",
    "reset_tenant_service",
]

