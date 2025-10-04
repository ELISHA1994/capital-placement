"""Provider utilities for audit logging services following hexagonal architecture."""

from __future__ import annotations

import asyncio
from typing import Optional

from app.domain.interfaces import IAuditService
from app.services.audit.audit_service import AuditService
from app.infrastructure.providers.postgres_provider import get_postgres_adapter

_audit_service: Optional[IAuditService] = None
_audit_lock = asyncio.Lock()


async def get_audit_service() -> IAuditService:
    """
    Return singleton audit service configured via provider pattern.
    
    This function follows the hexagonal architecture pattern by:
    - Returning the domain interface (IAuditService)
    - Using dependency injection via providers
    - Ensuring singleton behavior for performance
    - Abstracting infrastructure concerns from consumers
    
    Returns:
        IAuditService: Configured audit service instance
    """
    global _audit_service

    if _audit_service is not None:
        return _audit_service

    async with _audit_lock:
        if _audit_service is not None:
            return _audit_service

        # Get database adapter via provider
        postgres_adapter = await get_postgres_adapter()
        
        # Create audit service with injected dependencies
        _audit_service = AuditService(database_adapter=postgres_adapter)
        
        return _audit_service


async def reset_audit_service() -> None:
    """
    Reset the audit service singleton for testing or reconfiguration.
    
    This is primarily used in test environments to ensure clean state
    between test runs.
    """
    global _audit_service
    
    async with _audit_lock:
        _audit_service = None


__all__ = ["get_audit_service", "reset_audit_service"]