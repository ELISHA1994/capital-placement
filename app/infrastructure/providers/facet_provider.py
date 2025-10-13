"""Provider for facet application service with dependency injection."""

from __future__ import annotations

import asyncio
from typing import Optional

from app.application.facet_application_service import FacetApplicationService
from app.infrastructure.adapters.facet_adapter import PostgresFacetAdapter
from app.infrastructure.providers.postgres_provider import get_postgres_adapter
from app.infrastructure.providers.cache_provider import get_cache_service

_facet_service: Optional[FacetApplicationService] = None
_lock = asyncio.Lock()


async def get_facet_application_service() -> FacetApplicationService:
    """Get or create facet application service."""
    global _facet_service

    if _facet_service is not None:
        return _facet_service

    async with _lock:
        if _facet_service is not None:
            return _facet_service

        # Get dependencies
        db_adapter = await get_postgres_adapter()
        cache_service = await get_cache_service()

        # Create adapters
        facet_adapter = PostgresFacetAdapter(db_adapter)

        # Create application service
        _facet_service = FacetApplicationService(
            facet_service=facet_adapter,
            cache_service=cache_service
        )

        return _facet_service


async def reset_facet_service() -> None:
    """Reset facet service (for testing)."""
    global _facet_service
    async with _lock:
        _facet_service = None


__all__ = ["get_facet_application_service", "reset_facet_service"]
