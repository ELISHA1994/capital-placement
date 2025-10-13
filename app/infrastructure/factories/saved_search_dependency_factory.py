"""Concrete factory for creating SavedSearchApplicationService dependencies."""

from __future__ import annotations

from app.application.dependencies.saved_search_dependencies import (
    SavedSearchDependencies,
)
from app.infrastructure.providers.repository_provider import get_saved_search_repository


async def get_saved_search_dependencies() -> SavedSearchDependencies:
    """
    Construct dependencies for the saved search application service.

    Follows the established pattern: uses repository provider for singleton
    repository instances with async lock protection.
    """
    # Get repository via provider (singleton pattern)
    saved_search_repository = await get_saved_search_repository()

    return SavedSearchDependencies(
        saved_search_repository=saved_search_repository
    )


__all__ = ["get_saved_search_dependencies"]
