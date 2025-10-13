"""Concrete factory for creating SearchHistoryApplicationService dependencies."""

from __future__ import annotations

from app.application.dependencies.search_history_dependencies import (
    SearchHistoryDependencies,
)
from app.infrastructure.providers.repository_provider import (
    get_search_history_repository,
)


async def get_search_history_dependencies() -> SearchHistoryDependencies:
    """
    Construct dependencies for the search history application service.

    Follows the established pattern: uses repository provider for singleton
    repository instances with async lock protection.
    """
    # Get repository via provider (singleton pattern)
    search_history_repository = await get_search_history_repository()

    return SearchHistoryDependencies(
        search_history_repository=search_history_repository
    )


__all__ = ["get_search_history_dependencies"]
