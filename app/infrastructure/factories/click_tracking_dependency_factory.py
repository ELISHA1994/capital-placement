"""Concrete factory for creating ClickTrackingApplicationService dependencies."""

from __future__ import annotations

from app.application.dependencies.click_tracking_dependencies import (
    ClickTrackingDependencies,
)
from app.infrastructure.providers.repository_provider import get_search_click_repository


async def get_click_tracking_dependencies() -> ClickTrackingDependencies:
    """
    Construct dependencies for the click tracking application service.

    Follows the established pattern: uses repository provider for singleton
    repository instances with async lock protection.
    """
    # Get repository via provider (singleton pattern)
    click_repository = await get_search_click_repository()

    return ClickTrackingDependencies(
        click_repository=click_repository
    )


__all__ = ["get_click_tracking_dependencies"]
