"""Dependency container for click tracking service."""

from dataclasses import dataclass

from app.domain.repositories.search_click_repository import ISearchClickRepository


@dataclass
class ClickTrackingDependencies:
    """Container for click tracking service dependencies."""

    click_repository: ISearchClickRepository


__all__ = ["ClickTrackingDependencies"]
