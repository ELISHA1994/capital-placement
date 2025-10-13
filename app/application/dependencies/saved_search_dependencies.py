"""Dependency container for saved search application service."""

from dataclasses import dataclass

from app.domain.repositories.saved_search_repository import ISavedSearchRepository


@dataclass
class SavedSearchDependencies:
    """Container for saved search service dependencies."""

    saved_search_repository: ISavedSearchRepository


__all__ = ["SavedSearchDependencies"]