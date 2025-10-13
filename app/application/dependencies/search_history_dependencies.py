"""Dependency container for search history application service."""

from dataclasses import dataclass

from app.domain.repositories.search_history_repository import ISearchHistoryRepository


@dataclass
class SearchHistoryDependencies:
    """Container for search history service dependencies."""

    search_history_repository: ISearchHistoryRepository


__all__ = ["SearchHistoryDependencies"]
