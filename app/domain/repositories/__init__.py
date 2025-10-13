"""Domain repository abstractions."""

from .profile_repository import IProfileRepository
from .saved_search_repository import ISavedSearchRepository
from .search_click_repository import ISearchClickRepository
from .search_history_repository import ISearchHistoryRepository
from .tenant_repository import ITenantRepository
from .user_repository import IUserRepository

__all__ = [
    "IProfileRepository",
    "ISavedSearchRepository",
    "ISearchClickRepository",
    "ISearchHistoryRepository",
    "ITenantRepository",
    "IUserRepository",
]
