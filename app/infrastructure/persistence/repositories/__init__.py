"""Repository implementations using PostgreSQL and domain mappers."""

from .profile_repository import PostgresProfileRepository
from .saved_search_repository import PostgresSavedSearchRepository
from .search_click_repository import PostgresSearchClickRepository
from .search_history_repository import PostgresSearchHistoryRepository
from .tenant_repository import PostgresTenantRepository
from .user_repository import PostgresUserRepository

__all__ = [
    "PostgresProfileRepository",
    "PostgresSavedSearchRepository",
    "PostgresSearchClickRepository",
    "PostgresSearchHistoryRepository",
    "PostgresTenantRepository",
    "PostgresUserRepository",
]