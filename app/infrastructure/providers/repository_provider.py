"""Repository provider utilities."""

from __future__ import annotations

import asyncio

from sqlmodel import SQLModel

from app.database.repositories.postgres import SQLModelRepository, VectorRepository
from app.database.sqlmodel_engine import get_sqlmodel_db_manager
from app.domain.repositories import IProfileRepository
from app.domain.repositories.saved_search_repository import ISavedSearchRepository
from app.domain.repositories.search_click_repository import ISearchClickRepository
from app.domain.repositories.search_history_repository import ISearchHistoryRepository
from app.domain.repositories.tenant_repository import ITenantRepository
from app.domain.repositories.user_repository import IUserRepository
from app.infrastructure.persistence.repositories import (
    PostgresProfileRepository,
    PostgresSavedSearchRepository,
    PostgresSearchClickRepository,
    PostgresSearchHistoryRepository,
    PostgresTenantRepository,
    PostgresUserRepository,
)

_sqlmodel_repositories: dict[type[SQLModel], SQLModelRepository] = {}
_vector_repositories: dict[type[SQLModel], VectorRepository] = {}
_profile_repository: IProfileRepository | None = None
_tenant_repository: ITenantRepository | None = None
_user_repository: IUserRepository | None = None
_saved_search_repository: ISavedSearchRepository | None = None
_search_history_repository: ISearchHistoryRepository | None = None
_search_click_repository: ISearchClickRepository | None = None

_sql_lock = asyncio.Lock()
_vector_lock = asyncio.Lock()
_profile_lock = asyncio.Lock()
_tenant_lock = asyncio.Lock()
_user_lock = asyncio.Lock()
_saved_search_lock = asyncio.Lock()
_search_history_lock = asyncio.Lock()
_search_click_lock = asyncio.Lock()


async def get_sqlmodel_repository(model_cls: type[SQLModel]) -> SQLModelRepository:
    """Return SQLModel repository for the given model class."""
    if model_cls in _sqlmodel_repositories:
        return _sqlmodel_repositories[model_cls]

    async with _sql_lock:
        if model_cls in _sqlmodel_repositories:
            return _sqlmodel_repositories[model_cls]

        db_manager = get_sqlmodel_db_manager()
        repository = SQLModelRepository(model_cls, db_manager)
        _sqlmodel_repositories[model_cls] = repository
        return repository


async def get_vector_repository(model_cls: type[SQLModel]) -> VectorRepository:
    """Return VectorRepository for the given embedding-capable model class."""
    if model_cls in _vector_repositories:
        return _vector_repositories[model_cls]

    async with _vector_lock:
        if model_cls in _vector_repositories:
            return _vector_repositories[model_cls]

        db_manager = get_sqlmodel_db_manager()
        repository = VectorRepository(model_cls, db_manager)
        _vector_repositories[model_cls] = repository
        return repository


async def get_profile_repository() -> IProfileRepository:
    """Return singleton profile repository adapter satisfying the domain interface."""
    global _profile_repository
    if _profile_repository is not None:
        return _profile_repository

    async with _profile_lock:
        if _profile_repository is not None:
            return _profile_repository

        _profile_repository = PostgresProfileRepository()
        return _profile_repository


async def get_tenant_repository() -> ITenantRepository:
    """Return singleton tenant repository implementation."""
    global _tenant_repository
    if _tenant_repository is not None:
        return _tenant_repository

    async with _tenant_lock:
        if _tenant_repository is not None:
            return _tenant_repository

        _tenant_repository = PostgresTenantRepository()
        return _tenant_repository


async def get_user_repository() -> IUserRepository:
    """Return singleton user repository implementation."""
    global _user_repository
    if _user_repository is not None:
        return _user_repository

    async with _user_lock:
        if _user_repository is not None:
            return _user_repository

        _user_repository = PostgresUserRepository()
        return _user_repository


async def get_saved_search_repository() -> ISavedSearchRepository:
    """Return singleton saved search repository implementation."""
    global _saved_search_repository
    if _saved_search_repository is not None:
        return _saved_search_repository

    async with _saved_search_lock:
        if _saved_search_repository is not None:
            return _saved_search_repository

        _saved_search_repository = PostgresSavedSearchRepository()
        return _saved_search_repository


async def get_search_history_repository() -> ISearchHistoryRepository:
    """Return singleton search history repository implementation."""
    global _search_history_repository
    if _search_history_repository is not None:
        return _search_history_repository

    async with _search_history_lock:
        if _search_history_repository is not None:
            return _search_history_repository

        _search_history_repository = PostgresSearchHistoryRepository()
        return _search_history_repository


async def get_search_click_repository() -> ISearchClickRepository:
    """Return singleton search click repository implementation."""
    global _search_click_repository
    if _search_click_repository is not None:
        return _search_click_repository

    async with _search_click_lock:
        if _search_click_repository is not None:
            return _search_click_repository

        _search_click_repository = PostgresSearchClickRepository()
        return _search_click_repository


__all__ = [
    "get_sqlmodel_repository",
    "get_vector_repository",
    "get_profile_repository",
    "get_tenant_repository",
    "get_user_repository",
    "get_saved_search_repository",
    "get_search_history_repository",
    "get_search_click_repository",
]
