"""Repository provider utilities."""

from __future__ import annotations

import asyncio
from typing import Dict, Optional, Type

from sqlmodel import SQLModel

from app.database.repositories.postgres import SQLModelRepository, VectorRepository
from app.database.sqlmodel_engine import get_sqlmodel_db_manager
from app.domain.repositories import IProfileRepository
from app.services.adapters import ProfileRepositoryAdapter

_sqlmodel_repositories: Dict[Type[SQLModel], SQLModelRepository] = {}
_vector_repositories: Dict[Type[SQLModel], VectorRepository] = {}
_profile_repository: Optional[IProfileRepository] = None

_sql_lock = asyncio.Lock()
_vector_lock = asyncio.Lock()
_profile_lock = asyncio.Lock()


async def get_sqlmodel_repository(model_cls: Type[SQLModel]) -> SQLModelRepository:
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


async def get_vector_repository(model_cls: Type[SQLModel]) -> VectorRepository:
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

        _profile_repository = ProfileRepositoryAdapter()
        return _profile_repository


__all__ = ["get_sqlmodel_repository", "get_vector_repository", "get_profile_repository"]
