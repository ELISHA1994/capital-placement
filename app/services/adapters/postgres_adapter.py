"""Simple Postgres adapter using SQLModel database manager."""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional

from sqlalchemy.ext.asyncio import AsyncConnection

from app.database.sqlmodel_engine import get_sqlmodel_db_manager, SQLModelDatabaseManager


class _AsyncConnectionWrapper:
    def __init__(self, connection: AsyncConnection):
        self._connection = connection

    async def fetchrow(self, query: str, *params: Any) -> Optional[Mapping[str, Any]]:
        result = await self._connection.exec_driver_sql(query, params)
        row = result.mappings().first()
        return dict(row) if row else None

    async def fetch(self, query: str, *params: Any) -> List[Mapping[str, Any]]:
        result = await self._connection.exec_driver_sql(query, params)
        rows = result.mappings().all()
        return [dict(row) for row in rows]

    async def execute(self, query: str, *params: Any) -> None:
        await self._connection.exec_driver_sql(query, params)


class _ConnectionContext:
    def __init__(self, db_manager: SQLModelDatabaseManager):
        self.db_manager = db_manager
        self._connection: Optional[AsyncConnection] = None

    async def __aenter__(self) -> _AsyncConnectionWrapper:
        self._connection = await self.db_manager.engine.connect()
        return _AsyncConnectionWrapper(self._connection)

    async def __aexit__(self, exc_type, exc, tb):
        if self._connection:
            await self._connection.close()
            self._connection = None


class PostgresAdapter:
    """Lightweight adapter exposing fetch/execute helpers."""

    def __init__(self, db_manager: Optional[SQLModelDatabaseManager] = None):
        self.db_manager = db_manager or get_sqlmodel_db_manager()

    async def fetch_all(self, query: str, *params: Any) -> List[Mapping[str, Any]]:
        async with self.db_manager.engine.connect() as connection:
            result = await connection.exec_driver_sql(query, params)
            rows = result.mappings().all()
            return [dict(row) for row in rows]

    async def execute(self, query: str, *params: Any) -> None:
        async with self.db_manager.engine.begin() as connection:
            await connection.exec_driver_sql(query, params)

    async def fetch_one(self, query: str, *params: Any) -> Optional[Mapping[str, Any]]:
        async with self.db_manager.engine.connect() as connection:
            result = await connection.exec_driver_sql(query, params)
            row = result.mappings().first()
            return dict(row) if row else None

    def get_connection(self) -> _ConnectionContext:
        """Provide context manager compatible with repository interface."""
        return _ConnectionContext(self.db_manager)


async def get_postgres_adapter() -> PostgresAdapter:
    return PostgresAdapter()
