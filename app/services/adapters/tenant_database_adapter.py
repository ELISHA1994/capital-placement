"""Tenant configuration database adapter implementing IDatabase interface."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

import structlog

from app.core.interfaces import IDatabase
from app.database.repositories.postgres import TenantRepository

logger = structlog.get_logger(__name__)


class TenantConfigDatabaseAdapter(IDatabase):
    """Adapt SQLModel tenant repository to the IDatabase interface."""

    CONTAINER_NAME = "tenant-config"

    def __init__(self, tenant_repository: TenantRepository):
        self.tenant_repository = tenant_repository

    async def check_health(self) -> Dict[str, Any]:
        return {"status": "healthy", "service": self.__class__.__name__}

    async def create_item(self, container: str, item: Dict[str, Any]) -> Dict[str, Any]:
        self._validate_container(container)
        if "id" not in item and item.get("tenant_id"):
            item["id"] = item["tenant_id"]
        return await self.tenant_repository.create(item)

    async def get_item(
        self,
        container: str,
        item_id: str,
        partition_key: str
    ) -> Optional[Dict[str, Any]]:
        self._validate_container(container)
        return await self.tenant_repository.get(item_id)

    async def update_item(
        self,
        container: str,
        item_id: str,
        item: Dict[str, Any]
    ) -> Dict[str, Any]:
        self._validate_container(container)
        if item.get("id") is None:
            item["id"] = item_id
        return await self.tenant_repository.update(item_id, item)

    async def delete_item(
        self,
        container: str,
        item_id: str,
        partition_key: str
    ) -> bool:
        self._validate_container(container)
        return await self.tenant_repository.delete(item_id)

    async def query_items(
        self,
        container: str,
        query: str,
        parameters: Optional[List[Dict[str, Any]]] = None,
        partition_key: str = None
    ) -> List[Dict[str, Any]]:
        self._validate_container(container)
        # Simple passthrough using criteria extracted from parameters when provided.
        criteria = {}
        if parameters:
            for param in parameters:
                name = param.get("name")
                value = param.get("value")
                if name and value is not None:
                    criteria[name] = value

        if not criteria and partition_key:
            criteria["tenant_id"] = partition_key

        return await self.tenant_repository.find_by_criteria(criteria)

    def _validate_container(self, container: str) -> None:
        if container != self.CONTAINER_NAME:
            raise ValueError(f"Unsupported container: {container}")
