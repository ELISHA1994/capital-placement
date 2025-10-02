"""Document store provider built on the service factory."""

from __future__ import annotations

import asyncio
from typing import Optional

from app.core.interfaces import IDocumentStore
from app.core.service_factory import get_service_factory

_document_store: Optional[IDocumentStore] = None
_lock = asyncio.Lock()


async def get_document_store() -> IDocumentStore:
    global _document_store

    if _document_store is not None:
        return _document_store

    async with _lock:
        if _document_store is not None:
            return _document_store

        factory = get_service_factory()
        _document_store = await factory.create_document_store()
        return _document_store


async def reset_document_store() -> None:
    global _document_store
    async with _lock:
        _document_store = None
