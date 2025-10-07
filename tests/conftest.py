"""Pytest fixtures for provider-based architecture."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from app.infrastructure.providers.ai_provider import reset_ai_services
from app.infrastructure.providers.analytics_provider import reset_analytics_service
from app.infrastructure.providers.cache_provider import reset_cache_service
from app.infrastructure.providers.database_provider import reset_database_service
from app.infrastructure.providers.event_provider import reset_event_publisher
from app.infrastructure.providers.message_queue_provider import reset_message_queue
from app.infrastructure.providers.notification_provider import reset_notification_service
from app.infrastructure.providers.postgres_provider import (
    get_postgres_adapter,
    reset_postgres_adapter,
)
from app.infrastructure.providers.search_provider import reset_search_services


@pytest.fixture(autouse=True)
async def reset_provider_state() -> AsyncIterator[None]:
    """Ensure each test starts with clean provider singletons."""
    await reset_ai_services()
    await reset_search_services()
    await reset_cache_service()
    await reset_database_service()
    await reset_message_queue()
    await reset_event_publisher()
    await reset_notification_service()
    await reset_analytics_service()
    await reset_postgres_adapter()
    yield
    # await reset_ai_services()
    await reset_search_services()
    await reset_cache_service()
    await reset_database_service()
    await reset_message_queue()
    await reset_event_publisher()
    await reset_notification_service()
    await reset_analytics_service()
    await reset_postgres_adapter()


@pytest.fixture
async def postgres_adapter():
    """Provide a fresh Postgres adapter for tests."""
    adapter = await get_postgres_adapter()
    try:
        yield adapter
    finally:
        await reset_postgres_adapter()
