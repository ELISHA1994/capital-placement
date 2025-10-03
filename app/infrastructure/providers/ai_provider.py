"""Provider utilities for AI-related services."""

from __future__ import annotations

import asyncio
from typing import Optional

from app.services.ai.openai_service import OpenAIService
from app.services.ai.embedding_service import EmbeddingService
from app.services.ai.prompt_manager import PromptManager
from app.services.ai.cache_manager import CacheManager

from app.infrastructure.providers.cache_provider import get_cache_service
from app.infrastructure.providers.postgres_provider import get_postgres_adapter

_openai_service: Optional[OpenAIService] = None
_embedding_service: Optional[EmbeddingService] = None
_prompt_manager: Optional[PromptManager] = None
_semantic_cache_manager: Optional[CacheManager] = None

_openai_lock = asyncio.Lock()
_embedding_lock = asyncio.Lock()
_prompt_lock = asyncio.Lock()
_cache_lock = asyncio.Lock()


async def get_openai_service() -> OpenAIService:
    """Return singleton OpenAI service configured via service factory."""
    global _openai_service

    if _openai_service is not None:
        return _openai_service

    async with _openai_lock:
        if _openai_service is not None:
            return _openai_service

        cache_service = await get_cache_service()
        _openai_service = await OpenAIService.create(cache_service=cache_service)
        return _openai_service


async def get_embedding_service() -> EmbeddingService:
    """Return embedding service backed by the shared Postgres adapter."""
    global _embedding_service

    if _embedding_service is not None:
        return _embedding_service

    async with _embedding_lock:
        if _embedding_service is not None:
            return _embedding_service

        postgres_adapter = await get_postgres_adapter()
        openai_service = await get_openai_service()
        cache_service = await get_cache_service()
        _embedding_service = EmbeddingService(
            openai_service=openai_service,
            db_adapter=postgres_adapter,
            cache_service=cache_service,
        )
        return _embedding_service


async def get_prompt_manager() -> PromptManager:
    """Return prompt manager instance."""
    global _prompt_manager

    if _prompt_manager is not None:
        return _prompt_manager

    async with _prompt_lock:
        if _prompt_manager is not None:
            return _prompt_manager

        cache_service = await get_cache_service()
        _prompt_manager = PromptManager(cache_service=cache_service)
        return _prompt_manager


async def get_semantic_cache_manager() -> CacheManager:
    """Return semantic cache manager used by AI services."""
    global _semantic_cache_manager

    if _semantic_cache_manager is not None:
        return _semantic_cache_manager

    async with _cache_lock:
        if _semantic_cache_manager is not None:
            return _semantic_cache_manager

        redis_client = None
        cache_service = await get_cache_service()
        if hasattr(cache_service, "redis"):
            redis_client = getattr(cache_service, "redis")

        embedding_service = await get_embedding_service()
        _semantic_cache_manager = CacheManager(
            redis_client=redis_client,
            embedding_service=embedding_service,
        )
        return _semantic_cache_manager


async def reset_ai_services() -> None:
    """Reset cached AI services (useful for tests)."""
    global _openai_service, _embedding_service, _prompt_manager, _semantic_cache_manager
    async with _openai_lock:
        _openai_service = None
    async with _embedding_lock:
        _embedding_service = None
    async with _prompt_lock:
        _prompt_manager = None
    async with _cache_lock:
        _semantic_cache_manager = None


__all__ = [
    "get_openai_service",
    "get_embedding_service",
    "get_prompt_manager",
    "get_semantic_cache_manager",
    "reset_ai_services",
]

