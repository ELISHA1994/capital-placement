"""Providers for search-related services."""

from __future__ import annotations

import asyncio
from typing import Optional

from app.infrastructure.search.vector_search import VectorSearchService
from app.infrastructure.search.query_processor import QueryProcessor
from app.infrastructure.search.search_analytics import SearchAnalyticsService
from app.application.search.hybrid_search import HybridSearchService
from app.application.search.result_reranker import ResultRerankerService

from app.infrastructure.providers.ai_provider import (
    get_embedding_service,
    get_openai_service,
    get_prompt_manager,
    get_semantic_cache_manager,
)
from app.infrastructure.providers.postgres_provider import get_postgres_adapter
from app.infrastructure.providers.notification_provider import get_notification_service

_vector_search_service: Optional[VectorSearchService] = None
_hybrid_search_service: Optional[HybridSearchService] = None
_reranker_service: Optional[ResultRerankerService] = None
_query_processor: Optional[QueryProcessor] = None
_analytics_service: Optional[SearchAnalyticsService] = None

_vector_lock = asyncio.Lock()
_hybrid_lock = asyncio.Lock()
_reranker_lock = asyncio.Lock()
_query_lock = asyncio.Lock()
_analytics_lock = asyncio.Lock()


async def get_vector_search_service() -> VectorSearchService:
    """Return singleton vector search service."""
    global _vector_search_service

    if _vector_search_service is not None:
        return _vector_search_service

    async with _vector_lock:
        if _vector_search_service is not None:
            return _vector_search_service

        postgres_adapter = await get_postgres_adapter()
        embedding_service = await get_embedding_service()
        cache_manager = await get_semantic_cache_manager()

        _vector_search_service = VectorSearchService(
            db_adapter=postgres_adapter,
            embedding_service=embedding_service,
            cache_manager=cache_manager,
        )
        return _vector_search_service


async def get_query_processor() -> QueryProcessor:
    """Return singleton query processor service."""
    global _query_processor

    if _query_processor is not None:
        return _query_processor

    async with _query_lock:
        if _query_processor is not None:
            return _query_processor

        openai_service = await get_openai_service()
        prompt_manager = await get_prompt_manager()
        cache_manager = await get_semantic_cache_manager()
        postgres_adapter = await get_postgres_adapter()

        _query_processor = QueryProcessor(
            openai_service=openai_service,
            prompt_manager=prompt_manager,
            cache_manager=cache_manager,
            db_adapter=postgres_adapter,
        )
        return _query_processor


async def get_hybrid_search_service() -> HybridSearchService:
    """Return singleton hybrid search service."""
    global _hybrid_search_service

    if _hybrid_search_service is not None:
        return _hybrid_search_service

    async with _hybrid_lock:
        if _hybrid_search_service is not None:
            return _hybrid_search_service

        postgres_adapter = await get_postgres_adapter()
        vector_search_service = await get_vector_search_service()
        query_processor = await get_query_processor()
        cache_manager = await get_semantic_cache_manager()

        _hybrid_search_service = HybridSearchService(
            db_adapter=postgres_adapter,
            vector_search_service=vector_search_service,
            query_processor=query_processor,
            cache_manager=cache_manager,
        )
        return _hybrid_search_service


async def get_result_reranker_service() -> ResultRerankerService:
    """Return singleton result reranker service."""
    global _reranker_service

    if _reranker_service is not None:
        return _reranker_service

    async with _reranker_lock:
        if _reranker_service is not None:
            return _reranker_service

        openai_service = await get_openai_service()
        prompt_manager = await get_prompt_manager()
        cache_manager = await get_semantic_cache_manager()
        postgres_adapter = await get_postgres_adapter()

        _reranker_service = ResultRerankerService(
            openai_service=openai_service,
            prompt_manager=prompt_manager,
            db_adapter=postgres_adapter,
            cache_manager=cache_manager,
        )
        return _reranker_service


async def get_search_analytics_service() -> SearchAnalyticsService:
    """Return singleton search analytics service."""
    global _analytics_service

    if _analytics_service is not None:
        return _analytics_service

    async with _analytics_lock:
        if _analytics_service is not None:
            return _analytics_service

        postgres_adapter = await get_postgres_adapter()
        notification_service = await get_notification_service()

        _analytics_service = SearchAnalyticsService(
            db_adapter=postgres_adapter,
            notification_service=notification_service,
        )
        return _analytics_service


async def reset_search_services() -> None:
    """Reset cached search-related services (used in tests)."""
    global _vector_search_service, _query_processor, _hybrid_search_service, _reranker_service, _analytics_service

    async with _vector_lock:
        _vector_search_service = None
    async with _query_lock:
        _query_processor = None
    async with _hybrid_lock:
        _hybrid_search_service = None
    async with _reranker_lock:
        _reranker_service = None
    async with _analytics_lock:
        _analytics_service = None


__all__ = [
    "get_vector_search_service",
    "get_query_processor",
    "get_hybrid_search_service",
    "get_result_reranker_service",
    "get_search_analytics_service",
    "reset_search_services",
]

