"""
Suggestion Provider

Provider for SuggestionApplicationService with dependency injection.
"""

from pathlib import Path
from typing import Optional
import structlog

from app.application.suggestion_service import SuggestionApplicationService
from app.infrastructure.persistence.repositories.popular_search_repository import PopularSearchRepository
from app.infrastructure.persistence.repositories.user_history_repository import UserHistoryRepository
from app.infrastructure.persistence.repositories.dictionary_repository import DictionaryRepository
from app.infrastructure.adapters.suggestion_cache_adapter import SuggestionCacheAdapter
from app.infrastructure.providers.database_provider import get_postgres_adapter
from app.infrastructure.providers.cache_provider import get_redis_cache

logger = structlog.get_logger(__name__)

_suggestion_service: Optional[SuggestionApplicationService] = None


async def get_suggestion_service() -> SuggestionApplicationService:
    """
    Provide initialized SuggestionApplicationService.

    Singleton pattern - initializes once and reuses.
    Initializes all repositories and cache adapters.

    Returns:
        Configured SuggestionApplicationService instance
    """
    global _suggestion_service

    if _suggestion_service is None:
        try:
            # Get dependencies
            db_adapter = await get_postgres_adapter()
            redis_cache = await get_redis_cache()

            # Initialize repositories
            popular_search_repo = PopularSearchRepository(db_adapter)
            user_history_repo = UserHistoryRepository(db_adapter)

            # Dictionary repository with initialization
            dict_path = Path(__file__).parent.parent / "data" / "dictionaries"
            dictionary_repo = DictionaryRepository(dict_path)
            await dictionary_repo.initialize()

            # Cache adapter
            cache_adapter = SuggestionCacheAdapter(redis_cache)

            # Create service
            _suggestion_service = SuggestionApplicationService(
                popular_search_repo=popular_search_repo,
                user_history_repo=user_history_repo,
                dictionary_repo=dictionary_repo,
                cache=cache_adapter
            )

            logger.info("Suggestion service initialized")

        except Exception as e:
            logger.error("Failed to initialize suggestion service", error=str(e))
            raise

    return _suggestion_service
