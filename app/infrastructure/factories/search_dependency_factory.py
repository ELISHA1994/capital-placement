"""Concrete factory for creating SearchApplicationService dependencies."""

from __future__ import annotations

from app.application.dependencies import SearchDependencies, ISearchDependencyFactory
from app.infrastructure.persistence.repositories import (
    PostgresProfileRepository,
    PostgresUserRepository,
    PostgresTenantRepository
)
from app.infrastructure.providers.search_provider import (
    get_hybrid_search_service,
    get_result_reranker_service,
    get_search_analytics_service,
)
from app.infrastructure.providers.cache_provider import get_cache_service
from app.infrastructure.providers.tenant_provider import get_tenant_service as get_tenant_manager
from app.infrastructure.adapters.event_publisher_adapter import EventPublisherAdapter


class SearchDependencyFactory(ISearchDependencyFactory):
    """Concrete factory for creating search dependencies using current providers."""

    async def create_dependencies(self) -> SearchDependencies:
        """Create and return search dependencies."""
        
        # Repository implementations
        profile_repository = PostgresProfileRepository()
        user_repository = PostgresUserRepository()
        tenant_repository = PostgresTenantRepository()
        
        # Service implementations (using existing providers)
        search_service = await get_hybrid_search_service()
        reranker_service = await get_result_reranker_service()
        analytics_service = await get_search_analytics_service()
        tenant_manager = await get_tenant_manager()
        cache_service = await get_cache_service()
        
        # Event publisher
        event_publisher = EventPublisherAdapter()
        
        return SearchDependencies(
            # Repositories
            profile_repository=profile_repository,
            user_repository=user_repository,
            tenant_repository=tenant_repository,
            
            # Services
            search_service=search_service,
            reranker_service=reranker_service,
            analytics_service=analytics_service,
            tenant_manager=tenant_manager,
            cache_service=cache_service,
            event_publisher=event_publisher
        )


# Singleton instance for global usage
_search_dependency_factory: SearchDependencyFactory | None = None


async def get_search_dependency_factory() -> SearchDependencyFactory:
    """Get singleton instance of search dependency factory."""
    global _search_dependency_factory
    if _search_dependency_factory is None:
        _search_dependency_factory = SearchDependencyFactory()
    return _search_dependency_factory


async def get_search_dependencies() -> SearchDependencies:
    """Helper function to get search dependencies directly."""
    factory = await get_search_dependency_factory()
    return await factory.create_dependencies()


__all__ = ["SearchDependencyFactory", "get_search_dependency_factory", "get_search_dependencies"]