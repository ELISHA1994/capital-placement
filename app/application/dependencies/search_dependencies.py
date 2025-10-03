"""Dependencies interface for SearchApplicationService."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from app.domain.interfaces import (
    ICacheService,
    IEventPublisher,
    ISearchAnalyticsService,
)
from app.domain.repositories.profile_repository import IProfileRepository
from app.domain.repositories.user_repository import IUserRepository
from app.domain.repositories.tenant_repository import ITenantRepository


@runtime_checkable
class IHybridSearchService(Protocol):
    """Interface for hybrid search services."""
    
    async def hybrid_search(
        self,
        query: str,
        tenant_id: Optional[str] = None,
        limit: int = 20,
        search_mode: Any = None,
        config: Any = None,
        search_filter: Any = None,
        use_cache: bool = True,
        include_explanations: bool = False,
    ) -> Any:
        """Perform hybrid search."""
        ...


@runtime_checkable
class IResultRerankerService(Protocol):
    """Interface for result reranking services."""
    
    async def rerank_results(
        self,
        query: str,
        results: List[Any],
        tenant_id: Optional[str] = None,
        config: Any = None,
    ) -> Any:
        """Rerank search results."""
        ...


@runtime_checkable
class ITenantManagerService(Protocol):
    """Interface for tenant management services."""
    
    async def get_tenant_configuration(self, tenant_id: str) -> Dict[str, Any]:
        """Get tenant configuration."""
        ...
    
    async def update_usage_metrics(self, tenant_id: str, metrics_update: Dict[str, Any]) -> None:
        """Update tenant usage metrics."""
        ...


@dataclass
class SearchDependencies:
    """Dependencies required by SearchApplicationService."""
    
    # Repositories (domain layer)
    profile_repository: IProfileRepository
    user_repository: IUserRepository
    tenant_repository: ITenantRepository
    
    # Services (infrastructure layer)
    search_service: IHybridSearchService
    reranker_service: Optional[IResultRerankerService]
    analytics_service: Optional[ISearchAnalyticsService]
    tenant_manager: ITenantManagerService
    cache_service: ICacheService
    event_publisher: IEventPublisher


class ISearchDependencyFactory(ABC):
    """Abstract factory for creating search dependencies."""
    
    @abstractmethod
    async def create_dependencies(self) -> SearchDependencies:
        """Create and return search dependencies."""
        pass