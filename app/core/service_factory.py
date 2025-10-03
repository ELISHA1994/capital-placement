"""
Service Factory - Creates cloud-agnostic services using local implementations and OpenAI
"""

import os
from typing import Dict, Any, Type, Optional
import structlog

from app.core.environment import get_service_strategy, ServiceStrategy, get_current_environment, Environment
from app.domain.interfaces import (
    ICacheService, IDocumentStore, IDatabase, ISearchService, 
    IAIService, IMessageQueue, INotificationService, IAnalyticsService
)

logger = structlog.get_logger(__name__)


class ServiceFactory:
    """Factory for creating environment-appropriate services"""
    
    def __init__(self):
        self.strategy = get_service_strategy()
        self.environment = get_current_environment()
        self._service_cache: Dict[Type, Any] = {}
    
    async def create_cache_service(self) -> ICacheService:
        """Create cache service (Redis or in-memory)"""
        if ICacheService in self._service_cache:
            return self._service_cache[ICacheService]
        
        if self.strategy.should_use_redis():
            service = await self._create_redis_cache()
        else:
            service = await self._create_memory_cache()
        
        self._service_cache[ICacheService] = service
        return service
    
    async def create_document_store(self) -> IDocumentStore:
        """Create document storage service"""
        if IDocumentStore in self._service_cache:
            return self._service_cache[IDocumentStore]
        
        # Always use local file storage (cloud-agnostic)
        service = await self._create_local_file_storage()
        
        self._service_cache[IDocumentStore] = service
        return service
    
    async def create_database(self) -> IDatabase:
        """Create database service"""
        if IDatabase in self._service_cache:
            return self._service_cache[IDatabase]
        
        # Always use PostgreSQL database (cloud-agnostic)
        service = await self._create_local_database()
        
        self._service_cache[IDatabase] = service
        return service
    
    async def create_search_service(self) -> ISearchService:
        """Create search service"""
        if ISearchService in self._service_cache:
            return self._service_cache[ISearchService]
        
        # Always use local search service with PostgreSQL pgvector
        service = await self._create_local_search()
        
        self._service_cache[ISearchService] = service
        return service
    
    async def create_ai_service(self) -> IAIService:
        """Create AI service"""
        if IAIService in self._service_cache:
            return self._service_cache[IAIService]
        
        # Use cloud-agnostic OpenAI service
        service = await self._create_openai_service()
        
        self._service_cache[IAIService] = service
        return service
    
    async def create_message_queue(self) -> IMessageQueue:
        """Create message queue service"""
        if IMessageQueue in self._service_cache:
            return self._service_cache[IMessageQueue]
        
        # Always use local queue service (cloud-agnostic)
        service = await self._create_local_queue()
        
        self._service_cache[IMessageQueue] = service
        return service
    
    async def create_notification_service(self) -> INotificationService:
        """Create notification service"""
        if INotificationService in self._service_cache:
            return self._service_cache[INotificationService]
        
        # Always use local notification service (cloud-agnostic)
        service = await self._create_local_notification()
        
        self._service_cache[INotificationService] = service
        return service
    
    async def create_analytics_service(self) -> IAnalyticsService:
        """Create analytics service"""
        if IAnalyticsService in self._service_cache:
            return self._service_cache[IAnalyticsService]
        
        # Always use local analytics service (cloud-agnostic)
        service = await self._create_local_analytics()
        
        self._service_cache[IAnalyticsService] = service
        return service
    
    # Redis cache service creation
    async def _create_redis_cache(self) -> ICacheService:
        """Create Redis cache service"""
        try:
            from app.services.adapters.redis_cache_adapter import RedisCacheService
            
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            return await RedisCacheService.create(redis_url)
        except ImportError:
            logger.warning("Redis not available, falling back to memory cache")
            return await self._create_memory_cache()
    
    # Cloud-agnostic OpenAI service creation
    async def _create_openai_service(self) -> IAIService:
        """Create cloud-agnostic OpenAI service"""
        try:
            from app.services.ai.openai_service import OpenAIService
            return await OpenAIService.create()
        except ImportError:
            logger.error("OpenAI SDK not available. Install with: pip install openai")
            raise
    
    # Local service creation methods
    async def _create_memory_cache(self) -> ICacheService:
        """Create in-memory cache service"""
        from app.services.adapters.memory_cache_adapter import MemoryCacheService
        return MemoryCacheService()
    
    async def _create_local_file_storage(self) -> IDocumentStore:
        """Create local file storage service"""
        from app.services.adapters.storage_adapters import FileSystemBlobStorage
        
        storage_path = os.getenv("LOCAL_STORAGE_PATH", "./data/documents")
        return FileSystemBlobStorage(storage_path)
    
    async def _create_local_database(self) -> IDatabase:
        """Create local database service with SQLModel"""
        # Use SQLModel with PostgreSQL for all environments (cloud-agnostic)
        from app.database.sqlmodel_engine import get_sqlmodel_db_manager
        
        return get_sqlmodel_db_manager()
    
    async def _create_local_search(self) -> ISearchService:
        """Create local search service with SQLModel repositories"""
        from app.services.search.vector_search import VectorSearchService
        from app.services.ai.openai_service import OpenAIService
        from app.services.ai.embedding_service import EmbeddingService
        from app.services.adapters.postgres_adapter import PostgresAdapter

        cache_service = await self.create_cache_service()
        openai_service = await OpenAIService.create(cache_service=cache_service)
        postgres_adapter = PostgresAdapter()

        embedding_service = EmbeddingService(
            openai_service=openai_service,
            db_adapter=postgres_adapter,
            cache_service=cache_service,
        )

        return VectorSearchService(
            db_adapter=postgres_adapter,
            embedding_service=embedding_service,
            cache_manager=None,
        )
    
    
    async def _create_local_queue(self) -> IMessageQueue:
        """Create local message queue service"""
        from app.services.adapters.messaging_adapters import InMemoryMessageQueue
        return InMemoryMessageQueue()
    
    async def _create_local_notification(self) -> INotificationService:
        """Create local notification service"""
        from app.services.adapters.notification_adapter import LocalNotificationService
        return LocalNotificationService()
    
    async def _create_local_analytics(self) -> IAnalyticsService:
        """Create local analytics service"""
        from app.services.adapters.postgres_adapter import PostgresAdapter
        from app.services.search.search_analytics import SearchAnalyticsService

        postgres_adapter = PostgresAdapter()
        notification_service = await self.create_notification_service()

        return SearchAnalyticsService(
            db_adapter=postgres_adapter,
            notification_service=notification_service,
        )
    
    def clear_cache(self):
        """Clear service cache"""
        self._service_cache.clear()


# Singleton factory instance
_factory: Optional[ServiceFactory] = None


def get_service_factory() -> ServiceFactory:
    """Get singleton service factory"""
    global _factory
    if _factory is None:
        _factory = ServiceFactory()
    return _factory


async def create_all_services() -> Dict[Type, Any]:
    """Create all services and return them as a dictionary"""
    factory = get_service_factory()
    
    services = {
        ICacheService: await factory.create_cache_service(),
        IDocumentStore: await factory.create_document_store(),
        IDatabase: await factory.create_database(),
        ISearchService: await factory.create_search_service(),
        IAIService: await factory.create_ai_service(),
        IMessageQueue: await factory.create_message_queue(),
        INotificationService: await factory.create_notification_service(),
        IAnalyticsService: await factory.create_analytics_service(),
    }
    
    logger.info("All services created successfully", 
                service_count=len(services),
                environment=get_current_environment().value)
    
    return services
