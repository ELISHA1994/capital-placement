"""Infrastructure adapters for external services."""

from .event_publisher_adapter import EventPublisherAdapter, get_event_publisher
from .storage_adapter import LocalFileStorageAdapter
from .notification_adapter import NotificationAdapter, LocalNotificationService
from .memory_cache_adapter import MemoryCacheService
from .redis_cache_adapter import RedisCacheService
from .messaging_adapters import InMemoryMessageQueue, RedisMessageQueue, LocalEventPublisher
from .postgres_adapter import PostgresAdapter, get_postgres_adapter
from .profile_repository_adapter import ProfileRepositoryAdapter
from .document_processor_adapter import DocumentProcessorAdapter
from .secrets_adapters import LocalSecretManager, EnvironmentSecretManager
from .tenant_database_adapter import TenantConfigDatabaseAdapter

__all__ = [
    # Events & Messaging
    "EventPublisherAdapter",
    "get_event_publisher",
    "InMemoryMessageQueue",
    "RedisMessageQueue",
    "LocalEventPublisher",

    # Storage
    "LocalFileStorageAdapter",

    # Notifications
    "NotificationAdapter",
    "LocalNotificationService",

    # Cache
    "MemoryCacheService",
    "RedisCacheService",

    # Database
    "PostgresAdapter",
    "get_postgres_adapter",
    "TenantConfigDatabaseAdapter",

    # Repositories
    "ProfileRepositoryAdapter",
    "DocumentProcessorAdapter",

    # Secrets
    "LocalSecretManager",
    "EnvironmentSecretManager",
]
