"""Infrastructure provider accessors package."""

from .ai_provider import (  # noqa: F401
    get_openai_service,
    get_embedding_service,
    get_prompt_manager,
    get_semantic_cache_manager,
    reset_ai_services,
)
from .analytics_provider import (  # noqa: F401
    get_analytics_service,
    reset_analytics_service,
)
from .cache_provider import get_cache_service, reset_cache_service  # noqa: F401
from .database_provider import (  # noqa: F401
    get_database_service,
    reset_database_service,
)
from .message_queue_provider import (  # noqa: F401
    get_message_queue,
    reset_message_queue,
)
from .event_provider import get_event_publisher, reset_event_publisher  # noqa: F401
from .notification_provider import (  # noqa: F401
    get_notification_service,
    reset_notification_service,
)
from .postgres_provider import get_postgres_adapter, reset_postgres_adapter  # noqa: F401
from .repository_provider import (  # noqa: F401
    get_sqlmodel_repository,
    get_vector_repository,
    get_profile_repository,
)
from .search_provider import (  # noqa: F401
    get_vector_search_service,
    get_query_processor,
    get_hybrid_search_service,
    get_result_reranker_service,
    get_search_analytics_service,
    reset_search_services,
)
from .storage_provider import (  # noqa: F401
    get_file_storage,
    shutdown_file_storage,
)

__all__ = [name for name in globals() if name.startswith("get_") or name.startswith("reset_") or name.startswith("shutdown_")]
