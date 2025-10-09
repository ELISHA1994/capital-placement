"""Factory for creating ProfileApplicationService dependencies."""

from __future__ import annotations

import structlog

from app.application.dependencies.profile_dependencies import ProfileDependencies
from app.infrastructure.providers.repository_provider import get_profile_repository
from app.infrastructure.providers.usage_provider import get_usage_service
from app.infrastructure.providers.audit_provider import get_audit_service
from app.infrastructure.providers.ai_provider import get_embedding_service

logger = structlog.get_logger(__name__)


async def get_profile_dependencies() -> ProfileDependencies:
    """Construct dependencies for the profile application service.

    Initializes all required and optional services for profile operations:
    - profile_repository: Core profile data access (REQUIRED)
    - usage_service: Usage tracking and metrics (optional)
    - audit_service: Audit logging for compliance (optional)
    - embedding_service: AI embedding generation (optional)
    - search_index_service: Search index management (not yet implemented)
    """
    # Required dependency
    profile_repository = await get_profile_repository()

    # Initialize optional services (failures don't block service creation)
    usage_service = None
    audit_service = None
    embedding_service = None
    search_index_service = None  # TODO: Implement when search index service is available

    try:
        usage_service = await get_usage_service()
        logger.debug("Usage service initialized for profile operations")
    except Exception as e:
        logger.warning(f"Usage service unavailable: {e}")

    try:
        audit_service = await get_audit_service()
        logger.debug("Audit service initialized for profile operations")
    except Exception as e:
        logger.warning(f"Audit service unavailable: {e}")

    try:
        embedding_service = await get_embedding_service()
        logger.debug("Embedding service initialized for profile operations")
    except Exception as e:
        logger.warning(f"Embedding service unavailable: {e}")

    # TODO: Implement search_index_service when available
    # try:
    #     search_index_service = await get_search_index_service()
    #     logger.debug("Search index service initialized for profile operations")
    # except Exception as e:
    #     logger.warning(f"Search index service unavailable: {e}")

    return ProfileDependencies(
        profile_repository=profile_repository,
        usage_service=usage_service,
        audit_service=audit_service,
        embedding_service=embedding_service,
        search_index_service=search_index_service,
    )


__all__ = ["get_profile_dependencies"]
