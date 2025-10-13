"""
API-specific dependencies for application services with dependency injection.

This module provides FastAPI dependency injection helpers for application services,
bridging the API layer with the hexagonal architecture's application services.
"""

from typing import Annotated

import structlog
from fastapi import Depends, HTTPException

from app.application.click_tracking_service import ClickTrackingApplicationService
from app.application.profile_service import ProfileApplicationService
from app.application.saved_search_service import SavedSearchApplicationService
from app.application.search_history_service import SearchHistoryApplicationService
from app.application.search_service import SearchApplicationService
from app.application.upload_service import UploadApplicationService
from app.domain.exceptions import (
    AuthorizationError,
    ConcurrencyError,
    ConfigurationError,
    DocumentNotFoundError,
    DocumentProcessingError,
    DomainException,
    EmbeddingGenerationError,
    InsufficientPermissionsError,
    NotFoundError,
    ProcessingError,
    ProfileNotFoundError,
    RateLimitExceededError,
    SearchError,
    TenantAccessDeniedError,
    TenantNotFoundError,
    UserNotFoundError,
    ValidationError,
    WebhookValidationError,
)
from app.infrastructure.factories.click_tracking_dependency_factory import (
    get_click_tracking_dependencies,
)
from app.infrastructure.factories.profile_dependency_factory import (
    get_profile_dependencies,
)
from app.infrastructure.factories.saved_search_dependency_factory import (
    get_saved_search_dependencies,
)
from app.infrastructure.factories.search_dependency_factory import (
    get_search_dependencies,
)
from app.infrastructure.factories.search_history_dependency_factory import (
    get_search_history_dependencies,
)
from app.infrastructure.factories.upload_dependency_factory import (
    get_upload_dependencies,
)

logger = structlog.get_logger(__name__)


# Application Service Dependencies
async def get_search_service() -> SearchApplicationService:
    """Create SearchApplicationService with injected dependencies."""
    try:
        dependencies = await get_search_dependencies()
        return SearchApplicationService(dependencies)
    except Exception as e:
        logger.error("Failed to create search service", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Search service unavailable"
        ) from e


async def get_upload_service() -> UploadApplicationService:
    """Create UploadApplicationService with injected dependencies."""
    try:
        dependencies = await get_upload_dependencies()
        return UploadApplicationService(dependencies)
    except Exception as e:
        logger.error("Failed to create upload service", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Upload service unavailable"
        ) from e


async def get_profile_service() -> ProfileApplicationService:
    """Create ProfileApplicationService with shared infrastructure."""
    try:
        dependencies = await get_profile_dependencies()
        return ProfileApplicationService(dependencies)
    except Exception as e:
        logger.error("Failed to create profile service", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Profile service unavailable"
        ) from e


async def get_saved_search_service() -> SavedSearchApplicationService:
    """Create SavedSearchApplicationService with injected dependencies."""
    try:
        dependencies = await get_saved_search_dependencies()
        return SavedSearchApplicationService(dependencies)
    except Exception as e:
        logger.error("Failed to create saved search service", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Saved search service unavailable"
        ) from e


async def get_search_history_service() -> SearchHistoryApplicationService:
    """Create SearchHistoryApplicationService with injected dependencies."""
    try:
        dependencies = await get_search_history_dependencies()
        return SearchHistoryApplicationService(dependencies)
    except Exception as e:
        logger.error("Failed to create search history service", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Search history service unavailable"
        ) from e


async def get_click_tracking_service() -> ClickTrackingApplicationService:
    """Create ClickTrackingApplicationService with injected dependencies."""
    try:
        dependencies = await get_click_tracking_dependencies()
        return ClickTrackingApplicationService(dependencies)
    except Exception as e:
        logger.error("Failed to create click tracking service", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Click tracking service unavailable"
        ) from e


# Type aliases for dependency injection
SearchServiceDep = Annotated[SearchApplicationService, Depends(get_search_service)]
UploadServiceDep = Annotated[UploadApplicationService, Depends(get_upload_service)]
ProfileServiceDep = Annotated[ProfileApplicationService, Depends(get_profile_service)]
SavedSearchServiceDep = Annotated[SavedSearchApplicationService, Depends(get_saved_search_service)]
SearchHistoryServiceDep = Annotated[SearchHistoryApplicationService, Depends(get_search_history_service)]
ClickTrackingServiceDep = Annotated[ClickTrackingApplicationService, Depends(get_click_tracking_service)]


# Domain Exception Handlers
def map_domain_exception_to_http(exception: Exception) -> HTTPException:
    """Map domain exceptions to appropriate HTTP responses."""

    # NotFoundError hierarchy - 404 Not Found
    if isinstance(exception, (DocumentNotFoundError, ProfileNotFoundError, UserNotFoundError, TenantNotFoundError, NotFoundError)):
        return HTTPException(status_code=404, detail=str(exception))

    # ValidationError hierarchy - 400 Bad Request
    elif isinstance(exception, (WebhookValidationError, ValidationError)):
        return HTTPException(status_code=400, detail=str(exception))

    # AuthorizationError hierarchy - 403 Forbidden
    elif isinstance(exception, (TenantAccessDeniedError, InsufficientPermissionsError, AuthorizationError)):
        return HTTPException(status_code=403, detail=str(exception))

    # ProcessingError hierarchy - 422 Unprocessable Entity
    elif isinstance(exception, (DocumentProcessingError, SearchError, EmbeddingGenerationError, ProcessingError)):
        return HTTPException(status_code=422, detail=str(exception))

    # ConfigurationError - 500 Internal Server Error (configuration issues)
    elif isinstance(exception, ConfigurationError):
        logger.error("Configuration error", error=str(exception))
        return HTTPException(status_code=500, detail="Service configuration error")

    # RateLimitExceededError - 429 Too Many Requests
    elif isinstance(exception, RateLimitExceededError):
        return HTTPException(status_code=429, detail=str(exception))

    # ConcurrencyError - 409 Conflict
    elif isinstance(exception, ConcurrencyError):
        return HTTPException(status_code=409, detail=str(exception))

    # Generic DomainException - 500 Internal Server Error
    elif isinstance(exception, DomainException):
        logger.error("Unhandled domain exception", exception_type=type(exception).__name__, error=str(exception))
        return HTTPException(status_code=500, detail="Domain operation failed")

    else:
        # Non-domain exception - log and return generic error
        logger.error("Non-domain exception in mapping", exception_type=type(exception).__name__, error=str(exception))
        return HTTPException(status_code=500, detail="Internal server error")


__all__ = [
    "get_search_service",
    "get_upload_service",
    "get_profile_service",
    "get_saved_search_service",
    "get_search_history_service",
    "get_click_tracking_service",
    "SearchServiceDep",
    "UploadServiceDep",
    "ProfileServiceDep",
    "SavedSearchServiceDep",
    "SearchHistoryServiceDep",
    "ClickTrackingServiceDep",
    "map_domain_exception_to_http"
]
