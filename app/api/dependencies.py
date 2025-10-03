"""
API-specific dependencies for application services with dependency injection.

This module provides FastAPI dependency injection helpers for application services,
bridging the API layer with the hexagonal architecture's application services.
"""

from typing import Annotated

from fastapi import Depends, HTTPException
import structlog

from app.application.search_service import SearchApplicationService
from app.application.upload_service import UploadApplicationService
from app.infrastructure.factories.search_dependency_factory import get_search_dependencies
from app.infrastructure.factories.upload_dependency_factory import get_upload_dependencies
from app.domain.exceptions import (
    ProfileNotFoundError, 
    InsufficientPermissionsError,
    DocumentProcessingError,
    TenantNotFoundError,
    UserNotFoundError,
    ValidationError as DomainValidationError
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
        )


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
        )


# Type aliases for dependency injection
SearchServiceDep = Annotated[SearchApplicationService, Depends(get_search_service)]
UploadServiceDep = Annotated[UploadApplicationService, Depends(get_upload_service)]


# Domain Exception Handlers
def map_domain_exception_to_http(exception: Exception) -> HTTPException:
    """Map domain exceptions to appropriate HTTP responses."""
    
    if isinstance(exception, ProfileNotFoundError):
        return HTTPException(status_code=404, detail=str(exception))
    
    elif isinstance(exception, UserNotFoundError):
        return HTTPException(status_code=404, detail=str(exception))
    
    elif isinstance(exception, TenantNotFoundError):
        return HTTPException(status_code=404, detail=str(exception))
    
    elif isinstance(exception, InsufficientPermissionsError):
        return HTTPException(status_code=403, detail=str(exception))
    
    elif isinstance(exception, DocumentProcessingError):
        return HTTPException(status_code=422, detail=str(exception))
    
    elif isinstance(exception, DomainValidationError):
        return HTTPException(status_code=400, detail=str(exception))
    
    else:
        # Unknown domain exception - log and return generic error
        logger.error("Unknown domain exception", exception_type=type(exception).__name__, error=str(exception))
        return HTTPException(status_code=500, detail="Internal server error")


__all__ = [
    "get_search_service",
    "get_upload_service", 
    "SearchServiceDep",
    "UploadServiceDep",
    "map_domain_exception_to_http"
]