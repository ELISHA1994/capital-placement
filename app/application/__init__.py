"""Application layer entry points.

Holds orchestrators and use-case services that coordinate domain logic with adapters.

Note: Services are imported directly from their modules to avoid circular imports.
Use:
    from app.application.search_service import SearchApplicationService
    from app.application.upload_service import UploadApplicationService, UploadError
    from app.application.search import SearchApplicationService (migrated from app/services/core/)
"""

# Services are NOT imported here to avoid circular dependencies with API layer
# Import directly from submodules when needed

__all__ = [
    "SearchApplicationService",
    "UploadApplicationService",
    "UploadError",
]
