"""
API Package

Central package for all API endpoints.
Provides versioned API routes with proper namespace management.

Note: Routers are imported lazily to avoid circular import issues with application services.
"""

from fastapi import APIRouter


def create_api_router() -> APIRouter:
    """
    Create and configure the main API router with all v1 routes.

    Uses lazy imports to avoid circular dependencies between:
    - Application layer services
    - API schemas
    - API dependencies
    - API routers
    """
    # Import v1 routes only when creating the router (lazy import)
    from app.api.v1.search import router as search_router
    from app.api.v1.upload import router as upload_router
    from app.api.v1.profiles import router as profiles_router
    from app.api.v1.auth import router as auth_router
    from app.api.v1.ai import router as ai_router
    from app.api.v1.tenants import router as tenants_router
    from app.api.v1.setup import router as setup_router
    from app.api.v1.audit import router as audit_router

    # Create main API router
    api_router = APIRouter()

    # Include v1 routers with version prefix
    api_router.include_router(
        search_router,
        prefix="/api/v1",
        tags=["search"]
    )

    api_router.include_router(
        upload_router,
        prefix="/api/v1",
        tags=["upload"]
    )

    api_router.include_router(
        profiles_router,
        prefix="/api/v1",
        tags=["profiles"]
    )

    api_router.include_router(
        auth_router,
        prefix="/api/v1",
        tags=["authentication"]
    )

    api_router.include_router(
        ai_router,
        prefix="/api/v1",
        tags=["ai"]
    )

    api_router.include_router(
        tenants_router,
        prefix="/api/v1",
        tags=["tenants"]
    )

    api_router.include_router(
        setup_router,
        prefix="/api/v1",
        tags=["setup"]
    )

    api_router.include_router(
        audit_router,
        prefix="/api/v1",
        tags=["audit"]
    )

    return api_router


# Router will be created lazily when needed
# Do NOT create api_router here to avoid circular imports
# Import api_router from main.py where it's created after all modules are loaded
api_router = None  # type: ignore  # Will be set in main.py

__all__ = ["api_router", "create_api_router"]