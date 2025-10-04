"""
API Package

Central package for all API endpoints.
Provides versioned API routes with proper namespace management.
"""

from fastapi import APIRouter

# Import v1 routes
from app.api.v1 import upload_router, profiles_router, auth_router, ai_router, tenants_router, search_router, setup_router, audit_router
print("Successfully imported v1 API routes")

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

__all__ = ["api_router"]