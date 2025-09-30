"""
API v1 Routes

Simplified API endpoints that avoid complex dependencies while providing
proper FastAPI documentation structure.
"""

# Import secure routers with proper authentication
from .search import router as search_router
from .upload import router as upload_router  
from .profiles import router as profiles_router
from .auth import router as auth_router
from .ai import router as ai_router
from .tenants import router as tenants_router
from .setup import router as setup_router

__all__ = [
    "search_router",
    "upload_router", 
    "profiles_router",
    "auth_router",
    "ai_router",
    "tenants_router",
    "setup_router"
]