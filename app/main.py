"""
CV Matching Backend - Main FastAPI Application
"""

import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import get_settings
from app.core.environment import log_environment_info
from app.core.service_factory import get_service_factory
from app.api import api_router
from app.infrastructure.providers.ai_provider import (
    get_openai_service,
    reset_ai_services,
)
from app.infrastructure.providers.analytics_provider import (
    reset_analytics_service,
)
from app.infrastructure.providers.auth_provider import (
    get_authentication_service,
    get_authorization_service,
    get_bootstrap_service,
    get_tenant_service,
    reset_authentication_service,
    reset_authorization_service,
    reset_bootstrap_service,
    reset_tenant_service,
)
from app.infrastructure.providers.cache_provider import (
    get_cache_service,
    reset_cache_service,
)
from app.infrastructure.providers.database_provider import (
    reset_database_service,
)
from app.infrastructure.providers.document_store_provider import (
    reset_document_store,
)
from app.infrastructure.providers.message_queue_provider import (
    reset_message_queue,
)
from app.infrastructure.providers.notification_provider import (
    reset_notification_service,
)
from app.infrastructure.providers.postgres_provider import (
    reset_postgres_adapter,
)
from app.infrastructure.providers.search_provider import reset_search_services
from app.infrastructure.providers.resource_provider import (
    get_file_resource_service,
    get_periodic_cleanup_service,
    shutdown_resource_services,
)
from app.infrastructure.providers.rate_limit_provider import (
    get_rate_limit_service,
    reset_rate_limit_service,
)
from app.infrastructure.task_manager import get_task_manager, shutdown_task_manager
from app.middleware import DefaultUsageTrackingMiddleware

# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer() if sys.stdout.isatty() else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting CV Matching Backend API", version=app.version)
    
    # Log environment information
    log_environment_info()
    
    # Initialize database first
    try:
        from app.database import initialize_database, get_database_manager
        
        settings = get_settings()
        logger.info("Initializing database connection and migrations")
        
        # Initialize database manager with migrations
        db_manager = await initialize_database(settings, run_migrations=True)
        
        # Initialize transaction manager with database manager
        from app.core.transaction_manager import initialize_transaction_manager
        tx_manager = initialize_transaction_manager(db_manager)
        
        # Check database health
        db_health = await db_manager.get_health_status()
        logger.info("Database and transaction manager initialized successfully", 
                   status=db_health["status"],
                   pgvector_available=db_health.get("pgvector_available", False))
        
    except Exception as e:
        logger.error("Failed to initialize database and transaction manager", error=str(e))
        if get_settings().ENVIRONMENT == "production":
            sys.exit(1)
    
    # Initialize services using provider-backed singletons
    try:
        # Initialize service factory
        factory = get_service_factory()
        
        # Pre-warm critical provider services
        await get_authentication_service()
        await get_authorization_service()
        await get_bootstrap_service()
        await get_tenant_service()

        cache_service = await get_cache_service()
        ai_service = await get_openai_service()
        
        # Initialize resource management services
        file_resource_service = await get_file_resource_service()
        periodic_cleanup_service = await get_periodic_cleanup_service()
        
        # Initialize rate limiting service (for health checks and dependency setup)
        rate_limit_service = await get_rate_limit_service()
        
        # Setup async-dependent middleware now that services are initialized
        await setup_async_middleware(app, get_settings())
        
        # Initialize task manager
        task_manager = get_task_manager()
        
        # Check health
        cache_health = await cache_service.check_health()
        ai_health = await ai_service.check_health()
        resource_health = await file_resource_service.check_health()
        cleanup_status = await periodic_cleanup_service.get_cleanup_status()
        rate_limit_health = await rate_limit_service.check_health()

        logger.info("All services initialized successfully",
                   cache_status=cache_health["status"],
                   ai_status=ai_health["status"],
                   resource_status=resource_health["status"],
                   cleanup_status=cleanup_status["status"],
                   rate_limit_status=rate_limit_health["status"],
                   auth_services="initialized",
                   task_manager="initialized")
        
    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
        # Don't exit in development, but log the error
        if get_settings().ENVIRONMENT == "production":
            sys.exit(1)
    
    yield
    
    # Cleanup
    logger.info("Shutting down CV Matching Backend API")
    try:
        # Shutdown task manager first (cancel all active tasks)
        await shutdown_task_manager()
        logger.info("Task manager shutdown completed")
        
        # Shutdown resource management services
        await shutdown_resource_services()
        logger.info("Resource management services shutdown completed")
        
        # Shutdown transaction manager (rollback any active transactions)
        from app.core.transaction_manager import shutdown_transaction_manager
        await shutdown_transaction_manager()
        logger.info("Transaction manager shutdown completed")
        
        # Cleanup database
        from app.database import shutdown_database
        await shutdown_database()
        logger.info("Database shutdown completed")

        # Reset provider singletons
        await reset_authentication_service()
        await reset_authorization_service()
        await reset_bootstrap_service()
        await reset_tenant_service()
        await reset_cache_service()
        await reset_ai_services()
        await reset_search_services()
        await reset_notification_service()
        await reset_message_queue()
        await reset_document_store()
        await reset_database_service()
        await reset_analytics_service()
        await reset_postgres_adapter()
        await reset_rate_limit_service()
        
        # Cleanup service factory
        factory = get_service_factory()
        factory.clear_cache()
        
        logger.info("Service cleanup completed")
    except Exception as e:
        logger.error("Cleanup error", error=str(e))


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()
    
    app = FastAPI(
        title="CV Matching Backend API",
        description="Intelligent CV/Resume matching system with semantic search capabilities",
        version="1.0.0",
        docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
        openapi_url="/openapi.json" if settings.ENVIRONMENT != "production" else None,
        lifespan=lifespan,
    )
    
    # Add basic middleware that doesn't require async initialization
    setup_basic_middleware(app, settings)
    
    # Include API routes
    app.include_router(api_router)
    
    # Basic health check endpoint
    @app.get("/health")
    async def health_check():
        """Basic health check endpoint"""
        return {
            "status": "healthy",
            "version": app.version,
            "environment": settings.ENVIRONMENT,
        }
    
    # Detailed health check with database
    @app.get("/health/detailed")
    async def detailed_health_check():
        """Detailed health check including database and services"""
        from app.database import get_database_manager
        
        health_status = {
            "status": "healthy",
            "version": app.version,
            "environment": settings.ENVIRONMENT,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "services": {}
        }
        
        # Database health
        try:
            db_manager = get_database_manager()
            db_health = await db_manager.get_detailed_health_status()
            health_status["services"]["database"] = db_health
        except Exception as e:
            health_status["services"]["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "unhealthy"
        
        # Service factory health
        try:
            factory = get_service_factory()
            cache_service = await factory.create_cache_service()
            cache_health = await cache_service.check_health()
            health_status["services"]["cache"] = cache_health
        except Exception as e:
            health_status["services"]["cache"] = {
                "status": "unhealthy", 
                "error": str(e)
            }
        
        return health_status
    
    # Database-specific health endpoint
    @app.get("/health/database")
    async def database_health_check():
        """Database-specific health check endpoint"""
        from app.database import get_database_manager
        
        try:
            db_manager = get_database_manager()
            db_health = await db_manager.get_detailed_health_status()
            return db_health
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
    
    # Migration status endpoint
    @app.get("/health/migrations")
    async def migration_status_check():
        """Migration status check endpoint"""
        from app.database import get_database_manager
        
        try:
            db_manager = get_database_manager()
            migration_status = await db_manager.migration_manager.get_migration_status()
            return {
                "status": "healthy",
                "migrations": migration_status,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
    
    # Transaction manager health endpoint
    @app.get("/health/transactions")
    async def transaction_health_check():
        """Transaction manager health check endpoint"""
        from app.core.transaction_manager import get_transaction_manager
        
        try:
            tx_manager = get_transaction_manager()
            tx_stats = await tx_manager.get_transaction_stats()
            return {
                "status": "healthy",
                "transaction_stats": tx_stats,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

    # API root
    @app.get("/")
    async def root():
        """API root endpoint"""
        return {
            "message": "CV Matching Backend API",
            "version": app.version,
            "docs_url": "/docs" if settings.ENVIRONMENT != "production" else None
        }
    
    return app




def setup_basic_middleware(app: FastAPI, settings) -> None:
    """Setup all middleware during app creation"""
    
    # Rate limiting middleware (first to protect against DDoS)
    # Using lazy initialization to handle async dependencies
    from app.middleware import LazyRateLimitMiddleware
    app.add_middleware(LazyRateLimitMiddleware, settings=settings)
    
    # Usage tracking middleware (before CORS to capture all requests)
    app.add_middleware(DefaultUsageTrackingMiddleware)
    
    # CORS middleware
    cors_origins = settings.get_cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    logger.info("All middleware configured successfully (rate limiting will initialize on first request)")


async def setup_async_middleware(app: FastAPI, settings) -> None:
    """Validate that async services are ready (middleware already configured)"""
    
    try:
        # Just validate that the rate limiting service can be initialized
        rate_limit_service = await get_rate_limit_service()
        rate_limit_health = await rate_limit_service.check_health()
        
        logger.info("Rate limiting service pre-validation successful", 
                   status=rate_limit_health["status"])
        
    except Exception as e:
        logger.error("Rate limiting service pre-validation failed", error=str(e))
        # The LazyRateLimitMiddleware will handle initialization failures gracefully


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development",
        log_config=None,  # Use our structured logging
    )
