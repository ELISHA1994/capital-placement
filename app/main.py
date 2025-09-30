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

from app.core.config import get_settings
from app.core.service_factory import get_service_factory
from app.core.environment import log_environment_info
from app.api import api_router

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
    
    # Initialize services using the factory and async container
    try:
        # Initialize service factory
        factory = get_service_factory()
        
        # Initialize async container and critical services
        from app.core.dependencies import get_async_container
        from app.services.auth import AuthenticationService, AuthorizationService
        
        # Initialize async container
        container = await get_async_container()
        
        # Pre-warm authentication services
        auth_service = await container.get_service(AuthenticationService)
        authz_service = await container.get_service(AuthorizationService)
        
        # Pre-initialize other critical services
        cache_service = await factory.create_cache_service()
        ai_service = await factory.create_ai_service()
        
        # Check health
        cache_health = await cache_service.check_health()
        ai_health = await ai_service.check_health()
        
        logger.info("All services initialized successfully",
                   cache_status=cache_health["status"],
                   ai_status=ai_health["status"],
                   auth_services="initialized")
        
    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
        # Don't exit in development, but log the error
        if get_settings().ENVIRONMENT == "production":
            sys.exit(1)
    
    yield
    
    # Cleanup
    logger.info("Shutting down CV Matching Backend API")
    try:
        # Shutdown transaction manager first (rollback any active transactions)
        from app.core.transaction_manager import shutdown_transaction_manager
        await shutdown_transaction_manager()
        logger.info("Transaction manager shutdown completed")
        
        # Cleanup database
        from app.database import shutdown_database
        await shutdown_database()
        logger.info("Database shutdown completed")
        
        # Cleanup async container  
        from app.core.container import reset_async_container
        await reset_async_container()
        
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
    
    # Add middleware
    setup_middleware(app, settings)
    
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


def setup_middleware(app: FastAPI, settings) -> None:
    """Setup application middleware"""
    
    # CORS middleware
    cors_origins = settings.get_cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


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