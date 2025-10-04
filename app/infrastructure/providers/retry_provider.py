"""Provider for retry and error recovery services."""

from __future__ import annotations

from typing import Optional

from app.domain.retry import IDeadLetterService, IRetryService, IRetryOperationExecutor
from app.services.retry.dead_letter_service import DeadLetterService
from app.services.retry.error_classifier import (
    DatabaseErrorClassifier, DefaultErrorClassifier, OpenAIErrorClassifier
)
from app.services.retry.retry_executor import RetryOperationExecutor
from app.services.retry.retry_service import RetryService


# Global service instances
_retry_service: Optional[RetryService] = None
_dead_letter_service: Optional[DeadLetterService] = None
_retry_executor: Optional[RetryOperationExecutor] = None
_default_error_classifier: Optional[DefaultErrorClassifier] = None
_openai_error_classifier: Optional[OpenAIErrorClassifier] = None
_database_error_classifier: Optional[DatabaseErrorClassifier] = None


async def get_retry_service() -> IRetryService:
    """Get the retry service instance."""
    global _retry_service
    
    if _retry_service is None:
        error_classifier = await get_default_error_classifier()
        _retry_service = RetryService(error_classifier=error_classifier)
    
    return _retry_service


async def get_dead_letter_service() -> IDeadLetterService:
    """Get the dead letter service instance."""
    global _dead_letter_service
    
    if _dead_letter_service is None:
        _dead_letter_service = DeadLetterService()
    
    return _dead_letter_service


async def get_retry_executor() -> IRetryOperationExecutor:
    """Get the retry operation executor instance."""
    global _retry_executor
    
    if _retry_executor is None:
        retry_service = await get_retry_service()
        error_classifier = await get_default_error_classifier()
        _retry_executor = RetryOperationExecutor(
            retry_service=retry_service,
            error_classifier=error_classifier
        )
    
    return _retry_executor


async def get_default_error_classifier() -> DefaultErrorClassifier:
    """Get the default error classifier instance."""
    global _default_error_classifier
    
    if _default_error_classifier is None:
        _default_error_classifier = DefaultErrorClassifier()
    
    return _default_error_classifier


async def get_openai_error_classifier() -> OpenAIErrorClassifier:
    """Get the OpenAI error classifier instance."""
    global _openai_error_classifier
    
    if _openai_error_classifier is None:
        _openai_error_classifier = OpenAIErrorClassifier()
    
    return _openai_error_classifier


async def get_database_error_classifier() -> DatabaseErrorClassifier:
    """Get the database error classifier instance."""
    global _database_error_classifier
    
    if _database_error_classifier is None:
        _database_error_classifier = DatabaseErrorClassifier()
    
    return _database_error_classifier


async def get_specialized_retry_service(operation_type: str) -> IRetryService:
    """Get a retry service with specialized error classifier for the operation type."""
    
    if operation_type in ["ai_processing", "openai_api", "embedding_generation"]:
        error_classifier = await get_openai_error_classifier()
    elif operation_type in ["database_operation", "data_processing"]:
        error_classifier = await get_database_error_classifier()
    else:
        error_classifier = await get_default_error_classifier()
    
    return RetryService(error_classifier=error_classifier)


async def get_specialized_retry_executor(operation_type: str) -> IRetryOperationExecutor:
    """Get a retry executor with specialized services for the operation type."""
    
    retry_service = await get_specialized_retry_service(operation_type)
    
    if operation_type in ["ai_processing", "openai_api", "embedding_generation"]:
        error_classifier = await get_openai_error_classifier()
    elif operation_type in ["database_operation", "data_processing"]:
        error_classifier = await get_database_error_classifier()
    else:
        error_classifier = await get_default_error_classifier()
    
    return RetryOperationExecutor(
        retry_service=retry_service,
        error_classifier=error_classifier
    )


async def health_check_retry_services() -> dict:
    """Perform health check on all retry services."""
    
    health_results = {}
    
    try:
        retry_service = await get_retry_service()
        health_results["retry_service"] = await retry_service.check_health()
    except Exception as e:
        health_results["retry_service"] = {"status": "unhealthy", "error": str(e)}
    
    try:
        dead_letter_service = await get_dead_letter_service()
        health_results["dead_letter_service"] = await dead_letter_service.check_health()
    except Exception as e:
        health_results["dead_letter_service"] = {"status": "unhealthy", "error": str(e)}
    
    try:
        retry_executor = await get_retry_executor()
        health_results["retry_executor"] = await retry_executor.check_health()
    except Exception as e:
        health_results["retry_executor"] = {"status": "unhealthy", "error": str(e)}
    
    try:
        default_classifier = await get_default_error_classifier()
        health_results["default_error_classifier"] = await default_classifier.check_health()
    except Exception as e:
        health_results["default_error_classifier"] = {"status": "unhealthy", "error": str(e)}
    
    # Determine overall health
    all_healthy = all(
        result.get("status") == "healthy" 
        for result in health_results.values()
    )
    
    health_results["overall_status"] = "healthy" if all_healthy else "degraded"
    
    return health_results


async def cleanup_retry_services() -> None:
    """Cleanup and reset retry service instances."""
    global _retry_service, _dead_letter_service, _retry_executor
    global _default_error_classifier, _openai_error_classifier, _database_error_classifier
    
    # Reset all global instances to force recreation
    _retry_service = None
    _dead_letter_service = None
    _retry_executor = None
    _default_error_classifier = None
    _openai_error_classifier = None
    _database_error_classifier = None


__all__ = [
    "get_retry_service",
    "get_dead_letter_service",
    "get_retry_executor",
    "get_default_error_classifier",
    "get_openai_error_classifier",
    "get_database_error_classifier",
    "get_specialized_retry_service",
    "get_specialized_retry_executor",
    "health_check_retry_services",
    "cleanup_retry_services"
]