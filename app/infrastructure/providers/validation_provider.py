"""
Validation service provider for file content validation.

Provides access to comprehensive file validation services following the provider pattern
for dependency injection throughout the application.
"""

from typing import Optional
import structlog

from app.domain.interfaces import IFileContentValidator, IWebhookValidator
from app.infrastructure.validation.file_content_validator import FileContentValidator
from app.domain.services.webhook_validator import WebhookValidator

logger = structlog.get_logger(__name__)

# Global singleton instances
_file_content_validator: Optional[IFileContentValidator] = None
_webhook_validator: Optional[IWebhookValidator] = None


async def get_file_content_validator() -> IFileContentValidator:
    """
    Get singleton instance of the file content validator.
    
    Provides comprehensive file validation including:
    - MIME type validation
    - File signature validation using magic bytes
    - Cross-validation for consistency
    - Security threat scanning
    
    Returns:
        IFileContentValidator: File content validation service
    """
    global _file_content_validator
    
    if _file_content_validator is None:
        logger.info("Initializing FileContentValidator service")
        _file_content_validator = FileContentValidator()
        
        # Perform health check on initialization
        try:
            health = await _file_content_validator.check_health()
            logger.info(
                "FileContentValidator initialized successfully",
                health_status=health.get("status"),
                supported_types=health.get("supported_types", [])
            )
        except Exception as e:
            logger.error(
                "FileContentValidator health check failed during initialization",
                error=str(e)
            )
            # Continue with initialization but log the error
    
    return _file_content_validator


def get_file_content_validator_sync() -> IFileContentValidator:
    """
    Get file content validator instance synchronously.
    
    Note: This should only be used in synchronous contexts where the async
    version cannot be used. The service will be initialized if needed.
    
    Returns:
        IFileContentValidator: File content validation service
    """
    global _file_content_validator
    
    if _file_content_validator is None:
        logger.info("Initializing FileContentValidator service (sync)")
        _file_content_validator = FileContentValidator()
    
    return _file_content_validator


async def get_webhook_validator() -> IWebhookValidator:
    """
    Get singleton instance of the webhook validator.
    
    Provides webhook URL validation with SSRF protection.
    
    Returns:
        IWebhookValidator: Webhook validation service
    """
    global _webhook_validator
    
    if _webhook_validator is None:
        logger.info("Initializing WebhookValidator service")
        _webhook_validator = WebhookValidator()
    
    return _webhook_validator


def get_webhook_validator_sync() -> IWebhookValidator:
    """
    Get webhook validator instance synchronously.
    
    Returns:
        IWebhookValidator: Webhook validation service
    """
    global _webhook_validator
    
    if _webhook_validator is None:
        logger.info("Initializing WebhookValidator service (sync)")
        _webhook_validator = WebhookValidator()
    
    return _webhook_validator


async def reset_validation_services() -> None:
    """
    Reset validation services (primarily for testing).
    
    This function resets all singleton instances to None, forcing them to be
    re-initialized on next access. Should only be used in testing scenarios.
    """
    global _file_content_validator, _webhook_validator
    
    logger.debug("Resetting validation services")
    _file_content_validator = None
    _webhook_validator = None


__all__ = [
    "get_file_content_validator",
    "get_file_content_validator_sync",
    "get_webhook_validator",
    "get_webhook_validator_sync",
    "reset_validation_services"
]