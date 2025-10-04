"""
Domain-level exceptions for the hexagonal architecture.

These exceptions represent business rule violations and domain logic errors.
They should be mapped to appropriate HTTP responses in the API layer.
"""


class DomainException(Exception):
    """Base exception for all domain-level errors."""
    pass


class ValidationError(DomainException):
    """Raised when domain validation rules are violated."""
    pass


class NotFoundError(DomainException):
    """Base exception for entities not found."""
    pass


class ProfileNotFoundError(NotFoundError):
    """Raised when a profile is not found."""
    pass


class UserNotFoundError(NotFoundError):
    """Raised when a user is not found."""
    pass


class TenantNotFoundError(NotFoundError):
    """Raised when a tenant is not found."""
    pass


class DocumentNotFoundError(NotFoundError):
    """Raised when a document is not found."""
    pass


class AuthorizationError(DomainException):
    """Base exception for authorization errors."""
    pass


class InsufficientPermissionsError(AuthorizationError):
    """Raised when user lacks required permissions."""
    pass


class TenantAccessDeniedError(AuthorizationError):
    """Raised when user is denied access to tenant resources."""
    pass


class ProcessingError(DomainException):
    """Base exception for processing errors."""
    pass


class DocumentProcessingError(ProcessingError):
    """Raised when document processing fails."""
    pass


class SearchError(ProcessingError):
    """Raised when search operations fail."""
    pass


class EmbeddingGenerationError(ProcessingError):
    """Raised when embedding generation fails."""
    pass


class ConfigurationError(DomainException):
    """Raised when configuration is invalid or missing."""
    pass


class RateLimitExceededError(DomainException):
    """Raised when rate limits are exceeded."""
    
    def __init__(
        self,
        limit_type: str,
        limit_value: int,
        time_window: str,
        retry_after: int = None,
        identifier: str = None
    ):
        self.limit_type = limit_type
        self.limit_value = limit_value
        self.time_window = time_window
        self.retry_after = retry_after
        self.identifier = identifier
        
        message = f"Rate limit exceeded: {limit_value} requests per {time_window}"
        if identifier:
            message += f" for {identifier}"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
            
        super().__init__(message)


class ConcurrencyError(DomainException):
    """Raised when concurrent operations conflict."""
    pass


class WebhookValidationError(ValidationError):
    """Raised when webhook URL validation fails."""
    pass


class FileSizeExceededError(ValidationError):
    """Raised when uploaded file exceeds size limits."""
    
    def __init__(self, actual_size: int, max_size: int, filename: str = None):
        self.actual_size = actual_size
        self.max_size = max_size
        self.filename = filename
        
        size_mb = actual_size / (1024 * 1024)
        max_mb = max_size / (1024 * 1024)
        
        filename_str = f" '{filename}'" if filename else ""
        message = f"File{filename_str} size {size_mb:.2f}MB exceeds maximum allowed size of {max_mb:.2f}MB"
        super().__init__(message)


class InvalidFileError(ValidationError):
    """Raised when uploaded file is invalid or corrupted."""
    
    def __init__(self, filename: str = None, reason: str = None):
        self.filename = filename
        self.reason = reason
        
        filename_str = f" '{filename}'" if filename else ""
        reason_str = f": {reason}" if reason else ""
        message = f"Invalid file{filename_str}{reason_str}"
        super().__init__(message)


__all__ = [
    "DomainException",
    "ValidationError",
    "NotFoundError",
    "ProfileNotFoundError",
    "UserNotFoundError", 
    "TenantNotFoundError",
    "DocumentNotFoundError",
    "AuthorizationError",
    "InsufficientPermissionsError",
    "TenantAccessDeniedError",
    "ProcessingError",
    "DocumentProcessingError",
    "SearchError", 
    "EmbeddingGenerationError",
    "ConfigurationError",
    "RateLimitExceededError",
    "ConcurrencyError",
    "WebhookValidationError",
    "FileSizeExceededError",
    "InvalidFileError"
]