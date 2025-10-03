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
    pass


class ConcurrencyError(DomainException):
    """Raised when concurrent operations conflict."""
    pass


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
    "ConcurrencyError"
]