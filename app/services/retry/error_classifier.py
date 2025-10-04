"""Error classification service for determining retry behavior."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, Optional

import structlog
from aiohttp import ClientError, ClientResponseError, ClientTimeout
from sqlalchemy.exc import (
    DatabaseError, DisconnectionError, OperationalError, 
    TimeoutError as SQLTimeoutError, IntegrityError
)

from app.domain.interfaces import IHealthCheck
from app.domain.retry import ErrorCategory, IErrorClassifier


logger = structlog.get_logger(__name__)


class DefaultErrorClassifier(IErrorClassifier, IHealthCheck):
    """Default implementation of error classification for retry logic."""
    
    def __init__(self):
        self._logger = structlog.get_logger(__name__)
        
        # HTTP status code mappings
        self._http_status_mappings = {
            # 4xx Client Errors (mostly non-retryable)
            400: ErrorCategory.MALFORMED_REQUEST,
            401: ErrorCategory.AUTHENTICATION_ERROR,
            403: ErrorCategory.AUTHORIZATION_ERROR,
            404: ErrorCategory.RESOURCE_NOT_FOUND,
            405: ErrorCategory.UNSUPPORTED_OPERATION,
            409: ErrorCategory.CONFLICT_ERROR,
            413: ErrorCategory.QUOTA_EXCEEDED,
            422: ErrorCategory.VALIDATION_ERROR,
            429: ErrorCategory.RATE_LIMITED,
            
            # 5xx Server Errors (mostly retryable)
            500: ErrorCategory.TEMPORARY_FAILURE,
            502: ErrorCategory.SERVICE_UNAVAILABLE,
            503: ErrorCategory.SERVICE_UNAVAILABLE,
            504: ErrorCategory.TIMEOUT_ERROR,
            507: ErrorCategory.QUOTA_EXCEEDED,
            
            # Other specific codes
            408: ErrorCategory.TIMEOUT_ERROR,
            423: ErrorCategory.TEMPORARY_FAILURE,
            424: ErrorCategory.TEMPORARY_FAILURE,
        }
        
        # Error message patterns for classification
        self._error_patterns = [
            # Network errors
            (re.compile(r'connection.*(?:refused|reset|aborted|closed)', re.IGNORECASE), 
             ErrorCategory.CONNECTION_ERROR),
            (re.compile(r'network.*(?:unreachable|error|timeout)', re.IGNORECASE), 
             ErrorCategory.NETWORK_ERROR),
            (re.compile(r'dns.*(?:resolution|lookup).*(?:failed|error)', re.IGNORECASE), 
             ErrorCategory.NETWORK_ERROR),
            (re.compile(r'host.*(?:unreachable|not found)', re.IGNORECASE), 
             ErrorCategory.NETWORK_ERROR),
            
            # Timeout errors
            (re.compile(r'timeout', re.IGNORECASE), 
             ErrorCategory.TIMEOUT_ERROR),
            (re.compile(r'timed out', re.IGNORECASE), 
             ErrorCategory.TIMEOUT_ERROR),
            (re.compile(r'deadline exceeded', re.IGNORECASE), 
             ErrorCategory.TIMEOUT_ERROR),
            
            # Service availability
            (re.compile(r'service.*(?:unavailable|down|overloaded)', re.IGNORECASE), 
             ErrorCategory.SERVICE_UNAVAILABLE),
            (re.compile(r'server.*(?:unavailable|overloaded|busy)', re.IGNORECASE), 
             ErrorCategory.SERVICE_UNAVAILABLE),
            (re.compile(r'temporarily unavailable', re.IGNORECASE), 
             ErrorCategory.TEMPORARY_FAILURE),
            
            # Rate limiting
            (re.compile(r'rate.?limit', re.IGNORECASE), 
             ErrorCategory.RATE_LIMITED),
            (re.compile(r'too many requests', re.IGNORECASE), 
             ErrorCategory.RATE_LIMITED),
            (re.compile(r'quota.*exceeded', re.IGNORECASE), 
             ErrorCategory.QUOTA_EXCEEDED),
            
            # Authentication/Authorization
            (re.compile(r'unauthorized', re.IGNORECASE), 
             ErrorCategory.AUTHENTICATION_ERROR),
            (re.compile(r'forbidden', re.IGNORECASE), 
             ErrorCategory.AUTHORIZATION_ERROR),
            (re.compile(r'access.*denied', re.IGNORECASE), 
             ErrorCategory.PERMISSION_ERROR),
            (re.compile(r'invalid.*(?:token|credentials|api.?key)', re.IGNORECASE), 
             ErrorCategory.AUTHENTICATION_ERROR),
            
            # Validation errors
            (re.compile(r'validation.*(?:failed|error)', re.IGNORECASE), 
             ErrorCategory.VALIDATION_ERROR),
            (re.compile(r'invalid.*(?:request|input|parameter)', re.IGNORECASE), 
             ErrorCategory.VALIDATION_ERROR),
            (re.compile(r'malformed.*(?:request|json|xml)', re.IGNORECASE), 
             ErrorCategory.MALFORMED_REQUEST),
            
            # Configuration errors
            (re.compile(r'configuration.*(?:error|invalid)', re.IGNORECASE), 
             ErrorCategory.CONFIGURATION_ERROR),
            (re.compile(r'misconfigured', re.IGNORECASE), 
             ErrorCategory.CONFIGURATION_ERROR),
            
            # Data integrity errors
            (re.compile(r'integrity.*(?:constraint|violation)', re.IGNORECASE), 
             ErrorCategory.DATA_INTEGRITY_ERROR),
            (re.compile(r'unique.*constraint', re.IGNORECASE), 
             ErrorCategory.DATA_INTEGRITY_ERROR),
            (re.compile(r'foreign key.*constraint', re.IGNORECASE), 
             ErrorCategory.DATA_INTEGRITY_ERROR),
            
            # Business logic errors
            (re.compile(r'business.*rule.*violation', re.IGNORECASE), 
             ErrorCategory.BUSINESS_RULE_VIOLATION),
            (re.compile(r'invalid.*state', re.IGNORECASE), 
             ErrorCategory.INVALID_STATE),
        ]
    
    def classify_error(
        self, 
        error: Exception, 
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorCategory:
        """
        Classify an error into a retry category.
        
        Args:
            error: The exception that occurred
            context: Additional context about the operation
            
        Returns:
            ErrorCategory for the error
        """
        context = context or {}
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        self._logger.debug(
            "Classifying error for retry logic",
            error_type=error_type,
            error_message=str(error)[:200],
            context=context
        )
        
        # Classify by exception type first
        category = self._classify_by_exception_type(error)
        if category != ErrorCategory.UNKNOWN_ERROR:
            self._logger.debug("Error classified by exception type", category=category.value)
            return category
        
        # Classify HTTP errors by status code
        if hasattr(error, 'status') or hasattr(error, 'status_code'):
            status_code = getattr(error, 'status', None) or getattr(error, 'status_code', None)
            if status_code and status_code in self._http_status_mappings:
                category = self._http_status_mappings[status_code]
                self._logger.debug("Error classified by HTTP status", status_code=status_code, category=category.value)
                return category
        
        # Classify by error message patterns
        for pattern, pattern_category in self._error_patterns:
            if pattern.search(error_str):
                self._logger.debug("Error classified by message pattern", pattern=pattern.pattern, category=pattern_category.value)
                return pattern_category
        
        # Check context for additional classification hints
        if context:
            category = self._classify_by_context(error, context)
            if category != ErrorCategory.UNKNOWN_ERROR:
                return category
        
        # Default classification based on error type
        category = self._get_default_category_for_type(error)
        self._logger.debug("Error classified with default category", category=category.value)
        return category
    
    def _classify_by_exception_type(self, error: Exception) -> ErrorCategory:
        """Classify error based on exception type."""
        
        # Network/HTTP errors
        if isinstance(error, (ClientError, ConnectionError)):
            return ErrorCategory.CONNECTION_ERROR
        
        if isinstance(error, ClientTimeout):
            return ErrorCategory.TIMEOUT_ERROR
        
        if isinstance(error, ClientResponseError):
            if hasattr(error, 'status'):
                return self._http_status_mappings.get(error.status, ErrorCategory.NETWORK_ERROR)
            return ErrorCategory.NETWORK_ERROR
        
        # Database errors
        if isinstance(error, DisconnectionError):
            return ErrorCategory.CONNECTION_ERROR
        
        if isinstance(error, (OperationalError, SQLTimeoutError)):
            error_msg = str(error).lower()
            if 'timeout' in error_msg or 'timed out' in error_msg:
                return ErrorCategory.TIMEOUT_ERROR
            elif 'connection' in error_msg:
                return ErrorCategory.CONNECTION_ERROR
            return ErrorCategory.TEMPORARY_FAILURE
        
        if isinstance(error, IntegrityError):
            return ErrorCategory.DATA_INTEGRITY_ERROR
        
        if isinstance(error, DatabaseError):
            return ErrorCategory.TEMPORARY_FAILURE
        
        # Asyncio errors
        if isinstance(error, asyncio.TimeoutError):
            return ErrorCategory.TIMEOUT_ERROR
        
        if isinstance(error, asyncio.CancelledError):
            return ErrorCategory.CANCELLED  # Special handling for cancellation
        
        # Standard Python errors
        if isinstance(error, TimeoutError):
            return ErrorCategory.TIMEOUT_ERROR
        
        if isinstance(error, ConnectionError):
            return ErrorCategory.CONNECTION_ERROR
        
        if isinstance(error, PermissionError):
            return ErrorCategory.PERMISSION_ERROR
        
        if isinstance(error, ValueError):
            return ErrorCategory.VALIDATION_ERROR
        
        if isinstance(error, (FileNotFoundError, AttributeError, KeyError)):
            return ErrorCategory.CONFIGURATION_ERROR
        
        return ErrorCategory.UNKNOWN_ERROR
    
    def _classify_by_context(self, error: Exception, context: Dict[str, Any]) -> ErrorCategory:
        """Classify error using additional context information."""
        
        # Check if context provides explicit classification
        if 'error_category' in context:
            try:
                return ErrorCategory(context['error_category'])
            except ValueError:
                pass
        
        # Check operation type for context-specific classification
        operation_type = context.get('operation_type', '')
        
        if operation_type == 'webhook_delivery':
            # For webhooks, connection errors are often temporary
            if 'connection' in str(error).lower():
                return ErrorCategory.TEMPORARY_FAILURE
        
        elif operation_type == 'ai_processing':
            # AI processing errors often indicate quota or rate limits
            if 'quota' in str(error).lower() or 'limit' in str(error).lower():
                return ErrorCategory.QUOTA_EXCEEDED
        
        elif operation_type == 'document_processing':
            # Document processing errors are often validation related
            if 'invalid' in str(error).lower() or 'malformed' in str(error).lower():
                return ErrorCategory.VALIDATION_ERROR
        
        # Check for retry attempt context
        attempt_count = context.get('attempt_count', 0)
        if attempt_count > 0:
            # On retry attempts, be more lenient about classifying as temporary
            error_str = str(error).lower()
            if any(word in error_str for word in ['failed', 'error', 'exception']):
                return ErrorCategory.TEMPORARY_FAILURE
        
        return ErrorCategory.UNKNOWN_ERROR
    
    def _get_default_category_for_type(self, error: Exception) -> ErrorCategory:
        """Get default category based on error type when no specific classification matches."""
        
        # Most exceptions should be considered unknown unless explicitly classified
        if isinstance(error, (OSError, IOError)):
            return ErrorCategory.TEMPORARY_FAILURE
        
        if isinstance(error, RuntimeError):
            return ErrorCategory.TEMPORARY_FAILURE
        
        # Be conservative - unknown errors are not retryable by default
        return ErrorCategory.UNKNOWN_ERROR
    
    def is_transient(self, error: Exception) -> bool:
        """Check if an error is likely transient and retryable."""
        category = self.classify_error(error)
        
        transient_categories = {
            ErrorCategory.NETWORK_ERROR,
            ErrorCategory.CONNECTION_ERROR,
            ErrorCategory.TIMEOUT_ERROR,
            ErrorCategory.SERVICE_UNAVAILABLE,
            ErrorCategory.RATE_LIMITED,
            ErrorCategory.TEMPORARY_FAILURE,
        }
        
        return category in transient_categories
    
    def extract_retry_after(self, error: Exception) -> Optional[float]:
        """Extract retry-after delay from error if available."""
        
        # Check for Retry-After header in HTTP errors
        if hasattr(error, 'headers') and error.headers:
            retry_after = error.headers.get('Retry-After') or error.headers.get('retry-after')
            if retry_after:
                try:
                    return float(retry_after)
                except (ValueError, TypeError):
                    pass
        
        # Check for rate limit specific errors with delay information
        error_str = str(error).lower()
        
        # Look for patterns like "retry after X seconds"
        import re
        retry_patterns = [
            r'retry.*after.*?(\d+(?:\.\d+)?)\s*seconds?',
            r'wait.*?(\d+(?:\.\d+)?)\s*seconds?',
            r'delay.*?(\d+(?:\.\d+)?)\s*seconds?',
        ]
        
        for pattern in retry_patterns:
            match = re.search(pattern, error_str)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, IndexError):
                    continue
        
        # Check for OpenAI rate limit errors (common pattern)
        if 'rate limit' in error_str or 'quota exceeded' in error_str:
            # Default retry-after for rate limits
            return 60.0
        
        return None
    
    async def check_health(self) -> Dict[str, Any]:
        """Check health of the error classifier."""
        return {
            "status": "healthy",
            "classifier": "DefaultErrorClassifier",
            "supported_categories": len(ErrorCategory),
            "http_status_mappings": len(self._http_status_mappings),
            "error_patterns": len(self._error_patterns),
        }


class OpenAIErrorClassifier(DefaultErrorClassifier):
    """Specialized error classifier for OpenAI API errors."""
    
    def __init__(self):
        super().__init__()
        
        # OpenAI specific error patterns
        self._openai_patterns = [
            (re.compile(r'rate limit exceeded', re.IGNORECASE), ErrorCategory.RATE_LIMITED),
            (re.compile(r'quota exceeded', re.IGNORECASE), ErrorCategory.QUOTA_EXCEEDED),
            (re.compile(r'invalid api key', re.IGNORECASE), ErrorCategory.AUTHENTICATION_ERROR),
            (re.compile(r'model not found', re.IGNORECASE), ErrorCategory.CONFIGURATION_ERROR),
            (re.compile(r'content policy violation', re.IGNORECASE), ErrorCategory.BUSINESS_RULE_VIOLATION),
            (re.compile(r'context length exceeded', re.IGNORECASE), ErrorCategory.VALIDATION_ERROR),
            (re.compile(r'engine overloaded', re.IGNORECASE), ErrorCategory.SERVICE_UNAVAILABLE),
        ]
        
        self._error_patterns = self._openai_patterns + self._error_patterns
    
    def extract_retry_after(self, error: Exception) -> Optional[float]:
        """Extract retry-after delay with OpenAI-specific logic."""
        
        # First try default extraction
        delay = super().extract_retry_after(error)
        if delay:
            return delay
        
        error_str = str(error).lower()
        
        # OpenAI rate limit specific delays
        if 'rate limit' in error_str:
            if 'gpt-4' in error_str:
                return 60.0  # GPT-4 has stricter limits
            elif 'embedding' in error_str:
                return 30.0  # Embedding limits are usually less strict
            else:
                return 45.0  # Default for other models
        
        if 'quota exceeded' in error_str:
            return 3600.0  # Quota resets are usually hourly
        
        if 'engine overloaded' in error_str:
            return 10.0  # Usually resolves quickly
        
        return None


class DatabaseErrorClassifier(DefaultErrorClassifier):
    """Specialized error classifier for database errors."""
    
    def __init__(self):
        super().__init__()
        
        # Database specific error patterns
        self._db_patterns = [
            (re.compile(r'deadlock detected', re.IGNORECASE), ErrorCategory.TEMPORARY_FAILURE),
            (re.compile(r'lock timeout', re.IGNORECASE), ErrorCategory.TIMEOUT_ERROR),
            (re.compile(r'connection pool exhausted', re.IGNORECASE), ErrorCategory.TEMPORARY_FAILURE),
            (re.compile(r'too many connections', re.IGNORECASE), ErrorCategory.TEMPORARY_FAILURE),
            (re.compile(r'duplicate key', re.IGNORECASE), ErrorCategory.DATA_INTEGRITY_ERROR),
            (re.compile(r'foreign key constraint', re.IGNORECASE), ErrorCategory.DATA_INTEGRITY_ERROR),
            (re.compile(r'not null constraint', re.IGNORECASE), ErrorCategory.DATA_INTEGRITY_ERROR),
            (re.compile(r'check constraint', re.IGNORECASE), ErrorCategory.DATA_INTEGRITY_ERROR),
            (re.compile(r'serialization failure', re.IGNORECASE), ErrorCategory.TEMPORARY_FAILURE),
            (re.compile(r'could not serialize', re.IGNORECASE), ErrorCategory.TEMPORARY_FAILURE),
        ]
        
        self._error_patterns = self._db_patterns + self._error_patterns
    
    def extract_retry_after(self, error: Exception) -> Optional[float]:
        """Extract retry-after delay with database-specific logic."""
        
        error_str = str(error).lower()
        
        # Database specific retry delays
        if 'deadlock' in error_str:
            return 0.1  # Deadlocks should be retried quickly
        
        if 'lock timeout' in error_str:
            return 0.5  # Lock timeouts resolve relatively quickly
        
        if 'connection pool' in error_str or 'too many connections' in error_str:
            return 2.0  # Connection issues need time to resolve
        
        if 'serialization failure' in error_str:
            return 0.2  # Serialization failures should be retried quickly
        
        return super().extract_retry_after(error)


__all__ = [
    "DefaultErrorClassifier",
    "OpenAIErrorClassifier", 
    "DatabaseErrorClassifier"
]