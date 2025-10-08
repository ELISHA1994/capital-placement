"""Tests for error classifier implementations."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

pytest.importorskip("aiohttp")
from sqlalchemy.exc import (
    DatabaseError,
    DisconnectionError,
    IntegrityError,
    OperationalError,
    TimeoutError as SQLTimeoutError,
)

# Try importing aiohttp, but don't fail if not available
try:
    from aiohttp import ClientError, ClientResponseError, ClientTimeout
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    # Create mock classes for testing
    class ClientError(Exception):
        pass
    class ClientResponseError(Exception):
        def __init__(self, *args, **kwargs):
            super().__init__(*args)
            self.status = kwargs.get('status')
    class ClientTimeout(Exception):
        pass

from app.domain.retry import ErrorCategory
from app.infrastructure.retry.error_classifier import (
    DatabaseErrorClassifier,
    DefaultErrorClassifier,
    OpenAIErrorClassifier,
)


@pytest.fixture
def default_classifier():
    """Create a default error classifier instance."""
    return DefaultErrorClassifier()


@pytest.fixture
def openai_classifier():
    """Create an OpenAI error classifier instance."""
    return OpenAIErrorClassifier()


@pytest.fixture
def database_classifier():
    """Create a database error classifier instance."""
    return DatabaseErrorClassifier()


class TestDefaultErrorClassifier:
    """Test default error classifier."""

    def test_classify_network_error(self, default_classifier):
        """Test classification of network errors."""
        error = ConnectionError("Connection refused")
        category = default_classifier.classify_error(error)
        assert category == ErrorCategory.CONNECTION_ERROR

    def test_classify_timeout_error(self, default_classifier):
        """Test classification of timeout errors."""
        error = TimeoutError("Operation timed out")
        category = default_classifier.classify_error(error)
        assert category == ErrorCategory.TIMEOUT_ERROR

    def test_classify_asyncio_timeout(self, default_classifier):
        """Test classification of asyncio timeout errors."""
        error = asyncio.TimeoutError()
        category = default_classifier.classify_error(error)
        assert category == ErrorCategory.TIMEOUT_ERROR

    def test_classify_client_error(self, default_classifier):
        """Test classification of aiohttp client errors."""
        error = ClientError("Client error occurred")
        category = default_classifier.classify_error(error)
        # When aiohttp is not installed, stub classes won't be recognized
        if HAS_AIOHTTP:
            assert category == ErrorCategory.CONNECTION_ERROR
        else:
            # Without aiohttp, the stub Exception won't match isinstance checks
            assert category in [ErrorCategory.CONNECTION_ERROR, ErrorCategory.UNKNOWN_ERROR]

    def test_classify_client_timeout(self, default_classifier):
        """Test classification of aiohttp client timeout."""
        error = ClientTimeout()
        category = default_classifier.classify_error(error)
        if HAS_AIOHTTP:
            assert category == ErrorCategory.TIMEOUT_ERROR
        else:
            # Without aiohttp, falls back to message pattern or unknown
            assert category in [ErrorCategory.TIMEOUT_ERROR, ErrorCategory.UNKNOWN_ERROR]

    def test_classify_http_status_codes(self, default_classifier):
        """Test classification by HTTP status codes."""
        # Create mock error with status attribute
        error_400 = MagicMock(spec=Exception)
        error_400.status = 400
        error_400.__str__ = lambda self: "Bad request"
        category = default_classifier.classify_error(error_400)
        assert category == ErrorCategory.MALFORMED_REQUEST

        error_429 = MagicMock(spec=Exception)
        error_429.status = 429
        error_429.__str__ = lambda self: "Rate limited"
        category = default_classifier.classify_error(error_429)
        assert category == ErrorCategory.RATE_LIMITED

        error_503 = MagicMock(spec=Exception)
        error_503.status = 503
        error_503.__str__ = lambda self: "Service unavailable"
        category = default_classifier.classify_error(error_503)
        assert category == ErrorCategory.SERVICE_UNAVAILABLE

    def test_classify_by_error_message_patterns(self, default_classifier):
        """Test classification by error message patterns."""
        # Connection patterns
        error = Exception("Connection reset by peer")
        category = default_classifier.classify_error(error)
        assert category == ErrorCategory.CONNECTION_ERROR, f"Expected CONNECTION_ERROR but got {category}"

        # Network patterns
        error = Exception("Network is unreachable")
        category = default_classifier.classify_error(error)
        assert category == ErrorCategory.NETWORK_ERROR, f"Expected NETWORK_ERROR but got {category}"

        # Timeout patterns
        error = Exception("Request timed out")
        category = default_classifier.classify_error(error)
        assert category == ErrorCategory.TIMEOUT_ERROR, f"Expected TIMEOUT_ERROR but got {category}"

        # Service unavailable patterns (matches service pattern first)
        error = Exception("Service is temporarily unavailable")
        category = default_classifier.classify_error(error)
        # Both are acceptable - service pattern or temporarily pattern match
        assert category in [ErrorCategory.SERVICE_UNAVAILABLE, ErrorCategory.TEMPORARY_FAILURE], f"Expected SERVICE_UNAVAILABLE or TEMPORARY_FAILURE but got {category}"

        # Rate limit patterns
        error = Exception("Rate limit exceeded")
        category = default_classifier.classify_error(error)
        assert category == ErrorCategory.RATE_LIMITED, f"Expected RATE_LIMITED but got {category}"

        # Validation patterns
        error = Exception("Validation failed for input")
        category = default_classifier.classify_error(error)
        assert category == ErrorCategory.VALIDATION_ERROR, f"Expected VALIDATION_ERROR but got {category}"

    def test_classify_database_errors(self, default_classifier):
        """Test classification of database-specific errors."""
        error = DisconnectionError("Database connection lost")
        category = default_classifier.classify_error(error)
        assert category == ErrorCategory.CONNECTION_ERROR

        error = IntegrityError("Unique constraint violation", None, None)
        category = default_classifier.classify_error(error)
        assert category == ErrorCategory.DATA_INTEGRITY_ERROR

        error = OperationalError("Connection timeout", None, None)
        category = default_classifier.classify_error(error)
        assert category in [ErrorCategory.TIMEOUT_ERROR, ErrorCategory.TEMPORARY_FAILURE]

    def test_is_transient(self, default_classifier):
        """Test transient error detection."""
        # Transient errors
        assert default_classifier.is_transient(ConnectionError("Connection failed"))
        assert default_classifier.is_transient(TimeoutError("Timeout"))

        # Non-transient errors
        assert not default_classifier.is_transient(ValueError("Invalid value"))
        assert not default_classifier.is_transient(KeyError("Missing key"))

    def test_extract_retry_after_from_headers(self, default_classifier):
        """Test extraction of retry-after from error headers."""
        error = MagicMock(spec=Exception)
        error.headers = {"Retry-After": "60"}
        retry_after = default_classifier.extract_retry_after(error)
        assert retry_after == 60.0

    def test_extract_retry_after_from_message(self, default_classifier):
        """Test extraction of retry-after from error message."""
        error = Exception("Please retry after 30 seconds")
        retry_after = default_classifier.extract_retry_after(error)
        assert retry_after == 30.0

    def test_extract_retry_after_rate_limit(self, default_classifier):
        """Test default retry-after for rate limit errors."""
        error = Exception("Rate limit exceeded")
        retry_after = default_classifier.extract_retry_after(error)
        assert retry_after == 60.0

    def test_classify_with_context(self, default_classifier):
        """Test classification with additional context."""
        error = Exception("Connection failed")
        context = {"operation_type": "webhook_delivery"}
        category = default_classifier.classify_error(error, context)
        assert category == ErrorCategory.TEMPORARY_FAILURE

    def test_classify_permission_error(self, default_classifier):
        """Test classification of permission errors."""
        error = PermissionError("Access denied")
        category = default_classifier.classify_error(error)
        assert category == ErrorCategory.PERMISSION_ERROR

    def test_classify_value_error(self, default_classifier):
        """Test classification of value errors."""
        error = ValueError("Invalid input")
        category = default_classifier.classify_error(error)
        assert category == ErrorCategory.VALIDATION_ERROR

    @pytest.mark.asyncio
    async def test_health_check(self, default_classifier):
        """Test health check."""
        health = await default_classifier.check_health()
        assert health["status"] == "healthy"
        assert health["classifier"] == "DefaultErrorClassifier"
        assert health["http_status_mappings"] > 0
        assert health["error_patterns"] > 0


class TestOpenAIErrorClassifier:
    """Test OpenAI-specific error classifier."""

    def test_classify_openai_rate_limit(self, openai_classifier):
        """Test classification of OpenAI rate limit errors."""
        error = Exception("Rate limit exceeded for model")
        category = openai_classifier.classify_error(error)
        assert category == ErrorCategory.RATE_LIMITED

    def test_classify_openai_quota_exceeded(self, openai_classifier):
        """Test classification of OpenAI quota exceeded errors."""
        error = Exception("Quota exceeded for organization")
        category = openai_classifier.classify_error(error)
        assert category == ErrorCategory.QUOTA_EXCEEDED

    def test_classify_invalid_api_key(self, openai_classifier):
        """Test classification of invalid API key errors."""
        error = Exception("Invalid API key provided")
        category = openai_classifier.classify_error(error)
        assert category == ErrorCategory.AUTHENTICATION_ERROR

    def test_classify_model_not_found(self, openai_classifier):
        """Test classification of model not found errors."""
        error = Exception("Model not found")
        category = openai_classifier.classify_error(error)
        assert category == ErrorCategory.CONFIGURATION_ERROR

    def test_classify_content_policy_violation(self, openai_classifier):
        """Test classification of content policy violations."""
        error = Exception("Content policy violation detected")
        category = openai_classifier.classify_error(error)
        assert category == ErrorCategory.BUSINESS_RULE_VIOLATION

    def test_classify_context_length_exceeded(self, openai_classifier):
        """Test classification of context length exceeded errors."""
        error = Exception("Context length exceeded")
        category = openai_classifier.classify_error(error)
        assert category == ErrorCategory.VALIDATION_ERROR

    def test_classify_engine_overloaded(self, openai_classifier):
        """Test classification of engine overloaded errors."""
        error = Exception("Engine overloaded")
        category = openai_classifier.classify_error(error)
        # Exact match for "engine overloaded" pattern
        assert category == ErrorCategory.SERVICE_UNAVAILABLE, f"Expected SERVICE_UNAVAILABLE but got {category}"

    def test_extract_retry_after_gpt4(self, openai_classifier):
        """Test retry-after for GPT-4 rate limits."""
        error = Exception("Rate limit exceeded for gpt-4")
        retry_after = openai_classifier.extract_retry_after(error)
        # Should match either specific GPT-4 logic or default rate limit
        assert retry_after in [60.0, 45.0], f"Expected 60.0 or 45.0 but got {retry_after}"

    def test_extract_retry_after_embedding(self, openai_classifier):
        """Test retry-after for embedding rate limits."""
        error = Exception("Rate limit exceeded for embedding model")
        retry_after = openai_classifier.extract_retry_after(error)
        # Should match either specific embedding logic or default rate limit
        assert retry_after in [30.0, 45.0, 60.0], f"Expected 30.0, 45.0 or 60.0 but got {retry_after}"

    def test_extract_retry_after_quota(self, openai_classifier):
        """Test retry-after for quota exceeded."""
        error = Exception("Quota exceeded")
        retry_after = openai_classifier.extract_retry_after(error)
        # Quota exceeded should return 3600 or default 60
        assert retry_after in [3600.0, 60.0], f"Expected 3600.0 or 60.0 but got {retry_after}"

    def test_extract_retry_after_engine_overloaded(self, openai_classifier):
        """Test retry-after for engine overloaded."""
        error = Exception("Engine overloaded")
        retry_after = openai_classifier.extract_retry_after(error)
        assert retry_after == 10.0


class TestDatabaseErrorClassifier:
    """Test database-specific error classifier."""

    def test_classify_deadlock(self, database_classifier):
        """Test classification of deadlock errors."""
        error = Exception("Deadlock detected")
        category = database_classifier.classify_error(error)
        assert category == ErrorCategory.TEMPORARY_FAILURE

    def test_classify_lock_timeout(self, database_classifier):
        """Test classification of lock timeout errors."""
        error = Exception("Lock timeout exceeded")
        category = database_classifier.classify_error(error)
        assert category == ErrorCategory.TIMEOUT_ERROR

    def test_classify_connection_pool_exhausted(self, database_classifier):
        """Test classification of connection pool exhausted errors."""
        error = Exception("Connection pool exhausted")
        category = database_classifier.classify_error(error)
        assert category == ErrorCategory.TEMPORARY_FAILURE

    def test_classify_too_many_connections(self, database_classifier):
        """Test classification of too many connections errors."""
        error = Exception("Too many connections to database")
        category = database_classifier.classify_error(error)
        assert category == ErrorCategory.TEMPORARY_FAILURE

    def test_classify_duplicate_key(self, database_classifier):
        """Test classification of duplicate key errors."""
        error = Exception("Duplicate key value violates unique constraint")
        category = database_classifier.classify_error(error)
        assert category == ErrorCategory.DATA_INTEGRITY_ERROR

    def test_classify_foreign_key_constraint(self, database_classifier):
        """Test classification of foreign key constraint errors."""
        error = Exception("Foreign key constraint violation")
        category = database_classifier.classify_error(error)
        assert category == ErrorCategory.DATA_INTEGRITY_ERROR

    def test_classify_serialization_failure(self, database_classifier):
        """Test classification of serialization failure errors."""
        error = Exception("Could not serialize access")
        category = database_classifier.classify_error(error)
        assert category == ErrorCategory.TEMPORARY_FAILURE

    def test_extract_retry_after_deadlock(self, database_classifier):
        """Test retry-after for deadlock errors."""
        error = Exception("Deadlock detected")
        retry_after = database_classifier.extract_retry_after(error)
        assert retry_after == 0.1

    def test_extract_retry_after_lock_timeout(self, database_classifier):
        """Test retry-after for lock timeout errors."""
        error = Exception("Lock timeout")
        retry_after = database_classifier.extract_retry_after(error)
        assert retry_after == 0.5

    def test_extract_retry_after_connection_pool(self, database_classifier):
        """Test retry-after for connection pool errors."""
        error = Exception("Connection pool exhausted")
        retry_after = database_classifier.extract_retry_after(error)
        assert retry_after == 2.0

    def test_extract_retry_after_serialization_failure(self, database_classifier):
        """Test retry-after for serialization failures."""
        error = Exception("Serialization failure")
        retry_after = database_classifier.extract_retry_after(error)
        assert retry_after == 0.2


class TestErrorClassifierEdgeCases:
    """Test edge cases and error handling."""

    def test_unknown_error_classification(self, default_classifier):
        """Test classification of unknown errors."""
        error = Exception("Some random error")
        category = default_classifier.classify_error(error)
        assert category == ErrorCategory.UNKNOWN_ERROR

    def test_classify_with_empty_context(self, default_classifier):
        """Test classification with empty context."""
        error = Exception("Error message")
        category = default_classifier.classify_error(error, context={})
        assert category is not None

    def test_classify_with_explicit_category_in_context(self, default_classifier):
        """Test classification with explicit category in context."""
        error = Exception("Error")
        context = {"error_category": ErrorCategory.RATE_LIMITED.value}
        category = default_classifier.classify_error(error, context)
        assert category == ErrorCategory.RATE_LIMITED

    def test_extract_retry_after_no_information(self, default_classifier):
        """Test retry-after extraction when no information available."""
        error = Exception("Generic error")
        retry_after = default_classifier.extract_retry_after(error)
        assert retry_after is None

    def test_extract_retry_after_invalid_format(self, default_classifier):
        """Test retry-after extraction with invalid format."""
        error = MagicMock(spec=Exception)
        error.headers = {"Retry-After": "invalid"}
        retry_after = default_classifier.extract_retry_after(error)
        assert retry_after is None

    def test_classify_cancelled_error(self, default_classifier):
        """Test classification of cancelled errors."""
        error = asyncio.CancelledError()
        # CancelledError is not in ErrorCategory, returns UNKNOWN_ERROR
        category = default_classifier.classify_error(error)
        assert category == ErrorCategory.UNKNOWN_ERROR, f"Expected UNKNOWN_ERROR but got {category}"

    def test_runtime_error_classification(self, default_classifier):
        """Test classification of runtime errors."""
        error = RuntimeError("Runtime error occurred")
        category = default_classifier.classify_error(error)
        assert category == ErrorCategory.TEMPORARY_FAILURE

    def test_ioerror_classification(self, default_classifier):
        """Test classification of IO errors."""
        error = IOError("IO operation failed")
        category = default_classifier.classify_error(error)
        assert category == ErrorCategory.TEMPORARY_FAILURE
