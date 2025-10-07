"""
Comprehensive tests for WebhookSignatureService.

This test module covers all signature generation and verification functionality
including basic signatures, timestamped signatures, and test utilities.
"""

import pytest
import time
from datetime import datetime, timedelta

from app.infrastructure.webhook.signature_service import WebhookSignatureService


class TestWebhookSignatureService:
    """Test webhook signature service comprehensive functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = WebhookSignatureService()
        self.test_payload = '{"event": "test", "data": {"id": 123}}'
        self.test_secret = "webhook_secret_key_123"

    # Basic Signature Generation Tests (3 tests)

    def test_generate_signature_sha256(self):
        """Test SHA256 signature generation."""
        signature = self.service.generate_signature(
            self.test_payload, self.test_secret, "sha256"
        )

        assert signature is not None
        assert signature.startswith("sha256=")
        assert len(signature) > 10

        # Signature should be deterministic
        signature2 = self.service.generate_signature(
            self.test_payload, self.test_secret, "sha256"
        )
        assert signature == signature2

    def test_generate_signature_sha1(self):
        """Test SHA1 signature generation."""
        signature = self.service.generate_signature(
            self.test_payload, self.test_secret, "sha1"
        )

        assert signature.startswith("sha1=")
        assert len(signature) > 10

    def test_generate_signature_unsupported_algorithm(self):
        """Test error handling for unsupported algorithm."""
        with pytest.raises(ValueError) as exc_info:
            self.service.generate_signature(
                self.test_payload, self.test_secret, "unsupported"
            )

        assert "Unsupported algorithm" in str(exc_info.value)

    # Signature Verification Tests (4 tests)

    def test_verify_signature_success(self):
        """Test successful signature verification."""
        signature = self.service.generate_signature(
            self.test_payload, self.test_secret, "sha256"
        )

        is_valid = self.service.verify_signature(
            self.test_payload, signature, self.test_secret, "sha256"
        )

        assert is_valid is True

    def test_verify_signature_wrong_secret(self):
        """Test signature verification with wrong secret."""
        signature = self.service.generate_signature(
            self.test_payload, self.test_secret, "sha256"
        )

        is_valid = self.service.verify_signature(
            self.test_payload, signature, "wrong_secret", "sha256"
        )

        assert is_valid is False

    def test_verify_signature_wrong_payload(self):
        """Test signature verification with wrong payload."""
        signature = self.service.generate_signature(
            self.test_payload, self.test_secret, "sha256"
        )

        wrong_payload = '{"event": "different", "data": {}}'
        is_valid = self.service.verify_signature(
            wrong_payload, signature, self.test_secret, "sha256"
        )

        assert is_valid is False

    def test_verify_signature_without_algorithm_prefix(self):
        """Test verification of signature without algorithm prefix."""
        # Generate signature
        full_signature = self.service.generate_signature(
            self.test_payload, self.test_secret, "sha256"
        )

        # Extract just the signature part (without "sha256=")
        sig_value = full_signature.split("=", 1)[1]

        # Verify should still work
        is_valid = self.service.verify_signature(
            self.test_payload, sig_value, self.test_secret, "sha256"
        )

        assert is_valid is True

    # Timestamped Signature Tests (5 tests)

    def test_generate_timestamp_signature(self):
        """Test timestamped signature generation."""
        timestamp = int(time.time())

        signature = self.service.generate_timestamp_signature(
            self.test_payload, timestamp, self.test_secret, "sha256"
        )

        assert signature is not None
        assert signature.startswith(f"t={timestamp},v1=")

        # Extract components
        parts = signature.split(",")
        assert len(parts) == 2
        assert parts[0] == f"t={timestamp}"
        assert parts[1].startswith("v1=")

    def test_verify_timestamp_signature_success(self):
        """Test successful timestamped signature verification."""
        timestamp = int(time.time())

        signature = self.service.generate_timestamp_signature(
            self.test_payload, timestamp, self.test_secret, "sha256"
        )

        is_valid = self.service.verify_timestamp_signature(
            self.test_payload, signature, self.test_secret, tolerance_seconds=300
        )

        assert is_valid is True

    def test_verify_timestamp_signature_expired(self):
        """Test timestamped signature with expired timestamp."""
        # Create timestamp 10 minutes in the past
        timestamp = int((datetime.utcnow() - timedelta(minutes=10)).timestamp())

        signature = self.service.generate_timestamp_signature(
            self.test_payload, timestamp, self.test_secret, "sha256"
        )

        # Verify with 5 minute tolerance should fail
        is_valid = self.service.verify_timestamp_signature(
            self.test_payload, signature, self.test_secret, tolerance_seconds=300
        )

        assert is_valid is False

    def test_verify_timestamp_signature_future(self):
        """Test timestamped signature with future timestamp."""
        # Create timestamp 10 minutes in the future
        timestamp = int((datetime.utcnow() + timedelta(minutes=10)).timestamp())

        signature = self.service.generate_timestamp_signature(
            self.test_payload, timestamp, self.test_secret, "sha256"
        )

        # Verify with 5 minute tolerance should fail
        is_valid = self.service.verify_timestamp_signature(
            self.test_payload, signature, self.test_secret, tolerance_seconds=300
        )

        assert is_valid is False

    def test_verify_timestamp_signature_invalid_format(self):
        """Test timestamped signature with invalid format."""
        invalid_signature = "invalid_format"

        is_valid = self.service.verify_timestamp_signature(
            self.test_payload, invalid_signature, self.test_secret
        )

        assert is_valid is False

    # Test Signature Generation Tests (3 tests)

    def test_generate_test_signature(self):
        """Test test signature generation."""
        payload, secret, signature = self.service.generate_test_signature()

        assert payload is not None
        assert secret is not None
        assert signature is not None
        assert signature.startswith("sha256=")

        # Verify generated test signature is valid
        is_valid = self.service.verify_signature(payload, signature, secret)
        assert is_valid is True

    def test_generate_test_signature_custom_payload(self):
        """Test test signature generation with custom payload."""
        custom_payload = '{"test": "custom", "id": 999}'

        payload, secret, signature = self.service.generate_test_signature(
            test_payload=custom_payload
        )

        assert payload == custom_payload
        assert secret is not None
        assert signature is not None

        # Verify generated test signature is valid
        is_valid = self.service.verify_signature(payload, signature, secret)
        assert is_valid is True

    def test_generate_test_signature_different_algorithms(self):
        """Test test signature generation with different algorithms."""
        payload_sha256, secret_sha256, sig_sha256 = self.service.generate_test_signature(
            algorithm="sha256"
        )
        payload_sha1, secret_sha1, sig_sha1 = self.service.generate_test_signature(
            algorithm="sha1"
        )

        # Both should be valid
        assert self.service.verify_signature(
            payload_sha256, sig_sha256, secret_sha256, "sha256"
        )
        assert self.service.verify_signature(
            payload_sha1, sig_sha1, secret_sha1, "sha1"
        )

        # Signatures should be different due to different algorithms
        assert sig_sha256 != sig_sha1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])