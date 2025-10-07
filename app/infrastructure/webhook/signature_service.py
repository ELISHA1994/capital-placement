"""
Webhook signature generation and verification service.

This module provides cryptographic signature generation and verification
for webhook payloads to ensure authenticity and prevent tampering.
"""

import hashlib
import hmac
import time
from typing import Optional

import structlog

from app.domain.interfaces import IWebhookSignatureService

logger = structlog.get_logger(__name__)


class WebhookSignatureService(IWebhookSignatureService):
    """Service for generating and verifying webhook signatures."""

    def __init__(self):
        """Initialize the signature service."""
        self._supported_algorithms = {"sha256", "sha1", "md5"}
        
    def generate_signature(
        self,
        payload: str,
        secret: str,
        algorithm: str = "sha256"
    ) -> str:
        """
        Generate webhook signature for payload.
        
        Args:
            payload: Webhook payload as string
            secret: Secret key for signing
            algorithm: Signature algorithm
            
        Returns:
            Generated signature in format "algorithm=signature"
            
        Raises:
            ValueError: If algorithm is not supported
        """
        if algorithm not in self._supported_algorithms:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. "
                f"Supported: {', '.join(self._supported_algorithms)}"
            )
        
        try:
            # Get the hash function
            hash_func = getattr(hashlib, algorithm)
            
            # Generate HMAC signature
            signature = hmac.new(
                secret.encode('utf-8'),
                payload.encode('utf-8'),
                hash_func
            ).hexdigest()
            
            # Return in format "algorithm=signature"
            return f"{algorithm}={signature}"
            
        except Exception as e:
            logger.error(
                "Failed to generate webhook signature",
                algorithm=algorithm,
                payload_length=len(payload),
                error=str(e)
            )
            raise ValueError(f"Failed to generate signature: {str(e)}")
    
    def verify_signature(
        self,
        payload: str,
        signature: str,
        secret: str,
        algorithm: str = "sha256"
    ) -> bool:
        """
        Verify webhook signature.
        
        Args:
            payload: Webhook payload as string
            signature: Provided signature to verify
            secret: Secret key for verification
            algorithm: Signature algorithm
            
        Returns:
            True if signature is valid
        """
        try:
            # Handle both formats: "algorithm=signature" and just "signature"
            if "=" in signature:
                sig_algorithm, sig_value = signature.split("=", 1)
                # Use the algorithm from the signature if provided
                if sig_algorithm in self._supported_algorithms:
                    algorithm = sig_algorithm
            else:
                sig_value = signature
            
            # Generate expected signature
            expected_signature = self.generate_signature(payload, secret, algorithm)
            
            # Extract the signature part (after "=")
            if "=" in expected_signature:
                expected_sig_value = expected_signature.split("=", 1)[1]
            else:
                expected_sig_value = expected_signature
            
            # Use constant-time comparison to prevent timing attacks
            return hmac.compare_digest(sig_value, expected_sig_value)
            
        except Exception as e:
            logger.warning(
                "Failed to verify webhook signature",
                algorithm=algorithm,
                signature_length=len(signature),
                payload_length=len(payload),
                error=str(e)
            )
            return False
    
    def generate_timestamp_signature(
        self,
        payload: str,
        timestamp: int,
        secret: str,
        algorithm: str = "sha256"
    ) -> str:
        """
        Generate timestamped signature for replay protection.
        
        The timestamp is included in the signed data to prevent replay attacks.
        
        Args:
            payload: Webhook payload as string
            timestamp: Unix timestamp
            secret: Secret key for signing
            algorithm: Signature algorithm
            
        Returns:
            Generated timestamped signature in format "t=timestamp,v1=signature"
        """
        try:
            # Create signed payload with timestamp
            signed_payload = f"{timestamp}.{payload}"
            
            # Generate signature
            signature = self.generate_signature(signed_payload, secret, algorithm)
            
            # Extract signature value
            if "=" in signature:
                sig_value = signature.split("=", 1)[1]
            else:
                sig_value = signature
            
            # Return in Stripe/standard webhook format
            return f"t={timestamp},v1={sig_value}"
            
        except Exception as e:
            logger.error(
                "Failed to generate timestamped webhook signature",
                timestamp=timestamp,
                algorithm=algorithm,
                payload_length=len(payload),
                error=str(e)
            )
            raise ValueError(f"Failed to generate timestamped signature: {str(e)}")
    
    def verify_timestamp_signature(
        self,
        payload: str,
        signature: str,
        secret: str,
        tolerance_seconds: int = 300,
        algorithm: str = "sha256"
    ) -> bool:
        """
        Verify timestamped signature with replay protection.
        
        Args:
            payload: Webhook payload as string
            signature: Timestamped signature to verify
            secret: Secret key for verification
            tolerance_seconds: Maximum age of signature in seconds
            algorithm: Signature algorithm
            
        Returns:
            True if signature is valid and within tolerance
        """
        try:
            # Parse timestamp and signature from the header
            # Format: "t=timestamp,v1=signature"
            parts = signature.split(",")
            timestamp_part = None
            signature_part = None
            
            for part in parts:
                if part.startswith("t="):
                    timestamp_part = part[2:]
                elif part.startswith("v1="):
                    signature_part = part[3:]
            
            if not timestamp_part or not signature_part:
                logger.warning(
                    "Invalid timestamped signature format",
                    signature=signature
                )
                return False
            
            # Convert timestamp
            try:
                timestamp = int(timestamp_part)
            except ValueError:
                logger.warning(
                    "Invalid timestamp in signature",
                    timestamp=timestamp_part
                )
                return False
            
            # Check timestamp tolerance
            current_time = int(time.time())
            if abs(current_time - timestamp) > tolerance_seconds:
                logger.warning(
                    "Timestamped signature outside tolerance",
                    timestamp=timestamp,
                    current_time=current_time,
                    tolerance_seconds=tolerance_seconds,
                    age_seconds=abs(current_time - timestamp)
                )
                return False
            
            # Verify signature
            signed_payload = f"{timestamp}.{payload}"
            expected_signature = self.generate_signature(signed_payload, secret, algorithm)
            
            # Extract signature value
            if "=" in expected_signature:
                expected_sig_value = expected_signature.split("=", 1)[1]
            else:
                expected_sig_value = expected_signature
            
            # Use constant-time comparison
            is_valid = hmac.compare_digest(signature_part, expected_sig_value)
            
            if not is_valid:
                logger.warning(
                    "Timestamped signature verification failed",
                    timestamp=timestamp,
                    algorithm=algorithm
                )
            
            return is_valid
            
        except Exception as e:
            logger.warning(
                "Failed to verify timestamped webhook signature",
                signature=signature,
                algorithm=algorithm,
                payload_length=len(payload),
                error=str(e)
            )
            return False
    
    def generate_test_signature(
        self,
        test_payload: Optional[str] = None,
        algorithm: str = "sha256"
    ) -> tuple[str, str, str]:
        """
        Generate test signature for webhook endpoint testing.
        
        Args:
            test_payload: Custom test payload, or default if None
            algorithm: Signature algorithm
            
        Returns:
            Tuple of (payload, secret, signature)
        """
        if test_payload is None:
            test_payload = '{"event": "test", "timestamp": "' + str(int(time.time())) + '"}'
        
        # Generate a test secret
        test_secret = "test_secret_" + str(int(time.time()))
        
        # Generate signature
        signature = self.generate_signature(test_payload, test_secret, algorithm)
        
        return test_payload, test_secret, signature


__all__ = ["WebhookSignatureService"]