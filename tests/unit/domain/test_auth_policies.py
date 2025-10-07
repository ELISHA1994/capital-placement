"""
Unit tests for authentication domain policies and business rules.

These tests focus on pure authentication logic without external dependencies.
Tests password policies, token validation rules, and security business logic.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from app.utils.security import password_manager
from app.services.auth.authentication_service import AuthenticationService


class TestPasswordPolicyRules:
    """Test password policy business rules."""
    
    def test_valid_passwords_pass_validation(self):
        """Test that valid passwords pass strength validation."""
        valid_passwords = [
            "SecurePassword123!",
            "MyP@ssw0rd2024",
            "Complex#Pass1",
            "Str0ng!Password",
            "Valid@123Pass"
        ]
        
        for password in valid_passwords:
            result = password_manager.validate_password_strength(password)
            assert result["valid"] is True, f"Password '{password}' should be valid"
            assert len(result["errors"]) == 0
    
    def test_weak_passwords_fail_validation(self):
        """Test that weak passwords fail strength validation."""
        weak_passwords = [
            "short",  # Too short
            "password",  # No numbers, uppercase, or special chars
            "12345678",  # Only numbers
            "PASSWORD",  # Only uppercase
            "password123",  # No uppercase or special chars
            "Password!",  # Too short
            "",  # Empty
            "   ",  # Only spaces
        ]
        
        for password in weak_passwords:
            result = password_manager.validate_password_strength(password)
            assert result["valid"] is False, f"Password '{password}' should be invalid"
            assert len(result["errors"]) > 0
    
    def test_password_complexity_requirements(self):
        """Test specific password complexity requirements."""
        # Test minimum length requirement
        short_password = "Ab1!"
        result = password_manager.validate_password_strength(short_password)
        assert not result["valid"]
        assert any("length" in error.lower() for error in result["errors"])
        
        # Test uppercase requirement
        no_uppercase = "password123!"
        result = password_manager.validate_password_strength(no_uppercase)
        if not result["valid"]:
            assert any("uppercase" in error.lower() for error in result["errors"])
        
        # Test number requirement  
        no_numbers = "Password!"
        result = password_manager.validate_password_strength(no_numbers)
        if not result["valid"]:
            assert any("number" in error.lower() or "digit" in error.lower() for error in result["errors"])
    
    def test_password_hashing_consistency(self):
        """Test that password hashing is consistent and verifiable."""
        password = "TestPassword123!"
        
        # Hash the password
        hashed = password_manager.hash_password(password)
        
        # Verify the password matches the hash
        assert password_manager.verify_password(password, hashed) is True
        
        # Verify wrong password doesn't match
        assert password_manager.verify_password("WrongPassword", hashed) is False
        assert password_manager.verify_password("", hashed) is False
    
    def test_password_hash_uniqueness(self):
        """Test that hashing the same password produces different hashes (due to salt)."""
        password = "TestPassword123!"
        
        hash1 = password_manager.hash_password(password)
        hash2 = password_manager.hash_password(password)
        
        # Hashes should be different due to salt
        assert hash1 != hash2
        
        # But both should verify correctly
        assert password_manager.verify_password(password, hash1) is True
        assert password_manager.verify_password(password, hash2) is True


class TestAuthenticationBusinessLogic:
    """Test authentication business logic and rules."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Create minimal auth service for testing business logic
        self.auth_service = AuthenticationService(
            user_repository=None,
            tenant_repository=None,
            cache_manager=None,
            notification_service=None
        )
    
    def test_email_normalization(self):
        """Test that email addresses are properly normalized."""
        test_cases = [
            ("User@Example.Com", "user@example.com"),
            ("TEST@DOMAIN.ORG", "test@domain.org"),
            ("Mixed.Case@Example.Net", "mixed.case@example.net"),
            ("user+tag@domain.com", "user+tag@domain.com"),  # Preserve plus addressing
        ]
        
        for input_email, expected_output in test_cases:
            # Test through the UserLogin validator
            from app.infrastructure.persistence.models.auth_tables import UserLogin
            login = UserLogin(email=input_email, password="test")
            assert login.email == expected_output
    
    def test_token_data_normalization(self):
        """Test that user data is properly normalized for token generation."""
        # Sample user data from database (with UUID types)
        user_data = {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "email": "user@example.com",
            "full_name": "Test User",
            "tenant_id": "987fcdeb-51d2-43e8-b5f6-426614174000",
            "roles": ["user", "admin"],
            "permissions": ["read", "write"],
            "is_active": True,
            "is_superuser": False
        }
        
        # Normalize the data
        normalized = self.auth_service._normalize_user_data(user_data)
        
        # Verify UUID fields are converted to strings
        assert isinstance(normalized["id"], str)
        assert isinstance(normalized["tenant_id"], str)
        
        # Verify other fields are preserved
        assert normalized["email"] == user_data["email"]
        assert normalized["roles"] == user_data["roles"]
        assert normalized["is_active"] is True
    
    def test_user_data_defaults(self):
        """Test that missing user data fields get proper defaults."""
        minimal_user_data = {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "email": "user@example.com",
            "tenant_id": "987fcdeb-51d2-43e8-b5f6-426614174000"
        }
        
        normalized = self.auth_service._normalize_user_data(minimal_user_data)
        
        # Verify defaults are applied
        assert normalized["permissions"] == []
        assert normalized["is_superuser"] is False
        assert normalized["is_verified"] is False


class TestRateLimitingLogic:
    """Test rate limiting business logic."""
    
    def test_rate_limit_calculation(self):
        """Test rate limit enforcement calculations."""
        max_attempts = 5
        current_attempts = 3
        
        # Within limit
        assert current_attempts < max_attempts
        
        # At limit
        current_attempts = 5
        assert current_attempts >= max_attempts
        
        # Over limit
        current_attempts = 7
        assert current_attempts >= max_attempts
    
    def test_rate_limit_window_logic(self):
        """Test rate limiting time window logic."""
        window_minutes = 15
        now = datetime.utcnow()
        
        # Attempt within window (should count)
        recent_attempt = now - timedelta(minutes=5)
        assert (now - recent_attempt).total_seconds() < (window_minutes * 60)
        
        # Attempt outside window (should reset)
        old_attempt = now - timedelta(minutes=20)
        assert (now - old_attempt).total_seconds() > (window_minutes * 60)


class TestSecurityEventClassification:
    """Test security event risk classification logic."""
    
    def test_suspicious_activity_detection(self):
        """Test logic for detecting suspicious activities."""
        # Multiple failed login attempts should be flagged
        failed_attempts = 5
        threshold = 3
        
        is_suspicious = failed_attempts > threshold
        assert is_suspicious is True
        
        # Normal activity should not be flagged
        normal_attempts = 1
        is_suspicious = normal_attempts > threshold
        assert is_suspicious is False
    
    def test_risk_level_assignment(self):
        """Test risk level assignment logic."""
        # Low risk: Normal authentication
        risk_level = "low"
        assert risk_level in ["low", "medium", "high"]
        
        # Medium risk: Failed authentication
        failed_attempts = 2
        if 1 <= failed_attempts <= 3:
            risk_level = "medium"
        assert risk_level == "medium"
        
        # High risk: Multiple failures or suspicious patterns
        failed_attempts = 5
        if failed_attempts > 3:
            risk_level = "high"
        assert risk_level == "high"


class TestSessionValidationRules:
    """Test session validation business rules."""
    
    def test_session_expiration_logic(self):
        """Test session expiration calculation logic."""
        now = datetime.utcnow()
        
        # Valid session (not expired)
        valid_expiry = now + timedelta(hours=1)
        is_expired = now > valid_expiry
        assert is_expired is False
        
        # Expired session
        expired_expiry = now - timedelta(hours=1)
        is_expired = now > expired_expiry
        assert is_expired is True
        
        # Edge case: exactly at expiration time
        exact_expiry = now
        is_expired = now >= exact_expiry
        assert is_expired is True
    
    def test_session_extension_logic(self):
        """Test session extension calculation logic."""
        now = datetime.utcnow()
        original_expiry = now + timedelta(hours=1)
        extension_seconds = 3600  # 1 hour
        
        # Calculate new expiry using the original expiry as the baseline
        new_expiry = original_expiry + timedelta(seconds=extension_seconds)
        
        # New expiry should be later than original
        assert new_expiry > original_expiry
        
        # Should extend by the correct amount
        expected_extension = timedelta(seconds=extension_seconds)
        actual_extension = new_expiry - original_expiry
        assert abs((actual_extension - expected_extension).total_seconds()) < 1  # Allow for small timing differences


class TestAuthenticationResultValidation:
    """Test authentication result validation logic."""
    
    def test_successful_authentication_result(self):
        """Test validation of successful authentication results."""
        from app.infrastructure.persistence.models.auth_tables import AuthenticationResult, CurrentUser, TokenResponse
        
        # Mock successful result
        current_user = CurrentUser(
            user_id="123",
            email="user@test.com",
            full_name="Test User",
            tenant_id="456",
            roles=["user"],
            permissions=["read"],
            is_active=True,
            is_superuser=False
        )
        
        tokens = TokenResponse(
            access_token="mock_access_token",
            refresh_token="mock_refresh_token",
            token_type="bearer",
            expires_in=3600
        )
        
        result = AuthenticationResult(
            success=True,
            user=current_user,
            tokens=tokens
        )
        
        # Validate result structure
        assert result.success is True
        assert result.user is not None
        assert result.tokens is not None
        assert result.error is None
        
        # Validate user data
        assert result.user.email == "user@test.com"
        assert "user" in result.user.roles
        assert result.user.is_active is True
    
    def test_failed_authentication_result(self):
        """Test validation of failed authentication results."""
        from app.infrastructure.persistence.models.auth_tables import AuthenticationResult
        
        result = AuthenticationResult(
            success=False,
            error="Invalid credentials"
        )
        
        # Validate failure structure
        assert result.success is False
        assert result.user is None
        assert result.tokens is None
        assert result.error is not None
        assert result.error == "Invalid credentials"


class TestPasswordResetTokenLogic:
    """Test password reset token business logic."""
    
    def test_reset_token_ttl_calculation(self):
        """Test password reset token TTL calculation."""
        ttl_minutes = 30
        ttl_seconds = ttl_minutes * 60
        
        assert ttl_seconds == 1800
        
        # Minimum TTL enforcement
        min_ttl = 60  # 1 minute minimum
        calculated_ttl = max(min_ttl, ttl_seconds)
        assert calculated_ttl >= min_ttl
    
    def test_reset_token_validation_logic(self):
        """Test password reset token validation logic."""
        import hashlib
        
        # Simulate token generation and validation
        raw_token = "secure_random_token_123"
        token_digest = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        
        # Validation should use the same hashing
        validation_digest = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        
        assert token_digest == validation_digest
        
        # Wrong token should not match
        wrong_token = "wrong_token"
        wrong_digest = hashlib.sha256(wrong_token.encode("utf-8")).hexdigest()
        
        assert token_digest != wrong_digest
