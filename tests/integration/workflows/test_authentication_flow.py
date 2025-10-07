"""
Integration tests for authentication workflows.

These tests verify complete authentication flows including
token generation, session management, and error handling.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import patch

from app.infrastructure.auth.authentication_service import AuthenticationService
from app.infrastructure.persistence.models.auth_tables import UserLogin, PasswordResetRequest, PasswordResetConfirm
from tests.mocks.mock_repositories import MockUserRepository, MockTenantRepository, MockUserSessionRepository
from tests.mocks.mock_services import MockCacheService, MockNotificationService
from tests.fixtures.tenant_fixtures import TenantTestBuilder


class TestAuthenticationWorkflow:
    """Test complete authentication workflows."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.mock_user_repo = MockUserRepository()
        self.mock_tenant_repo = MockTenantRepository()
        self.mock_session_repo = MockUserSessionRepository()
        self.mock_cache = MockCacheService()
        self.mock_notification = MockNotificationService()
        
        self.auth_service = AuthenticationService(
            user_repository=self.mock_user_repo,
            tenant_repository=self.mock_tenant_repo,
            cache_manager=self.mock_cache,
            notification_service=self.mock_notification,
            session_repository=self.mock_session_repo
        )
        
        # Setup test tenant
        self.test_tenant = TenantTestBuilder().build()
        self.mock_tenant_repo.add_test_tenant(self.test_tenant)
        
        # Setup test user with a consistent user ID
        self.test_user_id = str(uuid4())
        self.test_user = {
            "id": self.test_user_id,
            "tenant_id": self.test_tenant["id"],
            "email": "user@test.com",
            "hashed_password": "$2b$12$test_hashed_password",  # Mock hash
            "full_name": "Test User",
            "roles": ["user"],
            "permissions": ["read"],
            "is_active": True,
            "is_verified": True,
            "is_superuser": False
        }
        self.mock_user_repo.add_test_user(self.test_user)
    
    @pytest.mark.asyncio
    async def test_successful_authentication_flow(self):
        """Test complete successful authentication flow."""
        # Arrange
        login_request = UserLogin(
            email="user@test.com",
            password="correct_password",
            tenant_id=self.test_tenant["id"]
        )
        
        # Mock password verification to succeed
        with patch('app.utils.security.password_manager.verify_password', return_value=True), \
             patch('app.utils.security.token_manager.create_access_token', return_value="mock_access_token"), \
             patch('app.utils.security.token_manager.create_refresh_token', return_value="mock_refresh_token"):
            
            # Act
            result = await self.auth_service.authenticate(login_request)
        
        # Assert
        assert result.success is True
        assert result.user is not None
        assert result.tokens is not None
        assert result.error is None
        
        # Verify user data
        assert result.user.email == "user@test.com"
        assert result.user.tenant_id == self.test_tenant["id"]
        assert "user" in result.user.roles
        
        # Verify tokens
        assert result.tokens.access_token == "mock_access_token"
        assert result.tokens.refresh_token == "mock_refresh_token"
        assert result.tokens.token_type == "bearer"
        
        # Verify session creation
        assert self.mock_session_repo.get_call_count("create") == 1
        
        # Verify last login update
        assert self.mock_user_repo.was_called_with("update_last_login", self.test_user_id)
        
        # Verify failed login attempts were cleared
        cache_deletes = [op for op in self.mock_cache.operations if op[0] == "delete"]
        assert any("login_attempts" in str(op) for op in cache_deletes)
    
    @pytest.mark.asyncio
    async def test_authentication_with_invalid_credentials(self):
        """Test authentication with invalid password."""
        # Arrange
        login_request = UserLogin(
            email="user@test.com",
            password="wrong_password",
            tenant_id=self.test_tenant["id"]
        )
        
        # Mock password verification to fail
        with patch('app.utils.security.password_manager.verify_password', return_value=False):
            # Act
            result = await self.auth_service.authenticate(login_request)
        
        # Assert
        assert result.success is False
        assert result.user is None
        assert result.tokens is None
        assert result.error == "Invalid credentials"
        
        # Verify failed login attempt was recorded
        cache_sets = [op for op in self.mock_cache.operations if op[0] == "set"]
        login_attempt_sets = [op for op in cache_sets if "login_attempts" in str(op)]
        assert len(login_attempt_sets) > 0
        
        # Verify no session was created
        assert self.mock_session_repo.get_call_count("create") == 0
    
    @pytest.mark.asyncio
    async def test_authentication_with_nonexistent_user(self):
        """Test authentication with user that doesn't exist."""
        # Arrange
        login_request = UserLogin(
            email="nonexistent@test.com",
            password="password",
            tenant_id=self.test_tenant["id"]
        )
        
        # Act
        result = await self.auth_service.authenticate(login_request)
        
        # Assert
        assert result.success is False
        assert result.user is None
        assert result.tokens is None
        assert result.error == "Invalid credentials"
        
        # Verify failed login attempt was recorded
        cache_sets = [op for op in self.mock_cache.operations if op[0] == "set"]
        login_attempt_sets = [op for op in cache_sets if "login_attempts" in str(op)]
        assert len(login_attempt_sets) > 0
    
    @pytest.mark.asyncio
    async def test_authentication_with_inactive_user(self):
        """Test authentication with inactive user account."""
        # Arrange
        inactive_user = self.test_user.copy()
        inactive_user["id"] = "inactive-user"
        inactive_user["email"] = "inactive@test.com"
        inactive_user["is_active"] = False
        self.mock_user_repo.add_test_user(inactive_user)
        
        login_request = UserLogin(
            email="inactive@test.com",
            password="password",
            tenant_id=self.test_tenant["id"]
        )
        
        # Act
        result = await self.auth_service.authenticate(login_request)
        
        # Assert
        assert result.success is False
        assert result.user is None
        assert result.tokens is None
        assert result.error == "Account is not active"
    
    @pytest.mark.asyncio
    async def test_authentication_rate_limiting(self):
        """Test authentication rate limiting after multiple failed attempts."""
        # Arrange - Simulate multiple failed attempts
        login_request = UserLogin(
            email="user@test.com",
            password="wrong_password",
            tenant_id=self.test_tenant["id"]
        )
        
        # Mock cache to return high attempt count
        async def mock_cache_get(key):
            if "login_attempts" in key:
                return 10  # Over the limit
            return None
        
        self.mock_cache.get = mock_cache_get
        
        # Act
        result = await self.auth_service.authenticate(login_request)
        
        # Assert
        assert result.success is False
        assert result.error == "Too many login attempts. Please try again later."
        
        # Verify rate limit check was performed
        assert self.mock_cache.get_operation_count("get") > 0
    
    @pytest.mark.asyncio
    async def test_authentication_with_suspended_tenant(self):
        """Test authentication when tenant is suspended."""
        # Arrange
        suspended_tenant = TenantTestBuilder().as_suspended("Policy violation").build()
        self.mock_tenant_repo.add_test_tenant(suspended_tenant)
        
        suspended_user = self.test_user.copy()
        suspended_user["id"] = str(uuid4())
        suspended_user["email"] = "user@suspended.com"
        suspended_user["tenant_id"] = suspended_tenant["id"]
        self.mock_user_repo.add_test_user(suspended_user)
        
        login_request = UserLogin(
            email="user@suspended.com",
            password="password",
            tenant_id=suspended_tenant["id"]
        )
        
        # Mock password verification to succeed (but should fail due to suspended tenant)
        with patch('app.utils.security.password_manager.verify_password', return_value=True):
            # Act & Assert
            with pytest.raises(ValueError, match="suspended"):
                await self.auth_service.authenticate(login_request)


class TestPasswordResetWorkflow:
    """Test password reset workflows."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.mock_user_repo = MockUserRepository()
        self.mock_tenant_repo = MockTenantRepository()
        self.mock_cache = MockCacheService()
        self.mock_notification = MockNotificationService()
        
        self.auth_service = AuthenticationService(
            user_repository=self.mock_user_repo,
            tenant_repository=self.mock_tenant_repo,
            cache_manager=self.mock_cache,
            notification_service=self.mock_notification
        )
        
        # Setup test user with consistent IDs
        self.test_user_id = str(uuid4())
        self.test_tenant_id = str(uuid4())
        self.test_user = {
            "id": self.test_user_id,
            "tenant_id": self.test_tenant_id,
            "email": "user@test.com",
            "full_name": "Test User",
            "hashed_password": "$2b$12$old_password_hash"
        }
        self.mock_user_repo.add_test_user(self.test_user)
    
    @pytest.mark.asyncio
    async def test_successful_password_reset_request(self):
        """Test successful password reset request flow."""
        # Arrange
        reset_request = PasswordResetRequest(
            email="user@test.com",
            redirect_url="https://app.example.com/reset-password"
        )
        
        # Act
        result = await self.auth_service.request_password_reset(reset_request)
        
        # Assert
        assert result is not None
        assert "token" in result
        assert "reset_link" in result
        assert result["email"] == "user@test.com"
        assert result["email_sent"] is True
        
        # Verify email was sent
        assert self.mock_notification.verify_email_sent(
            "user@test.com", 
            subject_contains="password reset"
        )
        
        # Verify token was cached
        cache_sets = [op for op in self.mock_cache.operations if op[0] == "set"]
        reset_token_sets = [op for op in cache_sets if "password_reset:" in str(op)]
        assert len(reset_token_sets) >= 2  # Token and user token key
    
    @pytest.mark.asyncio
    async def test_password_reset_request_nonexistent_user(self):
        """Test password reset request for nonexistent user."""
        # Arrange
        reset_request = PasswordResetRequest(email="nonexistent@test.com")
        
        # Act
        result = await self.auth_service.request_password_reset(reset_request)
        
        # Assert - Should return None to prevent user enumeration
        assert result is None
        
        # Verify no email was sent
        assert len(self.mock_notification.sent_emails) == 0
    
    @pytest.mark.asyncio
    async def test_password_reset_request_throttling(self):
        """Test password reset request throttling."""
        # Arrange - Mock cache to show recent request
        async def mock_cache_exists(key):
            if "password_reset_throttle:" in key:
                return True
            return False
        
        self.mock_cache.exists = mock_cache_exists
        
        reset_request = PasswordResetRequest(email="user@test.com")
        
        # Act
        result = await self.auth_service.request_password_reset(reset_request)
        
        # Assert
        assert result is None  # Should be throttled
        
        # Verify no email was sent
        assert len(self.mock_notification.sent_emails) == 0
    
    @pytest.mark.asyncio
    async def test_password_reset_confirmation_success(self):
        """Test successful password reset confirmation."""
        # Arrange
        reset_token = "secure_reset_token_123"
        
        # Mock cache to return valid reset payload
        async def mock_cache_get(key):
            if "password_reset:" in key:
                return {
                    "user_id": self.test_user_id,
                    "tenant_id": self.test_tenant_id,
                    "email": "user@test.com",
                    "token_digest": "hashed_token"
                }
            return None
        
        self.mock_cache.get = mock_cache_get
        
        reset_confirm = PasswordResetConfirm(
            token=reset_token,
            new_password="NewSecurePassword123!"
        )
        
        # Mock password validation and hashing
        with patch('app.utils.security.password_manager.validate_password_strength', 
                  return_value={"valid": True, "errors": []}), \
             patch('app.utils.security.password_manager.hash_password', 
                  return_value="$2b$12$new_password_hash"):
            
            # Act
            result = await self.auth_service.confirm_password_reset(reset_confirm)
        
        # Assert
        assert result is True
        
        # Verify password was updated
        assert self.mock_user_repo.was_called_with("update", self.test_user_id)
        
        # Verify sessions were revoked
        assert self.mock_session_repo.was_called_with("delete_by_user", self.test_user_id)
        
        # Verify caches were cleared
        cache_deletes = [op for op in self.mock_cache.operations if op[0] == "delete"]
        assert len(cache_deletes) > 0
    
    @pytest.mark.asyncio
    async def test_password_reset_confirmation_invalid_token(self):
        """Test password reset confirmation with invalid token."""
        # Arrange
        reset_confirm = PasswordResetConfirm(
            token="invalid_token",
            new_password="NewSecurePassword123!"
        )
        
        # Mock cache to return no reset payload
        async def mock_cache_get(key):
            return None
        
        self.mock_cache.get = mock_cache_get
        
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid or expired reset token"):
            await self.auth_service.confirm_password_reset(reset_confirm)
        
        # Verify no password update occurred
        assert self.mock_user_repo.get_call_count("update") == 0
    
    @pytest.mark.asyncio
    async def test_password_reset_confirmation_weak_password(self):
        """Test password reset confirmation with weak password."""
        # Arrange
        reset_token = "valid_token"
        
        # Mock cache to return valid reset payload
        async def mock_cache_get(key):
            if "password_reset:" in key:
                return {
                    "user_id": self.test_user_id,
                    "tenant_id": self.test_tenant_id,
                    "email": "user@test.com"
                }
            return None
        
        self.mock_cache.get = mock_cache_get
        
        reset_confirm = PasswordResetConfirm(
            token=reset_token,
            new_password="weak"  # Weak password
        )
        
        # Mock password validation to fail
        with patch('app.utils.security.password_manager.validate_password_strength', 
                  return_value={"valid": False, "errors": ["Password too weak"]}):
            
            # Act & Assert
            with pytest.raises(ValueError, match="Password too weak"):
                await self.auth_service.confirm_password_reset(reset_confirm)
        
        # Verify no password update occurred
        assert self.mock_user_repo.get_call_count("update") == 0
    
    @pytest.mark.asyncio
    async def test_password_reset_email_failure_handling(self):
        """Test password reset when email sending fails."""
        # Arrange
        self.mock_notification.should_fail_email = True
        
        reset_request = PasswordResetRequest(email="user@test.com")
        
        # Act
        result = await self.auth_service.request_password_reset(reset_request)
        
        # Assert
        assert result is not None
        assert result["email_sent"] is False
        assert "token" in result  # Token should still be generated
        
        # Verify token was still cached even though email failed
        cache_sets = [op for op in self.mock_cache.operations if op[0] == "set"]
        reset_token_sets = [op for op in cache_sets if "password_reset:" in str(op)]
        assert len(reset_token_sets) >= 2


class TestTokenValidationWorkflow:
    """Test token validation workflows."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.mock_user_repo = MockUserRepository()
        self.mock_cache = MockCacheService()
        
        self.auth_service = AuthenticationService(
            user_repository=self.mock_user_repo,
            tenant_repository=None,
            cache_manager=self.mock_cache
        )
        
        # Setup test user with consistent IDs
        self.test_user_id = str(uuid4())
        self.test_tenant_id = str(uuid4())
        self.test_user = {
            "id": self.test_user_id,
            "tenant_id": self.test_tenant_id,
            "email": "user@test.com",
            "full_name": "Test User",
            "roles": ["user"],
            "permissions": ["read"],
            "is_active": True
        }
        self.mock_user_repo.add_test_user(self.test_user)
    
    @pytest.mark.asyncio
    async def test_valid_token_verification(self):
        """Test verification of valid token."""
        # Arrange
        valid_token = "valid_jwt_token"
        
        # Mock token verification and data extraction
        with patch('app.utils.security.token_manager.verify_token', 
                  return_value={"sub": self.test_user_id, "tenant_id": self.test_tenant_id}), \
             patch('app.utils.security.token_manager.extract_token_data') as mock_extract:
            
            # Mock token data
            from unittest.mock import Mock
            mock_token_data = Mock()
            mock_token_data.sub = self.test_user_id
            mock_token_data.tenant_id = self.test_tenant_id
            mock_token_data.email = "user@test.com"
            mock_token_data.roles = ["user"]
            mock_token_data.permissions = ["read"]
            mock_token_data.token_type = "access"
            mock_extract.return_value = mock_token_data
            
            # Act
            result = await self.auth_service.verify_token(valid_token)
        
        # Assert
        assert result is not None
        assert result["user_id"] == self.test_user_id
        assert result["email"] == "user@test.com"
        assert result["tenant_id"] == self.test_tenant_id
        assert "user" in result["roles"]
    
    @pytest.mark.asyncio
    async def test_blacklisted_token_rejection(self):
        """Test rejection of blacklisted token."""
        # Arrange
        blacklisted_token = "blacklisted_token"
        
        # Mock cache to show token is blacklisted
        async def mock_cache_exists(key):
            if "blacklist:" in key:
                return True
            return False
        
        self.mock_cache.exists = mock_cache_exists
        
        # Act
        result = await self.auth_service.verify_token(blacklisted_token)
        
        # Assert
        assert result is None
    
    @pytest.mark.asyncio
    async def test_expired_token_rejection(self):
        """Test rejection of expired token."""
        # Arrange
        expired_token = "expired_token"
        
        # Mock token verification to return None (expired)
        with patch('app.utils.security.token_manager.verify_token', return_value=None):
            # Act
            result = await self.auth_service.verify_token(expired_token)
        
        # Assert
        assert result is None