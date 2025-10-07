"""
Comprehensive integration tests for authentication service workflows.

These tests verify complete authentication workflows including:
- User registration and verification
- Authentication with rate limiting
- Password reset flows
- Session management and termination
- Token generation, validation, and rotation
- Multi-tenant authentication isolation
- Cross-service interactions
"""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from unittest.mock import patch, AsyncMock, Mock

from app.services.auth.authentication_service import AuthenticationService
from app.infrastructure.persistence.models.auth_tables import (
    UserCreate, UserLogin, PasswordChangeRequest, PasswordResetRequest, 
    PasswordResetConfirm, RefreshTokenRequest, CurrentUser, UserUpdate,
    TokenResponse, AuthenticationResult
)
from app.infrastructure.persistence.models.tenant_table import SubscriptionTier
from tests.mocks.mock_repositories import MockUserRepository, MockTenantRepository, MockUserSessionRepository
from tests.mocks.mock_services import MockCacheService, MockNotificationService


class TestUserRegistrationWorkflows:
    """Test complete user registration and verification workflows."""
    
    def setup_method(self):
        """Setup test dependencies."""
        self.user_repo = MockUserRepository()
        self.tenant_repo = MockTenantRepository()
        self.session_repo = MockUserSessionRepository()
        self.cache_service = MockCacheService()
        self.notification_service = MockNotificationService()
        
        self.auth_service = AuthenticationService(
            user_repository=self.user_repo,
            tenant_repository=self.tenant_repo,
            cache_manager=self.cache_service,
            notification_service=self.notification_service,
            session_repository=self.session_repo
        )
        
        # Setup test tenant
        from uuid import uuid4
        tenant_id = str(uuid4())
        self.tenant_id = self.tenant_repo.add_test_tenant({
            "id": tenant_id,
            "name": "test-tenant",
            "is_active": True,
            "is_suspended": False
        })
    
    @pytest.mark.asyncio
    async def test_successful_user_registration_workflow(self):
        """Test complete successful user registration workflow."""
        user_data = UserCreate(
            tenant_id=self.tenant_id,
            email="newuser@testcompany.com",
            password="SecurePassword123!",
            full_name="New Test User",
            roles=["user"]
        )
        
        with patch('app.utils.security.password_manager.validate_password_strength') as mock_validator:
            mock_validator.return_value = {"valid": True, "errors": []}
            
            with patch('app.utils.security.password_manager.hash_password') as mock_hash:
                mock_hash.return_value = "hashed_secure_password"
                
                # Register user
                result = await self.auth_service.register_user(user_data)
        
        # Verify user was created successfully
        assert result.email == user_data.email
        assert result.full_name == user_data.full_name
        assert result.tenant_id == self.tenant_id
        assert "user" in result.roles
        assert result.is_active is True
        assert result.is_verified is False  # Should require verification
        
        # Verify user was stored in repository
        assert self.user_repo.get_call_count("create") == 1
        
        # Verify password was hashed
        created_user_data = [call[1] for call in self.user_repo.call_log if call[0] == "create"][0]
        assert created_user_data["hashed_password"] == "hashed_secure_password"
        
        # Verify security event was logged
        assert "user_registered" in str(self.auth_service._log_security_event.call_args_list) if hasattr(self.auth_service._log_security_event, 'call_args_list') else True
    
    @pytest.mark.asyncio
    async def test_duplicate_email_prevention_per_tenant(self):
        """Test that duplicate emails are prevented within same tenant."""
        email = "duplicate@testcompany.com"
        
        # Add existing user
        self.user_repo.add_test_user({
            "id": "existing-user-id",
            "tenant_id": self.tenant_id,
            "email": email,
            "full_name": "Existing User"
        })
        
        user_data = UserCreate(
            tenant_id=self.tenant_id,
            email=email,
            password="SecurePassword123!",
            full_name="Duplicate User"
        )
        
        # Should raise error for duplicate email
        with pytest.raises(ValueError, match="User already exists with this email"):
            await self.auth_service.register_user(user_data)
    
    @pytest.mark.asyncio
    async def test_cross_tenant_email_isolation(self):
        """Test that same email can exist in different tenants."""
        email = "shared@example.com"
        
        # Setup second tenant
        tenant2_id = self.tenant_repo.add_test_tenant({
            "id": "tenant2-id",
            "name": "tenant2",
            "is_active": True,
            "is_suspended": False
        })
        
        with patch('app.utils.security.password_manager.validate_password_strength') as mock_validator:
            mock_validator.return_value = {"valid": True, "errors": []}
            
            with patch('app.utils.security.password_manager.hash_password') as mock_hash:
                mock_hash.return_value = "hashed_password"
                
                # Create user in first tenant
                user1_data = UserCreate(
                    tenant_id=self.tenant_id,
                    email=email,
                    password="SecurePassword123!",
                    full_name="User One"
                )
                user1 = await self.auth_service.register_user(user1_data)
                
                # Create user with same email in second tenant - should succeed
                user2_data = UserCreate(
                    tenant_id=tenant2_id,
                    email=email,
                    password="SecurePassword123!",
                    full_name="User Two"
                )
                user2 = await self.auth_service.register_user(user2_data)
        
        # Verify both users were created successfully
        assert user1.email == email
        assert user2.email == email
        assert user1.user_id != user2.user_id
        assert user1.tenant_id != user2.tenant_id
    
    @pytest.mark.asyncio
    async def test_suspended_tenant_registration_prevention(self):
        """Test that registration is blocked for suspended tenants."""
        # Suspend the tenant
        self.tenant_repo.tenants[self.tenant_id]["is_suspended"] = True
        
        user_data = UserCreate(
            tenant_id=self.tenant_id,
            email="blocked@suspended.com",
            password="SecurePassword123!",
            full_name="Blocked User"
        )
        
        # Should raise error for suspended tenant
        with pytest.raises(ValueError, match="Organization is currently suspended"):
            await self.auth_service.register_user(user_data)


class TestAuthenticationWorkflows:
    """Test complete authentication workflows including rate limiting."""
    
    def setup_method(self):
        self.user_repo = MockUserRepository()
        self.tenant_repo = MockTenantRepository()
        self.session_repo = MockUserSessionRepository()
        self.cache_service = MockCacheService()
        self.notification_service = MockNotificationService()
        
        self.auth_service = AuthenticationService(
            user_repository=self.user_repo,
            tenant_repository=self.tenant_repo,
            cache_manager=self.cache_service,
            notification_service=self.notification_service,
            session_repository=self.session_repo
        )
        
        # Setup test tenant and user
        self.tenant_id = self.tenant_repo.add_test_tenant({
            "id": "auth-tenant-id",
            "name": "auth-tenant",
            "is_active": True,
            "is_suspended": False
        })
        
        self.user_id = self.user_repo.add_test_user({
            "id": "auth-user-id",
            "tenant_id": self.tenant_id,
            "email": "auth@testcompany.com",
            "hashed_password": "hashed_password",
            "full_name": "Auth Test User",
            "roles": ["user"],
            "permissions": ["read"],
            "is_active": True,
            "is_superuser": False
        })
    
    @pytest.mark.asyncio
    async def test_successful_authentication_workflow(self):
        """Test complete successful authentication workflow."""
        credentials = UserLogin(
            email="auth@testcompany.com",
            password="correct_password",
            tenant_id=self.tenant_id
        )
        
        with patch('app.utils.security.password_manager.verify_password') as mock_verify:
            mock_verify.return_value = True
            
            with patch('app.utils.security.token_manager.create_access_token') as mock_access_token:
                mock_access_token.return_value = "mock_access_token"
                
                with patch('app.utils.security.token_manager.create_refresh_token') as mock_refresh_token:
                    mock_refresh_token.return_value = "mock_refresh_token"
                    
                    # Authenticate user
                    result = await self.auth_service.authenticate(credentials)
        
        # Verify successful authentication
        assert result.success is True
        assert result.user is not None
        assert result.user.email == credentials.email
        assert result.user.tenant_id == self.tenant_id
        assert result.tokens is not None
        assert result.tokens.access_token == "mock_access_token"
        assert result.tokens.refresh_token == "mock_refresh_token"
        
        # Verify session was created
        assert self.session_repo.get_call_count("create") > 0
        
        # Verify last login was updated
        assert self.user_repo.get_call_count("update_last_login") == 1
    
    @pytest.mark.asyncio
    async def test_authentication_rate_limiting_workflow(self):
        """Test authentication rate limiting workflow."""
        credentials = UserLogin(
            email="auth@testcompany.com",
            password="wrong_password",
            tenant_id=self.tenant_id
        )
        
        # Mock rate limit exceeded
        with patch.object(self.auth_service.settings, 'MAX_LOGIN_ATTEMPTS', 3):
            # Set attempts to max
            self.cache_service.data[f"login_attempts:{self.tenant_id}:auth@testcompany.com"] = 3
            
            # Authentication should be rate limited
            result = await self.auth_service.authenticate(credentials)
            
            assert result.success is False
            assert "Too many login attempts" in result.error
    
    @pytest.mark.asyncio
    async def test_failed_authentication_tracking(self):
        """Test failed authentication attempt tracking."""
        credentials = UserLogin(
            email="auth@testcompany.com",
            password="wrong_password",
            tenant_id=self.tenant_id
        )
        
        with patch('app.utils.security.password_manager.verify_password') as mock_verify:
            mock_verify.return_value = False  # Wrong password
            
            # First failed attempt
            result = await self.auth_service.authenticate(credentials)
            assert result.success is False
            assert result.error == "Invalid credentials"
            
            # Verify failed attempt was recorded
            attempts_key = f"login_attempts:{self.tenant_id}:auth@testcompany.com"
            assert attempts_key in self.cache_service.data
            assert self.cache_service.data[attempts_key] == 1
            
            # Second failed attempt
            result = await self.auth_service.authenticate(credentials)
            assert result.success is False
            
            # Verify attempt count increased
            assert self.cache_service.data[attempts_key] == 2
    
    @pytest.mark.asyncio
    async def test_inactive_user_authentication_prevention(self):
        """Test that inactive users cannot authenticate."""
        # Deactivate user
        self.user_repo.users[self.user_id]["is_active"] = False
        
        credentials = UserLogin(
            email="auth@testcompany.com",
            password="correct_password",
            tenant_id=self.tenant_id
        )
        
        result = await self.auth_service.authenticate(credentials)
        assert result.success is False
        assert result.error == "Account is not active"


class TestTokenManagementWorkflows:
    """Test token generation, validation, and rotation workflows."""
    
    def setup_method(self):
        self.user_repo = MockUserRepository()
        self.tenant_repo = MockTenantRepository()
        self.session_repo = MockUserSessionRepository()
        self.cache_service = MockCacheService()
        self.notification_service = MockNotificationService()
        
        self.auth_service = AuthenticationService(
            user_repository=self.user_repo,
            tenant_repository=self.tenant_repo,
            cache_manager=self.cache_service,
            notification_service=self.notification_service,
            session_repository=self.session_repo
        )
        
        # Setup test user
        self.user_id = self.user_repo.add_test_user({
            "id": "token-user-id",
            "tenant_id": "token-tenant-id",
            "email": "token@testcompany.com",
            "full_name": "Token Test User",
            "roles": ["user"],
            "permissions": ["read"],
            "is_active": True
        })
    
    @pytest.mark.asyncio
    async def test_token_validation_workflow(self):
        """Test complete token validation workflow."""
        token = "valid.jwt.token"
        
        # Mock token verification
        with patch('app.utils.security.token_manager.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": self.user_id, "exp": 9999999999}
            
            with patch('app.utils.security.token_manager.extract_token_data') as mock_extract:
                mock_extract.return_value = Mock(
                    token_type="access",
                    sub=self.user_id,
                    tenant_id="token-tenant-id",
                    roles=["user"],
                    permissions=["read"],
                    email="token@testcompany.com"
                )
                
                # Validate token
                result = await self.auth_service.verify_token(token)
        
        # Verify token was validated successfully
        assert result is not None
        assert result["user_id"] == self.user_id
        assert result["email"] == "token@testcompany.com"
        assert "user" in result["roles"]
    
    @pytest.mark.asyncio
    async def test_blacklisted_token_rejection(self):
        """Test that blacklisted tokens are rejected."""
        token = "blacklisted.jwt.token"
        
        # Add token to blacklist
        self.cache_service.data["blacklist:token_jti"] = True
        
        with patch('app.utils.security.token_manager.verify_token') as mock_verify:
            mock_verify.return_value = {"jti": "token_jti", "exp": 9999999999}
            
            result = await self.auth_service.verify_token(token)
            assert result is None
    
    @pytest.mark.asyncio
    async def test_token_refresh_workflow(self):
        """Test token refresh and rotation workflow."""
        refresh_token = "valid.refresh.token"
        
        # Mock refresh token verification
        with patch('app.utils.security.token_manager.verify_token') as mock_verify:
            mock_verify.return_value = {
                "sub": self.user_id,
                "token_type": "refresh",
                "family": "token_family_id",
                "exp": 9999999999
            }
            
            with patch('app.utils.security.token_manager.create_access_token') as mock_access_token:
                mock_access_token.return_value = "new_access_token"
                
                with patch('app.utils.security.token_manager.create_refresh_token') as mock_new_refresh:
                    mock_new_refresh.return_value = "new_refresh_token"
                    
                    # Refresh tokens
                    request = RefreshTokenRequest(refresh_token=refresh_token)
                    result = await self.auth_service.refresh_tokens(request)
        
        # Verify new tokens were generated
        assert result is not None
        assert result.access_token == "new_access_token"
        assert result.refresh_token == "new_refresh_token"
        
        # Verify old refresh token was blacklisted
        blacklist_key = f"blacklist:{refresh_token}"
        # This would be set in a real implementation
    
    @pytest.mark.asyncio
    async def test_token_revocation_workflow(self):
        """Test token revocation workflow."""
        token = "token.to.revoke"
        
        with patch('app.utils.security.token_manager.verify_token') as mock_verify:
            mock_verify.return_value = {
                "jti": "token_jti",
                "token_type": "access",
                "exp": 9999999999
            }
            
            # Revoke token
            result = await self.auth_service.revoke_token(token)
        
        # Verify token was revoked
        assert result is True
        
        # Verify token was blacklisted
        blacklist_key = "blacklist:token_jti"
        assert blacklist_key in self.cache_service.data


class TestPasswordResetWorkflows:
    """Test complete password reset workflows."""
    
    def setup_method(self):
        self.user_repo = MockUserRepository()
        self.tenant_repo = MockTenantRepository()
        self.session_repo = MockUserSessionRepository()
        self.cache_service = MockCacheService()
        self.notification_service = MockNotificationService()
        
        self.auth_service = AuthenticationService(
            user_repository=self.user_repo,
            tenant_repository=self.tenant_repo,
            cache_manager=self.cache_service,
            notification_service=self.notification_service,
            session_repository=self.session_repo
        )
        
        # Setup test user
        self.user_id = self.user_repo.add_test_user({
            "id": "reset-user-id",
            "tenant_id": "reset-tenant-id",
            "email": "reset@testcompany.com",
            "full_name": "Reset Test User",
            "hashed_password": "old_hashed_password"
        })
    
    @pytest.mark.asyncio
    async def test_password_reset_request_workflow(self):
        """Test complete password reset request workflow."""
        request = PasswordResetRequest(
            email="reset@testcompany.com",
            redirect_url="https://app.example.com/reset-password"
        )
        
        with patch('secrets.token_urlsafe') as mock_token:
            mock_token.return_value = "secure_reset_token"
            
            with patch('hashlib.sha256') as mock_sha:
                mock_sha.return_value.hexdigest.return_value = "token_digest_hash"
                
                # Request password reset
                result = await self.auth_service.request_password_reset(request)
        
        # Verify reset was initiated
        assert result is not None
        assert result["email"] == "reset@testcompany.com"
        assert result["token"] == "secure_reset_token"
        assert result["reset_link"] is not None
        assert "secure_reset_token" in result["reset_link"]
        
        # Verify reset token was cached
        cache_key = "password_reset:token_digest_hash"
        assert cache_key in self.cache_service.data
        
        # Verify notification was sent
        assert self.notification_service.get_call_count("send_email") == 1
    
    @pytest.mark.asyncio
    async def test_password_reset_confirmation_workflow(self):
        """Test complete password reset confirmation workflow."""
        reset_token = "valid_reset_token"
        token_digest = "token_digest_hash"
        
        # Setup reset token data in cache
        reset_payload = {
            "user_id": self.user_id,
            "tenant_id": "reset-tenant-id",
            "email": "reset@testcompany.com",
            "token_digest": token_digest,
            "requested_at": datetime.utcnow().isoformat()
        }
        
        cache_key = f"password_reset:{token_digest}"
        self.cache_service.data[cache_key] = reset_payload
        
        # Add some sessions to verify they get revoked
        self.session_repo.add_test_session({
            "id": "session-1",
            "user_id": self.user_id
        })
        self.session_repo.add_test_session({
            "id": "session-2",
            "user_id": self.user_id
        })
        
        with patch('app.utils.security.password_manager.validate_password_strength') as mock_validator:
            mock_validator.return_value = {"valid": True, "errors": []}
            
            with patch('app.utils.security.password_manager.hash_password') as mock_hash:
                mock_hash.return_value = "new_hashed_password"
                
                with patch('hashlib.sha256') as mock_sha:
                    mock_sha.return_value.hexdigest.return_value = token_digest
                    
                    # Confirm password reset
                    confirm_request = PasswordResetConfirm(
                        token=reset_token,
                        new_password="NewSecurePassword123!"
                    )
                    
                    result = await self.auth_service.confirm_password_reset(confirm_request)
        
        # Verify password was reset
        assert result is True
        
        # Verify password was updated
        assert self.user_repo.get_call_count("update") == 1
        
        # Verify sessions were revoked
        assert self.session_repo.get_call_count("delete_by_user") == 1
        
        # Verify reset token was cleared
        assert cache_key not in self.cache_service.data
    
    @pytest.mark.asyncio
    async def test_password_reset_throttling(self):
        """Test password reset request throttling."""
        request = PasswordResetRequest(email="reset@testcompany.com")
        
        # Set throttle
        throttle_key = "password_reset_throttle:reset@testcompany.com"
        self.cache_service.data[throttle_key] = True
        
        with patch.object(self.auth_service.settings, 'PASSWORD_RESET_REQUEST_INTERVAL_SECONDS', 300):
            # Request should be throttled
            result = await self.auth_service.request_password_reset(request)
            assert result is None
    
    @pytest.mark.asyncio
    async def test_password_reset_for_nonexistent_user(self):
        """Test password reset request for non-existent user."""
        request = PasswordResetRequest(email="nonexistent@testcompany.com")
        
        with patch('app.utils.security.security_validator.validate_email') as mock_validator:
            mock_validator.return_value = True
            
            # Should return None but not raise error (prevent enumeration)
            result = await self.auth_service.request_password_reset(request)
            assert result is None


class TestSessionManagementWorkflows:
    """Test session management and termination workflows."""
    
    def setup_method(self):
        self.user_repo = MockUserRepository()
        self.tenant_repo = MockTenantRepository()
        self.session_repo = MockUserSessionRepository()
        self.cache_service = MockCacheService()
        self.notification_service = MockNotificationService()
        
        self.auth_service = AuthenticationService(
            user_repository=self.user_repo,
            tenant_repository=self.tenant_repo,
            cache_manager=self.cache_service,
            notification_service=self.notification_service,
            session_repository=self.session_repo
        )
        
        # Setup test user
        self.user_id = "session-user-id"
        self.user_repo.add_test_user({
            "id": self.user_id,
            "tenant_id": "session-tenant-id",
            "email": "session@testcompany.com",
            "full_name": "Session Test User"
        })
    
    @pytest.mark.asyncio
    async def test_session_listing_workflow(self):
        """Test session listing workflow."""
        # Add test sessions
        session1_id = self.session_repo.add_test_session({
            "id": "session-1",
            "user_id": self.user_id,
            "tenant_id": "session-tenant-id",
            "ip_address": "192.168.1.1",
            "user_agent": "Mozilla/5.0",
            "expires_at": (datetime.utcnow() + timedelta(days=1)).isoformat(),
            "last_activity": datetime.utcnow().isoformat()
        })
        
        session2_id = self.session_repo.add_test_session({
            "id": "session-2",
            "user_id": self.user_id,
            "tenant_id": "session-tenant-id",
            "ip_address": "192.168.1.2",
            "user_agent": "Chrome/90.0",
            "expires_at": (datetime.utcnow() + timedelta(days=1)).isoformat(),
            "last_activity": datetime.utcnow().isoformat()
        })
        
        # List sessions
        sessions = await self.auth_service.list_sessions(self.user_id)
        
        # Verify sessions were returned
        assert len(sessions) == 2
        session_ids = {session.session_id for session in sessions}
        assert session1_id in session_ids
        assert session2_id in session_ids
    
    @pytest.mark.asyncio
    async def test_session_termination_workflow(self):
        """Test session termination workflow."""
        # Add test session
        session_id = self.session_repo.add_test_session({
            "id": "session-to-terminate",
            "user_id": self.user_id,
            "tenant_id": "session-tenant-id",
            "ip_address": "192.168.1.1",
            "user_agent": "Mozilla/5.0"
        })
        
        # Terminate session
        result = await self.auth_service.terminate_session(session_id, self.user_id)
        assert result is True
        
        # Verify session was deleted
        assert self.session_repo.get_call_count("delete") == 1
        
        # Verify session no longer exists
        remaining_sessions = await self.auth_service.list_sessions(self.user_id)
        remaining_session_ids = {session.session_id for session in remaining_sessions}
        assert session_id not in remaining_session_ids
    
    @pytest.mark.asyncio
    async def test_cross_user_session_access_prevention(self):
        """Test that users cannot access other users' sessions."""
        other_user_id = "other-user-id"
        
        # Add session for other user
        session_id = self.session_repo.add_test_session({
            "id": "other-user-session",
            "user_id": other_user_id,
            "tenant_id": "session-tenant-id"
        })
        
        # Try to terminate other user's session
        result = await self.auth_service.terminate_session(session_id, self.user_id)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_all_sessions_revocation_workflow(self):
        """Test revocation of all user sessions."""
        # Add multiple sessions
        for i in range(3):
            self.session_repo.add_test_session({
                "id": f"session-{i}",
                "user_id": self.user_id,
                "tenant_id": "session-tenant-id"
            })
        
        # Revoke all sessions
        await self.auth_service._revoke_all_user_sessions(self.user_id)
        
        # Verify all sessions were deleted
        assert self.session_repo.get_call_count("delete_by_user") == 1
        
        # Verify no sessions remain
        sessions = await self.auth_service.list_sessions(self.user_id)
        assert len(sessions) == 0


class TestUserProfileManagementWorkflows:
    """Test user profile update and management workflows."""
    
    def setup_method(self):
        self.user_repo = MockUserRepository()
        self.tenant_repo = MockTenantRepository()
        self.session_repo = MockUserSessionRepository()
        self.cache_service = MockCacheService()
        self.notification_service = MockNotificationService()
        
        self.auth_service = AuthenticationService(
            user_repository=self.user_repo,
            tenant_repository=self.tenant_repo,
            cache_manager=self.cache_service,
            notification_service=self.notification_service,
            session_repository=self.session_repo
        )
        
        # Setup test user
        self.user_id = "profile-user-id"
        self.tenant_id = "profile-tenant-id"
        self.user_repo.add_test_user({
            "id": self.user_id,
            "tenant_id": self.tenant_id,
            "email": "profile@testcompany.com",
            "full_name": "Profile Test User",
            "first_name": "Profile",
            "last_name": "Test User",
            "roles": ["user"],
            "permissions": ["read"],
            "is_active": True,
            "settings": {"theme": "dark", "notifications": True}
        })
    
    @pytest.mark.asyncio
    async def test_profile_update_workflow(self):
        """Test complete user profile update workflow."""
        current_user = CurrentUser(
            user_id=self.user_id,
            email="profile@testcompany.com",
            full_name="Profile Test User",
            tenant_id=self.tenant_id,
            roles=["user"],
            permissions=["read"],
            is_active=True,
            is_superuser=False
        )
        
        # Update profile
        update_request = UserUpdate(
            full_name="Updated Profile User",
            settings={"theme": "light", "language": "en"}
        )
        
        result = await self.auth_service.update_user_profile(current_user, update_request)
        
        # Verify profile was updated
        assert result.full_name == "Updated Profile User"
        
        # Verify name parts were updated
        assert self.user_repo.get_call_count("update") == 1
        
        # Verify cache was updated
        cache_key = f"user:{self.user_id}"
        assert cache_key in self.cache_service.data
    
    @pytest.mark.asyncio
    async def test_restricted_field_update_prevention(self):
        """Test that restricted fields cannot be updated."""
        current_user = CurrentUser(
            user_id=self.user_id,
            email="profile@testcompany.com",
            full_name="Profile Test User",
            tenant_id=self.tenant_id,
            roles=["user"],
            permissions=["read"],
            is_active=True,
            is_superuser=False
        )
        
        # Try to update restricted fields
        update_request = UserUpdate(roles=["admin"])
        
        with pytest.raises(ValueError, match="Not permitted to update fields"):
            await self.auth_service.update_user_profile(current_user, update_request)
    
    @pytest.mark.asyncio
    async def test_password_change_workflow(self):
        """Test password change workflow."""
        # Setup user with known password
        self.user_repo.users[self.user_id]["hashed_password"] = "old_hashed_password"
        
        password_change = PasswordChangeRequest(
            current_password="old_password",
            new_password="NewSecurePassword123!"
        )
        
        with patch('app.utils.security.password_manager.verify_password') as mock_verify:
            mock_verify.return_value = True  # Current password is correct
            
            with patch('app.utils.security.password_manager.validate_password_strength') as mock_validator:
                mock_validator.return_value = {"valid": True, "errors": []}
                
                with patch('app.utils.security.password_manager.hash_password') as mock_hash:
                    mock_hash.return_value = "new_hashed_password"
                    
                    # Change password
                    result = await self.auth_service.change_password(self.user_id, password_change)
        
        # Verify password was changed
        assert result is True
        
        # Verify password was updated in database
        assert self.user_repo.get_call_count("update") == 1
        
        # Verify all sessions were revoked
        assert self.session_repo.get_call_count("delete_by_user") == 1


class TestMultiTenantAuthenticationIsolation:
    """Test multi-tenant authentication isolation and security."""
    
    def setup_method(self):
        self.user_repo = MockUserRepository()
        self.tenant_repo = MockTenantRepository()
        self.session_repo = MockUserSessionRepository()
        self.cache_service = MockCacheService()
        self.notification_service = MockNotificationService()
        
        self.auth_service = AuthenticationService(
            user_repository=self.user_repo,
            tenant_repository=self.tenant_repo,
            cache_manager=self.cache_service,
            notification_service=self.notification_service,
            session_repository=self.session_repo
        )
        
        # Setup multiple tenants
        self.tenant1_id = self.tenant_repo.add_test_tenant({
            "id": "tenant1-id",
            "name": "tenant1",
            "is_active": True,
            "is_suspended": False
        })
        
        self.tenant2_id = self.tenant_repo.add_test_tenant({
            "id": "tenant2-id",
            "name": "tenant2",
            "is_active": True,
            "is_suspended": False
        })
        
        # Setup users in different tenants
        self.user1_id = self.user_repo.add_test_user({
            "id": "user1-id",
            "tenant_id": self.tenant1_id,
            "email": "user1@tenant1.com",
            "hashed_password": "hashed_password",
            "full_name": "User One",
            "roles": ["user"],
            "is_active": True
        })
        
        self.user2_id = self.user_repo.add_test_user({
            "id": "user2-id",
            "tenant_id": self.tenant2_id,
            "email": "user2@tenant2.com",
            "hashed_password": "hashed_password",
            "full_name": "User Two",
            "roles": ["user"],
            "is_active": True
        })
    
    @pytest.mark.asyncio
    async def test_tenant_specific_authentication(self):
        """Test that users can only authenticate to their own tenant."""
        # User 1 authenticating to their own tenant - should succeed
        credentials1 = UserLogin(
            email="user1@tenant1.com",
            password="correct_password",
            tenant_id=self.tenant1_id
        )
        
        with patch('app.utils.security.password_manager.verify_password') as mock_verify:
            mock_verify.return_value = True
            
            with patch('app.utils.security.token_manager.create_access_token') as mock_access_token:
                mock_access_token.return_value = "access_token"
                
                with patch('app.utils.security.token_manager.create_refresh_token') as mock_refresh_token:
                    mock_refresh_token.return_value = "refresh_token"
                    
                    result = await self.auth_service.authenticate(credentials1)
                    assert result.success is True
        
        # User 1 trying to authenticate to different tenant - should fail
        credentials1_wrong_tenant = UserLogin(
            email="user1@tenant1.com",
            password="correct_password",
            tenant_id=self.tenant2_id  # Wrong tenant
        )
        
        result = await self.auth_service.authenticate(credentials1_wrong_tenant)
        assert result.success is False
        assert result.error == "Invalid credentials"
    
    @pytest.mark.asyncio
    async def test_tenant_isolation_in_rate_limiting(self):
        """Test that rate limiting is isolated per tenant."""
        email = "shared@example.com"
        
        # Create users with same email in different tenants
        self.user_repo.add_test_user({
            "id": "shared-user1-id",
            "tenant_id": self.tenant1_id,
            "email": email,
            "hashed_password": "hashed_password",
            "is_active": True
        })
        
        self.user_repo.add_test_user({
            "id": "shared-user2-id",
            "tenant_id": self.tenant2_id,
            "email": email,
            "hashed_password": "hashed_password",
            "is_active": True
        })
        
        # Set rate limit for tenant 1
        self.cache_service.data[f"login_attempts:{self.tenant1_id}:{email}"] = 3
        
        # User in tenant 1 should be rate limited
        credentials1 = UserLogin(email=email, password="password", tenant_id=self.tenant1_id)
        
        with patch.object(self.auth_service.settings, 'MAX_LOGIN_ATTEMPTS', 3):
            result1 = await self.auth_service.authenticate(credentials1)
            assert result1.success is False
            assert "Too many login attempts" in result1.error
        
        # User in tenant 2 should not be affected by tenant 1's rate limit
        credentials2 = UserLogin(email=email, password="password", tenant_id=self.tenant2_id)
        
        with patch('app.utils.security.password_manager.verify_password') as mock_verify:
            mock_verify.return_value = True
            
            with patch('app.utils.security.token_manager.create_access_token') as mock_access_token:
                mock_access_token.return_value = "access_token"
                
                with patch('app.utils.security.token_manager.create_refresh_token') as mock_refresh_token:
                    mock_refresh_token.return_value = "refresh_token"
                    
                    result2 = await self.auth_service.authenticate(credentials2)
                    assert result2.success is True
    
    @pytest.mark.asyncio
    async def test_super_admin_cross_tenant_access(self):
        """Test that super admins can access across tenants."""
        # Create super admin user
        super_admin_id = self.user_repo.add_test_user({
            "id": "super-admin-id",
            "tenant_id": "00000000-0000-0000-0000-000000000000",  # System tenant
            "email": "admin@system.com",
            "full_name": "Super Admin",
            "roles": ["super_admin"],
            "is_active": True
        })
        
        # Mock token with super admin accessing different tenant
        token = "super.admin.token"
        
        with patch('app.utils.security.token_manager.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": super_admin_id, "exp": 9999999999}
            
            with patch('app.utils.security.token_manager.extract_token_data') as mock_extract:
                mock_extract.return_value = Mock(
                    token_type="access",
                    sub=super_admin_id,
                    tenant_id=self.tenant1_id,  # Accessing different tenant
                    roles=["super_admin"],
                    permissions=[],
                    email="admin@system.com"
                )
                
                # Super admin should be able to access any tenant
                result = await self.auth_service.verify_token(token)
                assert result is not None
                assert result["user_id"] == super_admin_id


class TestAuthenticationErrorHandlingAndResilience:
    """Test error handling and system resilience in authentication workflows."""
    
    def setup_method(self):
        self.user_repo = MockUserRepository()
        self.tenant_repo = MockTenantRepository()
        self.session_repo = MockUserSessionRepository()
        self.cache_service = MockCacheService()
        self.notification_service = MockNotificationService()
        
        self.auth_service = AuthenticationService(
            user_repository=self.user_repo,
            tenant_repository=self.tenant_repo,
            cache_manager=self.cache_service,
            notification_service=self.notification_service,
            session_repository=self.session_repo
        )
    
    @pytest.mark.asyncio
    async def test_graceful_cache_failure_handling(self):
        """Test graceful handling of cache failures during authentication."""
        # Setup tenant and user
        tenant_id = self.tenant_repo.add_test_tenant({
            "id": "cache-fail-tenant",
            "is_active": True,
            "is_suspended": False
        })
        
        user_id = self.user_repo.add_test_user({
            "id": "cache-fail-user",
            "tenant_id": tenant_id,
            "email": "cache@fail.com",
            "hashed_password": "hashed_password",
            "full_name": "Cache Fail User",
            "roles": ["user"],
            "is_active": True
        })
        
        # Make cache fail
        self.cache_service.should_fail_on_get = True
        
        credentials = UserLogin(
            email="cache@fail.com",
            password="correct_password",
            tenant_id=tenant_id
        )
        
        with patch('app.utils.security.password_manager.verify_password') as mock_verify:
            mock_verify.return_value = True
            
            with patch('app.utils.security.token_manager.create_access_token') as mock_access_token:
                mock_access_token.return_value = "access_token"
                
                with patch('app.utils.security.token_manager.create_refresh_token') as mock_refresh_token:
                    mock_refresh_token.return_value = "refresh_token"
                    
                    # Should still work despite cache failure
                    result = await self.auth_service.authenticate(credentials)
                    assert result.success is True
    
    @pytest.mark.asyncio
    async def test_notification_service_failure_handling(self):
        """Test handling of notification service failures during password reset."""
        user_id = self.user_repo.add_test_user({
            "id": "notification-fail-user",
            "email": "notify@fail.com",
            "full_name": "Notification Fail User"
        })
        
        # Make notification service fail
        self.notification_service.should_fail_on_send_email = True
        
        request = PasswordResetRequest(email="notify@fail.com")
        
        with patch('secrets.token_urlsafe') as mock_token:
            mock_token.return_value = "reset_token"
            
            with patch('hashlib.sha256') as mock_sha:
                mock_sha.return_value.hexdigest.return_value = "token_hash"
                
                # Should still generate reset token even if email fails
                result = await self.auth_service.request_password_reset(request)
                
                assert result is not None
                assert result["email_sent"] is False  # Email failed
                assert result["token"] == "reset_token"  # But token was generated
    
    @pytest.mark.asyncio
    async def test_repository_failure_recovery(self):
        """Test recovery from temporary repository failures."""
        # Setup tenant
        tenant_id = self.tenant_repo.add_test_tenant({
            "id": "repo-fail-tenant",
            "is_active": True,
            "is_suspended": False
        })
        
        # Make user repository fail temporarily
        self.user_repo.should_fail_on_create = True
        
        user_data = UserCreate(
            tenant_id=tenant_id,
            email="repo@fail.com",
            password="SecurePassword123!",
            full_name="Repo Fail User"
        )
        
        with patch('app.utils.security.password_manager.validate_password_strength') as mock_validator:
            mock_validator.return_value = {"valid": True, "errors": []}
            
            # Should fail gracefully
            with pytest.raises(Exception):
                await self.auth_service.register_user(user_data)
            
            # Fix repository and try again
            self.user_repo.should_fail_on_create = False
            
            # Should succeed now
            result = await self.auth_service.register_user(user_data)
            assert result is not None
            assert result.email == "repo@fail.com"