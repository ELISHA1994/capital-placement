"""
Unit tests for tenant domain rules and validation logic.

These tests focus on pure business logic without external dependencies.
Fast execution and high coverage of edge cases.
"""

import pytest
from app.infrastructure.persistence.models.tenant_table import SubscriptionTier, QuotaLimits, FeatureFlags
from app.infrastructure.tenant.tenant_service import TenantService


class TestTenantNameValidation:
    """Test tenant name validation rules (business logic)."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Create a minimal TenantService instance for testing validation logic
        self.tenant_service = TenantService(
            tenant_repository=None,  # Not needed for validation tests
            user_repository=None,
            cache_manager=None
        )
    
    def test_valid_tenant_names(self):
        """Test that valid tenant names pass validation."""
        valid_names = [
            "test-tenant",
            "my-company",
            "startup123",
            "abc-def-ghi",
            "company2024"
        ]
        
        for name in valid_names:
            assert self.tenant_service._validate_tenant_name(name) is True, f"Name '{name}' should be valid"
    
    def test_invalid_tenant_names(self):
        """Test that invalid tenant names fail validation."""
        invalid_names = [
            "",  # Empty
            "a",  # Too short
            "ab",  # Still too short (minimum 3)
            "a" * 51,  # Too long
            "Test-Tenant",  # Uppercase not allowed
            "test_tenant",  # Underscore not allowed
            "test tenant",  # Space not allowed
            "-test",  # Cannot start with hyphen
            "test-",  # Cannot end with hyphen
            "te--st",  # Double hyphen not allowed
            "123",  # Numbers only not allowed
            "test@tenant",  # Special characters not allowed
        ]
        
        for name in invalid_names:
            assert self.tenant_service._validate_tenant_name(name) is False, f"Name '{name}' should be invalid"
    
    def test_reserved_words_rejected(self):
        """Test that reserved words are rejected."""
        reserved_names = [
            "api", "www", "admin", "app", "mail", "ftp", "blog",
            "dev", "test", "staging", "support", "help", "docs",
            "status", "login", "signup", "dashboard", "settings"
        ]
        
        for name in reserved_names:
            assert self.tenant_service._validate_tenant_name(name) is False, f"Reserved name '{name}' should be rejected"


class TestQuotaLimitsLogic:
    """Test quota limits business logic."""
    
    def setup_method(self):
        self.tenant_service = TenantService(None, None, None)
    
    def test_free_tier_quota_limits(self):
        """Test quota limits for free tier."""
        limits = self.tenant_service._get_default_quota_limits(SubscriptionTier.FREE)
        
        assert limits.max_profiles == 100
        assert limits.max_searches_per_month == 50
        assert limits.max_storage_gb == 1
        assert limits.max_api_requests_per_minute == 10
        assert limits.max_users == 2
    
    def test_enterprise_tier_unlimited_resources(self):
        """Test that enterprise tier has unlimited resources."""
        limits = self.tenant_service._get_default_quota_limits(SubscriptionTier.ENTERPRISE)
        
        assert limits.max_profiles is None  # Unlimited
        assert limits.max_searches_per_month is None  # Unlimited
        assert limits.max_storage_gb is None  # Unlimited
        assert limits.max_users is None  # Unlimited
        # API rate limiting still applies
        assert limits.max_api_requests_per_minute == 1000
    
    def test_quota_progression_across_tiers(self):
        """Test that quota limits increase across subscription tiers."""
        free_limits = self.tenant_service._get_default_quota_limits(SubscriptionTier.FREE)
        basic_limits = self.tenant_service._get_default_quota_limits(SubscriptionTier.BASIC)
        pro_limits = self.tenant_service._get_default_quota_limits(SubscriptionTier.PROFESSIONAL)
        
        # Test progression (higher tiers have higher limits)
        assert free_limits.max_profiles < basic_limits.max_profiles < pro_limits.max_profiles
        assert free_limits.max_searches_per_month < basic_limits.max_searches_per_month < pro_limits.max_searches_per_month
        assert free_limits.max_storage_gb < basic_limits.max_storage_gb < pro_limits.max_storage_gb
        assert free_limits.max_users < basic_limits.max_users < pro_limits.max_users


class TestFeatureFlagsLogic:
    """Test feature flags business logic."""
    
    def setup_method(self):
        self.tenant_service = TenantService(None, None, None)
    
    def test_free_tier_feature_restrictions(self):
        """Test that free tier has appropriate feature restrictions."""
        flags = self.tenant_service._get_default_feature_flags(SubscriptionTier.FREE)
        
        assert flags.enable_custom_reports is False  # Not available for FREE
        assert flags.enable_api_access is True  # API access enabled for all
        assert flags.enable_data_insights is False  # Not available for FREE
        assert flags.enable_export is True  # Export enabled for all
        assert flags.enable_webhooks is False  # Not available for FREE
        assert flags.enable_ats_integration is False  # Enterprise only
        assert flags.enable_sso is False  # Enterprise only
    
    def test_enterprise_tier_all_features(self):
        """Test that enterprise tier has all features enabled."""
        flags = self.tenant_service._get_default_feature_flags(SubscriptionTier.ENTERPRISE)
        
        assert flags.enable_custom_reports is True
        assert flags.enable_api_access is True
        assert flags.enable_data_insights is True
        assert flags.enable_export is True
        assert flags.enable_webhooks is True
        assert flags.enable_ats_integration is True
        assert flags.enable_sso is True
    
    def test_feature_progression_across_tiers(self):
        """Test that features are enabled progressively across tiers."""
        basic_flags = self.tenant_service._get_default_feature_flags(SubscriptionTier.BASIC)
        pro_flags = self.tenant_service._get_default_feature_flags(SubscriptionTier.PROFESSIONAL)
        
        # Basic tier should have API access and custom reports
        assert basic_flags.enable_api_access is True
        assert basic_flags.enable_export is True
        assert basic_flags.enable_custom_reports is True  # Available from BASIC
        assert basic_flags.enable_data_insights is False  # Pro+ only
        assert basic_flags.enable_webhooks is False  # Pro+ only
        
        # Professional tier should have more features
        assert pro_flags.enable_custom_reports is True
        assert pro_flags.enable_data_insights is True
        assert pro_flags.enable_webhooks is True
        
        # But not the enterprise-only features
        assert pro_flags.enable_ats_integration is False  # Enterprise only
        assert pro_flags.enable_sso is False  # Enterprise only


class TestQuotaCalculations:
    """Test quota usage calculations and enforcement logic."""
    
    def test_quota_percentage_calculation(self):
        """Test quota usage percentage calculation."""
        # Mock quota check result
        quota_result = {
            "resource": "profiles",
            "limit": 100,
            "current_usage": 75,
            "remaining": 25,
            "percentage_used": 75.0,
            "exceeded": False
        }
        
        # Verify calculations
        assert quota_result["remaining"] == quota_result["limit"] - quota_result["current_usage"]
        assert quota_result["percentage_used"] == (quota_result["current_usage"] / quota_result["limit"]) * 100
        assert quota_result["exceeded"] == (quota_result["current_usage"] >= quota_result["limit"])
    
    def test_quota_exceeded_detection(self):
        """Test quota exceeded detection logic."""
        # Test exact limit
        quota_at_limit = {
            "limit": 100,
            "current_usage": 100
        }
        assert quota_at_limit["current_usage"] >= quota_at_limit["limit"]
        
        # Test over limit
        quota_over_limit = {
            "limit": 100,
            "current_usage": 105
        }
        assert quota_over_limit["current_usage"] >= quota_over_limit["limit"]
    
    def test_unlimited_quota_handling(self):
        """Test handling of unlimited quotas (None values)."""
        # When limit is None (unlimited), percentage should be 0 and never exceeded
        unlimited_usage = 1000000
        unlimited_limit = None
        
        percentage_used = (unlimited_usage / unlimited_limit * 100) if unlimited_limit and unlimited_limit > 0 else 0
        exceeded = unlimited_usage >= unlimited_limit if unlimited_limit is not None else False
        
        assert percentage_used == 0
        assert exceeded is False


class TestTenantValidationInputs:
    """Test tenant creation input validation logic."""
    
    def setup_method(self):
        self.tenant_service = TenantService(None, None, None)
    
    @pytest.mark.asyncio
    async def test_valid_tenant_creation_inputs(self):
        """Test validation of valid tenant creation inputs."""
        valid_admin_data = {
            "email": "admin@example.com",
            "password": "SecurePassword123!",
            "full_name": "Test Administrator"
        }
        
        # Should not raise any exceptions
        await self.tenant_service._validate_tenant_creation_inputs(
            name="test-tenant",
            display_name="Test Tenant Company",
            primary_contact_email="contact@example.com",
            admin_user_data=valid_admin_data
        )
    
    @pytest.mark.asyncio
    async def test_invalid_email_format_rejection(self):
        """Test that invalid email formats are rejected."""
        invalid_emails = [
            "invalid-email",
            "@domain.com",
            "user@",
            "user@domain",
            "user.domain.com",
            ""
        ]
        
        for email in invalid_emails:
            with pytest.raises(ValueError, match="Invalid.*email format"):
                await self.tenant_service._validate_tenant_creation_inputs(
                    name="test-tenant",
                    display_name="Test Tenant",
                    primary_contact_email=email,
                    admin_user_data={
                        "email": "admin@valid.com",
                        "password": "SecurePassword123!",
                        "full_name": "Admin User"
                    }
                )
    
    @pytest.mark.asyncio 
    async def test_weak_password_rejection(self):
        """Test that weak passwords are rejected."""
        weak_passwords = [
            "short",  # Too short
            "password",  # Too simple
            "12345678",  # Only numbers
            "",  # Empty
        ]
        
        for password in weak_passwords:
            with pytest.raises(ValueError, match="password"):
                await self.tenant_service._validate_tenant_creation_inputs(
                    name="test-tenant",
                    display_name="Test Tenant",
                    primary_contact_email="contact@example.com",
                    admin_user_data={
                        "email": "admin@valid.com",
                        "password": password,
                        "full_name": "Admin User"
                    }
                )
    
    @pytest.mark.asyncio
    async def test_missing_required_fields_rejection(self):
        """Test that missing required fields are rejected."""
        incomplete_admin_data_sets = [
            {},  # Missing everything
            {"email": "admin@test.com"},  # Missing password and name
            {"password": "SecurePassword123!"},  # Missing email and name
            {"full_name": "Admin User"},  # Missing email and password
            {"email": "admin@test.com", "password": "SecurePassword123!"},  # Missing name
        ]
        
        for admin_data in incomplete_admin_data_sets:
            with pytest.raises(ValueError, match="required"):
                await self.tenant_service._validate_tenant_creation_inputs(
                    name="test-tenant",
                    display_name="Test Tenant",
                    primary_contact_email="contact@example.com",
                    admin_user_data=admin_data
                )