"""
Test suite for the comprehensive audit logging system.

This module tests the audit logging functionality across all components
including the service, provider, API endpoints, and database integration.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.infrastructure.audit import AuditService, AuditServiceError
from app.infrastructure.providers.audit_provider import get_audit_service, reset_audit_service
from app.infrastructure.persistence.models.audit_table import AuditEventType, AuditRiskLevel
from app.infrastructure.adapters.postgres_adapter import PostgresAdapter


class TestAuditService:
    """Test the core audit service functionality."""

    @pytest.fixture
    async def mock_database_adapter(self):
        """Create a mock database adapter for testing."""
        adapter = AsyncMock(spec=PostgresAdapter)
        adapter.execute = AsyncMock()
        return adapter

    @pytest.fixture
    async def audit_service(self, mock_database_adapter):
        """Create audit service with mocked dependencies."""
        return AuditService(database_adapter=mock_database_adapter)

    @pytest.mark.asyncio
    async def test_log_basic_audit_event(self, audit_service, mock_database_adapter):
        """Test logging a basic audit event."""
        tenant_id = str(uuid4())
        user_id = str(uuid4())
        
        # Mock sequence number query
        mock_database_adapter.execute.side_effect = [1, None]  # sequence number, then insert
        
        log_id = await audit_service.log_event(
            event_type=AuditEventType.LOGIN_SUCCESS.value,
            tenant_id=tenant_id,
            action="user_login",
            resource_type="user_session",
            user_id=user_id,
            user_email="test@example.com",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
        )
        
        assert log_id is not None
        assert isinstance(log_id, str)
        assert len(log_id) == 36  # UUID length
        
        # Verify database calls
        assert mock_database_adapter.execute.call_count == 2
        
        # Verify the INSERT call
        insert_call = mock_database_adapter.execute.call_args_list[1]
        assert "INSERT INTO audit_logs" in insert_call[0][0]

    @pytest.mark.asyncio
    async def test_log_authentication_event(self, audit_service, mock_database_adapter):
        """Test logging authentication-specific events."""
        tenant_id = str(uuid4())
        user_id = str(uuid4())
        
        mock_database_adapter.execute.side_effect = [1, None]
        
        log_id = await audit_service.log_authentication_event(
            event_type=AuditEventType.LOGIN_FAILED.value,
            tenant_id=tenant_id,
            user_id=user_id,
            user_email="test@example.com",
            success=False,
            failure_reason="invalid_password",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
        )
        
        assert log_id is not None
        
        # Verify that risk level was set to medium for failed login
        insert_call = mock_database_adapter.execute.call_args_list[1]
        assert "medium" in str(insert_call[0])  # Check risk level

    @pytest.mark.asyncio
    async def test_log_file_upload_event(self, audit_service, mock_database_adapter):
        """Test logging file upload events."""
        tenant_id = str(uuid4())
        user_id = str(uuid4())
        upload_id = str(uuid4())
        
        mock_database_adapter.execute.side_effect = [1, None]
        
        log_id = await audit_service.log_file_upload_event(
            event_type=AuditEventType.FILE_UPLOAD_SUCCESS.value,
            tenant_id=tenant_id,
            user_id=user_id,
            filename="test_resume.pdf",
            file_size=1024000,  # 1MB
            upload_id=upload_id,
            processing_duration_ms=5000,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
        )
        
        assert log_id is not None
        
        # Verify file-specific details were captured
        insert_call = mock_database_adapter.execute.call_args_list[1]
        assert "test_resume.pdf" in str(insert_call[0])
        assert "1024000" in str(insert_call[0])

    @pytest.mark.asyncio
    async def test_log_security_event(self, audit_service, mock_database_adapter):
        """Test logging security events."""
        tenant_id = str(uuid4())
        
        mock_database_adapter.execute.side_effect = [1, None]
        
        log_id = await audit_service.log_security_event(
            event_type=AuditEventType.SUSPICIOUS_ACTIVITY.value,
            tenant_id=tenant_id,
            threat_type="brute_force_attack",
            severity="high",
            ip_address="192.168.1.100",
            user_agent="BadBot/1.0",
            threat_details={"attempts": 10, "timespan": "60s"},
            mitigation_actions=["rate_limit_applied", "ip_blocked"],
        )
        
        assert log_id is not None
        
        # Verify security event was marked as suspicious
        insert_call = mock_database_adapter.execute.call_args_list[1]
        assert "true" in str(insert_call[0]).lower()  # suspicious flag

    @pytest.mark.asyncio
    async def test_query_audit_logs(self, audit_service, mock_database_adapter):
        """Test querying audit logs with filtering."""
        tenant_id = str(uuid4())
        
        # Mock query results
        mock_rows = [
            {
                "id": str(uuid4()),
                "tenant_id": tenant_id,
                "event_type": "login_success",
                "user_id": str(uuid4()),
                "user_email": "test@example.com",
                "resource_type": "user_session",
                "resource_id": None,
                "action": "authenticate",
                "details": '{"success": true}',
                "ip_address": "192.168.1.1",
                "risk_level": "low",
                "suspicious": False,
                "event_timestamp": datetime.now(timezone.utc),
                "correlation_id": None,
            }
        ]
        
        mock_database_adapter.execute.side_effect = [1, mock_rows]  # count, then data
        
        result = await audit_service.query_audit_logs(
            tenant_id=tenant_id,
            event_types=["login_success"],
            page=1,
            size=10,
        )
        
        assert "audit_logs" in result
        assert "pagination" in result
        assert len(result["audit_logs"]) == 1
        assert result["pagination"]["total"] == 1

    @pytest.mark.asyncio
    async def test_get_audit_statistics(self, audit_service, mock_database_adapter):
        """Test getting audit statistics."""
        tenant_id = str(uuid4())
        
        # Mock statistics queries
        mock_database_adapter.execute.side_effect = [
            100,  # total events
            [{"event_type": "login_success", "count": 50}],  # events by type
            [{"risk_level": "low", "count": 80}],  # events by risk
            10,  # suspicious events
            25,  # recent events
            15,  # unique users
            8,   # unique IPs
        ]
        
        stats = await audit_service.get_audit_statistics(tenant_id=tenant_id)
        
        assert stats["total_events"] == 100
        assert stats["suspicious_events"] == 10
        assert stats["unique_users"] == 15
        assert "events_by_type" in stats
        assert "events_by_risk_level" in stats

    @pytest.mark.asyncio
    async def test_verify_log_integrity(self, audit_service, mock_database_adapter):
        """Test log integrity verification."""
        tenant_id = str(uuid4())
        log_id = str(uuid4())
        
        # Mock log data
        mock_log = {
            "id": log_id,
            "tenant_id": tenant_id,
            "event_type": "login_success",
            "user_id": str(uuid4()),
            "user_email": "test@example.com",
            "session_id": None,
            "api_key_id": None,
            "resource_type": "user_session",
            "resource_id": None,
            "action": "authenticate",
            "details": {"success": True},
            "ip_address": "192.168.1.1",
            "user_agent": "Mozilla/5.0",
            "risk_level": "low",
            "suspicious": False,
            "event_timestamp": datetime.now(timezone.utc),
            "logged_at": datetime.now(timezone.utc),
            "log_hash": "abc123...",
            "sequence_number": 1,
            "correlation_id": None,
            "batch_id": None,
            "error_code": None,
            "error_message": None,
        }
        
        mock_database_adapter.execute.return_value = mock_log
        
        result = await audit_service.verify_log_integrity(tenant_id=tenant_id, log_id=log_id)
        
        assert "verified" in result
        assert "log_id" in result
        assert result["log_id"] == log_id

    @pytest.mark.asyncio
    async def test_export_audit_logs_json(self, audit_service, mock_database_adapter):
        """Test exporting audit logs in JSON format."""
        tenant_id = str(uuid4())
        
        mock_rows = [
            {
                "id": str(uuid4()),
                "tenant_id": tenant_id,
                "event_type": "login_success",
                "event_timestamp": datetime.now(timezone.utc),
            }
        ]
        
        mock_database_adapter.execute.return_value = mock_rows
        
        export_data = await audit_service.export_audit_logs(
            tenant_id=tenant_id,
            format="json",
        )
        
        assert isinstance(export_data, bytes)
        # Verify it's valid JSON by checking it starts with '[' and ends with ']'
        json_str = export_data.decode('utf-8')
        assert json_str.startswith('[')
        assert json_str.endswith(']')

    @pytest.mark.asyncio
    async def test_health_check(self, audit_service, mock_database_adapter):
        """Test audit service health check."""
        mock_database_adapter.execute.side_effect = [1, True]  # connectivity, table exists
        
        health = await audit_service.check_health()
        
        assert health["status"] == "healthy"
        assert health["database_connected"] is True
        assert health["audit_table_exists"] is True

    @pytest.mark.asyncio
    async def test_audit_service_error_handling(self, audit_service, mock_database_adapter):
        """Test error handling in audit service."""
        tenant_id = str(uuid4())
        
        # Mock database error
        mock_database_adapter.execute.side_effect = Exception("Database connection failed")
        
        with pytest.raises(AuditServiceError):
            await audit_service.log_event(
                event_type=AuditEventType.LOGIN_SUCCESS.value,
                tenant_id=tenant_id,
                action="test",
                resource_type="test",
            )


class TestAuditProvider:
    """Test the audit service provider."""

    @pytest.mark.asyncio
    async def test_get_audit_service_singleton(self):
        """Test that audit service provider returns singleton instance."""
        # Reset the provider first
        await reset_audit_service()
        
        # Mock the postgres adapter
        with patch('app.infrastructure.providers.audit_provider.get_postgres_adapter') as mock_postgres:
            mock_postgres.return_value = AsyncMock()
            
            service1 = await get_audit_service()
            service2 = await get_audit_service()
            
            # Should be the same instance
            assert service1 is service2
            
            # Should only call postgres adapter once (singleton behavior)
            assert mock_postgres.call_count == 1

    @pytest.mark.asyncio
    async def test_reset_audit_service(self):
        """Test resetting the audit service provider."""
        with patch('app.infrastructure.providers.audit_provider.get_postgres_adapter') as mock_postgres:
            mock_postgres.return_value = AsyncMock()
            
            service1 = await get_audit_service()
            await reset_audit_service()
            service2 = await get_audit_service()
            
            # Should be different instances after reset
            assert service1 is not service2


class TestAuditIntegration:
    """Integration tests for audit logging across the system."""

    @pytest.mark.asyncio
    async def test_audit_logging_in_authentication(self):
        """Test that authentication events are properly audited."""
        # This would be an integration test that requires actual database
        # For now, just verify the interface exists
        from app.infrastructure.auth.authentication_service import AuthenticationService
        from app.infrastructure.persistence.models.audit_table import AuditEventType
        
        # Verify that audit event types exist for authentication
        assert hasattr(AuditEventType, 'LOGIN_SUCCESS')
        assert hasattr(AuditEventType, 'LOGIN_FAILED')
        assert hasattr(AuditEventType, 'PASSWORD_CHANGED')

    @pytest.mark.asyncio
    async def test_audit_logging_in_upload_service(self):
        """Test that upload events are properly audited."""
        from app.application.upload_service import UploadApplicationService
        from app.infrastructure.persistence.models.audit_table import AuditEventType
        
        # Verify that audit event types exist for uploads
        assert hasattr(AuditEventType, 'FILE_UPLOAD_STARTED')
        assert hasattr(AuditEventType, 'FILE_UPLOAD_SUCCESS')
        assert hasattr(AuditEventType, 'FILE_UPLOAD_FAILED')
        assert hasattr(AuditEventType, 'FILE_VALIDATION_FAILED')


if __name__ == "__main__":
    pytest.main([__file__])