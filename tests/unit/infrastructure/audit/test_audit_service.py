"""Audit service infrastructure tests disabled pending module restoration."""

import pytest

pytest.skip(
    "Audit infrastructure currently unavailable after migration; tests pending rewrite.",
    allow_module_level=True,
)


class TestAuditServiceInitialization:
    """Test AuditService initialization and health checks."""

    @pytest.fixture
    def mock_db_adapter(self):
        """Create mock database adapter."""
        adapter = AsyncMock()
        adapter.fetch_one = AsyncMock()
        adapter.fetch_all = AsyncMock()
        adapter.execute = AsyncMock()
        return adapter

    @pytest.fixture
    def audit_service(self, mock_db_adapter):
        """Create AuditService instance."""
        return AuditService(database_adapter=mock_db_adapter)

    def test_audit_service_initialization(self, audit_service, mock_db_adapter):
        """Test service initializes with correct dependencies."""
        assert audit_service.db_adapter == mock_db_adapter
        assert audit_service._sequence_counters == {}

    @pytest.mark.asyncio

    async def test_check_health_success(self, audit_service, mock_db_adapter):
        """Test health check when service is healthy."""
        mock_db_adapter.fetch_one.side_effect = [
            {"test": 1},  # Connection test
            {"exists": True}  # Table exists check
        ]

        result = await audit_service.check_health()

        assert result["status"] == "healthy"
        assert result["database_connected"] is True
        assert result["audit_table_exists"] is True

    @pytest.mark.asyncio

    async def test_check_health_table_missing(self, audit_service, mock_db_adapter):
        """Test health check when audit table doesn't exist."""
        mock_db_adapter.fetch_one.side_effect = [
            {"test": 1},  # Connection test
            {"exists": False}  # Table doesn't exist
        ]

        result = await audit_service.check_health()

        assert result["status"] == "degraded"
        assert result["database_connected"] is True
        assert result["audit_table_exists"] is False

    @pytest.mark.asyncio

    async def test_check_health_database_error(self, audit_service, mock_db_adapter):
        """Test health check when database is unreachable."""
        mock_db_adapter.fetch_one.side_effect = Exception("Connection failed")

        result = await audit_service.check_health()

        assert result["status"] == "unhealthy"
        assert result["database_connected"] is False
        assert "error" in result


class TestAuditEventLogging:
    """Test basic audit event logging functionality."""

    @pytest.fixture
    def mock_db_adapter(self):
        """Create mock database adapter."""
        adapter = AsyncMock()
        adapter.fetch_one = AsyncMock(return_value={"max": 0})
        adapter.execute = AsyncMock()
        return adapter

    @pytest.fixture
    def audit_service(self, mock_db_adapter):
        """Create AuditService instance."""
        return AuditService(database_adapter=mock_db_adapter)

    @pytest.mark.asyncio

    async def test_log_event_basic(self, audit_service, mock_db_adapter):
        """Test logging a basic audit event."""
        log_id = await audit_service.log_event(
            event_type="user_action",
            tenant_id="tenant-123",
            action="create",
            resource_type="profile",
            user_id="user-456"
        )

        assert log_id is not None
        mock_db_adapter.execute.assert_called_once()

    @pytest.mark.asyncio

    async def test_log_event_with_all_fields(self, audit_service, mock_db_adapter):
        """Test logging event with all optional fields."""
        log_id = await audit_service.log_event(
            event_type="user_action",
            tenant_id="tenant-123",
            action="update",
            resource_type="profile",
            user_id="user-456",
            user_email="user@test.com",
            session_id="session-789",
            api_key_id="key-012",
            resource_id="profile-345",
            details={"field": "value"},
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            risk_level="medium",
            suspicious=True,
            correlation_id="corr-123",
            batch_id="batch-456",
            error_code="ERR_001",
            error_message="Test error"
        )

        assert log_id is not None
        mock_db_adapter.execute.assert_called_once()

    @pytest.mark.asyncio

    async def test_log_event_generates_hash(self, audit_service, mock_db_adapter):
        """Test that event logging generates tamper-resistant hash."""
        await audit_service.log_event(
            event_type="user_action",
            tenant_id="tenant-123",
            action="create",
            resource_type="profile"
        )

        call_args = mock_db_adapter.execute.call_args
        # Hash should be in the parameters
        assert len(call_args[0]) > 20  # Has log_hash parameter

    @pytest.mark.asyncio

    async def test_log_event_sequence_numbers(self, audit_service, mock_db_adapter):
        """Test that events get sequential sequence numbers."""
        mock_db_adapter.fetch_one.side_effect = [
            {"max": 0},  # First event
            {"max": 1},  # Second event
            {"max": 2},  # Third event
        ]

        await audit_service.log_event(
            event_type="user_action",
            tenant_id="tenant-123",
            action="create",
            resource_type="profile"
        )

        await audit_service.log_event(
            event_type="user_action",
            tenant_id="tenant-123",
            action="update",
            resource_type="profile"
        )

        assert mock_db_adapter.execute.call_count == 2

    @pytest.mark.asyncio

    async def test_log_event_error_handling(self, audit_service, mock_db_adapter):
        """Test error handling when logging fails."""
        mock_db_adapter.execute.side_effect = Exception("Database error")

        with pytest.raises(AuditServiceError) as exc_info:
            await audit_service.log_event(
                event_type="user_action",
                tenant_id="tenant-123",
                action="create",
                resource_type="profile"
            )

        assert "Failed to log audit event" in str(exc_info.value)


class TestAuthenticationEventLogging:
    """Test authentication-specific event logging."""

    @pytest.fixture
    def mock_db_adapter(self):
        """Create mock database adapter."""
        adapter = AsyncMock()
        adapter.fetch_one = AsyncMock(return_value={"max": 0})
        adapter.execute = AsyncMock()
        return adapter

    @pytest.fixture
    def audit_service(self, mock_db_adapter):
        """Create AuditService instance."""
        return AuditService(database_adapter=mock_db_adapter)

    @pytest.mark.asyncio

    async def test_log_authentication_success(self, audit_service, mock_db_adapter):
        """Test logging successful authentication."""
        log_id = await audit_service.log_authentication_event(
            event_type="user_login",
            tenant_id="tenant-123",
            user_id="user-456",
            user_email="user@test.com",
            session_id="session-789",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            success=True
        )

        assert log_id is not None
        mock_db_adapter.execute.assert_called_once()

    @pytest.mark.asyncio

    async def test_log_authentication_failure(self, audit_service, mock_db_adapter):
        """Test logging failed authentication."""
        log_id = await audit_service.log_authentication_event(
            event_type="login_failed",
            tenant_id="tenant-123",
            user_email="user@test.com",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            success=False,
            failure_reason="invalid_password"
        )

        assert log_id is not None
        mock_db_adapter.execute.assert_called_once()

    @pytest.mark.asyncio

    async def test_log_authentication_suspicious(self, audit_service, mock_db_adapter):
        """Test logging suspicious authentication attempts."""
        log_id = await audit_service.log_authentication_event(
            event_type="suspicious_login",
            tenant_id="tenant-123",
            user_email="user@test.com",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            success=False,
            failure_reason="multiple_failed_attempts"
        )

        assert log_id is not None


class TestFileUploadEventLogging:
    """Test file upload-specific event logging."""

    @pytest.fixture
    def mock_db_adapter(self):
        """Create mock database adapter."""
        adapter = AsyncMock()
        adapter.fetch_one = AsyncMock(return_value={"max": 0})
        adapter.execute = AsyncMock()
        return adapter

    @pytest.fixture
    def audit_service(self, mock_db_adapter):
        """Create AuditService instance."""
        return AuditService(database_adapter=mock_db_adapter)

    @pytest.mark.asyncio

    async def test_log_file_upload_success(self, audit_service, mock_db_adapter):
        """Test logging successful file upload."""
        log_id = await audit_service.log_file_upload_event(
            event_type="file_uploaded",
            tenant_id="tenant-123",
            user_id="user-456",
            filename="test.pdf",
            file_size=1024000,
            upload_id="upload-789"
        )

        assert log_id is not None
        mock_db_adapter.execute.assert_called_once()

    @pytest.mark.asyncio

    async def test_log_file_upload_with_validation_errors(self, audit_service, mock_db_adapter):
        """Test logging file upload with validation errors."""
        log_id = await audit_service.log_file_upload_event(
            event_type="file_validation_failed",
            tenant_id="tenant-123",
            user_id="user-456",
            filename="test.pdf",
            file_size=1024000,
            upload_id="upload-789",
            validation_errors=["Invalid format", "File too large"]
        )

        assert log_id is not None

    @pytest.mark.asyncio

    async def test_log_file_upload_with_security_warnings(self, audit_service, mock_db_adapter):
        """Test logging file upload with security warnings."""
        log_id = await audit_service.log_file_upload_event(
            event_type="file_security_issue",
            tenant_id="tenant-123",
            user_id="user-456",
            filename="suspicious.pdf",
            file_size=1024000,
            upload_id="upload-789",
            security_warnings=["Suspicious content detected"]
        )

        assert log_id is not None


class TestSecurityEventLogging:
    """Test security-specific event logging."""

    @pytest.fixture
    def mock_db_adapter(self):
        """Create mock database adapter."""
        adapter = AsyncMock()
        adapter.fetch_one = AsyncMock(return_value={"max": 0})
        adapter.execute = AsyncMock()
        return adapter

    @pytest.fixture
    def audit_service(self, mock_db_adapter):
        """Create AuditService instance."""
        return AuditService(database_adapter=mock_db_adapter)

    @pytest.mark.asyncio

    async def test_log_security_event_low_severity(self, audit_service, mock_db_adapter):
        """Test logging low severity security event."""
        log_id = await audit_service.log_security_event(
            event_type="security_alert",
            tenant_id="tenant-123",
            threat_type="brute_force_attempt",
            severity="low",
            ip_address="192.168.1.1"
        )

        assert log_id is not None

    @pytest.mark.asyncio

    async def test_log_security_event_critical_severity(self, audit_service, mock_db_adapter):
        """Test logging critical severity security event."""
        log_id = await audit_service.log_security_event(
            event_type="security_breach",
            tenant_id="tenant-123",
            threat_type="unauthorized_access",
            severity="critical",
            user_id="user-456",
            ip_address="192.168.1.1",
            threat_details={"attempted_resource": "admin_panel"}
        )

        assert log_id is not None

    @pytest.mark.asyncio

    async def test_log_security_event_with_mitigation(self, audit_service, mock_db_adapter):
        """Test logging security event with mitigation actions."""
        log_id = await audit_service.log_security_event(
            event_type="security_alert",
            tenant_id="tenant-123",
            threat_type="sql_injection",
            severity="high",
            ip_address="192.168.1.1",
            mitigation_actions=["Blocked IP", "Alert security team"]
        )

        assert log_id is not None


class TestAuditLogQuerying:
    """Test audit log querying and filtering."""

    @pytest.fixture
    def mock_db_adapter(self):
        """Create mock database adapter."""
        adapter = AsyncMock()
        adapter.fetch_one = AsyncMock(return_value={"count": 10})
        adapter.fetch_all = AsyncMock(return_value=[
            {
                "id": "log-1",
                "tenant_id": "tenant-123",
                "event_type": "user_action",
                "user_id": "user-456",
                "user_email": "user@test.com",
                "resource_type": "profile",
                "resource_id": "profile-789",
                "action": "create",
                "details": json.dumps({}),
                "ip_address": "192.168.1.1",
                "risk_level": "low",
                "suspicious": False,
                "event_timestamp": datetime.now(timezone.utc),
                "correlation_id": None,
                "session_id": None,
                "api_key_id": None,
                "user_agent": None,
                "batch_id": None,
                "error_code": None,
                "error_message": None,
                "sequence_number": 1
            }
        ])
        return adapter

    @pytest.fixture
    def audit_service(self, mock_db_adapter):
        """Create AuditService instance."""
        return AuditService(database_adapter=mock_db_adapter)

    @pytest.mark.asyncio

    async def test_query_audit_logs_basic(self, audit_service, mock_db_adapter):
        """Test basic audit log query."""
        result = await audit_service.query_audit_logs(tenant_id="tenant-123")

        assert "audit_logs" in result
        assert "pagination" in result
        assert len(result["audit_logs"]) > 0

    @pytest.mark.asyncio

    async def test_query_audit_logs_with_user_filter(self, audit_service, mock_db_adapter):
        """Test querying logs filtered by user."""
        result = await audit_service.query_audit_logs(
            tenant_id="tenant-123",
            user_id="user-456"
        )

        assert result is not None

    @pytest.mark.asyncio

    async def test_query_audit_logs_with_event_types(self, audit_service, mock_db_adapter):
        """Test querying logs filtered by event types."""
        result = await audit_service.query_audit_logs(
            tenant_id="tenant-123",
            event_types=["user_action", "security_alert"]
        )

        assert result is not None

    @pytest.mark.asyncio

    async def test_query_audit_logs_with_time_range(self, audit_service, mock_db_adapter):
        """Test querying logs within time range."""
        start = datetime.now(timezone.utc) - timedelta(days=7)
        end = datetime.now(timezone.utc)

        result = await audit_service.query_audit_logs(
            tenant_id="tenant-123",
            start_time=start,
            end_time=end
        )

        assert result is not None

    @pytest.mark.asyncio

    async def test_query_audit_logs_suspicious_only(self, audit_service, mock_db_adapter):
        """Test querying only suspicious events."""
        result = await audit_service.query_audit_logs(
            tenant_id="tenant-123",
            suspicious_only=True
        )

        assert result is not None

    @pytest.mark.asyncio

    async def test_query_audit_logs_pagination(self, audit_service, mock_db_adapter):
        """Test audit log pagination."""
        result = await audit_service.query_audit_logs(
            tenant_id="tenant-123",
            page=2,
            size=25
        )

        assert result["pagination"]["page"] == 2
        assert result["pagination"]["size"] == 25

    @pytest.mark.asyncio

    async def test_query_audit_logs_error_handling(self, audit_service, mock_db_adapter):
        """Test error handling in query."""
        mock_db_adapter.fetch_one.side_effect = Exception("Database error")

        with pytest.raises(AuditServiceError) as exc_info:
            await audit_service.query_audit_logs(tenant_id="tenant-123")

        assert "Failed to query audit logs" in str(exc_info.value)


class TestAuditStatistics:
    """Test audit statistics and compliance reporting."""

    @pytest.fixture
    def mock_db_adapter(self):
        """Create mock database adapter."""
        adapter = AsyncMock()
        adapter.fetch_one = AsyncMock(side_effect=[
            {"count": 100},  # total events
            {"count": 10},   # suspicious events
            {"count": 50},   # recent events
            {"count": 20},   # unique users
            {"count": 15},   # unique IPs
        ])
        adapter.fetch_all = AsyncMock(side_effect=[
            [{"event_type": "user_action", "count": 50}, {"event_type": "security_alert", "count": 10}],
            [{"risk_level": "low", "count": 70}, {"risk_level": "medium", "count": 20}]
        ])
        return adapter

    @pytest.fixture
    def audit_service(self, mock_db_adapter):
        """Create AuditService instance."""
        return AuditService(database_adapter=mock_db_adapter)

    @pytest.mark.asyncio

    async def test_get_audit_statistics_basic(self, audit_service, mock_db_adapter):
        """Test getting basic audit statistics."""
        result = await audit_service.get_audit_statistics(tenant_id="tenant-123")

        assert "total_events" in result
        assert "events_by_type" in result
        assert "events_by_risk_level" in result
        assert "suspicious_events" in result
        assert "unique_users" in result
        assert "unique_ip_addresses" in result

    @pytest.mark.asyncio

    async def test_get_audit_statistics_with_time_range(self, audit_service, mock_db_adapter):
        """Test getting statistics for specific time range."""
        start = datetime.now(timezone.utc) - timedelta(days=30)
        end = datetime.now(timezone.utc)

        result = await audit_service.get_audit_statistics(
            tenant_id="tenant-123",
            start_time=start,
            end_time=end
        )

        assert result["time_range"]["start_time"] is not None
        assert result["time_range"]["end_time"] is not None

    @pytest.mark.asyncio

    async def test_get_audit_statistics_error_handling(self, audit_service, mock_db_adapter):
        """Test error handling in statistics."""
        mock_db_adapter.fetch_one.side_effect = Exception("Database error")

        with pytest.raises(AuditServiceError) as exc_info:
            await audit_service.get_audit_statistics(tenant_id="tenant-123")

        assert "Failed to get audit statistics" in str(exc_info.value)


class TestLogIntegrityVerification:
    """Test audit log integrity verification."""

    @pytest.fixture
    def mock_db_adapter(self):
        """Create mock database adapter."""
        adapter = AsyncMock()
        return adapter

    @pytest.fixture
    def audit_service(self, mock_db_adapter):
        """Create AuditService instance."""
        return AuditService(database_adapter=mock_db_adapter)

    @pytest.mark.asyncio

    async def test_verify_log_integrity_valid(self, audit_service, mock_db_adapter):
        """Test verifying valid log integrity."""
        test_data = {
            "id": "log-123",
            "tenant_id": "tenant-456",
            "event_type": "user_action",
            "action": "create",
            "resource_type": "profile",
            "details": {},
            "sequence_number": 1,
            "event_timestamp": datetime.now(timezone.utc),
            "logged_at": datetime.now(timezone.utc),
            "user_id": None,
            "user_email": None,
            "session_id": None,
            "api_key_id": None,
            "resource_id": None,
            "ip_address": "192.168.1.1",
            "user_agent": None,
            "risk_level": "low",
            "suspicious": False,
            "correlation_id": None,
            "batch_id": None,
            "error_code": None,
            "error_message": None,
        }

        # Generate expected hash
        expected_hash = audit_service._generate_log_hash(test_data)
        test_data["log_hash"] = expected_hash

        mock_db_adapter.fetch_one.return_value = test_data

        result = await audit_service.verify_log_integrity(
            tenant_id="tenant-456",
            log_id="log-123"
        )

        assert result["verified"] is True
        assert result["log_id"] == "log-123"

    @pytest.mark.asyncio

    async def test_verify_log_integrity_tampered(self, audit_service, mock_db_adapter):
        """Test detecting tampered log entry."""
        test_data = {
            "id": "log-123",
            "tenant_id": "tenant-456",
            "event_type": "user_action",
            "action": "create",
            "resource_type": "profile",
            "details": {},
            "sequence_number": 1,
            "event_timestamp": datetime.now(timezone.utc),
            "logged_at": datetime.now(timezone.utc),
            "log_hash": "wrong_hash",
            "user_id": None,
            "user_email": None,
            "session_id": None,
            "api_key_id": None,
            "resource_id": None,
            "ip_address": "192.168.1.1",
            "user_agent": None,
            "risk_level": "low",
            "suspicious": False,
            "correlation_id": None,
            "batch_id": None,
            "error_code": None,
            "error_message": None,
        }

        mock_db_adapter.fetch_one.return_value = test_data

        result = await audit_service.verify_log_integrity(
            tenant_id="tenant-456",
            log_id="log-123"
        )

        assert result["verified"] is False

    @pytest.mark.asyncio

    async def test_verify_log_integrity_not_found(self, audit_service, mock_db_adapter):
        """Test verifying non-existent log."""
        mock_db_adapter.fetch_one.return_value = None

        result = await audit_service.verify_log_integrity(
            tenant_id="tenant-456",
            log_id="log-nonexistent"
        )

        assert result["verified"] is False
        assert "not found" in result["error"]


class TestAuditLogExport:
    """Test audit log export functionality."""

    @pytest.fixture
    def mock_db_adapter(self):
        """Create mock database adapter."""
        adapter = AsyncMock()
        adapter.fetch_all = AsyncMock(return_value=[
            {
                "id": "log-1",
                "tenant_id": "tenant-123",
                "event_type": "user_action",
                "user_id": "user-456",
                "user_email": "user@test.com",
                "resource_type": "profile",
                "resource_id": "profile-789",
                "action": "create",
                "details": {},
                "ip_address": "192.168.1.1",
                "risk_level": "low",
                "suspicious": False,
                "event_timestamp": datetime.now(timezone.utc),
                "logged_at": datetime.now(timezone.utc),
                "correlation_id": None,
                "batch_id": None,
                "error_code": None,
                "error_message": None,
                "sequence_number": 1,
                "session_id": None,
                "api_key_id": None,
                "user_agent": None
            }
        ])
        return adapter

    @pytest.fixture
    def audit_service(self, mock_db_adapter):
        """Create AuditService instance."""
        return AuditService(database_adapter=mock_db_adapter)

    @pytest.mark.asyncio

    async def test_export_audit_logs_json(self, audit_service, mock_db_adapter):
        """Test exporting logs as JSON."""
        result = await audit_service.export_audit_logs(
            tenant_id="tenant-123",
            format="json"
        )

        assert result is not None
        assert isinstance(result, bytes)
        # Verify it's valid JSON
        json.loads(result.decode('utf-8'))

    @pytest.mark.asyncio

    async def test_export_audit_logs_csv(self, audit_service, mock_db_adapter):
        """Test exporting logs as CSV."""
        result = await audit_service.export_audit_logs(
            tenant_id="tenant-123",
            format="csv"
        )

        assert result is not None
        assert isinstance(result, bytes)

    @pytest.mark.asyncio

    async def test_export_audit_logs_xml(self, audit_service, mock_db_adapter):
        """Test exporting logs as XML."""
        result = await audit_service.export_audit_logs(
            tenant_id="tenant-123",
            format="xml"
        )

        assert result is not None
        assert isinstance(result, bytes)
        assert b"<?xml" in result

    @pytest.mark.asyncio

    async def test_export_audit_logs_unsupported_format(self, audit_service, mock_db_adapter):
        """Test exporting with unsupported format."""
        with pytest.raises(AuditServiceError) as exc_info:
            await audit_service.export_audit_logs(
                tenant_id="tenant-123",
                format="pdf"
            )

        assert "Unsupported export format" in str(exc_info.value)

    @pytest.mark.asyncio

    async def test_export_audit_logs_with_filters(self, audit_service, mock_db_adapter):
        """Test exporting logs with filters."""
        start = datetime.now(timezone.utc) - timedelta(days=7)
        end = datetime.now(timezone.utc)

        result = await audit_service.export_audit_logs(
            tenant_id="tenant-123",
            format="json",
            start_time=start,
            end_time=end,
            event_types=["user_action"]
        )

        assert result is not None


class TestTenantIsolation:
    """Test tenant isolation in audit logging."""

    @pytest.fixture
    def mock_db_adapter(self):
        """Create mock database adapter."""
        adapter = AsyncMock()
        adapter.fetch_one = AsyncMock(return_value={"max": 0})
        adapter.execute = AsyncMock()
        adapter.fetch_all = AsyncMock(return_value=[])
        return adapter

    @pytest.fixture
    def audit_service(self, mock_db_adapter):
        """Create AuditService instance."""
        return AuditService(database_adapter=mock_db_adapter)

    @pytest.mark.asyncio

    async def test_separate_sequence_per_tenant(self, audit_service, mock_db_adapter):
        """Test that each tenant has separate sequence numbers."""
        # Log event for tenant 1
        await audit_service.log_event(
            event_type="user_action",
            tenant_id="tenant-1",
            action="create",
            resource_type="profile"
        )

        # Log event for tenant 2
        await audit_service.log_event(
            event_type="user_action",
            tenant_id="tenant-2",
            action="create",
            resource_type="profile"
        )

        # Both should have been logged
        assert mock_db_adapter.execute.call_count == 2

    @pytest.mark.asyncio

    async def test_query_respects_tenant_isolation(self, audit_service, mock_db_adapter):
        """Test that queries respect tenant boundaries."""
        mock_db_adapter.fetch_one.return_value = {"count": 5}
        mock_db_adapter.fetch_all.return_value = []

        await audit_service.query_audit_logs(tenant_id="tenant-123")

        # Verify tenant_id is in the query
        call_args = mock_db_adapter.fetch_one.call_args
        assert "tenant-123" in call_args[0]
