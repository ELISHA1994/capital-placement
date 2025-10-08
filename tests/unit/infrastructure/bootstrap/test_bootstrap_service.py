"""Bootstrap service tests disabled until infrastructure is restored."""

import pytest

pytest.skip(
    "Bootstrap workflow removed during migration; tests pending rewrite.",
    allow_module_level=True,
)


class TestBootstrapServiceInitialization:
    """Test BootstrapService initialization and basic operations."""

    @pytest.fixture
    def mock_tenant_repo(self):
        """Create mock tenant repository."""
        repo = AsyncMock()
        repo.get = AsyncMock(return_value=None)
        repo.create = AsyncMock(return_value={
            "id": SYSTEM_TENANT_ID,
            "name": SYSTEM_TENANT_NAME,
            "is_active": True
        })
        repo.list_all = AsyncMock(return_value=[])
        return repo

    @pytest.fixture
    def mock_user_repo(self):
        """Create mock user repository."""
        repo = AsyncMock()
        repo.get_by_tenant = AsyncMock(return_value=[])
        repo.create = AsyncMock(return_value={
            "id": "user-123",
            "email": "admin@test.com",
            "tenant_id": SYSTEM_TENANT_ID,
            "roles": [SUPER_ADMIN_ROLE],
            "is_active": True,
            "first_name": "Admin",
            "last_name": "User"
        })
        return repo

    @pytest.fixture
    def mock_auth_service(self):
        """Create mock authentication service."""
        return AsyncMock()

    @pytest.fixture
    def bootstrap_service(self, mock_tenant_repo, mock_user_repo, mock_auth_service):
        """Create BootstrapService instance with mocked dependencies."""
        return BootstrapService(
            tenant_repository=mock_tenant_repo,
            user_repository=mock_user_repo,
            auth_service=mock_auth_service
        )

    @pytest.mark.asyncio

    async def test_initialization(self, bootstrap_service, mock_tenant_repo, mock_user_repo):
        """Test service initializes with correct dependencies."""
        assert bootstrap_service.tenant_repo == mock_tenant_repo
        assert bootstrap_service.user_repo == mock_user_repo
        assert bootstrap_service.auth_service is not None


class TestSystemTenantCreation:
    """Test system tenant creation and management."""

    @pytest.fixture
    def mock_tenant_repo(self):
        """Create mock tenant repository."""
        repo = AsyncMock()
        repo.get = AsyncMock(return_value=None)
        repo.create = AsyncMock(return_value={
            "id": SYSTEM_TENANT_ID,
            "name": SYSTEM_TENANT_NAME,
            "slug": "system",
            "display_name": "System Administration",
            "is_system_tenant": True,
            "is_active": True
        })
        return repo

    @pytest.fixture
    def mock_user_repo(self):
        """Create mock user repository."""
        return AsyncMock()

    @pytest.fixture
    def mock_auth_service(self):
        """Create mock authentication service."""
        return AsyncMock()

    @pytest.fixture
    def bootstrap_service(self, mock_tenant_repo, mock_user_repo, mock_auth_service):
        """Create BootstrapService instance."""
        return BootstrapService(
            tenant_repository=mock_tenant_repo,
            user_repository=mock_user_repo,
            auth_service=mock_auth_service
        )

    @pytest.mark.asyncio

    async def test_ensure_system_tenant_creates_new(self, bootstrap_service, mock_tenant_repo):
        """Test creating system tenant when it doesn't exist."""
        result = await bootstrap_service.ensure_system_tenant()

        assert result is not None
        assert result["id"] == SYSTEM_TENANT_ID
        assert result["name"] == SYSTEM_TENANT_NAME
        assert result["is_system_tenant"] is True
        mock_tenant_repo.create.assert_called_once()

    @pytest.mark.asyncio

    async def test_ensure_system_tenant_returns_existing(self, bootstrap_service, mock_tenant_repo):
        """Test returning existing system tenant."""
        existing_tenant = {
            "id": SYSTEM_TENANT_ID,
            "name": SYSTEM_TENANT_NAME,
            "is_active": True
        }
        mock_tenant_repo.get.return_value = existing_tenant

        result = await bootstrap_service.ensure_system_tenant()

        assert result == existing_tenant
        mock_tenant_repo.create.assert_not_called()

    @pytest.mark.asyncio

    async def test_ensure_system_tenant_error_handling(self, bootstrap_service, mock_tenant_repo):
        """Test error handling when tenant creation fails."""
        mock_tenant_repo.create.side_effect = Exception("Database error")

        with pytest.raises(Exception) as exc_info:
            await bootstrap_service.ensure_system_tenant()

        assert "Database error" in str(exc_info.value)


class TestSuperAdminManagement:
    """Test super admin creation and checking."""

    @pytest.fixture
    def mock_tenant_repo(self):
        """Create mock tenant repository."""
        repo = AsyncMock()
        repo.get = AsyncMock(return_value={
            "id": SYSTEM_TENANT_ID,
            "name": SYSTEM_TENANT_NAME
        })
        return repo

    @pytest.fixture
    def mock_user_repo(self):
        """Create mock user repository."""
        repo = AsyncMock()
        repo.get_by_tenant = AsyncMock(return_value=[])
        repo.create = AsyncMock(return_value={
            "id": "user-123",
            "email": "admin@test.com",
            "tenant_id": SYSTEM_TENANT_ID,
            "roles": [SUPER_ADMIN_ROLE],
            "is_active": True,
            "first_name": "Admin",
            "last_name": "User"
        })
        return repo

    @pytest.fixture
    def mock_auth_service(self):
        """Create mock authentication service."""
        return AsyncMock()

    @pytest.fixture
    def bootstrap_service(self, mock_tenant_repo, mock_user_repo, mock_auth_service):
        """Create BootstrapService instance."""
        return BootstrapService(
            tenant_repository=mock_tenant_repo,
            user_repository=mock_user_repo,
            auth_service=mock_auth_service
        )

    @pytest.mark.asyncio

    async def test_has_super_admin_returns_false_when_none(self, bootstrap_service, mock_user_repo):
        """Test checking for super admin when none exists."""
        mock_user_repo.get_by_tenant.return_value = []

        result = await bootstrap_service.has_super_admin()

        assert result is False

    @pytest.mark.asyncio

    async def test_has_super_admin_returns_true_when_exists(self, bootstrap_service, mock_user_repo):
        """Test checking for super admin when one exists."""
        mock_user_repo.get_by_tenant.return_value = [
            {"id": "user-1", "roles": [SUPER_ADMIN_ROLE]}
        ]

        result = await bootstrap_service.has_super_admin()

        assert result is True

    @pytest.mark.asyncio

    async def test_has_super_admin_handles_errors(self, bootstrap_service, mock_user_repo):
        """Test error handling when checking for super admin."""
        mock_user_repo.get_by_tenant.side_effect = Exception("Database error")

        result = await bootstrap_service.has_super_admin()

        assert result is False

    @patch('app.infrastructure.bootstrap.bootstrap_service.get_transaction_manager')
    @patch('app.infrastructure.bootstrap.bootstrap_service.password_manager')
    @pytest.mark.asyncio

    async def test_create_super_admin_success(
        self,
        mock_password_manager,
        mock_get_tx_manager,
        bootstrap_service,
        mock_user_repo,
        mock_tenant_repo
    ):
        """Test successful super admin creation."""
        # Mock password validation
        mock_password_manager.validate_password_strength.return_value = {
            "valid": True,
            "errors": []
        }
        mock_password_manager.hash_password.return_value = "hashed_password"

        # Mock transaction manager
        mock_session = AsyncMock()
        mock_tx_context = AsyncMock()
        mock_tx_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_tx_context.__aexit__ = AsyncMock(return_value=None)

        mock_tx_manager = AsyncMock()
        mock_tx_manager.transaction.return_value = mock_tx_context
        mock_get_tx_manager.return_value = mock_tx_manager

        # Mock repository returns within transaction
        mock_user_repo.get_by_tenant.return_value = []
        mock_tenant_repo.get.return_value = {
            "id": SYSTEM_TENANT_ID,
            "name": SYSTEM_TENANT_NAME
        }
        mock_user_repo.create.return_value = {
            "id": "user-123",
            "email": "admin@test.com",
            "tenant_id": SYSTEM_TENANT_ID,
            "roles": [SUPER_ADMIN_ROLE],
            "is_active": True,
            "first_name": "Admin",
            "last_name": "User"
        }

        result = await bootstrap_service.create_super_admin(
            email="admin@test.com",
            password="StrongPassword123!",
            full_name="Admin User"
        )

        assert result is not None
        assert "user" in result
        assert result["user"]["email"] == "admin@test.com"
        assert SUPER_ADMIN_ROLE in result["user"]["roles"]

    @patch('app.infrastructure.bootstrap.bootstrap_service.get_transaction_manager')
    @pytest.mark.asyncio

    async def test_create_super_admin_fails_when_exists(
        self,
        mock_get_tx_manager,
        bootstrap_service,
        mock_user_repo
    ):
        """Test super admin creation fails when one already exists."""
        # Mock transaction manager
        mock_session = AsyncMock()
        mock_tx_context = AsyncMock()
        mock_tx_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_tx_context.__aexit__ = AsyncMock(return_value=None)

        mock_tx_manager = AsyncMock()
        mock_tx_manager.transaction.return_value = mock_tx_context
        mock_get_tx_manager.return_value = mock_tx_manager

        # Existing super admin
        mock_user_repo.get_by_tenant.return_value = [
            {"id": "user-1", "roles": [SUPER_ADMIN_ROLE]}
        ]

        with pytest.raises(ValueError) as exc_info:
            await bootstrap_service.create_super_admin(
                email="admin@test.com",
                password="StrongPassword123!",
                full_name="Admin User"
            )

        assert "already exists" in str(exc_info.value)

    @patch('app.infrastructure.bootstrap.bootstrap_service.get_transaction_manager')
    @patch('app.infrastructure.bootstrap.bootstrap_service.password_manager')
    @pytest.mark.asyncio

    async def test_create_super_admin_validates_password(
        self,
        mock_password_manager,
        mock_get_tx_manager,
        bootstrap_service,
        mock_user_repo
    ):
        """Test password validation during super admin creation."""
        # Mock password validation failure
        mock_password_manager.validate_password_strength.return_value = {
            "valid": False,
            "errors": ["Password too weak"]
        }

        # Mock transaction manager
        mock_session = AsyncMock()
        mock_tx_context = AsyncMock()
        mock_tx_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_tx_context.__aexit__ = AsyncMock(return_value=None)

        mock_tx_manager = AsyncMock()
        mock_tx_manager.transaction.return_value = mock_tx_context
        mock_get_tx_manager.return_value = mock_tx_manager

        mock_user_repo.get_by_tenant.return_value = []

        with pytest.raises(ValueError) as exc_info:
            await bootstrap_service.create_super_admin(
                email="admin@test.com",
                password="weak",
                full_name="Admin User"
            )

        assert "Password validation failed" in str(exc_info.value)


class TestSystemStatus:
    """Test system status reporting."""

    @pytest.fixture
    def mock_tenant_repo(self):
        """Create mock tenant repository."""
        repo = AsyncMock()
        repo.get = AsyncMock(return_value={
            "id": SYSTEM_TENANT_ID,
            "name": SYSTEM_TENANT_NAME
        })
        repo.list_all = AsyncMock(return_value=[
            {"id": SYSTEM_TENANT_ID, "name": SYSTEM_TENANT_NAME},
            {"id": "tenant-2", "name": "Tenant 2"}
        ])
        return repo

    @pytest.fixture
    def mock_user_repo(self):
        """Create mock user repository."""
        repo = AsyncMock()
        repo.get_by_tenant = AsyncMock(return_value=[
            {"id": "user-1", "roles": [SUPER_ADMIN_ROLE]}
        ])
        return repo

    @pytest.fixture
    def mock_auth_service(self):
        """Create mock authentication service."""
        return AsyncMock()

    @pytest.fixture
    def bootstrap_service(self, mock_tenant_repo, mock_user_repo, mock_auth_service):
        """Create BootstrapService instance."""
        return BootstrapService(
            tenant_repository=mock_tenant_repo,
            user_repository=mock_user_repo,
            auth_service=mock_auth_service
        )

    @pytest.mark.asyncio

    async def test_get_system_status_initialized(self, bootstrap_service):
        """Test system status when fully initialized."""
        result = await bootstrap_service.get_system_status()

        assert result["initialized"] is True
        assert result["has_system_tenant"] is True
        assert result["has_super_admin"] is True
        assert result["tenant_count"] == 1  # Excludes system tenant
        assert result["system_tenant_id"] == SYSTEM_TENANT_ID

    @pytest.mark.asyncio

    async def test_get_system_status_not_initialized(self, bootstrap_service, mock_tenant_repo):
        """Test system status when not initialized."""
        mock_tenant_repo.get.return_value = None

        result = await bootstrap_service.get_system_status()

        assert result["initialized"] is False
        assert result["has_system_tenant"] is False
        assert result["system_tenant_id"] is None

    @pytest.mark.asyncio

    async def test_initialize_system_success(self, bootstrap_service):
        """Test successful system initialization."""
        result = await bootstrap_service.initialize_system()

        assert result is True

    @pytest.mark.asyncio

    async def test_initialize_system_handles_errors(self, bootstrap_service, mock_tenant_repo):
        """Test system initialization error handling."""
        mock_tenant_repo.get.side_effect = Exception("Database error")

        result = await bootstrap_service.initialize_system()

        assert result is False
