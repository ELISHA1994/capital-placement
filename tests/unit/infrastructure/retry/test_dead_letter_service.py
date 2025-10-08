"""Dead letter service tests disabled until retry infrastructure returns."""

import pytest

pytest.skip(
    "Retry dead-letter infrastructure removed during migration; tests pending rewrite.",
    allow_module_level=True,
)


@pytest.fixture
def dead_letter_service():
    """Create dead letter service instance."""
    return DeadLetterService()


@pytest.mark.asyncio
class TestDeadLetterService:
    """Test DeadLetterService functionality."""

    @patch("app.infrastructure.retry.dead_letter_service.get_postgres_adapter")
    async def test_add_to_dead_letter_queue(
        self, mock_get_adapter, dead_letter_service
    ):
        """Test adding entry to dead letter queue."""
        mock_adapter = AsyncMock()
        mock_get_adapter.return_value = mock_adapter
        mock_adapter.execute = AsyncMock(return_value=str(uuid4()))

        dead_letter_id = await dead_letter_service.add_to_dead_letter_queue(
            retry_id=str(uuid4()),
            operation_id="test_op",
            operation_type="test_type",
            tenant_id=str(uuid4()),
            error_message="Test error",
            context={"key": "value"}
        )

        assert dead_letter_id is not None
        assert mock_adapter.execute.called

    @patch("app.infrastructure.retry.dead_letter_service.get_postgres_adapter")
    async def test_get_dead_letter_entry(
        self, mock_get_adapter, dead_letter_service
    ):
        """Test getting dead letter entry."""
        mock_adapter = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        entry_id = str(uuid4())
        mock_entry = MagicMock(
            id=entry_id,
            operation_id="test_op",
            resolved=False
        )
        mock_adapter.fetch_one = AsyncMock(return_value=mock_entry)

        entry = await dead_letter_service.get_dead_letter_entry(entry_id)

        assert entry is not None
        assert entry.id == entry_id
        assert mock_adapter.fetch_one.called

    @patch("app.infrastructure.retry.dead_letter_service.get_postgres_adapter")
    async def test_get_dead_letter_queue_unresolved(
        self, mock_get_adapter, dead_letter_service
    ):
        """Test getting unresolved dead letter entries."""
        mock_adapter = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        mock_entries = [
            MagicMock(id=str(uuid4()), resolved=False)
            for _ in range(10)
        ]
        mock_adapter.fetch_all = AsyncMock(return_value=mock_entries)
        mock_adapter.fetch_one = AsyncMock(return_value={"total": 10})

        result = await dead_letter_service.get_dead_letter_queue(
            resolved=False,
            limit=20
        )

        assert result["total_count"] == 10
        assert len(result["items"]) == 10
        assert all(not entry.resolved for entry in result["items"])

    @patch("app.infrastructure.retry.dead_letter_service.get_postgres_adapter")
    async def test_get_dead_letter_queue_with_filters(
        self, mock_get_adapter, dead_letter_service
    ):
        """Test getting dead letter queue with filters."""
        mock_adapter = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        tenant_id = str(uuid4())
        mock_entries = [MagicMock(id=str(uuid4()), tenant_id=tenant_id)]
        mock_adapter.fetch_all = AsyncMock(return_value=mock_entries)
        mock_adapter.fetch_one = AsyncMock(return_value={"total": 1})

        result = await dead_letter_service.get_dead_letter_queue(
            tenant_id=tenant_id,
            operation_type="test_type",
            start_date=datetime.utcnow() - timedelta(days=7),
            end_date=datetime.utcnow()
        )

        assert result["total_count"] == 1
        assert mock_adapter.fetch_all.called

    @patch("app.infrastructure.retry.dead_letter_service.get_postgres_adapter")
    async def test_resolve_dead_letter_entry(
        self, mock_get_adapter, dead_letter_service
    ):
        """Test resolving dead letter entry."""
        mock_adapter = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        entry_id = str(uuid4())
        mock_entry = MagicMock(id=entry_id, resolved=False)
        mock_adapter.fetch_one = AsyncMock(return_value=mock_entry)
        mock_adapter.execute = AsyncMock()

        success = await dead_letter_service.resolve_dead_letter_entry(
            entry_id=entry_id,
            resolution_notes="Manually resolved"
        )

        assert success is True
        assert mock_adapter.execute.called

    @patch("app.infrastructure.retry.dead_letter_service.get_postgres_adapter")
    async def test_resolve_dead_letter_entry_not_found(
        self, mock_get_adapter, dead_letter_service
    ):
        """Test resolving non-existent entry."""
        mock_adapter = AsyncMock()
        mock_get_adapter.return_value = mock_adapter
        mock_adapter.fetch_one = AsyncMock(return_value=None)

        success = await dead_letter_service.resolve_dead_letter_entry(
            entry_id=str(uuid4())
        )

        assert success is False

    @patch("app.infrastructure.retry.dead_letter_service.get_postgres_adapter")
    async def test_replay_dead_letter_entry(
        self, mock_get_adapter, dead_letter_service
    ):
        """Test replaying dead letter entry."""
        mock_adapter = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        entry_id = str(uuid4())
        mock_entry = MagicMock(
            id=entry_id,
            retry_id=str(uuid4()),
            operation_id="test_op",
            resolved=False
        )
        mock_adapter.fetch_one = AsyncMock(return_value=mock_entry)
        mock_adapter.execute = AsyncMock(return_value=str(uuid4()))

        new_retry_id = await dead_letter_service.replay_dead_letter_entry(
            entry_id=entry_id
        )

        assert new_retry_id is not None
        assert mock_adapter.execute.called

    @patch("app.infrastructure.retry.dead_letter_service.get_postgres_adapter")
    async def test_get_dead_letter_statistics(
        self, mock_get_adapter, dead_letter_service
    ):
        """Test getting dead letter statistics."""
        mock_adapter = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        mock_stats = {
            "total_entries": 100,
            "resolved_entries": 80,
            "unresolved_entries": 20
        }
        mock_adapter.fetch_one = AsyncMock(return_value=mock_stats)

        stats = await dead_letter_service.get_dead_letter_statistics(
            start_time=datetime.utcnow() - timedelta(days=30),
            end_time=datetime.utcnow()
        )

        assert stats is not None
        assert "total_entries" in stats

    @patch("app.infrastructure.retry.dead_letter_service.get_postgres_adapter")
    async def test_cleanup_old_resolved_entries(
        self, mock_get_adapter, dead_letter_service
    ):
        """Test cleanup of old resolved entries."""
        mock_adapter = AsyncMock()
        mock_get_adapter.return_value = mock_adapter
        mock_adapter.execute = AsyncMock(return_value=15)

        deleted_count = await dead_letter_service.cleanup_old_resolved_entries(
            days_to_keep=30
        )

        assert deleted_count == 15
        assert mock_adapter.execute.called

    @patch("app.infrastructure.retry.dead_letter_service.get_postgres_adapter")
    async def test_check_health(
        self, mock_get_adapter, dead_letter_service
    ):
        """Test health check."""
        mock_adapter = AsyncMock()
        mock_get_adapter.return_value = mock_adapter
        mock_adapter.fetch_one = AsyncMock(return_value={"count": 50})

        health = await dead_letter_service.check_health()

        assert "status" in health
        assert "unresolved_count" in health
        assert health["status"] == "healthy"
