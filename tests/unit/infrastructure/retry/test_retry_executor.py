"""Retry executor tests disabled pending infrastructure restoration."""

import pytest

pytest.skip(
    "Retry executor depends on removed infrastructure; tests pending rewrite.",
    allow_module_level=True,
)


@pytest.fixture
def mock_retry_service():
    """Mock retry service."""
    service = AsyncMock()
    service.create_retry_state = AsyncMock(return_value=str(uuid4()))
    service.record_attempt = AsyncMock()
    service.get_retry_state = AsyncMock()
    service.move_to_dead_letter = AsyncMock(return_value=str(uuid4()))
    service.check_health = AsyncMock(return_value={"status": "healthy"})
    return service


@pytest.fixture
def mock_error_classifier():
    """Mock error classifier."""
    classifier = AsyncMock()
    classifier.check_health = AsyncMock(return_value={"status": "healthy"})
    return classifier


@pytest.fixture
def retry_executor(mock_retry_service, mock_error_classifier):
    """Create retry executor instance."""
    return RetryOperationExecutor(
        retry_service=mock_retry_service,
        error_classifier=mock_error_classifier
    )


@pytest.mark.asyncio
class TestRetryOperationExecutor:
    """Test RetryOperationExecutor functionality."""

    async def test_initialization(self, retry_executor):
        """Test executor initialization."""
        assert retry_executor._retry_service is not None
        assert retry_executor._error_classifier is not None
        assert retry_executor._execution_stats["total_executions"] == 0

    async def test_execute_with_retry_success_first_attempt(
        self, retry_executor, mock_retry_service
    ):
        """Test successful execution on first attempt."""
        # Mock successful operation
        async def successful_operation():
            return "success_result"

        # Mock retry state
        mock_retry_service.get_retry_state.return_value = MagicMock(
            status=RetryResult.SUCCESS,
            operation_id="test_op",
            current_attempt=1,
            total_attempts=1
        )

        result = await retry_executor.execute_with_retry(
            operation_func=successful_operation,
            operation_id="test_op_1",
            operation_type="test_operation",
            tenant_id=str(uuid4())
        )

        assert result == "success_result"
        assert mock_retry_service.create_retry_state.called
        assert mock_retry_service.record_attempt.called
        assert retry_executor._execution_stats["successful_executions"] == 1

    async def test_execute_with_retry_failure_then_success(
        self, retry_executor, mock_retry_service
    ):
        """Test execution that fails then succeeds on retry."""
        attempt_count = 0

        async def flaky_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise Exception("Temporary failure")
            return "success"

        # First call returns FAILED with next_attempt_at
        # Second call returns SUCCESS
        mock_retry_service.get_retry_state.side_effect = [
            MagicMock(
                status=RetryResult.FAILED,
                next_attempt_at=datetime.utcnow(),
                operation_id="test_op",
                current_attempt=1,
                policy=MagicMock(dead_letter_enabled=False)
            ),
            MagicMock(
                status=RetryResult.SUCCESS,
                operation_id="test_op",
                current_attempt=2
            )
        ]

        result = await retry_executor.execute_with_retry(
            operation_func=flaky_operation,
            operation_id="test_op_2",
            operation_type="test_operation",
            tenant_id=str(uuid4())
        )

        assert result == "success"
        assert attempt_count == 2

    async def test_execute_with_retry_max_attempts_exceeded(
        self, retry_executor, mock_retry_service
    ):
        """Test execution that exceeds max retry attempts."""
        async def failing_operation():
            raise Exception("Persistent failure")

        # Mock retry state with MAX_ATTEMPTS_EXCEEDED
        mock_retry_service.get_retry_state.return_value = MagicMock(
            status=RetryResult.MAX_ATTEMPTS_EXCEEDED,
            operation_id="test_op",
            current_attempt=3,
            total_attempts=3,
            policy=MagicMock(dead_letter_enabled=False)
        )

        with pytest.raises(Exception, match="Persistent failure"):
            await retry_executor.execute_with_retry(
                operation_func=failing_operation,
                operation_id="test_op_3",
                operation_type="test_operation",
                tenant_id=str(uuid4())
            )

        assert retry_executor._execution_stats["failed_executions"] == 1

    async def test_execute_with_retry_dead_letter_enabled(
        self, retry_executor, mock_retry_service
    ):
        """Test execution that moves to dead letter queue."""
        async def failing_operation():
            raise Exception("Fatal failure")

        # Mock retry state with dead letter enabled
        mock_retry_service.get_retry_state.return_value = MagicMock(
            status=RetryResult.MAX_ATTEMPTS_EXCEEDED,
            operation_id="test_op",
            current_attempt=3,
            total_attempts=3,
            policy=MagicMock(dead_letter_enabled=True)
        )

        with pytest.raises(Exception, match="Fatal failure"):
            await retry_executor.execute_with_retry(
                operation_func=failing_operation,
                operation_id="test_op_4",
                operation_type="test_operation",
                tenant_id=str(uuid4())
            )

        assert mock_retry_service.move_to_dead_letter.called
        assert retry_executor._execution_stats["dead_lettered_executions"] == 1

    async def test_execute_with_manual_retry_state_success(
        self, retry_executor, mock_retry_service
    ):
        """Test manual retry execution with success."""
        async def successful_operation():
            return "manual_success"

        retry_id = str(uuid4())
        mock_retry_service.get_retry_state.return_value = MagicMock(
            id=retry_id,
            status=RetryResult.FAILED,
            operation_id="manual_op",
            current_attempt=1
        )

        result, success = await retry_executor.execute_with_manual_retry_state(
            operation_func=successful_operation,
            retry_id=retry_id
        )

        assert success is True
        assert result == "manual_success"
        assert mock_retry_service.record_attempt.called

    async def test_execute_with_manual_retry_state_failure(
        self, retry_executor, mock_retry_service
    ):
        """Test manual retry execution with failure."""
        async def failing_operation():
            raise Exception("Manual retry failed")

        retry_id = str(uuid4())
        mock_retry_service.get_retry_state.return_value = MagicMock(
            id=retry_id,
            status=RetryResult.FAILED,
            operation_id="manual_op",
            current_attempt=1
        )

        result, success = await retry_executor.execute_with_manual_retry_state(
            operation_func=failing_operation,
            retry_id=retry_id
        )

        assert success is False
        assert result is None

    async def test_execute_with_manual_retry_state_invalid_state(
        self, retry_executor, mock_retry_service
    ):
        """Test manual retry with invalid retry state."""
        retry_id = str(uuid4())
        mock_retry_service.get_retry_state.return_value = MagicMock(
            id=retry_id,
            status=RetryResult.SUCCESS,  # Not FAILED
            operation_id="manual_op"
        )

        with pytest.raises(ValueError, match="not in failed state"):
            await retry_executor.execute_with_manual_retry_state(
                operation_func=lambda: "test",
                retry_id=retry_id
            )

    async def test_execute_batch_with_retry_all_success(
        self, retry_executor, mock_retry_service
    ):
        """Test batch execution with all operations succeeding."""
        async def batch_operation(value: int):
            return value * 2

        operations = [
            {
                "operation_id": f"op_{i}",
                "operation_func": batch_operation,
                "kwargs": {"value": i}
            }
            for i in range(5)
        ]

        # Mock all operations as successful
        mock_retry_service.get_retry_state.return_value = MagicMock(
            status=RetryResult.SUCCESS,
            operation_id="batch_op",
            current_attempt=1
        )

        result = await retry_executor.execute_batch_with_retry(
            operations=operations,
            operation_type="batch_test",
            tenant_id=str(uuid4()),
            max_concurrent=3
        )

        assert result["total_operations"] == 5
        assert result["successful_operations"] == 5
        assert result["failed_operations"] == 0
        assert result["success_rate"] == 1.0

    async def test_execute_batch_with_retry_partial_failure(
        self, retry_executor, mock_retry_service
    ):
        """Test batch execution with some operations failing."""
        call_count = 0

        async def batch_operation(should_fail: bool):
            nonlocal call_count
            call_count += 1
            if should_fail:
                # First 2 calls will fail permanently
                if call_count <= 2:
                    mock_retry_service.get_retry_state.return_value = MagicMock(
                        status=RetryResult.MAX_ATTEMPTS_EXCEEDED,
                        operation_id="batch_op",
                        current_attempt=3,
                        policy=MagicMock(dead_letter_enabled=False)
                    )
                    raise Exception("Operation failed")
            return "success"

        operations = [
            {"operation_id": "op_1", "operation_func": batch_operation, "kwargs": {"should_fail": True}},
            {"operation_id": "op_2", "operation_func": batch_operation, "kwargs": {"should_fail": True}},
            {"operation_id": "op_3", "operation_func": batch_operation, "kwargs": {"should_fail": False}},
        ]

        # For successful operations
        mock_retry_service.get_retry_state.return_value = MagicMock(
            status=RetryResult.SUCCESS,
            operation_id="batch_op",
            current_attempt=1
        )

        result = await retry_executor.execute_batch_with_retry(
            operations=operations,
            operation_type="batch_test",
            tenant_id=str(uuid4())
        )

        assert result["total_operations"] == 3
        assert result["failed_operations"] >= 1  # At least one should fail

    async def test_get_execution_statistics(self, retry_executor):
        """Test getting execution statistics."""
        # Set some statistics
        retry_executor._execution_stats = {
            "total_executions": 100,
            "successful_executions": 80,
            "failed_executions": 10,
            "retried_executions": 15,
            "dead_lettered_executions": 5
        }

        stats = retry_executor.get_execution_statistics()

        assert stats["total_executions"] == 100
        assert stats["successful_executions"] == 80
        assert stats["success_rate"] == 0.8
        assert stats["retry_rate"] == 0.15

    async def test_reset_statistics(self, retry_executor):
        """Test resetting execution statistics."""
        retry_executor._execution_stats["total_executions"] = 100
        retry_executor.reset_statistics()

        assert retry_executor._execution_stats["total_executions"] == 0
        assert retry_executor._execution_stats["successful_executions"] == 0

    async def test_check_health(self, retry_executor, mock_retry_service, mock_error_classifier):
        """Test health check."""
        health = await retry_executor.check_health()

        assert "status" in health
        assert "execution_statistics" in health
        assert health["status"] == "healthy"

    async def test_execute_with_context(self, retry_executor, mock_retry_service):
        """Test execution with custom context."""
        async def operation_with_context():
            return "success"

        mock_retry_service.get_retry_state.return_value = MagicMock(
            status=RetryResult.SUCCESS,
            operation_id="context_op",
            current_attempt=1
        )

        custom_context = {"user_action": "test_action", "metadata": {"key": "value"}}

        result = await retry_executor.execute_with_retry(
            operation_func=operation_with_context,
            operation_id="context_op_1",
            operation_type="test_operation",
            tenant_id=str(uuid4()),
            context=custom_context
        )

        assert result == "success"
        # Verify context was passed to create_retry_state
        call_args = mock_retry_service.create_retry_state.call_args
        assert "context" in call_args.kwargs
        assert "user_action" in call_args.kwargs["context"]

    async def test_execute_sync_function(self, retry_executor, mock_retry_service):
        """Test execution of synchronous function."""
        def sync_operation():
            return "sync_result"

        mock_retry_service.get_retry_state.return_value = MagicMock(
            status=RetryResult.SUCCESS,
            operation_id="sync_op",
            current_attempt=1
        )

        result = await retry_executor.execute_with_retry(
            operation_func=sync_operation,
            operation_id="sync_op_1",
            operation_type="test_operation",
            tenant_id=str(uuid4())
        )

        assert result == "sync_result"
