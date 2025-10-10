"""
Integration tests for profile restoration workflows.

These tests verify complete profile restoration flows including
soft delete, restoration, embedding regeneration, and search index updates.
"""

import pytest
from datetime import datetime
from uuid import UUID, uuid4
from typing import Any, Dict, List, Optional

from app.application.profile_service import ProfileApplicationService
from app.application.dependencies.profile_dependencies import ProfileDependencies
from app.domain.entities.profile import (
    Profile,
    ProfileData,
    ProfileStatus,
    ProcessingStatus,
    ProcessingMetadata,
    PrivacySettings,
    ExperienceLevel,
    ProfileEmbeddings,
)
from app.domain.value_objects import ProfileId, TenantId, EmailAddress, EmbeddingVector


# ========================================
# Mock Repository and Service Classes
# ========================================


class MockProfileRepository:
    """Mock profile repository with in-memory storage."""

    def __init__(self):
        self.profiles: Dict[str, Profile] = {}
        self.save_count = 0
        self.delete_count = 0
        self.call_log: List[tuple] = []

    async def get_by_id(
        self, profile_id: ProfileId, tenant_id: TenantId
    ) -> Optional[Profile]:
        """Get profile by ID and tenant."""
        self.call_log.append(("get_by_id", str(profile_id.value), str(tenant_id.value)))
        key = str(profile_id.value)
        profile = self.profiles.get(key)

        # Verify tenant matches
        if profile and profile.tenant_id.value != tenant_id.value:
            return None

        return profile

    async def save(self, profile: Profile) -> Profile:
        """Save profile to in-memory storage."""
        from copy import deepcopy
        self.call_log.append(("save", str(profile.id.value)))
        # Store a deep copy to avoid reference issues
        self.profiles[str(profile.id.value)] = deepcopy(profile)
        self.save_count += 1
        return profile

    async def delete(self, profile_id: ProfileId, tenant_id: TenantId) -> bool:
        """Delete profile from storage."""
        self.call_log.append(("delete", str(profile_id.value), str(tenant_id.value)))
        key = str(profile_id.value)
        if key in self.profiles:
            profile = self.profiles[key]
            # Verify tenant matches
            if profile.tenant_id.value == tenant_id.value:
                del self.profiles[key]
                self.delete_count += 1
                return True
        return False

    async def list_by_tenant(
        self,
        tenant_id: TenantId,
        status: Optional[ProcessingStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Profile]:
        """List profiles by tenant."""
        self.call_log.append(("list_by_tenant", str(tenant_id.value), status, limit, offset))
        profiles = [
            p for p in self.profiles.values()
            if p.tenant_id.value == tenant_id.value
        ]
        if status:
            profiles = [p for p in profiles if p.processing.status == status]
        return profiles[offset:offset + limit]

    # Test helper methods
    def get_call_count(self, method: str) -> int:
        """Get count of method calls."""
        return len([call for call in self.call_log if call[0] == method])

    def add_test_profile(self, profile: Profile):
        """Add profile directly to storage."""
        self.profiles[str(profile.id.value)] = profile


class MockEmbeddingService:
    """Mock embedding service."""

    def __init__(self):
        self.generate_embedding_count = 0
        self.call_log: List[tuple] = []
        self.should_fail = False

    async def generate_embedding(self, text: str, **kwargs) -> List[float]:
        """Generate mock embedding vector."""
        self.call_log.append(("generate_embedding", text[:50], kwargs))
        self.generate_embedding_count += 1

        if self.should_fail:
            raise Exception("Mock embedding generation failed")

        # Return mock 3072-dimensional embedding
        return [0.1] * 3072

    def get_call_count(self) -> int:
        """Get count of generate_embedding calls."""
        return self.generate_embedding_count


class MockSearchIndexService:
    """Mock search index service."""

    def __init__(self):
        self.indexed_profiles: Dict[str, Dict[str, Any]] = {}
        self.removed_profiles: List[str] = []
        self.call_log: List[tuple] = []
        self.should_fail = False

    async def update_profile_index(self, profile_id: str, profile_data: Dict) -> None:
        """Update search index for profile."""
        self.call_log.append(("update_profile_index", profile_id))
        if self.should_fail:
            raise Exception("Mock search index update failed")
        self.indexed_profiles[profile_id] = profile_data

    async def remove_profile_index(self, profile_id: str) -> None:
        """Remove profile from search index."""
        self.call_log.append(("remove_profile_index", profile_id))
        if self.should_fail:
            raise Exception("Mock search index removal failed")
        self.removed_profiles.append(profile_id)
        self.indexed_profiles.pop(profile_id, None)

    async def delete_profile_index(self, profile_id: str) -> None:
        """Remove profile from search index (alias)."""
        await self.remove_profile_index(profile_id)

    def get_call_count(self, method: str) -> int:
        """Get count of method calls."""
        return len([call for call in self.call_log if call[0] == method])


class MockUsageService:
    """Mock usage tracking service."""

    def __init__(self):
        self.usage_records: List[Dict[str, Any]] = []
        self.call_log: List[tuple] = []
        self.should_fail = False

    async def track_usage(
        self,
        tenant_id: str,
        resource_type: str,
        amount: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Track usage."""
        self.call_log.append(("track_usage", tenant_id, resource_type, amount, metadata))
        if self.should_fail:
            raise Exception("Mock usage tracking failed")

        self.usage_records.append({
            "tenant_id": tenant_id,
            "resource_type": resource_type,
            "amount": amount,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat(),
        })
        return True

    def get_call_count(self) -> int:
        """Get count of track_usage calls."""
        return len(self.usage_records)

    def get_usage_by_operation(self, operation: str) -> List[Dict[str, Any]]:
        """Get usage records by operation type."""
        return [
            record for record in self.usage_records
            if record["metadata"].get("operation") == operation
        ]


class MockAuditService:
    """Mock audit logging service."""

    def __init__(self):
        self.audit_events: List[Dict[str, Any]] = []
        self.call_log: List[tuple] = []
        self.should_fail = False

    async def log_event(
        self,
        event_type: str,
        tenant_id: str,
        action: str,
        resource_type: str,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> bool:
        """Log audit event."""
        self.call_log.append(("log_event", event_type, tenant_id, action, resource_type))
        if self.should_fail:
            raise Exception("Mock audit logging failed")

        self.audit_events.append({
            "event_type": event_type,
            "tenant_id": tenant_id,
            "action": action,
            "resource_type": resource_type,
            "user_id": user_id,
            "resource_id": resource_id,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat(),
        })
        return True

    def get_call_count(self) -> int:
        """Get count of log_event calls."""
        return len(self.audit_events)

    def get_events_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Get audit events by type."""
        return [event for event in self.audit_events if event["event_type"] == event_type]


class MockBackgroundTasks:
    """Mock background task scheduler."""

    def __init__(self):
        self.tasks: List[tuple] = []
        self.executed_tasks: List[tuple] = []

    def add_task(self, func, *args, **kwargs):
        """Add task to queue."""
        self.tasks.append((func, args, kwargs))

    async def execute_all(self):
        """Execute all scheduled tasks."""
        for func, args, kwargs in self.tasks:
            result = func(*args, **kwargs)
            # If coroutine, await it
            if hasattr(result, '__await__'):
                await result
            self.executed_tasks.append((func.__name__, args, kwargs))

    def get_task_count(self) -> int:
        """Get count of scheduled tasks."""
        return len(self.tasks)

    def get_executed_task_count(self) -> int:
        """Get count of executed tasks."""
        return len(self.executed_tasks)


# ========================================
# Test Fixtures and Helpers
# ========================================


def create_test_profile(
    profile_id: Optional[UUID] = None,
    tenant_id: Optional[UUID] = None,
    status: ProfileStatus = ProfileStatus.ACTIVE,
    email: str = "test@example.com",
) -> Profile:
    """Create a test profile with sensible defaults."""
    profile_id = profile_id or uuid4()
    tenant_id = tenant_id or uuid4()

    profile_data = ProfileData(
        name="John Doe",
        email=EmailAddress(email),
        summary="Experienced software engineer with 5 years of experience",
        headline="Senior Software Engineer",
        experience=[],
        education=[],
        skills=[],
    )

    processing = ProcessingMetadata(
        status=ProcessingStatus.COMPLETED,
        quality_score=0.85,
    )

    profile = Profile(
        id=ProfileId(profile_id),
        tenant_id=TenantId(tenant_id),
        status=status,
        profile_data=profile_data,
        processing=processing,
    )

    return profile


# ========================================
# Integration Tests
# ========================================


class TestProfileRestorationWorkflow:
    """Integration tests for complete profile restoration workflow."""

    def setup_method(self):
        """Setup for each test method."""
        # Create mock services
        self.mock_repo = MockProfileRepository()
        self.mock_embedding = MockEmbeddingService()
        self.mock_search = MockSearchIndexService()
        self.mock_usage = MockUsageService()
        self.mock_audit = MockAuditService()

        # Create dependencies
        self.dependencies = ProfileDependencies(
            profile_repository=self.mock_repo,
            embedding_service=self.mock_embedding,
            search_index_service=self.mock_search,
            usage_service=self.mock_usage,
            audit_service=self.mock_audit,
        )

        # Create service
        self.profile_service = ProfileApplicationService(self.dependencies)

        # Test data
        self.test_tenant_id = TenantId(uuid4())
        self.test_user_id = "user-123"

    @pytest.mark.asyncio
    async def test_complete_restoration_workflow(self):
        """Test end-to-end profile restoration workflow."""
        # Arrange - Create a profile, soft delete it, then restore
        profile = create_test_profile(tenant_id=self.test_tenant_id.value)
        self.mock_repo.add_test_profile(profile)

        # Act - Soft delete the profile
        deleted_profile = await self.profile_service.soft_delete_profile(
            tenant_id=self.test_tenant_id,
            profile_id=profile.id,
            user_id=self.test_user_id,
            reason="Test deletion",
        )

        # Assert - Profile is soft deleted
        assert deleted_profile is not None
        assert deleted_profile.status == ProfileStatus.DELETED
        assert deleted_profile.privacy.deletion_requested is True
        assert "deletion_reason" in deleted_profile.metadata
        assert deleted_profile.metadata["deletion_reason"] == "Test deletion"

        # Act - Restore the profile
        restored_profile = await self.profile_service.restore_profile(
            tenant_id=self.test_tenant_id,
            profile_id=profile.id,
            user_id=self.test_user_id,
        )

        # Assert - Profile is restored
        assert restored_profile is not None
        assert restored_profile.status == ProfileStatus.ACTIVE
        assert restored_profile.privacy.deletion_requested is False
        assert "deletion_reason" not in restored_profile.metadata
        assert "deleted_at" not in restored_profile.metadata

        # Verify repository persistence
        fetched_profile = await self.mock_repo.get_by_id(profile.id, self.test_tenant_id)
        assert fetched_profile is not None
        assert fetched_profile.status == ProfileStatus.ACTIVE
        assert fetched_profile.id == profile.id

    @pytest.mark.asyncio
    async def test_restoration_workflow_with_background_tasks(self):
        """Test restoration with background task scheduling."""
        # Arrange
        profile = create_test_profile(tenant_id=self.test_tenant_id.value)
        profile.status = ProfileStatus.DELETED
        profile.privacy.deletion_requested = True
        self.mock_repo.add_test_profile(profile)

        background_tasks = MockBackgroundTasks()

        # Act - Restore with background tasks
        restored_profile = await self.profile_service.restore_profile(
            tenant_id=self.test_tenant_id,
            profile_id=profile.id,
            user_id=self.test_user_id,
            schedule_task=background_tasks,
        )

        # Assert - Profile is restored
        assert restored_profile is not None
        assert restored_profile.status == ProfileStatus.ACTIVE

        # Verify background tasks were scheduled
        assert background_tasks.get_task_count() > 0
        task_names = [task[0].__name__ for task in background_tasks.tasks]

        # Should schedule: embedding regeneration, search index update, usage tracking, audit logging
        expected_task_names = [
            "_regenerate_profile_embeddings",
            "_update_search_index",
            "_track_profile_restoration_usage",
            "_log_profile_restore_audit",
        ]

        for expected_name in expected_task_names:
            assert any(expected_name in name for name in task_names), \
                f"Expected task '{expected_name}' not found in {task_names}"

        # Execute tasks and verify they run
        await background_tasks.execute_all()

        # Verify services were called
        assert self.mock_embedding.get_call_count() > 0, "Embedding service should be called"
        assert self.mock_search.get_call_count("update_profile_index") > 0, "Search index should be updated"
        assert self.mock_usage.get_call_count() > 0, "Usage should be tracked"
        assert self.mock_audit.get_call_count() > 0, "Audit should be logged"

    @pytest.mark.asyncio
    async def test_restoration_workflow_profile_not_found(self):
        """Test restoration when profile doesn't exist."""
        # Arrange - No profile in repository
        nonexistent_profile_id = ProfileId(uuid4())

        # Act
        result = await self.profile_service.restore_profile(
            tenant_id=self.test_tenant_id,
            profile_id=nonexistent_profile_id,
            user_id=self.test_user_id,
        )

        # Assert - Should return None
        assert result is None

        # Verify repository was queried
        assert self.mock_repo.get_call_count("get_by_id") == 1

        # Verify no side effects occurred
        assert self.mock_repo.save_count == 0
        assert self.mock_embedding.get_call_count() == 0
        assert self.mock_search.get_call_count("update_profile_index") == 0

    @pytest.mark.asyncio
    async def test_restoration_workflow_validation_failure(self):
        """Test restoration when profile is not deleted."""
        # Arrange - Create an active profile (not deleted)
        profile = create_test_profile(
            tenant_id=self.test_tenant_id.value,
            status=ProfileStatus.ACTIVE,
        )
        self.mock_repo.add_test_profile(profile)

        # Act & Assert - Should raise ValueError
        with pytest.raises(ValueError, match="Cannot restore profile"):
            await self.profile_service.restore_profile(
                tenant_id=self.test_tenant_id,
                profile_id=profile.id,
                user_id=self.test_user_id,
            )

        # Verify no side effects occurred
        assert self.mock_repo.save_count == 0
        assert self.mock_embedding.get_call_count() == 0
        assert self.mock_search.get_call_count("update_profile_index") == 0
        assert self.mock_usage.get_call_count() == 0
        assert self.mock_audit.get_call_count() == 0

    @pytest.mark.asyncio
    async def test_restoration_workflow_with_repository_persistence(self):
        """Test that repository properly persists restoration."""
        # Arrange - Create and soft delete profile
        profile = create_test_profile(tenant_id=self.test_tenant_id.value)
        self.mock_repo.add_test_profile(profile)

        # Soft delete
        await self.profile_service.soft_delete_profile(
            tenant_id=self.test_tenant_id,
            profile_id=profile.id,
            user_id=self.test_user_id,
            reason="Test",
        )

        initial_save_count = self.mock_repo.save_count

        # Act - Restore profile
        restored = await self.profile_service.restore_profile(
            tenant_id=self.test_tenant_id,
            profile_id=profile.id,
            user_id=self.test_user_id,
        )

        # Assert - Repository save was called
        assert self.mock_repo.save_count > initial_save_count

        # Fetch from repository and verify state
        fetched = await self.mock_repo.get_by_id(profile.id, self.test_tenant_id)
        assert fetched is not None
        assert fetched.status == ProfileStatus.ACTIVE
        assert fetched.privacy.deletion_requested is False
        assert "deletion_reason" not in fetched.metadata
        assert fetched.id == restored.id

    @pytest.mark.asyncio
    async def test_restoration_workflow_triggers_all_background_operations(self):
        """Test all background operations are triggered during restoration."""
        # Arrange
        profile = create_test_profile(tenant_id=self.test_tenant_id.value)
        profile.status = ProfileStatus.DELETED
        profile.privacy.deletion_requested = True
        self.mock_repo.add_test_profile(profile)

        background_tasks = MockBackgroundTasks()

        # Act - Restore profile
        await self.profile_service.restore_profile(
            tenant_id=self.test_tenant_id,
            profile_id=profile.id,
            user_id=self.test_user_id,
            schedule_task=background_tasks,
        )

        # Execute all background tasks
        await background_tasks.execute_all()

        # Assert - Verify embedding regeneration was called
        assert self.mock_embedding.get_call_count() == 1, \
            "Embedding regeneration should be called once"

        # Verify search index update was called
        assert self.mock_search.get_call_count("update_profile_index") == 1, \
            "Search index update should be called once"

        # Verify usage tracking was called with correct operation
        usage_records = self.mock_usage.get_usage_by_operation("restore")
        assert len(usage_records) == 1, "Usage tracking should record restoration"
        assert usage_records[0]["resource_type"] == "profile"
        assert usage_records[0]["tenant_id"] == str(self.test_tenant_id.value)

        # Verify audit logging was called with correct event type
        audit_events = self.mock_audit.get_events_by_type("profile_restored")
        assert len(audit_events) == 1, "Audit should log restoration event"
        assert audit_events[0]["action"] == "restore"
        assert audit_events[0]["resource_type"] == "profile"
        assert audit_events[0]["resource_id"] == str(profile.id.value)

    @pytest.mark.asyncio
    async def test_soft_delete_then_restore_maintains_profile_data(self):
        """Test that profile data is preserved through delete and restore cycle."""
        # Arrange - Create profile with specific data
        profile = create_test_profile(
            tenant_id=self.test_tenant_id.value,
            email="preserve@example.com",
        )
        profile.profile_data.summary = "Original summary that should be preserved"
        profile.profile_data.headline = "Original headline"
        profile.metadata["custom_field"] = "custom_value"
        self.mock_repo.add_test_profile(profile)

        original_name = profile.profile_data.name
        original_email = str(profile.profile_data.email)
        original_summary = profile.profile_data.summary
        original_headline = profile.profile_data.headline

        # Act - Soft delete
        deleted = await self.profile_service.soft_delete_profile(
            tenant_id=self.test_tenant_id,
            profile_id=profile.id,
            user_id=self.test_user_id,
            reason="Test preservation",
        )

        # Restore
        restored = await self.profile_service.restore_profile(
            tenant_id=self.test_tenant_id,
            profile_id=profile.id,
            user_id=self.test_user_id,
        )

        # Assert - All data preserved
        assert restored.profile_data.name == original_name
        assert str(restored.profile_data.email) == original_email
        assert restored.profile_data.summary == original_summary
        assert restored.profile_data.headline == original_headline
        assert restored.metadata.get("custom_field") == "custom_value"

        # Status changed correctly
        assert deleted.status == ProfileStatus.DELETED
        assert restored.status == ProfileStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_restoration_regenerates_embeddings_with_correct_text(self):
        """Test that embedding regeneration uses correct profile text."""
        # Arrange
        profile = create_test_profile(tenant_id=self.test_tenant_id.value)
        profile.status = ProfileStatus.DELETED
        # Update the profile data and trigger searchable text rebuild
        profile.profile_data.summary = "Test summary for embedding generation"
        profile._update_computed_fields()  # Rebuild searchable text
        self.mock_repo.add_test_profile(profile)

        background_tasks = MockBackgroundTasks()

        # Act - Restore profile
        await self.profile_service.restore_profile(
            tenant_id=self.test_tenant_id,
            profile_id=profile.id,
            user_id=self.test_user_id,
            schedule_task=background_tasks,
        )

        # Execute background tasks
        await background_tasks.execute_all()

        # Assert - Embedding service was called
        assert self.mock_embedding.get_call_count() == 1

        # Verify embedding was called with searchable text
        embedding_calls = [
            call for call in self.mock_embedding.call_log
            if call[0] == "generate_embedding"
        ]
        assert len(embedding_calls) == 1

        # Verify the text contains profile data (name, email, summary)
        text_used = embedding_calls[0][1]
        assert "Test summary for embedding generation" in text_used or "John Doe" in text_used

        # Verify profile has new embeddings
        updated_profile = await self.mock_repo.get_by_id(profile.id, self.test_tenant_id)
        assert updated_profile.embeddings.overall is not None
        assert updated_profile.embeddings.overall.dimensions == 3072

    @pytest.mark.asyncio
    async def test_restoration_updates_search_index_with_correct_data(self):
        """Test that search index is updated with correct profile data."""
        # Arrange
        profile = create_test_profile(tenant_id=self.test_tenant_id.value)
        profile.status = ProfileStatus.DELETED
        profile.profile_data.headline = "Senior Backend Engineer"
        self.mock_repo.add_test_profile(profile)

        background_tasks = MockBackgroundTasks()

        # Act - Restore profile
        await self.profile_service.restore_profile(
            tenant_id=self.test_tenant_id,
            profile_id=profile.id,
            user_id=self.test_user_id,
            schedule_task=background_tasks,
        )

        # Execute background tasks
        await background_tasks.execute_all()

        # Assert - Search index was updated
        assert self.mock_search.get_call_count("update_profile_index") == 1

        # Verify indexed data
        indexed_data = self.mock_search.indexed_profiles.get(str(profile.id.value))
        assert indexed_data is not None
        assert indexed_data["profile_id"] == str(profile.id.value)
        assert indexed_data["tenant_id"] == str(self.test_tenant_id.value)
        assert indexed_data["name"] == profile.profile_data.name
        assert indexed_data["headline"] == "Senior Backend Engineer"
        assert indexed_data["status"] == ProfileStatus.ACTIVE.value

    @pytest.mark.asyncio
    async def test_restoration_workflow_with_service_failures(self):
        """Test restoration workflow when background services fail gracefully."""
        # Arrange
        profile = create_test_profile(tenant_id=self.test_tenant_id.value)
        profile.status = ProfileStatus.DELETED
        self.mock_repo.add_test_profile(profile)

        # Make embedding service fail
        self.mock_embedding.should_fail = True

        background_tasks = MockBackgroundTasks()

        # Act - Restore should succeed even if background tasks fail
        restored = await self.profile_service.restore_profile(
            tenant_id=self.test_tenant_id,
            profile_id=profile.id,
            user_id=self.test_user_id,
            schedule_task=background_tasks,
        )

        # Assert - Profile is restored despite embedding failure
        assert restored is not None
        assert restored.status == ProfileStatus.ACTIVE

        # Execute background tasks (some will fail)
        # This should not raise exceptions
        try:
            await background_tasks.execute_all()
        except Exception:
            pass  # Expected to fail gracefully

        # Verify profile is still restored in repository
        fetched = await self.mock_repo.get_by_id(profile.id, self.test_tenant_id)
        assert fetched.status == ProfileStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_restoration_workflow_updates_timestamps(self):
        """Test that restoration properly updates profile timestamps."""
        # Arrange
        profile = create_test_profile(tenant_id=self.test_tenant_id.value)
        profile.status = ProfileStatus.DELETED
        original_created_at = profile.created_at
        original_updated_at = profile.updated_at
        self.mock_repo.add_test_profile(profile)

        # Act - Restore profile
        restored = await self.profile_service.restore_profile(
            tenant_id=self.test_tenant_id,
            profile_id=profile.id,
            user_id=self.test_user_id,
        )

        # Assert - Timestamps updated correctly
        assert restored.created_at == original_created_at, "created_at should not change"
        assert restored.updated_at > original_updated_at, "updated_at should be newer"

    @pytest.mark.asyncio
    async def test_multiple_restore_attempts_on_same_profile(self):
        """Test that restoring an already active profile fails validation."""
        # Arrange - Create and restore profile
        profile = create_test_profile(tenant_id=self.test_tenant_id.value)
        profile.status = ProfileStatus.DELETED
        self.mock_repo.add_test_profile(profile)

        # First restoration
        await self.profile_service.restore_profile(
            tenant_id=self.test_tenant_id,
            profile_id=profile.id,
            user_id=self.test_user_id,
        )

        # Act & Assert - Second restoration should fail
        with pytest.raises(ValueError, match="Cannot restore profile"):
            await self.profile_service.restore_profile(
                tenant_id=self.test_tenant_id,
                profile_id=profile.id,
                user_id=self.test_user_id,
            )


class TestProfileSoftDeleteWorkflow:
    """Integration tests for profile soft deletion workflow."""

    def setup_method(self):
        """Setup for each test method."""
        # Create mock services
        self.mock_repo = MockProfileRepository()
        self.mock_search = MockSearchIndexService()
        self.mock_usage = MockUsageService()
        self.mock_audit = MockAuditService()

        # Create dependencies
        self.dependencies = ProfileDependencies(
            profile_repository=self.mock_repo,
            search_index_service=self.mock_search,
            usage_service=self.mock_usage,
            audit_service=self.mock_audit,
        )

        # Create service
        self.profile_service = ProfileApplicationService(self.dependencies)

        # Test data
        self.test_tenant_id = TenantId(uuid4())
        self.test_user_id = "user-456"

    @pytest.mark.asyncio
    async def test_soft_delete_marks_profile_as_deleted(self):
        """Test that soft delete properly marks profile as deleted."""
        # Arrange
        profile = create_test_profile(tenant_id=self.test_tenant_id.value)
        self.mock_repo.add_test_profile(profile)

        # Act
        deleted = await self.profile_service.soft_delete_profile(
            tenant_id=self.test_tenant_id,
            profile_id=profile.id,
            user_id=self.test_user_id,
            reason="User request",
        )

        # Assert
        assert deleted is not None
        assert deleted.status == ProfileStatus.DELETED
        assert deleted.privacy.deletion_requested is True
        assert deleted.metadata["deletion_reason"] == "User request"
        assert "deleted_at" in deleted.metadata

    @pytest.mark.asyncio
    async def test_soft_delete_removes_from_search_index(self):
        """Test that soft delete removes profile from search index."""
        # Arrange
        profile = create_test_profile(tenant_id=self.test_tenant_id.value)
        self.mock_repo.add_test_profile(profile)

        background_tasks = MockBackgroundTasks()

        # Act
        await self.profile_service.soft_delete_profile(
            tenant_id=self.test_tenant_id,
            profile_id=profile.id,
            user_id=self.test_user_id,
            schedule_task=background_tasks,
        )

        # Execute background tasks
        await background_tasks.execute_all()

        # Assert
        assert self.mock_search.get_call_count("remove_profile_index") == 1
        assert str(profile.id.value) in self.mock_search.removed_profiles

    @pytest.mark.asyncio
    async def test_soft_delete_tracks_usage_and_logs_audit(self):
        """Test that soft delete tracks usage and logs audit event."""
        # Arrange
        profile = create_test_profile(tenant_id=self.test_tenant_id.value)
        self.mock_repo.add_test_profile(profile)

        background_tasks = MockBackgroundTasks()

        # Act
        await self.profile_service.soft_delete_profile(
            tenant_id=self.test_tenant_id,
            profile_id=profile.id,
            user_id=self.test_user_id,
            reason="Compliance",
            schedule_task=background_tasks,
        )

        # Execute background tasks
        await background_tasks.execute_all()

        # Assert - Usage tracked
        usage_records = self.mock_usage.get_usage_by_operation("delete")
        assert len(usage_records) == 1
        assert usage_records[0]["metadata"]["deletion_type"] == "soft_delete"

        # Assert - Audit logged
        audit_events = self.mock_audit.get_events_by_type("profile_deleted")
        assert len(audit_events) == 1
        assert audit_events[0]["details"]["deletion_type"] == "soft_delete"
        assert audit_events[0]["details"]["reason"] == "Compliance"