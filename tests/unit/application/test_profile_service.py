"""
Comprehensive unit tests for ProfileApplicationService methods.

This test suite covers:

ProfileRestoration Tests:
- Successful profile restoration workflow
- Profile not found scenarios
- Validation failures (profile not deleted)
- Background task scheduling
- Optional service handling
- Deletion metadata cleanup

ProfileAnalytics Tests:
- Successful analytics retrieval
- Profile not found scenarios
- View count and search appearances tracking
- Completeness score calculation
- Last viewed timestamp handling
- Advanced metrics placeholders (MVP)
- Repository error propagation
- Edge cases (zero views, different time ranges)

ProfileViewTracking Tests:
- get_profile() schedules view tracking in background
- View tracking works without background tasks
- _record_profile_view() increments view count
- _record_profile_view() updates last_viewed_at timestamp
- Graceful handling when profile not found during view recording
- Error handling when repository save fails
- No tracking scheduled when profile doesn't exist
- View tracking preserves other analytics fields
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from app.application.dependencies.profile_dependencies import ProfileDependencies
from app.application.profile_service import ProfileApplicationService
from app.domain.entities.profile import (
    Education,
    Experience,
    Location,
    PrivacySettings,
    ProcessingMetadata,
    ProcessingStatus,
    Profile,
    ProfileData,
    ProfileStatus,
    Skill,
)
from app.domain.repositories.profile_repository import IProfileRepository
from app.domain.value_objects import EmailAddress, ProfileId, SkillName, TenantId

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_profile_repository():
    """Mock profile repository."""
    repo = Mock(spec=IProfileRepository)
    repo.get_by_id = AsyncMock()
    repo.save = AsyncMock()
    repo.delete = AsyncMock()
    repo.list_by_tenant = AsyncMock()
    return repo


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    service = Mock()
    service.generate_embedding = AsyncMock(return_value=[0.1] * 3072)
    return service


@pytest.fixture
def mock_search_index_service():
    """Mock search index service."""
    service = Mock()
    service.update_profile_index = AsyncMock()
    service.remove_profile_index = AsyncMock()
    return service


@pytest.fixture
def mock_usage_service():
    """Mock usage tracking service."""
    service = Mock()
    service.track_usage = AsyncMock()
    return service


@pytest.fixture
def mock_audit_service():
    """Mock audit logging service."""
    service = Mock()
    service.log_event = AsyncMock()
    return service


@pytest.fixture
def profile_dependencies(
    mock_profile_repository,
    mock_embedding_service,
    mock_search_index_service,
    mock_usage_service,
    mock_audit_service,
):
    """ProfileDependencies with all mocked services."""
    return ProfileDependencies(
        profile_repository=mock_profile_repository,
        embedding_service=mock_embedding_service,
        search_index_service=mock_search_index_service,
        usage_service=mock_usage_service,
        audit_service=mock_audit_service,
    )


@pytest.fixture
def profile_service(profile_dependencies: ProfileDependencies):
    """ProfileApplicationService with mocked dependencies."""
    return ProfileApplicationService(dependencies=profile_dependencies)


@pytest.fixture
def sample_deleted_profile():
    """A deleted Profile entity for testing restoration."""
    profile_id = ProfileId(uuid4())
    tenant_id = TenantId(uuid4())

    profile_data = ProfileData(
        name="John Doe",
        email=EmailAddress("john.doe@example.com"),
        phone=None,
        location=Location(city="San Francisco", state="CA", country="USA"),
        summary="Senior Software Engineer with 10 years of experience",
        headline="Senior Software Engineer",
        experience=[
            Experience(
                title="Senior Software Engineer",
                company="Tech Corp",
                start_date="2020-01-01",
                description="Lead development of microservices",
                current=True,
                skills=[SkillName("Python"), SkillName("FastAPI")],
            )
        ],
        education=[
            Education(
                institution="University of California",
                degree="BS",
                field="Computer Science",
                end_date="2015-05-15",
            )
        ],
        skills=[
            Skill(name=SkillName("Python"), category="technical", proficiency=5),
            Skill(name=SkillName("FastAPI"), category="technical", proficiency=4),
        ],
        languages=["English", "Spanish"],
    )

    # Create deleted profile with deletion metadata
    profile = Profile(
        id=profile_id,
        tenant_id=tenant_id,
        status=ProfileStatus.DELETED,
        profile_data=profile_data,
        processing=ProcessingMetadata(status=ProcessingStatus.COMPLETED, quality_score=0.85),
        privacy=PrivacySettings(
            consent_given=True,
            consent_date=datetime.utcnow(),
            deletion_requested=True,
        ),
        metadata={
            "deletion_reason": "User requested deletion",
            "deleted_at": datetime.utcnow().isoformat(),
            "original_filename": "resume.pdf",
        },
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    return profile


@pytest.fixture
def mock_background_tasks():
    """Mock for FastAPI BackgroundTasks."""
    tasks = Mock()
    tasks.add_task = Mock()
    return tasks


@pytest.fixture
def minimal_profile_data():
    """Minimal profile data for testing."""
    return ProfileData(
        name="Jane Smith",
        email=EmailAddress("jane.smith@example.com"),
        phone=None,
        location=Location(city="New York", state="NY", country="USA"),
        summary="Experienced data scientist with expertise in ML",
        headline="Senior Data Scientist",
        experience=[
            Experience(
                title="Data Scientist",
                company="Data Corp",
                start_date="2020-01-01",
                description="Machine learning model development",
                current=True,
                skills=[SkillName("Python"), SkillName("TensorFlow")],
            )
        ],
        education=[
            Education(
                institution="MIT",
                degree="MS",
                field="Data Science",
                end_date="2019-05-15",
            )
        ],
        skills=[
            Skill(name=SkillName("Python"), category="technical", proficiency=5),
            Skill(name=SkillName("TensorFlow"), category="technical", proficiency=4),
        ],
        languages=["English"],
    )


@pytest.fixture
def sample_profile_with_analytics(minimal_profile_data: ProfileData) -> Profile:
    """Create profile with analytics data for testing."""
    profile = Profile(
        id=ProfileId(uuid4()),
        tenant_id=TenantId(uuid4()),
        status=ProfileStatus.ACTIVE,
        profile_data=minimal_profile_data,
        processing=ProcessingMetadata(status=ProcessingStatus.COMPLETED, quality_score=0.9),
    )

    # Set analytics data
    profile.analytics.view_count = 42
    profile.analytics.search_appearances = 15
    profile.analytics.last_viewed_at = datetime(2025, 10, 1, 12, 0, 0)

    return profile


# =============================================================================
# PROFILE RESTORATION TESTS
# =============================================================================


@pytest.mark.asyncio
class TestProfileRestoration:
    """Tests for profile restoration functionality in ProfileApplicationService."""

    async def test_restore_profile_success(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
        sample_deleted_profile: Profile,
        mock_background_tasks,
    ):
        """Should successfully restore a deleted profile."""
        # Arrange
        profile_id = sample_deleted_profile.id
        tenant_id = sample_deleted_profile.tenant_id
        user_id = "test_user_123"

        # Create a copy for the restored state
        restored_profile = Profile(
            id=profile_id,
            tenant_id=tenant_id,
            status=ProfileStatus.ACTIVE,  # Will be updated by restore()
            profile_data=sample_deleted_profile.profile_data,
            processing=sample_deleted_profile.processing,
            privacy=PrivacySettings(
                consent_given=True,
                consent_date=sample_deleted_profile.privacy.consent_date,
                deletion_requested=False,  # Will be reset
            ),
            metadata={"original_filename": "resume.pdf"},  # Deletion metadata removed
        )

        mock_profile_repository.get_by_id.return_value = sample_deleted_profile
        mock_profile_repository.save.return_value = restored_profile

        # Act
        result = await profile_service.restore_profile(
            profile_id=profile_id,
            tenant_id=tenant_id,
            user_id=user_id,
            schedule_task=mock_background_tasks,
        )

        # Assert
        assert result is not None
        assert result.status == ProfileStatus.ACTIVE
        assert result.privacy.deletion_requested is False
        assert "deletion_reason" not in result.metadata
        assert "deleted_at" not in result.metadata

        # Verify repository interactions
        mock_profile_repository.get_by_id.assert_called_once_with(profile_id, tenant_id)
        mock_profile_repository.save.assert_called_once()

        # Verify background tasks were scheduled
        assert mock_background_tasks.add_task.call_count == 4  # embeddings, search, usage, audit

    async def test_restore_profile_not_found(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
        mock_background_tasks,
    ):
        """Should return None when profile doesn't exist."""
        # Arrange
        profile_id = ProfileId(uuid4())
        tenant_id = TenantId(uuid4())
        user_id = "test_user_123"

        mock_profile_repository.get_by_id.return_value = None

        # Act
        result = await profile_service.restore_profile(
            profile_id=profile_id,
            tenant_id=tenant_id,
            user_id=user_id,
            schedule_task=mock_background_tasks,
        )

        # Assert
        assert result is None
        mock_profile_repository.get_by_id.assert_called_once_with(profile_id, tenant_id)
        mock_profile_repository.save.assert_not_called()
        mock_background_tasks.add_task.assert_not_called()

    async def test_restore_profile_validation_fails(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
        sample_deleted_profile: Profile,
        mock_background_tasks,
    ):
        """Should raise ValueError when profile is not deleted."""
        # Arrange
        profile_id = sample_deleted_profile.id
        tenant_id = sample_deleted_profile.tenant_id
        user_id = "test_user_123"

        # Create an active profile (not deleted)
        active_profile = Profile(
            id=profile_id,
            tenant_id=tenant_id,
            status=ProfileStatus.ACTIVE,  # Not deleted!
            profile_data=sample_deleted_profile.profile_data,
        )

        mock_profile_repository.get_by_id.return_value = active_profile

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot restore profile"):
            await profile_service.restore_profile(
                profile_id=profile_id,
                tenant_id=tenant_id,
                user_id=user_id,
                schedule_task=mock_background_tasks,
            )

        # Verify repository was called but save was not
        mock_profile_repository.get_by_id.assert_called_once_with(profile_id, tenant_id)
        mock_profile_repository.save.assert_not_called()
        mock_background_tasks.add_task.assert_not_called()

    async def test_restore_profile_schedules_background_tasks(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
        sample_deleted_profile: Profile,
        mock_background_tasks,
        mock_embedding_service,
        mock_search_index_service,
        mock_usage_service,
        mock_audit_service,
    ):
        """Should schedule all required background tasks."""
        # Arrange
        profile_id = sample_deleted_profile.id
        tenant_id = sample_deleted_profile.tenant_id
        user_id = "test_user_123"

        restored_profile = Profile(
            id=profile_id,
            tenant_id=tenant_id,
            status=ProfileStatus.ACTIVE,
            profile_data=sample_deleted_profile.profile_data,
        )

        mock_profile_repository.get_by_id.return_value = sample_deleted_profile
        mock_profile_repository.save.return_value = restored_profile

        # Act
        result = await profile_service.restore_profile(
            profile_id=profile_id,
            tenant_id=tenant_id,
            user_id=user_id,
            schedule_task=mock_background_tasks,
        )

        # Assert
        assert result is not None

        # Verify all background tasks were scheduled
        assert mock_background_tasks.add_task.call_count == 4

        # Get all task calls
        task_calls = [call.args[0].__name__ for call in mock_background_tasks.add_task.call_args_list]

        # Verify each background task was scheduled
        assert "_regenerate_profile_embeddings" in task_calls
        assert "_update_search_index" in task_calls
        assert "_track_profile_restoration_usage" in task_calls
        assert "_log_profile_restore_audit" in task_calls

    async def test_restore_profile_without_optional_services(
        self,
        mock_profile_repository,
        sample_deleted_profile: Profile,
        mock_background_tasks,
    ):
        """Should restore successfully when optional services are None."""
        # Arrange
        # Create dependencies with None for optional services
        dependencies = ProfileDependencies(
            profile_repository=mock_profile_repository,
            embedding_service=None,  # No embedding service
            search_index_service=None,  # No search index service
            usage_service=None,
            audit_service=None,
        )

        profile_service = ProfileApplicationService(dependencies=dependencies)

        profile_id = sample_deleted_profile.id
        tenant_id = sample_deleted_profile.tenant_id
        user_id = "test_user_123"

        restored_profile = Profile(
            id=profile_id,
            tenant_id=tenant_id,
            status=ProfileStatus.ACTIVE,
            profile_data=sample_deleted_profile.profile_data,
            metadata={"original_filename": "resume.pdf"},
        )

        mock_profile_repository.get_by_id.return_value = sample_deleted_profile
        mock_profile_repository.save.return_value = restored_profile

        # Act - Should not raise errors
        result = await profile_service.restore_profile(
            profile_id=profile_id,
            tenant_id=tenant_id,
            user_id=user_id,
            schedule_task=mock_background_tasks,
        )

        # Assert
        assert result is not None
        assert result.status == ProfileStatus.ACTIVE

        # Verify repository was called
        mock_profile_repository.get_by_id.assert_called_once()
        mock_profile_repository.save.assert_called_once()

        # Should still schedule usage task (even though it won't do anything)
        # Only usage task is scheduled when all optional services are None
        # (audit task is inside an if check, so it's not scheduled when audit_service is None)
        assert mock_background_tasks.add_task.call_count == 1  # Only usage task scheduled

    async def test_restore_profile_metadata_cleanup(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
        sample_deleted_profile: Profile,
        mock_background_tasks,
    ):
        """Should properly clear deletion metadata after restoration."""
        # Arrange
        profile_id = sample_deleted_profile.id
        tenant_id = sample_deleted_profile.tenant_id
        user_id = "test_user_123"

        # Verify sample profile has deletion metadata
        assert "deletion_reason" in sample_deleted_profile.metadata
        assert "deleted_at" in sample_deleted_profile.metadata
        assert sample_deleted_profile.privacy.deletion_requested is True

        # Mock repository to capture the saved profile
        saved_profile = None

        async def capture_save(profile):
            nonlocal saved_profile
            saved_profile = profile
            return profile

        mock_profile_repository.get_by_id.return_value = sample_deleted_profile
        mock_profile_repository.save.side_effect = capture_save

        # Act
        result = await profile_service.restore_profile(
            profile_id=profile_id,
            tenant_id=tenant_id,
            user_id=user_id,
            schedule_task=mock_background_tasks,
        )

        # Assert
        assert result is not None

        # Verify deletion metadata is cleared in the saved profile
        assert "deletion_reason" not in saved_profile.metadata
        assert "deleted_at" not in saved_profile.metadata
        assert saved_profile.privacy.deletion_requested is False

        # Verify other metadata is preserved
        assert saved_profile.metadata.get("original_filename") == "resume.pdf"

    async def test_restore_profile_without_background_tasks(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
        sample_deleted_profile: Profile,
    ):
        """Should restore successfully without background task scheduler."""
        # Arrange
        profile_id = sample_deleted_profile.id
        tenant_id = sample_deleted_profile.tenant_id
        user_id = "test_user_123"

        restored_profile = Profile(
            id=profile_id,
            tenant_id=tenant_id,
            status=ProfileStatus.ACTIVE,
            profile_data=sample_deleted_profile.profile_data,
        )

        mock_profile_repository.get_by_id.return_value = sample_deleted_profile
        mock_profile_repository.save.return_value = restored_profile

        # Act - No schedule_task provided
        result = await profile_service.restore_profile(
            profile_id=profile_id,
            tenant_id=tenant_id,
            user_id=user_id,
            schedule_task=None,  # No background task scheduler
        )

        # Assert
        assert result is not None
        assert result.status == ProfileStatus.ACTIVE

        # Verify repository interactions
        mock_profile_repository.get_by_id.assert_called_once()
        mock_profile_repository.save.assert_called_once()

    async def test_restore_profile_preserves_profile_data(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
        sample_deleted_profile: Profile,
        mock_background_tasks,
    ):
        """Should preserve all profile data during restoration."""
        # Arrange
        profile_id = sample_deleted_profile.id
        tenant_id = sample_deleted_profile.tenant_id
        user_id = "test_user_123"

        # Capture the saved profile
        saved_profile = None

        async def capture_save(profile):
            nonlocal saved_profile
            saved_profile = profile
            return profile

        mock_profile_repository.get_by_id.return_value = sample_deleted_profile
        mock_profile_repository.save.side_effect = capture_save

        # Act
        result = await profile_service.restore_profile(
            profile_id=profile_id,
            tenant_id=tenant_id,
            user_id=user_id,
            schedule_task=mock_background_tasks,
        )

        # Assert
        assert result is not None

        # Verify profile data is preserved
        assert saved_profile.profile_data.name == sample_deleted_profile.profile_data.name
        assert saved_profile.profile_data.email == sample_deleted_profile.profile_data.email
        assert saved_profile.profile_data.summary == sample_deleted_profile.profile_data.summary
        assert len(saved_profile.profile_data.skills) == len(sample_deleted_profile.profile_data.skills)
        assert len(saved_profile.profile_data.experience) == len(
            sample_deleted_profile.profile_data.experience
        )
        assert len(saved_profile.profile_data.education) == len(
            sample_deleted_profile.profile_data.education
        )

        # Verify processing metadata is preserved
        assert saved_profile.processing.quality_score == sample_deleted_profile.processing.quality_score
        assert saved_profile.processing.status == sample_deleted_profile.processing.status

    async def test_restore_profile_updates_timestamp(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
        sample_deleted_profile: Profile,
        mock_background_tasks,
    ):
        """Should update the profile timestamp during restoration."""
        # Arrange
        profile_id = sample_deleted_profile.id
        tenant_id = sample_deleted_profile.tenant_id
        user_id = "test_user_123"

        original_updated_at = sample_deleted_profile.updated_at

        # Capture the saved profile
        saved_profile = None

        async def capture_save(profile):
            nonlocal saved_profile
            saved_profile = profile
            return profile

        mock_profile_repository.get_by_id.return_value = sample_deleted_profile
        mock_profile_repository.save.side_effect = capture_save

        # Act
        result = await profile_service.restore_profile(
            profile_id=profile_id,
            tenant_id=tenant_id,
            user_id=user_id,
            schedule_task=mock_background_tasks,
        )

        # Assert
        assert result is not None

        # Verify timestamp was updated (domain entity updates it automatically)
        # The restore() method updates updated_at
        assert saved_profile.updated_at >= original_updated_at


# =============================================================================
# PROFILE ANALYTICS TESTS
# =============================================================================


@pytest.mark.asyncio
class TestProfileAnalytics:
    """Tests for profile analytics functionality in ProfileApplicationService."""

    async def test_get_profile_analytics_success(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
        sample_profile_with_analytics: Profile,
    ):
        """Should successfully retrieve profile analytics."""
        # Arrange
        profile_id = sample_profile_with_analytics.id
        tenant_id = sample_profile_with_analytics.tenant_id

        mock_profile_repository.get_by_id.return_value = sample_profile_with_analytics

        # Act
        result = await profile_service.get_profile_analytics(
            profile_id=profile_id,
            tenant_id=tenant_id,
            time_range_days=30,
        )

        # Assert
        assert result is not None
        assert result["profile_id"] == str(profile_id.value)
        assert result["view_count"] == 42
        assert result["search_appearances"] == 15
        assert result["profile_completeness"] > 0.0
        assert "last_viewed" in result
        assert "match_score_distribution" in result
        assert "popular_searches" in result
        assert "skill_demand_score" in result
        mock_profile_repository.get_by_id.assert_called_once_with(profile_id, tenant_id)

    async def test_get_profile_analytics_profile_not_found(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
    ):
        """Should return None when profile doesn't exist."""
        # Arrange
        profile_id = ProfileId(uuid4())
        tenant_id = TenantId(uuid4())

        mock_profile_repository.get_by_id.return_value = None

        # Act
        result = await profile_service.get_profile_analytics(
            profile_id=profile_id,
            tenant_id=tenant_id,
            time_range_days=30,
        )

        # Assert
        assert result is None
        mock_profile_repository.get_by_id.assert_called_once_with(profile_id, tenant_id)

    async def test_get_profile_analytics_returns_correct_view_count(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
        sample_profile_with_analytics: Profile,
    ):
        """Should return correct view_count from profile analytics."""
        # Arrange
        profile_id = sample_profile_with_analytics.id
        tenant_id = sample_profile_with_analytics.tenant_id

        # Set specific view count
        sample_profile_with_analytics.analytics.view_count = 123

        mock_profile_repository.get_by_id.return_value = sample_profile_with_analytics

        # Act
        result = await profile_service.get_profile_analytics(
            profile_id=profile_id,
            tenant_id=tenant_id,
        )

        # Assert
        assert result is not None
        assert result["view_count"] == 123

    async def test_get_profile_analytics_returns_correct_search_appearances(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
        sample_profile_with_analytics: Profile,
    ):
        """Should return correct search_appearances from profile analytics."""
        # Arrange
        profile_id = sample_profile_with_analytics.id
        tenant_id = sample_profile_with_analytics.tenant_id

        # Set specific search appearances
        sample_profile_with_analytics.analytics.search_appearances = 456

        mock_profile_repository.get_by_id.return_value = sample_profile_with_analytics

        # Act
        result = await profile_service.get_profile_analytics(
            profile_id=profile_id,
            tenant_id=tenant_id,
        )

        # Assert
        assert result is not None
        assert result["search_appearances"] == 456

    async def test_get_profile_analytics_calculates_completeness_score(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
        minimal_profile_data: ProfileData,
    ):
        """Should calculate completeness score using domain method."""
        # Arrange
        # Create profile with high completeness (all fields populated)
        complete_profile = Profile(
            id=ProfileId(uuid4()),
            tenant_id=TenantId(uuid4()),
            status=ProfileStatus.ACTIVE,
            profile_data=minimal_profile_data,
            processing=ProcessingMetadata(status=ProcessingStatus.COMPLETED),
        )

        # Calculate expected score using domain method
        expected_score = complete_profile.calculate_completeness_score()

        mock_profile_repository.get_by_id.return_value = complete_profile

        # Act
        result = await profile_service.get_profile_analytics(
            profile_id=complete_profile.id,
            tenant_id=complete_profile.tenant_id,
        )

        # Assert
        assert result is not None
        assert result["profile_completeness"] == expected_score
        assert result["profile_completeness"] > 0.0

    async def test_get_profile_analytics_returns_placeholder_for_advanced_metrics(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
        sample_profile_with_analytics: Profile,
    ):
        """Should return empty/None placeholders for advanced metrics."""
        # Arrange
        profile_id = sample_profile_with_analytics.id
        tenant_id = sample_profile_with_analytics.tenant_id

        mock_profile_repository.get_by_id.return_value = sample_profile_with_analytics

        # Act
        result = await profile_service.get_profile_analytics(
            profile_id=profile_id,
            tenant_id=tenant_id,
        )

        # Assert - Advanced metrics should be empty/None (MVP implementation)
        assert result is not None
        assert result["match_score_distribution"] == {}
        assert result["popular_searches"] == []
        assert result["skill_demand_score"] is None

    async def test_get_profile_analytics_includes_last_viewed_timestamp(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
        sample_profile_with_analytics: Profile,
    ):
        """Should include last_viewed timestamp from analytics."""
        # Arrange
        profile_id = sample_profile_with_analytics.id
        tenant_id = sample_profile_with_analytics.tenant_id

        # Set specific last viewed timestamp
        last_viewed = datetime(2025, 10, 5, 14, 30, 0)
        sample_profile_with_analytics.analytics.last_viewed_at = last_viewed

        mock_profile_repository.get_by_id.return_value = sample_profile_with_analytics

        # Act
        result = await profile_service.get_profile_analytics(
            profile_id=profile_id,
            tenant_id=tenant_id,
        )

        # Assert
        assert result is not None
        assert result["last_viewed"] == last_viewed

    async def test_get_profile_analytics_repository_error(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
    ):
        """Should propagate exception when repository raises error."""
        # Arrange
        profile_id = ProfileId(uuid4())
        tenant_id = TenantId(uuid4())

        mock_profile_repository.get_by_id.side_effect = Exception("Database connection failed")

        # Act & Assert
        with pytest.raises(Exception, match="Database connection failed"):
            await profile_service.get_profile_analytics(
                profile_id=profile_id,
                tenant_id=tenant_id,
            )

        mock_profile_repository.get_by_id.assert_called_once_with(profile_id, tenant_id)

    async def test_get_profile_analytics_with_zero_views(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
        sample_profile_with_analytics: Profile,
    ):
        """Should handle profile with zero views correctly."""
        # Arrange
        profile_id = sample_profile_with_analytics.id
        tenant_id = sample_profile_with_analytics.tenant_id

        # Set zero views
        sample_profile_with_analytics.analytics.view_count = 0
        sample_profile_with_analytics.analytics.search_appearances = 0
        sample_profile_with_analytics.analytics.last_viewed_at = None

        mock_profile_repository.get_by_id.return_value = sample_profile_with_analytics

        # Act
        result = await profile_service.get_profile_analytics(
            profile_id=profile_id,
            tenant_id=tenant_id,
        )

        # Assert
        assert result is not None
        assert result["view_count"] == 0
        assert result["search_appearances"] == 0
        assert result["last_viewed"] is None

    async def test_get_profile_analytics_respects_time_range_parameter(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
        sample_profile_with_analytics: Profile,
    ):
        """Should accept time_range_days parameter (for future implementation)."""
        # Arrange
        profile_id = sample_profile_with_analytics.id
        tenant_id = sample_profile_with_analytics.tenant_id

        mock_profile_repository.get_by_id.return_value = sample_profile_with_analytics

        # Act - Call with different time ranges
        result_30 = await profile_service.get_profile_analytics(
            profile_id=profile_id,
            tenant_id=tenant_id,
            time_range_days=30,
        )

        result_90 = await profile_service.get_profile_analytics(
            profile_id=profile_id,
            tenant_id=tenant_id,
            time_range_days=90,
        )

        # Assert - Both calls should succeed (MVP doesn't filter by time, but parameter is accepted)
        assert result_30 is not None
        assert result_90 is not None
        assert mock_profile_repository.get_by_id.call_count == 2

    async def test_get_profile_analytics_includes_all_required_fields(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
        sample_profile_with_analytics: Profile,
    ):
        """Should include all required fields in analytics response."""
        # Arrange
        profile_id = sample_profile_with_analytics.id
        tenant_id = sample_profile_with_analytics.tenant_id

        mock_profile_repository.get_by_id.return_value = sample_profile_with_analytics

        # Act
        result = await profile_service.get_profile_analytics(
            profile_id=profile_id,
            tenant_id=tenant_id,
        )

        # Assert - Verify all required fields are present
        assert result is not None
        required_fields = [
            "profile_id",
            "view_count",
            "search_appearances",
            "last_viewed",
            "profile_completeness",
            "match_score_distribution",
            "popular_searches",
            "skill_demand_score",
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"


# =============================================================================
# PROFILE VIEW TRACKING TESTS
# =============================================================================


@pytest.mark.asyncio
class TestProfileViewTracking:
    """Tests for profile view tracking functionality in ProfileApplicationService."""

    async def test_get_profile_tracks_view_in_background(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
        sample_profile_with_analytics: Profile,
        mock_background_tasks,
    ):
        """Should schedule view tracking when background tasks provided."""
        # Arrange
        profile_id = sample_profile_with_analytics.id
        tenant_id = sample_profile_with_analytics.tenant_id

        mock_profile_repository.get_by_id.return_value = sample_profile_with_analytics

        # Act
        result = await profile_service.get_profile(
            profile_id=profile_id,
            tenant_id=tenant_id,
            schedule_task=mock_background_tasks,
        )

        # Assert
        assert result is not None
        assert result.id == profile_id
        mock_background_tasks.add_task.assert_called_once()

        # Verify the task is _record_profile_view
        task_call = mock_background_tasks.add_task.call_args
        assert task_call[0][0].__name__ == "_record_profile_view"

        # Verify task arguments are profile_id and tenant_id as strings
        assert task_call[0][1] == str(profile_id.value)
        assert task_call[0][2] == str(tenant_id.value)

    async def test_get_profile_without_background_tasks(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
        sample_profile_with_analytics: Profile,
    ):
        """Should return profile without errors when schedule_task is None."""
        # Arrange
        profile_id = sample_profile_with_analytics.id
        tenant_id = sample_profile_with_analytics.tenant_id

        mock_profile_repository.get_by_id.return_value = sample_profile_with_analytics

        # Act - Call without schedule_task
        result = await profile_service.get_profile(
            profile_id=profile_id,
            tenant_id=tenant_id,
            schedule_task=None,
        )

        # Assert
        assert result is not None
        assert result.id == profile_id
        mock_profile_repository.get_by_id.assert_called_once_with(profile_id, tenant_id)

    async def test_record_profile_view_increments_count(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
        sample_profile_with_analytics: Profile,
    ):
        """Should increment view count when recording profile view."""
        # Arrange
        profile_id = sample_profile_with_analytics.id
        tenant_id = sample_profile_with_analytics.tenant_id

        # Set initial view count
        initial_view_count = 10
        sample_profile_with_analytics.analytics.view_count = initial_view_count

        mock_profile_repository.get_by_id.return_value = sample_profile_with_analytics
        mock_profile_repository.save.return_value = sample_profile_with_analytics

        # Act - Call the background task directly
        await profile_service._record_profile_view(
            profile_id=str(profile_id.value),
            tenant_id=str(tenant_id.value),
        )

        # Assert
        # Verify repository methods were called
        mock_profile_repository.get_by_id.assert_called_once_with(profile_id, tenant_id)
        mock_profile_repository.save.assert_called_once()

        # Verify view count was incremented (record_view() increments by 1)
        saved_profile = mock_profile_repository.save.call_args[0][0]
        assert saved_profile.analytics.view_count == initial_view_count + 1

    async def test_record_profile_view_updates_timestamp(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
        sample_profile_with_analytics: Profile,
    ):
        """Should update last_viewed_at timestamp when recording view."""
        # Arrange
        profile_id = sample_profile_with_analytics.id
        tenant_id = sample_profile_with_analytics.tenant_id

        # Start with no last_viewed_at
        sample_profile_with_analytics.analytics.last_viewed_at = None

        mock_profile_repository.get_by_id.return_value = sample_profile_with_analytics
        mock_profile_repository.save.return_value = sample_profile_with_analytics

        # Act
        await profile_service._record_profile_view(
            profile_id=str(profile_id.value),
            tenant_id=str(tenant_id.value),
        )

        # Assert
        # Verify timestamp was set
        saved_profile = mock_profile_repository.save.call_args[0][0]
        assert saved_profile.analytics.last_viewed_at is not None
        assert isinstance(saved_profile.analytics.last_viewed_at, datetime)

    async def test_record_profile_view_profile_not_found(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
    ):
        """Should handle gracefully when profile doesn't exist during view recording."""
        # Arrange
        profile_id = ProfileId(uuid4())
        tenant_id = TenantId(uuid4())

        mock_profile_repository.get_by_id.return_value = None

        # Act - Should not raise error
        await profile_service._record_profile_view(
            profile_id=str(profile_id.value),
            tenant_id=str(tenant_id.value),
        )

        # Assert
        mock_profile_repository.get_by_id.assert_called_once_with(profile_id, tenant_id)
        mock_profile_repository.save.assert_not_called()

    async def test_record_profile_view_handles_repository_error(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
        sample_profile_with_analytics: Profile,
    ):
        """Should log error but not raise when repository save fails."""
        # Arrange
        profile_id = sample_profile_with_analytics.id
        tenant_id = sample_profile_with_analytics.tenant_id

        mock_profile_repository.get_by_id.return_value = sample_profile_with_analytics
        mock_profile_repository.save.side_effect = Exception("Database connection failed")

        # Act - Should not raise error (errors are caught and logged)
        await profile_service._record_profile_view(
            profile_id=str(profile_id.value),
            tenant_id=str(tenant_id.value),
        )

        # Assert - Both methods were called despite error
        mock_profile_repository.get_by_id.assert_called_once()
        mock_profile_repository.save.assert_called_once()

    async def test_get_profile_returns_none_when_not_found(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
        mock_background_tasks,
    ):
        """Should return None and not schedule view tracking when profile doesn't exist."""
        # Arrange
        profile_id = ProfileId(uuid4())
        tenant_id = TenantId(uuid4())

        mock_profile_repository.get_by_id.return_value = None

        # Act
        result = await profile_service.get_profile(
            profile_id=profile_id,
            tenant_id=tenant_id,
            schedule_task=mock_background_tasks,
        )

        # Assert
        assert result is None
        mock_profile_repository.get_by_id.assert_called_once_with(profile_id, tenant_id)
        # Should not schedule background task when profile is None
        mock_background_tasks.add_task.assert_not_called()

    async def test_record_profile_view_preserves_other_analytics(
        self,
        profile_service: ProfileApplicationService,
        mock_profile_repository,
        sample_profile_with_analytics: Profile,
    ):
        """Should preserve other analytics fields when recording view."""
        # Arrange
        profile_id = sample_profile_with_analytics.id
        tenant_id = sample_profile_with_analytics.tenant_id

        # Set existing analytics
        sample_profile_with_analytics.analytics.view_count = 5
        sample_profile_with_analytics.analytics.search_appearances = 20

        mock_profile_repository.get_by_id.return_value = sample_profile_with_analytics
        mock_profile_repository.save.return_value = sample_profile_with_analytics

        # Act
        await profile_service._record_profile_view(
            profile_id=str(profile_id.value),
            tenant_id=str(tenant_id.value),
        )

        # Assert
        saved_profile = mock_profile_repository.save.call_args[0][0]
        # View count incremented
        assert saved_profile.analytics.view_count == 6
        # Search appearances preserved
        assert saved_profile.analytics.search_appearances == 20


__all__ = ["TestProfileRestoration", "TestProfileAnalytics", "TestProfileViewTracking"]
