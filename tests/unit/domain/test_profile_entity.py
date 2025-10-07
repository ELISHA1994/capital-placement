"""Comprehensive test suite for Profile domain entity business logic.

Tests focus exclusively on domain behavior - no persistence, no infrastructure.
All tests are pure and fast, validating business rules and domain logic.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from hypothesis import given, strategies as st

from app.domain.entities.profile import (
    Profile,
    ProfileData,
    ProfileStatus,
    ExperienceLevel,
    ProcessingMetadata,
    ProcessingStatus,
    PrivacySettings,
    ProfileAnalytics,
    Location,
    Experience,
    Education,
    Skill,
    ProfileEmbeddings,
)
from app.domain.value_objects import (
    ProfileId,
    TenantId,
    EmailAddress,
    PhoneNumber,
    SkillName,
    EmbeddingVector,
    MatchScore,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def valid_email() -> EmailAddress:
    """Create valid email address."""
    return EmailAddress("john.doe@example.com")


@pytest.fixture
def valid_phone() -> PhoneNumber:
    """Create valid phone number."""
    return PhoneNumber("+1-555-123-4567")


@pytest.fixture
def minimal_profile_data(valid_email: EmailAddress) -> ProfileData:
    """Create minimal valid profile data."""
    return ProfileData(
        name="John Doe",
        email=valid_email,
    )


@pytest.fixture
def complete_profile_data(valid_email: EmailAddress, valid_phone: PhoneNumber) -> ProfileData:
    """Create complete profile data with all fields."""
    return ProfileData(
        name="Jane Smith",
        email=valid_email,
        phone=valid_phone,
        location=Location(city="San Francisco", state="CA", country="USA"),
        summary="Experienced software engineer with 8 years in backend development.",
        headline="Senior Software Engineer",
        experience=[
            Experience(
                title="Senior Software Engineer",
                company="TechCorp",
                start_date="2020-01-01",
                end_date=None,
                current=True,
                description="Leading backend development team",
                achievements=["Reduced latency by 40%", "Mentored 5 junior engineers"],
                skills=[SkillName("Python"), SkillName("FastAPI")],
            ),
            Experience(
                title="Software Engineer",
                company="StartupXYZ",
                start_date="2016-06-01",
                end_date="2019-12-31",
                current=False,
                description="Backend development",
                achievements=["Built microservices architecture"],
                skills=[SkillName("Python"), SkillName("Django")],
            ),
        ],
        education=[
            Education(
                institution="Stanford University",
                degree="Bachelor of Science",
                field="Computer Science",
                start_date="2012-09-01",
                end_date="2016-05-31",
                gpa=3.8,
                achievements=["Dean's List", "CS Department Award"],
            ),
        ],
        skills=[
            Skill(name=SkillName("Python"), category="technical", proficiency=5, years_of_experience=8),
            Skill(name=SkillName("FastAPI"), category="technical", proficiency=4, years_of_experience=3),
            Skill(name=SkillName("PostgreSQL"), category="technical", proficiency=4, years_of_experience=6),
        ],
        languages=["English", "Spanish"],
    )


@pytest.fixture
def profile_with_minimal_data(minimal_profile_data: ProfileData) -> Profile:
    """Create profile with minimal data."""
    return Profile(
        id=ProfileId(uuid4()),
        tenant_id=TenantId(uuid4()),
        status=ProfileStatus.ACTIVE,
        profile_data=minimal_profile_data,
    )


@pytest.fixture
def profile_with_complete_data(complete_profile_data: ProfileData) -> Profile:
    """Create profile with complete data."""
    return Profile(
        id=ProfileId(uuid4()),
        tenant_id=TenantId(uuid4()),
        status=ProfileStatus.ACTIVE,
        profile_data=complete_profile_data,
    )


# ============================================================================
# 1. Entity Creation & Initialization Tests (5 tests)
# ============================================================================

class TestProfileCreation:
    """Test profile entity creation and initialization."""

    def test_create_profile_with_required_fields(self, minimal_profile_data: ProfileData):
        """Should create profile with only required fields."""
        # Arrange
        profile_id = ProfileId(uuid4())
        tenant_id = TenantId(uuid4())

        # Act
        profile = Profile(
            id=profile_id,
            tenant_id=tenant_id,
            status=ProfileStatus.ACTIVE,
            profile_data=minimal_profile_data,
        )

        # Assert
        assert profile.id == profile_id
        assert profile.tenant_id == tenant_id
        assert profile.status == ProfileStatus.ACTIVE
        assert profile.profile_data == minimal_profile_data
        assert profile.created_at is not None
        assert profile.updated_at is not None
        assert isinstance(profile.created_at, datetime)

    def test_create_profile_with_all_optional_fields(self, complete_profile_data: ProfileData):
        """Should create profile with all optional fields populated."""
        # Arrange
        profile_id = ProfileId(uuid4())
        tenant_id = TenantId(uuid4())
        embeddings = ProfileEmbeddings(
            overall=EmbeddingVector(dimensions=3, values=[0.1, 0.2, 0.3])
        )
        metadata = {"source": "linkedin", "imported_at": "2025-01-01"}

        # Act
        profile = Profile(
            id=profile_id,
            tenant_id=tenant_id,
            status=ProfileStatus.ACTIVE,
            profile_data=complete_profile_data,
            embeddings=embeddings,
            metadata=metadata,
            keywords=["python", "backend", "senior"],
        )

        # Assert
        assert profile.embeddings == embeddings
        assert profile.metadata == metadata
        assert profile.keywords == ["python", "backend", "senior"]
        assert profile.profile_data.name == "Jane Smith"

    def test_profile_default_values_on_creation(self, minimal_profile_data: ProfileData):
        """Should initialize with correct default values."""
        # Act
        profile = Profile(
            id=ProfileId(uuid4()),
            tenant_id=TenantId(uuid4()),
            status=ProfileStatus.ACTIVE,
            profile_data=minimal_profile_data,
        )

        # Assert - searchable_text is computed in __post_init__, so it won't be empty
        assert profile.searchable_text != ""  # Computed from profile data
        assert profile.normalized_skills == []
        assert profile.keywords == []
        assert profile.experience_level == ExperienceLevel.ENTRY
        assert isinstance(profile.embeddings, ProfileEmbeddings)
        assert isinstance(profile.processing, ProcessingMetadata)
        assert isinstance(profile.privacy, PrivacySettings)
        assert isinstance(profile.analytics, ProfileAnalytics)
        assert profile.metadata == {}
        assert profile.last_activity_at is None

    def test_post_init_computes_derived_fields(self, complete_profile_data: ProfileData):
        """Should compute normalized_skills, experience_level, and searchable_text in __post_init__."""
        # Act
        profile = Profile(
            id=ProfileId(uuid4()),
            tenant_id=TenantId(uuid4()),
            status=ProfileStatus.ACTIVE,
            profile_data=complete_profile_data,
        )

        # Assert - computed fields should be populated
        assert len(profile.normalized_skills) == 3
        assert "python" in profile.normalized_skills
        assert "fastapi" in profile.normalized_skills
        assert "postgresql" in profile.normalized_skills

        # Experience level should be calculated from ~8.5 years (2020-now + 2016-2019)
        # This equals LEAD level (>= 7 years)
        assert profile.experience_level == ExperienceLevel.LEAD

        # Searchable text should contain profile content
        assert "Jane Smith" in profile.searchable_text
        assert "Senior Software Engineer" in profile.searchable_text
        assert "TechCorp" in profile.searchable_text

    def test_post_init_handles_empty_profile_data(self, minimal_profile_data: ProfileData):
        """Should handle profile data with no experience gracefully."""
        # Act
        profile = Profile(
            id=ProfileId(uuid4()),
            tenant_id=TenantId(uuid4()),
            status=ProfileStatus.DRAFT,
            profile_data=minimal_profile_data,
        )

        # Assert
        assert profile.normalized_skills == []
        assert profile.experience_level == ExperienceLevel.ENTRY
        assert profile.searchable_text != ""  # Should at least contain name and email


# ============================================================================
# 2. Status Transition Methods Tests (8 tests)
# ============================================================================

class TestProfileStatusTransitions:
    """Test profile status transition methods."""

    def test_activate_sets_status_to_active(self, profile_with_minimal_data: Profile):
        """Should change status to ACTIVE and update timestamp."""
        # Arrange
        profile_with_minimal_data.status = ProfileStatus.INACTIVE
        original_updated_at = profile_with_minimal_data.updated_at

        # Act
        profile_with_minimal_data.activate()

        # Assert
        assert profile_with_minimal_data.status == ProfileStatus.ACTIVE
        assert profile_with_minimal_data.updated_at > original_updated_at

    def test_activate_raises_error_for_deleted_profile(self, profile_with_minimal_data: Profile):
        """Should raise ValueError when trying to activate DELETED profile."""
        # Arrange
        profile_with_minimal_data.status = ProfileStatus.DELETED

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot activate deleted profile"):
            profile_with_minimal_data.activate()

    def test_archive_sets_status_to_archived(self, profile_with_minimal_data: Profile):
        """Should change status to ARCHIVED and update timestamp."""
        # Arrange
        profile_with_minimal_data.status = ProfileStatus.ACTIVE
        original_updated_at = profile_with_minimal_data.updated_at

        # Act
        profile_with_minimal_data.archive()

        # Assert
        assert profile_with_minimal_data.status == ProfileStatus.ARCHIVED
        assert profile_with_minimal_data.updated_at > original_updated_at

    def test_mark_deleted_sets_status_to_deleted(self, profile_with_minimal_data: Profile):
        """Should soft-delete profile by changing status to DELETED."""
        # Arrange
        profile_with_minimal_data.status = ProfileStatus.ACTIVE
        original_updated_at = profile_with_minimal_data.updated_at

        # Act
        profile_with_minimal_data.mark_deleted()

        # Assert
        assert profile_with_minimal_data.status == ProfileStatus.DELETED
        assert profile_with_minimal_data.updated_at > original_updated_at

    def test_status_transition_updates_timestamp(self, profile_with_minimal_data: Profile):
        """Should always update updated_at on status transitions."""
        # Arrange
        original_updated_at = profile_with_minimal_data.updated_at

        # Act - wait a moment to ensure timestamp difference
        import time
        time.sleep(0.01)
        profile_with_minimal_data.archive()

        # Assert
        assert profile_with_minimal_data.updated_at > original_updated_at

    def test_activate_idempotency_already_active(self, profile_with_minimal_data: Profile):
        """Should handle activating already ACTIVE profile gracefully."""
        # Arrange
        profile_with_minimal_data.status = ProfileStatus.ACTIVE

        # Act - activate already active profile
        profile_with_minimal_data.activate()

        # Assert - should succeed without error
        assert profile_with_minimal_data.status == ProfileStatus.ACTIVE

    def test_all_profile_status_enum_values_exist(self):
        """Should verify all ProfileStatus enum values are defined."""
        # Assert
        assert ProfileStatus.ACTIVE == "active"
        assert ProfileStatus.INACTIVE == "inactive"
        assert ProfileStatus.DRAFT == "draft"
        assert ProfileStatus.ARCHIVED == "archived"
        assert ProfileStatus.DELETED == "deleted"

    def test_status_transitions_preserve_other_fields(self, profile_with_complete_data: Profile):
        """Should not modify other fields during status transitions."""
        # Arrange
        original_id = profile_with_complete_data.id
        original_tenant_id = profile_with_complete_data.tenant_id
        original_name = profile_with_complete_data.profile_data.name

        # Act
        profile_with_complete_data.archive()

        # Assert - only status and updated_at should change
        assert profile_with_complete_data.id == original_id
        assert profile_with_complete_data.tenant_id == original_tenant_id
        assert profile_with_complete_data.profile_data.name == original_name


# ============================================================================
# 3. Business Calculation Methods Tests (10 tests)
# ============================================================================

class TestProfileBusinessCalculations:
    """Test business calculation methods."""

    def test_calculate_completeness_score_minimal_data(self, profile_with_minimal_data: Profile):
        """Should return low completeness score for minimal data."""
        # Act
        score = profile_with_minimal_data.calculate_completeness_score()

        # Assert - only name (10) + email (10) = 20
        assert score == 20.0

    def test_calculate_completeness_score_complete_data(self, profile_with_complete_data: Profile):
        """Should return high completeness score for complete data."""
        # Act
        score = profile_with_complete_data.calculate_completeness_score()

        # Assert - should be close to 100
        # name(10) + email(10) + summary(15) + experience(15) + skills(10) + education(10)
        # + location(10) + phone(5) + headline(5) + languages(5) = 95
        assert score >= 90.0
        assert score <= 100.0

    def test_calculate_completeness_score_partial_data(self, minimal_profile_data: ProfileData):
        """Should return moderate score for partial data."""
        # Arrange - add some but not all fields
        minimal_profile_data.summary = "A brief summary"
        minimal_profile_data.skills = [Skill(name=SkillName("Python"))]

        profile = Profile(
            id=ProfileId(uuid4()),
            tenant_id=TenantId(uuid4()),
            status=ProfileStatus.ACTIVE,
            profile_data=minimal_profile_data,
        )

        # Act
        score = profile.calculate_completeness_score()

        # Assert - name(10) + email(10) + summary(15) + skills(10) = 45
        assert 40.0 <= score <= 50.0

    def test_calculate_completeness_score_max_capped_at_100(self, profile_with_complete_data: Profile):
        """Should never exceed 100% completeness."""
        # Act
        score = profile_with_complete_data.calculate_completeness_score()

        # Assert
        assert score <= 100.0

    def test_get_quality_issues_for_incomplete_profile(self, minimal_profile_data: ProfileData):
        """Should return quality issues for incomplete profile."""
        # Arrange - create profile with missing fields
        minimal_profile_data.summary = None
        profile = Profile(
            id=ProfileId(uuid4()),
            tenant_id=TenantId(uuid4()),
            status=ProfileStatus.ACTIVE,
            profile_data=minimal_profile_data,
        )

        # Act
        issues = profile.get_quality_issues()

        # Assert
        assert len(issues) > 0
        assert any("required fields" in issue.lower() for issue in issues)
        assert "No skills listed" in issues
        assert "No professional experience" in issues
        assert "Missing vector embeddings" in issues
        assert "Processing not completed" in issues

    def test_get_quality_issues_for_complete_profile(self, profile_with_complete_data: Profile):
        """Should return fewer issues for complete profile."""
        # Arrange - mark processing as completed and add embeddings
        profile_with_complete_data.processing.mark_processing_completed()
        profile_with_complete_data.embeddings = ProfileEmbeddings(
            overall=EmbeddingVector(dimensions=3, values=[0.1, 0.2, 0.3]),
            skills=EmbeddingVector(dimensions=3, values=[0.4, 0.5, 0.6]),
        )

        # Act
        issues = profile_with_complete_data.get_quality_issues()

        # Assert - should have no or minimal issues
        assert len(issues) == 0

    def test_get_quality_issues_checks_required_fields(self, valid_email: EmailAddress):
        """Should detect missing required fields."""
        # Arrange - profile with name but no summary or experience
        profile_data = ProfileData(name="John Doe", email=valid_email)
        profile = Profile(
            id=ProfileId(uuid4()),
            tenant_id=TenantId(uuid4()),
            status=ProfileStatus.ACTIVE,
            profile_data=profile_data,
        )

        # Act
        issues = profile.get_quality_issues()

        # Assert
        assert any("required fields" in issue.lower() for issue in issues)

    def test_calculate_experience_level_from_years(self):
        """Should correctly calculate experience level from total years."""
        # This is tested via the experience level tests below
        pass

    def test_skill_count_calculation(self, profile_with_complete_data: Profile):
        """Should correctly count skills."""
        # Act
        skill_count = len(profile_with_complete_data.profile_data.skills)

        # Assert
        assert skill_count == 3

    def test_companies_extraction(self, profile_with_complete_data: Profile):
        """Should extract company names from experience."""
        # Act
        companies = profile_with_complete_data.profile_data.get_companies()

        # Assert
        assert "TechCorp" in companies
        assert "StartupXYZ" in companies
        assert len(companies) == 2


# ============================================================================
# 4. Data Validation Tests (8 tests)
# ============================================================================

class TestProfileDataValidation:
    """Test profile data validation."""

    def test_profile_data_has_required_fields_returns_true(self, complete_profile_data: ProfileData):
        """Should return True when all required fields present."""
        # Act
        result = complete_profile_data.has_required_fields()

        # Assert
        assert result is True

    def test_profile_data_has_required_fields_returns_false(self, valid_email: EmailAddress):
        """Should return False when required fields missing."""
        # Arrange - profile with no summary or experience
        profile_data = ProfileData(name="John Doe", email=valid_email)

        # Act
        result = profile_data.has_required_fields()

        # Assert
        assert result is False

    def test_email_validation_rejects_invalid_format(self):
        """Should reject invalid email formats."""
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid email address format"):
            EmailAddress("not-an-email")

    def test_email_validation_accepts_valid_format(self):
        """Should accept valid email formats."""
        # Act
        email = EmailAddress("valid@example.com")

        # Assert
        assert str(email) == "valid@example.com"

    def test_phone_validation_rejects_too_short(self):
        """Should reject phone numbers that are too short."""
        # Act & Assert
        with pytest.raises(ValueError, match="must be at least 10 characters"):
            PhoneNumber("123")

    def test_phone_validation_accepts_valid_number(self):
        """Should accept valid phone numbers."""
        # Act
        phone = PhoneNumber("+1-555-123-4567")

        # Assert
        assert str(phone) == "+1-555-123-4567"

    def test_skill_name_validation_rejects_empty(self):
        """Should reject empty skill names."""
        # Act & Assert
        with pytest.raises(ValueError, match="cannot be empty"):
            SkillName("")

    def test_skill_name_normalization(self):
        """Should normalize skill names to lowercase."""
        # Act
        skill = SkillName("Python")

        # Assert
        assert skill.value == "Python"
        assert skill.normalized == "python"


# ============================================================================
# 5. Experience Level Calculation Tests (7 tests)
# ============================================================================

class TestExperienceLevelCalculation:
    """Test experience level auto-calculation from years."""

    def test_entry_level_0_years(self, minimal_profile_data: ProfileData):
        """Should classify as ENTRY for 0 years experience."""
        # Arrange - no experience entries
        profile = Profile(
            id=ProfileId(uuid4()),
            tenant_id=TenantId(uuid4()),
            status=ProfileStatus.ACTIVE,
            profile_data=minimal_profile_data,
        )

        # Act
        level = profile.experience_level

        # Assert
        assert level == ExperienceLevel.ENTRY

    def test_junior_level_1_5_years(self, valid_email: EmailAddress):
        """Should classify as JUNIOR for 1-2 years experience (>= 1, < 2)."""
        # Arrange - 1.5 years of experience
        profile_data = ProfileData(
            name="Junior Dev",
            email=valid_email,
            experience=[
                Experience(
                    title="Developer",
                    company="CompanyA",
                    start_date="2023-07-01",
                    end_date="2025-01-01",
                    description="Development work",
                )
            ],
        )
        profile = Profile(
            id=ProfileId(uuid4()),
            tenant_id=TenantId(uuid4()),
            status=ProfileStatus.ACTIVE,
            profile_data=profile_data,
        )

        # Act
        level = profile.experience_level

        # Assert
        assert level == ExperienceLevel.JUNIOR

    def test_mid_level_3_years(self, valid_email: EmailAddress):
        """Should classify as MID for 2-4 years experience (>= 2, < 4)."""
        # Arrange - 3 years of experience
        profile_data = ProfileData(
            name="Mid-Level Dev",
            email=valid_email,
            experience=[
                Experience(
                    title="Software Engineer",
                    company="CompanyB",
                    start_date="2022-01-01",
                    end_date="2025-01-01",
                    description="Development work",
                )
            ],
        )
        profile = Profile(
            id=ProfileId(uuid4()),
            tenant_id=TenantId(uuid4()),
            status=ProfileStatus.ACTIVE,
            profile_data=profile_data,
        )

        # Act
        level = profile.experience_level

        # Assert
        assert level == ExperienceLevel.MID

    def test_senior_level_5_years(self, valid_email: EmailAddress):
        """Should classify as SENIOR for 4-7 years experience (>= 4, < 7)."""
        # Arrange - 5 years of experience
        profile_data = ProfileData(
            name="Senior Dev",
            email=valid_email,
            experience=[
                Experience(
                    title="Senior Engineer",
                    company="CompanyC",
                    start_date="2020-01-01",
                    end_date="2025-01-01",
                    description="Senior development work",
                )
            ],
        )
        profile = Profile(
            id=ProfileId(uuid4()),
            tenant_id=TenantId(uuid4()),
            status=ProfileStatus.ACTIVE,
            profile_data=profile_data,
        )

        # Assert
        assert profile.experience_level == ExperienceLevel.SENIOR

    def test_lead_level_8_years(self, valid_email: EmailAddress):
        """Should classify as LEAD for 7-10 years experience (>= 7, < 10)."""
        # Arrange - 8 years of experience
        profile_data = ProfileData(
            name="Lead Engineer",
            email=valid_email,
            experience=[
                Experience(
                    title="Lead Engineer",
                    company="CompanyD",
                    start_date="2017-01-01",
                    end_date="2025-01-01",
                    description="Leading teams",
                )
            ],
        )
        profile = Profile(
            id=ProfileId(uuid4()),
            tenant_id=TenantId(uuid4()),
            status=ProfileStatus.ACTIVE,
            profile_data=profile_data,
        )

        # Act
        level = profile.experience_level

        # Assert
        assert level == ExperienceLevel.LEAD

    def test_principal_level_11_years(self, valid_email: EmailAddress):
        """Should classify as PRINCIPAL for 11 years experience."""
        # Arrange - 11 years of experience
        profile_data = ProfileData(
            name="Principal Engineer",
            email=valid_email,
            experience=[
                Experience(
                    title="Principal Engineer",
                    company="CompanyD",
                    start_date="2014-01-01",
                    end_date="2025-01-01",
                    description="Principal engineering work",
                )
            ],
        )
        profile = Profile(
            id=ProfileId(uuid4()),
            tenant_id=TenantId(uuid4()),
            status=ProfileStatus.ACTIVE,
            profile_data=profile_data,
        )

        # Act
        level = profile.experience_level

        # Assert
        assert level == ExperienceLevel.PRINCIPAL

    def test_executive_level_15_years(self, valid_email: EmailAddress):
        """Should classify as EXECUTIVE for 15+ years experience."""
        # Arrange - 15 years of experience
        profile_data = ProfileData(
            name="CTO",
            email=valid_email,
            experience=[
                Experience(
                    title="CTO",
                    company="CompanyE",
                    start_date="2010-01-01",
                    end_date="2025-01-01",
                    description="Executive leadership",
                )
            ],
        )
        profile = Profile(
            id=ProfileId(uuid4()),
            tenant_id=TenantId(uuid4()),
            status=ProfileStatus.ACTIVE,
            profile_data=profile_data,
        )

        # Act
        level = profile.experience_level

        # Assert
        assert level == ExperienceLevel.EXECUTIVE


# ============================================================================
# 6. Profile Data Management Tests (6 tests)
# ============================================================================

class TestProfileDataManagement:
    """Test profile data update and management methods."""

    def test_update_profile_data_changes_data(self, profile_with_minimal_data: Profile, valid_email: EmailAddress):
        """Should update profile data and trigger recomputation."""
        # Arrange
        new_data = ProfileData(
            name="Updated Name",
            email=valid_email,
            summary="New summary",
        )

        # Act
        profile_with_minimal_data.update_profile_data(new_data)

        # Assert
        assert profile_with_minimal_data.profile_data.name == "Updated Name"
        assert profile_with_minimal_data.profile_data.summary == "New summary"

    def test_update_profile_data_preserves_id_and_tenant(self, profile_with_minimal_data: Profile, valid_email: EmailAddress):
        """Should preserve profile id and tenant_id during update."""
        # Arrange
        original_id = profile_with_minimal_data.id
        original_tenant_id = profile_with_minimal_data.tenant_id
        new_data = ProfileData(name="New Name", email=valid_email)

        # Act
        profile_with_minimal_data.update_profile_data(new_data)

        # Assert
        assert profile_with_minimal_data.id == original_id
        assert profile_with_minimal_data.tenant_id == original_tenant_id

    def test_update_profile_data_updates_timestamps(self, profile_with_minimal_data: Profile, valid_email: EmailAddress):
        """Should update both updated_at and last_activity_at."""
        # Arrange
        original_updated_at = profile_with_minimal_data.updated_at
        new_data = ProfileData(name="New Name", email=valid_email)

        # Act
        import time
        time.sleep(0.01)
        profile_with_minimal_data.update_profile_data(new_data)

        # Assert
        assert profile_with_minimal_data.updated_at > original_updated_at
        assert profile_with_minimal_data.last_activity_at is not None
        assert profile_with_minimal_data.last_activity_at >= profile_with_minimal_data.updated_at

    def test_update_profile_data_recomputes_derived_fields(self, profile_with_minimal_data: Profile, valid_email: EmailAddress):
        """Should recompute normalized_skills, experience_level, and searchable_text."""
        # Arrange
        new_data = ProfileData(
            name="New Name",
            email=valid_email,
            skills=[Skill(name=SkillName("Rust"))],
        )

        # Act
        profile_with_minimal_data.update_profile_data(new_data)

        # Assert - derived fields should be updated
        assert "rust" in profile_with_minimal_data.normalized_skills
        assert "Rust" in profile_with_minimal_data.searchable_text

    def test_set_embeddings_updates_profile(self, profile_with_minimal_data: Profile):
        """Should set embeddings and update timestamp."""
        # Arrange
        embeddings = ProfileEmbeddings(
            overall=EmbeddingVector(dimensions=3, values=[0.1, 0.2, 0.3]),
            skills=EmbeddingVector(dimensions=3, values=[0.4, 0.5, 0.6]),
        )
        original_updated_at = profile_with_minimal_data.updated_at

        # Act
        import time
        time.sleep(0.01)
        profile_with_minimal_data.set_embeddings(embeddings)

        # Assert
        assert profile_with_minimal_data.embeddings == embeddings
        assert profile_with_minimal_data.updated_at > original_updated_at

    def test_searchable_text_includes_all_content(self, profile_with_complete_data: Profile):
        """Should build searchable text from all profile sections."""
        # Act
        searchable = profile_with_complete_data.searchable_text

        # Assert - check all sections are included
        assert "Jane Smith" in searchable
        assert "TechCorp" in searchable
        assert "Senior Software Engineer" in searchable
        assert "Stanford University" in searchable
        assert "Python" in searchable


# ============================================================================
# 7. Privacy & Consent Tests (4 tests)
# ============================================================================

class TestProfilePrivacySettings:
    """Test privacy and consent management."""

    def test_default_privacy_settings(self, profile_with_minimal_data: Profile):
        """Should have correct default privacy settings."""
        # Assert
        assert profile_with_minimal_data.privacy.consent_given is False
        assert profile_with_minimal_data.privacy.consent_date is None
        assert profile_with_minimal_data.privacy.deletion_requested is False
        assert profile_with_minimal_data.privacy.gdpr_export_requested is False

    def test_give_consent_updates_settings(self, profile_with_minimal_data: Profile):
        """Should set consent_given to True and record timestamp."""
        # Act
        profile_with_minimal_data.privacy.give_consent()

        # Assert
        assert profile_with_minimal_data.privacy.consent_given is True
        assert profile_with_minimal_data.privacy.consent_date is not None
        assert isinstance(profile_with_minimal_data.privacy.consent_date, datetime)

    def test_request_deletion_sets_flag(self, profile_with_minimal_data: Profile):
        """Should set deletion_requested flag."""
        # Act
        profile_with_minimal_data.privacy.request_deletion()

        # Assert
        assert profile_with_minimal_data.privacy.deletion_requested is True

    def test_request_gdpr_export_sets_flag(self, profile_with_minimal_data: Profile):
        """Should set gdpr_export_requested flag."""
        # Act
        profile_with_minimal_data.privacy.request_gdpr_export()

        # Assert
        assert profile_with_minimal_data.privacy.gdpr_export_requested is True


# ============================================================================
# 8. Analytics Methods Tests (4 tests)
# ============================================================================

class TestProfileAnalytics:
    """Test analytics tracking methods."""

    def test_record_view_increments_count(self, profile_with_minimal_data: Profile):
        """Should increment view_count and update last_viewed_at."""
        # Arrange
        original_count = profile_with_minimal_data.analytics.view_count

        # Act
        profile_with_minimal_data.record_view()

        # Assert
        assert profile_with_minimal_data.analytics.view_count == original_count + 1
        assert profile_with_minimal_data.analytics.last_viewed_at is not None
        assert profile_with_minimal_data.last_activity_at is not None

    def test_record_search_appearance_increments_count(self, profile_with_minimal_data: Profile):
        """Should increment search_appearances count."""
        # Arrange
        original_count = profile_with_minimal_data.analytics.search_appearances

        # Act
        profile_with_minimal_data.record_search_appearance()

        # Assert
        assert profile_with_minimal_data.analytics.search_appearances == original_count + 1

    def test_multiple_views_tracked_correctly(self, profile_with_minimal_data: Profile):
        """Should track multiple views correctly."""
        # Act
        profile_with_minimal_data.record_view()
        profile_with_minimal_data.record_view()
        profile_with_minimal_data.record_view()

        # Assert
        assert profile_with_minimal_data.analytics.view_count == 3

    def test_last_viewed_at_updates_on_each_view(self, profile_with_minimal_data: Profile):
        """Should update last_viewed_at timestamp on each view."""
        # Act
        profile_with_minimal_data.record_view()
        first_view_time = profile_with_minimal_data.analytics.last_viewed_at

        import time
        time.sleep(0.01)

        profile_with_minimal_data.record_view()
        second_view_time = profile_with_minimal_data.analytics.last_viewed_at

        # Assert
        assert second_view_time > first_view_time


# ============================================================================
# 9. Searchability & Business Rules Tests (5 tests)
# ============================================================================

class TestProfileSearchability:
    """Test is_searchable business rule."""

    def test_is_searchable_returns_true_when_all_conditions_met(self, profile_with_complete_data: Profile):
        """Should return True when profile meets all searchability criteria."""
        # Arrange
        profile_with_complete_data.status = ProfileStatus.ACTIVE
        profile_with_complete_data.processing.mark_processing_completed()
        profile_with_complete_data.privacy.give_consent()

        # Act
        result = profile_with_complete_data.is_searchable()

        # Assert
        assert result is True

    def test_is_searchable_returns_false_when_not_active(self, profile_with_complete_data: Profile):
        """Should return False when status is not ACTIVE."""
        # Arrange
        profile_with_complete_data.status = ProfileStatus.INACTIVE
        profile_with_complete_data.processing.mark_processing_completed()
        profile_with_complete_data.privacy.give_consent()

        # Act
        result = profile_with_complete_data.is_searchable()

        # Assert
        assert result is False

    def test_is_searchable_returns_false_when_processing_not_completed(self, profile_with_complete_data: Profile):
        """Should return False when processing is not completed."""
        # Arrange
        profile_with_complete_data.status = ProfileStatus.ACTIVE
        profile_with_complete_data.processing.status = ProcessingStatus.PENDING
        profile_with_complete_data.privacy.give_consent()

        # Act
        result = profile_with_complete_data.is_searchable()

        # Assert
        assert result is False

    def test_is_searchable_returns_false_when_no_consent(self, profile_with_complete_data: Profile):
        """Should return False when consent not given."""
        # Arrange
        profile_with_complete_data.status = ProfileStatus.ACTIVE
        profile_with_complete_data.processing.mark_processing_completed()
        profile_with_complete_data.privacy.consent_given = False

        # Act
        result = profile_with_complete_data.is_searchable()

        # Assert
        assert result is False

    def test_is_searchable_returns_false_when_deletion_requested(self, profile_with_complete_data: Profile):
        """Should return False when deletion requested."""
        # Arrange
        profile_with_complete_data.status = ProfileStatus.ACTIVE
        profile_with_complete_data.processing.mark_processing_completed()
        profile_with_complete_data.privacy.give_consent()
        profile_with_complete_data.privacy.request_deletion()

        # Act
        result = profile_with_complete_data.is_searchable()

        # Assert
        assert result is False


# ============================================================================
# 10. Match Score Calculation Tests (3 tests)
# ============================================================================

class TestMatchScoreCalculation:
    """Test match score calculation against job requirements."""

    def test_calculate_match_score_with_matching_skills(self, profile_with_complete_data: Profile):
        """Should return high score when skills match job requirements."""
        # Arrange
        job_requirements = ["Python", "FastAPI", "PostgreSQL"]

        # Act
        score = profile_with_complete_data.calculate_match_score(job_requirements)

        # Assert
        assert isinstance(score, MatchScore)
        assert score.value == 1.0  # All 3 skills match

    def test_calculate_match_score_with_partial_match(self, profile_with_complete_data: Profile):
        """Should return partial score for partial skill match."""
        # Arrange
        job_requirements = ["Python", "Rust", "Go", "JavaScript"]

        # Act
        score = profile_with_complete_data.calculate_match_score(job_requirements)

        # Assert
        assert 0.0 < score.value < 1.0  # Only Python matches out of 4

    def test_calculate_match_score_with_no_match(self, profile_with_complete_data: Profile):
        """Should return zero score when no skills match."""
        # Arrange
        job_requirements = ["Rust", "Go", "JavaScript"]

        # Act
        score = profile_with_complete_data.calculate_match_score(job_requirements)

        # Assert
        assert score.value == 0.0


# ============================================================================
# 11. Processing Metadata Tests (5 tests)
# ============================================================================

class TestProcessingMetadata:
    """Test processing metadata state management."""

    def test_mark_processing_started(self, profile_with_minimal_data: Profile):
        """Should mark processing as started with timestamp."""
        # Act
        profile_with_minimal_data.processing.mark_processing_started()

        # Assert
        assert profile_with_minimal_data.processing.status == ProcessingStatus.PROCESSING
        assert profile_with_minimal_data.processing.last_processed is not None

    def test_mark_processing_completed(self, profile_with_minimal_data: Profile):
        """Should mark processing as completed with quality score."""
        # Act
        profile_with_minimal_data.processing.mark_processing_completed(quality_score=0.95)

        # Assert
        assert profile_with_minimal_data.processing.status == ProcessingStatus.COMPLETED
        assert profile_with_minimal_data.processing.quality_score == 0.95
        assert profile_with_minimal_data.processing.last_processed is not None

    def test_mark_processing_failed(self, profile_with_minimal_data: Profile):
        """Should mark processing as failed with error message."""
        # Act
        profile_with_minimal_data.processing.mark_processing_failed("Extraction error")

        # Assert
        assert profile_with_minimal_data.processing.status == ProcessingStatus.FAILED
        assert profile_with_minimal_data.processing.error_message == "Extraction error"
        assert profile_with_minimal_data.processing.last_processed is not None

    def test_mark_processing_cancelled(self, profile_with_minimal_data: Profile):
        """Should mark processing as cancelled with reason."""
        # Act
        profile_with_minimal_data.processing.mark_processing_cancelled("User cancelled")

        # Assert
        assert profile_with_minimal_data.processing.status == ProcessingStatus.CANCELLED
        assert "cancelled" in profile_with_minimal_data.processing.error_message.lower()

    def test_processing_metadata_default_state(self, profile_with_minimal_data: Profile):
        """Should have correct default processing state."""
        # Assert
        assert profile_with_minimal_data.processing.status == ProcessingStatus.PENDING
        assert profile_with_minimal_data.processing.version == "1.0"
        assert profile_with_minimal_data.processing.last_processed is None
        assert profile_with_minimal_data.processing.error_message is None


# ============================================================================
# 12. Location Value Object Tests (3 tests)
# ============================================================================

class TestLocationValueObject:
    """Test Location value object behavior."""

    def test_location_is_complete_with_city_and_country(self):
        """Should return True when city and country are present."""
        # Arrange
        location = Location(city="San Francisco", state="CA", country="USA")

        # Act
        result = location.is_complete()

        # Assert
        assert result is True

    def test_location_is_incomplete_without_city(self):
        """Should return False when city is missing."""
        # Arrange
        location = Location(state="CA", country="USA")

        # Act
        result = location.is_complete()

        # Assert
        assert result is False

    def test_location_is_incomplete_without_country(self):
        """Should return False when country is missing."""
        # Arrange
        location = Location(city="San Francisco", state="CA")

        # Act
        result = location.is_complete()

        # Assert
        assert result is False


# ============================================================================
# 13. Experience Value Object Tests (3 tests)
# ============================================================================

class TestExperienceValueObject:
    """Test Experience value object methods."""

    def test_is_current_role_returns_true(self):
        """Should return True for current roles."""
        # Arrange
        exp = Experience(
            title="Engineer",
            company="TechCorp",
            start_date="2023-01-01",
            current=True,
            end_date=None,
        )

        # Act
        result = exp.is_current_role()

        # Assert
        assert result is True

    def test_is_current_role_returns_false_with_end_date(self):
        """Should return False when end_date is set."""
        # Arrange
        exp = Experience(
            title="Engineer",
            company="TechCorp",
            start_date="2023-01-01",
            end_date="2024-01-01",
            current=False,
        )

        # Act
        result = exp.is_current_role()

        # Assert
        assert result is False

    def test_duration_years_calculation(self):
        """Should calculate duration in years correctly."""
        # Arrange
        exp = Experience(
            title="Engineer",
            company="TechCorp",
            start_date="2020-01-01T00:00:00Z",
            end_date="2023-01-01T00:00:00Z",
        )

        # Act
        duration = exp.duration_years()

        # Assert
        assert duration is not None
        assert 2.9 < duration < 3.1  # Approximately 3 years


# ============================================================================
# 14. Education Value Object Tests (1 test)
# ============================================================================

class TestEducationValueObject:
    """Test Education value object methods."""

    def test_is_completed_returns_true_with_end_date(self):
        """Should return True when education has end_date."""
        # Arrange
        edu = Education(
            institution="Stanford",
            degree="BS",
            field="CS",
            end_date="2016-05-31",
        )

        # Act
        result = edu.is_completed()

        # Assert
        assert result is True


# ============================================================================
# 15. Skill Value Object Tests (1 test)
# ============================================================================

class TestSkillValueObject:
    """Test Skill value object methods."""

    def test_is_expert_level_with_high_proficiency(self):
        """Should return True for expert-level skills."""
        # Arrange
        skill = Skill(name=SkillName("Python"), proficiency=5, years_of_experience=8)

        # Act
        result = skill.is_expert_level()

        # Assert
        assert result is True


# ============================================================================
# 16. Property-Based Tests with Hypothesis (3 tests)
# ============================================================================

class TestProfilePropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        name=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        email_local=st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
    )
    def test_completeness_score_always_between_0_and_100(self, name: str, email_local: str):
        """Property: Any valid profile should have completeness score between 0-100."""
        # Arrange
        email = EmailAddress(f"{email_local}@example.com")
        profile_data = ProfileData(name=name, email=email)
        profile = Profile(
            id=ProfileId(uuid4()),
            tenant_id=TenantId(uuid4()),
            status=ProfileStatus.ACTIVE,
            profile_data=profile_data,
        )

        # Act
        score = profile.calculate_completeness_score()

        # Assert
        assert 0.0 <= score <= 100.0

    @given(
        status=st.sampled_from([ProfileStatus.ACTIVE, ProfileStatus.INACTIVE, ProfileStatus.DRAFT])
    )
    def test_status_transitions_always_update_timestamp(self, status: ProfileStatus):
        """Property: Status transitions should always update updated_at."""
        # Arrange
        profile_data = ProfileData(name="Test", email=EmailAddress("test@example.com"))
        profile = Profile(
            id=ProfileId(uuid4()),
            tenant_id=TenantId(uuid4()),
            status=status,
            profile_data=profile_data,
        )
        original_updated_at = profile.updated_at

        # Act
        import time
        time.sleep(0.01)
        profile.archive()

        # Assert
        assert profile.updated_at > original_updated_at

    @given(
        name=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    )
    def test_validation_is_consistent(self, name: str):
        """Property: has_required_fields should return same result for same input."""
        # Arrange
        email = EmailAddress("test@example.com")
        profile_data = ProfileData(name=name, email=email)

        # Act - call twice
        result1 = profile_data.has_required_fields()
        result2 = profile_data.has_required_fields()

        # Assert - should be consistent
        assert result1 == result2