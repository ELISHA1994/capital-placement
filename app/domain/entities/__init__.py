"""Domain entities exposed for application layer use."""

from .document_processing import (
    DocumentProcessing,
    ProcessingError,
    ProcessingTiming,
    QualityMetrics,
)
from .profile import (
    ExperienceLevel,
    ProcessingMetadata,
    ProcessingStatus,
    Profile,
    ProfileAnalytics,
    ProfileStatus,
    PrivacySettings,
    Location,
)

__all__ = [
    # Document Processing
    "DocumentProcessing",
    "ProcessingError",
    "ProcessingTiming",
    "QualityMetrics",
    # Profile
    "ExperienceLevel",
    "ProcessingMetadata",
    "ProcessingStatus",
    "Profile",
    "ProfileAnalytics",
    "ProfileStatus",
    "PrivacySettings",
    "Location",
]
