"""Factory for creating ProfileApplicationService dependencies."""

from __future__ import annotations

from app.application.dependencies.profile_dependencies import ProfileDependencies
from app.infrastructure.providers.repository_provider import get_profile_repository


async def get_profile_dependencies() -> ProfileDependencies:
    """Construct dependencies for the profile application service."""
    profile_repository = await get_profile_repository()
    return ProfileDependencies(profile_repository=profile_repository)


__all__ = ["get_profile_dependencies"]
