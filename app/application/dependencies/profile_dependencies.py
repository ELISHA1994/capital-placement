"""Dependency contracts for ProfileApplicationService."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Any

from app.domain.repositories.profile_repository import IProfileRepository
from app.domain.interfaces import (
    IAuditService,
    ITaskManager,
    IUsageService,
)


@runtime_checkable
class IEmbeddingService(Protocol):
    """Interface for embedding generation services."""

    async def generate_embedding(self, text: str, **kwargs) -> list[float]:
        """Generate embedding vector for text."""
        ...


@runtime_checkable
class ISearchIndexService(Protocol):
    """Interface for search index update services."""

    async def update_profile_index(self, profile_id: str, profile_data: dict) -> None:
        """Update search index for a profile."""
        ...

    async def delete_profile_index(self, profile_id: str) -> None:
        """Remove profile from search index."""
        ...

    async def remove_profile_index(self, profile_id: str) -> None:
        """Remove profile from search index (alias for delete_profile_index)."""
        ...


@dataclass
class ProfileDependencies:
    """Dependencies required by ProfileApplicationService."""

    # Core repository
    profile_repository: IProfileRepository

    # Optional services for extended functionality
    usage_service: IUsageService | None = None
    audit_service: IAuditService | None = None
    task_manager: ITaskManager | None = None
    embedding_service: IEmbeddingService | None = None
    search_index_service: ISearchIndexService | None = None


__all__ = ["ProfileDependencies", "IEmbeddingService", "ISearchIndexService"]
