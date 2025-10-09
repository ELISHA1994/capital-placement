"""Dependency contracts for ProfileApplicationService."""

from __future__ import annotations

from dataclasses import dataclass

from app.domain.repositories.profile_repository import IProfileRepository


@dataclass
class ProfileDependencies:
    """Dependencies required by ProfileApplicationService."""

    profile_repository: IProfileRepository


__all__ = ["ProfileDependencies"]
