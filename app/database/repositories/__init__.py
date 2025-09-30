"""
Database repositories for the CV Matching Platform.

This module contains repository implementations using SQLModel/SQLAlchemy.
Updated to use SQLModel for 70-85% code reduction and improved maintainability.
"""

from .postgres import (
    SQLModelRepository,
    VectorRepository, 
    UserRepository,
    TenantRepository,
    ProfileRepository,
)

__all__ = [
    "SQLModelRepository",
    "VectorRepository",
    "UserRepository", 
    "TenantRepository",
    "ProfileRepository",
]