"""Provider for profile export service.

Implements singleton pattern for ProfileExportService lifecycle management
following the provider pattern used throughout the infrastructure layer.
"""

from app.application.profile_export_service import ProfileExportService
from app.infrastructure.factories.profile_dependency_factory import get_profile_dependencies

_export_service: ProfileExportService | None = None


async def get_export_service() -> ProfileExportService:
    """Get singleton ProfileExportService instance.

    Returns:
        Singleton ProfileExportService configured with profile dependencies

    Note:
        Uses profile dependencies which include profile repository and
        other services needed for export operations.
    """
    global _export_service
    if _export_service is None:
        deps = await get_profile_dependencies()
        _export_service = ProfileExportService(dependencies=deps)
    return _export_service


__all__ = ["get_export_service"]
