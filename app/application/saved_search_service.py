"""Application layer orchestrator for saved search workflows following hexagonal architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional
from uuid import uuid4

import structlog

from app.api.schemas.search_schemas import SearchRequest
from app.domain.entities.saved_search import (
    AlertFrequency,
    SavedSearch,
    SavedSearchStatus,
    SearchConfiguration,
)
from app.domain.exceptions import DomainException
from app.domain.value_objects import SavedSearchId, TenantId, UserId

if TYPE_CHECKING:
    from app.application.dependencies.saved_search_dependencies import SavedSearchDependencies


logger = structlog.get_logger(__name__)


class SavedSearchApplicationService:
    """Coordinates saved search operations across domain services and infrastructure.

    This application service follows hexagonal architecture principles by:
    - Using dependency injection via constructor
    - Depending only on domain interfaces (ports)
    - Orchestrating workflow without implementing business logic
    - Maintaining separation between domain and infrastructure concerns
    """

    def __init__(self, dependencies: SavedSearchDependencies) -> None:
        """Initialize with injected dependencies.

        Args:
            dependencies: All required services and repositories
        """
        self._deps = dependencies
        self._logger = structlog.get_logger(__name__)

    async def create_saved_search(
        self,
        name: str,
        search_request: SearchRequest,
        tenant_id: str,
        user_id: str,
        description: Optional[str] = None,
        is_alert: bool = False,
        alert_frequency: Optional[str] = None,
    ) -> SavedSearch:
        """Create a new saved search.

        Args:
            name: Name for the saved search
            search_request: Complete search configuration
            tenant_id: Tenant identifier
            user_id: User creating the saved search
            description: Optional description
            is_alert: Whether to create as an alert
            alert_frequency: Alert frequency if is_alert is True

        Returns:
            Created SavedSearch domain entity

        Raises:
            DomainException: If validation fails or duplicate name exists
        """
        self._logger.info(
            "Creating saved search",
            name=name,
            tenant_id=tenant_id,
            user_id=user_id,
            is_alert=is_alert,
        )

        tenant_value_object = TenantId(tenant_id)
        user_value_object = UserId(user_id)

        # Check for duplicate name
        existing = await self._deps.saved_search_repository.get_by_name(
            name=name,
            user_id=user_value_object,
            tenant_id=tenant_value_object,
        )

        if existing:
            raise DomainException(
                message=f"Saved search with name '{name}' already exists",
                error_code="duplicate_saved_search_name",
            )

        # Map SearchRequest to SearchConfiguration
        configuration = self._map_search_request_to_configuration(search_request)

        # Create domain entity
        saved_search = SavedSearch(
            id=SavedSearchId(uuid4()),
            tenant_id=tenant_value_object,
            created_by=user_value_object,
            name=name,
            description=description,
            configuration=configuration,
            is_alert=is_alert,
            alert_frequency=AlertFrequency(alert_frequency or "never"),
        )

        # Persist
        saved_search = await self._deps.saved_search_repository.save(saved_search)

        self._logger.info(
            "Saved search created successfully",
            saved_search_id=str(saved_search.id),
            name=name,
            tenant_id=tenant_id,
        )

        return saved_search

    async def get_saved_search(
        self,
        saved_search_id: str,
        tenant_id: str,
        user_id: str,
    ) -> SavedSearch:
        """Get a saved search by ID with access control.

        Args:
            saved_search_id: Saved search identifier
            tenant_id: Tenant identifier for isolation
            user_id: User requesting access

        Returns:
            SavedSearch domain entity

        Raises:
            DomainException: If not found or access denied
        """
        saved_search = await self._deps.saved_search_repository.get_by_id(
            saved_search_id=SavedSearchId(saved_search_id),
            tenant_id=TenantId(tenant_id),
        )

        if not saved_search:
            raise DomainException(
                message="Saved search not found",
                error_code="saved_search_not_found",
            )

        # Check access control
        user_value_object = UserId(user_id)
        if not saved_search.can_be_accessed_by(user_value_object):
            raise DomainException(
                message="Access denied to saved search",
                error_code="saved_search_access_denied",
            )

        return saved_search

    async def list_user_saved_searches(
        self,
        user_id: str,
        tenant_id: str,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[SavedSearch]:
        """List saved searches for a user.

        Args:
            user_id: User identifier
            tenant_id: Tenant identifier
            status: Optional status filter
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            List of SavedSearch domain entities
        """
        self._logger.debug(
            "Listing saved searches for user",
            user_id=user_id,
            tenant_id=tenant_id,
            status=status,
        )

        status_enum = SavedSearchStatus(status) if status else None

        saved_searches = await self._deps.saved_search_repository.list_by_user(
            user_id=UserId(user_id),
            tenant_id=TenantId(tenant_id),
            status=status_enum,
            limit=limit,
            offset=offset,
        )

        return saved_searches

    async def update_saved_search(
        self,
        saved_search_id: str,
        tenant_id: str,
        user_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        search_request: Optional[SearchRequest] = None,
        is_alert: Optional[bool] = None,
        alert_frequency: Optional[str] = None,
    ) -> SavedSearch:
        """Update a saved search.

        Args:
            saved_search_id: Saved search identifier
            tenant_id: Tenant identifier
            user_id: User making the update
            name: New name (optional)
            description: New description (optional)
            search_request: New search configuration (optional)
            is_alert: New alert flag (optional)
            alert_frequency: New alert frequency (optional)

        Returns:
            Updated SavedSearch domain entity

        Raises:
            DomainException: If not found or access denied
        """
        saved_search = await self.get_saved_search(
            saved_search_id=saved_search_id,
            tenant_id=tenant_id,
            user_id=user_id,
        )

        user_value_object = UserId(user_id)

        # Check ownership for updates
        if saved_search.created_by != user_value_object:
            raise DomainException(
                message="Only the creator can update a saved search",
                error_code="saved_search_update_denied",
            )

        # Apply updates
        if name is not None:
            saved_search.name = name

        if description is not None:
            saved_search.description = description

        if search_request is not None:
            configuration = self._map_search_request_to_configuration(search_request)
            saved_search.update_configuration(configuration, user_value_object)

        if is_alert is not None and alert_frequency is not None:
            if is_alert:
                saved_search.enable_alert(
                    AlertFrequency(alert_frequency),
                    user_value_object,
                )
            else:
                saved_search.disable_alert(user_value_object)

        # Persist
        saved_search = await self._deps.saved_search_repository.save(saved_search)

        self._logger.info(
            "Saved search updated",
            saved_search_id=saved_search_id,
            tenant_id=tenant_id,
        )

        return saved_search

    async def delete_saved_search(
        self,
        saved_search_id: str,
        tenant_id: str,
        user_id: str,
    ) -> bool:
        """Delete a saved search.

        Args:
            saved_search_id: Saved search identifier
            tenant_id: Tenant identifier
            user_id: User requesting deletion

        Returns:
            True if deleted successfully

        Raises:
            DomainException: If not found or access denied
        """
        saved_search = await self.get_saved_search(
            saved_search_id=saved_search_id,
            tenant_id=tenant_id,
            user_id=user_id,
        )

        user_value_object = UserId(user_id)

        # Check ownership for deletion
        if saved_search.created_by != user_value_object:
            raise DomainException(
                message="Only the creator can delete a saved search",
                error_code="saved_search_delete_denied",
            )

        success = await self._deps.saved_search_repository.delete(
            saved_search_id=SavedSearchId(saved_search_id),
            tenant_id=TenantId(tenant_id),
        )

        if success:
            self._logger.info(
                "Saved search deleted",
                saved_search_id=saved_search_id,
                tenant_id=tenant_id,
            )

        return success

    @staticmethod
    def _map_search_request_to_configuration(
        search_request: SearchRequest
    ) -> SearchConfiguration:
        """Map API SearchRequest to domain SearchConfiguration."""

        return SearchConfiguration(
            query=search_request.query,
            search_mode=search_request.search_mode.value,
            max_results=search_request.max_results,
            basic_filters=[f.model_dump() for f in search_request.basic_filters],
            range_filters=[f.model_dump() for f in search_request.range_filters],
            location_filter=search_request.location_filter.model_dump() if search_request.location_filter else None,
            salary_filter=search_request.salary_filter.model_dump() if search_request.salary_filter else None,
            skill_requirements=[s.model_dump() for s in search_request.skill_requirements],
            experience_requirements=search_request.experience_requirements.model_dump() if search_request.experience_requirements else None,
            education_requirements=search_request.education_requirements.model_dump() if search_request.education_requirements else None,
            include_inactive=search_request.include_inactive,
            min_match_score=search_request.min_match_score,
            enable_query_expansion=search_request.enable_query_expansion,
            vector_weight=search_request.vector_weight,
            keyword_weight=search_request.keyword_weight,
        )


__all__ = ["SavedSearchApplicationService"]