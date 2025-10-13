"""Application layer orchestrator for search history workflows following hexagonal architecture."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, List, Dict, Any, Optional

import structlog

from app.domain.entities.search_history import SearchHistory, SearchOutcome, InteractionType
from app.domain.exceptions import DomainException
from app.domain.value_objects import SearchHistoryId, TenantId, UserId

if TYPE_CHECKING:
    from app.application.dependencies.search_history_dependencies import SearchHistoryDependencies


logger = structlog.get_logger(__name__)


class SearchHistoryApplicationService:
    """
    Coordinates search history operations across domain services and infrastructure.

    This application service follows hexagonal architecture principles by:
    - Using dependency injection via constructor
    - Depending only on domain interfaces (ports)
    - Orchestrating workflow without implementing business logic
    - Maintaining separation between domain and infrastructure concerns
    """

    def __init__(self, dependencies: SearchHistoryDependencies) -> None:
        """Initialize with injected dependencies.

        Args:
            dependencies: All required services and repositories
        """
        self._deps = dependencies
        self._logger = structlog.get_logger(__name__)

    async def get_user_search_history(
        self,
        user_id: str,
        tenant_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[SearchHistory]:
        """Get search history for a user.

        Args:
            user_id: User identifier
            tenant_id: Tenant identifier
            start_date: Filter searches from this date
            end_date: Filter searches until this date
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            List of SearchHistory domain entities
        """
        self._logger.debug(
            "Retrieving search history",
            user_id=user_id,
            tenant_id=tenant_id,
            limit=limit
        )

        history = await self._deps.search_history_repository.list_by_user(
            user_id=UserId(user_id),
            tenant_id=TenantId(tenant_id),
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset
        )

        return history

    async def get_search_analytics(
        self,
        tenant_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get aggregated search analytics for a tenant.

        Args:
            tenant_id: Tenant identifier
            start_date: Start of analysis period
            end_date: End of analysis period

        Returns:
            Dictionary with analytics metrics
        """
        # Default to last 30 days if not specified
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)

        self._logger.debug(
            "Generating search analytics",
            tenant_id=tenant_id,
            start_date=start_date,
            end_date=end_date
        )

        analytics = await self._deps.search_history_repository.get_analytics_summary(
            tenant_id=TenantId(tenant_id),
            start_date=start_date,
            end_date=end_date
        )

        # Get popular queries
        popular_queries = await self._deps.search_history_repository.get_popular_queries(
            tenant_id=TenantId(tenant_id),
            limit=10,
            days_back=30
        )

        analytics["popular_queries"] = popular_queries

        return analytics

    async def record_interaction(
        self,
        search_history_id: str,
        tenant_id: str,
        user_id: str,
        interaction_type: str,
        profile_id: Optional[str] = None,
        position: Optional[int] = None
    ) -> None:
        """Record a user interaction with search results.

        Args:
            search_history_id: Search history identifier
            tenant_id: Tenant identifier
            user_id: User identifier
            interaction_type: Type of interaction (clicked, contacted, etc.)
            profile_id: Profile that was interacted with
            position: Position in search results
        """
        self._logger.debug(
            "Recording search interaction",
            search_history_id=search_history_id,
            interaction_type=interaction_type,
            profile_id=profile_id
        )

        # Get search history
        history = await self._deps.search_history_repository.get_by_id(
            search_history_id=SearchHistoryId(search_history_id),
            tenant_id=TenantId(tenant_id)
        )

        if not history:
            raise DomainException(
                message="Search history not found",
                error_code="search_history_not_found"
            )

        # Verify user owns this search
        if str(history.user_id.value) != user_id:
            raise DomainException(
                message="Access denied to search history",
                error_code="search_history_access_denied"
            )

        # Record interaction
        interaction = InteractionType(interaction_type)
        history.record_interaction(interaction, profile_id, position)

        # Save updated history
        await self._deps.search_history_repository.save(history)

        self._logger.info(
            "Search interaction recorded",
            search_history_id=search_history_id,
            interaction_type=interaction_type
        )

    async def get_query_suggestions(
        self,
        user_id: str,
        tenant_id: str,
        query_prefix: str,
        limit: int = 5
    ) -> List[str]:
        """Get search query suggestions based on user's history.

        Args:
            user_id: User identifier
            tenant_id: Tenant identifier
            query_prefix: Query prefix to match
            limit: Maximum suggestions to return

        Returns:
            List of suggested queries
        """
        suggestions = await self._deps.search_history_repository.get_user_query_suggestions(
            user_id=UserId(user_id),
            tenant_id=TenantId(tenant_id),
            query_prefix=query_prefix,
            limit=limit
        )

        return suggestions

    async def cleanup_old_history(
        self,
        tenant_id: str,
        retention_days: int = 90
    ) -> int:
        """Clean up old search history records for data retention compliance.

        Args:
            tenant_id: Tenant identifier
            retention_days: Number of days to retain history

        Returns:
            Number of records deleted
        """
        before_date = datetime.utcnow() - timedelta(days=retention_days)

        self._logger.info(
            "Cleaning up old search history",
            tenant_id=tenant_id,
            retention_days=retention_days,
            before_date=before_date
        )

        deleted_count = await self._deps.search_history_repository.delete_old_records(
            tenant_id=TenantId(tenant_id),
            before_date=before_date
        )

        self._logger.info(
            "Search history cleanup completed",
            tenant_id=tenant_id,
            deleted_count=deleted_count
        )

        return deleted_count


__all__ = ["SearchHistoryApplicationService"]
