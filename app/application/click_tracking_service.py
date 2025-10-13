"""
Application service for click tracking with async processing and analytics.

Handles high-volume click events with background processing to avoid blocking requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

import structlog

from app.domain.entities.search_click import SearchClick, ClickContext, ClickSource, ClickDevice
from app.domain.exceptions import DomainException
from app.domain.value_objects import SearchClickId, TenantId, UserId, ProfileId

if TYPE_CHECKING:
    from app.application.dependencies.click_tracking_dependencies import ClickTrackingDependencies


logger = structlog.get_logger(__name__)


class ClickTrackingApplicationService:
    """
    Orchestrates click tracking workflows with async processing.

    Key responsibilities:
    - Accept click events from API layer
    - Enrich events with context
    - Queue for async processing
    - Generate analytics
    - Handle fraud detection
    """

    def __init__(self, dependencies: ClickTrackingDependencies) -> None:
        """Initialize with injected dependencies."""
        self._deps = dependencies
        self._logger = structlog.get_logger(__name__)

    async def track_click(
        self,
        search_id: str,
        profile_id: str,
        position: int,
        tenant_id: str,
        user_id: str,
        context_data: Optional[Dict[str, Any]] = None,
        relevance_score: Optional[float] = None,
    ) -> SearchClick:
        """
        Track a single search result click.

        This is a fast, fire-and-forget operation optimized for high throughput.

        Args:
            search_id: Search execution identifier
            profile_id: Profile that was clicked
            position: Position in results (0-based)
            tenant_id: Tenant identifier
            user_id: User who clicked
            context_data: Optional enrichment data
            relevance_score: Score of clicked result

        Returns:
            Created SearchClick domain entity
        """
        self._logger.debug(
            "Tracking search click",
            search_id=search_id,
            profile_id=profile_id,
            position=position,
            tenant_id=tenant_id,
        )

        try:
            # Create context from provided data
            context = self._build_click_context(context_data or {})

            # Create domain entity
            click = SearchClick(
                id=SearchClickId(uuid4()),
                tenant_id=TenantId(tenant_id),
                user_id=UserId(user_id),
                search_id=search_id,
                profile_id=ProfileId(profile_id),
                position=position,
                clicked_at=datetime.utcnow(),
                context=context,
                relevance_score=relevance_score,
            )

            # Persist (fast write)
            click = await self._deps.click_repository.save(click)

            self._logger.info(
                "Click tracked successfully",
                click_id=str(click.id),
                search_id=search_id,
                position=position,
            )

            return click

        except Exception as e:
            # Don't fail the request if click tracking fails
            self._logger.error(
                "Failed to track click",
                error=str(e),
                search_id=search_id,
                profile_id=profile_id,
            )
            raise DomainException(
                message="Failed to track click event",
                error_code="click_tracking_failed",
            )

    async def track_clicks_batch(
        self,
        clicks_data: List[Dict[str, Any]],
        tenant_id: str,
    ) -> int:
        """
        Batch track multiple clicks (for bulk operations).

        Args:
            clicks_data: List of click event dictionaries
            tenant_id: Tenant identifier

        Returns:
            Number of clicks successfully tracked
        """
        self._logger.info(
            "Batch tracking clicks",
            count=len(clicks_data),
            tenant_id=tenant_id,
        )

        try:
            clicks = []
            tenant_vo = TenantId(tenant_id)

            for data in clicks_data:
                context = self._build_click_context(data.get("context", {}))

                click = SearchClick(
                    id=SearchClickId(uuid4()),
                    tenant_id=tenant_vo,
                    user_id=UserId(data["user_id"]),
                    search_id=data["search_id"],
                    profile_id=ProfileId(data["profile_id"]),
                    position=data["position"],
                    clicked_at=data.get("clicked_at", datetime.utcnow()),
                    context=context,
                    relevance_score=data.get("relevance_score"),
                )
                clicks.append(click)

            # Batch insert
            saved_count = await self._deps.click_repository.save_batch(clicks)

            self._logger.info(
                "Batch clicks tracked",
                requested=len(clicks_data),
                saved=saved_count,
            )

            return saved_count

        except Exception as e:
            self._logger.error("Batch click tracking failed", error=str(e))
            raise DomainException(
                message="Failed to batch track clicks",
                error_code="batch_click_tracking_failed",
            )

    async def get_search_analytics(
        self,
        search_id: str,
        tenant_id: str,
    ) -> Dict[str, Any]:
        """
        Get analytics for a specific search.

        Includes:
        - Total clicks
        - Click-through rate
        - Position distribution
        - Engagement signals
        """
        clicks = await self._deps.click_repository.get_clicks_for_search(
            search_id=search_id,
            tenant_id=TenantId(tenant_id),
        )

        if not clicks:
            return {
                "search_id": search_id,
                "total_clicks": 0,
                "position_distribution": {},
                "engagement_signals": {},
            }

        # Calculate analytics
        position_clicks = {}
        engagement_counts = {"strong": 0, "medium": 0, "weak": 0}

        for click in clicks:
            position_clicks[click.position] = position_clicks.get(click.position, 0) + 1
            signal = click.get_engagement_signal()
            engagement_counts[signal] += 1

        return {
            "search_id": search_id,
            "total_clicks": len(clicks),
            "unique_profiles": len(set(str(c.profile_id) for c in clicks)),
            "position_distribution": position_clicks,
            "engagement_signals": engagement_counts,
            "avg_position": sum(c.position for c in clicks) / len(clicks),
            "top_3_clicks": sum(1 for c in clicks if c.position < 3),
        }

    async def get_profile_analytics(
        self,
        profile_id: str,
        tenant_id: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get click analytics for a profile (popularity tracking).

        Args:
            profile_id: Profile identifier
            tenant_id: Tenant identifier
            days: Number of days to analyze

        Returns:
            Analytics dictionary with click trends
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        clicks = await self._deps.click_repository.get_clicks_for_profile(
            profile_id=ProfileId(profile_id),
            tenant_id=TenantId(tenant_id),
            start_date=start_date,
            end_date=end_date,
        )

        if not clicks:
            return {
                "profile_id": profile_id,
                "total_clicks": 0,
                "period_days": days,
            }

        # Analyze trends
        avg_position = sum(c.position for c in clicks) / len(clicks)
        searches = set(c.search_id for c in clicks)

        return {
            "profile_id": profile_id,
            "total_clicks": len(clicks),
            "unique_searches": len(searches),
            "avg_position": avg_position,
            "top_result_count": sum(1 for c in clicks if c.is_top_result()),
            "first_page_count": sum(1 for c in clicks if c.is_first_page()),
            "period_days": days,
            "clicks_per_day": len(clicks) / days,
        }

    async def get_tenant_ctr_report(
        self,
        tenant_id: str,
        start_date: datetime,
        end_date: datetime,
        group_by: str = "day",
    ) -> List[Dict[str, Any]]:
        """
        Generate click-through rate report for tenant.

        Args:
            tenant_id: Tenant identifier
            start_date: Report start date
            end_date: Report end date
            group_by: Grouping level (hour, day, week, month)

        Returns:
            Time-series CTR data
        """
        return await self._deps.click_repository.get_click_through_rate(
            tenant_id=TenantId(tenant_id),
            start_date=start_date,
            end_date=end_date,
            group_by=group_by,
        )

    async def get_position_performance(
        self,
        tenant_id: str,
        days: int = 7,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Analyze click performance by result position.

        Useful for understanding ranking quality and position bias.
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        return await self._deps.click_repository.get_position_analytics(
            tenant_id=TenantId(tenant_id),
            start_date=start_date,
            end_date=end_date,
        )

    @staticmethod
    def _build_click_context(context_data: Dict[str, Any]) -> ClickContext:
        """Build ClickContext from API data."""
        return ClickContext(
            session_id=context_data.get("session_id"),
            ip_address=context_data.get("ip_address"),
            user_agent=context_data.get("user_agent"),
            device_type=ClickDevice(context_data.get("device_type", "unknown")),
            time_to_click_ms=context_data.get("time_to_click_ms"),
            scroll_position=context_data.get("scroll_position"),
            viewport_height=context_data.get("viewport_height"),
            previous_clicks=context_data.get("previous_clicks", 0),
            query_length=context_data.get("query_length", 0),
            results_shown=context_data.get("results_shown", 0),
            filter_count=context_data.get("filter_count", 0),
            source=ClickSource(context_data.get("source", "search_results")),
            referrer_url=context_data.get("referrer_url"),
        )


__all__ = ["ClickTrackingApplicationService"]