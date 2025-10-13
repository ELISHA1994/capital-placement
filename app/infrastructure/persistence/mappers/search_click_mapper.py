"""
Mapper between SearchClick domain entities and SearchClickTable persistence models.
"""

from __future__ import annotations

from app.domain.entities.search_click import SearchClick, ClickContext
from app.domain.value_objects import SearchClickId, TenantId, UserId, ProfileId
from app.infrastructure.persistence.models.search_click_table import SearchClickTable


class SearchClickMapper:
    """Maps between SearchClick domain entities and SearchClickTable persistence models."""

    @staticmethod
    def to_domain(table: SearchClickTable) -> SearchClick:
        """Convert SearchClickTable (persistence) to SearchClick (domain entity)."""

        # Reconstruct context from JSONB
        context = ClickContext.from_dict(table.context or {})

        return SearchClick(
            id=SearchClickId(table.id),
            tenant_id=TenantId(table.tenant_id),
            user_id=UserId(table.user_id),
            search_id=table.search_id,
            profile_id=ProfileId(table.profile_id),
            position=table.position,
            clicked_at=table.clicked_at,
            context=context,
            relevance_score=table.relevance_score,
            rank_quality=table.rank_quality,
            metadata=table.extra_metadata or {},
        )

    @staticmethod
    def to_table(entity: SearchClick) -> SearchClickTable:
        """Convert SearchClick (domain entity) to SearchClickTable (persistence)."""

        return SearchClickTable(
            id=entity.id.value,
            tenant_id=entity.tenant_id.value,
            user_id=entity.user_id.value,
            search_id=entity.search_id,
            profile_id=entity.profile_id.value,
            position=entity.position,
            clicked_at=entity.clicked_at,
            relevance_score=entity.relevance_score,
            rank_quality=entity.rank_quality,
            engagement_signal=entity.get_engagement_signal(),
            context=entity.context.to_dict(),
            session_id=entity.context.session_id,
            ip_address=entity.context.ip_address,
            extra_metadata=entity.metadata,
            created_at=entity.clicked_at,
            updated_at=entity.clicked_at,
        )

    @staticmethod
    def to_table_batch(entities: list[SearchClick]) -> list[SearchClickTable]:
        """Convert multiple entities for batch insert."""
        return [SearchClickMapper.to_table(entity) for entity in entities]


__all__ = ["SearchClickMapper"]
