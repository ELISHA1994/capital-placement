"""
Mapper between SavedSearch domain entities and SavedSearchTable persistence models.

Handles bidirectional conversion with proper value object transformations.
"""

from __future__ import annotations

from typing import Any, Dict

from app.domain.entities.saved_search import (
    AlertFrequency,
    ExecutionStatistics,
    SavedSearch,
    SavedSearchStatus,
    SearchConfiguration,
)
from app.domain.value_objects import SavedSearchId, TenantId, UserId
from app.infrastructure.persistence.models.saved_search_table import SavedSearchTable


class SavedSearchMapper:
    """Maps between SavedSearch domain entities and SavedSearchTable persistence models."""

    @staticmethod
    def to_domain(table: SavedSearchTable) -> SavedSearch:
        """Convert SavedSearchTable (persistence) to SavedSearch (domain entity)."""

        # Map configuration from JSONB
        configuration = SavedSearchMapper._map_configuration_to_domain(
            table.configuration or {}
        )

        # Map statistics
        statistics = ExecutionStatistics(
            total_executions=table.total_executions,
            last_run=table.last_run,
            last_result_count=table.last_result_count,
            new_results_since_last_run=table.new_results_since_last_run,
            average_result_count=table.average_result_count,
            average_execution_time_ms=table.average_execution_time_ms,
        )

        # Map shared users
        shared_with_users = [
            UserId(user_id) for user_id in (table.shared_with_users or [])
        ]

        return SavedSearch(
            id=SavedSearchId(table.id),
            tenant_id=TenantId(table.tenant_id),
            created_by=UserId(table.created_by),
            name=table.name,
            description=table.description,
            configuration=configuration,
            is_alert=table.is_alert,
            alert_frequency=AlertFrequency(table.alert_frequency),
            next_alert_at=table.next_alert_at,
            is_shared=table.is_shared,
            shared_with_users=shared_with_users,
            status=SavedSearchStatus(table.status),
            statistics=statistics,
            created_at=table.created_at,
            updated_at=table.updated_at,
            updated_by=UserId(table.updated_by) if table.updated_by else None,
            metadata=table.extra_metadata or {},
        )

    @staticmethod
    def to_table(entity: SavedSearch) -> SavedSearchTable:
        """Convert SavedSearch (domain entity) to SavedSearchTable (persistence)."""

        # Map configuration to JSONB
        configuration_dict = SavedSearchMapper._map_configuration_to_persistence(
            entity.configuration
        )

        # Map shared users to string list
        shared_with_users = [
            str(user_id.value) for user_id in entity.shared_with_users
        ]

        return SavedSearchTable(
            id=entity.id.value,
            tenant_id=entity.tenant_id.value,
            created_by=entity.created_by.value,
            name=entity.name,
            description=entity.description,
            configuration=configuration_dict,
            is_alert=entity.is_alert,
            alert_frequency=entity.alert_frequency.value,
            next_alert_at=entity.next_alert_at,
            is_shared=entity.is_shared,
            shared_with_users=shared_with_users,
            status=entity.status.value,
            total_executions=entity.statistics.total_executions,
            last_run=entity.statistics.last_run,
            last_result_count=entity.statistics.last_result_count,
            new_results_since_last_run=entity.statistics.new_results_since_last_run,
            average_result_count=entity.statistics.average_result_count,
            average_execution_time_ms=entity.statistics.average_execution_time_ms,
            extra_metadata=entity.metadata,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            updated_by=entity.updated_by.value if entity.updated_by else None,
        )

    @staticmethod
    def update_table_from_domain(
        table: SavedSearchTable,
        entity: SavedSearch
    ) -> SavedSearchTable:
        """Update existing SavedSearchTable with data from SavedSearch domain entity."""

        table.name = entity.name
        table.description = entity.description
        table.configuration = SavedSearchMapper._map_configuration_to_persistence(
            entity.configuration
        )
        table.is_alert = entity.is_alert
        table.alert_frequency = entity.alert_frequency.value
        table.next_alert_at = entity.next_alert_at
        table.is_shared = entity.is_shared
        table.shared_with_users = [
            str(user_id.value) for user_id in entity.shared_with_users
        ]
        table.status = entity.status.value
        table.total_executions = entity.statistics.total_executions
        table.last_run = entity.statistics.last_run
        table.last_result_count = entity.statistics.last_result_count
        table.new_results_since_last_run = entity.statistics.new_results_since_last_run
        table.average_result_count = entity.statistics.average_result_count
        table.average_execution_time_ms = entity.statistics.average_execution_time_ms
        table.extra_metadata = entity.metadata
        table.updated_at = entity.updated_at
        table.updated_by = entity.updated_by.value if entity.updated_by else None

        return table

    @staticmethod
    def _map_configuration_to_domain(config: Dict[str, Any]) -> SearchConfiguration:
        """Map JSONB configuration dictionary to domain SearchConfiguration."""

        return SearchConfiguration(
            query=config.get("query", ""),
            search_mode=config.get("search_mode", "hybrid"),
            max_results=config.get("max_results", 100),
            basic_filters=config.get("basic_filters", []),
            range_filters=config.get("range_filters", []),
            location_filter=config.get("location_filter"),
            salary_filter=config.get("salary_filter"),
            skill_requirements=config.get("skill_requirements", []),
            experience_requirements=config.get("experience_requirements"),
            education_requirements=config.get("education_requirements"),
            include_inactive=config.get("include_inactive", False),
            min_match_score=config.get("min_match_score", 0.0),
            enable_query_expansion=config.get("enable_query_expansion", True),
            vector_weight=config.get("vector_weight", 0.7),
            keyword_weight=config.get("keyword_weight", 0.3),
        )

    @staticmethod
    def _map_configuration_to_persistence(
        config: SearchConfiguration
    ) -> Dict[str, Any]:
        """Map domain SearchConfiguration to JSONB dictionary."""

        return {
            "query": config.query,
            "search_mode": config.search_mode,
            "max_results": config.max_results,
            "basic_filters": config.basic_filters,
            "range_filters": config.range_filters,
            "location_filter": config.location_filter,
            "salary_filter": config.salary_filter,
            "skill_requirements": config.skill_requirements,
            "experience_requirements": config.experience_requirements,
            "education_requirements": config.education_requirements,
            "include_inactive": config.include_inactive,
            "min_match_score": config.min_match_score,
            "enable_query_expansion": config.enable_query_expansion,
            "vector_weight": config.vector_weight,
            "keyword_weight": config.keyword_weight,
        }


__all__ = ["SavedSearchMapper"]