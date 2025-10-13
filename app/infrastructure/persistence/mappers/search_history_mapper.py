"""
Mapper between SearchHistory domain entities and SearchHistoryTable persistence models.

Handles bidirectional conversion with proper value object transformations.
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.domain.entities.search_history import (
    SearchHistory,
    SearchParameters,
    SearchResultsSummary,
    PerformanceMetrics,
    UserEngagement,
    SearchOutcome,
)
from app.domain.value_objects import SearchHistoryId, TenantId, UserId
from app.infrastructure.persistence.models.search_history_table import SearchHistoryTable


class SearchHistoryMapper:
    """Maps between SearchHistory domain entities and SearchHistoryTable persistence models."""

    @staticmethod
    def to_domain(table: SearchHistoryTable) -> SearchHistory:
        """Convert SearchHistoryTable (persistence) to SearchHistory (domain entity)."""

        # Map search parameters
        search_parameters = SearchHistoryMapper._map_parameters_to_domain(
            table.search_parameters or {}
        )

        # Map results summary
        results_summary = SearchHistoryMapper._map_results_to_domain(
            table.results_summary or {},
            table.total_results,
            table.returned_results
        )

        # Map performance metrics
        performance_metrics = SearchHistoryMapper._map_performance_to_domain(
            table.performance_metrics or {},
            table.search_duration_ms,
            table.cache_hit
        )

        # Map engagement
        engagement = SearchHistoryMapper._map_engagement_to_domain(
            table.engagement_data or {},
            table.results_clicked,
            table.profiles_contacted
        )

        return SearchHistory(
            id=SearchHistoryId(table.id),
            tenant_id=TenantId(table.tenant_id),
            user_id=UserId(table.user_id),
            search_parameters=search_parameters,
            results_summary=results_summary,
            performance_metrics=performance_metrics,
            search_outcome=SearchOutcome(table.search_outcome),
            engagement=engagement,
            search_context=table.search_context,
            source=table.source,
            user_agent=table.user_agent,
            ip_address=table.ip_address,
            executed_at=table.executed_at,
            last_interaction_at=table.last_interaction_at,
            metadata=table.extra_metadata or {},
        )

    @staticmethod
    def to_table(entity: SearchHistory) -> SearchHistoryTable:
        """Convert SearchHistory (domain entity) to SearchHistoryTable (persistence)."""

        # Map parameters to JSONB
        parameters_dict = SearchHistoryMapper._map_parameters_to_persistence(
            entity.search_parameters
        )

        # Map results to JSONB
        results_dict = SearchHistoryMapper._map_results_to_persistence(
            entity.results_summary
        )

        # Map performance to JSONB
        performance_dict = SearchHistoryMapper._map_performance_to_persistence(
            entity.performance_metrics
        )

        # Map engagement to JSONB
        engagement_dict = SearchHistoryMapper._map_engagement_to_persistence(
            entity.engagement
        )

        return SearchHistoryTable(
            id=entity.id.value,
            tenant_id=entity.tenant_id.value,
            user_id=entity.user_id.value,
            query_text=entity.search_parameters.query,
            search_mode=entity.search_parameters.search_mode,
            search_parameters=parameters_dict,
            total_results=entity.results_summary.total_count,
            returned_results=entity.results_summary.returned_count,
            results_summary=results_dict,
            search_duration_ms=entity.performance_metrics.total_duration_ms,
            cache_hit=entity.performance_metrics.cache_hit,
            performance_metrics=performance_dict,
            search_outcome=entity.search_outcome.value,
            results_clicked=entity.engagement.results_clicked,
            profiles_contacted=entity.engagement.profiles_contacted,
            engagement_data=engagement_dict,
            engagement_score=entity.engagement.get_engagement_score(),
            search_context=entity.search_context,
            source=entity.source,
            user_agent=entity.user_agent,
            ip_address=entity.ip_address,
            executed_at=entity.executed_at,
            last_interaction_at=entity.last_interaction_at,
            extra_metadata=entity.metadata,
            created_at=entity.executed_at,  # Use executed_at as created_at
            updated_at=entity.last_interaction_at or entity.executed_at,
        )

    @staticmethod
    def update_table_from_domain(
        table: SearchHistoryTable,
        entity: SearchHistory
    ) -> SearchHistoryTable:
        """Update existing SearchHistoryTable with data from SearchHistory domain entity."""

        # Update engagement data (most likely to change)
        table.results_clicked = entity.engagement.results_clicked
        table.profiles_contacted = entity.engagement.profiles_contacted
        table.engagement_data = SearchHistoryMapper._map_engagement_to_persistence(
            entity.engagement
        )
        table.engagement_score = entity.engagement.get_engagement_score()

        # Update outcome and interaction time
        table.search_outcome = entity.search_outcome.value
        table.last_interaction_at = entity.last_interaction_at
        table.extra_metadata = entity.metadata
        table.updated_at = entity.last_interaction_at or entity.executed_at

        return table

    @staticmethod
    def _map_parameters_to_domain(params: Dict[str, Any]) -> SearchParameters:
        """Map JSONB parameters dictionary to domain SearchParameters."""
        return SearchParameters(
            query=params.get("query", ""),
            search_mode=params.get("search_mode", "hybrid"),
            max_results=params.get("max_results", 100),
            basic_filters=params.get("basic_filters", []),
            range_filters=params.get("range_filters", []),
            location_filter=params.get("location_filter"),
            salary_filter=params.get("salary_filter"),
            skill_requirements=params.get("skill_requirements", []),
            experience_requirements=params.get("experience_requirements"),
            education_requirements=params.get("education_requirements"),
            include_inactive=params.get("include_inactive", False),
            min_match_score=params.get("min_match_score", 0.0),
            enable_query_expansion=params.get("enable_query_expansion", True),
            vector_weight=params.get("vector_weight", 0.7),
            keyword_weight=params.get("keyword_weight", 0.3),
        )

    @staticmethod
    def _map_parameters_to_persistence(params: SearchParameters) -> Dict[str, Any]:
        """Map domain SearchParameters to JSONB dictionary."""
        return {
            "query": params.query,
            "search_mode": params.search_mode,
            "max_results": params.max_results,
            "basic_filters": params.basic_filters,
            "range_filters": params.range_filters,
            "location_filter": params.location_filter,
            "salary_filter": params.salary_filter,
            "skill_requirements": params.skill_requirements,
            "experience_requirements": params.experience_requirements,
            "education_requirements": params.education_requirements,
            "include_inactive": params.include_inactive,
            "min_match_score": params.min_match_score,
            "enable_query_expansion": params.enable_query_expansion,
            "vector_weight": params.vector_weight,
            "keyword_weight": params.keyword_weight,
        }

    @staticmethod
    def _map_results_to_domain(
        results: Dict[str, Any],
        total_count: int,
        returned_count: int
    ) -> SearchResultsSummary:
        """Map JSONB results dictionary to domain SearchResultsSummary."""
        return SearchResultsSummary(
            total_count=total_count,
            returned_count=returned_count,
            has_more=results.get("has_more", False),
            average_match_score=results.get("average_match_score", 0.0),
            high_match_count=results.get("high_match_count", 0),
            top_skills_found=results.get("top_skills_found", []),
            location_distribution=results.get("location_distribution", {}),
            experience_range=results.get("experience_range", {}),
            cache_hit=results.get("cache_hit", False),
            vector_search_used=results.get("vector_search_used", False),
            query_expanded=results.get("query_expanded", False),
            expanded_terms=results.get("expanded_terms", []),
        )

    @staticmethod
    def _map_results_to_persistence(results: SearchResultsSummary) -> Dict[str, Any]:
        """Map domain SearchResultsSummary to JSONB dictionary."""
        return {
            "has_more": results.has_more,
            "average_match_score": results.average_match_score,
            "high_match_count": results.high_match_count,
            "top_skills_found": results.top_skills_found,
            "location_distribution": results.location_distribution,
            "experience_range": results.experience_range,
            "cache_hit": results.cache_hit,
            "vector_search_used": results.vector_search_used,
            "query_expanded": results.query_expanded,
            "expanded_terms": results.expanded_terms,
        }

    @staticmethod
    def _map_performance_to_domain(
        perf: Dict[str, Any],
        total_duration_ms: int,
        cache_hit: bool
    ) -> PerformanceMetrics:
        """Map JSONB performance dictionary to domain PerformanceMetrics."""
        return PerformanceMetrics(
            total_duration_ms=total_duration_ms,
            vector_search_duration_ms=perf.get("vector_search_duration_ms"),
            text_search_duration_ms=perf.get("text_search_duration_ms"),
            reranking_duration_ms=perf.get("reranking_duration_ms"),
            post_processing_duration_ms=perf.get("post_processing_duration_ms"),
            cache_hit=cache_hit,
            cache_hit_rate=perf.get("cache_hit_rate"),
            database_queries=perf.get("database_queries", 0),
            ai_tokens_used=perf.get("ai_tokens_used"),
            ai_model_used=perf.get("ai_model_used"),
            ai_cost_estimate=perf.get("ai_cost_estimate"),
        )

    @staticmethod
    def _map_performance_to_persistence(perf: PerformanceMetrics) -> Dict[str, Any]:
        """Map domain PerformanceMetrics to JSONB dictionary."""
        return {
            "vector_search_duration_ms": perf.vector_search_duration_ms,
            "text_search_duration_ms": perf.text_search_duration_ms,
            "reranking_duration_ms": perf.reranking_duration_ms,
            "post_processing_duration_ms": perf.post_processing_duration_ms,
            "cache_hit_rate": perf.cache_hit_rate,
            "database_queries": perf.database_queries,
            "ai_tokens_used": perf.ai_tokens_used,
            "ai_model_used": perf.ai_model_used,
            "ai_cost_estimate": perf.ai_cost_estimate,
        }

    @staticmethod
    def _map_engagement_to_domain(
        engagement: Dict[str, Any],
        results_clicked: List[str],
        profiles_contacted: List[str]
    ) -> UserEngagement:
        """Map JSONB engagement dictionary to domain UserEngagement."""
        return UserEngagement(
            results_viewed=engagement.get("results_viewed", []),
            results_clicked=results_clicked,
            profiles_contacted=profiles_contacted,
            profiles_shortlisted=engagement.get("profiles_shortlisted", []),
            time_on_results_seconds=engagement.get("time_on_results_seconds", 0),
            search_refined=engagement.get("search_refined", False),
            session_abandoned=engagement.get("session_abandoned", False),
            satisfaction_rating=engagement.get("satisfaction_rating"),
            feedback_provided=engagement.get("feedback_provided"),
        )

    @staticmethod
    def _map_engagement_to_persistence(engagement: UserEngagement) -> Dict[str, Any]:
        """Map domain UserEngagement to JSONB dictionary."""
        return {
            "results_viewed": engagement.results_viewed,
            "profiles_shortlisted": engagement.profiles_shortlisted,
            "time_on_results_seconds": engagement.time_on_results_seconds,
            "search_refined": engagement.search_refined,
            "session_abandoned": engagement.session_abandoned,
            "satisfaction_rating": engagement.satisfaction_rating,
            "feedback_provided": engagement.feedback_provided,
        }


__all__ = ["SearchHistoryMapper"]
