"""API layer converters for domain-to-API schema transformations.

This module contains helper functions for converting between domain entities
and API request/response schemas. These are API-layer concerns and kept
separate from the router to maintain slim controllers.
"""

from uuid import UUID

from app.api.schemas.search_schemas import (
    EducationRequirement,
    ExperienceRequirement,
    LocationFilter,
    RangeFilter,
    SalaryFilter,
    SearchFilter,
    SearchMode,
    SearchRequest,
    SkillRequirement,
)
from app.domain.entities.saved_search import SearchConfiguration
from app.infrastructure.persistence.models.base import PaginationModel


def build_search_request_from_config(
    config: SearchConfiguration, tenant_id: UUID
) -> SearchRequest:
    """
    Build SearchRequest from domain SearchConfiguration with tenant context.

    This helper reconstructs the API SearchRequest model from the domain's
    SearchConfiguration, which is needed because the API response includes
    the full search_request object.

    Args:
        config: Domain SearchConfiguration entity
        tenant_id: Tenant identifier for the request

    Returns:
        SearchRequest API schema ready for search execution
    """
    return SearchRequest(
        query=config.query,
        search_mode=SearchMode(config.search_mode),
        tenant_id=tenant_id,
        max_results=config.max_results,
        pagination=PaginationModel(page=1, page_size=config.max_results),
        basic_filters=[SearchFilter(**f) for f in config.basic_filters],
        range_filters=[RangeFilter(**f) for f in config.range_filters],
        location_filter=LocationFilter(**config.location_filter)
        if config.location_filter
        else None,
        salary_filter=SalaryFilter(**config.salary_filter)
        if config.salary_filter
        else None,
        skill_requirements=[SkillRequirement(**s) for s in config.skill_requirements],
        experience_requirements=ExperienceRequirement(**config.experience_requirements)
        if config.experience_requirements
        else None,
        education_requirements=EducationRequirement(**config.education_requirements)
        if config.education_requirements
        else None,
        include_inactive=config.include_inactive,
        min_match_score=config.min_match_score,
        enable_query_expansion=config.enable_query_expansion,
        vector_weight=config.vector_weight,
        keyword_weight=config.keyword_weight,
    )


__all__ = ["build_search_request_from_config"]
