"""
Profile Management API Endpoints

Comprehensive CV profile management with:
- Profile retrieval and detailed views
- Profile updates and modifications
- Bulk operations and exports
- Analytics and insights
- Privacy and compliance features
- Multi-tenant data isolation
"""

from datetime import datetime
from uuid import UUID

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Path, Query
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.dependencies import ProfileServiceDep, map_domain_exception_to_http
from app.api.schemas.base import PaginatedResponse
from app.api.schemas.profile_schemas import (
    Profile as ProfileSchema,
)
from app.api.schemas.profile_schemas import (
    ProfileAnalyticsSummary,
    ProfileDeletionResponse,
    ProfileRestorationResponse,
    ProfileSummary,
    ProfileUpdate,
    SimilarProfileItem,
    SimilarProfilesResponse,
)
from app.core.dependencies import (
    CurrentUserDep,
)
from app.domain.entities.profile import ProcessingStatus
from app.domain.exceptions import DomainException
from app.domain.value_objects import ProfileId, TenantId
from app.infrastructure.persistence.mappers.profile_mapper import ProfileMapper
from app.infrastructure.persistence.models.base import PaginationModel
from app.infrastructure.providers.export_provider import get_export_service
from app.infrastructure.providers.tenant_provider import (
    get_tenant_service as get_tenant_manager,
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/profiles", tags=["profiles"])


@router.get("/", response_model=PaginatedResponse)
async def list_profiles(
    current_user: CurrentUserDep,
    profile_service: ProfileServiceDep,
    pagination: PaginationModel = Depends(),
    status_filter: ProcessingStatus = Query(
        None, description="Filter by processing status"
    ),
    skill_filter: str = Query(
        None, description="Filter by skill (partial match)"
    ),
    experience_min: int = Query(
        None, ge=0, description="Minimum years of experience"
    ),
    experience_max: int = Query(
        None, ge=0, description="Maximum years of experience"
    ),
    quality_min: float = Query(
        None, ge=0.0, le=1.0, description="Minimum quality score"
    ),
    sort_by: str = Query(
        "last_updated",
        description="Sort field (last_updated, quality_score, experience)",
    ),
    sort_order: str = Query("desc", description="Sort order (asc, desc)"),
    include_incomplete: bool = Query(
        False, description="Include profiles with incomplete processing"
    ),
) -> PaginatedResponse:
    """
    List CV profiles with advanced filtering and sorting.

    Supports filtering by:
    - Processing status and quality scores
    - Skills and experience levels
    - Custom date ranges
    - Profile completeness

    Provides efficient pagination with:
    - Configurable page sizes
    - Multiple sort options
    - Total count tracking
    - Performance optimization
    """
    try:
        logger.info(
            "Profile list requested",
            tenant_id=current_user.tenant_id,
            user_id=current_user.user_id,
            filters={
                "status": status_filter,
                "skill": skill_filter,
                "experience_range": f"{experience_min}-{experience_max}",
                "quality_min": quality_min,
            },
        )

        result = await profile_service.list_profiles(
            tenant_id=UUID(str(current_user.tenant_id)),
            pagination_offset=pagination.offset,
            pagination_limit=pagination.limit,
            status_filter=status_filter,
            include_incomplete=include_incomplete,
            quality_min=quality_min,
            experience_min=experience_min,
            experience_max=experience_max,
            skill_filter=skill_filter,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        page_items = [
            ProfileSummary(
                profile_id=item.profile_id,
                email=item.email,
                full_name=item.full_name,
                title=item.title,
                current_company=item.current_company,
                total_experience_years=item.total_experience_years,
                top_skills=item.top_skills,
                last_updated=item.last_updated,
                processing_status=item.processing_status,
                quality_score=item.quality_score,
            )
            for item in result.items
        ]

        return PaginatedResponse.create(
            items=page_items,
            total=result.total,
            page=pagination.page,
            size=pagination.size,
        )

    except DomainException as domain_exc:
        # Map domain exceptions to appropriate HTTP responses
        raise map_domain_exception_to_http(domain_exc)
    except Exception as e:
        logger.error(f"Failed to list profiles: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve profiles")


@router.get("/{profile_id}", response_model=ProfileSchema)
async def get_profile(
    current_user: CurrentUserDep,
    profile_service: ProfileServiceDep,
    profile_id: str = Path(..., description="Profile identifier"),
    include_analytics: bool = Query(False, description="Include profile analytics"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> ProfileSchema:
    """
    Get detailed CV profile by ID.

    Returns complete profile information including:
    - All extracted structured data
    - Processing metadata and quality scores
    - Search optimization fields
    - Optional analytics and insights
    """
    try:
        logger.debug("Profile retrieval requested", profile_id=profile_id)

        domain_profile = await profile_service.get_profile(
            tenant_id=TenantId(current_user.tenant_id),
            profile_id=ProfileId(profile_id),
            schedule_task=background_tasks,  # Enable view tracking
        )

        if domain_profile is None:
            raise HTTPException(status_code=404, detail="Profile not found")

        profile_table = ProfileMapper.to_table(domain_profile)
        profile_schema = ProfileSchema.from_table(profile_table)

        if not include_analytics:
            profile_schema.analytics = None

        return profile_schema

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve profile")


@router.put("/{profile_id}", response_model=ProfileSchema)
async def update_profile(
    profile_id: str,
    profile_update: ProfileUpdate,
    current_user: CurrentUserDep,
    profile_service: ProfileServiceDep,
    regenerate_embeddings: bool = Query(
        False, description="Regenerate embeddings after update"
    ),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> ProfileSchema:
    """
    Update the CV profile with new information.

    Supports partial updates to:
    - Professional information (title, summary)
    - Skills and competencies
    - Experience and education
    - Contact information
    - Job preferences and availability

    Options:
    - Automatic embedding regeneration
    - Search index updates
    - Quality score recalculation
    """
    try:
        # Get update data (only fields that were set)
        update_data = profile_update.model_dump(exclude_unset=True)

        # Call application service - it handles ALL business logic
        updated_profile = await profile_service.update_profile(
            tenant_id=TenantId(current_user.tenant_id),
            profile_id=ProfileId(profile_id),
            update_data=update_data,
            user_id=current_user.user_id,
            regenerate_embeddings=regenerate_embeddings,
            schedule_task=background_tasks,
        )

        if updated_profile is None:
            raise HTTPException(status_code=404, detail="Profile not found")

        # Convert domain entity to API schema and return
        profile_table = ProfileMapper.to_table(updated_profile)
        return ProfileSchema.from_table(profile_table)

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Profile update validation failed: {e}", profile_id=profile_id)
        raise HTTPException(status_code=400, detail=f"Invalid update data: {str(e)}")
    except DomainException as domain_exc:
        raise map_domain_exception_to_http(domain_exc)
    except Exception as e:
        logger.error(f"Failed to update profile: {e}", profile_id=profile_id)
        raise HTTPException(status_code=500, detail="Failed to update profile")


@router.delete("/{profile_id}")
async def delete_profile(
    profile_id: str,
    current_user: CurrentUserDep,
    profile_service: ProfileServiceDep,
    permanent: bool = Query(False, description="Permanently delete (vs soft delete)"),
    reason: str = Query(None, description="Reason for deletion"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> JSONResponse:
    """
    Delete CV profile (soft delete by default).

    Options:
    - **Soft delete**: Marks profile as deleted but preserves data (can be restored)
    - **Permanent delete**: Completely removes profile and associated data (cannot be restored)

    Includes:
    - Automatic cleanup of search index
    - Document storage cleanup (permanent delete only)
    - Analytics data retention (anonymized)
    - Complete audit trail

    Args:
        profile_id: Profile identifier to delete
        permanent: If True, permanently deletes. If False (default), soft deletes.
        reason: Optional reason for deletion (audit purposes)
        current_user: Authenticated user context
        background_tasks: FastAPI background task scheduler
        profile_service: Profile application service (dependency injection)

    Returns:
        ProfileDeletionResponse with deletion status and details

    Raises:
        404: Profile not found
        400: Validation error (e.g., already deleted, processing in progress)
        500: Internal server error
    """
    try:
        logger.info(
            "Profile deletion requested via API",
            profile_id=profile_id,
            permanent=permanent,
            user_id=current_user.user_id,
            tenant_id=current_user.tenant_id,
        )

        # Convert IDs to value objects
        profile_vo = ProfileId(profile_id)
        tenant_vo = TenantId(current_user.tenant_id)

        # Call service - it handles ALL business logic
        result = await profile_service.delete_profile(
            tenant_id=tenant_vo,
            profile_id=profile_vo,
            user_id=current_user.user_id,
            permanent=permanent,
            reason=reason,
            schedule_task=background_tasks,
        )

        # Map service result to HTTP response
        if not result.success:
            return JSONResponse(
                status_code=404,
                content={
                    "detail": result.message,
                    "profile_id": profile_id,
                },
            )

        # Convert service result to API schema
        response = ProfileDeletionResponse(
            success=result.success,
            deletion_type=result.deletion_type,
            profile_id=result.profile_id,
            message=result.message,
            can_restore=result.can_restore,
        )

        logger.info(
            "Profile deletion completed via API",
            profile_id=profile_id,
            deletion_type=result.deletion_type,
            user_id=current_user.user_id,
        )

        return JSONResponse(
            status_code=200,
            content=response.model_dump(),
        )

    except ValueError as e:
        # Validation errors (already deleted, processing in progress, etc.)
        logger.warning(
            "Profile deletion validation failed",
            profile_id=profile_id,
            error=str(e),
            user_id=current_user.user_id,
        )
        return JSONResponse(
            status_code=400,
            content={
                "detail": str(e),
                "profile_id": profile_id,
            },
        )

    except Exception as e:
        # Unexpected errors
        logger.error(
            "Profile deletion failed",
            profile_id=profile_id,
            error=str(e),
            user_id=current_user.user_id,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error during profile deletion",
                "profile_id": profile_id,
            },
        )


@router.post("/{profile_id}/restore", response_model=ProfileRestorationResponse)
async def restore_profile(
    profile_id: str,
    current_user: CurrentUserDep,
    profile_service: ProfileServiceDep,
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> ProfileRestorationResponse:
    """
    Restore a soft-deleted profile.

    Restores a previously soft-deleted profile, including:
    - Profile data restoration
    - Search index re-indexing
    - Analytics data reactivation
    - Embedding regeneration

    Args:
        profile_id: Profile identifier to restore
        current_user: Authenticated user context
        profile_service: Profile application service (dependency injection)
        background_tasks: FastAPI background task scheduler

    Returns:
        ProfileRestorationResponse with restoration status and details

    Raises:
        404: Profile not found or not deleted
        400: Validation error (e.g., cannot restore this profile)
        500: Internal server error
    """
    try:
        logger.info(
            "Profile restoration requested via API",
            profile_id=profile_id,
            user_id=current_user.user_id,
            tenant_id=current_user.tenant_id,
        )

        # Convert IDs to value objects
        profile_vo = ProfileId(profile_id)
        tenant_vo = TenantId(current_user.tenant_id)

        # Call service - it handles ALL business logic
        restored_profile = await profile_service.restore_profile(
            tenant_id=tenant_vo,
            profile_id=profile_vo,
            user_id=current_user.user_id,
            schedule_task=background_tasks,
        )

        # Map service result to HTTP response
        if not restored_profile:
            raise HTTPException(
                status_code=404, detail="Profile not found or not deleted"
            )

        logger.info(
            "Profile restoration completed via API",
            profile_id=profile_id,
            user_id=current_user.user_id,
        )

        # Return ProfileRestorationResponse schema
        return ProfileRestorationResponse(
            success=True,
            profile_id=profile_id,
            message="Profile restored successfully and re-indexed for search",
        )

    except ValueError as e:
        # Validation errors (cannot restore, etc.)
        logger.warning(
            "Profile restoration validation failed",
            profile_id=profile_id,
            error=str(e),
            user_id=current_user.user_id,
        )
        raise HTTPException(status_code=400, detail=str(e))

    except DomainException as domain_exc:
        # Map domain exceptions to appropriate HTTP responses
        raise map_domain_exception_to_http(domain_exc)

    except Exception as e:
        # Unexpected errors
        logger.error(
            "Profile restoration failed",
            profile_id=profile_id,
            error=str(e),
            user_id=current_user.user_id,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Internal server error during profile restoration"
        )


@router.get("/{profile_id}/analytics", response_model=ProfileAnalyticsSummary)
async def get_profile_analytics(
    profile_id: str,
    current_user: CurrentUserDep,
    profile_service: ProfileServiceDep,
    time_range_days: int = Query(
        30, ge=1, le=365, description="Analytics time range in days"
    ),
) -> ProfileAnalyticsSummary:
    """
    Get comprehensive analytics for a profile.

    Provides insights on:
    - View patterns and engagement
    - Search performance and visibility
    - Market demand for skills (MVP: placeholder)
    - Profile optimization suggestions
    - Competitive benchmarking (MVP: placeholder)
    """
    try:
        logger.info(
            "Profile analytics requested",
            profile_id=profile_id,
            tenant_id=current_user.tenant_id,
            time_range_days=time_range_days,
        )

        # Convert to value objects
        profile_vo = ProfileId(profile_id)
        tenant_vo = TenantId(current_user.tenant_id)

        # Call service - it handles ALL business logic
        analytics_data = await profile_service.get_profile_analytics(
            profile_id=profile_vo,
            tenant_id=tenant_vo,
            time_range_days=time_range_days,
        )

        # Handle not found
        if analytics_data is None:
            raise HTTPException(
                status_code=404,
                detail="Profile not found",
            )

        # Construct response from service result
        analytics = ProfileAnalyticsSummary(**analytics_data)

        logger.info(
            "Profile analytics retrieved successfully",
            profile_id=profile_id,
            view_count=analytics.view_count,
            search_appearances=analytics.search_appearances,
        )

        return analytics

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(
            "Profile analytics validation failed", profile_id=profile_id, error=str(e)
        )
        raise HTTPException(status_code=400, detail=str(e))
    except DomainException as domain_exc:
        raise map_domain_exception_to_http(domain_exc)
    except Exception as e:
        logger.error(
            "Failed to get profile analytics",
            profile_id=profile_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve profile analytics"
        )


@router.get("/export/csv")
async def export_profiles_csv(
    current_user: CurrentUserDep,
    status_filter: ProcessingStatus = Query(
        None, description="Filter by processing status"
    ),
    skill_filter: str = Query(None, description="Filter by skill (partial match)"),
    include_contact_info: bool = Query(
        False, description="Include contact information (email, phone, location)"
    ),
    include_full_text: bool = Query(
        False, description="Include full profile text (summary, searchable text)"
    ),
) -> StreamingResponse:
    """
    Export profiles to CSV format with streaming for large datasets.

    This endpoint uses a streaming response to efficiently handle exports
    of 100k+ profiles without memory bloat. Profiles are processed in
    batches and CSV rows are yielded incrementally.

    Features:
    - Memory-efficient streaming (handles 1M+ records)
    - Configurable field selection (privacy compliance)
    - Advanced filtering (status, skills)
    - Proper CSV formatting with headers
    - Timestamped filenames

    Query Parameters:
    - status_filter: Filter by processing status
    - skill_filter: Filter by skill (partial, case-insensitive match)
    - include_contact_info: Include email, phone, location (default: False)
    - include_full_text: Include summary and searchable text (default: False)

    Returns:
        StreamingResponse with text/csv content and attachment disposition
    """
    try:
        logger.info(
            "CSV export requested",
            tenant_id=current_user.tenant_id,
            user_id=current_user.user_id,
            filters={
                "status": status_filter.value if status_filter else None,
                "skill": skill_filter,
            },
            include_contact_info=include_contact_info,
            include_full_text=include_full_text,
        )

        # Check export permissions (async)
        tenant_manager = await get_tenant_manager()
        has_access = await tenant_manager.check_feature_access(
            tenant_id=current_user.tenant_id, feature_name="export"
        )

        if not has_access:
            raise HTTPException(
                status_code=403,
                detail="Export functionality not available in your subscription tier",
            )

        # Get export service and generate CSV stream
        export_service = await get_export_service()

        # Convert tenant_id to TenantId value object
        tenant_vo = TenantId(current_user.tenant_id)

        # Generate CSV using service (returns synchronous generator)
        csv_generator = export_service.generate_csv_export(
            tenant_id=tenant_vo,
            status_filter=status_filter,
            skill_filter=skill_filter,
            include_contact_info=include_contact_info,
            include_full_text=include_full_text,
        )

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"profiles_export_{timestamp}.csv"

        logger.info(
            "CSV export generation started",
            tenant_id=current_user.tenant_id,
            filename=filename,
        )

        # Return streaming response
        return StreamingResponse(
            csv_generator,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "no-cache",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "CSV export failed",
            tenant_id=current_user.tenant_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to export profiles")


@router.post("/{profile_id}/similar")
async def find_similar_profiles(
    profile_id: str,
    current_user: CurrentUserDep,
    profile_service: ProfileServiceDep,
    similarity_threshold: float = Query(
        0.7, ge=0.0, le=1.0, description="Minimum similarity score"
    ),
    max_results: int = Query(
        10, ge=1, le=50, description="Maximum number of similar profiles"
    ),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> JSONResponse:
    """
    Find profiles similar to the given profile.

    Uses advanced similarity matching based on:
    - Skill overlap and complementarity
    - Experience patterns and progression
    - Education and certification alignment
    - Industry and role similarity
    """
    try:
        logger.debug(
            "Similar profiles search requested",
            profile_id=profile_id,
            threshold=similarity_threshold,
        )

        # Convert to value objects
        profile_vo = ProfileId(profile_id)
        tenant_vo = TenantId(current_user.tenant_id)

        # Call application service - it handles ALL business logic
        similar_profiles_data = await profile_service.find_similar_profiles(
            profile_id=profile_vo,
            tenant_id=tenant_vo,
            similarity_threshold=similarity_threshold,
            max_results=max_results,
            schedule_task=background_tasks,
        )

        # Map to API response schema
        similar_profiles = [
            SimilarProfileItem(**item) for item in similar_profiles_data
        ]

        response = SimilarProfilesResponse(
            target_profile_id=profile_id,
            similar_profiles=similar_profiles,
            similarity_threshold=similarity_threshold,
            results_count=len(similar_profiles),
        )

        return JSONResponse(content=response.model_dump())

    except ValueError as e:
        logger.warning(
            "Similar profiles validation failed",
            profile_id=profile_id,
            error=str(e),
        )
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Similar profiles search failed: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to find similar profiles"
        ) from e
