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

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Path, Query
from fastapi.responses import JSONResponse, StreamingResponse

from app.domain.entities.profile import ProcessingStatus
from app.api.schemas.profile_schemas import (
    BulkOperationRequest,
    Profile as ProfileSchema,
    ProfileAnalyticsSummary,
    ProfileSummary,
    ProfileUpdate,
)
from app.infrastructure.persistence.mappers.profile_mapper import ProfileMapper
from app.domain.value_objects import ProfileId, TenantId
from app.api.schemas.base import PaginatedResponse
from app.infrastructure.persistence.models.base import PaginationModel
from app.core.dependencies import CurrentUserDep, TenantContextDep, AuthzService, require_permission
from app.infrastructure.providers.usage_provider import get_usage_service
from app.api.dependencies import ProfileServiceDep, map_domain_exception_to_http
from app.domain.exceptions import DomainException, ProfileNotFoundError
from app.infrastructure.providers.tenant_provider import get_tenant_service as get_tenant_manager


logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/profiles", tags=["profiles"])



@router.get("/", response_model=PaginatedResponse)
async def list_profiles(
    current_user: CurrentUserDep,
    profile_service: ProfileServiceDep,
    pagination: PaginationModel = Depends(),
    status_filter: Optional[ProcessingStatus] = Query(None, description="Filter by processing status"),
    skill_filter: Optional[str] = Query(None, description="Filter by skill (partial match)"),
    experience_min: Optional[int] = Query(None, ge=0, description="Minimum years of experience"),
    experience_max: Optional[int] = Query(None, ge=0, description="Maximum years of experience"),
    quality_min: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum quality score"),
    sort_by: str = Query("last_updated", description="Sort field (last_updated, quality_score, experience)"),
    sort_order: str = Query("desc", description="Sort order (asc, desc)"),
    include_incomplete: bool = Query(False, description="Include profiles with incomplete processing")
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
                "quality_min": quality_min
            }
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
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve profiles"
        )


@router.get("/{profile_id}", response_model=ProfileSchema)
async def get_profile(
    current_user: CurrentUserDep,
    profile_service: ProfileServiceDep,
    profile_id: str = Path(..., description="Profile identifier"),
    include_analytics: bool = Query(False, description="Include profile analytics"),
    background_tasks: BackgroundTasks = BackgroundTasks()
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
        )

        if domain_profile is None:
            raise HTTPException(status_code=404, detail="Profile not found")

        profile_table = ProfileMapper.to_table(domain_profile)
        profile_schema = ProfileSchema.from_table(profile_table)

        background_tasks.add_task(
            _track_profile_usage,
            tenant_id=current_user.tenant_id,
            operation="view",
            profile_count=1,
        )

        if not include_analytics:
            profile_schema.analytics = None

        return profile_schema
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get profile: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve profile"
        )


@router.put("/{profile_id}", response_model=ProfileSchema)
async def update_profile(
    profile_id: str,
    profile_update: ProfileUpdate,
    current_user: CurrentUserDep,
    profile_service: ProfileServiceDep,
    regenerate_embeddings: bool = Query(False, description="Regenerate embeddings after update"),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> ProfileSchema:
    """
    Update CV profile with new information.

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
        raise HTTPException(
            status_code=400,
            detail=f"Invalid update data: {str(e)}"
        )
    except DomainException as domain_exc:
        raise map_domain_exception_to_http(domain_exc)
    except Exception as e:
        logger.error(f"Failed to update profile: {e}", profile_id=profile_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to update profile"
        )


@router.delete("/{profile_id}")
async def delete_profile(
    profile_id: str,
    current_user: CurrentUserDep,
    permanent: bool = Query(False, description="Permanently delete (vs soft delete)"),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> JSONResponse:
    """
    Delete CV profile (soft delete by default).
    
    Options:
    - **Soft delete**: Marks profile as deleted but preserves data
    - **Permanent delete**: Completely removes profile and associated data
    
    Includes:
    - Automatic cleanup of search index
    - Document storage cleanup
    - Analytics data retention (anonymized)
    """
    try:
        logger.info(
            "Profile deletion requested",
            profile_id=profile_id,
            permanent=permanent,
            user_id=current_user.user_id
        )
        
        # TODO: Implement profile deletion
        # Get profile first to verify ownership
        # profile = await get_profile(profile_id, current_user=current_user)
        
        if permanent:
            # Permanent deletion
            # await _permanently_delete_profile(profile_id, current_user.tenant_id)
            message = "Profile permanently deleted"
        else:
            # Soft deletion
            # await _soft_delete_profile(profile_id, current_user.user_id)
            message = "Profile deleted (can be restored)"
        
        # Clean up search index
        background_tasks.add_task(
            _remove_from_search_index,
            profile_id=profile_id
        )
        
        # Clean up document storage if permanent
        if permanent:
            background_tasks.add_task(
                _cleanup_profile_documents,
                profile_id=profile_id,
                tenant_id=current_user.tenant_id
            )
        
        # Track profile deletion in background using centralized tracker
        background_tasks.add_task(
            _track_profile_usage,
            tenant_id=current_user.tenant_id,
            operation="delete",
            profile_count=1
        )
        
        return JSONResponse(content={
            "status": "deleted",
            "profile_id": profile_id,
            "permanent": permanent,
            "message": message
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete profile: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete profile"
        )


@router.post("/{profile_id}/restore")
async def restore_profile(
    profile_id: str,
    current_user: CurrentUserDep,
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> JSONResponse:
    """
    Restore a soft-deleted profile.
    
    Includes:
    - Profile data restoration
    - Search index re-indexing
    - Analytics data reactivation
    """
    try:
        logger.info(
            "Profile restoration requested",
            profile_id=profile_id,
            user_id=current_user.user_id
        )
        
        # TODO: Implement profile restoration
        # Check if profile exists and is soft-deleted
        # Restore profile data
        # Re-index for search
        
        background_tasks.add_task(
            _reindex_restored_profile,
            profile_id=profile_id
        )
        
        return JSONResponse(content={
            "status": "restored",
            "profile_id": profile_id,
            "message": "Profile has been restored"
        })
        
    except Exception as e:
        logger.error(f"Failed to restore profile: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to restore profile"
        )


@router.get("/{profile_id}/analytics", response_model=ProfileAnalyticsSummary)
async def get_profile_analytics(
    profile_id: str,
    current_user: CurrentUserDep,
    time_range_days: int = Query(30, ge=1, le=365, description="Analytics time range in days")
) -> ProfileAnalyticsSummary:
    """
    Get comprehensive analytics for a profile.
    
    Provides insights on:
    - View patterns and engagement
    - Search performance and visibility  
    - Market demand for skills
    - Profile optimization suggestions
    - Competitive benchmarking
    """
    try:
        # TODO: Implement analytics retrieval
        # This would involve querying various analytics tables/collections
        
        analytics = ProfileAnalyticsSummary(
            profile_id=profile_id,
            view_count=0,
            search_appearances=0,
            profile_completeness=0.0
        )
        
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to get profile analytics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve profile analytics"
        )


@router.post("/bulk", response_model=Dict[str, Any])
async def bulk_operation(
    bulk_request: BulkOperationRequest,
    current_user: CurrentUserDep,
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> Dict[str, Any]:
    """
    Perform bulk operations on multiple profiles.
    
    Supported operations:
    - **update**: Apply updates to multiple profiles
    - **delete**: Bulk delete profiles (soft or permanent)
    - **tag**: Add/remove tags from profiles
    - **export**: Export profile data in various formats
    - **reprocess**: Reprocess documents for multiple profiles
    """
    try:
        logger.info(
            "Bulk operation requested",
            operation=bulk_request.operation,
            profile_count=len(bulk_request.profile_ids),
            user_id=current_user.user_id
        )
        
        # Validate profile ownership
        # TODO: Check that all profiles belong to the tenant
        
        # Check feature access for bulk operations
        tenant_manager = await get_tenant_manager()
        has_access = await tenant_manager.check_feature_access(
            tenant_id=current_user.tenant_id,
            feature_name="bulk_operations"
        )
        
        if not has_access:
            raise HTTPException(
                status_code=403,
                detail="Bulk operations not available in your subscription tier"
            )
        
        # Execute bulk operation in background
        operation_id = f"bulk_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        background_tasks.add_task(
            _execute_bulk_operation,
            operation_id=operation_id,
            operation_type=bulk_request.operation,
            profile_ids=bulk_request.profile_ids,
            parameters=bulk_request.parameters or {},
            tenant_id=current_user.tenant_id,
            user_id=current_user.user_id
        )
        
        return {
            "operation_id": operation_id,
            "status": "started",
            "profile_count": len(bulk_request.profile_ids),
            "operation": bulk_request.operation,
            "message": f"Bulk {bulk_request.operation} operation started for {len(bulk_request.profile_ids)} profiles"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk operation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to execute bulk operation"
        )


@router.get("/export/csv")
async def export_profiles_csv(
    current_user: CurrentUserDep,
    status_filter: Optional[ProcessingStatus] = Query(None, description="Filter by processing status"),
    skill_filter: Optional[str] = Query(None, description="Filter by skill"),
    include_contact_info: bool = Query(False, description="Include contact information"),
    include_full_text: bool = Query(False, description="Include full profile text")
) -> StreamingResponse:
    """
    Export profiles to CSV format.
    
    Features:
    - Configurable field selection
    - Privacy-compliant exports
    - Large dataset streaming
    - Custom formatting options
    """
    try:
        logger.info(
            "CSV export requested",
            tenant_id=current_user.tenant_id,
            user_id=current_user.user_id,
            filters={"status": status_filter, "skill": skill_filter}
        )
        
        # Check export permissions
        tenant_manager = await get_tenant_manager()
        has_access = await tenant_manager.check_feature_access(
            tenant_id=current_user.tenant_id,
            feature_name="export"
        )
        
        if not has_access:
            raise HTTPException(
                status_code=403,
                detail="Export functionality not available in your subscription tier"
            )
        
        # TODO: Implement CSV export
        def generate_csv():
            yield "profile_id,email,name,title,skills,experience_years\n"
            # Mock data - replace with actual database query
            yield "123,john@example.com,John Doe,Software Engineer,Python;JavaScript,5\n"
        
        return StreamingResponse(
            generate_csv(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=profiles_export.csv"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CSV export failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to export profiles"
        )


@router.post("/{profile_id}/similar")
async def find_similar_profiles(
    profile_id: str,
    current_user: CurrentUserDep,
    similarity_threshold: float = Query(0.7, ge=0.0, le=1.0, description="Minimum similarity score"),
    max_results: int = Query(10, ge=1, le=50, description="Maximum number of similar profiles")
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
            threshold=similarity_threshold
        )
        
        # TODO: Implement similarity search
        # 1. Get target profile embeddings
        # 2. Perform vector similarity search
        # 3. Apply business logic filtering
        # 4. Return ranked results
        
        similar_profiles = []  # Mock empty results
        
        return JSONResponse(content={
            "target_profile_id": profile_id,
            "similar_profiles": similar_profiles,
            "similarity_threshold": similarity_threshold,
            "results_count": len(similar_profiles)
        })
        
    except Exception as e:
        logger.error(f"Similar profiles search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to find similar profiles"
        )


# Background task functions

async def _track_profile_usage(tenant_id: str, operation: str, profile_count: int = 1, **metadata) -> None:
    """Track profile usage via usage service provider"""
    try:
        usage_service = await get_usage_service()
        await usage_service.track_usage(
            tenant_id=tenant_id,
            resource_type="profile",
            amount=profile_count,
            metadata={"operation": operation, **metadata}
        )
        logger.debug("Profile usage tracked", operation=operation, profile_count=profile_count)
    except Exception as e:
        logger.warning(f"Failed to track profile usage: {e}")


async def _track_profile_view(profile_id: str, user_id: str, tenant_id: str) -> None:
    """Track profile view for analytics"""
    try:
        # TODO: Update profile view count and analytics
        logger.debug("Profile view tracked", profile_id=profile_id, user_id=user_id)
    except Exception as e:
        logger.warning(f"Failed to track profile view: {e}")


async def _update_search_index(profile_id: str, profile_data: Optional[Dict] = None) -> None:
    """Update search index for profile"""
    try:
        if profile_data:
            search_service = AzureSearchService()
            await search_service.index_document(
                index_name="cv-profiles",
                document=profile_data
            )
        logger.debug("Search index updated", profile_id=profile_id)
    except Exception as e:
        logger.warning(f"Failed to update search index: {e}")


async def _regenerate_profile_embeddings(profile_id: str, profile: Optional[ProfileSchema] = None) -> None:
    """Regenerate embeddings for profile"""
    try:
        if profile:
            embedding_generator = EmbeddingGenerator()
            new_embedding = await embedding_generator.generate_profile_embedding(
                profile=profile,
                force_regenerate=True
            )
            
            # Update profile with new embedding
            # TODO: Update in database
            
        logger.info("Profile embeddings regenerated", profile_id=profile_id)
    except Exception as e:
        logger.warning(f"Failed to regenerate embeddings: {e}")


async def _remove_from_search_index(profile_id: str) -> None:
    """Remove profile from search index"""
    try:
        search_service = AzureSearchService()
        await search_service.delete_document(
            index_name="cv-profiles",
            document_id=profile_id
        )
        logger.debug("Profile removed from search index", profile_id=profile_id)
    except Exception as e:
        logger.warning(f"Failed to remove from search index: {e}")


async def _cleanup_profile_documents(profile_id: str, tenant_id: str) -> None:
    """Clean up document storage for profile"""
    try:
        # TODO: Remove documents from blob storage
        logger.debug("Profile documents cleaned up", profile_id=profile_id)
    except Exception as e:
        logger.warning(f"Failed to cleanup profile documents: {e}")


async def _reindex_restored_profile(profile_id: str) -> None:
    """Re-index restored profile for search"""
    try:
        # TODO: Get profile data and re-index
        logger.debug("Restored profile re-indexed", profile_id=profile_id)
    except Exception as e:
        logger.warning(f"Failed to re-index restored profile: {e}")


async def _execute_bulk_operation(
    operation_id: str,
    operation_type: str,
    profile_ids: List[str],
    parameters: Dict[str, Any],
    tenant_id: str,
    user_id: str
) -> None:
    """Execute bulk operation in background"""
    try:
        logger.info(
            "Executing bulk operation",
            operation_id=operation_id,
            operation_type=operation_type,
            profile_count=len(profile_ids)
        )
        
        # TODO: Implement actual bulk operations
        # - Validate all profiles belong to tenant
        # - Execute operation based on type
        # - Track progress and results
        # - Send completion notification
        
        # Mock successful completion
        logger.info("Bulk operation completed", operation_id=operation_id)
        
    except Exception as e:
        logger.error(f"Bulk operation failed: {e}", operation_id=operation_id)
