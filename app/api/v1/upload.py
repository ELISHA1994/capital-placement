"""
Upload API Endpoints

Document upload and processing endpoints with:
- Multiple file format support (PDF, DOC, DOCX)
- Batch upload capabilities
- Real-time processing status tracking
- Error handling and validation
- Progress monitoring and webhooks
- Multi-tenant file isolation
"""

from datetime import datetime
from typing import List, Optional
import structlog

from fastapi import (
    APIRouter, Depends, HTTPException, UploadFile, File,
    BackgroundTasks, Form, Query, status,
)
from fastapi.responses import JSONResponse

from app.application.upload_service import UploadError
from app.domain.entities.profile import ProcessingStatus
from app.api.schemas.base import PaginatedResponse
from app.infrastructure.persistence.models.base import PaginationModel
from app.core.dependencies import CurrentUserDep
from app.infrastructure.persistence.models.auth_tables import CurrentUser
from app.api.schemas.upload_schemas import (
    UploadResponse,
    BatchUploadResponse,
    ProcessingStatusResponse,
)
from app.api.dependencies import UploadServiceDep, map_domain_exception_to_http
from app.api.rate_limit import UploadRateLimitDep
from app.domain.exceptions import DomainException
from app.core.config import get_settings

# Core Services
logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/upload", tags=["upload"])
settings = get_settings()



@router.post("/", response_model=UploadResponse)
async def upload_cv_document(
    background_tasks: BackgroundTasks,
    current_user: CurrentUserDep,
    upload_service: UploadServiceDep,
    file: UploadFile = File(..., description="CV document file (PDF, DOC, DOCX)"),
    webhook_url: Optional[str] = Form(None, description="Webhook URL for processing updates"),
    auto_process: bool = Form(True, description="Automatically start processing"),
    extract_embeddings: bool = Form(True, description="Generate embeddings for search"),
    processing_priority: str = Form("normal", description="Processing priority (low, normal, high)"),
    _rate_limit_check: None = UploadRateLimitDep,
) -> UploadResponse:
    """
    Upload a single CV document for processing.
    
    Supports multiple file formats:
    - **PDF**: Preferred format with best extraction accuracy
    - **DOC/DOCX**: Microsoft Word documents
    - **TXT**: Plain text files
    
    Processing includes:
    - Advanced PDF parsing with layout analysis
    - Intelligent CV structure extraction
    - Skills, experience, and education parsing
    - Text embedding generation for search
    - Quality assessment and validation
    """
    try:
        return await upload_service.upload_document(
            file=file,
            current_user=current_user,
            schedule_task=background_tasks,
            webhook_url=webhook_url,
            auto_process=auto_process,
            extract_embeddings=extract_embeddings,
            processing_priority=processing_priority,
        )
    except DomainException as domain_exc:
        # Map domain exceptions to appropriate HTTP responses
        raise map_domain_exception_to_http(domain_exc)
    except UploadError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except Exception as exc:  # pragma: no cover - unexpected failures bubble as 500
        logger.error("CV upload failed", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "upload_failed",
                "message": "Document upload could not be completed",
                "details": str(exc) if settings.ENVIRONMENT in ("local", "development") else None,
            },
        )


@router.post("/batch", response_model=BatchUploadResponse)
async def upload_cv_documents_batch(
    background_tasks: BackgroundTasks,
    current_user: CurrentUserDep,
    upload_service: UploadServiceDep,
    files: List[UploadFile] = File(..., description="Multiple CV document files"),
    webhook_url: Optional[str] = Form(None, description="Webhook URL for batch progress updates"),
    auto_process: bool = Form(True, description="Automatically start processing all files"),
    extract_embeddings: bool = Form(True, description="Generate embeddings for search"),
    max_concurrent: int = Form(3, ge=1, le=10, description="Maximum concurrent processing jobs"),
    _rate_limit_check: None = UploadRateLimitDep,
) -> BatchUploadResponse:
    """
    Upload multiple CV documents in batch for efficient processing.
    
    Features:
    - Concurrent processing with configurable limits
    - Individual file validation and error handling
    - Progress tracking and webhook notifications
    - Automatic retry for transient failures
    - Batch-level analytics and reporting
    """
    try:
        return await upload_service.upload_documents_batch(
            files=files,
            current_user=current_user,
            schedule_task=background_tasks,
            webhook_url=webhook_url,
            auto_process=auto_process,
            extract_embeddings=extract_embeddings,
            max_concurrent=max_concurrent,
        )
    except DomainException as domain_exc:
        # Map domain exceptions to appropriate HTTP responses
        raise map_domain_exception_to_http(domain_exc)
    except UploadError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except Exception as exc:  # pragma: no cover
        logger.error("Batch CV upload failed", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "batch_upload_failed",
                "message": "Batch upload could not be completed",
                "details": str(exc) if settings.ENVIRONMENT in ("local", "development") else None,
            },
        )


@router.get("/status/{upload_id}", response_model=ProcessingStatusResponse)
async def get_processing_status(
    upload_id: str,
    current_user: CurrentUserDep,
    upload_service: UploadServiceDep
) -> ProcessingStatusResponse:
    """
    Get detailed processing status for an uploaded document.
    
    Returns:
    - Current processing stage and progress
    - Quality assessment scores
    - Extracted data preview
    - Error details if processing failed
    - Performance metrics
    """
    try:
        return await upload_service.get_processing_status(
            upload_id=upload_id,
            tenant_id=str(current_user.tenant_id),
            user_id=current_user.user_id,
        )
    except DomainException as domain_exc:
        # Map domain exceptions to appropriate HTTP responses
        raise map_domain_exception_to_http(domain_exc)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to get processing status", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve processing status",
        )


@router.get("/batch/{batch_id}/status")
async def get_batch_processing_status(
    batch_id: str,
    current_user: CurrentUserDep,
    upload_service: UploadServiceDep
) -> JSONResponse:
    """Get processing status for an entire batch of uploads."""
    try:
        batch_status = await upload_service.get_batch_processing_status(
            batch_id=batch_id,
            tenant_id=str(current_user.tenant_id),
            user_id=current_user.user_id,
        )
        return JSONResponse(content=batch_status)
        
    except DomainException as domain_exc:
        # Map domain exceptions to appropriate HTTP responses
        raise map_domain_exception_to_http(domain_exc)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to get batch status", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve batch status",
        )


@router.post("/{upload_id}/reprocess")
async def reprocess_document(
    upload_id: str,
    background_tasks: BackgroundTasks,
    current_user: CurrentUserDep,
    upload_service: UploadServiceDep,
    force_reprocess: bool = Query(False, description="Force reprocessing even if already completed"),
    extract_embeddings: bool = Query(True, description="Generate new embeddings"),
) -> JSONResponse:
    """
    Reprocess a previously uploaded document.

    Useful for:
    - Retrying failed processing jobs
    - Applying improved extraction algorithms
    - Regenerating embeddings with updated models
    - Fixing processing errors
    """
    try:
        payload = await upload_service.reprocess_document(
            upload_id=upload_id,
            tenant_id=current_user.tenant_id,
            user_id=current_user.user_id,
            force_reprocess=force_reprocess,
            extract_embeddings=extract_embeddings,
            schedule_task=background_tasks,
        )
        return JSONResponse(content=payload)

    except DomainException as domain_exc:
        # Map domain exceptions to appropriate HTTP responses
        raise map_domain_exception_to_http(domain_exc)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to start reprocessing", error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to start reprocessing")


@router.delete("/{upload_id}")
async def cancel_processing(
    upload_id: str,
    current_user: CurrentUserDep,
    upload_service: UploadServiceDep
) -> JSONResponse:
    """
    Cancel an in-progress processing job.
    
    Note: Only pending and processing jobs can be cancelled.
    Completed jobs cannot be cancelled but can be deleted via profiles API.
    """
    try:
        payload = await upload_service.cancel_processing(
            upload_id=upload_id,
            user_id=current_user.user_id,
            tenant_id=str(current_user.tenant_id),
        )
        return JSONResponse(content=payload)
        
    except DomainException as domain_exc:
        # Map domain exceptions to appropriate HTTP responses
        raise map_domain_exception_to_http(domain_exc)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to cancel processing", error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to cancel processing")


@router.delete("/batch/{batch_id}")
async def cancel_batch_processing(
    batch_id: str,
    current_user: CurrentUserDep,
    upload_service: UploadServiceDep
) -> JSONResponse:
    """
    Cancel all processing jobs in a batch upload.
    
    This will attempt to cancel all individual upload processing tasks
    within the specified batch. Returns detailed results showing which
    tasks were successfully cancelled and which failed.
    
    Note: Only pending and processing jobs can be cancelled.
    """
    try:
        payload = await upload_service.cancel_batch_processing(
            batch_id=batch_id,
            user_id=current_user.user_id,
            tenant_id=str(current_user.tenant_id),
        )
        return JSONResponse(content=payload)
        
    except DomainException as domain_exc:
        # Map domain exceptions to appropriate HTTP responses
        raise map_domain_exception_to_http(domain_exc)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to cancel batch processing", error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to cancel batch processing")


@router.get("/history", response_model=PaginatedResponse)
async def get_upload_history(
    current_user: CurrentUserDep,
    upload_service: UploadServiceDep,
    pagination: PaginationModel = Depends(),
    status_filter: Optional[ProcessingStatus] = Query(None, description="Filter by processing status"),
    start_date: Optional[datetime] = Query(None, description="Filter uploads from this date"),
    end_date: Optional[datetime] = Query(None, description="Filter uploads until this date")
) -> PaginatedResponse:
    """
    Get upload and processing history for the current user/tenant.

    Provides comprehensive upload history with:
    - Processing status and duration
    - Quality scores and metrics
    - Error analysis and patterns
    - Usage analytics over time

    Returns paginated list of uploaded documents with their processing status.
    """
    try:
        # Application layer returns domain data (UploadHistoryResult)
        result = await upload_service.get_upload_history(
            tenant_id=str(current_user.tenant_id),
            pagination=pagination,
            status_filter=status_filter,
            start_date=start_date,
            end_date=end_date
        )

        # API layer wraps in PaginatedResponse for HTTP response
        return PaginatedResponse.create(
            items=result.items,
            total=result.total,
            page=result.page,
            size=result.size
        )
    except DomainException as domain_exc:
        # Map domain exceptions to appropriate HTTP responses
        raise map_domain_exception_to_http(domain_exc)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to retrieve upload history", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve upload history"
        )
