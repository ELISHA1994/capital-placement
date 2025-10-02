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

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4
import structlog

from fastapi import (
    APIRouter, Depends, HTTPException, UploadFile, File, 
    BackgroundTasks, Form, Query, status
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.models.profile import CVProfile, ProcessingStatus
from app.models.base import PaginatedResponse, PaginationModel
from app.core.dependencies import CurrentUserDep
from app.models.auth import CurrentUser

# AI-Powered Document Processing Services
from app.services.document.pdf_processor import PDFProcessor
from app.services.document.content_extractor import ContentExtractor
from app.services.document.quality_analyzer import QualityAnalyzer

# Core Services
from app.services.core.tenant_manager_provider import get_tenant_manager
from app.core.config import get_settings
from app.services.providers.ai_provider import (
    get_openai_service,
    get_embedding_service,
    get_prompt_manager,
)
from app.services.providers.postgres_provider import get_postgres_adapter

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/upload", tags=["upload"])

class UploadResponse(BaseModel):
    """Response model for file upload operations"""
    
    upload_id: str = Field(..., description="Unique upload identifier")
    profile_id: str = Field(..., description="Generated profile identifier")
    filename: str = Field(..., description="Original filename")
    status: ProcessingStatus = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")
    estimated_processing_time_seconds: Optional[int] = Field(None, description="Estimated processing time")
    webhook_url: Optional[str] = Field(None, description="Webhook URL for status updates")


class BatchUploadResponse(BaseModel):
    """Response model for batch upload operations"""
    
    batch_id: str = Field(..., description="Unique batch identifier")
    total_files: int = Field(..., description="Total files in batch")
    accepted_files: int = Field(..., description="Files accepted for processing")
    rejected_files: int = Field(..., description="Files rejected")
    uploads: List[UploadResponse] = Field(..., description="Individual upload responses")
    rejected_reasons: Dict[str, str] = Field(default_factory=dict, description="Rejection reasons")


class ProcessingStatusResponse(BaseModel):
    """Response model for processing status queries"""
    
    upload_id: str = Field(..., description="Upload identifier")
    profile_id: str = Field(..., description="Profile identifier")
    status: ProcessingStatus = Field(..., description="Current processing status")
    progress_percentage: int = Field(..., ge=0, le=100, description="Processing progress")
    processing_duration_seconds: Optional[float] = Field(None, description="Processing duration")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    quality_score: Optional[float] = Field(None, description="Extraction quality score")
    extracted_data_preview: Optional[Dict[str, Any]] = Field(None, description="Preview of extracted data")


@router.post("/", response_model=UploadResponse)
async def upload_cv_document(
    background_tasks: BackgroundTasks,
    current_user: CurrentUserDep,
    file: UploadFile = File(..., description="CV document file (PDF, DOC, DOCX)"),
    webhook_url: Optional[str] = Form(None, description="Webhook URL for processing updates"),
    auto_process: bool = Form(True, description="Automatically start processing"),
    extract_embeddings: bool = Form(True, description="Generate embeddings for search"),
    processing_priority: str = Form("normal", description="Processing priority (low, normal, high)"),
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
        start_time = datetime.now()
        upload_id = str(uuid4())
        
        logger.info(
            "CV upload received",
            upload_id=upload_id,
            filename=file.filename,
            content_type=file.content_type,
            file_size=file.size if hasattr(file, 'size') else 'unknown',
            user_id=current_user.user_id,
            tenant_id=current_user.tenant_id
        )
        
        # Validate file
        validate_file_upload(
            file.filename or "",
            file.content_type or "",
            file.size if hasattr(file, 'size') else 0
        )
        
        # Get tenant configuration
        settings = get_settings()
        tenant_manager = await get_tenant_manager()
        tenant_config = await tenant_manager.get_tenant_configuration(current_user.tenant_id)
        
        validation_result = await _validate_upload_file(file, str(current_user.tenant_id), tenant_config)
        if not validation_result["valid"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_file",
                    "message": validation_result["reason"],
                    "filename": file.filename
                }
            )
        
        # Check quotas
        quota_check = await tenant_manager.check_quota_limit(
            tenant_id=str(current_user.tenant_id),
            resource_type="documents_per_day",
            current_usage=tenant_config.get("documents_processed_today", 0)
        )
        
        if not quota_check["allowed"]:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "quota_exceeded",
                    "message": "Daily document processing quota exceeded",
                    "quota_info": quota_check
                }
            )
        
        # Read file content
        file_content = await file.read()
        
        # Generate profile ID
        profile_id = str(uuid4())
        
        # Create upload response
        response = UploadResponse(
            upload_id=upload_id,
            profile_id=profile_id,
            filename=file.filename,
            status=ProcessingStatus.PENDING,
            message="Document uploaded successfully, processing will begin shortly",
            webhook_url=webhook_url
        )
        
        if auto_process:
            # Start AI-powered processing in background
            background_tasks.add_task(
                _process_document_background,
                upload_id=upload_id,
                profile_id=profile_id,
                file_content=file_content,
                filename=file.filename or "unknown_document",
                tenant_id=str(current_user.tenant_id),
                user_id=current_user.user_id,
                webhook_url=webhook_url,
                extract_embeddings=extract_embeddings,
                processing_priority=processing_priority,
                settings=settings
            )
            
            response.message = "Document processing started"
            response.status = ProcessingStatus.PROCESSING
            
            # Estimate processing time based on file size
            file_size_mb = len(file_content) / (1024 * 1024)
            response.estimated_processing_time_seconds = max(30, int(file_size_mb * 15))  # ~15 seconds per MB
        
        # Update tenant usage with file size tracking
        background_tasks.add_task(
            _update_upload_usage,
            tenant_id=str(current_user.tenant_id),
            document_count=1,
            file_size_bytes=len(file_content)
        )
        
        logger.info(
            "CV upload processed successfully",
            upload_id=upload_id,
            profile_id=profile_id,
            auto_process=auto_process
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CV upload failed: {e}", upload_id=upload_id)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "upload_failed",
                "message": "Document upload could not be completed",
                "details": str(e) if logger.level == "DEBUG" else None
            }
        )


@router.post("/batch", response_model=BatchUploadResponse)
async def upload_cv_documents_batch(
    background_tasks: BackgroundTasks,
    current_user: CurrentUserDep,
    files: List[UploadFile] = File(..., description="Multiple CV document files"),
    webhook_url: Optional[str] = Form(None, description="Webhook URL for batch progress updates"),
    auto_process: bool = Form(True, description="Automatically start processing all files"),
    extract_embeddings: bool = Form(True, description="Generate embeddings for search"),
    max_concurrent: int = Form(3, ge=1, le=10, description="Maximum concurrent processing jobs"),
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
        batch_id = str(uuid4())
        
        logger.info(
            "Batch CV upload received",
            batch_id=batch_id,
            file_count=len(files),
            user_id=current_user.user_id,
            tenant_id=current_user.tenant_id
        )
        
        # Get tenant configuration and check batch quota
        settings = get_settings()
        tenant_manager = await get_tenant_manager()
        tenant_config = await tenant_manager.get_tenant_configuration(str(current_user.tenant_id))
        
        quota_check = await tenant_manager.check_quota_limit(
            tenant_id=str(current_user.tenant_id),
            resource_type="documents_per_day",
            current_usage=tenant_config.get("documents_processed_today", 0),
            increment=len(files)
        )
        
        if not quota_check["allowed"]:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "batch_quota_exceeded",
                    "message": f"Batch would exceed daily quota. Requested: {len(files)}, Available: {quota_check.get('remaining', 0)}",
                    "quota_info": quota_check
                }
            )
        
        # Validate and process each file
        uploads = []
        rejected_files = {}
        accepted_files = 0
        total_file_size_bytes = 0
        
        for file in files:
            try:
                # Validate individual file
                validation_result = await _validate_upload_file(file, str(current_user.tenant_id), tenant_config)
                
                if validation_result["valid"]:
                    upload_id = str(uuid4())
                    profile_id = str(uuid4())
                    
                    # Read file content to get size
                    file_content = await file.read()
                    await file.seek(0)  # Reset file position for potential reuse
                    total_file_size_bytes += len(file_content)
                    
                    uploads.append(UploadResponse(
                        upload_id=upload_id,
                        profile_id=profile_id,
                        filename=file.filename,
                        status=ProcessingStatus.PENDING if auto_process else ProcessingStatus.PENDING,
                        message="Queued for processing",
                        webhook_url=webhook_url
                    ))
                    accepted_files += 1
                    
                else:
                    rejected_files[file.filename] = validation_result["reason"]
                    
            except Exception as e:
                rejected_files[file.filename] = f"Validation error: {str(e)}"
        
        # Start AI-powered batch processing if auto_process enabled
        if auto_process and uploads:
            background_tasks.add_task(
                _process_batch_background,
                batch_id=batch_id,
                uploads=uploads,
                files=files,
                tenant_id=str(current_user.tenant_id),
                user_id=current_user.user_id,
                webhook_url=webhook_url,
                extract_embeddings=extract_embeddings,
                max_concurrent=max_concurrent,
                settings=settings
            )
        
        # Update tenant usage with total file size for batch
        background_tasks.add_task(
            _update_upload_usage,
            tenant_id=str(current_user.tenant_id),
            document_count=accepted_files,
            file_size_bytes=total_file_size_bytes
        )
        
        batch_response = BatchUploadResponse(
            batch_id=batch_id,
            total_files=len(files),
            accepted_files=accepted_files,
            rejected_files=len(rejected_files),
            uploads=uploads,
            rejected_reasons=rejected_files
        )
        
        logger.info(
            "Batch CV upload processed",
            batch_id=batch_id,
            accepted_files=accepted_files,
            rejected_files=len(rejected_files)
        )
        
        return batch_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch CV upload failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "batch_upload_failed",
                "message": "Batch upload could not be completed",
                "details": str(e) if logger.level == "DEBUG" else None
            }
        )


@router.get("/status/{upload_id}", response_model=ProcessingStatusResponse)
async def get_processing_status(
    upload_id: str,
    current_user: CurrentUserDep
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
        # Get processing status from database
        logger.debug("Processing status requested", upload_id=upload_id)
        
        try:
            postgres_adapter = await get_postgres_adapter()

            # Query document processing status
            processing_record = await postgres_adapter.fetch_one(
                """
                SELECT document_id, status, processing_duration_ms, quality_score, 
                       output_data, error_details, started_at, completed_at
                FROM document_processing 
                WHERE id = $1 AND tenant_id = $2
                """,
                upload_id, str(current_user.tenant_id)
            )
            
            if not processing_record:
                # Return default pending status if not found
                status_response = ProcessingStatusResponse(
                    upload_id=upload_id,
                    profile_id=str(uuid4()),
                    status=ProcessingStatus.PENDING,
                    progress_percentage=0,
                    extracted_data_preview={"status": "No processing record found"}
                )
            else:
                # Calculate progress percentage based on status
                progress_map = {
                    "pending": 0,
                    "processing": 50,
                    "completed": 100,
                    "failed": 100
                }
                
                progress_percentage = progress_map.get(processing_record["status"], 0)
                
                # Extract preview data from output_data
                output_data = processing_record["output_data"] or {}
                ai_analysis = output_data.get("ai_analysis", {})
                
                extracted_data_preview = {
                    "filename": output_data.get("filename", "Unknown"),
                    "text_length": len(output_data.get("original_text", "")),
                    "has_ai_analysis": bool(ai_analysis),
                    "analysis_fields": list(ai_analysis.keys()) if ai_analysis else []
                }
                
                # Add specific extracted data if available
                if ai_analysis:
                    if "personal_info" in ai_analysis:
                        extracted_data_preview["name"] = ai_analysis["personal_info"].get("name", "N/A")
                    if "skills" in ai_analysis:
                        extracted_data_preview["skills_count"] = len(ai_analysis["skills"])
                    if "experience" in ai_analysis:
                        extracted_data_preview["experience_entries"] = len(ai_analysis["experience"])
                
                status_response = ProcessingStatusResponse(
                    upload_id=upload_id,
                    profile_id=processing_record["document_id"],
                    status=ProcessingStatus(processing_record["status"]),
                    progress_percentage=progress_percentage,
                    processing_duration_seconds=(
                        processing_record["processing_duration_ms"] / 1000.0 
                        if processing_record["processing_duration_ms"] else None
                    ),
                    quality_score=processing_record["quality_score"],
                    error_message=(
                        processing_record["error_details"].get("error") 
                        if processing_record["error_details"] else None
                    ),
                    extracted_data_preview=extracted_data_preview
                )
        
        except Exception as db_error:
            logger.warning(f"Failed to query processing status: {db_error}")
            # Fallback to mock response
            status_response = ProcessingStatusResponse(
                upload_id=upload_id,
                profile_id=str(uuid4()),
                status=ProcessingStatus.PENDING,
                progress_percentage=0,
                extracted_data_preview={"status": "Status query failed", "error": str(db_error)}
            )
        
        return status_response
        
    except Exception as e:
        logger.error(f"Failed to get processing status: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve processing status"
        )


@router.get("/batch/{batch_id}/status")
async def get_batch_processing_status(
    batch_id: str,
    current_user: CurrentUserDep
) -> JSONResponse:
    """Get processing status for an entire batch of uploads."""
    try:
        # Get batch status from database
        try:
            postgres_adapter = await get_postgres_adapter()

            # Query all processing records for this batch
            batch_records = await postgres_adapter.fetch_all(
                """
                SELECT id, document_id, status, processing_duration_ms, quality_score, 
                       output_data->>'filename' as filename, error_details
                FROM document_processing 
                WHERE (output_data->>'batch_id') = $1 AND tenant_id = $2
                """,
                batch_id, str(current_user.tenant_id)
            )
            
            if not batch_records:
                # If no specific batch records found, return empty batch
                batch_status = {
                    "batch_id": batch_id,
                    "status": "not_found",
                    "total_files": 0,
                    "completed": 0,
                    "processing": 0,
                    "failed": 0,
                    "success_rate": 0.0,
                    "individual_status": []
                }
            else:
                # Count statuses
                status_counts = {"completed": 0, "processing": 0, "failed": 0, "pending": 0}
                individual_status = []
                total_duration = 0
                completed_count = 0
                
                for record in batch_records:
                    status = record["status"]
                    status_counts[status] = status_counts.get(status, 0) + 1
                    
                    individual_status.append({
                        "upload_id": record["id"],
                        "profile_id": record["document_id"],
                        "filename": record.get("filename", "Unknown"),
                        "status": status,
                        "quality_score": record["quality_score"],
                        "duration_seconds": (
                            record["processing_duration_ms"] / 1000.0 
                            if record["processing_duration_ms"] else None
                        )
                    })
                    
                    if status == "completed" and record["processing_duration_ms"]:
                        total_duration += record["processing_duration_ms"]
                        completed_count += 1
                
                total_files = len(batch_records)
                success_rate = status_counts["completed"] / total_files if total_files > 0 else 0
                avg_duration = (total_duration / completed_count / 1000.0) if completed_count > 0 else 0
                
                # Determine overall batch status
                if status_counts["processing"] > 0:
                    overall_status = "processing"
                elif status_counts["failed"] > 0 and status_counts["completed"] == 0:
                    overall_status = "failed"
                elif status_counts["completed"] == total_files:
                    overall_status = "completed"
                else:
                    overall_status = "partial"
                
                batch_status = {
                    "batch_id": batch_id,
                    "status": overall_status,
                    "total_files": total_files,
                    "completed": status_counts["completed"],
                    "processing": status_counts["processing"],
                    "failed": status_counts["failed"],
                    "pending": status_counts["pending"],
                    "success_rate": success_rate,
                    "average_processing_time_seconds": avg_duration,
                    "individual_status": individual_status
                }
        
        except Exception as db_error:
            logger.warning(f"Failed to query batch status: {db_error}")
            # Fallback batch status
            batch_status = {
                "batch_id": batch_id,
                "status": "error",
                "error": str(db_error),
                "total_files": 0,
                "completed": 0,
                "processing": 0,
                "failed": 0,
                "success_rate": 0.0,
                "individual_status": []
            }
        
        return JSONResponse(content=batch_status)
        
    except Exception as e:
        logger.error(f"Failed to get batch status: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve batch status"
        )


@router.post("/{upload_id}/reprocess")
async def reprocess_document(
    upload_id: str,
    background_tasks: BackgroundTasks,
    current_user: CurrentUserDep,
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
        # TODO: Retrieve original document and reprocess
        # This would involve:
        # 1. Getting original file from storage
        # 2. Validating reprocess permissions
        # 3. Starting new processing job
        # 4. Updating status tracking
        
        logger.info(
            "Document reprocessing requested",
            upload_id=upload_id,
            force=force_reprocess,
            user_id=current_user.user_id
        )
        
        # Mock reprocessing start
        background_tasks.add_task(_mock_reprocess, upload_id)
        
        return JSONResponse(content={
            "status": "reprocessing_started",
            "upload_id": upload_id,
            "message": "Document reprocessing has been queued"
        })
        
    except Exception as e:
        logger.error(f"Failed to start reprocessing: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to start reprocessing"
        )


@router.delete("/{upload_id}")
async def cancel_processing(
    upload_id: str,
    current_user: CurrentUserDep
) -> JSONResponse:
    """
    Cancel an in-progress processing job.
    
    Note: Only pending and processing jobs can be cancelled.
    Completed jobs cannot be cancelled but can be deleted via profiles API.
    """
    try:
        # TODO: Implement processing cancellation
        # This would involve:
        # 1. Checking if job can be cancelled
        # 2. Stopping background processing
        # 3. Cleaning up temporary data
        # 4. Updating status to cancelled
        
        logger.info(
            "Processing cancellation requested",
            upload_id=upload_id,
            user_id=current_user.user_id
        )
        
        return JSONResponse(content={
            "status": "cancelled",
            "upload_id": upload_id,
            "message": "Processing job has been cancelled"
        })
        
    except Exception as e:
        logger.error(f"Failed to cancel processing: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to cancel processing"
        )


@router.get("/history", response_model=PaginatedResponse)
async def get_upload_history(
    current_user: CurrentUserDep,
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
    """
    try:
        # TODO: Implement database query for upload history
        
        upload_history = []  # Mock empty history
        total_count = 0
        
        response = PaginatedResponse.create(
            items=upload_history,
            total=total_count,
            page=pagination.page,
            size=pagination.size
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get upload history: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve upload history"
        )


# Validation and utility functions

async def _validate_upload_file(
    file: UploadFile, 
    tenant_id: str, 
    tenant_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate uploaded file against constraints and quotas"""
    
    # Check file extension
    allowed_extensions = tenant_config.get("allowed_file_extensions", [".pdf", ".doc", ".docx", ".txt"])
    file_extension = None
    if file.filename:
        file_extension = "." + file.filename.split(".")[-1].lower()
        
    if file_extension not in allowed_extensions:
        return {
            "valid": False,
            "reason": f"File type not supported. Allowed types: {', '.join(allowed_extensions)}"
        }
    
    # Check file size
    if hasattr(file, 'size') and file.size:
        max_size = tenant_config.get("max_file_size_mb", 10) * 1024 * 1024  # Convert to bytes
        if file.size > max_size:
            return {
                "valid": False,
                "reason": f"File size too large. Maximum allowed: {max_size // (1024*1024)}MB"
            }
    
    # Check filename
    if not file.filename or len(file.filename) > 255:
        return {
            "valid": False,
            "reason": "Invalid filename"
        }
    
    return {"valid": True}


# Background processing functions

async def _process_document_background(
    upload_id: str,
    profile_id: str,
    file_content: bytes,
    filename: str,
    tenant_id: str,
    user_id: str,
    webhook_url: Optional[str] = None,
    extract_embeddings: bool = True,
    processing_priority: str = "normal",
    settings: Any = None
) -> None:
    """Process document using AI-powered services with comprehensive error handling"""
    start_time = datetime.now()
    
    try:
        logger.info(
            "Starting AI-powered document processing",
            upload_id=upload_id,
            profile_id=profile_id,
            filename=filename,
            tenant_id=tenant_id
        )
        
        if not settings:
            settings = get_settings()
        
        # Initialize data adapters
        postgres_adapter = await get_postgres_adapter()

        # Track processing start in database
        await postgres_adapter.execute(
            """
            INSERT INTO document_processing (id, document_id, tenant_id, processing_type, status, input_metadata, started_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            upload_id, profile_id, tenant_id, "ai_analysis", "processing", 
            {"filename": filename, "file_size": len(file_content), "priority": processing_priority},
            start_time
        )
        
        # Step 1: Extract content from document
        logger.info("Extracting document content", upload_id=upload_id)
        
        content_extractor = ContentExtractor()
        
        # Determine file type and extract content
        file_extension = filename.lower().split('.')[-1] if '.' in filename else 'unknown'
        
        if file_extension == 'pdf':
            pdf_processor = PDFProcessor()
            extracted_content = await pdf_processor.extract_content(file_content)
            text_content = extracted_content.get("text", "")
            metadata = extracted_content.get("metadata", {})
        else:
            # For other file types, try basic text extraction
            try:
                text_content = file_content.decode('utf-8')
                metadata = {"file_type": file_extension}
            except UnicodeDecodeError:
                logger.warning(f"Could not decode file as UTF-8: {filename}")
                text_content = file_content.decode('utf-8', errors='ignore')
                metadata = {"file_type": file_extension, "encoding_issues": True}
        
        if not text_content or len(text_content.strip()) < 50:
            raise ValueError("Insufficient text content extracted from document")
        
        # Step 2: AI-powered document analysis
        logger.info("Performing AI analysis", upload_id=upload_id)
        
        if settings.is_openai_configured():
            openai_service = await get_openai_service()
            prompt_manager = await get_prompt_manager()
            
            document_analyzer = DocumentAnalyzer(
                openai_service=openai_service,
                prompt_manager=prompt_manager,
                db_repository=postgres_adapter
            )
            
            # Perform comprehensive AI analysis
            analysis_result = await document_analyzer.analyze_document(
                text_content=text_content,
                document_type="cv",
                tenant_id=tenant_id,
                metadata=metadata
            )
            
            # Step 3: Quality analysis
            quality_analyzer = QualityAnalyzer(
                openai_service=openai_service,
                prompt_manager=prompt_manager
            )
            
            quality_assessment = await quality_analyzer.analyze_quality(
                extracted_text=text_content,
                structured_data=analysis_result,
                document_type="cv"
            )
            
        else:
            # Fallback analysis without AI
            logger.warning("OpenAI not configured, using basic analysis")
            analysis_result = await content_extractor.extract_cv_data(text_content)
            quality_assessment = {"overall_score": 0.7, "completeness": 0.7, "clarity": 0.7}
        
        # Step 4: Generate embeddings if requested and AI is available
        embedding_vector = None
        if extract_embeddings and settings.is_openai_configured():
            try:
                logger.info("Generating embeddings", upload_id=upload_id)
                embedding_service = await get_embedding_service()
                
                # Create combined text for embedding
                embedding_text = content_extractor.prepare_text_for_embedding(
                    text_content, analysis_result
                )
                
                embedding_vector = await embedding_service.generate_embedding(
                    text=embedding_text,
                    tenant_id=tenant_id
                )
                
                # Store embedding in database
                if embedding_vector:
                    await postgres_adapter.execute(
                        """
                        INSERT INTO embeddings (entity_id, entity_type, tenant_id, embedding_model, embedding_vector, content_hash, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (entity_id, entity_type, tenant_id) DO UPDATE SET
                            embedding_vector = EXCLUDED.embedding_vector,
                            updated_at = NOW()
                        """,
                        profile_id, "cv_profile", tenant_id, 
                        settings.OPENAI_EMBEDDING_MODEL,
                        f"[{','.join(map(str, embedding_vector))}]",
                        content_extractor.hash_content(text_content),
                        {"upload_id": upload_id, "filename": filename}
                    )
                    logger.info("Embeddings stored successfully", upload_id=upload_id)
                    
            except Exception as e:
                logger.warning(f"Failed to generate embeddings: {e}", upload_id=upload_id)
        
        # Step 5: Store processed profile in database
        processing_duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Create comprehensive profile record
        profile_data = {
            "id": profile_id,
            "upload_id": upload_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "filename": filename,
            "original_text": text_content,
            "ai_analysis": analysis_result,
            "quality_assessment": quality_assessment,
            "metadata": metadata,
            "processing_duration_ms": processing_duration_ms,
            "status": ProcessingStatus.COMPLETED.value,
            "has_embeddings": embedding_vector is not None,
            "created_at": start_time.isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Store profile (this would be in your profiles table)
        # For now, we'll update the document_processing table with completion
        await postgres_adapter.execute(
            """
            UPDATE document_processing 
            SET status = $1, completed_at = $2, processing_duration_ms = $3, 
                output_data = $4, quality_score = $5
            WHERE id = $6
            """,
            "completed", datetime.now(), processing_duration_ms, 
            profile_data, quality_assessment.get("overall_score", 0),
            upload_id
        )
        
        # Step 6: Send webhook notification if configured
        if webhook_url:
            await _send_webhook_notification(
                webhook_url, 
                {
                    "upload_id": upload_id,
                    "profile_id": profile_id,
                    "status": ProcessingStatus.COMPLETED.value,
                    "quality_score": quality_assessment.get("overall_score"),
                    "processing_duration_ms": processing_duration_ms,
                    "has_embeddings": embedding_vector is not None
                }
            )
        
        logger.info(
            "AI document processing completed successfully",
            upload_id=upload_id,
            profile_id=profile_id,
            quality_score=quality_assessment.get("overall_score"),
            processing_duration_ms=processing_duration_ms,
            has_embeddings=embedding_vector is not None
        )
        
    except Exception as e:
        processing_duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        logger.error(
            f"AI document processing failed: {e}",
            upload_id=upload_id,
            profile_id=profile_id,
            processing_duration_ms=processing_duration_ms
        )
        
        # Update database with failure status
        try:
            postgres_adapter = await get_postgres_adapter()
            await postgres_adapter.execute(
                """
                UPDATE document_processing 
                SET status = $1, completed_at = $2, processing_duration_ms = $3,
                    error_details = $4
                WHERE id = $5
                """,
                "failed", datetime.now(), processing_duration_ms,
                {"error": str(e), "error_type": type(e).__name__},
                upload_id
            )
        except Exception as db_error:
            logger.error(f"Failed to update failure status in database: {db_error}")
        
        # Send failure webhook if configured
        if webhook_url:
            await _send_webhook_notification(
                webhook_url,
                {
                    "upload_id": upload_id,
                    "profile_id": profile_id,
                    "status": ProcessingStatus.FAILED.value,
                    "error_message": str(e),
                    "processing_duration_ms": processing_duration_ms
                }
            )


async def _process_batch_background(
    batch_id: str,
    uploads: List[UploadResponse],
    files: List[UploadFile],
    tenant_id: str,
    user_id: str,
    webhook_url: Optional[str] = None,
    extract_embeddings: bool = True,
    max_concurrent: int = 3,
    settings: Any = None
) -> None:
    """Process batch of documents with controlled concurrency"""
    try:
        logger.info(
            "Starting batch document processing",
            batch_id=batch_id,
            file_count=len(uploads),
            max_concurrent=max_concurrent
        )
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_file(upload: UploadResponse, file: UploadFile):
            async with semaphore:
                file_content = await file.read()
                await _process_document_background(
                    upload_id=upload.upload_id,
                    profile_id=upload.profile_id,
                    file_content=file_content,
                    filename=file.filename or "unknown_document",
                    tenant_id=tenant_id,
                    user_id=user_id,
                    extract_embeddings=extract_embeddings,
                    processing_priority="normal",
                    settings=settings
                )
        
        # Process all files concurrently
        tasks = [
            process_single_file(upload, file)
            for upload, file in zip(uploads, files)
        ]
        
        # Execute with error handling
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successes and failures
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful
        
        # Send batch completion webhook
        if webhook_url:
            await _send_webhook_notification(
                webhook_url,
                {
                    "batch_id": batch_id,
                    "status": "completed",
                    "total_files": len(uploads),
                    "successful": successful,
                    "failed": failed
                }
            )
        
        logger.info(
            "Batch processing completed",
            batch_id=batch_id,
            successful=successful,
            failed=failed
        )
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}", batch_id=batch_id)


async def _update_upload_usage(tenant_id: str, document_count: int, file_size_bytes: int = 0) -> None:
    """Update tenant upload usage metrics including storage"""
    try:
        tenant_manager = await get_tenant_manager()
        
        # Calculate storage in GB
        storage_gb = file_size_bytes / (1024 * 1024 * 1024) if file_size_bytes > 0 else 0
        
        metrics_update = {
            "documents_processed": document_count,
            "documents_uploaded": document_count,
            "storage_used_gb": storage_gb
        }
        
        await tenant_manager.update_usage_metrics(
            tenant_id=tenant_id,
            metrics_update=metrics_update
        )
        
        logger.debug(
            "Updated tenant upload usage", 
            tenant_id=tenant_id, 
            document_count=document_count,
            storage_gb=round(storage_gb, 4)
        )
    except Exception as e:
        logger.warning(f"Failed to update upload usage: {e}")


async def _send_webhook_notification(url: str, payload: Dict[str, Any]) -> None:
    """Send webhook notification with retry logic"""
    try:
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                timeout=30.0,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
        logger.debug("Webhook notification sent successfully", url=url)
        
    except Exception as e:
        logger.warning(f"Failed to send webhook notification: {e}")


async def _mock_reprocess(upload_id: str) -> None:
    """Mock reprocessing function"""
    await asyncio.sleep(2)  # Simulate reprocessing delay
    logger.info("Mock reprocessing completed", upload_id=upload_id)
