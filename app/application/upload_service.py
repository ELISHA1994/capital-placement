"""Application layer orchestrator for upload workflows following hexagonal architecture."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog
from fastapi import UploadFile
from sqlmodel import select

from app.core.config import get_settings
from app.database.sqlmodel_engine import get_sqlmodel_db_manager
from app.domain.entities.profile import ProcessingStatus
from app.api.schemas.upload_schemas import (
    BatchUploadResponse,
    ProcessingStatusResponse,
    UploadResponse,
)
from app.application.dependencies import UploadDependencies
from app.domain.entities.profile import Profile, ProfileStatus
from app.domain.value_objects import ProfileId, TenantId
from app.domain.utils import FileSizeValidator
from app.domain.exceptions import FileSizeExceededError, InvalidFileError
from app.domain.task_types import TaskType

# NOTE: CurrentUser is an API-layer DTO passed from FastAPI dependencies.
# Application layer accepts it as Any to avoid depending on infrastructure/API layers.
# AuditEventType, get_audit_service, and get_task_manager should be injected via dependencies.

logger = structlog.get_logger(__name__)


class UploadError(Exception):
    """Raised when upload workflows encounter a recoverable failure."""

    def __init__(self, status_code: int, detail: Any):
        super().__init__(str(detail))
        self.status_code = status_code
        self.detail = detail


@dataclass
class UploadHistoryResult:
    """Application layer result for upload history queries.

    This dataclass represents the result of upload history queries from the
    application layer, keeping the application layer free from HTTP concerns.
    The API layer wraps this in PaginatedResponse for HTTP responses.
    """
    items: List[Any]
    total: int
    page: int
    size: int


class UploadApplicationService:
    """Coordinates upload and document-processing workflows.
    
    This application service follows hexagonal architecture principles by:
    - Using dependency injection via constructor
    - Depending only on domain interfaces (ports)
    - Orchestrating workflow without implementing business logic
    - Maintaining separation between domain and infrastructure concerns
    """

    def __init__(self, dependencies: UploadDependencies) -> None:
        """Initialize with injected dependencies.

        Args:
            dependencies: All required services and repositories
        """
        self._deps = dependencies
        self._logger = structlog.get_logger(__name__)

    @property
    def deps(self):
        """Provide access to dependencies for consistency."""
        return self._deps

    async def upload_document(
        self,
        *,
        file: UploadFile,
        current_user: Any,
        schedule_task: Optional[Any] = None,
        webhook_url: Optional[str] = None,
        auto_process: bool = True,
        extract_embeddings: bool = True,
        processing_priority: str = "normal",
    ) -> UploadResponse:
        start_time = datetime.now()
        upload_id = str(uuid4())

        logger.info(
            "CV upload received",
            upload_id=upload_id,
            filename=file.filename,
            content_type=file.content_type,
            file_size=getattr(file, "size", "unknown"),
            user_id=current_user.user_id,
            tenant_id=current_user.tenant_id,
        )

        # Log file upload start event
        try:
            audit_service = self._deps.audit_service
            await audit_service.log_file_upload_event(
                event_type="audit_event_type_FILE_UPLOAD_STARTED",
                tenant_id=current_user.tenant_id,
                user_id=current_user.user_id,
                filename=file.filename or "unknown",
                file_size=getattr(file, "size", 0),
                upload_id=upload_id,
                ip_address=None,  # TODO: Extract from request context
                user_agent=None,  # TODO: Extract from request context
            )
        except Exception as e:
            logger.warning("Failed to log file upload start audit event", error=str(e))

        settings = get_settings()
        tenant_id = TenantId(current_user.tenant_id)
        
        # Get tenant configuration
        tenant_config = await self._deps.tenant_manager.get_tenant_configuration(str(current_user.tenant_id))

        # Validate webhook URL if provided
        if webhook_url:
            self._deps.webhook_validator.validate_webhook_url(webhook_url)

        # Perform comprehensive file content validation 
        # This includes MIME type, magic bytes, and security scanning
        validation_result = await self._deps.file_content_validator.validate_upload_file(
            file=file,
            tenant_config=tenant_config,
        )
        
        if not validation_result.is_valid:
            logger.warning(
                "File validation failed",
                upload_id=upload_id,
                filename=file.filename,
                validation_errors=validation_result.validation_errors,
                security_warnings=validation_result.security_warnings,
                confidence_score=validation_result.confidence_score
            )
            
            # Log file validation failure audit event
            try:
                audit_service = self._deps.audit_service
                await audit_service.log_file_upload_event(
                    event_type="audit_event_type_FILE_VALIDATION_FAILED",
                    tenant_id=current_user.tenant_id,
                    user_id=current_user.user_id,
                    filename=file.filename or "unknown",
                    file_size=getattr(file, "size", 0),
                    upload_id=upload_id,
                    validation_errors=validation_result.validation_errors,
                    security_warnings=validation_result.security_warnings,
                    error_message="File validation failed",
                    ip_address=None,  # TODO: Extract from request context
                    user_agent=None,  # TODO: Extract from request context
                )
            except Exception as e:
                logger.warning("Failed to log file validation failure audit event", error=str(e))
            
            # Construct detailed error message
            error_details = {
                "error": "invalid_file",
                "message": "File validation failed",
                "filename": file.filename,
                "validation_errors": validation_result.validation_errors,
                "details": validation_result.validation_details
            }
            
            # Add security warnings if present
            if validation_result.security_warnings:
                error_details["security_warnings"] = validation_result.security_warnings
            
            raise UploadError(status_code=400, detail=error_details)
        
        # Check if security warnings should cause rejection
        # Handle both dict and domain entity (TenantConfiguration)
        if hasattr(tenant_config, 'get'):
            # It's a dictionary
            processing_config = tenant_config.get("processing_configuration", {})
            file_validation_config = processing_config.get("file_validation", {})
            reject_on_warnings = file_validation_config.get("reject_on_security_warnings", False)
        else:
            # It's a domain entity - access attributes directly
            processing_config = tenant_config.processing_configuration
            file_validation_config = processing_config.file_validation
            reject_on_warnings = file_validation_config.reject_on_security_warnings

        if validation_result.security_warnings and reject_on_warnings:
            logger.warning(
                "File rejected due to security warnings",
                upload_id=upload_id,
                filename=file.filename,
                security_warnings=validation_result.security_warnings
            )
            raise UploadError(
                status_code=400,
                detail={
                    "error": "security_risk",
                    "message": "File contains potential security risks",
                    "filename": file.filename,
                    "security_warnings": validation_result.security_warnings,
                }
            )

        # Get validated file size for usage tracking
        validated_file_size = validation_result.file_size
        
        logger.info(
            "File validation completed successfully",
            upload_id=upload_id,
            filename=file.filename,
            validated_size=validated_file_size,
            size_mb=validated_file_size / (1024 * 1024)
        )

        # Check quota limits
        # Handle both dict and domain entity for usage tracking
        if hasattr(tenant_config, 'get'):
            current_docs_today = tenant_config.get("documents_processed_today", 0)
        else:
            # Domain entity - field doesn't exist in current schema, default to 0
            current_docs_today = 0

        quota_check = await self._deps.tenant_manager.check_quota_limit(
            tenant_id=str(current_user.tenant_id),
            resource_type="documents_per_day",
            current_usage=current_docs_today,
        )
        if not quota_check["allowed"]:
            raise UploadError(
                status_code=429,
                detail={
                    "error": "quota_exceeded",
                    "message": "Daily document processing quota exceeded",
                    "quota_info": quota_check,
                },
            )

        # File size has already been validated, safe to read content now
        # The file pointer was reset during validation
        logger.debug(
            "Reading validated file content",
            upload_id=upload_id,
            filename=file.filename,
            expected_size=validated_file_size
        )
        file_content = await file.read()
        profile_id = str(uuid4())

        # Track file content for automatic cleanup
        content_resource_id = await self._deps.file_resource_manager.track_file_content(
            content=file_content,
            filename=file.filename or "unknown_document",
            upload_id=upload_id,
            tenant_id=str(current_user.tenant_id),
            auto_cleanup_after=3600,  # Auto cleanup after 1 hour
        )

        if not content_resource_id:
            logger.warning("Failed to track file content for cleanup", upload_id=upload_id)
        else:
            logger.debug("File content tracked for cleanup",
                        upload_id=upload_id, resource_id=content_resource_id)

        # Save file to storage using IFileStorage
        storage_path = None
        try:
            storage_path = await self._deps.file_storage.save_file(
                tenant_id=str(current_user.tenant_id),
                upload_id=str(upload_id),
                filename=file.filename or "unknown_document",
                content=file_content
            )
            logger.info(
                "File saved to storage",
                upload_id=str(upload_id),
                storage_path=storage_path,
                file_size_bytes=len(file_content)
            )
        except Exception as storage_error:
            logger.error(
                "Failed to save file to storage",
                upload_id=str(upload_id),
                filename=file.filename,
                error=str(storage_error),
                error_type=type(storage_error).__name__
            )
            # Continue processing even if storage fails (graceful degradation)
            # The file content is still in memory and will be processed

        response = UploadResponse(
            upload_id=upload_id,
            profile_id=profile_id,
            filename=file.filename,
            status=ProcessingStatus.PENDING,
            message="Document uploaded successfully, processing will begin shortly",
            webhook_url=webhook_url,
        )

        if auto_process:
            await self._enqueue_background(
                schedule_task,
                self.process_document_background,
                upload_id,
                profile_id,
                file_content,
                file.filename or "unknown_document",
                str(current_user.tenant_id),
                current_user.user_id,
                webhook_url,
                extract_embeddings,
                processing_priority,
                settings,
            )

            response.message = "Document processing started"
            response.status = ProcessingStatus.PROCESSING
            file_size_mb = len(file_content) / (1024 * 1024)
            response.estimated_processing_time_seconds = max(30, int(file_size_mb * 15))

        await self._enqueue_background(
            schedule_task,
            self._update_upload_usage,
            str(current_user.tenant_id),
            1,
            len(file_content),
        )

        # Log successful file upload
        try:
            audit_service = self._deps.audit_service
            processing_duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            await audit_service.log_file_upload_event(
                event_type="audit_event_type_FILE_UPLOAD_SUCCESS",
                tenant_id=current_user.tenant_id,
                user_id=current_user.user_id,
                filename=file.filename or "unknown",
                file_size=len(file_content),
                upload_id=upload_id,
                processing_duration_ms=processing_duration_ms,
                ip_address=None,  # TODO: Extract from request context
                user_agent=None,  # TODO: Extract from request context
            )
        except Exception as e:
            logger.warning("Failed to log file upload success audit event", error=str(e))

        logger.info(
            "CV upload processed successfully",
            upload_id=upload_id,
            profile_id=profile_id,
            auto_process=auto_process,
        )

        return response

    async def upload_documents_batch(
        self,
        *,
        files: List[UploadFile],
        current_user: Any,
        schedule_task: Optional[Any] = None,
        webhook_url: Optional[str] = None,
        auto_process: bool = True,
        extract_embeddings: bool = True,
        max_concurrent: int = 3,
    ) -> BatchUploadResponse:
        batch_id = str(uuid4())

        logger.info(
            "Batch CV upload received",
            batch_id=batch_id,
            file_count=len(files),
            user_id=current_user.user_id,
            tenant_id=current_user.tenant_id,
        )

        settings = get_settings()
        tenant_config = await self._deps.tenant_manager.get_tenant_configuration(str(current_user.tenant_id))

        # Validate webhook URL if provided
        if webhook_url:
            self._deps.webhook_validator.validate_webhook_url(webhook_url)

        # Handle both dict and domain entity for usage tracking
        if hasattr(tenant_config, 'get'):
            current_docs_today = tenant_config.get("documents_processed_today", 0)
        else:
            # Domain entity - field doesn't exist in current schema, default to 0
            current_docs_today = 0

        quota_check = await self._deps.tenant_manager.check_quota_limit(
            tenant_id=str(current_user.tenant_id),
            resource_type="documents_per_day",
            current_usage=current_docs_today,
            increment=len(files),
        )

        if not quota_check["allowed"]:
            raise UploadError(
                status_code=429,
                detail={
                    "error": "batch_quota_exceeded",
                    "message": (
                        f"Batch would exceed daily quota. Requested: {len(files)}, "
                        f"Available: {quota_check.get('remaining', 0)}"
                    ),
                    "quota_info": quota_check,
                },
            )

        uploads: List[UploadResponse] = []
        rejected_files: Dict[str, str] = {}
        accepted_files = 0
        total_file_size_bytes = 0

        for upload_file in files:
            try:
                # Use comprehensive file content validation for batch uploads
                validation_result = await self._deps.file_content_validator.validate_upload_file(
                    file=upload_file,
                    tenant_config=tenant_config,
                )

                # Check if security warnings should cause rejection
                # Handle both dict and domain entity (TenantConfiguration)
                if hasattr(tenant_config, 'get'):
                    # It's a dictionary
                    processing_config = tenant_config.get("processing_configuration", {})
                    file_validation_config = processing_config.get("file_validation", {})
                    reject_on_warnings = file_validation_config.get("reject_on_security_warnings", False)
                else:
                    # It's a domain entity - access attributes directly
                    processing_config = tenant_config.processing_configuration
                    file_validation_config = processing_config.file_validation
                    reject_on_warnings = file_validation_config.reject_on_security_warnings

                should_reject_security = validation_result.security_warnings and reject_on_warnings

                if validation_result.is_valid and not should_reject_security:
                    upload_id = str(uuid4())
                    profile_id = str(uuid4())

                    # File size has already been validated via comprehensive validation
                    validated_file_size = validation_result.file_size
                    
                    logger.debug(
                        "Reading validated file content for batch processing",
                        upload_id=upload_id,
                        filename=upload_file.filename,
                        expected_size=validated_file_size,
                        validation_confidence=validation_result.confidence_score
                    )
                    
                    file_content = await upload_file.read()
                    await upload_file.seek(0)  # Reset for potential later processing
                    
                    # Track file content for automatic cleanup  
                    content_resource_id = await self._deps.file_resource_manager.track_file_content(
                        content=file_content,
                        filename=upload_file.filename or "unknown_document",
                        upload_id=upload_id,
                        tenant_id=str(current_user.tenant_id),
                        auto_cleanup_after=3600,  # Auto cleanup after 1 hour
                    )
                    
                    if not content_resource_id:
                        logger.warning("Failed to track batch file content for cleanup", 
                                     upload_id=upload_id, filename=upload_file.filename)
                    
                    # Use validated size for consistency and performance
                    total_file_size_bytes += validated_file_size

                    uploads.append(
                        UploadResponse(
                            upload_id=upload_id,
                            profile_id=profile_id,
                            filename=upload_file.filename,
                            status=ProcessingStatus.PENDING,
                            message="Queued for processing",
                            webhook_url=webhook_url,
                        )
                    )
                    accepted_files += 1
                else:
                    # Construct detailed rejection reason
                    if should_reject_security:
                        reason = f"Security risk detected: {', '.join(validation_result.security_warnings)}"
                    elif validation_result.validation_errors:
                        reason = f"Validation failed: {', '.join(validation_result.validation_errors)}"
                    else:
                        reason = "File validation failed for unknown reason"
                    
                    rejected_files[upload_file.filename] = reason
                    
                    logger.warning(
                        "File rejected in batch upload",
                        filename=upload_file.filename,
                        validation_errors=validation_result.validation_errors,
                        security_warnings=validation_result.security_warnings,
                        confidence_score=validation_result.confidence_score
                    )
            except Exception as exc:  # pragma: no cover - defensive
                rejected_files[upload_file.filename] = f"Validation error: {exc}"

        if auto_process and uploads:
            await self._enqueue_background(
                schedule_task,
                self.process_batch_background,
                batch_id,
                uploads,
                files,
                str(current_user.tenant_id),
                current_user.user_id,
                webhook_url,
                extract_embeddings,
                max_concurrent,
                settings,
            )

        await self._enqueue_background(
            schedule_task,
            self._update_upload_usage,
            str(current_user.tenant_id),
            accepted_files,
            total_file_size_bytes,
        )

        return BatchUploadResponse(
            batch_id=batch_id,
            total_files=len(files),
            accepted_files=accepted_files,
            rejected_files=len(rejected_files),
            uploads=uploads,
            rejected_reasons=rejected_files,
        )

    async def get_processing_status(
        self,
        *,
        upload_id: str,
        tenant_id: str,
        user_id: str,
    ) -> ProcessingStatusResponse:
        logger.debug("Processing status requested", upload_id=upload_id)

        try:
            processing_record = await self._deps.database_adapter.fetch_one(
                """
                SELECT document_id, status, processing_duration_ms, quality_score,
                       output_data, error_details, started_at, completed_at
                FROM document_processing
                WHERE id = $1 AND tenant_id = $2
                """,
                upload_id,
                tenant_id,
            )

            if not processing_record:
                return ProcessingStatusResponse(
                    upload_id=upload_id,
                    profile_id=str(uuid4()),
                    status=ProcessingStatus.PENDING,
                    progress_percentage=0,
                    extracted_data_preview={"status": "No processing record found"},
                )

            progress_map = {
                "pending": 0,
                "processing": 50,
                "completed": 100,
                "failed": 100,
            }

            progress_percentage = progress_map.get(processing_record["status"], 0)
            output_data = processing_record.get("output_data") or {}
            ai_analysis = output_data.get("ai_analysis", {})

            extracted_data_preview: Dict[str, Any] = {
                "filename": output_data.get("filename", "Unknown"),
                "text_length": len(output_data.get("original_text", "")),
                "has_ai_analysis": bool(ai_analysis),
                "analysis_fields": list(ai_analysis.keys()) if ai_analysis else [],
            }

            if ai_analysis:
                if "personal_info" in ai_analysis:
                    extracted_data_preview["name"] = ai_analysis["personal_info"].get("name", "N/A")
                if "skills" in ai_analysis:
                    extracted_data_preview["skills_count"] = len(ai_analysis["skills"])
                if "experience" in ai_analysis:
                    extracted_data_preview["experience_entries"] = len(ai_analysis["experience"])

            return ProcessingStatusResponse(
                upload_id=upload_id,
                profile_id=str(processing_record["document_id"]),
                status=ProcessingStatus(processing_record["status"]),
                progress_percentage=progress_percentage,
                processing_duration_seconds=(
                    (processing_record["processing_duration_ms"] or 0) / 1000.0
                    if processing_record["processing_duration_ms"]
                    else None
                ),
                quality_score=processing_record.get("quality_score"),
                error_message=(
                    (processing_record.get("error_details") or {}).get("error")
                    if processing_record.get("error_details")
                    else None
                ),
                extracted_data_preview=extracted_data_preview,
            )
        except Exception as exc:  # pragma: no cover - fallback behaviour
            logger.warning("Failed to query processing status", error=str(exc))
            return ProcessingStatusResponse(
                upload_id=upload_id,
                profile_id=str(uuid4()),
                status=ProcessingStatus.PENDING,
                progress_percentage=0,
                extracted_data_preview={
                    "status": "Status query failed",
                    "error": str(exc),
                },
            )

    async def get_batch_processing_status(
        self,
        *,
        batch_id: str,
        tenant_id: str,
        user_id: str,
    ) -> Dict[str, Any]:
        logger.debug("Batch status requested", batch_id=batch_id)

        try:
            postgres_adapter = await get_postgres_adapter()
            batch_records = await postgres_adapter.fetch_all(
                """
                SELECT id, document_id, status, processing_duration_ms, quality_score,
                       output_data->>'filename' as filename, error_details
                FROM document_processing
                WHERE (output_data->>'batch_id') = $1 AND tenant_id = $2
                """,
                batch_id,
                tenant_id,
            )

            if not batch_records:
                return {
                    "batch_id": batch_id,
                    "status": "not_found",
                    "total_files": 0,
                    "completed": 0,
                    "processing": 0,
                    "failed": 0,
                    "success_rate": 0.0,
                    "individual_status": [],
                }

            status_counts = {"completed": 0, "processing": 0, "failed": 0, "pending": 0}
            individual_status: List[Dict[str, Any]] = []
            total_duration = 0
            completed_count = 0

            for record in batch_records:
                status = record["status"]
                status_counts[status] = status_counts.get(status, 0) + 1

                duration_seconds = (
                    record["processing_duration_ms"] / 1000.0
                    if record.get("processing_duration_ms")
                    else None
                )

                individual_status.append(
                    {
                        "upload_id": str(record["id"]),
                        "profile_id": str(record["document_id"]),
                        "filename": record.get("filename", "Unknown"),
                        "status": status,
                        "quality_score": record.get("quality_score"),
                        "duration_seconds": duration_seconds,
                    }
                )

                if status == "completed" and record.get("processing_duration_ms"):
                    total_duration += record["processing_duration_ms"]
                    completed_count += 1

            total_files = len(batch_records)
            success_rate = (
                status_counts["completed"] / total_files if total_files > 0 else 0.0
            )
            avg_duration = (
                (total_duration / completed_count / 1000.0)
                if completed_count > 0
                else 0.0
            )

            if status_counts["processing"] > 0:
                overall_status = "processing"
            elif status_counts["failed"] > 0 and status_counts["completed"] == 0:
                overall_status = "failed"
            elif status_counts["completed"] == total_files:
                overall_status = "completed"
            else:
                overall_status = "partial"

            return {
                "batch_id": batch_id,
                "status": overall_status,
                "total_files": total_files,
                "completed": status_counts["completed"],
                "processing": status_counts["processing"],
                "failed": status_counts["failed"],
                "pending": status_counts["pending"],
                "success_rate": success_rate,
                "average_processing_time_seconds": avg_duration,
                "individual_status": individual_status,
            }
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to query batch status", error=str(exc))
            return {
                "batch_id": batch_id,
                "status": "error",
                "error": str(exc),
                "total_files": 0,
                "completed": 0,
                "processing": 0,
                "failed": 0,
                "success_rate": 0.0,
                "individual_status": [],
            }


    async def process_document_background(
        self,
        upload_id: str,
        profile_id: str,
        file_content: bytes,
        filename: str,
        tenant_id: str,
        user_id: str,
        webhook_url: Optional[str] = None,
        extract_embeddings: bool = True,
        processing_priority: str = "normal",
        settings: Any | None = None,
        is_reprocessing: bool = False,
    ) -> None:
        start_time = datetime.now()
        logger.info(
            "Starting AI-powered document processing",
            upload_id=upload_id,
            profile_id=profile_id,
            filename=filename,
            tenant_id=tenant_id,
        )

        if settings is None:
            settings = get_settings()
        
        # Mark file content as in use during processing (if resource tracking is available)
        try:
            # Try to find and mark the resource as in use
            stats = await self._deps.file_resource_manager.get_resource_stats()
            if stats.get("total_resources", 0) > 0:
                # There are resources to potentially mark in use
                await self._deps.file_resource_manager.mark_resource_in_use(f"file_content_{upload_id}")
        except Exception as mark_error:
            logger.warning("Failed to mark resource in use", upload_id=upload_id, error=str(mark_error))

        try:
            # Record processing start (skip if reprocessing - record was already updated)
            if not is_reprocessing:
                await self._deps.database_adapter.execute(
                    """
                    INSERT INTO document_processing (id, created_at, updated_at, document_id, tenant_id, processing_type, status, input_metadata, started_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    upload_id,
                    start_time,  # created_at
                    start_time,  # updated_at
                    profile_id,
                    tenant_id,
                    "ai_analysis",
                    "processing",
                    json.dumps({
                        "filename": filename,
                        "file_size": len(file_content),
                        "priority": processing_priority,
                    }),
                    start_time,  # started_at
                )

            # Process document content
            processed_data = await self._deps.document_processor.process_document(
                file_content=file_content,
                filename=filename,
                tenant_id=tenant_id
            )

            text_content = processed_data.get("text", "")
            metadata = processed_data.get("metadata", {})

            if not text_content or len(text_content.strip()) < 50:
                raise ValueError("Insufficient text content extracted from document")

            logger.info("Performing AI analysis", upload_id=upload_id)

            # Extract structured data
            analysis_result = await self._deps.content_extractor.extract_cv_data(text_content)
            logger.info("Processing result", upload_id=upload_id, analysis_result=analysis_result)

            # Analyze quality
            quality_assessment = await self._deps.quality_analyzer.analyze_document_quality(
                text=text_content,
                document_type="cv",
                structured_data=analysis_result,
                use_ai=True
            )

            embedding_vector = None
            if extract_embeddings and settings.is_openai_configured():
                try:
                    logger.info("Generating embeddings", upload_id=upload_id)
                    embedding_text = self._deps.content_extractor.prepare_text_for_embedding(
                        text_content,
                        analysis_result,
                    )

                    embedding_vector = await self._deps.embedding_service.generate_embedding(
                        text=embedding_text,
                        tenant_id=tenant_id,
                    )

                    if embedding_vector:
                        embedding_id = str(uuid4())
                        current_time = datetime.now()
                        await self._deps.database_adapter.execute(
                            """
                            INSERT INTO embeddings (id, created_at, updated_at, entity_id, entity_type, tenant_id, embedding_model, embedding)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                            ON CONFLICT (entity_id, entity_type, tenant_id) DO UPDATE SET
                                embedding = EXCLUDED.embedding,
                                updated_at = NOW()
                            """,
                            embedding_id,
                            current_time,
                            current_time,
                            profile_id,
                            "cv_profile",
                            tenant_id,
                            settings.OPENAI_EMBEDDING_MODEL,
                            f"[{','.join(map(str, embedding_vector))}]",
                        )
                        logger.info("Embeddings stored successfully", upload_id=upload_id)
                except Exception as exc:  # pragma: no cover - embeddings optional
                    logger.warning("Failed to generate embeddings", upload_id=upload_id, error=str(exc))

            processing_duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

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
                "updated_at": datetime.now().isoformat(),
            }

            await self._deps.database_adapter.execute(
                """
                UPDATE document_processing
                SET status = $1, completed_at = $2, processing_duration_ms = $3,
                    output_data = $4, quality_score = $5
                WHERE id = $6
                """,
                "completed",
                datetime.now(),
                processing_duration_ms,
                json.dumps(profile_data),
                quality_assessment.get("overall_score", 0),
                upload_id,
            )

            if webhook_url:
                await self._deps.notification_service.send_webhook(
                    webhook_url,
                    {
                        "upload_id": upload_id,
                        "profile_id": profile_id,
                        "status": ProcessingStatus.COMPLETED.value,
                        "quality_score": quality_assessment.get("overall_score"),
                        "processing_duration_ms": processing_duration_ms,
                        "has_embeddings": embedding_vector is not None,
                    },
                )

            logger.info(
                "AI document processing completed",
                upload_id=upload_id,
                profile_id=profile_id,
                quality_score=quality_assessment.get("overall_score"),
                processing_duration_ms=processing_duration_ms,
                has_embeddings=embedding_vector is not None,
            )

        except Exception as exc:
            processing_duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.error(
                "AI document processing failed",
                upload_id=upload_id,
                profile_id=profile_id,
                error=str(exc),
                processing_duration_ms=processing_duration_ms,
            )

            try:
                await self._deps.database_adapter.execute(
                    """
                    UPDATE document_processing
                    SET status = $1, completed_at = $2, processing_duration_ms = $3,
                        error_details = $4
                    WHERE id = $5
                    """,
                    "failed",
                    datetime.now(),
                    processing_duration_ms,
                    json.dumps({"error": str(exc), "error_type": type(exc).__name__}),
                    upload_id,
                )
            except Exception as db_error:  # pragma: no cover - best effort logging
                logger.error(
                    "Failed to update failure status in database",
                    upload_id=upload_id,
                    error=str(db_error),
                )

            if webhook_url:
                await self._deps.notification_service.send_webhook(
                    webhook_url,
                    {
                        "upload_id": upload_id,
                        "profile_id": profile_id,
                        "status": ProcessingStatus.FAILED.value,
                        "error_message": str(exc),
                        "processing_duration_ms": processing_duration_ms,
                    },
                )
        
        finally:
            # Clean up file resources after processing (success or failure)
            try:
                cleanup_stats = await self._deps.file_resource_manager.release_upload_resources(upload_id)
                logger.debug(
                    "Upload resources cleaned up",
                    upload_id=upload_id,
                    cleanup_stats=cleanup_stats
                )
            except Exception as cleanup_error:
                logger.warning(
                    "Failed to cleanup upload resources",
                    upload_id=upload_id,
                    error=str(cleanup_error)
                )


    async def process_batch_background(
        self,
        batch_id: str,
        uploads: List[UploadResponse],
        files: List[UploadFile],
        tenant_id: str,
        user_id: str,
        webhook_url: Optional[str] = None,
        extract_embeddings: bool = True,
        max_concurrent: int = 3,
        settings: Any | None = None,
    ) -> None:
        logger.info(
            "Starting batch document processing",
            batch_id=batch_id,
            file_count=len(uploads),
            max_concurrent=max_concurrent,
        )

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single(upload: UploadResponse, file: UploadFile):
            async with semaphore:
                try:
                    file_content = await file.read()
                    await self.process_document_background(
                        upload.upload_id,
                        upload.profile_id,
                        file_content,
                        file.filename or "unknown_document",
                        tenant_id,
                        user_id,
                        webhook_url=webhook_url,
                        extract_embeddings=extract_embeddings,
                        processing_priority="normal",
                        settings=settings,
                    )
                except Exception as e:
                    logger.error(
                        "Error processing single file in batch",
                        upload_id=upload.upload_id,
                        filename=file.filename,
                        error=str(e)
                    )
                    raise

        tasks = [process_single(upload, file) for upload, file in zip(uploads, files)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = sum(1 for result in results if not isinstance(result, Exception))
        failed = len(results) - successful

        if webhook_url:
            await self._send_webhook_notification(
                webhook_url,
                {
                    "batch_id": batch_id,
                    "status": "completed",
                    "total_files": len(uploads),
                    "successful": successful,
                    "failed": failed,
                },
            )

        logger.info(
            "Batch processing completed",
            batch_id=batch_id,
            successful=successful,
            failed=failed,
        )
        
        # Clean up any remaining batch resources
        try:
            total_cleanup_stats = {"released_count": 0, "total_size_mb": 0}
            for upload in uploads:
                cleanup_stats = await self._deps.file_resource_manager.release_upload_resources(
                    upload.upload_id,
                    exclude_types=[]  # Clean up all resource types
                )
                total_cleanup_stats["released_count"] += cleanup_stats.get("released_count", 0)
                total_cleanup_stats["total_size_mb"] += cleanup_stats.get("total_size_mb", 0)
            
            if total_cleanup_stats["released_count"] > 0:
                logger.info(
                    "Batch resources cleaned up",
                    batch_id=batch_id,
                    total_cleanup_stats=total_cleanup_stats
                )
        except Exception as cleanup_error:
            logger.warning(
                "Failed to cleanup batch resources",
                batch_id=batch_id,
                error=str(cleanup_error)
            )

    async def _validate_upload_file(
        self,
        *,
        file: UploadFile,
        tenant_id: str,
        tenant_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Comprehensive file validation with stream-based size checking.
        
        This method performs validation without loading the entire file into memory,
        providing protection against memory exhaustion attacks.
        """
        filename = getattr(file, 'filename', 'unknown')
        
        logger.debug(
            "Starting file validation",
            filename=filename,
            tenant_id=tenant_id,
            has_size_attr=hasattr(file, 'size')
        )
        
        # 1. Validate filename
        if not file.filename or len(file.filename) > 255:
            logger.warning(
                "Invalid filename during validation",
                filename=filename,
                filename_length=len(file.filename) if file.filename else 0
            )
            return {"valid": False, "reason": "Invalid filename"}
        
        # 2. Validate file extension
        allowed_extensions = tenant_config.get(
            "allowed_file_extensions",
            [".pdf", ".doc", ".docx", ".txt"],
        )
        file_extension = None
        if file.filename:
            file_extension = "." + file.filename.split(".")[-1].lower()

        if file_extension not in allowed_extensions:
            logger.warning(
                "Unsupported file type during validation",
                filename=filename,
                file_extension=file_extension,
                allowed_extensions=allowed_extensions
            )
            return {
                "valid": False,
                "reason": "File type not supported. Allowed types: "
                + ", ".join(allowed_extensions),
            }
        
        # 3. Validate file size using stream-based approach
        try:
            max_size_bytes = FileSizeValidator.get_tenant_max_file_size(
                tenant_config=tenant_config,
                default_mb=10
            )
            
            logger.debug(
                "Performing file size validation",
                filename=filename,
                max_size_mb=max_size_bytes / (1024 * 1024)
            )
            
            # Use stream-based validation that works with or without file.size
            size_validation = await FileSizeValidator.validate_file_size(
                file=file,
                max_size_bytes=max_size_bytes,
                tenant_config=tenant_config
            )
            
            logger.info(
                "File size validation completed",
                filename=filename,
                file_size=size_validation["size"],
                size_mb=size_validation["size"] / (1024 * 1024),
                validation_method=size_validation.get("validation_method", "unknown")
            )
            
            return {"valid": True, "file_size": size_validation["size"]}
            
        except FileSizeExceededError as e:
            logger.warning(
                "File size exceeded during validation",
                filename=filename,
                actual_size=e.actual_size,
                max_size=e.max_size,
                actual_mb=e.actual_size / (1024 * 1024),
                max_mb=e.max_size / (1024 * 1024)
            )
            return {
                "valid": False,
                "reason": str(e)
            }
        except InvalidFileError as e:
            logger.error(
                "Invalid file error during validation",
                filename=filename,
                error=str(e),
                reason=e.reason
            )
            return {
                "valid": False,
                "reason": str(e)
            }
        except Exception as e:
            logger.error(
                "Unexpected error during file validation",
                filename=filename,
                error=str(e),
                error_type=type(e).__name__
            )
            return {
                "valid": False,
                "reason": f"File validation failed: {str(e)}"
            }

    async def _update_upload_usage(
        self,
        tenant_id: str,
        document_count: int,
        file_size_bytes: int = 0,
    ) -> None:
        try:
            storage_gb = (
                file_size_bytes / (1024 * 1024 * 1024) if file_size_bytes > 0 else 0
            )
            await self._deps.tenant_manager.update_usage_metrics(
                tenant_id=tenant_id,
                metrics_update={
                    "documents_processed": document_count,
                    "documents_uploaded": document_count,
                    "storage_used_gb": storage_gb,
                },
            )
            logger.debug(
                "Updated tenant upload usage",
                tenant_id=tenant_id,
                document_count=document_count,
                storage_gb=round(storage_gb, 4),
            )
        except Exception as exc:  # pragma: no cover - usage updates best effort
            logger.warning("Failed to update upload usage", error=str(exc))

    async def get_batch_processing_status(
        self,
        *,
        batch_id: str,
        tenant_id: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """Get batch processing status using dependency injection."""
        logger.debug("Batch status requested", batch_id=batch_id)

        try:
            batch_records = await self._deps.database_adapter.fetch_all(
                """
                SELECT id, document_id, status, processing_duration_ms, quality_score,
                       output_data->>'filename' as filename, error_details
                FROM document_processing
                WHERE (output_data->>'batch_id') = $1 AND tenant_id = $2
                """,
                batch_id,
                tenant_id,
            )

            if not batch_records:
                return {
                    "batch_id": batch_id,
                    "status": "not_found",
                    "total_files": 0,
                    "completed": 0,
                    "processing": 0,
                    "failed": 0,
                    "success_rate": 0.0,
                    "individual_status": [],
                }

            status_counts = {"completed": 0, "processing": 0, "failed": 0, "pending": 0}
            individual_status: List[Dict[str, Any]] = []
            total_duration = 0
            completed_count = 0

            for record in batch_records:
                status = record["status"]
                status_counts[status] = status_counts.get(status, 0) + 1

                duration_seconds = (
                    record["processing_duration_ms"] / 1000.0
                    if record.get("processing_duration_ms")
                    else None
                )

                individual_status.append(
                    {
                        "upload_id": str(record["id"]),
                        "profile_id": str(record["document_id"]),
                        "filename": record.get("filename", "Unknown"),
                        "status": status,
                        "quality_score": record.get("quality_score"),
                        "duration_seconds": duration_seconds,
                    }
                )

                if status == "completed" and record.get("processing_duration_ms"):
                    total_duration += record["processing_duration_ms"]
                    completed_count += 1

            total_files = len(batch_records)
            success_rate = (
                status_counts["completed"] / total_files if total_files > 0 else 0.0
            )
            avg_duration = (
                (total_duration / completed_count / 1000.0)
                if completed_count > 0
                else 0.0
            )

            if status_counts["processing"] > 0:
                overall_status = "processing"
            elif status_counts["failed"] > 0 and status_counts["completed"] == 0:
                overall_status = "failed"
            elif status_counts["completed"] == total_files:
                overall_status = "completed"
            else:
                overall_status = "partial"

            return {
                "batch_id": batch_id,
                "status": overall_status,
                "total_files": total_files,
                "completed": status_counts["completed"],
                "processing": status_counts["processing"],
                "failed": status_counts["failed"],
                "pending": status_counts["pending"],
                "success_rate": success_rate,
                "average_processing_time_seconds": avg_duration,
                "individual_status": individual_status,
            }
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to query batch status", error=str(exc))
            return {
                "batch_id": batch_id,
                "status": "error",
                "error": str(exc),
                "total_files": 0,
                "completed": 0,
                "processing": 0,
                "failed": 0,
                "success_rate": 0.0,
                "individual_status": [],
            }

    async def reprocess_document(
        self,
        *,
        upload_id: str,
        tenant_id: str,
        user_id: str,
        force_reprocess: bool = False,
        extract_embeddings: bool = True,
        schedule_task: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Reprocess a document using SQLModel ORM.

        Args:
            upload_id: The document processing record ID
            tenant_id: The tenant identifier for multi-tenant isolation
            user_id: The user requesting reprocessing
            force_reprocess: Force reprocessing even if already completed
            extract_embeddings: Generate new embeddings during reprocessing
            schedule_task: Optional task scheduler for background processing

        Returns:
            Dictionary with reprocessing status and details
        """
        logger.info(
            "Document reprocessing requested",
            upload_id=upload_id,
            tenant_id=tenant_id,
            user_id=user_id,
            force_reprocess=force_reprocess
        )

        try:
            # Get SQLModel database manager
            db_manager = get_sqlmodel_db_manager()

            if not db_manager.is_initialized:
                logger.error("SQLModel database manager not initialized")
                return {
                    "status": "error",
                    "upload_id": upload_id,
                    "message": "Database manager not initialized",
                    "error_code": "DB_NOT_INITIALIZED"
                }

            # Query the document_processing record using SQLModel ORM
            async with db_manager.get_session() as session:
                statement = select(DocumentProcessingTable).where(
                    DocumentProcessingTable.id == upload_id,
                    DocumentProcessingTable.tenant_id == tenant_id
                )
                result = await session.execute(statement)
                processing_record = result.scalar_one_or_none()

                if not processing_record:
                    logger.warning(
                        "Document processing record not found",
                        upload_id=upload_id,
                        tenant_id=tenant_id
                    )
                    return {
                        "status": "error",
                        "upload_id": upload_id,
                        "message": "Document processing record not found",
                        "error_code": "RECORD_NOT_FOUND"
                    }

                # Store original status for response
                original_status = processing_record.status

                # Validate status - check if reprocessing is allowed
                if processing_record.status == "processing":
                    logger.warning(
                        "Cannot reprocess - document is currently being processed",
                        upload_id=upload_id,
                        current_status=processing_record.status
                    )
                    return {
                        "status": "error",
                        "upload_id": upload_id,
                        "message": "Cannot reprocess - document is currently being processed",
                        "current_status": processing_record.status,
                        "error_code": "ALREADY_PROCESSING"
                    }

                # Check if force_reprocess is required
                if processing_record.status == "completed" and not force_reprocess:
                    logger.warning(
                        "Document already processed - use force_reprocess to override",
                        upload_id=upload_id,
                        current_status=processing_record.status
                    )
                    return {
                        "status": "error",
                        "upload_id": upload_id,
                        "message": "Document already processed. Use force_reprocess=true to reprocess",
                        "current_status": processing_record.status,
                        "error_code": "ALREADY_COMPLETED"
                    }

                # Extract file metadata from input_metadata
                input_metadata = processing_record.input_metadata or {}
                filename = input_metadata.get("filename", "unknown_document")
                file_size = input_metadata.get("file_size", 0)

                logger.info(
                    "Document metadata extracted",
                    upload_id=upload_id,
                    filename=filename,
                    file_size=file_size,
                    original_status=original_status
                )

                # Retrieve file from storage
                try:
                    file_content = await self._deps.file_storage.retrieve_file(
                        tenant_id=str(tenant_id),
                        upload_id=upload_id
                    )

                    logger.info(
                        "File retrieved from storage for reprocessing",
                        upload_id=upload_id,
                        filename=filename,
                        file_size=len(file_content)
                    )

                except FileNotFoundError:
                    logger.error(
                        "File not found in storage - cannot reprocess",
                        upload_id=upload_id,
                        tenant_id=str(tenant_id)
                    )
                    return {
                        "status": "error",
                        "error_code": "FILE_NOT_FOUND",
                        "message": "Original file not found in storage. Please re-upload the document.",
                    }
                except Exception as storage_error:
                    logger.error(
                        "Failed to retrieve file from storage",
                        upload_id=upload_id,
                        error=str(storage_error)
                    )
                    return {
                        "status": "error",
                        "error_code": "STORAGE_ERROR",
                        "message": "Failed to retrieve file from storage",
                        "details": str(storage_error),
                    }

                # Update status to "processing" and reset timestamps
                current_time = datetime.utcnow()  # Use timezone-naive to match database column type
                processing_record.status = "processing"
                processing_record.started_at = current_time
                processing_record.completed_at = None
                processing_record.error_details = None
                processing_record.updated_at = current_time
                processing_record.processing_duration_ms = None
                processing_record.quality_score = None
                processing_record.output_data = None

                # Update input_metadata to indicate this is a reprocessing operation
                updated_input_metadata = input_metadata.copy()
                updated_input_metadata["reprocessing"] = True
                updated_input_metadata["force_reprocess"] = force_reprocess
                updated_input_metadata["extract_embeddings"] = extract_embeddings
                updated_input_metadata["reprocessed_at"] = current_time.isoformat()
                updated_input_metadata["reprocessed_by_user"] = user_id
                processing_record.input_metadata = updated_input_metadata

                session.add(processing_record)
                await session.commit()
                await session.refresh(processing_record)

                logger.info(
                    "Document status updated for reprocessing",
                    upload_id=upload_id,
                    previous_status=original_status,
                    new_status=processing_record.status,
                    filename=filename
                )

            # Schedule background processing with retrieved file
            settings = get_settings()
            await self._enqueue_background(
                schedule_task,
                self.process_document_background,
                upload_id,
                str(processing_record.document_id),
                file_content,
                filename,
                str(tenant_id),
                user_id,
                None,  # webhook_url
                extract_embeddings,
                "normal",  # processing_priority
                settings,
                True,  # is_reprocessing
            )

            # Log audit event for reprocessing
            try:
                audit_service = self._deps.audit_service
                await audit_service.log_file_upload_event(
                    event_type="audit_event_type_FILE_UPLOAD_STARTED",  # Reusing upload event type
                    tenant_id=tenant_id,
                    user_id=user_id,
                    filename=filename,
                    file_size=file_size,
                    upload_id=upload_id,
                    ip_address=None,
                    user_agent=None,
                )
            except Exception as audit_error:
                logger.warning(
                    "Failed to log reprocessing audit event",
                    upload_id=upload_id,
                    error=str(audit_error)
                )

            # Return success response
            return {
                "status": "reprocess_initiated",
                "upload_id": upload_id,
                "message": "Document reprocessing initiated successfully. File retrieved from storage and sent for processing.",
                "document_id": str(processing_record.document_id),
                "previous_status": original_status,
                "new_status": "processing",
                "filename": filename,
                "file_size_bytes": len(file_content),
                "extract_embeddings": extract_embeddings,
                "force_reprocess": force_reprocess,
            }

        except Exception as exc:
            logger.error(
                "Reprocess failed",
                upload_id=upload_id,
                tenant_id=tenant_id,
                error=str(exc),
                error_type=type(exc).__name__
            )
            return {
                "status": "error",
                "upload_id": upload_id,
                "message": f"Reprocessing failed: {str(exc)}",
                "error_details": str(exc),
                "error_type": type(exc).__name__
            }

    async def cancel_processing(self, *, upload_id: str, user_id: str, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Cancel document processing with actual task cancellation."""
        logger.info("Processing cancellation requested", upload_id=upload_id, user_id=user_id)
        
        try:
            # First check if the processing can be cancelled
            if tenant_id:
                current_status = await self._get_processing_status_from_db(upload_id, tenant_id)
                if current_status and current_status in ["completed", "failed", "cancelled"]:
                    return {
                        "status": "error",
                        "upload_id": upload_id,
                        "message": f"Cannot cancel processing - job is already {current_status}",
                        "current_status": current_status
                    }
            
            # Cancel background tasks using task manager
            task_manager = self._deps.task_manager
            cancellation_result = await task_manager.cancel_tasks_for_upload(
                upload_id, 
                reason=f"Cancelled by user {user_id}"
            )
            
            # Update database status to cancelled if tasks were found and cancelled
            cancelled_in_db = False
            if cancellation_result["cancelled_count"] > 0 or not cancellation_result["task_ids"]:
                try:
                    await self._update_processing_status_to_cancelled(
                        upload_id=upload_id,
                        tenant_id=tenant_id,
                        reason=f"Cancelled by user {user_id}"
                    )
                    cancelled_in_db = True
                except Exception as db_error:
                    logger.warning(
                        "Failed to update database status to cancelled",
                        upload_id=upload_id,
                        error=str(db_error)
                    )
            
            # Log cancellation audit event
            try:
                audit_service = self._deps.audit_service
                await audit_service.log_file_upload_event(
                    event_type="audit_event_type_PROCESSING_CANCELLED",
                    tenant_id=tenant_id or "unknown",
                    user_id=user_id,
                    filename="unknown",
                    file_size=0,
                    upload_id=upload_id,
                    processing_duration_ms=0,
                    error_message=f"Processing cancelled by user {user_id}",
                    ip_address=None,
                    user_agent=None,
                )
            except Exception as audit_error:
                logger.warning("Failed to log cancellation audit event", error=str(audit_error))
            
            # Determine overall success
            total_operations = cancellation_result["cancelled_count"] + cancellation_result["failed_count"]
            
            if cancellation_result["cancelled_count"] > 0 or cancelled_in_db:
                status = "cancelled"
                message = f"Processing cancelled successfully. {cancellation_result['message']}"
                if cancelled_in_db:
                    message += " Database status updated."
            elif total_operations == 0:
                # No active tasks found, but try to mark as cancelled in DB anyway
                try:
                    await self._update_processing_status_to_cancelled(
                        upload_id=upload_id,
                        tenant_id=tenant_id,
                        reason=f"Cancelled by user {user_id}"
                    )
                    status = "cancelled"
                    message = "No active tasks found, but processing marked as cancelled in database"
                except Exception:
                    status = "warning"
                    message = "No active tasks found and unable to update database status"
            else:
                status = "partial"
                message = f"Partial cancellation: {cancellation_result['message']}"
            
            result = {
                "status": status,
                "upload_id": upload_id,
                "message": message,
                "task_cancellation": cancellation_result,
                "database_updated": cancelled_in_db
            }
            
            # Include failure details if any
            if cancellation_result["failed_count"] > 0:
                result["failed_cancellations"] = cancellation_result["failed_cancellations"]
            
            logger.info(
                "Processing cancellation completed",
                upload_id=upload_id,
                user_id=user_id,
                status=status,
                cancelled_tasks=cancellation_result["cancelled_count"],
                failed_tasks=cancellation_result["failed_count"],
                db_updated=cancelled_in_db
            )
            
            # Clean up upload resources after cancellation
            try:
                cleanup_stats = await self._deps.file_resource_manager.release_upload_resources(upload_id)
                result["cleanup_stats"] = cleanup_stats
                if cleanup_stats.get("released_count", 0) > 0:
                    logger.info(
                        "Upload resources cleaned up after cancellation",
                        upload_id=upload_id,
                        cleanup_stats=cleanup_stats
                    )
            except Exception as cleanup_error:
                logger.warning(
                    "Failed to cleanup resources after cancellation",
                    upload_id=upload_id,
                    error=str(cleanup_error)
                )
                result["cleanup_error"] = str(cleanup_error)
            
            return result
            
        except Exception as exc:
            logger.error(
                "Processing cancellation failed",
                upload_id=upload_id,
                user_id=user_id,
                error=str(exc)
            )
            return {
                "status": "error",
                "upload_id": upload_id,
                "message": f"Cancellation failed: {str(exc)}",
                "error_details": str(exc)
            }

    async def _mock_reprocess(self, upload_id: str) -> None:
        await asyncio.sleep(2)
        logger.info("Mock reprocessing completed", upload_id=upload_id)
    
    async def _get_processing_status_from_db(self, upload_id: str, tenant_id: str) -> Optional[str]:
        """Get the current processing status from database."""
        try:
            record = await self._deps.database_adapter.fetch_one(
                """
                SELECT status FROM document_processing 
                WHERE id = $1 AND tenant_id = $2
                """,
                upload_id,
                tenant_id
            )
            return record["status"] if record else None
        except Exception as e:
            logger.warning(
                "Failed to fetch processing status from database", 
                upload_id=upload_id, 
                error=str(e)
            )
            return None
    
    async def _update_processing_status_to_cancelled(
        self, 
        upload_id: str, 
        tenant_id: Optional[str] = None,
        reason: Optional[str] = None
    ) -> bool:
        """Update processing status to cancelled in database."""
        try:
            if tenant_id:
                # Update with tenant_id if provided
                result = await self._deps.database_adapter.execute(
                    """
                    UPDATE document_processing
                    SET status = $1, completed_at = $2, error_details = $3
                    WHERE id = $4 AND tenant_id = $5 AND status NOT IN ('completed', 'failed', 'cancelled')
                    """,
                    "cancelled",
                    datetime.now(),
                    json.dumps({"error": reason or "Processing cancelled", "cancelled_by": "user"}),
                    upload_id,
                    tenant_id
                )
            else:
                # Update without tenant_id for backward compatibility
                result = await self._deps.database_adapter.execute(
                    """
                    UPDATE document_processing
                    SET status = $1, completed_at = $2, error_details = $3
                    WHERE id = $4 AND status NOT IN ('completed', 'failed', 'cancelled')
                    """,
                    "cancelled",
                    datetime.now(),
                    json.dumps({"error": reason or "Processing cancelled", "cancelled_by": "user"}),
                    upload_id
                )
            
            logger.info(
                "Processing status updated to cancelled",
                upload_id=upload_id,
                tenant_id=tenant_id,
                reason=reason
            )
            return True
            
        except Exception as e:
            logger.error(
                "Failed to update processing status to cancelled",
                upload_id=upload_id,
                tenant_id=tenant_id,
                error=str(e)
            )
            return False
    
    async def cancel_batch_processing(
        self,
        *,
        batch_id: str,
        user_id: str,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Cancel all processing tasks for a batch upload."""
        logger.info(
            "Batch processing cancellation requested",
            batch_id=batch_id,
            user_id=user_id,
            tenant_id=tenant_id
        )
        
        try:
            # Get all upload IDs for this batch from database
            upload_ids = await self._get_upload_ids_for_batch(batch_id, tenant_id)
            
            if not upload_ids:
                return {
                    "status": "error",
                    "batch_id": batch_id,
                    "message": "No uploads found for this batch",
                    "cancelled_count": 0
                }
            
            # Cancel each upload's processing
            batch_results = []
            total_cancelled = 0
            total_failed = 0
            
            for upload_id in upload_ids:
                try:
                    result = await self.cancel_processing(
                        upload_id=upload_id,
                        user_id=user_id,
                        tenant_id=tenant_id
                    )
                    
                    batch_results.append({
                        "upload_id": upload_id,
                        "status": result["status"],
                        "message": result["message"]
                    })
                    
                    if result["status"] == "cancelled":
                        total_cancelled += 1
                    else:
                        total_failed += 1
                        
                except Exception as e:
                    batch_results.append({
                        "upload_id": upload_id,
                        "status": "error",
                        "message": f"Cancellation failed: {str(e)}"
                    })
                    total_failed += 1
            
            # Log batch cancellation audit event
            try:
                audit_service = self._deps.audit_service
                await audit_service.log_file_upload_event(
                    event_type="audit_event_type_PROCESSING_CANCELLED",
                    tenant_id=tenant_id or "unknown",
                    user_id=user_id,
                    filename=f"batch_{batch_id}",
                    file_size=0,
                    upload_id=batch_id,
                    processing_duration_ms=0,
                    error_message=f"Batch processing cancelled by user {user_id}",
                    ip_address=None,
                    user_agent=None,
                )
            except Exception as audit_error:
                logger.warning("Failed to log batch cancellation audit event", error=str(audit_error))
            
            # Determine overall status
            if total_cancelled == len(upload_ids):
                status = "cancelled"
                message = f"All {total_cancelled} uploads in batch cancelled successfully"
            elif total_cancelled > 0:
                status = "partial"
                message = f"Partial success: {total_cancelled} of {len(upload_ids)} uploads cancelled"
            else:
                status = "failed"
                message = f"Failed to cancel any uploads in batch"
            
            result = {
                "status": status,
                "batch_id": batch_id,
                "message": message,
                "total_uploads": len(upload_ids),
                "cancelled_count": total_cancelled,
                "failed_count": total_failed,
                "individual_results": batch_results
            }
            
            logger.info(
                "Batch processing cancellation completed",
                batch_id=batch_id,
                user_id=user_id,
                status=status,
                total_uploads=len(upload_ids),
                cancelled_count=total_cancelled,
                failed_count=total_failed
            )
            
            return result
            
        except Exception as exc:
            logger.error(
                "Batch processing cancellation failed",
                batch_id=batch_id,
                user_id=user_id,
                error=str(exc)
            )
            return {
                "status": "error",
                "batch_id": batch_id,
                "message": f"Batch cancellation failed: {str(exc)}",
                "error_details": str(exc)
            }
    
    async def _get_upload_ids_for_batch(self, batch_id: str, tenant_id: Optional[str] = None) -> List[str]:
        """Get all upload IDs associated with a batch."""
        try:
            if tenant_id:
                records = await self._deps.database_adapter.fetch_all(
                    """
                    SELECT id FROM document_processing 
                    WHERE (output_data->>'batch_id') = $1 AND tenant_id = $2
                    """,
                    batch_id,
                    tenant_id
                )
            else:
                records = await self._deps.database_adapter.fetch_all(
                    """
                    SELECT id FROM document_processing 
                    WHERE (output_data->>'batch_id') = $1
                    """,
                    batch_id
                )
            
            return [record["id"] for record in records]
            
        except Exception as e:
            logger.warning(
                "Failed to fetch upload IDs for batch",
                batch_id=batch_id,
                error=str(e)
            )
            return []

    async def get_upload_history(
        self,
        *,
        tenant_id: str,
        pagination: Any,
        status_filter: Optional[Any] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> UploadHistoryResult:
        """
        Get paginated upload history for a tenant.

        Args:
            tenant_id: Tenant identifier
            pagination: Pagination parameters (page, size, offset, limit)
            status_filter: Optional status filter
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            UploadHistoryResult with list of upload history items and pagination metadata
        """
        from app.api.schemas.upload_schemas import UploadHistoryItem

        logger.debug(
            "Upload history requested",
            tenant_id=tenant_id,
            page=pagination.page,
            size=pagination.size,
            status_filter=status_filter,
            start_date=start_date,
            end_date=end_date
        )

        try:
            # Build WHERE clause with filters
            where_conditions = ["tenant_id = $1"]
            params = [tenant_id]
            param_counter = 2

            if status_filter:
                where_conditions.append(f"status = ${param_counter}")
                params.append(status_filter.value if hasattr(status_filter, 'value') else str(status_filter))
                param_counter += 1

            if start_date:
                where_conditions.append(f"started_at >= ${param_counter}")
                params.append(start_date)
                param_counter += 1

            if end_date:
                where_conditions.append(f"started_at <= ${param_counter}")
                params.append(end_date)
                param_counter += 1

            where_clause = " AND ".join(where_conditions)

            # Get total count
            count_query = f"""
                SELECT COUNT(*) as total
                FROM document_processing
                WHERE {where_clause}
            """

            count_result = await self._deps.database_adapter.fetch_one(count_query, *params)
            total_count = count_result["total"] if count_result else 0

            # Get paginated records
            query = f"""
                SELECT
                    id as upload_id,
                    document_id as profile_id,
                    input_metadata->>'filename' as filename,
                    status,
                    started_at as created_at,
                    completed_at,
                    processing_duration_ms,
                    quality_score,
                    error_details->>'error' as error_message,
                    (input_metadata->>'file_size')::bigint as file_size_bytes
                FROM document_processing
                WHERE {where_clause}
                ORDER BY started_at DESC
                OFFSET ${param_counter} LIMIT ${param_counter + 1}
            """

            params.extend([pagination.offset, pagination.limit])

            records = await self._deps.database_adapter.fetch_all(query, *params)

            # Convert records to UploadHistoryItem models
            upload_history = []
            for record in records:
                try:
                    history_item = UploadHistoryItem(
                        upload_id=str(record["upload_id"]),
                        profile_id=str(record["profile_id"]),
                        filename=record.get("filename") or "Unknown",
                        status=record["status"],
                        created_at=record["created_at"],
                        completed_at=record.get("completed_at"),
                        processing_duration_seconds=(
                            record["processing_duration_ms"] / 1000.0
                            if record.get("processing_duration_ms")
                            else None
                        ),
                        quality_score=record.get("quality_score"),
                        error_message=record.get("error_message"),
                        file_size_bytes=record.get("file_size_bytes")
                    )
                    upload_history.append(history_item)
                except Exception as item_error:
                    logger.warning(
                        "Failed to parse upload history item",
                        upload_id=record.get("upload_id"),
                        error=str(item_error)
                    )
                    continue

            logger.info(
                "Upload history retrieved",
                tenant_id=tenant_id,
                total_count=total_count,
                returned_count=len(upload_history),
                page=pagination.page
            )

            return UploadHistoryResult(
                items=upload_history,
                total=total_count,
                page=pagination.page,
                size=pagination.size
            )

        except Exception as exc:
            logger.error(
                "Failed to retrieve upload history",
                tenant_id=tenant_id,
                error=str(exc)
            )
            # Return empty result as fallback
            return UploadHistoryResult(
                items=[],
                total=0,
                page=pagination.page,
                size=pagination.size
            )

    async def _enqueue_background(self, scheduler: Optional[Any], func, *args) -> Optional[str]:
        """Enqueue a background task with task manager tracking."""

        # Extract task information from args for task manager
        upload_id = None
        tenant_id = None
        user_id = None
        task_type = TaskType.DOCUMENT_PROCESSING  # Default

        # Try to extract common parameters from args
        if len(args) >= 3:
            if isinstance(args[0], str):  # upload_id typically first
                upload_id = args[0]
            if len(args) >= 5 and isinstance(args[4], str):  # tenant_id typically 5th
                tenant_id = args[4]
            if len(args) >= 6 and isinstance(args[5], str):  # user_id typically 6th
                user_id = args[5]

        # Determine task type based on function name
        if hasattr(func, '__name__'):
            if 'batch' in func.__name__:
                task_type = TaskType.BATCH_PROCESSING
            elif 'reprocess' in func.__name__:
                task_type = TaskType.REPROCESSING

        # Use task manager if we have required information
        if upload_id and tenant_id and user_id:
            task_manager = self._deps.task_manager
            task_id = f"{upload_id}_{task_type.value}_{int(datetime.now().timestamp())}"

            result = func(*args)
            if asyncio.iscoroutine(result):
                task = task_manager.create_task(
                    result,
                    task_id=task_id,
                    upload_id=upload_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    task_type=task_type
                )
                return task_id
        
        # Fallback to original behavior for compatibility
        if scheduler is None:
            result = func(*args)
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)
            return None

        add_task = getattr(scheduler, "add_task", None)
        if callable(add_task):
            add_task(func, *args)
            return None

        if callable(scheduler):
            scheduler(func, *args)
            return None

        raise AttributeError("Provided scheduler does not support task scheduling")


__all__ = ["UploadApplicationService", "UploadError", "UploadHistoryResult"]
