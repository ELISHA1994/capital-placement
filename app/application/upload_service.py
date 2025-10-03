"""Application layer orchestrator for upload workflows following hexagonal architecture."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog
from fastapi import UploadFile

from app.core.config import get_settings
from app.models.auth import CurrentUser
from app.models.profile import ProcessingStatus
from app.models.upload_models import (
    BatchUploadResponse,
    ProcessingStatusResponse,
    UploadResponse,
)
from app.application.dependencies import UploadDependencies
from app.domain.entities.profile import Profile, ProfileStatus, ProcessingStatus as DomainProcessingStatus
from app.domain.value_objects import ProfileId, TenantId

logger = structlog.get_logger(__name__)


class UploadError(Exception):
    """Raised when upload workflows encounter a recoverable failure."""

    def __init__(self, status_code: int, detail: Any):
        super().__init__(str(detail))
        self.status_code = status_code
        self.detail = detail


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

    async def upload_document(
        self,
        *,
        file: UploadFile,
        current_user: CurrentUser,
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

        settings = get_settings()
        tenant_id = TenantId(current_user.tenant_id)
        
        # Get tenant configuration
        tenant_config = await self._deps.tenant_manager.get_tenant_configuration(str(current_user.tenant_id))

        validation_result = await self._validate_upload_file(
            file=file,
            tenant_id=str(current_user.tenant_id),
            tenant_config=tenant_config,
        )
        if not validation_result["valid"]:
            raise UploadError(
                status_code=400,
                detail={
                    "error": "invalid_file",
                    "message": validation_result["reason"],
                    "filename": file.filename,
                },
            )

        # Check quota limits
        quota_check = await self._deps.tenant_manager.check_quota_limit(
            tenant_id=str(current_user.tenant_id),
            resource_type="documents_per_day",
            current_usage=tenant_config.get("documents_processed_today", 0),
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

        file_content = await file.read()
        profile_id = str(uuid4())

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
        current_user: CurrentUser,
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

        quota_check = await self._deps.tenant_manager.check_quota_limit(
            tenant_id=str(current_user.tenant_id),
            resource_type="documents_per_day",
            current_usage=tenant_config.get("documents_processed_today", 0),
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
                validation_result = await self._validate_upload_file(
                    file=upload_file,
                    tenant_id=str(current_user.tenant_id),
                    tenant_config=tenant_config,
                )

                if validation_result["valid"]:
                    upload_id = str(uuid4())
                    profile_id = str(uuid4())

                    file_content = await upload_file.read()
                    await upload_file.seek(0)
                    total_file_size_bytes += len(file_content)

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
                    rejected_files[upload_file.filename] = validation_result["reason"]
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
                profile_id=processing_record["document_id"],
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
                        "upload_id": record["id"],
                        "profile_id": record["document_id"],
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
        schedule_task: Optional[Any] = None,
    ) -> Dict[str, Any]:
        logger.info("Document reprocessing requested", upload_id=upload_id)
        await self._enqueue_background(schedule_task, self._mock_reprocess, upload_id)
        return {
            "status": "reprocessing_started",
            "upload_id": upload_id,
            "message": "Document reprocessing has been queued",
        }

    async def cancel_processing(self, *, upload_id: str, user_id: str) -> Dict[str, Any]:
        logger.info("Processing cancellation requested", upload_id=upload_id, user_id=user_id)
        # TODO: integrate with actual cancellation mechanics
        return {
            "status": "cancelled",
            "upload_id": upload_id,
            "message": "Processing job has been cancelled",
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

        try:
            # Record processing start
            await self._deps.database_adapter.execute(
                """
                INSERT INTO document_processing (id, document_id, tenant_id, processing_type, status, input_metadata, started_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                upload_id,
                profile_id,
                tenant_id,
                "ai_analysis",
                "processing",
                {
                    "filename": filename,
                    "file_size": len(file_content),
                    "priority": processing_priority,
                },
                start_time,
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

            # Analyze quality
            quality_assessment = await self._deps.quality_analyzer.analyze_quality(
                extracted_text=text_content,
                structured_data=analysis_result,
                document_type="cv",
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
                        await self._deps.database_adapter.execute(
                            """
                            INSERT INTO embeddings (entity_id, entity_type, tenant_id, embedding_model, embedding_vector, content_hash, metadata)
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            ON CONFLICT (entity_id, entity_type, tenant_id) DO UPDATE SET
                                embedding_vector = EXCLUDED.embedding_vector,
                                updated_at = NOW()
                            """,
                            profile_id,
                            "cv_profile",
                            tenant_id,
                            settings.OPENAI_EMBEDDING_MODEL,
                            f"[{','.join(map(str, embedding_vector))}]",
                            self._deps.content_extractor.hash_content(text_content),
                            {"upload_id": upload_id, "filename": filename},
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
                profile_data,
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
                    {"error": str(exc), "error_type": type(exc).__name__},
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

    async def _validate_upload_file(
        self,
        *,
        file: UploadFile,
        tenant_id: str,
        tenant_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        allowed_extensions = tenant_config.get(
            "allowed_file_extensions",
            [".pdf", ".doc", ".docx", ".txt"],
        )
        file_extension = None
        if file.filename:
            file_extension = "." + file.filename.split(".")[-1].lower()

        if file_extension not in allowed_extensions:
            return {
                "valid": False,
                "reason": "File type not supported. Allowed types: "
                + ", ".join(allowed_extensions),
            }

        if hasattr(file, "size") and file.size:
            max_size = tenant_config.get("max_file_size_mb", 10) * 1024 * 1024
            if file.size > max_size:
                return {
                    "valid": False,
                    "reason": f"File size too large. Maximum allowed: {max_size // (1024*1024)}MB",
                }

        if not file.filename or len(file.filename) > 255:
            return {"valid": False, "reason": "Invalid filename"}

        return {"valid": True}

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
                        "upload_id": record["id"],
                        "profile_id": record["document_id"],
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
        schedule_task: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Reprocess a document."""
        logger.info("Document reprocessing requested", upload_id=upload_id)
        await self._enqueue_background(schedule_task, self._mock_reprocess, upload_id)
        return {
            "status": "reprocessing_started",
            "upload_id": upload_id,
            "message": "Document reprocessing has been queued",
        }

    async def cancel_processing(self, *, upload_id: str, user_id: str) -> Dict[str, Any]:
        """Cancel document processing."""
        logger.info("Processing cancellation requested", upload_id=upload_id, user_id=user_id)
        # TODO: integrate with actual cancellation mechanics
        return {
            "status": "cancelled",
            "upload_id": upload_id,
            "message": "Processing job has been cancelled",
        }

    async def _mock_reprocess(self, upload_id: str) -> None:
        await asyncio.sleep(2)
        logger.info("Mock reprocessing completed", upload_id=upload_id)

    async def _enqueue_background(self, scheduler: Optional[Any], func, *args) -> None:
        if scheduler is None:
            result = func(*args)
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)
            return

        add_task = getattr(scheduler, "add_task", None)
        if callable(add_task):
            add_task(func, *args)
            return

        if callable(scheduler):
            scheduler(func, *args)
            return

        raise AttributeError("Provided scheduler does not support task scheduling")


__all__ = ["UploadApplicationService", "UploadError"]
