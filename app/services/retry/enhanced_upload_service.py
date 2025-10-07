"""Enhanced upload service with retry mechanism integration."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog

from app.domain.retry import RetryPolicy
from app.infrastructure.providers.retry_provider import (
    get_retry_executor, get_specialized_retry_executor
)


logger = structlog.get_logger(__name__)


class EnhancedUploadService:
    """Upload service enhanced with comprehensive retry mechanisms."""
    
    def __init__(self):
        self._logger = structlog.get_logger(__name__)
    
    async def process_document_with_retry(
        self,
        upload_id: str,
        file_content: bytes,
        filename: str,
        tenant_id: str,
        user_id: str,
        *,
        retry_policy: Optional[RetryPolicy] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a document with comprehensive retry mechanism.
        
        This replaces the problematic section at line 601 in the original upload service.
        """
        
        processing_context = {
            **(context or {}),
            "upload_id": upload_id,
            "filename": filename,
            "file_size": len(file_content),
            "tenant_id": tenant_id,
            "user_id": user_id
        }
        
        # Get specialized retry executor for document processing
        retry_executor = await get_specialized_retry_executor("document_processing")
        
        try:
            # Execute document processing with retry
            result = await retry_executor.execute_with_retry(
                operation_func=self._process_document_operation,
                operation_id=upload_id,
                operation_type="document_processing",
                tenant_id=tenant_id,
                user_id=user_id,
                args=(file_content, filename),
                kwargs={
                    "upload_id": upload_id,
                    "context": processing_context
                },
                policy=retry_policy,
                context=processing_context
            )
            
            return {
                "status": "success",
                "upload_id": upload_id,
                "processing_result": result
            }
        
        except Exception as error:
            self._logger.error(
                "Document processing failed after all retries",
                upload_id=upload_id,
                filename=filename,
                error=str(error),
                tenant_id=tenant_id
            )
            
            return {
                "status": "failed",
                "upload_id": upload_id,
                "error": str(error)
            }
    
    async def _process_document_operation(
        self,
        file_content: bytes,
        filename: str,
        *,
        upload_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Core document processing operation that can be retried.
        
        This encapsulates the actual processing logic that was failing at line 601.
        """
        
        processing_start = datetime.utcnow()
        
        self._logger.info(
            "Starting document processing operation",
            upload_id=upload_id,
            filename=filename,
            file_size=len(file_content)
        )
        
        try:
            # Step 1: Content extraction with retry capability
            extracted_content = await self._extract_document_content_with_retry(
                file_content, filename, upload_id, context
            )
            
            # Step 2: AI analysis with retry capability
            ai_analysis = await self._perform_ai_analysis_with_retry(
                extracted_content, upload_id, context
            )
            
            # Step 3: Embedding generation with retry capability
            embeddings = await self._generate_embeddings_with_retry(
                extracted_content, upload_id, context
            )
            
            # Step 4: Database persistence with retry capability
            persistence_result = await self._persist_document_data_with_retry(
                extracted_content, ai_analysis, embeddings, upload_id, context
            )
            
            processing_duration = (datetime.utcnow() - processing_start).total_seconds()
            
            result = {
                "extracted_content": extracted_content,
                "ai_analysis": ai_analysis,
                "embeddings": embeddings,
                "persistence_result": persistence_result,
                "processing_duration_seconds": processing_duration,
                "processed_at": datetime.utcnow().isoformat()
            }
            
            self._logger.info(
                "Document processing operation completed successfully",
                upload_id=upload_id,
                processing_duration_seconds=processing_duration
            )
            
            return result
        
        except Exception as error:
            processing_duration = (datetime.utcnow() - processing_start).total_seconds()
            
            self._logger.error(
                "Document processing operation failed",
                upload_id=upload_id,
                error=str(error),
                processing_duration_seconds=processing_duration
            )
            
            # Re-raise to allow retry mechanism to handle
            raise
    
    async def _extract_document_content_with_retry(
        self,
        file_content: bytes,
        filename: str,
        upload_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract document content with retry capability."""
        
        retry_executor = await get_specialized_retry_executor("document_processing")
        
        extraction_context = {
            **context,
            "operation_step": "content_extraction",
            "filename": filename,
            "file_size": len(file_content)
        }
        
        return await retry_executor.execute_with_retry(
            operation_func=self._extract_content_operation,
            operation_id=f"{upload_id}_extraction",
            operation_type="content_extraction",
            tenant_id=context["tenant_id"],
            user_id=context["user_id"],
            args=(file_content, filename),
            context=extraction_context
        )
    
    async def _extract_content_operation(
        self,
        file_content: bytes,
        filename: str
    ) -> Dict[str, Any]:
        """Core content extraction operation."""

        # Import here to avoid circular dependencies
        from app.infrastructure.providers.document_provider import get_pdf_processor

        pdf_processor = await get_pdf_processor()

        # Process PDF using correct method which returns a PDFDocument object
        pdf_document = await pdf_processor.process_pdf(
            pdf_content=file_content,
            filename=filename
        )

        if not pdf_document.full_text:
            raise ValueError("No text content could be extracted from document")

        return {
            "raw_text": pdf_document.full_text,
            "metadata": {**pdf_document.metadata, **pdf_document.processing_info},
            "page_count": pdf_document.total_pages,
            "word_count": pdf_document.total_words,
            "extraction_method": "pdf_processing"
        }
    
    async def _perform_ai_analysis_with_retry(
        self,
        extracted_content: Dict[str, Any],
        upload_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform AI analysis with retry capability."""
        
        retry_executor = await get_specialized_retry_executor("ai_processing")
        
        ai_context = {
            **context,
            "operation_step": "ai_analysis",
            "text_length": len(extracted_content.get("raw_text", "")),
            "word_count": extracted_content.get("word_count", 0)
        }
        
        return await retry_executor.execute_with_retry(
            operation_func=self._ai_analysis_operation,
            operation_id=f"{upload_id}_ai_analysis",
            operation_type="ai_processing",
            tenant_id=context["tenant_id"],
            user_id=context["user_id"],
            args=(extracted_content,),
            context=ai_context
        )
    
    async def _ai_analysis_operation(
        self,
        extracted_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Core AI analysis operation."""
        
        # Import here to avoid circular dependencies
        from app.infrastructure.providers.ai_provider import get_content_extractor
        
        content_extractor = await get_content_extractor()
        
        raw_text = extracted_content["raw_text"]
        
        # Extract structured data using AI
        structured_data = await content_extractor.extract_cv_data(raw_text)
        
        if not structured_data:
            raise ValueError("AI analysis failed to extract structured data")
        
        return {
            "structured_data": structured_data,
            "analysis_confidence": structured_data.get("confidence", 0.0),
            "extracted_sections": list(structured_data.keys()),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _generate_embeddings_with_retry(
        self,
        extracted_content: Dict[str, Any],
        upload_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate embeddings with retry capability."""
        
        retry_executor = await get_specialized_retry_executor("embedding_generation")
        
        embedding_context = {
            **context,
            "operation_step": "embedding_generation",
            "text_length": len(extracted_content.get("raw_text", ""))
        }
        
        return await retry_executor.execute_with_retry(
            operation_func=self._embedding_generation_operation,
            operation_id=f"{upload_id}_embeddings",
            operation_type="embedding_generation",
            tenant_id=context["tenant_id"],
            user_id=context["user_id"],
            args=(extracted_content,),
            context=embedding_context
        )
    
    async def _embedding_generation_operation(
        self,
        extracted_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Core embedding generation operation."""
        
        # Import here to avoid circular dependencies
        from app.infrastructure.providers.ai_provider import get_embedding_service
        
        embedding_service = await get_embedding_service()
        
        raw_text = extracted_content["raw_text"]
        
        if len(raw_text.strip()) < 10:
            raise ValueError("Text content too short for embedding generation")
        
        # Generate embeddings
        embedding_vector = await embedding_service.generate_embedding(raw_text)
        
        if not embedding_vector or len(embedding_vector) == 0:
            raise ValueError("Failed to generate embeddings")
        
        return {
            "embedding_vector": embedding_vector,
            "embedding_dimension": len(embedding_vector),
            "text_length": len(raw_text),
            "generation_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _persist_document_data_with_retry(
        self,
        extracted_content: Dict[str, Any],
        ai_analysis: Dict[str, Any],
        embeddings: Dict[str, Any],
        upload_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Persist document data with retry capability."""
        
        retry_executor = await get_specialized_retry_executor("database_operation")
        
        persistence_context = {
            **context,
            "operation_step": "data_persistence",
            "data_size": len(str(extracted_content)) + len(str(ai_analysis)) + len(str(embeddings))
        }
        
        return await retry_executor.execute_with_retry(
            operation_func=self._data_persistence_operation,
            operation_id=f"{upload_id}_persistence",
            operation_type="database_operation",
            tenant_id=context["tenant_id"],
            user_id=context["user_id"],
            args=(extracted_content, ai_analysis, embeddings, upload_id),
            context=persistence_context
        )
    
    async def _data_persistence_operation(
        self,
        extracted_content: Dict[str, Any],
        ai_analysis: Dict[str, Any],
        embeddings: Dict[str, Any],
        upload_id: str
    ) -> Dict[str, Any]:
        """Core data persistence operation."""
        
        # Import here to avoid circular dependencies
        from app.database.sqlmodel_engine import get_sqlmodel_db_manager
        from app.api.schemas.document_schemas import Document, DocumentContent, DocumentStatus
        
        db_manager = get_sqlmodel_db_manager()
        
        # Create document content object
        document_content = DocumentContent(
            raw_text=extracted_content["raw_text"],
            formatted_text=extracted_content.get("formatted_text"),
            word_count=extracted_content.get("word_count", 0),
            page_count=extracted_content.get("page_count", 1),
            extracted_entities=ai_analysis["structured_data"].get("entities", {}),
            extracted_skills=ai_analysis["structured_data"].get("skills", []),
            extracted_experience=ai_analysis["structured_data"].get("experience", []),
            extracted_education=ai_analysis["structured_data"].get("education", []),
            extraction_confidence=ai_analysis.get("analysis_confidence", 0.0)
        )
        
        async with db_manager.get_session() as session:
            # Update document with processed content
            from sqlalchemy import select
            from app.api.schemas.document_schemas import Document as DocumentModel
            
            stmt = select(DocumentModel).where(DocumentModel.id == upload_id)
            result = await session.execute(stmt)
            document = result.scalar_one_or_none()
            
            if not document:
                raise ValueError(f"Document {upload_id} not found in database")
            
            # Update document with processing results
            document.content = document_content.dict()
            document.status = DocumentStatus.COMPLETED.value
            document.processing_completed_at = datetime.utcnow()
            document.embedding_vector = embeddings["embedding_vector"]
            document.is_indexed = True
            document.index_updated_at = datetime.utcnow()
            
            await session.commit()
            await session.refresh(document)
        
        return {
            "document_id": upload_id,
            "status": "persisted",
            "content_size": len(document_content.raw_text),
            "embedding_dimension": len(embeddings["embedding_vector"]),
            "persistence_timestamp": datetime.utcnow().isoformat()
        }
    
    async def process_batch_documents_with_retry(
        self,
        documents: List[Dict[str, Any]],
        tenant_id: str,
        user_id: str,
        *,
        max_concurrent: int = 5,
        retry_policy: Optional[RetryPolicy] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process multiple documents with retry capability."""
        
        batch_id = str(uuid4())
        batch_context = {
            **(context or {}),
            "batch_id": batch_id,
            "batch_size": len(documents),
            "tenant_id": tenant_id,
            "user_id": user_id
        }
        
        self._logger.info(
            "Starting batch document processing with retry",
            batch_id=batch_id,
            document_count=len(documents),
            tenant_id=tenant_id,
            max_concurrent=max_concurrent
        )
        
        # Prepare operations for batch execution
        operations = []
        for doc_info in documents:
            operation = {
                "operation_id": doc_info["upload_id"],
                "operation_func": self._process_document_operation,
                "args": (doc_info["file_content"], doc_info["filename"]),
                "kwargs": {
                    "upload_id": doc_info["upload_id"],
                    "context": {
                        **batch_context,
                        "filename": doc_info["filename"],
                        "file_size": len(doc_info["file_content"])
                    }
                },
                "context": {
                    "document_filename": doc_info["filename"],
                    "document_size": len(doc_info["file_content"])
                }
            }
            operations.append(operation)
        
        # Get retry executor and execute batch
        retry_executor = await get_specialized_retry_executor("document_processing")
        
        batch_result = await retry_executor.execute_batch_with_retry(
            operations=operations,
            operation_type="document_processing",
            tenant_id=tenant_id,
            user_id=user_id,
            policy=retry_policy,
            max_concurrent=max_concurrent,
            context=batch_context
        )
        
        self._logger.info(
            "Batch document processing completed",
            batch_id=batch_id,
            total_documents=batch_result["total_operations"],
            successful_documents=batch_result["successful_operations"],
            failed_documents=batch_result["failed_operations"],
            success_rate=batch_result["success_rate"]
        )
        
        return batch_result
    
    async def retry_failed_document_processing(
        self,
        retry_id: str,
        *,
        new_retry_policy: Optional[RetryPolicy] = None
    ) -> Dict[str, Any]:
        """Manually retry a failed document processing operation."""
        
        retry_executor = await get_retry_executor()
        
        # Get retry state to understand what failed
        from app.infrastructure.providers.retry_provider import get_retry_service
        retry_service = await get_retry_service()
        
        retry_state = await retry_service.get_retry_state(retry_id)
        if not retry_state:
            raise ValueError(f"Retry state {retry_id} not found")
        
        # Extract original operation parameters from context
        operation_context = retry_state.operation_context
        
        # Prepare for manual retry
        manual_context = {
            **operation_context,
            "manual_retry": True,
            "original_retry_id": retry_id,
            "retry_timestamp": datetime.utcnow().isoformat()
        }
        
        # If new policy provided, create new retry state
        if new_retry_policy:
            new_retry_id = await retry_service.create_retry_state(
                operation_id=retry_state.operation_id,
                operation_type=retry_state.operation_type,
                tenant_id=retry_state.tenant_id,
                user_id=retry_state.user_id,
                policy=new_retry_policy,
                context=manual_context
            )
            
            # Execute with new retry state
            result = await retry_executor.execute_with_retry(
                operation_func=self._process_document_operation,
                operation_id=retry_state.operation_id,
                operation_type=retry_state.operation_type,
                tenant_id=retry_state.tenant_id,
                user_id=retry_state.user_id,
                policy=new_retry_policy,
                context=manual_context
            )
            
            return {
                "status": "success",
                "original_retry_id": retry_id,
                "new_retry_id": new_retry_id,
                "result": result
            }
        
        else:
            # Execute with existing retry state
            result, success = await retry_executor.execute_with_manual_retry_state(
                operation_func=self._process_document_operation,
                retry_id=retry_id,
                context=manual_context
            )
            
            return {
                "status": "success" if success else "failed",
                "retry_id": retry_id,
                "result": result,
                "success": success
            }


__all__ = ["EnhancedUploadService"]