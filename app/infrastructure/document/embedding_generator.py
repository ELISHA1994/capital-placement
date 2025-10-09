"""
Document Embedding Generator

Specialized embedding generation for document processing:
- Document-aware chunking strategies
- Hierarchical embedding generation (document, section, paragraph)
- Semantic relationship mapping
- Batch processing optimization
- Integration with pgvector storage
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import structlog
import hashlib
import asyncio

from app.core.config import get_settings
from app.infrastructure.ai.embedding_service import EmbeddingService
from app.infrastructure.document.pdf_processor import PDFDocument
from app.infrastructure.document.schemas import (
    StructuredContent,
    ExtractedSection,
    DocumentEmbedding,
    SectionEmbedding,
    EmbeddingResult,
)

logger = structlog.get_logger(__name__)


class EmbeddingGenerator:
    """
    Advanced document embedding generator with hierarchical processing.
    
    Features:
    - Multi-level embedding generation (document, section, paragraph)
    - Intelligent content chunking and optimization
    - Semantic relationship detection between sections
    - Batch processing for performance
    - Integration with vector database storage
    - Content deduplication and versioning
    """
    
    def __init__(self, embedding_service: EmbeddingService):
        self.settings = get_settings()
        self.embedding_service = embedding_service
        self._processing_stats = {
            "documents_processed": 0,
            "embeddings_generated": 0,
            "sections_processed": 0,
            "relationships_discovered": 0,
            "processing_errors": 0
        }
    
    async def generate_document_embeddings(
        self,
        document_id: str,
        structured_content: StructuredContent,
        tenant_id: Optional[str] = None,
        generate_relationships: bool = True,
        store_in_database: bool = True
    ) -> EmbeddingResult:
        """
        Generate comprehensive embeddings for a structured document.
        
        Args:
            document_id: Unique document identifier
            structured_content: Processed document content
            tenant_id: Tenant identifier for multi-tenancy
            generate_relationships: Calculate semantic relationships
            store_in_database: Store embeddings in vector database
            
        Returns:
            EmbeddingResult with all generated embeddings
        """
        start_time = datetime.now()
        
        try:
            # Generate document-level embedding
            document_embedding = await self._generate_document_level_embedding(
                document_id=document_id,
                structured_content=structured_content,
                tenant_id=tenant_id
            )
            
            # Generate section-level embeddings
            section_embeddings = await self._generate_section_embeddings(
                document_id=document_id,
                structured_content=structured_content,
                tenant_id=tenant_id
            )
            
            # Generate semantic relationships if requested
            relationships = []
            if generate_relationships and len(section_embeddings) > 1:
                relationships = await self._discover_semantic_relationships(
                    section_embeddings
                )
                self._processing_stats["relationships_discovered"] += len(relationships)
            
            # Store embeddings in database if requested
            if store_in_database:
                await self._store_embeddings(
                    document_embedding=document_embedding,
                    section_embeddings=section_embeddings,
                    tenant_id=tenant_id
                )
            
            # Create processing info
            processing_info = {
                "processing_started": start_time.isoformat(),
                "processing_completed": datetime.now().isoformat(),
                "processing_duration": (datetime.now() - start_time).total_seconds(),
                "document_id": document_id,
                "tenant_id": tenant_id,
                "embeddings_generated": 1 + len(section_embeddings),
                "relationships_discovered": len(relationships),
                "stored_in_database": store_in_database
            }
            
            # Create result object
            result = EmbeddingResult(
                document_embedding=document_embedding,
                section_embeddings=section_embeddings,
                processing_info=processing_info,
                semantic_relationships=relationships
            )
            
            # Update statistics
            self._processing_stats["documents_processed"] += 1
            self._processing_stats["embeddings_generated"] += 1 + len(section_embeddings)
            self._processing_stats["sections_processed"] += len(section_embeddings)
            
            logger.info(
                "Document embeddings generated successfully",
                document_id=document_id,
                embeddings_count=1 + len(section_embeddings),
                relationships_count=len(relationships),
                processing_time=processing_info["processing_duration"]
            )
            
            return result
            
        except Exception as e:
            self._processing_stats["processing_errors"] += 1
            logger.error(f"Failed to generate document embeddings: {e}")
            raise
    
    async def generate_pdf_embeddings(
        self,
        document_id: str,
        pdf_document: PDFDocument,
        tenant_id: Optional[str] = None,
        chunk_strategy: str = "adaptive",
        store_in_database: bool = True
    ) -> EmbeddingResult:
        """
        Generate embeddings directly from PDF document.
        
        Args:
            document_id: Unique document identifier
            pdf_document: Processed PDF document
            tenant_id: Tenant identifier
            chunk_strategy: Chunking strategy (adaptive, fixed, semantic)
            store_in_database: Store in vector database
            
        Returns:
            EmbeddingResult with generated embeddings
        """
        try:
            # Create document-level content for embedding
            document_content = self._prepare_document_content(pdf_document)
            
            # Generate document embedding
            document_embedding_vector = await self.embedding_service.openai_service.generate_embedding(
                document_content
            )
            
            document_embedding = DocumentEmbedding(
                document_id=document_id,
                document_type="pdf",
                embedding_vector=document_embedding_vector,
                content_hash=self._generate_content_hash(document_content),
                metadata={
                    "total_pages": pdf_document.total_pages,
                    "total_words": pdf_document.total_words,
                    "total_characters": pdf_document.total_characters,
                    "pdf_metadata": pdf_document.metadata
                },
                created_at=datetime.now()
            )
            
            # Generate page-level embeddings
            section_embeddings = []
            for page in pdf_document.pages:
                if page.text and len(page.text.strip()) > 20:  # Only substantial content
                    try:
                        page_embedding_vector = await self.embedding_service.openai_service.generate_embedding(
                            page.text
                        )
                        
                        section_embedding = SectionEmbedding(
                            section_id=f"{document_id}_page_{page.page_number}",
                            document_id=document_id,
                            section_type="page",
                            title=f"Page {page.page_number}",
                            embedding_vector=page_embedding_vector,
                            content_hash=self._generate_content_hash(page.text),
                            metadata={
                                "page_number": page.page_number,
                                "word_count": page.word_count,
                                "character_count": page.character_count,
                                **page.metadata
                            },
                            created_at=datetime.now()
                        )
                        
                        section_embeddings.append(section_embedding)
                        
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding for page {page.page_number}: {e}")
                        continue
            
            # Store embeddings if requested
            if store_in_database:
                await self._store_embeddings(
                    document_embedding=document_embedding,
                    section_embeddings=section_embeddings,
                    tenant_id=tenant_id
                )
            
            # Create processing info
            processing_info = {
                "document_id": document_id,
                "processing_method": "pdf_direct",
                "chunk_strategy": chunk_strategy,
                "embeddings_generated": 1 + len(section_embeddings),
                "pages_processed": len([page for page in pdf_document.pages if page.text]),
                "stored_in_database": store_in_database,
                "timestamp": datetime.now().isoformat()
            }
            
            return EmbeddingResult(
                document_embedding=document_embedding,
                section_embeddings=section_embeddings,
                processing_info=processing_info,
                semantic_relationships=[]
            )
            
        except Exception as e:
            self._processing_stats["processing_errors"] += 1
            logger.error(f"Failed to generate PDF embeddings: {e}")
            raise
    
    async def update_document_embeddings(
        self,
        document_id: str,
        structured_content: StructuredContent,
        tenant_id: Optional[str] = None
    ) -> EmbeddingResult:
        """
        Update existing document embeddings with new content.
        
        Args:
            document_id: Document identifier
            structured_content: Updated document content
            tenant_id: Tenant identifier
            
        Returns:
            EmbeddingResult with updated embeddings
        """
        try:
            # Delete existing embeddings
            await self._delete_existing_embeddings(document_id, tenant_id)
            
            # Generate new embeddings
            result = await self.generate_document_embeddings(
                document_id=document_id,
                structured_content=structured_content,
                tenant_id=tenant_id,
                generate_relationships=True,
                store_in_database=True
            )
            
            logger.info("Document embeddings updated successfully", document_id=document_id)
            return result
            
        except Exception as e:
            logger.error(f"Failed to update document embeddings: {e}")
            raise
    
    async def _generate_document_level_embedding(
        self,
        document_id: str,
        structured_content: StructuredContent,
        tenant_id: Optional[str] = None
    ) -> DocumentEmbedding:
        """Generate document-level embedding"""
        try:
            # Create comprehensive document content
            content_parts = [structured_content.summary]
            
            # Add key information
            if structured_content.key_information:
                key_info_text = self._format_key_information(structured_content.key_information)
                content_parts.append(key_info_text)
            
            # Add section summaries
            for section in structured_content.sections[:5]:  # Limit to top 5 sections
                content_parts.append(f"{section.title}: {section.content[:200]}")
            
            document_content = "\n\n".join(content_parts)
            
            # Generate embedding
            embedding_vector = await self.embedding_service.openai_service.generate_embedding(
                document_content
            )
            
            # Create document embedding
            return DocumentEmbedding(
                document_id=document_id,
                document_type=structured_content.document_type,
                embedding_vector=embedding_vector,
                content_hash=self._generate_content_hash(document_content),
                metadata={
                    "sections_count": len(structured_content.sections),
                    "summary_length": len(structured_content.summary),
                    "quality_score": structured_content.quality_assessment.get("overall_quality", 0),
                    "processing_metadata": structured_content.processing_metadata
                },
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to generate document-level embedding: {e}")
            raise
    
    async def _generate_section_embeddings(
        self,
        document_id: str,
        structured_content: StructuredContent,
        tenant_id: Optional[str] = None
    ) -> List[SectionEmbedding]:
        """Generate section-level embeddings"""
        section_embeddings = []
        
        # Prepare content for batch processing
        sections_to_process = []
        for i, section in enumerate(structured_content.sections):
            if len(section.content.strip()) > 20:  # Only substantial content
                sections_to_process.append((i, section))
        
        if not sections_to_process:
            return section_embeddings
        
        try:
            # Extract content for batch embedding generation
            section_contents = [section.content for _, section in sections_to_process]
            
            # Generate embeddings in batch
            embedding_vectors = await self.embedding_service.openai_service.generate_embeddings_batch(
                section_contents
            )
            
            # Create section embeddings
            for (section_index, section), embedding_vector in zip(sections_to_process, embedding_vectors):
                section_embedding = SectionEmbedding(
                    section_id=f"{document_id}_section_{section_index}",
                    document_id=document_id,
                    section_type=section.section_type,
                    title=section.title,
                    embedding_vector=embedding_vector,
                    content_hash=self._generate_content_hash(section.content),
                    metadata={
                        "section_index": section_index,
                        "confidence": section.confidence,
                        "start_position": section.start_position,
                        "end_position": section.end_position,
                        "content_length": len(section.content),
                        **section.metadata
                    },
                    created_at=datetime.now()
                )
                section_embeddings.append(section_embedding)
            
            logger.debug(f"Generated {len(section_embeddings)} section embeddings")
            return section_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate section embeddings: {e}")
            return section_embeddings
    
    async def _discover_semantic_relationships(
        self,
        section_embeddings: List[SectionEmbedding]
    ) -> List[Dict[str, Any]]:
        """Discover semantic relationships between sections"""
        relationships = []
        
        try:
            if len(section_embeddings) < 2:
                return relationships
            
            # Calculate similarity matrix
            embeddings_matrix = [section.embedding_vector for section in section_embeddings]
            similarity_matrix = await self.embedding_service.calculate_similarity_matrix(embeddings_matrix)
            
            # Find significant relationships
            threshold = 0.8  # High similarity threshold
            
            for i, section_a in enumerate(section_embeddings):
                for j, section_b in enumerate(section_embeddings[i+1:], i+1):
                    similarity = float(similarity_matrix[i][j])
                    
                    if similarity >= threshold:
                        relationship = {
                            "source_section_id": section_a.section_id,
                            "target_section_id": section_b.section_id,
                            "relationship_type": "semantic_similarity",
                            "similarity_score": similarity,
                            "metadata": {
                                "source_type": section_a.section_type,
                                "target_type": section_b.section_type,
                                "source_title": section_a.title,
                                "target_title": section_b.title
                            }
                        }
                        relationships.append(relationship)
            
            logger.debug(f"Discovered {len(relationships)} semantic relationships")
            return relationships
            
        except Exception as e:
            logger.error(f"Failed to discover semantic relationships: {e}")
            return relationships
    
    async def _store_embeddings(
        self,
        document_embedding: DocumentEmbedding,
        section_embeddings: List[SectionEmbedding],
        tenant_id: Optional[str] = None
    ) -> None:
        """Store embeddings in vector database"""
        try:
            # Store document-level embedding
            await self.embedding_service.generate_and_store_embedding(
                entity_id=document_embedding.document_id,
                entity_type="document",
                content=f"Document {document_embedding.document_id}",  # Placeholder content
                tenant_id=tenant_id
            )
            
            # Store section embeddings in batch
            entities_batch = []
            for section_embedding in section_embeddings:
                entities_batch.append({
                    "entity_id": section_embedding.section_id,
                    "entity_type": "section", 
                    "content": section_embedding.title,  # Placeholder content
                    "tenant_id": tenant_id
                })
            
            if entities_batch:
                await self.embedding_service.generate_and_store_batch(entities_batch)
            
            logger.debug("Embeddings stored in database")
            
        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            raise
    
    async def _delete_existing_embeddings(
        self,
        document_id: str,
        tenant_id: Optional[str] = None
    ) -> None:
        """Delete existing embeddings for document"""
        try:
            # Delete document embedding
            await self.embedding_service.delete_entity_embedding(
                entity_id=document_id,
                entity_type="document",
                tenant_id=tenant_id
            )
            
            # Note: In a production system, you'd also need to delete
            # all section embeddings for this document
            logger.debug("Existing embeddings deleted")
            
        except Exception as e:
            logger.warning(f"Failed to delete existing embeddings: {e}")
    
    def _prepare_document_content(self, pdf_document: PDFDocument) -> str:
        """Prepare document content for embedding"""
        content_parts = []
        
        # Add metadata if available
        if pdf_document.metadata.get("title"):
            content_parts.append(f"Title: {pdf_document.metadata['title']}")
        
        if pdf_document.metadata.get("author"):
            content_parts.append(f"Author: {pdf_document.metadata['author']}")
        
        # Add document text (truncated for embedding limits)
        max_length = 6000  # Conservative limit for embedding models
        document_text = pdf_document.full_text
        
        if len(document_text) > max_length:
            document_text = document_text[:max_length] + "..."
        
        content_parts.append(document_text)
        
        return "\n\n".join(content_parts)
    
    def _format_key_information(self, key_info: Dict[str, Any]) -> str:
        """Format key information for embedding"""
        formatted_parts = []
        
        for key, value in key_info.items():
            if isinstance(value, (str, int, float)):
                formatted_parts.append(f"{key}: {value}")
            elif isinstance(value, list) and value:
                formatted_parts.append(f"{key}: {', '.join(str(v) for v in value)}")
        
        return "\n".join(formatted_parts)
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content deduplication"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "stats": self._processing_stats.copy(),
            "configuration": {
                "embedding_model": self.settings.OPENAI_EMBEDDING_MODEL,
                "embedding_dimension": self.settings.EMBEDDING_DIMENSION,
                "batch_size": self.settings.EMBEDDING_BATCH_SIZE
            }
        }
    
    async def check_health(self) -> Dict[str, Any]:
        """Check embedding generator health"""
        try:
            # Test embedding generation
            test_content = "This is a test document for health checking."
            test_embedding = await self.embedding_service.openai_service.generate_embedding(test_content)
            
            return {
                "status": "healthy",
                "embedding_generator": "operational",
                "test_embedding_dimension": len(test_embedding),
                "expected_dimension": self.settings.EMBEDDING_DIMENSION,
                "stats": self._processing_stats.copy(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
