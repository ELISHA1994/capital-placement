"""
PDF Processor with Advanced Content Extraction

Comprehensive PDF processing for CV and document analysis:
- PDF parsing and text extraction
- Structure preservation and metadata extraction
- Multi-page document handling
- Error handling for corrupted or encrypted PDFs
- Content quality assessment and validation
"""

import io
import re
from typing import Dict, List, Optional, Any, Tuple, BinaryIO
from datetime import datetime
import structlog
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError, PdfReadWarning

from app.core.config import get_settings

logger = structlog.get_logger(__name__)


class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors"""
    pass


class PDFPage:
    """Represents a processed PDF page"""
    
    def __init__(self, page_number: int, text: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_number = page_number
        self.text = text
        self.metadata = metadata or {}
        self.word_count = len(text.split()) if text else 0
        self.character_count = len(text) if text else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "page_number": self.page_number,
            "text": self.text,
            "metadata": self.metadata,
            "word_count": self.word_count,
            "character_count": self.character_count
        }


class PDFDocument:
    """Represents a processed PDF document"""
    
    def __init__(
        self,
        pages: List[PDFPage],
        metadata: Optional[Dict[str, Any]] = None,
        processing_info: Optional[Dict[str, Any]] = None
    ):
        self.pages = pages
        self.metadata = metadata or {}
        self.processing_info = processing_info or {}
        self.full_text = "\n\n".join([page.text for page in pages if page.text])
        self.total_pages = len(pages)
        self.total_words = sum(page.word_count for page in pages)
        self.total_characters = sum(page.character_count for page in pages)
    
    def get_page(self, page_number: int) -> Optional[PDFPage]:
        """Get specific page by number"""
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None
    
    def get_pages_range(self, start: int, end: int) -> List[PDFPage]:
        """Get pages in range"""
        return [page for page in self.pages if start <= page.page_number <= end]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "pages": [page.to_dict() for page in self.pages],
            "metadata": self.metadata,
            "processing_info": self.processing_info,
            "full_text": self.full_text,
            "total_pages": self.total_pages,
            "total_words": self.total_words,
            "total_characters": self.total_characters
        }


class PDFProcessor:
    """
    Advanced PDF processor for document analysis and content extraction.
    
    Features:
    - Robust PDF parsing with error handling
    - Text extraction with structure preservation
    - Metadata extraction and document analysis
    - Multi-page processing optimization
    - Content quality assessment
    - Support for encrypted and corrupted PDFs
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._processing_stats = {
            "documents_processed": 0,
            "pages_extracted": 0,
            "errors_encountered": 0,
            "warnings_encountered": 0
        }
    
    async def process_pdf(
        self,
        pdf_content: bytes,
        filename: Optional[str] = None,
        extract_metadata: bool = True,
        validate_content: bool = True
    ) -> PDFDocument:
        """
        Process PDF document and extract content.
        
        Args:
            pdf_content: Raw PDF bytes
            filename: Original filename for context
            extract_metadata: Extract document metadata
            validate_content: Validate extracted content quality
            
        Returns:
            PDFDocument object with extracted content
        """
        if not pdf_content:
            raise PDFProcessingError("PDF content is empty")
        
        if len(pdf_content) > self.settings.MAX_DOCUMENT_SIZE:
            raise PDFProcessingError(
                f"PDF size ({len(pdf_content)} bytes) exceeds maximum allowed "
                f"({self.settings.MAX_DOCUMENT_SIZE} bytes)"
            )
        
        start_time = datetime.now()
        
        try:
            # Create PDF reader from bytes
            pdf_stream = io.BytesIO(pdf_content)
            pdf_reader = PdfReader(pdf_stream)
            
            # Extract basic metadata
            metadata = {}
            processing_info = {
                "filename": filename,
                "file_size": len(pdf_content),
                "processing_started": start_time.isoformat(),
                "processor": "PyPDF2"
            }
            
            if extract_metadata:
                metadata = await self._extract_metadata(pdf_reader)
            
            # Check for encryption
            if pdf_reader.is_encrypted:
                # Try to decrypt with empty password
                try:
                    pdf_reader.decrypt("")
                except Exception:
                    raise PDFProcessingError("PDF is password protected and cannot be processed")
            
            # Process pages
            pages = await self._extract_pages(pdf_reader)
            
            # Create document object
            processing_info.update({
                "processing_completed": datetime.now().isoformat(),
                "processing_duration": (datetime.now() - start_time).total_seconds(),
                "pages_processed": len(pages),
                "extraction_method": "standard"
            })
            
            document = PDFDocument(
                pages=pages,
                metadata=metadata,
                processing_info=processing_info
            )
            
            # Validate content if requested
            if validate_content:
                validation_results = await self._validate_content(document)
                document.processing_info["validation"] = validation_results
            
            self._processing_stats["documents_processed"] += 1
            self._processing_stats["pages_extracted"] += len(pages)
            
            logger.info(
                "PDF processed successfully",
                filename=filename,
                pages=len(pages),
                words=document.total_words,
                processing_time=processing_info["processing_duration"]
            )
            
            return document
            
        except PdfReadError as e:
            self._processing_stats["errors_encountered"] += 1
            logger.error(f"PDF read error: {e}", filename=filename)
            raise PDFProcessingError(f"Failed to read PDF: {e}")
        
        except Exception as e:
            self._processing_stats["errors_encountered"] += 1
            logger.error(f"PDF processing error: {e}", filename=filename)
            raise PDFProcessingError(f"PDF processing failed: {e}")
    
    async def process_pdf_file(
        self,
        file_path: str,
        extract_metadata: bool = True,
        validate_content: bool = True
    ) -> PDFDocument:
        """
        Process PDF file from file path.
        
        Args:
            file_path: Path to PDF file
            extract_metadata: Extract document metadata
            validate_content: Validate extracted content quality
            
        Returns:
            PDFDocument object with extracted content
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_content = file.read()
            
            filename = file_path.split('/')[-1]
            return await self.process_pdf(
                pdf_content=pdf_content,
                filename=filename,
                extract_metadata=extract_metadata,
                validate_content=validate_content
            )
            
        except FileNotFoundError:
            raise PDFProcessingError(f"PDF file not found: {file_path}")
        except PermissionError:
            raise PDFProcessingError(f"Permission denied reading PDF file: {file_path}")
    
    async def _extract_metadata(self, pdf_reader: PdfReader) -> Dict[str, Any]:
        """Extract metadata from PDF"""
        metadata = {}
        
        try:
            if pdf_reader.metadata:
                # Standard metadata fields
                metadata_fields = {
                    '/Title': 'title',
                    '/Author': 'author',
                    '/Subject': 'subject',
                    '/Creator': 'creator',
                    '/Producer': 'producer',
                    '/CreationDate': 'creation_date',
                    '/ModDate': 'modification_date',
                    '/Keywords': 'keywords'
                }
                
                for pdf_key, standard_key in metadata_fields.items():
                    if pdf_key in pdf_reader.metadata:
                        value = pdf_reader.metadata[pdf_key]
                        
                        # Convert datetime objects to strings
                        if hasattr(value, 'isoformat'):
                            value = value.isoformat()
                        elif isinstance(value, str):
                            value = value.strip()
                        
                        if value:
                            metadata[standard_key] = value
            
            # Additional document info
            metadata.update({
                'page_count': len(pdf_reader.pages),
                'is_encrypted': pdf_reader.is_encrypted,
                'pdf_version': getattr(pdf_reader, 'pdf_version', 'unknown')
            })
            
        except Exception as e:
            logger.warning(f"Failed to extract PDF metadata: {e}")
            self._processing_stats["warnings_encountered"] += 1
        
        return metadata
    
    async def _extract_pages(self, pdf_reader: PdfReader) -> List[PDFPage]:
        """Extract text from all pages"""
        pages = []
        
        for page_num, page in enumerate(pdf_reader.pages, 1):
            try:
                # Extract text from page
                text = page.extract_text()
                
                # Clean and normalize text
                cleaned_text = self._clean_text(text)
                
                # Extract page metadata
                page_metadata = {
                    'rotation': getattr(page, 'rotation', 0),
                    'mediabox': str(getattr(page, 'mediabox', 'unknown'))
                }
                
                # Create page object
                pdf_page = PDFPage(
                    page_number=page_num,
                    text=cleaned_text,
                    metadata=page_metadata
                )
                
                pages.append(pdf_page)
                
                logger.debug(
                    "Page extracted",
                    page_number=page_num,
                    words=pdf_page.word_count,
                    characters=pdf_page.character_count
                )
                
            except Exception as e:
                logger.warning(f"Failed to extract page {page_num}: {e}")
                self._processing_stats["warnings_encountered"] += 1
                
                # Create empty page to maintain page numbering
                pages.append(PDFPage(
                    page_number=page_num,
                    text="",
                    metadata={"extraction_error": str(e)}
                ))
        
        return pages
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-printable characters except common ones
        text = re.sub(r'[^\x20-\x7E\n\t]', '', text)
        
        # Fix common PDF extraction issues
        text = text.replace('\x00', '')  # Remove null characters
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Fix multiple newlines
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    async def _validate_content(self, document: PDFDocument) -> Dict[str, Any]:
        """Validate extracted content quality"""
        validation = {
            "is_valid": True,
            "issues": [],
            "quality_score": 100,
            "recommendations": []
        }
        
        # Check for minimum content
        if document.total_words < 10:
            validation["issues"].append("Document contains very little text content")
            validation["quality_score"] -= 30
        
        # Check for excessive special characters (OCR artifacts)
        text = document.full_text
        if text:
            special_char_ratio = len(re.findall(r'[^\w\s]', text)) / len(text)
            if special_char_ratio > 0.3:
                validation["issues"].append("High ratio of special characters detected")
                validation["quality_score"] -= 20
        
        # Check for repeated patterns (scanning artifacts)
        if text and len(set(text.split())) < len(text.split()) * 0.3:
            validation["issues"].append("High repetition detected in content")
            validation["quality_score"] -= 15
        
        # Check page distribution
        non_empty_pages = [page for page in document.pages if page.word_count > 5]
        if len(non_empty_pages) < document.total_pages * 0.7:
            validation["issues"].append("Many pages contain little or no text")
            validation["quality_score"] -= 10
        
        # Set overall validity
        validation["is_valid"] = validation["quality_score"] >= 50
        
        # Generate recommendations
        if validation["quality_score"] < 80:
            validation["recommendations"].append("Consider using OCR for scanned documents")
        
        if not validation["is_valid"]:
            validation["recommendations"].append("Manual review recommended")
        
        return validation
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "stats": self._processing_stats.copy(),
            "configuration": {
                "max_document_size": self.settings.MAX_DOCUMENT_SIZE,
                "processor_version": "PyPDF2"
            }
        }
    
    async def check_health(self) -> Dict[str, Any]:
        """Check processor health"""
        try:
            # Test with a minimal PDF
            test_pdf = self._create_test_pdf()
            document = await self.process_pdf(
                pdf_content=test_pdf,
                filename="health_check.pdf",
                extract_metadata=False,
                validate_content=False
            )
            
            return {
                "status": "healthy",
                "processor": "operational",
                "test_processing": "successful",
                "stats": self._processing_stats.copy(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _create_test_pdf(self) -> bytes:
        """Create a minimal test PDF for health checks"""
        # This would typically create a minimal PDF programmatically
        # For now, return a simple placeholder
        # In production, you might want to include a minimal PDF binary
        return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\nxref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n0000000074 00000 n \ntrailer\n<<\n/Size 3\n/Root 1 0 R\n>>\nstartxref\n128\n%%EOF"