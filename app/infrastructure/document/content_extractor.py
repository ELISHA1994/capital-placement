"""
Content Extractor with LangChain Integration

Advanced content analysis and structured extraction:
- LangChain integration for document processing
- Structured content extraction with AI analysis
- Text chunking for optimal processing
- Content classification and categorization
- Information extraction with confidence scoring
"""

import re
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
import structlog
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.infrastructure.ai.openai_service import OpenAIService
from app.infrastructure.ai.prompt_manager import PromptManager, PromptType
from app.infrastructure.document.pdf_processor import PDFDocument

logger = structlog.get_logger(__name__)


@dataclass
class ExtractedSection:
    """Represents an extracted document section"""
    section_type: str
    title: str
    content: str
    confidence: float
    metadata: Dict[str, Any]
    start_position: int
    end_position: int


@dataclass
class StructuredContent:
    """Represents structured content extracted from document"""
    document_type: str
    sections: List[ExtractedSection]
    summary: str
    key_information: Dict[str, Any]
    quality_assessment: Dict[str, Any]
    processing_metadata: Dict[str, Any]


class ContentExtractor:
    """
    Advanced content extraction service with LangChain integration.

    Features:
    - Intelligent document chunking and processing
    - AI-powered content analysis and extraction
    - Structured information extraction with confidence scoring
    - Content classification and categorization
    - Multi-format document support
    - Quality assessment and validation
    """

    def __init__(
        self,
        openai_service: OpenAIService,
        prompt_manager: PromptManager
    ):
        self.settings = get_settings()
        self.openai_service = openai_service
        self.prompt_manager = prompt_manager

        # Initialize LangChain components
        self._init_langchain_components()

        # Processing statistics
        self._stats = {
            "documents_processed": 0,
            "sections_extracted": 0,
            "ai_analysis_calls": 0,
            "extraction_errors": 0
        }

    def _init_langchain_components(self):
        """Initialize LangChain components"""
        try:
            # Get OpenAI configuration
            openai_config = self.settings.get_openai_config()

            # Initialize text splitter for chunking
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.settings.DOCUMENT_CHUNK_SIZE,
                chunk_overlap=self.settings.DOCUMENT_CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )

            # Initialize ChatOpenAI if using OpenAI provider
            if openai_config["provider"] == "openai":
                self.chat_model = ChatOpenAI(
                    api_key=openai_config["api_key"],
                    model=openai_config["model"],
                    temperature=openai_config["temperature"],
                    max_tokens=openai_config["max_tokens"]
                )
            else:
                # For Azure OpenAI, we'll use our OpenAI service directly
                self.chat_model = None

            logger.info("LangChain components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize LangChain components: {e}")
            raise

    async def extract_structured_content(
        self,
        pdf_document: PDFDocument,
        content_type: str = "cv",
        extract_sections: bool = True,
        analyze_quality: bool = True
    ) -> StructuredContent:
        """
        Extract structured content from PDF document.

        Args:
            pdf_document: Processed PDF document
            content_type: Type of content (cv, job_description, etc.)
            extract_sections: Extract document sections
            analyze_quality: Perform quality analysis

        Returns:
            StructuredContent object with extracted information
        """
        start_time = datetime.now()

        try:
            # Create LangChain documents from PDF pages
            documents = await self._create_langchain_documents(pdf_document)

            # Chunk documents for processing
            chunked_docs = await self._chunk_documents(documents)

            # Extract sections if requested
            sections = []
            if extract_sections:
                sections = await self._extract_sections(chunked_docs, content_type)

            # Generate document summary
            summary = await self._generate_summary(pdf_document.full_text, content_type)

            # Extract key information
            key_info = await self._extract_key_information(pdf_document.full_text, content_type)

            # Perform quality assessment if requested
            quality_assessment = {}
            if analyze_quality:
                quality_assessment = await self._assess_content_quality(
                    pdf_document.full_text,
                    sections,
                    content_type
                )

            # Create structured content object
            processing_metadata = {
                "extraction_started": start_time.isoformat(),
                "extraction_completed": datetime.now().isoformat(),
                "processing_duration": (datetime.now() - start_time).total_seconds(),
                "content_type": content_type,
                "pages_processed": pdf_document.total_pages,
                "chunks_created": len(chunked_docs),
                "sections_found": len(sections)
            }

            structured_content = StructuredContent(
                document_type=content_type,
                sections=sections,
                summary=summary,
                key_information=key_info,
                quality_assessment=quality_assessment,
                processing_metadata=processing_metadata
            )

            # Update statistics
            self._stats["documents_processed"] += 1
            self._stats["sections_extracted"] += len(sections)

            logger.info(
                "Content extraction completed",
                content_type=content_type,
                sections=len(sections),
                processing_time=processing_metadata["processing_duration"]
            )

            return structured_content

        except Exception as e:
            self._stats["extraction_errors"] += 1
            logger.error(f"Content extraction failed: {e}")
            raise

    async def extract_from_text(
        self,
        text: str,
        content_type: str = "generic",
        extract_sections: bool = True,
        analyze_quality: bool = True
    ) -> StructuredContent:
        """
        Extract structured content from raw text.

        Args:
            text: Raw text content
            content_type: Type of content
            extract_sections: Extract document sections
            analyze_quality: Perform quality analysis

        Returns:
            StructuredContent object with extracted information
        """
        if not text or not text.strip():
            raise ValueError("Text content cannot be empty")

        start_time = datetime.now()

        try:
            # Create LangChain document
            documents = [Document(page_content=text, metadata={"source": "text_input"})]

            # Chunk documents
            chunked_docs = await self._chunk_documents(documents)

            # Extract sections if requested
            sections = []
            if extract_sections:
                sections = await self._extract_sections(chunked_docs, content_type)

            # Generate summary
            summary = await self._generate_summary(text, content_type)

            # Extract key information
            key_info = await self._extract_key_information(text, content_type)

            # Quality assessment
            quality_assessment = {}
            if analyze_quality:
                quality_assessment = await self._assess_content_quality(
                    text,
                    sections,
                    content_type
                )

            # Create structured content
            processing_metadata = {
                "extraction_started": start_time.isoformat(),
                "extraction_completed": datetime.now().isoformat(),
                "processing_duration": (datetime.now() - start_time).total_seconds(),
                "content_type": content_type,
                "text_length": len(text),
                "chunks_created": len(chunked_docs),
                "sections_found": len(sections)
            }

            return StructuredContent(
                document_type=content_type,
                sections=sections,
                summary=summary,
                key_information=key_info,
                quality_assessment=quality_assessment,
                processing_metadata=processing_metadata
            )

        except Exception as e:
            self._stats["extraction_errors"] += 1
            logger.error(f"Text extraction failed: {e}")
            raise

    async def _create_langchain_documents(self, pdf_document: PDFDocument) -> List[Document]:
        """Create LangChain documents from PDF pages"""
        documents = []

        for page in pdf_document.pages:
            if page.text and page.text.strip():
                metadata = {
                    "source": "pdf",
                    "page_number": page.page_number,
                    "word_count": page.word_count,
                    "character_count": page.character_count
                }
                metadata.update(page.metadata)

                doc = Document(
                    page_content=page.text,
                    metadata=metadata
                )
                documents.append(doc)

        return documents

    async def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents for optimal processing"""
        try:
            chunked_docs = []

            for doc in documents:
                chunks = self.text_splitter.split_documents([doc])

                # Add chunk metadata
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunk_size": len(chunk.page_content)
                    })

                chunked_docs.extend(chunks)

            logger.debug(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
            return chunked_docs

        except Exception as e:
            logger.error(f"Document chunking failed: {e}")
            raise

    async def _extract_sections(
        self,
        chunked_docs: List[Document],
        content_type: str
    ) -> List[ExtractedSection]:
        """Extract document sections using AI analysis"""
        sections = []

        try:
            # Combine chunks for section analysis
            full_content = "\n\n".join([doc.page_content for doc in chunked_docs])

            # Generate prompt for section extraction
            prompt = await self.prompt_manager.generate_prompt(
                PromptType.CONTENT_CLASSIFICATION,
                {"content": full_content[:3000]},  # Limit for prompt size
                custom_instructions=f"Extract and identify sections in this {content_type} document"
            )

            # Get AI analysis
            response = await self.openai_service.chat_completion(
                messages=prompt["messages"],
                temperature=prompt["temperature"],
                max_tokens=prompt["max_tokens"]
            )

            self._stats["ai_analysis_calls"] += 1

            # Parse response and create sections
            analysis_text = response["choices"][0]["message"]["content"]
            sections = await self._parse_section_analysis(analysis_text, full_content)

            logger.debug(f"Extracted {len(sections)} sections")

        except Exception as e:
            logger.error(f"Section extraction failed: {e}")
            # Continue without sections rather than failing completely

        return sections

    async def _generate_summary(self, text: str, content_type: str) -> str:
        """Generate document summary using AI"""
        try:
            # Truncate text if too long
            max_text_length = 4000
            truncated_text = text[:max_text_length] if len(text) > max_text_length else text

            # Generate summary prompt
            prompt = await self.prompt_manager.generate_prompt(
                PromptType.DOCUMENT_SUMMARY,
                {"document_content": truncated_text}
            )

            # Get AI summary
            response = await self.openai_service.chat_completion(
                messages=prompt["messages"],
                temperature=prompt["temperature"],
                max_tokens=prompt["max_tokens"]
            )

            self._stats["ai_analysis_calls"] += 1

            summary_response = response["choices"][0]["message"]["content"]

            # Try to extract structured summary
            try:
                summary_data = json.loads(summary_response)
                return summary_data.get("executive_summary", summary_response)
            except json.JSONDecodeError:
                return summary_response

        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return f"Summary unavailable for {content_type} document"

    async def _extract_key_information(self, text: str, content_type: str) -> Dict[str, Any]:
        """Extract key information based on content type"""
        try:
            if content_type.lower() == "cv":
                return await self._extract_cv_information(text)
            elif content_type.lower() == "job_description":
                return await self._extract_job_information(text)
            else:
                return await self._extract_generic_information(text)
        except Exception as e:
            logger.error(f"Key information extraction failed: {e}")
            return {"extraction_error": str(e)}

    async def _extract_cv_information(self, text: str) -> Dict[str, Any]:
        """Extract CV-specific information"""
        try:
            prompt = await self.prompt_manager.generate_prompt(
                PromptType.CV_ANALYSIS,
                {"cv_content": text[:5000]}  # Limit for prompt size
            )

            response = await self.openai_service.chat_completion(
                messages=prompt["messages"],
                temperature=prompt["temperature"],
                max_tokens=prompt["max_tokens"]
            )

            self._stats["ai_analysis_calls"] += 1

            # Parse structured CV information
            analysis_text = response["choices"][0]["message"]["content"]

            try:
                return json.loads(analysis_text)
            except json.JSONDecodeError:
                # Fallback to basic extraction
                return {"raw_analysis": analysis_text}

        except Exception as e:
            logger.error(f"CV information extraction failed: {e}")
            return {"error": str(e)}

    async def _extract_job_information(self, text: str) -> Dict[str, Any]:
        """Extract job description information"""
        try:
            # Use skill extraction for job descriptions
            prompt = await self.prompt_manager.generate_prompt(
                PromptType.SKILL_EXTRACTION,
                {"text_content": text[:5000]}
            )

            response = await self.openai_service.chat_completion(
                messages=prompt["messages"],
                temperature=prompt["temperature"],
                max_tokens=prompt["max_tokens"]
            )

            self._stats["ai_analysis_calls"] += 1

            analysis_text = response["choices"][0]["message"]["content"]

            try:
                job_info = json.loads(analysis_text)
                job_info["document_type"] = "job_description"
                return job_info
            except json.JSONDecodeError:
                return {"raw_analysis": analysis_text}

        except Exception as e:
            logger.error(f"Job information extraction failed: {e}")
            return {"error": str(e)}

    async def _extract_generic_information(self, text: str) -> Dict[str, Any]:
        """Extract generic document information"""
        # Basic information extraction without AI
        word_count = len(text.split())
        char_count = len(text)

        # Extract basic patterns
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        phones = re.findall(r'[\+]?[1-9]?[0-9]{7,15}', text)
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)

        return {
            "word_count": word_count,
            "character_count": char_count,
            "emails_found": emails,
            "phone_numbers_found": phones,
            "urls_found": urls,
            "extraction_method": "pattern_matching"
        }

    async def _assess_content_quality(
        self,
        text: str,
        sections: List[ExtractedSection],
        content_type: str
    ) -> Dict[str, Any]:
        """Assess content quality using AI"""
        try:
            prompt = await self.prompt_manager.generate_prompt(
                PromptType.QUALITY_ASSESSMENT,
                {"content": text[:3000]},  # Limit for prompt size
                custom_instructions=f"Assess quality for {content_type} document"
            )

            response = await self.openai_service.chat_completion(
                messages=prompt["messages"],
                temperature=prompt["temperature"],
                max_tokens=prompt["max_tokens"]
            )

            self._stats["ai_analysis_calls"] += 1

            assessment_text = response["choices"][0]["message"]["content"]

            try:
                quality_data = json.loads(assessment_text)
                quality_data["sections_count"] = len(sections)
                return quality_data
            except json.JSONDecodeError:
                return {
                    "raw_assessment": assessment_text,
                    "sections_count": len(sections)
                }

        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {
                "error": str(e),
                "sections_count": len(sections)
            }

    async def _parse_section_analysis(self, analysis_text: str, full_content: str) -> List[ExtractedSection]:
        """Parse AI section analysis into structured sections"""
        sections = []

        try:
            # Try to parse as JSON first
            try:
                analysis_data = json.loads(analysis_text)
                # Process structured data into sections
                # This would depend on the specific format returned by the AI
                pass
            except json.JSONDecodeError:
                # Parse text format
                pass

            # For now, create basic sections based on content structure
            # This is a simplified implementation
            paragraphs = full_content.split('\n\n')

            for i, paragraph in enumerate(paragraphs):
                if len(paragraph.strip()) > 50:  # Only substantial paragraphs
                    section = ExtractedSection(
                        section_type="paragraph",
                        title=f"Section {i+1}",
                        content=paragraph.strip(),
                        confidence=0.8,
                        metadata={"paragraph_index": i},
                        start_position=full_content.find(paragraph),
                        end_position=full_content.find(paragraph) + len(paragraph)
                    )
                    sections.append(section)

        except Exception as e:
            logger.error(f"Section parsing failed: {e}")

        return sections

    async def extract_cv_data(self, text_content: str) -> Dict[str, Any]:
        """
        Extract structured data from CV text.

        This method implements the IContentExtractor interface and provides
        backward compatibility with existing code.

        Args:
            text_content: Raw CV text content

        Returns:
            Dictionary with extracted CV data
        """
        if not text_content or not text_content.strip():
            logger.warning("Empty text content provided for CV extraction")
            return {}

        try:
            # Use the existing _extract_cv_information method
            cv_data = await self._extract_cv_information(text_content)

            logger.debug(
                "CV data extraction completed",
                has_data=bool(cv_data),
                data_keys=list(cv_data.keys()) if cv_data else []
            )

            return cv_data

        except Exception as e:
            logger.error(f"CV data extraction failed: {e}")
            self._stats["extraction_errors"] += 1
            return {"error": str(e), "extraction_failed": True}

    def prepare_text_for_embedding(
        self,
        text_content: str,
        structured_data: Dict[str, Any]
    ) -> str:
        """
        Prepare text for embedding generation by combining raw text with structured data.

        This creates an optimized representation that includes both the original content
        and key extracted information for better semantic search.

        Args:
            text_content: Original text content
            structured_data: Extracted structured data

        Returns:
            Prepared text string for embedding
        """
        try:
            # Start with the original text (truncated if needed)
            max_text_length = 6000
            prepared_text = text_content[:max_text_length] if len(text_content) > max_text_length else text_content

            # Add structured data for enrichment
            enrichment_parts = []

            # Add personal information
            if "personal_info" in structured_data:
                personal = structured_data["personal_info"]
                if isinstance(personal, dict):
                    if personal.get("name"):
                        enrichment_parts.append(f"Name: {personal['name']}")
                    if personal.get("email"):
                        enrichment_parts.append(f"Email: {personal['email']}")
                    if personal.get("location"):
                        enrichment_parts.append(f"Location: {personal['location']}")

            # Add skills
            if "skills" in structured_data:
                skills = structured_data["skills"]
                if isinstance(skills, list) and skills:
                    enrichment_parts.append(f"Skills: {', '.join(str(s) for s in skills[:20])}")
                elif isinstance(skills, dict) and skills.get("technical_skills"):
                    tech_skills = skills["technical_skills"]
                    if isinstance(tech_skills, list):
                        enrichment_parts.append(f"Technical Skills: {', '.join(str(s) for s in tech_skills[:20])}")

            # Add experience summary
            if "experience" in structured_data:
                experience = structured_data["experience"]
                if isinstance(experience, list) and experience:
                    exp_summaries = []
                    for exp in experience[:3]:  # Top 3 experiences
                        if isinstance(exp, dict):
                            title = exp.get("title", "")
                            company = exp.get("company", "")
                            if title and company:
                                exp_summaries.append(f"{title} at {company}")
                    if exp_summaries:
                        enrichment_parts.append(f"Experience: {'; '.join(exp_summaries)}")

            # Add education
            if "education" in structured_data:
                education = structured_data["education"]
                if isinstance(education, list) and education:
                    edu_summaries = []
                    for edu in education[:2]:  # Top 2 education entries
                        if isinstance(edu, dict):
                            degree = edu.get("degree", "")
                            institution = edu.get("institution", "")
                            if degree and institution:
                                edu_summaries.append(f"{degree} from {institution}")
                    if edu_summaries:
                        enrichment_parts.append(f"Education: {'; '.join(edu_summaries)}")

            # Combine everything
            if enrichment_parts:
                enrichment_text = "\n\n--- Extracted Information ---\n" + "\n".join(enrichment_parts)
                prepared_text = prepared_text + enrichment_text

            logger.debug(
                "Text prepared for embedding",
                original_length=len(text_content),
                prepared_length=len(prepared_text),
                enrichment_added=bool(enrichment_parts)
            )

            return prepared_text

        except Exception as e:
            logger.error(f"Text preparation for embedding failed: {e}")
            # Return original text as fallback
            return text_content[:8000]

    def hash_content(self, text: str) -> str:
        """
        Generate a hash of content for deduplication and caching.

        Args:
            text: Text content to hash

        Returns:
            SHA256 hash of the content
        """
        try:
            # Normalize text before hashing (lowercase, strip whitespace)
            normalized_text = " ".join(text.lower().split())

            # Generate SHA256 hash
            content_hash = hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()

            logger.debug(
                "Content hash generated",
                text_length=len(text),
                hash=content_hash[:16] + "..."
            )

            return content_hash

        except Exception as e:
            logger.error(f"Content hashing failed: {e}")
            # Return a basic hash as fallback
            return hashlib.sha256(text.encode('utf-8', errors='ignore')).hexdigest()

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "stats": self._stats.copy(),
            "configuration": {
                "chunk_size": self.settings.DOCUMENT_CHUNK_SIZE,
                "chunk_overlap": self.settings.DOCUMENT_CHUNK_OVERLAP,
                "langchain_enabled": True
            }
        }

    async def check_health(self) -> Dict[str, Any]:
        """Check content extractor health"""
        try:
            # Test with sample text
            test_text = "This is a test document for health checking the content extraction service."

            structured_content = await self.extract_from_text(
                text=test_text,
                content_type="test",
                extract_sections=False,
                analyze_quality=False
            )

            return {
                "status": "healthy",
                "extractor": "operational",
                "langchain": "initialized",
                "test_extraction": "successful",
                "stats": self._stats.copy(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


__all__ = ["ContentExtractor", "StructuredContent", "ExtractedSection"]