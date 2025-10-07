"""
Document Processor Service

Comprehensive document processing service with:
- PDF processing and structure extraction
- Intelligent CV parsing with section detection
- Multi-language support with confidence scoring
- Error handling and fallback mechanisms
- Performance optimization and parallel processing
- Quality assessment and validation
"""

import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, BinaryIO
import structlog


    CVProfile, ContactInfo, Skill, Experience as ExperienceEntry, Education as EducationEntry,
    Certification, Project, Language, ProcessingStatus, ProcessingMetadata
)
from app.services.azure.document_intelligence_service import AzureDocumentIntelligenceService
from app.services.azure.blob_service import AzureBlobService

logger = structlog.get_logger(__name__)


class DocumentProcessor:
    """
    Production-ready document processing service for CV analysis.
    
    Provides comprehensive document processing with:
    - Advanced PDF parsing with layout analysis
    - Intelligent CV structure extraction
    - Multi-language support and confidence scoring
    - Quality assessment and validation
    - Parallel processing for performance
    - Error handling with detailed diagnostics
    """
    
    def __init__(
        self,
        document_intelligence_service: Optional[AzureDocumentIntelligenceService] = None,
        blob_service: Optional[AzureBlobService] = None
    ):
        self.doc_intelligence = document_intelligence_service or AzureDocumentIntelligenceService()
        self.blob_service = blob_service
        self._processing_stats = {
            "documents_processed": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "average_processing_time": 0.0
        }
        
        # Quality thresholds
        self._quality_thresholds = {
            "min_confidence": 0.7,
            "min_text_length": 100,
            "required_sections": ["contact", "experience"],
            "min_skills_count": 1
        }
        
        # Processing configuration
        self._processing_config = {
            "max_document_size_mb": 50,
            "supported_formats": [".pdf", ".doc", ".docx"],
            "processing_timeout_minutes": 10,
            "enable_parallel_processing": True,
            "max_concurrent_documents": 5
        }
    
    async def process_document(
        self,
        document_content: bytes,
        filename: str,
        tenant_id: str,
        user_id: Optional[str] = None,
        processing_options: Optional[Dict[str, Any]] = None
    ) -> CVProfile:
        """
        Process a CV document and extract structured profile data.
        
        Args:
            document_content: Document content as bytes
            filename: Original filename
            tenant_id: Tenant identifier
            user_id: User who uploaded the document
            processing_options: Optional processing configuration
            
        Returns:
            Structured CV profile with extracted data
            
        Raises:
            ValueError: If document is invalid or unsupported
            RuntimeError: If processing fails
        """
        start_time = datetime.now()
        
        try:
            # Validate document
            self._validate_document(document_content, filename)
            
            # Initialize profile with metadata
            profile = await self._initialize_profile(
                document_content, filename, tenant_id, user_id
            )
            
            # Update processing status
            profile.processing.processing_status = ProcessingStatus.PROCESSING
            profile.processing.processing_started_at = start_time
            
            logger.info(
                "Starting document processing",
                filename=filename,
                size_bytes=len(document_content),
                tenant_id=tenant_id
            )
            
            # Extract document structure and content
            extraction_result = await self._extract_document_content(
                document_content, filename, processing_options
            )
            
            # Parse CV structure from extracted content
            cv_structure = await self._parse_cv_structure(extraction_result)
            
            # Populate profile with structured data
            await self._populate_profile_data(profile, cv_structure, extraction_result)
            
            # Validate and assess quality
            quality_assessment = await self._assess_quality(profile, extraction_result)
            profile.processing.quality_score = quality_assessment["overall_score"]
            profile.processing.confidence_score = quality_assessment["confidence_score"]
            
            # Update computed fields and search content
            profile.update_computed_fields()
            
            # Store original document if blob service available
            if self.blob_service:
                await self._store_original_document(profile, document_content, filename)
            
            # Finalize processing
            processing_duration = (datetime.now() - start_time).total_seconds()
            profile.processing.processing_completed_at = datetime.now()
            profile.processing.processing_duration_seconds = processing_duration
            profile.processing.processing_status = ProcessingStatus.COMPLETED
            
            # Update stats
            self._processing_stats["documents_processed"] += 1
            self._processing_stats["successful_extractions"] += 1
            self._update_average_processing_time(processing_duration)
            
            logger.info(
                "Document processing completed successfully",
                filename=filename,
                profile_id=profile.profile_id,
                duration_seconds=processing_duration,
                quality_score=profile.processing.quality_score
            )
            
            return profile
            
        except Exception as e:
            # Handle processing failure
            processing_duration = (datetime.now() - start_time).total_seconds()
            
            if 'profile' in locals():
                profile.processing.processing_status = ProcessingStatus.FAILED
                profile.processing.processing_completed_at = datetime.now()
                profile.processing.processing_duration_seconds = processing_duration
                profile.processing.error_message = str(e)
            
            self._processing_stats["failed_extractions"] += 1
            
            logger.error(
                "Document processing failed",
                filename=filename,
                error=str(e),
                duration_seconds=processing_duration
            )
            
            raise RuntimeError(f"Document processing failed: {e}")
    
    async def process_documents_batch(
        self,
        documents: List[Tuple[bytes, str]],
        tenant_id: str,
        user_id: Optional[str] = None,
        max_concurrent: int = 5
    ) -> List[CVProfile]:
        """
        Process multiple documents concurrently with controlled parallelism.
        
        Args:
            documents: List of (content, filename) tuples
            tenant_id: Tenant identifier
            user_id: User who uploaded documents
            max_concurrent: Maximum concurrent processing tasks
            
        Returns:
            List of processed CV profiles
        """
        if not documents:
            return []
        
        logger.info(
            "Starting batch document processing",
            document_count=len(documents),
            tenant_id=tenant_id,
            max_concurrent=max_concurrent
        )
        
        # Process in controlled batches
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(content: bytes, filename: str) -> Optional[CVProfile]:
            async with semaphore:
                try:
                    return await self.process_document(content, filename, tenant_id, user_id)
                except Exception as e:
                    logger.error(f"Failed to process {filename}: {e}")
                    return None
        
        # Create tasks for all documents
        tasks = [
            process_single(content, filename)
            for content, filename in documents
        ]
        
        # Execute with progress tracking
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        for result in completed_results:
            if isinstance(result, CVProfile):
                results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Batch processing error: {result}")
        
        logger.info(
            "Batch document processing completed",
            total_documents=len(documents),
            successful_documents=len(results),
            failed_documents=len(documents) - len(results)
        )
        
        return results
    
    def _validate_document(self, content: bytes, filename: str) -> None:
        """Validate document before processing"""
        # Check file size
        size_mb = len(content) / (1024 * 1024)
        if size_mb > self._processing_config["max_document_size_mb"]:
            raise ValueError(f"Document size {size_mb:.1f}MB exceeds limit of {self._processing_config['max_document_size_mb']}MB")
        
        # Check file format
        file_extension = self._get_file_extension(filename)
        if file_extension not in self._processing_config["supported_formats"]:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Basic content validation
        if len(content) < 100:  # Minimum viable document size
            raise ValueError("Document appears to be empty or too small")
    
    async def _initialize_profile(
        self,
        content: bytes,
        filename: str,
        tenant_id: str,
        user_id: Optional[str]
    ) -> CVProfile:
        """Initialize CV profile with basic metadata"""
        import uuid
        
        # Generate document hash for deduplication
        document_hash = hashlib.sha256(content).hexdigest()
        
        profile = CVProfile(
            tenant_id=uuid.UUID(tenant_id),
            profile_id=str(uuid.uuid4()),
            email="",  # Will be populated during extraction
            original_filename=filename,
            document_format=self._get_file_extension(filename).lstrip('.'),
            document_hash=document_hash,
            created_by=uuid.UUID(user_id) if user_id else None
        )
        
        # Initialize processing metadata
        profile.processing.extraction_method = "azure_document_intelligence"
        profile.processing.processing_status = ProcessingStatus.PENDING
        
        return profile
    
    async def _extract_document_content(
        self,
        content: bytes,
        filename: str,
        options: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract structured content using Document Intelligence"""
        file_extension = self._get_file_extension(filename).lstrip('.')
        
        # Use CV-specific extraction for better structure detection
        extraction_result = await self.doc_intelligence.extract_cv_structure(
            document_content=content,
            document_type=file_extension
        )
        
        return extraction_result
    
    async def _parse_cv_structure(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and structure CV data from extraction results"""
        cv_structure = {
            "contact_info": {},
            "personal_info": {},
            "experience_entries": [],
            "education_entries": [],
            "skills": [],
            "certifications": [],
            "projects": [],
            "languages": [],
            "summary": "",
            "metadata": {}
        }
        
        # Extract contact information
        if extraction_result.get("contact_info"):
            cv_structure["contact_info"] = extraction_result["contact_info"]
        
        # Extract sections
        sections = extraction_result.get("sections", {})
        
        # Process experience section
        if "experience" in sections:
            cv_structure["experience_entries"] = await self._parse_experience_entries(
                sections["experience"]
            )
        
        # Process education section
        if "education" in sections:
            cv_structure["education_entries"] = await self._parse_education_entries(
                sections["education"]
            )
        
        # Process skills section
        if "skills" in sections:
            cv_structure["skills"] = await self._parse_skills(sections["skills"])
        
        # Process other sections
        for section_name, content in sections.items():
            if section_name == "summary":
                cv_structure["summary"] = content
            elif section_name == "certifications":
                cv_structure["certifications"] = await self._parse_certifications(content)
            elif section_name == "projects":
                cv_structure["projects"] = await self._parse_projects(content)
        
        return cv_structure
    
    async def _populate_profile_data(
        self,
        profile: CVProfile,
        cv_structure: Dict[str, Any],
        extraction_result: Dict[str, Any]
    ) -> None:
        """Populate profile with extracted and structured data"""
        
        # Contact information
        contact_data = cv_structure.get("contact_info", {})
        if contact_data:
            try:
                profile.contact_info = ContactInfo(**contact_data)
                profile.email = contact_data.get("email", "")
            except Exception as e:
                logger.warning(f"Failed to parse contact info: {e}")
                # Extract email at minimum
                profile.email = contact_data.get("email", f"unknown_{profile.profile_id}@extracted.local")
        
        # Basic information
        if cv_structure.get("personal_info"):
            personal = cv_structure["personal_info"]
            profile.first_name = personal.get("first_name")
            profile.last_name = personal.get("last_name")
            profile.title = personal.get("title")
        
        # Summary
        profile.summary = cv_structure.get("summary", "")
        
        # Experience
        for exp_data in cv_structure.get("experience_entries", []):
            try:
                experience = ExperienceEntry(**exp_data)
                profile.experience_entries.append(experience)
            except Exception as e:
                logger.warning(f"Failed to parse experience entry: {e}")
        
        # Education
        for edu_data in cv_structure.get("education_entries", []):
            try:
                education = EducationEntry(**edu_data)
                profile.education_entries.append(education)
            except Exception as e:
                logger.warning(f"Failed to parse education entry: {e}")
        
        # Skills
        for skill_data in cv_structure.get("skills", []):
            try:
                if isinstance(skill_data, str):
                    skill = Skill(name=skill_data)
                else:
                    skill = Skill(**skill_data)
                profile.skills.append(skill)
            except Exception as e:
                logger.warning(f"Failed to parse skill: {e}")
        
        # Certifications
        for cert_data in cv_structure.get("certifications", []):
            try:
                certification = Certification(**cert_data)
                profile.certifications.append(certification)
            except Exception as e:
                logger.warning(f"Failed to parse certification: {e}")
        
        # Projects
        for proj_data in cv_structure.get("projects", []):
            try:
                project = Project(**proj_data)
                profile.projects.append(project)
            except Exception as e:
                logger.warning(f"Failed to parse project: {e}")
        
        # Languages
        for lang_data in cv_structure.get("languages", []):
            try:
                language = Language(**lang_data)
                profile.languages.append(language)
            except Exception as e:
                logger.warning(f"Failed to parse language: {e}")
        
        # Update processing metadata
        profile.processing.pages_processed = extraction_result.get("metadata", {}).get("pages_analyzed", 1)
        profile.processing.document_language = extraction_result.get("metadata", {}).get("document_language", "en")
    
    async def _parse_experience_entries(self, experience_text: str) -> List[Dict[str, Any]]:
        """Parse experience section into structured entries"""
        import re
        from dateutil import parser as date_parser
        
        entries = []
        
        # Split by common separators
        potential_entries = re.split(r'\n\s*\n|\n(?=\d{4})', experience_text)
        
        for entry_text in potential_entries:
            if not entry_text.strip():
                continue
            
            entry = {
                "title": "",
                "company": "",
                "location": None,
                "start_date": None,
                "end_date": None,
                "description": "",
                "skills_used": [],
                "achievements": []
            }
            
            lines = [line.strip() for line in entry_text.split('\n') if line.strip()]
            if not lines:
                continue
            
            # First line often contains title/company
            first_line = lines[0]
            
            # Try to parse title and company from first line
            # Common patterns: "Title at Company", "Title - Company", "Title | Company"
            title_company_patterns = [
                r'^(.+?)\s+at\s+(.+)$',
                r'^(.+?)\s*-\s*(.+)$',
                r'^(.+?)\s*\|\s*(.+)$',
                r'^(.+?)\s*,\s*(.+)$'
            ]
            
            for pattern in title_company_patterns:
                match = re.match(pattern, first_line, re.IGNORECASE)
                if match:
                    entry["title"] = match.group(1).strip()
                    entry["company"] = match.group(2).strip()
                    break
            else:
                # If no pattern matches, assume entire first line is title
                entry["title"] = first_line
            
            # Look for dates in the entry
            date_pattern = r'\b(\d{1,2}/?20\d{2}|\w{3,9}\s+20\d{2}|20\d{2})\b'
            dates_found = re.findall(date_pattern, entry_text)
            
            if len(dates_found) >= 2:
                # Try to parse start and end dates
                try:
                    start_date = date_parser.parse(dates_found[0], fuzzy=True).date()
                    entry["start_date"] = start_date
                    
                    if dates_found[1].lower() not in ['present', 'current', 'now']:
                        end_date = date_parser.parse(dates_found[1], fuzzy=True).date()
                        entry["end_date"] = end_date
                except Exception:
                    pass
            elif len(dates_found) == 1:
                try:
                    start_date = date_parser.parse(dates_found[0], fuzzy=True).date()
                    entry["start_date"] = start_date
                except Exception:
                    pass
            
            # Remaining lines are description
            if len(lines) > 1:
                entry["description"] = '\n'.join(lines[1:])
            
            entries.append(entry)
        
        return entries
    
    async def _parse_education_entries(self, education_text: str) -> List[Dict[str, Any]]:
        """Parse education section into structured entries"""
        import re
        
        entries = []
        potential_entries = re.split(r'\n\s*\n', education_text)
        
        for entry_text in potential_entries:
            if not entry_text.strip():
                continue
            
            entry = {
                "degree": "",
                "institution": "",
                "location": None,
                "graduation_year": None,
                "major": None,
                "gpa": None
            }
            
            lines = [line.strip() for line in entry_text.split('\n') if line.strip()]
            if not lines:
                continue
            
            # First line often contains degree
            entry["degree"] = lines[0]
            
            # Look for institution name
            for line in lines[1:]:
                if any(keyword in line.lower() for keyword in ['university', 'college', 'school', 'institute']):
                    entry["institution"] = line
                    break
            
            # Look for graduation year
            year_pattern = r'\b(19|20)\d{2}\b'
            year_match = re.search(year_pattern, entry_text)
            if year_match:
                entry["graduation_year"] = int(year_match.group())
            
            entries.append(entry)
        
        return entries
    
    async def _parse_skills(self, skills_text: str) -> List[Dict[str, Any]]:
        """Parse skills section into structured skill objects"""
        skills = []
        
        # Split by common separators
        separators = [',', '•', '·', '|', '\n', ';', '/', '\\']
        skill_items = [skills_text]
        
        for sep in separators:
            new_items = []
            for item in skill_items:
                new_items.extend(item.split(sep))
            skill_items = new_items
        
        # Clean and create skill objects
        for skill_text in skill_items:
            cleaned = skill_text.strip()
            if cleaned and len(cleaned) > 1 and len(cleaned) < 100:
                # Try to detect skill level from text
                level = None
                years = None
                
                # Look for experience indicators
                import re
                years_match = re.search(r'(\d+)\s*(?:years?|yrs?)', cleaned.lower())
                if years_match:
                    years = int(years_match.group(1))
                    # Remove years info from skill name
                    cleaned = re.sub(r'\s*\(\s*\d+\s*(?:years?|yrs?)\s*\)\s*', '', cleaned)
                
                # Look for level indicators
                level_indicators = {
                    'expert': 'expert',
                    'advanced': 'advanced', 
                    'intermediate': 'intermediate',
                    'beginner': 'beginner',
                    'basic': 'beginner'
                }
                
                for indicator, level_value in level_indicators.items():
                    if indicator in cleaned.lower():
                        level = level_value
                        # Remove level info from skill name
                        cleaned = re.sub(f'\\s*\\(?{indicator}\\)?\\s*', '', cleaned, flags=re.IGNORECASE)
                        break
                
                skill_data = {"name": cleaned.strip()}
                if level:
                    skill_data["level"] = level
                if years:
                    skill_data["years_experience"] = years
                
                skills.append(skill_data)
        
        return skills
    
    async def _parse_certifications(self, cert_text: str) -> List[Dict[str, Any]]:
        """Parse certifications section"""
        import re
        
        certifications = []
        lines = [line.strip() for line in cert_text.split('\n') if line.strip()]
        
        for line in lines:
            # Remove bullet points
            cleaned = re.sub(r'^[•·\-\*]\s*', '', line)
            
            if cleaned:
                cert_data = {"name": cleaned, "issuer": ""}
                
                # Try to extract issuer if separated by common patterns
                issuer_patterns = [
                    r'^(.+?)\s+(?:from|by)\s+(.+)$',
                    r'^(.+?)\s*-\s*(.+)$',
                    r'^(.+?)\s*\|\s*(.+)$'
                ]
                
                for pattern in issuer_patterns:
                    match = re.match(pattern, cleaned)
                    if match:
                        cert_data["name"] = match.group(1).strip()
                        cert_data["issuer"] = match.group(2).strip()
                        break
                
                certifications.append(cert_data)
        
        return certifications
    
    async def _parse_projects(self, projects_text: str) -> List[Dict[str, Any]]:
        """Parse projects section"""
        projects = []
        
        # Split by project separators
        import re
        potential_projects = re.split(r'\n\s*\n|\n(?=\w+:)', projects_text)
        
        for project_text in potential_projects:
            if not project_text.strip():
                continue
            
            project_data = {
                "name": "",
                "description": None,
                "technologies": [],
                "url": None
            }
            
            lines = [line.strip() for line in project_text.split('\n') if line.strip()]
            if lines:
                project_data["name"] = lines[0]
                if len(lines) > 1:
                    project_data["description"] = '\n'.join(lines[1:])
                
                # Look for URLs in the description
                url_pattern = r'https?://[^\s]+'
                urls = re.findall(url_pattern, project_text)
                if urls:
                    project_data["url"] = urls[0]
                
                projects.append(project_data)
        
        return projects
    
    async def _assess_quality(
        self,
        profile: CVProfile,
        extraction_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """Assess the quality of extracted CV data"""
        
        quality_scores = {
            "contact_completeness": 0.0,
            "content_richness": 0.0,
            "structure_quality": 0.0,
            "data_consistency": 0.0
        }
        
        # Contact information completeness
        if profile.contact_info:
            contact_fields = ['email', 'phone', 'location']
            filled_fields = sum(1 for field in contact_fields 
                              if getattr(profile.contact_info, field, None))
            quality_scores["contact_completeness"] = filled_fields / len(contact_fields)
        
        # Content richness
        content_indicators = {
            "has_summary": bool(profile.summary),
            "has_experience": len(profile.experience_entries) > 0,
            "has_education": len(profile.education_entries) > 0,
            "has_skills": len(profile.skills) > 0,
            "sufficient_text": len(profile.generate_searchable_text()) > 200
        }
        quality_scores["content_richness"] = sum(content_indicators.values()) / len(content_indicators)
        
        # Structure quality
        structure_indicators = {
            "experience_with_dates": any(exp.start_date for exp in profile.experience_entries),
            "education_with_degrees": any(edu.degree for edu in profile.education_entries),
            "skills_categorized": len(profile.skills) >= 3,
            "proper_contact": bool(profile.email and '@' in profile.email)
        }
        quality_scores["structure_quality"] = sum(structure_indicators.values()) / len(structure_indicators)
        
        # Data consistency
        consistency_score = 1.0
        
        # Check for obvious inconsistencies
        if profile.experience_entries:
            # Check date consistency in experience
            for exp in profile.experience_entries:
                if exp.start_date and exp.end_date:
                    if exp.start_date > exp.end_date:
                        consistency_score -= 0.2
        
        quality_scores["data_consistency"] = max(0.0, consistency_score)
        
        # Calculate overall scores
        overall_score = sum(quality_scores.values()) / len(quality_scores)
        confidence_score = extraction_result.get("metadata", {}).get("confidence_score", overall_score)
        
        return {
            "overall_score": overall_score,
            "confidence_score": confidence_score,
            "detailed_scores": quality_scores
        }
    
    async def _store_original_document(
        self,
        profile: CVProfile,
        content: bytes,
        filename: str
    ) -> None:
        """Store original document in blob storage"""
        try:
            # Generate storage path
            storage_path = f"tenant_{profile.tenant_id}/documents/{profile.profile_id}/{filename}"
            
            # Store document
            document_url = await self.blob_service.store_document(
                container="documents",
                path=storage_path,
                content=content,
                metadata={
                    "profile_id": profile.profile_id,
                    "tenant_id": str(profile.tenant_id),
                    "original_filename": filename,
                    "processing_timestamp": datetime.now().isoformat()
                }
            )
            
            profile.document_url = document_url
            
        except Exception as e:
            logger.warning(f"Failed to store original document: {e}")
    
    def _get_file_extension(self, filename: str) -> str:
        """Extract file extension from filename"""
        import os
        return os.path.splitext(filename)[1].lower()
    
    def _update_average_processing_time(self, duration: float) -> None:
        """Update average processing time statistics"""
        current_avg = self._processing_stats["average_processing_time"]
        processed_count = self._processing_stats["documents_processed"]
        
        if processed_count == 1:
            self._processing_stats["average_processing_time"] = duration
        else:
            # Weighted average
            self._processing_stats["average_processing_time"] = (
                (current_avg * (processed_count - 1) + duration) / processed_count
            )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return {
            **self._processing_stats,
            "success_rate": (
                self._processing_stats["successful_extractions"] / 
                max(1, self._processing_stats["documents_processed"])
            ),
            "supported_formats": self._processing_config["supported_formats"],
            "quality_thresholds": self._quality_thresholds
        }