"""
Quality Analyzer for Document Processing

Advanced document quality assessment and validation:
- Multi-dimensional quality scoring
- Content completeness analysis
- Data integrity validation
- Structural analysis and scoring
- AI-powered quality recommendations
"""

import re
import statistics
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import structlog

from app.core.config import get_settings
from app.services.ai.openai_service import OpenAIService
from app.services.ai.prompt_manager import PromptManager, PromptType
from app.services.document.pdf_processor import PDFDocument
from app.services.document.content_extractor import StructuredContent

logger = structlog.get_logger(__name__)


class QualityDimension(Enum):
    """Quality assessment dimensions"""
    CONTENT_COMPLETENESS = "content_completeness"
    STRUCTURAL_INTEGRITY = "structural_integrity"
    INFORMATION_DENSITY = "information_density"
    PROFESSIONAL_PRESENTATION = "professional_presentation"
    DATA_ACCURACY = "data_accuracy"
    SEARCHABILITY = "searchability"
    PROCESSING_SUITABILITY = "processing_suitability"


@dataclass
class QualityScore:
    """Individual quality score for a dimension"""
    dimension: QualityDimension
    score: float  # 0-100
    confidence: float  # 0-1
    issues: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]


@dataclass
class QualityAssessment:
    """Complete quality assessment result"""
    overall_score: float  # 0-100
    dimension_scores: List[QualityScore]
    is_acceptable: bool
    critical_issues: List[str]
    improvement_recommendations: List[str]
    processing_metadata: Dict[str, Any]
    assessment_confidence: float


class QualityAnalyzer:
    """
    Comprehensive document quality analyzer.
    
    Features:
    - Multi-dimensional quality scoring
    - AI-powered content assessment
    - Structural and data integrity validation
    - Professional presentation evaluation
    - Automated quality recommendations
    - Content suitability for AI processing
    """
    
    def __init__(
        self,
        openai_service: OpenAIService,
        prompt_manager: PromptManager
    ):
        self.settings = get_settings()
        self.openai_service = openai_service
        self.prompt_manager = prompt_manager
        
        # Quality thresholds
        self._quality_thresholds = {
            "excellent": 90,
            "good": 75,
            "acceptable": 60,
            "poor": 40,
            "unacceptable": 0
        }
        
        # Processing statistics
        self._stats = {
            "assessments_completed": 0,
            "ai_analysis_calls": 0,
            "critical_issues_found": 0,
            "recommendations_generated": 0
        }
    
    async def analyze_document_quality(
        self,
        pdf_document: Optional[PDFDocument] = None,
        structured_content: Optional[StructuredContent] = None,
        content_type: str = "generic",
        use_ai_analysis: bool = True
    ) -> QualityAssessment:
        """
        Perform comprehensive quality analysis on a document.
        
        Args:
            pdf_document: Original PDF document
            structured_content: Processed structured content
            content_type: Type of content being analyzed
            use_ai_analysis: Use AI for advanced quality assessment
            
        Returns:
            QualityAssessment with comprehensive quality metrics
        """
        if not pdf_document and not structured_content:
            raise ValueError("Either pdf_document or structured_content must be provided")
        
        start_time = datetime.now()
        
        try:
            # Initialize dimension scores
            dimension_scores = []
            
            # Analyze each quality dimension
            for dimension in QualityDimension:
                score = await self._analyze_dimension(
                    dimension=dimension,
                    pdf_document=pdf_document,
                    structured_content=structured_content,
                    content_type=content_type,
                    use_ai_analysis=use_ai_analysis
                )
                dimension_scores.append(score)
            
            # Calculate overall quality score
            overall_score = self._calculate_overall_score(dimension_scores)
            
            # Determine if quality is acceptable
            is_acceptable = overall_score >= self._quality_thresholds["acceptable"]
            
            # Collect critical issues
            critical_issues = []
            improvement_recommendations = []
            
            for score in dimension_scores:
                if score.score < self._quality_thresholds["acceptable"]:
                    critical_issues.extend(score.issues)
                improvement_recommendations.extend(score.recommendations)
            
            # Calculate assessment confidence
            confidence_scores = [score.confidence for score in dimension_scores]
            assessment_confidence = statistics.mean(confidence_scores) if confidence_scores else 0.0
            
            # Create processing metadata
            processing_metadata = {
                "analysis_started": start_time.isoformat(),
                "analysis_completed": datetime.now().isoformat(),
                "analysis_duration": (datetime.now() - start_time).total_seconds(),
                "content_type": content_type,
                "ai_analysis_used": use_ai_analysis,
                "dimensions_analyzed": len(dimension_scores)
            }
            
            # Create quality assessment
            assessment = QualityAssessment(
                overall_score=overall_score,
                dimension_scores=dimension_scores,
                is_acceptable=is_acceptable,
                critical_issues=critical_issues,
                improvement_recommendations=improvement_recommendations,
                processing_metadata=processing_metadata,
                assessment_confidence=assessment_confidence
            )
            
            # Update statistics
            self._stats["assessments_completed"] += 1
            self._stats["critical_issues_found"] += len(critical_issues)
            self._stats["recommendations_generated"] += len(improvement_recommendations)
            
            logger.info(
                "Quality analysis completed",
                overall_score=overall_score,
                is_acceptable=is_acceptable,
                critical_issues=len(critical_issues),
                analysis_duration=processing_metadata["analysis_duration"]
            )
            
            return assessment
            
        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            raise
    
    async def _analyze_dimension(
        self,
        dimension: QualityDimension,
        pdf_document: Optional[PDFDocument],
        structured_content: Optional[StructuredContent],
        content_type: str,
        use_ai_analysis: bool
    ) -> QualityScore:
        """Analyze a specific quality dimension"""
        
        if dimension == QualityDimension.CONTENT_COMPLETENESS:
            return await self._analyze_content_completeness(
                pdf_document, structured_content, content_type, use_ai_analysis
            )
        elif dimension == QualityDimension.STRUCTURAL_INTEGRITY:
            return await self._analyze_structural_integrity(
                pdf_document, structured_content
            )
        elif dimension == QualityDimension.INFORMATION_DENSITY:
            return await self._analyze_information_density(
                pdf_document, structured_content
            )
        elif dimension == QualityDimension.PROFESSIONAL_PRESENTATION:
            return await self._analyze_professional_presentation(
                pdf_document, structured_content, use_ai_analysis
            )
        elif dimension == QualityDimension.DATA_ACCURACY:
            return await self._analyze_data_accuracy(
                pdf_document, structured_content, use_ai_analysis
            )
        elif dimension == QualityDimension.SEARCHABILITY:
            return await self._analyze_searchability(
                pdf_document, structured_content
            )
        elif dimension == QualityDimension.PROCESSING_SUITABILITY:
            return await self._analyze_processing_suitability(
                pdf_document, structured_content
            )
        else:
            # Default scoring
            return QualityScore(
                dimension=dimension,
                score=50.0,
                confidence=0.5,
                issues=["Analysis not implemented for this dimension"],
                recommendations=[],
                metadata={}
            )
    
    async def _analyze_content_completeness(
        self,
        pdf_document: Optional[PDFDocument],
        structured_content: Optional[StructuredContent],
        content_type: str,
        use_ai_analysis: bool
    ) -> QualityScore:
        """Analyze content completeness"""
        issues = []
        recommendations = []
        score = 100.0
        confidence = 0.9
        
        # Get text for analysis
        text = ""
        if structured_content:
            text = " ".join([section.content for section in structured_content.sections])
        elif pdf_document:
            text = pdf_document.full_text
        
        # Basic completeness checks
        if not text or len(text.strip()) < 100:
            issues.append("Document contains very little text content")
            score -= 40
        
        if len(text.split()) < 50:
            issues.append("Document has fewer than 50 words")
            score -= 30
            recommendations.append("Add more detailed content")
        
        # Content-type specific checks
        if content_type.lower() == "cv":
            score, cv_issues, cv_recommendations = await self._check_cv_completeness(text, use_ai_analysis)
            issues.extend(cv_issues)
            recommendations.extend(cv_recommendations)
        
        elif content_type.lower() == "job_description":
            score, job_issues, job_recommendations = await self._check_job_description_completeness(text, use_ai_analysis)
            issues.extend(job_issues)
            recommendations.extend(job_recommendations)
        
        return QualityScore(
            dimension=QualityDimension.CONTENT_COMPLETENESS,
            score=max(0.0, score),
            confidence=confidence,
            issues=issues,
            recommendations=recommendations,
            metadata={
                "text_length": len(text),
                "word_count": len(text.split()),
                "content_type": content_type
            }
        )
    
    async def _analyze_structural_integrity(
        self,
        pdf_document: Optional[PDFDocument],
        structured_content: Optional[StructuredContent]
    ) -> QualityScore:
        """Analyze document structural integrity"""
        issues = []
        recommendations = []
        score = 100.0
        confidence = 0.8
        
        if pdf_document:
            # Check page extraction success
            non_empty_pages = [page for page in pdf_document.pages if page.text and len(page.text.strip()) > 10]
            if len(non_empty_pages) < len(pdf_document.pages) * 0.8:
                issues.append("Many pages failed to extract content properly")
                score -= 25
                recommendations.append("Consider using OCR for scanned documents")
            
            # Check for processing errors in pages
            error_pages = [page for page in pdf_document.pages if "extraction_error" in page.metadata]
            if error_pages:
                issues.append(f"{len(error_pages)} pages had extraction errors")
                score -= 15
        
        if structured_content:
            # Check section extraction
            if not structured_content.sections:
                issues.append("No document sections were identified")
                score -= 30
                recommendations.append("Improve document formatting and structure")
            
            # Check section quality
            poor_sections = [s for s in structured_content.sections if s.confidence < 0.5]
            if len(poor_sections) > len(structured_content.sections) * 0.3:
                issues.append("Many sections have low confidence scores")
                score -= 20
        
        return QualityScore(
            dimension=QualityDimension.STRUCTURAL_INTEGRITY,
            score=max(0.0, score),
            confidence=confidence,
            issues=issues,
            recommendations=recommendations,
            metadata={}
        )
    
    async def _analyze_information_density(
        self,
        pdf_document: Optional[PDFDocument],
        structured_content: Optional[StructuredContent]
    ) -> QualityScore:
        """Analyze information density and richness"""
        issues = []
        recommendations = []
        score = 100.0
        confidence = 0.7
        
        # Get text for analysis
        text = ""
        if structured_content:
            text = " ".join([section.content for section in structured_content.sections])
        elif pdf_document:
            text = pdf_document.full_text
        
        if not text:
            return QualityScore(
                dimension=QualityDimension.INFORMATION_DENSITY,
                score=0.0,
                confidence=0.9,
                issues=["No text content to analyze"],
                recommendations=["Ensure document contains readable text"],
                metadata={}
            )
        
        # Calculate density metrics
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        paragraphs = text.split('\n\n')
        
        # Average words per sentence
        avg_words_per_sentence = len(words) / max(len(sentences), 1)
        if avg_words_per_sentence < 5:
            issues.append("Sentences are very short, may lack detail")
            score -= 15
        elif avg_words_per_sentence > 25:
            issues.append("Sentences are very long, may be hard to process")
            score -= 10
        
        # Vocabulary diversity
        unique_words = set(word.lower().strip('.,!?;:') for word in words)
        vocabulary_ratio = len(unique_words) / max(len(words), 1)
        
        if vocabulary_ratio < 0.3:
            issues.append("Limited vocabulary diversity")
            score -= 20
            recommendations.append("Use more varied vocabulary")
        
        # Information patterns
        has_dates = bool(re.search(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b', text))
        has_numbers = bool(re.search(r'\b\d+\b', text))
        has_emails = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        
        info_richness = sum([has_dates, has_numbers, has_emails])
        if info_richness == 0:
            issues.append("Document lacks specific details (dates, numbers, contact info)")
            score -= 25
        
        return QualityScore(
            dimension=QualityDimension.INFORMATION_DENSITY,
            score=max(0.0, score),
            confidence=confidence,
            issues=issues,
            recommendations=recommendations,
            metadata={
                "avg_words_per_sentence": avg_words_per_sentence,
                "vocabulary_ratio": vocabulary_ratio,
                "has_dates": has_dates,
                "has_numbers": has_numbers,
                "has_emails": has_emails
            }
        )
    
    async def _analyze_professional_presentation(
        self,
        pdf_document: Optional[PDFDocument],
        structured_content: Optional[StructuredContent],
        use_ai_analysis: bool
    ) -> QualityScore:
        """Analyze professional presentation quality"""
        issues = []
        recommendations = []
        score = 100.0
        confidence = 0.6
        
        # Get text for analysis
        text = ""
        if structured_content:
            text = " ".join([section.content for section in structured_content.sections])
        elif pdf_document:
            text = pdf_document.full_text
        
        # Basic presentation checks
        if text:
            # Check for excessive special characters
            special_char_ratio = len(re.findall(r'[^\w\s.,!?;:\-()"]', text)) / max(len(text), 1)
            if special_char_ratio > 0.1:
                issues.append("High ratio of unusual characters detected")
                score -= 20
                recommendations.append("Clean up formatting and encoding issues")
            
            # Check for repeated whitespace/formatting issues
            if re.search(r'\s{4,}', text):
                issues.append("Irregular spacing detected")
                score -= 10
            
            # Check capitalization patterns
            words = text.split()
            all_caps_words = [w for w in words if w.isupper() and len(w) > 2]
            if len(all_caps_words) > len(words) * 0.1:
                issues.append("Excessive use of ALL CAPS")
                score -= 15
        
        # Use AI analysis if enabled
        if use_ai_analysis and text:
            try:
                ai_assessment = await self._get_ai_presentation_assessment(text[:2000])  # Limit text
                
                # Parse AI feedback
                if "unprofessional" in ai_assessment.lower():
                    issues.append("AI detected unprofessional presentation")
                    score -= 25
                
                if "formatting issues" in ai_assessment.lower():
                    issues.append("AI detected formatting issues")
                    score -= 15
                
                confidence = 0.8  # Higher confidence with AI analysis
                
            except Exception as e:
                logger.warning(f"AI presentation analysis failed: {e}")
        
        return QualityScore(
            dimension=QualityDimension.PROFESSIONAL_PRESENTATION,
            score=max(0.0, score),
            confidence=confidence,
            issues=issues,
            recommendations=recommendations,
            metadata={}
        )
    
    async def _analyze_data_accuracy(
        self,
        pdf_document: Optional[PDFDocument],
        structured_content: Optional[StructuredContent],
        use_ai_analysis: bool
    ) -> QualityScore:
        """Analyze data accuracy and consistency"""
        issues = []
        recommendations = []
        score = 100.0
        confidence = 0.5  # Lower confidence without external validation
        
        # Get text for analysis
        text = ""
        if structured_content:
            text = " ".join([section.content for section in structured_content.sections])
        elif pdf_document:
            text = pdf_document.full_text
        
        if not text:
            return QualityScore(
                dimension=QualityDimension.DATA_ACCURACY,
                score=0.0,
                confidence=0.9,
                issues=["No text content to validate"],
                recommendations=[],
                metadata={}
            )
        
        # Basic validation checks
        # Check for date consistency
        dates = re.findall(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b', text)
        if dates:
            current_year = datetime.now().year
            future_dates = [d for d in dates if re.match(r'\d{4}', d) and int(d) > current_year + 1]
            if future_dates:
                issues.append("Found dates in the future")
                score -= 15
        
        # Check for email format validity
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        for email in emails:
            if not re.match(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$', email):
                issues.append("Invalid email format detected")
                score -= 10
                break
        
        # Check for phone number patterns
        phones = re.findall(r'[\+]?[1-9]?[0-9]{7,15}', text)
        # Basic validation could be added here
        
        # Use AI for advanced accuracy assessment if enabled
        if use_ai_analysis:
            try:
                ai_accuracy = await self._get_ai_accuracy_assessment(text[:2000])
                confidence = 0.7  # Higher confidence with AI
            except Exception as e:
                logger.warning(f"AI accuracy analysis failed: {e}")
        
        return QualityScore(
            dimension=QualityDimension.DATA_ACCURACY,
            score=max(0.0, score),
            confidence=confidence,
            issues=issues,
            recommendations=recommendations,
            metadata={
                "dates_found": len(dates),
                "emails_found": len(emails),
                "phones_found": len(phones)
            }
        )
    
    async def _analyze_searchability(
        self,
        pdf_document: Optional[PDFDocument],
        structured_content: Optional[StructuredContent]
    ) -> QualityScore:
        """Analyze searchability and keyword richness"""
        issues = []
        recommendations = []
        score = 100.0
        confidence = 0.8
        
        # Get text for analysis
        text = ""
        if structured_content:
            text = " ".join([section.content for section in structured_content.sections])
        elif pdf_document:
            text = pdf_document.full_text
        
        if not text:
            return QualityScore(
                dimension=QualityDimension.SEARCHABILITY,
                score=0.0,
                confidence=0.9,
                issues=["No text content for searching"],
                recommendations=["Ensure document contains searchable text"],
                metadata={}
            )
        
        # Analyze searchable content
        words = text.split()
        
        # Check for minimum keyword density
        if len(words) < 100:
            issues.append("Document too short for effective searching")
            score -= 30
            recommendations.append("Add more descriptive content")
        
        # Check for keyword diversity
        unique_words = set(word.lower().strip('.,!?;:') for word in words)
        if len(unique_words) < 20:
            issues.append("Limited keyword diversity")
            score -= 25
            recommendations.append("Include more varied terminology")
        
        # Check for professional keywords (basic industry terms)
        professional_keywords = {
            'experience', 'skills', 'education', 'responsibilities', 'achievements',
            'management', 'development', 'analysis', 'project', 'team', 'leadership'
        }
        found_keywords = [word for word in unique_words if word in professional_keywords]
        
        if len(found_keywords) < 3:
            issues.append("Limited professional terminology")
            score -= 20
            recommendations.append("Include more industry-relevant keywords")
        
        return QualityScore(
            dimension=QualityDimension.SEARCHABILITY,
            score=max(0.0, score),
            confidence=confidence,
            issues=issues,
            recommendations=recommendations,
            metadata={
                "total_words": len(words),
                "unique_words": len(unique_words),
                "professional_keywords": found_keywords
            }
        )
    
    async def _analyze_processing_suitability(
        self,
        pdf_document: Optional[PDFDocument],
        structured_content: Optional[StructuredContent]
    ) -> QualityScore:
        """Analyze suitability for AI processing"""
        issues = []
        recommendations = []
        score = 100.0
        confidence = 0.9
        
        # Check PDF processing quality
        if pdf_document:
            extraction_success = len([page for page in pdf_document.pages if page.text]) / max(len(pdf_document.pages), 1)
            if extraction_success < 0.8:
                issues.append("Poor text extraction from PDF")
                score -= 30
                recommendations.append("Consider using OCR or providing text-based PDF")
        
        # Check structured content quality
        if structured_content:
            if not structured_content.sections:
                issues.append("No structured sections identified")
                score -= 25
                recommendations.append("Improve document structure and formatting")
            
            # Check quality assessment from content extractor
            if structured_content.quality_assessment:
                extractor_quality = structured_content.quality_assessment.get("overall_quality", 100)
                if extractor_quality < 60:
                    issues.append("Content extraction quality is low")
                    score -= 20
        
        # Check text quality for AI processing
        text = ""
        if structured_content:
            text = " ".join([section.content for section in structured_content.sections])
        elif pdf_document:
            text = pdf_document.full_text
        
        if text:
            # Check for minimum content for AI analysis
            if len(text.split()) < 50:
                issues.append("Insufficient content for reliable AI analysis")
                score -= 35
                recommendations.append("Provide more detailed content")
            
            # Check encoding and character issues
            try:
                text.encode('utf-8')
            except UnicodeEncodeError:
                issues.append("Text encoding issues detected")
                score -= 15
                recommendations.append("Fix text encoding problems")
        
        return QualityScore(
            dimension=QualityDimension.PROCESSING_SUITABILITY,
            score=max(0.0, score),
            confidence=confidence,
            issues=issues,
            recommendations=recommendations,
            metadata={}
        )
    
    async def _check_cv_completeness(self, text: str, use_ai: bool) -> Tuple[float, List[str], List[str]]:
        """Check CV-specific completeness"""
        score = 100.0
        issues = []
        recommendations = []
        
        # Check for essential CV sections
        essential_indicators = {
            "experience": ["experience", "work", "employment", "position", "role"],
            "education": ["education", "degree", "university", "college", "school"],
            "skills": ["skills", "competencies", "expertise", "proficient"],
            "contact": ["email", "@", "phone", "contact"]
        }
        
        text_lower = text.lower()
        missing_sections = []
        
        for section, indicators in essential_indicators.items():
            if not any(indicator in text_lower for indicator in indicators):
                missing_sections.append(section)
        
        if missing_sections:
            penalty = len(missing_sections) * 15
            score -= penalty
            issues.append(f"Missing essential CV sections: {', '.join(missing_sections)}")
            recommendations.append(f"Add {', '.join(missing_sections)} information")
        
        return score, issues, recommendations
    
    async def _check_job_description_completeness(self, text: str, use_ai: bool) -> Tuple[float, List[str], List[str]]:
        """Check job description completeness"""
        score = 100.0
        issues = []
        recommendations = []
        
        # Check for essential job description elements
        essential_indicators = {
            "responsibilities": ["responsibilities", "duties", "role", "tasks"],
            "requirements": ["requirements", "qualifications", "skills", "experience"],
            "company": ["company", "organization", "employer"],
            "location": ["location", "city", "remote", "office"]
        }
        
        text_lower = text.lower()
        missing_elements = []
        
        for element, indicators in essential_indicators.items():
            if not any(indicator in text_lower for indicator in indicators):
                missing_elements.append(element)
        
        if missing_elements:
            penalty = len(missing_elements) * 12
            score -= penalty
            issues.append(f"Missing job description elements: {', '.join(missing_elements)}")
            recommendations.append(f"Add {', '.join(missing_elements)} details")
        
        return score, issues, recommendations
    
    async def _get_ai_presentation_assessment(self, text: str) -> str:
        """Get AI assessment of presentation quality"""
        try:
            prompt = await self.prompt_manager.generate_prompt(
                PromptType.QUALITY_ASSESSMENT,
                {"content": text},
                custom_instructions="Focus on professional presentation and formatting quality"
            )
            
            response = await self.openai_service.chat_completion(
                messages=prompt["messages"],
                temperature=0.1,
                max_tokens=200
            )
            
            self._stats["ai_analysis_calls"] += 1
            return response["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"AI presentation assessment failed: {e}")
            return "Assessment unavailable"
    
    async def _get_ai_accuracy_assessment(self, text: str) -> str:
        """Get AI assessment of data accuracy"""
        try:
            prompt = await self.prompt_manager.generate_prompt(
                PromptType.QUALITY_ASSESSMENT,
                {"content": text},
                custom_instructions="Focus on data accuracy and consistency"
            )
            
            response = await self.openai_service.chat_completion(
                messages=prompt["messages"],
                temperature=0.1,
                max_tokens=200
            )
            
            self._stats["ai_analysis_calls"] += 1
            return response["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"AI accuracy assessment failed: {e}")
            return "Assessment unavailable"
    
    def _calculate_overall_score(self, dimension_scores: List[QualityScore]) -> float:
        """Calculate weighted overall quality score"""
        if not dimension_scores:
            return 0.0
        
        # Define weights for different dimensions
        weights = {
            QualityDimension.CONTENT_COMPLETENESS: 0.25,
            QualityDimension.STRUCTURAL_INTEGRITY: 0.15,
            QualityDimension.INFORMATION_DENSITY: 0.15,
            QualityDimension.PROFESSIONAL_PRESENTATION: 0.15,
            QualityDimension.DATA_ACCURACY: 0.10,
            QualityDimension.SEARCHABILITY: 0.10,
            QualityDimension.PROCESSING_SUITABILITY: 0.10
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for score in dimension_scores:
            weight = weights.get(score.dimension, 1.0 / len(dimension_scores))
            weighted_sum += score.score * weight * score.confidence
            total_weight += weight * score.confidence
        
        return weighted_sum / max(total_weight, 0.01)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "stats": self._stats.copy(),
            "thresholds": self._quality_thresholds,
            "dimensions": [dim.value for dim in QualityDimension]
        }
    
    async def check_health(self) -> Dict[str, Any]:
        """Check quality analyzer health"""
        try:
            # Test basic analysis
            test_text = "This is a test document with some professional experience and education details."
            
            # Create minimal test content
            from app.services.document.content_extractor import StructuredContent, ExtractedSection
            test_section = ExtractedSection(
                section_type="test",
                title="Test Section",
                content=test_text,
                confidence=0.8,
                metadata={},
                start_position=0,
                end_position=len(test_text)
            )
            
            test_structured_content = StructuredContent(
                document_type="test",
                sections=[test_section],
                summary="Test document summary",
                key_information={"test": "value"},
                quality_assessment={},
                processing_metadata={}
            )
            
            # Perform test analysis
            assessment = await self.analyze_document_quality(
                structured_content=test_structured_content,
                content_type="test",
                use_ai_analysis=False
            )
            
            return {
                "status": "healthy",
                "analyzer": "operational",
                "test_assessment_score": assessment.overall_score,
                "dimensions_analyzed": len(assessment.dimension_scores),
                "stats": self._stats.copy(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }