"""
Prompt Manager for AI Operations

Centralized prompt management for consistent AI interactions:
- Templated prompts for different use cases
- Dynamic prompt generation with context
- Prompt optimization and versioning
- Multi-language support
- Performance monitoring and A/B testing
"""

import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import structlog

from app.core.config import get_settings

logger = structlog.get_logger(__name__)


class PromptType(Enum):
    """Types of prompts supported by the system"""
    QUERY_EXPANSION = "query_expansion"
    RESULT_RANKING = "result_ranking"
    CV_ANALYSIS = "cv_analysis"
    JOB_MATCHING = "job_matching"
    SKILL_EXTRACTION = "skill_extraction"
    DOCUMENT_SUMMARY = "document_summary"
    CONTENT_CLASSIFICATION = "content_classification"
    QUALITY_ASSESSMENT = "quality_assessment"


class PromptManager:
    """
    Comprehensive prompt management system for AI operations.
    
    Features:
    - Template-based prompt generation
    - Context-aware dynamic prompts
    - Prompt versioning and optimization
    - Performance tracking and analytics
    - Multi-language support
    - Caching for frequently used prompts
    """
    
    def __init__(self, cache_service=None):
        self.settings = get_settings()
        self.cache_service = cache_service
        self._prompt_templates = self._initialize_templates()
        self._metrics = {
            "prompts_generated": 0,
            "template_usage": {},
            "cache_hits": 0
        }
        
    def _initialize_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize prompt templates"""
        return {
            PromptType.QUERY_EXPANSION.value: {
                "system": """You are an expert at understanding job search queries and expanding them to find relevant opportunities. 
Your task is to expand and enhance search queries to improve job matching results.

Guidelines:
- Identify core skills, technologies, and job roles
- Add relevant synonyms and related terms
- Include industry-standard abbreviations and variations
- Consider seniority levels and experience requirements
- Maintain the original intent while broadening scope""",
                "user": """Original query: "{query}"

Expand this query to include:
1. Related skills and technologies
2. Alternative job titles and roles
3. Industry synonyms and abbreviations
4. Experience level variations

Return a JSON object with:
- "expanded_terms": list of additional search terms
- "primary_skills": list of main skills identified
- "job_roles": list of relevant job titles
- "experience_level": estimated experience level
- "industry": likely industry/domain""",
                "temperature": 0.3,
                "max_tokens": 500
            },
            
            PromptType.RESULT_RANKING.value: {
                "system": """You are an expert recruiter who excels at matching candidates to job requirements. 
Your task is to rank search results based on relevance and fit quality.

Consider these factors:
- Skill alignment and proficiency levels
- Experience relevance and duration
- Industry background and domain knowledge
- Career progression and growth trajectory
- Cultural and role fit indicators""",
                "user": """Job Requirements: {job_requirements}

Candidate Results to Rank:
{candidate_results}

Rank these candidates from 1 (best fit) to N (least fit) and provide:
- "ranking": list of candidate IDs in order of fit
- "scores": relevance score (0-100) for each candidate
- "reasoning": brief explanation for top 3 candidates
- "red_flags": any concerns or misalignments noted""",
                "temperature": 0.2,
                "max_tokens": 800
            },
            
            PromptType.CV_ANALYSIS.value: {
                "system": """You are an experienced HR professional specializing in CV analysis and candidate assessment.
Your task is to extract and analyze key information from CVs for recruitment purposes.

Focus on:
- Professional experience and career progression
- Technical skills and proficiency levels
- Educational background and certifications
- Achievements and measurable results
- Industry expertise and domain knowledge""",
                "user": """Analyze this CV content and extract structured information:

CV Content:
{cv_content}

Provide a comprehensive analysis with:
- "summary": 2-3 sentence professional summary
- "experience": work history with roles, companies, durations
- "skills": technical and soft skills with proficiency estimates
- "education": degrees, certifications, relevant training
- "achievements": quantifiable accomplishments
- "keywords": important terms for search matching
- "seniority_level": estimated experience level (junior/mid/senior/lead)
- "industries": relevant industry experience""",
                "temperature": 0.1,
                "max_tokens": 1000
            },
            
            PromptType.JOB_MATCHING.value: {
                "system": """You are a specialized job matching algorithm that understands both candidate profiles and job requirements.
Your task is to assess compatibility between candidates and job opportunities.

Evaluate:
- Skill requirements vs candidate abilities
- Experience level and seniority match
- Industry and domain alignment
- Cultural fit indicators
- Growth potential and career trajectory""",
                "user": """Candidate Profile: {candidate_profile}

Job Description: {job_description}

Assess the match quality and provide:
- "overall_score": compatibility score (0-100)
- "skill_match": technical skills alignment (0-100)
- "experience_match": experience level fit (0-100)
- "industry_fit": industry background relevance (0-100)
- "strengths": top 3 reasons this is a good match
- "gaps": areas where candidate doesn't fully meet requirements
- "recommendation": hire/interview/pass with reasoning""",
                "temperature": 0.2,
                "max_tokens": 600
            },
            
            PromptType.SKILL_EXTRACTION.value: {
                "system": """You are a skills taxonomy expert who identifies and categorizes professional skills from text.
Your task is to extract and classify skills mentioned in job descriptions or CVs.

Categories to identify:
- Technical skills (programming languages, tools, frameworks)
- Domain expertise (industry knowledge, methodologies)
- Soft skills (leadership, communication, problem-solving)
- Certifications and qualifications
- Years of experience for each skill when mentioned""",
                "user": """Extract skills from this text:

Text: {text_content}

Provide structured output:
- "technical_skills": list of technical skills with proficiency level if mentioned
- "domain_skills": industry/domain specific knowledge
- "soft_skills": interpersonal and management skills
- "certifications": mentioned certifications or qualifications
- "tools": software, platforms, and tools mentioned
- "experience_years": skill experience durations when specified
- "skill_categories": main skill domains identified""",
                "temperature": 0.1,
                "max_tokens": 500
            },
            
            PromptType.DOCUMENT_SUMMARY.value: {
                "system": """You are an expert at creating concise, informative summaries of professional documents.
Your task is to distill key information while maintaining important context and details.

Focus on:
- Key achievements and qualifications
- Most relevant experience and skills
- Important metrics and quantifiable results
- Career highlights and progression
- Critical information for decision making""",
                "user": """Create a professional summary of this document:

Document Content: {document_content}

Generate:
- "executive_summary": 2-3 sentence overview
- "key_highlights": top 5 most important points
- "relevant_experience": most applicable work history
- "core_competencies": primary skills and expertise areas
- "notable_achievements": measurable accomplishments
- "professional_level": experience level assessment""",
                "temperature": 0.3,
                "max_tokens": 400
            },
            
            PromptType.CONTENT_CLASSIFICATION.value: {
                "system": """You are a content classification expert who categorizes and labels professional content.
Your task is to identify the type, quality, and characteristics of submitted content.

Categories to assess:
- Content type (CV, cover letter, portfolio, etc.)
- Industry/domain focus
- Experience level indicators
- Content quality and completeness
- Relevance for recruitment purposes""",
                "user": """Classify and analyze this content:

Content: {content}

Provide classification:
- "content_type": type of document/content
- "industry": primary industry focus
- "experience_level": apparent experience level
- "quality_score": content quality (0-100)
- "completeness": how complete the information is (0-100)
- "relevance": relevance for job matching (0-100)
- "tags": descriptive tags for categorization
- "improvements": suggestions for content enhancement""",
                "temperature": 0.2,
                "max_tokens": 300
            },
            
            PromptType.QUALITY_ASSESSMENT.value: {
                "system": """You are a quality assurance expert for recruitment content and data.
Your task is to assess the quality, completeness, and usability of candidate information.

Evaluate:
- Information completeness and accuracy
- Professional presentation quality
- Relevance and specificity of details
- Consistency and coherence
- Suitability for automated processing""",
                "user": """Assess the quality of this candidate information:

Content: {content}

Provide quality assessment:
- "overall_quality": overall quality score (0-100)
- "completeness": information completeness (0-100)
- "accuracy": apparent accuracy and consistency (0-100)
- "presentation": professional presentation quality (0-100)
- "usability": suitability for matching algorithms (0-100)
- "missing_elements": important information that's missing
- "quality_issues": specific problems identified
- "recommendations": suggestions for improvement""",
                "temperature": 0.1,
                "max_tokens": 400
            }
        }
    
    async def generate_prompt(
        self,
        prompt_type: Union[PromptType, str],
        context: Dict[str, Any],
        custom_instructions: Optional[str] = None,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Generate a prompt for specific AI operations.
        
        Args:
            prompt_type: Type of prompt to generate
            context: Context variables for template substitution
            custom_instructions: Additional custom instructions
            language: Language for the prompt
            
        Returns:
            Dictionary containing the generated prompt
        """
        if isinstance(prompt_type, str):
            prompt_type_str = prompt_type
        else:
            prompt_type_str = prompt_type.value
        
        if prompt_type_str not in self._prompt_templates:
            raise ValueError(f"Unknown prompt type: {prompt_type_str}")
        
        # Check cache first
        cache_key = self._get_cache_key(prompt_type_str, context, custom_instructions)
        if self.cache_service:
            cached_prompt = await self.cache_service.get(cache_key)
            if cached_prompt:
                self._metrics["cache_hits"] += 1
                return cached_prompt
        
        try:
            template = self._prompt_templates[prompt_type_str]
            
            # Generate system and user messages
            system_message = template["system"]
            user_message = template["user"].format(**context)
            
            # Add custom instructions if provided
            if custom_instructions:
                user_message += f"\n\nAdditional Instructions: {custom_instructions}"
            
            # Build prompt structure
            prompt = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "temperature": template.get("temperature", 0.7),
                "max_tokens": template.get("max_tokens", 500),
                "prompt_type": prompt_type_str,
                "language": language,
                "generated_at": datetime.now().isoformat()
            }
            
            # Cache the generated prompt
            if self.cache_service:
                await self.cache_service.set(
                    cache_key,
                    prompt,
                    ttl=self.settings.CACHE_TTL
                )
            
            # Update metrics
            self._metrics["prompts_generated"] += 1
            if prompt_type_str not in self._metrics["template_usage"]:
                self._metrics["template_usage"][prompt_type_str] = 0
            self._metrics["template_usage"][prompt_type_str] += 1
            
            logger.debug(
                "Generated prompt",
                prompt_type=prompt_type_str,
                language=language,
                context_keys=list(context.keys())
            )
            
            return prompt
            
        except Exception as e:
            logger.error(f"Failed to generate prompt: {e}")
            raise
    
    async def create_query_expansion_prompt(
        self,
        query: str,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create prompt for query expansion"""
        context = {"query": query}
        if additional_context:
            context.update(additional_context)
        
        return await self.generate_prompt(
            PromptType.QUERY_EXPANSION,
            context
        )
    
    async def create_cv_analysis_prompt(
        self,
        cv_content: str,
        analysis_focus: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create prompt for CV analysis"""
        context = {"cv_content": cv_content}
        
        custom_instructions = None
        if analysis_focus:
            custom_instructions = f"Focus particularly on: {analysis_focus}"
        
        return await self.generate_prompt(
            PromptType.CV_ANALYSIS,
            context,
            custom_instructions
        )
    
    async def create_job_matching_prompt(
        self,
        candidate_profile: str,
        job_description: str,
        matching_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create prompt for job matching assessment"""
        context = {
            "candidate_profile": candidate_profile,
            "job_description": job_description
        }
        
        custom_instructions = None
        if matching_criteria:
            criteria_text = json.dumps(matching_criteria, indent=2)
            custom_instructions = f"Use these specific matching criteria: {criteria_text}"
        
        return await self.generate_prompt(
            PromptType.JOB_MATCHING,
            context,
            custom_instructions
        )
    
    async def create_result_ranking_prompt(
        self,
        job_requirements: str,
        candidate_results: List[Dict[str, Any]],
        ranking_factors: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create prompt for result ranking"""
        # Format candidate results for prompt
        formatted_results = []
        for i, candidate in enumerate(candidate_results):
            formatted_results.append(f"Candidate {i+1} (ID: {candidate.get('id', 'unknown')}):\n{candidate.get('summary', 'No summary available')}")
        
        context = {
            "job_requirements": job_requirements,
            "candidate_results": "\n\n".join(formatted_results)
        }
        
        custom_instructions = None
        if ranking_factors:
            custom_instructions = f"Prioritize these ranking factors: {', '.join(ranking_factors)}"
        
        return await self.generate_prompt(
            PromptType.RESULT_RANKING,
            context,
            custom_instructions
        )
    
    def get_available_prompt_types(self) -> List[str]:
        """Get list of available prompt types"""
        return list(self._prompt_templates.keys())
    
    def get_template_info(self, prompt_type: Union[PromptType, str]) -> Dict[str, Any]:
        """Get information about a specific prompt template"""
        if isinstance(prompt_type, str):
            prompt_type_str = prompt_type
        else:
            prompt_type_str = prompt_type.value
        
        if prompt_type_str not in self._prompt_templates:
            raise ValueError(f"Unknown prompt type: {prompt_type_str}")
        
        template = self._prompt_templates[prompt_type_str]
        return {
            "prompt_type": prompt_type_str,
            "system_message_length": len(template["system"]),
            "user_template_length": len(template["user"]),
            "default_temperature": template.get("temperature", 0.7),
            "default_max_tokens": template.get("max_tokens", 500),
            "required_context_vars": self._extract_template_variables(template["user"])
        }
    
    def _extract_template_variables(self, template: str) -> List[str]:
        """Extract variable names from template string"""
        import re
        variables = re.findall(r'\{([^}]+)\}', template)
        return list(set(variables))
    
    def _get_cache_key(
        self,
        prompt_type: str,
        context: Dict[str, Any],
        custom_instructions: Optional[str] = None
    ) -> str:
        """Generate cache key for prompt"""
        import hashlib
        
        cache_data = {
            "prompt_type": prompt_type,
            "context": context,
            "custom_instructions": custom_instructions
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        cache_hash = hashlib.sha256(cache_string.encode()).hexdigest()[:16]
        
        return f"prompt:{prompt_type}:{cache_hash}"
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get prompt manager metrics"""
        return {
            "generation_stats": self._metrics.copy(),
            "available_templates": len(self._prompt_templates),
            "cache_enabled": self.cache_service is not None
        }
    
    async def check_health(self) -> Dict[str, Any]:
        """Check prompt manager health"""
        try:
            # Test template generation
            test_context = {"query": "test query"}
            test_prompt = await self.generate_prompt(
                PromptType.QUERY_EXPANSION,
                test_context
            )
            
            return {
                "status": "healthy",
                "templates_loaded": len(self._prompt_templates),
                "test_generation": "successful",
                "cache_available": self.cache_service is not None,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }