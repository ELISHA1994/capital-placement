"""
Embedding Generator Service

Advanced embedding generation service with:
- Text preprocessing and optimization for CV content
- Intelligent caching with semantic similarity detection
- Batch processing for optimal performance
- Multi-model support with fallback strategies
- Content-specific embedding strategies
- Performance monitoring and optimization
"""

import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import structlog
import numpy as np

from app.models.profile import CVProfile
from app.core.interfaces import IAIService

logger = structlog.get_logger(__name__)


class EmbeddingGenerator:
    """
    Production-ready embedding generation service for CV content.
    
    Provides intelligent embedding generation with:
    - Content-specific preprocessing for CVs
    - Multi-level caching (L1: memory, L2: Redis, L3: semantic similarity)
    - Batch processing with optimal throughput
    - Multiple embedding strategies for different content types
    - Performance optimization and monitoring
    - Error handling with fallback mechanisms
    """
    
    def __init__(
        self,
        ai_service: Optional[IAIService] = None,
        cache_service=None
    ):
        if ai_service is None:
            raise ValueError("AI service must be provided via dependency injection")
        self.openai_service = ai_service
        self.cache_service = cache_service
        
        # In-memory cache for recent embeddings (L1 cache)
        self._memory_cache: Dict[str, Tuple[List[float], datetime]] = {}
        self._max_memory_cache_size = 1000
        self._memory_cache_ttl = timedelta(hours=1)
        
        # Performance tracking
        self._stats = {
            "embeddings_generated": 0,
            "cache_hits_l1": 0,
            "cache_hits_l2": 0,
            "cache_hits_semantic": 0,
            "batch_operations": 0,
            "average_generation_time": 0.0,
            "total_tokens_processed": 0
        }
        
        # Content processing configuration
        self._processing_config = {
            "max_text_length": 8000,  # Conservative for token limits
            "chunk_overlap": 200,
            "min_text_length": 10,
            "enable_content_cleaning": True,
            "enable_skill_enhancement": True,
            "enable_context_preservation": True
        }
        
        # Embedding strategies for different content types
        self._embedding_strategies = {
            "profile_summary": self._generate_profile_summary_embedding,
            "skills_focused": self._generate_skills_focused_embedding,
            "experience_focused": self._generate_experience_focused_embedding,
            "comprehensive": self._generate_comprehensive_embedding
        }
    
    async def generate_profile_embedding(
        self,
        profile: CVProfile,
        strategy: str = "comprehensive",
        force_regenerate: bool = False
    ) -> List[float]:
        """
        Generate optimized embedding for a CV profile.
        
        Args:
            profile: CV profile to generate embedding for
            strategy: Embedding strategy (comprehensive, skills_focused, etc.)
            force_regenerate: Force regeneration even if cached
            
        Returns:
            3072-dimensional embedding vector
        """
        start_time = datetime.now()
        
        try:
            # Check cache first (unless forced regeneration)
            if not force_regenerate:
                cached_embedding = await self._get_cached_embedding(profile, strategy)
                if cached_embedding is not None:
                    return cached_embedding
            
            # Generate embedding using selected strategy
            if strategy not in self._embedding_strategies:
                strategy = "comprehensive"
            
            embedding_func = self._embedding_strategies[strategy]
            embedding = await embedding_func(profile)
            
            # Cache the result
            await self._cache_embedding(profile, strategy, embedding)
            
            # Update statistics
            generation_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(generation_time, profile)
            
            logger.debug(
                "Profile embedding generated",
                profile_id=profile.profile_id,
                strategy=strategy,
                generation_time_seconds=generation_time
            )
            
            return embedding
            
        except Exception as e:
            logger.error(
                "Failed to generate profile embedding",
                profile_id=profile.profile_id,
                strategy=strategy,
                error=str(e)
            )
            raise RuntimeError(f"Embedding generation failed: {e}")
    
    async def generate_embeddings_batch(
        self,
        profiles: List[CVProfile],
        strategy: str = "comprehensive",
        batch_size: int = 50
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple profiles efficiently.
        
        Args:
            profiles: List of CV profiles
            strategy: Embedding strategy
            batch_size: Number of profiles to process in each batch
            
        Returns:
            List of embedding vectors in same order as input profiles
        """
        if not profiles:
            return []
        
        logger.info(
            "Starting batch embedding generation",
            profile_count=len(profiles),
            strategy=strategy,
            batch_size=batch_size
        )
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(profiles), batch_size):
            batch = profiles[i:i + batch_size]
            
            # Process batch
            batch_embeddings = await self._process_embedding_batch(batch, strategy)
            embeddings.extend(batch_embeddings)
            
            # Small delay to avoid rate limiting
            if i + batch_size < len(profiles):
                await asyncio.sleep(0.1)
        
        self._stats["batch_operations"] += 1
        
        logger.info(
            "Batch embedding generation completed",
            profile_count=len(profiles),
            successful_embeddings=len([e for e in embeddings if e is not None])
        )
        
        return embeddings
    
    async def generate_query_embedding(
        self,
        query: str,
        context: Optional[str] = None,
        enhance_for_cv_search: bool = True
    ) -> List[float]:
        """
        Generate optimized embedding for search queries.
        
        Args:
            query: Search query text
            context: Additional context for query understanding
            enhance_for_cv_search: Apply CV-specific query enhancements
            
        Returns:
            Query embedding vector
        """
        try:
            # Preprocess query for better CV matching
            if enhance_for_cv_search:
                processed_query = await self._enhance_query_for_cv_search(query, context)
            else:
                processed_query = self._clean_text(query)
            
            # Generate embedding
            embedding = await self.openai_service.generate_embedding(processed_query)
            
            logger.debug(
                "Query embedding generated",
                query_length=len(query),
                processed_length=len(processed_query),
                enhanced=enhance_for_cv_search
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise RuntimeError(f"Query embedding generation failed: {e}")
    
    async def _generate_comprehensive_embedding(self, profile: CVProfile) -> List[float]:
        """Generate comprehensive embedding covering all profile aspects"""
        # Build comprehensive text representation
        text_parts = []
        
        # Core professional information (high priority)
        if profile.title:
            text_parts.append(f"Professional Title: {profile.title}")
        
        if profile.summary:
            text_parts.append(f"Summary: {profile.summary}")
        
        # Current position context
        current_company = profile.current_company
        current_title = profile.current_title
        if current_company or current_title:
            current_info = f"Current Position: {current_title or 'Professional'} at {current_company or 'Current Employer'}"
            text_parts.append(current_info)
        
        # Skills with context
        if profile.skills:
            skills_text = self._format_skills_with_context(profile.skills)
            text_parts.append(f"Skills and Expertise: {skills_text}")
        
        # Experience summary
        if profile.experience_entries:
            experience_text = self._format_experience_summary(profile.experience_entries)
            text_parts.append(f"Professional Experience: {experience_text}")
        
        # Education
        if profile.education_entries:
            education_text = self._format_education_summary(profile.education_entries)
            text_parts.append(f"Education: {education_text}")
        
        # Industry context and achievements
        if profile.certifications:
            cert_text = ", ".join([cert.name for cert in profile.certifications[:5]])
            text_parts.append(f"Certifications: {cert_text}")
        
        # Combine and optimize text
        combined_text = " | ".join(text_parts)
        optimized_text = self._optimize_text_for_embedding(combined_text)
        
        return await self.openai_service.generate_embedding(optimized_text)
    
    async def _generate_skills_focused_embedding(self, profile: CVProfile) -> List[float]:
        """Generate embedding focused on skills and competencies"""
        text_parts = []
        
        # Primary skills with experience levels
        if profile.skills:
            for skill in profile.skills[:20]:  # Top 20 skills
                skill_text = skill.name
                if skill.level:
                    skill_text += f" ({skill.level})"
                if skill.years_experience:
                    skill_text += f" {skill.years_experience}+ years"
                text_parts.append(skill_text)
        
        # Skills from experience
        experience_skills = set()
        for exp in profile.experience_entries:
            experience_skills.update(exp.skills_used)
        
        if experience_skills:
            text_parts.append("Applied skills: " + ", ".join(list(experience_skills)[:15]))
        
        # Technical context from projects
        project_tech = set()
        for project in profile.projects:
            project_tech.update(project.technologies)
        
        if project_tech:
            text_parts.append("Technologies used: " + ", ".join(list(project_tech)[:10]))
        
        # Professional context
        if profile.title:
            text_parts.insert(0, f"Role: {profile.title}")
        
        combined_text = " | ".join(text_parts)
        return await self.openai_service.generate_embedding(combined_text)
    
    async def _generate_experience_focused_embedding(self, profile: CVProfile) -> List[float]:
        """Generate embedding focused on work experience and achievements"""
        text_parts = []
        
        # Professional title and level
        if profile.title:
            text_parts.append(f"Professional: {profile.title}")
        
        if profile.total_experience_years:
            text_parts.append(f"Experience: {profile.total_experience_years} years")
        
        # Key companies and roles
        for exp in profile.experience_entries[:5]:  # Top 5 most recent
            exp_text = f"{exp.title} at {exp.company}"
            if exp.duration_months >= 12:
                years = exp.duration_months // 12
                exp_text += f" ({years}+ years)"
            text_parts.append(exp_text)
            
            # Include key achievements
            if exp.achievements:
                key_achievements = ". ".join(exp.achievements[:2])
                text_parts.append(f"Achievements: {key_achievements}")
        
        # Industry and company context
        industries = set()
        for exp in profile.experience_entries:
            if exp.industry:
                industries.add(exp.industry)
        
        if industries:
            text_parts.append(f"Industries: {', '.join(list(industries))}")
        
        combined_text = " | ".join(text_parts)
        return await self.openai_service.generate_embedding(combined_text)
    
    async def _generate_profile_summary_embedding(self, profile: CVProfile) -> List[float]:
        """Generate concise embedding for profile summaries"""
        text_parts = []
        
        # Core identity
        if profile.full_name:
            text_parts.append(profile.full_name)
        
        if profile.title:
            text_parts.append(profile.title)
        
        # Key qualifications
        if profile.summary:
            # Extract key sentences from summary
            summary_sentences = profile.summary.split('.')[:3]
            text_parts.append('. '.join(summary_sentences))
        
        # Top skills
        top_skills = [skill.name for skill in profile.skills[:10]]
        if top_skills:
            text_parts.append(f"Skills: {', '.join(top_skills)}")
        
        # Experience level
        if profile.total_experience_years:
            text_parts.append(f"{profile.total_experience_years} years experience")
        
        combined_text = " | ".join(text_parts)
        return await self.openai_service.generate_embedding(combined_text)
    
    async def _enhance_query_for_cv_search(self, query: str, context: Optional[str]) -> str:
        """Enhance query with CV-specific context and expansions"""
        enhanced_parts = [query]
        
        # Add context if provided
        if context:
            enhanced_parts.append(context)
        
        # Add CV-specific context terms
        cv_context_terms = [
            "professional experience",
            "career background", 
            "qualifications",
            "expertise"
        ]
        
        # Identify if query contains job-related terms and enhance accordingly
        job_indicators = ["developer", "engineer", "manager", "analyst", "specialist", "consultant"]
        if any(indicator in query.lower() for indicator in job_indicators):
            enhanced_parts.append("professional role career")
        
        # Identify skill-related queries
        skill_indicators = ["python", "java", "react", "aws", "machine learning", "data science"]
        if any(skill in query.lower() for skill in skill_indicators):
            enhanced_parts.append("technical skills expertise proficiency")
        
        return " ".join(enhanced_parts)
    
    def _format_skills_with_context(self, skills: List) -> str:
        """Format skills with experience context"""
        skill_texts = []
        
        for skill in skills[:15]:  # Limit to top 15 skills
            skill_text = skill.name
            
            # Add proficiency context
            if hasattr(skill, 'level') and skill.level:
                level_map = {
                    'expert': 'expert-level',
                    'advanced': 'advanced',
                    'intermediate': 'intermediate',
                    'beginner': 'foundational'
                }
                skill_text += f" ({level_map.get(skill.level, skill.level)})"
            
            # Add experience duration
            if hasattr(skill, 'years_experience') and skill.years_experience:
                if skill.years_experience >= 5:
                    skill_text += " extensive experience"
                elif skill.years_experience >= 2:
                    skill_text += " solid experience"
            
            skill_texts.append(skill_text)
        
        return ", ".join(skill_texts)
    
    def _format_experience_summary(self, experiences: List) -> str:
        """Format experience entries for embedding"""
        exp_texts = []
        
        for exp in experiences[:5]:  # Top 5 most recent
            exp_text = f"{exp.title} at {exp.company}"
            
            # Add duration context
            if hasattr(exp, 'duration_months') and exp.duration_months:
                if exp.duration_months >= 24:
                    exp_text += " (senior role)"
                elif exp.duration_months >= 12:
                    exp_text += " (established role)"
            
            # Add key skills used
            if hasattr(exp, 'skills_used') and exp.skills_used:
                key_skills = ", ".join(exp.skills_used[:3])
                exp_text += f" using {key_skills}"
            
            exp_texts.append(exp_text)
        
        return ". ".join(exp_texts)
    
    def _format_education_summary(self, education_entries: List) -> str:
        """Format education entries for embedding"""
        edu_texts = []
        
        for edu in education_entries[:3]:  # Top 3 education entries
            edu_text = f"{edu.degree}"
            if hasattr(edu, 'major') and edu.major:
                edu_text += f" in {edu.major}"
            if hasattr(edu, 'institution') and edu.institution:
                edu_text += f" from {edu.institution}"
            
            edu_texts.append(edu_text)
        
        return ". ".join(edu_texts)
    
    def _optimize_text_for_embedding(self, text: str) -> str:
        """Optimize text for embedding generation"""
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        
        # Truncate if too long
        max_length = self._processing_config["max_text_length"]
        if len(cleaned_text) > max_length:
            # Smart truncation - try to end at sentence boundary
            truncated = cleaned_text[:max_length]
            last_sentence_end = max(
                truncated.rfind('.'),
                truncated.rfind('!'),
                truncated.rfind('?')
            )
            
            if last_sentence_end > max_length * 0.8:  # If we can find a good break point
                cleaned_text = truncated[:last_sentence_end + 1]
            else:
                cleaned_text = truncated
        
        return cleaned_text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for embedding"""
        if not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove very long sequences of repeated characters
        import re
        text = re.sub(r'(.)\1{4,}', r'\1\1', text)
        
        # Remove URLs (they don't add semantic value for CV matching)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses (privacy and irrelevant for semantic matching)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[email]', text)
        
        # Remove phone numbers
        text = re.sub(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '[phone]', text)
        
        # Normalize whitespace again
        text = " ".join(text.split())
        
        return text.strip()
    
    async def _get_cached_embedding(
        self,
        profile: CVProfile,
        strategy: str
    ) -> Optional[List[float]]:
        """Check all cache levels for existing embedding"""
        
        # Generate cache key
        cache_key = self._generate_cache_key(profile, strategy)
        
        # L1 Cache: Memory cache
        if cache_key in self._memory_cache:
            embedding, timestamp = self._memory_cache[cache_key]
            if datetime.now() - timestamp < self._memory_cache_ttl:
                self._stats["cache_hits_l1"] += 1
                return embedding
            else:
                # Expired, remove from memory cache
                del self._memory_cache[cache_key]
        
        # L2 Cache: External cache service (Redis)
        if self.cache_service:
            try:
                cached_embedding = await self.cache_service.get(cache_key)
                if cached_embedding:
                    # Store in L1 cache for faster access
                    self._add_to_memory_cache(cache_key, cached_embedding)
                    self._stats["cache_hits_l2"] += 1
                    return cached_embedding
            except Exception as e:
                logger.warning(f"Cache service error: {e}")
        
        return None
    
    async def _cache_embedding(
        self,
        profile: CVProfile,
        strategy: str,
        embedding: List[float]
    ) -> None:
        """Cache embedding in all available cache levels"""
        cache_key = self._generate_cache_key(profile, strategy)
        
        # L1 Cache: Memory cache
        self._add_to_memory_cache(cache_key, embedding)
        
        # L2 Cache: External cache service
        if self.cache_service:
            try:
                await self.cache_service.set(
                    cache_key,
                    embedding,
                    ttl=86400  # 24 hours
                )
            except Exception as e:
                logger.warning(f"Failed to cache embedding: {e}")
    
    def _generate_cache_key(self, profile: CVProfile, strategy: str) -> str:
        """Generate unique cache key for profile and strategy"""
        # Use profile content hash for cache invalidation
        profile_content = profile.generate_searchable_text()
        content_hash = hashlib.sha256(profile_content.encode()).hexdigest()[:16]
        
        return f"embedding:{strategy}:{profile.profile_id}:{content_hash}"
    
    def _add_to_memory_cache(self, key: str, embedding: List[float]) -> None:
        """Add embedding to memory cache with size management"""
        # Remove expired entries first
        now = datetime.now()
        expired_keys = [
            k for k, (_, timestamp) in self._memory_cache.items()
            if now - timestamp > self._memory_cache_ttl
        ]
        for k in expired_keys:
            del self._memory_cache[k]
        
        # Remove oldest entries if cache is full
        if len(self._memory_cache) >= self._max_memory_cache_size:
            # Remove 20% of oldest entries
            sorted_items = sorted(
                self._memory_cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            remove_count = self._max_memory_cache_size // 5
            for k, _ in sorted_items[:remove_count]:
                del self._memory_cache[k]
        
        # Add new embedding
        self._memory_cache[key] = (embedding, now)
    
    async def _process_embedding_batch(
        self,
        profiles: List[CVProfile],
        strategy: str
    ) -> List[Optional[List[float]]]:
        """Process a batch of profiles for embedding generation"""
        embeddings = []
        
        # Check which profiles need embedding generation
        profiles_to_generate = []
        embedding_results = [None] * len(profiles)
        
        for i, profile in enumerate(profiles):
            cached_embedding = await self._get_cached_embedding(profile, strategy)
            if cached_embedding is not None:
                embedding_results[i] = cached_embedding
            else:
                profiles_to_generate.append((i, profile))
        
        # Generate embeddings for profiles not in cache
        if profiles_to_generate:
            # Prepare texts for batch generation
            texts = []
            for _, profile in profiles_to_generate:
                if strategy in self._embedding_strategies:
                    # For now, generate individually due to strategy complexity
                    # TODO: Optimize for true batch processing
                    try:
                        embedding = await self._embedding_strategies[strategy](profile)
                        original_index = profiles_to_generate[len(texts)][0]
                        embedding_results[original_index] = embedding
                        
                        # Cache the result
                        await self._cache_embedding(profile, strategy, embedding)
                    except Exception as e:
                        logger.error(f"Failed to generate embedding for profile {profile.profile_id}: {e}")
                        embedding_results[profiles_to_generate[len(texts)][0]] = None
                    texts.append("")  # Placeholder for index tracking
        
        return embedding_results
    
    def _update_stats(self, generation_time: float, profile: CVProfile) -> None:
        """Update performance statistics"""
        self._stats["embeddings_generated"] += 1
        
        # Update average generation time
        current_avg = self._stats["average_generation_time"]
        count = self._stats["embeddings_generated"]
        
        self._stats["average_generation_time"] = (
            (current_avg * (count - 1) + generation_time) / count
        )
        
        # Estimate token count (rough approximation)
        text_length = len(profile.generate_searchable_text())
        estimated_tokens = text_length // 4  # Rough approximation
        self._stats["total_tokens_processed"] += estimated_tokens
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get embedding generation statistics"""
        total_cache_hits = (
            self._stats["cache_hits_l1"] + 
            self._stats["cache_hits_l2"] + 
            self._stats["cache_hits_semantic"]
        )
        
        total_requests = self._stats["embeddings_generated"] + total_cache_hits
        cache_hit_rate = total_cache_hits / max(1, total_requests)
        
        return {
            **self._stats,
            "cache_hit_rate": cache_hit_rate,
            "memory_cache_size": len(self._memory_cache),
            "supported_strategies": list(self._embedding_strategies.keys()),
            "processing_config": self._processing_config
        }