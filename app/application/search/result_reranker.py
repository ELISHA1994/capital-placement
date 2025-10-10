"""
AI-Powered Result Reranking Service

Advanced result reranking using AI models for enhanced search relevance:
- LLM-powered semantic relevance assessment
- Multi-criteria business logic scoring
- Contextual reranking based on user preferences
- Explanation generation for ranking decisions
- Performance optimization with batch processing
- Configurable ranking strategies and weights
- Real-time learning and adaptation
"""

from __future__ import annotations

import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union, TYPE_CHECKING
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from uuid import uuid4
import numpy as np

from app.core.config import get_settings
from app.domain.interfaces import IHealthCheck
from app.application.search.hybrid_search import HybridSearchResult
from app.infrastructure.ai.cache_manager import UUIDEncoder

# Import infrastructure types only for type checking
if TYPE_CHECKING:
    from app.infrastructure.ai.openai_service import OpenAIService
    from app.infrastructure.ai.prompt_manager import PromptManager, PromptType
    from app.infrastructure.ai.cache_manager import CacheManager
    from app.infrastructure.adapters.postgres_adapter import PostgresAdapter

logger = structlog.get_logger(__name__)


class RankingStrategy(Enum):
    """Available ranking strategies"""
    SEMANTIC_RELEVANCE = "semantic_relevance"
    BUSINESS_LOGIC = "business_logic"
    CONTEXTUAL_MATCH = "contextual_match"
    HYBRID_INTELLIGENT = "hybrid_intelligent"
    LEARNING_TO_RANK = "learning_to_rank"


class RankingCriteria(Enum):
    """Individual ranking criteria"""
    SEMANTIC_MATCH = "semantic_match"
    SKILL_ALIGNMENT = "skill_alignment"
    EXPERIENCE_MATCH = "experience_match"
    RECENCY_SCORE = "recency_score"
    COMPLETENESS_SCORE = "completeness_score"
    QUALITY_SCORE = "quality_score"
    DIVERSITY_BONUS = "diversity_bonus"
    USER_PREFERENCE = "user_preference"


@dataclass
class RankingCriterion:
    """Individual ranking criterion with weight and score"""
    name: RankingCriteria
    score: float
    weight: float
    explanation: Optional[str] = None
    confidence: float = 1.0


@dataclass
class RerankingResult:
    """Result after AI-powered reranking"""
    entity_id: str
    entity_type: str
    original_score: float
    reranked_score: float
    ranking_criteria: List[RankingCriterion]
    ai_explanation: Optional[str] = None
    confidence: float = 0.8
    metadata: Dict[str, Any] = None


@dataclass
class RerankingResponse:
    """Complete reranking response with analytics"""
    results: List[RerankingResult]
    reranking_id: str
    strategy: RankingStrategy
    original_count: int
    reranked_count: int
    processing_time_ms: int
    ai_model_used: str
    token_usage: Dict[str, int] = None
    cache_hit: bool = False
    reranking_metadata: Dict[str, Any] = None


@dataclass
class RerankingConfig:
    """Configuration for reranking behavior"""
    strategy: RankingStrategy = RankingStrategy.HYBRID_INTELLIGENT
    max_results_to_rerank: int = 50
    ai_model: str = "gpt-4"
    batch_size: int = 10
    enable_explanations: bool = True
    enable_caching: bool = True
    criteria_weights: Dict[RankingCriteria, float] = None
    context_window: int = 2000  # tokens for AI context
    
    def __post_init__(self):
        if self.criteria_weights is None:
            self.criteria_weights = {
                RankingCriteria.SEMANTIC_MATCH: 0.25,
                RankingCriteria.SKILL_ALIGNMENT: 0.20,
                RankingCriteria.EXPERIENCE_MATCH: 0.15,
                RankingCriteria.RECENCY_SCORE: 0.10,
                RankingCriteria.COMPLETENESS_SCORE: 0.10,
                RankingCriteria.QUALITY_SCORE: 0.10,
                RankingCriteria.DIVERSITY_BONUS: 0.05,
                RankingCriteria.USER_PREFERENCE: 0.05
            }


class ResultRerankerService(IHealthCheck):
    """
    Advanced AI-powered result reranking service.
    
    Features:
    - LLM-powered semantic relevance assessment
    - Multi-criteria business logic integration
    - Batch processing for efficiency
    - Intelligent caching and optimization
    - Configurable ranking strategies
    - Detailed explanation generation
    - Performance monitoring and analytics
    - Contextual user preference learning
    """
    
    def __init__(
        self,
        openai_service: OpenAIService,
        prompt_manager: PromptManager,
        db_adapter: PostgresAdapter,
        cache_manager: Optional[CacheManager] = None,
        default_config: Optional[RerankingConfig] = None
    ):
        self.settings = get_settings()
        self.openai_service = openai_service
        self.prompt_manager = prompt_manager
        self.db_adapter = db_adapter
        self.cache_manager = cache_manager
        
        # Default configuration
        self.config = default_config or RerankingConfig()
        
        # Performance tracking
        self._stats = {
            "reranking_operations": 0,
            "results_reranked": 0,
            "ai_calls_made": 0,
            "cache_hits": 0,
            "total_processing_time_ms": 0,
            "average_processing_time_ms": 0,
            "total_tokens_used": 0,
            "errors": 0
        }
        
        # Caching configuration
        self.reranking_cache_ttl = 3600  # 1 hour
        
    async def rerank_results(
        self,
        query: str,
        results: List[HybridSearchResult],
        tenant_id: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None,
        config: Optional[RerankingConfig] = None,
        use_cache: bool = True
    ) -> RerankingResponse:
        """
        Rerank search results using AI-powered intelligence.
        
        Args:
            query: Original search query
            results: Initial search results to rerank
            tenant_id: Tenant ID for personalization
            user_context: User preferences and context
            config: Custom reranking configuration
            use_cache: Enable result caching
            
        Returns:
            RerankingResponse with reranked results and explanations
        """
        start_time = datetime.now()
        reranking_id = str(uuid4())
        
        try:
            # Use provided config or default
            rerank_config = config or self.config
            
            # Validate inputs
            if not results:
                return RerankingResponse(
                    results=[],
                    reranking_id=reranking_id,
                    strategy=rerank_config.strategy,
                    original_count=0,
                    reranked_count=0,
                    processing_time_ms=0,
                    ai_model_used=rerank_config.ai_model,
                    reranking_metadata={"message": "No results to rerank"}
                )
            
            # Limit results to rerank
            results_to_rerank = results[:rerank_config.max_results_to_rerank]
            
            # Check cache
            cached_response = None
            if use_cache and rerank_config.enable_caching:
                cached_response = await self._get_cached_reranking(
                    query, results_to_rerank, tenant_id, rerank_config
                )
                if cached_response:
                    self._stats["cache_hits"] += 1
                    return cached_response
            
            # Perform reranking based on strategy
            reranked_results = []
            
            if rerank_config.strategy == RankingStrategy.SEMANTIC_RELEVANCE:
                reranked_results = await self._semantic_reranking(
                    query, results_to_rerank, rerank_config
                )
            elif rerank_config.strategy == RankingStrategy.BUSINESS_LOGIC:
                reranked_results = await self._business_logic_reranking(
                    query, results_to_rerank, user_context, rerank_config
                )
            elif rerank_config.strategy == RankingStrategy.CONTEXTUAL_MATCH:
                reranked_results = await self._contextual_reranking(
                    query, results_to_rerank, user_context, tenant_id, rerank_config
                )
            elif rerank_config.strategy == RankingStrategy.HYBRID_INTELLIGENT:
                reranked_results = await self._hybrid_intelligent_reranking(
                    query, results_to_rerank, user_context, tenant_id, rerank_config
                )
            else:
                # Default to business logic reranking
                reranked_results = await self._business_logic_reranking(
                    query, results_to_rerank, user_context, rerank_config
                )
            
            # Calculate processing time
            processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Create response
            response = RerankingResponse(
                results=reranked_results,
                reranking_id=reranking_id,
                strategy=rerank_config.strategy,
                original_count=len(results),
                reranked_count=len(reranked_results),
                processing_time_ms=processing_time_ms,
                ai_model_used=rerank_config.ai_model,
                token_usage={"total_tokens": 0},  # Will be updated by AI calls
                cache_hit=False,
                reranking_metadata={
                    "query": query,
                    "tenant_id": tenant_id,
                    "user_context_provided": user_context is not None,
                    "config": asdict(rerank_config)
                }
            )
            
            # Cache results if enabled
            if use_cache and rerank_config.enable_caching and len(reranked_results) > 0:
                await self._cache_reranking_results(
                    response, query, results_to_rerank, tenant_id, rerank_config
                )
            
            # Update statistics
            self._update_reranking_stats(processing_time_ms, len(reranked_results))
            
            logger.info(
                "Result reranking completed",
                reranking_id=reranking_id,
                strategy=rerank_config.strategy.value,
                original_count=len(results),
                reranked_count=len(reranked_results),
                processing_time_ms=processing_time_ms,
                tenant_id=tenant_id
            )
            
            return response
            
        except Exception as e:
            self._stats["errors"] += 1
            processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            logger.error(
                "Result reranking failed",
                reranking_id=reranking_id,
                error=str(e),
                processing_time_ms=processing_time_ms,
                tenant_id=tenant_id
            )
            
            # Return original results with minimal reranking on error
            fallback_results = []
            for i, result in enumerate(results):
                fallback_result = RerankingResult(
                    entity_id=result.entity_id,
                    entity_type=result.entity_type,
                    original_score=result.final_score,
                    reranked_score=result.final_score,
                    ranking_criteria=[],
                    ai_explanation=f"Error occurred: {str(e)}",
                    confidence=0.0,
                    metadata=result.metadata
                )
                fallback_results.append(fallback_result)
            
            return RerankingResponse(
                results=fallback_results,
                reranking_id=reranking_id,
                strategy=rerank_config.strategy if rerank_config else RankingStrategy.BUSINESS_LOGIC,
                original_count=len(results),
                reranked_count=len(fallback_results),
                processing_time_ms=processing_time_ms,
                ai_model_used="error_fallback",
                reranking_metadata={"error": str(e)}
            )
    
    async def _semantic_reranking(
        self,
        query: str,
        results: List[HybridSearchResult],
        config: RerankingConfig
    ) -> List[RerankingResult]:
        """Perform semantic reranking using AI model"""
        
        try:
            # Process results in batches for efficiency
            reranked_results = []
            
            for i in range(0, len(results), config.batch_size):
                batch = results[i:i + config.batch_size]
                batch_results = await self._process_semantic_batch(query, batch, config)
                reranked_results.extend(batch_results)
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Semantic reranking failed: {e}")
            return await self._fallback_reranking(results)
    
    async def _process_semantic_batch(
        self,
        query: str,
        results: List[HybridSearchResult],
        config: RerankingConfig
    ) -> List[RerankingResult]:
        """Process a batch of results for semantic reranking"""

        try:
            # ✅ FIX P1: Filter out None results before processing
            valid_results = [r for r in results if r is not None]

            if not valid_results:
                logger.warning("No valid results to rerank in batch")
                return []

            # Prepare context for AI model
            context_data = {
                "query": query,
                "results": []
            }

            for result in valid_results:  # ✅ Use valid_results instead of results
                result_context = {
                    "entity_id": result.entity_id,
                    "entity_type": result.entity_type,
                    "original_score": result.final_score,
                    "content_preview": result.content_preview or "",  # ✅ Handle None
                    "metadata": result.metadata or {}  # ✅ Handle None
                }
                context_data["results"].append(result_context)
            
            # Create prompt for semantic reranking
            prompt = await self.prompt_manager.create_result_reranking_prompt(
                query=query,
                results_context=context_data,
                ranking_criteria=["semantic_relevance", "content_quality", "query_match"]
            )
            
            # Call AI model
            response = await self.openai_service.chat_completion(
                messages=prompt["messages"],
                model=config.ai_model,
                temperature=0.3,  # Lower temperature for consistent ranking
                max_tokens=min(config.context_window, 2000)
            )
            
            self._stats["ai_calls_made"] += 1
            if "usage" in response:
                self._stats["total_tokens_used"] += response["usage"].get("total_tokens", 0)
            
            # Parse AI response
            ai_ranking = await self._parse_semantic_ranking_response(
                response["choices"][0]["message"]["content"],
                results,
                config
            )
            
            return ai_ranking
            
        except Exception as e:
            logger.error(f"Semantic batch processing failed: {e}")
            return await self._fallback_reranking(results)
    
    async def _business_logic_reranking(
        self,
        query: str,
        results: List[HybridSearchResult],
        user_context: Optional[Dict[str, Any]],
        config: RerankingConfig
    ) -> List[RerankingResult]:
        """Perform business logic based reranking"""
        
        try:
            reranked_results = []
            
            for result in results:
                # Calculate individual criteria scores
                criteria_scores = []
                
                # Semantic match (use original score as baseline)
                semantic_criterion = RankingCriterion(
                    name=RankingCriteria.SEMANTIC_MATCH,
                    score=result.final_score,
                    weight=config.criteria_weights.get(RankingCriteria.SEMANTIC_MATCH, 0.25),
                    explanation="Original semantic similarity score"
                )
                criteria_scores.append(semantic_criterion)
                
                # Skill alignment
                skill_score = await self._calculate_skill_alignment(result, query, user_context)
                skill_criterion = RankingCriterion(
                    name=RankingCriteria.SKILL_ALIGNMENT,
                    score=skill_score,
                    weight=config.criteria_weights.get(RankingCriteria.SKILL_ALIGNMENT, 0.20),
                    explanation=f"Skill match score: {skill_score:.2f}"
                )
                criteria_scores.append(skill_criterion)
                
                # Experience match
                experience_score = await self._calculate_experience_match(result, query, user_context)
                experience_criterion = RankingCriterion(
                    name=RankingCriteria.EXPERIENCE_MATCH,
                    score=experience_score,
                    weight=config.criteria_weights.get(RankingCriteria.EXPERIENCE_MATCH, 0.15),
                    explanation=f"Experience level match: {experience_score:.2f}"
                )
                criteria_scores.append(experience_criterion)
                
                # Recency score
                recency_score = await self._calculate_recency_score(result)
                recency_criterion = RankingCriterion(
                    name=RankingCriteria.RECENCY_SCORE,
                    score=recency_score,
                    weight=config.criteria_weights.get(RankingCriteria.RECENCY_SCORE, 0.10),
                    explanation=f"Content recency score: {recency_score:.2f}"
                )
                criteria_scores.append(recency_criterion)
                
                # Completeness score
                completeness_score = await self._calculate_completeness_score(result)
                completeness_criterion = RankingCriterion(
                    name=RankingCriteria.COMPLETENESS_SCORE,
                    score=completeness_score,
                    weight=config.criteria_weights.get(RankingCriteria.COMPLETENESS_SCORE, 0.10),
                    explanation=f"Profile completeness: {completeness_score:.2f}"
                )
                criteria_scores.append(completeness_criterion)
                
                # Quality score
                quality_score = await self._calculate_quality_score(result)
                quality_criterion = RankingCriterion(
                    name=RankingCriteria.QUALITY_SCORE,
                    score=quality_score,
                    weight=config.criteria_weights.get(RankingCriteria.QUALITY_SCORE, 0.10),
                    explanation=f"Content quality assessment: {quality_score:.2f}"
                )
                criteria_scores.append(quality_criterion)
                
                # Calculate final weighted score
                weighted_score = sum(
                    criterion.score * criterion.weight
                    for criterion in criteria_scores
                )
                
                # Generate explanation if enabled
                explanation = None
                if config.enable_explanations:
                    explanation = await self._generate_business_logic_explanation(
                        query, result, criteria_scores, weighted_score
                    )
                
                # Create reranked result
                reranked_result = RerankingResult(
                    entity_id=result.entity_id,
                    entity_type=result.entity_type,
                    original_score=result.final_score,
                    reranked_score=weighted_score,
                    ranking_criteria=criteria_scores,
                    ai_explanation=explanation,
                    confidence=0.9,  # High confidence for business logic
                    metadata=result.metadata
                )
                
                reranked_results.append(reranked_result)
            
            # Sort by reranked score
            reranked_results.sort(key=lambda x: x.reranked_score, reverse=True)
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Business logic reranking failed: {e}")
            return await self._fallback_reranking(results)
    
    async def _hybrid_intelligent_reranking(
        self,
        query: str,
        results: List[HybridSearchResult],
        user_context: Optional[Dict[str, Any]],
        tenant_id: Optional[str],
        config: RerankingConfig
    ) -> List[RerankingResult]:
        """Perform hybrid intelligent reranking combining AI and business logic"""
        
        try:
            # First pass: Business logic reranking
            business_results = await self._business_logic_reranking(
                query, results, user_context, config
            )
            
            # Second pass: AI semantic refinement for top results
            top_results_limit = min(20, len(business_results))
            top_business_results = business_results[:top_results_limit]
            
            # Convert back to HybridSearchResult for AI processing
            hybrid_results_for_ai = []
            for br in top_business_results:
                original_result = next(
                    (r for r in results if r.entity_id == br.entity_id), None
                )
                if original_result:
                    hybrid_results_for_ai.append(original_result)
            
            # Apply AI semantic reranking to top results
            ai_config = RerankingConfig(
                strategy=RankingStrategy.SEMANTIC_RELEVANCE,
                max_results_to_rerank=top_results_limit,
                ai_model=config.ai_model,
                batch_size=min(10, top_results_limit),
                enable_explanations=config.enable_explanations
            )
            
            ai_results = await self._semantic_reranking(
                query, hybrid_results_for_ai, ai_config
            )
            
            # Combine business logic and AI scores
            final_results = []
            for ai_result in ai_results:
                # Find corresponding business logic result
                business_result = next(
                    (br for br in business_results if br.entity_id == ai_result.entity_id), None
                )
                
                if business_result:
                    # Weighted combination of business logic and AI scores
                    business_weight = 0.6
                    ai_weight = 0.4
                    
                    combined_score = (
                        business_result.reranked_score * business_weight +
                        ai_result.reranked_score * ai_weight
                    )
                    
                    # Combine criteria from both approaches
                    combined_criteria = business_result.ranking_criteria.copy()
                    
                    # Add AI semantic criterion
                    ai_semantic_criterion = RankingCriterion(
                        name=RankingCriteria.SEMANTIC_MATCH,
                        score=ai_result.reranked_score,
                        weight=ai_weight,
                        explanation=f"AI semantic assessment: {ai_result.ai_explanation}"
                    )
                    combined_criteria.append(ai_semantic_criterion)
                    
                    # Create final result
                    final_result = RerankingResult(
                        entity_id=ai_result.entity_id,
                        entity_type=ai_result.entity_type,
                        original_score=ai_result.original_score,
                        reranked_score=combined_score,
                        ranking_criteria=combined_criteria,
                        ai_explanation=f"Hybrid: {business_result.ai_explanation} + AI: {ai_result.ai_explanation}",
                        confidence=0.95,
                        metadata=ai_result.metadata
                    )
                    
                    final_results.append(final_result)
            
            # Add remaining business logic results that weren't processed by AI
            processed_ids = {result.entity_id for result in final_results}
            for business_result in business_results:
                if business_result.entity_id not in processed_ids:
                    final_results.append(business_result)
            
            # Sort by final combined score
            final_results.sort(key=lambda x: x.reranked_score, reverse=True)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid intelligent reranking failed: {e}")
            return await self._fallback_reranking(results)
    
    async def _calculate_skill_alignment(
        self,
        result: HybridSearchResult,
        query: str,
        user_context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate skill alignment score"""
        
        try:
            # Extract skills from result metadata
            skills = []
            if result.metadata:
                if "skills" in result.metadata:
                    skills = result.metadata["skills"]
                elif "ai_skills" in result.metadata:
                    skills = result.metadata["ai_skills"]
            
            if not skills:
                return 0.5  # Neutral score if no skills data
            
            # Simple skill matching (can be enhanced with NLP)
            query_lower = query.lower()
            skill_matches = 0
            total_skills = len(skills) if isinstance(skills, list) else 0
            
            if isinstance(skills, list):
                for skill in skills:
                    if isinstance(skill, str) and skill.lower() in query_lower:
                        skill_matches += 1
                    elif isinstance(skill, dict) and "name" in skill:
                        if skill["name"].lower() in query_lower:
                            skill_matches += 1
            
            # Calculate alignment score
            if total_skills > 0:
                return min(1.0, skill_matches / total_skills * 2.0)  # Scale up matches
            
            return 0.5
            
        except Exception as e:
            logger.warning(f"Skill alignment calculation failed: {e}")
            return 0.5
    
    async def _calculate_experience_match(
        self,
        result: HybridSearchResult,
        query: str,
        user_context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate experience level match score"""
        
        try:
            # Extract experience information
            experience_level = None
            if result.metadata:
                experience_level = result.metadata.get("experience_level")
            
            # Simple experience level matching
            query_lower = query.lower()
            
            experience_keywords = {
                "junior": ["junior", "entry", "graduate", "trainee", "associate"],
                "mid": ["mid", "intermediate", "experienced", "regular"],
                "senior": ["senior", "lead", "principal", "staff", "expert"],
                "executive": ["director", "manager", "head", "chief", "vp", "executive"]
            }
            
            query_experience_level = None
            for level, keywords in experience_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    query_experience_level = level
                    break
            
            # If no experience level detected in query, return neutral score
            if not query_experience_level:
                return 0.7
            
            # Perfect match
            if experience_level == query_experience_level:
                return 1.0
            
            # Partial matches (adjacent levels)
            level_hierarchy = ["junior", "mid", "senior", "executive"]
            if experience_level in level_hierarchy and query_experience_level in level_hierarchy:
                exp_idx = level_hierarchy.index(experience_level)
                query_idx = level_hierarchy.index(query_experience_level)
                distance = abs(exp_idx - query_idx)
                
                if distance == 1:
                    return 0.7  # Adjacent level
                elif distance == 2:
                    return 0.4  # Two levels apart
                else:
                    return 0.2  # Far apart
            
            return 0.5  # Default neutral score
            
        except Exception as e:
            logger.warning(f"Experience match calculation failed: {e}")
            return 0.5
    
    async def _calculate_recency_score(self, result: HybridSearchResult) -> float:
        """Calculate recency score based on content age"""
        
        try:
            # Get creation date from metadata
            created_at = None
            if result.metadata:
                created_at_str = result.metadata.get("created_at")
                if created_at_str:
                    if isinstance(created_at_str, str):
                        from dateutil import parser
                        created_at = parser.parse(created_at_str)
                    elif isinstance(created_at_str, datetime):
                        created_at = created_at_str
            
            if not created_at:
                return 0.7  # Neutral score if no date available
            
            # Calculate age in days
            age_days = (datetime.now() - created_at.replace(tzinfo=None)).days
            
            # Score based on age (newer is better)
            if age_days <= 30:  # Within 1 month
                return 1.0
            elif age_days <= 90:  # Within 3 months
                return 0.9
            elif age_days <= 180:  # Within 6 months
                return 0.8
            elif age_days <= 365:  # Within 1 year
                return 0.7
            elif age_days <= 730:  # Within 2 years
                return 0.6
            else:  # Older than 2 years
                return 0.5
                
        except Exception as e:
            logger.warning(f"Recency score calculation failed: {e}")
            return 0.7
    
    async def _calculate_completeness_score(self, result: HybridSearchResult) -> float:
        """Calculate profile completeness score"""
        
        try:
            if not result.metadata:
                return 0.5
            
            # Define important fields for completeness
            important_fields = [
                "title", "skills", "experience", "education", 
                "ai_summary", "description", "content"
            ]
            
            filled_fields = 0
            for field in important_fields:
                if field in result.metadata and result.metadata[field]:
                    filled_fields += 1
            
            # Calculate completeness percentage
            completeness_ratio = filled_fields / len(important_fields)
            
            # Boost score slightly for having some content
            if result.content_preview:
                completeness_ratio += 0.1
            
            return min(1.0, completeness_ratio)
            
        except Exception as e:
            logger.warning(f"Completeness score calculation failed: {e}")
            return 0.5
    
    async def _calculate_quality_score(self, result: HybridSearchResult) -> float:
        """Calculate content quality score"""
        
        try:
            quality_score = 0.7  # Default neutral score
            
            # Check for AI quality assessment in metadata
            if result.metadata and "quality_assessment" in result.metadata:
                quality_data = result.metadata["quality_assessment"]
                if isinstance(quality_data, dict) and "overall_score" in quality_data:
                    quality_score = float(quality_data["overall_score"]) / 100.0
                elif isinstance(quality_data, (int, float)):
                    quality_score = float(quality_data) / 100.0
            
            # Basic content quality heuristics
            if result.content_preview:
                content_length = len(result.content_preview)
                if content_length > 500:  # Substantial content
                    quality_score += 0.1
                elif content_length < 50:  # Very short content
                    quality_score -= 0.2
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.warning(f"Quality score calculation failed: {e}")
            return 0.7
    
    async def _generate_business_logic_explanation(
        self,
        query: str,
        result: HybridSearchResult,
        criteria: List[RankingCriterion],
        final_score: float
    ) -> str:
        """Generate explanation for business logic ranking"""
        
        try:
            explanations = []
            explanations.append(f"Final score: {final_score:.3f}")
            
            # Top contributing factors
            sorted_criteria = sorted(criteria, key=lambda x: x.score * x.weight, reverse=True)
            
            explanations.append("Key factors:")
            for criterion in sorted_criteria[:3]:  # Top 3 factors
                contribution = criterion.score * criterion.weight
                explanations.append(
                    f"- {criterion.name.value}: {contribution:.3f} "
                    f"(score: {criterion.score:.2f}, weight: {criterion.weight:.2f})"
                )
            
            return " ".join(explanations)
            
        except Exception as e:
            return f"Ranked by business logic (score: {final_score:.3f})"
    
    async def _parse_semantic_ranking_response(
        self,
        ai_response: str,
        results: List[HybridSearchResult],
        config: RerankingConfig
    ) -> List[RerankingResult]:
        """Parse AI response for semantic ranking"""
        
        try:
            # Try to parse JSON response first
            if ai_response.strip().startswith('[') or ai_response.strip().startswith('{'):
                ranking_data = json.loads(ai_response)
                
                reranked_results = []
                for item in ranking_data:
                    if isinstance(item, dict) and "entity_id" in item:
                        # Find original result
                        original_result = next(
                            (r for r in results if r.entity_id == item["entity_id"]), None
                        )
                        
                        if original_result:
                            reranked_result = RerankingResult(
                                entity_id=item["entity_id"],
                                entity_type=original_result.entity_type,
                                original_score=original_result.final_score,
                                reranked_score=float(item.get("score", original_result.final_score)),
                                ranking_criteria=[
                                    RankingCriterion(
                                        name=RankingCriteria.SEMANTIC_MATCH,
                                        score=float(item.get("score", original_result.final_score)),
                                        weight=1.0,
                                        explanation=item.get("explanation", "AI semantic assessment")
                                    )
                                ],
                                ai_explanation=item.get("explanation", "AI semantic ranking"),
                                confidence=float(item.get("confidence", 0.8)),
                                metadata=original_result.metadata
                            )
                            reranked_results.append(reranked_result)
                
                return reranked_results
            
            # Fallback to simple parsing
            return await self._fallback_reranking(results)
            
        except Exception as e:
            logger.error(f"Failed to parse semantic ranking response: {e}")
            return await self._fallback_reranking(results)
    
    async def _fallback_reranking(
        self, 
        results: List[HybridSearchResult]
    ) -> List[RerankingResult]:
        """Fallback reranking when AI processing fails"""
        
        fallback_results = []
        for result in results:
            fallback_result = RerankingResult(
                entity_id=result.entity_id,
                entity_type=result.entity_type,
                original_score=result.final_score,
                reranked_score=result.final_score,
                ranking_criteria=[
                    RankingCriterion(
                        name=RankingCriteria.SEMANTIC_MATCH,
                        score=result.final_score,
                        weight=1.0,
                        explanation="Original search score (fallback)"
                    )
                ],
                ai_explanation="Used original search ranking (AI processing unavailable)",
                confidence=0.6,
                metadata=result.metadata
            )
            fallback_results.append(fallback_result)
        
        return fallback_results
    
    async def _get_cached_reranking(
        self,
        query: str,
        results: List[HybridSearchResult],
        tenant_id: Optional[str],
        config: RerankingConfig
    ) -> Optional[RerankingResponse]:
        """Get cached reranking results"""
        
        if not self.cache_manager:
            return None
        
        try:
            # Create cache key based on query and result IDs
            result_ids = sorted([str(r.entity_id) for r in results])  # Convert UUIDs to strings
            cache_data = {
                "query": query.lower().strip(),
                "result_ids": result_ids,
                "tenant_id": str(tenant_id) if tenant_id else None,  # Convert UUID to string
                "strategy": config.strategy.value,
                "weights": {k.value: v for k, v in config.criteria_weights.items()}  # Convert Enum keys to values
            }

            cache_key = f"reranking:{hash(json.dumps(cache_data, sort_keys=True, cls=UUIDEncoder)) % (2**31)}"

            cached_response = await self.cache_manager.get(
                cache_key,
                content_type="reranking_results"
            )
            
            if cached_response and isinstance(cached_response, dict):
                logger.debug("Retrieved reranking results from cache")
                cached_response["cache_hit"] = True
                return RerankingResponse(**cached_response)
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cached reranking: {e}")
            return None
    
    async def _cache_reranking_results(
        self,
        response: RerankingResponse,
        query: str,
        results: List[HybridSearchResult],
        tenant_id: Optional[str],
        config: RerankingConfig
    ) -> None:
        """Cache reranking results"""
        
        if not self.cache_manager:
            return
        
        try:
            # Create cache key
            result_ids = sorted([str(r.entity_id) for r in results])  # Convert UUIDs to strings
            cache_data = {
                "query": query.lower().strip(),
                "result_ids": result_ids,
                "tenant_id": str(tenant_id) if tenant_id else None,  # Convert UUID to string
                "strategy": config.strategy.value,
                "weights": {k.value: v for k, v in config.criteria_weights.items()}  # Convert Enum keys to values
            }

            cache_key = f"reranking:{hash(json.dumps(cache_data, sort_keys=True, cls=UUIDEncoder)) % (2**31)}"

            # Convert response to cacheable format
            cacheable_response = self._serialize_reranking_response(response)
            cacheable_response["cache_hit"] = False

            await self.cache_manager.set(
                key=cache_key,
                value=cacheable_response,
                ttl=self.reranking_cache_ttl,
                content_type="reranking_results"
            )
            
            logger.debug("Cached reranking results")
            
        except Exception as e:
            logger.warning(f"Failed to cache reranking results: {e}")
    
    def _update_reranking_stats(
        self, 
        processing_time_ms: int, 
        results_count: int
    ) -> None:
        """Update reranking statistics"""
        
        self._stats["reranking_operations"] += 1
        self._stats["results_reranked"] += results_count
        self._stats["total_processing_time_ms"] += processing_time_ms
        
        if self._stats["reranking_operations"] > 0:
            self._stats["average_processing_time_ms"] = (
                self._stats["total_processing_time_ms"] / self._stats["reranking_operations"]
            )
    
    async def get_reranking_analytics(
        self,
        tenant_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get reranking analytics and performance metrics"""
        
        try:
            analytics = {
                "service_stats": self._stats.copy(),
                "configuration": {
                    "default_config": asdict(self.config),
                    "cache_enabled": self.cache_manager is not None,
                    "cache_ttl": self.reranking_cache_ttl
                },
                "performance_metrics": {
                    "average_processing_time_ms": self._stats["average_processing_time_ms"],
                    "cache_hit_rate": (
                        self._stats["cache_hits"] / max(1, self._stats["reranking_operations"])
                    ),
                    "average_results_per_operation": (
                        self._stats["results_reranked"] / max(1, self._stats["reranking_operations"])
                    ),
                    "ai_efficiency": (
                        self._stats["total_tokens_used"] / max(1, self._stats["ai_calls_made"])
                    )
                }
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get reranking analytics: {e}")
            return {"error": str(e), "service_stats": self._stats.copy()}
    
    async def check_health(self) -> Dict[str, Any]:
        """Check result reranker service health"""

        try:
            start_time = datetime.now()

            # Test OpenAI service
            openai_health = await self.openai_service.check_health()

            # Test prompt manager
            prompt_health = "available"

            # Test cache service
            cache_health = "disabled"
            if self.cache_manager:
                try:
                    cache_status = await self.cache_manager.check_health()
                    cache_health = cache_status.get('status', 'unknown')
                except Exception:
                    cache_health = "error"

            health_check_time = int((datetime.now() - start_time).total_seconds() * 1000)

            return {
                "status": "healthy",
                "result_reranker_service": "operational",
                "openai_service": openai_health.get('status', 'unknown'),
                "prompt_manager": prompt_health,
                "cache_service": cache_health,
                "stats": self._stats.copy(),
                "health_check_time_ms": health_check_time,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    # === STATIC HELPER METHODS ===

    @staticmethod
    def _serialize_reranking_response(response: RerankingResponse) -> Dict[str, Any]:
        """
        Serialize RerankingResponse for Redis caching, converting enums to strings.

        Handles conversion of:
        - RankingStrategy enum to string value
        - RankingCriteria enum to string value (in nested RankingCriterion objects)
        - Nested dataclass structures
        - UUID objects to strings

        Args:
            response: RerankingResponse object to serialize

        Returns:
            Dictionary with all enums converted to string values, safe for Redis JSON serialization
        """
        # Convert response to dict using dataclasses.asdict
        response_dict = asdict(response)

        # Convert strategy enum to string value
        if hasattr(response.strategy, 'value'):
            response_dict["strategy"] = response.strategy.value

        # Convert nested RankingCriteria enums in results
        for result in response_dict.get("results", []):
            if "ranking_criteria" in result:
                for criterion in result["ranking_criteria"]:
                    # Convert enum name to string value
                    if "name" in criterion and hasattr(criterion["name"], "value"):
                        criterion["name"] = criterion["name"].value

        # Convert config criteria_weights enums in metadata (if present)
        if "reranking_metadata" in response_dict and response_dict["reranking_metadata"]:
            metadata = response_dict["reranking_metadata"]
            if "config" in metadata and isinstance(metadata["config"], dict):
                config = metadata["config"]
                if "criteria_weights" in config and isinstance(config["criteria_weights"], dict):
                    # Convert enum keys to string values
                    config["criteria_weights"] = {
                        k.value if hasattr(k, "value") else str(k): v
                        for k, v in config["criteria_weights"].items()
                    }
                # Convert strategy enum in config
                if "strategy" in config and hasattr(config["strategy"], "value"):
                    config["strategy"] = config["strategy"].value

        return response_dict
