"""
Search Application Service

Advanced multi-stage search engine with:
- Multi-modal search (vector + keyword + semantic + hybrid)
- Query expansion and international terminology handling
- Multi-stage retrieval: vector search → reranking → business logic scoring
- Result diversification and quality filtering
- Performance optimization with intelligent caching
- Analytics and search quality monitoring

Migrated from app/services/core/search_engine.py to application layer.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from collections import defaultdict
import structlog
import json

from app.api.schemas.search_schemas import (
    SearchRequest, SearchResponse, SearchResult, MatchScore,
    SearchFacet, FacetValue, SearchAnalytics, SearchMode, SortOrder
)
from app.domain.interfaces import ISearchService, IAIService

# Import infrastructure types only for type checking
if TYPE_CHECKING:
    from app.infrastructure.document.embedding_generator import EmbeddingGenerator

logger = structlog.get_logger(__name__)


class SearchApplicationService:
    """
    Production-ready multi-stage search engine for CV matching.

    Application service orchestrating complex search workflows:
    - Multi-modal search combining vector, keyword, and semantic approaches
    - Intelligent query expansion with domain-specific terminology
    - Multi-stage retrieval with cross-encoder reranking
    - Business logic scoring with customizable weights
    - Result diversification and quality filtering
    - Performance monitoring and search analytics
    - Multi-tenant search isolation and configuration
    """

    def __init__(
        self,
        search_service: Optional[ISearchService] = None,
        ai_service: Optional[IAIService] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        cache_service=None
    ):
        self.search_service = search_service
        self.ai_service = ai_service
        self.embedding_generator = embedding_generator  # Must be provided via dependency injection
        self.cache_service = cache_service

        # Performance and analytics tracking
        self._search_stats = {
            "total_searches": 0,
            "vector_searches": 0,
            "hybrid_searches": 0,
            "semantic_searches": 0,
            "cached_searches": 0,
            "average_search_time_ms": 0.0,
            "average_results_returned": 0.0,
            "query_expansions_performed": 0
        }

        # Query expansion dictionaries
        self._skill_synonyms = {
            # Programming languages
            "javascript": ["js", "node.js", "nodejs", "ecmascript"],
            "python": ["py", "python3", "django", "flask", "fastapi"],
            "java": ["j2ee", "spring", "spring boot", "maven", "gradle"],
            "csharp": ["c#", ".net", "dotnet", "asp.net"],
            "typescript": ["ts", "angular", "react"],

            # Technologies
            "kubernetes": ["k8s", "container orchestration"],
            "aws": ["amazon web services", "cloud computing"],
            "azure": ["microsoft azure", "cloud platform"],
            "docker": ["containerization", "containers"],
            "react": ["reactjs", "react.js", "frontend"],

            # Roles
            "software engineer": ["developer", "programmer", "software developer"],
            "data scientist": ["ml engineer", "machine learning engineer", "data analyst"],
            "devops": ["site reliability", "platform engineer", "infrastructure"],
            "product manager": ["pm", "product owner", "product lead"],
        }

        # Industry terminology mappings
        self._industry_terms = {
            "fintech": ["financial technology", "banking", "payments", "trading"],
            "healthtech": ["healthcare", "medical technology", "digital health"],
            "edtech": ["education technology", "learning platforms", "e-learning"],
            "blockchain": ["crypto", "cryptocurrency", "web3", "defi"],
        }

        # Search configuration
        self._search_config = {
            "default_cache_ttl": 300,  # 5 minutes
            "max_query_expansion_terms": 10,
            "reranking_enabled": True,
            "diversification_enabled": True,
            "quality_filtering_enabled": True,
            "min_match_score": 0.3,
            "max_results_for_reranking": 100
        }

    async def search(
        self,
        request: SearchRequest,
        tenant_config: Optional[Dict[str, Any]] = None
    ) -> SearchResponse:
        """
        Execute comprehensive search with multi-stage retrieval.

        Args:
            request: Search request with query and filters
            tenant_config: Tenant-specific search configuration

        Returns:
            Complete search response with results, facets, and analytics
        """
        search_start_time = datetime.now()
        search_id = self._generate_search_id()

        try:
            logger.info(
                "Starting search execution",
                search_id=search_id,
                query=request.query[:100],
                search_mode=request.search_mode,
                tenant_id=str(request.tenant_id)
            )

            # Check cache first
            if request.track_search:
                cached_response = await self._get_cached_search_results(request)
                if cached_response:
                    self._search_stats["cached_searches"] += 1
                    logger.info("Returning cached search results", search_id=search_id)
                    return cached_response

            # Initialize analytics tracking
            analytics = SearchAnalytics(
                total_search_time_ms=0,
                total_candidates=0,
                candidates_after_filters=0,
                query_expanded=False,
                expanded_terms=[],
                synonyms_used=[]
            )

            # Stage 1: Query preprocessing and expansion
            processed_query = await self._preprocess_query(request, analytics)

            # Stage 2: Execute primary search
            primary_results = await self._execute_primary_search(
                request, processed_query, analytics
            )

            # Stage 3: Apply business logic filtering
            filtered_results = await self._apply_business_logic_filtering(
                primary_results, request, analytics
            )

            # Stage 4: Rerank results if enabled
            if (self._search_config["reranking_enabled"] and
                len(filtered_results) > 1 and
                request.search_mode in [SearchMode.HYBRID, SearchMode.MULTI_STAGE]):
                reranked_results = await self._rerank_results(
                    filtered_results, request, analytics
                )
            else:
                reranked_results = filtered_results

            # Stage 5: Apply diversification and final scoring
            final_results = await self._finalize_results(
                reranked_results, request, analytics
            )

            # Stage 6: Generate facets
            facets = await self._generate_search_facets(primary_results, request)

            # Complete analytics
            search_duration = (datetime.now() - search_start_time).total_seconds() * 1000
            analytics.total_search_time_ms = int(search_duration)

            # Build response
            response = SearchResponse(
                results=final_results[:request.max_results],
                total_count=len(final_results),
                page=request.pagination.page,
                page_size=request.pagination.page_size,
                total_pages=(len(final_results) + request.pagination.page_size - 1) // request.pagination.page_size,
                has_next_page=len(final_results) > ((request.pagination.page - 1) * request.pagination.page_size + request.pagination.page_size),
                has_prev_page=request.pagination.page > 1,
                search_id=search_id,
                search_mode=request.search_mode,
                query=request.query,
                facets=facets,
                analytics=analytics
            )

            # Cache results if tracking enabled
            if request.track_search:
                await self._cache_search_results(request, response)

            # Update statistics
            self._update_search_stats(search_duration, request, len(final_results))

            logger.info(
                "Search completed successfully",
                search_id=search_id,
                results_count=len(final_results),
                duration_ms=search_duration,
                search_mode=request.search_mode
            )

            return response

        except Exception as e:
            search_duration = (datetime.now() - search_start_time).total_seconds() * 1000

            logger.error(
                "Search execution failed",
                search_id=search_id,
                error=str(e),
                duration_ms=search_duration
            )

            raise RuntimeError(f"Search failed: {e}")

    async def _preprocess_query(
        self,
        request: SearchRequest,
        analytics: SearchAnalytics
    ) -> str:
        """Preprocess and expand query for better matching"""

        processed_query = request.query.strip()
        expanded_terms = []
        synonyms_used = []

        if request.enable_query_expansion:
            # Skill-based expansion
            query_lower = processed_query.lower()
            for skill, synonyms in self._skill_synonyms.items():
                if skill in query_lower:
                    for synonym in synonyms[:3]:  # Limit synonyms per skill
                        if synonym not in query_lower:
                            expanded_terms.append(synonym)
                            synonyms_used.append(f"{skill} -> {synonym}")

            # Industry terminology expansion
            for industry, terms in self._industry_terms.items():
                if industry in query_lower:
                    for term in terms[:2]:
                        if term not in query_lower:
                            expanded_terms.append(term)
                            synonyms_used.append(f"{industry} -> {term}")

            # Add expanded terms to query
            if expanded_terms:
                # Limit expansion terms
                expanded_terms = expanded_terms[:self._search_config["max_query_expansion_terms"]]
                processed_query += " " + " ".join(expanded_terms)

                analytics.query_expanded = True
                analytics.expanded_terms = expanded_terms
                analytics.synonyms_used = synonyms_used

                self._search_stats["query_expansions_performed"] += 1

        return processed_query

    async def _execute_primary_search(
        self,
        request: SearchRequest,
        processed_query: str,
        analytics: SearchAnalytics
    ) -> List[Dict[str, Any]]:
        """Execute primary search using specified search mode"""

        start_time = datetime.now()

        # Determine index name (could be tenant-specific)
        index_name = self._get_index_name(request.tenant_id)

        if request.search_mode == SearchMode.VECTOR:
            # Pure vector search
            query_embedding = await self.embedding_generator.generate_query_embedding(
                processed_query,
                enhance_for_cv_search=True
            )

            search_result = await self.search_service.vector_search(
                index_name=index_name,
                vector=query_embedding,
                top=min(request.max_results * 2, 200),  # Get more for filtering
                filters=self._build_search_filters(request)
            )

            primary_results = search_result["documents"]
            self._search_stats["vector_searches"] += 1

        elif request.search_mode == SearchMode.SEMANTIC:
            # Semantic search with configured search service
            search_result = await self.search_service.search(
                index_name=index_name,
                query=processed_query,
                filters=self._build_search_filters(request),
                top=min(request.max_results * 2, 200),
                search_mode="semantic"
            )

            primary_results = search_result["documents"]
            self._search_stats["semantic_searches"] += 1

        elif request.search_mode in [SearchMode.HYBRID, SearchMode.MULTI_STAGE]:
            # Hybrid search combining vector and keyword
            search_result = await self.search_service.search(
                index_name=index_name,
                query=processed_query,
                filters=self._build_search_filters(request),
                top=min(request.max_results * 2, 200),
                search_mode="hybrid"
            )

            primary_results = search_result["documents"]
            self._search_stats["hybrid_searches"] += 1

        else:  # KEYWORD
            # Traditional keyword search
            search_result = await self.search_service.search(
                index_name=index_name,
                query=processed_query,
                filters=self._build_search_filters(request),
                top=min(request.max_results * 2, 200),
                search_mode="keyword"
            )

            primary_results = search_result["documents"]

        search_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Update analytics
        analytics.total_candidates = len(primary_results)
        if request.search_mode == SearchMode.VECTOR:
            analytics.vector_search_time_ms = int(search_time_ms)
        else:
            analytics.keyword_search_time_ms = int(search_time_ms)

        return primary_results

    async def _apply_business_logic_filtering(
        self,
        results: List[Dict[str, Any]],
        request: SearchRequest,
        analytics: SearchAnalytics
    ) -> List[Dict[str, Any]]:
        """Apply business logic filtering and scoring"""

        filtered_results = []

        for result in results:
            # Calculate comprehensive match score
            match_score = await self._calculate_match_score(result, request)

            # Apply minimum score threshold
            if match_score.overall_score >= request.min_match_score:
                # Add match score to result
                result["match_score"] = match_score.model_dump()
                result["overall_match_score"] = match_score.overall_score
                filtered_results.append(result)

        # Sort by match score initially
        filtered_results.sort(key=lambda x: x["overall_match_score"], reverse=True)

        analytics.candidates_after_filters = len(filtered_results)

        return filtered_results

    async def _calculate_match_score(
        self,
        result: Dict[str, Any],
        request: SearchRequest
    ) -> MatchScore:
        """Calculate comprehensive match score for a search result"""

        # Base search relevance score
        relevance_score = result.get("@search_score", 0.0)
        if relevance_score > 1.0:
            relevance_score = relevance_score / 100.0  # Normalize if needed

        # Initialize match score
        match_score = MatchScore(
            overall_score=relevance_score,
            relevance_score=relevance_score
        )

        # Skills matching
        if request.skill_requirements:
            skill_match = self._calculate_skill_match_score(result, request.skill_requirements)
            match_score.skill_match_score = skill_match["score"]
            match_score.matched_skills = skill_match["matched_skills"]
            match_score.missing_skills = skill_match["missing_skills"]
            match_score.skill_gaps = skill_match["gaps"]

        # Experience matching
        if request.experience_requirements:
            experience_match = self._calculate_experience_match_score(
                result, request.experience_requirements
            )
            match_score.experience_match_score = experience_match

        # Education matching
        if request.education_requirements:
            education_match = self._calculate_education_match_score(
                result, request.education_requirements
            )
            match_score.education_match_score = education_match

        # Location matching
        if request.location_filter:
            location_match = self._calculate_location_match_score(
                result, request.location_filter
            )
            match_score.location_match_score = location_match

        # Salary compatibility
        if request.salary_filter:
            salary_match = self._calculate_salary_match_score(
                result, request.salary_filter
            )
            match_score.salary_match_score = salary_match

        # Calculate weighted overall score
        weights = request.custom_scoring or {
            "relevance": 0.3,
            "skills": 0.4,
            "experience": 0.2,
            "education": 0.05,
            "location": 0.03,
            "salary": 0.02
        }

        weighted_score = (
            weights.get("relevance", 0.3) * match_score.relevance_score +
            weights.get("skills", 0.4) * match_score.skill_match_score +
            weights.get("experience", 0.2) * match_score.experience_match_score +
            weights.get("education", 0.05) * match_score.education_match_score +
            weights.get("location", 0.03) * match_score.location_match_score +
            weights.get("salary", 0.02) * match_score.salary_match_score
        )

        match_score.overall_score = min(1.0, weighted_score)

        # Add explanation
        match_score.match_explanation = self._generate_match_explanation(
            match_score, request
        )

        return match_score

    def _calculate_skill_match_score(
        self,
        result: Dict[str, Any],
        skill_requirements: List
    ) -> Dict[str, Any]:
        """Calculate skill matching score and details"""

        profile_skills = self._extract_profile_skills(result)
        profile_skills_lower = [skill.lower() for skill in profile_skills]

        matched_skills = []
        missing_skills = []
        skill_gaps = {}
        total_weight = 0.0
        matched_weight = 0.0

        for req in skill_requirements:
            skill_name = req.name.lower()
            weight = req.weight
            total_weight += weight

            # Check for exact match or alternatives
            found = False
            for profile_skill in profile_skills_lower:
                if (skill_name in profile_skill or
                    profile_skill in skill_name or
                    any(alt.lower() in profile_skill for alt in req.alternatives)):
                    matched_skills.append(req.name)
                    matched_weight += weight
                    found = True
                    break

            if not found:
                missing_skills.append(req.name)
                if req.required:
                    skill_gaps[req.name] = "Required skill missing"

        # Calculate score
        if total_weight > 0:
            skill_score = matched_weight / total_weight
        else:
            skill_score = 1.0 if not skill_requirements else 0.0

        return {
            "score": skill_score,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "gaps": skill_gaps
        }

    def _calculate_experience_match_score(
        self,
        result: Dict[str, Any],
        experience_req
    ) -> float:
        """Calculate experience matching score"""

        total_experience = result.get("total_experience_years", 0)
        score = 1.0

        # Check minimum experience requirement
        if experience_req.min_total_years:
            if total_experience >= experience_req.min_total_years:
                # Bonus for having more experience than minimum
                excess = total_experience - experience_req.min_total_years
                score = min(1.0, 0.8 + (excess * 0.02))  # 2% bonus per extra year
            else:
                # Penalty for not meeting minimum
                shortfall = experience_req.min_total_years - total_experience
                score = max(0.0, 0.8 - (shortfall * 0.1))  # 10% penalty per missing year

        # Check maximum experience (if candidate is overqualified)
        if experience_req.max_total_years and total_experience > experience_req.max_total_years:
            over_qualified = total_experience - experience_req.max_total_years
            if over_qualified > 5:  # Significantly overqualified
                score *= 0.9  # Small penalty for being overqualified

        return score

    def _calculate_education_match_score(
        self,
        result: Dict[str, Any],
        education_req
    ) -> float:
        """Calculate education matching score"""

        # This is a simplified implementation
        # In production, you'd want more sophisticated education matching
        highest_degree = result.get("highest_degree", "").lower()

        if not education_req.required_degree_levels:
            return 1.0

        required_levels = [level.lower() for level in education_req.required_degree_levels]

        # Simple degree level matching
        degree_hierarchy = {
            "phd": 6, "doctorate": 6, "doctoral": 6,
            "master": 5, "masters": 5, "mba": 5,
            "bachelor": 4, "bachelors": 4,
            "associate": 3, "associates": 3,
            "certificate": 2, "diploma": 1
        }

        candidate_level = 0
        for degree_type, level in degree_hierarchy.items():
            if degree_type in highest_degree:
                candidate_level = level
                break

        required_level = max([degree_hierarchy.get(req_level, 0) for req_level in required_levels])

        if candidate_level >= required_level:
            return 1.0
        elif candidate_level > 0:
            return candidate_level / required_level
        else:
            return 0.5  # Some education, but can't determine level

    def _calculate_location_match_score(self, result: Dict[str, Any], location_filter) -> float:
        """Calculate location matching score"""

        current_location = result.get("current_location", "").lower()

        if location_filter.include_remote:
            # Check if candidate mentions remote work capability
            remote_indicators = ["remote", "distributed", "anywhere", "global"]
            if any(indicator in current_location for indicator in remote_indicators):
                return 1.0

        # Check preferred locations
        for preferred in location_filter.preferred_locations:
            if preferred.lower() in current_location:
                return 1.0

        # For radius-based matching, this would require more sophisticated geocoding
        # For now, return a moderate score for any location match
        if current_location:
            return 0.7

        return 0.3  # Default for no location information

    def _calculate_salary_match_score(self, result: Dict[str, Any], salary_filter) -> float:
        """Calculate salary compatibility score"""

        # This would require salary expectation data in the profile
        # For now, return a default score
        return 0.8

    async def _rerank_results(
        self,
        results: List[Dict[str, Any]],
        request: SearchRequest,
        analytics: SearchAnalytics
    ) -> List[Dict[str, Any]]:
        """Rerank top results using cross-encoder or advanced scoring"""

        if len(results) <= 1:
            return results

        start_time = datetime.now()

        # Take top results for reranking
        top_results = results[:self._search_config["max_results_for_reranking"]]

        # For now, implement a simple reranking based on comprehensive scoring
        # In production, you might use a cross-encoder model here

        reranked_results = []

        for result in top_results:
            # Enhanced scoring for reranking
            enhanced_score = await self._calculate_enhanced_rerank_score(result, request)
            result["rerank_score"] = enhanced_score
            reranked_results.append(result)

        # Sort by rerank score
        reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Add remaining results without reranking
        reranked_results.extend(results[len(top_results):])

        rerank_time = (datetime.now() - start_time).total_seconds() * 1000
        analytics.reranking_time_ms = int(rerank_time)
        analytics.candidates_reranked = len(top_results)

        return reranked_results

    async def _calculate_enhanced_rerank_score(
        self,
        result: Dict[str, Any],
        request: SearchRequest
    ) -> float:
        """Calculate enhanced score for reranking"""

        base_score = result.get("overall_match_score", 0.0)

        # Boost factors
        boosts = []

        # Boost recent activity
        if request.boost_recent_activity:
            last_updated = result.get("last_updated")
            if last_updated:
                # Boost profiles updated in last 30 days
                days_ago = (datetime.now() - datetime.fromisoformat(last_updated.replace('Z', '+00:00'))).days
                if days_ago <= 30:
                    activity_boost = max(0, (30 - days_ago) / 30 * 0.1)
                    boosts.append(activity_boost)

        # Boost profile completeness
        completeness = result.get("profile_completeness", 0.0)
        if completeness > 0.8:
            boosts.append(0.05)

        # Apply boosts
        final_score = base_score + sum(boosts)
        return min(1.0, final_score)

    async def _finalize_results(
        self,
        results: List[Dict[str, Any]],
        request: SearchRequest,
        analytics: SearchAnalytics
    ) -> List[SearchResult]:
        """Apply final processing and convert to SearchResult objects"""

        final_results = []

        # Apply diversification if enabled
        if self._search_config["diversification_enabled"] and len(results) > 10:
            results = self._apply_result_diversification(results)

        # Convert to SearchResult objects
        for result in results:
            try:
                search_result = self._convert_to_search_result(result, request)
                final_results.append(search_result)
            except Exception as e:
                logger.warning(f"Failed to convert search result: {e}")
                continue

        return final_results

    def _apply_result_diversification(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply diversification to avoid too similar results"""

        if len(results) <= 10:
            return results

        diversified = []
        company_counts = defaultdict(int)

        for result in results:
            company = result.get("current_company", "Unknown")

            # Limit results from same company
            if company_counts[company] < 2:  # Max 2 per company in top results
                diversified.append(result)
                company_counts[company] += 1
            elif len(diversified) < len(results) * 0.7:  # Until we have 70% of original
                diversified.append(result)

        # Add remaining results
        remaining = [r for r in results if r not in diversified]
        diversified.extend(remaining)

        return diversified

    def _convert_to_search_result(
        self,
        result: Dict[str, Any],
        request: SearchRequest
    ) -> SearchResult:
        """Convert search engine result to SearchResult object"""

        # Extract match score
        match_score_data = result.get("match_score", {})
        match_score = MatchScore(**match_score_data) if match_score_data else MatchScore(
            overall_score=result.get("overall_match_score", 0.0),
            relevance_score=result.get("@search_score", 0.0)
        )

        # Build search result
        search_result = SearchResult(
            profile_id=result.get("profile_id", result.get("id", "")),
            email=result.get("email", ""),
            tenant_id=result.get("tenant_id", request.tenant_id),
            full_name=result.get("full_name", ""),
            title=result.get("title", ""),
            summary=result.get("summary", ""),
            current_company=result.get("current_company", ""),
            current_location=result.get("current_location", ""),
            total_experience_years=result.get("total_experience_years", 0),
            top_skills=result.get("top_skills", result.get("skills", [])[:10]),
            highest_degree=result.get("highest_degree", ""),
            match_score=match_score,
            search_highlights=result.get("@search.highlights", {}),
            last_updated=datetime.fromisoformat(result.get("last_updated", datetime.now().isoformat()).replace('Z', '+00:00')),
            profile_completeness=result.get("profile_completeness", 0.0),
            availability_status=result.get("availability_status", "")
        )

        return search_result

    def _generate_match_explanation(
        self,
        match_score: MatchScore,
        request: SearchRequest
    ) -> List[str]:
        """Generate human-readable match explanations"""

        explanations = []

        # Skills explanation
        if match_score.matched_skills:
            explanations.append(f"Matches {len(match_score.matched_skills)} required skills: {', '.join(match_score.matched_skills[:3])}")

        if match_score.missing_skills:
            explanations.append(f"Missing {len(match_score.missing_skills)} skills: {', '.join(match_score.missing_skills[:3])}")

        # Experience explanation
        if match_score.experience_match_score > 0.8:
            explanations.append("Strong experience match for role requirements")
        elif match_score.experience_match_score < 0.5:
            explanations.append("Limited experience for role requirements")

        # Overall score explanation
        if match_score.overall_score > 0.8:
            explanations.append("Excellent overall match for position")
        elif match_score.overall_score > 0.6:
            explanations.append("Good match with some gaps")
        elif match_score.overall_score > 0.4:
            explanations.append("Partial match - may require training")
        else:
            explanations.append("Limited match for role requirements")

        return explanations

    async def _generate_search_facets(
        self,
        results: List[Dict[str, Any]],
        request: SearchRequest
    ) -> List[SearchFacet]:
        """Generate search facets for result refinement"""

        facets = []

        if not results:
            return facets

        # Skills facet
        skill_counts = defaultdict(int)
        for result in results:
            skills = result.get("skills", [])
            if isinstance(skills, str):
                skills = [s.strip() for s in skills.split(",")]
            for skill in skills[:10]:  # Top skills only
                if skill:
                    skill_counts[skill] += 1

        if skill_counts:
            skill_facet_values = [
                FacetValue(value=skill, count=count)
                for skill, count in sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            ]
            facets.append(SearchFacet(
                field="skills",
                display_name="Skills",
                values=skill_facet_values,
                facet_type="terms"
            ))

        # Experience years facet
        exp_ranges = {"0-2 years": 0, "3-5 years": 0, "6-10 years": 0, "10+ years": 0}
        for result in results:
            exp_years = result.get("total_experience_years", 0)
            if exp_years <= 2:
                exp_ranges["0-2 years"] += 1
            elif exp_years <= 5:
                exp_ranges["3-5 years"] += 1
            elif exp_years <= 10:
                exp_ranges["6-10 years"] += 1
            else:
                exp_ranges["10+ years"] += 1

        exp_facet_values = [
            FacetValue(value=range_name, count=count)
            for range_name, count in exp_ranges.items() if count > 0
        ]

        if exp_facet_values:
            facets.append(SearchFacet(
                field="total_experience_years",
                display_name="Experience Level",
                values=exp_facet_values,
                facet_type="range"
            ))

        return facets

    def _build_search_filters(self, request: SearchRequest) -> Dict[str, Any]:
        """Build search service filters from search request"""

        filters = {}

        # Add tenant isolation
        filters["tenant_id"] = str(request.tenant_id)

        # Add basic filters
        for filter_item in request.basic_filters:
            filters[filter_item.field] = filter_item.value

        # Add range filters
        for range_filter in request.range_filters:
            range_dict = {}
            if range_filter.min_value is not None:
                range_dict["min"] = range_filter.min_value
            if range_filter.max_value is not None:
                range_dict["max"] = range_filter.max_value

            if range_dict:
                filters[range_filter.field] = range_dict

        # Add location filters
        if request.location_filter:
            if request.location_filter.preferred_locations:
                filters["current_location"] = request.location_filter.preferred_locations

        return filters

    def _get_index_name(self, tenant_id) -> str:
        """Get appropriate search index name for tenant"""
        # This could be tenant-specific in production
        return "cv-profiles"

    def _generate_search_id(self) -> str:
        """Generate unique search identifier"""
        import uuid
        return str(uuid.uuid4())[:8]

    async def _get_cached_search_results(
        self,
        request: SearchRequest
    ) -> Optional[SearchResponse]:
        """Get cached search results if available"""

        if not self.cache_service:
            return None

        try:
            # Generate cache key from request
            import hashlib
            request_hash = hashlib.sha256(
                json.dumps(request.dict(), sort_keys=True, default=str).encode()
            ).hexdigest()[:16]

            cache_key = f"search_results:{request_hash}"
            cached_result = await self.cache_service.get(cache_key)

            if cached_result:
                return SearchResponse(**cached_result)

        except Exception as e:
            logger.warning(f"Failed to get cached search results: {e}")

        return None

    async def _cache_search_results(
        self,
        request: SearchRequest,
        response: SearchResponse
    ) -> None:
        """Cache search results for future use"""

        if not self.cache_service:
            return

        try:
            import hashlib
            request_hash = hashlib.sha256(
                json.dumps(request.dict(), sort_keys=True, default=str).encode()
            ).hexdigest()[:16]

            cache_key = f"search_results:{request_hash}"
            await self.cache_service.set(
                cache_key,
                response.dict(),
                ttl=self._search_config["default_cache_ttl"]
            )

        except Exception as e:
            logger.warning(f"Failed to cache search results: {e}")

    def _update_search_stats(
        self,
        duration_ms: float,
        request: SearchRequest,
        result_count: int
    ) -> None:
        """Update search performance statistics"""

        self._search_stats["total_searches"] += 1

        # Update average search time
        current_avg = self._search_stats["average_search_time_ms"]
        count = self._search_stats["total_searches"]

        self._search_stats["average_search_time_ms"] = (
            (current_avg * (count - 1) + duration_ms) / count
        )

        # Update average results count
        current_avg_results = self._search_stats["average_results_returned"]
        self._search_stats["average_results_returned"] = (
            (current_avg_results * (count - 1) + result_count) / count
        )

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search engine performance statistics"""

        return {
            **self._search_stats,
            "search_config": self._search_config,
            "skill_synonyms_count": len(self._skill_synonyms),
            "industry_terms_count": len(self._industry_terms)
        }

    @staticmethod
    def _extract_profile_skills(result: Dict[str, Any]) -> List[str]:
        """Collect skills from various result sources."""

        potential_sources = []
        potential_sources.append(result.get("skills"))
        potential_sources.append(result.get("top_skills"))

        metadata = result.get("metadata")
        if isinstance(metadata, dict):
            potential_sources.append(metadata.get("skills"))
            potential_sources.append(metadata.get("top_skills"))
            potential_sources.append(metadata.get("matched_skills"))

        collected: List[str] = []

        for source in potential_sources:
            if not source:
                continue

            if isinstance(source, str):
                collected.extend(
                    [item.strip() for item in source.split(",") if item and item.strip()]
                )
            elif isinstance(source, list):
                for item in source:
                    if isinstance(item, str):
                        cleaned = item.strip()
                        if cleaned:
                            collected.append(cleaned)
                    elif isinstance(item, dict) and item.get("name"):
                        cleaned = str(item["name"]).strip()
                        if cleaned:
                            collected.append(cleaned)

        # Preserve order while removing duplicates
        unique_skills = []
        seen = set()
        for skill in collected:
            key = skill.lower()
            if key not in seen:
                seen.add(key)
                unique_skills.append(skill)

        return unique_skills
