"""
Query Processor with AI-Powered Query Expansion

Advanced query understanding and expansion for enhanced search:
- Natural language query analysis
- AI-powered query expansion with synonyms and related terms
- Intent detection and context understanding
- Query optimization and restructuring
- Caching for performance optimization
"""

import re
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import structlog

from app.core.config import get_settings
from app.services.ai.openai_service import OpenAIService
from app.services.ai.prompt_manager import PromptManager, PromptType
from app.services.ai.cache_manager import CacheManager
from app.services.adapters.postgres_adapter import PostgresAdapter

logger = structlog.get_logger(__name__)


@dataclass
class QueryExpansion:
    """Represents an expanded query with additional terms and metadata"""
    original_query: str
    expanded_terms: List[str]
    primary_skills: List[str]
    job_roles: List[str]
    experience_level: Optional[str]
    industry: Optional[str]
    confidence: float
    intent: str
    metadata: Dict[str, Any]


@dataclass
class ProcessedQuery:
    """Complete processed query with all enhancements"""
    original_query: str
    normalized_query: str
    expansion: QueryExpansion
    filters: Dict[str, Any]
    search_strategy: str
    processing_metadata: Dict[str, Any]


class QueryProcessor:
    """
    Advanced query processor with AI-powered expansion and understanding.
    
    Features:
    - Natural language query analysis and normalization
    - AI-powered query expansion with industry-specific terms
    - Intent detection and context understanding
    - Smart filtering and search strategy selection
    - Intelligent caching for performance optimization
    - Query optimization and restructuring
    """
    
    def __init__(
        self,
        openai_service: OpenAIService,
        prompt_manager: PromptManager,
        cache_manager: CacheManager,
        db_adapter: PostgresAdapter
    ):
        self.settings = get_settings()
        self.openai_service = openai_service
        self.prompt_manager = prompt_manager
        self.cache_manager = cache_manager
        self.db_adapter = db_adapter
        
        # Processing statistics
        self._stats = {
            "queries_processed": 0,
            "expansions_generated": 0,
            "cache_hits": 0,
            "ai_calls": 0,
            "processing_errors": 0
        }
    
    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
        expand_query: bool = True,
        cache_results: bool = True
    ) -> ProcessedQuery:
        """
        Process and enhance a search query with AI-powered expansion.
        
        Args:
            query: Original search query
            context: Additional context for query processing
            tenant_id: Tenant identifier for multi-tenancy
            user_preferences: User preferences for personalization
            expand_query: Enable AI-powered query expansion
            cache_results: Cache expansion results for performance
            
        Returns:
            ProcessedQuery with expansion and enhancements
        """
        start_time = datetime.now()
        
        try:
            # Normalize the query
            normalized_query = self._normalize_query(query)
            
            # Generate expansion if enabled
            expansion = None
            if expand_query and normalized_query:
                expansion = await self._expand_query(
                    normalized_query,
                    context=context,
                    tenant_id=tenant_id,
                    cache_results=cache_results
                )
            else:
                # Create basic expansion for consistency
                expansion = QueryExpansion(
                    original_query=query,
                    expanded_terms=[],
                    primary_skills=[],
                    job_roles=[],
                    experience_level=None,
                    industry=None,
                    confidence=0.5,
                    intent="basic_search",
                    metadata={}
                )
            
            # Extract filters from query
            filters = self._extract_filters(normalized_query, expansion)
            
            # Determine search strategy
            search_strategy = self._determine_search_strategy(expansion, filters, context)
            
            # Create processing metadata
            processing_metadata = {
                "processing_started": start_time.isoformat(),
                "processing_completed": datetime.now().isoformat(),
                "processing_duration": (datetime.now() - start_time).total_seconds(),
                "query_length": len(query),
                "normalized_length": len(normalized_query),
                "expansion_enabled": expand_query,
                "cache_enabled": cache_results,
                "tenant_id": tenant_id
            }
            
            # Create processed query
            processed_query = ProcessedQuery(
                original_query=query,
                normalized_query=normalized_query,
                expansion=expansion,
                filters=filters,
                search_strategy=search_strategy,
                processing_metadata=processing_metadata
            )
            
            # Update statistics
            self._stats["queries_processed"] += 1
            if expansion and expansion.expanded_terms:
                self._stats["expansions_generated"] += 1
            
            logger.info(
                "Query processed successfully",
                original_query=query[:50] + "..." if len(query) > 50 else query,
                expanded_terms_count=len(expansion.expanded_terms) if expansion else 0,
                search_strategy=search_strategy,
                processing_time=processing_metadata["processing_duration"]
            )
            
            return processed_query
            
        except Exception as e:
            self._stats["processing_errors"] += 1
            logger.error(f"Query processing failed: {e}")
            
            # Return basic processed query on error
            return ProcessedQuery(
                original_query=query,
                normalized_query=self._normalize_query(query),
                expansion=QueryExpansion(
                    original_query=query,
                    expanded_terms=[],
                    primary_skills=[],
                    job_roles=[],
                    experience_level=None,
                    industry=None,
                    confidence=0.0,
                    intent="error",
                    metadata={"error": str(e)}
                ),
                filters={},
                search_strategy="basic",
                processing_metadata={"error": str(e)}
            )
    
    async def _expand_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
        cache_results: bool = True
    ) -> QueryExpansion:
        """Expand query using AI-powered analysis"""
        
        # Check cache first if enabled
        if cache_results:
            cached_expansion = await self._get_cached_expansion(query, tenant_id)
            if cached_expansion:
                self._stats["cache_hits"] += 1
                logger.debug("Retrieved query expansion from cache")
                return cached_expansion
        
        try:
            # Generate query expansion prompt
            expansion_context = {"query": query}
            if context:
                expansion_context.update(context)
            
            prompt = await self.prompt_manager.create_query_expansion_prompt(
                query=query,
                additional_context=context
            )
            
            # Get AI expansion
            response = await self.openai_service.chat_completion(
                messages=prompt["messages"],
                temperature=prompt["temperature"],
                max_tokens=prompt["max_tokens"]
            )
            
            self._stats["ai_calls"] += 1
            
            # Parse AI response
            expansion_data = self._parse_expansion_response(
                response["choices"][0]["message"]["content"],
                query
            )
            
            # Create expansion object
            expansion = QueryExpansion(
                original_query=query,
                expanded_terms=expansion_data.get("expanded_terms", []),
                primary_skills=expansion_data.get("primary_skills", []),
                job_roles=expansion_data.get("job_roles", []),
                experience_level=expansion_data.get("experience_level"),
                industry=expansion_data.get("industry"),
                confidence=expansion_data.get("confidence", 0.8),
                intent=expansion_data.get("intent", "job_search"),
                metadata={
                    "ai_model": response.get("model", "unknown"),
                    "tokens_used": response.get("usage", {}).get("total_tokens", 0),
                    "processing_time": datetime.now().isoformat()
                }
            )
            
            # Cache the expansion if enabled
            if cache_results:
                await self._cache_expansion(expansion, tenant_id)
            
            return expansion
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return QueryExpansion(
                original_query=query,
                expanded_terms=[],
                primary_skills=[],
                job_roles=[],
                experience_level=None,
                industry=None,
                confidence=0.0,
                intent="error",
                metadata={"error": str(e)}
            )
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query text for better processing"""
        if not query:
            return ""
        
        # Convert to lowercase
        normalized = query.lower()
        
        # Remove excessive whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove special characters but preserve important ones
        normalized = re.sub(r'[^\w\s\-+.#]', ' ', normalized)
        
        # Handle common abbreviations and expansions
        abbreviation_map = {
            'js': 'javascript',
            'py': 'python',
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
            'ui': 'user interface',
            'ux': 'user experience',
            'api': 'application programming interface',
            'db': 'database',
            'devops': 'development operations',
            'qa': 'quality assurance',
            'hr': 'human resources'
        }
        
        words = normalized.split()
        for i, word in enumerate(words):
            if word in abbreviation_map:
                words[i] = abbreviation_map[word]
        
        normalized = ' '.join(words)
        
        # Remove duplicates while preserving order
        words = []
        seen = set()
        for word in normalized.split():
            if word not in seen:
                words.append(word)
                seen.add(word)
        
        return ' '.join(words).strip()
    
    def _parse_expansion_response(self, response_text: str, original_query: str) -> Dict[str, Any]:
        """Parse AI expansion response into structured data"""
        try:
            # Try to parse as JSON first
            if response_text.strip().startswith('{'):
                data = json.loads(response_text)
                return data
        except json.JSONDecodeError:
            pass
        
        # Fallback to text parsing
        expansion_data = {
            "expanded_terms": [],
            "primary_skills": [],
            "job_roles": [],
            "experience_level": None,
            "industry": None,
            "confidence": 0.6,
            "intent": "job_search"
        }
        
        # Extract terms using simple patterns
        lines = response_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect sections
            if 'expanded' in line.lower() and 'terms' in line.lower():
                current_section = 'expanded_terms'
                continue
            elif 'skills' in line.lower():
                current_section = 'primary_skills'
                continue
            elif 'roles' in line.lower() or 'titles' in line.lower():
                current_section = 'job_roles'
                continue
            elif 'experience' in line.lower():
                current_section = 'experience_level'
                continue
            elif 'industry' in line.lower():
                current_section = 'industry'
                continue
            
            # Extract items based on current section
            if current_section in ['expanded_terms', 'primary_skills', 'job_roles']:
                # Extract list items
                items = re.findall(r'[-*•]\s*(.+)', line)
                if items:
                    expansion_data[current_section].extend([item.strip() for item in items])
                elif line.startswith('-') or line.startswith('*'):
                    expansion_data[current_section].append(line[1:].strip())
            elif current_section == 'experience_level' and line:
                levels = ['junior', 'mid', 'senior', 'lead', 'principal', 'entry']
                for level in levels:
                    if level in line.lower():
                        expansion_data['experience_level'] = level
                        break
            elif current_section == 'industry' and line:
                expansion_data['industry'] = line.split(':')[-1].strip()
        
        # If no structured data found, extract keywords from the response
        if not any(expansion_data[key] for key in ['expanded_terms', 'primary_skills', 'job_roles']):
            # Simple keyword extraction
            words = re.findall(r'\b[a-zA-Z]{3,}\b', response_text.lower())
            unique_words = list(set(words))[:10]  # Limit to 10 terms
            expansion_data['expanded_terms'] = unique_words
        
        return expansion_data
    
    def _extract_filters(self, query: str, expansion: QueryExpansion) -> Dict[str, Any]:
        """Extract search filters from query and expansion"""
        filters = {}
        
        # Experience level filters
        experience_keywords = {
            'junior': ['junior', 'entry', 'graduate', 'trainee'],
            'mid': ['mid', 'intermediate', 'experienced'],
            'senior': ['senior', 'lead', 'principal', 'staff'],
            'executive': ['director', 'manager', 'head', 'chief', 'vp']
        }
        
        for level, keywords in experience_keywords.items():
            if any(keyword in query.lower() for keyword in keywords):
                filters['experience_level'] = level
                break
        
        # If not found in query, use expansion data
        if 'experience_level' not in filters and expansion.experience_level:
            filters['experience_level'] = expansion.experience_level
        
        # Industry filters
        if expansion.industry:
            filters['industry'] = expansion.industry
        
        # Skills filters (use primary skills from expansion)
        if expansion.primary_skills:
            filters['required_skills'] = expansion.primary_skills[:5]  # Limit to top 5
        
        # Remote/location filters
        location_keywords = ['remote', 'onsite', 'hybrid', 'work from home']
        for keyword in location_keywords:
            if keyword in query.lower():
                filters['work_arrangement'] = keyword
                break
        
        # Salary filters (basic detection)
        salary_match = re.search(r'\$?(\d{1,3}[,.]?\d{0,3})[k]?\s*[-to]?\s*\$?(\d{1,3}[,.]?\d{0,3})?[k]?', query.lower())
        if salary_match:
            filters['salary_range'] = salary_match.group(0)
        
        return filters
    
    def _determine_search_strategy(
        self,
        expansion: QueryExpansion,
        filters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Determine optimal search strategy based on query analysis"""
        
        # If query is very short or has low confidence, use basic search
        if len(expansion.original_query.split()) <= 2 or expansion.confidence < 0.3:
            return "basic"
        
        # If we have good expansion data, use semantic search
        if expansion.expanded_terms or expansion.primary_skills:
            return "semantic"
        
        # If we have specific filters, use filtered search
        if filters:
            return "filtered"
        
        # If intent suggests specific search type
        if expansion.intent in ["skill_based", "role_specific"]:
            return "semantic"
        elif expansion.intent in ["location_based", "company_based"]:
            return "filtered"
        
        # Default to hybrid search
        return "hybrid"
    
    async def _get_cached_expansion(
        self,
        query: str,
        tenant_id: Optional[str] = None
    ) -> Optional[QueryExpansion]:
        """Retrieve cached query expansion"""
        try:
            # Generate cache key
            cache_key = self._generate_expansion_cache_key(query, tenant_id)
            
            # Try cache manager first
            if self.cache_manager:
                cached_data = await self.cache_manager.get(
                    cache_key,
                    content_type="query_expansion",
                    semantic_search=False  # Exact match for expansions
                )
                
                if cached_data:
                    if isinstance(cached_data, dict):
                        return QueryExpansion(**cached_data)
                    elif isinstance(cached_data, QueryExpansion):
                        return cached_data
            
            # Try database cache
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            
            async with self.db_adapter.get_connection() as conn:
                result = await conn.fetchrow("""
                    SELECT * FROM query_expansions 
                    WHERE query_hash = $1 
                    AND (tenant_id = $2 OR tenant_id IS NULL)
                    AND expires_at > NOW()
                    ORDER BY usage_count DESC
                    LIMIT 1
                """, query_hash, tenant_id)
                
                if result:
                    # Update usage count
                    await conn.execute("""
                        UPDATE query_expansions 
                        SET usage_count = usage_count + 1, last_used_at = NOW()
                        WHERE id = $1
                    """, result['id'])
                    
                    return QueryExpansion(
                        original_query=result['original_query'],
                        expanded_terms=result['expanded_terms'],
                        primary_skills=result['primary_skills'],
                        job_roles=result['job_roles'],
                        experience_level=result['experience_level'],
                        industry=result['industry'],
                        confidence=float(result['confidence_score']) if result['confidence_score'] else 0.8,
                        intent="cached",
                        metadata={"cached_at": result['created_at'].isoformat()}
                    )
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cached expansion: {e}")
            return None
    
    async def _cache_expansion(
        self,
        expansion: QueryExpansion,
        tenant_id: Optional[str] = None
    ) -> None:
        """Cache query expansion for future use"""
        try:
            # Cache in cache manager
            if self.cache_manager:
                cache_key = self._generate_expansion_cache_key(expansion.original_query, tenant_id)
                await self.cache_manager.set(
                    key=cache_key,
                    value=expansion.__dict__,
                    ttl=self.settings.SEMANTIC_CACHE_TTL,
                    content_type="query_expansion",
                    generate_embedding=False
                )
            
            # Cache in database
            query_hash = hashlib.sha256(expansion.original_query.encode()).hexdigest()
            expires_at = datetime.now() + timedelta(seconds=self.settings.SEMANTIC_CACHE_TTL)
            
            async with self.db_adapter.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO query_expansions (
                        original_query, query_hash, expanded_terms, primary_skills,
                        job_roles, experience_level, industry, confidence_score,
                        tenant_id, ai_model_used, expires_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (query_hash, tenant_id) 
                    DO UPDATE SET
                        expanded_terms = EXCLUDED.expanded_terms,
                        primary_skills = EXCLUDED.primary_skills,
                        job_roles = EXCLUDED.job_roles,
                        experience_level = EXCLUDED.experience_level,
                        industry = EXCLUDED.industry,
                        confidence_score = EXCLUDED.confidence_score,
                        expires_at = EXCLUDED.expires_at,
                        last_used_at = NOW()
                """, 
                expansion.original_query,
                query_hash,
                json.dumps(expansion.expanded_terms),
                json.dumps(expansion.primary_skills),
                json.dumps(expansion.job_roles),
                expansion.experience_level,
                expansion.industry,
                expansion.confidence,
                tenant_id,
                expansion.metadata.get("ai_model", "unknown"),
                expires_at
                )
            
        except Exception as e:
            logger.warning(f"Failed to cache expansion: {e}")
    
    def _generate_expansion_cache_key(self, query: str, tenant_id: Optional[str] = None) -> str:
        """Generate cache key for query expansion"""
        key_parts = [query.lower().strip()]
        if tenant_id:
            key_parts.append(tenant_id)
        
        key_string = "|".join(key_parts)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        
        return f"query_expansion:{key_hash}"
    
    async def suggest_queries(
        self,
        partial_query: str,
        limit: int = 5,
        tenant_id: Optional[str] = None
    ) -> List[str]:
        """Generate query suggestions based on partial input"""
        try:
            # Get popular queries from cache
            async with self.db_adapter.get_connection() as conn:
                results = await conn.fetch("""
                    SELECT original_query, usage_count
                    FROM query_expansions
                    WHERE original_query ILIKE $1
                    AND (tenant_id = $2 OR tenant_id IS NULL)
                    AND expires_at > NOW()
                    ORDER BY usage_count DESC
                    LIMIT $3
                """, f"%{partial_query}%", tenant_id, limit)
                
                suggestions = [row['original_query'] for row in results]
                
                # If not enough suggestions, add some common patterns
                if len(suggestions) < limit:
                    common_patterns = [
                        f"{partial_query} developer",
                        f"{partial_query} engineer",
                        f"senior {partial_query}",
                        f"{partial_query} remote",
                        f"{partial_query} manager"
                    ]
                    
                    for pattern in common_patterns:
                        if len(suggestions) >= limit:
                            break
                        if pattern not in suggestions:
                            suggestions.append(pattern)
                
                return suggestions[:limit]
                
        except Exception as e:
            logger.error(f"Query suggestion failed: {e}")
            return []
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get query processing statistics"""
        return {
            "stats": self._stats.copy(),
            "configuration": {
                "expansion_enabled": True,
                "cache_enabled": self.cache_manager is not None,
                "ai_model": self.settings.OPENAI_MODEL
            }
        }
    
    async def check_health(self) -> Dict[str, Any]:
        """Check query processor health"""
        try:
            # Test query processing
            test_query = "python developer remote"
            processed = await self.process_query(
                query=test_query,
                expand_query=True,
                cache_results=False
            )
            
            return {
                "status": "healthy",
                "query_processor": "operational",
                "expansion_service": "operational",
                "test_query_processed": processed.normalized_query is not None,
                "cache_available": self.cache_manager is not None,
                "stats": self._stats.copy(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
