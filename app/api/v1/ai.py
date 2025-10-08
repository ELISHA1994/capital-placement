"""
AI API Endpoints

Dedicated AI-powered endpoints for:
- Document analysis and processing
- Chat completion and assistance
- Embedding generation and similarity
- Query expansion and suggestions
- Quality analysis and scoring
- Real-time AI operations monitoring
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
import structlog

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator

from app.infrastructure.persistence.models.auth_tables import CurrentUser
from app.api.schemas.base import PaginatedResponse
from app.infrastructure.persistence.models.base import PaginationModel
from app.core.dependencies import CurrentUserDep
from app.api.dependencies import map_domain_exception_to_http
from app.domain.exceptions import DomainException

# DocumentAnalyzer replaced with QualityAnalyzer for document analysis
from app.infrastructure.providers.document_provider import get_quality_analyzer

# Core Services
from app.core.config import get_settings
from app.infrastructure.providers.ai_provider import (
    get_openai_service,
    get_embedding_service,
    get_prompt_manager,
)
from app.infrastructure.providers.postgres_provider import get_postgres_adapter
from app.infrastructure.providers.search_provider import get_query_processor

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/ai", tags=["ai"])


# Request/Response Models

class ChatCompletionRequest(BaseModel):
    """Request model for AI chat completion"""
    
    messages: List[Dict[str, str]] = Field(..., description="Conversation messages")
    model: Optional[str] = Field(None, description="AI model to use")
    max_tokens: Optional[int] = Field(1000, ge=1, le=4000, description="Maximum response tokens")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Response creativity")
    stream: bool = Field(False, description="Enable streaming response")
    context: Optional[str] = Field(None, description="Additional context for the conversation")
    
    @validator('messages')
    def validate_messages(cls, messages):
        if not messages:
            raise ValueError("At least one message is required")
        
        for msg in messages:
            if 'role' not in msg or 'content' not in msg:
                raise ValueError("Each message must have 'role' and 'content'")
            if msg['role'] not in ['user', 'assistant', 'system']:
                raise ValueError("Message role must be 'user', 'assistant', or 'system'")
        
        return messages


class DocumentAnalysisRequest(BaseModel):
    """Request model for document analysis"""
    
    text_content: str = Field(..., min_length=50, description="Document text content")
    document_type: str = Field("cv", description="Type of document (cv, resume, job_description)")
    analysis_type: str = Field("comprehensive", description="Analysis depth (basic, comprehensive, detailed)")
    extract_skills: bool = Field(True, description="Extract skills and technologies")
    extract_experience: bool = Field(True, description="Extract work experience")
    extract_education: bool = Field(True, description="Extract education information")
    generate_summary: bool = Field(True, description="Generate AI summary")


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation"""
    
    texts: List[str] = Field(..., description="Texts to generate embeddings for")
    model: Optional[str] = Field(None, description="Embedding model to use")
    normalize: bool = Field(True, description="Normalize embedding vectors")
    
    @validator('texts')
    def validate_texts(cls, texts):
        if not texts:
            raise ValueError("At least one text is required")
        if len(texts) > 100:
            raise ValueError("Maximum 100 texts per request")
        return texts


class SimilarityRequest(BaseModel):
    """Request model for similarity calculation"""
    
    query_text: str = Field(..., min_length=1, description="Query text")
    candidate_texts: List[str] = Field(..., description="Candidate texts to compare")
    threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Similarity threshold")
    top_k: Optional[int] = Field(10, ge=1, le=100, description="Number of top results")


class QueryExpansionRequest(BaseModel):
    """Request model for query expansion"""
    
    query: str = Field(..., min_length=2, description="Original search query")
    domain: str = Field("general", description="Domain context (general, tech, hr, finance)")
    max_expansions: int = Field(5, ge=1, le=20, description="Maximum number of expansions")
    include_synonyms: bool = Field(True, description="Include synonym expansions")
    include_related: bool = Field(True, description="Include related terms")


# Response Models

class ChatCompletionResponse(BaseModel):
    """Response model for AI chat completion"""
    
    id: str
    message: str
    model_used: str
    usage: Dict[str, int]
    response_time_ms: int
    created_at: datetime


class DocumentAnalysisResponse(BaseModel):
    """Response model for document analysis"""
    
    analysis_id: str
    document_type: str
    analysis_result: Dict[str, Any]
    quality_scores: Dict[str, float]
    extracted_entities: Dict[str, List[str]]
    summary: Optional[str]
    processing_time_ms: int
    confidence_scores: Dict[str, float]


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation"""
    
    embeddings: List[List[float]]
    model_used: str
    dimensions: int
    processing_time_ms: int
    usage: Dict[str, int]


class SimilarityResponse(BaseModel):
    """Response model for similarity calculation"""
    
    results: List[Dict[str, Any]]
    query_embedding: List[float]
    processing_time_ms: int
    total_comparisons: int


class QueryExpansionResponse(BaseModel):
    """Response model for query expansion"""
    
    original_query: str
    expanded_terms: List[str]
    synonyms: List[str]
    related_terms: List[str]
    confidence_score: float
    processing_time_ms: int


# API Endpoints

@router.post("/chat/completion", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    current_user: CurrentUserDep,
) -> ChatCompletionResponse:
    """
    Generate AI chat completion with conversation context.
    
    Supports:
    - Multi-turn conversations with context
    - Streaming and non-streaming responses
    - Configurable model parameters
    - Usage tracking and rate limiting
    - CV/HR domain-specific assistance
    """
    start_time = datetime.now()
    
    try:
        
        settings = get_settings()
        if not settings.is_openai_configured():
            raise HTTPException(
                status_code=503,
                detail="AI services not configured"
            )
        
        # Get AI services
        openai_service = await get_openai_service()
        prompt_manager = await get_prompt_manager()
        
        # Prepare messages with system context
        messages = request.messages.copy()
        
        # Add system context for CV/HR domain
        if request.context:
            system_message = {
                "role": "system",
                "content": await prompt_manager.get_prompt("chat_system_context", {
                    "domain": "cv_analysis",
                    "context": request.context
                })
            }
            messages.insert(0, system_message)
        
        # Generate completion
        completion_result = await openai_service.create_chat_completion(
            messages=messages,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=request.stream
        )
        
        response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Track usage in background
        background_tasks.add_task(
            _track_ai_usage,
            tenant_id=str(current_user.tenant_id),
            user_id=current_user.user_id,
            operation_type="chat_completion",
            model_used=completion_result.get("model", request.model or "gpt-4"),
            usage=completion_result.get("usage", {}),
            processing_time_ms=response_time_ms
        )
        
        response = ChatCompletionResponse(
            id=str(uuid4()),
            message=completion_result["message"],
            model_used=completion_result.get("model", request.model or "gpt-4"),
            usage=completion_result.get("usage", {}),
            response_time_ms=response_time_ms,
            created_at=start_time
        )
        
        logger.info(
            "Chat completion generated successfully",
            user_id=current_user.user_id,
            model=response.model_used,
            response_time_ms=response_time_ms
        )
        
        return response
        
    except DomainException as domain_exc:
        # Map domain exceptions to appropriate HTTP responses
        raise map_domain_exception_to_http(domain_exc)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate chat completion"
        )


@router.post("/analyze/document", response_model=DocumentAnalysisResponse)
async def analyze_document(
    request: DocumentAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: CurrentUserDep,
) -> DocumentAnalysisResponse:
    """
    Perform comprehensive AI-powered document analysis.
    
    Features:
    - Intelligent CV/resume parsing
    - Skills and experience extraction
    - Quality assessment and scoring
    - Entity recognition and classification
    - Multi-language support
    """
    start_time = datetime.now()
    
    try:
        
        settings = get_settings()
        if not settings.is_openai_configured():
            raise HTTPException(
                status_code=503,
                detail="AI services not configured"
            )
        
        # Get AI services
        openai_service = await get_openai_service()
        prompt_manager = await get_prompt_manager()
        postgres_adapter = await get_postgres_adapter()

        # Get document analyzer (using QualityAnalyzer via provider)
        document_analyzer = await get_quality_analyzer()
        
        # Perform comprehensive analysis
        analysis_result = await document_analyzer.analyze_document_quality(
            text_content=request.text_content,
            document_type=request.document_type,
            tenant_id=str(current_user.tenant_id),
            metadata={
                "analysis_type": request.analysis_type,
                "extract_skills": request.extract_skills,
                "extract_experience": request.extract_experience,
                "extract_education": request.extract_education
            }
        )
        
        # Perform quality analysis (reuse same analyzer instance)
        quality_scores = await document_analyzer.analyze_document_quality(
            text=request.text_content,
            document_type=request.document_type,
            structured_data=analysis_result,
            use_ai=True
        )
        
        # Generate summary if requested
        summary = None
        if request.generate_summary:
            summary = analysis_result.get("summary", "")
        
        # Extract entities for response
        extracted_entities = {}
        if analysis_result.get("skills"):
            extracted_entities["skills"] = analysis_result["skills"]
        if analysis_result.get("experience"):
            extracted_entities["experience"] = [exp.get("title", "") for exp in analysis_result["experience"]]
        if analysis_result.get("education"):
            extracted_entities["education"] = [edu.get("degree", "") for edu in analysis_result["education"]]
        
        response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Track usage in background
        background_tasks.add_task(
            _track_ai_usage,
            tenant_id=str(current_user.tenant_id),
            user_id=current_user.user_id,
            operation_type="document_analysis",
            model_used="gpt-4",
            processing_time_ms=response_time_ms,
            input_size=len(request.text_content)
        )
        
        response = DocumentAnalysisResponse(
            analysis_id=str(uuid4()),
            document_type=request.document_type,
            analysis_result=analysis_result,
            quality_scores=quality_scores,
            extracted_entities=extracted_entities,
            summary=summary,
            processing_time_ms=response_time_ms,
            confidence_scores=analysis_result.get("confidence_scores", {})
        )
        
        logger.info(
            "Document analysis completed successfully",
            user_id=current_user.user_id,
            document_type=request.document_type,
            processing_time_ms=response_time_ms
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze document"
        )


@router.post("/embeddings/generate", response_model=EmbeddingResponse)
async def generate_embeddings(
    request: EmbeddingRequest,
    background_tasks: BackgroundTasks,
    current_user: CurrentUserDep,
) -> EmbeddingResponse:
    """
    Generate vector embeddings for text content.
    
    Features:
    - Batch embedding generation
    - Multiple embedding models support
    - Vector normalization options
    - Usage tracking and optimization
    """
    start_time = datetime.now()
    
    try:
        
        settings = get_settings()
        if not settings.is_openai_configured():
            raise HTTPException(
                status_code=503,
                detail="AI services not configured"
            )
        
        # Get embedding service
        embedding_service = await get_embedding_service()
        
        # Generate embeddings for all texts
        all_embeddings = []
        total_usage = {"prompt_tokens": 0, "total_tokens": 0}
        
        for text in request.texts:
            embedding = await embedding_service.generate_embedding(
                text=text,
                tenant_id=str(current_user.tenant_id),
                model=request.model,
                normalize=request.normalize
            )
            all_embeddings.append(embedding)
        
        response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        model_used = request.model or settings.OPENAI_EMBEDDING_MODEL
        dimensions = len(all_embeddings[0]) if all_embeddings else 0
        
        # Track usage in background
        background_tasks.add_task(
            _track_ai_usage,
            tenant_id=str(current_user.tenant_id),
            user_id=current_user.user_id,
            operation_type="embedding_generation",
            model_used=model_used,
            usage=total_usage,
            processing_time_ms=response_time_ms,
            input_size=sum(len(text) for text in request.texts)
        )
        
        response = EmbeddingResponse(
            embeddings=all_embeddings,
            model_used=model_used,
            dimensions=dimensions,
            processing_time_ms=response_time_ms,
            usage=total_usage
        )
        
        logger.info(
            "Embeddings generated successfully",
            user_id=current_user.user_id,
            text_count=len(request.texts),
            model=model_used,
            processing_time_ms=response_time_ms
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate embeddings"
        )


@router.post("/similarity/calculate", response_model=SimilarityResponse)
async def calculate_similarity(
    request: SimilarityRequest,
    background_tasks: BackgroundTasks,
    current_user: CurrentUserDep,
) -> SimilarityResponse:
    """
    Calculate semantic similarity between texts using embeddings.
    
    Features:
    - Vector-based similarity calculation
    - Configurable similarity thresholds
    - Batch similarity processing
    - Result ranking and filtering
    """
    start_time = datetime.now()
    
    try:
        
        settings = get_settings()
        if not settings.is_openai_configured():
            raise HTTPException(
                status_code=503,
                detail="AI services not configured"
            )
        
        # Get embedding service
        embedding_service = await get_embedding_service()
        
        # Generate embedding for query
        query_embedding = await embedding_service.generate_embedding(
            text=request.query_text,
            tenant_id=str(current_user.tenant_id)
        )
        
        # Generate embeddings for candidates
        candidate_embeddings = []
        for text in request.candidate_texts:
            embedding = await embedding_service.generate_embedding(
                text=text,
                tenant_id=str(current_user.tenant_id)
            )
            candidate_embeddings.append(embedding)
        
        # Calculate similarities
        similarities = []
        for i, candidate_embedding in enumerate(candidate_embeddings):
            similarity_score = embedding_service.calculate_similarity(
                query_embedding, candidate_embedding
            )
            
            if similarity_score >= request.threshold:
                similarities.append({
                    "index": i,
                    "text": request.candidate_texts[i],
                    "similarity_score": similarity_score
                })
        
        # Sort by similarity and take top K
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        similarities = similarities[:request.top_k]
        
        response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Track usage in background
        background_tasks.add_task(
            _track_ai_usage,
            tenant_id=str(current_user.tenant_id),
            user_id=current_user.user_id,
            operation_type="similarity_calculation",
            processing_time_ms=response_time_ms,
            input_size=len(request.query_text) + sum(len(text) for text in request.candidate_texts)
        )
        
        response = SimilarityResponse(
            results=similarities,
            query_embedding=query_embedding,
            processing_time_ms=response_time_ms,
            total_comparisons=len(request.candidate_texts)
        )
        
        logger.info(
            "Similarity calculation completed successfully",
            user_id=current_user.user_id,
            query_length=len(request.query_text),
            candidate_count=len(request.candidate_texts),
            results_count=len(similarities),
            processing_time_ms=response_time_ms
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similarity calculation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to calculate similarity"
        )


@router.post("/query/expand", response_model=QueryExpansionResponse)
async def expand_query(
    request: QueryExpansionRequest,
    background_tasks: BackgroundTasks,
    current_user: CurrentUserDep,
) -> QueryExpansionResponse:
    """
    Expand search queries using AI-powered term expansion.
    
    Features:
    - Domain-specific query expansion
    - Synonym and related term generation
    - Confidence scoring for expansions
    - Context-aware term suggestions
    """
    start_time = datetime.now()
    
    try:
        
        settings = get_settings()
        if not settings.is_openai_configured():
            raise HTTPException(
                status_code=503,
                detail="AI services not configured"
            )
        
        # Get query processor service
        query_processor = await get_query_processor()
        
        # Expand query
        expansion_result = await query_processor.expand_query(
            query=request.query,
            tenant_id=str(current_user.tenant_id),
            domain=request.domain,
            max_expansions=request.max_expansions,
            include_synonyms=request.include_synonyms,
            include_related=request.include_related
        )
        
        response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Track usage in background
        background_tasks.add_task(
            _track_ai_usage,
            tenant_id=str(current_user.tenant_id),
            user_id=current_user.user_id,
            operation_type="query_expansion",
            processing_time_ms=response_time_ms,
            input_size=len(request.query)
        )
        
        response = QueryExpansionResponse(
            original_query=request.query,
            expanded_terms=expansion_result.get("expanded_terms", []),
            synonyms=expansion_result.get("synonyms", []),
            related_terms=expansion_result.get("related_terms", []),
            confidence_score=expansion_result.get("confidence", 0.8),
            processing_time_ms=response_time_ms
        )
        
        logger.info(
            "Query expansion completed successfully",
            user_id=current_user.user_id,
            original_query=request.query,
            expansion_count=len(response.expanded_terms),
            processing_time_ms=response_time_ms
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query expansion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to expand query"
        )


@router.get("/usage/analytics")
async def get_ai_usage_analytics(
    current_user: CurrentUserDep,
    start_date: Optional[datetime] = Query(None, description="Start date for analytics"),
    end_date: Optional[datetime] = Query(None, description="End date for analytics"),
    operation_type: Optional[str] = Query(None, description="Filter by operation type")
) -> JSONResponse:
    """
    Get AI usage analytics and cost insights.
    
    Provides:
    - Token usage and cost tracking
    - Performance metrics by operation
    - Model usage distribution
    - Error rates and success metrics
    """
    try:
        postgres_adapter = await get_postgres_adapter()
        
        # Build analytics query
        where_conditions = ["tenant_id = $1"]
        params = [str(current_user.tenant_id)]
        param_count = 1
        
        if start_date:
            param_count += 1
            where_conditions.append(f"created_at >= ${param_count}")
            params.append(start_date)
        
        if end_date:
            param_count += 1
            where_conditions.append(f"created_at <= ${param_count}")
            params.append(end_date)
        
        if operation_type:
            param_count += 1
            where_conditions.append(f"operation_type = ${param_count}")
            params.append(operation_type)
        
        where_clause = " AND ".join(where_conditions)
        
        # Get usage analytics
        analytics_data = await postgres_adapter.fetch_all(
            f"""
            SELECT 
                operation_type,
                ai_model,
                COUNT(*) as operation_count,
                SUM(COALESCE((token_usage->>'total_tokens')::INTEGER, 0)) as total_tokens,
                AVG(processing_time_ms) as avg_processing_time,
                SUM(COALESCE(cost_estimate, 0)) as total_cost,
                COUNT(CASE WHEN success THEN 1 END)::DECIMAL / COUNT(*) as success_rate
            FROM ai_analytics 
            WHERE {where_clause}
            GROUP BY operation_type, ai_model
            ORDER BY total_tokens DESC
            """,
            *params
        )
        
        # Format response
        analytics_summary = {
            "time_range": {
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None
            },
            "operations": [
                {
                    "operation_type": row["operation_type"],
                    "ai_model": row["ai_model"],
                    "operation_count": row["operation_count"],
                    "total_tokens": row["total_tokens"],
                    "avg_processing_time_ms": float(row["avg_processing_time"]) if row["avg_processing_time"] else 0,
                    "total_cost": float(row["total_cost"]),
                    "success_rate": float(row["success_rate"])
                }
                for row in analytics_data
            ],
            "totals": {
                "total_operations": sum(row["operation_count"] for row in analytics_data),
                "total_tokens": sum(row["total_tokens"] for row in analytics_data),
                "total_cost": sum(float(row["total_cost"]) for row in analytics_data),
                "overall_success_rate": (
                    sum(row["operation_count"] * float(row["success_rate"]) for row in analytics_data) /
                    sum(row["operation_count"] for row in analytics_data)
                    if analytics_data else 0
                )
            }
        }
        
        return JSONResponse(content=analytics_summary)
        
    except Exception as e:
        logger.error(f"Failed to get AI usage analytics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve AI usage analytics"
        )


# Background task functions

async def _track_ai_usage(
    tenant_id: str,
    user_id: str,
    operation_type: str,
    model_used: str = "unknown",
    usage: Dict[str, int] = None,
    processing_time_ms: int = 0,
    input_size: int = 0,
    output_size: int = 0
) -> None:
    """Track AI usage for analytics and billing"""
    try:
        postgres_adapter = await get_postgres_adapter()
        
        # Calculate estimated cost (simplified pricing)
        cost_estimate = 0.0
        if usage:
            # Rough pricing estimates (adjust based on actual pricing)
            if "gpt-4" in model_used.lower():
                cost_estimate = (usage.get("prompt_tokens", 0) * 0.03 + usage.get("completion_tokens", 0) * 0.06) / 1000
            elif "gpt-3.5" in model_used.lower():
                cost_estimate = (usage.get("total_tokens", 0) * 0.002) / 1000
            elif "embedding" in model_used.lower():
                cost_estimate = (usage.get("total_tokens", 0) * 0.0001) / 1000
        
        # Store analytics record
        await postgres_adapter.execute(
            """
            INSERT INTO ai_analytics (
                tenant_id, operation_type, ai_model, token_usage, processing_time_ms,
                input_size, output_size, success, cost_estimate, user_id
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            tenant_id, operation_type, model_used, 
            usage or {}, processing_time_ms, input_size, output_size,
            True, cost_estimate, user_id
        )
        
        logger.debug(
            "AI usage tracked",
            tenant_id=tenant_id,
            operation_type=operation_type,
            model_used=model_used,
            cost_estimate=cost_estimate
        )
        
    except Exception as e:
        logger.warning(f"Failed to track AI usage: {e}")
