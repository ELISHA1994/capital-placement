"""
Cloud-Agnostic OpenAI Service Implementation

Production-ready OpenAI service supporting both direct OpenAI and Azure OpenAI:
- Unified OpenAI SDK integration (replaces Azure-specific implementation)
- Text embedding generation with automatic model detection
- Chat completions for query expansion and analysis
- Intelligent error handling and retry logic
- Comprehensive caching and rate limiting
- Health monitoring and performance metrics
"""

import asyncio
import hashlib
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import structlog
from openai import AsyncOpenAI
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion

from app.domain.interfaces import IAIService
from app.core.config import get_settings

logger = structlog.get_logger(__name__)


class OpenAIService(IAIService):
    """
    Cloud-agnostic OpenAI service that works with both OpenAI and Azure OpenAI.
    
    Features:
    - Automatic provider detection (OpenAI vs Azure)
    - Unified API interface regardless of provider
    - Intelligent model mapping and configuration
    - Advanced error handling with exponential backoff
    - Comprehensive metrics and monitoring
    - Semantic caching for performance optimization
    """
    
    def __init__(self, cache_service=None):
        self.settings = get_settings()
        self.cache_service = cache_service
        self._client: Optional[AsyncOpenAI] = None
        self._config = self._initialize_config()
        self._model_specs = self._initialize_model_specs()
        self._metrics = {"embeddings": 0, "chat": 0, "errors": 0, "cache_hits": 0}
    
    @classmethod
    async def create(cls, cache_service=None) -> "OpenAIService":
        """Create OpenAI service with validation and initialization"""
        try:
            service = cls(cache_service)
            
            # Test OpenAI client initialization
            _ = service.client
            
            logger.info("OpenAI service created successfully")
            return service
            
        except Exception as e:
            logger.error(f"Failed to create OpenAI service: {e}")
            raise
        
    def _initialize_config(self) -> Dict[str, Any]:
        """Initialize OpenAI configuration with provider detection"""
        try:
            config = self.settings.get_openai_config()
            logger.info(
                "OpenAI configuration initialized",
                provider=config["provider"],
                model=config["model"],
                embedding_model=config["embedding_model"]
            )
            return config
        except ValueError as e:
            logger.error(f"OpenAI configuration error: {e}")
            raise
    
    def _initialize_model_specs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize model specifications for different providers"""
        if self._config["provider"] == "azure":
            return {
                "embeddings": {
                    "model": self._config["embedding_model"],
                    "dimensions": 1536 if "ada-002" in self._config["embedding_model"] else 3072,
                    "max_tokens": 8192,
                    "batch_size": 100
                },
                "chat": {
                    "model": self._config["model"],
                    "max_tokens": self._config["max_tokens"],
                    "temperature": self._config["temperature"]
                }
            }
        else:
            # OpenAI direct
            return {
                "embeddings": {
                    "model": self._config["embedding_model"],
                    "dimensions": self._get_embedding_dimensions(self._config["embedding_model"]),
                    "max_tokens": 8192,
                    "batch_size": 2000
                },
                "chat": {
                    "model": self._config["model"],
                    "max_tokens": self._config["max_tokens"],
                    "temperature": self._config["temperature"]
                }
            }
    
    def _get_embedding_dimensions(self, model: str) -> int:
        """Get embedding dimensions for OpenAI models"""
        dimension_map = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536
        }
        return dimension_map.get(model, 1536)
    
    @property
    def client(self) -> AsyncOpenAI:
        """Get or create OpenAI client"""
        if self._client is None:
            try:
                client_kwargs = {
                    "api_key": self._config["api_key"],
                    "timeout": self.settings.REQUEST_TIMEOUT,
                    "max_retries": 3
                }
                
                # Add provider-specific configuration
                if self._config["provider"] == "azure":
                    client_kwargs.update({
                        "base_url": self._config["base_url"],
                        "default_headers": {"api-version": self._config["api_version"]}
                    })
                elif self._config["base_url"] != "https://api.openai.com/v1":
                    client_kwargs["base_url"] = self._config["base_url"]
                
                self._client = AsyncOpenAI(**client_kwargs)
                logger.info(
                    "OpenAI client initialized successfully",
                    provider=self._config["provider"]
                )
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                raise
        
        return self._client
    
    async def generate_embedding(
        self, 
        text: str, 
        model: Optional[str] = None
    ) -> List[float]:
        """
        Generate single text embedding.
        
        Args:
            text: Text to embed
            model: Override model (optional)
            
        Returns:
            List of floats representing the text embedding
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        model = model or self._model_specs["embeddings"]["model"]
        
        # Clean and validate text
        text = self._clean_text(text)
        
        # Check cache first
        cache_key = self._get_cache_key("embedding", text, model)
        if self.cache_service:
            cached = await self.cache_service.get(cache_key)
            if cached:
                self._metrics["cache_hits"] += 1
                logger.debug("Retrieved embedding from cache")
                return cached
        
        try:
            start_time = time.time()
            
            response: CreateEmbeddingResponse = await self.client.embeddings.create(
                input=text,
                model=model
            )
            
            embedding = response.data[0].embedding
            self._metrics["embeddings"] += 1
            
            # Cache the result
            if self.cache_service:
                await self.cache_service.set(
                    cache_key,
                    embedding,
                    ttl=self.settings.SEMANTIC_CACHE_TTL
                )
            
            duration = time.time() - start_time
            logger.debug(
                "Generated embedding",
                model=model,
                text_length=len(text),
                dimensions=len(embedding),
                duration_ms=int(duration * 1000)
            )
            
            return embedding
            
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error(f"Failed to generate embedding: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")
    
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            model: Override model (optional)
            
        Returns:
            List of embedding vectors in same order as input
        """
        if not texts:
            return []
        
        model = model or self._model_specs["embeddings"]["model"]
        batch_size = self._model_specs["embeddings"]["batch_size"]
        
        # Clean texts and check cache
        cleaned_texts = [self._clean_text(text) for text in texts]
        results = [None] * len(texts)
        texts_to_process = []
        indices_to_process = []
        
        if self.cache_service:
            for i, text in enumerate(cleaned_texts):
                if text:
                    cache_key = self._get_cache_key("embedding", text, model)
                    cached = await self.cache_service.get(cache_key)
                    if cached:
                        results[i] = cached
                        self._metrics["cache_hits"] += 1
                    else:
                        texts_to_process.append(text)
                        indices_to_process.append(i)
                else:
                    # Empty text - return zero vector
                    dimensions = self._model_specs["embeddings"]["dimensions"]
                    results[i] = [0.0] * dimensions
        else:
            for i, text in enumerate(cleaned_texts):
                if text:
                    texts_to_process.append(text)
                    indices_to_process.append(i)
                else:
                    dimensions = self._model_specs["embeddings"]["dimensions"]
                    results[i] = [0.0] * dimensions
        
        if not texts_to_process:
            return results
        
        try:
            # Process in batches
            for i in range(0, len(texts_to_process), batch_size):
                batch_texts = texts_to_process[i:i + batch_size]
                batch_indices = indices_to_process[i:i + batch_size]
                
                response: CreateEmbeddingResponse = await self.client.embeddings.create(
                    input=batch_texts,
                    model=model
                )
                
                for j, embedding_data in enumerate(response.data):
                    original_idx = batch_indices[j]
                    embedding = embedding_data.embedding
                    results[original_idx] = embedding
                    
                    # Cache individual embeddings
                    if self.cache_service:
                        text = batch_texts[j]
                        cache_key = self._get_cache_key("embedding", text, model)
                        await self.cache_service.set(
                            cache_key,
                            embedding,
                            ttl=self.settings.SEMANTIC_CACHE_TTL
                        )
                
                # Rate limiting
                if i + batch_size < len(texts_to_process):
                    await asyncio.sleep(0.1)
            
            self._metrics["embeddings"] += len(texts_to_process)
            
            logger.info(
                "Generated batch embeddings",
                total_texts=len(texts),
                processed=len(texts_to_process),
                cached=len(texts) - len(texts_to_process),
                model=model
            )
            
            return results
            
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise RuntimeError(f"Batch embedding generation failed: {e}")
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate chat completion for AI analysis and query expansion.
        
        Args:
            messages: List of message dictionaries
            model: Override model (optional)
            max_tokens: Override max tokens (optional)
            temperature: Override temperature (optional)
            **kwargs: Additional parameters
            
        Returns:
            Chat completion response
        """
        if not messages:
            raise ValueError("Messages cannot be empty")
        
        model = model or self._model_specs["chat"]["model"]
        max_tokens = max_tokens or self._model_specs["chat"]["max_tokens"]
        temperature = temperature if temperature is not None else self._model_specs["chat"]["temperature"]
        
        try:
            start_time = time.time()
            
            response: ChatCompletion = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            self._metrics["chat"] += 1
            
            # Convert to standardized format
            result = {
                "choices": [
                    {
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content
                        },
                        "finish_reason": choice.finish_reason,
                        "index": choice.index
                    }
                    for choice in response.choices
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                },
                "model": response.model,
                "id": response.id
            }
            
            duration = time.time() - start_time
            logger.debug(
                "Generated chat completion",
                model=model,
                tokens=result["usage"]["total_tokens"],
                duration_ms=int(duration * 1000)
            )
            
            return result
            
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error(f"Failed to generate chat completion: {e}")
            raise RuntimeError(f"Chat completion failed: {e}")
    
    async def extract_text_from_document(
        self,
        document_content: bytes,
        document_type: str = "pdf"
    ) -> Dict[str, Any]:
        """
        Document extraction using AI analysis (fallback method).
        For comprehensive document processing, use DocumentProcessor service.
        """
        logger.warning("Basic document extraction - use DocumentProcessor for full features")
        
        if document_type.lower() in ["txt", "text"]:
            try:
                text = document_content.decode("utf-8")
                return {
                    "text": text,
                    "pages": [{"text": text, "page_number": 1}],
                    "metadata": {
                        "document_type": document_type,
                        "extraction_method": "text_decode",
                        "character_count": len(text)
                    }
                }
            except UnicodeDecodeError as e:
                raise ValueError(f"Failed to decode text document: {e}")
        
        raise NotImplementedError(
            "Advanced document extraction requires DocumentProcessor service"
        )
    
    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """Get embedding vector dimension"""
        model = model or self._model_specs["embeddings"]["model"]
        return self._model_specs["embeddings"]["dimensions"]
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for processing"""
        if not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Truncate if too long (conservative token estimate: 1 token ≈ 4 chars)
        max_chars = self._model_specs["embeddings"]["max_tokens"] * 3
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.warning(f"Text truncated to {max_chars} characters")
        
        return text
    
    def _get_cache_key(self, operation: str, text: str, model: str) -> str:
        """Generate cache key for operations"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"openai:{operation}:{model}:{text_hash}"
    
    async def check_health(self) -> Dict[str, Any]:
        """Check service health with comprehensive diagnostics"""
        try:
            start_time = time.time()
            
            # Test embedding generation
            test_text = "Health check test"
            embedding = await self.generate_embedding(test_text)
            
            response_time = time.time() - start_time
            expected_dim = self.get_embedding_dimension()
            
            health_status = {
                "status": "healthy",
                "provider": self._config["provider"],
                "models": {
                    "chat": self._model_specs["chat"]["model"],
                    "embedding": self._model_specs["embeddings"]["model"]
                },
                "response_time_ms": int(response_time * 1000),
                "embedding_dimension": len(embedding),
                "expected_dimension": expected_dim,
                "dimension_correct": len(embedding) == expected_dim,
                "metrics": self._metrics.copy(),
                "cache_configured": self.cache_service is not None,
                "timestamp": datetime.now().isoformat()
            }
            
            if response_time > 5.0:
                health_status["status"] = "degraded"
                health_status["warning"] = "High response time"
            
            logger.info("OpenAI service health check passed", **health_status)
            return health_status
            
        except Exception as e:
            error_status = {
                "status": "unhealthy",
                "error": str(e),
                "provider": self._config.get("provider", "unknown"),
                "metrics": self._metrics.copy(),
                "timestamp": datetime.now().isoformat()
            }
            logger.error("OpenAI service health check failed", **error_status)
            return error_status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        return {
            "requests": self._metrics.copy(),
            "config": {
                "provider": self._config["provider"],
                "model": self._model_specs["chat"]["model"],
                "embedding_model": self._model_specs["embeddings"]["model"]
            },
            "cache_enabled": self.cache_service is not None
        }
