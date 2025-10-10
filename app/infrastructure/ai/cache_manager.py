"""
Semantic Cache Manager with Similarity Matching

Advanced caching system for AI operations with semantic similarity:
- Multi-tier caching (memory + Redis)
- Semantic similarity matching for cache retrieval
- Intelligent cache invalidation strategies
- Performance optimization and analytics
- Support for different content types and operations
"""

import json
import time
import hashlib
import uuid
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import structlog
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import get_settings

logger = structlog.get_logger(__name__)


# ✅ FIX P0: Custom JSON encoder to handle UUID, Enum, and datetime serialization
class UUIDEncoder(json.JSONEncoder):
    """JSON encoder that handles UUID, Enum, and datetime objects."""

    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


class CacheEntry:
    """Represents a cache entry with metadata"""
    
    def __init__(
        self,
        key: str,
        value: Any,
        embedding: Optional[List[float]] = None,
        ttl: Optional[int] = None,
        content_type: str = "generic",
        created_at: Optional[datetime] = None
    ):
        self.key = key
        self.value = value
        self.embedding = embedding
        self.ttl = ttl
        self.content_type = content_type
        self.created_at = created_at or datetime.now()
        self.access_count = 1
        self.last_accessed = self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "key": self.key,
            "value": self.value,
            "embedding": self.embedding,
            "ttl": self.ttl,
            "content_type": self.content_type,
            "created_at": self.created_at.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create from dictionary"""
        entry = cls(
            key=data["key"],
            value=data["value"],
            embedding=data.get("embedding"),
            ttl=data.get("ttl"),
            content_type=data.get("content_type", "generic"),
            created_at=datetime.fromisoformat(data["created_at"])
        )
        entry.access_count = data.get("access_count", 1)
        entry.last_accessed = datetime.fromisoformat(data.get("last_accessed", data["created_at"]))
        return entry
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl is None:
            return False
        
        expiry_time = self.created_at + timedelta(seconds=self.ttl)
        return datetime.now() > expiry_time
    
    def update_access(self) -> None:
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = datetime.now()


class CacheManager:
    """
    Advanced semantic cache manager with multi-tier caching and similarity matching.
    
    Features:
    - Memory cache for ultra-fast access
    - Redis cache for persistence and scaling
    - Semantic similarity matching for intelligent retrieval
    - Configurable similarity thresholds
    - Automatic cache warming and invalidation
    - Comprehensive metrics and analytics
    """
    
    def __init__(self, redis_client=None, embedding_service=None):
        self.settings = get_settings()
        self.redis_client = redis_client
        self.embedding_service = embedding_service
        
        # Memory cache for ultra-fast access
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._max_memory_entries = 1000
        
        # Metrics and monitoring
        self._metrics = {
            "memory_hits": 0,
            "redis_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "sets": 0,
            "invalidations": 0,
            "similarity_searches": 0
        }
        
        # Similarity threshold for semantic matching
        self._similarity_threshold = self.settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD
        
        # Background cleanup task
        self._cleanup_task = None
        if redis_client:
            self._start_cleanup_task()
    
    def _start_cleanup_task(self) -> None:
        """Start background cleanup task"""
        async def cleanup_expired():
            while True:
                try:
                    await self._cleanup_expired_entries()
                    await asyncio.sleep(300)  # 5 minutes
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
                    await asyncio.sleep(60)  # 1 minute on error
        
        self._cleanup_task = asyncio.create_task(cleanup_expired())
    
    async def get(
        self,
        key: str,
        content_type: str = "generic",
        semantic_search: bool = True
    ) -> Optional[Any]:
        """
        Retrieve item from cache with semantic similarity fallback.
        
        Args:
            key: Cache key
            content_type: Type of content for filtering
            semantic_search: Enable semantic similarity search if exact match fails
            
        Returns:
            Cached value or None if not found
        """
        # First, try exact key match in memory
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if not entry.is_expired():
                entry.update_access()
                self._metrics["memory_hits"] += 1
                logger.debug("Memory cache hit", key=key)
                return entry.value
            else:
                # Remove expired entry
                del self._memory_cache[key]
        
        # Try exact key match in Redis
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(f"cache:{key}")
                if cached_data:
                    entry_dict = json.loads(cached_data)
                    entry = CacheEntry.from_dict(entry_dict)
                    
                    if not entry.is_expired():
                        # Store in memory for fast future access
                        self._store_in_memory(entry)
                        entry.update_access()
                        self._metrics["redis_hits"] += 1
                        logger.debug("Redis cache hit", key=key)
                        return entry.value
                    else:
                        # Remove expired entry
                        await self.redis_client.delete(f"cache:{key}")
            except Exception as e:
                logger.warning(f"Redis cache retrieval error: {e}")
        
        # If semantic search is enabled and we have embedding service
        if semantic_search and self.embedding_service:
            semantic_result = await self._semantic_search(key, content_type)
            if semantic_result:
                self._metrics["semantic_hits"] += 1
                logger.debug("Semantic cache hit", key=key, similar_key=semantic_result[0])
                return semantic_result[1]
        
        self._metrics["misses"] += 1
        logger.debug("Cache miss", key=key)
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        content_type: str = "generic",
        generate_embedding: bool = True
    ) -> bool:
        """
        Store item in cache with optional embedding generation.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            content_type: Type of content
            generate_embedding: Generate embedding for semantic search
            
        Returns:
            True if successfully stored
        """
        try:
            # Generate embedding if requested and service available
            embedding = None
            if generate_embedding and self.embedding_service and isinstance(value, str):
                try:
                    embedding = await self.embedding_service.generate_embedding(value)
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for cache: {e}")
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                embedding=embedding,
                ttl=ttl or self.settings.SEMANTIC_CACHE_TTL,
                content_type=content_type
            )
            
            # Store in memory
            self._store_in_memory(entry)
            
            # Store in Redis if available
            if self.redis_client:
                try:
                    cache_key = f"cache:{key}"
                    entry_json = json.dumps(entry.to_dict(), cls=UUIDEncoder)  # ✅ FIX P2: Use UUID encoder
                    
                    if ttl:
                        await self.redis_client.setex(cache_key, ttl, entry_json)
                    else:
                        await self.redis_client.set(cache_key, entry_json)
                    
                    # Store embedding separately for similarity search
                    if embedding:
                        embedding_key = f"embedding:{key}"
                        embedding_data = {
                            "key": key,
                            "embedding": embedding,
                            "content_type": content_type,
                            "created_at": entry.created_at.isoformat()
                        }
                        await self.redis_client.setex(
                            embedding_key,
                            ttl or self.settings.SEMANTIC_CACHE_TTL,
                            json.dumps(embedding_data, cls=UUIDEncoder)  # ✅ FIX P2: Use UUID encoder
                        )
                        
                except Exception as e:
                    logger.warning(f"Redis cache storage error: {e}")
            
            self._metrics["sets"] += 1
            logger.debug("Cache set", key=key, has_embedding=embedding is not None)
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete item from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if deleted
        """
        deleted = False
        
        # Remove from memory cache
        if key in self._memory_cache:
            del self._memory_cache[key]
            deleted = True
        
        # Remove from Redis
        if self.redis_client:
            try:
                redis_deleted = await self.redis_client.delete(f"cache:{key}")
                await self.redis_client.delete(f"embedding:{key}")  # Also remove embedding
                deleted = deleted or redis_deleted > 0
            except Exception as e:
                logger.warning(f"Redis cache deletion error: {e}")
        
        if deleted:
            self._metrics["invalidations"] += 1
            logger.debug("Cache delete", key=key)
        
        return deleted
    
    async def clear(self, content_type: Optional[str] = None) -> int:
        """
        Clear cache entries, optionally by content type.
        
        Args:
            content_type: Clear only entries of this type
            
        Returns:
            Number of entries cleared
        """
        cleared_count = 0
        
        # Clear memory cache
        if content_type:
            keys_to_remove = [
                key for key, entry in self._memory_cache.items()
                if entry.content_type == content_type
            ]
        else:
            keys_to_remove = list(self._memory_cache.keys())
        
        for key in keys_to_remove:
            del self._memory_cache[key]
            cleared_count += 1
        
        # Clear Redis cache
        if self.redis_client:
            try:
                if content_type:
                    # This is more complex - would need to scan keys
                    # For now, just clear all cache entries
                    pattern = "cache:*"
                else:
                    pattern = "cache:*"
                
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
                
                # Also clear embeddings
                embedding_keys = await self.redis_client.keys("embedding:*")
                if embedding_keys:
                    await self.redis_client.delete(*embedding_keys)
                
            except Exception as e:
                logger.warning(f"Redis cache clear error: {e}")
        
        logger.info("Cache cleared", cleared_count=cleared_count, content_type=content_type)
        return cleared_count
    
    async def _semantic_search(
        self,
        query_key: str,
        content_type: str,
        limit: int = 5
    ) -> Optional[Tuple[str, Any]]:
        """
        Perform semantic similarity search in cached embeddings.
        
        Args:
            query_key: Key to generate embedding for
            content_type: Type of content to search
            limit: Maximum number of candidates to check
            
        Returns:
            (similar_key, cached_value) tuple or None
        """
        if not self.embedding_service or not self.redis_client:
            return None
        
        try:
            # Generate embedding for query
            query_embedding = await self.embedding_service.generate_embedding(query_key)
            
            # Get cached embeddings from Redis
            embedding_keys = await self.redis_client.keys("embedding:*")
            if not embedding_keys:
                return None
            
            candidates = []
            for embedding_key in embedding_keys[:limit]:  # Limit for performance
                try:
                    embedding_data_str = await self.redis_client.get(embedding_key)
                    if embedding_data_str:
                        embedding_data = json.loads(embedding_data_str)
                        
                        # Filter by content type if specified
                        if content_type != "generic" and embedding_data.get("content_type") != content_type:
                            continue
                        
                        candidates.append(embedding_data)
                        
                except Exception as e:
                    logger.warning(f"Error processing embedding {embedding_key}: {e}")
                    continue
            
            if not candidates:
                return None
            
            # Calculate similarities
            candidate_embeddings = [candidate["embedding"] for candidate in candidates]
            query_array = np.array([query_embedding])
            candidate_array = np.array(candidate_embeddings)
            
            similarities = cosine_similarity(query_array, candidate_array)[0]
            
            # Find best match above threshold
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            if best_similarity >= self._similarity_threshold:
                best_candidate = candidates[best_idx]
                similar_key = best_candidate["key"]
                
                # Retrieve the actual cached value
                cached_value = await self.get(similar_key, semantic_search=False)
                if cached_value:
                    self._metrics["similarity_searches"] += 1
                    logger.debug(
                        "Semantic similarity match found",
                        query_key=query_key,
                        similar_key=similar_key,
                        similarity=best_similarity
                    )
                    return (similar_key, cached_value)
            
            return None
            
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return None
    
    def _store_in_memory(self, entry: CacheEntry) -> None:
        """Store entry in memory cache with LRU eviction"""
        # Remove expired entries first
        expired_keys = [
            key for key, cached_entry in self._memory_cache.items()
            if cached_entry.is_expired()
        ]
        for key in expired_keys:
            del self._memory_cache[key]
        
        # Evict least recently used if at capacity
        if len(self._memory_cache) >= self._max_memory_entries:
            # Find LRU entry
            lru_key = min(
                self._memory_cache.keys(),
                key=lambda k: self._memory_cache[k].last_accessed
            )
            del self._memory_cache[lru_key]
        
        self._memory_cache[entry.key] = entry
    
    async def _cleanup_expired_entries(self) -> None:
        """Clean up expired entries from caches"""
        try:
            # Clean memory cache
            expired_keys = [
                key for key, entry in self._memory_cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self._memory_cache[key]
            
            if expired_keys:
                logger.debug("Cleaned expired memory cache entries", count=len(expired_keys))
            
            # Note: Redis handles TTL expiry automatically
            
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        memory_stats = {
            "entries": len(self._memory_cache),
            "max_entries": self._max_memory_entries,
            "utilization": len(self._memory_cache) / self._max_memory_entries
        }
        
        redis_stats = {}
        if self.redis_client:
            try:
                # Get Redis info
                info = await self.redis_client.info()
                redis_stats = {
                    "connected": True,
                    "used_memory": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients")
                }
            except Exception:
                redis_stats = {"connected": False}
        
        return {
            "metrics": self._metrics.copy(),
            "memory_cache": memory_stats,
            "redis_cache": redis_stats,
            "semantic_search": {
                "enabled": self.embedding_service is not None,
                "threshold": self._similarity_threshold
            },
            "configuration": {
                "semantic_cache_ttl": self.settings.SEMANTIC_CACHE_TTL,
                "cache_enabled": self.settings.SEMANTIC_CACHE_ENABLED
            }
        }
    
    async def check_health(self) -> Dict[str, Any]:
        """Check cache manager health"""
        try:
            health_status = {
                "status": "healthy",
                "memory_cache": "operational",
                "timestamp": datetime.now().isoformat()
            }
            
            # Test memory cache
            test_key = "health_check_test"
            await self.set(test_key, "test_value", ttl=10, generate_embedding=False)
            test_value = await self.get(test_key, semantic_search=False)
            
            if test_value != "test_value":
                health_status["status"] = "degraded"
                health_status["memory_cache"] = "failed"
            
            # Clean up test entry
            await self.delete(test_key)
            
            # Test Redis connection
            if self.redis_client:
                try:
                    await self.redis_client.ping()
                    health_status["redis_cache"] = "operational"
                except Exception:
                    health_status["redis_cache"] = "failed"
                    if health_status["status"] == "healthy":
                        health_status["status"] = "degraded"
            else:
                health_status["redis_cache"] = "not_configured"
            
            # Test semantic search
            if self.embedding_service:
                health_status["semantic_search"] = "operational"
            else:
                health_status["semantic_search"] = "not_configured"
            
            return health_status
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def __del__(self):
        """Cleanup background tasks"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()