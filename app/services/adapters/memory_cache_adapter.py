"""
Memory Cache Service - In-memory cache implementation for local development
"""

import asyncio
import time
import json
import pickle
from typing import Dict, Any, Optional
from dataclasses import dataclass
import structlog

from app.core.interfaces import ICacheService

logger = structlog.get_logger(__name__)


@dataclass
class CacheItem:
    """Cache item with expiration"""
    value: Any
    expires_at: float
    created_at: float


class MemoryCacheService(ICacheService):
    """In-memory cache service for local development"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: Dict[str, CacheItem] = {}
        self._lock = asyncio.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0
        }
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_expired())
    
    async def check_health(self) -> Dict[str, Any]:
        """Check service health"""
        return {
            "status": "healthy",
            "service": "MemoryCacheService",
            "cache_size": len(self._cache),
            "max_size": self.max_size,
            "stats": self._stats.copy()
        }
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self._lock:
            item = self._cache.get(key)
            
            if item is None:
                self._stats["misses"] += 1
                logger.debug("Cache miss", key=key)
                return None
            
            # Check expiration
            if time.time() > item.expires_at:
                del self._cache[key]
                self._stats["misses"] += 1
                logger.debug("Cache expired", key=key)
                return None
            
            self._stats["hits"] += 1
            logger.debug("Cache hit", key=key)
            return item.value
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL"""
        async with self._lock:
            try:
                # Check if we need to evict items
                if len(self._cache) >= self.max_size and key not in self._cache:
                    await self._evict_lru()
                
                expires_at = time.time() + ttl
                self._cache[key] = CacheItem(
                    value=value,
                    expires_at=expires_at,
                    created_at=time.time()
                )
                
                self._stats["sets"] += 1
                logger.debug("Cache set", key=key, ttl=ttl)
                return True
                
            except Exception as e:
                logger.error("Cache set failed", key=key, error=str(e))
                return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats["deletes"] += 1
                logger.debug("Cache delete", key=key)
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        async with self._lock:
            item = self._cache.get(key)
            if item is None:
                return False
            
            # Check expiration
            if time.time() > item.expires_at:
                del self._cache[key]
                return False
            
            return True
    
    async def clear(self, pattern: str = "*") -> int:
        """Clear keys matching pattern"""
        async with self._lock:
            if pattern == "*":
                count = len(self._cache)
                self._cache.clear()
                logger.info("Cache cleared completely", count=count)
                return count
            
            # Simple pattern matching (only supports prefix with *)
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                keys_to_delete = [key for key in self._cache.keys() if key.startswith(prefix)]
            else:
                # Exact match
                keys_to_delete = [key for key in self._cache.keys() if key == pattern]
            
            for key in keys_to_delete:
                del self._cache[key]
            
            logger.info("Cache pattern clear", pattern=pattern, count=len(keys_to_delete))
            return len(keys_to_delete)
    
    async def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self._cache:
            return
        
        # Find oldest item
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        del self._cache[oldest_key]
        self._stats["evictions"] += 1
        
        logger.debug("Cache LRU eviction", key=oldest_key)
    
    async def _cleanup_expired(self) -> None:
        """Periodic cleanup of expired items"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                async with self._lock:
                    current_time = time.time()
                    expired_keys = [
                        key for key, item in self._cache.items()
                        if current_time > item.expires_at
                    ]
                    
                    for key in expired_keys:
                        del self._cache[key]
                    
                    if expired_keys:
                        logger.debug("Cache cleanup", expired_count=len(expired_keys))
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cache cleanup error", error=str(e))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            **self._stats,
            "cache_size": len(self._cache),
            "max_size": self.max_size,
            "hit_rate": self._stats["hits"] / max(1, self._stats["hits"] + self._stats["misses"])
        }
    
    async def close(self):
        """Close cache and cleanup resources"""
        if hasattr(self, '_cleanup_task'):
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        async with self._lock:
            self._cache.clear()
        
        logger.info("Memory cache closed")