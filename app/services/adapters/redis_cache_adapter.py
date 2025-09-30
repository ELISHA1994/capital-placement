"""
Redis Cache Service - Redis-based cache implementation for production environments
"""

import asyncio
import json
import pickle
import time
from typing import Dict, Any, Optional, Union
import structlog

from app.core.interfaces import ICacheService

logger = structlog.get_logger(__name__)

try:
    import redis.asyncio as aioredis
    from redis.exceptions import RedisError, ConnectionError, TimeoutError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available. Install with: pip install redis[hiredis]")


class RedisCacheService(ICacheService):
    """Redis-based cache service for production environments"""
    
    def __init__(self, redis_client: Optional[Any] = None):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis[hiredis]")
        
        self.redis = redis_client
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
        self._connection_pool = None
        
    @classmethod
    async def create(cls, redis_url: str = "redis://localhost:6379/0") -> "RedisCacheService":
        """Create Redis cache service with connection pool"""
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis[hiredis]")
        
        try:
            # Create connection pool for better performance
            redis_client = aioredis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=False,  # We handle encoding manually
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await redis_client.ping()
            
            service = cls(redis_client)
            service._connection_pool = redis_client.connection_pool
            
            logger.info("Redis cache service initialized", redis_url=redis_url)
            return service
            
        except Exception as e:
            logger.error("Failed to initialize Redis cache", error=str(e))
            raise
    
    async def check_health(self) -> Dict[str, Any]:
        """Check Redis service health"""
        try:
            # Test connection with ping
            await self.redis.ping()
            
            # Get Redis info
            info = await self.redis.info()
            
            return {
                "status": "healthy",
                "service": "RedisCacheService",
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory": info.get("used_memory"),
                "stats": self._stats.copy()
            }
            
        except Exception as e:
            logger.error("Redis health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "service": "RedisCacheService",
                "error": str(e),
                "stats": self._stats.copy()
            }
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        try:
            # Get value from Redis
            raw_value = await self.redis.get(key)
            
            if raw_value is None:
                self._stats["misses"] += 1
                logger.debug("Cache miss", key=key)
                return None
            
            # Deserialize value
            try:
                # Try JSON first (for simple types)
                value = json.loads(raw_value)
            except (json.JSONDecodeError, TypeError):
                # Fall back to pickle for complex objects
                value = pickle.loads(raw_value)
            
            self._stats["hits"] += 1
            logger.debug("Cache hit", key=key)
            return value
            
        except (RedisError, ConnectionError, TimeoutError) as e:
            self._stats["errors"] += 1
            logger.error("Redis get failed", key=key, error=str(e))
            return None
        except Exception as e:
            self._stats["errors"] += 1
            logger.error("Cache get failed", key=key, error=str(e))
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in Redis cache with TTL"""
        try:
            # Serialize value
            try:
                # Try JSON first (for better performance and readability)
                serialized_value = json.dumps(value, default=str)
            except (TypeError, ValueError):
                # Fall back to pickle for complex objects
                serialized_value = pickle.dumps(value)
            
            # Set with TTL
            result = await self.redis.setex(key, ttl, serialized_value)
            
            if result:
                self._stats["sets"] += 1
                logger.debug("Cache set", key=key, ttl=ttl)
                return True
            
            return False
            
        except (RedisError, ConnectionError, TimeoutError) as e:
            self._stats["errors"] += 1
            logger.error("Redis set failed", key=key, error=str(e))
            return False
        except Exception as e:
            self._stats["errors"] += 1
            logger.error("Cache set failed", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache"""
        try:
            result = await self.redis.delete(key)
            
            if result > 0:
                self._stats["deletes"] += 1
                logger.debug("Cache delete", key=key)
                return True
            
            return False
            
        except (RedisError, ConnectionError, TimeoutError) as e:
            self._stats["errors"] += 1
            logger.error("Redis delete failed", key=key, error=str(e))
            return False
        except Exception as e:
            self._stats["errors"] += 1
            logger.error("Cache delete failed", key=key, error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        try:
            result = await self.redis.exists(key)
            return result > 0
            
        except (RedisError, ConnectionError, TimeoutError) as e:
            self._stats["errors"] += 1
            logger.error("Redis exists check failed", key=key, error=str(e))
            return False
        except Exception as e:
            self._stats["errors"] += 1
            logger.error("Cache exists check failed", key=key, error=str(e))
            return False
    
    async def clear(self, pattern: str = "*") -> int:
        """Clear keys matching pattern"""
        try:
            if pattern == "*":
                # Clear entire cache (use with caution!)
                result = await self.redis.flushdb()
                logger.warning("Redis cache cleared completely")
                return 1 if result else 0
            
            # Find keys matching pattern
            keys = await self.redis.keys(pattern)
            
            if not keys:
                return 0
            
            # Delete matching keys in batches
            batch_size = 100
            deleted_count = 0
            
            for i in range(0, len(keys), batch_size):
                batch = keys[i:i + batch_size]
                deleted_count += await self.redis.delete(*batch)
            
            logger.info("Cache pattern clear", pattern=pattern, count=deleted_count)
            return deleted_count
            
        except (RedisError, ConnectionError, TimeoutError) as e:
            self._stats["errors"] += 1
            logger.error("Redis clear failed", pattern=pattern, error=str(e))
            return 0
        except Exception as e:
            self._stats["errors"] += 1
            logger.error("Cache clear failed", pattern=pattern, error=str(e))
            return 0
    
    async def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> Optional[int]:
        """Increment a counter (Redis-specific feature)"""
        try:
            # Increment the key
            result = await self.redis.incr(key, amount)
            
            # Set TTL if specified and this is a new key
            if ttl is not None and result == amount:
                await self.redis.expire(key, ttl)
            
            return result
            
        except (RedisError, ConnectionError, TimeoutError) as e:
            self._stats["errors"] += 1
            logger.error("Redis increment failed", key=key, error=str(e))
            return None
        except Exception as e:
            self._stats["errors"] += 1
            logger.error("Cache increment failed", key=key, error=str(e))
            return None
    
    async def set_multiple(self, mapping: Dict[str, Any], ttl: int = 3600) -> bool:
        """Set multiple key-value pairs atomically"""
        try:
            # Prepare pipeline for atomic operations
            pipe = self.redis.pipeline()
            
            for key, value in mapping.items():
                # Serialize value
                try:
                    serialized_value = json.dumps(value, default=str)
                except (TypeError, ValueError):
                    serialized_value = pickle.dumps(value)
                
                pipe.setex(key, ttl, serialized_value)
            
            # Execute pipeline
            results = await pipe.execute()
            
            # Check if all operations succeeded
            success = all(results)
            
            if success:
                self._stats["sets"] += len(mapping)
                logger.debug("Cache set multiple", count=len(mapping), ttl=ttl)
            
            return success
            
        except (RedisError, ConnectionError, TimeoutError) as e:
            self._stats["errors"] += 1
            logger.error("Redis set multiple failed", error=str(e))
            return False
        except Exception as e:
            self._stats["errors"] += 1
            logger.error("Cache set multiple failed", error=str(e))
            return False
    
    async def get_multiple(self, keys: list[str]) -> Dict[str, Any]:
        """Get multiple values atomically"""
        try:
            # Get multiple values
            raw_values = await self.redis.mget(keys)
            
            result = {}
            for key, raw_value in zip(keys, raw_values):
                if raw_value is not None:
                    try:
                        # Try JSON first
                        value = json.loads(raw_value)
                    except (json.JSONDecodeError, TypeError):
                        # Fall back to pickle
                        value = pickle.loads(raw_value)
                    
                    result[key] = value
                    self._stats["hits"] += 1
                else:
                    self._stats["misses"] += 1
            
            logger.debug("Cache get multiple", requested=len(keys), found=len(result))
            return result
            
        except (RedisError, ConnectionError, TimeoutError) as e:
            self._stats["errors"] += 1
            logger.error("Redis get multiple failed", error=str(e))
            return {}
        except Exception as e:
            self._stats["errors"] += 1
            logger.error("Cache get multiple failed", error=str(e))
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            **self._stats,
            "hit_rate": self._stats["hits"] / max(1, self._stats["hits"] + self._stats["misses"]),
            "error_rate": self._stats["errors"] / max(1, sum(self._stats.values()))
        }
    
    async def close(self):
        """Close Redis connection and cleanup resources"""
        try:
            if self.redis:
                await self.redis.close()
            
            if self._connection_pool:
                await self._connection_pool.disconnect()
            
            logger.info("Redis cache closed")
            
        except Exception as e:
            logger.error("Error closing Redis cache", error=str(e))


# Backwards compatibility alias
RedisCacheAdapter = RedisCacheService