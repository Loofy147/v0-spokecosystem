"""Caching infrastructure with Redis and LRU fallback."""
from __future__ import annotations
import pickle
import time
from typing import Any, Optional
from collections import OrderedDict
import warnings

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    warnings.warn("Redis not available. Install with: pip install redis")


class LRUCache:
    """Simple LRU (Least Recently Used) cache implementation."""
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        else:
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # Remove least recently used item
                self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def delete(self, key: str):
        """Delete key from cache."""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """Clear all cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache),
            'capacity': self.capacity
        }


class RedisCache:
    """Redis-based cache with automatic serialization."""
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = 'spokecosystem:'
    ):
        if not REDIS_AVAILABLE:
            raise RuntimeError("Redis is required. Install with: pip install redis")
        
        self.prefix = prefix
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False
        )
        
        # Test connection
        try:
            self.client.ping()
        except redis.ConnectionError as e:
            raise RuntimeError(f"Failed to connect to Redis: {e}")
    
    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        full_key = self._make_key(key)
        value = self.client.get(full_key)
        
        if value is None:
            return None
        
        try:
            return pickle.loads(value)
        except Exception as e:
            warnings.warn(f"Failed to deserialize cached value: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None for no expiration)
        """
        full_key = self._make_key(key)
        serialized = pickle.dumps(value)
        
        if ttl is not None:
            self.client.setex(full_key, ttl, serialized)
        else:
            self.client.set(full_key, serialized)
    
    def delete(self, key: str):
        """Delete key from cache."""
        full_key = self._make_key(key)
        self.client.delete(full_key)
    
    def clear(self):
        """Clear all keys with the prefix."""
        pattern = f"{self.prefix}*"
        keys = self.client.keys(pattern)
        if keys:
            self.client.delete(*keys)
    
    def stats(self) -> dict:
        """Get cache statistics from Redis INFO."""
        info = self.client.info('stats')
        return {
            'hits': info.get('keyspace_hits', 0),
            'misses': info.get('keyspace_misses', 0),
            'hit_rate': info.get('keyspace_hits', 0) / max(1, info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0))
        }


class CacheManager:
    """
    Unified cache manager with Redis primary and LRU fallback.
    Automatically falls back to LRU if Redis is unavailable.
    """
    def __init__(
        self,
        use_redis: bool = True,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        lru_capacity: int = 1000,
        **redis_kwargs
    ):
        self.backend = None
        
        if use_redis and REDIS_AVAILABLE:
            try:
                self.backend = RedisCache(host=redis_host, port=redis_port, **redis_kwargs)
                self.backend_type = 'redis'
            except Exception as e:
                warnings.warn(f"Failed to initialize Redis, falling back to LRU: {e}")
                self.backend = LRUCache(capacity=lru_capacity)
                self.backend_type = 'lru'
        else:
            self.backend = LRUCache(capacity=lru_capacity)
            self.backend_type = 'lru'
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self.backend.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        self.backend.set(key, value, ttl)
    
    def delete(self, key: str):
        """Delete key from cache."""
        self.backend.delete(key)
    
    def clear(self):
        """Clear all cache."""
        self.backend.clear()
    
    def stats(self) -> dict:
        """Get cache statistics."""
        stats = self.backend.stats()
        stats['backend'] = self.backend_type
        return stats
