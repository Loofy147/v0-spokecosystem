"""
Performance optimization patterns and utilities.
"""
import time
import functools
from typing import Any, Callable, Dict, Optional, Tuple
from collections import OrderedDict
import threading
from utils.logging_config import get_logger

logger = get_logger(__name__)


class LRUCache:
    """Thread-safe Least Recently Used cache."""
    
    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.logger = get_logger(self.__class__.__name__)
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                self.hits += 1
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            else:
                self.misses += 1
                return None
    
    def put(self, key: Any, value: Any):
        """Put value in cache."""
        with self.lock:
            if key in self.cache:
                # Update existing key
                self.cache.move_to_end(key)
            else:
                # Add new key
                if len(self.cache) >= self.capacity:
                    # Remove least recently used
                    oldest = next(iter(self.cache))
                    del self.cache[oldest]
            
            self.cache[key] = value
    
    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache),
            'capacity': self.capacity
        }


def memoize(maxsize: int = 128):
    """
    Memoization decorator with LRU cache.
    
    Args:
        maxsize: Maximum cache size
    """
    def decorator(func: Callable) -> Callable:
        cache = LRUCache(capacity=maxsize)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = (args, tuple(sorted(kwargs.items())))
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result
        
        # Attach cache for inspection
        wrapper.cache = cache
        return wrapper
    
    return decorator


class LazyProperty:
    """Lazy evaluation property descriptor."""
    
    def __init__(self, func: Callable):
        self.func = func
        self.name = func.__name__
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        
        # Compute value and cache it
        value = self.func(obj)
        setattr(obj, self.name, value)
        return value


def lazy_property(func: Callable) -> LazyProperty:
    """Decorator for lazy property evaluation."""
    return LazyProperty(func)


class BatchProcessor:
    """Batch processing for improved throughput."""
    
    def __init__(
        self,
        process_func: Callable,
        batch_size: int = 32,
        timeout: float = 1.0
    ):
        """
        Initialize batch processor.
        
        Args:
            process_func: Function to process batches
            batch_size: Maximum batch size
            timeout: Maximum wait time before processing partial batch
        """
        self.process_func = process_func
        self.batch_size = batch_size
        self.timeout = timeout
        self.buffer = []
        self.lock = threading.Lock()
        self.last_process_time = time.time()
        self.logger = get_logger(self.__class__.__name__)
    
    def add(self, item: Any) -> Optional[Any]:
        """Add item to batch."""
        with self.lock:
            self.buffer.append(item)
            
            # Process if batch is full or timeout exceeded
            if len(self.buffer) >= self.batch_size or \
               (time.time() - self.last_process_time) > self.timeout:
                return self._process_batch()
        
        return None
    
    def _process_batch(self) -> Any:
        """Process current batch."""
        if not self.buffer:
            return None
        
        batch = self.buffer
        self.buffer = []
        self.last_process_time = time.time()
        
        self.logger.debug(f"Processing batch of {len(batch)} items")
        return self.process_func(batch)
    
    def flush(self) -> Optional[Any]:
        """Process remaining items in buffer."""
        with self.lock:
            return self._process_batch()


def profile_time(func: Callable) -> Callable:
    """
    Decorator to profile function execution time.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.debug(
            f"{func.__name__} executed in {execution_time:.4f} seconds"
        )
        
        return result
    
    return wrapper


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, list] = {}
        self.lock = threading.Lock()
        self.logger = get_logger(self.__class__.__name__)
    
    def record(self, metric_name: str, value: float):
        """Record a performance metric."""
        with self.lock:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            self.metrics[metric_name].append(value)
    
    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        with self.lock:
            if metric_name not in self.metrics:
                return {}
            
            values = self.metrics[metric_name]
            return {
                'count': len(values),
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'total': sum(values)
            }
    
    def clear(self, metric_name: Optional[str] = None):
        """Clear metrics."""
        with self.lock:
            if metric_name:
                if metric_name in self.metrics:
                    del self.metrics[metric_name]
            else:
                self.metrics.clear()
    
    def monitor(self, metric_name: str):
        """Decorator to monitor function execution time."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                self.record(metric_name, execution_time)
                return result
            
            return wrapper
        return decorator


class ObjectPool:
    """Object pool for expensive-to-create objects."""
    
    def __init__(
        self,
        factory: Callable[[], Any],
        reset: Optional[Callable[[Any], None]] = None,
        max_size: int = 10
    ):
        """
        Initialize object pool.
        
        Args:
            factory: Function to create new objects
            reset: Optional function to reset objects before reuse
            max_size: Maximum pool size
        """
        self.factory = factory
        self.reset = reset
        self.max_size = max_size
        self.pool = []
        self.lock = threading.Lock()
        self.logger = get_logger(self.__class__.__name__)
    
    def acquire(self) -> Any:
        """Acquire an object from the pool."""
        with self.lock:
            if self.pool:
                obj = self.pool.pop()
                if self.reset:
                    self.reset(obj)
                self.logger.debug("Acquired object from pool")
                return obj
            else:
                obj = self.factory()
                self.logger.debug("Created new object")
                return obj
    
    def release(self, obj: Any):
        """Release an object back to the pool."""
        with self.lock:
            if len(self.pool) < self.max_size:
                self.pool.append(obj)
                self.logger.debug("Released object to pool")


class ComputationCache:
    """Cache for expensive computations with TTL."""
    
    def __init__(self, ttl: float = 3600):
        """
        Initialize computation cache.
        
        Args:
            ttl: Time to live in seconds
        """
        self.ttl = ttl
        self.cache: Dict[Any, Tuple[Any, float]] = {}
        self.lock = threading.Lock()
        self.logger = get_logger(self.__class__.__name__)
    
    def get(self, key: Any) -> Optional[Any]:
        """Get cached value if not expired."""
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    return value
                else:
                    # Expired, remove from cache
                    del self.cache[key]
        return None
    
    def put(self, key: Any, value: Any):
        """Cache a value with current timestamp."""
        with self.lock:
            self.cache[key] = (value, time.time())
    
    def clear_expired(self):
        """Remove expired entries."""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, (_, timestamp) in self.cache.items()
                if current_time - timestamp >= self.ttl
            ]
            for key in expired_keys:
                del self.cache[key]
            
            if expired_keys:
                self.logger.debug(f"Cleared {len(expired_keys)} expired entries")


def vectorize_operation(func: Callable) -> Callable:
    """
    Decorator to encourage vectorized operations.
    Logs warning if called with small batches.
    """
    @functools.wraps(func)
    def wrapper(data, *args, **kwargs):
        if hasattr(data, '__len__') and len(data) < 10:
            logger.warning(
                f"{func.__name__} called with small batch size ({len(data)}). "
                "Consider batching for better performance."
            )
        return func(data, *args, **kwargs)
    
    return wrapper
