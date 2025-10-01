"""
Resource management and cleanup utilities.
"""
import gc
import psutil
import threading
from typing import Any, Callable, Dict, List, Optional
from contextlib import contextmanager
from utils.logging_config import get_logger
from utils.exceptions import ResourceError

logger = get_logger(__name__)


class ResourceTracker:
    """Tracks resource usage (memory, CPU, etc.)."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.logger = get_logger(self.__class__.__name__)
        self._snapshots: List[Dict[str, Any]] = []
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        mem_info = self.process.memory_info()
        return {
            'rss': mem_info.rss / 1024 / 1024,  # Resident Set Size
            'vms': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': self.process.memory_percent()
        }
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent(interval=0.1)
    
    def get_thread_count(self) -> int:
        """Get number of active threads."""
        return threading.active_count()
    
    def snapshot(self, label: str = ""):
        """Take a snapshot of current resource usage."""
        snapshot = {
            'label': label,
            'memory': self.get_memory_usage(),
            'cpu': self.get_cpu_usage(),
            'threads': self.get_thread_count()
        }
        self._snapshots.append(snapshot)
        self.logger.debug(f"Resource snapshot '{label}': {snapshot}")
        return snapshot
    
    def get_snapshots(self) -> List[Dict[str, Any]]:
        """Get all resource snapshots."""
        return self._snapshots
    
    def clear_snapshots(self):
        """Clear all snapshots."""
        self._snapshots.clear()
    
    def check_memory_limit(self, limit_mb: float) -> bool:
        """Check if memory usage exceeds limit."""
        current = self.get_memory_usage()['rss']
        if current > limit_mb:
            self.logger.warning(
                f"Memory usage ({current:.2f} MB) exceeds limit ({limit_mb} MB)"
            )
            return False
        return True


class ResourcePool:
    """Pool for managing reusable resources."""
    
    def __init__(
        self,
        factory: Callable[[], Any],
        max_size: int = 10,
        cleanup: Optional[Callable[[Any], None]] = None
    ):
        """
        Initialize resource pool.
        
        Args:
            factory: Function to create new resources
            max_size: Maximum pool size
            cleanup: Optional cleanup function for resources
        """
        self.factory = factory
        self.max_size = max_size
        self.cleanup = cleanup
        self._pool: List[Any] = []
        self._in_use: List[Any] = []
        self._lock = threading.Lock()
        self.logger = get_logger(self.__class__.__name__)
    
    def acquire(self) -> Any:
        """Acquire a resource from the pool."""
        with self._lock:
            if self._pool:
                resource = self._pool.pop()
                self.logger.debug("Acquired resource from pool")
            else:
                resource = self.factory()
                self.logger.debug("Created new resource")
            
            self._in_use.append(resource)
            return resource
    
    def release(self, resource: Any):
        """Release a resource back to the pool."""
        with self._lock:
            if resource in self._in_use:
                self._in_use.remove(resource)
                
                if len(self._pool) < self.max_size:
                    self._pool.append(resource)
                    self.logger.debug("Released resource to pool")
                else:
                    if self.cleanup:
                        self.cleanup(resource)
                    self.logger.debug("Cleaned up excess resource")
    
    def clear(self):
        """Clear all resources from pool."""
        with self._lock:
            if self.cleanup:
                for resource in self._pool:
                    self.cleanup(resource)
            self._pool.clear()
            self.logger.info("Cleared resource pool")
    
    @contextmanager
    def get_resource(self):
        """Context manager for acquiring and releasing resources."""
        resource = self.acquire()
        try:
            yield resource
        finally:
            self.release(resource)


class MemoryManager:
    """Manages memory allocation and cleanup."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self._tracked_objects: Dict[str, Any] = {}
    
    def track(self, name: str, obj: Any):
        """Track an object for memory management."""
        self._tracked_objects[name] = obj
        self.logger.debug(f"Tracking object: {name}")
    
    def untrack(self, name: str):
        """Stop tracking an object."""
        if name in self._tracked_objects:
            del self._tracked_objects[name]
            self.logger.debug(f"Untracked object: {name}")
    
    def cleanup(self, name: Optional[str] = None):
        """Clean up tracked objects."""
        if name:
            if name in self._tracked_objects:
                del self._tracked_objects[name]
                self.logger.debug(f"Cleaned up object: {name}")
        else:
            self._tracked_objects.clear()
            self.logger.info("Cleaned up all tracked objects")
        
        # Force garbage collection
        gc.collect()
    
    def get_tracked_count(self) -> int:
        """Get number of tracked objects."""
        return len(self._tracked_objects)
    
    @contextmanager
    def managed_object(self, name: str, obj: Any):
        """Context manager for automatic object cleanup."""
        self.track(name, obj)
        try:
            yield obj
        finally:
            self.cleanup(name)


class ResourceLimiter:
    """Enforces resource limits."""
    
    def __init__(
        self,
        max_memory_mb: Optional[float] = None,
        max_threads: Optional[int] = None
    ):
        self.max_memory_mb = max_memory_mb
        self.max_threads = max_threads
        self.tracker = ResourceTracker()
        self.logger = get_logger(self.__class__.__name__)
    
    def check_limits(self):
        """Check if resource limits are exceeded."""
        violations = []
        
        if self.max_memory_mb:
            memory = self.tracker.get_memory_usage()['rss']
            if memory > self.max_memory_mb:
                violations.append(
                    f"Memory limit exceeded: {memory:.2f} MB > {self.max_memory_mb} MB"
                )
        
        if self.max_threads:
            threads = self.tracker.get_thread_count()
            if threads > self.max_threads:
                violations.append(
                    f"Thread limit exceeded: {threads} > {self.max_threads}"
                )
        
        if violations:
            error_msg = "; ".join(violations)
            self.logger.error(error_msg)
            raise ResourceError(error_msg, details={'violations': violations})
    
    @contextmanager
    def enforce_limits(self):
        """Context manager that enforces resource limits."""
        self.check_limits()
        try:
            yield
        finally:
            self.check_limits()


@contextmanager
def cleanup_on_error(*cleanup_funcs: Callable):
    """
    Context manager that runs cleanup functions on error.
    
    Args:
        *cleanup_funcs: Functions to call on error
    """
    try:
        yield
    except Exception as e:
        logger.error(f"Error occurred, running cleanup: {e}")
        for func in cleanup_funcs:
            try:
                func()
            except Exception as cleanup_error:
                logger.error(f"Cleanup error: {cleanup_error}", exc_info=True)
        raise


@contextmanager
def temporary_resource(
    acquire: Callable[[], Any],
    release: Callable[[Any], None]
):
    """
    Context manager for temporary resources.
    
    Args:
        acquire: Function to acquire resource
        release: Function to release resource
    """
    resource = acquire()
    try:
        yield resource
    finally:
        try:
            release(resource)
        except Exception as e:
            logger.error(f"Error releasing resource: {e}", exc_info=True)


class CleanupRegistry:
    """Registry for cleanup callbacks."""
    
    def __init__(self):
        self._callbacks: List[Callable] = []
        self.logger = get_logger(self.__class__.__name__)
    
    def register(self, callback: Callable):
        """Register a cleanup callback."""
        self._callbacks.append(callback)
        self.logger.debug(f"Registered cleanup callback: {callback.__name__}")
    
    def cleanup_all(self):
        """Run all cleanup callbacks."""
        self.logger.info(f"Running {len(self._callbacks)} cleanup callbacks")
        errors = []
        
        for callback in self._callbacks:
            try:
                callback()
            except Exception as e:
                errors.append((callback.__name__, e))
                self.logger.error(
                    f"Error in cleanup callback {callback.__name__}: {e}",
                    exc_info=True
                )
        
        self._callbacks.clear()
        
        if errors:
            self.logger.warning(f"{len(errors)} cleanup callbacks failed")
    
    def clear(self):
        """Clear all callbacks without running them."""
        self._callbacks.clear()
        self.logger.debug("Cleared all cleanup callbacks")


# Global cleanup registry
_global_cleanup_registry = CleanupRegistry()


def register_cleanup(callback: Callable):
    """Register a global cleanup callback."""
    _global_cleanup_registry.register(callback)


def cleanup_all():
    """Run all global cleanup callbacks."""
    _global_cleanup_registry.cleanup_all()
