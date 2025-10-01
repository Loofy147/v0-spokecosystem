"""
Singleton pattern for single-instance classes.
"""
from typing import Any, Dict
import threading
from utils.logging_config import get_logger

logger = get_logger(__name__)


class SingletonMeta(type):
    """
    Thread-safe Singleton metaclass.
    """
    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        """Create or return existing instance."""
        if cls not in cls._instances:
            with cls._lock:
                # Double-checked locking
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
                    logger.debug(f"Created singleton instance of {cls.__name__}")
        
        return cls._instances[cls]


class Singleton(metaclass=SingletonMeta):
    """Base class for singleton objects."""
    pass


class ConfigurationManager(Singleton):
    """Singleton configuration manager."""
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._config: Dict[str, Any] = {}
            self._initialized = True
            self.logger = get_logger(self.__class__.__name__)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self._config[key] = value
        self.logger.debug(f"Set config: {key} = {value}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def update(self, config: Dict[str, Any]):
        """Update multiple configuration values."""
        self._config.update(config)
        self.logger.debug(f"Updated config with {len(config)} values")
    
    def clear(self):
        """Clear all configuration."""
        self._config.clear()
        self.logger.debug("Cleared all configuration")
