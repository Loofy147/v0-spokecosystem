"""
Factory pattern implementations for creating objects.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, Callable
from utils.logging_config import get_logger
from utils.exceptions import ConfigurationError

logger = get_logger(__name__)


class Factory(ABC):
    """Abstract factory base class."""
    
    _registry: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, name: str, implementation: Type):
        """Register an implementation with a name."""
        cls._registry[name] = implementation
        logger.debug(f"Registered {name} in {cls.__name__}")
    
    @classmethod
    def create(cls, name: str, **kwargs) -> Any:
        """Create an instance by name."""
        if name not in cls._registry:
            raise ConfigurationError(
                f"Unknown type: {name}",
                details={'available_types': list(cls._registry.keys())}
            )
        
        implementation = cls._registry[name]
        return implementation(**kwargs)
    
    @classmethod
    def list_available(cls) -> list:
        """List all registered implementations."""
        return list(cls._registry.keys())


class OptimizerFactory(Factory):
    """Factory for creating optimizers."""
    _registry: Dict[str, Type] = {}


class ActivationFactory(Factory):
    """Factory for creating activation functions."""
    _registry: Dict[str, Type] = {}


class LossFactory(Factory):
    """Factory for creating loss functions."""
    _registry: Dict[str, Type] = {}


class ModelFactory(Factory):
    """Factory for creating models."""
    _registry: Dict[str, Type] = {}


def register_optimizer(name: str):
    """Decorator for registering optimizers."""
    def decorator(cls):
        OptimizerFactory.register(name, cls)
        return cls
    return decorator


def register_activation(name: str):
    """Decorator for registering activation functions."""
    def decorator(cls):
        ActivationFactory.register(name, cls)
        return cls
    return decorator


def register_loss(name: str):
    """Decorator for registering loss functions."""
    def decorator(cls):
        LossFactory.register(name, cls)
        return cls
    return decorator


def register_model(name: str):
    """Decorator for registering models."""
    def decorator(cls):
        ModelFactory.register(name, cls)
        return cls
    return decorator
