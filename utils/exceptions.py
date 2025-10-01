"""
Custom exception hierarchy for SpokeEcosystem.
"""
from typing import Any, Dict, Optional


class SpokeEcosystemError(Exception):
    """Base exception for all SpokeEcosystem errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details
        }


# Core Engine Exceptions
class CoreEngineError(SpokeEcosystemError):
    """Base exception for core engine errors."""
    pass


class GradientError(CoreEngineError):
    """Raised when gradient computation fails."""
    pass


class ShapeError(CoreEngineError):
    """Raised when tensor shapes are incompatible."""
    pass


class NumericalError(CoreEngineError):
    """Raised when numerical instability is detected."""
    pass


# AutoML Exceptions
class AutoMLError(SpokeEcosystemError):
    """Base exception for AutoML errors."""
    pass


class ArchitectureError(AutoMLError):
    """Raised when architecture is invalid."""
    pass


class FitnessEvaluationError(AutoMLError):
    """Raised when fitness evaluation fails."""
    pass


class OptimizationError(AutoMLError):
    """Raised when optimization fails."""
    pass


# Infrastructure Exceptions
class InfrastructureError(SpokeEcosystemError):
    """Base exception for infrastructure errors."""
    pass


class CacheError(InfrastructureError):
    """Raised when cache operations fail."""
    pass


class VectorSearchError(InfrastructureError):
    """Raised when vector search operations fail."""
    pass


class ObservabilityError(InfrastructureError):
    """Raised when observability operations fail."""
    pass


# Production Exceptions
class ProductionError(SpokeEcosystemError):
    """Base exception for production errors."""
    pass


class ModelRegistryError(ProductionError):
    """Raised when model registry operations fail."""
    pass


class DeploymentError(ProductionError):
    """Raised when deployment operations fail."""
    pass


class ServingError(ProductionError):
    """Raised when model serving fails."""
    pass


# Data Exceptions
class DataError(SpokeEcosystemError):
    """Base exception for data errors."""
    pass


class ValidationError(DataError):
    """Raised when data validation fails."""
    pass


class LoadingError(DataError):
    """Raised when data loading fails."""
    pass


# Configuration Exceptions
class ConfigurationError(SpokeEcosystemError):
    """Raised when configuration is invalid."""
    pass


# Resource Exceptions
class ResourceError(SpokeEcosystemError):
    """Base exception for resource errors."""
    pass


class MemoryError(ResourceError):
    """Raised when memory limits are exceeded."""
    pass


class TimeoutError(ResourceError):
    """Raised when operations timeout."""
    pass
