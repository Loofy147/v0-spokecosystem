"""
Builder pattern for complex object construction.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from utils.logging_config import get_logger

logger = get_logger(__name__)


class Builder(ABC):
    """Abstract builder base class."""
    
    @abstractmethod
    def reset(self):
        """Reset the builder."""
        pass
    
    @abstractmethod
    def build(self) -> Any:
        """Build and return the final product."""
        pass


class ModelBuilder(Builder):
    """Builder for constructing complex models."""
    
    def __init__(self):
        self.reset()
        self.logger = get_logger(self.__class__.__name__)
    
    def reset(self):
        """Reset the builder state."""
        self._layers = []
        self._config = {}
        self._optimizer = None
        self._loss = None
    
    def add_layer(self, layer_type: str, **kwargs):
        """Add a layer to the model."""
        self._layers.append({'type': layer_type, 'config': kwargs})
        self.logger.debug(f"Added layer: {layer_type}")
        return self
    
    def set_optimizer(self, optimizer_type: str, **kwargs):
        """Set the optimizer."""
        self._optimizer = {'type': optimizer_type, 'config': kwargs}
        self.logger.debug(f"Set optimizer: {optimizer_type}")
        return self
    
    def set_loss(self, loss_type: str, **kwargs):
        """Set the loss function."""
        self._loss = {'type': loss_type, 'config': kwargs}
        self.logger.debug(f"Set loss: {loss_type}")
        return self
    
    def set_config(self, **kwargs):
        """Set model configuration."""
        self._config.update(kwargs)
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build and return the model specification."""
        model_spec = {
            'layers': self._layers,
            'optimizer': self._optimizer,
            'loss': self._loss,
            'config': self._config
        }
        self.logger.info(f"Built model with {len(self._layers)} layers")
        return model_spec


class PipelineBuilder(Builder):
    """Builder for constructing data pipelines."""
    
    def __init__(self):
        self.reset()
        self.logger = get_logger(self.__class__.__name__)
    
    def reset(self):
        """Reset the builder state."""
        self._steps = []
    
    def add_step(self, step_name: str, transform: Any, **kwargs):
        """Add a step to the pipeline."""
        self._steps.append({
            'name': step_name,
            'transform': transform,
            'config': kwargs
        })
        self.logger.debug(f"Added pipeline step: {step_name}")
        return self
    
    def build(self) -> list:
        """Build and return the pipeline."""
        self.logger.info(f"Built pipeline with {len(self._steps)} steps")
        return self._steps
