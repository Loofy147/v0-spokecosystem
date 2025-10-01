"""
Strategy pattern for interchangeable algorithms.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from utils.logging_config import get_logger

logger = get_logger(__name__)


class Strategy(ABC):
    """Abstract strategy base class."""
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the strategy."""
        pass


class TrainingStrategy(Strategy):
    """Base class for training strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    def train_step(self, model, batch, optimizer) -> Dict[str, float]:
        """Execute one training step."""
        pass
    
    @abstractmethod
    def validation_step(self, model, batch) -> Dict[str, float]:
        """Execute one validation step."""
        pass


class StandardTrainingStrategy(TrainingStrategy):
    """Standard supervised training strategy."""
    
    def train_step(self, model, batch, optimizer) -> Dict[str, float]:
        """Standard training step with forward and backward pass."""
        x, y = batch
        
        # Forward pass
        predictions = model(x)
        loss = model.compute_loss(predictions, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {'loss': float(loss)}
    
    def validation_step(self, model, batch) -> Dict[str, float]:
        """Standard validation step."""
        x, y = batch
        predictions = model(x)
        loss = model.compute_loss(predictions, y)
        
        return {'val_loss': float(loss)}


class MixedPrecisionTrainingStrategy(TrainingStrategy):
    """Mixed precision training strategy."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.scaler = self.config.get('scaler')
    
    def train_step(self, model, batch, optimizer) -> Dict[str, float]:
        """Training step with automatic mixed precision."""
        x, y = batch
        
        # Forward pass with autocast
        with self.autocast():
            predictions = model(x)
            loss = model.compute_loss(predictions, y)
        
        # Backward pass with gradient scaling
        optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        return {'loss': float(loss)}
    
    def validation_step(self, model, batch) -> Dict[str, float]:
        """Validation step with autocast."""
        x, y = batch
        
        with self.autocast():
            predictions = model(x)
            loss = model.compute_loss(predictions, y)
        
        return {'val_loss': float(loss)}
    
    def autocast(self):
        """Context manager for automatic mixed precision."""
        # Placeholder for actual autocast implementation
        from contextlib import nullcontext
        return nullcontext()


class Context:
    """Context that uses a strategy."""
    
    def __init__(self, strategy: Strategy):
        self._strategy = strategy
        self.logger = get_logger(self.__class__.__name__)
    
    @property
    def strategy(self) -> Strategy:
        """Get current strategy."""
        return self._strategy
    
    @strategy.setter
    def strategy(self, strategy: Strategy):
        """Set new strategy."""
        self.logger.info(f"Switching strategy to {strategy.__class__.__name__}")
        self._strategy = strategy
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute the current strategy."""
        return self._strategy.execute(*args, **kwargs)
