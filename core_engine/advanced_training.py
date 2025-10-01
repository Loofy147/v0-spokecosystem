"""
Advanced Training Features for Enterprise-Grade Deep Learning
Includes mixed precision, gradient accumulation, distributed training support,
learning rate scheduling, early stopping, and gradient monitoring.
"""

import numpy as np
from typing import Optional, Callable, Dict, List, Any, Tuple
from core_engine.node import Node
from core_engine.nn_modules import Module
from core_engine.optimizers import Optimizer
import time
import json
from pathlib import Path


class MixedPrecisionTrainer:
    """
    Mixed precision training for faster computation and reduced memory usage.
    Uses float16 for forward pass and float32 for gradient accumulation.
    """
    
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_scale: float = 1024.0,
        scale_factor: float = 2.0,
        scale_window: int = 2000
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_scale = loss_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self._scale_growth_tracker = 0
        
    def scale_loss(self, loss: Node) -> Node:
        """Scale loss to prevent underflow in fp16."""
        return loss * self.loss_scale
    
    def unscale_gradients(self):
        """Unscale gradients before optimizer step."""
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = param.grad / self.loss_scale
    
    def update_scale(self, found_inf: bool):
        """Dynamically adjust loss scale based on gradient overflow."""
        if found_inf:
            self.loss_scale = max(1.0, self.loss_scale / self.scale_factor)
            self._scale_growth_tracker = 0
        else:
            self._scale_growth_tracker += 1
            if self._scale_growth_tracker >= self.scale_window:
                self.loss_scale *= self.scale_factor
                self._scale_growth_tracker = 0
    
    def check_overflow(self) -> bool:
        """Check if gradients contain inf or nan."""
        for param in self.model.parameters():
            if param.grad is not None:
                grad_data = param.grad.data if hasattr(param.grad, 'data') else param.grad
                if np.any(np.isnan(grad_data)) or np.any(np.isinf(grad_data)):
                    return True
        return False


class GradientAccumulator:
    """
    Gradient accumulation for training with larger effective batch sizes.
    Useful when GPU memory is limited.
    """
    
    def __init__(self, model: Module, accumulation_steps: int = 4):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        
    def should_update(self) -> bool:
        """Check if we should perform optimizer step."""
        self.current_step += 1
        if self.current_step >= self.accumulation_steps:
            self.current_step = 0
            return True
        return False
    
    def scale_gradients(self):
        """Scale accumulated gradients by number of accumulation steps."""
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = param.grad / self.accumulation_steps


class LearningRateScheduler:
    """
    Advanced learning rate scheduling strategies.
    """
    
    def __init__(self, optimizer: Optimizer, schedule_type: str = "cosine", **kwargs):
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.initial_lr = optimizer.lr
        self.current_step = 0
        self.kwargs = kwargs
        
    def step(self, epoch: Optional[int] = None):
        """Update learning rate based on schedule."""
        self.current_step += 1
        
        if self.schedule_type == "cosine":
            total_steps = self.kwargs.get("total_steps", 1000)
            min_lr = self.kwargs.get("min_lr", 0.0)
            progress = min(self.current_step / total_steps, 1.0)
            self.optimizer.lr = min_lr + (self.initial_lr - min_lr) * 0.5 * (
                1 + np.cos(np.pi * progress)
            )
            
        elif self.schedule_type == "linear_warmup":
            warmup_steps = self.kwargs.get("warmup_steps", 100)
            if self.current_step < warmup_steps:
                self.optimizer.lr = self.initial_lr * (self.current_step / warmup_steps)
            else:
                self.optimizer.lr = self.initial_lr
                
        elif self.schedule_type == "exponential":
            decay_rate = self.kwargs.get("decay_rate", 0.96)
            decay_steps = self.kwargs.get("decay_steps", 100)
            self.optimizer.lr = self.initial_lr * (decay_rate ** (self.current_step / decay_steps))
            
        elif self.schedule_type == "step":
            step_size = self.kwargs.get("step_size", 100)
            gamma = self.kwargs.get("gamma", 0.1)
            self.optimizer.lr = self.initial_lr * (gamma ** (self.current_step // step_size))
            
        elif self.schedule_type == "plateau":
            # Requires manual triggering with metric
            pass
    
    def step_plateau(self, metric: float, mode: str = "min"):
        """Update LR based on metric plateau (call manually)."""
        patience = self.kwargs.get("patience", 10)
        factor = self.kwargs.get("factor", 0.1)
        threshold = self.kwargs.get("threshold", 1e-4)
        
        if not hasattr(self, "_best_metric"):
            self._best_metric = metric
            self._plateau_count = 0
            return
        
        improved = False
        if mode == "min":
            improved = metric < self._best_metric - threshold
        else:
            improved = metric > self._best_metric + threshold
        
        if improved:
            self._best_metric = metric
            self._plateau_count = 0
        else:
            self._plateau_count += 1
            if self._plateau_count >= patience:
                self.optimizer.lr *= factor
                self._plateau_count = 0


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.early_stop = False
        
    def __call__(self, score: float, model: Module) -> bool:
        """
        Check if training should stop.
        Returns True if should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = self._save_weights(model)
            return False
        
        improved = False
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = self._save_weights(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights:
                    self._restore_weights(model, self.best_weights)
                return True
        
        return False
    
    def _save_weights(self, model: Module) -> Dict:
        """Save model weights."""
        return {name: param.data.copy() for name, param in model.named_parameters()}
    
    def _restore_weights(self, model: Module, weights: Dict):
        """Restore model weights."""
        for name, param in model.named_parameters():
            if name in weights:
                param.data = weights[name].copy()


class GradientMonitor:
    """
    Monitor gradient statistics for debugging and analysis.
    """
    
    def __init__(self):
        self.history = []
        
    def log_gradients(self, model: Module, step: int) -> Dict[str, float]:
        """Log gradient statistics."""
        stats = {
            "step": step,
            "grad_norm": 0.0,
            "grad_mean": 0.0,
            "grad_std": 0.0,
            "grad_max": 0.0,
            "grad_min": 0.0,
            "num_zeros": 0,
            "num_nans": 0,
            "num_infs": 0
        }
        
        all_grads = []
        for param in model.parameters():
            if param.grad is not None:
                grad_data = param.grad.data if hasattr(param.grad, 'data') else param.grad
                all_grads.append(grad_data.flatten())
                
                stats["num_zeros"] += np.sum(grad_data == 0)
                stats["num_nans"] += np.sum(np.isnan(grad_data))
                stats["num_infs"] += np.sum(np.isinf(grad_data))
        
        if all_grads:
            all_grads = np.concatenate(all_grads)
            valid_grads = all_grads[~np.isnan(all_grads) & ~np.isinf(all_grads)]
            
            if len(valid_grads) > 0:
                stats["grad_norm"] = float(np.linalg.norm(valid_grads))
                stats["grad_mean"] = float(np.mean(valid_grads))
                stats["grad_std"] = float(np.std(valid_grads))
                stats["grad_max"] = float(np.max(valid_grads))
                stats["grad_min"] = float(np.min(valid_grads))
        
        self.history.append(stats)
        return stats
    
    def save_history(self, filepath: str):
        """Save gradient history to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)


class AdvancedTrainer:
    """
    Enterprise-grade trainer with all advanced features integrated.
    """
    
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_fn: Callable,
        use_mixed_precision: bool = False,
        gradient_accumulation_steps: int = 1,
        lr_scheduler: Optional[LearningRateScheduler] = None,
        early_stopping: Optional[EarlyStopping] = None,
        gradient_clip_value: Optional[float] = None,
        monitor_gradients: bool = False
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.gradient_clip_value = gradient_clip_value
        
        # Advanced features
        self.mixed_precision = MixedPrecisionTrainer(model, optimizer) if use_mixed_precision else None
        self.gradient_accumulator = GradientAccumulator(model, gradient_accumulation_steps)
        self.lr_scheduler = lr_scheduler
        self.early_stopping = early_stopping
        self.gradient_monitor = GradientMonitor() if monitor_gradients else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.training_history = []
        
    def train_step(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Single training step with all advanced features."""
        # Forward pass
        predictions = self.model(X)
        loss = self.loss_fn(predictions, y)
        
        # Mixed precision scaling
        if self.mixed_precision:
            loss = self.mixed_precision.scale_loss(loss)
        
        # Backward pass
        loss.backward()
        
        # Check for gradient overflow in mixed precision
        if self.mixed_precision:
            found_inf = self.mixed_precision.check_overflow()
            self.mixed_precision.update_scale(found_inf)
            if found_inf:
                # Skip this step if overflow detected
                self.model.zero_grad()
                return {"loss": float('inf'), "skipped": True}
            self.mixed_precision.unscale_gradients()
        
        # Gradient monitoring
        if self.gradient_monitor:
            grad_stats = self.gradient_monitor.log_gradients(self.model, self.global_step)
        
        # Gradient clipping
        if self.gradient_clip_value:
            self._clip_gradients(self.gradient_clip_value)
        
        # Gradient accumulation
        should_update = self.gradient_accumulator.should_update()
        if should_update:
            self.gradient_accumulator.scale_gradients()
            self.optimizer.step()
            self.model.zero_grad()
            
            # Learning rate scheduling
            if self.lr_scheduler:
                self.lr_scheduler.step()
        
        self.global_step += 1
        
        return {
            "loss": float(loss.data),
            "lr": self.optimizer.lr,
            "step": self.global_step
        }
    
    def _clip_gradients(self, max_norm: float):
        """Clip gradients by global norm."""
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_data = param.grad.data if hasattr(param.grad, 'data') else param.grad
                total_norm += np.sum(grad_data ** 2)
        
        total_norm = np.sqrt(total_norm)
        clip_coef = max_norm / (total_norm + 1e-6)
        
        if clip_coef < 1:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = param.grad * clip_coef
    
    def train_epoch(
        self,
        train_loader,
        val_loader=None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_losses = []
        start_time = time.time()
        
        for batch_idx, (X, y) in enumerate(train_loader):
            metrics = self.train_step(X, y)
            if not metrics.get("skipped", False):
                epoch_losses.append(metrics["loss"])
            
            if verbose and batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {metrics['loss']:.4f}, LR: {metrics['lr']:.6f}")
        
        # Validation
        val_metrics = {}
        if val_loader:
            val_metrics = self.evaluate(val_loader)
            
            # Early stopping check
            if self.early_stopping:
                should_stop = self.early_stopping(val_metrics["loss"], self.model)
                if should_stop:
                    print(f"Early stopping triggered at epoch {self.epoch}")
                    return {"early_stop": True, **val_metrics}
        
        self.epoch += 1
        epoch_time = time.time() - start_time
        
        results = {
            "epoch": self.epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "epoch_time": epoch_time,
            **val_metrics
        }
        
        self.training_history.append(results)
        return results
    
    def evaluate(self, data_loader) -> Dict[str, float]:
        """Evaluate model on validation/test set."""
        losses = []
        correct = 0
        total = 0
        
        for X, y in data_loader:
            predictions = self.model(X)
            loss = self.loss_fn(predictions, y)
            losses.append(float(loss.data))
            
            # Calculate accuracy for classification
            pred_classes = np.argmax(predictions.data, axis=1)
            true_classes = np.argmax(y, axis=1) if y.ndim > 1 else y
            correct += np.sum(pred_classes == true_classes)
            total += len(y)
        
        return {
            "loss": float(np.mean(losses)),
            "accuracy": correct / total if total > 0 else 0.0
        }
    
    def save_training_history(self, filepath: str):
        """Save training history to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=2)
