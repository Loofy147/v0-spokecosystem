"""Enhanced fitness functions with multi-objective optimization."""
import math
import numpy as np
from typing import Dict, Any

def basic_fitness(metrics: dict) -> float:
    """A basic fitness function based on validation accuracy."""
    return metrics.get('val_accuracy', -1.0)

def count_parameters(model) -> int:
    """Count trainable parameters in a model."""
    return sum(p.value.size for p in model.parameters())

def estimate_flops(genome, input_size: int = 784) -> int:
    """
    Estimate FLOPs for a given genome architecture.
    
    Args:
        genome: Genome specification
        input_size: Input feature dimension
        
    Returns:
        Estimated FLOPs count
    """
    flops = 0
    current_size = input_size
    
    for token in genome.architecture:
        t = token.lower().strip()
        
        if t.startswith('linear') or t.startswith('fc'):
            parts = t.split('-')
            out_size = int(parts[1]) if len(parts) > 1 else 128
            # FLOPs for matrix multiplication: 2 * in * out
            flops += 2 * current_size * out_size
            current_size = out_size
            
        elif t in ['relu', 'sigmoid', 'tanh']:
            # Activation functions: 1 FLOP per element
            flops += current_size
    
    return flops

def composite_fitness(
    val_acc: float,
    params: int,
    flops: int,
    w_accuracy: float = 1.0,
    w_params: float = 0.12,
    w_flops: float = 0.18
) -> float:
    """
    Multi-objective fitness function balancing accuracy, parameters, and FLOPs.
    
    Args:
        val_acc: Validation accuracy
        params: Number of parameters
        flops: Estimated FLOPs
        w_accuracy: Weight for accuracy
        w_params: Weight for parameter penalty
        w_flops: Weight for FLOPs penalty
        
    Returns:
        Composite fitness score
    """
    param_penalty = 1.0 / (1.0 + math.log1p(params))
    flops_penalty = 1.0 / (1.0 + math.log1p(flops / 1e6 + 1.0))
    
    fitness = (val_acc ** w_accuracy) * (param_penalty ** w_params) * (flops_penalty ** w_flops)
    return float(fitness)

def evaluate_model(
    genome_dict: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 10,
    fitness_cfg: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Evaluate a model genome on training and validation data.
    
    Args:
        genome_dict: Genome specification
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Number of training epochs
        fitness_cfg: Fitness function configuration
        
    Returns:
        Dictionary with fitness metrics
    """
    from core_engine import Node, CrossEntropyLoss, SGD
    from automl.model_builder import Genome, build_model_from_genome
    
    try:
        genome = Genome(**genome_dict) if isinstance(genome_dict, dict) else genome_dict
        
        input_size = X_train.shape[1]
        num_classes = int(y_train.max()) + 1
        
        model = build_model_from_genome(genome, input_size, num_classes, use_dropout=True)
        optimizer = SGD(model.parameters(), lr=genome.lr, weight_decay=genome.wd)
        loss_fn = CrossEntropyLoss()
        
        # Training loop
        model.train()
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training
            X_node = Node(X_train, requires_grad=False)
            y_node = Node(y_train, requires_grad=False)
            
            predictions = model(X_node)
            loss = loss_fn(predictions, y_node)
            
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            X_val_node = Node(X_val, requires_grad=False)
            val_predictions = model(X_val_node)
            val_pred_classes = np.argmax(val_predictions.value, axis=1)
            val_acc = np.mean(val_pred_classes == y_val)
            best_val_acc = max(best_val_acc, val_acc)
            model.train()
        
        # Calculate metrics
        params = count_parameters(model)
        flops = estimate_flops(genome, input_size)
        
        cfg = fitness_cfg or {}
        fitness = composite_fitness(best_val_acc, params, flops, **cfg)
        
        return {
            'fitness': float(fitness),
            'val_accuracy': float(best_val_acc),
            'params': int(params),
            'flops': int(flops)
        }
        
    except Exception as e:
        print(f"Evaluation error: {e}")
        return {
            'fitness': 0.0,
            'val_accuracy': 0.0,
            'params': 0,
            'flops': 0
        }
