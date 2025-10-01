"""Model builder: converts genome representation to neural network model."""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from core_engine import Module, Linear, ReLU, Sigmoid, Tanh, Sequential, Dropout

@dataclass
class Genome:
    """Genome representation for neural architecture."""
    architecture: List[str]
    lr: float = 1e-3
    wd: float = 5e-4
    batch_size: int = 128
    meta: Optional[dict] = None

def build_model_from_genome(genome: Genome, input_size: int, num_classes: int, use_dropout: bool = False) -> Module:
    """
    Builds a neural network model from a genome specification.
    
    Args:
        genome: Genome specification with architecture and hyperparameters
        input_size: Input feature dimension
        num_classes: Number of output classes
        use_dropout: Whether to add dropout layers
        
    Returns:
        A Module representing the neural network
    """
    layers = []
    current_size = input_size
    
    for token in genome.architecture:
        t = token.lower().strip()
        
        if t.startswith('linear') or t.startswith('fc'):
            # Parse layer size: "linear-128" or "fc-64"
            parts = t.split('-')
            out_size = int(parts[1]) if len(parts) > 1 else 128
            layers.append(Linear(current_size, out_size))
            current_size = out_size
            
        elif t == 'relu':
            layers.append(ReLU())
            
        elif t == 'sigmoid':
            layers.append(Sigmoid())
            
        elif t == 'tanh':
            layers.append(Tanh())
            
        elif t.startswith('dropout'):
            if use_dropout:
                # Parse dropout rate: "dropout-0.5" or default 0.5
                parts = t.split('-')
                p = float(parts[1]) if len(parts) > 1 else 0.5
                layers.append(Dropout(p))
    
    # Add output layer if not already present
    if not layers or not isinstance(layers[-1], Linear):
        layers.append(Linear(current_size, num_classes))
    else:
        # Adjust last linear layer to match num_classes
        last_layer = layers[-1]
        if isinstance(last_layer, Linear):
            layers[-1] = Linear(last_layer.weight.value.shape[0], num_classes)
    
    return Sequential(*layers)
