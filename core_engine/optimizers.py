# core_engine/optimizers.py

import numpy as np
from .node import Node
from typing import List, Optional

def clip_grad_norm_(parameters: List[Node], max_norm: float) -> float:
    """
    Clips gradient norm of parameters.
    
    Args:
        parameters: List of parameters to clip
        max_norm: Maximum norm value
        
    Returns:
        Total norm of the parameters
    """
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = np.linalg.norm(p.grad)
            total_norm += param_norm ** 2
    total_norm = np.sqrt(total_norm)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad *= clip_coef
    
    return total_norm

class SGD:
    """
    Stochastic Gradient Descent optimizer with momentum and weight decay.
    """
    def __init__(self, parameters: List[Node], lr: float, momentum: float = 0.0, weight_decay: float = 0.0):
        """
        Initializes the SGD optimizer.

        Args:
            parameters: A list of Node objects representing the model parameters.
            lr: The learning rate.
            momentum: Momentum factor (default: 0.0).
            weight_decay: The L2 regularization strength (default: 0.0).
        """
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(p.value, dtype=np.float64) for p in parameters]

    def step(self):
        """
        Updates the parameters based on their gradients using SGD with momentum.
        """
        for i, p in enumerate(self.parameters):
            if p.grad is not None:
                # Convert gradient to float32 for parameter update
                grad = p.grad.astype(np.float32)
                
                # Add L2 regularization gradient
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * p.value
                
                # Apply momentum
                if self.momentum != 0:
                    self.velocity[i] = self.momentum * self.velocity[i] + grad
                    grad = self.velocity[i]
                
                # Update parameters
                p.value -= self.lr * grad

    def zero_grad(self):
        """Sets the gradients of all parameters to None."""
        for p in self.parameters:
            p.grad = None


class Adam:
    """
    Adam optimizer with bias correction.
    """
    def __init__(self, parameters: List[Node], lr: float = 0.001, 
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8, 
                 weight_decay: float = 0.0):
        """
        Initializes the Adam optimizer.

        Args:
            parameters: List of parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages (beta1, beta2)
            eps: Term added for numerical stability
            weight_decay: Weight decay (L2 penalty)
        """
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize moment estimates
        self.m = [np.zeros_like(p.value, dtype=np.float64) for p in parameters]
        self.v = [np.zeros_like(p.value, dtype=np.float64) for p in parameters]
        self.t = 0  # Time step

    def step(self):
        """Performs a single optimization step."""
        self.t += 1
        
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            
            # Convert gradient to float64 for computation
            grad = p.grad.astype(np.float64)
            
            # Add weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.value.astype(np.float64)
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected moment estimates
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters (convert back to float32)
            p.value -= (self.lr * m_hat / (np.sqrt(v_hat) + self.eps)).astype(np.float32)

    def zero_grad(self):
        """Sets the gradients of all parameters to None."""
        for p in self.parameters:
            p.grad = None
