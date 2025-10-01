# core_engine/optimizers.py

import numpy as np
from .node import Node
from typing import List

class SGD:
    """
    Stochastic Gradient Descent optimizer for custom Node-based models with optional L2 regularization.
    """
    def __init__(self, parameters: List[Node], lr: float, weight_decay: float = 0.0):
        """
        Initializes the SGD optimizer.

        Args:
            parameters: A list of Node objects representing the model parameters.
            lr: The learning rate.
            weight_decay: The L2 regularization strength (default: 0.0).
        """
        self.parameters = parameters
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        """
        Updates the parameters based on their gradients using the SGD algorithm, with optional L2 regularization.
        """
        for p in self.parameters:
            if p.grad is not None:
                # Add L2 regularization gradient (weight_decay * weight)
                grad_with_decay = p.grad + self.weight_decay * p.value
                p.value -= self.lr * grad_with_decay
                # Note: Gradient is not reset here, it should be zeroed out
                # after the step or before the next backward pass.

    def zero_grad(self):
        """
        Sets the gradients of all parameters to None.
        """
        for p in self.parameters:
            p.grad = None
