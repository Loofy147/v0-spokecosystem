# core_engine/nn_modules.py

import numpy as np
from .node import Node
from typing import List

class Module:
    """Base class for all neural network modules."""
    def parameters(self) -> List[Node]:
        params = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Node) and attr.requires_grad:
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
        return params

    def zero_grad(self):
        """Sets gradients of all parameters to zero."""
        for p in self.parameters():
            p.grad = np.zeros_like(p.value)


    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Linear(Module):
    """A standard linear layer (y = xW + b)."""
    def __init__(self, in_features: int, out_features: int):
        # Kaiming initialization for better training dynamics
        limit = np.sqrt(2.0 / in_features)
        self.weight = Node(np.random.randn(in_features, out_features) * limit)
        self.bias = Node(np.zeros(out_features))

    def forward(self, x: Node) -> Node:
        return x @ self.weight + self.bias

class ReLU(Module):
    """ReLU activation function."""
    def forward(self, x: Node) -> Node:
        return x.relu()

class Sigmoid(Module):
    """Sigmoid activation function."""
    def forward(self, x: Node) -> Node:
        # Numerically stable sigmoid
        out_value = 1 / (1 + np.exp(-x.value))
        out = Node(out_value, (x,), 'Sigmoid')

        def _backward():
            if x.requires_grad and out.grad is not None:
                # Gradient of sigmoid: sigmoid(x) * (1 - sigmoid(x))
                grad_contrib = out.value * (1 - out.value) * out.grad
                if x.grad is None:
                    x.grad = grad_contrib
                else:
                    x.grad += grad_contrib
        out._backward = _backward
        return out

class Tanh(Module):
    """Tanh activation function."""
    def forward(self, x: Node) -> Node:
        out_value = np.tanh(x.value)
        out = Node(out_value, (x,), 'Tanh')

        def _backward():
            if x.requires_grad and out.grad is not None:
                # Gradient of tanh: 1 - tanh(x)^2
                grad_contrib = (1 - out.value**2) * out.grad
                if x.grad is None:
                    x.grad = grad_contrib
                else:
                    x.grad += grad_contrib
        out._backward = _backward
        return out


class Sequential(Module):
    """A container to chain modules together."""
    def __init__(self, *layers: Module):
        self.layers = layers

    def forward(self, x: Node) -> Node:
        for layer in self.layers:
            x = layer(x)
        return x
