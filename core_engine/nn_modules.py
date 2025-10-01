# core_engine/nn_modules.py

import numpy as np
from .node import Node, Parameter
from typing import List, Optional
import pickle

class Module:
    """Base class for all neural network modules with enhanced functionality."""
    def parameters(self) -> List[Node]:
        """Returns all trainable parameters in the module."""
        params = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, (Node, Parameter)) and attr.requires_grad:
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self):
        """Sets gradients of all parameters to zero."""
        for p in self.parameters():
            p.grad = np.zeros_like(p.value, dtype=np.float64)

    def train(self):
        """Sets the module in training mode."""
        self.training = True
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Module):
                attr.train()

    def eval(self):
        """Sets the module in evaluation mode."""
        self.training = False
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Module):
                attr.eval()

    def save(self, filepath: str):
        """Saves module parameters to a file."""
        params = {i: p.value for i, p in enumerate(self.parameters())}
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)

    def load(self, filepath: str):
        """Loads module parameters from a file."""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        for i, p in enumerate(self.parameters()):
            if i in params:
                p.value = params[i]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Linear(Module):
    """A standard linear layer (y = xW + b) with improved initialization."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        limit = np.sqrt(2.0 / in_features)
        self.weight = Parameter(np.random.randn(in_features, out_features).astype(np.float32) * limit)
        self.bias = Parameter(np.zeros(out_features, dtype=np.float32)) if bias else None
        self.training = True

    def forward(self, x: Node) -> Node:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

class ReLU(Module):
    """ReLU activation function."""
    def __init__(self):
        self.training = True

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

class Dropout(Module):
    """Dropout layer for regularization."""
    def __init__(self, p: float = 0.5):
        self.p = p
        self.training = True

    def forward(self, x: Node) -> Node:
        if not self.training or self.p == 0:
            return x
        
        # Create dropout mask
        mask = np.random.binomial(1, 1 - self.p, size=x.value.shape).astype(np.float32)
        # Scale by 1/(1-p) during training (inverted dropout)
        mask = mask / (1 - self.p)
        
        out_value = x.value * mask
        out = Node(out_value, (x,), 'Dropout')
        
        def _backward():
            if x.requires_grad and out.grad is not None:
                x.grad = x.grad.astype(np.float64) if x.grad is not None else np.zeros_like(x.value, dtype=np.float64)
                x.grad += (out.grad * mask).astype(np.float64)
        out._backward = _backward
        return out

class Softmax(Module):
    """Softmax activation function."""
    def __init__(self, axis: int = -1):
        self.axis = axis
        self.training = True

    def forward(self, x: Node) -> Node:
        # Numerically stable softmax
        exp_x = (x - Node(np.max(x.value, axis=self.axis, keepdims=True), requires_grad=False)).exp()
        sum_exp_x = exp_x.sum(axis=self.axis, keepdims=True)
        out = exp_x / sum_exp_x
        return out

class Sequential(Module):
    """A container to chain modules together."""
    def __init__(self, *layers: Module):
        self.layers = list(layers)
        self.training = True

    def forward(self, x: Node) -> Node:
        for layer in self.layers:
            x = layer(x)
        return x

    def add(self, layer: Module):
        """Adds a layer to the sequential container."""
        self.layers.append(layer)
