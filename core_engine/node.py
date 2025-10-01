# core_engine/node.py

import numpy as np
from typing import Any, Iterable, Optional

def _sum_to_shape(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """A utility to broadcast gradients correctly."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, (g_dim, t_dim) in enumerate(zip(grad.shape, shape)):
        if t_dim == 1 and g_dim != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)

class Node:
    """
    A Node represents a value in the computation graph. It stores the value,
    its parents, the operation that produced it, and its gradient.
    
    Enhanced with float64 gradient accumulation for numerical stability.
    """
    def __init__(self, value: Any, parents: Iterable['Node'] = (), op: str = '', requires_grad: bool = True):
        # print(f"Node.__init__ called with value type: {type(value)}, op: {op}") # Debug print
        # Ensure value is a numpy array and handle cases where value is already a Node
        if isinstance(value, Node):
            # If value is already a Node, we extract its numerical value and requires_grad status
            # This should prevent recursively wrapping a Node in another Node
            self.value = np.array(value.value, dtype=np.float32)
            self.requires_grad = value.requires_grad
            # print(f"  Node.__init__ received a Node, extracting value: {self.value.shape}, requires_grad: {self.requires_grad}") # Debug print
        else:
            try:
                self.value = np.array(value, dtype=np.float32)
                # print(f"  Node.__init__ converting value to numpy array: {self.value.shape}") # Debug print
            except Exception as e:
                print(f"  Node.__init__ failed to convert value to numpy array: {e}") # Debug print
                raise # Re-raise the exception after printing
            self.requires_grad = requires_grad # Use the requires_grad argument


        self.parents = tuple(parents)
        self.op = op
        self.grad: Optional[np.ndarray] = None
        self._backward = lambda: None

    def _ensure(self, other: Any) -> 'Node':
        """Ensures the other operand is a Node."""
        if isinstance(other, self.__class__):
            return other
        else:
            return Node(other, requires_grad=False)

    def __add__(self, other: Any) -> 'Node':
        # print(f"Node.__add__ called with self type: {type(self)}, other type: {type(other)}") # Debug print
        other = self._ensure(other)
        # Perform the operation on the numerical values
        out_value = self.value + other.value
        out = Node(out_value, (self, other), '+')
        def _backward():
            if out.grad is None: return
            if self.requires_grad: 
                self.grad = self.grad.astype(np.float64) if self.grad is not None else np.zeros_like(self.value, dtype=np.float64)
                self.grad += _sum_to_shape(out.grad.astype(np.float64), self.value.shape)
            if other.requires_grad: 
                other.grad = other.grad.astype(np.float64) if other.grad is not None else np.zeros_like(other.value, dtype=np.float64)
                other.grad += _sum_to_shape(out.grad.astype(np.float64), other.value.shape)
        out._backward = _backward
        return out

    def __mul__(self, other: Any) -> 'Node':
        # print(f"Node.__mul__ called with self type: {type(self)}, other type: {type(other)}") # Debug print
        other = self._ensure(other)
        # Perform the operation on the numerical values
        out_value = self.value * other.value
        out = Node(out_value, (self, other), '*')
        def _backward():
            if out.grad is None: return
            # Gradients should be multiplied by the other operand's value
            if self.requires_grad: 
                self.grad = self.grad.astype(np.float64) if self.grad is not None else np.zeros_like(self.value, dtype=np.float64)
                self.grad += _sum_to_shape((out.grad * other.value).astype(np.float64), self.value.shape)
            if other.requires_grad: 
                other.grad = other.grad.astype(np.float64) if other.grad is not None else np.zeros_like(other.value, dtype=np.float64)
                other.grad += _sum_to_shape((out.grad * self.value).astype(np.float64), other.value.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other: Any) -> 'Node':
        # print(f"Node.__matmul__ called with self type: {type(self)}, other type: {type(other)}") # Debug print
        other = self._ensure(other)
        # Ensure we are using the numerical values for the matrix multiplication
        out_value = self.value @ other.value
        out = Node(out_value, (self, other), '@')
        def _backward():
            if out.grad is None: return
            # Gradients for matrix multiplication
            if self.requires_grad:
                # Gradient w.r.t. self: out.grad @ other.value.T
                self.grad += _sum_to_shape(out.grad @ other.value.T, self.value.shape)
            if other.requires_grad:
                # Gradient w.r.t. other: self.value.T @ out.grad
                other.grad += _sum_to_shape(self.value.T @ out.grad, other.value.shape)
        out._backward = _backward
        return out

    def __pow__(self, power: float) -> 'Node':
        # print(f"Node.__pow__ called with self type: {type(self)}, power type: {type(power)}") # Debug print
        # Perform the operation on the numerical value
        out_value = np.power(self.value, power)
        out = Node(out_value, (self,), f'**{power}')
        def _backward():
            if self.requires_grad and out.grad is not None:
                # Gradient of x^n is n * x^(n-1)
                grad_contrib = power * np.power(self.value, power - 1) * out.grad
                if self.grad is None:
                    self.grad = grad_contrib
                else:
                    self.grad += grad_contrib
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False) -> 'Node':
        # print(f"Node.sum called with self type: {type(self)}") # Debug print
        # Perform the sum operation on the numerical value
        out_value = self.value.sum(axis=axis, keepdims=keepdims)
        out = Node(out_value, (self,), 'sum')
        def _backward():
            if self.requires_grad and out.grad is not None:
                # Gradient of sum is broadcasting the output gradient
                grad = out.grad if keepdims else np.expand_dims(out.grad, axis)
                if self.grad is None:
                     self.grad = np.broadcast_to(grad, self.value.shape)
                else:
                     self.grad += np.broadcast_to(grad, self.value.shape)
        out._backward = _backward
        return out

    def relu(self) -> 'Node':
        # print(f"Node.relu called with self type: {type(self)}") # Debug print
        # Perform the ReLU operation on the numerical value
        out_value = np.maximum(0, self.value)
        out = Node(out_value, (self,), 'ReLU')
        def _backward():
            if self.requires_grad and out.grad is not None:
                # Gradient of ReLU is 1 if input > 0, 0 otherwise
                grad_contrib = (self.value > 0) * out.grad
                if self.grad is None:
                     self.grad = grad_contrib
                else:
                     self.grad += grad_contrib
        out._backward = _backward
        return out

    def __neg__(self) -> 'Node':
        """Negation: -x"""
        out_value = -self.value
        out = Node(out_value, (self,), 'neg')
        def _backward():
            if self.requires_grad and out.grad is not None:
                self.grad = self.grad.astype(np.float64) if self.grad is not None else np.zeros_like(self.value, dtype=np.float64)
                self.grad += -out.grad.astype(np.float64)
        out._backward = _backward
        return out

    def __sub__(self, other: Any) -> 'Node':
        """Subtraction: self - other"""
        return self + (-self._ensure(other))

    def __rsub__(self, other: Any) -> 'Node':
        """Reverse subtraction: other - self"""
        return self._ensure(other) + (-self)

    def __truediv__(self, other: Any) -> 'Node':
        """Division: self / other"""
        other = self._ensure(other)
        out_value = self.value / other.value
        out = Node(out_value, (self, other), '/')
        def _backward():
            if out.grad is None: return
            if self.requires_grad:
                self.grad = self.grad.astype(np.float64) if self.grad is not None else np.zeros_like(self.value, dtype=np.float64)
                self.grad += _sum_to_shape((out.grad / other.value).astype(np.float64), self.value.shape)
            if other.requires_grad:
                other.grad = other.grad.astype(np.float64) if other.grad is not None else np.zeros_like(other.value, dtype=np.float64)
                other.grad += _sum_to_shape((-out.grad * self.value / (other.value ** 2)).astype(np.float64), other.value.shape)
        out._backward = _backward
        return out

    def __rtruediv__(self, other: Any) -> 'Node':
        """Reverse division: other / self"""
        return self._ensure(other) / self

    def __radd__(self, other: Any) -> 'Node':
        """Reverse addition: other + self"""
        return self + other

    def __rmul__(self, other: Any) -> 'Node':
        """Reverse multiplication: other * self"""
        return self * other

    def exp(self) -> 'Node':
        """Exponential: e^x"""
        out_value = np.exp(self.value)
        out = Node(out_value, (self,), 'exp')
        def _backward():
            if self.requires_grad and out.grad is not None:
                self.grad = self.grad.astype(np.float64) if self.grad is not None else np.zeros_like(self.value, dtype=np.float64)
                self.grad += (out.grad * out.value).astype(np.float64)
        out._backward = _backward
        return out

    def log(self) -> 'Node':
        """Natural logarithm: ln(x)"""
        out_value = np.log(self.value + 1e-15)  # Add epsilon for numerical stability
        out = Node(out_value, (self,), 'log')
        def _backward():
            if self.requires_grad and out.grad is not None:
                self.grad = self.grad.astype(np.float64) if self.grad is not None else np.zeros_like(self.value, dtype=np.float64)
                self.grad += (out.grad / (self.value + 1e-15)).astype(np.float64)
        out._backward = _backward
        return out

    def backward(self):
        """Performs backpropagation to compute gradients for all nodes."""
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for p in v.parents: build_topo(p)
                topo.append(v)

        build_topo(self)

        for v in topo:
            if v.requires_grad:
                v.grad = np.zeros_like(v.value, dtype=np.float64)

        if self.requires_grad:
            if self.grad is None:
                self.grad = np.ones_like(self.value, dtype=np.float64)
            else:
                self.grad += np.ones_like(self.value, dtype=np.float64)
        else:
            return

        for v in reversed(topo):
            v._backward()

    def __repr__(self) -> str:
        return f"Node(value={self.value.shape}, op='{self.op}')"

class Parameter(Node):
    """
    A Parameter is a special Node that represents a trainable parameter.
    It always requires gradients and provides additional utilities.
    """
    def __init__(self, value: Any):
        super().__init__(value, requires_grad=True)
        self.is_parameter = True

    def __repr__(self) -> str:
        return f"Parameter(shape={self.value.shape})"
