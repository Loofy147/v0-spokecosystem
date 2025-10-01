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
        self.grad: Optional[np.ndarray] = None # Gradient is None initially
        self._backward = lambda: None


    def _ensure(self, other: Any) -> 'Node':
        """Ensures the other operand is a Node."""
        # print(f"Node._ensure called with type: {type(other)}") # Debug print
        # Corrected check to use the current Node class
        if isinstance(other, self.__class__):
            # print("  _ensure: Input is already a Node, returning it.") # Debug print
            return other
        else:
            # print("  _ensure: Input is not a Node, creating a new Node.") # Debug print
            return Node(other, requires_grad=False) # Ensure constants don't require grad


    def __add__(self, other: Any) -> 'Node':
        # print(f"Node.__add__ called with self type: {type(self)}, other type: {type(other)}") # Debug print
        other = self._ensure(other)
        # Perform the operation on the numerical values
        out_value = self.value + other.value
        out = Node(out_value, (self, other), '+')
        def _backward():
            if out.grad is None: return
            if self.requires_grad: self.grad += _sum_to_shape(out.grad, self.value.shape)
            if other.requires_grad: other.grad += _sum_to_shape(out.grad, other.value.shape)
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
            if self.requires_grad: self.grad += _sum_to_shape(out.grad * other.value, self.value.shape)
            if other.requires_grad: other.grad += _sum_to_shape(out.grad * self.value, other.value.shape)
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


    def backward(self):
        """Performs backpropagation to compute gradients for all nodes."""
        # print(f"Node.backward called on node with op: {self.op}") # Debug print
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for p in v.parents: build_topo(p)
                topo.append(v)

        build_topo(self)

        # Initialize gradients to zero for all nodes that require grad
        for v in topo:
            if v.requires_grad:
                 v.grad = np.zeros_like(v.value, dtype=np.float32)

        # Set the gradient of the output node to 1 if it requires grad
        if self.requires_grad:
            # Ensure the gradient is initialized before adding to it
            if self.grad is None:
                 self.grad = np.ones_like(self.value, dtype=np.float32)
            else:
                self.grad += np.ones_like(self.value, dtype=np.float32)
        else:
             # If the output does not require gradients, there is nothing to backpropagate
             # print("  backward: Output node does not require grad, stopping.") # Debug print
             return


        for v in reversed(topo):
            # print(f"  backward: Processing node with op: {v.op}, requires_grad: {v.requires_grad}") # Debug print
            v._backward()
            # if v.grad is not None:
            #      print(f"  backward: Gradient calculated for node with op: {v.op}, shape: {v.grad.shape}") # Debug print
            # else:
            #      print(f"  backward: No gradient for node with op: {v.op}") # Debug print


    def __repr__(self) -> str:
        return f"Node(value={self.value.shape}, op='{self.op}')"
