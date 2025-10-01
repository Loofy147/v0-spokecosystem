"""Tests for core engine components."""
import pytest
import numpy as np
from core_engine import Node, Linear, ReLU, Sigmoid, Tanh, Sequential, Dropout
from core_engine import SGD, Adam, CrossEntropyLoss, MSELoss


class TestNode:
    """Tests for Node class."""
    
    def test_node_creation(self):
        """Test creating a node."""
        data = np.array([1.0, 2.0, 3.0])
        node = Node(data)
        assert np.allclose(node.value, data)
        assert node.grad is None
    
    def test_node_addition(self):
        """Test node addition."""
        a = Node(np.array([1.0, 2.0]))
        b = Node(np.array([3.0, 4.0]))
        c = a + b
        assert np.allclose(c.value, np.array([4.0, 6.0]))
    
    def test_node_multiplication(self):
        """Test node multiplication."""
        a = Node(np.array([2.0, 3.0]))
        b = Node(np.array([4.0, 5.0]))
        c = a * b
        assert np.allclose(c.value, np.array([8.0, 15.0]))
    
    def test_backward_simple(self):
        """Test simple backward pass."""
        a = Node(np.array([2.0]), requires_grad=True)
        b = Node(np.array([3.0]), requires_grad=True)
        c = a * b
        c.backward()
        assert np.allclose(a.grad, np.array([3.0]))
        assert np.allclose(b.grad, np.array([2.0]))


class TestLinear:
    """Tests for Linear layer."""
    
    def test_linear_forward(self):
        """Test linear layer forward pass."""
        layer = Linear(3, 2)
        x = Node(np.random.randn(5, 3))
        output = layer(x)
        assert output.value.shape == (5, 2)
    
    def test_linear_backward(self):
        """Test linear layer backward pass."""
        layer = Linear(3, 2)
        x = Node(np.random.randn(5, 3), requires_grad=True)
        output = layer(x)
        output.backward()
        assert x.grad is not None
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None


class TestActivations:
    """Tests for activation functions."""
    
    def test_relu_forward(self):
        """Test ReLU forward pass."""
        relu = ReLU()
        x = Node(np.array([-1.0, 0.0, 1.0]))
        output = relu(x)
        assert np.allclose(output.value, np.array([0.0, 0.0, 1.0]))
    
    def test_sigmoid_forward(self):
        """Test Sigmoid forward pass."""
        sigmoid = Sigmoid()
        x = Node(np.array([0.0]))
        output = sigmoid(x)
        assert np.allclose(output.value, np.array([0.5]))
    
    def test_tanh_forward(self):
        """Test Tanh forward pass."""
        tanh = Tanh()
        x = Node(np.array([0.0]))
        output = tanh(x)
        assert np.allclose(output.value, np.array([0.0]))


class TestSequential:
    """Tests for Sequential container."""
    
    def test_sequential_forward(self):
        """Test sequential forward pass."""
        model = Sequential(
            Linear(10, 5),
            ReLU(),
            Linear(5, 2)
        )
        x = Node(np.random.randn(3, 10))
        output = model(x)
        assert output.value.shape == (3, 2)
    
    def test_sequential_parameters(self):
        """Test getting parameters from sequential."""
        model = Sequential(
            Linear(10, 5),
            ReLU(),
            Linear(5, 2)
        )
        params = list(model.parameters())
        assert len(params) == 4  # 2 weights + 2 biases


class TestOptimizers:
    """Tests for optimizers."""
    
    def test_sgd_step(self):
        """Test SGD optimizer step."""
        layer = Linear(3, 2)
        optimizer = SGD(layer.parameters(), lr=0.1)
        
        # Forward and backward
        x = Node(np.random.randn(5, 3))
        output = layer(x)
        output.backward()
        
        # Store old weights
        old_weight = layer.weight.value.copy()
        
        # Optimizer step
        optimizer.step()
        
        # Check weights changed
        assert not np.allclose(layer.weight.value, old_weight)
    
    def test_adam_step(self):
        """Test Adam optimizer step."""
        layer = Linear(3, 2)
        optimizer = Adam(layer.parameters(), lr=0.001)
        
        # Forward and backward
        x = Node(np.random.randn(5, 3))
        output = layer(x)
        output.backward()
        
        # Store old weights
        old_weight = layer.weight.value.copy()
        
        # Optimizer step
        optimizer.step()
        
        # Check weights changed
        assert not np.allclose(layer.weight.value, old_weight)


class TestLosses:
    """Tests for loss functions."""
    
    def test_mse_loss(self):
        """Test MSE loss."""
        loss_fn = MSELoss()
        predictions = Node(np.array([1.0, 2.0, 3.0]))
        targets = Node(np.array([1.5, 2.5, 3.5]))
        loss = loss_fn(predictions, targets)
        expected = np.mean((np.array([1.0, 2.0, 3.0]) - np.array([1.5, 2.5, 3.5])) ** 2)
        assert np.allclose(loss.value, expected)
    
    def test_cross_entropy_loss(self):
        """Test cross entropy loss."""
        loss_fn = CrossEntropyLoss()
        logits = Node(np.random.randn(5, 10))
        targets = Node(np.array([0, 1, 2, 3, 4]))
        loss = loss_fn(logits, targets)
        assert loss.value > 0
