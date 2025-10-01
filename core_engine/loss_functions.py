# core_engine/loss_functions.py

import numpy as np
from .node import Node

def softmax(x: Node, axis: int = -1) -> Node:
    """
    Applies the Softmax function to a Node's value.
    Numerically stable implementation.
    """
    x = x if isinstance(x, Node) else Node(x, requires_grad=False)
    
    # Numerically stable softmax
    exp_x = (x - Node(np.max(x.value, axis=axis, keepdims=True), requires_grad=False)).exp()
    sum_exp_x = exp_x.sum(axis=axis, keepdims=True)
    out = exp_x / sum_exp_x
    return out


class MSELoss:
    """Mean Squared Error Loss function."""
    def __call__(self, predictions: Node, targets: Node) -> Node:
        """
        Calculates the MSE Loss.

        Args:
            predictions: Model predictions
            targets: Target values

        Returns:
            A Node containing the calculated loss
        """
        diff = predictions - targets
        squared_diff = diff * diff
        loss = squared_diff.sum() / Node(predictions.value.size, requires_grad=False)
        return loss


class CrossEntropyLoss:
    """
    Cross Entropy Loss function for classification.
    Includes a Softmax operation internally.
    """
    def __call__(self, predictions: Node, targets: Node) -> Node:
        """
        Calculates the Cross Entropy Loss.

        Args:
            predictions: A Node containing the model predictions (logits).
            targets: A Node containing the target class indices (integer array).

        Returns:
            A Node containing the calculated loss.
        """
        # Apply softmax to get probabilities
        probabilities = softmax(predictions, axis=-1)

        # Numerically stable log
        epsilon = 1e-15
        log_probabilities = (probabilities + Node(epsilon, requires_grad=False)).log()

        # Create one-hot encoded targets
        targets_np = targets.value.astype(int)
        num_classes = predictions.value.shape[-1]
        targets_one_hot = np.zeros_like(predictions.value)
        targets_one_hot[np.arange(targets_np.shape[0]), targets_np] = 1
        targets_one_hot_node = Node(targets_one_hot, requires_grad=False)

        # Calculate negative log likelihood
        negative_log_likelihood = -(targets_one_hot_node * log_probabilities).sum()

        # Calculate the mean over the batch
        batch_size = Node(predictions.value.shape[0], requires_grad=False)
        loss = negative_log_likelihood / batch_size

        def _backward():
            if loss.grad is None: return
            
            # Gradient of Cross-Entropy + Softmax w.r.t. logits
            grad_wrt_predictions_value = (probabilities.value - targets_one_hot_node.value) * (loss.grad / batch_size.value)
            
            if predictions.requires_grad:
                predictions.grad = predictions.grad.astype(np.float64) if predictions.grad is not None else np.zeros_like(predictions.value, dtype=np.float64)
                predictions.grad += grad_wrt_predictions_value.astype(np.float64)

        loss._backward = _backward
        return loss


class SoftmaxCrossEntropyLoss(CrossEntropyLoss):
    """Alias for CrossEntropyLoss for compatibility."""
    pass
