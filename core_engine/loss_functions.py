# core_engine/loss_functions.py

import numpy as np
from .node import Node

# Add Softmax as a Node method or a separate function if not already present
# Assuming softmax is not a Node method, let's implement it as a function that returns a Node.
def softmax(x: Node) -> Node:
    """
    Applies the Softmax function element-wise to a Node's value.
    Numerically stable implementation.
    """
    # Ensure input is a Node
    x = x if isinstance(x, Node) else Node(x, requires_grad=False)

    # Numerically stable softmax: exp(x - max(x)) / sum(exp(x - max(x)))
    exp_x = (x - Node(np.max(x.value, axis=-1, keepdims=True), requires_grad=False)).exp()
    sum_exp_x = exp_x.sum(axis=-1, keepdims=True)
    out = exp_x / sum_exp_x

    # Define the backward pass for softmax
    def _backward():
        if out.grad is None: return

        # Gradient of softmax: diag(softmax) - outer(softmax, softmax)
        # For a batch of inputs, the gradient is more complex.
        # For simplicity, we'll use the fact that the gradient of softmax
        # followed by cross-entropy loss w.r.t. logits is simply probabilities - target (one-hot).
        # This backward pass is specifically for softmax alone, which is more complex.
        # Since we are primarily using this within CrossEntropyLoss, where the combined
        # gradient is simpler, we might rely on that.
        # However, for a standalone softmax Node, the backward would involve:
        # grad_output * softmax - sum(grad_output * softmax, axis=-1, keepdims=True) * softmax
        # Let's implement the standalone backward for completeness.

        s = out.value # softmax output values
        grad_input = out.grad # gradient from the next layer

        # Compute the Jacobian of softmax
        # J_ij = s_i * (delta_ij - s_j)
        # For a batch, this becomes more complex.
        # A simplified approach for batch: grad_input * s - (grad_input * s).sum(axis=-1, keepdims=True) * s
        grad_wrt_input = grad_input * s - np.sum(grad_input * s, axis=-1, keepdims=True) * s

        if x.requires_grad:
            if x.grad is None:
                 x.grad = grad_wrt_input
            else:
                 x.grad += grad_wrt_input

    out._backward = _backward

    return out


class CrossEntropyLoss:
    """
    Cross Entropy Loss function for classification using custom Node.
    Includes a Softmax operation internally.
    Loss = -sum(target_one_hot * log(softmax(predictions))) / batch_size
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
        probabilities = softmax(predictions)

        # Numerically stable log: log(probabilities + epsilon)
        epsilon = 1e-15
        log_probabilities = (probabilities + Node(epsilon, requires_grad=False)).log()

        # Create one-hot encoded targets
        # This part needs to be handled carefully with Node or converted to numpy temporarily
        # Let's convert targets to numpy for one-hot encoding, as Node doesn't directly support this
        targets_np = targets.value.astype(int)
        num_classes = predictions.value.shape[-1]
        targets_one_hot = np.zeros_like(predictions.value)
        targets_one_hot[np.arange(targets_np.shape[0]), targets_np] = 1

        targets_one_hot_node = Node(targets_one_hot, requires_grad=False)


        # Calculate negative log likelihood: -sum(target_one_hot * log_probabilities)
        negative_log_likelihood = -(targets_one_hot_node * log_probabilities).sum()

        # Calculate the mean over the batch
        batch_size = Node(predictions.value.shape[0], requires_grad=False)
        loss = negative_log_likelihood / batch_size

        # Define the backward pass for Cross Entropy Loss (combined with Softmax)
        # The gradient of Cross-Entropy Loss with Softmax w.r.t. the logits (predictions)
        # is simply (probabilities - target_one_hot) / batch_size
        def _backward():
            if loss.grad is None: return

            # Calculate the gradient w.r.t. predictions (logits)
            grad_wrt_predictions_value = (probabilities.value - targets_one_hot_node.value) * (loss.grad / batch_size.value)

            if predictions.requires_grad:
                 if predictions.grad is None:
                      predictions.grad = grad_wrt_predictions_value
                 else:
                      predictions.grad += grad_wrt_predictions_value


        # Attach the backward function to the loss node
        loss._backward = _backward

        return loss
