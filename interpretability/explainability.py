"""
Model Interpretability and Explainability Tools
Includes feature importance, SHAP-like values, attention visualization, and saliency maps.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from core_engine.node import Node
from core_engine.nn_modules import Module
import json
from pathlib import Path


class FeatureImportance:
    """
    Calculate feature importance using various methods.
    """
    
    @staticmethod
    def permutation_importance(
        model: Module,
        X: np.ndarray,
        y: np.ndarray,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        n_repeats: int = 10,
        random_state: Optional[int] = None
    ) -> Dict[int, float]:
        """
        Calculate feature importance via permutation.
        
        Args:
            model: Trained model
            X: Input features
            y: True labels
            metric_fn: Metric function (higher is better)
            n_repeats: Number of permutation repeats
            random_state: Random seed
            
        Returns:
            Dictionary mapping feature index to importance score
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Baseline score
        baseline_pred = model(X).data
        baseline_score = metric_fn(y, baseline_pred)
        
        n_features = X.shape[1]
        importances = {}
        
        for feature_idx in range(n_features):
            scores = []
            
            for _ in range(n_repeats):
                # Permute feature
                X_permuted = X.copy()
                X_permuted[:, feature_idx] = np.random.permutation(X_permuted[:, feature_idx])
                
                # Calculate score with permuted feature
                permuted_pred = model(X_permuted).data
                permuted_score = metric_fn(y, permuted_pred)
                
                # Importance is decrease in performance
                scores.append(baseline_score - permuted_score)
            
            importances[feature_idx] = float(np.mean(scores))
        
        return importances
    
    @staticmethod
    def gradient_based_importance(
        model: Module,
        X: np.ndarray,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Calculate feature importance using gradients.
        
        Args:
            model: Trained model
            X: Input features
            target_class: Target class for classification (if None, uses predicted class)
            
        Returns:
            Feature importance scores (same shape as X)
        """
        # Forward pass
        X_node = Node(X)
        output = model(X_node.data)
        
        # Select target
        if target_class is not None:
            target_output = output.data[:, target_class]
        else:
            target_output = np.max(output.data, axis=1)
        
        # Compute gradients
        output.backward()
        
        if hasattr(X_node, 'grad') and X_node.grad is not None:
            # Importance is absolute gradient
            importance = np.abs(X_node.grad)
        else:
            # Fallback: use finite differences
            importance = FeatureImportance._finite_difference_importance(model, X)
        
        return importance
    
    @staticmethod
    def _finite_difference_importance(
        model: Module,
        X: np.ndarray,
        epsilon: float = 1e-4
    ) -> np.ndarray:
        """Calculate importance using finite differences."""
        baseline_output = model(X).data
        importance = np.zeros_like(X)
        
        for i in range(X.shape[1]):
            X_perturbed = X.copy()
            X_perturbed[:, i] += epsilon
            perturbed_output = model(X_perturbed).data
            
            # Gradient approximation
            importance[:, i] = np.sum(np.abs(perturbed_output - baseline_output), axis=1) / epsilon
        
        return importance


class SHAPExplainer:
    """
    SHAP-like (SHapley Additive exPlanations) values for model explanations.
    Simplified implementation using sampling.
    """
    
    def __init__(self, model: Module, background_data: np.ndarray, n_samples: int = 100):
        self.model = model
        self.background_data = background_data
        self.n_samples = n_samples
        self.baseline_value = np.mean(model(background_data).data, axis=0)
    
    def explain_instance(self, instance: np.ndarray) -> np.ndarray:
        """
        Calculate SHAP values for a single instance.
        
        Args:
            instance: Single instance to explain (1D array)
            
        Returns:
            SHAP values for each feature
        """
        n_features = len(instance)
        shap_values = np.zeros(n_features)
        
        # Sample feature coalitions
        for _ in range(self.n_samples):
            # Random subset of features
            mask = np.random.randint(0, 2, n_features).astype(bool)
            
            # Create instance with masked features replaced by background
            background_idx = np.random.randint(0, len(self.background_data))
            masked_instance = instance.copy()
            masked_instance[~mask] = self.background_data[background_idx, ~mask]
            
            # Prediction with and without each feature
            pred_with = self.model(masked_instance.reshape(1, -1)).data[0]
            
            for feature_idx in range(n_features):
                if mask[feature_idx]:
                    # Remove this feature
                    masked_instance_without = masked_instance.copy()
                    masked_instance_without[feature_idx] = self.background_data[background_idx, feature_idx]
                    pred_without = self.model(masked_instance_without.reshape(1, -1)).data[0]
                    
                    # Marginal contribution
                    shap_values[feature_idx] += np.sum(pred_with - pred_without)
        
        # Average over samples
        shap_values /= self.n_samples
        
        return shap_values
    
    def explain_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate SHAP values for multiple instances.
        
        Args:
            X: Batch of instances
            
        Returns:
            SHAP values for each instance and feature
        """
        shap_values = np.zeros_like(X)
        
        for i in range(len(X)):
            shap_values[i] = self.explain_instance(X[i])
        
        return shap_values


class SaliencyMap:
    """
    Generate saliency maps for neural networks.
    Shows which input features are most important for predictions.
    """
    
    @staticmethod
    def vanilla_gradient(
        model: Module,
        X: np.ndarray,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Vanilla gradient saliency map.
        
        Args:
            model: Trained model
            X: Input data
            target_class: Target class (if None, uses predicted class)
            
        Returns:
            Saliency map (same shape as X)
        """
        return FeatureImportance.gradient_based_importance(model, X, target_class)
    
    @staticmethod
    def integrated_gradients(
        model: Module,
        X: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        n_steps: int = 50,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Integrated gradients for attribution.
        
        Args:
            model: Trained model
            X: Input data
            baseline: Baseline input (if None, uses zeros)
            n_steps: Number of integration steps
            target_class: Target class
            
        Returns:
            Attribution map
        """
        if baseline is None:
            baseline = np.zeros_like(X)
        
        # Generate interpolated inputs
        alphas = np.linspace(0, 1, n_steps)
        interpolated_inputs = [baseline + alpha * (X - baseline) for alpha in alphas]
        
        # Calculate gradients at each step
        gradients = []
        for interpolated in interpolated_inputs:
            grad = FeatureImportance.gradient_based_importance(model, interpolated, target_class)
            gradients.append(grad)
        
        # Average gradients
        avg_gradients = np.mean(gradients, axis=0)
        
        # Integrated gradients
        integrated_grads = (X - baseline) * avg_gradients
        
        return integrated_grads
    
    @staticmethod
    def smooth_grad(
        model: Module,
        X: np.ndarray,
        n_samples: int = 50,
        noise_level: float = 0.1,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        SmoothGrad: Average gradients over noisy samples.
        
        Args:
            model: Trained model
            X: Input data
            n_samples: Number of noisy samples
            noise_level: Standard deviation of noise
            target_class: Target class
            
        Returns:
            Smoothed saliency map
        """
        gradients = []
        
        for _ in range(n_samples):
            # Add noise
            noise = np.random.normal(0, noise_level, X.shape)
            noisy_input = X + noise
            
            # Calculate gradient
            grad = FeatureImportance.gradient_based_importance(model, noisy_input, target_class)
            gradients.append(grad)
        
        # Average gradients
        smooth_grad = np.mean(gradients, axis=0)
        
        return smooth_grad


class AttentionVisualizer:
    """
    Visualize attention patterns in models with attention mechanisms.
    """
    
    @staticmethod
    def extract_attention_weights(
        model: Module,
        X: np.ndarray,
        layer_name: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract attention weights from model.
        
        Args:
            model: Model with attention layers
            X: Input data
            layer_name: Specific layer to extract (if None, extracts all)
            
        Returns:
            Dictionary of attention weights by layer
        """
        attention_weights = {}
        
        # Forward pass
        _ = model(X)
        
        # Extract attention from layers
        for name, module in model.named_modules():
            if hasattr(module, 'attention_weights'):
                if layer_name is None or name == layer_name:
                    attention_weights[name] = module.attention_weights
        
        return attention_weights
    
    @staticmethod
    def visualize_attention_matrix(
        attention_weights: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Create visualization data for attention matrix.
        
        Args:
            attention_weights: Attention weight matrix
            feature_names: Names for features/tokens
            
        Returns:
            Visualization data
        """
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(attention_weights.shape[0])]
        
        # Normalize attention weights
        normalized_weights = attention_weights / (np.max(attention_weights) + 1e-8)
        
        return {
            "weights": normalized_weights.tolist(),
            "feature_names": feature_names,
            "shape": attention_weights.shape
        }


class ModelAnalyzer:
    """
    Comprehensive model analysis and debugging tools.
    """
    
    @staticmethod
    def analyze_predictions(
        model: Module,
        X: np.ndarray,
        y: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Analyze model predictions and identify patterns.
        
        Args:
            model: Trained model
            X: Input features
            y: True labels
            class_names: Names for classes
            
        Returns:
            Analysis results
        """
        predictions = model(X).data
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1) if y.ndim > 1 else y
        
        # Confusion matrix
        n_classes = predictions.shape[1]
        confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
        for true, pred in zip(true_classes, pred_classes):
            confusion_matrix[int(true), int(pred)] += 1
        
        # Per-class metrics
        per_class_metrics = {}
        for class_idx in range(n_classes):
            class_name = class_names[class_idx] if class_names else f"Class_{class_idx}"
            
            tp = confusion_matrix[class_idx, class_idx]
            fp = np.sum(confusion_matrix[:, class_idx]) - tp
            fn = np.sum(confusion_matrix[class_idx, :]) - tp
            tn = np.sum(confusion_matrix) - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[class_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "support": int(np.sum(true_classes == class_idx))
            }
        
        # Confidence analysis
        confidence_scores = np.max(predictions, axis=1)
        correct_mask = pred_classes == true_classes
        
        analysis = {
            "accuracy": float(np.mean(correct_mask)),
            "confusion_matrix": confusion_matrix.tolist(),
            "per_class_metrics": per_class_metrics,
            "confidence_analysis": {
                "mean_confidence": float(np.mean(confidence_scores)),
                "mean_confidence_correct": float(np.mean(confidence_scores[correct_mask])),
                "mean_confidence_incorrect": float(np.mean(confidence_scores[~correct_mask])) if np.any(~correct_mask) else 0.0,
                "low_confidence_errors": int(np.sum((confidence_scores < 0.5) & ~correct_mask))
            }
        }
        
        return analysis
    
    @staticmethod
    def find_adversarial_examples(
        model: Module,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float = 0.1,
        n_steps: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find adversarial examples using FGSM-like approach.
        
        Args:
            model: Trained model
            X: Input features
            y: True labels
            epsilon: Perturbation magnitude
            n_steps: Number of attack steps
            
        Returns:
            (adversarial_examples, success_mask)
        """
        X_adv = X.copy()
        true_classes = np.argmax(y, axis=1) if y.ndim > 1 else y
        
        for _ in range(n_steps):
            # Get gradients
            gradients = FeatureImportance.gradient_based_importance(model, X_adv)
            
            # Update adversarial examples
            X_adv += epsilon / n_steps * np.sign(gradients)
            
            # Clip to valid range (assuming normalized data)
            X_adv = np.clip(X_adv, X.min(), X.max())
        
        # Check if attack succeeded
        adv_predictions = model(X_adv).data
        adv_classes = np.argmax(adv_predictions, axis=1)
        success_mask = adv_classes != true_classes
        
        return X_adv, success_mask
    
    @staticmethod
    def analyze_layer_activations(
        model: Module,
        X: np.ndarray,
        layer_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze activation statistics for each layer.
        
        Args:
            model: Trained model
            X: Input data
            layer_names: Specific layers to analyze
            
        Returns:
            Activation statistics by layer
        """
        # Forward pass
        _ = model(X)
        
        activation_stats = {}
        
        # Analyze each layer
        for name, module in model.named_modules():
            if layer_names is None or name in layer_names:
                if hasattr(module, 'output'):
                    activations = module.output
                    
                    activation_stats[name] = {
                        "mean": float(np.mean(activations)),
                        "std": float(np.std(activations)),
                        "min": float(np.min(activations)),
                        "max": float(np.max(activations)),
                        "sparsity": float(np.mean(activations == 0)),
                        "dead_neurons": int(np.sum(np.all(activations == 0, axis=0)))
                    }
        
        return activation_stats


class ExplanationReport:
    """
    Generate comprehensive explanation reports for model predictions.
    """
    
    def __init__(self, model: Module):
        self.model = model
    
    def generate_report(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Generate comprehensive explanation report.
        
        Args:
            X: Input features
            y: True labels
            feature_names: Names for features
            class_names: Names for classes
            output_path: Path to save report
            
        Returns:
            Report dictionary
        """
        # Feature importance
        def accuracy_metric(y_true, y_pred):
            pred_classes = np.argmax(y_pred, axis=1)
            true_classes = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
            return np.mean(pred_classes == true_classes)
        
        feature_importance = FeatureImportance.permutation_importance(
            self.model, X, y, accuracy_metric, n_repeats=5
        )
        
        # Prediction analysis
        prediction_analysis = ModelAnalyzer.analyze_predictions(
            self.model, X, y, class_names
        )
        
        # Saliency maps for sample instances
        sample_indices = np.random.choice(len(X), min(5, len(X)), replace=False)
        saliency_maps = {}
        for idx in sample_indices:
            saliency = SaliencyMap.vanilla_gradient(self.model, X[idx:idx+1])
            saliency_maps[f"sample_{idx}"] = saliency.tolist()
        
        # Compile report
        report = {
            "model_summary": {
                "n_samples": len(X),
                "n_features": X.shape[1],
                "feature_names": feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            },
            "feature_importance": feature_importance,
            "prediction_analysis": prediction_analysis,
            "saliency_maps": saliency_maps,
            "generated_at": str(np.datetime64('now'))
        }
        
        # Save report
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {output_path}")
        
        return report
