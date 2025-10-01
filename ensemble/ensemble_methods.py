"""
Advanced Ensemble Methods and Model Fusion
Includes voting, stacking, boosting, bagging, and knowledge distillation.
"""

import numpy as np
from typing import List, Dict, Optional, Callable, Any, Tuple
from core_engine.node import Node
from core_engine.nn_modules import Module
import json
from pathlib import Path


class EnsembleBase:
    """Base class for ensemble methods."""
    
    def __init__(self, models: List[Module]):
        self.models = models
        self.weights = None
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the ensemble."""
        raise NotImplementedError
    
    def save(self, filepath: str):
        """Save ensemble configuration."""
        raise NotImplementedError
    
    @classmethod
    def load(cls, filepath: str):
        """Load ensemble from file."""
        raise NotImplementedError


class VotingEnsemble(EnsembleBase):
    """
    Voting ensemble for classification.
    Supports hard voting (majority) and soft voting (average probabilities).
    """
    
    def __init__(
        self,
        models: List[Module],
        voting: str = "soft",  # 'hard' or 'soft'
        weights: Optional[List[float]] = None
    ):
        super().__init__(models)
        self.voting = voting
        
        if weights is None:
            self.weights = np.ones(len(models)) / len(models)
        else:
            self.weights = np.array(weights)
            self.weights = self.weights / np.sum(self.weights)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        predictions = [model(X).data for model in self.models]
        
        if self.voting == "soft":
            # Average weighted probabilities
            weighted_preds = np.array([pred * w for pred, w in zip(predictions, self.weights)])
            return np.sum(weighted_preds, axis=0)
        else:
            # Hard voting - majority class
            class_preds = [np.argmax(pred, axis=1) for pred in predictions]
            class_preds = np.array(class_preds).T
            
            # Weighted voting
            final_preds = []
            for sample_votes in class_preds:
                vote_counts = {}
                for vote, weight in zip(sample_votes, self.weights):
                    vote_counts[vote] = vote_counts.get(vote, 0) + weight
                final_preds.append(max(vote_counts.items(), key=lambda x: x[1])[0])
            
            # Convert to one-hot
            n_classes = predictions[0].shape[1]
            result = np.zeros((len(final_preds), n_classes))
            result[np.arange(len(final_preds)), final_preds] = 1
            return result
    
    def optimize_weights(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        method: str = "grid_search"
    ):
        """
        Optimize ensemble weights on validation set.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            method: Optimization method ('grid_search', 'gradient')
        """
        if method == "grid_search":
            self._grid_search_weights(X_val, y_val)
        elif method == "gradient":
            self._gradient_optimize_weights(X_val, y_val)
    
    def _grid_search_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """Grid search for optimal weights."""
        best_weights = self.weights.copy()
        best_acc = self._compute_accuracy(X_val, y_val)
        
        # Try different weight combinations
        n_steps = 11
        for i in range(len(self.models)):
            for weight in np.linspace(0, 1, n_steps):
                test_weights = self.weights.copy()
                test_weights[i] = weight
                test_weights = test_weights / np.sum(test_weights)
                
                self.weights = test_weights
                acc = self._compute_accuracy(X_val, y_val)
                
                if acc > best_acc:
                    best_acc = acc
                    best_weights = test_weights.copy()
        
        self.weights = best_weights
    
    def _gradient_optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray, lr: float = 0.01, n_iter: int = 100):
        """Gradient-based weight optimization."""
        # Get all model predictions
        predictions = [model(X_val).data for model in self.models]
        y_true = np.argmax(y_val, axis=1) if y_val.ndim > 1 else y_val
        
        # Optimize with gradient descent
        for _ in range(n_iter):
            # Compute weighted prediction
            weighted_pred = np.sum([pred * w for pred, w in zip(predictions, self.weights)], axis=0)
            pred_classes = np.argmax(weighted_pred, axis=1)
            
            # Compute gradient (simplified)
            for i in range(len(self.models)):
                correct = (pred_classes == y_true).astype(float)
                self.weights[i] += lr * np.mean(correct)
            
            # Normalize
            self.weights = np.maximum(self.weights, 0)
            self.weights = self.weights / np.sum(self.weights)
    
    def _compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy on dataset."""
        predictions = self.predict(X)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1) if y.ndim > 1 else y
        return np.mean(pred_classes == true_classes)


class AveragingEnsemble(EnsembleBase):
    """
    Simple averaging ensemble for regression.
    """
    
    def __init__(
        self,
        models: List[Module],
        weights: Optional[List[float]] = None
    ):
        super().__init__(models)
        
        if weights is None:
            self.weights = np.ones(len(models)) / len(models)
        else:
            self.weights = np.array(weights)
            self.weights = self.weights / np.sum(self.weights)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions by weighted averaging."""
        predictions = [model(X).data for model in self.models]
        weighted_preds = np.array([pred * w for pred, w in zip(predictions, self.weights)])
        return np.sum(weighted_preds, axis=0)


class StackingEnsemble(EnsembleBase):
    """
    Stacking ensemble with meta-learner.
    Base models make predictions, meta-model learns to combine them.
    """
    
    def __init__(
        self,
        base_models: List[Module],
        meta_model: Module,
        use_original_features: bool = True
    ):
        super().__init__(base_models)
        self.meta_model = meta_model
        self.use_original_features = use_original_features
    
    def fit_meta_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        optimizer,
        loss_fn: Callable,
        epochs: int = 10
    ):
        """
        Train the meta-model on base model predictions.
        
        Args:
            X_train: Training features
            y_train: Training labels
            optimizer: Optimizer for meta-model
            loss_fn: Loss function
            epochs: Number of training epochs
        """
        # Get base model predictions
        base_predictions = [model(X_train).data for model in self.models]
        
        # Stack predictions
        if self.use_original_features:
            meta_features = np.concatenate([X_train] + base_predictions, axis=1)
        else:
            meta_features = np.concatenate(base_predictions, axis=1)
        
        # Train meta-model
        for epoch in range(epochs):
            predictions = self.meta_model(meta_features)
            loss = loss_fn(predictions, y_train)
            
            loss.backward()
            optimizer.step()
            self.meta_model.zero_grad()
            
            if (epoch + 1) % 5 == 0:
                print(f"Meta-model epoch {epoch+1}/{epochs}, Loss: {loss.data:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make stacked ensemble predictions."""
        # Get base model predictions
        base_predictions = [model(X).data for model in self.models]
        
        # Stack predictions
        if self.use_original_features:
            meta_features = np.concatenate([X] + base_predictions, axis=1)
        else:
            meta_features = np.concatenate(base_predictions, axis=1)
        
        # Meta-model prediction
        return self.meta_model(meta_features).data


class BaggingEnsemble:
    """
    Bagging (Bootstrap Aggregating) ensemble.
    Trains multiple models on bootstrap samples of the data.
    """
    
    def __init__(
        self,
        model_factory: Callable[[], Module],
        n_models: int = 10,
        sample_ratio: float = 1.0,
        feature_ratio: float = 1.0
    ):
        self.model_factory = model_factory
        self.n_models = n_models
        self.sample_ratio = sample_ratio
        self.feature_ratio = feature_ratio
        self.models = []
        self.feature_indices = []
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        optimizer_factory: Callable,
        loss_fn: Callable,
        epochs: int = 10,
        verbose: bool = True
    ):
        """
        Train bagging ensemble.
        
        Args:
            X_train: Training features
            y_train: Training labels
            optimizer_factory: Function that creates optimizer for a model
            loss_fn: Loss function
            epochs: Training epochs per model
            verbose: Print progress
        """
        n_samples = len(X_train)
        n_features = X_train.shape[1]
        
        for i in range(self.n_models):
            if verbose:
                print(f"\nTraining model {i+1}/{self.n_models}")
            
            # Bootstrap sample
            sample_size = int(n_samples * self.sample_ratio)
            sample_indices = np.random.choice(n_samples, sample_size, replace=True)
            
            # Random feature subset
            feature_size = int(n_features * self.feature_ratio)
            feature_indices = np.random.choice(n_features, feature_size, replace=False)
            
            X_sample = X_train[sample_indices][:, feature_indices]
            y_sample = y_train[sample_indices]
            
            # Create and train model
            model = self.model_factory()
            optimizer = optimizer_factory(model)
            
            for epoch in range(epochs):
                predictions = model(X_sample)
                loss = loss_fn(predictions, y_sample)
                
                loss.backward()
                optimizer.step()
                model.zero_grad()
            
            self.models.append(model)
            self.feature_indices.append(feature_indices)
            
            if verbose:
                print(f"Model {i+1} final loss: {loss.data:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make bagging ensemble predictions."""
        predictions = []
        
        for model, feature_indices in zip(self.models, self.feature_indices):
            X_subset = X[:, feature_indices]
            pred = model(X_subset).data
            predictions.append(pred)
        
        # Average predictions
        return np.mean(predictions, axis=0)


class ModelFusion:
    """
    Advanced model fusion techniques including knowledge distillation.
    """
    
    @staticmethod
    def knowledge_distillation(
        teacher_model: Module,
        student_model: Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        optimizer,
        temperature: float = 3.0,
        alpha: float = 0.5,
        epochs: int = 20,
        verbose: bool = True
    ):
        """
        Train student model to mimic teacher model (knowledge distillation).
        
        Args:
            teacher_model: Pre-trained teacher model
            student_model: Student model to train
            X_train: Training features
            y_train: Training labels
            optimizer: Optimizer for student
            temperature: Softmax temperature for soft targets
            alpha: Weight for distillation loss vs hard target loss
            epochs: Training epochs
            verbose: Print progress
        """
        for epoch in range(epochs):
            # Get teacher predictions (soft targets)
            teacher_logits = teacher_model(X_train).data
            teacher_soft = ModelFusion._softmax_with_temperature(teacher_logits, temperature)
            
            # Get student predictions
            student_logits = student_model(X_train)
            student_soft = ModelFusion._softmax_with_temperature(student_logits.data, temperature)
            
            # Distillation loss (KL divergence)
            distillation_loss = np.mean(
                np.sum(teacher_soft * np.log(teacher_soft / (student_soft + 1e-10) + 1e-10), axis=1)
            )
            
            # Hard target loss
            hard_loss = np.mean(np.sum((student_logits.data - y_train) ** 2, axis=1))
            
            # Combined loss
            total_loss = alpha * distillation_loss + (1 - alpha) * hard_loss
            
            # Backward pass (simplified - using hard loss for gradients)
            loss_node = Node(hard_loss)
            student_logits.backward()
            optimizer.step()
            student_model.zero_grad()
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Distillation Loss: {distillation_loss:.4f}, Hard Loss: {hard_loss:.4f}")
    
    @staticmethod
    def _softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
        """Apply softmax with temperature scaling."""
        scaled_logits = logits / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    @staticmethod
    def model_averaging(
        models: List[Module],
        weights: Optional[List[float]] = None
    ) -> Module:
        """
        Create a new model by averaging weights of multiple models.
        All models must have the same architecture.
        
        Args:
            models: List of models to average
            weights: Optional weights for weighted averaging
            
        Returns:
            New model with averaged weights
        """
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        # Create new model (copy first model's structure)
        averaged_model = models[0].__class__.__new__(models[0].__class__)
        averaged_model.__dict__.update(models[0].__dict__)
        
        # Average parameters
        param_names = [name for name, _ in models[0].named_parameters()]
        
        for param_name in param_names:
            # Get parameters from all models
            params = []
            for model in models:
                for name, param in model.named_parameters():
                    if name == param_name:
                        params.append(param.data)
                        break
            
            # Weighted average
            avg_param = np.sum([p * w for p, w in zip(params, weights)], axis=0)
            
            # Set averaged parameter
            for name, param in averaged_model.named_parameters():
                if name == param_name:
                    param.data = avg_param
                    break
        
        return averaged_model
    
    @staticmethod
    def snapshot_ensemble(
        model: Module,
        checkpoints: List[str]
    ) -> VotingEnsemble:
        """
        Create ensemble from model snapshots saved during training.
        
        Args:
            model: Model class
            checkpoints: List of checkpoint file paths
            
        Returns:
            VotingEnsemble of snapshot models
        """
        models = []
        for checkpoint_path in checkpoints:
            snapshot_model = Module.load(checkpoint_path)
            models.append(snapshot_model)
        
        return VotingEnsemble(models, voting="soft")


class DiversityMetrics:
    """
    Measure diversity between ensemble members.
    Higher diversity often leads to better ensemble performance.
    """
    
    @staticmethod
    def disagreement_measure(
        predictions: List[np.ndarray],
        y_true: np.ndarray
    ) -> float:
        """
        Measure disagreement between ensemble members.
        
        Args:
            predictions: List of prediction arrays from each model
            y_true: True labels
            
        Returns:
            Disagreement score (0 to 1)
        """
        n_models = len(predictions)
        pred_classes = [np.argmax(pred, axis=1) for pred in predictions]
        
        disagreements = 0
        comparisons = 0
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                disagreements += np.sum(pred_classes[i] != pred_classes[j])
                comparisons += len(y_true)
        
        return disagreements / comparisons if comparisons > 0 else 0.0
    
    @staticmethod
    def q_statistic(
        predictions: List[np.ndarray],
        y_true: np.ndarray
    ) -> np.ndarray:
        """
        Compute Q-statistic between all pairs of models.
        Q close to 1 means models make similar errors (low diversity).
        Q close to -1 means models make different errors (high diversity).
        
        Returns:
            Matrix of Q-statistics between model pairs
        """
        n_models = len(predictions)
        pred_classes = [np.argmax(pred, axis=1) for pred in predictions]
        true_classes = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
        
        Q_matrix = np.zeros((n_models, n_models))
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                # Count agreements and disagreements
                both_correct = np.sum((pred_classes[i] == true_classes) & (pred_classes[j] == true_classes))
                both_wrong = np.sum((pred_classes[i] != true_classes) & (pred_classes[j] != true_classes))
                i_correct_j_wrong = np.sum((pred_classes[i] == true_classes) & (pred_classes[j] != true_classes))
                i_wrong_j_correct = np.sum((pred_classes[i] != true_classes) & (pred_classes[j] == true_classes))
                
                # Q-statistic
                numerator = both_correct * both_wrong - i_correct_j_wrong * i_wrong_j_correct
                denominator = both_correct * both_wrong + i_correct_j_wrong * i_wrong_j_correct
                
                Q = numerator / denominator if denominator != 0 else 0
                Q_matrix[i, j] = Q
                Q_matrix[j, i] = Q
        
        return Q_matrix
    
    @staticmethod
    def correlation_coefficient(
        predictions: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute correlation between model predictions.
        
        Returns:
            Correlation matrix
        """
        n_models = len(predictions)
        
        # Flatten predictions
        flat_preds = [pred.flatten() for pred in predictions]
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(flat_preds)
        
        return corr_matrix
