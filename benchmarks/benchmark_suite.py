"""
Comprehensive Benchmark Suite & Validation Framework
Scientific validation, performance benchmarking, and reproducibility testing.
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple
import time
import json
from pathlib import Path
from datetime import datetime
from core_engine.nn_modules import Module
from core_engine.optimizers import Optimizer


class BenchmarkResult:
    """Container for benchmark results."""
    
    def __init__(
        self,
        benchmark_name: str,
        model_name: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict] = None
    ):
        self.benchmark_name = benchmark_name
        self.model_name = model_name
        self.metrics = metrics
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "model_name": self.model_name,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class PerformanceBenchmark:
    """
    Benchmark model performance (speed, memory, throughput).
    """
    
    @staticmethod
    def measure_inference_latency(
        model: Module,
        input_shape: Tuple[int, ...],
        n_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Measure inference latency statistics.
        
        Args:
            model: Model to benchmark
            input_shape: Shape of input data
            n_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Latency statistics
        """
        # Warmup
        for _ in range(warmup_iterations):
            X = np.random.randn(*input_shape)
            _ = model(X)
        
        # Benchmark
        latencies = []
        for _ in range(n_iterations):
            X = np.random.randn(*input_shape)
            
            start = time.time()
            _ = model(X)
            latency = (time.time() - start) * 1000  # Convert to ms
            
            latencies.append(latency)
        
        return {
            "mean_ms": float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99))
        }
    
    @staticmethod
    def measure_throughput(
        model: Module,
        input_shape: Tuple[int, ...],
        duration_seconds: float = 10.0
    ) -> Dict[str, float]:
        """
        Measure throughput (samples per second).
        
        Args:
            model: Model to benchmark
            input_shape: Shape of input data
            duration_seconds: Duration to run benchmark
            
        Returns:
            Throughput metrics
        """
        start_time = time.time()
        n_samples = 0
        
        while (time.time() - start_time) < duration_seconds:
            X = np.random.randn(*input_shape)
            _ = model(X)
            n_samples += input_shape[0]
        
        elapsed = time.time() - start_time
        throughput = n_samples / elapsed
        
        return {
            "samples_per_second": float(throughput),
            "duration_seconds": float(elapsed),
            "total_samples": int(n_samples)
        }
    
    @staticmethod
    def measure_training_speed(
        model: Module,
        optimizer: Optimizer,
        loss_fn: Callable,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        n_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Measure training speed (iterations per second).
        
        Args:
            model: Model to train
            optimizer: Optimizer
            loss_fn: Loss function
            input_shape: Input data shape
            output_shape: Output data shape
            n_iterations: Number of training iterations
            
        Returns:
            Training speed metrics
        """
        iteration_times = []
        
        for _ in range(n_iterations):
            X = np.random.randn(*input_shape)
            y = np.random.randn(*output_shape)
            
            start = time.time()
            
            # Forward pass
            predictions = model(X)
            loss = loss_fn(predictions, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            model.zero_grad()
            
            iteration_time = time.time() - start
            iteration_times.append(iteration_time)
        
        return {
            "mean_iteration_time_ms": float(np.mean(iteration_times) * 1000),
            "iterations_per_second": float(1.0 / np.mean(iteration_times)),
            "total_time_seconds": float(np.sum(iteration_times))
        }
    
    @staticmethod
    def profile_model_size(model: Module) -> Dict[str, Any]:
        """
        Profile model size and parameter count.
        
        Args:
            model: Model to profile
            
        Returns:
            Model size metrics
        """
        total_params = 0
        trainable_params = 0
        layer_params = {}
        
        for name, param in model.named_parameters():
            n_params = np.prod(param.data.shape)
            total_params += n_params
            
            if param.requires_grad:
                trainable_params += n_params
            
            layer_params[name] = {
                "shape": param.data.shape,
                "n_params": int(n_params),
                "trainable": param.requires_grad
            }
        
        # Estimate memory size (assuming float32)
        memory_mb = (total_params * 4) / (1024 * 1024)
        
        return {
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "non_trainable_parameters": int(total_params - trainable_params),
            "estimated_memory_mb": float(memory_mb),
            "layer_parameters": layer_params
        }


class AccuracyBenchmark:
    """
    Benchmark model accuracy on standard datasets and tasks.
    """
    
    @staticmethod
    def classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "macro"
    ) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities or labels
            average: Averaging method ('macro', 'micro', 'weighted')
            
        Returns:
            Classification metrics
        """
        # Convert to class labels if probabilities
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            pred_classes = np.argmax(y_pred, axis=1)
        else:
            pred_classes = y_pred.flatten()
        
        if y_true.ndim > 1:
            true_classes = np.argmax(y_true, axis=1)
        else:
            true_classes = y_true.flatten()
        
        # Overall accuracy
        accuracy = np.mean(pred_classes == true_classes)
        
        # Per-class metrics
        n_classes = int(max(np.max(true_classes), np.max(pred_classes)) + 1)
        
        precisions = []
        recalls = []
        f1_scores = []
        
        for class_idx in range(n_classes):
            tp = np.sum((pred_classes == class_idx) & (true_classes == class_idx))
            fp = np.sum((pred_classes == class_idx) & (true_classes != class_idx))
            fn = np.sum((pred_classes != class_idx) & (true_classes == class_idx))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        # Aggregate metrics
        if average == "macro":
            avg_precision = np.mean(precisions)
            avg_recall = np.mean(recalls)
            avg_f1 = np.mean(f1_scores)
        elif average == "weighted":
            weights = [np.sum(true_classes == i) for i in range(n_classes)]
            total_weight = sum(weights)
            avg_precision = sum(p * w for p, w in zip(precisions, weights)) / total_weight
            avg_recall = sum(r * w for r, w in zip(recalls, weights)) / total_weight
            avg_f1 = sum(f * w for f, w in zip(f1_scores, weights)) / total_weight
        else:  # micro
            all_tp = sum(np.sum((pred_classes == i) & (true_classes == i)) for i in range(n_classes))
            all_fp = sum(np.sum((pred_classes == i) & (true_classes != i)) for i in range(n_classes))
            all_fn = sum(np.sum((pred_classes != i) & (true_classes == i)) for i in range(n_classes))
            
            avg_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
            avg_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
            avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        return {
            "accuracy": float(accuracy),
            "precision": float(avg_precision),
            "recall": float(avg_recall),
            "f1_score": float(avg_f1)
        }
    
    @staticmethod
    def regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Regression metrics
        """
        # Mean Squared Error
        mse = np.mean((y_true - y_pred) ** 2)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(y_true - y_pred))
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2),
            "mape": float(mape)
        }


class ReproducibilityTest:
    """
    Test model reproducibility and determinism.
    """
    
    @staticmethod
    def test_determinism(
        model_factory: Callable[[], Module],
        X: np.ndarray,
        n_runs: int = 5,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Test if model produces deterministic results.
        
        Args:
            model_factory: Function that creates a new model
            X: Input data
            n_runs: Number of runs to test
            seed: Random seed
            
        Returns:
            Determinism test results
        """
        predictions = []
        
        for run in range(n_runs):
            np.random.seed(seed)
            model = model_factory()
            pred = model(X).data
            predictions.append(pred)
        
        # Check if all predictions are identical
        is_deterministic = all(np.allclose(predictions[0], pred) for pred in predictions[1:])
        
        # Calculate variance across runs
        pred_array = np.array(predictions)
        variance = np.var(pred_array, axis=0)
        max_variance = float(np.max(variance))
        mean_variance = float(np.mean(variance))
        
        return {
            "is_deterministic": is_deterministic,
            "max_variance": max_variance,
            "mean_variance": mean_variance,
            "n_runs": n_runs
        }
    
    @staticmethod
    def test_training_reproducibility(
        model_factory: Callable[[], Module],
        optimizer_factory: Callable[[Module], Optimizer],
        loss_fn: Callable,
        X: np.ndarray,
        y: np.ndarray,
        n_epochs: int = 10,
        n_runs: int = 3,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Test if training is reproducible.
        
        Args:
            model_factory: Function that creates a new model
            optimizer_factory: Function that creates optimizer
            loss_fn: Loss function
            X: Training features
            y: Training labels
            n_epochs: Number of training epochs
            n_runs: Number of independent runs
            seed: Random seed
            
        Returns:
            Training reproducibility results
        """
        final_losses = []
        final_weights = []
        
        for run in range(n_runs):
            np.random.seed(seed)
            model = model_factory()
            optimizer = optimizer_factory(model)
            
            # Train
            for epoch in range(n_epochs):
                predictions = model(X)
                loss = loss_fn(predictions, y)
                loss.backward()
                optimizer.step()
                model.zero_grad()
            
            final_losses.append(float(loss.data))
            
            # Save final weights
            weights = [param.data.copy() for param in model.parameters()]
            final_weights.append(weights)
        
        # Check reproducibility
        loss_variance = np.var(final_losses)
        
        # Check if weights are identical
        weights_identical = all(
            all(np.allclose(final_weights[0][i], final_weights[j][i]) 
                for i in range(len(final_weights[0])))
            for j in range(1, n_runs)
        )
        
        return {
            "is_reproducible": weights_identical,
            "final_losses": final_losses,
            "loss_variance": float(loss_variance),
            "n_runs": n_runs
        }


class RobustnessTest:
    """
    Test model robustness to various perturbations.
    """
    
    @staticmethod
    def test_noise_robustness(
        model: Module,
        X: np.ndarray,
        noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2],
        n_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Test robustness to input noise.
        
        Args:
            model: Model to test
            X: Clean input data
            noise_levels: List of noise standard deviations
            n_samples: Number of noisy samples per level
            
        Returns:
            Noise robustness results
        """
        baseline_pred = model(X).data
        baseline_class = np.argmax(baseline_pred, axis=1)
        
        results = {}
        
        for noise_level in noise_levels:
            predictions_match = []
            
            for _ in range(n_samples):
                # Add Gaussian noise
                noise = np.random.normal(0, noise_level, X.shape)
                X_noisy = X + noise
                
                # Predict
                noisy_pred = model(X_noisy).data
                noisy_class = np.argmax(noisy_pred, axis=1)
                
                # Check if prediction matches
                match_rate = np.mean(baseline_class == noisy_class)
                predictions_match.append(match_rate)
            
            results[f"noise_{noise_level}"] = {
                "mean_match_rate": float(np.mean(predictions_match)),
                "std_match_rate": float(np.std(predictions_match)),
                "min_match_rate": float(np.min(predictions_match))
            }
        
        return results
    
    @staticmethod
    def test_distribution_shift(
        model: Module,
        X_train: np.ndarray,
        X_test_shifted: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Test performance under distribution shift.
        
        Args:
            model: Trained model
            X_train: Training distribution data
            X_test_shifted: Test data with distribution shift
            y_test: Test labels
            
        Returns:
            Distribution shift metrics
        """
        # Baseline performance on training distribution
        train_pred = model(X_train).data
        
        # Performance on shifted distribution
        test_pred = model(X_test_shifted).data
        
        # Calculate metrics
        test_metrics = AccuracyBenchmark.classification_metrics(y_test, test_pred)
        
        # Measure distribution difference
        train_mean = np.mean(X_train, axis=0)
        test_mean = np.mean(X_test_shifted, axis=0)
        mean_shift = np.linalg.norm(train_mean - test_mean)
        
        return {
            **test_metrics,
            "distribution_shift_magnitude": float(mean_shift)
        }


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for model evaluation.
    """
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def run_full_benchmark(
        self,
        model: Module,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        input_shape: Tuple[int, ...],
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        """
        Run complete benchmark suite.
        
        Args:
            model: Model to benchmark
            model_name: Name of the model
            X_test: Test features
            y_test: Test labels
            input_shape: Input shape for performance tests
            task_type: 'classification' or 'regression'
            
        Returns:
            Complete benchmark results
        """
        print(f"\n{'='*60}")
        print(f"Running Full Benchmark Suite for: {model_name}")
        print(f"{'='*60}\n")
        
        results = {
            "model_name": model_name,
            "task_type": task_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # 1. Performance Benchmarks
        print("1. Performance Benchmarks...")
        results["performance"] = {
            "latency": PerformanceBenchmark.measure_inference_latency(model, input_shape),
            "throughput": PerformanceBenchmark.measure_throughput(model, input_shape),
            "model_size": PerformanceBenchmark.profile_model_size(model)
        }
        
        # 2. Accuracy Benchmarks
        print("2. Accuracy Benchmarks...")
        predictions = model(X_test).data
        
        if task_type == "classification":
            results["accuracy"] = AccuracyBenchmark.classification_metrics(y_test, predictions)
        else:
            results["accuracy"] = AccuracyBenchmark.regression_metrics(y_test, predictions)
        
        # 3. Robustness Tests
        print("3. Robustness Tests...")
        results["robustness"] = {
            "noise_robustness": RobustnessTest.test_noise_robustness(model, X_test[:100])
        }
        
        # Save results
        self._save_results(model_name, results)
        
        print(f"\n{'='*60}")
        print("Benchmark Complete!")
        print(f"{'='*60}\n")
        
        return results
    
    def compare_models(
        self,
        results_list: List[Dict[str, Any]],
        metrics: List[str] = ["accuracy.accuracy", "performance.latency.mean_ms"]
    ) -> Dict[str, Any]:
        """
        Compare multiple model benchmark results.
        
        Args:
            results_list: List of benchmark results
            metrics: Metrics to compare
            
        Returns:
            Comparison results
        """
        comparison = {
            "models": [r["model_name"] for r in results_list],
            "metrics": {}
        }
        
        for metric_path in metrics:
            metric_values = {}
            
            for result in results_list:
                # Navigate nested dict
                value = result
                for key in metric_path.split('.'):
                    value = value.get(key, None)
                    if value is None:
                        break
                
                if value is not None:
                    metric_values[result["model_name"]] = value
            
            comparison["metrics"][metric_path] = metric_values
        
        return comparison
    
    def _save_results(self, model_name: str, results: Dict[str, Any]):
        """Save benchmark results to file."""
        filename = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable benchmark report."""
        report = []
        report.append(f"\n{'='*60}")
        report.append(f"Benchmark Report: {results['model_name']}")
        report.append(f"{'='*60}\n")
        
        # Performance
        if "performance" in results:
            report.append("Performance Metrics:")
            report.append(f"  Latency (mean): {results['performance']['latency']['mean_ms']:.2f} ms")
            report.append(f"  Throughput: {results['performance']['throughput']['samples_per_second']:.2f} samples/sec")
            report.append(f"  Model Size: {results['performance']['model_size']['total_parameters']:,} parameters")
            report.append("")
        
        # Accuracy
        if "accuracy" in results:
            report.append("Accuracy Metrics:")
            for metric, value in results['accuracy'].items():
                report.append(f"  {metric}: {value:.4f}")
            report.append("")
        
        # Robustness
        if "robustness" in results:
            report.append("Robustness Tests:")
            for test_name, test_results in results['robustness'].items():
                report.append(f"  {test_name}:")
                for key, value in test_results.items():
                    if isinstance(value, dict):
                        report.append(f"    {key}:")
                        for k, v in value.items():
                            report.append(f"      {k}: {v:.4f}")
                    else:
                        report.append(f"    {key}: {value:.4f}")
            report.append("")
        
        report.append(f"{'='*60}\n")
        
        return "\n".join(report)
