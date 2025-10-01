"""
Production Model Serving Infrastructure
Includes REST API server, batch prediction, model monitoring, and A/B testing.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
import json
import time
from pathlib import Path
from datetime import datetime
from collections import deque
from core_engine.nn_modules import Module
from production.model_registry import ModelRegistry


class PredictionRequest:
    """Represents a prediction request."""
    
    def __init__(self, request_id: str, features: np.ndarray, metadata: Optional[Dict] = None):
        self.request_id = request_id
        self.features = features
        self.metadata = metadata or {}
        self.timestamp = time.time()


class PredictionResponse:
    """Represents a prediction response."""
    
    def __init__(
        self,
        request_id: str,
        predictions: np.ndarray,
        model_version: str,
        latency_ms: float,
        metadata: Optional[Dict] = None
    ):
        self.request_id = request_id
        self.predictions = predictions
        self.model_version = model_version
        self.latency_ms = latency_ms
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "request_id": self.request_id,
            "predictions": self.predictions.tolist(),
            "model_version": self.model_version,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class ModelMonitor:
    """
    Monitor model performance in production.
    Tracks latency, throughput, prediction distribution, and data drift.
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.prediction_counts = deque(maxlen=window_size)
        self.error_counts = deque(maxlen=window_size)
        self.prediction_distributions = deque(maxlen=window_size)
        self.feature_stats = {}
        self.start_time = time.time()
        self.total_requests = 0
        self.total_errors = 0
    
    def log_prediction(
        self,
        latency_ms: float,
        predictions: np.ndarray,
        features: Optional[np.ndarray] = None,
        error: bool = False
    ):
        """Log a prediction for monitoring."""
        self.latencies.append(latency_ms)
        self.prediction_counts.append(1)
        self.error_counts.append(1 if error else 0)
        self.total_requests += 1
        
        if error:
            self.total_errors += 1
        
        # Track prediction distribution
        if predictions.ndim > 1:
            pred_class = np.argmax(predictions, axis=1)[0]
        else:
            pred_class = predictions[0]
        self.prediction_distributions.append(pred_class)
        
        # Track feature statistics for drift detection
        if features is not None:
            self._update_feature_stats(features)
    
    def _update_feature_stats(self, features: np.ndarray):
        """Update running statistics for features."""
        if len(self.feature_stats) == 0:
            self.feature_stats = {
                "mean": features.mean(axis=0),
                "std": features.std(axis=0),
                "min": features.min(axis=0),
                "max": features.max(axis=0),
                "count": len(features)
            }
        else:
            # Update running statistics
            n = self.feature_stats["count"]
            m = len(features)
            
            # Update mean
            old_mean = self.feature_stats["mean"]
            new_mean = (n * old_mean + m * features.mean(axis=0)) / (n + m)
            
            self.feature_stats["mean"] = new_mean
            self.feature_stats["min"] = np.minimum(self.feature_stats["min"], features.min(axis=0))
            self.feature_stats["max"] = np.maximum(self.feature_stats["max"], features.max(axis=0))
            self.feature_stats["count"] = n + m
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current monitoring metrics."""
        if len(self.latencies) == 0:
            return {}
        
        uptime_hours = (time.time() - self.start_time) / 3600
        
        metrics = {
            "latency": {
                "mean_ms": float(np.mean(self.latencies)),
                "median_ms": float(np.median(self.latencies)),
                "p95_ms": float(np.percentile(self.latencies, 95)),
                "p99_ms": float(np.percentile(self.latencies, 99)),
                "max_ms": float(np.max(self.latencies))
            },
            "throughput": {
                "requests_per_second": len(self.prediction_counts) / (uptime_hours * 3600) if uptime_hours > 0 else 0,
                "total_requests": self.total_requests
            },
            "errors": {
                "error_rate": self.total_errors / self.total_requests if self.total_requests > 0 else 0,
                "total_errors": self.total_errors
            },
            "predictions": {
                "distribution": self._get_prediction_distribution()
            },
            "uptime_hours": uptime_hours
        }
        
        return metrics
    
    def _get_prediction_distribution(self) -> Dict[int, int]:
        """Get distribution of predicted classes."""
        if len(self.prediction_distributions) == 0:
            return {}
        
        unique, counts = np.unique(list(self.prediction_distributions), return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}
    
    def detect_drift(self, reference_stats: Dict[str, np.ndarray], threshold: float = 0.1) -> Dict[str, bool]:
        """
        Detect data drift by comparing current feature statistics to reference.
        
        Args:
            reference_stats: Reference statistics from training data
            threshold: Drift threshold (relative change)
            
        Returns:
            Dictionary indicating drift for each feature
        """
        if len(self.feature_stats) == 0:
            return {}
        
        drift_detected = {}
        
        # Compare means
        if "mean" in reference_stats and "mean" in self.feature_stats:
            ref_mean = reference_stats["mean"]
            curr_mean = self.feature_stats["mean"]
            
            relative_change = np.abs(curr_mean - ref_mean) / (np.abs(ref_mean) + 1e-8)
            
            for i, change in enumerate(relative_change):
                drift_detected[f"feature_{i}"] = bool(change > threshold)
        
        return drift_detected


class ABTestManager:
    """
    Manage A/B testing between different model versions.
    """
    
    def __init__(self):
        self.experiments = {}
        self.results = {}
    
    def create_experiment(
        self,
        experiment_id: str,
        model_a: Module,
        model_b: Module,
        traffic_split: float = 0.5,
        metadata: Optional[Dict] = None
    ):
        """
        Create A/B test experiment.
        
        Args:
            experiment_id: Unique experiment identifier
            model_a: Control model (A)
            model_b: Treatment model (B)
            traffic_split: Fraction of traffic to model B (0 to 1)
            metadata: Additional experiment metadata
        """
        self.experiments[experiment_id] = {
            "model_a": model_a,
            "model_b": model_b,
            "traffic_split": traffic_split,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat()
        }
        
        self.results[experiment_id] = {
            "model_a": {"requests": 0, "latencies": [], "predictions": []},
            "model_b": {"requests": 0, "latencies": [], "predictions": []}
        }
    
    def route_request(self, experiment_id: str, features: np.ndarray) -> Tuple[str, np.ndarray, float]:
        """
        Route request to model A or B based on traffic split.
        
        Returns:
            (model_variant, predictions, latency_ms)
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        # Randomly assign to A or B
        use_model_b = np.random.random() < experiment["traffic_split"]
        
        if use_model_b:
            model = experiment["model_b"]
            variant = "model_b"
        else:
            model = experiment["model_a"]
            variant = "model_a"
        
        # Make prediction and measure latency
        start_time = time.time()
        predictions = model(features).data
        latency_ms = (time.time() - start_time) * 1000
        
        # Log results
        self.results[experiment_id][variant]["requests"] += 1
        self.results[experiment_id][variant]["latencies"].append(latency_ms)
        self.results[experiment_id][variant]["predictions"].append(predictions)
        
        return variant, predictions, latency_ms
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get A/B test results."""
        if experiment_id not in self.results:
            return {}
        
        results = self.results[experiment_id]
        
        summary = {}
        for variant in ["model_a", "model_b"]:
            variant_results = results[variant]
            
            if len(variant_results["latencies"]) > 0:
                summary[variant] = {
                    "requests": variant_results["requests"],
                    "mean_latency_ms": float(np.mean(variant_results["latencies"])),
                    "p95_latency_ms": float(np.percentile(variant_results["latencies"], 95)),
                    "prediction_distribution": self._get_variant_distribution(variant_results["predictions"])
                }
            else:
                summary[variant] = {"requests": 0}
        
        return summary
    
    def _get_variant_distribution(self, predictions: List[np.ndarray]) -> Dict:
        """Get prediction distribution for a variant."""
        if len(predictions) == 0:
            return {}
        
        all_preds = np.concatenate([np.argmax(p, axis=1) if p.ndim > 1 else p for p in predictions])
        unique, counts = np.unique(all_preds, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}


class ModelServer:
    """
    Production model serving infrastructure.
    Handles model loading, prediction serving, monitoring, and A/B testing.
    """
    
    def __init__(
        self,
        registry: ModelRegistry,
        enable_monitoring: bool = True,
        enable_caching: bool = True,
        cache_size: int = 1000
    ):
        self.registry = registry
        self.loaded_models = {}
        self.enable_monitoring = enable_monitoring
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self.prediction_cache = {}
        
        if enable_monitoring:
            self.monitors = {}
        
        self.ab_test_manager = ABTestManager()
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = "production"
    ) -> bool:
        """
        Load model from registry into memory.
        
        Args:
            model_name: Name of the model
            version: Specific version (if None, loads latest)
            stage: Filter by stage
            
        Returns:
            True if successful
        """
        model = self.registry.get_model(model_name, version, stage)
        
        if model is None:
            print(f"Model {model_name} not found")
            return False
        
        model_version = self.registry.get_model_version(model_name, version, stage)
        model_key = f"{model_name}:{model_version.version}"
        
        self.loaded_models[model_key] = {
            "model": model,
            "version": model_version.version,
            "metadata": model_version.metadata,
            "loaded_at": time.time()
        }
        
        if self.enable_monitoring:
            self.monitors[model_key] = ModelMonitor()
        
        print(f"Loaded model {model_key}")
        return True
    
    def predict(
        self,
        model_name: str,
        features: np.ndarray,
        version: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> PredictionResponse:
        """
        Make prediction using loaded model.
        
        Args:
            model_name: Name of the model
            features: Input features
            version: Specific version (if None, uses latest loaded)
            request_id: Optional request identifier
            
        Returns:
            PredictionResponse
        """
        # Find model
        model_key = self._find_model_key(model_name, version)
        
        if model_key is None:
            raise ValueError(f"Model {model_name} not loaded")
        
        model_info = self.loaded_models[model_key]
        
        # Generate request ID if not provided
        if request_id is None:
            request_id = f"{model_name}_{int(time.time() * 1000)}"
        
        # Check cache
        if self.enable_caching:
            cache_key = self._get_cache_key(features)
            if cache_key in self.prediction_cache:
                cached_response = self.prediction_cache[cache_key]
                cached_response.request_id = request_id
                return cached_response
        
        # Make prediction
        start_time = time.time()
        try:
            predictions = model_info["model"](features).data
            latency_ms = (time.time() - start_time) * 1000
            error = False
        except Exception as e:
            print(f"Prediction error: {e}")
            predictions = np.array([])
            latency_ms = (time.time() - start_time) * 1000
            error = True
        
        # Create response
        response = PredictionResponse(
            request_id=request_id,
            predictions=predictions,
            model_version=model_info["version"],
            latency_ms=latency_ms,
            metadata={"model_name": model_name}
        )
        
        # Log to monitor
        if self.enable_monitoring and model_key in self.monitors:
            self.monitors[model_key].log_prediction(latency_ms, predictions, features, error)
        
        # Cache response
        if self.enable_caching and not error:
            cache_key = self._get_cache_key(features)
            self.prediction_cache[cache_key] = response
            
            # Limit cache size
            if len(self.prediction_cache) > self.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.prediction_cache))
                del self.prediction_cache[oldest_key]
        
        return response
    
    def batch_predict(
        self,
        model_name: str,
        features_batch: List[np.ndarray],
        version: Optional[str] = None,
        batch_size: int = 32
    ) -> List[PredictionResponse]:
        """
        Make batch predictions efficiently.
        
        Args:
            model_name: Name of the model
            features_batch: List of feature arrays
            version: Specific version
            batch_size: Batch size for processing
            
        Returns:
            List of PredictionResponse
        """
        responses = []
        
        # Process in batches
        for i in range(0, len(features_batch), batch_size):
            batch = features_batch[i:i + batch_size]
            
            # Stack features
            stacked_features = np.vstack(batch)
            
            # Make prediction
            response = self.predict(model_name, stacked_features, version)
            
            # Split response for each sample
            for j, features in enumerate(batch):
                sample_response = PredictionResponse(
                    request_id=f"{response.request_id}_{j}",
                    predictions=response.predictions[j:j+1],
                    model_version=response.model_version,
                    latency_ms=response.latency_ms / len(batch),
                    metadata=response.metadata
                )
                responses.append(sample_response)
        
        return responses
    
    def get_model_metrics(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get monitoring metrics for a model."""
        model_key = self._find_model_key(model_name, version)
        
        if model_key is None or model_key not in self.monitors:
            return {}
        
        return self.monitors[model_key].get_metrics()
    
    def _find_model_key(self, model_name: str, version: Optional[str] = None) -> Optional[str]:
        """Find model key in loaded models."""
        if version:
            model_key = f"{model_name}:{version}"
            if model_key in self.loaded_models:
                return model_key
        else:
            # Find latest version
            matching_keys = [k for k in self.loaded_models.keys() if k.startswith(f"{model_name}:")]
            if matching_keys:
                return matching_keys[-1]
        
        return None
    
    def _get_cache_key(self, features: np.ndarray) -> str:
        """Generate cache key from features."""
        return str(hash(features.tobytes()))
    
    def health_check(self) -> Dict[str, Any]:
        """Check server health status."""
        return {
            "status": "healthy",
            "loaded_models": len(self.loaded_models),
            "models": list(self.loaded_models.keys()),
            "monitoring_enabled": self.enable_monitoring,
            "caching_enabled": self.enable_caching,
            "cache_size": len(self.prediction_cache) if self.enable_caching else 0
        }


class BatchProcessor:
    """
    Efficient batch processing for large-scale inference.
    """
    
    def __init__(self, model: Module, batch_size: int = 32):
        self.model = model
        self.batch_size = batch_size
    
    def process_file(
        self,
        input_file: str,
        output_file: str,
        feature_columns: Optional[List[int]] = None,
        verbose: bool = True
    ):
        """
        Process predictions from input file and save to output file.
        
        Args:
            input_file: Path to input CSV/NPY file
            output_file: Path to output file
            feature_columns: Columns to use as features (for CSV)
            verbose: Print progress
        """
        # Load data
        if input_file.endswith('.npy'):
            data = np.load(input_file)
        elif input_file.endswith('.csv'):
            data = np.loadtxt(input_file, delimiter=',')
            if feature_columns:
                data = data[:, feature_columns]
        else:
            raise ValueError("Unsupported file format")
        
        # Process in batches
        all_predictions = []
        n_batches = (len(data) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            predictions = self.model(batch).data
            all_predictions.append(predictions)
            
            if verbose and (i // self.batch_size) % 10 == 0:
                print(f"Processed batch {i // self.batch_size + 1}/{n_batches}")
        
        # Concatenate and save
        all_predictions = np.vstack(all_predictions)
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_file, all_predictions)
        
        if verbose:
            print(f"Saved predictions to {output_file}")
