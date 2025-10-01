"""Observability infrastructure with Prometheus metrics support."""
from __future__ import annotations
import time
from typing import Dict, Any, Optional
from collections import defaultdict
import warnings

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    warnings.warn("Prometheus client not available. Install with: pip install prometheus-client")


class MetricsCollector:
    """
    Metrics collector with Prometheus support.
    Falls back to simple in-memory tracking if Prometheus is unavailable.
    """
    def __init__(self, namespace: str = 'spokecosystem', use_prometheus: bool = True):
        self.namespace = namespace
        self.use_prometheus = use_prometheus and PROMETHEUS_AVAILABLE
        
        if self.use_prometheus:
            self.registry = CollectorRegistry()
            self._init_prometheus_metrics()
        else:
            self._init_fallback_metrics()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        # Counters
        self.training_steps = Counter(
            f'{self.namespace}_training_steps_total',
            'Total number of training steps',
            registry=self.registry
        )
        
        self.evaluations = Counter(
            f'{self.namespace}_evaluations_total',
            'Total number of model evaluations',
            registry=self.registry
        )
        
        self.errors = Counter(
            f'{self.namespace}_errors_total',
            'Total number of errors',
            ['error_type'],
            registry=self.registry
        )
        
        # Histograms
        self.training_duration = Histogram(
            f'{self.namespace}_training_duration_seconds',
            'Training duration in seconds',
            registry=self.registry
        )
        
        self.evaluation_duration = Histogram(
            f'{self.namespace}_evaluation_duration_seconds',
            'Evaluation duration in seconds',
            registry=self.registry
        )
        
        # Gauges
        self.current_fitness = Gauge(
            f'{self.namespace}_current_fitness',
            'Current best fitness score',
            registry=self.registry
        )
        
        self.population_size = Gauge(
            f'{self.namespace}_population_size',
            'Current population size',
            registry=self.registry
        )
        
        self.cache_hit_rate = Gauge(
            f'{self.namespace}_cache_hit_rate',
            'Cache hit rate',
            registry=self.registry
        )
        
        # Info
        self.build_info = Info(
            f'{self.namespace}_build',
            'Build information',
            registry=self.registry
        )
    
    def _init_fallback_metrics(self):
        """Initialize fallback in-memory metrics."""
        self.metrics = defaultdict(float)
        self.histograms = defaultdict(list)
    
    def inc_training_steps(self, amount: float = 1.0):
        """Increment training steps counter."""
        if self.use_prometheus:
            self.training_steps.inc(amount)
        else:
            self.metrics['training_steps'] += amount
    
    def inc_evaluations(self, amount: float = 1.0):
        """Increment evaluations counter."""
        if self.use_prometheus:
            self.evaluations.inc(amount)
        else:
            self.metrics['evaluations'] += amount
    
    def inc_errors(self, error_type: str = 'unknown', amount: float = 1.0):
        """Increment errors counter."""
        if self.use_prometheus:
            self.errors.labels(error_type=error_type).inc(amount)
        else:
            self.metrics[f'errors_{error_type}'] += amount
    
    def observe_training_duration(self, duration: float):
        """Record training duration."""
        if self.use_prometheus:
            self.training_duration.observe(duration)
        else:
            self.histograms['training_duration'].append(duration)
    
    def observe_evaluation_duration(self, duration: float):
        """Record evaluation duration."""
        if self.use_prometheus:
            self.evaluation_duration.observe(duration)
        else:
            self.histograms['evaluation_duration'].append(duration)
    
    def set_current_fitness(self, fitness: float):
        """Set current best fitness."""
        if self.use_prometheus:
            self.current_fitness.set(fitness)
        else:
            self.metrics['current_fitness'] = fitness
    
    def set_population_size(self, size: int):
        """Set population size."""
        if self.use_prometheus:
            self.population_size.set(size)
        else:
            self.metrics['population_size'] = size
    
    def set_cache_hit_rate(self, rate: float):
        """Set cache hit rate."""
        if self.use_prometheus:
            self.cache_hit_rate.set(rate)
        else:
            self.metrics['cache_hit_rate'] = rate
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        if self.use_prometheus:
            return generate_latest(self.registry).decode('utf-8')
        else:
            # Return simple text format
            lines = []
            for key, value in self.metrics.items():
                lines.append(f"{key}: {value}")
            for key, values in self.histograms.items():
                if values:
                    lines.append(f"{key}_count: {len(values)}")
                    lines.append(f"{key}_sum: {sum(values)}")
                    lines.append(f"{key}_avg: {sum(values) / len(values)}")
            return '\n'.join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get metrics as dictionary."""
        if self.use_prometheus:
            # Parse Prometheus metrics
            return {'prometheus_metrics': self.get_metrics()}
        else:
            stats = dict(self.metrics)
            for key, values in self.histograms.items():
                if values:
                    stats[f'{key}_count'] = len(values)
                    stats[f'{key}_sum'] = sum(values)
                    stats[f'{key}_avg'] = sum(values) / len(values)
            return stats


class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None, metric_name: Optional[str] = None):
        self.metrics_collector = metrics_collector
        self.metric_name = metric_name
        self.start_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - self.start_time
        
        if self.metrics_collector and self.metric_name:
            if self.metric_name == 'training':
                self.metrics_collector.observe_training_duration(self.duration)
            elif self.metric_name == 'evaluation':
                self.metrics_collector.observe_evaluation_duration(self.duration)
