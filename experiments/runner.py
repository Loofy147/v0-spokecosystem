"""Experiment runner for training and evaluation."""
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import json

from core_engine import Node, CrossEntropyLoss, SGD, Adam
from automl.orchestrator.orchestrator import Orchestrator
from automl.fitness_functions import evaluate_model
from infrastructure import MetricsCollector, CacheManager, Timer
from .config import ExperimentConfig
from .checkpoint import CheckpointManager


class ExperimentRunner:
    """
    Unified experiment runner for supervised learning, AutoML, and RL experiments.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir, config.name)
        
        # Initialize infrastructure
        self.metrics_collector = MetricsCollector() if config.use_metrics else None
        self.cache_manager = CacheManager(
            use_redis=(config.cache_backend == 'redis')
        ) if config.use_cache else None
        
        # Set random seed
        np.random.seed(config.seed)
        
        # Results tracking
        self.results = {
            'train_losses': [],
            'val_accuracies': [],
            'test_accuracy': None,
            'best_val_accuracy': -float('inf'),
            'best_epoch': 0
        }
    
    def run(self) -> Dict[str, Any]:
        """
        Run the experiment based on configuration.
        
        Returns:
            Dictionary with experiment results
        """
        if self.config.model_type == "automl":
            return self.run_automl()
        elif self.config.model_type == "rl_agent":
            return self.run_rl()
        else:
            return self.run_supervised()
    
    def run_supervised(self) -> Dict[str, Any]:
        """Run supervised learning experiment."""
        print(f"[ExperimentRunner] Starting supervised learning experiment: {self.config.name}")
        
        # Load data
        X_train, y_train, X_val, y_val, X_test, y_test = self._load_data()
        
        # Build model
        from core_engine import Sequential, Linear, ReLU, Dropout
        
        input_size = X_train.shape[1]
        num_classes = int(y_train.max()) + 1
        hidden_size = self.config.model_config.get('hidden_size', 128)
        num_layers = self.config.model_config.get('num_layers', 2)
        
        layers = [Linear(input_size, hidden_size), ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([
                Dropout(0.2),
                Linear(hidden_size, hidden_size),
                ReLU()
            ])
        layers.append(Linear(hidden_size, num_classes))
        
        model = Sequential(*layers)
        
        # Setup optimizer
        if self.config.optimizer == "adam":
            optimizer = Adam(model.parameters(), lr=self.config.learning_rate)
        else:
            optimizer = SGD(model.parameters(), lr=self.config.learning_rate, 
                          weight_decay=self.config.weight_decay)
        
        loss_fn = CrossEntropyLoss()
        
        # Training loop
        print(f"[ExperimentRunner] Training for {self.config.epochs} epochs...")
        
        for epoch in range(self.config.epochs):
            with Timer(self.metrics_collector, 'training') as timer:
                # Training
                model.train()
                X_node = Node(X_train, requires_grad=False)
                y_node = Node(y_train, requires_grad=False)
                
                predictions = model(X_node)
                loss = loss_fn(predictions, y_node)
                
                model.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss = loss.value
                
                # Validation
                model.eval()
                X_val_node = Node(X_val, requires_grad=False)
                val_predictions = model(X_val_node)
                val_pred_classes = np.argmax(val_predictions.value, axis=1)
                val_accuracy = np.mean(val_pred_classes == y_val)
                
                # Track results
                self.results['train_losses'].append(float(train_loss))
                self.results['val_accuracies'].append(float(val_accuracy))
                
                if val_accuracy > self.results['best_val_accuracy']:
                    self.results['best_val_accuracy'] = val_accuracy
                    self.results['best_epoch'] = epoch
                
                # Update metrics
                if self.metrics_collector:
                    self.metrics_collector.inc_training_steps()
                    self.metrics_collector.set_current_fitness(val_accuracy)
                
                # Logging
                if epoch % self.config.log_every == 0:
                    print(f"Epoch {epoch}/{self.config.epochs} - "
                          f"Loss: {train_loss:.4f}, Val Acc: {val_accuracy:.4f}")
                
                # Checkpointing
                if epoch % self.config.save_every == 0:
                    model_state = {param.name: param.value for param in model.parameters()}
                    self.checkpoint_manager.save_checkpoint(
                        epoch=epoch,
                        model_state=model_state,
                        metrics={'train_loss': train_loss, 'val_accuracy': val_accuracy}
                    )
        
        # Final evaluation on test set
        model.eval()
        X_test_node = Node(X_test, requires_grad=False)
        test_predictions = model(X_test_node)
        test_pred_classes = np.argmax(test_predictions.value, axis=1)
        test_accuracy = np.mean(test_pred_classes == y_test)
        self.results['test_accuracy'] = float(test_accuracy)
        
        print(f"\n[ExperimentRunner] Training complete!")
        print(f"Best Val Accuracy: {self.results['best_val_accuracy']:.4f} (epoch {self.results['best_epoch']})")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return self.results
    
    def run_automl(self) -> Dict[str, Any]:
        """Run AutoML experiment."""
        print(f"[ExperimentRunner] Starting AutoML experiment: {self.config.name}")
        
        # Load data
        X_train, y_train, X_val, y_val, X_test, y_test = self._load_data()
        
        # Define objective function
        def objective_fn(config):
            return evaluate_model(
                config,
                X_train, y_train,
                X_val, y_val,
                epochs=10
            )
        
        # Create orchestrator
        orchestrator = Orchestrator(
            objective_fn=objective_fn,
            base_config={'architecture': [], 'lr': 1e-3, 'wd': 1e-4, 'batch_size': 128},
            population_size=self.config.automl_population_size,
            num_islands=self.config.automl_num_islands
        )
        
        # Run optimization
        best_performer, history = orchestrator.launch(
            n_trials=self.config.automl_generations,
            migrate_every=self.config.automl_migration_every,
            verbose=True
        )
        
        self.results['best_config'] = best_performer['config']
        self.results['best_fitness'] = best_performer['fitness']
        self.results['history'] = history
        
        print(f"\n[ExperimentRunner] AutoML complete!")
        print(f"Best Fitness: {best_performer['fitness']:.4f}")
        
        return self.results
    
    def run_rl(self) -> Dict[str, Any]:
        """Run RL experiment."""
        print(f"[ExperimentRunner] Starting RL experiment: {self.config.name}")
        print("[ExperimentRunner] RL training not yet implemented in this version")
        return self.results
    
    def _load_data(self):
        """Load and split dataset."""
        # For now, generate synthetic data
        # In production, this would load real datasets
        n_samples = 1000
        n_features = 20
        n_classes = 10
        
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, n_classes, n_samples)
        
        # Split data
        n_train = int(n_samples * self.config.train_split)
        n_val = int(n_samples * self.config.val_split)
        
        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
        X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def save_results(self, filepath: Optional[str] = None):
        """Save experiment results to JSON."""
        if filepath is None:
            filepath = Path(self.config.log_dir) / self.config.name / "results.json"
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"[ExperimentRunner] Results saved to {filepath}")


class GridSearchRunner:
    """Runner for grid search experiments."""
    
    def __init__(self, grid_config):
        self.grid_config = grid_config
        self.results = []
    
    def run(self) -> list:
        """Run all experiments in the grid."""
        configs = self.grid_config.generate_configs()
        
        print(f"[GridSearchRunner] Running {len(configs)} experiments...")
        
        for i, config in enumerate(configs):
            print(f"\n[GridSearchRunner] Experiment {i+1}/{len(configs)}")
            
            runner = ExperimentRunner(config)
            result = runner.run()
            
            self.results.append({
                'config': config.to_dict(),
                'results': result
            })
        
        # Find best configuration
        best_idx = max(range(len(self.results)), 
                      key=lambda i: self.results[i]['results'].get('best_val_accuracy', -float('inf')))
        
        print(f"\n[GridSearchRunner] Grid search complete!")
        print(f"Best configuration: Experiment {best_idx + 1}")
        print(f"Best Val Accuracy: {self.results[best_idx]['results']['best_val_accuracy']:.4f}")
        
        return self.results
