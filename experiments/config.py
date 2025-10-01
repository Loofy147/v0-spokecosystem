"""Experiment configuration management."""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
import yaml
import json
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    # Experiment metadata
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Model configuration
    model_type: str = "mlp"  # mlp, automl, rl_agent
    model_config: Dict[str, Any] = field(default_factory=dict)
    
    # Training configuration
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 100
    weight_decay: float = 1e-4
    optimizer: str = "sgd"  # sgd, adam, adamw
    
    # Data configuration
    dataset: str = "mnist"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # AutoML configuration (if model_type == "automl")
    automl_population_size: int = 24
    automl_num_islands: int = 4
    automl_generations: int = 20
    automl_migration_every: int = 5
    
    # RL configuration (if model_type == "rl_agent")
    rl_env: str = "CartPole-v1"
    rl_episodes: int = 1000
    rl_gamma: float = 0.99
    
    # Infrastructure configuration
    use_cache: bool = True
    use_metrics: bool = True
    cache_backend: str = "lru"  # lru, redis
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10
    keep_best_only: bool = False
    
    # Logging
    log_dir: str = "logs"
    log_every: int = 1
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    
    # Reproducibility
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_yaml(self, filepath: str):
        """Save config to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def to_json(self, filepath: str):
        """Save config to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'ExperimentConfig':
        """Load config from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'ExperimentConfig':
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class GridSearchConfig:
    """Configuration for grid search experiments."""
    base_config: ExperimentConfig
    param_grid: Dict[str, List[Any]]
    n_trials: Optional[int] = None  # None = exhaustive search
    random_search: bool = False
    
    def generate_configs(self) -> List[ExperimentConfig]:
        """Generate all experiment configurations from grid."""
        import itertools
        
        configs = []
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        if self.random_search and self.n_trials:
            # Random search
            import random
            for _ in range(self.n_trials):
                config = ExperimentConfig(**asdict(self.base_config))
                for param_name in param_names:
                    value = random.choice(self.param_grid[param_name])
                    setattr(config, param_name, value)
                configs.append(config)
        else:
            # Grid search
            for values in itertools.product(*param_values):
                config = ExperimentConfig(**asdict(self.base_config))
                for param_name, value in zip(param_names, values):
                    setattr(config, param_name, value)
                configs.append(config)
                
                if self.n_trials and len(configs) >= self.n_trials:
                    break
        
        return configs
