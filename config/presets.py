"""
Predefined configuration presets for common use cases.
"""
from typing import Dict, Any


class ConfigPresets:
    """Collection of predefined configuration presets."""
    
    @staticmethod
    def quick_experiment() -> Dict[str, Any]:
        """Fast configuration for quick experiments."""
        return {
            'training': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 10,
                'optimizer': 'adam'
            },
            'model': {
                'architecture': 'simple_mlp',
                'hidden_dims': [128, 64]
            },
            'logging': {
                'log_level': 'INFO',
                'log_dir': 'logs/quick'
            }
        }
    
    @staticmethod
    def production() -> Dict[str, Any]:
        """Production-ready configuration."""
        return {
            'training': {
                'learning_rate': 0.0001,
                'batch_size': 64,
                'epochs': 100,
                'optimizer': 'adamw',
                'weight_decay': 0.01,
                'gradient_clip': 1.0
            },
            'model': {
                'architecture': 'deep_network',
                'dropout': 0.1,
                'batch_norm': True
            },
            'logging': {
                'log_level': 'INFO',
                'log_dir': 'logs/production',
                'enable_structured': True
            },
            'monitoring': {
                'enable_metrics': True,
                'checkpoint_frequency': 5
            }
        }
    
    @staticmethod
    def research() -> Dict[str, Any]:
        """Configuration for research experiments."""
        return {
            'training': {
                'learning_rate': 0.0003,
                'batch_size': 128,
                'epochs': 200,
                'optimizer': 'adam',
                'lr_scheduler': 'cosine'
            },
            'model': {
                'architecture': 'custom',
                'experimental_features': True
            },
            'logging': {
                'log_level': 'DEBUG',
                'log_dir': 'logs/research',
                'enable_structured': True
            },
            'experiment': {
                'seed': 42,
                'reproducible': True,
                'track_gradients': True
            }
        }
    
    @staticmethod
    def automl() -> Dict[str, Any]:
        """Configuration for AutoML experiments."""
        return {
            'automl': {
                'search_space': 'default',
                'max_trials': 100,
                'optimization_metric': 'accuracy'
            },
            'training': {
                'epochs': 50,
                'early_stopping': True,
                'patience': 10
            },
            'logging': {
                'log_level': 'INFO',
                'log_dir': 'logs/automl'
            }
        }
    
    @staticmethod
    def distributed() -> Dict[str, Any]:
        """Configuration for distributed training."""
        return {
            'training': {
                'learning_rate': 0.001,
                'batch_size': 256,
                'epochs': 100,
                'optimizer': 'adamw',
                'distributed': True,
                'world_size': 4,
                'backend': 'nccl'
            },
            'model': {
                'sync_batch_norm': True
            },
            'logging': {
                'log_level': 'INFO',
                'log_dir': 'logs/distributed'
            }
        }
