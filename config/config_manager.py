"""
Comprehensive configuration management system.
"""
import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, List
from copy import deepcopy
from utils.logging_config import get_logger
from utils.exceptions import ConfigurationError
from validation.schema import Schema

logger = get_logger(__name__)


class Config:
    """Configuration container with dot notation access."""
    
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self._data = data or {}
    
    def __getattr__(self, key: str) -> Any:
        """Get config value using dot notation."""
        if key.startswith('_'):
            return object.__getattribute__(self, key)
        
        if key not in self._data:
            raise AttributeError(f"Config has no attribute '{key}'")
        
        value = self._data[key]
        if isinstance(value, dict):
            return Config(value)
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Get config value using bracket notation."""
        return self._data[key]
    
    def __setitem__(self, key: str, value: Any):
        """Set config value using bracket notation."""
        self._data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default."""
        try:
            keys = key.split('.')
            value = self._data
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set config value using dot notation."""
        keys = key.split('.')
        data = self._data
        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]
        data[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return deepcopy(self._data)
    
    def update(self, other: Dict[str, Any]):
        """Update configuration with another dict."""
        self._deep_update(self._data, other)
    
    @staticmethod
    def _deep_update(base: Dict, update: Dict):
        """Recursively update nested dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                Config._deep_update(base[key], value)
            else:
                base[key] = value


class ConfigManager:
    """
    Centralized configuration management with multiple sources.
    """
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._config = Config()
        self._schemas: Dict[str, Schema] = {}
        self.logger = get_logger(self.__class__.__name__)
    
    def load_from_file(self, filepath: str, validate: bool = True):
        """
        Load configuration from file (JSON or YAML).
        
        Args:
            filepath: Path to configuration file
            validate: Whether to validate against schema
        """
        path = Path(filepath)
        
        if not path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {filepath}",
                details={'filepath': str(path)}
            )
        
        try:
            with open(path, 'r') as f:
                if path.suffix in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif path.suffix == '.json':
                    data = json.load(f)
                else:
                    raise ConfigurationError(
                        f"Unsupported file format: {path.suffix}",
                        details={'filepath': str(path)}
                    )
            
            # Validate if schema exists
            if validate and path.stem in self._schemas:
                schema = self._schemas[path.stem]
                data = schema.validate(data)
            
            self._config.update(data)
            self.logger.info(f"Loaded configuration from {filepath}")
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration from {filepath}: {e}",
                details={'filepath': str(path), 'error': str(e)}
            )
    
    def load_from_env(self, prefix: str = "SPOKE_"):
        """
        Load configuration from environment variables.
        
        Args:
            prefix: Prefix for environment variables
        """
        env_config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix):].lower()
                
                # Try to parse as JSON for complex types
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = value
                
                # Convert nested keys (SPOKE_MODEL_LR -> model.lr)
                if '_' in config_key:
                    parts = config_key.split('_')
                    nested_key = '.'.join(parts)
                    self._config.set(nested_key, parsed_value)
                else:
                    self._config[config_key] = parsed_value
        
        self.logger.info(f"Loaded {len(env_config)} configuration values from environment")
    
    def load_from_dict(self, data: Dict[str, Any], validate: bool = False, schema_name: Optional[str] = None):
        """
        Load configuration from dictionary.
        
        Args:
            data: Configuration dictionary
            validate: Whether to validate against schema
            schema_name: Name of schema to use for validation
        """
        if validate and schema_name and schema_name in self._schemas:
            schema = self._schemas[schema_name]
            data = schema.validate(data)
        
        self._config.update(data)
        self.logger.info("Loaded configuration from dictionary")
    
    def save_to_file(self, filepath: str, format: str = 'yaml'):
        """
        Save configuration to file.
        
        Args:
            filepath: Path to save configuration
            format: File format ('yaml' or 'json')
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, 'w') as f:
                if format == 'yaml':
                    yaml.dump(self._config.to_dict(), f, default_flow_style=False)
                elif format == 'json':
                    json.dump(self._config.to_dict(), f, indent=2)
                else:
                    raise ConfigurationError(f"Unsupported format: {format}")
            
            self.logger.info(f"Saved configuration to {filepath}")
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to save configuration to {filepath}: {e}",
                details={'filepath': str(path), 'error': str(e)}
            )
    
    def register_schema(self, name: str, schema: Schema):
        """Register a validation schema."""
        self._schemas[name] = schema
        self.logger.debug(f"Registered schema: {name}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self._config.set(key, value)
        self.logger.debug(f"Set config: {key} = {value}")
    
    def get_config(self) -> Config:
        """Get the full configuration object."""
        return self._config
    
    def merge_configs(self, *configs: Dict[str, Any]):
        """Merge multiple configuration dictionaries."""
        for config in configs:
            self._config.update(config)
        self.logger.info(f"Merged {len(configs)} configurations")
    
    def clear(self):
        """Clear all configuration."""
        self._config = Config()
        self.logger.info("Cleared all configuration")


class ConfigBuilder:
    """Builder for constructing configurations."""
    
    def __init__(self):
        self._config = {}
        self.logger = get_logger(self.__class__.__name__)
    
    def set_training_config(
        self,
        learning_rate: float,
        batch_size: int,
        epochs: int,
        optimizer: str = 'adam'
    ):
        """Set training configuration."""
        self._config['training'] = {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'optimizer': optimizer
        }
        return self
    
    def set_model_config(self, architecture: str, **kwargs):
        """Set model configuration."""
        self._config['model'] = {
            'architecture': architecture,
            **kwargs
        }
        return self
    
    def set_data_config(self, dataset: str, **kwargs):
        """Set data configuration."""
        self._config['data'] = {
            'dataset': dataset,
            **kwargs
        }
        return self
    
    def set_logging_config(self, log_dir: str = 'logs', log_level: str = 'INFO'):
        """Set logging configuration."""
        self._config['logging'] = {
            'log_dir': log_dir,
            'log_level': log_level
        }
        return self
    
    def add_custom(self, key: str, value: Any):
        """Add custom configuration."""
        self._config[key] = value
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build and return the configuration."""
        self.logger.info("Built configuration")
        return deepcopy(self._config)


# Global configuration manager instance
_global_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    return _global_config_manager


def load_config(filepath: str):
    """Load configuration from file into global manager."""
    manager = get_config_manager()
    manager.load_from_file(filepath)


def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value from global manager."""
    manager = get_config_manager()
    return manager.get(key, default)


def set_config(key: str, value: Any):
    """Set configuration value in global manager."""
    manager = get_config_manager()
    manager.set(key, value)
