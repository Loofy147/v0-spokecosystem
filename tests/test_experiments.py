"""Tests for experiment management."""
import pytest
import tempfile
from pathlib import Path
from experiments import ExperimentConfig, CheckpointManager


class TestExperimentConfig:
    """Tests for experiment configuration."""
    
    def test_config_creation(self):
        """Test creating experiment config."""
        config = ExperimentConfig(
            name="test_experiment",
            learning_rate=0.001,
            batch_size=128
        )
        
        assert config.name == "test_experiment"
        assert config.learning_rate == 0.001
        assert config.batch_size == 128
    
    def test_config_to_dict(self):
        """Test converting config to dict."""
        config = ExperimentConfig(name="test")
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'name' in config_dict
    
    def test_config_save_load_yaml(self):
        """Test saving and loading config as YAML."""
        config = ExperimentConfig(name="test", learning_rate=0.001)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.to_yaml(f.name)
            loaded_config = ExperimentConfig.from_yaml(f.name)
        
        assert loaded_config.name == config.name
        assert loaded_config.learning_rate == config.learning_rate


class TestCheckpointManager:
    """Tests for checkpoint manager."""
    
    def test_save_checkpoint(self):
        """Test saving checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, "test_experiment")
            
            checkpoint_path = manager.save_checkpoint(
                epoch=1,
                model_state={'weight': [1, 2, 3]},
                metrics={'val_accuracy': 0.85}
            )
            
            assert Path(checkpoint_path).exists()
    
    def test_load_checkpoint(self):
        """Test loading checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, "test_experiment")
            
            # Save
            manager.save_checkpoint(
                epoch=1,
                model_state={'weight': [1, 2, 3]},
                metrics={'val_accuracy': 0.85}
            )
            
            # Load
            checkpoint = manager.load_checkpoint()
            
            assert checkpoint['epoch'] == 1
            assert checkpoint['metrics']['val_accuracy'] == 0.85
