"""Checkpoint management for experiments."""
import os
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


class CheckpointManager:
    """Manages saving and loading experiment checkpoints."""
    
    def __init__(self, checkpoint_dir: str, experiment_name: str):
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = -float('inf')
        self.best_checkpoint_path = None
    
    def save_checkpoint(
        self,
        epoch: int,
        model_state: Dict[str, Any],
        optimizer_state: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            epoch: Current epoch number
            model_state: Model state dictionary
            optimizer_state: Optimizer state dictionary
            metrics: Training metrics
            metadata: Additional metadata
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state': model_state,
            'optimizer_state': optimizer_state,
            'metrics': metrics or {},
            'metadata': metadata or {}
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pkl"
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Save metrics separately as JSON for easy inspection
        if metrics:
            metrics_path = self.checkpoint_dir / f"metrics_epoch_{epoch}.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        # Track best checkpoint
        if metrics and 'val_accuracy' in metrics:
            if metrics['val_accuracy'] > self.best_metric:
                self.best_metric = metrics['val_accuracy']
                self.best_checkpoint_path = checkpoint_path
                
                # Save best checkpoint separately
                best_path = self.checkpoint_dir / "best_checkpoint.pkl"
                with open(best_path, 'wb') as f:
                    pickle.dump(checkpoint, f)
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint. If None, loads best checkpoint.
            
        Returns:
            Checkpoint dictionary
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "best_checkpoint.pkl"
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        return checkpoint
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pkl"))
        return [str(cp) for cp in checkpoints]
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) > keep_last_n:
            for checkpoint_path in checkpoints[:-keep_last_n]:
                # Don't delete best checkpoint
                if checkpoint_path != str(self.best_checkpoint_path):
                    os.remove(checkpoint_path)
                    # Also remove corresponding metrics file
                    metrics_path = checkpoint_path.replace('checkpoint_', 'metrics_').replace('.pkl', '.json')
                    if os.path.exists(metrics_path):
                        os.remove(metrics_path)
