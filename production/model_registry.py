"""
Enterprise Model Registry & Versioning System
Tracks model versions, metadata, performance metrics, and lineage.
Supports model promotion workflows and A/B testing.
"""

import json
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
from core_engine.nn_modules import Module


class ModelVersion:
    """Represents a single version of a model."""
    
    def __init__(
        self,
        model_name: str,
        version: str,
        model_path: str,
        metadata: Dict[str, Any],
        metrics: Dict[str, float],
        tags: List[str] = None,
        stage: str = "development"
    ):
        self.model_name = model_name
        self.version = version
        self.model_path = model_path
        self.metadata = metadata
        self.metrics = metrics
        self.tags = tags or []
        self.stage = stage  # development, staging, production, archived
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.checksum = self._compute_checksum()
        
    def _compute_checksum(self) -> str:
        """Compute SHA256 checksum of model file."""
        if Path(self.model_path).exists():
            sha256 = hashlib.sha256()
            with open(self.model_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        return ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "model_path": self.model_path,
            "metadata": self.metadata,
            "metrics": self.metrics,
            "tags": self.tags,
            "stage": self.stage,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "checksum": self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelVersion':
        """Create from dictionary."""
        obj = cls(
            model_name=data["model_name"],
            version=data["version"],
            model_path=data["model_path"],
            metadata=data["metadata"],
            metrics=data["metrics"],
            tags=data.get("tags", []),
            stage=data.get("stage", "development")
        )
        obj.created_at = data.get("created_at", obj.created_at)
        obj.updated_at = data.get("updated_at", obj.updated_at)
        obj.checksum = data.get("checksum", obj.checksum)
        return obj


class ModelRegistry:
    """
    Central registry for managing model versions and lifecycle.
    """
    
    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.registry_path / "registry_index.json"
        self.models: Dict[str, Dict[str, ModelVersion]] = {}
        self._load_index()
        
    def _load_index(self):
        """Load registry index from disk."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                data = json.load(f)
                for model_name, versions in data.items():
                    self.models[model_name] = {
                        v: ModelVersion.from_dict(vdata) 
                        for v, vdata in versions.items()
                    }
    
    def _save_index(self):
        """Save registry index to disk."""
        data = {
            model_name: {
                version: mv.to_dict() 
                for version, mv in versions.items()
            }
            for model_name, versions in self.models.items()
        }
        with open(self.index_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_model(
        self,
        model: Module,
        model_name: str,
        version: Optional[str] = None,
        metadata: Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        stage: str = "development"
    ) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            model: The model to register
            model_name: Name of the model
            version: Version string (auto-generated if None)
            metadata: Additional metadata (hyperparameters, dataset info, etc.)
            metrics: Performance metrics
            tags: Tags for categorization
            stage: Lifecycle stage
            
        Returns:
            ModelVersion object
        """
        # Auto-generate version if not provided
        if version is None:
            version = self._generate_version(model_name)
        
        # Create model directory
        model_dir = self.registry_path / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = str(model_dir / "model.pkl")
        model.save(model_path)
        
        # Save metadata and metrics separately
        if metadata:
            with open(model_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        
        if metrics:
            with open(model_dir / "metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
        
        # Create model version
        model_version = ModelVersion(
            model_name=model_name,
            version=version,
            model_path=model_path,
            metadata=metadata or {},
            metrics=metrics or {},
            tags=tags or [],
            stage=stage
        )
        
        # Add to registry
        if model_name not in self.models:
            self.models[model_name] = {}
        self.models[model_name][version] = model_version
        
        self._save_index()
        
        print(f"Registered {model_name} version {version} in stage '{stage}'")
        return model_version
    
    def _generate_version(self, model_name: str) -> str:
        """Auto-generate version number."""
        if model_name not in self.models or not self.models[model_name]:
            return "v1.0.0"
        
        versions = list(self.models[model_name].keys())
        # Extract numeric versions
        numeric_versions = []
        for v in versions:
            if v.startswith('v'):
                try:
                    parts = v[1:].split('.')
                    numeric_versions.append(tuple(map(int, parts)))
                except:
                    pass
        
        if numeric_versions:
            latest = max(numeric_versions)
            return f"v{latest[0]}.{latest[1]}.{latest[2] + 1}"
        
        return f"v1.0.{len(versions)}"
    
    def get_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ) -> Optional[Module]:
        """
        Load a model from registry.
        
        Args:
            model_name: Name of the model
            version: Specific version (if None, gets latest)
            stage: Filter by stage (e.g., 'production')
            
        Returns:
            Loaded model or None
        """
        model_version = self.get_model_version(model_name, version, stage)
        if model_version:
            return Module.load(model_version.model_path)
        return None
    
    def get_model_version(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ) -> Optional[ModelVersion]:
        """Get model version metadata."""
        if model_name not in self.models:
            return None
        
        versions = self.models[model_name]
        
        # Filter by stage if specified
        if stage:
            versions = {v: mv for v, mv in versions.items() if mv.stage == stage}
        
        if not versions:
            return None
        
        # Get specific version or latest
        if version:
            return versions.get(version)
        else:
            # Return latest version
            latest = max(versions.items(), key=lambda x: x[1].created_at)
            return latest[1]
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self.models.keys())
    
    def list_versions(
        self,
        model_name: str,
        stage: Optional[str] = None
    ) -> List[ModelVersion]:
        """List all versions of a model."""
        if model_name not in self.models:
            return []
        
        versions = list(self.models[model_name].values())
        
        if stage:
            versions = [v for v in versions if v.stage == stage]
        
        return sorted(versions, key=lambda x: x.created_at, reverse=True)
    
    def promote_model(
        self,
        model_name: str,
        version: str,
        target_stage: str
    ) -> bool:
        """
        Promote a model to a different stage.
        
        Args:
            model_name: Name of the model
            version: Version to promote
            target_stage: Target stage (staging, production, etc.)
            
        Returns:
            True if successful
        """
        if model_name not in self.models or version not in self.models[model_name]:
            return False
        
        model_version = self.models[model_name][version]
        old_stage = model_version.stage
        model_version.stage = target_stage
        model_version.updated_at = datetime.now().isoformat()
        
        self._save_index()
        
        print(f"Promoted {model_name} {version} from '{old_stage}' to '{target_stage}'")
        return True
    
    def compare_versions(
        self,
        model_name: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """Compare two model versions."""
        if model_name not in self.models:
            return {}
        
        v1 = self.models[model_name].get(version1)
        v2 = self.models[model_name].get(version2)
        
        if not v1 or not v2:
            return {}
        
        comparison = {
            "version1": version1,
            "version2": version2,
            "metrics_diff": {},
            "metadata_diff": {}
        }
        
        # Compare metrics
        all_metrics = set(v1.metrics.keys()) | set(v2.metrics.keys())
        for metric in all_metrics:
            val1 = v1.metrics.get(metric, None)
            val2 = v2.metrics.get(metric, None)
            if val1 is not None and val2 is not None:
                comparison["metrics_diff"][metric] = {
                    "v1": val1,
                    "v2": val2,
                    "diff": val2 - val1,
                    "percent_change": ((val2 - val1) / val1 * 100) if val1 != 0 else 0
                }
        
        # Compare metadata
        all_keys = set(v1.metadata.keys()) | set(v2.metadata.keys())
        for key in all_keys:
            val1 = v1.metadata.get(key)
            val2 = v2.metadata.get(key)
            if val1 != val2:
                comparison["metadata_diff"][key] = {"v1": val1, "v2": val2}
        
        return comparison
    
    def delete_version(
        self,
        model_name: str,
        version: str,
        force: bool = False
    ) -> bool:
        """
        Delete a model version.
        
        Args:
            model_name: Name of the model
            version: Version to delete
            force: Allow deletion of production models
            
        Returns:
            True if successful
        """
        if model_name not in self.models or version not in self.models[model_name]:
            return False
        
        model_version = self.models[model_name][version]
        
        # Prevent accidental deletion of production models
        if model_version.stage == "production" and not force:
            print(f"Cannot delete production model without force=True")
            return False
        
        # Delete model files
        model_dir = Path(model_version.model_path).parent
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # Remove from registry
        del self.models[model_name][version]
        
        # Remove model entry if no versions left
        if not self.models[model_name]:
            del self.models[model_name]
        
        self._save_index()
        
        print(f"Deleted {model_name} version {version}")
        return True
    
    def search_models(
        self,
        tags: Optional[List[str]] = None,
        min_metrics: Optional[Dict[str, float]] = None,
        stage: Optional[str] = None
    ) -> List[ModelVersion]:
        """
        Search for models matching criteria.
        
        Args:
            tags: Filter by tags (any match)
            min_metrics: Minimum metric values
            stage: Filter by stage
            
        Returns:
            List of matching model versions
        """
        results = []
        
        for model_name, versions in self.models.items():
            for version, model_version in versions.items():
                # Filter by stage
                if stage and model_version.stage != stage:
                    continue
                
                # Filter by tags
                if tags and not any(tag in model_version.tags for tag in tags):
                    continue
                
                # Filter by metrics
                if min_metrics:
                    meets_criteria = True
                    for metric, min_val in min_metrics.items():
                        if metric not in model_version.metrics or model_version.metrics[metric] < min_val:
                            meets_criteria = False
                            break
                    if not meets_criteria:
                        continue
                
                results.append(model_version)
        
        return sorted(results, key=lambda x: x.created_at, reverse=True)
    
    def export_model(
        self,
        model_name: str,
        version: str,
        export_path: str,
        format: str = "pkl"
    ) -> bool:
        """
        Export model to external location.
        
        Args:
            model_name: Name of the model
            version: Version to export
            export_path: Destination path
            format: Export format (pkl, onnx, etc.)
            
        Returns:
            True if successful
        """
        model_version = self.get_model_version(model_name, version)
        if not model_version:
            return False
        
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "pkl":
            shutil.copy(model_version.model_path, export_path)
        else:
            print(f"Export format '{format}' not yet supported")
            return False
        
        # Also export metadata
        metadata_path = export_path.parent / f"{export_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_version.to_dict(), f, indent=2)
        
        print(f"Exported {model_name} {version} to {export_path}")
        return True
    
    def get_lineage(self, model_name: str, version: str) -> Dict[str, Any]:
        """
        Get model lineage information (parent models, training data, etc.).
        """
        model_version = self.get_model_version(model_name, version)
        if not model_version:
            return {}
        
        lineage = {
            "model_name": model_name,
            "version": version,
            "created_at": model_version.created_at,
            "metadata": model_version.metadata,
            "metrics": model_version.metrics,
            "tags": model_version.tags,
            "stage": model_version.stage
        }
        
        # Extract lineage info from metadata
        if "parent_model" in model_version.metadata:
            lineage["parent"] = model_version.metadata["parent_model"]
        
        if "training_data" in model_version.metadata:
            lineage["training_data"] = model_version.metadata["training_data"]
        
        if "experiment_id" in model_version.metadata:
            lineage["experiment_id"] = model_version.metadata["experiment_id"]
        
        return lineage
