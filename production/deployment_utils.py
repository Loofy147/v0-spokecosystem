"""
Deployment utilities for model packaging and containerization.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from core_engine.nn_modules import Module


class ModelPackager:
    """
    Package models for deployment with all dependencies.
    """
    
    @staticmethod
    def package_model(
        model: Module,
        output_dir: str,
        model_name: str,
        version: str,
        metadata: Optional[Dict] = None,
        include_examples: bool = True
    ):
        """
        Package model with metadata and examples for deployment.
        
        Args:
            model: Model to package
            output_dir: Output directory
            model_name: Model name
            version: Model version
            metadata: Additional metadata
            include_examples: Include example usage code
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_path / "model.pkl"
        model.save(str(model_path))
        
        # Create deployment manifest
        manifest = {
            "model_name": model_name,
            "version": version,
            "model_file": "model.pkl",
            "metadata": metadata or {},
            "deployment_config": {
                "batch_size": 32,
                "timeout_seconds": 30,
                "max_retries": 3
            }
        }
        
        with open(output_path / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Create requirements file
        requirements = [
            "numpy>=1.20.0",
            "scipy>=1.7.0"
        ]
        
        with open(output_path / "requirements.txt", 'w') as f:
            f.write('\n'.join(requirements))
        
        # Create example usage
        if include_examples:
            example_code = f"""
# Example usage for {model_name} v{version}

import numpy as np
from core_engine.nn_modules import Module

# Load model
model = Module.load('model.pkl')

# Make prediction
features = np.random.randn(1, 10)  # Replace with actual features
predictions = model(features)

print("Predictions:", predictions.data)
"""
            with open(output_path / "example.py", 'w') as f:
                f.write(example_code)
        
        print(f"Model packaged successfully in {output_dir}")
    
    @staticmethod
    def create_dockerfile(
        output_dir: str,
        model_name: str,
        port: int = 8000
    ):
        """
        Create Dockerfile for containerized deployment.
        
        Args:
            output_dir: Directory containing packaged model
            model_name: Model name
            port: Port for API server
        """
        dockerfile_content = f"""
FROM python:3.9-slim

WORKDIR /app

# Copy model files
COPY . /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE {port}

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:{port}/health')"

# Run server
CMD ["python", "server.py"]
"""
        
        output_path = Path(output_dir)
        with open(output_path / "Dockerfile", 'w') as f:
            f.write(dockerfile_content)
        
        # Create simple server script
        server_code = f"""
# Simple HTTP server for model serving
import json
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from core_engine.nn_modules import Module

# Load model
model = Module.load('model.pkl')

class ModelHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data)
                features = np.array(data['features'])
                
                predictions = model(features).data
                
                response = {{
                    'predictions': predictions.tolist(),
                    'model': '{model_name}'
                }}
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({{'error': str(e)}}).encode())
    
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({{'status': 'healthy'}}).encode())

if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', {port}), ModelHandler)
    print('Server running on port {port}')
    server.serve_forever()
"""
        
        with open(output_path / "server.py", 'w') as f:
            f.write(server_code)
        
        print(f"Dockerfile created in {output_dir}")


class DeploymentValidator:
    """
    Validate model deployment before production.
    """
    
    @staticmethod
    def validate_model(
        model: Module,
        test_data: np.ndarray,
        expected_output_shape: tuple,
        latency_threshold_ms: float = 100.0
    ) -> Dict[str, bool]:
        """
        Validate model meets deployment requirements.
        
        Args:
            model: Model to validate
            test_data: Test input data
            expected_output_shape: Expected output shape
            latency_threshold_ms: Maximum acceptable latency
            
        Returns:
            Validation results
        """
        results = {
            "output_shape_valid": False,
            "latency_acceptable": False,
            "no_errors": False,
            "deterministic": False
        }
        
        try:
            # Test output shape
            import time
            start = time.time()
            output = model(test_data)
            latency_ms = (time.time() - start) * 1000
            
            results["output_shape_valid"] = output.data.shape[1:] == expected_output_shape[1:]
            results["latency_acceptable"] = latency_ms < latency_threshold_ms
            results["no_errors"] = True
            
            # Test determinism
            output2 = model(test_data)
            results["deterministic"] = np.allclose(output.data, output2.data)
            
        except Exception as e:
            print(f"Validation error: {e}")
            results["no_errors"] = False
        
        return results
