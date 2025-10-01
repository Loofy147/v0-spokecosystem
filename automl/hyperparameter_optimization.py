"""
Advanced Hyperparameter Optimization
Includes Bayesian Optimization, Multi-Fidelity Optimization, and Population-Based Training.
"""

import numpy as np
from typing import Dict, List, Callable, Optional, Any, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
import time


@dataclass
class HyperparameterSpace:
    """Define hyperparameter search space."""
    name: str
    type: str  # 'continuous', 'discrete', 'categorical'
    bounds: Optional[Tuple] = None  # For continuous/discrete
    choices: Optional[List] = None  # For categorical
    log_scale: bool = False  # Use log scale for continuous params
    
    def sample(self) -> Any:
        """Sample a value from this hyperparameter space."""
        if self.type == 'continuous':
            if self.log_scale:
                log_low, log_high = np.log(self.bounds[0]), np.log(self.bounds[1])
                return np.exp(np.random.uniform(log_low, log_high))
            return np.random.uniform(self.bounds[0], self.bounds[1])
        elif self.type == 'discrete':
            return np.random.randint(self.bounds[0], self.bounds[1] + 1)
        elif self.type == 'categorical':
            return np.random.choice(self.choices)
        else:
            raise ValueError(f"Unknown hyperparameter type: {self.type}")


class GaussianProcess:
    """
    Simple Gaussian Process for Bayesian Optimization.
    Uses RBF kernel for modeling the objective function.
    """
    
    def __init__(self, length_scale: float = 1.0, noise: float = 1e-6):
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = []
        self.y_train = []
        self.K_inv = None
        
    def rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF (Gaussian) kernel."""
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return np.exp(-0.5 * sqdist / self.length_scale**2)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit GP to observed data."""
        self.X_train = X
        self.y_train = y
        
        K = self.rbf_kernel(X, X)
        K += self.noise * np.eye(len(X))
        
        try:
            self.K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            # Add more noise if singular
            K += 1e-3 * np.eye(len(X))
            self.K_inv = np.linalg.inv(K)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance at new points.
        
        Returns:
            mean, variance
        """
        if len(self.X_train) == 0:
            return np.zeros(len(X)), np.ones(len(X))
        
        K_s = self.rbf_kernel(self.X_train, X)
        K_ss = self.rbf_kernel(X, X)
        
        # Mean prediction
        mean = K_s.T.dot(self.K_inv).dot(self.y_train)
        
        # Variance prediction
        var = np.diag(K_ss) - np.sum(K_s.T.dot(self.K_inv) * K_s.T, axis=1)
        var = np.maximum(var, 1e-8)  # Ensure positive
        
        return mean, var


class AcquisitionFunction:
    """Acquisition functions for Bayesian Optimization."""
    
    @staticmethod
    def expected_improvement(mean: np.ndarray, std: np.ndarray, best_y: float, xi: float = 0.01) -> np.ndarray:
        """Expected Improvement acquisition function."""
        improvement = mean - best_y - xi
        Z = improvement / (std + 1e-9)
        
        # Use approximation for normal CDF and PDF
        ei = improvement * 0.5 * (1 + np.tanh(Z / np.sqrt(2))) + std * np.exp(-Z**2 / 2) / np.sqrt(2 * np.pi)
        return ei
    
    @staticmethod
    def upper_confidence_bound(mean: np.ndarray, std: np.ndarray, kappa: float = 2.0) -> np.ndarray:
        """Upper Confidence Bound acquisition function."""
        return mean + kappa * std
    
    @staticmethod
    def probability_of_improvement(mean: np.ndarray, std: np.ndarray, best_y: float, xi: float = 0.01) -> np.ndarray:
        """Probability of Improvement acquisition function."""
        improvement = mean - best_y - xi
        Z = improvement / (std + 1e-9)
        return 0.5 * (1 + np.tanh(Z / np.sqrt(2)))


class BayesianOptimizer:
    """
    Bayesian Optimization for hyperparameter tuning.
    Uses Gaussian Process to model the objective function.
    """
    
    def __init__(
        self,
        search_space: List[HyperparameterSpace],
        acquisition: str = "ei",  # 'ei', 'ucb', 'poi'
        n_initial_points: int = 5,
        xi: float = 0.01,
        kappa: float = 2.0
    ):
        self.search_space = search_space
        self.acquisition = acquisition
        self.n_initial_points = n_initial_points
        self.xi = xi
        self.kappa = kappa
        
        self.gp = GaussianProcess()
        self.X_observed = []
        self.y_observed = []
        self.best_params = None
        self.best_score = -np.inf
        
    def _params_to_array(self, params: Dict) -> np.ndarray:
        """Convert parameter dict to array for GP."""
        arr = []
        for hp in self.search_space:
            val = params[hp.name]
            if hp.type == 'categorical':
                # One-hot encode categorical
                idx = hp.choices.index(val)
                arr.append(idx)
            elif hp.log_scale:
                arr.append(np.log(val))
            else:
                arr.append(val)
        return np.array(arr)
    
    def _array_to_params(self, arr: np.ndarray) -> Dict:
        """Convert array back to parameter dict."""
        params = {}
        for i, hp in enumerate(self.search_space):
            val = arr[i]
            if hp.type == 'categorical':
                idx = int(round(val))
                idx = max(0, min(idx, len(hp.choices) - 1))
                params[hp.name] = hp.choices[idx]
            elif hp.type == 'discrete':
                params[hp.name] = int(round(val))
            elif hp.log_scale:
                params[hp.name] = np.exp(val)
            else:
                params[hp.name] = val
        return params
    
    def _sample_random_params(self) -> Dict:
        """Sample random parameters from search space."""
        return {hp.name: hp.sample() for hp in self.search_space}
    
    def _get_acquisition_value(self, X: np.ndarray) -> np.ndarray:
        """Compute acquisition function value."""
        mean, var = self.gp.predict(X)
        std = np.sqrt(var)
        
        if self.acquisition == "ei":
            return AcquisitionFunction.expected_improvement(mean, std, self.best_score, self.xi)
        elif self.acquisition == "ucb":
            return AcquisitionFunction.upper_confidence_bound(mean, std, self.kappa)
        elif self.acquisition == "poi":
            return AcquisitionFunction.probability_of_improvement(mean, std, self.best_score, self.xi)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition}")
    
    def suggest(self) -> Dict:
        """Suggest next hyperparameters to evaluate."""
        # Random sampling for initial points
        if len(self.X_observed) < self.n_initial_points:
            return self._sample_random_params()
        
        # Fit GP to observed data
        X_train = np.array(self.X_observed)
        y_train = np.array(self.y_observed)
        self.gp.fit(X_train, y_train)
        
        # Optimize acquisition function
        best_acq = -np.inf
        best_params = None
        
        # Random search over acquisition function
        n_candidates = 1000
        for _ in range(n_candidates):
            candidate_params = self._sample_random_params()
            candidate_array = self._params_to_array(candidate_params)
            acq_value = self._get_acquisition_value(candidate_array.reshape(1, -1))[0]
            
            if acq_value > best_acq:
                best_acq = acq_value
                best_params = candidate_params
        
        return best_params
    
    def observe(self, params: Dict, score: float):
        """Record observation of hyperparameters and their score."""
        param_array = self._params_to_array(params)
        self.X_observed.append(param_array)
        self.y_observed.append(score)
        
        if score > self.best_score:
            self.best_score = score
            self.best_params = params
    
    def optimize(
        self,
        objective_fn: Callable[[Dict], float],
        n_iterations: int = 50,
        verbose: bool = True
    ) -> Tuple[Dict, float]:
        """
        Run Bayesian optimization.
        
        Args:
            objective_fn: Function that takes params dict and returns score
            n_iterations: Number of optimization iterations
            verbose: Print progress
            
        Returns:
            best_params, best_score
        """
        for i in range(n_iterations):
            # Suggest next parameters
            params = self.suggest()
            
            # Evaluate objective
            score = objective_fn(params)
            
            # Record observation
            self.observe(params, score)
            
            if verbose:
                print(f"Iteration {i+1}/{n_iterations}: Score = {score:.4f}, Best = {self.best_score:.4f}")
        
        return self.best_params, self.best_score


class MultiFidelityOptimizer:
    """
    Multi-Fidelity Optimization using Successive Halving.
    Efficiently allocates resources by early stopping poor configurations.
    """
    
    def __init__(
        self,
        search_space: List[HyperparameterSpace],
        min_fidelity: int = 1,
        max_fidelity: int = 81,
        reduction_factor: int = 3
    ):
        self.search_space = search_space
        self.min_fidelity = min_fidelity
        self.max_fidelity = max_fidelity
        self.reduction_factor = reduction_factor
        
        self.best_params = None
        self.best_score = -np.inf
        self.history = []
    
    def _sample_random_params(self) -> Dict:
        """Sample random parameters."""
        return {hp.name: hp.sample() for hp in self.search_space}
    
    def optimize(
        self,
        objective_fn: Callable[[Dict, int], float],
        n_configurations: int = 27,
        verbose: bool = True
    ) -> Tuple[Dict, float]:
        """
        Run multi-fidelity optimization.
        
        Args:
            objective_fn: Function that takes (params, fidelity) and returns score
            n_configurations: Number of initial configurations
            verbose: Print progress
            
        Returns:
            best_params, best_score
        """
        # Generate initial configurations
        configurations = [self._sample_random_params() for _ in range(n_configurations)]
        
        fidelity = self.min_fidelity
        
        while len(configurations) > 0 and fidelity <= self.max_fidelity:
            if verbose:
                print(f"\nEvaluating {len(configurations)} configurations at fidelity {fidelity}")
            
            # Evaluate all configurations at current fidelity
            scores = []
            for i, params in enumerate(configurations):
                score = objective_fn(params, fidelity)
                scores.append(score)
                
                self.history.append({
                    "params": params,
                    "fidelity": fidelity,
                    "score": score
                })
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
                
                if verbose:
                    print(f"  Config {i+1}/{len(configurations)}: Score = {score:.4f}")
            
            # Keep top configurations
            n_keep = max(1, len(configurations) // self.reduction_factor)
            top_indices = np.argsort(scores)[-n_keep:]
            configurations = [configurations[i] for i in top_indices]
            
            # Increase fidelity
            fidelity = min(fidelity * self.reduction_factor, self.max_fidelity)
        
        return self.best_params, self.best_score


class PopulationBasedTraining:
    """
    Population-Based Training for joint optimization of hyperparameters and model weights.
    """
    
    def __init__(
        self,
        search_space: List[HyperparameterSpace],
        population_size: int = 10,
        exploit_threshold: float = 0.2,
        explore_probability: float = 0.25
    ):
        self.search_space = search_space
        self.population_size = population_size
        self.exploit_threshold = exploit_threshold
        self.explore_probability = explore_probability
        
        self.population = []
        self.best_params = None
        self.best_score = -np.inf
    
    def _sample_random_params(self) -> Dict:
        """Sample random parameters."""
        return {hp.name: hp.sample() for hp in self.search_space}
    
    def _perturb_params(self, params: Dict) -> Dict:
        """Perturb hyperparameters for exploration."""
        new_params = params.copy()
        for hp in self.search_space:
            if np.random.random() < self.explore_probability:
                new_params[hp.name] = hp.sample()
        return new_params
    
    def initialize_population(self) -> List[Dict]:
        """Initialize population with random hyperparameters."""
        self.population = [
            {
                "params": self._sample_random_params(),
                "score": -np.inf,
                "model_state": None
            }
            for _ in range(self.population_size)
        ]
        return [member["params"] for member in self.population]
    
    def step(
        self,
        scores: List[float],
        model_states: List[Any]
    ) -> List[Dict]:
        """
        Perform one PBT step: exploit and explore.
        
        Args:
            scores: Current scores for each population member
            model_states: Current model states (for copying weights)
            
        Returns:
            New hyperparameters for each member
        """
        # Update population with scores and states
        for i, (score, state) in enumerate(zip(scores, model_states)):
            self.population[i]["score"] = score
            self.population[i]["model_state"] = state
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = self.population[i]["params"]
        
        # Sort by score
        sorted_pop = sorted(self.population, key=lambda x: x["score"], reverse=True)
        
        # Exploit and explore
        new_params = []
        for i, member in enumerate(self.population):
            # Bottom performers exploit top performers
            if member["score"] < sorted_pop[int(self.population_size * self.exploit_threshold)]["score"]:
                # Copy from top performer
                top_member = sorted_pop[np.random.randint(0, max(1, int(self.population_size * self.exploit_threshold)))]
                member["params"] = self._perturb_params(top_member["params"])
                member["model_state"] = top_member["model_state"]  # Copy weights
            
            new_params.append(member["params"])
        
        return new_params


class HyperparameterOptimizer:
    """
    Unified interface for hyperparameter optimization.
    """
    
    def __init__(
        self,
        search_space: List[HyperparameterSpace],
        method: str = "bayesian",  # 'bayesian', 'multifidelity', 'pbt', 'random'
        **kwargs
    ):
        self.search_space = search_space
        self.method = method
        
        if method == "bayesian":
            self.optimizer = BayesianOptimizer(search_space, **kwargs)
        elif method == "multifidelity":
            self.optimizer = MultiFidelityOptimizer(search_space, **kwargs)
        elif method == "pbt":
            self.optimizer = PopulationBasedTraining(search_space, **kwargs)
        elif method == "random":
            self.optimizer = None  # Random search doesn't need optimizer
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def optimize(
        self,
        objective_fn: Callable,
        n_iterations: int = 50,
        verbose: bool = True
    ) -> Tuple[Dict, float]:
        """Run hyperparameter optimization."""
        if self.method == "random":
            return self._random_search(objective_fn, n_iterations, verbose)
        elif self.method in ["bayesian", "multifidelity"]:
            return self.optimizer.optimize(objective_fn, n_iterations, verbose)
        else:
            raise NotImplementedError(f"Optimize not implemented for {self.method}")
    
    def _random_search(
        self,
        objective_fn: Callable[[Dict], float],
        n_iterations: int,
        verbose: bool
    ) -> Tuple[Dict, float]:
        """Random search baseline."""
        best_params = None
        best_score = -np.inf
        
        for i in range(n_iterations):
            params = {hp.name: hp.sample() for hp in self.search_space}
            score = objective_fn(params)
            
            if score > best_score:
                best_score = score
                best_params = params
            
            if verbose:
                print(f"Iteration {i+1}/{n_iterations}: Score = {score:.4f}, Best = {best_score:.4f}")
        
        return best_params, best_score
    
    def save_results(self, filepath: str):
        """Save optimization results."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            "method": self.method,
            "search_space": [
                {
                    "name": hp.name,
                    "type": hp.type,
                    "bounds": hp.bounds,
                    "choices": hp.choices,
                    "log_scale": hp.log_scale
                }
                for hp in self.search_space
            ]
        }
        
        if hasattr(self.optimizer, 'best_params'):
            results["best_params"] = self.optimizer.best_params
            results["best_score"] = float(self.optimizer.best_score)
        
        if hasattr(self.optimizer, 'history'):
            results["history"] = self.optimizer.history
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
