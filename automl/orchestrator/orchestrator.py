"""Enhanced orchestrator with island-based evolution, migration, and surrogate modeling."""
import numpy as np
import time
import json
import warnings
from copy import deepcopy
from typing import List, Dict, Any, Optional, Callable
from ..proposal_engine import ProposalEngine
from ..fitness_functions import basic_fitness

class Orchestrator:
    """
    Meta-cognitive orchestrator for Neural Architecture and Hyperparameter Search.
    Enhanced with island-based evolution, migration, and surrogate modeling.
    """
    def __init__(
        self,
        objective_fn: Callable,
        base_config: Dict[str, Any],
        population_size: int = 24,
        num_islands: int = 4,
        elitism: int = 2,
        surrogate_model: Optional[Any] = None
    ):
        self.objective_fn = objective_fn
        self.base_config = base_config
        self.population_size = population_size
        self.num_islands = num_islands
        self.elitism = elitism
        self.proposal_engine = ProposalEngine(base_config)
        self.history = []
        self.best_performer = None
        self.surrogate_model = surrogate_model
        self.surrogate_data_X = []
        self.surrogate_data_y = []

    def run_trial(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a single trial with a given configuration."""
        start_time = time.time()
        try:
            metrics = self.objective_fn(config)
            runtime = time.time() - start_time
            fitness = metrics.get('fitness', basic_fitness(metrics))

            result = {
                'status': 'ok',
                'config': config,
                'metrics': metrics,
                'fitness': fitness,
                'runtime': runtime
            }
        except Exception as e:
            result = {
                'status': 'error',
                'config': config,
                'error': str(e),
                'fitness': -1.0,
                'runtime': time.time() - start_time
            }
        return result

    def evaluate_individual(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate an individual genome."""
        try:
            result = self.run_trial(individual['genome'])
            individual.update(result)
            return individual
        except Exception as e:
            warnings.warn(f"Evaluation failure: {e}")
            individual.update({
                'fitness': 0.0,
                'status': 'error',
                'error': str(e)
            })
            return individual

    def batch_feature_matrix(self, individuals: List[Dict[str, Any]]) -> np.ndarray:
        """Convert batch of individuals to feature matrix for surrogate."""
        features = [self.proposal_engine.genome_to_feature_vector(ind['genome']) 
                   for ind in individuals]
        return np.vstack(features).astype(np.float32)

    def launch(
        self,
        n_trials: int = 100,
        migrate_every: int = 5,
        migration_k: int = 2,
        surrogate_warmup: int = 40,
        verbose: bool = True
    ) -> tuple:
        """
        Launches the full AutoML optimization loop with island-based evolution.
        
        Args:
            n_trials: Total number of generations
            migrate_every: Migrate individuals every N generations
            migration_k: Number of individuals to migrate
            surrogate_warmup: Number of evaluations before using surrogate
            verbose: Whether to print progress
            
        Returns:
            Tuple of (best_performer, history)
        """
        if verbose:
            print("[Orchestrator] Launching Island-Based Evolution")

        # Initialize islands
        islands = [[] for _ in range(self.num_islands)]
        island_size = self.population_size // self.num_islands
        
        # Create initial population
        for i in range(self.population_size):
            genome = self.proposal_engine.random_genome()
            individual = {'genome': genome, 'fitness': None}
            islands[i % self.num_islands].append(individual)

        # Initial evaluation
        if verbose:
            print("[Orchestrator] Evaluating initial population...")
        
        for island in islands:
            for ind in island:
                self.evaluate_individual(ind)

        # Collect initial surrogate data
        all_individuals = [ind for island in islands for ind in island]
        X = self.batch_feature_matrix(all_individuals)
        y = np.array([ind.get('fitness', 0.0) for ind in all_individuals], dtype=np.float32)
        self.surrogate_data_X.append(X)
        self.surrogate_data_y.append(y)

        # Track best
        all_individuals.sort(key=lambda x: x.get('fitness', -1e9), reverse=True)
        self.best_performer = deepcopy(all_individuals[0])
        self.history.append(self.best_performer['fitness'])

        if verbose:
            print(f"[Orchestrator] Initial best fitness: {self.best_performer['fitness']:.4f}")

        # Evolution loop
        for gen in range(1, n_trials):
            # Evolve each island
            for island_idx, island in enumerate(islands):
                # Sort by fitness
                island.sort(key=lambda x: x.get('fitness', -1e9), reverse=True)
                
                # Keep elites
                elites = island[:self.elitism]
                new_population = elites[:]
                
                # Generate offspring
                while len(new_population) < island_size:
                    # Tournament selection
                    parent1 = max(random.sample(island, min(3, len(island))), 
                                key=lambda x: x.get('fitness', -1e9))
                    parent2 = max(random.sample(island, min(3, len(island))), 
                                key=lambda x: x.get('fitness', -1e9))
                    
                    # Crossover and mutation
                    child_genome = self.proposal_engine.crossover(parent1['genome'], parent2['genome'])
                    child_genome = self.proposal_engine.mutate(child_genome)
                    
                    child = {'genome': child_genome}
                    child = self.evaluate_individual(child)
                    new_population.append(child)
                
                islands[island_idx] = new_population

            # Migration
            if migrate_every and gen % migrate_every == 0 and self.num_islands > 1:
                migrants = [sorted(island, key=lambda x: x.get('fitness', -1e9), reverse=True)[:migration_k] 
                           for island in islands]
                
                for i in range(len(islands)):
                    dest = (i + 1) % len(islands)
                    islands[dest][-migration_k:] = [deepcopy(m) for m in migrants[i]]

            # Update surrogate
            all_individuals = [ind for island in islands for ind in island]
            X = self.batch_feature_matrix(all_individuals)
            y = np.array([ind.get('fitness', 0.0) for ind in all_individuals], dtype=np.float32)
            self.surrogate_data_X.append(X)
            self.surrogate_data_y.append(y)

            if self.surrogate_model is not None and len(self.surrogate_data_X) >= surrogate_warmup:
                try:
                    X_all = np.vstack(self.surrogate_data_X)
                    y_all = np.concatenate(self.surrogate_data_y)
                    self.surrogate_model.fit(X_all, y_all)
                except Exception as e:
                    warnings.warn(f"Surrogate training failed: {e}")

            # Track best
            all_individuals.sort(key=lambda x: x.get('fitness', -1e9), reverse=True)
            current_best = all_individuals[0]
            
            if current_best['fitness'] > self.best_performer['fitness']:
                self.best_performer = deepcopy(current_best)
                if verbose:
                    print(f"[Orchestrator] Gen {gen}: New best! Fitness: {self.best_performer['fitness']:.4f}")
            elif verbose and gen % 5 == 0:
                print(f"[Orchestrator] Gen {gen}: Best fitness: {self.best_performer['fitness']:.4f}")

            self.history.append(self.best_performer['fitness'])

        if verbose:
            print("\n[Orchestrator] Optimization complete!")
            print("--- Best Configuration ---")
            print(json.dumps(self.best_performer['config'], indent=2))
            print(f"Best Fitness: {self.best_performer['fitness']:.4f}")

        return self.best_performer, self.history
