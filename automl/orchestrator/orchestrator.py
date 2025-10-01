# automl/orchestrator/orchestrator.py

import numpy as np
import time
import json
from ..proposal_engine import ProposalEngine
from ..fitness_functions import basic_fitness

class Orchestrator:
    """
    A meta-cognitive orchestrator that performs end-to-end
    Neural Architecture and Hyperparameter Search.
    """
    def __init__(self, objective_fn, base_config):
        self.objective_fn = objective_fn
        self.base_config = base_config
        self.proposal_engine = ProposalEngine(base_config)
        self.history = []
        self.best_performer = None

    def run_trial(self, config):
        """Executes a single trial with a given configuration."""
        start_time = time.time()
        try:
            metrics = self.objective_fn(config)
            runtime = time.time() - start_time
            fitness = basic_fitness(metrics) # Use a standardized fitness function

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
                'fitness': -1.0, # Penalize errors heavily
                'runtime': time.time() - start_time
            }
        return result

    def launch(self, n_trials: int = 100):
        """Launches the full AutoML optimization loop."""
        print("🚀 [Orchestrator] Launching Autonomous Optimization Loop.")

        # Initial baseline run
        print("🔬 [Orchestrator] Running initial baseline trial...")
        self.best_performer = self.run_trial(self.base_config)
        self.history.append(self.best_performer)
        print(f" baseline fitness: {self.best_performer['fitness']:.4f}")

        for i in range(1, n_trials):
            # Propose a new candidate configuration
            candidate_config = self.proposal_engine.propose_modification(
                self.best_performer['config']
            )

            # Run the new trial
            candidate_result = self.run_trial(candidate_config)
            self.history.append(candidate_result)

            # Decide whether to promote the candidate
            if candidate_result['fitness'] > self.best_performer['fitness']:
                self.best_performer = candidate_result
                print(f"🏆 [Orchestrator] Trial {i}: New best found! "
                      f"Fitness: {self.best_performer['fitness']:.4f}")
            else:
                print(f"📉 [Orchestrator] Trial {i}: Candidate rejected. "
                      f"Fitness: {candidate_result['fitness']:.4f}")

        print("
🏁 [Orchestrator] Optimization loop finished.")
        print("--- Best Configuration Found ---")
        print(json.dumps(self.best_performer['config'], indent=2))
        print(f"Best Fitness Score: {self.best_performer['fitness']:.4f}")
        return self.best_performer, self.history
