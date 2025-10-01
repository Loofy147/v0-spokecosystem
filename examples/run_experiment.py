"""Example script for running experiments."""
from experiments import ExperimentConfig, ExperimentRunner, GridSearchConfig, GridSearchRunner


def run_simple_experiment():
    """Run a simple supervised learning experiment."""
    config = ExperimentConfig(
        name="simple_mlp",
        description="Simple MLP on synthetic data",
        model_type="mlp",
        learning_rate=1e-3,
        batch_size=128,
        epochs=50,
        log_every=5,
        save_every=10
    )
    
    runner = ExperimentRunner(config)
    results = runner.run()
    runner.save_results()
    
    return results


def run_automl_experiment():
    """Run an AutoML experiment."""
    config = ExperimentConfig(
        name="automl_search",
        description="AutoML architecture search",
        model_type="automl",
        automl_population_size=16,
        automl_num_islands=4,
        automl_generations=20,
        automl_migration_every=5
    )
    
    runner = ExperimentRunner(config)
    results = runner.run()
    runner.save_results()
    
    return results


def run_grid_search():
    """Run a grid search over hyperparameters."""
    base_config = ExperimentConfig(
        name="grid_search",
        model_type="mlp",
        epochs=30
    )
    
    grid_config = GridSearchConfig(
        base_config=base_config,
        param_grid={
            'learning_rate': [1e-4, 1e-3, 1e-2],
            'batch_size': [64, 128, 256],
            'weight_decay': [1e-5, 1e-4, 1e-3]
        }
    )
    
    runner = GridSearchRunner(grid_config)
    results = runner.run()
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Running Simple Experiment")
    print("=" * 60)
    run_simple_experiment()
    
    print("\n" + "=" * 60)
    print("Running AutoML Experiment")
    print("=" * 60)
    run_automl_experiment()
    
    print("\n" + "=" * 60)
    print("Running Grid Search")
    print("=" * 60)
    run_grid_search()
