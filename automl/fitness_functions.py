# automl/fitness_functions.py

def basic_fitness(metrics: dict) -> float:
    """A basic fitness function based on validation accuracy."""
    return metrics.get('val_accuracy', -1.0) # Return -1.0 if accuracy is not available
