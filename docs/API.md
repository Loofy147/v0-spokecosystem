# API Documentation

## Core Engine

### Node

The fundamental building block for automatic differentiation.

\`\`\`python
Node(value, requires_grad=False, name=None)
\`\`\`

**Parameters:**
- `value` (np.ndarray): The data stored in the node
- `requires_grad` (bool): Whether to compute gradients for this node
- `name` (str): Optional name for debugging

**Methods:**
- `backward()`: Compute gradients via backpropagation
- `zero_grad()`: Reset gradients to zero

### Linear

Fully connected layer.

\`\`\`python
Linear(in_features, out_features)
\`\`\`

**Parameters:**
- `in_features` (int): Input dimension
- `out_features` (int): Output dimension

### Optimizers

#### SGD

\`\`\`python
SGD(parameters, lr=0.01, momentum=0.0, weight_decay=0.0)
\`\`\`

#### Adam

\`\`\`python
Adam(parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
\`\`\`

## AutoML

### Orchestrator

Main class for running AutoML experiments.

\`\`\`python
Orchestrator(objective_fn, base_config, population_size=24, num_islands=4)
\`\`\`

**Methods:**
- `launch(n_trials, migrate_every, migration_k)`: Run optimization

### ProposalEngine

Generates architecture proposals.

\`\`\`python
ProposalEngine(base_config, layer_library=None)
\`\`\`

**Methods:**
- `random_genome()`: Generate random architecture
- `crossover(genome1, genome2)`: Combine two architectures
- `mutate(genome)`: Apply random mutations

## Infrastructure

### VectorIndex

Efficient similarity search.

\`\`\`python
create_vector_index(dimension, index_type='flat', **kwargs)
\`\`\`

**Methods:**
- `add(vectors)`: Add vectors to index
- `search(queries, k)`: Find k nearest neighbors

### CacheManager

Unified caching interface.

\`\`\`python
CacheManager(use_redis=True, redis_host='localhost', lru_capacity=1000)
\`\`\`

**Methods:**
- `get(key)`: Retrieve cached value
- `set(key, value, ttl)`: Store value with optional TTL
- `stats()`: Get cache statistics

## Experiments

### ExperimentConfig

Configuration for experiments.

\`\`\`python
ExperimentConfig(name, model_type='mlp', learning_rate=1e-3, ...)
\`\`\`

### ExperimentRunner

Run experiments with full infrastructure.

\`\`\`python
ExperimentRunner(config)
\`\`\`

**Methods:**
- `run()`: Execute experiment
- `save_results(filepath)`: Save results to file
