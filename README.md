# SpokEcosystem

A comprehensive machine learning ecosystem featuring neural architecture search, reinforcement learning, and advanced training infrastructure.

## Features

### Core Engine
- **Custom Autograd System**: Efficient automatic differentiation with computational graph
- **Neural Network Modules**: Linear layers, activations (ReLU, Sigmoid, Tanh), Dropout, BatchNorm
- **Optimizers**: SGD, Adam, AdamW with weight decay and momentum
- **Loss Functions**: MSE, CrossEntropy, Huber Loss

### AutoML System (SpokNAS)
- **Island-Based Evolution**: Parallel population evolution with migration
- **Multi-Objective Optimization**: Balance accuracy, parameters, and FLOPs
- **Surrogate Modeling**: Sample-efficient architecture search
- **Intelligent Proposal Engine**: Crossover, mutation, and adaptive search

### Reinforcement Learning
- **Policy Gradient Methods**: REINFORCE, Actor-Critic
- **Value-Based Methods**: DQN with experience replay
- **Environment Support**: Gymnasium integration

### Infrastructure
- **Vector Search**: Faiss-based similarity search with IVF/PQ support
- **Caching**: Redis primary with LRU fallback
- **Observability**: Prometheus metrics for monitoring
- **Experiment Management**: Configuration, checkpointing, and grid search

## Installation

\`\`\`bash
# Clone the repository
git clone https://github.com/yourusername/spokecosystem.git
cd spokecosystem

# Install dependencies
pip install -r requirements.txt
\`\`\`

## Quick Start

### Simple Neural Network Training

\`\`\`python
from core_engine import Sequential, Linear, ReLU, SGD, CrossEntropyLoss, Node
import numpy as np

# Create model
model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10)
)

# Setup training
optimizer = SGD(model.parameters(), lr=0.01)
loss_fn = CrossEntropyLoss()

# Training loop
for epoch in range(100):
    X = Node(np.random.randn(32, 784), requires_grad=False)
    y = Node(np.random.randint(0, 10, 32), requires_grad=False)
    
    predictions = model(X)
    loss = loss_fn(predictions, y)
    
    model.zero_grad()
    loss.backward()
    optimizer.step()
\`\`\`

### AutoML Architecture Search

\`\`\`python
from experiments import ExperimentConfig, ExperimentRunner

config = ExperimentConfig(
    name="automl_search",
    model_type="automl",
    automl_population_size=24,
    automl_num_islands=4,
    automl_generations=20
)

runner = ExperimentRunner(config)
results = runner.run()
\`\`\`

### Reinforcement Learning

\`\`\`python
from rl_agents import PolicyGradientAgent
import gymnasium as gym

env = gym.make('CartPole-v1')
agent = PolicyGradientAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    hidden_dim=128
)

agent.train(env, episodes=1000)
\`\`\`

## Project Structure

\`\`\`
spokecosystem/
├── core_engine/          # Neural network engine with autograd
│   ├── node.py          # Computational graph nodes
│   ├── modules.py       # Neural network layers
│   ├── optimizers.py    # Optimization algorithms
│   └── losses.py        # Loss functions
├── automl/              # AutoML and NAS
│   ├── orchestrator/    # Evolution orchestrator
│   ├── proposal_engine.py
│   ├── fitness_functions.py
│   └── model_builder.py
├── rl_agents/           # Reinforcement learning
│   ├── policy_gradient.py
│   ├── actor_critic.py
│   └── dqn.py
├── infrastructure/      # Infrastructure layer
│   ├── vector_search.py
│   ├── cache.py
│   └── observability.py
├── experiments/         # Experiment management
│   ├── config.py
│   ├── runner.py
│   └── checkpoint.py
├── examples/            # Example scripts
└── tests/              # Test suite
\`\`\`

## Configuration

Experiments are configured using YAML or Python dataclasses:

\`\`\`yaml
name: my_experiment
model_type: mlp
learning_rate: 0.001
batch_size: 128
epochs: 100
use_cache: true
use_metrics: true
\`\`\`

## Testing

\`\`\`bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_core_engine.py

# Run with coverage
pytest --cov=. tests/
\`\`\`

## Advanced Features

### Vector Search

\`\`\`python
from infrastructure import create_vector_index

index = create_vector_index(dimension=128, index_type='ivf_pq')
index.train(training_vectors)
index.add(vectors)
distances, indices = index.search(queries, k=10)
\`\`\`

### Caching

\`\`\`python
from infrastructure import CacheManager

cache = CacheManager(use_redis=True)
cache.set('key', value, ttl=3600)
result = cache.get('key')
\`\`\`

### Metrics

\`\`\`python
from infrastructure import MetricsCollector

metrics = MetricsCollector()
metrics.inc_training_steps()
metrics.set_current_fitness(0.95)
print(metrics.get_stats())
\`\`\`

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use SpokEcosystem in your research, please cite:

```bibtex
@software{spokecosystem2025,
  title={SpokEcosystem: A Comprehensive Machine Learning Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/spokecosystem}
}
