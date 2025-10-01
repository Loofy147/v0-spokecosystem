"""Enhanced proposal engine with island-based evolution and surrogate modeling."""
import random
import copy
import numpy as np
from typing import List, Dict, Any, Optional

class ProposalEngine:
    """
    Intelligent engine for proposing modifications to model and training configurations.
    Enhanced with island-based evolution and surrogate modeling support.
    """
    def __init__(self, base_config: Dict[str, Any], layer_library: Optional[List[str]] = None):
        self.base_config = base_config
        self.layer_library = layer_library or [
            'linear-32', 'linear-64', 'linear-128', 'linear-256',
            'relu', 'sigmoid', 'tanh',
            'dropout-0.2', 'dropout-0.5'
        ]
        self.mutation_rate = 0.06
        self.crossover_rate = 0.6

    def random_genome(self, min_layers: int = 3, max_layers: int = 10) -> Dict[str, Any]:
        """Generate a random genome."""
        arch = [random.choice(self.layer_library) for _ in range(random.randint(min_layers, max_layers))]
        
        # Ensure at least one linear layer
        if not any(t.startswith('linear') or t.startswith('fc') for t in arch):
            arch.insert(0, random.choice([t for t in self.layer_library if t.startswith('linear')]))
        
        return {
            'architecture': arch,
            'lr': 10 ** random.uniform(-4, -2),
            'wd': 10 ** random.uniform(-6, -3),
            'batch_size': random.choice([32, 64, 128])
        }

    def propose_modification(self, best_config: Dict[str, Any]) -> Dict[str, Any]:
        """Proposes a single, targeted modification to the current best config."""
        config = copy.deepcopy(best_config)

        mutation_type = random.choice([
            'change_lr', 'change_depth', 'change_width', 'change_activation'
        ])

        if mutation_type == 'change_lr':
            config['lr'] *= 10 ** np.random.uniform(-0.5, 0.5)

        elif mutation_type == 'change_depth' and len(config['architecture']) < 16:
            # Add a new layer
            new_layer = random.choice(self.layer_library)
            insert_pos = random.randint(0, len(config['architecture']))
            config['architecture'].insert(insert_pos, new_layer)

        elif mutation_type == 'change_width':
            # Change the width of a linear layer
            linear_indices = [i for i, l in enumerate(config['architecture']) 
                            if l.startswith('linear') or l.startswith('fc')]
            if linear_indices:
                idx = random.choice(linear_indices)
                layer = config['architecture'][idx]
                parts = layer.split('-')
                if len(parts) > 1:
                    current_size = int(parts[1])
                    factor = random.choice([0.5, 0.75, 1.5, 2.0])
                    new_size = max(8, int(current_size * factor))
                    config['architecture'][idx] = f"{parts[0]}-{new_size}"

        elif mutation_type == 'change_activation':
            # Change an activation function
            activation_indices = [i for i, l in enumerate(config['architecture']) 
                                if l in ['relu', 'sigmoid', 'tanh']]
            if activation_indices:
                idx = random.choice(activation_indices)
                config['architecture'][idx] = random.choice(['relu', 'sigmoid', 'tanh'])

        return config

    def crossover(self, genome1: Dict[str, Any], genome2: Dict[str, Any]) -> Dict[str, Any]:
        """Perform crossover between two genomes."""
        arch1 = genome1['architecture']
        arch2 = genome2['architecture']
        
        if len(arch1) < 2 or len(arch2) < 2:
            return {'architecture': arch1[:], 'lr': genome1['lr'], 
                   'wd': genome1['wd'], 'batch_size': genome1['batch_size']}
        
        # Single-point crossover
        p1 = random.randint(1, len(arch1) - 1)
        p2 = random.randint(1, len(arch2) - 1)
        child_arch = arch1[:p1] + arch2[p2:]
        
        # Limit architecture length
        child_arch = child_arch[:16]
        
        return {
            'architecture': child_arch,
            'lr': (genome1['lr'] + genome2['lr']) / 2.0,
            'wd': genome1['wd'],
            'batch_size': random.choice([32, 64, 128])
        }

    def mutate(self, genome: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mutation to a genome."""
        arch = genome['architecture'][:]
        
        if random.random() < self.mutation_rate:
            op = random.choice(['add', 'delete', 'swap'])
            
            if op == 'add' and len(arch) < 16:
                arch.insert(random.randint(0, len(arch)), random.choice(self.layer_library))
            elif op == 'delete' and len(arch) > 3:
                del arch[random.randint(0, len(arch) - 1)]
            elif op == 'swap' and len(arch) > 1:
                i, j = random.sample(range(len(arch)), 2)
                arch[i], arch[j] = arch[j], arch[i]
        
        # Mutate learning rate
        lr = genome['lr'] * (10 ** random.uniform(-0.1, 0.1)) if random.random() < self.mutation_rate else genome['lr']
        
        return {
            'architecture': arch,
            'lr': lr,
            'wd': genome['wd'],
            'batch_size': genome.get('batch_size', 64)
        }

    def genome_to_feature_vector(self, genome: Dict[str, Any]) -> np.ndarray:
        """Convert genome to feature vector for surrogate modeling."""
        arch = genome['architecture']
        length = len(arch)
        num_linear = sum(1 for t in arch if t.startswith('linear') or t.startswith('fc'))
        num_relu = sum(1 for t in arch if t == 'relu')
        num_dropout = sum(1 for t in arch if t.startswith('dropout'))
        
        # Extract layer sizes
        sizes = []
        for t in arch:
            if '-' in t:
                try:
                    sizes.append(float(t.split('-')[1]))
                except:
                    pass
        
        avg_size = float(np.mean(sizes)) if sizes else 0.0
        
        return np.array([
            length, num_linear, num_relu, num_dropout,
            avg_size, genome['lr'], genome['wd'], genome['batch_size']
        ], dtype=np.float32)
