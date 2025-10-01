# automl/proposal_engine.py

import random
import copy
import numpy as np

class ProposalEngine:
    """
    An intelligent engine for proposing modifications to model
    and training configurations.
    """
    def __init__(self, base_config):
        self.base_config = base_config

    def propose_modification(self, best_config):
        """Proposes a single, targeted modification to the current best config."""
        config = copy.deepcopy(best_config)

        mutation_type = random.choice([
            'change_lr', 'change_depth', 'change_width',
            'change_activation' # Uncomment and implement these later
        ])

        if mutation_type == 'change_lr':
            config['optimizer']['lr'] *= 10**np.random.uniform(-0.5, 0.5)

        elif mutation_type == 'change_depth' and len(config['model']['layers']) < 6:
            last_linear_out_features = 32 * 32 * 3

            for layer in reversed(config['model']['layers']):
              if layer['type'] == 'Linear':
                last_linear_out_features = layer['out_features']
                break

            new_layer = {'type': 'Linear', 'out_features': random.choice([32, 64, 128])}
            new_activation = {'type': random.choice(['ReLU', 'Sigmoid', 'Tanh'])} # Include new activations
            insert_pos = random.randint(0, len(config['model']['layers']))

            if insert_pos < len(config['model']['layers']):
                 next_layer = config['model']['layers'][insert_pos]
                 if next_layer['type'] == 'Linear':
                      if insert_pos > 0 and config['model']['layers'][insert_pos - 1]['type'] == 'Linear':
                           config['model']['layers'][insert_pos]['in_features'] = config['model']['layers'][insert_pos - 1]['out_features']
                      elif insert_pos == 0:
                            config['model']['layers'][insert_pos]['in_features'] = new_layer['out_features']


            config['model']['layers'].insert(insert_pos, new_activation)
            config['model']['layers'].insert(insert_pos, new_layer)


        elif mutation_type == 'change_width':
            linear_layers_indices = [i for i, l in enumerate(config['model']['layers']) if l['type'] == 'Linear']
            if linear_layers_indices:
                layer_index_to_change = random.choice(linear_layers_indices)
                layer_to_change = config['model']['layers'][layer_index_to_change]

                factor = random.choice([0.5, 0.75, 1.5, 2.0])
                new_out_features = max(8, int(layer_to_change['out_features'] * factor))
                layer_to_change['out_features'] = new_out_features

                if layer_index_to_change + 1 < len(config['model']['layers']):
                    next_layer = config['model']['layers'][layer_index_to_change + 1]
                    if next_layer['type'] == 'Linear':
                         next_layer['in_features'] = new_out_features

        elif mutation_type == 'change_activation':
             activation_layer_indices = [i for i, l in enumerate(config['model']['layers']) if l['type'] in ['ReLU', 'Sigmoid', 'Tanh']]
             if activation_layer_indices:
                  layer_index_to_change = random.choice(activation_layer_indices)
                  config['model']['layers'][layer_index_to_change]['type'] = random.choice(['ReLU', 'Sigmoid', 'Tanh']) # Change to a new activation

        return config
