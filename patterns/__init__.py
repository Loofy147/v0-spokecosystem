"""
Design patterns for SpokeEcosystem.
"""
from .factory import (
    Factory,
    OptimizerFactory,
    ActivationFactory,
    LossFactory,
    ModelFactory,
    register_optimizer,
    register_activation,
    register_loss,
    register_model
)
from .strategy import (
    Strategy,
    TrainingStrategy,
    StandardTrainingStrategy,
    MixedPrecisionTrainingStrategy,
    Context
)
from .observer import (
    Observer,
    Subject,
    CallbackObserver,
    TrainingObserver
)
from .singleton import (
    Singleton,
    SingletonMeta,
    ConfigurationManager
)
from .builder import (
    Builder,
    ModelBuilder,
    PipelineBuilder
)

__all__ = [
    'Factory',
    'OptimizerFactory',
    'ActivationFactory',
    'LossFactory',
    'ModelFactory',
    'register_optimizer',
    'register_activation',
    'register_loss',
    'register_model',
    'Strategy',
    'TrainingStrategy',
    'StandardTrainingStrategy',
    'MixedPrecisionTrainingStrategy',
    'Context',
    'Observer',
    'Subject',
    'CallbackObserver',
    'TrainingObserver',
    'Singleton',
    'SingletonMeta',
    'ConfigurationManager',
    'Builder',
    'ModelBuilder',
    'PipelineBuilder',
]
