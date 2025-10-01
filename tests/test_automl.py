"""Tests for AutoML components."""
import pytest
import numpy as np
from automl import ProposalEngine, basic_fitness, composite_fitness
from automl.model_builder import Genome, build_model_from_genome


class TestProposalEngine:
    """Tests for ProposalEngine."""
    
    def test_random_genome(self):
        """Test random genome generation."""
        engine = ProposalEngine(base_config={})
        genome = engine.random_genome()
        
        assert 'architecture' in genome
        assert 'lr' in genome
        assert 'wd' in genome
        assert len(genome['architecture']) >= 3
    
    def test_crossover(self):
        """Test genome crossover."""
        engine = ProposalEngine(base_config={})
        genome1 = engine.random_genome()
        genome2 = engine.random_genome()
        
        child = engine.crossover(genome1, genome2)
        
        assert 'architecture' in child
        assert len(child['architecture']) > 0
    
    def test_mutate(self):
        """Test genome mutation."""
        engine = ProposalEngine(base_config={})
        genome = engine.random_genome()
        
        mutated = engine.mutate(genome)
        
        assert 'architecture' in mutated
        assert len(mutated['architecture']) > 0
    
    def test_genome_to_feature_vector(self):
        """Test converting genome to feature vector."""
        engine = ProposalEngine(base_config={})
        genome = engine.random_genome()
        
        features = engine.genome_to_feature_vector(genome)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0


class TestFitnessFunctions:
    """Tests for fitness functions."""
    
    def test_basic_fitness(self):
        """Test basic fitness function."""
        metrics = {'val_accuracy': 0.85}
        fitness = basic_fitness(metrics)
        assert fitness == 0.85
    
    def test_composite_fitness(self):
        """Test composite fitness function."""
        fitness = composite_fitness(
            val_acc=0.9,
            params=10000,
            flops=1000000
        )
        assert 0 < fitness <= 1.0


class TestModelBuilder:
    """Tests for model builder."""
    
    def test_build_simple_model(self):
        """Test building a simple model."""
        genome = Genome(
            architecture=['linear-64', 'relu', 'linear-10'],
            lr=0.001,
            wd=1e-4
        )
        
        model = build_model_from_genome(genome, input_size=784, num_classes=10)
        
        assert model is not None
        params = list(model.parameters())
        assert len(params) > 0
