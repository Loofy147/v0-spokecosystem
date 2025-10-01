"""Tests for infrastructure components."""
import pytest
import numpy as np
from infrastructure import create_vector_index, LRUCache, CacheManager, MetricsCollector


class TestVectorSearch:
    """Tests for vector search."""
    
    def test_create_flat_index(self):
        """Test creating flat index."""
        index = create_vector_index(dimension=128, index_type='flat')
        assert index is not None
    
    def test_add_and_search(self):
        """Test adding vectors and searching."""
        index = create_vector_index(dimension=10, index_type='flat')
        
        # Add vectors
        vectors = np.random.randn(100, 10).astype(np.float32)
        index.add(vectors)
        
        assert index.ntotal == 100
        
        # Search
        queries = np.random.randn(5, 10).astype(np.float32)
        distances, indices = index.search(queries, k=5)
        
        assert distances.shape == (5, 5)
        assert indices.shape == (5, 5)


class TestCache:
    """Tests for caching."""
    
    def test_lru_cache(self):
        """Test LRU cache."""
        cache = LRUCache(capacity=10)
        
        cache.set('key1', 'value1')
        assert cache.get('key1') == 'value1'
        
        cache.set('key2', 'value2')
        assert cache.get('key2') == 'value2'
        
        # Test miss
        assert cache.get('nonexistent') is None
        
        # Test stats
        stats = cache.stats()
        assert stats['hits'] == 2
        assert stats['misses'] == 1
    
    def test_cache_manager(self):
        """Test cache manager."""
        cache = CacheManager(use_redis=False)
        
        cache.set('test_key', {'data': [1, 2, 3]})
        result = cache.get('test_key')
        
        assert result == {'data': [1, 2, 3]}


class TestMetrics:
    """Tests for metrics collector."""
    
    def test_metrics_collector(self):
        """Test metrics collector."""
        metrics = MetricsCollector(use_prometheus=False)
        
        metrics.inc_training_steps(5)
        metrics.inc_evaluations(2)
        metrics.set_current_fitness(0.85)
        
        stats = metrics.get_stats()
        
        assert stats['training_steps'] == 5
        assert stats['evaluations'] == 2
        assert stats['current_fitness'] == 0.85
