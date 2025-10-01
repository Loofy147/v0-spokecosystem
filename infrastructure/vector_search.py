"""Vector search infrastructure with Faiss support for efficient similarity search."""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Literal
import warnings

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn("Faiss not available. Install with: pip install faiss-cpu or faiss-gpu")

class VectorIndex:
    """
    Vector search index with multiple backend support.
    Supports flat (exact), IVF (inverted file), and PQ (product quantization) indices.
    """
    def __init__(
        self,
        dimension: int,
        index_type: Literal['flat', 'ivf', 'pq', 'ivf_pq'] = 'flat',
        nlist: int = 100,
        nprobe: int = 10,
        m: int = 8,
        nbits: int = 8
    ):
        """
        Initialize vector index.
        
        Args:
            dimension: Vector dimension
            index_type: Type of index ('flat', 'ivf', 'pq', 'ivf_pq')
            nlist: Number of clusters for IVF (default: 100)
            nprobe: Number of clusters to search (default: 10)
            m: Number of subquantizers for PQ (default: 8)
            nbits: Bits per subquantizer for PQ (default: 8)
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("Faiss is required for VectorIndex. Install with: pip install faiss-cpu")
        
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.m = m
        self.nbits = nbits
        self.index = None
        self.is_trained = False
        
        self._build_index()
    
    def _build_index(self):
        """Build the appropriate Faiss index based on index_type."""
        if self.index_type == 'flat':
            # Exact search using L2 distance
            self.index = faiss.IndexFlatL2(self.dimension)
            self.is_trained = True
            
        elif self.index_type == 'ivf':
            # IVF index for faster approximate search
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            
        elif self.index_type == 'pq':
            # Product quantization for memory efficiency
            self.index = faiss.IndexPQ(self.dimension, self.m, self.nbits)
            
        elif self.index_type == 'ivf_pq':
            # Combined IVF + PQ for speed and memory efficiency
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFPQ(quantizer, self.dimension, self.nlist, self.m, self.nbits)
        
        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")
        
        # Set nprobe for IVF-based indices
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe
    
    def train(self, vectors: np.ndarray):
        """
        Train the index (required for IVF and PQ indices).
        
        Args:
            vectors: Training vectors of shape (n_samples, dimension)
        """
        if self.is_trained:
            return
        
        vectors = self._ensure_float32(vectors)
        
        if vectors.shape[0] < self.nlist:
            warnings.warn(f"Training data size ({vectors.shape[0]}) < nlist ({self.nlist}). "
                         f"Reducing nlist to {vectors.shape[0] // 2}")
            self.nlist = max(1, vectors.shape[0] // 2)
            self._build_index()
        
        self.index.train(vectors)
        self.is_trained = True
    
    def add(self, vectors: np.ndarray):
        """
        Add vectors to the index.
        
        Args:
            vectors: Vectors to add of shape (n_samples, dimension)
        """
        if not self.is_trained:
            self.train(vectors)
        
        vectors = self._ensure_float32(vectors)
        self.index.add(vectors)
    
    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        
        Args:
            queries: Query vectors of shape (n_queries, dimension)
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (distances, indices) arrays
        """
        queries = self._ensure_float32(queries)
        distances, indices = self.index.search(queries, k)
        return distances, indices
    
    def _ensure_float32(self, vectors: np.ndarray) -> np.ndarray:
        """Ensure vectors are float32 and contiguous."""
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        return np.ascontiguousarray(vectors)
    
    def save(self, filepath: str):
        """Save index to disk."""
        faiss.write_index(self.index, filepath)
    
    def load(self, filepath: str):
        """Load index from disk."""
        self.index = faiss.read_index(filepath)
        self.is_trained = True
    
    def reset(self):
        """Reset the index."""
        self._build_index()
        self.is_trained = False
    
    @property
    def ntotal(self) -> int:
        """Return number of vectors in the index."""
        return self.index.ntotal


class FallbackVectorIndex:
    """Fallback vector index using numpy when Faiss is not available."""
    def __init__(self, dimension: int, **kwargs):
        self.dimension = dimension
        self.vectors = None
        self.is_trained = True
    
    def train(self, vectors: np.ndarray):
        """No training needed for fallback."""
        pass
    
    def add(self, vectors: np.ndarray):
        """Add vectors to the index."""
        vectors = vectors.astype(np.float32)
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
    
    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Brute force search using numpy."""
        if self.vectors is None or len(self.vectors) == 0:
            return np.array([]), np.array([])
        
        queries = queries.astype(np.float32)
        
        # Compute L2 distances
        distances = np.sum((self.vectors[np.newaxis, :, :] - queries[:, np.newaxis, :]) ** 2, axis=2)
        
        # Get top k
        k = min(k, len(self.vectors))
        indices = np.argsort(distances, axis=1)[:, :k]
        distances = np.take_along_axis(distances, indices, axis=1)
        
        return distances, indices
    
    def save(self, filepath: str):
        """Save vectors to disk."""
        np.save(filepath, self.vectors)
    
    def load(self, filepath: str):
        """Load vectors from disk."""
        self.vectors = np.load(filepath)
    
    def reset(self):
        """Reset the index."""
        self.vectors = None
    
    @property
    def ntotal(self) -> int:
        """Return number of vectors."""
        return 0 if self.vectors is None else len(self.vectors)


def create_vector_index(dimension: int, **kwargs) -> VectorIndex | FallbackVectorIndex:
    """
    Factory function to create a vector index.
    Falls back to numpy implementation if Faiss is not available.
    """
    if FAISS_AVAILABLE:
        return VectorIndex(dimension, **kwargs)
    else:
        warnings.warn("Using fallback numpy-based vector index. Install Faiss for better performance.")
        return FallbackVectorIndex(dimension, **kwargs)
