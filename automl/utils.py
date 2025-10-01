"""Utilities: vectorized helpers, pairwise distances, serialization, timing"""
from __future__ import annotations
import numpy as np
import time
from typing import Tuple

def now_ms() -> float:
    """Returns current time in milliseconds."""
    return time.time() * 1000.0

def as_f32_contiguous(x: np.ndarray) -> np.ndarray:
    """Converts array to contiguous float32."""
    return np.ascontiguousarray(x, dtype=np.float32)

def pairwise_sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute pairwise squared euclidean distances between X (n,d) and Y (m,d).
    Vectorized implementation for efficiency.
    """
    X = as_f32_contiguous(X)
    Y = as_f32_contiguous(Y)
    X2 = np.sum(X * X, axis=1).reshape(-1, 1)
    Y2 = np.sum(Y * Y, axis=1).reshape(1, -1)
    XY = X @ Y.T
    d2 = X2 + Y2 - 2.0 * XY
    return np.maximum(d2, 0.0)

def cosine_sim_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between X and Y."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    return Xn @ Yn.T
