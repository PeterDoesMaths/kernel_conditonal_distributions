"""
Kernel functions used by CMMD estimators and hypothesis tests.
Includes continuous, discrete, and utility bandwidth-selection kernels.
"""

import numpy as np
from typing import Optional


def gaussian_kernel(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    bandwidth: float = 1.0
) -> np.ndarray:
    """
    Compute the Gaussian (RBF) kernel matrix.
    
    k(x, x') = exp(-0.5 * bandwidth * ||x - x'||_2^2)
    
    Parameters
    ----------
    X : np.ndarray, shape (n, d) or (n,)
        First set of samples. If 1D, will be reshaped to (n, 1).
    Y : np.ndarray, shape (m, d) or (m,), optional
        Second set of samples. If None, computes k(X, X).
        If 1D, will be reshaped to (m, 1).
    bandwidth : float, default=1.0
        Bandwidth parameter h in the kernel formula.
    
    Returns
    -------
    K : np.ndarray, shape (n, m) or (n, n)
        Kernel matrix where K[i, j] = k(X[i], Y[j]).
    """
    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # If Y is None, compute K(X, X)
    if Y is None:
        Y = X
    elif Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    # Compute pairwise squared Euclidean distances
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
    X_sq = np.sum(X**2, axis=1, keepdims=True)  # (n, 1)
    Y_sq = np.sum(Y**2, axis=1, keepdims=True)  # (m, 1)
    XY = X @ Y.T  # (n, m)
    
    sq_dists = X_sq + Y_sq.T - 2 * XY  # (n, m)
    
    # Kernel computation
    K = np.exp(-0.5 * bandwidth * sq_dists)
    
    return K


def median_heuristic(X: np.ndarray, Y: Optional[np.ndarray] = None) -> float:
    """
    Compute the median heuristic for bandwidth selection.
    
    Returns the median of pairwise distances, which is a common
    heuristic for setting the bandwidth parameter.
    
    Parameters
    ----------
    X : np.ndarray, shape (n, d) or (n,)
        First set of samples.
    Y : np.ndarray, shape (m, d) or (m,), optional
        Second set of samples. If None, uses X only.
    
    Returns
    -------
    bandwidth : float
        Suggested bandwidth parameter h = 1 / median_distance^2.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    if Y is None:
        # Use only X
        from scipy.spatial.distance import pdist
        dists = pdist(X, metric='euclidean')
    else:
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        from scipy.spatial.distance import cdist
        dists = cdist(X, Y, metric='euclidean').ravel()
    
    median_dist = np.median(dists[dists > 0])  # Exclude zero distances
    bandwidth = 1.0 / (median_dist ** 2) if median_dist > 0 else 1.0
    
    return bandwidth


def kronecker_delta_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
    """
    Compute the Kronecker delta kernel matrix.
    
    k(x, y) = 1 if x == y else 0
    
    Parameters
    ----------
    X : np.ndarray, shape (n,) or (n, 1)
        First set of samples.
    Y : np.ndarray, shape (m,) or (m, 1), optional
        Second set of samples. If None, computes k(X, X).
    **kwargs
        Additional arguments (ignored, for compatibility with other kernels).
    
    Returns
    -------
    K : np.ndarray, shape (n, m) or (n, n)
        Kernel matrix where K[i, j] = k(X[i], Y[j]).
    """
    if Y is None:
        Y = X
    
    # Flatten to handle both (n,) and (n, 1) shapes
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    
    K = (X_flat[:, None] == Y_flat[None, :]).astype(float)
    
    return K

def polynomial_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None, degree: int = 2, coef0: float = 1.0, gamma: float = 1.0) -> np.ndarray:
    """
    Compute the polynomial kernel matrix.
    
    k(x, y) = (gamma * <x, y> + coef0)^degree
    
    Parameters
    ----------
    X : np.ndarray, shape (n, d) or (n,)
        First set of samples. If 1D, will be reshaped to (n, 1).
    Y : np.ndarray, shape (m, d) or (m,), optional
        Second set of samples. If None, computes k(X, X).
        If 1D, will be reshaped to (m, 1).
    degree : int, default=2
        Degree of the polynomial kernel.
    coef0 : float, default=1.0
        Independent term in the polynomial kernel.
    gamma : float, default=1.0
        Scaling factor for the inner product.
    
    Returns
    -------
    K : np.ndarray, shape (n, m) or (n, n)
        Kernel matrix where K[i, j] = k(X[i], Y[j]).
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    if Y is None:
        Y = X
    elif Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    K = (gamma * (X @ Y.T) + coef0) ** degree
    
    return K

def linear_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
    """
    Compute the linear kernel matrix.
    
    k(x, y) = <x, y>
    
    Parameters
    ----------
    X : np.ndarray, shape (n, d) or (n,)
        First set of samples. If 1D, will be reshaped to (n, 1).
    Y : np.ndarray, shape (m, d) or (m,), optional
        Second set of samples. If None, computes k(X, X).
        If 1D, will be reshaped to (m, 1).
    **kwargs
        Additional arguments (ignored, for compatibility with other kernels).
    
    Returns
    -------
    K : np.ndarray, shape (n, m) or (n, n)
        Kernel matrix where K[i, j] = k(X[i], Y[j]).
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    if Y is None:
        Y = X
    elif Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    K = X @ Y.T
    
    return K

def indicator_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
    """
    Compute the indicator kernel matrix.
    
    k(x, y) = 1 if x == y and x == 1 else 0
    
    Parameters
    ----------
    X : np.ndarray, shape (n,) or (n, 1)
        First set of samples.
    Y : np.ndarray, shape (m,) or (m, 1), optional
        Second set of samples. If None, computes k(X, X).
    **kwargs
        Additional arguments (ignored, for compatibility with other kernels).
    
    Returns
    -------
    K : np.ndarray, shape (n, m) or (n, n)
        Kernel matrix where K[i, j] = k(X[i], Y[j]).
    """
    if Y is None:
        Y = X
    
    # Flatten to handle both (n,) and (n, 1) shapes
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    
    K = (X_flat[:, None] == Y_flat[None, :]) & (X_flat[:, None] == 1)
    K = K.astype(float)
    
    return K