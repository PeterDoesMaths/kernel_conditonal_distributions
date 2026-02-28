"""
Data generating processes for conditional distribution experiments.
"""

import numpy as np
from typing import Tuple, Callable


def sample_covariate_p(n: int, seed: int = None) -> np.ndarray:
    """
    Sample covariates from the base distribution X ~ U(0, 1).
    
    Parameters
    ----------
    n : int
        Number of samples.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    X : np.ndarray, shape (n,)
        Covariate samples.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, size=n)

def sample_covariate_q(n: int, seed: int = None) -> np.ndarray:
    """
    Sample covariates from the base distribution X ~ Beta(0.5, 0.5).
    
    Parameters
    ----------
    n : int
        Number of samples.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    X : np.ndarray, shape (n,)
        Covariate samples.
    """
    rng = np.random.default_rng(seed)
    return rng.beta(0.5, 0.5, size=n)


def conditional_y(X: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Sample from conditional distribution Y|X = cos(4 pi X) + 0.5X^2 + Noise
    
    Parameters
    ----------
    X : np.ndarray, shape (n,)
        Covariate values.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    Y : np.ndarray, shape (n,)
        Conditional outcome samples.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    Y = np.cos(4 * np.pi * X) + 0.5 * X**2 + rng.normal(0, 0.5, size=n)
    return Y


def conditional_z(X: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Sample from conditional distribution Z|X = cos(4 pi X) + Noise
    
    Parameters
    ----------
    X : np.ndarray, shape (n,)
        Covariate values.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    Z : np.ndarray, shape (n,)
        Conditional outcome samples.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    Z = np.cos(4 * np.pi * X) + rng.normal(0, 0.5, size=n)
    return Z


def sample_joint(
    n: int,
    marginal_fn: Callable,
    conditional_fn: Callable,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample from joint distribution P_{Y|X} ⊗ P_X.
    
    Parameters
    ----------
    n : int
        Number of samples.
    marginal_fn : callable
        Function (n, seed) -> X that generates covariate samples.
    conditional_fn : callable
        Function (X, noise_std, seed) -> Y that generates conditional samples.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    X : np.ndarray, shape (n,)
        Covariate samples.
    Y : np.ndarray, shape (n,)
        Conditional outcome samples.
    """
    rng = np.random.default_rng(seed)
    seed_x = rng.integers(0, 2**31)
    seed_y = rng.integers(0, 2**31)
    
    X = marginal_fn(n, seed=seed_x)
    Y = conditional_fn(X, seed=seed_y)
    
    return X, Y

def propensity(X: np.ndarray) -> np.ndarray:
    """
    Propensity score function.

    e(x) = 1 / (1 + (1/π) * x^(-0.5) * (1-x)^(-0.5)) for Beta(0.5, 0.5) vs Uniform(0, 1)
    
    Parameters
    ----------
    X : np.ndarray, shape (n,) or (n, 1)
        Covariate values.
    
    Returns
    -------
    e : np.ndarray, shape (n,)
        Propensity scores in (0, 1).
    """
    X = np.asarray(X).flatten()
    e = (1 / (1 + (1 / np.pi) * X**(-0.5) * (1 - X)**(-0.5)))
    # Clip to avoid extreme values
    e = np.clip(e, 1e-1, 1 - 1e-1)
    return e

if __name__ == "__main__":
    # Test sampling functions
    n_samples = 5
    X_p, Y = sample_joint(n_samples, sample_covariate_p, conditional_y, noise_std=0.5, seed=42)
    X_q, Z = sample_joint(n_samples, sample_covariate_q, conditional_z, noise_std=0.5, seed=43)
    
    print("Sample from P:")
    for x, y in zip(X_p, Y):
        print(f"X={x:.3f}, Y={y}")
    
    print("\nSample from Q:")
    for x, z in zip(X_q, Z):
        print(f"X={x:.3f}, Z={z}")