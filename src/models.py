"""
Data generating processes for conditional distribution experiments.
"""

import numpy as np
from typing import Tuple, Callable


def sample_covariate(n: int, seed: int = None) -> np.ndarray:
    """
    Sample covariates from the base distribution X ~ N(0, 1).
    
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
    return rng.normal(0, 1, size=n)


def conditional_y(X: np.ndarray, noise_std: float = 0.3, seed: int = None) -> np.ndarray:
    """
    Sample from conditional distribution Y|X = exp(-0.5 X^2) sin(2X) + epsilon,
    where epsilon ~ N(0, noise_std^2).
    
    Parameters
    ----------
    X : np.ndarray, shape (n,)
        Covariate values.
    noise_std : float, default=0.3
        Standard deviation of the noise term.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    Y : np.ndarray, shape (n,)
        Conditional outcome samples.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    mean = np.exp(-0.5 * X**2) * np.sin(2 * X)
    noise = rng.normal(0, noise_std, size=n)
    return mean + noise


def conditional_z(X: np.ndarray, noise_std: float = 0.3, seed: int = None) -> np.ndarray:
    """
    Sample from conditional distribution Z|X = X + epsilon,
    where epsilon ~ N(0, noise_std^2).
    
    Parameters
    ----------
    X : np.ndarray, shape (n,)
        Covariate values.
    noise_std : float, default=0.3
        Standard deviation of the noise term.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    Z : np.ndarray, shape (n,)
        Conditional outcome samples.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    noise = rng.normal(0, noise_std, size=n)
    return X + noise


def sample_joint(
    n: int,
    conditional_fn: Callable,
    noise_std: float = 0.3,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample from joint distribution P_{Y|X} ⊗ P_X.
    
    Parameters
    ----------
    n : int
        Number of samples.
    conditional_fn : callable
        Function (X, noise_std, seed) -> Y that generates conditional samples.
    noise_std : float, default=0.3
        Standard deviation of the noise term.
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
    
    X = sample_covariate(n, seed=seed_x)
    Y = conditional_fn(X, noise_std=noise_std, seed=seed_y)
    
    return X, Y
