"""
Data generating processes for conditional distribution experiments.
"""

import numpy as np
from typing import Tuple, Callable
import inspect


def sample_covariate(n: int, seed: int = None) -> np.ndarray:
    """
    Sample covariates from the base distribution.

    X ~ Beta(4, 4).
    
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

    return rng.beta(4, 4, size=n)


def conditional_y(X: np.ndarray, theta: float, noise_std: float = 0.5, seed: int = None) -> np.ndarray:
    """
    Sample from conditional distribution Y|X = sin(np.pi * X) + epsilon,
    where epsilon ~ N(0, noise_std^2).
    
    Parameters
    ----------
    X : np.ndarray, shape (n,)
        Covariate values.
    theta : float
        Parameter controlling the mean of the conditional distribution.
    noise_std : float, default=0.5
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
    mean = np.sin(np.pi * X)
    noise = rng.normal(0, noise_std, size=n)
    return mean + noise


def conditional_z(
    X: np.ndarray,
    theta: float,
    noise_std: float = 0.5,
    setting: int = 1,
    seed: int = None,
) -> np.ndarray:
    """
    Sample from conditional distribution 
    Setting 1: Z|X = (1 - theta) * sin(np.pi * X) + theta * (3 * X - 0.5) + epsilon
    Setting 2: Z|X = (1 - theta) * sin(np.pi * X) + 0.5 * theta + epsilon,
    where epsilon ~ N(0, noise_std^2).
    
    Parameters
    ----------
    X : np.ndarray, shape (n,)
        Covariate values.
    theta : float
        Parameter controlling the mean of the conditional distribution.
    noise_std : float, default=0.5
        Standard deviation of the noise term.
    setting : int, default=1
        Conditional distribution setting.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    Z : np.ndarray, shape (n,)
        Conditional outcome samples.
    """
    rng = np.random.default_rng(seed)
    n = len(X)

    if setting == 1:
        mean = (1 - theta) * np.sin(np.pi * X) + theta * (3 * X - 0.5)
    elif setting == 2:
        mean = (1 - theta) * np.sin(np.pi * X) + 0.5 * theta
    else:
        raise ValueError(f"Unknown conditional Z setting: {setting}")

    noise = rng.normal(0, noise_std, size=n)
    return mean + noise


def sample_joint(
    n: int,
    theta: float,
    conditional_fn: Callable,
    noise_std: float = 0.5,
    setting: int = 1,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample from joint distribution P_{Y|X} ⊗ P_X.
    
    Parameters
    ----------
    n : int
        Number of samples.
    theta : float
        Parameter controlling the mean of the conditional distribution.
    conditional_fn : callable
        Function (X, theta, noise_std, seed) -> Y that generates conditional samples.
    noise_std : float, default=0.5
        Standard deviation of the noise term.
    setting : int, default=1
        Covariate distribution setting.
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
    conditional_kwargs = {
        "noise_std": noise_std,
        "seed": seed_y,
    }
    if "setting" in inspect.signature(conditional_fn).parameters:
        conditional_kwargs["setting"] = setting

    Y = conditional_fn(X, theta, **conditional_kwargs)
    
    return X, Y
