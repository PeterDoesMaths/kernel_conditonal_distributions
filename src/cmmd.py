"""
Conditional Maximum Mean Discrepancy (CMMD) test statistics and algorithms.
"""

from abc import ABC, abstractmethod
from typing import Callable, Tuple
import numpy as np


def _compute_x_kernels(
    X_P: np.ndarray,
    X_Q: np.ndarray,
    kernel_fn: Callable,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    K_XpXp = kernel_fn(X_P, X_P, **kwargs) # (n, n)
    K_XqXq = kernel_fn(X_Q, X_Q, **kwargs) # (m, m)
    K_XqXp = kernel_fn(X_Q, X_P, **kwargs) # (m, n)
    return K_XpXp, K_XqXq, K_XqXp


def _compute_yz_kernels(
    Y: np.ndarray,
    Z: np.ndarray,
    kernel_fn: Callable,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    L_YY = kernel_fn(Y, Y, **kwargs) # (n, n)
    L_ZZ = kernel_fn(Z, Z, **kwargs) # (m, m)
    L_YZ = kernel_fn(Y, Z, **kwargs) # (n, m)
    return L_YY, L_ZZ, L_YZ


def _compute_w_matrices(
    K_XpXp: np.ndarray,
    K_XqXq: np.ndarray,
    lam_p: float,
    lam_q: float
) -> Tuple[np.ndarray, np.ndarray]:
    n = K_XpXp.shape[0]
    m = K_XqXq.shape[0]
    I_n = np.eye(n)
    I_m = np.eye(m)
    W_Xp = np.linalg.inv(K_XpXp + lam_p * n * I_n)
    W_Xq = np.linalg.inv(K_XqXq + lam_q * m * I_m)
    return W_Xp, W_Xq


def _compute_tilde_kernels(
    X_P: np.ndarray,
    X_Q: np.ndarray,
    kernel_fn: Callable,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_tilde = np.vstack([X_P, X_Q])
    K_Xp_Xtilde = kernel_fn(X_P, X_tilde, **kwargs) # (n, n+m)
    K_Xq_Xtilde = kernel_fn(X_Q, X_tilde, **kwargs) # (m, n+m)
    K_Xtilde_Xtilde = kernel_fn(X_tilde, X_tilde, **kwargs) # (n+m, n+m)
    return K_Xp_Xtilde, K_Xq_Xtilde, K_Xtilde_Xtilde


class TestStatistic(ABC):
    """
    Abstract base class for CMMD-based test statistics.
    
    A test statistic takes samples from two conditional distributions and
    returns a scalar value. Different implementations may use different
    estimators or kernels.
    """
    
    @abstractmethod
    def compute(
        self,
        X_P: np.ndarray,
        Y: np.ndarray,
        X_Q: np.ndarray,
        Z: np.ndarray,
        lam_p: float,
        lam_q: float,
        kernel_fn: Callable,
        **kwargs
    ) -> float:
        """
        Compute the test statistic.
        
        Parameters
        ----------
        X_P : np.ndarray, shape (n, d)
            Covariates from distribution P.
        Y : np.ndarray, shape (n, k)
            Outcomes from distribution P (conditional on X_P).
        X_Q : np.ndarray, shape (m, d)
            Covariates from distribution Q.
        Z : np.ndarray, shape (m, k)
            Outcomes from distribution Q (conditional on X_Q).
        lam_p : float
            Regularization parameter for W_X = (K_XX + lam_p * n * I_n)^{-1}.
        lam_q : float
            Regularization parameter for W_X' = (K_X'X' + lam_q * m * I_m)^{-1}.
        kernel_fn : callable
            Kernel function k(X, Y, **kwargs) that computes kernel matrices.
        **kwargs
            Additional arguments passed to kernel_fn (e.g., bandwidth).
        
        Returns
        -------
        stat : float
            The computed test statistic value.
        """
        pass


class CMMD0(TestStatistic):
    """
    CMMD0 test statistic.
    
    Compares two conditional distributions P(Y|X) and Q(Z|X) using conditional mean operators.
    """
    
    def compute(
        self,
        X_P: np.ndarray,
        Y: np.ndarray,
        X_Q: np.ndarray,
        Z: np.ndarray,
        lam_p: float,
        lam_q: float,
        kernel_fn: Callable,
        **kwargs
    ) -> float:
        """
        Compute CMMD0 test statistic.
        """
        K_XpXp, K_XqXq, K_XqXp = _compute_x_kernels(X_P, X_Q, kernel_fn, **kwargs)
        L_YY, L_ZZ, L_YZ = _compute_yz_kernels(Y, Z, kernel_fn, **kwargs)
        W_Xp, W_Xq = _compute_w_matrices(K_XpXp, K_XqXq, lam_p, lam_q)
        
        # Compute the three terms of the CMMD statistic
        # Term 1: Tr(W_X L_YY W_X K_XX)
        term1 = np.trace(W_Xp @ L_YY @ W_Xp @ K_XpXp)
        
        # Term 2: -2 * Tr(W_X L_YZ W_X' K_X'X)
        term2 = -2.0 * np.trace(W_Xp @ L_YZ @ W_Xq @ K_XqXp)
        
        # Term 3: Tr(W_X' L_ZZ W_X' K_X'X')
        term3 = np.trace(W_Xq @ L_ZZ @ W_Xq @ K_XqXq)
        
        # CMMD statistic
        cmmd = term1 + term2 + term3
        
        return float(cmmd)


class CMMD1(TestStatistic):
    """
    CMMD1 test statistic.

    Compares two conditional distributions P(Y|X) and Q(Z|X) using conditional mean embeddings.
    
    Uses pooled covariates \tilde{X} = (X, X')
    """
    
    def compute(
        self,
        X_P: np.ndarray,
        Y: np.ndarray,
        X_Q: np.ndarray,
        Z: np.ndarray,
        lam_p: float,
        lam_q: float,
        kernel_fn: Callable,
        **kwargs
    ) -> float:
        """
        Compute CMMD1 test statistic.
        """
        n = X_P.shape[0]
        m = X_Q.shape[0]

        K_XpXp, K_XqXq, _ = _compute_x_kernels(X_P, X_Q, kernel_fn, **kwargs)
        K_Xp_Xtilde, K_Xq_Xtilde, _ = _compute_tilde_kernels(
            X_P, X_Q, kernel_fn, **kwargs
        )
        K_Xtilde_Xp = K_Xp_Xtilde.T
        K_Xtilde_Xq = K_Xq_Xtilde.T
        L_YY, L_ZZ, L_YZ = _compute_yz_kernels(Y, Z, kernel_fn, **kwargs)
        W_Xp, W_Xq = _compute_w_matrices(K_XpXp, K_XqXq, lam_p, lam_q)

        # Compute the three terms of the CMMD statistic
        # Term 1: Tr(W_X L_YY W_X K_Xtilde K_tildeX)
        term1 = np.trace(W_Xp @ L_YY @ W_Xp @ K_Xp_Xtilde @ K_Xtilde_Xp)

        # Term 2: -2 * Tr(W_X L_YZ W_X' K_X'tilde K_tildeX)
        term2 = -2.0 * np.trace(W_Xp @ L_YZ @ W_Xq @ K_Xq_Xtilde @ K_Xtilde_Xp)

        # Term 3: Tr(W_X' L_ZZ W_X' K_X'tilde K_tildeX')
        term3 = np.trace(W_Xq @ L_ZZ @ W_Xq @ K_Xq_Xtilde @ K_Xtilde_Xq)

        cmmd = (term1 + term2 + term3) / (n + m)

        return float(cmmd)


class CMMD2(TestStatistic):
    """
    CMMD2 test statistic.

    Compares two conditional distributions P(Y|X) and Q(Z|X) using joint mean embeddings.
    """
    
    def compute(
        self,
        X_P: np.ndarray,
        Y: np.ndarray,
        X_Q: np.ndarray,
        Z: np.ndarray,
        lam_p: float,
        lam_q: float,
        kernel_fn: Callable,
        estimator: str = "cmmd",
        **kwargs
    ) -> float:
        """
        Compute CMMD2 test statistic.
        """
        estimator = estimator.lower()
        n = X_P.shape[0]
        m = X_Q.shape[0]

        K_XpXp, K_XqXq, K_XqXp = _compute_x_kernels(X_P, X_Q, kernel_fn, **kwargs)
        L_YY, L_ZZ, L_YZ = _compute_yz_kernels(Y, Z, kernel_fn, **kwargs)

        if estimator == "jmmd":
            # Term 1: Tr(L_YY K_XX)
            term1 = np.trace(L_YY @ K_XpXp)

            # Term 2: -2 * Tr(L_YZ K_X'X)
            term2 = -2.0 * np.trace(L_YZ @ K_XqXp)

            # Term 3: Tr(L_ZZ K_X'X')
            term3 = np.trace(L_ZZ @ K_XqXq)

            cmmd = (term1 + term2 + term3) / (n + m) ** 2
            return float(cmmd)

        if estimator == "cmmd":
            W_Xp, W_Xq = _compute_w_matrices(K_XpXp, K_XqXq, lam_p, lam_q)
            K_Xp_Xtilde, K_Xq_Xtilde, K_Xtilde_Xtilde = _compute_tilde_kernels(X_P, X_Q, kernel_fn, **kwargs)
            K_Xtilde_Xp = K_Xp_Xtilde.T
            K_Xtilde_Xq = K_Xq_Xtilde.T

            # Term 1: Tr(W_X L_YY W_X K_Xp_Xtilde K_Xtilde_Xtilde K_Xtilde_Xp)
            term1 = np.trace(
                W_Xp @ L_YY @ W_Xp @ K_Xp_Xtilde @ K_Xtilde_Xtilde @ K_Xtilde_Xp
            )

            # Term 2: -2 * Tr(W_X L_YZ W_X' K_Xq_Xtilde K_Xtilde_Xtilde K_Xtilde_Xp)
            term2 = -2.0 * np.trace(
                W_Xp @ L_YZ @ W_Xq @ K_Xq_Xtilde @ K_Xtilde_Xtilde @ K_Xtilde_Xp
            )

            # Term 3: Tr(W_X' L_ZZ W_X' K_Xq_Xtilde K_Xtilde_Xtilde K_Xtilde_Xq)
            term3 = np.trace(
                W_Xq @ L_ZZ @ W_Xq @ K_Xq_Xtilde @ K_Xtilde_Xtilde @ K_Xtilde_Xq
            )

            cmmd = (term1 + term2 + term3) / (n + m) ** 2
            return float(cmmd)

        raise ValueError(
            f"Unknown estimator '{estimator}'. Use 'jmmd' or 'cmmd'."
        )


class Algorithm(ABC):
    """
    Abstract base class for hypothesis testing algorithms.
    
    An algorithm takes raw samples and a test statistic, and performs
    the actual hypothesis test (e.g., permutation test, bootstrap, etc.).
    """
    
    @abstractmethod
    def test(
        self,
        X_P: np.ndarray,
        Y: np.ndarray,
        X_Q: np.ndarray,
        Z: np.ndarray,
        test_statistic: TestStatistic,
        kernel_fn: Callable,
        **kwargs
    ) -> Tuple[float, float]:
        """
        Perform hypothesis test.
        
        Parameters
        ----------
        X_P : np.ndarray, shape (n, d)
            Covariates from distribution P.
        Y : np.ndarray, shape (n, k)
            Outcomes from distribution P (conditional on X_P).
        X_Q : np.ndarray, shape (m, d)
            Covariates from distribution Q.
        Z : np.ndarray, shape (m, k)
            Outcomes from distribution Q (conditional on X_Q).
        test_statistic : TestStatistic
            The test statistic to use.
        kernel_fn : callable
            Kernel function k(X, Y, **kwargs).
        **kwargs
            Additional arguments (bandwidth, significance level, etc.).
        
        Returns
        -------
        stat : float
            The computed test statistic on the original data.
        p_value : float
            The estimated p-value for the test.
        """
        pass


class Test_Same_Marginal(Algorithm):
    """
    Kernel two-sample test for conditional distribution (P_X = Q_X).
    """
    
    def test(
        self,
        X_P: np.ndarray,
        Y: np.ndarray,
        X_Q: np.ndarray,
        Z: np.ndarray,
        test_statistic: TestStatistic,
        kernel_fn: Callable,
        **kwargs
    ) -> Tuple[float, float]:
        """
        Kernel two-sample test for conditional distribution (P_X = Q_X).
        """
        alpha = float(kwargs.get("alpha", 0.05))
        B = int(kwargs.get("B", 1000))
        lam_p = float(kwargs.get("lam_p", kwargs.get("lam", 1e-3)))
        lam_q = float(kwargs.get("lam_q", kwargs.get("lam", 1e-3)))
        rng = np.random.default_rng(kwargs.get("random_state", None))

        # Avoid passing algorithm-only arguments into the test statistic.
        stat_kwargs = dict(kwargs)
        for key in ("alpha", "B", "lam_p", "lam_q", "lam", "random_state"):
            stat_kwargs.pop(key, None)

        # Compute test statistic on original data
        stat = test_statistic.compute(
            X_P,
            Y,
            X_Q,
            Z,
            lam_p,
            lam_q,
            kernel_fn,
            **stat_kwargs,
        )

        X_all = np.vstack([X_P, X_Q])
        Y_all = np.vstack([Y, Z])
        n = X_P.shape[0]
        total = X_all.shape[0]

        # Bootstrap/permutation under same-marginal null
        stats_boot = np.zeros(B, dtype=float)
        indices = np.arange(total)
        for b in range(B):
            idx_p = rng.choice(indices, size=n, replace=False)
            mask_p = np.zeros(total, dtype=bool)
            mask_p[idx_p] = True
            idx_q = indices[~mask_p]

            X_P_b = X_all[idx_p]
            Y_b = Y_all[idx_p]
            X_Q_b = X_all[idx_q]
            Z_b = Y_all[idx_q]

            stats_boot[b] = test_statistic.compute(
                X_P_b,
                Y_b,
                X_Q_b,
                Z_b,
                lam_p,
                lam_q,
                kernel_fn,
                **stat_kwargs,
            )

        p_value = (1.0 + np.sum(stats_boot > stat)) / (1.0 + B)

        # Decision is p_value < alpha (not returned, but computed for callers)
        _ = p_value < alpha

        return float(stat), float(p_value)


class AlgorithmV2(Algorithm):
    """
    Second algorithm for hypothesis testing (template).
    
    To be implemented based on your specific method.
    """
    
    def test(
        self,
        X_P: np.ndarray,
        Y: np.ndarray,
        X_Q: np.ndarray,
        Z: np.ndarray,
        test_statistic: TestStatistic,
        kernel_fn: Callable,
        **kwargs
    ) -> Tuple[float, float]:
        """
        Perform hypothesis test using Algorithm V2.
        
        This is a template - implement your specific algorithm here.
        """
        raise NotImplementedError("Implement your specific algorithm here.")


def run_experiment(
    n_trials: int,
    n_samples: int,
    algorithm: Algorithm,
    test_statistic: TestStatistic,
    data_generator: Callable,
    kernel_fn: Callable,
    **kwargs
) -> np.ndarray:
    """
    Run hypothesis test multiple times and collect results.
    
    Parameters
    ----------
    n_trials : int
        Number of independent trials to run.
    n_samples : int
        Number of samples in each trial.
    algorithm : Algorithm
        The hypothesis testing algorithm to use.
    test_statistic : TestStatistic
        The test statistic to compute.
    data_generator : callable
        Function that generates (X_P, Y, X_Q, Z) samples.
    kernel_fn : callable
        Kernel function to use.
    **kwargs
        Additional arguments passed to algorithm.test().
    
    Returns
    -------
    results : np.ndarray, shape (n_trials, 2)
        Array where each row is [test_statistic, p_value] for each trial.
    """
    results = np.zeros((n_trials, 2))
    
    for trial in range(n_trials):
        # Generate data for this trial
        X_P, Y, X_Q, Z = data_generator(n_samples)
        
        # Run test
        stat, pval = algorithm.test(
            X_P, Y, X_Q, Z,
            test_statistic,
            kernel_fn,
            **kwargs
        )
        
        results[trial] = [stat, pval]
    
    return results
