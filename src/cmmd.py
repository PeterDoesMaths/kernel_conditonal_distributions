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
    

def cme_diff(
    x_test: np.ndarray,
    X_P: np.ndarray,
    Y: np.ndarray,
    X_Q: np.ndarray,
    Z: np.ndarray,
    lam_p: float,
    lam_q: float,
    kernel_x: Callable,
    kernel_y: Callable,
    norm: bool = True,
    **kwargs
) -> np.ndarray:
    """
    Evaluate CME difference at test values of x.
    
    Parameters
    ----------
    x_test : np.ndarray, shape (N,)
        Test points where CME difference is evaluated.
    norm : bool, default=True
        If True, returns ||μ̂_{Y|x} - μ̂_{Z|x}|| (RKHS norm, always non-negative).
        If False, returns μ̂_{Y|x} - μ̂_{Z|x} (scalar difference, can be negative).
        For binary outcomes with kronecker delta kernel, this is the difference
        in conditional probabilities P(Y=1|x) - P(Z=1|x).
    
    Returns
    -------
    cme_diff : np.ndarray, shape (N,)
        CME difference at each test point.
    """

    K_XpXp, K_XqXq, _ = _compute_x_kernels(X_P, X_Q, kernel_x, **kwargs)
    W_Xp, W_Xq = _compute_w_matrices(K_XpXp, K_XqXq, lam_p, lam_q)

    K_xXp = kernel_x(x_test, X_P, **kwargs) # (N, n)
    K_xXq = kernel_x(x_test, X_Q, **kwargs) # (N, m)
    K_Xpx = K_xXp.T # (n, N)
    K_Xqx = K_xXq.T # (m, N)

    if not norm:
        # Compute scalar difference: μ̂_{Y|x} - μ̂_{Z|x}
        # For binary Y with kronecker delta kernel: P(Y=1|x) - P(Z=1|x)
        Y_flat = Y.flatten()  # (n,)
        Z_flat = Z.flatten()  # (m,)
        
        mu_Y = Y_flat @ W_Xp @ K_Xpx  # (N,)
        mu_Z = Z_flat @ W_Xq @ K_Xqx  # (N,)
        
        cme_diff = mu_Y - mu_Z  # (N,)
    else:
        # Compute RKHS norm: ||μ̂_{Y|x} - μ̂_{Z|x}||
        L_YY, L_ZZ, L_YZ = _compute_yz_kernels(Y, Z, kernel_y, **kwargs)
        L_ZY = L_YZ.T
        
        # Compute intermediate matrices (one column per test point)
        M1 = W_Xp @ L_YY @ W_Xp @ K_Xpx  # (n, N)
        M2 = W_Xp @ L_YZ @ W_Xq @ K_Xqx  # (n, N)
        M3 = W_Xq @ L_ZY @ W_Xp @ K_Xpx  # (m, N)
        M4 = W_Xq @ L_ZZ @ W_Xq @ K_Xqx  # (m, N)

        term1_diag = np.sum(K_xXp * M1.T, axis=1)  # (N,)
        term2_diag = np.sum(K_xXp * M2.T, axis=1)  # (N,)
        term3_diag = np.sum(K_xXq * M3.T, axis=1)  # (N,)
        term4_diag = np.sum(K_xXq * M4.T, axis=1)  # (N,)

        cme_diff_sq = term1_diag - term2_diag - term3_diag + term4_diag  # (N,)
        cme_diff = np.sqrt(np.maximum(cme_diff_sq, 0))  # (N,)

    return cme_diff


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
        algo_kwargs: dict | None = None,
        stat_kwargs: dict | None = None
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
        algo_kwargs : dict, optional
            Algorithm-level parameters (alpha, B, lam_p, lam_q, random_state, etc.).
        stat_kwargs : dict, optional
            Test-statistic/kernel parameters (bandwidth, estimator, etc.).
        
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
        algo_kwargs: dict | None = None,
        stat_kwargs: dict | None = None
    ) -> Tuple[float, float]:
        """
        Kernel two-sample test for conditional distribution (P_X = Q_X).
        """
        algo_kwargs = algo_kwargs or {}
        stat_kwargs = stat_kwargs or {}

        alpha = float(algo_kwargs.get("alpha", 0.05))
        B = int(algo_kwargs.get("B", 1000))
        lam_p = float(algo_kwargs.get("lam_p", algo_kwargs.get("lam", 1e-3)))
        lam_q = float(algo_kwargs.get("lam_q", algo_kwargs.get("lam", 1e-3)))
        rng = np.random.default_rng(algo_kwargs.get("random_state", None))

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


class Test_Diff_Marginal(Algorithm):
    """
    Kernel two-sample test for conditional distribution (P_X != Q_X).
    
    Uses propensity score-based bootstrap for the case where marginal distributions
    of X differ between the two groups.
    """
    
    def test(
        self,
        X_P: np.ndarray,
        Y: np.ndarray,
        X_Q: np.ndarray,
        Z: np.ndarray,
        test_statistic: TestStatistic,
        kernel_fn: Callable,
        algo_kwargs: dict | None = None,
        stat_kwargs: dict | None = None
    ) -> Tuple[float, float]:
        """
        Kernel two-sample test for conditional distribution (P_X != Q_X).
        """
        algo_kwargs = algo_kwargs or {}
        stat_kwargs = stat_kwargs or {}

        alpha = float(algo_kwargs.get("alpha", 0.05))
        B = int(algo_kwargs.get("B", 1000))
        lam_p = float(algo_kwargs.get("lam_p", algo_kwargs.get("lam", 1e-3)))
        lam_q = float(algo_kwargs.get("lam_q", algo_kwargs.get("lam", 1e-3)))
        propensity_fn = algo_kwargs.get("propensity_fn", None)
        rng = np.random.default_rng(algo_kwargs.get("random_state", None))

        if propensity_fn is None:
            raise ValueError("propensity_fn must be provided for Test_Diff_Marginal")

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

        n = X_P.shape[0]
        m = X_Q.shape[0]

        # Bootstrap with propensity score resampling
        stats_boot = np.zeros(B, dtype=float)
        
        # Compute propensity scores
        e_P = propensity_fn(X_P.flatten())  # P(group=P | X_P)
        e_Q = propensity_fn(X_Q.flatten())  # P(group=P | X_Q)
        
        for b in range(B):
            # Initialize empty lists for bootstrap samples
            X_P_b_list = []
            Y_b_list = []
            X_Q_b_list = []
            Z_b_list = []
            
            # Resample from P using propensity scores
            for i in range(n):
                t_i = rng.binomial(1, e_P[i])  # Bernoulli(e(x_i))
                if t_i == 1:
                    X_P_b_list.append(X_P[i])
                    Y_b_list.append(Y[i])
                else:
                    X_Q_b_list.append(X_P[i])
                    Z_b_list.append(Y[i])
            
            # Resample from Q using propensity scores
            for j in range(m):
                t_j = rng.binomial(1, e_Q[j])  # Bernoulli(e(x_j'))
                if t_j == 1:
                    X_P_b_list.append(X_Q[j])
                    Y_b_list.append(Z[j])
                else:
                    X_Q_b_list.append(X_Q[j])
                    Z_b_list.append(Z[j])
            
            # Only compute statistic if both groups have at least one sample
            if len(X_P_b_list) > 0 and len(X_Q_b_list) > 0:
                X_P_b = np.array(X_P_b_list)
                Y_b = np.array(Y_b_list)
                X_Q_b = np.array(X_Q_b_list)
                Z_b = np.array(Z_b_list)
                
                # Reshape if needed (ensure 2D)
                if X_P_b.ndim == 1:
                    X_P_b = X_P_b.reshape(-1, 1)
                if Y_b.ndim == 1:
                    Y_b = Y_b.reshape(-1, 1)
                if X_Q_b.ndim == 1:
                    X_Q_b = X_Q_b.reshape(-1, 1)
                if Z_b.ndim == 1:
                    Z_b = Z_b.reshape(-1, 1)
                
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
            else:
                # If one group is empty, set stat to 0 (or inf for safety)
                stats_boot[b] = 0.0

        p_value = (1.0 + np.sum(stats_boot > stat)) / (1.0 + B)

        # Decision is p_value < alpha (not returned, but computed for callers)
        _ = p_value < alpha

        return float(stat), float(p_value)


def run_experiment(
    n_trials: int,
    n_samples: int,
    algorithm: Algorithm,
    test_statistic: TestStatistic,
    data_generator: Callable,
    kernel_fn: Callable,
    algo_kwargs: dict | None = None,
    stat_kwargs: dict | None = None
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
    algo_kwargs : dict, optional
        Algorithm-level parameters passed to algorithm.test().
    stat_kwargs : dict, optional
        Test-statistic/kernel parameters passed to algorithm.test().
    
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
            algo_kwargs=algo_kwargs,
            stat_kwargs=stat_kwargs,
        )
        
        results[trial] = [stat, pval]
    
    return results
