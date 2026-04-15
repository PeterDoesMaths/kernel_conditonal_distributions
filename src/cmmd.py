"""
Conditional Maximum Mean Discrepancy (CMMD) test statistics and algorithms.
"""

from abc import ABC, abstractmethod
from typing import Callable, Tuple
import numpy as np


def _compute_x_kernels(
    X_P: np.ndarray,
    X_Q: np.ndarray,
    kernel_x: Callable,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    K_XpXp = kernel_x(X_P, X_P, **kwargs) # (n, n)
    K_XqXq = kernel_x(X_Q, X_Q, **kwargs) # (m, m)
    K_XqXp = kernel_x(X_Q, X_P, **kwargs) # (m, n)
    return K_XpXp, K_XqXq, K_XqXp


def _compute_yz_kernels(
    Y: np.ndarray,
    Z: np.ndarray,
    kernel_y: Callable,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    L_YY = kernel_y(Y, Y, **kwargs) # (n, n)
    L_ZZ = kernel_y(Z, Z, **kwargs) # (m, m)
    L_YZ = kernel_y(Y, Z, **kwargs) # (n, m)
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
    kernel_x: Callable,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_tilde = np.vstack([X_P, X_Q])
    K_Xp_Xtilde = kernel_x(X_P, X_tilde, **kwargs) # (n, n+m)
    K_Xq_Xtilde = kernel_x(X_Q, X_tilde, **kwargs) # (m, n+m)
    K_Xtilde_Xtilde = kernel_x(X_tilde, X_tilde, **kwargs) # (n+m, n+m)
    return K_Xp_Xtilde, K_Xq_Xtilde, K_Xtilde_Xtilde


def _psd_matrix_power(K: np.ndarray, power: float) -> np.ndarray:
    """
    Compute K**power for a PSD matrix K via eigen-decomposition.
    """
    K_sym = 0.5 * (K + K.T)
    eigenvalues, eigenvectors = np.linalg.eigh(K_sym)
    eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)
    return (eigenvectors * (eigenvalues ** power)) @ eigenvectors.T


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
        kernel_x: Callable,
        kernel_y: Callable | None = None,
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
        kernel_x : callable
            Kernel function for covariates X.
        kernel_y : callable, optional
            Kernel function for outcomes Y/Z. If None, uses kernel_x.
        **kwargs
            Additional arguments passed to kernels (e.g., bandwidth).
        
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
        kernel_x: Callable,
        kernel_y: Callable | None = None,
        **kwargs
    ) -> float:
        """
        Compute CMMD0 test statistic.
        """
        if kernel_y is None:
            kernel_y = kernel_x
        
        K_XpXp, K_XqXq, K_XqXp = _compute_x_kernels(X_P, X_Q, kernel_x, **kwargs)
        L_YY, L_ZZ, L_YZ = _compute_yz_kernels(Y, Z, kernel_y, **kwargs)
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
    

class CMMD0_dr(TestStatistic):
    """
    CMMD0 DR test statistic.
    
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
        kernel_x: Callable,
        kernel_y: Callable | None = None,
        **kwargs
    ) -> float:
        """
        Compute CMMD0 DR test statistic.
        """
        if kernel_y is None:
            kernel_y = kernel_x

        propensity = kwargs.pop("propensity", None)
        if propensity is None:
            raise ValueError("CMMD0_dr requires 'propensity' in stat_kwargs")
        
        cme_y = kwargs.pop("cme_y", None)
        cme_z = kwargs.pop("cme_z", None)

        lam = lam_p
        
        # Merge test set 
        X = np.concatenate([X_P, X_Q])
        YZ_combined = np.concatenate([Y, Z]).reshape(-1)

        # T indicates which samples are from P (T=1) vs Q (T=0)
        T = np.concatenate([np.ones_like(Y), np.zeros_like(Z)]).reshape(-1)

        # Construct propensity vector
        E = propensity(X)
        E = np.clip(np.asarray(E).reshape(-1), 1e-6, 1 - 1e-6)
        E_tilde = (T - E) / (E * (1 - E))

        # CME models for Y and Z
        mu_Y = cme_y(X)
        mu_Z = cme_z(X)

        # Compute feat operator
        Phi_W = np.asarray(YZ_combined).reshape(-1)

        # Pseudo-outcome operator
        Phi = np.asarray((Phi_W - (1 - E) * mu_Y - E * mu_Z) * E_tilde).reshape(-1)

        # Kernel matrix and inverse for DR test statistic
        K_X = kernel_x(X, X, **kwargs)
        n_x = X.shape[0]
        W_X = np.linalg.inv(K_X + lam * n_x * np.eye(n_x))
        
        # CMMD statistic
        cmmd = float(Phi @ W_X @ K_X @ W_X @ Phi.T)
        
        return cmmd
    

class CMMD1_dr(TestStatistic):
    """
    CMMD1 DR test statistic.
    
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
        kernel_x: Callable,
        kernel_y: Callable | None = None,
        **kwargs
    ) -> float:
        """
        Compute CMMD1 DR test statistic.
        """
        if kernel_y is None:
            kernel_y = kernel_x

        propensity = kwargs.pop("propensity", None)

        cme_y = kwargs.pop("cme_y", None)
        cme_z = kwargs.pop("cme_z", None)

        lam = lam_p

        # Merge test set 
        X = np.concatenate([X_P, X_Q])
        YZ_combined = np.concatenate([Y, Z]).reshape(-1)

        # T indicates which samples are from P (T=1) vs Q (T=0)
        T = np.concatenate([np.ones_like(Y), np.zeros_like(Z)]).reshape(-1)

        # Construct propensity vector
        E = propensity(X)
        E = np.clip(np.asarray(E).reshape(-1), 1e-6, 1 - 1e-6)
        E_tilde = (T - E) / (E * (1 - E))

        # # CME models for Y and Z
        mu_Y = cme_y(X)
        mu_Z = cme_z(X)

        # Compute feat operator
        Phi_W = np.asarray(YZ_combined).reshape(-1)

        # Pseudo-outcome operator
        Phi = np.asarray((Phi_W - (1 - E) * mu_Y - E * mu_Z) * E_tilde).reshape(-1)

        # Kernel matrix and inverse for DR test statistic
        K_X = kernel_x(X, X, **kwargs)
        n_x = X.shape[0]
        W_X = np.linalg.inv(K_X + lam * n_x * np.eye(n_x))

        # CMMD1
        cmmd = float(Phi @ W_X @ K_X @ K_X @ W_X @ Phi.T) / n_x
        
        return cmmd
    

class CMMD2_dr(TestStatistic):
    """
    CMMD2 DR test statistic.
    
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
        kernel_x: Callable,
        kernel_y: Callable | None = None,
        **kwargs
    ) -> float:
        """
        Compute CMMD2 DR test statistic.
        """
        if kernel_y is None:
            kernel_y = kernel_x

        propensity = kwargs.pop("propensity", None)

        cme_y = kwargs.pop("cme_y", None)
        cme_z = kwargs.pop("cme_z", None)

        lam = lam_p

        # Merge test set 
        X = np.concatenate([X_P, X_Q])
        YZ_combined = np.concatenate([Y, Z]).reshape(-1)

        # T indicates which samples are from P (T=1) vs Q (T=0)
        T = np.concatenate([np.ones_like(Y), np.zeros_like(Z)]).reshape(-1)

        # Construct propensity vector
        E = propensity(X)
        E = np.clip(np.asarray(E).reshape(-1), 1e-6, 1 - 1e-6)
        E_tilde = (T - E) / (E * (1 - E))

        # CME models for Y and Z
        mu_Y = cme_y(X)
        mu_Z = cme_z(X)

        # Compute feat operator
        Phi_W = np.asarray(YZ_combined).reshape(-1)

        # Pseudo-outcome operator
        Phi = np.asarray((Phi_W - (1 - E) * mu_Y - E * mu_Z) * E_tilde).reshape(-1)

        # Kernel matrix and inverse for DR test statistic
        K_X = kernel_x(X, X, **kwargs)
        n_x = X.shape[0]
        W_X = np.linalg.inv(K_X + lam * n_x * np.eye(n_x))

        # CMMD2
        cmmd = float(Phi @ W_X @ K_X @ K_X @ K_X @ W_X @ Phi.T) / (n_x ** 2)
        
        return cmmd


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
        kernel_x: Callable,
        kernel_y: Callable | None = None,
        **kwargs
    ) -> float:
        """
        Compute CMMD1 test statistic.
        """
        if kernel_y is None:
            kernel_y = kernel_x
        
        n = X_P.shape[0]
        m = X_Q.shape[0]

        K_XpXp, K_XqXq, _ = _compute_x_kernels(X_P, X_Q, kernel_x, **kwargs)
        K_Xp_Xtilde, K_Xq_Xtilde, _ = _compute_tilde_kernels(
            X_P, X_Q, kernel_x, **kwargs
        )
        K_Xtilde_Xp = K_Xp_Xtilde.T
        K_Xtilde_Xq = K_Xq_Xtilde.T
        L_YY, L_ZZ, L_YZ = _compute_yz_kernels(Y, Z, kernel_y, **kwargs)
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
        kernel_x: Callable,
        kernel_y: Callable | None = None,
        estimator: str = "cmmd",
        **kwargs
    ) -> float:
        """
        Compute CMMD2 test statistic.
        """
        if kernel_y is None:
            kernel_y = kernel_x
        
        estimator = estimator.lower()
        n = X_P.shape[0]
        m = X_Q.shape[0]

        K_XpXp, K_XqXq, K_XqXp = _compute_x_kernels(X_P, X_Q, kernel_x, **kwargs)
        L_YY, L_ZZ, L_YZ = _compute_yz_kernels(Y, Z, kernel_y, **kwargs)

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
            K_Xp_Xtilde, K_Xq_Xtilde, K_Xtilde_Xtilde = _compute_tilde_kernels(X_P, X_Q, kernel_x, **kwargs)
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
    
class CMMD0_primal(TestStatistic):
    """
    CMMD0 test statistic using primal estimator of CMO.
    
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
        kernel_x: Callable,
        kernel_y: Callable | None = None,
        **kwargs
    ) -> float:
        """
        Compute CMMD0 test statistic.

        X_P, X_Q are categorical so we use kronecker delta kernel and compute CMO in primal form.
        """
        if kernel_y is None:
            kernel_y = kernel_x

        # Flatten X_P and X_Q since they're categorical
        X_P_flat = np.asarray(X_P).flatten()
        X_Q_flat = np.asarray(X_Q).flatten()

        # Compute one hot encoding of X_P and X_Q
        unique_X = np.unique(np.concatenate([np.unique(X_P_flat), np.unique(X_Q_flat)]))
        Phi_Xp = np.zeros((unique_X.size, X_P_flat.shape[0]))
        for i, x in enumerate(unique_X):
            Phi_Xp[i, X_P_flat == x] = 1.0

        Phi_Xq = np.zeros((unique_X.size, X_Q_flat.shape[0]))
        for i, x in enumerate(unique_X):
            Phi_Xq[i, X_Q_flat == x] = 1.0

        # K_XpXp, K_XqXq, K_XqXp = _compute_x_kernels(X_P, X_Q, kernel_x, **kwargs)
        L_YY, L_ZZ, L_YZ = _compute_yz_kernels(Y, Z, kernel_y, **kwargs)
        W_Xp, W_Xq = _compute_w_matrices(Phi_Xp @ Phi_Xp.T, Phi_Xq @ Phi_Xq.T, lam_p, lam_q)
        
        # Compute the three terms of the CMMD statistic
        # Term 1: Tr(W_X Phi_X L_YY Phi_X^* W_X)
        term1 = np.trace(W_Xp @ Phi_Xp @ L_YY @ Phi_Xp.T @ W_Xp)
        
        # Term 2: -2 * Tr(W_X Phi_X L_YZ Phi_X'^* W_X')
        term2 = -2.0 * np.trace(W_Xp @ Phi_Xp @ L_YZ @ Phi_Xq.T @ W_Xq)
        
        # Term 3: Tr(W_X' Phi_X' L_ZZ Phi_X'^* W_X')
        term3 = np.trace(W_Xq @ Phi_Xq @ L_ZZ @ Phi_Xq.T @ W_Xq)
        
        # CMMD statistic
        cmmd = term1 + term2 + term3
        
        return float(cmmd)
    
class CMMD1_primal(TestStatistic):
    """
    CMMD1 test statistic using primal estimator of CMO.
    
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
        kernel_x: Callable,
        kernel_y: Callable | None = None,
        **kwargs
    ) -> float:
        """
        Compute CMMD1 test statistic.

        X_P, X_Q are categorical so we use kronecker delta kernel and compute CMO in primal form.
        """
        if kernel_y is None:
            kernel_y = kernel_x

        n = X_P.shape[0]
        m = X_Q.shape[0]

        # Flatten X_P and X_Q since they're categorical
        X_P_flat = np.asarray(X_P).flatten()
        X_Q_flat = np.asarray(X_Q).flatten()

        # Compute one hot encoding of X_P and X_Q
        unique_X = np.unique(np.concatenate([np.unique(X_P_flat), np.unique(X_Q_flat)]))
        Phi_Xp = np.zeros((unique_X.size, X_P_flat.shape[0]))
        for i, x in enumerate(unique_X):
            Phi_Xp[i, X_P_flat == x] = 1.0

        Phi_Xq = np.zeros((unique_X.size, X_Q_flat.shape[0]))
        for i, x in enumerate(unique_X):
            Phi_Xq[i, X_Q_flat == x] = 1.0

        # K_XpXp, K_XqXq, K_XqXp = _compute_x_kernels(X_P, X_Q, kernel_x, **kwargs)
        L_YY, L_ZZ, L_YZ = _compute_yz_kernels(Y, Z, kernel_y, **kwargs)
        W_Xp, W_Xq = _compute_w_matrices(Phi_Xp @ Phi_Xp.T, Phi_Xq @ Phi_Xq.T, lam_p, lam_q)
        
        # Estimate covariance matrix
        C_XX = (1 / (n + m)) * (Phi_Xp @ Phi_Xp.T + Phi_Xq @ Phi_Xq.T)

        # Compute the three terms of the CMMD statistic
        # Term 1: Tr(W_X Phi_X L_YY Phi_X^* W_X C_XX)
        term1 = np.trace(W_Xp @ Phi_Xp @ L_YY @ Phi_Xp.T @ W_Xp @ C_XX)
        
        # Term 2: -2 * Tr(W_X Phi_X L_YZ Phi_X'^* W_X' C_XX)
        term2 = -2.0 * np.trace(W_Xp @ Phi_Xp @ L_YZ @ Phi_Xq.T @ W_Xq @ C_XX)
        
        # Term 3: Tr(W_X' Phi_X' L_ZZ Phi_X'^* W_X' C_XX)
        term3 = np.trace(W_Xq @ Phi_Xq @ L_ZZ @ Phi_Xq.T @ W_Xq @ C_XX)
        
        # CMMD statistic
        cmmd = term1 + term2 + term3
        
        return float(cmmd)

class CMMD2_primal(TestStatistic):
    """
    CMMD2 test statistic using primal estimator of CMO.
    
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
        kernel_x: Callable,
        kernel_y: Callable | None = None,
        **kwargs
    ) -> float:
        """
        Compute CMMD2 test statistic.

        X_P, X_Q are categorical so we use kronecker delta kernel and compute CMO in primal form.
        """
        if kernel_y is None:
            kernel_y = kernel_x

        n = X_P.shape[0]
        m = X_Q.shape[0]

        # Flatten X_P and X_Q since they're categorical
        X_P_flat = np.asarray(X_P).flatten()
        X_Q_flat = np.asarray(X_Q).flatten()

        # Compute one hot encoding of X_P and X_Q
        unique_X = np.unique(np.concatenate([np.unique(X_P_flat), np.unique(X_Q_flat)]))
        Phi_Xp = np.zeros((unique_X.size, X_P_flat.shape[0]))
        for i, x in enumerate(unique_X):
            Phi_Xp[i, X_P_flat == x] = 1.0

        Phi_Xq = np.zeros((unique_X.size, X_Q_flat.shape[0]))
        for i, x in enumerate(unique_X):
            Phi_Xq[i, X_Q_flat == x] = 1.0

        # K_XpXp, K_XqXq, K_XqXp = _compute_x_kernels(X_P, X_Q, kernel_x, **kwargs)
        L_YY, L_ZZ, L_YZ = _compute_yz_kernels(Y, Z, kernel_y, **kwargs)
        W_Xp, W_Xq = _compute_w_matrices(Phi_Xp @ Phi_Xp.T, Phi_Xq @ Phi_Xq.T, lam_p, lam_q)
        
        # Estimate covariance matrix
        C_XX = (1 / (n + m)) * (Phi_Xp @ Phi_Xp.T + Phi_Xq @ Phi_Xq.T)

        # Compute the three terms of the CMMD statistic
        # Term 1: Tr(W_X Phi_X L_YY Phi_X^* W_X C_XX^2)
        term1 = np.trace(W_Xp @ Phi_Xp @ L_YY @ Phi_Xp.T @ W_Xp @ C_XX @ C_XX)
        
        # Term 2: -2 * Tr(W_X Phi_X L_YZ Phi_X'^* W_X' C_XX^2)
        term2 = -2.0 * np.trace(W_Xp @ Phi_Xp @ L_YZ @ Phi_Xq.T @ W_Xq @ C_XX @ C_XX)
        
        # Term 3: Tr(W_X' Phi_X' L_ZZ Phi_X'^* W_X' C_XX^2)
        term3 = np.trace(W_Xq @ Phi_Xq @ L_ZZ @ Phi_Xq.T @ W_Xq @ C_XX @ C_XX)
        
        # CMMD statistic
        cmmd = term1 + term2 + term3
        
        return float(cmmd)


class CMMDs(TestStatistic):
    """
    CMMDs test statistic.

    Compares two conditional distributions P(Y|X) and Q(Z|X) using conditional mean embeddings.
    
    Uses pooled covariates \tilde{X} = (X, X')
    """

    # initialize with level parameter for smoothing matrix
    def __init__(self, level: float = 0.5):
        self.level = float(level)
    
    def compute(
        self,
        X_P: np.ndarray,
        Y: np.ndarray,
        X_Q: np.ndarray,
        Z: np.ndarray,
        lam_p: float,
        lam_q: float,
        kernel_x: Callable,
        kernel_y: Callable | None = None,
        **kwargs
    ) -> float:
        """
        Compute CMMDs test statistic.
        """
        if kernel_y is None:
            kernel_y = kernel_x
        
        n = X_P.shape[0]
        m = X_Q.shape[0]

        # Compute kernel matrices and inverses
        K_XpXp, K_XqXq, _ = _compute_x_kernels(X_P, X_Q, kernel_x, **kwargs)
        X_tilde = np.vstack([X_P, X_Q])
        K_Xtilde_Xtilde = kernel_x(X_tilde, X_tilde, **kwargs) # (n+m, n+m)
        L_YY, L_ZZ, L_YZ = _compute_yz_kernels(Y, Z, kernel_y, **kwargs)
        W_Xp, W_Xq = _compute_w_matrices(K_XpXp, K_XqXq, lam_p, lam_q)

        # Compute smoothing matrix
        s = self.level
        K = _psd_matrix_power(K_Xtilde_Xtilde, s + 1)

        # Get projection matrices
        P_Xp = np.concatenate([np.eye(n), np.zeros((n, m))], axis=1) # (n, n+m)
        P_Xq = np.concatenate([np.zeros((m, n)), np.eye(m)], axis=1) # (m, n+m)

        # Compute the three terms of the CMMD statistic
        # Term 1: Tr(W_X L_YY W_X K_Xtilde K_tildeX)
        term1 = np.trace(W_Xp @ L_YY @ W_Xp @ P_Xp @ K @ P_Xp.T)

        # Term 2: -2 * Tr(W_X L_YZ W_X' K_X'tilde K_tildeX)
        term2 = -2.0 * np.trace(W_Xp @ L_YZ @ W_Xq @ P_Xq @ K @ P_Xp.T)

        # Term 3: Tr(W_X' L_ZZ W_X' K_X'tilde K_tildeX')
        term3 = np.trace(W_Xq @ L_ZZ @ W_Xq @ P_Xq @ K @ P_Xq.T)

        cmmd = (term1 + term2 + term3) / (n + m)

        return float(cmmd)

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
        kernel_x: Callable,
        kernel_y: Callable | None = None,
        algo_kwargs: dict | None = None,
        stat_kwargs: dict | None = None
    ) -> Tuple[float, float]:
        """
        Kernel two-sample test for conditional distribution (P_X = Q_X).
        """
        if kernel_y is None:
            kernel_y = kernel_x
        
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
            kernel_x,
            kernel_y,
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
                kernel_x,
                kernel_y,
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
        kernel_x: Callable,
        kernel_y: Callable | None = None,
        algo_kwargs: dict | None = None,
        stat_kwargs: dict | None = None
    ) -> Tuple[float, float]:
        """
        Kernel two-sample test for conditional distribution (P_X != Q_X).
        """
        if kernel_y is None:
            kernel_y = kernel_x
        
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
            kernel_x,
            kernel_y,
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
                    kernel_x,
                    kernel_y,
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
