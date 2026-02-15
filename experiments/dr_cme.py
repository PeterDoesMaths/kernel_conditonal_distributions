"""
Experiment: Conditional Mean Embedding (CME) using Kernel Ridge Regression.

This script samples from the models defined in dr_models.py and computes
the conditional mean embeddings for Y and Z using kernel ridge regression.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.dr_models import sample_joint, sample_covariate_p, sample_covariate_q, conditional_y, conditional_z, propensity
from src.kernels import gaussian_kernel, kronecker_delta_kernel

def cme_model(
    x_test: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    lam: float,
    kernel_x: Callable,
    **kwargs
) -> np.ndarray:
    """
    Evaluate CME at test values of x.
    
    Parameters
    ----------
    x_test : np.ndarray, shape (N,)
        Test points where CME  is evaluated.
    
    Returns
    -------
    cme : np.ndarray, shape (N,)
        CME at each test point.
    """

    K = kernel_x(X, X, **kwargs)  # (n, n)
    W_X = np.linalg.inv(K + lam * X.shape[0] * np.eye(X.shape[0]))  # (n, n)

    K_Xx = kernel_x(X, x_test, **kwargs) # (n, N)

    # Compute scalar difference: μ̂_{Y|x}
    Y_flat = Y.flatten()  # (n,)
    
    mu_Y = Y_flat @ W_X @ K_Xx  # (N,)

    return mu_Y

def plot_cme_difference(x_eval: np.ndarray, true_diff: np.ndarray, standard_cme_diff: np.ndarray, dr_cme_diff: np.ndarray):
    """
    Plot the true difference, standard CME difference, and doubly robust CME difference.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(x_eval, true_diff, '--', label="True Difference", 
            linewidth=2, alpha=0.7, color='black')
    ax.plot(x_eval, standard_cme_diff, label="Standard Estimator", 
            linewidth=2, color='blue')
    ax.plot(x_eval, dr_cme_diff, label="Doubly Robust Estimator", 
            linewidth=2, color='red')
    ax.set_xlabel("$x$", fontsize=20)
    ax.set_ylabel(r"$\mu_{Y|x} - \mu_{Z|x}$", fontsize=20)
    ax.set_title("Difference in Conditional Mean Embeddings", fontsize=24)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent.parent / "figs" / "dr"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cme_difference.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    plt.show()


def main():
    # Set random seed for reproducibility
    seed = 1
    
    # Sample size
    n_samples = 1000
    
    # Sample from P and Q
    X_p, Y = sample_joint(
        n_samples, sample_covariate_p, conditional_y, seed=seed
    )
    X_q, Z = sample_joint(
        n_samples, sample_covariate_q, conditional_z, seed=seed + 1
    )
    
    # Points to evaluate CME difference
    x_eval = np.linspace(0, 1, 200)

    # True conditional mean difference
    true_diff = 0.5 * x_eval**2
    
    # Set bandwidth 
    bandwidth = 100
    
    # Regularization parameter
    lam_p = 1e-3
    lam_q = 1e-3
    
    # Compute CME difference
    cme_Y = cme_model(x_eval, X_p, Y, lam_p, gaussian_kernel, bandwidth=bandwidth)
    cme_Z = cme_model(x_eval, X_q, Z, lam_q, gaussian_kernel, bandwidth=bandwidth)
    standard_cme_diff = cme_Y - cme_Z

    # split data into train and test sets for DR estimation
    n_test = n_samples // 2
    X_p_train, Y_train = X_p[:n_test], Y[:n_test]
    X_q_train, Z_train = X_q[:n_test], Z[:n_test]
    X_p_test, Y_test = X_p[n_test:], Y[n_test:]
    X_q_test, Z_test = X_q[n_test:], Z[n_test:]

    # Merge test set 
    X_test = np.concatenate([X_p_test, X_q_test])
    YZ_test = np.concatenate([Y_test, Z_test])
    
    # T indicates which samples are from P (T=1) vs Q (T=0)
    T = np.concatenate([np.ones_like(Y_test), np.zeros_like(Z_test)])

    # Compute Propensity scores on test set
    E = propensity(X_test)

    # CME models for Y and Z
    cme_Y_train = cme_model(X_test, X_p_train, Y_train, lam_p, gaussian_kernel, bandwidth=bandwidth)
    cme_Z_train = cme_model(X_test, X_q_train, Z_train, lam_q, gaussian_kernel, bandwidth=bandwidth)

    # RKHS difference psuedo-outcome for doubly robust estimation
    psuedo_outcome = (T - E) / (E * (1 - E)) * (YZ_test - (1 - E) * cme_Y_train - E * cme_Z_train)

    # Compute doubly robust CME difference
    bandwidth_dr = 1
    dr_cme_diff = cme_model(x_eval, X_test, psuedo_outcome, lam_p, gaussian_kernel, bandwidth=bandwidth_dr)
    
    # Plot results
    plot_cme_difference(x_eval, true_diff, standard_cme_diff, dr_cme_diff)


if __name__ == "__main__":
    main()
