"""
Compare standard and doubly robust CMMD_1 test statistics from conditional distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import sys
import os

# Add project root to path (robust to different working directories)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.cmmd import CMMD1, CMMD2
from src.kernels import gaussian_kernel, median_heuristic
from src.dr_models import sample_joint, sample_covariate_p, sample_covariate_q, conditional_y, conditional_z, propensity

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


def run_dr_cmmd_experiment(
    n_trials: int = 5,
    n_samples: int = 100,
    lam_p: float = 1e-3,
    lam_q: float = 1e-3,
    bandwidth_model: float = 100,
    bandwidth_dr: float = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run multiple trials of standard and doubly robust CMMD1 test statistic computation.
    
    Parameters
    ----------
    n_trials : int, default=5
        Number of independent trials.
    n_samples : int, default=100
        Number of samples per trial.
    lam_p : float, default=0.01
        Regularization parameter for distribution P.
    lam_q : float, default=0.01
        Regularization parameter for distribution Q.
    bandwidth : float, default=0.1
        Bandwidth parameter for Gaussian kernel.
    
    Returns
    -------
    cmmd1_standard_stats : np.ndarray, shape (n_trials,)
        Standard CMMD1 test statistics for each trial.
    cmmd1_dr_stats : np.ndarray, shape (n_trials,)
        Doubly robust CMMD1 test statistics for each trial.
    """
    cmmd1_standard_stat = CMMD1()

    cmmd1_standard_stats = np.zeros(n_trials)
    cmmd1_dr_stats = np.zeros(n_trials)
    
    print(f"Running {n_trials} trials with:")
    print(f"  - n_samples per trial: {n_samples}")
    print(f"  - lam_p: {lam_p}, lam_q: {lam_q}")
    print(f"  - bandwidth (model): {bandwidth_model:.4f}")
    print(f"  - bandwidth (DR): {bandwidth_dr:.4f}")
    print()
    
    # Run trials
    for trial in range(n_trials):
        # Generate data from two conditional distributions
        
        # P: Y|X
        X_P, Y = sample_joint(n_samples, sample_covariate_p, conditional_y, seed=trial*2)
        
        # Q: Z|X
        X_Q, Z = sample_joint(n_samples, sample_covariate_q, conditional_z, seed=trial*2 + 1)
        
        # Ensure data is 2D for kernel computation
        X_p = X_P.reshape(-1, 1)
        Y_p = Y.reshape(-1, 1)
        X_q = X_Q.reshape(-1, 1)
        Z_q = Z.reshape(-1, 1)
        
        # Compute standard and doubly robust CMMD1 test statistics
        stat_standard = cmmd1_standard_stat.compute(
            X_p, Y_p, X_q, Z_q,
            lam_p, lam_q,
            gaussian_kernel,
            bandwidth=bandwidth_model
        )

        # split data into train and test sets for DR estimation
        n_test = n_samples // 2
        X_p_train, Y_train = X_P[:n_test], Y[:n_test]
        X_q_train, Z_train = X_Q[:n_test], Z[:n_test]
        X_p_test, Y_test = X_P[n_test:], Y[n_test:]
        X_q_test, Z_test = X_Q[n_test:], Z[n_test:]

        # Merge test set 
        X_test = np.concatenate([X_p_test, X_q_test])
        YZ_test = np.concatenate([Y_test, Z_test])

        # T indicates which samples are from P (T=1) vs Q (T=0)
        T = np.concatenate([np.ones_like(Y_test), np.zeros_like(Z_test)])

        # Compute Propensity scores on test set
        E = propensity(X_test)

        # CME models for Y and Z
        cme_Y = cme_model(X_test, X_p_train, Y_train, lam_p, gaussian_kernel, bandwidth=bandwidth_model)
        cme_Z = cme_model(X_test, X_q_train, Z_train, lam_q, gaussian_kernel, bandwidth=bandwidth_model)

        # RKHS difference psuedo-outcome for doubly robust estimation
        psuedo_outcome = (T - E) / (E * (1 - E)) * (YZ_test - (1 - E) * cme_Y - E * cme_Z)

        # Compute doubly robust CME difference
        dr_cme_diff = cme_model(X_test, X_test, psuedo_outcome, lam_p, gaussian_kernel, bandwidth=bandwidth_dr)

        stat_dr = sum(dr_cme_diff**2) / len(X_test)

        cmmd1_standard_stats[trial] = stat_standard
        cmmd1_dr_stats[trial] = stat_dr

        # print every 100th trial for progress
        if (trial + 1) % 100 == 0:
            print(
                f"Trial {trial+1:d}: CMMD1 (standard) = {stat_standard:.6f}, "
                f"CMMD1 (doubly robust) = {stat_dr:.6f}"
            )

    return cmmd1_standard_stats, cmmd1_dr_stats

def true_cmmd(n_samples: int = 1000) -> float:
    """
    Compute the true CMMD1^2 between P and Q using numerical integration.
    
    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples to use for numerical approximation.
    """
    # Sample covariate from combined distribution for numerical approximation
    X_P, _ = sample_joint(n_samples, sample_covariate_p, conditional_y, seed=1)
    X_Q, _ = sample_joint(n_samples, sample_covariate_q, conditional_z, seed=2) 

    X_test = np.concatenate([X_P, X_Q])

    cmmd1 = sum((0.5 * X_test**2)**2) / len(X_test)
    return cmmd1

def plot_test_statistics(
    cmmd1_standard_stats: np.ndarray,
    cmmd1_dr_stats: np.ndarray,
    cmmd1_true: float
):
    """
    Plot histograms of CMMD1 test statistics for standard and doubly robust versions.
    
    Parameters
    ----------
    cmmd1_standard_stats : np.ndarray
        CMMD1 test statistics from multiple trials for standard version.
    cmmd1_dr_stats : np.ndarray
        CMMD1 test statistics from multiple trials for doubly robust version.
    cmmd1_true : float
        True CMMD1^2 value for reference.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Add vertical line for true CMMD1^2
    ax.axvline(cmmd1_true, color='black', linestyle='--', linewidth=2, label='True Value')
    
    # Create overlaid histograms
    ax.hist(
        cmmd1_standard_stats,
        bins='auto',
        edgecolor='black',
        alpha=0.6,
        color='blue',
        label='Standard'
    )
    ax.hist(
        cmmd1_dr_stats,
        bins='auto',
        edgecolor='black',
        alpha=0.6,
        color='red',
        label='Doubly Robust'
    )
    
    # Add labels and title
    ax.set_xlabel(r'$\widehat{CMM}D_1^2$', fontsize=20)
    ax.set_ylabel('Frequency', fontsize=20)
    ax.set_title('Distribution of Test Statistic', fontsize=24)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=16)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig, ax


if __name__ == '__main__':

    # Run experiment
    cmmd1_standard_stats, cmmd1_dr_stats = run_dr_cmmd_experiment(
        n_trials=250,
        n_samples=1000,
        lam_p=1e-3,
        lam_q=1e-3,
        bandwidth_model=100,
        bandwidth_dr=1
    )

    # Compute true CMMD1^2 for reference using numerical approximation
    cmmd1_true = true_cmmd(n_samples=1000)
    
    # Plot results
    fig, ax = plot_test_statistics(cmmd1_standard_stats, cmmd1_dr_stats, cmmd1_true)
    
    # Save figure (use absolute path)
    figs_dir = os.path.join(script_dir, '..', 'figs/dr')
    os.makedirs(figs_dir, exist_ok=True)
    fig_path = os.path.join(figs_dir, f'cmmd1_test_statistics.pdf')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")
    
    # Display plot
    plt.show()
