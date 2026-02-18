"""
Generate and plot CMMD test statistics from conditional distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path (robust to different working directories)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.cmmd import CMMD0, CMMD1, CMMD2
from src.kernels import gaussian_kernel, median_heuristic
from src.models import sample_joint, sample_joint_theta, sample_covariate, sample_covariate_p, sample_covariate_q, conditional_y, conditional_z


def run_cmmd_experiment(
    n_trials: int = 5,
    n_samples: int = 100,
    lam_p: float = 0.01,
    lam_q: float = 0.01,
    bandwidth: float = 0.1,
    noise_std: float = 0.3,
    cmmd2_estimator: str = "jmmd",
    setting: str = "diff_marginal"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run multiple trials of CMMD0, CMMD1, and CMMD2 test statistic computation.
    
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
    noise_std : float, default=0.3
        Standard deviation of noise in data generation.
    cmmd2_estimator : str, default="cmmd"
        CMMD2 estimator to use ("cmmd" or "jmmd").
    setting : str, default="diff_marginal"
        Setting for marginal distributions ("same_marginal" or "diff_marginal").
    
    Returns
    -------
    cmmd0_stats : np.ndarray, shape (n_trials,)
        CMMD0 test statistics for each trial.
    cmmd1_stats : np.ndarray, shape (n_trials,)
        CMMD1 test statistics for each trial.
    cmmd2_stats : np.ndarray, shape (n_trials,)
        CMMD2 test statistics for each trial.
    """
    cmmd0_stat = CMMD0()
    cmmd1_stat = CMMD1()
    cmmd2_stat = CMMD2()

    cmmd0_stats = np.zeros(n_trials)
    cmmd1_stats = np.zeros(n_trials)
    cmmd2_stats = np.zeros(n_trials)
    
    # Estimate bandwidth using median heuristic on first data sample
    # X_pilot, _ = sample_joint(n_samples, conditional_y, noise_std=noise_std, seed=42)
    # bandwidth = median_heuristic(X_pilot)
    
    print(f"Running {n_trials} trials with:")
    print(f"  - n_samples per trial: {n_samples}")
    print(f"  - lam_p: {lam_p}, lam_q: {lam_q}")
    print(f"  - bandwidth: {bandwidth:.4f}")
    print()
    
    # Run trials
    for trial in range(n_trials):
        # Generate data from two conditional distributions
        # Select marginal distributions based on setting
        if setting == 'same_marginal':
            marginal_p = sample_covariate
            marginal_q = sample_covariate
            cmmd2_estimator = "jmmd"
        elif setting == 'diff_marginal':
            marginal_p = sample_covariate_p
            marginal_q = sample_covariate_q
            cmmd2_estimator = "cmmd"
        elif setting == 'same_marginal_theta':
            theta = 0.0
            cmmd2_estimator = "jmmd"
        else:
            raise ValueError(f"Unknown setting: {setting}. Must be 'same_marginal' or 'diff_marginal'.")
        

        if setting == 'same_marginal_theta':
            X_P, Y = sample_joint_theta(n_samples, theta, conditional_y, noise_std=noise_std, seed=trial*2)
            X_Q, Z = sample_joint_theta(n_samples, theta, conditional_z, noise_std=noise_std, seed=trial*2 + 1)
        else:
            X_P, Y = sample_joint(n_samples, marginal_p, conditional_y, noise_std=noise_std, seed=trial*2)
            X_Q, Z = sample_joint(n_samples, marginal_q, conditional_z, noise_std=noise_std, seed=trial*2 + 1)
        
        # Ensure data is 2D for kernel computation
        X_P = X_P.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        X_Q = X_Q.reshape(-1, 1)
        Z = Z.reshape(-1, 1)
        
        # Compute CMMD0, CMMD1, and CMMD2 test statistics
        stat0 = cmmd0_stat.compute(
            X_P, Y, X_Q, Z,
            lam_p, lam_q,
            gaussian_kernel,
            bandwidth=bandwidth
        )
        stat1 = cmmd1_stat.compute(
            X_P, Y, X_Q, Z,
            lam_p, lam_q,
            gaussian_kernel,
            bandwidth=bandwidth
        )
        stat2 = cmmd2_stat.compute(
            X_P, Y, X_Q, Z,
            lam_p, lam_q,
            gaussian_kernel,
            estimator=cmmd2_estimator,
            bandwidth=bandwidth
        )

        cmmd0_stats[trial] = stat0
        cmmd1_stats[trial] = stat1
        cmmd2_stats[trial] = stat2

        # print every 100th trial for progress
        if (trial + 1) % 100 == 0:
            print(
                f"Trial {trial+1:d}: CMMD0 = {stat0:.6f}, "
                f"CMMD1 = {stat1:.6f}, CMMD2 = {stat2:.6f}"
            )

    return cmmd0_stats, cmmd1_stats, cmmd2_stats


def plot_test_statistics(
    cmmd0_stats: np.ndarray,
    cmmd1_stats: np.ndarray,
    cmmd2_stats: np.ndarray,
    setting: str = "same_marginal_theta"
):
    """
    Plot histograms of CMMD0, CMMD1, and CMMD2 test statistics.
    
    Parameters
    ----------
    cmmd0_stats : np.ndarray
        CMMD0 test statistics from multiple trials.
    cmmd1_stats : np.ndarray
        CMMD1 test statistics from multiple trials.
    cmmd2_stats : np.ndarray
        CMMD2 test statistics from multiple trials.
    setting : str, default="same_marginal_theta"
        Setting for plot ("same_marginal", "diff_marginal", or "same_marginal_theta").
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create overlaid histograms
    ax.hist(
        cmmd0_stats,
        bins='auto',
        edgecolor='black',
        alpha=0.6,
        color='steelblue',
        label='CMMD$_0$'
    )
    ax.hist(
        cmmd1_stats,
        bins='auto',
        edgecolor='black',
        alpha=0.6,
        color='darkorange',
        label='CMMD$_1$'
    )
    ax.hist(
        cmmd2_stats,
        bins='auto',
        edgecolor='black',
        alpha=0.6,
        color='seagreen',
        label='CMMD$_2$'
    )
    
    # Add labels and title
    ax.set_xlabel(r'$\widehat{CMM}D^2$', fontsize=20)
    ax.set_ylabel('Frequency', fontsize=20)
    ax.set_title('Distribution of Test Statistic',
                 fontsize=24)
    ax.set_xscale('log')
    ax.tick_params(axis="both", which="major", labelsize=14)
    if setting == "same_marginal_theta":
        ax.legend(fontsize=16)
    else:
        ax.legend(fontsize=16, loc=(0.6, 0.75))
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig, ax


if __name__ == '__main__':
    # Setting
    setting="same_marginal_theta"
    # setting="same_marginal"
    # setting="diff_marginal"

    if setting == "same_marginal_theta":
        n_samples = 100
    else:
        n_samples = 250

    # Run experiment
    cmmd0_stats, cmmd1_stats, cmmd2_stats = run_cmmd_experiment(
        n_trials=250,
        n_samples=n_samples,
        lam_p=0.1,
        lam_q=0.1,
        bandwidth=0.1,
        noise_std=0.5,
        setting=setting
    )
    
    # Plot results
    fig, ax = plot_test_statistics(cmmd0_stats, cmmd1_stats, cmmd2_stats, setting)
    
    # Save figure (use absolute path)
    figs_dir = os.path.join(script_dir, '..', 'figs/synthetic')
    os.makedirs(figs_dir, exist_ok=True)
    fig_path = os.path.join(figs_dir, f'cmmd_test_statistics_{setting}.pdf')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")
    
    # Display plot
    plt.show()
