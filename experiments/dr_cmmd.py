"""
Compare standard and doubly robust CMMD_1 test statistics from conditional distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import sys
import os
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

# Add project root to path (robust to different working directories)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.cmmd import CMMD1, CMMD1_dr
from src.kernels import polynomial_kernel, linear_kernel
from src.dr_models import sample_joint, sample_covariate_p, sample_covariate_q, conditional_y, conditional_z, propensity


def compute_cmmd1_estimates(
	n_samples: int,
	seed: int
) -> Tuple[float, float]:
	"""
	Compute standard and doubly robust CMMD1^2 estimates for one trial.
	"""

	cmmd1_standard_stat = CMMD1()
	cmmd1_dr_stat = CMMD1_dr()

	X_P, Y = sample_joint(n_samples, sample_covariate_p, conditional_y, seed=seed)
	X_Q, Z = sample_joint(n_samples, sample_covariate_q, conditional_z, seed=seed + 1)

	# Ensure data is 2D for kernel computation
	X_P = X_P.reshape(-1, 1)
	Y = Y.reshape(-1, 1)
	X_Q = X_Q.reshape(-1, 1)
	Z = Z.reshape(-1, 1)
	
    # Fit KRR models for DR estimation
	alpha_grid = {"alpha": np.logspace(-4, 1, 5)}
	krr_y = GridSearchCV(
		KernelRidge(kernel='polynomial', degree=2, coef0=1, gamma=1),
		alpha_grid, cv=5)
	krr_y.fit(X_P, Y.flatten())

	krr_z = GridSearchCV(KernelRidge(kernel='polynomial', degree=2, coef0=1, gamma=1), alpha_grid, cv=5)
	krr_z.fit(X_Q, Z.flatten())

	alpha_p = float(krr_y.best_params_["alpha"])
	alpha_q = float(krr_z.best_params_["alpha"])
	lam_p = alpha_p / X_P.shape[0]
	lam_q = alpha_q / X_Q.shape[0]

	stat_standard = cmmd1_standard_stat.compute(
		X_P, Y, X_Q, Z,
		lam_p, lam_q,
		polynomial_kernel,
		linear_kernel
	)

	# Build pseudo-outcomes on pooled training data, then tune KRR alpha for DR lambda
	X_train = np.concatenate([X_P, X_Q], axis=0)
	YZ_train = np.concatenate([Y.flatten(), Z.flatten()])
	T_train = np.concatenate([
		np.ones(X_P.shape[0], dtype=float),
		np.zeros(X_Q.shape[0], dtype=float),
	])
	E_train = propensity(X_train.flatten())

	mu_y_train = krr_y.predict(X_train)
	mu_z_train = krr_z.predict(X_train)
	pseudo_outcome_train = (T_train - E_train) / (E_train * (1 - E_train)) * (
		YZ_train - (1 - E_train) * mu_y_train - E_train * mu_z_train
	)

	krr_dr = GridSearchCV(
		KernelRidge(kernel='polynomial', degree=2, coef0=1, gamma=1),
		alpha_grid, cv=5,
	)
	krr_dr.fit(X_train, pseudo_outcome_train)
	alpha_dr = float(krr_dr.best_params_["alpha"])
	lam_dr = alpha_dr / X_train.shape[0]

	# Create callable model functions
	def cme_y(X):
		return krr_y.predict(X.reshape(-1, 1) if X.ndim == 1 else X)

	def cme_z(X):
		return krr_z.predict(X.reshape(-1, 1) if X.ndim == 1 else X)
	
	stat_kwargs={"propensity": propensity, "cme_y": cme_y, "cme_z": cme_z}

	stat_dr = cmmd1_dr_stat.compute(
		X_P, Y, X_Q, Z,
		lam_dr, lam_dr,
		polynomial_kernel,
		linear_kernel,
		**stat_kwargs,
	)

	return stat_standard, stat_dr

def run_dr_cmmd_experiment(
    n_trials: int = 5,
    n_samples: int = 100,
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

    cmmd1_standard_stats = np.zeros(n_trials)
    cmmd1_dr_stats = np.zeros(n_trials)
    
    print(f"Running {n_trials} trials with:")
    print(f"  - n_samples per trial: {n_samples}")
    print()
    
    # Run trials
    for trial in range(n_trials):

        seed = trial * 5 + n_samples

        stat_standard, stat_dr = compute_cmmd1_estimates(
            n_samples=n_samples,
            seed=seed,
        )

        cmmd1_standard_stats[trial] = stat_standard
        cmmd1_dr_stats[trial] = stat_dr

        # print every 50th trial for progress
        if (trial + 1) % 50 == 0:
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
        n_samples=500,
    )

    # Compute true CMMD1^2 for reference using numerical approximation
    cmmd1_true = true_cmmd(n_samples=1000)
    
    # Plot results
    fig, ax = plot_test_statistics(cmmd1_standard_stats, cmmd1_dr_stats, cmmd1_true)
    
    # Save figure (use absolute path)
    # figs_dir = os.path.join(script_dir, '..', 'figs/dr')
    # os.makedirs(figs_dir, exist_ok=True)
    # fig_path = os.path.join(figs_dir, f'cmmd1_test_statistics.pdf')
    # fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    # print(f"\nFigure saved to: {fig_path}")
    
    # Display plot
    plt.show()
