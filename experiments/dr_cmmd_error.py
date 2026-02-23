"""
Plot squared error of CMMD1 estimators for standard vs doubly robust methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple
import sys
import os

# Add project root to path (robust to different working directories)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.cmmd import CMMD1
from src.kernels import gaussian_kernel
from src.dr_models import (
	sample_joint,
	sample_covariate_p,
	sample_covariate_q,
	conditional_y,
	conditional_z,
	propensity,
)


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
	"""
	K = kernel_x(X, X, **kwargs)
	W_X = np.linalg.inv(K + lam * X.shape[0] * np.eye(X.shape[0]))
	K_Xx = kernel_x(X, x_test, **kwargs)

	Y_flat = Y.flatten()
	mu_Y = Y_flat @ W_X @ K_Xx

	return mu_Y


def true_cmmd(n_samples: int = 2000) -> float:
	"""
	Compute the true CMMD1^2 between P and Q using numerical integration.
	"""
	X_P, _ = sample_joint(n_samples, sample_covariate_p, conditional_y, seed=1)
	X_Q, _ = sample_joint(n_samples, sample_covariate_q, conditional_z, seed=2)

	X_test = np.concatenate([X_P, X_Q])
	cmmd1 = np.mean((0.5 * X_test**2) ** 2)
	return cmmd1


def compute_cmmd1_estimates(
	n_samples: int,
	bandwidth: float,
	seed: int
) -> Tuple[float, float]:
	"""
	Compute standard and doubly robust CMMD1^2 estimates for one trial.
	"""
	cmmd1_standard_stat = CMMD1()

	X_P, Y = sample_joint(n_samples, sample_covariate_p, conditional_y, seed=seed)
	X_Q, Z = sample_joint(n_samples, sample_covariate_q, conditional_z, seed=seed + 1)
	
	lam_p = 0.01 * n_samples **(-0.25)
	lam_q = 0.01 * n_samples **(-0.25)

	stat_standard = cmmd1_standard_stat.compute(
		X_P.reshape(-1, 1), Y.reshape(-1, 1), 
		X_Q.reshape(-1, 1), Z.reshape(-1, 1),
		lam_p, lam_q,
		gaussian_kernel,
		bandwidth=bandwidth,
	)

	X_test = np.concatenate([X_P, X_Q])
	YZ_test = np.concatenate([Y, Z])
	T = np.concatenate([np.ones_like(Y), np.zeros_like(Z)])

	E = propensity(X_test)

	cme_Y = cme_model(
		X_test, X_P, Y, lam_p, gaussian_kernel, bandwidth=bandwidth
	)
	cme_Z = cme_model(
		X_test, X_Q, Z, lam_q, gaussian_kernel, bandwidth=bandwidth
	)

	psuedo_outcome = (T - E) / (E * (1 - E)) * (YZ_test - (1 - E) * cme_Y - E * cme_Z)

	dr_cme_diff = cme_model(
		X_test, X_test, psuedo_outcome, lam_p, gaussian_kernel, bandwidth=bandwidth
	)
	stat_dr = np.mean(dr_cme_diff**2)

	return stat_standard, stat_dr


def run_error_experiment(
	sample_sizes: np.ndarray,
	n_trials: int,
	bandwidth: float,
	true_samples: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""
	Run squared error experiments across sample sizes.
	"""
	cmmd1_true = true_cmmd(n_samples=true_samples)

	standard_errors = np.zeros_like(sample_sizes, dtype=float)
	dr_errors = np.zeros_like(sample_sizes, dtype=float)
	standard_se = np.zeros_like(sample_sizes, dtype=float)
	dr_se = np.zeros_like(sample_sizes, dtype=float)

	print("Running CMMD1 squared-error experiment")
	print(f"  - trials per size: {n_trials}")
	print(f"  - bandwidth: {bandwidth:.4f}")
	print()

	for i, n_samples in enumerate(sample_sizes):
		standard_trials = np.zeros(n_trials)
		dr_trials = np.zeros(n_trials)

		for trial in range(n_trials):
			stat_standard, stat_dr = compute_cmmd1_estimates(
				n_samples=n_samples,
				bandwidth=bandwidth,
				seed=trial * 5 + n_samples,
			)

			standard_trials[trial] = (stat_standard - cmmd1_true) ** 2
			dr_trials[trial] = (stat_dr - cmmd1_true) ** 2

		standard_errors[i] = np.mean(standard_trials)
		dr_errors[i] = np.mean(dr_trials)
		standard_se[i] = np.std(standard_trials, ddof=1) / np.sqrt(n_trials)
		dr_se[i] = np.std(dr_trials, ddof=1) / np.sqrt(n_trials)

		print(
			f"n={n_samples:4d}: SE std={standard_errors[i]:.6f}, "
			f"SE dr={dr_errors[i]:.6f}"
		)

	return standard_errors, dr_errors, standard_se, dr_se


def plot_error_curves(
	sample_sizes: np.ndarray,
	standard_errors: np.ndarray,
	dr_errors: np.ndarray,
	standard_se: np.ndarray,
	dr_se: np.ndarray,
	bandwidth: float
):
	"""
	Plot squared error vs number of samples for both estimators.
	"""
	fig, ax = plt.subplots(figsize=(8, 6))

	ax.errorbar(
		sample_sizes,
		standard_errors,
		yerr=standard_se,
		marker="o",
		linestyle="-",
		color="blue",
		label="Standard",
		linewidth=2,
	)
	ax.errorbar(
		sample_sizes,
		dr_errors,
		yerr=dr_se,
		marker="o",
		linestyle="-",
		color="red",
		label="Doubly Robust",
		linewidth=2,
	)

	ax.set_xlabel("Sample Size ($n$)", fontsize=30)
	ax.set_ylabel("CMMD$_1$ error", fontsize=30)
	ax.set_title(f"Bandwidth h={bandwidth:.1f}", fontsize=36)
	ax.tick_params(axis="both", which="major", labelsize=21)
	ax.grid(True, alpha=0.3)
	ax.legend(fontsize=24)

	plt.tight_layout()

	return fig, ax


if __name__ == "__main__":
	sample_sizes = np.array([30, 50, 100, 200, 300, 400, 500])
	bandwidth = 0.1
	n_trials = 100

	standard_errors, dr_errors, standard_se, dr_se = run_error_experiment(
		sample_sizes=sample_sizes,
		n_trials=n_trials,
		bandwidth=bandwidth
	)

	fig, ax = plot_error_curves(
		sample_sizes, standard_errors, dr_errors, standard_se, dr_se, bandwidth=bandwidth
	)

	# Save figure (optional)
	figs_dir = os.path.join(script_dir, "..", "figs", "dr")
	os.makedirs(figs_dir, exist_ok=True)
	fig_path = os.path.join(figs_dir, f"cmmd1_squared_error_h{bandwidth}.pdf")
	fig.savefig(fig_path, dpi=300, bbox_inches="tight")
	print(f"\nFigure saved to: {fig_path}")

	plt.show()
