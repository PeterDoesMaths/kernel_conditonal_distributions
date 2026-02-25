"""
Plot squared error of CMMD1 estimators for standard vs doubly robust methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple
import sys
import os
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

# Add project root to path (robust to different working directories)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.cmmd import CMMD1
from src.kernels import gaussian_kernel, indicator_kernel
from src.dr_models import (
	sample_joint,
	sample_covariate_p,
	sample_covariate_q,
	conditional_y,
	conditional_z,
	propensity,
)


def fit_krr_predict(
	x_train: np.ndarray,
	y_train: np.ndarray,
	x_test: np.ndarray,
	bandwidth: float,
	label: str,
	alpha_grid: np.ndarray | None = None,
	cv: int = 5,
) -> np.ndarray:
	"""
	Fit KernelRidge with CV over alpha and predict on x_test.
	"""
	if alpha_grid is None:
		alpha_grid = np.logspace(-4, 1, 12)

	x_train_2d = x_train.reshape(-1, 1) if x_train.ndim == 1 else x_train
	x_test_2d = x_test.reshape(-1, 1) if x_test.ndim == 1 else x_test
	y_train_1d = y_train.ravel()

	gamma = 0.5 * bandwidth
	krr = KernelRidge(kernel="rbf", gamma=gamma)

	search = GridSearchCV(
		estimator=krr,
		param_grid={"alpha": alpha_grid},
		cv=min(cv, x_train_2d.shape[0]),
		scoring="neg_mean_squared_error",
	)
	search.fit(x_train_2d, y_train_1d)

	best_model = search.best_estimator_
	return best_model.predict(x_test_2d)


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

	X_P, Y = sample_joint(n_samples, sample_covariate_p, conditional_y, seed=seed)
	X_Q, Z = sample_joint(n_samples, sample_covariate_q, conditional_z, seed=seed + 1)

	X_test = np.concatenate([X_P, X_Q])
	YZ_test = np.concatenate([Y, Z])
	T = np.concatenate([np.ones_like(Y), np.zeros_like(Z)])

	E = propensity(X_test)

	cme_Y = fit_krr_predict(
		x_train=X_P,
		y_train=Y,
		x_test=X_test,
		bandwidth=bandwidth,
		label="E[Y|X]",
	)
	cme_Z = fit_krr_predict(
		x_train=X_Q,
		y_train=Z,
		x_test=X_test,
		bandwidth=bandwidth,
		label="E[Z|X]",
	)

	stat_standard = np.mean((cme_Y - cme_Z) ** 2)

	pseudo_outcome = (T - E) / (E * (1 - E)) * (YZ_test - (1 - E) * cme_Y - E * cme_Z)

	dr_cme_diff = fit_krr_predict(
		x_train=X_test,
		y_train=pseudo_outcome,
		x_test=X_test,
		bandwidth=bandwidth,
		label="DR correction",
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
	sample_sizes = np.array([100, 200, 300, 400, 500])
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
