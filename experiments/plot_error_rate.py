"""
Plot power against sample size for CMMD-based tests.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path (robust to different working directories)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.cmmd import CMMD0, CMMD1, CMMD2, Test_Same_Marginal, Test_Diff_Marginal
from src.kernels import gaussian_kernel, median_heuristic
from src.models import sample_joint, sample_covariate, sample_covariate_p, sample_covariate_q, conditional_y, conditional_z, propensity


def run_power_experiment(
	error: str,
	sample_sizes: list[int],
	n_trials: int = 100,
	alpha: float = 0.05,
	B: int = 200,
	lam_p: float = 0.1,
	lam_q: float = 0.1,
	bandwidth: float | None = 0.1,
	noise_std: float = 0.3,
	cmmd2_estimator: str = "jmmd",
	setting: str = "diff_marginal",
	seed: int = 42
) -> dict[str, np.ndarray]:
	"""
	Estimate power (rejection rate) across sample sizes for CMMD tests.
	"""
	cmmd0_stat = CMMD0()
	cmmd1_stat = CMMD1()
	cmmd2_stat = CMMD2()

	if setting == "same_marginal":
		algo = Test_Same_Marginal()
	if setting == "diff_marginal":
		algo = Test_Diff_Marginal()

	rng = np.random.default_rng(seed)

	results_cmmd0 = np.zeros(len(sample_sizes))
	results_cmmd1 = np.zeros(len(sample_sizes))
	results_cmmd2 = np.zeros(len(sample_sizes))

	for i, n in enumerate(sample_sizes):
		# print current sample size
		print(f"Running power experiment for sample size: {n}")
		
		reject0 = 0
		reject1 = 0
		reject2 = 0

		for trial in range(n_trials):
			seed_y = int(rng.integers(1, 2**31))
			seed_z = int(rng.integers(1, 2**31))
			seed_perm = int(rng.integers(1, 2**31))

			# Select marginal distributions based on setting
			if setting == 'same_marginal':
				marginal_p = sample_covariate
				marginal_q = sample_covariate
				cmmd2_estimator = "jmmd"
			elif setting == 'diff_marginal':
				marginal_p = sample_covariate_p
				marginal_q = sample_covariate_q
				cmmd2_estimator = "cmmd"
			else:
				raise ValueError(f"Unknown setting: {setting}. Must be 'same_marginal' or 'diff_marginal'.")

			X_P, Y = sample_joint(n, marginal_p, conditional_y, noise_std=noise_std, seed=seed_y)
			if error == "type1":
				X_Q, Z = sample_joint(n, marginal_q, conditional_y, noise_std=noise_std, seed=seed_z)
			if error == "type2":
				X_Q, Z = sample_joint(n, marginal_q, conditional_z, noise_std=noise_std, seed=seed_z)

			X_P = X_P.reshape(-1, 1)
			Y = Y.reshape(-1, 1)
			X_Q = X_Q.reshape(-1, 1)
			Z = Z.reshape(-1, 1)

			# Prepare kwargs for test method
			algo_kwargs = {
				"alpha": alpha,
				"B": B,
				"lam_p": lam_p,
				"lam_q": lam_q,
				"random_state": seed_perm,
			}
			stat_kwargs = {
				"bandwidth": bandwidth,
			}
			# Add propensity function if using different marginals
			if setting == "diff_marginal":
				algo_kwargs["propensity_fn"] = propensity

			_, p0 = algo.test(
				X_P, Y, X_Q, Z,
				cmmd0_stat,
				gaussian_kernel,
				algo_kwargs=algo_kwargs,
				stat_kwargs=stat_kwargs,
			)
			_, p1 = algo.test(
				X_P, Y, X_Q, Z,
				cmmd1_stat,
				gaussian_kernel,
				algo_kwargs={**algo_kwargs, "random_state": seed_perm + 1},
				stat_kwargs=stat_kwargs,
			)
			_, p2 = algo.test(
				X_P, Y, X_Q, Z,
				cmmd2_stat,
				gaussian_kernel,
				algo_kwargs={**algo_kwargs, "random_state": seed_perm + 2},
				stat_kwargs={**stat_kwargs, "estimator": cmmd2_estimator},
			)

			reject0 += int(p0 < alpha)
			reject1 += int(p1 < alpha)
			reject2 += int(p2 < alpha)

			# print every 50th trial for progress
			if (trial + 1) % 50 == 0:
				print(
					f"Trial {trial+1:d}/{n_trials:d} for n={n:d}: "
					f"CMMD0 p-value = {p0:.4f}, CMMD1 p-value = {p1:.4f}, CMMD2 p-value = {p2:.4f}"
				)

		results_cmmd0[i] = reject0 / n_trials
		results_cmmd1[i] = reject1 / n_trials
		results_cmmd2[i] = reject2 / n_trials

	return {
		"cmmd0": results_cmmd0,
		"cmmd1": results_cmmd1,
		"cmmd2": results_cmmd2,
	}


def plot_power_vs_sample_size(
	sample_sizes: list[int],
	results: dict[str, np.ndarray],
	error: str = "type1",
	setting: str = "same_marginal"
):
	"""
	Plot power/type I error vs sample size for CMMD0, CMMD1, and CMMD2.
	"""
	fig, ax = plt.subplots(figsize=(8, 6))
	
	# if error is type1, plot significance level line at alpha=0.05
	if error == "type1":
		ax.axhline(0.05, color="black", linestyle="--")

	ax.plot(sample_sizes, results["cmmd0"], marker="o", label="CMMD$_0$")
	ax.plot(sample_sizes, results["cmmd1"], marker="s", label="CMMD$_1$")
	ax.plot(sample_sizes, results["cmmd2"], marker="^", label="CMMD$_2$")

	ax.set_xlabel("Sample size ($n$)", fontsize=20)
	if error == "type1":
		if setting == "same_marginal":
			ax.set_title("$P_X = Q_X$", fontsize=20)
		if setting == "diff_marginal":
			ax.set_title("$P_X \\neq Q_X$", fontsize=20)
		ax.set_ylabel("Type I Error", fontsize=20)
		ax.set_ylim(0.0, 1.0)
		ax.legend(fontsize=16)
	if error == "type2":
		ax.set_ylabel("Power", fontsize=20)
		ax.set_ylim(0.0, 1.1)
	ax.tick_params(axis="both", which="major", labelsize=14)
	ax.grid(True, alpha=0.3)

	plt.tight_layout()

	return fig, ax


if __name__ == "__main__":
	sample_sizes = [10, 50, 100, 150, 200, 250]
	n_trials = 250
	# error = "type1"
	error = "type2"
	# setting = "same_marginal"
	setting = "diff_marginal"

	results = run_power_experiment(
		error=error,
		sample_sizes=sample_sizes,
		n_trials=n_trials,
		alpha=0.05,
		B=250,
		lam_p=0.1,
		lam_q=0.1,
		bandwidth=0.1,
		noise_std=0.5,
		setting=setting,
		seed=42,
	)

	fig, _ = plot_power_vs_sample_size(sample_sizes, results, error=error, setting=setting)

	figs_dir = os.path.join(project_root, "figs")
	os.makedirs(figs_dir, exist_ok=True)
	fig_path = os.path.join(figs_dir, f"synthetic_{error}_vs_sample_size_{setting}.pdf")
	fig.savefig(fig_path, dpi=300, bbox_inches="tight")
	print(f"Figure saved to: {fig_path}")

	plt.show()
