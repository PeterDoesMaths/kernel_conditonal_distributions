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
from src.models import sample_joint_theta, conditional_y, conditional_z, propensity


def run_power_experiment(
	thetas: list[float],
	sample_size: int,
	n_trials: int = 100,
	alpha: float = 0.05,
	B: int = 200,
	lam_p: float = 0.1,
	lam_q: float = 0.1,
	bandwidth: float | None = 0.1,
	noise_std: float = 0.3,
	cmmd2_estimator: str = "jmmd",
	seed: int = 42
) -> dict[str, np.ndarray]:
	"""
	Estimate power (rejection rate) across sample sizes for CMMD tests.
	"""
	cmmd0_stat = CMMD0()
	cmmd1_stat = CMMD1()
	cmmd2_stat = CMMD2()
	
	algo = Test_Same_Marginal()

	rng = np.random.default_rng(seed)

	n = sample_size

	results_cmmd0 = np.zeros(len(thetas))
	results_cmmd1 = np.zeros(len(thetas))
	results_cmmd2 = np.zeros(len(thetas))

	for i, theta in enumerate(thetas):
		# print current theta
		print(f"Running power experiment for theta: {theta:.2f}")
		
		reject0 = 0
		reject1 = 0
		reject2 = 0


		for trial in range(n_trials):
			seed_y = int(rng.integers(1, 2**31))
			seed_z = int(rng.integers(1, 2**31))
			seed_perm = int(rng.integers(1, 2**31))

			X_P, Y = sample_joint_theta(n, theta, conditional_y, noise_std=noise_std, seed=seed_y)
			X_Q, Z = sample_joint_theta(n, theta, conditional_z, noise_std=noise_std, seed=seed_z)

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

		# print power for current theta
		print(
			f"Power for theta={theta:.2f}: "
			f"CMMD0 = {results_cmmd0[i]:.3f}, "
			f"CMMD1 = {results_cmmd1[i]:.3f}, "
			f"CMMD2 = {results_cmmd2[i]:.3f}"
		)

	return {
		"cmmd0": results_cmmd0,
		"cmmd1": results_cmmd1,
		"cmmd2": results_cmmd2,
	}


def plot_power_vs_sample_size(
	thetas: list[float],
	results: dict[str, np.ndarray]
):
	"""
	Plot power/type I error vs sample size for CMMD0, CMMD1, and CMMD2.
	"""
	fig, ax = plt.subplots(figsize=(8, 6))

	ax.plot(thetas, results["cmmd0"], marker="o", label="CMMD$_0$")
	ax.plot(thetas, results["cmmd1"], marker="s", label="CMMD$_1$")
	ax.plot(thetas, results["cmmd2"], marker="^", label="CMMD$_2$")

	ax.set_title("Power curve for CMMD Test", fontsize=24)
	ax.set_xlabel("$\\theta$", fontsize=20)
	ax.set_ylabel("Power", fontsize=20)
	ax.set_ylim(0.0, 1.1)
	ax.legend(fontsize=16)
	ax.tick_params(axis="both", which="major", labelsize=14)
	ax.grid(True, alpha=0.3)

	plt.tight_layout()

	return fig, ax


if __name__ == "__main__":
	thetas = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]

	results = run_power_experiment(
		thetas = thetas,
		sample_size=100,
		n_trials=250,
		alpha=0.05,
		B=250,
		lam_p=0.1,
		lam_q=0.1,
		bandwidth=0.1,
		noise_std=0.5,
		seed=42,
	)

	fig, _ = plot_power_vs_sample_size(thetas, results)

	figs_dir = os.path.join(project_root, "figs/synthetic")
	os.makedirs(figs_dir, exist_ok=True)
	fig_path = os.path.join(figs_dir, f"power_vs_theta.pdf")
	fig.savefig(fig_path, dpi=300, bbox_inches="tight")
	print(f"Figure saved to: {fig_path}")

	plt.show()
