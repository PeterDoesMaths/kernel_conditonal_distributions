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

from src.cmmd import CMMD0, CMMD1, CMMD2, Test_Same_Marginal
from src.kernels import gaussian_kernel, median_heuristic
from src.models import sample_joint, conditional_y, conditional_z


def run_power_experiment(
	sample_sizes: list[int],
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
	algo = Test_Same_Marginal()
	cmmd0_stat = CMMD0()
	cmmd1_stat = CMMD1()
	cmmd2_stat = CMMD2()

	rng = np.random.default_rng(seed)

	power_cmmd0 = np.zeros(len(sample_sizes))
	power_cmmd1 = np.zeros(len(sample_sizes))
	power_cmmd2 = np.zeros(len(sample_sizes))

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

			X_P, Y = sample_joint(n, conditional_y, noise_std=noise_std, seed=seed_y)
			# X_Q, Z = sample_joint(n, conditional_z, noise_std=noise_std, seed=seed_z)
			if error == "type1":
				X_Q, Z = sample_joint(n, conditional_y, noise_std=noise_std, seed=seed_z)
			else:
				X_Q, Z = sample_joint(n, conditional_z, noise_std=noise_std, seed=seed_z)

			X_P = X_P.reshape(-1, 1)
			Y = Y.reshape(-1, 1)
			X_Q = X_Q.reshape(-1, 1)
			Z = Z.reshape(-1, 1)

			_, p0 = algo.test(
				X_P,
				Y,
				X_Q,
				Z,
				cmmd0_stat,
				gaussian_kernel,
				alpha=alpha,
				B=B,
				lam_p=lam_p,
				lam_q=lam_q,
				bandwidth=bandwidth,
				random_state=seed_perm,
			)
			_, p1 = algo.test(
				X_P,
				Y,
				X_Q,
				Z,
				cmmd1_stat,
				gaussian_kernel,
				alpha=alpha,
				B=B,
				lam_p=lam_p,
				lam_q=lam_q,
				bandwidth=bandwidth,
				random_state=seed_perm + 1,
			)
			_, p2 = algo.test(
				X_P,
				Y,
				X_Q,
				Z,
				cmmd2_stat,
				gaussian_kernel,
				alpha=alpha,
				B=B,
				lam_p=lam_p,
				lam_q=lam_q,
				bandwidth=bandwidth,
				estimator=cmmd2_estimator,
				random_state=seed_perm + 2,
			)

			reject0 += int(p0 < alpha)
			reject1 += int(p1 < alpha)
			reject2 += int(p2 < alpha)

		power_cmmd0[i] = reject0 / n_trials
		power_cmmd1[i] = reject1 / n_trials
		power_cmmd2[i] = reject2 / n_trials

	return {
		"cmmd0": power_cmmd0,
		"cmmd1": power_cmmd1,
		"cmmd2": power_cmmd2,
	}


def plot_power_vs_sample_size(
	sample_sizes: list[int],
	power_results: dict[str, np.ndarray],
	error: str = "Power"
):
	"""
	Plot power/type I error vs sample size for CMMD0, CMMD1, and CMMD2.
	"""
	fig, ax = plt.subplots(figsize=(8, 6))

	ax.plot(sample_sizes, power_results["cmmd0"], marker="o", label="CMMD$_0$")
	ax.plot(sample_sizes, power_results["cmmd1"], marker="s", label="CMMD$_1$")
	ax.plot(sample_sizes, power_results["cmmd2"], marker="^", label="CMMD$_2$")

	ax.set_xlabel("Sample size ($n$)", fontsize=20)
	if error == "Power":
		ax.set_ylabel("Power", fontsize=20)
		ax.set_ylim(0.0, 1.1)
	else:
		ax.set_ylabel("Type I Error", fontsize=20)
		ax.set_ylim(0.0, 1.1)
	ax.tick_params(axis="both", which="major", labelsize=14)
	ax.grid(True, alpha=0.3)
	ax.legend(fontsize=16)

	plt.tight_layout()

	return fig, ax


if __name__ == "__main__":
	sample_sizes = [10, 50, 100, 150, 200, 250]
	n_trials = 100 # Change to 250 for final results
	error = "Power" # Change to "type1" for type I error results

	power_results = run_power_experiment(
		sample_sizes=sample_sizes,
		n_trials=n_trials,
		alpha=0.05,
		B=250,
		lam_p=0.1,
		lam_q=0.1,
		bandwidth=0.1,
		noise_std=0.5,
		cmmd2_estimator="jmmd",
		seed=42,
	)

	fig, _ = plot_power_vs_sample_size(sample_sizes, power_results)

	figs_dir = os.path.join(project_root, "figs")
	os.makedirs(figs_dir, exist_ok=True)
	fig_path = os.path.join(figs_dir, "power_vs_sample_size.pdf")
	fig.savefig(fig_path, dpi=300, bbox_inches="tight")
	print(f"Figure saved to: {fig_path}")

	plt.show()
