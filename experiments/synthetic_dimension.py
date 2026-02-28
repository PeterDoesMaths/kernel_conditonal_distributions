"""
Plot test power against dimension for CMMD-based tests.
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
from src.models import sample_joint_theta, conditional_y, conditional_z


def sample_joint_theta_multidim(
	n: int,
	dimension: int,
	theta: float,
	conditional_fn,
	noise_std: float = 0.5,
	seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Sample (X, Y) in d dimensions, with dimension 1 informative and other dimensions noise.
	"""
	rng = np.random.default_rng(seed)

	X_cols = []
	Y_cols = []
	
	for d in range(dimension):
		seed_d = rng.integers(0, 2**31)
		X_d, Y_d = sample_joint_theta(
			n,
			theta,
			conditional_fn,
			noise_std=(0.1*d+1)*noise_std,
			seed=seed_d,
		)
		X_cols.append(X_d)
		Y_cols.append(Y_d)

	X = np.column_stack(X_cols)
	Y = np.column_stack(Y_cols)

	return X, Y


def run_dimension_power_experiment(
	dimensions: list[int],
	theta: float = 0.5,
	sample_size: int = 100,
	n_trials: int = 100,
	alpha: float = 0.05,
	B: int = 200,
	lam_p: float = 0.1,
	lam_q: float = 0.1,
	noise_std: float = 0.5,
	cmmd2_estimator: str = "jmmd",
	seed: int = 42,
) -> dict[str, np.ndarray]:
	"""
	Estimate rejection rates (power) over a grid of dimensions.
	"""
	cmmd0_stat = CMMD0()
	cmmd1_stat = CMMD1()
	cmmd2_stat = CMMD2()
	algo = Test_Same_Marginal()

	rng = np.random.default_rng(seed)

	results_cmmd0 = np.zeros(len(dimensions))
	results_cmmd1 = np.zeros(len(dimensions))
	results_cmmd2 = np.zeros(len(dimensions))

	for i, d in enumerate(dimensions):
		print(f"Running dimension experiment for d={d:d}")

		reject0 = 0
		reject1 = 0
		reject2 = 0

		# get bandwidth for current dimension via median heuristic
		X_cols = []
		for _ in range(d):
			X_noise = rng.normal(theta, 0.75, size=sample_size)
			X_cols.append(X_noise)
		X = np.column_stack(X_cols)
		bandwidth = median_heuristic(X)

		for trial in range(n_trials):
			seed_y = int(rng.integers(1, 2**31))
			seed_z = int(rng.integers(1, 2**31))
			seed_perm = int(rng.integers(1, 2**31))

			X_P, Y = sample_joint_theta_multidim(
				n=sample_size,
				dimension=d,
				theta=theta,
				conditional_fn=conditional_y,
				noise_std=noise_std,
				seed=seed_y,
			)
			X_Q, Z = sample_joint_theta_multidim(
				n=sample_size,
				dimension=d,
				theta=theta,
				conditional_fn=conditional_z,
				noise_std=noise_std,
				seed=seed_z,
			)

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
				X_P,
				Y,
				X_Q,
				Z,
				cmmd0_stat,
				gaussian_kernel,
				algo_kwargs=algo_kwargs,
				stat_kwargs=stat_kwargs,
			)
			_, p1 = algo.test(
				X_P,
				Y,
				X_Q,
				Z,
				cmmd1_stat,
				gaussian_kernel,
				algo_kwargs={**algo_kwargs, "random_state": seed_perm + 1},
				stat_kwargs=stat_kwargs,
			)
			_, p2 = algo.test(
				X_P,
				Y,
				X_Q,
				Z,
				cmmd2_stat,
				gaussian_kernel,
				algo_kwargs={**algo_kwargs, "random_state": seed_perm + 2},
				stat_kwargs={**stat_kwargs, "estimator": cmmd2_estimator},
			)

			reject0 += int(p0 < alpha)
			reject1 += int(p1 < alpha)
			reject2 += int(p2 < alpha)

			if (trial + 1) % 25 == 0:
				print(
					f"Trial {trial+1:d}/{n_trials:d} for d={d:d}: "
					f"CMMD0 p-value = {p0:.4f}, CMMD1 p-value = {p1:.4f}, CMMD2 p-value = {p2:.4f}"
				)

		results_cmmd0[i] = reject0 / n_trials
		results_cmmd1[i] = reject1 / n_trials
		results_cmmd2[i] = reject2 / n_trials

		print(
			f"Power for d={d:d}: "
			f"CMMD0 = {results_cmmd0[i]:.3f}, "
			f"CMMD1 = {results_cmmd1[i]:.3f}, "
			f"CMMD2 = {results_cmmd2[i]:.3f}"
		)

	return {
		"cmmd0": results_cmmd0,
		"cmmd1": results_cmmd1,
		"cmmd2": results_cmmd2,
	}


def plot_power_vs_dimension(
	dimensions: list[int],
	results: dict[str, np.ndarray],
):
	"""
	Plot power vs dimension for CMMD0, CMMD1, and CMMD2.
	"""
	fig, ax = plt.subplots(figsize=(8, 6))

	ax.plot(dimensions, results["cmmd0"], marker="o", label="CMMD$_0$")
	ax.plot(dimensions, results["cmmd1"], marker="s", label="CMMD$_1$")
	ax.plot(dimensions, results["cmmd2"], marker="^", label="CMMD$_2$")

	ax.set_title("Power Decay", fontsize=24)
	ax.set_xlabel("Dimension ($D$)", fontsize=20)
	ax.set_ylabel("Power", fontsize=20)
	ax.set_ylim(0.0, 1.1)
	ax.legend(fontsize=16)
	ax.tick_params(axis="both", which="major", labelsize=14)
	ax.grid(True, alpha=0.3)

	plt.tight_layout()
	return fig, ax


if __name__ == "__main__":
	dimensions = [1, 20, 40, 60, 80, 100, 120, 140, 160]

	results = run_dimension_power_experiment(
		dimensions=dimensions,
		theta=0.5,
		sample_size=100,
		n_trials=200,
		alpha=0.05,
		B=200,
		lam_p=0.1,
		lam_q=0.1,
		noise_std=0.5,
		seed=42,
	)

	fig, _ = plot_power_vs_dimension(dimensions, results)

	figs_dir = os.path.join(project_root, "figs/synthetic")
	os.makedirs(figs_dir, exist_ok=True)
	fig_path = os.path.join(figs_dir, "power_vs_dimension.pdf")
	fig.savefig(fig_path, dpi=300, bbox_inches="tight")
	print(f"Figure saved to: {fig_path}")

	plt.show()
