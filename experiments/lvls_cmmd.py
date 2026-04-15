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

from src.cmmd import CMMD0, CMMD1, CMMD2, CMMDs, Test_Same_Marginal
from src.kernels import gaussian_kernel, median_heuristic
from src.lvls_models import sample_joint, conditional_y, conditional_z


def run_power_experiment(
	thetas: list[float],
	sample_size: int,
	setting: int = 2,
	n_trials: int = 100,
	alpha: float = 0.05,
	B: int = 200,
	lam_p: float = 0.1,
	lam_q: float = 0.1,
	noise_std: float = 0.3,
	cmmd2_estimator: str = "jmmd",
	seed: int = 42
) -> dict[str, np.ndarray]:
	"""
	Estimate power (rejection rate) across theta values for CMMD tests.
	"""
	stat_configs = [
		{"level": 0.0, "label": "CMMD$_0$", "stat": CMMD0(), "extra_stat_kwargs": {}},
		{"level": 0.5, "label": "CMMD$_{0.5}$", "stat": CMMDs(level=0.5), "extra_stat_kwargs": {}},
		{"level": 1.0, "label": "CMMD$_1$", "stat": CMMD1(), "extra_stat_kwargs": {}},
		{"level": 1.5, "label": "CMMD$_{1.5}$", "stat": CMMDs(level=1.5), "extra_stat_kwargs": {}},
		{
			"level": 2.0,
			"label": "CMMD$_2$",
			"stat": CMMD2(),
			"extra_stat_kwargs": {"estimator": cmmd2_estimator},
		},
		{"level": 2.5, "label": "CMMD$_{2.5}$", "stat": CMMDs(level=2.5), "extra_stat_kwargs": {}},
	]

	if setting == 1:
		setting_name = "Setting 1"
	elif setting == 2:
		setting_name = "Setting 2"
	else:
		raise ValueError(f"Unknown covariate setting: {setting}")

	print(f"Running {setting_name}")

	algo = Test_Same_Marginal()
	rng = np.random.default_rng(seed)
	n = sample_size
	results = {cfg["label"]: np.zeros(len(thetas)) for cfg in stat_configs}

	for i, theta in enumerate(thetas):
		print(f"Running power experiment for theta: {theta:.2f}")
		reject_counts = {cfg["label"]: 0 for cfg in stat_configs}

		for trial in range(n_trials):
			seed_y = int(rng.integers(1, 2**31))
			seed_z = int(rng.integers(1, 2**31))
			seed_perm = int(rng.integers(1, 2**31))

			X_P, Y = sample_joint(
				n,
				theta,
				conditional_y,
				noise_std=noise_std,
				setting=setting,
				seed=seed_y,
			)
			X_Q, Z = sample_joint(
				n,
				theta,
				conditional_z,
				noise_std=noise_std,
				setting=setting,
				seed=seed_z,
			)

			X_P = X_P.reshape(-1, 1)
			Y = Y.reshape(-1, 1)
			X_Q = X_Q.reshape(-1, 1)
			Z = Z.reshape(-1, 1)

			# get bandwidth for current dimension via median heuristic
			bandwidth = median_heuristic(X_P)

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

			trial_p_values = {}
			for j, cfg in enumerate(stat_configs):
				_, p_value = algo.test(
					X_P,
					Y,
					X_Q,
					Z,
					cfg["stat"],
					gaussian_kernel,
					algo_kwargs={**algo_kwargs, "random_state": seed_perm + j},
					stat_kwargs={**stat_kwargs, **cfg["extra_stat_kwargs"]},
				)

				reject_counts[cfg["label"]] += int(p_value < alpha)
				trial_p_values[cfg["label"]] = p_value

			# print every 50th trial for progress
			if (trial + 1) % 50 == 0:
				pval_str = ", ".join(
					f"{label} p-value = {trial_p_values[label]:.4f}" for label in trial_p_values
				)
				print(f"Trial {trial+1:d}/{n_trials:d} for n={n:d}: {pval_str}")

		for cfg in stat_configs:
			results[cfg["label"]][i] = reject_counts[cfg["label"]] / n_trials

		# print power for current theta
		power_str = ", ".join(
			f"{cfg['label']} = {results[cfg['label']][i]:.3f}" for cfg in stat_configs
		)
		print(f"Power for theta={theta:.2f}: {power_str}")

	return results


def plot_power_vs_theta(
	thetas: list[float],
	results: dict[str, np.ndarray],
	setting: int = 1,
):
	"""
	Plot power/type I error vs theta for CMMD levels.
	"""
	fig, ax = plt.subplots(figsize=(8, 6))

	markers = ["o", "s", "^", "D", "v", "8"]
	label_colors = {
		"CMMD$_0$": "tab:blue",
		"CMMD$_1$": "tab:orange",
		"CMMD$_2$": "tab:green",
		"CMMD$_{0.5}$": "tab:brown",
		"CMMD$_{1.5}$": "tab:pink",
		"CMMD$_{2.5}$": "tab:olive",
	}
	for marker, label in zip(markers, results.keys()):
		ax.plot(
			thetas,
			results[label],
			marker=marker,
			label=label,
			color=label_colors.get(label),
		)

	if setting == 1:
		setting_name = "Setting 1"
	elif setting == 2:
		setting_name = "Setting 2"
	else:
		raise ValueError(f"Unknown covariate setting: {setting}")

	ax.set_title(f"Power Curve - {setting_name}", fontsize=24)
	ax.set_xlabel("Difference parameter ($\\theta$)", fontsize=20)
	ax.set_ylabel("Power", fontsize=20)
	ax.set_ylim(0.0, 1.1)
	ax.legend(fontsize=16)
	ax.tick_params(axis="both", which="major", labelsize=14)
	ax.grid(True, alpha=0.3)

	plt.tight_layout()

	return fig, ax


if __name__ == "__main__":
	thetas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	setting = 2

	results = run_power_experiment(
			thetas=thetas,
			sample_size=100,
			setting=setting,
			n_trials=200,
			alpha=0.05,
			B=200,
			lam_p=0.1,
			lam_q=0.1,
			noise_std=0.5,
			seed=42,
		)

	fig, _ = plot_power_vs_theta(thetas, results, setting=setting)

	# figs_dir = os.path.join(project_root, "figs/lvls")
	# os.makedirs(figs_dir, exist_ok=True)
	# fig_path = os.path.join(figs_dir, f"power_curve_setting_{setting}.pdf")
	# fig.savefig(fig_path, dpi=300, bbox_inches="tight")
	# print(f"Figure saved to: {fig_path}")

	plt.show()
