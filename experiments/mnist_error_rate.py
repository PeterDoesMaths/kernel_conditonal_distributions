"""
Plot power against sample size for CMMD-based tests using MNIST data.

Tests conditional distribution differences under covariate shift (different marginals).
- Covariate X: digit class (0-9)
- Conditional outcome Y|X: images of digit X
- Null: Z|X = Y|X (same conditional distribution)
- Alternative: Z|X biased towards brightest images
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.cmmd import CMMD0_primal, CMMD1_primal, CMMD2_primal, Test_Diff_Marginal
from src.kernels import gaussian_kernel, kronecker_delta_kernel, median_heuristic


def load_mnist_data() -> pd.DataFrame:
    """
    Load cleaned MNIST test data from CSV.
    
    Returns
    -------
    df : pd.DataFrame
        Columns: X (digit label), Y_0 to Y_783 (normalized pixel values)
    """
    data_path = Path(project_root) / "data" / "clean_mnist" / "mnist_test.csv"
    df = pd.read_csv(data_path)
    return df


def sample_mnist_images(
    df: pd.DataFrame,
    digit: int,
    n_samples: int,
    bias_brightness: bool = False,
    seed: int = None
) -> np.ndarray:
    """
    Sample images of a given digit from MNIST.
    
    Parameters
    ----------
    df : pd.DataFrame
        MNIST data with X (label) and Y_* (pixel columns).
    digit : int
        Which digit (0-9) to sample images from.
    n_samples : int
        Number of images to sample.
    bias_brightness : bool, default=False
        If True, bias sampling towards brightest images.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    images : np.ndarray, shape (n_samples, d)
        Sampled images.
    """
    rng = np.random.default_rng(seed)
    
    # Get all images of this digit
    pixel_cols = [col for col in df.columns if col.startswith('Y_')]
    digit_images = df[df['X'] == digit][pixel_cols].values  # (n_digit, d)
    
    if len(digit_images) == 0:
        raise ValueError(f"No images found for digit {digit}")
    
    if bias_brightness:
        # Compute brightness (sum of pixel values) for each image
        brightness = digit_images.sum(axis=1)
        # Higher brightness gets higher probability
        probs = brightness - brightness.min()
        probs = probs / probs.sum()
        
        # Sample with bias towards bright images
        indices = rng.choice(len(digit_images), size=n_samples, p=probs, replace=True)
    else:
        # Uniform sampling
        indices = rng.choice(len(digit_images), size=n_samples, replace=True)
    
    return digit_images[indices]


def sample_mnist_joint(
    df: pd.DataFrame,
    probs: np.ndarray,
    n_samples: int,
    bias_brightness: bool = False,
    seed: int = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample joint (digit, image) from MNIST under given marginal.
    
    Parameters
    ----------
    df : pd.DataFrame
        MNIST data.
    probs : np.ndarray, shape (n_digits,)
        Probabilities for each digit.
    n_samples : int
        Number of samples.
    bias_brightness : bool, default=False
        If True, bias image sampling towards brightest.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    X : np.ndarray, shape (n_samples,)
        Digit labels.
    Y : np.ndarray, shape (n_samples, d)
        Images.
    """
    rng = np.random.default_rng(seed)
    
    X = rng.choice(len(probs), size=n_samples, p=probs)  # Sample digit labels according to probs
    
    # Sample images for each digit
    Y_list = []
    for digit in X:
        img = sample_mnist_images(df, digit, 1, bias_brightness=bias_brightness, seed=rng.integers(0, 2**31))
        Y_list.append(img[0])
    
    Y = np.array(Y_list)  # (n_samples, d)
    
    return X, Y


def run_mnist_power_experiment(
    df: pd.DataFrame,
    error: str,
    sample_sizes: list[int],
    n_trials: int = 50,
    alpha: float = 0.05,
    B: int = 200,
    seed: int = 42
) -> dict[str, np.ndarray]:
    """
    Estimate power/type I error across sample sizes for CMMD tests on MNIST.
    
    Parameters
    ----------
    df : pd.DataFrame
        MNIST data.
    error : str
        "type1" for Type I error (null hypothesis) or "type2" for power (alternative).
    sample_sizes : list[int]
        Sample sizes to test.
    n_trials : int
        Number of trials per sample size.
    alpha : float
        Type I error level for test.
    B : int
        Number of permutations for test.
    seed : int
        Random seed.
    
    Returns
    -------
    results : dict
        Keys: "cmmd0", "cmmd1", "cmmd2" with arrays of rejection rates.
    """
    cmmd0_stat = CMMD0_primal()
    cmmd1_stat = CMMD1_primal()
    cmmd2_stat = CMMD2_primal()
    algo = Test_Diff_Marginal()
    
    rng = np.random.default_rng(seed)
    
    # Marginals: P_X uniform over all digits, Q_X uniform but shifted
    digits_P = np.arange(10)  # All digits 0-9
    digits_Q = np.arange(10)  # Same for now (can modify for stronger shift)
    Px_probs = np.ones(len(digits_P)) / len(digits_P)
    Qx_probs = np.array([0.145-0.01*x for x in digits_Q])  # Slightly shift towards lower digits
    
    results_cmmd0 = np.zeros(len(sample_sizes))
    results_cmmd1 = np.zeros(len(sample_sizes))
    results_cmmd2 = np.zeros(len(sample_sizes))
    
    for i, n in enumerate(sample_sizes):
        print(f"\nRunning {error} experiment for sample size: {n}")
        
        reject0 = 0
        reject1 = 0
        reject2 = 0
        
        for trial in range(n_trials):
            seed_y = int(rng.integers(1, 2**31))
            seed_z = int(rng.integers(1, 2**31))
            seed_perm = int(rng.integers(1, 2**31))
            
            # Sample from P: (X_P, Y) where Y ~ P_{Y|X}
            X_P, Y = sample_mnist_joint(
                df, Px_probs, n, bias_brightness=False, seed=seed_y
            )
            
            # Sample from Q: (X_Q, Z) where Z ~ Q_{Z|X}
            if error == "type1":
                # Null: Z|X = Y|X (same conditional)
                X_Q, Z = sample_mnist_joint(
                    df, Qx_probs, n, bias_brightness=False, seed=seed_z
                )
            elif error == "type2":
                # Alternative: Z|X biased towards brighter images
                X_Q, Z = sample_mnist_joint(
                    df, Qx_probs, n, bias_brightness=True, seed=seed_z
                )
            else:
                raise ValueError(f"Unknown error type: {error}")
            
            # Reshape covariates to 2D for kernel computation
            X_P = X_P.reshape(-1, 1)
            X_Q = X_Q.reshape(-1, 1)
            
            # Compute bandwidth via median heuristic
            all_images = np.vstack([Y, Z])
            bandwidth = median_heuristic(all_images)
            
            # Regularization parameter: n^{-1/4}
            lam = n ** (-0.25)
            
            # Test statistic kwargs
            stat_kwargs = {"bandwidth": bandwidth}
            algo_kwargs = {
                "alpha": alpha,
                "B": B,
                "lam_p": lam,
                "lam_q": lam,
                "random_state": seed_perm,
            }
            
            # Propensity function (uniform marginals assumed)
            def propensity_fn(X):
                # For uniform marginals, propensity ≈ 0.5
                # X_flat = np.asarray(X).flatten()
                # return np.full(X_flat.shape[0], 0.5, dtype=float)
                # e(x) = 1 / (2.45 - 0.1 x)
                e = 1 / (2.45 - 0.1 * X.flatten())
                return e
            
            algo_kwargs["propensity_fn"] = propensity_fn
            
            # Run tests
            try:
                _, p0 = algo.test(
                    X_P, Y, X_Q, Z,
                    cmmd0_stat,
                    kronecker_delta_kernel,
                    gaussian_kernel,
                    algo_kwargs=algo_kwargs,
                    stat_kwargs=stat_kwargs,
                )
                _, p1 = algo.test(
                    X_P, Y, X_Q, Z,
                    cmmd1_stat,
                    kronecker_delta_kernel,
                    gaussian_kernel,
                    algo_kwargs={**algo_kwargs, "random_state": seed_perm + 1},
                    stat_kwargs=stat_kwargs,
                )
                _, p2 = algo.test(
                    X_P, Y, X_Q, Z,
                    cmmd2_stat,
                    kronecker_delta_kernel,
                    gaussian_kernel,
                    algo_kwargs={**algo_kwargs, "random_state": seed_perm + 2},
                    stat_kwargs=stat_kwargs,
                )
                
                reject0 += int(p0 < alpha)
                reject1 += int(p1 < alpha)
                reject2 += int(p2 < alpha)
                
            except Exception as e:
                print(f"  Warning in trial {trial}: {e}")
                continue
            
            if (trial + 1) % 20 == 0:
                print(
                    f"  Trial {trial+1:d}/{n_trials:d}: "
                    f"p0={p0:.4f}, p1={p1:.4f}, p2={p2:.4f}"
                )
        
        results_cmmd0[i] = reject0 / n_trials
        results_cmmd1[i] = reject1 / n_trials
        results_cmmd2[i] = reject2 / n_trials
        
        print(f"  Results for n={n}: CMMD0={results_cmmd0[i]:.3f}, CMMD1={results_cmmd1[i]:.3f}, CMMD2={results_cmmd2[i]:.3f}")
    
    return {
        "cmmd0": results_cmmd0,
        "cmmd1": results_cmmd1,
        "cmmd2": results_cmmd2,
    }


def plot_error_vs_sample_size(
    sample_sizes: list[int],
    results: dict[str, np.ndarray],
    error: str = "type1"
):
    """
    Plot type I error / power vs sample size for CMMD0, CMMD1, and CMMD2.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Type I error reference line
    if error == "type1":
        ax.axhline(0.05, color="black", linestyle="--", linewidth=2)
    
    ax.plot(sample_sizes, results["cmmd0"], marker="o", markersize=8, linewidth=2, label="CMMD$_0$")
    ax.plot(sample_sizes, results["cmmd1"], marker="s", markersize=8, linewidth=2, label="CMMD$_1$")
    ax.plot(sample_sizes, results["cmmd2"], marker="^", markersize=8, linewidth=2, label="CMMD$_2$")
    
    ax.set_xlabel("Sample size ($n$)", fontsize=20)
    
    if error == "type1":
        ax.set_title(r"$P_{Y|X} = Q_{Z|X}$", fontsize=24)
        ax.set_ylabel("Type I Error", fontsize=20)
        ax.set_ylim([0.0, 0.3])
        ax.legend(fontsize=16, loc="best")
    else:
        ax.set_title(r"$P_{Y|X} \neq Q_{Z|X}$", fontsize=24)
        ax.set_ylabel("Power", fontsize=20)
        ax.set_ylim([0.0, 1.1])
        ax.legend(fontsize=16, loc="best")
    
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # Configuration
    sample_sizes = [200, 400, 600, 800, 1000, 1200]
    n_trials = 200
    # error = "type1"
    error = "type2"
    
    # Load MNIST data
    print("Loading MNIST data...")
    df = load_mnist_data()
    print(f"  Loaded {len(df)} MNIST images")
    
    # Run experiment
    results = run_mnist_power_experiment(
        df,
        error=error,
        sample_sizes=sample_sizes,
        n_trials=n_trials,
        alpha=0.05,
        B=200,
        seed=42,
    )
    
    # Plot
    fig, ax = plot_error_vs_sample_size(sample_sizes, results, error=error)
    
    # Save figure
    figs_dir = Path(project_root) / "figs" / "mnist"
    figs_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figs_dir / f"{error}_vs_sample_size.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {fig_path}")
    
    plt.show()
