"""
Experiment: Conditional Mean Embedding (CME) using Kernel Ridge Regression.

This script samples from the models defined in dr_models.py and computes
the conditional mean embeddings for Y and Z using kernel ridge regression.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import sys
from pathlib import Path
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.dr_models import sample_joint, sample_covariate_p, sample_covariate_q, conditional_y, conditional_z, propensity

def cme_model(
    X: np.ndarray,
    Y: np.ndarray,
    alpha_grid: Optional[np.ndarray] = None,
    cv: int = 5,
) -> KernelRidge:
    """
    Fit CME model via sklearn KernelRidge with CV-tuned regularization.
    
    Parameters
    ----------
    X : np.ndarray, shape (n,) or (n, d)
        Training covariates.
    Y : np.ndarray, shape (n,) or (n, 1)
        Training responses.
    alpha_grid : np.ndarray, optional
        Candidate regularization values for CV.
    cv : int, default=5
        Number of cross-validation folds.
    
    Returns
    -------
    KernelRidge
        Best fitted model selected by cross-validation.
    """
    if alpha_grid is None:
        alpha_grid = np.logspace(-4, 1, 5)

    x_train_2d = X.reshape(-1, 1) if X.ndim == 1 else X
    y_train_1d = Y.ravel()

    base_model = KernelRidge(kernel="polynomial", degree=2, coef0=1, gamma=1)
    cv_folds = min(cv, x_train_2d.shape[0])

    if cv_folds < 2:
        fallback_model = KernelRidge(kernel="polynomial", degree=2, coef0=1, gamma=1, alpha=float(alpha_grid[0]))
        fallback_model.fit(x_train_2d, y_train_1d)
        return fallback_model

    search = GridSearchCV(
        estimator=base_model,
        param_grid={"alpha": alpha_grid},
        cv=cv_folds,
        scoring="neg_mean_squared_error",
    )
    search.fit(x_train_2d, y_train_1d)

    return search.best_estimator_

def plot_cme_data(x_eval: np.ndarray, cme_Y: np.ndarray, cme_Z: np.ndarray, X_p: np.ndarray, Y: np.ndarray, X_q: np.ndarray, Z: np.ndarray):
    """
    Plot the CME estimates for Y and Z, as well as the training data. Have different plots for P and Q samples.
    """

    true_y = np.cos(4 * np.pi * x_eval) + 0.5 * x_eval**2 
    true_z = np.cos(4 * np.pi * x_eval)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

    # plot 1, CME for Y with P samples
    axes[0].plot(x_eval, cme_Y, label="CME Y|X", linewidth=2, color='blue')
    axes[0].plot(x_eval, true_y, label="True Y|X", linewidth=2, color='black', linestyle='--')
    axes[0].scatter(X_p, Y, label="P samples", alpha=0.5, color='blue', edgecolor='k', s=50)
    axes[0].set_xlabel("$x$", fontsize=20)
    axes[0].set_ylabel(r"$\mu_{Y|x}$", fontsize=20)
    axes[0].set_title("Conditional Mean Embedding for Y|X", fontsize=20)
    axes[0].tick_params(axis="both", which="major", labelsize=14)
    axes[0].legend(fontsize=16)
    axes[0].grid(True, alpha=0.3)

    # plot 2, CME for Z with Q samples
    axes[1].plot(x_eval, cme_Z, label="CME Z|X", linewidth=2, color='red')
    axes[1].plot(x_eval, true_z, label="True Z|X", linewidth=2, color='black', linestyle='--')
    axes[1].scatter(X_q, Z, label="Q samples", alpha=0.5, color='red', edgecolor='k', s=50)
    axes[1].set_xlabel("$x$", fontsize=20)
    axes[1].set_ylabel(r"$\mu_{Z|x}$", fontsize=20)
    axes[1].set_title("Conditional Mean Embedding for Z|X", fontsize=20)
    axes[1].tick_params(axis="both", which="major", labelsize=14)
    axes[1].legend(fontsize=16)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    plt.show()

def plot_pseudo_outcome(X_test: np.ndarray, pseudo_outcome: np.ndarray, peudo_cme_diff: np.ndarray):
    """Plot the pseudo-outcome used for doubly robust estimation.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sc = ax.scatter(X_test, pseudo_outcome, alpha=0.6, edgecolors='w')
    ax.plot(np.sort(X_test), peudo_cme_diff[np.argsort(X_test)], color='red', linewidth=2, label='KRR Fit of Pseudo-outcome')
    ax.set_xlabel("$x$", fontsize=20)
    ax.set_ylabel("Pseudo-outcome", fontsize=20)
    ax.set_title("Pseudo-outcome for Doubly Robust Estimation", fontsize=24)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_cme_difference(x_eval: np.ndarray, true_diff: np.ndarray, standard_cme_diff: np.ndarray, dr_cme_diff: np.ndarray):
    """
    Plot the true difference, standard CME difference, and doubly robust CME difference.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(x_eval, true_diff, '--', label="True Difference", 
            linewidth=2, alpha=0.7, color='black')
    ax.plot(x_eval, standard_cme_diff, label="Standard Estimator", 
            linewidth=2, color='blue')
    ax.plot(x_eval, dr_cme_diff, label="DR Estimator", 
            linewidth=2, color='red')
    ax.set_xlabel("$x$", fontsize=30)
    ax.set_ylabel(r"$\mu_{Y|x} - \mu_{Z|x}$", fontsize=30)
    ax.set_title("Difference in CMEs", fontsize=36)
    ax.tick_params(axis="both", which="major", labelsize=21)
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent.parent / "figs" / "dr"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cme_difference.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    plt.show()


def main():
    # Set random seed for reproducibility
    seed = 123
    
    # Sample size
    n_samples = 500
    
    # Sample from P and Q
    X_p, Y = sample_joint(
        n_samples, sample_covariate_p, conditional_y, seed=seed
    )
    X_q, Z = sample_joint(
        n_samples, sample_covariate_q, conditional_z, seed=seed + 1
    )
    
    # Points to evaluate CME difference
    x_eval = np.linspace(0, 1, 200)

    # True conditional mean difference
    true_diff = 0.5 * x_eval**2
    
    # Fit CME models once
    model_Y = cme_model(X_p, Y)
    model_Z = cme_model(X_q, Z)

    x_eval_2d = x_eval.reshape(-1, 1)

    # Evaluate fitted models at x_eval
    cme_Y = model_Y.predict(x_eval_2d)
    cme_Z = model_Z.predict(x_eval_2d)
    standard_cme_diff = cme_Y - cme_Z

    # merge data for DR estimation
    X_test = np.concatenate([X_p, X_q])
    X_test_2d = X_test.reshape(-1, 1)
    YZ_test = np.concatenate([Y, Z])
    
    # T indicates which samples are from P (T=1) vs Q (T=0)
    T = np.concatenate([np.ones_like(Y), np.zeros_like(Z)])

    # Compute Propensity scores on test set
    E = propensity(X_test)

    # Reuse the same fitted CME models at X_test
    cme_Y_train = model_Y.predict(X_test_2d)
    cme_Z_train = model_Z.predict(X_test_2d)

    # RKHS difference psuedo-outcome for doubly robust estimation
    psuedo_outcome = (T - E) / (E * (1 - E)) * (YZ_test - (1 - E) * cme_Y_train - E * cme_Z_train)

    plot_cme_data(x_eval, cme_Y, cme_Z, X_p, Y, X_q, Z)

    # Compute doubly robust CME difference
    model_dr = cme_model(X_test, psuedo_outcome)

    peudo_cme_diff = model_dr.predict(X_test_2d)
    plot_pseudo_outcome(X_test, psuedo_outcome, peudo_cme_diff)

    dr_cme_diff = model_dr.predict(x_eval_2d)
    
    # Plot results
    plot_cme_difference(x_eval, true_diff, standard_cme_diff, dr_cme_diff)


if __name__ == "__main__":
    main()
