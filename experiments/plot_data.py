"""
Visualize the conditional distributions from the synthetic data model.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import sample_joint, sample_covariate, sample_covariate_p, sample_covariate_q, conditional_y, conditional_z


def plot_conditional_distributions(setting: str = 'diff_marginal', n_samples: int = 250, noise_std: float = 0.5,  seed: int = 42):
    """
    Plot samples from both conditional distributions P_{Y|X} and P_{Z|X}.
    """
    # Sample from both distributions
    if setting == 'same_marginal':
        X_P, Y = sample_joint(n_samples, sample_covariate, conditional_y,  noise_std=noise_std, seed=seed)
        X_Q, Z = sample_joint(n_samples, sample_covariate, conditional_z, noise_std=noise_std, seed=seed+1)
    elif setting == 'diff_marginal':
        X_P, Y = sample_joint(n_samples, sample_covariate_p, conditional_y,  noise_std=noise_std, seed=seed+1)
        X_Q, Z = sample_joint(n_samples, sample_covariate_q, conditional_z, noise_std=noise_std, seed=seed+3)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot both distributions on same axes
    ax.scatter(X_P, Y, alpha=0.6, s=30, c='blue', marker='o', label='$P$')
    ax.scatter(X_Q, Z, alpha=0.6, s=30, c='red', marker='s', label='$Q$')
    
    ax.set_xlabel('$X$', fontsize=20)
    ax.set_ylabel('$Y$, $Z$', fontsize=20)
    ax.set_title('Simulated Data', fontsize=24)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent.parent / 'figs'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'synthetic_data_plot_{setting}.pdf'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    
    plt.show()


if __name__ == '__main__':
    # setting = 'same_marginal'
    setting = 'diff_marginal'
    plot_conditional_distributions(setting=setting)
