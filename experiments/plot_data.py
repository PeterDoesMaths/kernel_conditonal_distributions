"""
Visualize the conditional distributions from the synthetic data model.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import sample_joint, conditional_y, conditional_z


def plot_conditional_distributions(n_samples: int = 500, seed: int = 42):
    """
    Plot samples from both conditional distributions P_{Y|X} and P_{Z|X}.
    """
    # Sample from both distributions
    X_y, Y = sample_joint(n_samples, conditional_y, noise_std=0.3, seed=seed)
    X_z, Z = sample_joint(n_samples, conditional_z, noise_std=0.3, seed=seed+1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot both distributions on same axes
    ax.scatter(X_y, Y, alpha=0.6, s=30, c='blue', marker='o', label='P')
    ax.scatter(X_z, Z, alpha=0.6, s=30, c='red', marker='s', label='Q')
    
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    ax.set_title('Simulated Data', fontsize=24)
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent.parent / 'figs'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'conditional_distributions.pdf'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    
    # plt.show()


if __name__ == '__main__':
    plot_conditional_distributions()
