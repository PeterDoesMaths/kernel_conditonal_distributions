# Measuring Differences between Conditional Distributions using Kernel Embeddings

## Experiments Overview

The experiments are designed to validate the CMMD framework across controlled synthetic settings and a real-data benchmark.

- Synthetic conditional-shift experiments (`experiments/synthetic_plot_data.py`, `experiments/synthetic_plot_error_rate.py`, `experiments/synthetic_plot_test_stat.py`, `experiments/synthetic_power_plot.py`, `experiments/synthetic_dimension.py`) plots data samples, evaluate Type I error, power, and statistic behavior for CMMD$_0$, CMMD$_1$, and CMMD$_2$ under both equal and unequal covariate marginals. These scripts also study sensitivity to signal strength (`\theta`) and input dimension.
- Level comparisons (`experiments/lvls_cmmd.py`) evaluate the broader CMMD$_s$ family (including intermediate levels such as $s=0.5,1.5,2.5$) to illustrate how the smoothing level changes test power.
- Doubly robust experiments (`experiments/dr_cme.py`, `experiments/dr_cmmd.py`, `experiments/dr_cmmd_error.py`, `experiments/dr_cmmd_rejection_rate.py`) compare standard and doubly robust estimators, including estimation-error curves and rejection-rate behavior under null and alternative settings.
- MNIST experiments (`experiments/mnist_error_rate.py`, with preprocessing in `src/clean_mnist.py`) treat digit label as the covariate and image representation as the conditional outcome, testing whether CMMD detects conditional changes under covariate shift.

Generated figures are saved under `figs/`, and numeric outputs used for plotting are stored in `data/experiment_data/`.


### Setting up a Virtual Environment

It's recommended to use a virtual environment to avoid dependency conflicts:

1. **Create a virtual environment**:
   ```bash
   python3 -m venv .venv
   ```

2. **Activate the virtual environment**:
   
   On macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```
   
   On Windows:
   ```bash
   .venv\Scripts\activate
   ```
   
   You should see `(.venv)` appear at the beginning of your command prompt.

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```