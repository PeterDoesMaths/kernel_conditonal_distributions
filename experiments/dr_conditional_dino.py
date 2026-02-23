import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

def pi_x(x):
    """
    Computes the posterior probability P(Group 2 | x), denoted pi(x).
    
    Assumes P(Group 1) = P(Group 2) = 0.5 (Equal Priors).
    f1(x) = 1 (Uniform(0,1))
    f2(x) = 6 * x * (1 - x) (Beta(2,2))
    
    pi(x) = (0.5 * f2(x)) / (0.5 * f1(x) + 0.5 * f2(x))
    """
    # Ensure x is an array for element-wise operation
    x = np.asarray(x)
    f2 = 6.0 * x * (1.0 - x)
    
    numerator = 0.5 * f2
    denominator = 0.5 * 1.0 + numerator
    
    # Handle division by zero for denominator = 0 (should not happen in [0, 1] range)
    # The Beta density is 0 at 0 and 1, which makes pi(x) = 0 there.
    # To prevent division by zero in the phi calculation, we use a small epsilon for pi(x).
    epsilon = 1e-6
    pi_val = np.clip(numerator / denominator, epsilon, 1 - epsilon)
    
    return pi_val


def generate_and_plot_data():
    # Set seed for reproducibility
    np.random.seed(180)

    # Number of samples
    N = 200
    
    # Variance is given as 0.1, so std_dev is sqrt(0.1)
    variance = 0.1
    std_dev = np.sqrt(variance)
    
    # KRR Bandwidth parameter setup
    bandwidth = 0.8
    gamma_val = 1.0 / (2 * (bandwidth ** 2))

    # ---------------------------------------------------------
    # 1. Data Generation
    # ---------------------------------------------------------
    
    # Dataset 1 (Group 1): Y | X, X ~ Uniform(0, 1)
    X = np.random.uniform(0, 1, N)
    mean_Y = np.cos(4 * np.pi * X)
    Y = np.random.normal(loc=mean_Y, scale=std_dev)

    # Dataset 2 (Group 2): Z | X', X' ~ Beta(2, 2)
    X_prime = np.random.beta(2, 2, N)
    mean_Z = np.cos(4 * np.pi * X_prime) + 0.5 * (X_prime**2)
    Z = np.random.normal(loc=mean_Z, scale=std_dev)

    # ---------------------------------------------------------
    # 2. Kernel Ridge Regression Fitting (m_Y and m_Z)
    # ---------------------------------------------------------
    
    # KRR for Y|X
    param_grid = {'alpha': [1e-3, 1e-2, 0.1, 1, 10]}
    krr_Y = GridSearchCV(KernelRidge(kernel='rbf', gamma=gamma_val), param_grid, cv=5)
    krr_Y.fit(X.reshape(-1, 1), Y)

    # KRR for Z|X'
    param_grid = {'alpha': [1e-3, 1e-2, 0.1, 1, 10]}
    krr_Z = GridSearchCV(KernelRidge(kernel='rbf', gamma=gamma_val), param_grid, cv=5)
    krr_Z.fit(X_prime.reshape(-1, 1), Z)
    
    # Prepare grid for prediction
    x_grid = np.linspace(0, 1, 200).reshape(-1, 1)

    # KRR Predictions on Grid
    y_krr_pred = krr_Y.predict(x_grid)
    z_krr_pred = krr_Z.predict(x_grid)
    
    # KRR Predictions on Sampled Data Points
    m_Y_X = krr_Y.predict(X.reshape(-1, 1))
    m_Z_X = krr_Z.predict(X.reshape(-1, 1))
    
    m_Y_Xprime = krr_Y.predict(X_prime.reshape(-1, 1))
    m_Z_Xprime = krr_Z.predict(X_prime.reshape(-1, 1))
    
    # ---------------------------------------------------------
    # 3. Compute pi(x) and the target variable phi
    # ---------------------------------------------------------
    
    # Calculate pi(x) for the sampled points
    pi_X = pi_x(X)
    pi_Xprime = pi_x(X_prime)

    # Phi for Group 1 (X, Y)
    # phi = m_Z(X) - m_Y(X) - (1-pi(X))^{-1} * (Y - m_Y(X))
    phi_group1 = m_Z_X - m_Y_X - (1.0 / (1.0 - pi_X)) * (Y - m_Y_X)

    # Phi for Group 2 (X', Z)
    # phi = m_Z(X') - m_Y(X') + (pi(X'))^{-1} * (Z - m_Z(X'))
    phi_group2 = m_Z_Xprime - m_Y_Xprime + (1.0 / pi_Xprime) * (Z - m_Z_Xprime)

    # Concatenate data
    X_all = np.concatenate([X, X_prime])
    Phi_all = np.concatenate([phi_group1, phi_group2])

    # ---------------------------------------------------------
    # 4. KRR on the new dataset (X_all, Phi_all)
    # ---------------------------------------------------------
    param_grid = {'alpha': [1e-3, 1e-2, 0.1, 1, 10]}
    krr_phi = GridSearchCV(KernelRidge(kernel='rbf', gamma=gamma_val), param_grid, cv=5)
    krr_phi.fit(X_all.reshape(-1, 1), Phi_all)
    phi_krr_pred = krr_phi.predict(x_grid)
    
    # True Difference: 0.5 * x^2
    true_diff = 0.5 * (x_grid**2)

    # ---------------------------------------------------------
    # 5. Visualization (3 Separate Figures)
    # ---------------------------------------------------------
    
    # --- FIGURE 1: Scatterplots Y|X and Z|X' ---
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
    fig1.suptitle('Figure 1: Observed Data with KRR Mean Estimation', fontsize=16)

    # Plot 1.1: Y vs X
    axes1[0].scatter(X, Y, alpha=0.6, edgecolors='w', s=40, c='royalblue', label='Data')
    axes1[0].set_title(r'$Y$ vs $X$ (Uniform Prior)', fontsize=14)
    axes1[0].set_xlabel(r'$X$', fontsize=12)
    axes1[0].set_ylabel(r'$Y$', fontsize=12)
    axes1[0].grid(True, alpha=0.3)
    axes1[0].plot(x_grid, np.cos(4 * np.pi * x_grid), 'k--', alpha=0.5, label='True Mean $\mu_Y$')
    axes1[0].plot(x_grid, y_krr_pred, 'r-', linewidth=2.5, label=r'KRR Fit $\hat{m}_Y$')
    axes1[0].legend()

    # Plot 1.2: Z vs X'
    axes1[1].scatter(X_prime, Z, alpha=0.6, edgecolors='w', s=40, c='crimson', label='Data')
    axes1[1].set_title(r'$Z$ vs $X^{\prime}$ (Beta Prior)', fontsize=14)
    axes1[1].set_xlabel(r'$X^{\prime}$', fontsize=12)
    axes1[1].set_ylabel(r'$Z$', fontsize=12)
    axes1[1].grid(True, alpha=0.3)
    z_mean_grid = np.cos(4 * np.pi * x_grid) + 0.5 * (x_grid**2)
    axes1[1].plot(x_grid, z_mean_grid, 'k--', alpha=0.5, label='True Mean $\mu_Z$')
    axes1[1].plot(x_grid, z_krr_pred, 'r-', linewidth=2.5, label=r'KRR Fit $\hat{m}_Z$')
    axes1[1].legend()

    plt.tight_layout()
    plt.show()


    # --- FIGURE 2: Scatterplot of Phi|X ---
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    fig2.suptitle(r'Figure 2: Constructed $\phi$ Variable Scatterplot', fontsize=16)

    # Scatterplot of the Phi data
    ax2.scatter(X_all, Phi_all, alpha=0.4, edgecolors='none', s=25, c='purple', label=r'$\phi$ Data')
    
    # True Difference line
    ax2.plot(x_grid, true_diff, 'k--', alpha=0.6, label=r'True Difference ($\mu_Z - \mu_Y$)')
    
    ax2.set_title(r'$\phi$ Values vs $X$ (Combined Dataset)', fontsize=14)
    ax2.set_xlabel(r'$X / X^{\prime}$', fontsize=12)
    ax2.set_ylabel(r'$\phi$ Value (Target)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


    # --- FIGURE 3: Combined Comparison of Difference Estimates (Former Figures 2 & 4) ---
    fig3 = plt.figure(figsize=(9, 7))
    ax3 = fig3.add_subplot(111)
    fig3.suptitle('Figure 3: Comparison of Difference Estimators', fontsize=16)
    
    est_diff_naive = z_krr_pred - y_krr_pred
    est_diff_phi = phi_krr_pred

    ax3.plot(x_grid, true_diff, 'k--', alpha=0.8, linewidth=3.0, label=r'True Difference ($\mu_Z - \mu_Y = 0.5 x^2$)')
    ax3.plot(x_grid, est_diff_naive, 'b-', linewidth=2.0, label=r'Naive Est ($\hat{m}_Z - \hat{m}_Y$)')
    ax3.plot(x_grid, est_diff_phi, 'g-', linewidth=2.0, label=r'Double Robust Est ($\hat{\phi}$)')
    
    ax3.set_title(r'Comparing Naive vs $\phi$-based Estimation of the Difference', fontsize=14)
    ax3.set_xlabel(r'$x$', fontsize=12)
    ax3.set_ylabel(r'Difference Value', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_and_plot_data()