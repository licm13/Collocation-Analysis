"""
Bayesian Three-Cornered Hat (BTCH) Example
==========================================

This example demonstrates the use of Bayesian Three-Cornered Hat (BTCH)
method for uncertainty quantification in collocation analysis.

BTCH is a simpler alternative to BTC (Bayesian Triple Collocation):
- Assumes constant error variances (homoscedastic)
- Provides full Bayesian uncertainty quantification
- Faster than time-varying BTC
- Suitable when time-varying errors are not expected

Comparison with other methods:
- TC/TCH: Classical methods, no uncertainty quantification
- BTC: Complex time-varying errors, full uncertainty (slower)
- BTCH: Constant errors, full uncertainty (faster)

Author: Converted from MATLAB and enhanced with Python features
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collocation import BAYESIAN_TCH_AVAILABLE

from collocation import tc

# If PyMC3 is unavailable, we'll fall back to a non-Bayesian bootstrap
# approximation so the example can still run and illustrate results.
USE_BAYES = bool(BAYESIAN_TCH_AVAILABLE)
if not USE_BAYES:
    print("=" * 70)
    print("WARNING: PyMC3 not available — running non-Bayesian bootstrap fallback.")
    print("To enable full Bayesian inference, install PyMC3:")
    print("  pip install pymc3==3.11.5 theano-pymc")
    print("=" * 70)

try:
    from collocation import BayesianTCH
except Exception:
    BayesianTCH = None

# ============================================================================
# Configuration
# ============================================================================

# Set random seed for reproducibility
np.random.seed(42)

# Data generation parameters
n_samples = 500
true_rmse = [0.5, 0.7, 0.9]  # True RMSE for each product

# Bayesian inference parameters
n_mcmc_iter = 2000        # MCMC iterations
n_advi_iter = 50000       # ADVI iterations for initialization
n_chains = 2              # Number of MCMC chains

# ============================================================================
# Generate Synthetic Data
# ============================================================================

print("="*70)
print("BAYESIAN THREE-CORNERED HAT (BTCH) EXAMPLE")
print("="*70)

print("\n1. Generating synthetic data...")
print("-" * 70)
print(f"Number of samples: {n_samples}")
print(f"True RMSE values: {true_rmse}")

# Generate true signal (unknown in real applications)
true_signal = 10 + 3 * np.sin(np.linspace(0, 4*np.pi, n_samples)) + \
              np.random.randn(n_samples) * 0.3

# Generate three products with known error characteristics
products = []
for i, rmse in enumerate(true_rmse):
    # Add Gaussian noise with specified RMSE
    error = np.random.randn(n_samples) * rmse
    product = true_signal + error
    products.append(product)
    print(f"Product {i+1}: RMSE = {rmse:.2f}")

# Stack products (shape: 3 x n_samples)
data = np.array(products)

# ============================================================================
# Classical Triple Collocation (for comparison)
# ============================================================================

print("\n2. Running Classical Triple Collocation (TC)...")
print("-" * 70)

tri = np.column_stack(products)
EeeT_tc, SNR_tc, rho2_tc, fMSE_tc = tc(tri)
rmse_tc = np.sqrt(np.diag(EeeT_tc))

print("Classical TC Results (no uncertainty quantification):")
for i, rmse in enumerate(rmse_tc):
    print(f"  Product {i+1}: RMSE = {rmse:.4f} (True: {true_rmse[i]:.2f})")

# ============================================================================
# Bayesian Three-Cornered Hat
# ============================================================================

print("\n3. Running Bayesian Three-Cornered Hat (BTCH) or fallback...")
print("-" * 70)

if USE_BAYES and BayesianTCH is not None:
    print("This will take a few minutes...")
    print(f"MCMC iterations: {n_mcmc_iter}")
    print(f"ADVI iterations: {n_advi_iter}")
    print(f"Chains: {n_chains}")

    # Initialize BTCH
    btch = BayesianTCH(data)

    # Run inference
    trace = btch.run_inference(
        niter=n_mcmc_iter,
        nadvi=n_advi_iter,
        nchains=n_chains,
        seed=42,
    )

    # Get results
    rmse_mean, rmse_std, rmse_quantiles = btch.get_error_estimates()

    # Print summary
    btch.summary()
    posterior_samples = None
else:
    # Non-Bayesian fallback: estimate uncertainty using simple bootstrap of TC
    print("PyMC3 not available — using bootstrap approximation for uncertainty (non-Bayesian).")
    B = 1000  # bootstrap replicates
    rng = np.random.RandomState(42)
    boot_rmse = np.zeros((B, data.shape[0]))
    tri = np.column_stack(products)
    n = tri.shape[0]
    print(f"Running {B} bootstrap replicates...")
    for b in range(B):
        idx = rng.randint(0, n, size=n)
        sample = tri[idx, :]
        try:
            EeeT_b, _, _, _ = tc(sample)
            rmse_b = np.sqrt(np.diag(EeeT_b))
        except Exception:
            # Fallback to sample RMSE if tc fails
            rmse_b = np.sqrt(np.mean((sample - sample.mean(axis=0)) ** 2, axis=0))
        boot_rmse[b, :] = rmse_b

    rmse_mean = boot_rmse.mean(axis=0)
    rmse_std = boot_rmse.std(axis=0)
    rmse_quantiles = np.percentile(boot_rmse, [2.5, 50, 97.5], axis=0).T
    posterior_samples = boot_rmse

# ============================================================================
# Compare Results
# ============================================================================

print("\n4. Comparison of Results")
print("="*70)
print(f"{'Method':<10} {'Product':<10} {'RMSE':<15} {'True RMSE':<12} {'Error':<10}")
print("-"*70)

for i in range(3):
    # Classical TC
    tc_error = abs(rmse_tc[i] - true_rmse[i])
    print(f"{'TC':<10} {'Product ' + str(i+1):<10} {rmse_tc[i]:<15.4f} {true_rmse[i]:<12.2f} {tc_error:<10.4f}")

for i in range(3):
    # Bayesian TCH
    btch_error = abs(rmse_mean[i] - true_rmse[i])
    ci_str = f"[{rmse_quantiles[i,0]:.4f}, {rmse_quantiles[i,2]:.4f}]"
    print(f"{'BTCH':<10} {'Product ' + str(i+1):<10} {rmse_mean[i]:.4f}±{rmse_std[i]:.4f} {true_rmse[i]:<12.2f} {btch_error:<10.4f}")
    print(f"{'  95% CI:':<10} {'':<10} {ci_str}")

# ============================================================================
# Visualization
# ============================================================================

print("\n5. Creating visualizations...")

fig = plt.figure(figsize=(12, 8))

# Plot 1: True signal and products
ax1 = plt.subplot(2, 2, 1)
time = np.arange(n_samples)
ax1.plot(time[:100], true_signal[:100], 'k-', linewidth=2, label='True Signal', alpha=0.7)
for i, product in enumerate(products):
    ax1.plot(time[:100], product[:100], alpha=0.6, label=f'Product {i+1}')
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Value')
ax1.set_title('(a) Time Series (first 100 samples)')
ax1.legend(loc='best', framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Plot 2: RMSE comparison
ax2 = plt.subplot(2, 2, 2)
x_pos = np.arange(3)
width = 0.35

# True RMSE
ax2.bar(x_pos - width/2, true_rmse, width, label='True RMSE',
        color='gray', alpha=0.7)

# Classical TC
ax2.bar(x_pos + width/2, rmse_tc, width, label='Classical TC',
        color='steelblue', alpha=0.7)

# BTCH with error bars
ax2.errorbar(x_pos, rmse_mean, yerr=rmse_std, fmt='ro',
             markersize=8, capsize=5, label='BTCH (mean±std)',
             linewidth=2, zorder=10)

ax2.set_xlabel('Product')
ax2.set_ylabel('RMSE')
ax2.set_title('(b) RMSE Estimates Comparison')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'P{i+1}' for i in range(3)])
ax2.legend(loc='best', framealpha=0.9)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Posterior distributions
ax3 = plt.subplot(2, 2, 3)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i in range(3):
    # Get posterior samples
    if USE_BAYES and BayesianTCH is not None:
        samples = trace.get_values('sigmap')[:, i]
    else:
        # use bootstrap samples as a proxy for posterior
        samples = posterior_samples[:, i]

    # Plot histogram
    ax3.hist(samples, bins=50, alpha=0.5, color=colors[i],
             label=f'Product {i+1}', density=True)

    # Plot true value
    ax3.axvline(true_rmse[i], color=colors[i], linestyle='--',
                linewidth=2, alpha=0.8)

ax3.set_xlabel('RMSE')
ax3.set_ylabel('Posterior Density')
ax3.set_title('(c) Posterior Distributions (dashed = true)')
ax3.legend(loc='best', framealpha=0.9)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Credible intervals
ax4 = plt.subplot(2, 2, 4)

for i in range(3):
    # 95% credible interval
    lower, median, upper = rmse_quantiles[i]

    # Plot interval
    ax4.plot([i, i], [lower, upper], 'o-', color=colors[i],
             linewidth=3, markersize=8, label=f'Product {i+1}')

    # Plot median
    ax4.plot(i, median, 's', color='white', markersize=6,
             markeredgecolor=colors[i], markeredgewidth=2, zorder=10)

    # Plot true value
    ax4.plot(i, true_rmse[i], '*', color='red', markersize=12,
             markeredgecolor='black', markeredgewidth=1, zorder=15)

ax4.set_xlabel('Product')
ax4.set_ylabel('RMSE')
ax4.set_title('(d) 95% Credible Intervals (red star = true)')
ax4.set_xticks(range(3))
ax4.set_xticklabels([f'P{i+1}' for i in range(3)])
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Save figure(s) in a `figures/` folder next to this script.
# Filename is derived from the script name so it relates to the example file.
script_dir = os.path.dirname(os.path.abspath(__file__))
script_base = os.path.splitext(os.path.basename(__file__))[0]
output_dir = os.path.join(script_dir, 'figures')
os.makedirs(output_dir, exist_ok=True)
output_filename = f"{script_base}.png"
output_path = os.path.join(output_dir, output_filename)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")

plt.show()

# ============================================================================
# Summary and Recommendations
# ============================================================================

print("\n" + "="*70)
print("SUMMARY AND RECOMMENDATIONS")
print("="*70)
print("""
Key Advantages of BTCH:
1. Full uncertainty quantification (credible intervals)
2. More reliable than point estimates (TC/TCH)
3. Faster than time-varying BTC
4. Suitable for constant error variance scenarios

When to use BTCH:
- You need uncertainty estimates (not just point estimates)
- Error variances are approximately constant over time
- You have sufficient data (n > 100 recommended)
- You can afford MCMC computation (~minutes)

When to use alternatives:
- TC/TCH: Very fast, no uncertainty needed
- BTC: Time-varying errors, complex heteroscedasticity
- EC: Four or more products available

Best practices:
1. Check MCMC convergence (trace plots)
2. Use at least 1000 MCMC samples
3. Run multiple chains (2-4) to verify consistency
4. Validate with simulated data first
""")

print("="*70)
print("Example completed successfully!")
print("="*70)
