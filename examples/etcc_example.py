"""
Extended Triple Collocation with Correlated Errors (ETCC) Example
==================================================================

This example demonstrates the use of the ETCC method for comprehensive
error analysis when all products may have correlated errors.

ETCC is the most general 3-way collocation method:
- TC: Assumes all error correlations = 0
- EIVD: Allows only one error correlation pair
- ETCC: Allows all pairwise error correlations (most general)

Use ETCC when:
1. You suspect multiple products have correlated errors
2. Products use similar algorithms or data sources
3. You need the most accurate error characterization
4. You have sufficient temporal resolution for lag analysis

Author: Collocation Analysis Package
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collocation import tc, eivd, etcc, compare_tc_eivd_etcc

# ============================================================================
# Configuration
# ============================================================================

# Set random seed for reproducibility
np.random.seed(42)

# Data generation parameters
n_samples = 500

# True error correlation structure (what we want to recover)
true_error_corr = np.array([
    [1.00, 0.35, 0.25],
    [0.35, 1.00, 0.45],
    [0.25, 0.45, 1.00]
])

# True error standard deviations
true_error_std = np.array([0.5, 0.7, 0.6])

# ============================================================================
# Generate Synthetic Data with Correlated Errors
# ============================================================================

print("="*80)
print("EXTENDED TRIPLE COLLOCATION WITH CORRELATED ERRORS (ETCC) EXAMPLE")
print("="*80)

print("\n1. Generating synthetic data with correlated errors...")
print("-" * 80)

# Generate true signal with temporal autocorrelation
t = np.linspace(0, 10 * np.pi, n_samples)
true_signal = 10 + 3 * np.sin(t) + 2 * np.sin(0.5 * t) + \
              np.random.randn(n_samples) * 0.3

print(f"Number of samples: {n_samples}")
print(f"\nTrue error correlation matrix:")
print(true_error_corr)
print(f"\nTrue error standard deviations: {true_error_std}")

# Construct true error covariance matrix
true_error_cov = np.outer(true_error_std, true_error_std) * true_error_corr

# Generate correlated errors using Cholesky decomposition
L = np.linalg.cholesky(true_error_cov)
independent_noise = np.random.randn(3, n_samples)
correlated_errors = L @ independent_noise

# Generate three products
products = []
for i in range(3):
    product = true_signal + correlated_errors[i]
    products.append(product)
    print(f"Product {i+1}: σ_error = {true_error_std[i]:.2f}")

# Stack products
tri = np.column_stack(products)

# ============================================================================
# Compare TC, EIVD, and ETCC
# ============================================================================

print("\n2. Comparing TC, EIVD, and ETCC methods...")
print("-" * 80)

# Run comprehensive comparison
comparison = compare_tc_eivd_etcc(tri)

print(f"\nRecommendation: {comparison['recommendation']}")

# Display results for each method
print("\n" + "="*80)
print("METHOD COMPARISON")
print("="*80)

# TC Results
print("\n(A) Triple Collocation (TC) - Assumes ZERO error correlations")
print("-" * 80)
tc_results = comparison['tc']
print("Error variance matrix (diagonal only):")
print(np.diag(tc_results['EeeT']))
print(f"SNR: {tc_results['SNR']}")
print(f"Note: {tc_results['assumption']}")

# EIVD Results
print("\n(B) Extended IVD (EIVD) - Allows ONE error correlation")
print("-" * 80)
eivd_results = comparison['eivd']
print("Error covariance matrix:")
print(eivd_results['EeeT'])
print(f"Error correlation (products 2 & 3): {eivd_results['EeeT'][1,2]:.4f}")
print(f"SNR: {eivd_results['SNR']}")
print(f"Note: {eivd_results['assumption']}")

# ETCC Results
print("\n(C) Extended Triple Collocation (ETCC) - FULL error correlation")
print("-" * 80)
etcc_results = comparison['etcc']
print("Full error covariance matrix:")
print(etcc_results['EeeT'])
print(f"\nEstimated error correlation matrix:")
print(etcc_results['error_corr_matrix'])
print(f"\nTrue error correlation matrix (for comparison):")
print(true_error_corr)
print(f"\nSNR: {etcc_results['SNR']}")
print(f"Note: {etcc_results['assumption']}")

# Diagnostics
print("\nETCC Diagnostics:")
print(f"  Condition number: {etcc_results['diagnostics']['condition_number']:.2e}")
print(f"  Positive definite: {etcc_results['diagnostics']['is_positive_definite']}")
print(f"  Method used: {etcc_results['diagnostics']['method_used']}")

# Differences
print("\n" + "="*80)
print("DIFFERENCES BETWEEN METHODS")
print("="*80)
print(f"\nMax error correlation detected: {comparison['differences']['max_error_correlation']:.4f}")
print(f"\nSNR difference (TC vs ETCC): {comparison['differences']['tc_vs_etcc_snr']}")
print(f"SNR difference (EIVD vs ETCC): {comparison['differences']['eivd_vs_etcc_snr']}")

# ============================================================================
# Test Different ETCC Methods
# ============================================================================

print("\n3. Testing different ETCC estimation methods...")
print("-" * 80)

methods = ['full', 'sequential', 'constrained']
etcc_results_methods = {}

for method in methods:
    try:
        EeeT, SNR, rho2, fMSE, diag = etcc(tri, method=method)
        etcc_results_methods[method] = {
            'EeeT': EeeT,
            'SNR': SNR,
            'corr_matrix': diag['error_corr_matrix'],
            'cond_num': diag['condition_number']
        }
        print(f"\nMethod '{method}':")
        print(f"  Correlation matrix:\n{diag['error_corr_matrix']}")
        print(f"  Condition number: {diag['condition_number']:.2e}")
    except Exception as e:
        print(f"\nMethod '{method}' failed: {e}")

# ============================================================================
# Visualization
# ============================================================================

print("\n4. Creating visualizations...")

fig = plt.figure(figsize=(16, 10))

# Plot 1: Time series
ax1 = plt.subplot(3, 3, 1)
time = np.arange(n_samples)
ax1.plot(time[:200], true_signal[:200], 'k-', linewidth=2,
         label='True Signal', alpha=0.8, zorder=10)
for i, product in enumerate(products):
    ax1.plot(time[:200], product[:200], alpha=0.6, label=f'Product {i+1}')
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Value')
ax1.set_title('(a) Time Series (first 200 samples)')
ax1.legend(loc='best', framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Plot 2: True correlation matrix
ax2 = plt.subplot(3, 3, 2)
im2 = ax2.imshow(true_error_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax2.set_title('(b) True Error Correlation')
ax2.set_xlabel('Product')
ax2.set_ylabel('Product')
ax2.set_xticks([0, 1, 2])
ax2.set_yticks([0, 1, 2])
ax2.set_xticklabels(['P1', 'P2', 'P3'])
ax2.set_yticklabels(['P1', 'P2', 'P3'])
# Add text annotations
for i in range(3):
    for j in range(3):
        text = ax2.text(j, i, f'{true_error_corr[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=10)
plt.colorbar(im2, ax=ax2, label='Correlation')

# Plot 3: ETCC estimated correlation
ax3 = plt.subplot(3, 3, 3)
etcc_corr = etcc_results['error_corr_matrix']
im3 = ax3.imshow(etcc_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax3.set_title('(c) ETCC Estimated Correlation')
ax3.set_xlabel('Product')
ax3.set_ylabel('Product')
ax3.set_xticks([0, 1, 2])
ax3.set_yticks([0, 1, 2])
ax3.set_xticklabels(['P1', 'P2', 'P3'])
ax3.set_yticklabels(['P1', 'P2', 'P3'])
# Add text annotations
for i in range(3):
    for j in range(3):
        text = ax3.text(j, i, f'{etcc_corr[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=10)
plt.colorbar(im3, ax=ax3, label='Correlation')

# Plot 4: SNR comparison
ax4 = plt.subplot(3, 3, 4)
x_pos = np.arange(3)
width = 0.25

ax4.bar(x_pos - width, tc_results['SNR'], width, label='TC',
        color='steelblue', alpha=0.7)
ax4.bar(x_pos, eivd_results['SNR'], width, label='EIVD',
        color='orange', alpha=0.7)
ax4.bar(x_pos + width, etcc_results['SNR'], width, label='ETCC',
        color='green', alpha=0.7)

ax4.set_xlabel('Product')
ax4.set_ylabel('SNR')
ax4.set_title('(d) Signal-to-Noise Ratio Comparison')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(['P1', 'P2', 'P3'])
ax4.legend(loc='best', framealpha=0.9)
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Error variance comparison
ax5 = plt.subplot(3, 3, 5)
tc_var = np.diag(tc_results['EeeT'])
eivd_var = np.diag(eivd_results['EeeT'])
etcc_var = np.diag(etcc_results['EeeT'])
true_var = true_error_std ** 2

ax5.bar(x_pos - 1.5*width, true_var, width, label='True',
        color='gray', alpha=0.7)
ax5.bar(x_pos - 0.5*width, tc_var, width, label='TC',
        color='steelblue', alpha=0.7)
ax5.bar(x_pos + 0.5*width, eivd_var, width, label='EIVD',
        color='orange', alpha=0.7)
ax5.bar(x_pos + 1.5*width, etcc_var, width, label='ETCC',
        color='green', alpha=0.7)

ax5.set_xlabel('Product')
ax5.set_ylabel('Error Variance')
ax5.set_title('(e) Error Variance Estimates')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(['P1', 'P2', 'P3'])
ax5.legend(loc='best', framealpha=0.9)
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Error correlation comparison
ax6 = plt.subplot(3, 3, 6)
correlation_pairs = ['P1-P2', 'P1-P3', 'P2-P3']
true_corrs = [true_error_corr[0,1], true_error_corr[0,2], true_error_corr[1,2]]
tc_corrs = [0, 0, 0]  # TC assumes zero
eivd_corrs = [0, 0, eivd_results['EeeT'][1,2] / np.sqrt(eivd_results['EeeT'][1,1] * eivd_results['EeeT'][2,2])]
etcc_corrs = [etcc_corr[0,1], etcc_corr[0,2], etcc_corr[1,2]]

x_corr = np.arange(3)
ax6.plot(x_corr, true_corrs, 'ko-', linewidth=2, markersize=8,
         label='True', zorder=10)
ax6.plot(x_corr, tc_corrs, 's--', linewidth=2, markersize=6,
         label='TC', alpha=0.7)
ax6.plot(x_corr, eivd_corrs, '^--', linewidth=2, markersize=6,
         label='EIVD', alpha=0.7)
ax6.plot(x_corr, etcc_corrs, 'o-', linewidth=2, markersize=6,
         label='ETCC', alpha=0.7)

ax6.set_xlabel('Product Pair')
ax6.set_ylabel('Error Correlation')
ax6.set_title('(f) Error Correlation Estimates')
ax6.set_xticks(x_corr)
ax6.set_xticklabels(correlation_pairs)
ax6.legend(loc='best', framealpha=0.9)
ax6.grid(True, alpha=0.3)
ax6.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

# Plot 7: Scatterplot P1 vs P2
ax7 = plt.subplot(3, 3, 7)
ax7.scatter(products[0], products[1], alpha=0.3, s=10)
ax7.plot(true_signal, true_signal, 'r-', linewidth=2, alpha=0.5, label='1:1 line')
ax7.set_xlabel('Product 1')
ax7.set_ylabel('Product 2')
ax7.set_title(f'(g) P1 vs P2 (corr={np.corrcoef(products[0], products[1])[0,1]:.3f})')
ax7.legend(loc='best', framealpha=0.9)
ax7.grid(True, alpha=0.3)
ax7.axis('equal')

# Plot 8: Scatterplot P1 vs P3
ax8 = plt.subplot(3, 3, 8)
ax8.scatter(products[0], products[2], alpha=0.3, s=10)
ax8.plot(true_signal, true_signal, 'r-', linewidth=2, alpha=0.5, label='1:1 line')
ax8.set_xlabel('Product 1')
ax8.set_ylabel('Product 3')
ax8.set_title(f'(h) P1 vs P3 (corr={np.corrcoef(products[0], products[2])[0,1]:.3f})')
ax8.legend(loc='best', framealpha=0.9)
ax8.grid(True, alpha=0.3)
ax8.axis('equal')

# Plot 9: Scatterplot P2 vs P3
ax9 = plt.subplot(3, 3, 9)
ax9.scatter(products[1], products[2], alpha=0.3, s=10)
ax9.plot(true_signal, true_signal, 'r-', linewidth=2, alpha=0.5, label='1:1 line')
ax9.set_xlabel('Product 2')
ax9.set_ylabel('Product 3')
ax9.set_title(f'(i) P2 vs P3 (corr={np.corrcoef(products[1], products[2])[0,1]:.3f})')
ax9.legend(loc='best', framealpha=0.9)
ax9.grid(True, alpha=0.3)
ax9.axis('equal')

plt.tight_layout()

# Save figure
output_dir = os.path.join(os.path.dirname(__file__), '..', 'docs')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'etcc_example.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")

plt.show()

# ============================================================================
# Summary and Recommendations
# ============================================================================

print("\n" + "="*80)
print("SUMMARY AND RECOMMENDATIONS")
print("="*80)
print(f"""
Key Findings:
1. True error correlations ranged from {true_error_corr[0,2]:.2f} to {true_error_corr[1,2]:.2f}
2. TC (assumes zero correlations) had SNR errors of {np.mean(comparison['differences']['tc_vs_etcc_snr']):.3f}
3. EIVD (one correlation) captured P2-P3 correlation but missed P1-P2 and P1-P3
4. ETCC (full correlations) recovered all correlations within {np.mean(np.abs(etcc_corr - true_error_corr)):.3f} error

When to use each method:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TC (Triple Collocation):
  ✓ Products are truly independent (different sensors, algorithms)
  ✓ Fastest computation
  ✓ Simple interpretation
  ✗ May overestimate SNR if correlations exist

EIVD (Extended IVD):
  ✓ Two products suspected to have correlated errors
  ✓ Faster than ETCC
  ✓ Good for similar sensors/algorithms (2 out of 3)
  ✗ Cannot capture multiple correlation pairs

ETCC (Extended Triple Collocation):
  ✓ Multiple products may have correlated errors
  ✓ Most accurate error characterization
  ✓ Products use related data or algorithms
  ✓ Comprehensive diagnostics
  ✗ Requires more data (temporal resolution)
  ✗ More complex computation

Recommendation for your data:
{comparison['recommendation']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Best Practices:
1. Always compare TC, EIVD, and ETCC results
2. Use compare_tc_eivd_etcc() for automatic method selection
3. Check ETCC diagnostics (condition number, positive-definiteness)
4. Verify results make physical sense for your application
5. Use method='constrained' if you get negative variance estimates
6. Ensure sufficient temporal resolution for lag-2 analysis
""")

print("="*80)
print("Example completed successfully!")
print("="*80)
