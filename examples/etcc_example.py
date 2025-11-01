"""
Comprehensive Example: ETCC (Extended Triple Collocation for Correlation)

This script demonstrates the use of ETCC for merging three precipitation products
with the goal of maximizing correlation with unknown truth.

Compares:
1. Traditional Triple Collocation (TC) - minimizes RMSE
2. Extended Triple Collocation for Correlation (ETCC) - maximizes correlation

Based on: Wei et al. (2023), Geophysical Research Letters
"""

import numpy as np
import matplotlib.pyplot as plt
from collocation import ETCC, TripleCollocation
from collocation.utils import calculate_all_metrics

# Set random seed for reproducibility
np.random.seed(42)

def generate_precipitation_data(n_samples=500):
    """
    Generate synthetic precipitation data with known truth.

    Simulates three precipitation products with different error characteristics:
    - Product 1: Low bias, moderate random error
    - Product 2: Higher bias, lower random error
    - Product 3: Moderate bias, higher random error

    Returns:
        truth, product1, product2, product3
    """
    # True precipitation (log-normal distribution, typical for precipitation)
    truth = np.random.lognormal(mean=1.5, sigma=0.8, size=n_samples)

    # Add realistic errors to each product
    # Product 1: Multiplicative bias 0.95, additive noise std=1.5
    product1 = truth * 0.95 + np.random.normal(0, 1.5, n_samples)

    # Product 2: Multiplicative bias 1.10, additive noise std=1.0
    product2 = truth * 1.10 + np.random.normal(0, 1.0, n_samples)

    # Product 3: Multiplicative bias 1.05, additive noise std=2.0
    product3 = truth * 1.05 + np.random.normal(0, 2.0, n_samples)

    # Ensure non-negative (precipitation cannot be negative)
    product1 = np.maximum(product1, 0)
    product2 = np.maximum(product2, 0)
    product3 = np.maximum(product3, 0)
    truth = np.maximum(truth, 0)

    return truth, product1, product2, product3


def compare_tc_etcc(truth, product1, product2, product3):
    """
    Compare Traditional TC and ETCC merging methods.

    Args:
        truth: True precipitation values
        product1, product2, product3: Three precipitation products

    Returns:
        Dictionary with results
    """
    print("=" * 80)
    print("COMPARISON: Traditional TC vs ETCC")
    print("=" * 80)

    # Traditional Triple Collocation (minimizes RMSE)
    print("\n1. Traditional Triple Collocation (TC)")
    print("-" * 80)
    tc = TripleCollocation()
    merged_tc = tc.merge(product1, product2, product3)

    print(f"\nTC Optimal Weights:")
    print(f"  Product 1: {tc.weights['wx']:.4f}")
    print(f"  Product 2: {tc.weights['wy']:.4f}")
    print(f"  Product 3: {tc.weights['wz']:.4f}")
    print(f"  Sum: {tc.weights['wx'] + tc.weights['wy'] + tc.weights['wz']:.4f}")

    print(f"\nTC Estimated Error Variances (RMSE²):")
    print(f"  Product 1: {tc.error_variances['sigma2_x']:.4f} (RMSE: {np.sqrt(tc.error_variances['sigma2_x']):.4f})")
    print(f"  Product 2: {tc.error_variances['sigma2_y']:.4f} (RMSE: {np.sqrt(tc.error_variances['sigma2_y']):.4f})")
    print(f"  Product 3: {tc.error_variances['sigma2_z']:.4f} (RMSE: {np.sqrt(tc.error_variances['sigma2_z']):.4f})")

    # Calculate metrics against truth
    tc_metrics = calculate_all_metrics(truth, merged_tc)
    print(f"\nTC Performance vs Truth:")
    print(f"  Correlation: {tc_metrics['r']:.4f}")
    print(f"  RMSE: {tc_metrics['rmse']:.4f}")
    print(f"  MAE: {tc_metrics['mae']:.4f}")
    print(f"  KGE: {tc_metrics['kge']:.4f}")
    print(f"  NSE: {tc_metrics['nse']:.4f}")
    print(f"  PBIAS: {tc_metrics['pbias']:.2f}%")

    # ETCC (maximizes correlation)
    print("\n2. Extended Triple Collocation for Correlation (ETCC)")
    print("-" * 80)
    etcc = ETCC(weight_increment=0.01, min_correlation=0.01)
    merged_etcc = etcc.merge(product1, product2, product3)

    print(f"\nETCC Optimal Weights:")
    print(f"  Product 1: {etcc.weights['wx']:.4f}")
    print(f"  Product 2: {etcc.weights['wy']:.4f}")
    print(f"  Product 3: {etcc.weights['wz']:.4f}")
    print(f"  Sum: {etcc.weights['wx'] + etcc.weights['wy'] + etcc.weights['wz']:.4f}")

    print(f"\nETCC Estimated Correlations with Truth:")
    print(f"  Product 1: {etcc.correlation_with_truth['rho_Rx']:.4f}")
    print(f"  Product 2: {etcc.correlation_with_truth['rho_Ry']:.4f}")
    print(f"  Product 3: {etcc.correlation_with_truth['rho_Rz']:.4f}")
    print(f"\nETCC Maximum Correlation: {etcc.max_correlation:.4f}")

    # Calculate metrics against truth
    etcc_metrics = calculate_all_metrics(truth, merged_etcc)
    print(f"\nETCC Performance vs Truth:")
    print(f"  Correlation: {etcc_metrics['r']:.4f}")
    print(f"  RMSE: {etcc_metrics['rmse']:.4f}")
    print(f"  MAE: {etcc_metrics['mae']:.4f}")
    print(f"  KGE: {etcc_metrics['kge']:.4f}")
    print(f"  NSE: {etcc_metrics['nse']:.4f}")
    print(f"  PBIAS: {etcc_metrics['pbias']:.2f}%")

    # Individual product metrics
    print("\n3. Individual Product Performance vs Truth")
    print("-" * 80)
    for i, product in enumerate([product1, product2, product3], 1):
        metrics = calculate_all_metrics(truth, product)
        print(f"\nProduct {i}:")
        print(f"  Correlation: {metrics['r']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")

    # Summary comparison
    print("\n4. Summary Comparison")
    print("=" * 80)
    print(f"{'Method':<20} {'Correlation':<15} {'RMSE':<15} {'KGE':<15}")
    print("-" * 80)

    for i, product in enumerate([product1, product2, product3], 1):
        m = calculate_all_metrics(truth, product)
        print(f"Product {i:<14} {m['r']:<15.4f} {m['rmse']:<15.4f} {m['kge']:<15.4f}")

    print(f"{'TC Merged':<20} {tc_metrics['r']:<15.4f} {tc_metrics['rmse']:<15.4f} {tc_metrics['kge']:<15.4f}")
    print(f"{'ETCC Merged':<20} {etcc_metrics['r']:<15.4f} {etcc_metrics['rmse']:<15.4f} {etcc_metrics['kge']:<15.4f}")
    print("-" * 80)

    # Improvement analysis
    best_product_corr = max([calculate_all_metrics(truth, p)['r'] for p in [product1, product2, product3]])
    best_product_rmse = min([calculate_all_metrics(truth, p)['rmse'] for p in [product1, product2, product3]])

    print(f"\nImprovement over best individual product:")
    print(f"  TC Correlation improvement: {(tc_metrics['r'] - best_product_corr):.4f}")
    print(f"  TC RMSE improvement: {(best_product_rmse - tc_metrics['rmse']):.4f}")
    print(f"  ETCC Correlation improvement: {(etcc_metrics['r'] - best_product_corr):.4f}")
    print(f"  ETCC RMSE improvement: {(best_product_rmse - etcc_metrics['rmse']):.4f}")

    print(f"\nETCC vs TC:")
    print(f"  Correlation difference: {(etcc_metrics['r'] - tc_metrics['r']):.4f}")
    print(f"  RMSE difference: {(etcc_metrics['rmse'] - tc_metrics['rmse']):.4f}")
    print(f"  {'ETCC has better correlation' if etcc_metrics['r'] > tc_metrics['r'] else 'TC has better correlation'}")
    print(f"  {'ETCC has lower RMSE' if etcc_metrics['rmse'] < tc_metrics['rmse'] else 'TC has lower RMSE'}")

    return {
        'tc': {'merged': merged_tc, 'metrics': tc_metrics, 'weights': tc.weights},
        'etcc': {'merged': merged_etcc, 'metrics': etcc_metrics, 'weights': etcc.weights},
        'truth': truth,
        'products': [product1, product2, product3]
    }


def create_visualizations(results):
    """
    Create comprehensive visualization comparing TC and ETCC.

    Args:
        results: Dictionary from compare_tc_etcc
    """
    truth = results['truth']
    tc_merged = results['tc']['merged']
    etcc_merged = results['etcc']['merged']
    products = results['products']

    fig = plt.figure(figsize=(16, 10))

    # 1. Time series comparison
    ax1 = plt.subplot(3, 3, 1)
    n_display = min(100, len(truth))
    ax1.plot(truth[:n_display], 'k-', linewidth=2, label='Truth', alpha=0.7)
    ax1.plot(tc_merged[:n_display], 'b--', linewidth=1.5, label='TC Merged', alpha=0.7)
    ax1.plot(etcc_merged[:n_display], 'r--', linewidth=1.5, label='ETCC Merged', alpha=0.7)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Precipitation (mm)')
    ax1.set_title('Time Series Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Scatter: TC vs Truth
    ax2 = plt.subplot(3, 3, 2)
    ax2.scatter(truth, tc_merged, alpha=0.5, s=20, c='blue', edgecolors='none')
    max_val = max(truth.max(), tc_merged.max())
    ax2.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Truth (mm)')
    ax2.set_ylabel('TC Merged (mm)')
    ax2.set_title(f'TC: R={results["tc"]["metrics"]["r"]:.3f}, RMSE={results["tc"]["metrics"]["rmse"]:.2f}')
    ax2.grid(True, alpha=0.3)

    # 3. Scatter: ETCC vs Truth
    ax3 = plt.subplot(3, 3, 3)
    ax3.scatter(truth, etcc_merged, alpha=0.5, s=20, c='red', edgecolors='none')
    ax3.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Truth (mm)')
    ax3.set_ylabel('ETCC Merged (mm)')
    ax3.set_title(f'ETCC: R={results["etcc"]["metrics"]["r"]:.3f}, RMSE={results["etcc"]["metrics"]["rmse"]:.2f}')
    ax3.grid(True, alpha=0.3)

    # 4-6. Individual products
    for i, product in enumerate(products, 1):
        ax = plt.subplot(3, 3, 3 + i)
        metrics = calculate_all_metrics(truth, product)
        ax.scatter(truth, product, alpha=0.5, s=20, edgecolors='none')
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Truth (mm)')
        ax.set_ylabel(f'Product {i} (mm)')
        ax.set_title(f'Product {i}: R={metrics["r"]:.3f}, RMSE={metrics["rmse"]:.2f}')
        ax.grid(True, alpha=0.3)

    # 7. Weights comparison
    ax7 = plt.subplot(3, 3, 7)
    x_pos = np.arange(3)
    width = 0.35
    tc_weights = [results['tc']['weights']['wx'], results['tc']['weights']['wy'], results['tc']['weights']['wz']]
    etcc_weights = [results['etcc']['weights']['wx'], results['etcc']['weights']['wy'], results['etcc']['weights']['wz']]

    ax7.bar(x_pos - width/2, tc_weights, width, label='TC', color='blue', alpha=0.7)
    ax7.bar(x_pos + width/2, etcc_weights, width, label='ETCC', color='red', alpha=0.7)
    ax7.set_xlabel('Product')
    ax7.set_ylabel('Weight')
    ax7.set_title('Optimal Weights Comparison')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(['Product 1', 'Product 2', 'Product 3'])
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')

    # 8. Performance metrics comparison
    ax8 = plt.subplot(3, 3, 8)
    metrics_names = ['r', 'kge', 'nse']
    tc_vals = [results['tc']['metrics'][m] for m in metrics_names]
    etcc_vals = [results['etcc']['metrics'][m] for m in metrics_names]

    x_pos = np.arange(len(metrics_names))
    ax8.bar(x_pos - width/2, tc_vals, width, label='TC', color='blue', alpha=0.7)
    ax8.bar(x_pos + width/2, etcc_vals, width, label='ETCC', color='red', alpha=0.7)
    ax8.set_xlabel('Metric')
    ax8.set_ylabel('Value')
    ax8.set_title('Performance Metrics Comparison')
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(['Correlation', 'KGE', 'NSE'])
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    ax8.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # 9. Error distribution
    ax9 = plt.subplot(3, 3, 9)
    tc_errors = tc_merged - truth
    etcc_errors = etcc_merged - truth

    ax9.hist(tc_errors, bins=30, alpha=0.5, label='TC Errors', color='blue', density=True)
    ax9.hist(etcc_errors, bins=30, alpha=0.5, label='ETCC Errors', color='red', density=True)
    ax9.axvline(x=0, color='k', linestyle='--', linewidth=1)
    ax9.set_xlabel('Error (Merged - Truth)')
    ax9.set_ylabel('Density')
    ax9.set_title('Error Distribution')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('etcc_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'etcc_comparison.png'")
    plt.show()


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("ETCC DEMONSTRATION: Extended Triple Collocation for Correlation")
    print("=" * 80)
    print("\nThis example compares Traditional TC (minimizes RMSE) with")
    print("ETCC (maximizes correlation) for precipitation data merging.")
    print("\nReference: Wei et al. (2023), Geophysical Research Letters")

    # Generate synthetic data
    print("\nGenerating synthetic precipitation data...")
    truth, product1, product2, product3 = generate_precipitation_data(n_samples=500)
    print(f"  Generated {len(truth)} samples")
    print(f"  Truth mean: {truth.mean():.2f} mm, std: {truth.std():.2f} mm")

    # Compare methods
    results = compare_tc_etcc(truth, product1, product2, product3)

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(results)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. TC optimizes for minimum RMSE (error variance)")
    print("2. ETCC optimizes for maximum correlation with truth")
    print("3. Different weights reflect different optimization objectives")
    print("4. Choice depends on application requirements:")
    print("   - Use TC when RMSE minimization is critical")
    print("   - Use ETCC when correlation maximization is more important")
    print("\nFor precipitation applications, ETCC often provides better")
    print("correlation with observed patterns, even if RMSE is slightly higher.")


if __name__ == '__main__':
    main()
