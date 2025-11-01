"""
Comprehensive Example: All Collocation Methods
===============================================

This example demonstrates all collocation analysis methods available
in the package using synthetic data with known error characteristics.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from collocation import ivd, ivs, tc, eivd, ec
from collocation.utils import calculate_all_metrics

# Set random seed for reproducibility
np.random.seed(42)


def generate_synthetic_data(n_samples=200):
    """
    Generate synthetic data with known error characteristics.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate

    Returns
    -------
    data_dict : dict
        Dictionary containing true signal and four products
    """
    # True signal
    true_signal = 10 + 2 * np.sin(np.linspace(0, 4 * np.pi, n_samples))
    true_signal += 0.5 * np.random.randn(n_samples)  # Add some variability

    # Product 1: Low error, independent
    error1 = 0.3 * np.random.randn(n_samples)
    product1 = true_signal + error1

    # Product 2: Medium error, independent
    error2 = 0.5 * np.random.randn(n_samples)
    product2 = true_signal + error2

    # Product 3: Medium error, partially correlated with product 2
    common_error = 0.2 * np.random.randn(n_samples)
    error3 = common_error + 0.4 * np.random.randn(n_samples)
    product3 = true_signal + error3

    # Product 4: High error, independent
    error4 = 0.7 * np.random.randn(n_samples)
    product4 = true_signal + error4

    return {
        'true_signal': true_signal,
        'product1': product1,
        'product2': product2,
        'product3': product3,
        'product4': product4,
        'error1': error1,
        'error2': error2,
        'error3': error3,
        'error4': error4
    }


def example_ivd():
    """Demonstrate IVD method with two products."""
    print("\n" + "=" * 70)
    print("IVD (Information Vector Dual) - Two Products")
    print("=" * 70)

    data = generate_synthetic_data()
    dual = np.column_stack([data['product1'], data['product2']])

    # Apply IVD
    EeeT, rho2, u = ivd(dual)

    print("\nError Covariance Matrix:")
    print(EeeT)
    print(f"\nTrue error variances: [{np.var(data['error1']):.4f}, {np.var(data['error2']):.4f}]")
    print(f"IVD error variances:  [{EeeT[0,0]:.4f}, {EeeT[1,1]:.4f}]")

    print(f"\nData-truth correlations: {rho2}")
    print(f"Merging weights: {u}")
    print(f"Sum of weights: {np.sum(u):.4f}")

    # Merge products
    merged = u[0] * data['product1'] + u[1] * data['product2']
    metrics = calculate_all_metrics(merged, data['true_signal'])
    print(f"\nMerged product performance:")
    print(f"  Correlation: {metrics['r']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")

    return merged


def example_ivs():
    """Demonstrate IVS method with bootstrap resampling."""
    print("\n" + "=" * 70)
    print("IVS (Information Vector with Scaling) - Bootstrap Resampling")
    print("=" * 70)

    data = generate_synthetic_data()
    X = np.column_stack([data['product1'], data['product2']])

    # Apply IVS (additive mode)
    print("\nRunning IVS with 500 bootstrap samples...")
    RMSE, rho2 = ivs(X, N_boot=500, column=1, M_A=0)

    print(f"\nRMSE estimates: {RMSE}")
    print(f"True RMSE: [{np.std(data['error1']):.4f}, {np.std(data['error2']):.4f}]")

    print(f"\nData-truth correlations: {rho2}")

    # Compare with multiplicative mode
    print("\n\nMultiplicative mode (for positive data):")
    X_pos = np.column_stack([
        np.abs(data['product1']) + 1.0,
        np.abs(data['product2']) + 1.0
    ])
    RMSE_mult, rho2_mult = ivs(X_pos, N_boot=500, column=1, M_A=1)
    print(f"RMSE estimates: {RMSE_mult}")
    print(f"Correlations: {rho2_mult}")


def example_tc():
    """Demonstrate Triple Collocation method."""
    print("\n" + "=" * 70)
    print("TC (Triple Collocation) - Three Products")
    print("=" * 70)

    data = generate_synthetic_data()
    tri = np.column_stack([data['product1'], data['product2'], data['product4']])

    # Apply TC
    EeeT, SNR, rho2, fMSE = tc(tri)

    print("\nError Covariance Matrix:")
    print(EeeT)

    print(f"\nTrue error variances: [{np.var(data['error1']):.4f}, "
          f"{np.var(data['error2']):.4f}, {np.var(data['error4']):.4f}]")
    print(f"TC error variances:   [{EeeT[0,0]:.4f}, {EeeT[1,1]:.4f}, {EeeT[2,2]:.4f}]")

    print(f"\nSignal-to-Noise Ratio: {SNR}")
    print(f"Data-truth correlation: {rho2}")
    print(f"Fractional MSE: {fMSE}")

    # Rank products by quality
    from collocation.tc import rank_products
    ranking = rank_products(tri)
    print(f"\nProduct quality ranking (best to worst): {ranking}")


def example_eivd():
    """Demonstrate EIVD method with error correlation."""
    print("\n" + "=" * 70)
    print("EIVD (Extended IVD) - Three Products with Error Correlation")
    print("=" * 70)

    data = generate_synthetic_data()
    # Use products 1, 2, 3 where 2 and 3 have correlated errors
    tri = np.column_stack([data['product1'], data['product2'], data['product3']])

    # Apply EIVD
    EeeT, SNR, rho2, fMSE, L = eivd(tri)

    print("\nError Covariance Matrix (with cross-correlation):")
    print(EeeT)

    print(f"\nError cross-correlation between products 2 and 3: {EeeT[1,2]:.4f}")
    print(f"True error correlation: {np.cov(data['error2'], data['error3'])[0,1]:.4f}")

    print(f"\nSignal-to-Noise Ratio: {SNR}")
    print(f"Data-truth correlation: {rho2}")
    print(f"Lag-1 autocorrelation: {L}")

    # Compare TC vs EIVD
    print("\n\nComparing TC and EIVD:")
    from collocation.eivd import compare_tc_eivd
    comparison = compare_tc_eivd(tri)
    print(f"Recommendation: {comparison['recommendation']}")
    print(f"SNR difference: {comparison['snr_diff']}")


def example_ec():
    """Demonstrate Extended Collocation (Quadruple) method."""
    print("\n" + "=" * 70)
    print("EC (Extended Collocation) - Four Products")
    print("=" * 70)

    data = generate_synthetic_data()
    qu = np.column_stack([
        data['product1'],
        data['product2'],
        data['product3'],
        data['product4']
    ])

    # Apply EC
    print("\nRunning Extended Collocation on all 6 combinations...")
    results = ec(qu)

    print(f"\nNumber of valid combinations: {len(results)}")

    # Select best combination
    from collocation.ec import select_best_combination
    best = select_best_combination(results, criterion='min_fMSE')
    print(f"\nBest combination: {best.combination}")
    print(f"Products used: {list(best.combination)}")

    # Show results from first reference scenario
    print("\nResults from best combination, first reference scenario:")
    print(f"Error covariance matrix:")
    print(best.EeeT[0])
    print(f"\nError variances: {np.diag(best.EeeT[0])}")
    print(f"Error cross-correlation: {best.EeeT[0][0,1]:.4f}")

    print(f"\nSignal-to-Noise Ratio: {best.SNR[0]}")
    print(f"Data-truth correlation: {best.rho2[0]}")
    print(f"Optimal weights: {best.re_weight[0]}")

    # Evaluate merged result
    merged = best.weighted_result[0]
    valid_idx = ~np.isnan(merged)
    metrics = calculate_all_metrics(
        merged[valid_idx],
        data['true_signal'][valid_idx]
    )
    print(f"\nMerged product performance:")
    print(f"  Correlation: {metrics['r']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")

    return merged


def plot_results(data, merged_ivd, merged_ec):
    """Create visualization of results."""
    print("\n" + "=" * 70)
    print("Creating visualization...")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: All products vs true signal
    ax = axes[0, 0]
    ax.plot(data['true_signal'], 'k-', label='True Signal', linewidth=2)
    ax.plot(data['product1'], 'r-', alpha=0.6, label='Product 1')
    ax.plot(data['product2'], 'g-', alpha=0.6, label='Product 2')
    ax.plot(data['product3'], 'b-', alpha=0.6, label='Product 3')
    ax.plot(data['product4'], 'm-', alpha=0.6, label='Product 4')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('All Products vs True Signal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Errors
    ax = axes[0, 1]
    ax.plot(data['error1'], 'r-', alpha=0.6, label='Error 1')
    ax.plot(data['error2'], 'g-', alpha=0.6, label='Error 2')
    ax.plot(data['error3'], 'b-', alpha=0.6, label='Error 3')
    ax.plot(data['error4'], 'm-', alpha=0.6, label='Error 4')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Error')
    ax.set_title('Product Errors')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: IVD merged result
    ax = axes[1, 0]
    ax.plot(data['true_signal'], 'k-', label='True Signal', linewidth=2)
    ax.plot(merged_ivd, 'c-', label='IVD Merged', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('IVD Merged Result')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: EC merged result
    ax = axes[1, 1]
    ax.plot(data['true_signal'], 'k-', label='True Signal', linewidth=2)
    valid_idx = ~np.isnan(merged_ec)
    ax.plot(np.where(valid_idx)[0], merged_ec[valid_idx],
            'orange', label='EC Merged', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('EC Merged Result (4 products)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    # save under ./figures with script name prefix
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(script_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    script_stem = os.path.splitext(os.path.basename(__file__))[0]
    out_file = os.path.join(fig_dir, f"{script_stem}_collocation_results.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved as '{out_file}'")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("COLLOCATION ANALYSIS METHODS - COMPREHENSIVE EXAMPLE")
    print("=" * 70)
    print("\nThis example demonstrates all collocation methods using")
    print("synthetic data with known error characteristics.")

    # Generate data
    data = generate_synthetic_data()
    print(f"\nGenerated {len(data['true_signal'])} samples")
    print(f"True error std devs: "
          f"[{np.std(data['error1']):.3f}, "
          f"{np.std(data['error2']):.3f}, "
          f"{np.std(data['error3']):.3f}, "
          f"{np.std(data['error4']):.3f}]")

    # Run examples
    merged_ivd = example_ivd()
    example_ivs()
    example_tc()
    example_eivd()
    merged_ec = example_ec()

    # Create plots
    plot_results(data, merged_ivd, merged_ec)

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
