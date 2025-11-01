"""
Basic example demonstrating TC and ETCC merging methods.

This example:
1. Generates synthetic precipitation data
2. Applies both TC and ETCC merging
3. Evaluates and compares results
4. Visualizes the comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from etcc import (TripleCollocation, ETCC, calculate_metrics, 
                  compare_products, print_comparison_table,
                  generate_synthetic_data, plot_comparison,
                  plot_scatter, plot_weights_comparison)


def main():
    print("=" * 70)
    print("ETCC Precipitation Merging - Basic Example")
    print("=" * 70 + "\n")
    
    # 1. Generate synthetic data
    print("1. Generating synthetic precipitation data...")
    n_samples = 1000
    truth, x, y, z = generate_synthetic_data(
        n_samples=n_samples,
        correlation=0.7,
        noise_level=0.3,
        seed=42
    )
    print(f"   Generated {n_samples} time steps")
    print(f"   Mean precipitation: {np.mean(truth):.2f} mm/day\n")
    
    # 2. Apply TC merging
    print("2. Applying Triple Collocation (TC) merging...")
    tc = TripleCollocation()
    merged_tc = tc.merge(x, y, z)
    print(f"   TC weights: wx={tc.weights['wx']:.3f}, "
          f"wy={tc.weights['wy']:.3f}, wz={tc.weights['wz']:.3f}")
    print(f"   Error variances: σx²={tc.error_variances['sigma2_x']:.3f}, "
          f"σy²={tc.error_variances['sigma2_y']:.3f}, "
          f"σz²={tc.error_variances['sigma2_z']:.3f}\n")
    
    # 3. Apply ETCC merging
    print("3. Applying ETCC (Extended TC with maximized correlation)...")
    etcc = ETCC(weight_increment=0.01)
    merged_etcc = etcc.merge(x, y, z)
    print(f"   ETCC weights: wx={etcc.weights['wx']:.3f}, "
          f"wy={etcc.weights['wy']:.3f}, wz={etcc.weights['wz']:.3f}")
    print(f"   Maximum correlation achieved: {etcc.max_correlation:.3f}")
    print(f"   Correlations with truth: ρRx={etcc.correlation_with_truth['rho_Rx']:.3f}, "
          f"ρRy={etcc.correlation_with_truth['rho_Ry']:.3f}, "
          f"ρRz={etcc.correlation_with_truth['rho_Rz']:.3f}\n")
    
    # 4. Evaluate all products
    print("4. Evaluating all products against reference...")
    products = {
        'Product X': x,
        'Product Y': y,
        'Product Z': z,
        'TC Merged': merged_tc,
        'ETCC Merged': merged_etcc
    }
    
    results = compare_products(products, truth)
    print_comparison_table(results)
    
    # 5. Analyze improvements
    print("5. Performance improvements:")
    best_input_cc = max(results['Product X']['cc'], 
                       results['Product Y']['cc'],
                       results['Product Z']['cc'])
    tc_improvement = (results['TC Merged']['cc'] - best_input_cc) / best_input_cc * 100
    etcc_improvement = (results['ETCC Merged']['cc'] - best_input_cc) / best_input_cc * 100
    
    print(f"   Best input product CC: {best_input_cc:.4f}")
    print(f"   TC improvement: +{tc_improvement:.2f}%")
    print(f"   ETCC improvement: +{etcc_improvement:.2f}%")
    print(f"   ETCC vs TC improvement: "
          f"+{(results['ETCC Merged']['cc'] - results['TC Merged']['cc']) / results['TC Merged']['cc'] * 100:.2f}%\n")
    
    # 6. Visualizations
    print("6. Creating visualizations...\n")
    
    # Plot time series comparison
    plot_comparison(
        products=products,
        reference=truth,
        figsize=(14, 6),
        save_path='comparison_timeseries.png'
    )
    
    # Plot scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # TC scatter
    plt.sca(axes[0])
    plot_scatter(merged_tc, truth, 'TC Merged')
    
    # ETCC scatter
    plt.sca(axes[1])
    plot_scatter(merged_etcc, truth, 'ETCC Merged')
    
    plt.tight_layout()
    plt.savefig('scatter_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot weights comparison
    plot_weights_comparison(
        tc_weights=tc.weights,
        etcc_weights=etcc.weights,
        product_names=['Product X', 'Product Y', 'Product Z'],
        save_path='weights_comparison.png'
    )
    
    print("=" * 70)
    print("Analysis complete!")
    print("Figures saved: comparison_timeseries.png, scatter_comparison.png, weights_comparison.png")
    print("=" * 70)


if __name__ == '__main__':
    main()
