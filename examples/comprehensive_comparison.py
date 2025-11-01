"""
Comprehensive Comparison of Collocation Methods
================================================

This script provides a comprehensive comparison of all collocation methods
across multiple scenarios with publication-quality figures following
Nature/Science journal standards.

Scenarios Compared:
1. Ideal case: Independent errors, constant variance
2. Correlated errors: Products with error cross-correlation
3. Time-varying errors: Heteroscedastic error structure
4. Systematic biases: Strong additive and multiplicative biases
5. Heavy-tailed errors: Non-Gaussian error distributions
6. Real-world case: Combined challenges

Methods Compared:
- IVD: Information Vector Dual (2-way)
- IVS: Information Vector with Scaling (2-way)
- TC: Triple Collocation (3-way)
- EIVD: Extended IVD (3-way with error correlation)
- EC: Extended Collocation (4-way)
- BTC: Bayesian Triple Collocation (3-way with full uncertainty)

Figure Style: Nature/Science standard
- Font: Arial or Helvetica
- Font size: 6-8pt (axis labels), 7-9pt (titles)
- Line width: 0.75-1.0pt
- Figure size: 89mm (single column) or 183mm (double column)
- DPI: 300-600 for final figures
- Colors: Colorblind-friendly palette
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import warnings
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collocation import ivd, ivs, tc, eivd, ec, BAYESIAN_AVAILABLE
from collocation.utils import calculate_all_metrics

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# Nature/Science Figure Configuration
# ============================================================================

def setup_publication_style():
    """Setup matplotlib for publication-quality figures (Nature/Science style)."""

    # Font settings
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    rcParams['font.size'] = 7
    rcParams['axes.labelsize'] = 8
    rcParams['axes.titlesize'] = 9
    rcParams['xtick.labelsize'] = 7
    rcParams['ytick.labelsize'] = 7
    rcParams['legend.fontsize'] = 6

    # Line and marker settings
    rcParams['lines.linewidth'] = 1.0
    rcParams['lines.markersize'] = 3
    rcParams['patch.linewidth'] = 0.75

    # Axes settings
    rcParams['axes.linewidth'] = 0.75
    rcParams['axes.grid'] = True
    rcParams['grid.alpha'] = 0.3
    rcParams['grid.linewidth'] = 0.5

    # Tick settings
    rcParams['xtick.major.width'] = 0.75
    rcParams['ytick.major.width'] = 0.75
    rcParams['xtick.minor.width'] = 0.5
    rcParams['ytick.minor.width'] = 0.5

    # Figure settings
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.pad_inches'] = 0.05

    # Legend settings
    rcParams['legend.frameon'] = True
    rcParams['legend.framealpha'] = 0.9
    rcParams['legend.edgecolor'] = 'gray'
    rcParams['legend.fancybox'] = False

# Colorblind-friendly palette (Wong 2011)
COLORS = {
    'blue': '#0173B2',
    'orange': '#DE8F05',
    'green': '#029E73',
    'red': '#CC3311',
    'purple': '#9558B2',
    'brown': '#8B4513',
    'pink': '#FBAFE4',
    'gray': '#949494',
}

METHOD_COLORS = {
    'IVD': COLORS['blue'],
    'IVS': COLORS['orange'],
    'TC': COLORS['green'],
    'EIVD': COLORS['red'],
    'EC': COLORS['purple'],
    'BTC': COLORS['brown'],
}

# ============================================================================
# Data Simulation Functions
# ============================================================================

def simulate_scenario_1_ideal(n=500, seed=42):
    """
    Scenario 1: Ideal case
    - Independent errors
    - Constant variance
    - Zero cross-correlation
    """
    np.random.seed(seed)

    # True signal
    t = np.linspace(0, 4*np.pi, n)
    truth = 0.2 + 0.15 * np.sin(t) + 0.1 * np.sin(2*t)

    # Error parameters
    rmse_true = np.array([0.02, 0.03, 0.04, 0.025])
    bias_add = np.array([0.0, 0.01, -0.01, 0.005])
    bias_mul = np.array([1.0, 1.05, 0.95, 1.02])

    # Generate products
    products = []
    for i in range(4):
        error = np.random.normal(0, rmse_true[i], n)
        product = bias_add[i] + bias_mul[i] * truth + error
        products.append(product)

    data = np.array(products)

    return {
        'data': data,
        'truth': truth,
        'rmse_true': rmse_true,
        'bias_add': bias_add,
        'bias_mul': bias_mul,
        'name': 'Ideal Case',
        'description': 'Independent errors, constant variance'
    }


def simulate_scenario_2_correlated(n=500, seed=43):
    """
    Scenario 2: Correlated errors
    - Error cross-correlation between products
    - Shared error source
    """
    np.random.seed(seed)

    # True signal
    t = np.linspace(0, 4*np.pi, n)
    truth = 0.2 + 0.15 * np.sin(t) + 0.1 * np.sin(2*t)

    # Error parameters
    rmse_true = np.array([0.02, 0.03, 0.04, 0.025])
    bias_add = np.array([0.0, 0.01, -0.01, 0.005])
    bias_mul = np.array([1.0, 1.05, 0.95, 1.02])

    # Generate correlated errors
    # Products 2 and 3 share some common error
    common_error = np.random.normal(0, 0.02, n)

    products = []
    for i in range(4):
        independent_error = np.random.normal(0, rmse_true[i], n)

        # Add common error to products 1 and 2
        if i in [1, 2]:
            error = 0.7 * independent_error + 0.3 * common_error
        else:
            error = independent_error

        product = bias_add[i] + bias_mul[i] * truth + error
        products.append(product)

    data = np.array(products)

    return {
        'data': data,
        'truth': truth,
        'rmse_true': rmse_true,
        'bias_add': bias_add,
        'bias_mul': bias_mul,
        'name': 'Correlated Errors',
        'description': 'Products 2-3 share common error source'
    }


def simulate_scenario_3_timevarying(n=500, seed=44):
    """
    Scenario 3: Time-varying errors
    - Heteroscedastic error structure
    - Error variance changes over time
    """
    np.random.seed(seed)

    # True signal
    t = np.linspace(0, 4*np.pi, n)
    truth = 0.2 + 0.15 * np.sin(t) + 0.1 * np.sin(2*t)

    # Base error parameters
    rmse_base = np.array([0.02, 0.03, 0.04, 0.025])
    bias_add = np.array([0.0, 0.01, -0.01, 0.005])
    bias_mul = np.array([1.0, 1.05, 0.95, 1.02])

    # Time-varying error variance (seasonal pattern)
    seasonal = 1 + 0.5 * np.sin(t / 2)

    products = []
    for i in range(4):
        # Error variance increases in certain seasons
        rmse_t = rmse_base[i] * seasonal
        error = np.random.normal(0, 1, n) * rmse_t
        product = bias_add[i] + bias_mul[i] * truth + error
        products.append(product)

    data = np.array(products)

    # Average RMSE for comparison
    rmse_true = rmse_base * np.mean(seasonal)

    return {
        'data': data,
        'truth': truth,
        'rmse_true': rmse_true,
        'bias_add': bias_add,
        'bias_mul': bias_mul,
        'name': 'Time-Varying Errors',
        'description': 'Heteroscedastic errors with seasonal pattern'
    }


def simulate_scenario_4_biased(n=500, seed=45):
    """
    Scenario 4: Systematic biases
    - Strong additive biases
    - Strong multiplicative biases
    """
    np.random.seed(seed)

    # True signal
    t = np.linspace(0, 4*np.pi, n)
    truth = 0.2 + 0.15 * np.sin(t) + 0.1 * np.sin(2*t)

    # Error parameters with strong biases
    rmse_true = np.array([0.02, 0.03, 0.04, 0.025])
    bias_add = np.array([0.0, 0.05, -0.04, 0.03])  # Strong additive bias
    bias_mul = np.array([1.0, 1.2, 0.8, 1.15])    # Strong multiplicative bias

    products = []
    for i in range(4):
        error = np.random.normal(0, rmse_true[i], n)
        product = bias_add[i] + bias_mul[i] * truth + error
        products.append(product)

    data = np.array(products)

    return {
        'data': data,
        'truth': truth,
        'rmse_true': rmse_true,
        'bias_add': bias_add,
        'bias_mul': bias_mul,
        'name': 'Systematic Biases',
        'description': 'Strong additive and multiplicative biases'
    }


def simulate_scenario_5_heavytail(n=500, seed=46):
    """
    Scenario 5: Heavy-tailed errors
    - Non-Gaussian error distributions
    - Student-t errors with outliers
    """
    np.random.seed(seed)

    # True signal
    t = np.linspace(0, 4*np.pi, n)
    truth = 0.2 + 0.15 * np.sin(t) + 0.1 * np.sin(2*t)

    # Error parameters
    rmse_true = np.array([0.02, 0.03, 0.04, 0.025])
    bias_add = np.array([0.0, 0.01, -0.01, 0.005])
    bias_mul = np.array([1.0, 1.05, 0.95, 1.02])

    # Generate products with Student-t errors (df=3)
    products = []
    for i in range(4):
        # Student-t with 3 degrees of freedom (heavy tails)
        error = np.random.standard_t(3, n) * rmse_true[i] * np.sqrt(3/(3-2))
        product = bias_add[i] + bias_mul[i] * truth + error
        products.append(product)

    data = np.array(products)

    return {
        'data': data,
        'truth': truth,
        'rmse_true': rmse_true,
        'bias_add': bias_add,
        'bias_mul': bias_mul,
        'name': 'Heavy-Tailed Errors',
        'description': 'Student-t errors with outliers (df=3)'
    }


def simulate_scenario_6_realistic(n=500, seed=47):
    """
    Scenario 6: Realistic case
    - Combined challenges from multiple scenarios
    - Time-varying + correlated + biased
    """
    np.random.seed(seed)

    # True signal with more complexity
    t = np.linspace(0, 4*np.pi, n)
    truth = 0.2 + 0.15 * np.sin(t) + 0.1 * np.sin(2*t) + 0.05 * np.random.randn(n).cumsum() * 0.001

    # Base error parameters
    rmse_base = np.array([0.02, 0.03, 0.04, 0.025])
    bias_add = np.array([0.0, 0.03, -0.02, 0.01])
    bias_mul = np.array([1.0, 1.1, 0.9, 1.05])

    # Time-varying error variance
    seasonal = 1 + 0.3 * np.sin(t / 2)

    # Common error source
    common_error = np.random.normal(0, 0.015, n)

    products = []
    for i in range(4):
        # Time-varying variance
        rmse_t = rmse_base[i] * seasonal
        independent_error = np.random.normal(0, 1, n) * rmse_t

        # Add correlation
        if i in [1, 2]:
            error = 0.8 * independent_error + 0.2 * common_error * seasonal
        else:
            error = independent_error

        product = bias_add[i] + bias_mul[i] * truth + error
        products.append(product)

    data = np.array(products)

    # Average RMSE
    rmse_true = rmse_base * np.mean(seasonal)

    return {
        'data': data,
        'truth': truth,
        'rmse_true': rmse_true,
        'bias_add': bias_add,
        'bias_mul': bias_mul,
        'name': 'Realistic Case',
        'description': 'Combined: time-varying + correlated + biased'
    }


# ============================================================================
# Method Application Functions
# ============================================================================

def apply_classical_methods(scenario):
    """Apply all classical collocation methods to a scenario."""
    data = scenario['data']
    n_products = data.shape[0]

    results = {}

    # IVD (2-way)
    try:
        dual = data[:2, :].T
        EeeT, rho2, weights = ivd(dual)
        rmse_ivd = np.sqrt(np.diag(EeeT))
        results['IVD'] = {
            'rmse': rmse_ivd,
            'rho2': rho2,
            'n_products': 2,
            'success': True
        }
    except Exception as e:
        results['IVD'] = {'success': False, 'error': str(e)}

    # IVS (2-way with bootstrap)
    try:
        dual = data[:2, :].T
        RMSE, rho2 = ivs(dual, N_boot=100)
        results['IVS'] = {
            'rmse': RMSE,
            'rho2': rho2,
            'n_products': 2,
            'success': True
        }
    except Exception as e:
        results['IVS'] = {'success': False, 'error': str(e)}

    # TC (3-way)
    try:
        tri = data[:3, :].T
        EeeT, SNR, rho2, fMSE = tc(tri)
        rmse_tc = np.sqrt(np.diag(EeeT))
        results['TC'] = {
            'rmse': rmse_tc,
            'SNR': SNR,
            'rho2': rho2,
            'fMSE': fMSE,
            'n_products': 3,
            'success': True
        }
    except Exception as e:
        results['TC'] = {'success': False, 'error': str(e)}

    # EIVD (3-way with error correlation)
    try:
        tri = data[:3, :].T
        EeeT, SNR, rho2, fMSE, L = eivd(tri)
        rmse_eivd = np.sqrt(np.diag(EeeT))
        results['EIVD'] = {
            'rmse': rmse_eivd,
            'SNR': SNR,
            'rho2': rho2,
            'fMSE': fMSE,
            'L': L,
            'n_products': 3,
            'success': True
        }
    except Exception as e:
        results['EIVD'] = {'success': False, 'error': str(e)}

    # EC (4-way)
    if n_products >= 4:
        try:
            quad = data[:4, :].T
            ec_results = ec(quad)
            # Use first combination as representative
            if len(ec_results) > 0:
                best = ec_results[0]
                rmse_ec = np.sqrt(np.diag(best.EeeT[0]))
                results['EC'] = {
                    'rmse': rmse_ec,
                    'SNR': best.SNR[0],
                    'rho2': best.rho2[0],
                    'fMSE': best.fMSE[0],
                    'n_products': 4,
                    'success': True
                }
            else:
                results['EC'] = {'success': False, 'error': 'No valid combinations'}
        except Exception as e:
            results['EC'] = {'success': False, 'error': str(e)}

    return results


def apply_bayesian_method(scenario, niter=1000, nadvi=50000):
    """Apply Bayesian Triple Collocation to a scenario."""
    if not BAYESIAN_AVAILABLE:
        return {'success': False, 'error': 'PyMC3 not available'}

    try:
        from collocation import BayesianTC

        data = scenario['data'][:3, :]  # Use first 3 products

        btc = BayesianTC(data)
        btc.run_inference(niter=niter, nadvi=nadvi, nchains=1, seed=42)

        rmse_mean, rmse_std, rmse_quantiles = btc.get_error_estimates()
        m_mean, l_mean = btc.get_calibration_parameters()

        return {
            'rmse': rmse_mean,
            'rmse_std': rmse_std,
            'rmse_quantiles': rmse_quantiles,
            'bias_add': np.concatenate([[0], m_mean]),
            'bias_mul': np.concatenate([[1], l_mean]),
            'n_products': 3,
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_scenario_comparison(scenario, results, save_path=None):
    """
    Create a comprehensive comparison plot for one scenario.

    Layout:
    - Top: Time series of products and truth
    - Middle: RMSE comparison
    - Bottom: Error distribution
    """
    setup_publication_style()

    # Create figure (double column width: 183mm)
    fig = plt.figure(figsize=(7.2, 8))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    data = scenario['data']
    truth = scenario['truth']
    n = len(truth)
    t = np.arange(n)

    # ---- Top panel: Time series ----
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, truth, 'k-', linewidth=1.5, label='Truth', zorder=10)

    product_labels = ['Product 1', 'Product 2', 'Product 3', 'Product 4']
    for i in range(min(4, data.shape[0])):
        ax1.plot(t, data[i, :], alpha=0.6, linewidth=0.75, label=product_labels[i])

    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Value')
    ax1.set_title(f'a) {scenario["name"]}: {scenario["description"]}',
                  fontweight='bold', loc='left')
    ax1.legend(ncol=3, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # ---- Middle left: RMSE comparison ----
    ax2 = fig.add_subplot(gs[1, 0])

    methods = ['IVD', 'IVS', 'TC', 'EIVD', 'EC', 'BTC']
    method_labels = []
    rmse_values = []
    colors = []

    for method in methods:
        if method in results and results[method]['success']:
            n_prod = results[method]['n_products']
            rmse = results[method]['rmse'][:n_prod]

            for i in range(n_prod):
                method_labels.append(f'{method}\nP{i+1}')
                rmse_values.append(rmse[i])
                colors.append(METHOD_COLORS[method])

    if rmse_values:
        x_pos = np.arange(len(rmse_values))
        bars = ax2.bar(x_pos, rmse_values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        # Add true RMSE as horizontal lines
        rmse_true = scenario['rmse_true']
        for i, rmse in enumerate(rmse_true):
            ax2.axhline(y=rmse, color='red', linestyle='--', linewidth=1, alpha=0.5)

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(method_labels, rotation=45, ha='right', fontsize=5)
        ax2.set_ylabel('RMSE')
        ax2.set_title('b) RMSE estimates by method', fontweight='bold', loc='left')
        ax2.grid(True, alpha=0.3, axis='y')

    # ---- Middle right: Correlation comparison ----
    ax3 = fig.add_subplot(gs[1, 1])

    method_labels_r = []
    rho2_values = []
    colors_r = []

    for method in methods:
        if method in results and results[method]['success'] and 'rho2' in results[method]:
            n_prod = results[method]['n_products']
            rho2 = results[method]['rho2'][:n_prod]

            for i in range(n_prod):
                method_labels_r.append(f'{method}\nP{i+1}')
                rho2_values.append(rho2[i])
                colors_r.append(METHOD_COLORS[method])

    if rho2_values:
        x_pos = np.arange(len(rho2_values))
        bars = ax3.bar(x_pos, rho2_values, color=colors_r, alpha=0.7, edgecolor='black', linewidth=0.5)

        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(method_labels_r, rotation=45, ha='right', fontsize=5)
        ax3.set_ylabel(r'$\rho^2$ (Data-truth correlation)')
        ax3.set_ylim([0, 1]) # type: ignore
        ax3.set_title('c) Data-truth correlations', fontweight='bold', loc='left')
        ax3.grid(True, alpha=0.3, axis='y')

    # ---- Bottom left: Error distributions for Product 1 ----
    ax4 = fig.add_subplot(gs[2, 0])

    # Calculate errors for product 1
    for i in range(min(3, data.shape[0])):
        errors = data[i, :] - truth
        ax4.hist(errors, bins=30, alpha=0.5, label=f'Product {i+1}', density=True)

    ax4.set_xlabel('Error')
    ax4.set_ylabel('Density')
    ax4.set_title('d) Error distributions', fontweight='bold', loc='left')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # ---- Bottom right: Relative error ----
    ax5 = fig.add_subplot(gs[2, 1])

    method_labels_re = []
    rel_errors = []
    colors_re = []

    for method in methods:
        if method in results and results[method]['success']:
            n_prod = results[method]['n_products']
            rmse_est = results[method]['rmse'][:n_prod]
            rmse_true_subset = scenario['rmse_true'][:n_prod]

            rel_err = np.abs(rmse_est - rmse_true_subset) / rmse_true_subset * 100

            for i in range(n_prod):
                method_labels_re.append(f'{method}\nP{i+1}')
                rel_errors.append(rel_err[i])
                colors_re.append(METHOD_COLORS[method])

    if rel_errors:
        x_pos = np.arange(len(rel_errors))
        bars = ax5.bar(x_pos, rel_errors, color=colors_re, alpha=0.7, edgecolor='black', linewidth=0.5)

        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(method_labels_re, rotation=45, ha='right', fontsize=5)
        ax5.set_ylabel('Relative error (%)')
        ax5.set_title('e) RMSE estimation error', fontweight='bold', loc='left')
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.axhline(y=10, color='red', linestyle='--', linewidth=0.75, alpha=0.5, label='10% threshold')
        ax5.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_overall_comparison(all_results, save_path=None):
    """
    Create overall comparison across all scenarios.
    """
    setup_publication_style()

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    fig.suptitle('Overall Method Performance Across Scenarios', fontweight='bold', fontsize=10)

    scenarios = list(all_results.keys())
    methods = ['IVD', 'IVS', 'TC', 'EIVD', 'EC', 'BTC']

    # Calculate metrics for each scenario and method
    for idx, scenario_name in enumerate(scenarios):
        ax = axes[idx // 3, idx % 3]

        scenario = all_results[scenario_name]['scenario']
        results = all_results[scenario_name]['results']

        # Calculate mean relative error for each method
        method_errors = []
        method_names = []
        colors_list = []

        for method in methods:
            if method in results and results[method]['success']:
                n_prod = results[method]['n_products']
                rmse_est = results[method]['rmse'][:n_prod]
                rmse_true = scenario['rmse_true'][:n_prod]

                rel_err = np.abs(rmse_est - rmse_true) / rmse_true * 100
                mean_rel_err = np.mean(rel_err)

                method_errors.append(mean_rel_err)
                method_names.append(method)
                colors_list.append(METHOD_COLORS[method])

        if method_errors:
            x_pos = np.arange(len(method_errors))
            ax.bar(x_pos, method_errors, color=colors_list, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(method_names, rotation=45, ha='right')
            ax.set_ylabel('Mean relative error (%)')
            ax.set_title(scenario_name, fontweight='bold')
            ax.axhline(y=10, color='red', linestyle='--', linewidth=0.75, alpha=0.5)
            ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def create_method_comparison_table(all_results):
    """Create a comprehensive comparison table."""
    print("\n" + "="*100)
    print("COMPREHENSIVE METHOD COMPARISON")
    print("="*100)

    scenarios = list(all_results.keys())
    methods = ['IVD', 'IVS', 'TC', 'EIVD', 'EC', 'BTC']

    for scenario_name in scenarios:
        print(f"\n{scenario_name}")
        print("-" * 100)

        scenario = all_results[scenario_name]['scenario']
        results = all_results[scenario_name]['results']

        print(f"Description: {scenario['description']}")
        print(f"\nTrue RMSE: {scenario['rmse_true']}")
        print(f"True additive bias: {scenario['bias_add']}")
        print(f"True multiplicative bias: {scenario['bias_mul']}")

        print(f"\n{'Method':<10} {'Products':<10} {'Est. RMSE':<40} {'Mean Rel. Error':<15} {'Status'}")
        print("-" * 100)

        for method in methods:
            if method in results:
                if results[method]['success']:
                    n_prod = results[method]['n_products']
                    rmse_est = results[method]['rmse'][:n_prod]
                    rmse_true = scenario['rmse_true'][:n_prod]

                    rel_err = np.abs(rmse_est - rmse_true) / rmse_true * 100
                    mean_rel_err = np.mean(rel_err)

                    rmse_str = '[' + ', '.join([f'{r:.4f}' for r in rmse_est]) + ']'

                    print(f"{method:<10} {n_prod:<10} {rmse_str:<40} {mean_rel_err:>6.2f}%      OK")
                else:
                    print(f"{method:<10} {'N/A':<10} {'FAILED':<40} {'N/A':<15} {results[method].get('error', 'Unknown error')}")


# =================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("="*100)
    print("COMPREHENSIVE COLLOCATION METHODS COMPARISON")
    print("="*100)
    print("\nThis analysis compares all collocation methods across multiple scenarios")
    print("with publication-quality figures (Nature/Science standard).\n")

    # Setup publication style
    setup_publication_style()

    # Generate all scenarios
    print("Generating scenarios...")
    scenarios = [
        simulate_scenario_1_ideal(),
        simulate_scenario_2_correlated(),
        simulate_scenario_3_timevarying(),
        simulate_scenario_4_biased(),
        simulate_scenario_5_heavytail(),
        simulate_scenario_6_realistic(),
    ]

    # Apply methods to all scenarios
    all_results = {}

    for scenario in scenarios:
        print(f"\nProcessing: {scenario['name']}")
        print(f"  {scenario['description']}")

        # Apply classical methods
        print("  Applying classical methods...")
        results = apply_classical_methods(scenario)

        # Apply Bayesian method (with reduced iterations for speed)
        if BAYESIAN_AVAILABLE:
            print("  Applying Bayesian method (this may take a few minutes)...")
            results['BTC'] = apply_bayesian_method(scenario, niter=500, nadvi=30000)
        else:
            print("  Skipping Bayesian method (PyMC3 not available)")
            results['BTC'] = {'success': False, 'error': 'PyMC3 not available'}

        all_results[scenario['name']] = {
            'scenario': scenario,
            'results': results
        }

    # Create visualizations
    print("\n" + "="*100)
    print("CREATING VISUALIZATIONS")
    print("="*100)

    # Create a 'figures' folder under this script
    fig_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    for scenario_name in all_results.keys():
        scenario = all_results[scenario_name]['scenario']
        results = all_results[scenario_name]['results']

        filename = scenario_name.lower().replace(' ', '_') + '_comparison.png'
        filepath = os.path.join(fig_dir, filename)

        print(f"\nCreating figure for {scenario_name}...")
        plot_scenario_comparison(scenario, results, save_path=filepath)

    # Create overall comparison
    print("\nCreating overall comparison figure...")
    overall_path = os.path.join(fig_dir, 'overall_comparison.png')
    plot_overall_comparison(all_results, save_path=overall_path)

    # Print comparison table
    create_method_comparison_table(all_results)

    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print(f"\nFigures saved in: {fig_dir}")
    print("\nSummary:")
    print("- Individual scenario comparisons show detailed method performance")
    print("- Overall comparison shows method robustness across scenarios")
    print("- All figures follow Nature/Science publication standards")
    print("- Use 300 DPI for final publication")

    plt.show()


if __name__ == "__main__":
    main()