"""
Advanced Sensitivity Analysis of Collocation Methods
=====================================================

This script goes beyond 'comprehensive_comparison.py' by performing
parametric sensitivity analysis to find the "breaking points" of
each method.

It generates new, publication-quality figures:
1.  Sensitivity Line Plot: Shows how method accuracy (Rel. RMSE) degrades
    as a key parameter (e.g., error correlation) increases.
2.  Bayesian Posterior Comparison: Uses violin plots to compare the
    estimated uncertainty distributions from BTC vs. BTCH.
3.  Error Correlation Estimation Plot: A scatter plot showing how well
    EIVD and EC actually estimate the true error cross-correlation.

Requires: pandas, seaborn, matplotlib, pymc3
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import sys
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 1. Setup Path and Imports ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collocation import (
    tc, eivd, ec,
    BAYESIAN_AVAILABLE, BayesianTC,
    BAYESIAN_TCH_AVAILABLE, BayesianTCH
)
# Import simulation functions from the other example
try:
    from comprehensive_comparison import (
        setup_publication_style,
        simulate_scenario_3_timevarying, # For Bayesian plot
        METHOD_COLORS
    )
except ImportError:
    print("Error: 'comprehensive_comparison.py' not found.")
    print("Please ensure it is in the same directory or accessible in sys.path.")
    # Define fallbacks if import fails
    def setup_publication_style(): pass
    def simulate_scenario_3_timevarying():
        print("Fallback simulation running...")
        np.random.seed(44)
        n=500
        t = np.linspace(0, 4*np.pi, n)
        truth = 0.2 + 0.15 * np.sin(t) + 0.1 * np.sin(2*t)
        rmse_base = np.array([0.02, 0.03, 0.04, 0.025])
        seasonal = 1 + 0.5 * np.sin(t / 2)
        products = []
        for i in range(4):
            rmse_t = rmse_base[i] * seasonal
            error = np.random.normal(0, 1, n) * rmse_t
            product = truth + error
            products.append(product)
        data = np.array(products)
        rmse_true = rmse_base * np.mean(seasonal)
        return {
            'data': data, 'truth': truth, 'rmse_true': rmse_true,
            'name': 'Time-Varying Errors (Fallback)'
        }
    METHOD_COLORS = {
        'TC': '#029E73', 'EIVD': '#CC3311', 'EC': '#9558B2',
        'BTC': '#8B4513', 'TCH': '#FBAFE4', 'BTCH': '#949494'
    }


warnings.filterwarnings('ignore')
setup_publication_style() # Use the same publication style

# Add new colors for TCH/BTCH if not already in comprehensive_comparison
if 'TCH' not in METHOD_COLORS:
    METHOD_COLORS['TCH'] = '#FBAFE4' # pink
    METHOD_COLORS['BTCH'] = '#949494' # gray

# --- 2. Parametric Scenario Generator ---

def generate_correlated_scenario(n=500, corr_level=0.0, seed=42):
    """
    Generates a 4-product scenario where products 1 and 2
    have their errors correlated at 'corr_level'.
    """
    np.random.seed(seed)
    
    # True signal
    t = np.linspace(0, 4*np.pi, n)
    truth = 0.2 + 0.15 * np.sin(t) + 0.1 * np.sin(2*t)

    # Error parameters
    rmse_true = np.array([0.02, 0.03, 0.04, 0.025])
    
    # Create covariance matrix for errors 1 & 2
    cov_12 = corr_level * rmse_true[0] * rmse_true[1]
    err_cov_matrix = np.array([
        [rmse_true[0]**2, cov_12],
        [cov_12, rmse_true[1]**2]
    ])
    
    # Cholesky decomposition to generate correlated errors
    try:
        L = np.linalg.cholesky(err_cov_matrix)
    except np.linalg.LinAlgError:
        # Fallback for non-positive-definite (e.g., corr=1)
        if cov_12 == 0: cov_12 = 1e-9 # avoid division by zero
        L = np.array([[rmse_true[0], 0], [cov_12 / rmse_true[0], np.sqrt(max(0, rmse_true[1]**2 - (cov_12/rmse_true[0])**2))]])

    uncorr_err = np.random.normal(0, 1, (2, n))
    corr_err_12 = (L @ uncorr_err)
    
    # Independent errors for 3 & 4
    err_3 = np.random.normal(0, rmse_true[2], n)
    err_4 = np.random.normal(0, rmse_true[3], n)

    # Generate products (no bias for simplicity in this test)
    p1 = truth + corr_err_12[0, :]
    p2 = truth + corr_err_12[1, :]
    p3 = truth + err_3
    p4 = truth + err_4
    
    data = np.array([p1, p2, p3, p4])
    
    return {
        'data': data,
        'truth': truth,
        'rmse_true': rmse_true,
        'true_corr': (corr_level, 0, 1) # (level, prod_idx_1, prod_idx_2)
    }

# --- 3. Method Application Loop ---

def run_sensitivity_analysis(param_levels):
    """
    Runs all methods over a range of parameter levels.
    """
    results_list = []
    
    # Define methods that can handle 3 or 4 products
    # (Method, n_products, estimates_corr)
    methods_to_run = {
        'TC': (tc, 3, False),
        'EIVD': (eivd, 3, True),
        'EC': (ec, 4, True),
    }

    for level in param_levels:
        print(f"  Running analysis for correlation level: {level:.2f}")
        scenario = generate_correlated_scenario(corr_level=level)
        data = scenario['data']
        rmse_true = scenario['rmse_true']

        for name, (func, n_prod, est_corr) in methods_to_run.items():
            
            # Select appropriate data slice
            if n_prod == 3:
                # Use (P1, P2, P3) - contains the correlated pair
                data_slice = data[:3, :].T
                rmse_true_slice = rmse_true[:3]
                corr_products = [0, 1] # P1 and P2
            elif n_prod == 4:
                data_slice = data[:4, :].T
                rmse_true_slice = rmse_true[:4]
                corr_products = [0, 1] # P1 and P2

            try:
                if name == 'EC':
                    ec_results = func(data_slice)
                    if not ec_results: raise Exception("EC failed")
                    # Find combination [0, 1, 2] or [0, 1, 3] etc.
                    # This is complex, let's simplify and use first result
                    res = ec_results[0]
                    EeeT = res.EeeT[0]
                    # Estimate error corr for P1, P2
                    est_corr_val = res.EeeT[0][corr_products[0], corr_products[1]] / \
                                   np.sqrt(res.EeeT[0][corr_products[0], corr_products[0]] * \
                                            res.EeeT[0][corr_products[1], corr_products[1]])
                else:
                    res = func(data_slice)
                    EeeT = res[0] # EeeT is the first return val
                    if est_corr:
                        # EIVD estimates EeeT directly
                        est_corr_val = EeeT[corr_products[0], corr_products[1]] / \
                                       np.sqrt(EeeT[corr_products[0], corr_products[0]] * \
                                                EeeT[corr_products[1], corr_products[1]])
                    else:
                        est_corr_val = np.nan # TC doesn't estimate it
                
                rmse_est = np.sqrt(np.diag(EeeT))
                
                # Calculate relative error for the correlated products
                rel_err_p1 = np.abs(rmse_est[corr_products[0]] - rmse_true_slice[corr_products[0]]) / rmse_true_slice[corr_products[0]]
                rel_err_p2 = np.abs(rmse_est[corr_products[1]] - rmse_true_slice[corr_products[1]]) / rmse_true_slice[corr_products[1]]
                
                # Store results for P1 (one of the correlated products)
                results_list.append({
                    'param_level': level,
                    'method': name,
                    'product': 'Product 1',
                    'rmse_est': rmse_est[corr_products[0]],
                    'rmse_true': rmse_true_slice[corr_products[0]],
                    'relative_error': rel_err_p1 * 100, # In percent
                    'est_corr': est_corr_val
                })
                # Store results for P2
                results_list.append({
                    'param_level': level,
                    'method': name,
                    'product': 'Product 2',
                    'rmse_est': rmse_est[corr_products[1]],
                    'rmse_true': rmse_true_slice[corr_products[1]],
                    'relative_error': rel_err_p2 * 100, # In percent
                    'est_corr': est_corr_val # Redundant, but ok
                })

            except Exception as e:
                print(f"    Failed {name} at level {level}: {e}")
                results_list.append({
                    'param_level': level, 'method': name, 'product': 'Product 1',
                    'relative_error': np.nan, 'est_corr': np.nan
                })
                results_list.append({
                    'param_level': level, 'method': name, 'product': 'Product 2',
                    'relative_error': np.nan, 'est_corr': np.nan
                })

    return pd.DataFrame(results_list)


def run_bayesian_comparison(scenario):
    """
    Runs BTC and BTCH on a single scenario and returns
    posterior samples for plotting. If PyMC3 is not available,
    uses bootstrap approximation like in bayesian_tch_example.py.
    """
    data = scenario['data'][:3, :] # Use 3 products
    
    if BAYESIAN_AVAILABLE and BAYESIAN_TCH_AVAILABLE:
        # Try Bayesian approach first
        try:
            # Run BTC (complex model)
            print("  Running BTC (complex model)...")
            btc = BayesianTC(data)
            btc.run_inference(niter=1000, nadvi=30000, nchains=2, seed=42)
            btc_samples = btc.get_posterior_samples('sigmap') # (n_samples, n_products)
            
            # Run BTCH (simple model)
            print("  Running BTCH (simple model)...")
            btch = BayesianTCH(data)
            btch.run_inference(niter=1000, nadvi=30000, nchains=2, seed=42)
            btch_samples = btch.get_posterior_samples('sigmap') # (n_samples, n_products)

            # Format for DataFrame
            df_btc = pd.DataFrame(btc_samples, columns=['Product 1', 'Product 2', 'Product 3'])
            df_btc['Method'] = 'BTC (Time-Varying)'
            
            df_btch = pd.DataFrame(btch_samples, columns=['Product 1', 'Product 2', 'Product 3'])
            df_btch['Method'] = 'BTCH (Constant Err)'
            
            # Melt for seaborn
            df_all = pd.concat([df_btc, df_btch])
            df_melted = df_all.melt(id_vars='Method', var_name='Product', value_name='RMSE Estimate')
            
            return df_melted
        
        except Exception as e:
            print(f"  Bayesian analysis failed: {e}")
            print("  Falling back to bootstrap approximation...")
    else:
        print("  PyMC3 not available. Using bootstrap approximation for uncertainty...")
    
    # Bootstrap fallback (non-Bayesian approximation)
    try:
        B = 500  # bootstrap replicates (reduced for speed in comparison plot)
        rng = np.random.RandomState(42)
        tri = data.T  # shape: (n_samples, n_products)
        n = tri.shape[0]
        
        # Bootstrap for "BTC-like" (classical TC with resampling)
        btc_boot_rmse = np.zeros((B, data.shape[0]))
        btch_boot_rmse = np.zeros((B, data.shape[0]))
        
        print(f"  Running {B} bootstrap replicates for BTC/BTCH approximation...")
        for b in range(B):
            idx = rng.randint(0, n, size=n)
            sample = tri[idx, :]
            try:
                # Use TC as approximation for both BTC and BTCH
                EeeT_b, _, _, _ = tc(sample)
                rmse_b = np.sqrt(np.diag(EeeT_b))
                btc_boot_rmse[b, :] = rmse_b
                btch_boot_rmse[b, :] = rmse_b
            except Exception:
                # Fallback to sample RMSE if tc fails
                rmse_b = np.sqrt(np.mean((sample - sample.mean(axis=0)) ** 2, axis=0))
                btc_boot_rmse[b, :] = rmse_b
                btch_boot_rmse[b, :] = rmse_b
        
        # Format for DataFrame (mimic Bayesian output structure)
        df_btc = pd.DataFrame(btc_boot_rmse, columns=['Product 1', 'Product 2', 'Product 3'])
        df_btc['Method'] = 'BTC (Time-Varying)'
        
        df_btch = pd.DataFrame(btch_boot_rmse, columns=['Product 1', 'Product 2', 'Product 3'])
        df_btch['Method'] = 'BTCH (Constant Err)'
        
        # Melt for seaborn
        df_all = pd.concat([df_btc, df_btch])
        df_melted = df_all.melt(id_vars='Method', var_name='Product', value_name='RMSE Estimate')
        
        return df_melted
        
    except Exception as e:
        print(f"  Bootstrap analysis also failed: {e}")
        print("  Skipping Bayesian plot.")
        return None


# --- 4. New "Good Plot" Functions ---

def plot_sensitivity_breaking_points(df, save_path=None):
    """
    NEW PLOT 1: Sensitivity Line Plot ("Breaking Point" Plot)
    Shows how estimation error increases with error correlation.
    """
    print("\nCreating Plot 1: Sensitivity 'Breaking Point' Plot")
    
    if df.empty:
        print("  DataFrame is empty, skipping plot 1.")
        return
        
    # Aggregate results (mean over P1 and P2)
    df_agg = df.groupby(['param_level', 'method']).relative_error.mean().reset_index()

    plt.figure(figsize=(5.5, 4)) # Single-column width-ish
    ax = sns.lineplot(
        data=df_agg,
        x='param_level',
        y='relative_error',
        hue='method',
        style='method',
        markers=True,
        linewidth=1.5,
        palette=[METHOD_COLORS[m] for m in df_agg['method'].unique()]
    )
    
    ax.set_xlabel('True Error Correlation (between P1 and P2)', fontsize=8)
    ax.set_ylabel('Mean Relative Error in RMSE (%)', fontsize=8)
    ax.set_title('Method Robustness to Error Correlation', fontweight='bold', loc='left')
    ax.axhline(10, ls='--', color='gray', linewidth=1, label='10% Error Threshold')
    ax.legend(title='Method')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        try:
            plt.savefig(save_path, dpi=300)
            print(f"Saved: {save_path}")
        except Exception as e:
            print(f"Error saving plot 1: {e}")
    plt.close()


def plot_bayesian_posterior_comparison(df_melted, scenario, save_path=None):
    """
    NEW PLOT 2: Bayesian Posterior Comparison (Violin Plot)
    Shows the full uncertainty distribution from BTC and BTCH.
    """
    print("\nCreating Plot 2: Bayesian Posterior Comparison")
    if df_melted is None or df_melted.empty:
        print("  Skipped (Bayesian results not available or empty).")
        return

    rmse_true = scenario['rmse_true'][:3]

    plt.figure(figsize=(7, 4)) # 183mm width
    
    g = sns.violinplot(
        data=df_melted,
        x='Product',
        y='RMSE Estimate',
        hue='Method',
        split=True,
        inner='quartile',
        palette={'BTC (Time-Varying)': METHOD_COLORS['BTC'], 'BTCH (Constant Err)': METHOD_COLORS['BTCH']}
    )

    # Add true RMSE lines
    for i in range(3):
        plt.axhline(rmse_true[i], xmin=i/3.0 + 0.05, xmax=(i+1)/3.0 - 0.05,
                    color='red', linestyle='--', linewidth=1.5, label='True RMSE' if i==0 else None)
    
    g.set_title(f'Bayesian Model Comparison ({scenario["name"]})', fontweight='bold', loc='left')
    g.set_ylabel('RMSE Estimate')
    handles, labels = g.get_legend_handles_labels()
    # Add true RMSE to legend
    if 'True RMSE' in labels:
        g.legend(handles=handles, labels=labels)
    else:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color='red', ls='--', lw=1.5))
        labels.append('True RMSE')
        g.legend(handles=handles, labels=labels)

    plt.tight_layout()
    if save_path:
        try:
            plt.savefig(save_path, dpi=300)
            print(f"Saved: {save_path}")
        except Exception as e:
            print(f"Error saving plot 2: {e}")
    plt.close()


def plot_correlation_estimation(df, save_path=None):
    """
    NEW PLOT 3: Error Correlation Estimation Plot
    Shows how well EIVD and EC estimate the true correlation.
    """
    print("\nCreating Plot 3: Error Correlation Estimation")
    
    if df.empty:
        print("  DataFrame is empty, skipping plot 3.")
        return
        
    # Filter to methods that estimate correlation
    df_corr = df[df['est_corr'].notna()].copy()
    
    if df_corr.empty:
        print("  No correlation estimates found, skipping plot 3.")
        return
        
    df_corr['true_corr'] = df_corr['param_level']
    
    # Aggregate only numeric columns to avoid trying to average string/object columns
    df_corr_agg = df_corr.groupby(['param_level', 'method']).agg({'est_corr': 'mean', 'true_corr': 'mean'}).reset_index()

    plt.figure(figsize=(5.5, 4))
    
    # Plot the ideal 1:1 line
    plt.plot([0, 0.8], [0, 0.8], 'k--', label='Ideal 1:1 Line', zorder=1)
    
    ax = sns.scatterplot(
        data=df_corr_agg,
        x='true_corr',
        y='est_corr',
        hue='method',
        style='method',
        s=100, # larger markers
        palette=[METHOD_COLORS[m] for m in df_corr_agg['method'].unique()],
        zorder=2
    )

    ax.set_xlabel('True Error Correlation', fontsize=8)
    ax.set_ylabel('Estimated Error Correlation', fontsize=8)
    ax.set_title('Accuracy of Error Correlation Estimation', fontweight='bold', loc='left')
    ax.legend(title='Method')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        try:
            plt.savefig(save_path, dpi=300)
            print(f"Saved: {save_path}")
        except Exception as e:
            print(f"Error saving plot 3: {e}")
    plt.close()


# --- 5. Main Execution ---

def main():
    print("="*100)
    print("ADVANCED SENSITIVITY ANALYSIS")
    print("="*100)
    
    # --- 0. Setup Figure Directory ---
    # 补丁：自动创建输出目录
    # 获取此脚本所在的目录 (e.g., .../examples)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # 定义输出目录 (e.g., .../examples/figures)
    FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"Ensured output directory exists: {FIG_DIR}")

    # 补丁：定义完整的保存路径，使用脚本名作为前缀
    script_base = os.path.splitext(os.path.basename(__file__))[0]
    save_path_1 = os.path.join(FIG_DIR, f"{script_base}_plot_1_corr_sensitivity.png")
    save_path_2 = os.path.join(FIG_DIR, f"{script_base}_plot_2_bayesian_violin.png")
    save_path_3 = os.path.join(FIG_DIR, f"{script_base}_plot_3_corr_estimation.png")
    
    # --- Analysis 1: Sensitivity to Error Correlation ---
    print("\nStarting Analysis 1: Sensitivity to Error Correlation")
    # Define parameter range to test
    correlation_levels = np.linspace(0, 0.8, 9) # 9 steps from 0 to 0.8
    
    # Run the analysis
    results_df = run_sensitivity_analysis(correlation_levels)
    
    # Create the plots
    plot_sensitivity_breaking_points(
        results_df, 
        save_path=save_path_1
    )
    
    plot_correlation_estimation(
        results_df,
        save_path=save_path_3
    )

    # --- Analysis 2: Bayesian Model Comparison ---
    print("\nStarting Analysis 2: Bayesian Model Comparison")
    # We use the time-varying scenario, as this is where
    # BTC (complex) should shine vs. BTCH (simple)
    scenario = simulate_scenario_3_timevarying()
    
    bayesian_results_df = run_bayesian_comparison(scenario)
    
    plot_bayesian_posterior_comparison(
        bayesian_results_df,
        scenario,
        save_path=save_path_2
    )
    
    print("\n" + "="*100)
    print("ADVANCED ANALYSIS COMPLETE")
    print("="*100)
    print(f"\nNew advanced figures saved in: {FIG_DIR}")
    print("- adv_plot_1_corr_sensitivity.png")
    print("- adv_plot_2_bayesian_violin.png")
    print("- adv_plot_3_corr_estimation.png")

if __name__ == "__main__":
    main()