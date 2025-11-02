import numpy as np
import os
import pytest
import sys

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
xr = pytest.importorskip("xarray")

# Chinese font configuration
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Ensure repository root is on sys.path so 'examples' and 'collocation' are importable.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from examples.covariance_method_comparison import run_covariance_comparison


def save_covariance_test_figure(result, script_name="test_covariance_example"):
    """Save covariance comparison test visualization"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    rmse = result["rmse"]
    
    # Plot 1: Correlated case
    methods = list(rmse["correlated"].keys())
    corr_values = list(rmse["correlated"].values())
    ax1.bar(methods, corr_values, color=['skyblue', 'lightcoral'])
    ax1.set_title('相关情况下的RMSE对比', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Small sample case
    small_methods = list(rmse["small_sample"].keys())
    small_values = list(rmse["small_sample"].values())
    ax2.bar(small_methods, small_values, color=['orange', 'lightgreen', 'purple'])
    ax2.set_title('小样本情况下的RMSE对比', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMSE')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Independent case
    indep_methods = list(rmse["independent"].keys())
    indep_values = list(rmse["independent"].values())
    ax3.bar(indep_methods, indep_values, color=['lightblue', 'pink'])
    ax3.set_title('独立情况下的RMSE对比', fontsize=12, fontweight='bold')
    ax3.set_ylabel('RMSE')
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Overall comparison
    scenarios = ['相关', '小样本', '独立']
    full_rmse = [rmse["correlated"]["full"], rmse["small_sample"]["full"], rmse["independent"]["full"]]
    diag_rmse = [rmse["correlated"]["diag"], rmse["small_sample"]["diag"], rmse["independent"]["diag"]]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    ax4.bar(x - width/2, full_rmse, width, label='完整协方差', color='skyblue')
    ax4.bar(x + width/2, diag_rmse, width, label='对角协方差', color='lightcoral')
    ax4.set_title('协方差方法总体对比', fontsize=12, fontweight='bold')
    ax4.set_ylabel('RMSE')
    ax4.set_xticks(x)
    ax4.set_xticklabels(scenarios)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save figure
    filepath = os.path.join(figures_dir, f'{script_name}_covariance_comparison.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Covariance test figure saved: {filepath}")


def test_covariance_comparison_example(tmp_path):
    output_path = tmp_path / "covariance_methods.png"
    result = run_covariance_comparison(output_path=output_path)
    
    # Generate test visualization
    save_covariance_test_figure(result)

    rmse = result["rmse"]

    # Full covariance should outperform diagonal in correlated regime.
    assert rmse["correlated"]["full"] < rmse["correlated"]["diag"]

    # Shrinkage stabilises the ill-conditioned sample.
    assert rmse["small_sample"]["shrunk"] < rmse["small_sample"]["full"]

    # Independent regime yields nearly identical performance for full and diagonal.
    np.testing.assert_allclose(
        rmse["independent"]["full"],
        rmse["independent"]["diag"],
        rtol=0,
        atol=0.02,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
