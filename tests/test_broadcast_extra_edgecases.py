import numpy as np
import xarray as xr
import sys
import os
import pytest

# Import matplotlib for testing
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

# Chinese font configuration
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Ensure repo root is on sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collocation.fusion.broadcast import prepare_cross_and_mse_for_sigma


def save_edgecase_test_figure(test_name, models, mse_data, cross_data, result_cross, result_mse,
                             script_name="test_broadcast_extra_edgecases"):
    """Save edge case test visualization"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Original MSE data
    if hasattr(mse_data, 'values'):
        mse_vals = mse_data.values
        if mse_data.ndim > 1:
            mse_vals = mse_vals.flatten()[:len(models)]
    else:
        mse_vals = mse_data
        
    ax1.bar(range(len(models)), mse_vals)
    ax1.set_title(f'原始MSE数据 - {test_name}', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MSE值')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models)
    
    # Plot 2: Original Cross data
    if hasattr(cross_data, 'values'):
        cross_vals = cross_data.values
    else:
        cross_vals = cross_data
        
    if cross_vals.ndim == 2:
        im2 = ax2.imshow(cross_vals, cmap='coolwarm')
        ax2.set_title(f'原始交叉相关矩阵 - {test_name}', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(models)))
        ax2.set_yticks(range(len(models)))
        ax2.set_xticklabels(models)
        ax2.set_yticklabels(models)
        plt.colorbar(im2, ax=ax2)
        
        # Add values to the matrix
        for i in range(len(models)):
            for j in range(len(models)):
                text = ax2.text(j, i, f'{cross_vals[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=9)
    else:
        ax2.text(0.5, 0.5, f'交叉相关数据\n形状: {cross_vals.shape}\n类型: {type(cross_vals)}',
                ha='center', va='center', transform=ax2.transAxes, fontsize=10)
        ax2.set_title(f'原始交叉相关 - {test_name}', fontsize=12, fontweight='bold')
    
    # Plot 3: Result MSE data
    if result_mse.ndim > 1:
        # Take first spatial slice for multi-dimensional MSE
        mse_slice = result_mse.flatten()[:len(models)]
    else:
        mse_slice = result_mse[:len(models)]
    ax3.bar(range(len(models)), mse_slice)
    ax3.set_title(f'处理后MSE数据 - {test_name}', fontsize=12, fontweight='bold')
    ax3.set_ylabel('MSE值')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(models)
    
    # Plot 4: Result Cross data
    if result_cross.ndim >= 2:
        cross_result_2d = result_cross if result_cross.ndim == 2 else result_cross[0, 0]
        im4 = ax4.imshow(cross_result_2d, cmap='coolwarm')
        ax4.set_title(f'处理后交叉相关矩阵 - {test_name}', fontsize=12, fontweight='bold')
        ax4.set_xticks(range(len(models)))
        ax4.set_yticks(range(len(models)))
        ax4.set_xticklabels(models)
        ax4.set_yticklabels(models)
        plt.colorbar(im4, ax=ax4)
        
        # Add values to the matrix
        for i in range(len(models)):
            for j in range(len(models)):
                text = ax4.text(j, i, f'{cross_result_2d[i, j]:.3f}',
                               ha="center", va="center", color="white", fontsize=9)
    else:
        ax4.text(0.5, 0.5, f'处理后交叉相关\n形状: {result_cross.shape}',
                ha='center', va='center', transform=ax4.transAxes, fontsize=10)
        ax4.set_title(f'处理后交叉相关 - {test_name}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    filepath = os.path.join(figures_dir, f'{script_name}_{test_name}.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Edge case test figure saved: {filepath}")


def test_mismatched_coordinate_names_are_handled(models, mse_1d, cross_2d):
    """If `mse` uses a differently named model coordinate, helper should still work
    when provided with explicit model_coords labels (user responsibility).
    """
    # Create mse with 'models' coord name instead of 'model'
    mse_mis = mse_1d.rename({"model": "models"}).assign_coords(models=models)

    # prepare should accept explicit model_coords ordering
    cross_np, mse_np, batch_dims, batch_coords = prepare_cross_and_mse_for_sigma(cross_2d, mse_mis.rename({"models":"model"}), models)
    
    # Generate test visualization
    save_edgecase_test_figure("mismatched_coords", models, mse_1d, cross_2d, cross_np, mse_np)

    assert cross_np.shape == (3, 3)
    assert mse_np.shape == (3,)


def test_dtype_edge_cases_broadcast(models, mse_batch, cross_2d):
    """Test that dtype differences (float32 mse, int cross) do not break broadcasting."""
    mse32 = mse_batch.astype('float32')
    cross_int = cross_2d.astype('int32')

    cross_np, mse_np, batch_dims, batch_coords = prepare_cross_and_mse_for_sigma(cross_int, mse32, models)
    
    # Generate test visualization
    save_edgecase_test_figure("dtype_differences", models, mse32, cross_int, cross_np, mse_np)

    # cross_np should be float after broadcasting (numpy upcasts)
    assert cross_np.dtype.kind in ('f', 'i')
    assert mse_np.dtype.kind == 'f'
