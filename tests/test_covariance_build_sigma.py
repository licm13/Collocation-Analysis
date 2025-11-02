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

from collocation.fusion.covariance import build_sigma


def save_sigma_build_test_figure(mse, cross, Sigma, script_name="test_covariance_build_sigma"):
    """Save sigma build test visualization"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: MSE heatmap
    mse_avg = mse.mean(dim=['lat', 'lon'])
    ax1.bar(range(len(mse_avg)), mse_avg.values)
    ax1.set_title('各模型平均MSE', fontsize=12, fontweight='bold')
    ax1.set_xlabel('模型')
    ax1.set_ylabel('MSE值')
    ax1.set_xticks(range(len(mse_avg)))
    ax1.set_xticklabels(mse_avg.model.values)
    
    # Plot 2: Cross-correlation matrix
    im2 = ax2.imshow(cross.values, cmap='coolwarm', vmin=-0.5, vmax=0.5)
    ax2.set_title('交叉相关矩阵', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(cross.model)))
    ax2.set_yticks(range(len(cross.model_2)))
    ax2.set_xticklabels(cross.model.values)
    ax2.set_yticklabels(cross.model_2.values)
    plt.colorbar(im2, ax=ax2, label='相关系数')
    
    # Add values to cross-correlation matrix
    for i in range(len(cross.model)):
        for j in range(len(cross.model_2)):
            text = ax2.text(j, i, f'{cross.values[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=9)
    
    # Plot 3: Sigma diagonal values (first spatial point)
    sigma_diag = [Sigma.sel(model=m, model_2=m).isel(lat=0, lon=0).values 
                  for m in Sigma.model.values]
    mse_first = [mse.sel(model=m).isel(lat=0, lon=0).values 
                 for m in mse.model.values]
    
    x = np.arange(len(Sigma.model))
    width = 0.35
    ax3.bar(x - width/2, sigma_diag, width, label='Sigma对角值', alpha=0.8)
    ax3.bar(x + width/2, mse_first, width, label='原始MSE值', alpha=0.8)
    ax3.set_title('Sigma对角值 vs 原始MSE (第一个空间点)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('模型')
    ax3.set_ylabel('值')
    ax3.set_xticks(x)
    ax3.set_xticklabels(Sigma.model.values)
    ax3.legend()
    
    # Plot 4: Sigma matrix (first spatial point)
    sigma_first = Sigma.isel(lat=0, lon=0)
    im4 = ax4.imshow(sigma_first.values, cmap='viridis', aspect='equal')
    ax4.set_title('Sigma矩阵 (第一个空间点)', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(len(sigma_first.model)))
    ax4.set_yticks(range(len(sigma_first.model_2)))
    ax4.set_xticklabels(sigma_first.model.values)
    ax4.set_yticklabels(sigma_first.model_2.values)
    plt.colorbar(im4, ax=ax4, label='协方差值')
    
    # Add values to sigma matrix
    for i in range(len(sigma_first.model)):
        for j in range(len(sigma_first.model_2)):
            text = ax4.text(j, i, f'{sigma_first.values[i, j]:.3f}',
                           ha="center", va="center", color="white", fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    filepath = os.path.join(figures_dir, f'{script_name}_sigma_build.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Sigma build test figure saved: {filepath}")


def test_build_sigma_broadcasts_cross_to_mse_batch_dims():
    """When `mse` has batch dims (lat, lon) but `cross` is 2D (model, model_2),
    `build_sigma` should broadcast `cross` to the batch dims and overwrite the
    diagonal with the per-model `mse` values."""

    models = ["A", "B", "C"]
    lat = [10.0, 20.0]
    lon = [100.0, 110.0]

    # Create mse with dims (lat, lon, model)
    rng = np.random.RandomState(0)
    mse_vals = np.abs(rng.randn(len(lat), len(lon), len(models))) + 0.1
    mse = xr.DataArray(
        mse_vals,
        dims=("lat", "lon", "model"),
        coords={"lat": lat, "lon": lon, "model": models},
    )

    # cross is a plain 2D matrix (model, model_2)
    cross_vals = np.array([[0.2, 0.01, 0.01], [0.01, 0.2, 0.01], [0.01, 0.01, 0.2]])
    cross = xr.DataArray(cross_vals, dims=("model", "model_2"), coords={"model": models, "model_2": models})

    # Build Sigma with no shrinkage so values should be preserved
    Sigma = build_sigma(mse, cross=cross, shrinkage="none")
    
    # Generate test visualization
    save_sigma_build_test_figure(mse, cross, Sigma)

    # Expect batch dims preserved and model dims present
    assert "model" in Sigma.dims
    assert "model_2" in Sigma.dims
    # The batch dims should match mse's batch dims
    for d in ("lat", "lon"):
        assert d in Sigma.dims

    # Diagonal of Sigma (for each model) should equal mse for that model
    for m in models:
        diag = Sigma.sel(model=m, model_2=m)
        mse_sel = mse.sel(model=m)
        np.testing.assert_allclose(diag.values, mse_sel.values)

    # Off-diagonal entries should equal the cross values broadcast to batch dims
    off_ab = Sigma.sel(model="A", model_2="B").values
    assert off_ab.shape == (len(lat), len(lon))
    np.testing.assert_allclose(off_ab, cross.sel(model="A", model_2="B").values)
