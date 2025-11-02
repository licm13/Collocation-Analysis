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


def save_reverse_test_figure(test_name, cross, mse, sigma, script_name="test_covariance_build_sigma_reverse"):
    """Save reverse test visualization"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Create simple visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot MSE data
    ax1.bar(range(len(mse)), mse.values)
    ax1.set_title(f'MSE数据 - {test_name}')
    ax1.set_ylabel('MSE值')
    if hasattr(mse, 'model'):
        ax1.set_xticks(range(len(mse)))
        ax1.set_xticklabels(mse.model.values)
    
    # Plot sigma shape info (first spatial slice)
    sigma_first = sigma.isel(lat=0, lon=0) if sigma.ndim > 2 else sigma
    im2 = ax2.imshow(sigma_first.values, cmap='viridis')
    ax2.set_title(f'Sigma矩阵 (第一个空间点) - {test_name}')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    filepath = os.path.join(figures_dir, f'{script_name}_{test_name}.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Reverse test figure saved: {filepath}")


def test_build_sigma_broadcasts_mse_to_cross_batch_dims():
    """When `cross` has batch dims (lat, lon) but `mse` is 1D (model),
    `build_sigma` should broadcast `mse` to the batch dims and overwrite the
    diagonal of the `cross` values accordingly.
    """

    models = ["A", "B", "C"]
    lat = [0.0, 1.0, 2.0]
    lon = [10.0, 11.0]

    # mse is 1D over model
    mse_vals = np.array([0.1, 0.2, 0.3])
    mse = xr.DataArray(mse_vals, dims=("model",), coords={"model": models})

    # cross has batch dims (lat, lon, model, model_2)
    rng = np.random.RandomState(1)
    cross_vals = rng.randn(len(lat), len(lon), len(models), len(models)) * 0.05
    # ensure symmetric-ish for covariance-like structure
    for i in range(len(models)):
        for j in range(len(models)):
            cross_vals[..., j, i] = cross_vals[..., i, j]

    cross = xr.DataArray(
        cross_vals,
        dims=("lat", "lon", "model", "model_2"),
        coords={"lat": lat, "lon": lon, "model": models, "model_2": models},
    )

    Sigma = build_sigma(mse, cross=cross, shrinkage="none")
    
    # Generate test visualization
    save_reverse_test_figure("mse_to_cross_batch", cross, mse, Sigma)

    # Sigma should preserve batch dims
    assert set(("lat", "lon")).issubset(set(Sigma.dims))
    assert "model" in Sigma.dims and "model_2" in Sigma.dims

    # For each model, diagonal should equal mse broadcasted to (lat, lon)
    for k, m in enumerate(models):
        diag = Sigma.sel(model=m, model_2=m).values
        expected = np.broadcast_to(mse_vals[k], (len(lat), len(lon)))
        np.testing.assert_allclose(diag, expected)

    # Off-diagonal slice should match original cross off-diagonal
    off_ab = Sigma.sel(model="A", model_2="B").values
    original_off_ab = cross.sel(model="A", model_2="B").values
    np.testing.assert_allclose(off_ab, original_off_ab)
