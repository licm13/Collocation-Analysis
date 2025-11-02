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


def save_broadcast_test_figure(test_name, cross_original, mse_original, cross_prepared, mse_prepared, 
                              batch_dims, script_name="test_broadcast_prepare"):
    """Save broadcast preparation test visualization"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Original Cross data (if it has batch dims)
    if hasattr(cross_original, 'dims') and len(cross_original.dims) > 2:
        # Show first spatial slice
        cross_slice = cross_original.isel({d: 0 for d in batch_dims if d in cross_original.dims})
        im1 = axes[0, 0].imshow(cross_slice.values, cmap='coolwarm')
        axes[0, 0].set_title(f'原始交叉相关 - {test_name}', fontsize=10, fontweight='bold')
        plt.colorbar(im1, ax=axes[0, 0])
    else:
        # Show as matrix
        if hasattr(cross_original, 'values'):
            im1 = axes[0, 0].imshow(cross_original.values, cmap='coolwarm')
        else:
            im1 = axes[0, 0].imshow(cross_original, cmap='coolwarm')
        axes[0, 0].set_title(f'原始交叉相关 - {test_name}', fontsize=10, fontweight='bold')
        plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: Original MSE data
    if hasattr(mse_original, 'dims') and len(mse_original.dims) > 1:
        # Average over batch dims
        mse_avg = mse_original.mean(dim=[d for d in mse_original.dims if d != 'model'])
        axes[0, 1].bar(range(len(mse_avg)), mse_avg.values)
        axes[0, 1].set_xticks(range(len(mse_avg)))
        axes[0, 1].set_xticklabels(mse_avg.model.values)
    else:
        if hasattr(mse_original, 'values'):
            axes[0, 1].bar(range(len(mse_original)), mse_original.values)
            if hasattr(mse_original, 'model'):
                axes[0, 1].set_xticks(range(len(mse_original)))
                axes[0, 1].set_xticklabels(mse_original.model.values)
        else:
            axes[0, 1].bar(range(len(mse_original)), mse_original)
    axes[0, 1].set_title(f'原始MSE - {test_name}', fontsize=10, fontweight='bold')
    axes[0, 1].set_ylabel('MSE值')
    
    # Plot 3: Batch dimensions info
    axes[0, 2].text(0.1, 0.8, f'测试: {test_name}', fontsize=12, fontweight='bold', transform=axes[0, 2].transAxes)
    axes[0, 2].text(0.1, 0.6, f'批处理维度: {batch_dims}', fontsize=10, transform=axes[0, 2].transAxes)
    if hasattr(cross_original, 'shape'):
        orig_shape = cross_original.shape
    else:
        orig_shape = np.array(cross_original).shape
    axes[0, 2].text(0.1, 0.4, f'原始Cross形状: {orig_shape}', fontsize=10, transform=axes[0, 2].transAxes)
    axes[0, 2].text(0.1, 0.2, f'准备后Cross形状: {cross_prepared.shape}', fontsize=10, transform=axes[0, 2].transAxes)
    axes[0, 2].set_xlim(0, 1)
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].axis('off')
    
    # Plot 4: Prepared Cross data (first slice)
    cross_slice = cross_prepared[0, 0] if cross_prepared.ndim > 2 else cross_prepared
    im4 = axes[1, 0].imshow(cross_slice, cmap='coolwarm')
    axes[1, 0].set_title('准备后交叉相关 (第一个空间点)', fontsize=10, fontweight='bold')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Plot 5: Prepared MSE data (first slice)
    mse_slice = mse_prepared[0, 0] if mse_prepared.ndim > 1 else mse_prepared
    axes[1, 1].bar(range(len(mse_slice)), mse_slice)
    axes[1, 1].set_title('准备后MSE (第一个空间点)', fontsize=10, fontweight='bold')
    axes[1, 1].set_ylabel('MSE值')
    axes[1, 1].set_xlabel('模型索引')
    
    # Plot 6: Shape comparison
    shapes_info = [
        f"Original Cross: {orig_shape}",
        f"Original MSE: {mse_original.shape if hasattr(mse_original, 'shape') else len(mse_original)}",
        f"Prepared Cross: {cross_prepared.shape}",
        f"Prepared MSE: {mse_prepared.shape}",
        f"Batch dims: {batch_dims}"
    ]
    
    for i, info in enumerate(shapes_info):
        axes[1, 2].text(0.1, 0.8 - i*0.15, info, fontsize=9, transform=axes[1, 2].transAxes)
    axes[1, 2].set_title('形状变换信息', fontsize=10, fontweight='bold')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    filepath = os.path.join(figures_dir, f'{script_name}_{test_name}.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Broadcast test figure saved: {filepath}")


def test_prepare_cross_cross2d_mse_batch():
    models = ["A", "B", "C"]
    lat = [0.0, 1.0]
    lon = [10.0, 11.0]

    # mse with batch dims
    mse_vals = np.abs(np.random.randn(len(lat), len(lon), len(models))) + 0.1
    mse = xr.DataArray(mse_vals, dims=("lat", "lon", "model"), coords={"lat": lat, "lon": lon, "model": models})

    # cross 2D
    cross = np.eye(3) * 0.2

    cross_np, mse_np, batch_dims, batch_coords = prepare_cross_and_mse_for_sigma(cross, mse, models)
    
    # Generate test visualization
    save_broadcast_test_figure("cross2d_mse_batch", cross, mse, cross_np, mse_np, batch_dims)

    assert cross_np.shape == (len(lat), len(lon), 3, 3)
    assert mse_np.shape == (len(lat), len(lon), 3)
    assert tuple(batch_dims) == ("lat", "lon")


def test_prepare_cross_crossbatch_mse1d():
    models = ["A", "B", "C"]
    lat = [0.0, 1.0, 2.0]
    lon = [10.0]

    # mse 1D
    mse = xr.DataArray(np.array([0.1, 0.2, 0.3]), dims=("model",), coords={"model": models})

    # cross has batch dims
    cross_vals = np.zeros((len(lat), len(lon), 3, 3))
    cross = xr.DataArray(cross_vals, dims=("lat", "lon", "model", "model_2"), coords={"lat": lat, "lon": lon, "model": models, "model_2": models})

    cross_np, mse_np, batch_dims, batch_coords = prepare_cross_and_mse_for_sigma(cross, mse, models)

    assert cross_np.shape == (len(lat), len(lon), 3, 3)
    assert mse_np.shape == (len(lat), len(lon), 3)
    assert tuple(batch_dims) == ("lat", "lon")


def test_prepare_cross_reindexing():
    models_mse = ["A", "B", "C"]
    models_cross = ["C", "A", "B"]

    mse = xr.DataArray(np.array([0.1, 0.2, 0.3]), dims=("model",), coords={"model": models_mse})

    cross_vals = np.zeros((3, 3))
    cross_vals[0, 1] = 0.05  # C->A
    cross = xr.DataArray(cross_vals, dims=("model", "model_2"), coords={"model": models_cross, "model_2": models_cross})

    cross_np, mse_np, batch_dims, batch_coords = prepare_cross_and_mse_for_sigma(cross, mse, models_mse)

    # After reindexing, the A->B position should reflect the value placed
    # at cross coordinate ('A','B') when reindexed to mse order
    reindexed = cross.reindex(model=models_mse, model_2=models_mse).values
    assert cross_np.shape == (3, 3)
    assert np.allclose(cross_np, reindexed)
