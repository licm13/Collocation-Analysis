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


def save_edgecase_figure(test_name, cross, mse, sigma, script_name="test_covariance_build_sigma_edgecases"):
    """Save edge case test visualization"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Create simple visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot cross matrix
    im1 = ax1.imshow(cross.values, cmap='coolwarm')
    ax1.set_title(f'交叉相关矩阵 - {test_name}')
    plt.colorbar(im1, ax=ax1)
    
    # Plot sigma diagonal
    sigma_diag = np.diag(sigma.values) if sigma.ndim == 2 else [sigma.sel(model=m, model_2=m).values for m in sigma.model.values]
    ax2.bar(range(len(sigma_diag)), sigma_diag)
    ax2.set_title(f'Sigma对角值 - {test_name}')
    ax2.set_ylabel('值')
    
    plt.tight_layout()
    filepath = os.path.join(figures_dir, f'{script_name}_{test_name}.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Edge case test figure saved: {filepath}")


def test_cross_with_reversed_model_dims_is_aligned():
    """If cross has dims ('model_2','model'), build_sigma should handle it by
    transposing to ('model','model_2') and produce correct diagonal and off-diagonals.
    """
    models = ["A", "B", "C"]
    mse = xr.DataArray(np.array([0.1, 0.2, 0.3]), dims=("model",), coords={"model": models})

    # cross provided with reversed dims order
    cross_vals = np.array([[0.2, 0.01, 0.01], [0.01, 0.2, 0.01], [0.01, 0.01, 0.2]])
    cross = xr.DataArray(cross_vals.T, dims=("model_2", "model"), coords={"model": models, "model_2": models})
    
    Sigma = build_sigma(mse, cross=cross, shrinkage="none")
    
    # Generate test visualization
    save_edgecase_figure("reversed_dims", cross, mse, Sigma)    # diagonal should equal mse
    for m in models:
        np.testing.assert_allclose(Sigma.sel(model=m, model_2=m).values, mse.sel(model=m).values)

    # off-diagonal A,B should match original cross A,B
    np.testing.assert_allclose(Sigma.sel(model="A", model_2="B").values, 0.01)


def test_cross_with_different_model_order_is_reindexed_to_mse():
    """If cross uses a different model ordering, build_sigma should reindex it
    to match mse's model ordering by label before constructing Sigma.
    """
    models_mse = ["A", "B", "C"]
    models_cross = ["C", "A", "B"]

    mse = xr.DataArray(np.array([0.1, 0.2, 0.3]), dims=("model",), coords={"model": models_mse})

    # cross has model dims but different ordering of labels
    cross_vals = np.zeros((3, 3))
    cross_vals[0, 1] = 0.05  # C->A
    cross_vals[1, 2] = 0.06  # A->B
    cross_vals[2, 0] = 0.07  # B->C

    cross = xr.DataArray(cross_vals, dims=("model", "model_2"), coords={"model": models_cross, "model_2": models_cross})

    Sigma = build_sigma(mse, cross=cross, shrinkage="none")

    # After reindexing to mse order, check that the A->B entry corresponds to value placed at
    # cross coordinate ('A','B') originally (which was at index (1,2) in models_cross)
    # We expect Sigma.sel(model='A', model_2='B') to equal cross.sel(model='A', model_2='B') after reindexing
    expected_ab = cross.reindex(model=models_mse, model_2=models_mse).sel(model="A", model_2="B").values
    np.testing.assert_allclose(Sigma.sel(model="A", model_2="B").values, expected_ab)


def test_non_square_cross_with_missing_labels_results_in_nans():
    """If cross is non-square by label (missing model_2 labels), build_sigma will
    reindex cross to mse model labels; missing entries become NaN in Sigma's
    off-diagonals (diagonal is still replaced by mse values).
    """
    models = ["A", "B", "C"]
    mse = xr.DataArray(np.array([0.1, 0.2, 0.3]), dims=("model",), coords={"model": models})

    # Create a non-square cross (3 x 2) with model_2 labels that don't match mse
    cross_vals = np.random.randn(3, 2)
    cross = xr.DataArray(cross_vals, dims=("model", "model_2"), coords={"model": models, "model_2": ["X", "Y"]})

    Sigma = build_sigma(mse, cross=cross, shrinkage="none")

    # Diagonal must equal mse (overwritten)
    for m in models:
        np.testing.assert_allclose(Sigma.sel(model=m, model_2=m).values, mse.sel(model=m).values)

    # Off-diagonal for model_2='C' (which wasn't present in original cross) should be NaN
    assert np.isnan(Sigma.sel(model="A", model_2="C").values)
