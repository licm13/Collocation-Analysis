import numpy as np
import xarray as xr

from collocation.fusion.covariance import build_sigma


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

    # diagonal should equal mse
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
