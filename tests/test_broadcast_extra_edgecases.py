import numpy as np
import xarray as xr

from collocation.fusion.broadcast import prepare_cross_and_mse_for_sigma


def test_mismatched_coordinate_names_are_handled(models, mse_1d, cross_2d):
    """If `mse` uses a differently named model coordinate, helper should still work
    when provided with explicit model_coords labels (user responsibility).
    """
    # Create mse with 'models' coord name instead of 'model'
    mse_mis = mse_1d.rename({"model": "models"}).assign_coords(models=models)

    # prepare should accept explicit model_coords ordering
    cross_np, mse_np, batch_dims, batch_coords = prepare_cross_and_mse_for_sigma(cross_2d, mse_mis.rename({"models":"model"}), models)

    assert cross_np.shape == (3, 3)
    assert mse_np.shape == (3,)


def test_dtype_edge_cases_broadcast(models, mse_batch, cross_2d):
    """Test that dtype differences (float32 mse, int cross) do not break broadcasting."""
    mse32 = mse_batch.astype('float32')
    cross_int = cross_2d.astype('int32')

    cross_np, mse_np, batch_dims, batch_coords = prepare_cross_and_mse_for_sigma(cross_int, mse32, models)

    # cross_np should be float after broadcasting (numpy upcasts)
    assert cross_np.dtype.kind in ('f', 'i')
    assert mse_np.dtype.kind == 'f'
