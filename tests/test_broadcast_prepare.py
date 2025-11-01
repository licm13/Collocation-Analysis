import numpy as np
import xarray as xr

from collocation.fusion.broadcast import prepare_cross_and_mse_for_sigma


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
