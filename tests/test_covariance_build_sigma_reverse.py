import numpy as np
import xarray as xr

from collocation.fusion.covariance import build_sigma


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
