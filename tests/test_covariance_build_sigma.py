import numpy as np
import xarray as xr

from collocation.fusion.covariance import build_sigma


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
