import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from collocation.covariance import build_sigma_from_collocation
from collocation.fuse import estimate_bias_from_collocation


def _make_dataset():
    models = ["m0", "m1", "m2"]
    var = xr.DataArray([1.0, 2.0, 3.0], dims=("model",), coords={"model": models})
    cov = xr.DataArray(
        [[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]],
        dims=("model", "model"),
        coords={"model": models},
    )
    bias = xr.DataArray([0.1, -0.2, 0.0], dims=("model",), coords={"model": models})
    return xr.Dataset({"var": var, "cov": cov, "bias": bias})


def test_build_sigma_from_collocation_defaults():
    ds = _make_dataset()
    sigma = build_sigma_from_collocation(ds)
    np.testing.assert_allclose(sigma.values, ds["cov"].values)
    assert sigma.dims == ("model", "model")


def test_build_sigma_from_collocation_mapping_reorder():
    ds = _make_dataset()
    sigma = build_sigma_from_collocation(ds, mapping={"model": ["m2", "m0", "m1"]})
    assert list(sigma["model"].values) == ["m2", "m0", "m1"]


def test_build_sigma_from_collocation_ridge():
    ds = _make_dataset()
    sigma = build_sigma_from_collocation(ds, shrinkage="ridge", lam=0.5)
    expected = ds["cov"].values + 0.5 * np.eye(3)
    np.testing.assert_allclose(sigma.values, expected)


def test_build_sigma_from_variance_only():
    ds = _make_dataset().drop_vars("cov")
    sigma = build_sigma_from_collocation(ds)
    expected = np.diag(ds["var"].values)
    np.testing.assert_allclose(sigma.values, expected)


def test_estimate_bias_from_collocation_prefers_bias():
    ds = _make_dataset()
    bias = estimate_bias_from_collocation(ds)
    np.testing.assert_allclose(bias.values, ds["bias"].values)


def test_estimate_bias_from_collocation_zero_fallback():
    ds = _make_dataset().drop_vars("bias")
    bias = estimate_bias_from_collocation(ds)
    np.testing.assert_allclose(bias.values, np.zeros(3))

