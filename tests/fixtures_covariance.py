import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def models():
    return ["A", "B", "C"]


@pytest.fixture
def lat():
    return [0.0, 1.0]


@pytest.fixture
def lon():
    return [10.0, 11.0]


@pytest.fixture
def mse_1d(models):
    return xr.DataArray(np.array([0.1, 0.2, 0.3]), dims=("model",), coords={"model": models})


@pytest.fixture
def mse_batch(models, lat, lon):
    vals = np.abs(np.random.RandomState(0).randn(len(lat), len(lon), len(models))) + 0.1
    return xr.DataArray(vals, dims=("lat", "lon", "model"), coords={"lat": lat, "lon": lon, "model": models})


@pytest.fixture
def cross_2d(models):
    vals = np.eye(len(models)) * 0.2
    return xr.DataArray(vals, dims=("model", "model_2"), coords={"model": models, "model_2": models})


@pytest.fixture
def cross_batch(models, lat, lon):
    vals = np.zeros((len(lat), len(lon), len(models), len(models)))
    return xr.DataArray(vals, dims=("lat", "lon", "model", "model_2"), coords={"lat": lat, "lon": lon, "model": models, "model_2": models})
