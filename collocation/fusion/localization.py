"""
Localization Strategies for Data Fusion
========================================

Implements spatial/temporal localization for heterogeneous error fields:

1. **Moving window**: Local MSE/covariance estimation in space/time
2. **Biome partitioning**: Separate weights per ecosystem/climate zone
3. **Climate-based**: Partition by Köppen-Geiger or aridity index

Localization improves fusion when error structure varies spatially.

Examples
--------
>>> import numpy as np
>>> import xarray as xr
>>> from collocation.fusion import apply_moving_window, partition_by_biome
>>>
>>> # Moving window MSE estimation
>>> data = xr.DataArray(
...     np.random.randn(100, 50, 50),
...     dims=["time", "lat", "lon"],
... )
>>> window_spec = {"time": 30, "space": 7}
>>> local_mean = apply_moving_window(data, window_spec, reduction="mean")
>>>
>>> # Biome-based partitioning
>>> biome_map = xr.DataArray(
...     np.random.randint(0, 5, (50, 50)),
...     dims=["lat", "lon"],
... )
>>> partitions = partition_by_biome(data, biome_map)
"""

from __future__ import annotations

from typing import Callable, Dict, Literal

import numpy as np
import xarray as xr


def apply_moving_window(
    data: xr.DataArray,
    window: Dict[str, int | str],
    reduction: Literal["mean", "var", "std", "median"] | Callable = "mean",
    min_samples: int = 10,
) -> xr.DataArray:
    """
    Apply moving window reduction along specified dimensions.

    Enables local estimation of statistics (MSE, variance, etc.) that vary
    in space/time.

    Parameters
    ----------
    data : xr.DataArray
        Input data with dimensions to window over.
    window : dict
        Window specification. Keys are dimension names or "space" (for lat/lon).
        Values are window sizes.
        Examples:
        - {"time": 30}: 30-timestep window
        - {"space": 7}: 7x7 spatial window (lat/lon)
        - {"time": "30D"}: 30-day window (requires datetime index)
    reduction : {"mean", "var", "std", "median"} or callable, default="mean"
        Reduction operation to apply within each window.
    min_samples : int, default=10
        Minimum number of valid samples required per window.

    Returns
    -------
    xr.DataArray
        Windowed result, same shape as input.

    Examples
    --------
    >>> data = xr.DataArray(
    ...     np.random.randn(100, 50, 50),
    ...     dims=["time", "lat", "lon"],
    ... )
    >>> # 7x7 spatial window
    >>> smoothed = apply_moving_window(data, {"space": 7}, reduction="mean")
    """
    result = data.copy()

    for dim_key, window_size in window.items():
        if dim_key == "space":
            # Apply to both lat and lon
            if "lat" in result.dims and "lon" in result.dims:
                result = _apply_spatial_window(
                    result,
                    window_size=int(window_size),
                    reduction=reduction,
                    min_samples=min_samples,
                )
            else:
                raise ValueError("'space' window requires 'lat' and 'lon' dimensions")

        elif dim_key in result.dims:
            # Single dimension window
            result = _apply_1d_window(
                result,
                dim=dim_key,
                window_size=window_size,
                reduction=reduction,
                min_samples=min_samples,
            )
        else:
            raise ValueError(f"Dimension '{dim_key}' not found in data")

    return result


def _apply_1d_window(
    data: xr.DataArray,
    dim: str,
    window_size: int | str,
    reduction: str | Callable,
    min_samples: int,
) -> xr.DataArray:
    """Apply 1D moving window along a single dimension."""
    # Use xarray's rolling window
    if isinstance(window_size, str):
        # Time-based window (e.g., "30D")
        window_obj = data.rolling({dim: window_size}, center=True, min_periods=min_samples)
    else:
        # Index-based window
        window_obj = data.rolling({dim: int(window_size)}, center=True, min_periods=min_samples)

    # Apply reduction
    if isinstance(reduction, str):
        if reduction == "mean":
            result = window_obj.mean()
        elif reduction == "var":
            result = window_obj.var()
        elif reduction == "std":
            result = window_obj.std()
        elif reduction == "median":
            result = window_obj.median()
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    elif callable(reduction):
        result = window_obj.reduce(reduction)
    else:
        raise TypeError("reduction must be string or callable")

    return result


def _apply_spatial_window(
    data: xr.DataArray,
    window_size: int,
    reduction: str | Callable,
    min_samples: int,
) -> xr.DataArray:
    """Apply 2D spatial moving window over lat/lon."""
    # Use coarsen for rectangular windows
    # For circular windows, would need custom implementation

    # For simplicity, use rectangular window via rolling
    window_dict = {"lat": window_size, "lon": window_size}
    window_obj = data.rolling(window_dict, center=True, min_periods=min_samples)

    if isinstance(reduction, str):
        if reduction == "mean":
            result = window_obj.mean()
        elif reduction == "var":
            result = window_obj.var()
        elif reduction == "std":
            result = window_obj.std()
        elif reduction == "median":
            result = window_obj.median()
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    elif callable(reduction):
        result = window_obj.reduce(reduction)
    else:
        raise TypeError("reduction must be string or callable")

    return result


def partition_by_biome(
    data: xr.DataArray,
    biome_map: xr.DataArray,
    biome_dim: str = "biome",
) -> Dict[int, xr.DataArray]:
    """
    Partition data by biome/land-cover type.

    Enables separate fusion weights per ecosystem type.

    Parameters
    ----------
    data : xr.DataArray
        Input data with spatial dimensions.
    biome_map : xr.DataArray
        Integer biome IDs, matching spatial dims of data.
    biome_dim : str, default="biome"
        Name for biome dimension in output.

    Returns
    -------
    dict of {int: xr.DataArray}
        Dictionary mapping biome ID to data subset.

    Examples
    --------
    >>> data = xr.DataArray(
    ...     np.random.randn(100, 50, 50, 3),
    ...     dims=["time", "lat", "lon", "model"],
    ... )
    >>> biome_map = xr.DataArray(
    ...     np.random.randint(0, 5, (50, 50)),
    ...     dims=["lat", "lon"],
    ... )
    >>> partitions = partition_by_biome(data, biome_map)
    >>> len(partitions)
    5
    """
    # Broadcast biome_map to match data dimensions
    biome_ids = np.unique(biome_map.values)
    biome_ids = biome_ids[~np.isnan(biome_ids)]  # Remove NaN

    partitions = {}

    for biome_id in biome_ids:
        # Mask for this biome
        mask = biome_map == biome_id

        # Apply mask and extract subset
        # Use where to mask, then dropna
        subset = data.where(mask, drop=False)

        partitions[int(biome_id)] = subset

    return partitions


def partition_by_climate(
    data: xr.DataArray,
    climate_zones: xr.DataArray | None = None,
    aridity_index: xr.DataArray | None = None,
    n_zones: int = 5,
) -> Dict[str, xr.DataArray]:
    """
    Partition data by climate zones.

    Parameters
    ----------
    data : xr.DataArray
        Input data with spatial dimensions.
    climate_zones : xr.DataArray, optional
        Predefined climate zone IDs (e.g., Köppen-Geiger classification).
    aridity_index : xr.DataArray, optional
        Aridity index (P/PET). Will be binned into n_zones categories.
    n_zones : int, default=5
        Number of zones if using aridity_index binning.

    Returns
    -------
    dict of {str: xr.DataArray}
        Dictionary mapping zone name to data subset.

    Examples
    --------
    >>> data = xr.DataArray(
    ...     np.random.randn(100, 50, 50),
    ...     dims=["time", "lat", "lon"],
    ... )
    >>> aridity = xr.DataArray(
    ...     np.random.rand(50, 50),
    ...     dims=["lat", "lon"],
    ... )
    >>> partitions = partition_by_climate(data, aridity_index=aridity, n_zones=3)
    """
    if climate_zones is not None:
        # Use predefined zones
        zone_ids = np.unique(climate_zones.values)
        zone_ids = zone_ids[~np.isnan(zone_ids)]

        partitions = {}
        for zone_id in zone_ids:
            mask = climate_zones == zone_id
            subset = data.where(mask, drop=False)
            partitions[f"zone_{int(zone_id)}"] = subset

    elif aridity_index is not None:
        # Bin aridity index
        # Aridity categories (approximate):
        # < 0.05: hyper-arid
        # 0.05-0.2: arid
        # 0.2-0.5: semi-arid
        # 0.5-0.65: dry sub-humid
        # > 0.65: humid

        if n_zones == 5:
            bins = [0, 0.05, 0.2, 0.5, 0.65, np.inf]
            labels = ["hyper_arid", "arid", "semi_arid", "dry_subhumid", "humid"]
        else:
            # Equal-width binning
            ai_min = float(aridity_index.min())
            ai_max = float(aridity_index.max())
            bins = np.linspace(ai_min, ai_max, n_zones + 1)
            labels = [f"zone_{i}" for i in range(n_zones)]

        # Digitize
        ai_values = aridity_index.values.ravel()
        zone_indices = np.digitize(ai_values, bins) - 1
        zone_map = xr.DataArray(
            zone_indices.reshape(aridity_index.shape),
            dims=aridity_index.dims,
            coords=aridity_index.coords,
        )

        partitions = {}
        for i, label in enumerate(labels):
            mask = zone_map == i
            subset = data.where(mask, drop=False)
            partitions[label] = subset

    else:
        raise ValueError("Must provide either climate_zones or aridity_index")

    return partitions


def compute_distance_matrix(
    lat: np.ndarray,
    lon: np.ndarray,
    metric: Literal["haversine", "euclidean"] = "haversine",
) -> np.ndarray:
    """
    Compute pairwise distance matrix for spatial locations.

    Useful for covariance tapering based on distance.

    Parameters
    ----------
    lat : np.ndarray
        Latitude values, shape (N,).
    lon : np.ndarray
        Longitude values, shape (N,).
    metric : {"haversine", "euclidean"}, default="haversine"
        Distance metric:
        - "haversine": Great-circle distance on sphere (km)
        - "euclidean": Euclidean distance in lat/lon degrees

    Returns
    -------
    np.ndarray
        Distance matrix, shape (N, N).

    Examples
    --------
    >>> lat = np.array([40.0, 41.0, 42.0])
    >>> lon = np.array([-75.0, -75.0, -75.0])
    >>> dist = compute_distance_matrix(lat, lon, metric="haversine")
    >>> dist[0, 1]  # Distance between point 0 and 1
    111.19...
    """
    lat = np.asarray(lat).ravel()
    lon = np.asarray(lon).ravel()

    if len(lat) != len(lon):
        raise ValueError("lat and lon must have same length")

    N = len(lat)

    if metric == "euclidean":
        # Simple Euclidean in degree space
        lat_grid, lat_grid_T = np.meshgrid(lat, lat, indexing="ij")
        lon_grid, lon_grid_T = np.meshgrid(lon, lon, indexing="ij")

        dist = np.sqrt((lat_grid - lat_grid_T) ** 2 + (lon_grid - lon_grid_T) ** 2)

    elif metric == "haversine":
        # Great-circle distance
        # Convert to radians
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)

        # Pairwise differences
        lat1_grid, lat2_grid = np.meshgrid(lat_rad, lat_rad, indexing="ij")
        lon1_grid, lon2_grid = np.meshgrid(lon_rad, lon_rad, indexing="ij")

        dlat = lat2_grid - lat1_grid
        dlon = lon2_grid - lon1_grid

        # Haversine formula
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_grid) * np.cos(lat2_grid) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        # Earth radius in km
        R = 6371.0
        dist = R * c

    else:
        raise ValueError(f"Unknown metric: {metric}")

    return dist


def localize_weights(
    X: xr.DataArray,
    Sigma: xr.DataArray,
    strategy: Literal["window", "biome", "climate"] = "window",
    **kwargs,
) -> xr.DataArray:
    """
    Compute localized fusion weights.

    Instead of global weights, computes spatially/temporally varying weights
    based on local error characteristics.

    Parameters
    ----------
    X : xr.DataArray
        Input data, shape (..., model).
    Sigma : xr.DataArray
        Covariance matrix, shape (..., model, model_2).
        Can be global or spatially varying.
    strategy : {"window", "biome", "climate"}, default="window"
        Localization strategy:
        - "window": Moving window weights
        - "biome": Per-biome weights
        - "climate": Per-climate-zone weights
    **kwargs
        Additional arguments passed to strategy-specific functions.

    Returns
    -------
    xr.DataArray
        Localized weights, shape (..., model).

    Examples
    --------
    >>> X = xr.DataArray(
    ...     np.random.randn(100, 50, 50, 3),
    ...     dims=["time", "lat", "lon", "model"],
    ... )
    >>> Sigma = xr.DataArray(
    ...     np.tile(np.eye(3), (50, 50, 1, 1)),
    ...     dims=["lat", "lon", "model", "model_2"],
    ... )
    >>> weights = localize_weights(X, Sigma, strategy="window", window={"space": 7})
    """
    from .weights import solve_weights_gls

    if strategy == "window":
        # Local GLS weights via moving window
        window_spec = kwargs.get("window", {"space": 7})

        # Apply window to Sigma to get local covariance
        # This is simplified; in practice, would recompute Sigma in each window
        Sigma_local = apply_moving_window(Sigma, window_spec, reduction="mean")

        # Compute weights pixel-by-pixel
        # For simplicity, assume Sigma is already local
        # Full implementation would recompute from local data

        # Extract numpy
        Sigma_np = Sigma_local.values  # (..., K, K)
        weights_np = np.zeros((*Sigma_np.shape[:-2], Sigma_np.shape[-1]))

        # Solve for each pixel
        for idx in np.ndindex(Sigma_np.shape[:-2]):
            S = Sigma_np[idx]
            w = solve_weights_gls(S)
            weights_np[idx] = w

        # Wrap in xarray
        weight_dims = [d for d in Sigma_local.dims if d not in ["model_2"]]
        weights = xr.DataArray(
            weights_np,
            dims=weight_dims,
            coords={d: Sigma_local.coords[d] for d in weight_dims},
        )

    elif strategy in {"biome", "climate"}:
        # Per-partition weights
        raise NotImplementedError(f"Strategy '{strategy}' not yet implemented")

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return weights
