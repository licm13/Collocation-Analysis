"""
Utility Functions for HCC-ET Framework
=======================================

Provides helper functions for:
- Trend analysis (Theil-Sen, Mann-Kendall)
- Spatial statistics
- Visualization
- Performance metrics

Author: [Your Name]
Date: 2025-11-09
"""

import logging
from typing import Tuple, Optional

import numpy as np
import xarray as xr
from scipy import stats
from scipy.stats import theilslopes, kendalltau

logger = logging.getLogger(__name__)


def compute_trend_pvalue(
    timeseries: np.ndarray,
    slope: float,
    method: str = 'mann_kendall'
) -> float:
    """
    Compute p-value for trend significance.

    Parameters
    ----------
    timeseries : np.ndarray
        Time series data
    slope : float
        Trend slope (from Theil-Sen or OLS)
    method : str
        Test method: 'mann_kendall', 'spearman'

    Returns
    -------
    float
        P-value (significance of trend)

    Examples
    --------
    >>> ts = np.array([1, 2, 3.5, 3.8, 5.2, 6.1, 7.3])
    >>> result = theilslopes(ts, np.arange(len(ts)))
    >>> pval = compute_trend_pvalue(ts, result.slope)
    >>> print(f"p-value: {pval:.4f}")
    """
    if method == 'mann_kendall':
        # Mann-Kendall test
        tau, pval = kendalltau(np.arange(len(timeseries)), timeseries)
        return pval

    elif method == 'spearman':
        # Spearman rank correlation
        rho, pval = stats.spearmanr(np.arange(len(timeseries)), timeseries)
        return pval

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_spatial_trends(
    data: xr.DataArray,
    dim: str = 'time',
    method: str = 'theil_sen'
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Compute pixel-wise trends over a specified dimension.

    Parameters
    ----------
    data : xr.DataArray
        Input data with dimensions including `dim`
    dim : str, default='time'
        Dimension over which to compute trends
    method : str, default='theil_sen'
        Trend estimation method: 'theil_sen', 'ols'

    Returns
    -------
    slopes : xr.DataArray
        Trend slopes (per unit of dim)
    pvalues : xr.DataArray
        P-values for trend significance

    Examples
    --------
    >>> data = xr.DataArray(
    ...     np.random.randn(100, 50, 50) + np.arange(100)[:, None, None] * 0.01,
    ...     dims=['time', 'lat', 'lon']
    ... )
    >>> slopes, pvals = compute_spatial_trends(data)
    >>> print(f"Mean trend: {slopes.mean().values:.4f}")
    """
    logger.info(f"Computing spatial trends using {method}...")

    # Get coordinates for trend dimension
    x = np.arange(data.sizes[dim])

    # Initialize output arrays
    other_dims = [d for d in data.dims if d != dim]
    shape = tuple(data.sizes[d] for d in other_dims)
    slopes = np.full(shape, np.nan)
    pvalues = np.full(shape, np.nan)

    # Compute trends pixel by pixel
    for idx in np.ndindex(shape):
        # Extract time series
        ts = data.isel(dict(zip(other_dims, idx))).values

        # Skip if insufficient valid data
        valid_mask = ~np.isnan(ts)
        if np.sum(valid_mask) < 5:
            continue

        x_valid = x[valid_mask]
        ts_valid = ts[valid_mask]

        # Compute trend
        if method == 'theil_sen':
            result = theilslopes(ts_valid, x_valid)
            slopes[idx] = result.slope
            pvalues[idx] = compute_trend_pvalue(ts_valid, result.slope)

        elif method == 'ols':
            # Ordinary least squares
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, ts_valid)
            slopes[idx] = slope
            pvalues[idx] = p_value

    # Convert to xarray
    coords = {d: data.coords[d] for d in other_dims}

    slopes_xr = xr.DataArray(
        slopes,
        dims=other_dims,
        coords=coords,
        name='trend_slope'
    )

    pvalues_xr = xr.DataArray(
        pvalues,
        dims=other_dims,
        coords=coords,
        name='trend_pvalue'
    )

    logger.info(f"  Trends computed for {np.sum(~np.isnan(slopes))} pixels")

    return slopes_xr, pvalues_xr


def compute_regional_statistics(
    data: xr.DataArray,
    regions: xr.DataArray,
    statistic: str = 'mean'
) -> xr.DataArray:
    """
    Compute statistics aggregated by regions (e.g., land cover classes, basins).

    Parameters
    ----------
    data : xr.DataArray
        Input data with spatial dimensions
    regions : xr.DataArray
        Region classification (integer codes)
    statistic : str, default='mean'
        Statistic to compute: 'mean', 'median', 'std', 'sum'

    Returns
    -------
    xr.DataArray
        Regional statistics, dims: (region_id, ...)

    Examples
    --------
    >>> data = xr.DataArray(
    ...     np.random.randn(100, 50, 50),
    ...     dims=['time', 'lat', 'lon']
    ... )
    >>> regions = xr.DataArray(
    ...     np.random.randint(0, 5, (50, 50)),
    ...     dims=['lat', 'lon']
    ... )
    >>> regional_mean = compute_regional_statistics(data, regions, statistic='mean')
    """
    logger.info(f"Computing regional {statistic}...")

    # Get unique region IDs
    region_ids = np.unique(regions.values[~np.isnan(regions.values)])

    # Initialize output
    results = []

    for region_id in region_ids:
        # Mask for this region
        mask = (regions == region_id)

        # Extract data for this region
        data_region = data.where(mask, drop=False)

        # Compute statistic
        if statistic == 'mean':
            stat_value = data_region.mean(dim=['lat', 'lon'], skipna=True)
        elif statistic == 'median':
            stat_value = data_region.median(dim=['lat', 'lon'], skipna=True)
        elif statistic == 'std':
            stat_value = data_region.std(dim=['lat', 'lon'], skipna=True)
        elif statistic == 'sum':
            stat_value = data_region.sum(dim=['lat', 'lon'], skipna=True)
        else:
            raise ValueError(f"Unknown statistic: {statistic}")

        # Add region dimension
        stat_value = stat_value.expand_dims(region_id=[int(region_id)])
        results.append(stat_value)

    # Concatenate
    combined = xr.concat(results, dim='region_id')

    logger.info(f"  Computed {statistic} for {len(region_ids)} regions")

    return combined


def compute_uncertainty_metrics(
    predictions: xr.DataArray,
    reference: xr.DataArray
) -> dict:
    """
    Compute comprehensive uncertainty metrics.

    Parameters
    ----------
    predictions : xr.DataArray
        Predicted/fused values
    reference : xr.DataArray
        Reference/validation data

    Returns
    -------
    dict
        Uncertainty metrics: bias, mae, rmse, r2, nse, kge

    Examples
    --------
    >>> pred = xr.DataArray(np.random.randn(100, 50, 50))
    >>> ref = pred + np.random.randn(100, 50, 50) * 0.1
    >>> metrics = compute_uncertainty_metrics(pred, ref)
    >>> print(f"RMSE: {metrics['rmse']:.4f}")
    """
    # Stack and remove NaN
    pred_flat = predictions.values.flatten()
    ref_flat = reference.values.flatten()

    valid_mask = ~(np.isnan(pred_flat) | np.isnan(ref_flat))
    pred_valid = pred_flat[valid_mask]
    ref_valid = ref_flat[valid_mask]

    # Compute metrics
    metrics = {}

    # Bias
    metrics['bias'] = float(np.mean(pred_valid - ref_valid))

    # MAE
    metrics['mae'] = float(np.mean(np.abs(pred_valid - ref_valid)))

    # RMSE
    metrics['rmse'] = float(np.sqrt(np.mean((pred_valid - ref_valid) ** 2)))

    # R²
    ss_res = np.sum((ref_valid - pred_valid) ** 2)
    ss_tot = np.sum((ref_valid - np.mean(ref_valid)) ** 2)
    metrics['r2'] = float(1 - (ss_res / ss_tot))

    # NSE (Nash-Sutcliffe Efficiency)
    metrics['nse'] = float(1 - (ss_res / ss_tot))

    # KGE (Kling-Gupta Efficiency)
    r = np.corrcoef(pred_valid, ref_valid)[0, 1]
    alpha = np.std(pred_valid) / np.std(ref_valid)
    beta = np.mean(pred_valid) / np.mean(ref_valid)
    metrics['kge'] = float(1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))

    # Correlation
    metrics['correlation'] = float(r)

    return metrics


def save_netcdf_with_metadata(
    data: xr.DataArray,
    output_file: str,
    metadata: Optional[dict] = None
) -> None:
    """
    Save xarray DataArray to NetCDF with comprehensive metadata.

    Parameters
    ----------
    data : xr.DataArray
        Data to save
    output_file : str
        Output file path
    metadata : dict, optional
        Additional metadata to include

    Examples
    --------
    >>> data = xr.DataArray(np.random.randn(100, 50, 50), dims=['time', 'lat', 'lon'])
    >>> save_netcdf_with_metadata(
    ...     data, 'output.nc',
    ...     metadata={'description': 'HCC-ET fused T/ET ratio'}
    ... )
    """
    logger.info(f"Saving data to {output_file}")

    # Add standard metadata
    data.attrs['creation_date'] = str(np.datetime64('today'))
    data.attrs['framework'] = 'HCC-ET'
    data.attrs['version'] = '1.0'

    # Add user metadata
    if metadata is not None:
        data.attrs.update(metadata)

    # Save
    data.to_netcdf(
        output_file,
        encoding={
            data.name: {
                'zlib': True,
                'complevel': 4,
                'dtype': 'float32'
            }
        }
    )

    logger.info(f"  Saved {output_file}")


def calculate_decadal_trends(
    data: xr.DataArray,
    time_dim: str = 'time'
) -> xr.DataArray:
    """
    Calculate trends normalized to per-decade rates.

    Parameters
    ----------
    data : xr.DataArray
        Input data with time dimension
    time_dim : str, default='time'
        Name of time dimension

    Returns
    -------
    xr.DataArray
        Decadal trends (units per decade)
    """
    # Compute annual trends
    slopes, pvals = compute_spatial_trends(data, dim=time_dim)

    # Convert to decadal (assuming annual time steps)
    # If time is in days/months, adjust accordingly
    decadal_trends = slopes * 10

    decadal_trends.attrs['units'] = 'per_decade'
    decadal_trends.attrs['description'] = 'Trend rate per decade'

    return decadal_trends


def spatial_aggregation_conservative(
    data_fine: xr.DataArray,
    data_coarse_template: xr.DataArray
) -> xr.DataArray:
    """
    Conservative spatial aggregation (area-weighted averaging).

    This is a placeholder - production code would use xESMF or similar.

    Parameters
    ----------
    data_fine : xr.DataArray
        High-resolution data
    data_coarse_template : xr.DataArray
        Coarse resolution template (for target grid)

    Returns
    -------
    xr.DataArray
        Aggregated data on coarse grid
    """
    logger.warning("Using simple interpolation (not conservative). Use xESMF in production.")

    # Simple interpolation (not conservative)
    # Production: use xesmf.Regridder with method='conservative'
    data_coarse = data_fine.interp_like(data_coarse_template, method='linear')

    return data_coarse
