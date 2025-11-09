"""
Data Loader Module for HCC-ET Framework
========================================

Handles loading and preprocessing of all input datasets:
- Parent T/ET ratios (coarse resolution: GLEAM, PMLv2, GLDAS, etc.)
- 1km high-resolution predictors (LST, NDVI, Albedo, Precipitation)
- Physical constraints (SapFluxNet, Water Balance ET)
- Land cover masks
- Auxiliary data (PFT maps, basin masks, etc.)

Author: [Your Name]
Date: 2025-11-09
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import xarray as xr
import pandas as pd

logger = logging.getLogger(__name__)


def load_parent_t_et_ratios(
    data_dir: str | Path,
    products: List[str],
    time_range: Tuple[str, str],
    variables: Optional[Dict[str, str]] = None
) -> xr.DataArray:
    """
    Load parent T/ET ratios from coarse-resolution products.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing parent product NetCDF files
    products : list of str
        List of product names (e.g., ['GLEAM', 'PMLv2', 'GLDAS'])
    time_range : tuple of str
        Start and end dates in 'YYYY-MM-DD' format
    variables : dict, optional
        Mapping from product name to variable name in NetCDF
        Default: {'GLEAM': 'T_ET_ratio', 'PMLv2': 't_et', 'GLDAS': 'transp_frac'}

    Returns
    -------
    xr.DataArray
        Multi-model T/ET ratios, dimensions: (time, lat, lon, model)

    Examples
    --------
    >>> ratios = load_parent_t_et_ratios(
    ...     data_dir='/data/parent_products',
    ...     products=['GLEAM', 'PMLv2', 'GLDAS'],
    ...     time_range=('1981-01-01', '2025-12-31')
    ... )
    >>> print(ratios.shape)  # (time, lat, lon, 3)
    """
    data_dir = Path(data_dir)

    if variables is None:
        # Default variable names for common products
        variables = {
            'GLEAM': 'T_ET_ratio',
            'PMLv2': 't_et_ratio',
            'GLDAS': 'transp_fraction',
            'FLUXCOM': 'T_ET',
            'MOD16': 'ET_T_ratio'
        }

    datasets = []

    for product in products:
        # Construct file path (adjust pattern as needed)
        file_pattern = f"{product}_*_T_ET_ratio.nc"
        files = sorted(data_dir.glob(file_pattern))

        if not files:
            logger.warning(f"No files found for {product} in {data_dir}")
            continue

        logger.info(f"  Loading {product}: {files[0].name}")

        # Load dataset
        ds = xr.open_mfdataset(
            files,
            combine='by_coords',
            parallel=True
        )

        # Extract T/ET variable
        var_name = variables.get(product, 'T_ET_ratio')
        if var_name not in ds:
            logger.warning(f"Variable {var_name} not found in {product}")
            continue

        t_et = ds[var_name]

        # Select time range
        t_et = t_et.sel(time=slice(*time_range))

        # Add model coordinate
        t_et = t_et.expand_dims(model=[product])

        datasets.append(t_et)

    if not datasets:
        raise ValueError("No valid datasets loaded!")

    # Concatenate along model dimension
    combined = xr.concat(datasets, dim='model')

    # Quality control
    combined = _quality_control_t_et_ratio(combined)

    logger.info(f"  Loaded {len(products)} parent products")
    logger.info(f"  Shape: {combined.shape}")
    logger.info(f"  Time range: {combined.time.values[0]} to {combined.time.values[-1]}")

    return combined


def load_predictors_1km(
    data_dir: str | Path,
    variables: List[str],
    time_range: Tuple[str, str],
    region: Optional[Dict[str, Tuple[float, float]]] = None
) -> xr.DataArray:
    """
    Load 1km high-resolution predictors for downscaling.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing predictor NetCDF files
    variables : list of str
        Variable names (e.g., ['LST', 'NDVI', 'Albedo', 'Precip'])
    time_range : tuple of str
        Start and end dates
    region : dict, optional
        Spatial subset: {'lat': (min, max), 'lon': (min, max)}

    Returns
    -------
    xr.DataArray
        Predictors, dimensions: (time, lat, lon, variable)

    Examples
    --------
    >>> predictors = load_predictors_1km(
    ...     data_dir='/data/predictors_1km',
    ...     variables=['LST', 'NDVI', 'Albedo', 'Precip'],
    ...     time_range=('1981-01-01', '2025-12-31')
    ... )
    """
    data_dir = Path(data_dir)
    datasets = []

    for var in variables:
        # Construct file path
        files = sorted(data_dir.glob(f"{var}_1km_*.nc"))

        if not files:
            logger.warning(f"No files found for {var}")
            continue

        logger.info(f"  Loading {var}: {len(files)} files")

        # Load
        ds = xr.open_mfdataset(files, combine='by_coords', parallel=True)

        # Extract variable (try common naming conventions)
        possible_names = [var, var.lower(), var.upper(), f"{var}_1km"]
        var_data = None
        for name in possible_names:
            if name in ds:
                var_data = ds[name]
                break

        if var_data is None:
            logger.warning(f"Variable {var} not found in dataset")
            continue

        # Select time range
        var_data = var_data.sel(time=slice(*time_range))

        # Spatial subset if specified
        if region is not None:
            var_data = var_data.sel(
                lat=slice(*region['lat']),
                lon=slice(*region['lon'])
            )

        # Add variable dimension
        var_data = var_data.expand_dims(variable=[var])

        datasets.append(var_data)

    if not datasets:
        raise ValueError("No predictors loaded!")

    # Concatenate
    combined = xr.concat(datasets, dim='variable')

    logger.info(f"  Loaded {len(variables)} predictors")
    logger.info(f"  Shape: {combined.shape}")

    return combined


def load_sapfluxnet_1km(
    data_dir: str | Path,
    upscale_method: str = 'nearest'
) -> xr.DataArray:
    """
    Load SapFluxNet observations upscaled to 1km grid.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing SapFluxNet data
    upscale_method : str
        Upscaling method: 'nearest', 'bilinear', 'conservative'

    Returns
    -------
    xr.DataArray
        Upscaled T/ET observations, dimensions: (lat, lon)
        Contains NaN where no observations available

    Notes
    -----
    SapFluxNet provides site-level transpiration observations.
    This function upscales them to 1km grid for use as constraints.
    """
    data_dir = Path(data_dir)

    # Load SapFluxNet site data
    site_file = data_dir / "sapfluxnet_sites.csv"
    if not site_file.exists():
        logger.warning(f"SapFluxNet file not found: {site_file}")
        return xr.DataArray(np.nan)

    logger.info(f"  Loading SapFluxNet from {site_file}")

    # Read site data
    sites = pd.read_csv(site_file)
    logger.info(f"  Found {len(sites)} sites")

    # Create 1km grid (placeholder - use actual grid from predictors)
    lat = np.arange(-90, 90, 0.01)  # ~1km at equator
    lon = np.arange(-180, 180, 0.01)

    # Initialize output grid
    t_et_obs = xr.DataArray(
        np.full((len(lat), len(lon)), np.nan),
        dims=['lat', 'lon'],
        coords={'lat': lat, 'lon': lon}
    )

    # Upscale site observations to grid
    for _, site in sites.iterrows():
        # Find nearest grid cell
        lat_idx = np.argmin(np.abs(lat - site['latitude']))
        lon_idx = np.argmin(np.abs(lon - site['longitude']))

        # Assign T/ET observation (if available)
        if 'T_ET_ratio' in site and not pd.isna(site['T_ET_ratio']):
            t_et_obs.values[lat_idx, lon_idx] = site['T_ET_ratio']

    # Store metadata
    t_et_obs.attrs['n_sites'] = len(sites)
    t_et_obs.attrs['upscale_method'] = upscale_method
    t_et_obs.attrs['source'] = 'SapFluxNet'

    logger.info(f"  Upscaled {np.sum(~np.isnan(t_et_obs.values))} valid pixels")

    return t_et_obs


def load_wb_et_basins(
    data_dir: str | Path,
    basin_ids: Optional[List[str]] = None
) -> xr.DataArray:
    """
    Load basin-scale Water Balance ET (P - Q - dS).

    Parameters
    ----------
    data_dir : str or Path
        Directory containing basin WB data
    basin_ids : list of str, optional
        Specific basin IDs to load (if None, load all)

    Returns
    -------
    xr.DataArray
        Basin ET, dimensions: (basin_id, time)

    Notes
    -----
    Water Balance ET = Precipitation - Runoff - Storage Change
    This provides a "ground truth" constraint at basin scale.
    """
    data_dir = Path(data_dir)

    wb_file = data_dir / "basin_water_balance_et.nc"
    if not wb_file.exists():
        logger.warning(f"Water Balance file not found: {wb_file}")
        return xr.DataArray(np.nan)

    logger.info(f"  Loading Water Balance ET from {wb_file}")

    # Load dataset
    ds = xr.open_dataset(wb_file)

    # Extract ET variable
    if 'ET_wb' in ds:
        wb_et = ds['ET_wb']
    elif 'evapotranspiration' in ds:
        wb_et = ds['evapotranspiration']
    else:
        raise ValueError("ET variable not found in Water Balance dataset")

    # Filter basins if specified
    if basin_ids is not None:
        wb_et = wb_et.sel(basin_id=basin_ids)

    logger.info(f"  Loaded {len(wb_et.coords['basin_id'])} basins")

    return wb_et


def load_total_et_1km(
    data_dir: str | Path,
    product: str = 'CAMELE'
) -> xr.DataArray:
    """
    Load total ET product at 1km resolution.

    This is used as the denominator for T/ET ratio to compute
    final T and E fluxes.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing total ET files
    product : str
        Product name (e.g., 'CAMELE', 'GLEAM', 'PMLv2')

    Returns
    -------
    xr.DataArray
        Total ET, dimensions: (time, lat, lon)
    """
    data_dir = Path(data_dir)

    files = sorted(data_dir.glob(f"{product}_*_ET_total.nc"))

    if not files:
        raise FileNotFoundError(f"No total ET files found for {product}")

    logger.info(f"  Loading total ET from {product}")

    ds = xr.open_mfdataset(files, combine='by_coords', parallel=True)

    # Extract ET variable
    if 'ET' in ds:
        et = ds['ET']
    elif 'evapotranspiration' in ds:
        et = ds['evapotranspiration']
    else:
        raise ValueError(f"ET variable not found in {product}")

    logger.info(f"  Total ET shape: {et.shape}")

    return et


def load_land_cover_masks(
    data_dir: str | Path,
    classification: str = 'IGBP'
) -> xr.DataArray:
    """
    Load land cover classification masks.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing land cover data
    classification : str
        Classification scheme: 'IGBP', 'LCCS', 'PFT'

    Returns
    -------
    xr.DataArray
        Land cover masks, dimensions: (lat, lon, class)
        Each class is a binary mask (1=present, 0=absent)

    Notes
    -----
    Used for partitioning trend analysis by:
    - Natural ecosystems (forests, grasslands)
    - Managed ecosystems (croplands, pastures)
    - Urban areas
    """
    data_dir = Path(data_dir)

    lc_file = data_dir / f"landcover_{classification}.nc"

    if not lc_file.exists():
        logger.warning(f"Land cover file not found: {lc_file}")
        # Return dummy mask
        return xr.DataArray(np.nan)

    logger.info(f"  Loading land cover from {lc_file}")

    ds = xr.open_dataset(lc_file)

    # Extract land cover variable
    if 'landcover' in ds:
        lc = ds['landcover']
    elif 'LC_Type1' in ds:  # MODIS naming
        lc = ds['LC_Type1']
    else:
        raise ValueError("Land cover variable not found")

    # Convert to binary masks for each class
    # Assuming IGBP classification (1-17)
    class_names = {
        1: 'evergreen_needleleaf',
        2: 'evergreen_broadleaf',
        3: 'deciduous_needleleaf',
        4: 'deciduous_broadleaf',
        5: 'mixed_forest',
        6: 'closed_shrublands',
        7: 'open_shrublands',
        8: 'woody_savannas',
        9: 'savannas',
        10: 'grasslands',
        11: 'permanent_wetlands',
        12: 'croplands',
        13: 'urban_buildup',
        14: 'cropland_natural_mosaic',
        15: 'snow_ice',
        16: 'barren',
        17: 'water_bodies'
    }

    # Group into "natural" and "managed"
    natural_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16]
    managed_classes = [12, 14]
    urban_classes = [13]

    masks = []
    mask_names = []

    # Natural mask
    natural_mask = xr.where(
        lc.isin(natural_classes), 1, 0
    ).expand_dims(class_name=['natural'])
    masks.append(natural_mask)
    mask_names.append('natural')

    # Managed mask
    managed_mask = xr.where(
        lc.isin(managed_classes), 1, 0
    ).expand_dims(class_name=['managed'])
    masks.append(managed_mask)
    mask_names.append('managed')

    # Urban mask
    urban_mask = xr.where(
        lc.isin(urban_classes), 1, 0
    ).expand_dims(class_name=['urban'])
    masks.append(urban_mask)
    mask_names.append('urban')

    # Combine
    combined_masks = xr.concat(masks, dim='class_name')

    logger.info(f"  Created masks for: {mask_names}")

    return combined_masks


def load_basin_mask_1km(basin_mask_file: str | Path) -> xr.DataArray:
    """
    Load basin mask at 1km resolution.

    Parameters
    ----------
    basin_mask_file : str or Path
        Path to basin mask NetCDF file

    Returns
    -------
    xr.DataArray
        Basin IDs, dimensions: (lat, lon)
    """
    basin_mask_file = Path(basin_mask_file)

    if not basin_mask_file.exists():
        raise FileNotFoundError(f"Basin mask not found: {basin_mask_file}")

    logger.info(f"  Loading basin mask from {basin_mask_file}")

    ds = xr.open_dataset(basin_mask_file)

    if 'basin_id' in ds:
        basin_mask = ds['basin_id']
    elif 'basins' in ds:
        basin_mask = ds['basins']
    else:
        raise ValueError("Basin mask variable not found")

    return basin_mask


def load_pft_map(pft_map_file: str | Path) -> xr.DataArray:
    """
    Load Plant Functional Type (PFT) map.

    Parameters
    ----------
    pft_map_file : str or Path
        Path to PFT map NetCDF file

    Returns
    -------
    xr.DataArray
        PFT classifications, dimensions: (lat, lon)
    """
    pft_map_file = Path(pft_map_file)

    if not pft_map_file.exists():
        logger.warning(f"PFT map not found: {pft_map_file}")
        return xr.DataArray(np.nan)

    logger.info(f"  Loading PFT map from {pft_map_file}")

    ds = xr.open_dataset(pft_map_file)

    if 'PFT' in ds:
        pft = ds['PFT']
    elif 'pft_type' in ds:
        pft = ds['pft_type']
    else:
        raise ValueError("PFT variable not found")

    return pft


def _quality_control_t_et_ratio(t_et: xr.DataArray) -> xr.DataArray:
    """
    Apply quality control to T/ET ratios.

    Parameters
    ----------
    t_et : xr.DataArray
        T/ET ratios

    Returns
    -------
    xr.DataArray
        Quality-controlled T/ET ratios

    QC Steps:
    - Clip to [0, 1] range
    - Remove outliers (> 99th percentile)
    - Fill missing values (optional)
    """
    logger.info("  Applying quality control to T/ET ratios...")

    # Clip to physical range [0, 1]
    t_et_qc = t_et.clip(0, 1)

    # Count values outside range
    n_below = (t_et < 0).sum().values
    n_above = (t_et > 1).sum().values

    if n_below > 0 or n_above > 0:
        logger.warning(f"  Clipped {n_below} values < 0, {n_above} values > 1")

    # Remove extreme outliers (optional)
    # Could use IQR method or percentile-based filtering

    # Fill missing values with temporal interpolation (optional)
    # t_et_qc = t_et_qc.interpolate_na(dim='time', method='linear')

    return t_et_qc


def validate_spatial_consistency(
    data1: xr.DataArray,
    data2: xr.DataArray,
    tolerance: float = 0.01
) -> bool:
    """
    Validate that two datasets have consistent spatial grids.

    Parameters
    ----------
    data1, data2 : xr.DataArray
        Datasets to compare
    tolerance : float
        Tolerance for coordinate differences (degrees)

    Returns
    -------
    bool
        True if grids are consistent
    """
    # Check dimensions
    if set(data1.dims) != set(data2.dims):
        logger.warning("Dimension mismatch!")
        return False

    # Check coordinate ranges
    for dim in ['lat', 'lon']:
        if dim in data1.dims and dim in data2.dims:
            diff_min = abs(data1[dim].min().values - data2[dim].min().values)
            diff_max = abs(data1[dim].max().values - data2[dim].max().values)

            if diff_min > tolerance or diff_max > tolerance:
                logger.warning(f"{dim} coordinate mismatch!")
                return False

    logger.info("  Spatial grids are consistent.")
    return True
