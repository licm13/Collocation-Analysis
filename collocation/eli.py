"""
Ecosystem Limitation Index (ELI) Calculation Module
====================================================

This module implements the ELI (Ecosystem Limitation Index) calculation framework
based on the paper "Widespread shift from ecosystem energy to water limitation
with climate change".

The ELI quantifies the relative importance of water vs. energy limitation in
terrestrial ecosystems using collocation analysis of multiple data products.

Variables processed:
- nsma: Near-surface soil moisture anomaly (0-10cm)
- ssma: Sub-surface soil moisture anomaly (10-100cm)
- tvega: Transpiration anomaly
- eta: Total evapotranspiration anomaly
- swa: Downward short-wave radiation flux anomaly

Data sources supported:
- ERA5-Land (ERA5L)
- GLEAM v3.8a (G38a)
- GLDAS v2.1 (G21)

Author: Converted from MATLAB to Python
Reference: licm_13@163.com
"""

import numpy as np
import xarray as xr
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings

# Import collocation methods
from .ivd import ivd
from .eivd import eivd
from .tc import tc
from .ivs import ivs
from .ec import ec
try:
    from .bayesian_tc import BayesianTC, PYMC3_AVAILABLE
except ImportError:
    PYMC3_AVAILABLE = False
    BayesianTC = None


class ELIProcessor:
    """
    Process ELI data using multiple collocation methods.

    This class handles:
    - Loading data from multiple sources
    - Applying various collocation methods (IVD, EIVD, TC, Bayesian TC)
    - Computing error statistics and merged products
    - Saving results to NetCDF

    Parameters
    ----------
    lat : array-like
        Latitude coordinates
    lon : array-like
        Longitude coordinates
    time_range : tuple
        (start_year, end_year) for data processing

    Attributes
    ----------
    variables : list
        List of variables to process: ['nsma', 'ssma', 'tvega', 'eta', 'swa']
    methods : dict
        Available collocation methods and their configurations
    """

    def __init__(self, lat: Optional[np.ndarray] = None,
                 lon: Optional[np.ndarray] = None,
                 time_range: Optional[Tuple[int, int]] = None):
        """Initialize ELI processor."""
        # Default spatial coordinates (from MATLAB code)
        if lat is None:
            self.lat = np.arange(89.75, -60, -0.25)  # 600 points
        else:
            self.lat = np.array(lat)

        if lon is None:
            self.lon = np.arange(-179.75, 180, 0.25)  # 1440 points
        else:
            self.lon = np.array(lon)

        self.time_range = time_range

        # Variables to process
        self.variables = ['nsma', 'ssma', 'tvega', 'eta', 'swa']

        # Results storage
        self.results = {}

    def process_dual_ivd(self, data1: np.ndarray, data2: np.ndarray,
                        variable: str = 'variable') -> Dict:
        """
        Process two data products using IVD method.

        This corresponds to the ELG38a_IVD.m MATLAB script, which processes
        ERA5L and GLEAM v3.8a data for 1980-1999.

        Parameters
        ----------
        data1 : np.ndarray
            First data product, shape (n_time, n_lat, n_lon) or (n_lat, n_lon, n_time, n_products)
        data2 : np.ndarray
            Second data product
        variable : str
            Variable name for logging

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'error_variance': Error variance for each product (n_lat, n_lon, 2)
            - 'rho2': Data-truth correlation (n_lat, n_lon, 2)
            - 'weights': Merging weights (n_lat, n_lon, 2)
            - 'merged': Merged product (n_lat, n_lon, n_time)
        """
        print(f"Processing {variable} with IVD method...")

        # Handle different input shapes
        if data1.ndim == 3 and data2.ndim == 3:
            # Shape: (n_time, n_lat, n_lon)
            n_time, n_lat, n_lon = data1.shape
            combined = np.stack([data1, data2], axis=-1)  # (n_time, n_lat, n_lon, 2)
        elif data1.ndim == 4:
            # Already combined: (n_lat, n_lon, n_time, 2)
            n_lat, n_lon, n_time, _ = data1.shape
            combined = data1
        else:
            raise ValueError(f"Unexpected data shape: {data1.shape}")

        # Initialize output arrays
        sige2 = np.full((n_lat, n_lon, 2), np.nan)
        rho2 = np.full((n_lat, n_lon, 2), np.nan)
        weights = np.full((n_lat, n_lon, 2), np.nan)
        merged = np.full((n_lat, n_lon, n_time), np.nan)

        # Process each grid point
        n_valid = 0
        for i in range(n_lat):
            for j in range(n_lon):
                # Extract time series for this location
                if combined.ndim == 4 and combined.shape[0] == n_lat:
                    dual = combined[i, j, :, :]  # (n_time, 2)
                else:
                    dual = combined[:, i, j, :]  # (n_time, 2)

                # Skip if mostly NaN
                if np.isnan(dual).sum() > dual.size * 0.5:
                    continue

                # Fill NaN with mean (as in MATLAB code)
                mean_val = np.nanmean(dual)
                dual_filled = dual.copy()
                dual_filled[np.isnan(dual_filled)] = mean_val

                # Remove rows with any NaN
                valid_rows = ~np.any(np.isnan(dual_filled), axis=1)
                dual_clean = dual_filled[valid_rows, :]

                if dual_clean.shape[0] < 4:
                    continue

                try:
                    # Apply IVD
                    EeeT, rho2_val, u = ivd(dual_clean)

                    # Store results
                    sige2[i, j, 0] = EeeT[0, 0]
                    sige2[i, j, 1] = EeeT[1, 1]
                    rho2[i, j, :] = rho2_val
                    weights[i, j, :] = u

                    # Compute merged product
                    if combined.ndim == 4 and combined.shape[0] == n_lat:
                        data_i = combined[i, j, :, :]
                    else:
                        data_i = combined[:, i, j, :]

                    merged_ts = data_i[:, 0] * u[0] + data_i[:, 1] * u[1]

                    # Fill NaN with mean (as in MATLAB)
                    mean_ts = np.nanmean(data_i, axis=1)
                    nan_idx = np.isnan(merged_ts)
                    merged_ts[nan_idx] = mean_ts[nan_idx]

                    merged[i, j, :] = merged_ts
                    n_valid += 1

                except Exception as e:
                    warnings.warn(f"IVD failed at ({i},{j}): {str(e)}")
                    continue

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{n_lat} rows, {n_valid} valid points")

        print(f"  Completed: {n_valid} valid grid points")

        return {
            'error_variance': sige2,
            'rho2': rho2,
            'weights': weights,
            'merged': merged,
            'method': 'IVD'
        }

    def process_triple_eivd(self, data1: np.ndarray, data2: np.ndarray,
                           data3: np.ndarray, variable: str = 'variable') -> Dict:
        """
        Process three data products using EIVD method.

        This corresponds to the ELG21G38_EIVD.m MATLAB script, which processes
        ERA5L, GLEAM v3.8a, and GLDAS v2.1 data for 2000-2022.

        Parameters
        ----------
        data1 : np.ndarray
            First data product (ERA5L)
        data2 : np.ndarray
            Second data product (GLEAM)
        data3 : np.ndarray
            Third data product (GLDAS)
        variable : str
            Variable name

        Returns
        -------
        results : dict
            Dictionary containing EIVD results including error cross-correlation
        """
        print(f"Processing {variable} with EIVD method...")

        # Handle different input shapes
        if data1.ndim == 3:
            n_time, n_lat, n_lon = data1.shape
            combined = np.stack([data1, data2, data3], axis=-1)
        elif data1.ndim == 4:
            n_lat, n_lon, n_time, _ = data1.shape
            combined = data1
        else:
            raise ValueError(f"Unexpected data shape: {data1.shape}")

        # Initialize outputs
        sige2 = np.full((n_lat, n_lon, 3), np.nan)
        rho2 = np.full((n_lat, n_lon, 3), np.nan)
        weights = np.full((n_lat, n_lon, 3), np.nan)
        merged = np.full((n_lat, n_lon, n_time), np.nan)
        ecc = np.full((n_lat, n_lon, 3), np.nan)  # Error cross-correlation

        n_valid = 0
        for i in range(n_lat):
            for j in range(n_lon):
                # Extract time series
                if combined.ndim == 4 and combined.shape[0] == n_lat:
                    tri = combined[i, j, :, :]
                else:
                    tri = combined[:, i, j, :]

                # Skip if mostly NaN
                if np.isnan(tri).sum() > tri.size * 0.5:
                    continue

                # Fill NaN with mean
                mean_val = np.nanmean(tri)
                tri_filled = tri.copy()
                tri_filled[np.isnan(tri_filled)] = mean_val

                # Remove rows with any NaN
                valid_rows = ~np.any(np.isnan(tri_filled), axis=1)
                tri_clean = tri_filled[valid_rows, :]

                # Use random data if empty (as in MATLAB)
                if tri_clean.shape[0] < 2:
                    tri_clean = np.random.rand(n_time, 3)

                try:
                    # Apply EIVD
                    EeeT, SNR, rho2_val, fMSE, L = eivd(tri_clean)

                    # Store results
                    sige2[i, j, :] = np.diag(EeeT)
                    rho2[i, j, :] = rho2_val

                    # Compute weights (inverse variance weighting)
                    inv_var = 1.0 / (sige2[i, j, :] + 1e-10)
                    weights[i, j, :] = inv_var / np.sum(inv_var)

                    # Store error cross-correlations
                    ecc[i, j, 0] = EeeT[1, 2]  # Between products 2 and 3
                    ecc[i, j, 1] = EeeT[0, 2]  # Between products 1 and 3
                    ecc[i, j, 2] = EeeT[0, 1]  # Between products 1 and 2

                    # Compute merged product
                    if combined.ndim == 4 and combined.shape[0] == n_lat:
                        data_i = combined[i, j, :, :]
                    else:
                        data_i = combined[:, i, j, :]

                    merged_ts = np.sum(data_i * weights[i, j, :], axis=1)

                    # Fill NaN with mean
                    mean_ts = np.nanmean(data_i, axis=1)
                    nan_idx = np.isnan(merged_ts)
                    merged_ts[nan_idx] = mean_ts[nan_idx]

                    merged[i, j, :] = merged_ts
                    n_valid += 1

                except Exception as e:
                    warnings.warn(f"EIVD failed at ({i},{j}): {str(e)}")
                    continue

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{n_lat} rows, {n_valid} valid points")

        print(f"  Completed: {n_valid} valid grid points")

        return {
            'error_variance': sige2,
            'rho2': rho2,
            'weights': weights,
            'merged': merged,
            'error_cross_correlation': ecc,
            'method': 'EIVD'
        }

    def process_triple_with_all_methods(self, data1: np.ndarray,
                                       data2: np.ndarray,
                                       data3: np.ndarray,
                                       variable: str = 'variable',
                                       use_bayesian: bool = False) -> Dict:
        """
        Process three data products using ALL available collocation methods.

        This applies:
        - TC (Triple Collocation)
        - EIVD (Extended IVD)
        - EC (Extended Collocation, if 4+ products)
        - Bayesian TC (optional, if PyMC3 available)

        Parameters
        ----------
        data1, data2, data3 : np.ndarray
            Data products to analyze
        variable : str
            Variable name
        use_bayesian : bool
            Whether to apply Bayesian TC (requires PyMC3)

        Returns
        -------
        results : dict
            Comprehensive results from all methods
        """
        print(f"\n{'='*70}")
        print(f"Processing {variable} with ALL METHODS")
        print(f"{'='*70}")

        all_results = {}

        # 1. EIVD (primary method for ELI)
        print("\n1. EIVD Method:")
        all_results['eivd'] = self.process_triple_eivd(data1, data2, data3, variable)

        # 2. Triple Collocation
        print("\n2. Triple Collocation (TC) Method:")
        all_results['tc'] = self.process_triple_tc(data1, data2, data3, variable)

        # 3. Bayesian TC (if requested and available)
        if use_bayesian and PYMC3_AVAILABLE:
            print("\n3. Bayesian Triple Collocation:")
            all_results['bayesian'] = self.process_triple_bayesian(
                data1, data2, data3, variable
            )
        elif use_bayesian:
            warnings.warn("Bayesian TC requested but PyMC3 not available")

        # 4. Comparison and recommendations
        print("\n4. Method Comparison:")
        all_results['comparison'] = self.compare_methods(all_results)

        return all_results

    def process_triple_tc(self, data1: np.ndarray, data2: np.ndarray,
                         data3: np.ndarray, variable: str = 'variable') -> Dict:
        """
        Process three data products using classic Triple Collocation.

        Parameters
        ----------
        data1, data2, data3 : np.ndarray
            Data products
        variable : str
            Variable name

        Returns
        -------
        results : dict
            TC analysis results
        """
        print(f"  Processing {variable} with TC...")

        # Handle input shapes
        if data1.ndim == 3:
            n_time, n_lat, n_lon = data1.shape
            combined = np.stack([data1, data2, data3], axis=-1)
        else:
            n_lat, n_lon, n_time, _ = data1.shape
            combined = data1

        # Initialize outputs
        sige2 = np.full((n_lat, n_lon, 3), np.nan)
        snr = np.full((n_lat, n_lon, 3), np.nan)
        rho2 = np.full((n_lat, n_lon, 3), np.nan)
        fMSE = np.full((n_lat, n_lon, 3), np.nan)

        n_valid = 0
        for i in range(n_lat):
            for j in range(n_lon):
                if combined.ndim == 4 and combined.shape[0] == n_lat:
                    tri = combined[i, j, :, :]
                else:
                    tri = combined[:, i, j, :]

                if np.isnan(tri).sum() > tri.size * 0.5:
                    continue

                # Remove NaN rows
                valid_rows = ~np.any(np.isnan(tri), axis=1)
                tri_clean = tri[valid_rows, :]

                if tri_clean.shape[0] < 100:
                    continue

                try:
                    EeeT, SNR, rho2_val, fMSE_val = tc(tri_clean)

                    sige2[i, j, :] = np.diag(EeeT)
                    snr[i, j, :] = SNR
                    rho2[i, j, :] = rho2_val
                    fMSE[i, j, :] = fMSE_val
                    n_valid += 1

                except Exception as e:
                    continue

            if (i + 1) % 100 == 0:
                print(f"    Processed {i+1}/{n_lat} rows, {n_valid} valid")

        print(f"    Completed: {n_valid} valid grid points")

        return {
            'error_variance': sige2,
            'snr': snr,
            'rho2': rho2,
            'fMSE': fMSE,
            'method': 'TC'
        }

    def process_triple_bayesian(self, data1: np.ndarray, data2: np.ndarray,
                               data3: np.ndarray, variable: str = 'variable',
                               n_samples: int = 1000) -> Dict:
        """
        Process three data products using Bayesian Triple Collocation.

        This provides time-varying error estimates and full uncertainty quantification.

        Parameters
        ----------
        data1, data2, data3 : np.ndarray
            Data products
        variable : str
            Variable name
        n_samples : int
            Number of MCMC samples

        Returns
        -------
        results : dict
            Bayesian TC results with uncertainty estimates
        """
        if not PYMC3_AVAILABLE or BayesianTC is None:
            raise ImportError("PyMC3 required for Bayesian TC")

        print(f"  Processing {variable} with Bayesian TC...")
        print(f"  (This may take a while...)")

        # For Bayesian TC, we'll process a sample of grid points
        # Full gridded Bayesian analysis would be computationally expensive

        # Handle input shapes
        if data1.ndim == 3:
            n_time, n_lat, n_lon = data1.shape
            combined = np.stack([data1, data2, data3], axis=-1)
        else:
            n_lat, n_lon, n_time, _ = data1.shape
            combined = data1

        # Sample grid points (e.g., 1% of points)
        n_sample = max(10, int(0.01 * n_lat * n_lon))
        sample_results = []

        for _ in range(n_sample):
            i = np.random.randint(0, n_lat)
            j = np.random.randint(0, n_lon)

            if combined.ndim == 4 and combined.shape[0] == n_lat:
                tri = combined[i, j, :, :]
            else:
                tri = combined[:, i, j, :]

            if np.isnan(tri).sum() > tri.size * 0.5:
                continue

            valid_rows = ~np.any(np.isnan(tri), axis=1)
            tri_clean = tri[valid_rows, :].T  # BayesianTC expects (n_products, n_samples)

            if tri_clean.shape[1] < 50:
                continue

            try:
                btc = BayesianTC(tri_clean)
                btc.setup_model()
                btc.run_inference(n_samples=n_samples)
                summary = btc.get_summary()

                sample_results.append({
                    'location': (i, j),
                    'summary': summary
                })
            except Exception as e:
                warnings.warn(f"Bayesian TC failed at ({i},{j}): {str(e)}")
                continue

        print(f"    Completed Bayesian analysis on {len(sample_results)} sample points")

        return {
            'sample_results': sample_results,
            'n_samples': len(sample_results),
            'method': 'Bayesian TC'
        }

    def compare_methods(self, results: Dict) -> Dict:
        """
        Compare results from different collocation methods.

        Parameters
        ----------
        results : dict
            Results from multiple methods

        Returns
        -------
        comparison : dict
            Comparison metrics and recommendations
        """
        comparison = {
            'methods_used': list(results.keys()),
            'recommendations': []
        }

        # Compare EIVD vs TC if both available
        if 'eivd' in results and 'tc' in results:
            eivd_ecc = results['eivd'].get('error_cross_correlation')
            if eivd_ecc is not None:
                mean_ecc = np.nanmean(np.abs(eivd_ecc))
                comparison['mean_error_correlation'] = mean_ecc

                if mean_ecc > 0.01:
                    comparison['recommendations'].append(
                        "Use EIVD: Significant error cross-correlation detected"
                    )
                else:
                    comparison['recommendations'].append(
                        "Either TC or EIVD: Error cross-correlation is negligible"
                    )

        # Compare error variances
        if 'eivd' in results and 'tc' in results:
            eivd_var = results['eivd']['error_variance']
            tc_var = results['tc']['error_variance']

            var_diff = np.nanmean(np.abs(eivd_var - tc_var) / (tc_var + 1e-10))
            comparison['mean_variance_difference'] = var_diff

            if var_diff > 0.1:
                comparison['recommendations'].append(
                    f"Significant difference between TC and EIVD ({var_diff:.2%})"
                )

        return comparison

    def save_to_netcdf(self, results: Dict, output_path: Union[str, Path],
                      variable: str, data_source: str = '',
                      metadata: Optional[Dict] = None):
        """
        Save results to NetCDF file.

        Parameters
        ----------
        results : dict
            Processing results
        output_path : str or Path
            Output file path
        variable : str
            Variable name
        data_source : str
            Data source identifier (e.g., 'ERA5L', 'GLEAM')
        metadata : dict, optional
            Additional metadata to include
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine actual dimensions from results
        if 'error_variance' in results:
            n_lat, n_lon = results['error_variance'].shape[:2]
        elif 'merged' in results:
            n_lat, n_lon = results['merged'].shape[:2]
        else:
            raise ValueError("Cannot determine dimensions from results")

        # Use actual lat/lon that matches data dimensions
        if len(self.lat) != n_lat or len(self.lon) != n_lon:
            # Generate appropriate coordinates for this data
            lat_coords = np.linspace(90, -90, n_lat)
            lon_coords = np.linspace(-180, 180, n_lon)
        else:
            lat_coords = self.lat
            lon_coords = self.lon

        # Create dataset
        ds = xr.Dataset(
            coords={
                'lat': lat_coords,
                'lon': lon_coords,
            }
        )

        # Add method results
        method = results.get('method', 'unknown')

        # Add error variance
        if 'error_variance' in results:
            n_products = results['error_variance'].shape[2]
            for i in range(n_products):
                ds[f'error_variance_product{i+1}'] = (
                    ['lat', 'lon'],
                    results['error_variance'][:, :, i],
                    {'long_name': f'Error variance of product {i+1}',
                     'units': 'variable_units^2'}
                )

        # Add rho2 (correlation)
        if 'rho2' in results:
            n_products = results['rho2'].shape[2]
            for i in range(n_products):
                ds[f'rho2_product{i+1}'] = (
                    ['lat', 'lon'],
                    results['rho2'][:, :, i],
                    {'long_name': f'Data-truth correlation of product {i+1}',
                     'units': 'dimensionless'}
                )

        # Add weights
        if 'weights' in results:
            n_products = results['weights'].shape[2]
            for i in range(n_products):
                ds[f'weight_product{i+1}'] = (
                    ['lat', 'lon'],
                    results['weights'][:, :, i],
                    {'long_name': f'Merging weight for product {i+1}',
                     'units': 'dimensionless'}
                )

        # Add merged product
        if 'merged' in results and results['merged'].ndim == 3:
            time_dim = results['merged'].shape[2]
            time_coord = pd.date_range('2000-01-01', periods=time_dim, freq='MS')
            ds = ds.assign_coords({'time': time_coord})
            ds['merged'] = (
                ['lat', 'lon', 'time'],
                results['merged'],
                {'long_name': f'Merged {variable} using {method}',
                 'method': method}
            )

        # Add error cross-correlation (EIVD only)
        if 'error_cross_correlation' in results:
            ecc = results['error_cross_correlation']
            for i in range(ecc.shape[2]):
                ds[f'error_cross_corr_{i+1}'] = (
                    ['lat', 'lon'],
                    ecc[:, :, i],
                    {'long_name': f'Error cross-correlation {i+1}'}
                )

        # Add global attributes
        ds.attrs['title'] = f'ELI {variable} collocation analysis'
        ds.attrs['method'] = method
        ds.attrs['variable'] = variable
        ds.attrs['data_source'] = data_source
        ds.attrs['reference'] = 'Widespread shift from ecosystem energy to water limitation with climate change'

        if metadata:
            ds.attrs.update(metadata)

        # Save to NetCDF
        ds.to_netcdf(output_path, mode='w', format='NETCDF4',
                    encoding={var: {'zlib': True, 'complevel': 5}
                             for var in ds.data_vars})

        print(f"  Saved results to: {output_path}")


def calculate_eli_index(merged_results: Dict) -> np.ndarray:
    """
    Calculate the Ecosystem Limitation Index from merged collocation results.

    The ELI quantifies the relative importance of water vs. energy limitation.

    Parameters
    ----------
    merged_results : dict
        Dictionary containing merged products for:
        - 'eta': Total evapotranspiration anomaly
        - 'tvega': Transpiration anomaly
        - 'nsma': Near-surface soil moisture anomaly
        - 'ssma': Sub-surface soil moisture anomaly
        - 'swa': Shortwave radiation anomaly

    Returns
    -------
    eli : np.ndarray
        ELI values (positive = water limited, negative = energy limited)
    """
    # This is a placeholder - the actual ELI calculation would depend on
    # the specific formulation in the reference paper

    # Example formulation (to be updated based on paper):
    # ELI = f(SM, ET, Radiation)

    sm_anom = (merged_results.get('nsma', 0) + merged_results.get('ssma', 0)) / 2
    et_anom = merged_results.get('eta', 0)
    rad_anom = merged_results.get('swa', 0)

    # Simplified ELI calculation (example)
    # Positive ELI = water limitation (low SM, high radiation)
    # Negative ELI = energy limitation (high SM, low radiation)

    eli = -sm_anom + 0.5 * rad_anom

    return eli


# Convenience function for quick processing
def process_eli_data(data_dict: Dict[str, np.ndarray],
                    method: str = 'auto',
                    use_bayesian: bool = False) -> Dict:
    """
    Convenience function to process ELI data with appropriate method.

    Parameters
    ----------
    data_dict : dict
        Dictionary with structure: {variable: [product1, product2, product3]}
    method : str
        Method to use: 'ivd' (2 products), 'eivd' (3 products),
        'tc' (3 products), 'all' (all methods), or 'auto' (detect)
    use_bayesian : bool
        Whether to include Bayesian TC

    Returns
    -------
    results : dict
        Processing results for all variables
    """
    processor = ELIProcessor()
    results = {}

    for var, products in data_dict.items():
        n_products = len(products)

        if method == 'auto':
            if n_products == 2:
                method_use = 'ivd'
            elif n_products >= 3:
                method_use = 'eivd'
            else:
                raise ValueError(f"Need at least 2 products, got {n_products}")
        else:
            method_use = method

        if method_use == 'ivd' and n_products >= 2:
            results[var] = processor.process_dual_ivd(
                products[0], products[1], var
            )
        elif method_use == 'eivd' and n_products >= 3:
            results[var] = processor.process_triple_eivd(
                products[0], products[1], products[2], var
            )
        elif method_use == 'all' and n_products >= 3:
            results[var] = processor.process_triple_with_all_methods(
                products[0], products[1], products[2], var, use_bayesian
            )
        else:
            raise ValueError(f"Invalid method '{method_use}' for {n_products} products")

    return results
