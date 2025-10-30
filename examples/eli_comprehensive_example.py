"""
Comprehensive Example: ELI (Ecosystem Limitation Index) Calculation
====================================================================

This example demonstrates how to use the ELI module with all available
collocation methods to analyze ecosystem water and energy limitation.

Based on the paper:
"Widespread shift from ecosystem energy to water limitation with climate change"

This script shows:
1. Processing dual datasets with IVD (ERA5L + GLEAM)
2. Processing triple datasets with EIVD (ERA5L + GLEAM + GLDAS)
3. Comparing all methods (IVD, EIVD, TC, Bayesian TC)
4. Calculating ELI indices
5. Exporting results to NetCDF

Author: Converted from MATLAB by Claude
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# Import collocation methods
from collocation import (
    ELIProcessor,
    process_eli_data,
    calculate_eli_index,
    ivd,
    eivd,
    tc,
    BAYESIAN_AVAILABLE
)

if BAYESIAN_AVAILABLE:
    from collocation import BayesianTC


def generate_synthetic_eli_data(n_time=240, n_lat=50, n_lon=50,
                                seed=42) -> dict:
    """
    Generate synthetic ELI data for demonstration.

    In real applications, this data would come from:
    - ERA5-Land reanalysis
    - GLEAM evapotranspiration
    - GLDAS land surface model

    Parameters
    ----------
    n_time : int
        Number of time steps (months)
    n_lat, n_lon : int
        Spatial dimensions
    seed : int
        Random seed

    Returns
    -------
    data : dict
        Dictionary with synthetic data for all variables and products
    """
    np.random.seed(seed)

    print("Generating synthetic ELI data...")
    print(f"  Dimensions: {n_time} months × {n_lat} lat × {n_lon} lon")

    # Generate synthetic "true" signals with spatial and temporal patterns
    lat_pattern = np.linspace(-1, 1, n_lat)[:, np.newaxis]
    lon_pattern = np.linspace(-1, 1, n_lon)[np.newaxis, :]

    # Spatial patterns
    spatial_pattern = np.sin(3 * lat_pattern) * np.cos(2 * lon_pattern)

    # Generate data for each variable
    variables = ['nsma', 'ssma', 'tvega', 'eta', 'swa']
    products = ['ERA5L', 'GLEAM', 'GLDAS']

    data = {}

    for var in variables:
        data[var] = {}

        # Create true signal with temporal variation
        true_signal = np.zeros((n_time, n_lat, n_lon))
        for t in range(n_time):
            # Seasonal pattern
            seasonal = np.sin(2 * np.pi * t / 12)
            # Trend
            trend = 0.01 * t / n_time

            true_signal[t, :, :] = spatial_pattern * (1 + 0.3 * seasonal + trend)

        # Add different noise levels for each product
        # ERA5L: lowest error
        noise_era5l = 0.15 * np.random.randn(n_time, n_lat, n_lon)
        data[var]['ERA5L'] = true_signal + noise_era5l

        # GLEAM: medium error
        noise_gleam = 0.25 * np.random.randn(n_time, n_lat, n_lon)
        data[var]['GLEAM'] = true_signal + noise_gleam

        # GLDAS: higher error, with some correlation to GLEAM
        common_error = 0.1 * np.random.randn(n_time, n_lat, n_lon)
        noise_gldas = common_error + 0.30 * np.random.randn(n_time, n_lat, n_lon)
        data[var]['GLDAS'] = true_signal + noise_gldas

        # Add some NaN values (realistic for satellite data)
        for product in products:
            mask = np.random.rand(n_time, n_lat, n_lon) < 0.05  # 5% missing
            data[var][product][mask] = np.nan

    print("  Synthetic data generated successfully")
    return data


def example_1_dual_ivd():
    """
    Example 1: Process two datasets with IVD
    (Corresponds to ELG38a_IVD.m in MATLAB code)
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Dual Dataset Processing with IVD")
    print("="*70)
    print("\nThis example processes ERA5L + GLEAM data (1980-1999 equivalent)")
    print("using the IVD (Instrumental Variable Design) method.\n")

    # Generate data
    data = generate_synthetic_eli_data(n_time=240, n_lat=30, n_lon=30)

    # Initialize processor
    processor = ELIProcessor()

    # Process each variable with IVD
    results_ivd = {}
    for var in ['nsma', 'ssma', 'tvega', 'eta']:
        print(f"\nProcessing {var.upper()}:")
        results_ivd[var] = processor.process_dual_ivd(
            data[var]['ERA5L'],
            data[var]['GLEAM'],
            variable=var
        )

    # Display summary statistics
    print("\n" + "-"*70)
    print("IVD RESULTS SUMMARY")
    print("-"*70)

    for var in results_ivd:
        result = results_ivd[var]
        print(f"\n{var.upper()}:")

        error_var = result['error_variance']
        print(f"  ERA5L  - Mean error variance: {np.nanmean(error_var[:,:,0]):.6f}")
        print(f"  GLEAM  - Mean error variance: {np.nanmean(error_var[:,:,1]):.6f}")

        rho2 = result['rho2']
        print(f"  ERA5L  - Mean correlation: {np.nanmean(rho2[:,:,0]):.4f}")
        print(f"  GLEAM  - Mean correlation: {np.nanmean(rho2[:,:,1]):.4f}")

        weights = result['weights']
        print(f"  ERA5L  - Mean weight: {np.nanmean(weights[:,:,0]):.4f}")
        print(f"  GLEAM  - Mean weight: {np.nanmean(weights[:,:,1]):.4f}")

    return results_ivd


def example_2_triple_eivd():
    """
    Example 2: Process three datasets with EIVD
    (Corresponds to ELG21G38_EIVD.m in MATLAB code)
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Triple Dataset Processing with EIVD")
    print("="*70)
    print("\nThis example processes ERA5L + GLEAM + GLDAS data (2000-2022 equivalent)")
    print("using the EIVD (Extended IVD) method.\n")

    # Generate data
    data = generate_synthetic_eli_data(n_time=276, n_lat=30, n_lon=30)

    # Initialize processor
    processor = ELIProcessor()

    # Process each variable with EIVD
    results_eivd = {}
    for var in ['nsma', 'ssma', 'tvega', 'eta']:
        print(f"\nProcessing {var.upper()}:")
        results_eivd[var] = processor.process_triple_eivd(
            data[var]['ERA5L'],
            data[var]['GLEAM'],
            data[var]['GLDAS'],
            variable=var
        )

    # Display summary statistics including error cross-correlation
    print("\n" + "-"*70)
    print("EIVD RESULTS SUMMARY")
    print("-"*70)

    for var in results_eivd:
        result = results_eivd[var]
        print(f"\n{var.upper()}:")

        error_var = result['error_variance']
        print(f"  ERA5L  - Mean error variance: {np.nanmean(error_var[:,:,0]):.6f}")
        print(f"  GLEAM  - Mean error variance: {np.nanmean(error_var[:,:,1]):.6f}")
        print(f"  GLDAS  - Mean error variance: {np.nanmean(error_var[:,:,2]):.6f}")

        ecc = result['error_cross_correlation']
        print(f"  Error cross-correlation (GLEAM-GLDAS): {np.nanmean(ecc[:,:,0]):.6f}")
        print(f"  Error cross-correlation (ERA5L-GLDAS): {np.nanmean(ecc[:,:,1]):.6f}")
        print(f"  Error cross-correlation (ERA5L-GLEAM): {np.nanmean(ecc[:,:,2]):.6f}")

        rho2 = result['rho2']
        print(f"  Mean correlations: ERA5L={np.nanmean(rho2[:,:,0]):.4f}, "
              f"GLEAM={np.nanmean(rho2[:,:,1]):.4f}, "
              f"GLDAS={np.nanmean(rho2[:,:,2]):.4f}")

    return results_eivd


def example_3_all_methods():
    """
    Example 3: Compare all collocation methods
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Comparison of All Methods")
    print("="*70)
    print("\nThis example applies ALL available collocation methods:")
    print("  - IVD (2 products)")
    print("  - EIVD (3 products, with error correlation)")
    print("  - TC (3 products, assumes independent errors)")
    if BAYESIAN_AVAILABLE:
        print("  - Bayesian TC (3 products, time-varying errors)")
    print()

    # Generate data
    data = generate_synthetic_eli_data(n_time=200, n_lat=20, n_lon=20)

    # Initialize processor
    processor = ELIProcessor()

    # Process one variable with all methods
    var = 'eta'  # Total evapotranspiration
    print(f"\nProcessing {var.upper()} with all methods:")

    results_all = processor.process_triple_with_all_methods(
        data[var]['ERA5L'],
        data[var]['GLEAM'],
        data[var]['GLDAS'],
        variable=var,
        use_bayesian=False  # Set to True to include Bayesian (slower)
    )

    # Compare methods
    print("\n" + "-"*70)
    print("METHOD COMPARISON")
    print("-"*70)

    methods = ['eivd', 'tc']
    for method in methods:
        if method in results_all:
            result = results_all[method]
            print(f"\n{method.upper()} Method:")

            if 'error_variance' in result:
                error_var = result['error_variance']
                print(f"  Mean error variances: "
                      f"[{np.nanmean(error_var[:,:,0]):.6f}, "
                      f"{np.nanmean(error_var[:,:,1]):.6f}, "
                      f"{np.nanmean(error_var[:,:,2]):.6f}]")

            if 'error_cross_correlation' in result:
                ecc = result['error_cross_correlation']
                print(f"  Mean error cross-correlation: {np.nanmean(np.abs(ecc)):.6f}")

    # Display comparison summary
    if 'comparison' in results_all:
        comp = results_all['comparison']
        print("\nRECOMMENDATIONS:")
        for rec in comp.get('recommendations', []):
            print(f"  • {rec}")

    return results_all


def example_4_calculate_eli():
    """
    Example 4: Calculate Ecosystem Limitation Index
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Calculate Ecosystem Limitation Index (ELI)")
    print("="*70)
    print("\nThe ELI quantifies water vs. energy limitation in ecosystems.")
    print("Positive values = water limitation")
    print("Negative values = energy limitation\n")

    # Generate data
    data = generate_synthetic_eli_data(n_time=200, n_lat=30, n_lon=30)

    # Process all variables
    processor = ELIProcessor()

    merged_results = {}
    for var in ['nsma', 'ssma', 'tvega', 'eta', 'swa']:
        print(f"\nProcessing {var.upper()}...")

        if var == 'swa':
            # SWA: only ERA5L and GLDAS (as in MATLAB code)
            result = processor.process_dual_ivd(
                data[var]['ERA5L'],
                data[var]['GLDAS'],
                variable=var
            )
        else:
            # Other variables: all three products
            result = processor.process_triple_eivd(
                data[var]['ERA5L'],
                data[var]['GLEAM'],
                data[var]['GLDAS'],
                variable=var
            )

        merged_results[var] = result['merged']

    # Calculate ELI
    print("\nCalculating ELI index...")
    # Note: This is a simplified calculation
    # The actual formulation should match the reference paper

    # Extract spatial means for display
    eli_components = {
        'soil_moisture': np.nanmean(
            (merged_results['nsma'] + merged_results['ssma']) / 2,
            axis=2
        ),
        'evapotranspiration': np.nanmean(merged_results['eta'], axis=2),
        'radiation': np.nanmean(merged_results['swa'], axis=2),
    }

    # Simple ELI formulation (example)
    eli = -eli_components['soil_moisture'] + 0.5 * eli_components['radiation']

    print(f"\nELI Statistics:")
    print(f"  Mean ELI: {np.nanmean(eli):.4f}")
    print(f"  Std ELI:  {np.nanstd(eli):.4f}")
    print(f"  Min ELI:  {np.nanmin(eli):.4f}")
    print(f"  Max ELI:  {np.nanmax(eli):.4f}")

    # Calculate fraction of water-limited vs energy-limited areas
    water_limited = np.sum(eli > 0) / np.sum(~np.isnan(eli))
    energy_limited = np.sum(eli < 0) / np.sum(~np.isnan(eli))

    print(f"\n  Water-limited areas:  {water_limited*100:.1f}%")
    print(f"  Energy-limited areas: {energy_limited*100:.1f}%")

    return eli, merged_results


def example_5_export_netcdf():
    """
    Example 5: Export results to NetCDF files
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Export Results to NetCDF")
    print("="*70)

    # Generate data
    data = generate_synthetic_eli_data(n_time=120, n_lat=20, n_lon=20)

    # Process one variable
    processor = ELIProcessor()

    var = 'eta'
    print(f"\nProcessing {var.upper()} for export...")

    result = processor.process_triple_eivd(
        data[var]['ERA5L'],
        data[var]['GLEAM'],
        data[var]['GLDAS'],
        variable=var
    )

    # Export to NetCDF
    output_dir = Path('eli_results')
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f'eli_{var}_eivd_results.nc'

    print(f"\nExporting to: {output_file}")

    processor.save_to_netcdf(
        result,
        output_file,
        variable=var,
        data_source='ERA5L+GLEAM+GLDAS',
        metadata={
            'description': 'ELI collocation analysis results',
            'reference': 'Widespread shift from ecosystem energy to water limitation',
            'author': 'Converted from MATLAB',
        }
    )

    print("\nNetCDF file structure:")
    print("  Variables:")
    print("    - error_variance_product[1-3]: Error variances")
    print("    - rho2_product[1-3]: Data-truth correlations")
    print("    - weight_product[1-3]: Merging weights")
    print("    - merged: Merged product time series")
    print("    - error_cross_corr_[1-3]: Error cross-correlations")

    return output_file


def example_6_time_series_analysis():
    """
    Example 6: Time series analysis at specific locations
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Time Series Analysis at Specific Locations")
    print("="*70)

    # Generate data
    data = generate_synthetic_eli_data(n_time=240, n_lat=30, n_lon=30)

    # Select a specific location
    i_lat, i_lon = 15, 15  # Middle of domain

    print(f"\nAnalyzing location: lat_idx={i_lat}, lon_idx={i_lon}")

    var = 'eta'

    # Extract time series for this location
    era5l_ts = data[var]['ERA5L'][:, i_lat, i_lon]
    gleam_ts = data[var]['GLEAM'][:, i_lat, i_lon]
    gldas_ts = data[var]['GLDAS'][:, i_lat, i_lon]

    # Apply collocation methods
    tri = np.column_stack([era5l_ts, gleam_ts, gldas_ts])

    # Remove NaN rows
    valid = ~np.any(np.isnan(tri), axis=1)
    tri_clean = tri[valid, :]

    print(f"\nApplying collocation methods to {tri_clean.shape[0]} valid time steps...")

    # EIVD
    EeeT_eivd, SNR_eivd, rho2_eivd, fMSE_eivd, L_eivd = eivd(tri_clean)

    print(f"\nEIVD Results:")
    print(f"  Error variances: {np.diag(EeeT_eivd)}")
    print(f"  Error cross-corr (GLEAM-GLDAS): {EeeT_eivd[1, 2]:.6f}")
    print(f"  Correlations: {rho2_eivd}")
    print(f"  Lag-1 autocorrelations: {L_eivd}")

    # TC
    EeeT_tc, SNR_tc, rho2_tc, fMSE_tc = tc(tri_clean)

    print(f"\nTC Results:")
    print(f"  Error variances: {np.diag(EeeT_tc)}")
    print(f"  SNR: {SNR_tc}")
    print(f"  Correlations: {rho2_tc}")

    # Compare
    var_diff = np.abs(np.diag(EeeT_eivd) - np.diag(EeeT_tc))
    print(f"\nDifference in error variance estimates:")
    print(f"  ERA5L: {var_diff[0]:.6f}")
    print(f"  GLEAM: {var_diff[1]:.6f}")
    print(f"  GLDAS: {var_diff[2]:.6f}")

    # Create merged product
    weights = 1.0 / (np.diag(EeeT_eivd) + 1e-10)
    weights = weights / np.sum(weights)

    merged = np.sum(tri_clean * weights, axis=1)

    print(f"\nMerged product created with weights:")
    print(f"  ERA5L: {weights[0]:.4f}")
    print(f"  GLEAM: {weights[1]:.4f}")
    print(f"  GLDAS: {weights[2]:.4f}")

    return tri_clean, merged, weights


def main():
    """
    Run all ELI examples.
    """
    print("\n" + "="*80)
    print(" "*20 + "ELI COMPREHENSIVE EXAMPLES")
    print("="*80)
    print("\nThis script demonstrates the complete ELI workflow using all")
    print("available collocation methods from the converted MATLAB code.")
    print("\nReference: 'Widespread shift from ecosystem energy to water")
    print("           limitation with climate change'")
    print("="*80)

    # Run examples
    print("\n\nPress Enter to run Example 1 (Dual IVD)...")
    # input()  # Uncomment for interactive mode
    results_1 = example_1_dual_ivd()

    print("\n\nPress Enter to run Example 2 (Triple EIVD)...")
    # input()
    results_2 = example_2_triple_eivd()

    print("\n\nPress Enter to run Example 3 (All Methods Comparison)...")
    # input()
    results_3 = example_3_all_methods()

    print("\n\nPress Enter to run Example 4 (Calculate ELI)...")
    # input()
    eli, merged = example_4_calculate_eli()

    print("\n\nPress Enter to run Example 5 (Export NetCDF)...")
    # input()
    output_file = example_5_export_netcdf()

    print("\n\nPress Enter to run Example 6 (Time Series Analysis)...")
    # input()
    ts_data, ts_merged, ts_weights = example_6_time_series_analysis()

    # Final summary
    print("\n\n" + "="*80)
    print(" "*25 + "EXAMPLES COMPLETED")
    print("="*80)
    print("\nAll ELI examples have been successfully executed!")
    print("\nKey takeaways:")
    print("  1. IVD is suitable for 2 data products")
    print("  2. EIVD handles 3+ products and error cross-correlation")
    print("  3. TC assumes independent errors (simpler but less general)")
    print("  4. Bayesian TC provides time-varying error estimates")
    print("  5. All methods are integrated in the ELIProcessor class")
    print("\nFor real data applications:")
    print("  - Replace synthetic data with actual NetCDF files")
    print("  - Use appropriate spatial/temporal domains")
    print("  - Validate results against ground truth where available")
    print("  - Consider computational resources for large domains")
    print("\nResults exported to: eli_results/")
    print("="*80)


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    main()
