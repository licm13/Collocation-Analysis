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
6. Visualizing spatial results in Nature/Science style.

Author: Converted from MATLAB by Claude
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib import rcParams
from pathlib import Path
import warnings

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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


# ============================================================================
# 🌟 VISUALIZATION FUNCTIONS (Nature/Science Style) 🌟
# ============================================================================

def setup_publication_style():
    """Setup matplotlib for publication-quality figures (Nature/Science style)."""

    # Font settings
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    rcParams['font.size'] = 7
    rcParams['axes.labelsize'] = 8
    rcParams['axes.titlesize'] = 9
    rcParams['xtick.labelsize'] = 7
    rcParams['ytick.labelsize'] = 7
    rcParams['legend.fontsize'] = 6

    # Line and marker settings
    rcParams['lines.linewidth'] = 1.0
    rcParams['lines.markersize'] = 3
    rcParams['patch.linewidth'] = 0.75

    # Axes settings
    rcParams['axes.linewidth'] = 0.75
    rcParams['axes.grid'] = False  # Grids are often avoided in spatial maps
    rcParams['grid.alpha'] = 0.3
    rcParams['grid.linewidth'] = 0.5

    # Tick settings
    rcParams['xtick.major.width'] = 0.75
    rcParams['ytick.major.width'] = 0.75
    rcParams['xtick.minor.width'] = 0.5
    rcParams['ytick.minor.width'] = 0.5

    # Figure settings
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.pad_inches'] = 0.05

    # Legend settings
    rcParams['legend.frameon'] = True
    rcParams['legend.framealpha'] = 0.9
    rcParams['legend.edgecolor'] = 'gray'
    rcParams['legend.fancybox'] = False


def plot_spatial_map(ax, data, title, cmap, cbar_label, vmin=None, vmax=None):
    """Helper function to plot a single spatial map."""
    
    # Handle NaNs by setting them to a specific color (e.g., gray)
    cmap_obj = cm.get_cmap(cmap)
    cmap_obj.set_bad('gray', 0.1) # type: ignore

    im = ax.imshow(data, cmap=cmap_obj, origin='lower', vmin=vmin, vmax=vmax,
                   interpolation='nearest')
    ax.set_title(title, fontweight='bold', loc='left')
    ax.set_xlabel('Longitude Index')
    ax.set_ylabel('Latitude Index')
    
    # Hide ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.8, pad=0.03)


def plot_ivd_results(result, var_name, fig_path):
    """Plots the results from a dual IVD analysis."""
    setup_publication_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.5)) # 183mm width
    
    var_data = result['error_variance']
    rho_data = result['rho2']
    
    # Plot Error Variances
    vmax_var = np.nanmax(var_data)
    plot_spatial_map(axes[0, 0], var_data[:,:,0], 'a) Error Variance (Prod 1)',
                     'viridis', 'Error Variance', vmin=0, vmax=vmax_var)
    plot_spatial_map(axes[0, 1], var_data[:,:,1], 'b) Error Variance (Prod 2)',
                     'viridis', 'Error Variance', vmin=0, vmax=vmax_var)
    
    # Plot Correlations
    plot_spatial_map(axes[1, 0], rho_data[:,:,0], 'c) Correlation (Prod 1)',
                     'RdYlBu_r', r'Correlation ($\rho^2$)', vmin=0, vmax=1)
    plot_spatial_map(axes[1, 1], rho_data[:,:,1], 'd) Correlation (Prod 2)',
                     'RdYlBu_r', r'Correlation ($\rho^2$)', vmin=0, vmax=1)
    
    fig.suptitle(f'IVD Results for {var_name.upper()}', fontweight='bold', y=1.02)
    plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=1.0)
    plt.savefig(fig_path)
    print(f"  Saved IVD plot: {fig_path}")
    plt.close(fig)


def plot_eivd_results(result, var_name, fig_path):
    """Plots the results from a triple EIVD analysis."""
    setup_publication_style()
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 9.5)) # Larger fig for 3x3
    
    var_data = result['error_variance']
    rho_data = result['rho2']
    ecc_data = result['error_cross_correlation']

    vmax_var = np.nanmax(var_data)
    
    # Row 1: Error Variances
    plot_spatial_map(axes[0, 0], var_data[:,:,0], 'a) Error Var (Prod 1)',
                     'viridis', 'Error Variance', vmin=0, vmax=vmax_var)
    plot_spatial_map(axes[0, 1], var_data[:,:,1], 'b) Error Var (Prod 2)',
                     'viridis', 'Error Variance', vmin=0, vmax=vmax_var)
    plot_spatial_map(axes[0, 2], var_data[:,:,2], 'c) Error Var (Prod 3)',
                     'viridis', 'Error Variance', vmin=0, vmax=vmax_var)
    
    # Row 2: Correlations
    plot_spatial_map(axes[1, 0], rho_data[:,:,0], r'd) $\rho^2$ (Prod 1)',
                     'RdYlBu_r', r'Correlation ($\rho^2$)', vmin=0, vmax=1)
    plot_spatial_map(axes[1, 1], rho_data[:,:,1], r'e) $\rho^2$ (Prod 2)',
                     'RdYlBu_r', r'Correlation ($\rho^2$)', vmin=0, vmax=1)
    plot_spatial_map(axes[1, 2], rho_data[:,:,2], r'f) $\rho^2$ (Prod 3)',
                     'RdYlBu_r', r'Correlation ($\rho^2$)', vmin=0, vmax=1)

    # Row 3: Error Cross-Correlations
    vmax_ecc = np.nanmax(np.abs(ecc_data))
    plot_spatial_map(axes[2, 0], ecc_data[:,:,0], 'g) ECC (P2-P3)',
                     'coolwarm', 'ECC', vmin=-vmax_ecc, vmax=vmax_ecc)
    plot_spatial_map(axes[2, 1], ecc_data[:,:,1], 'h) ECC (P1-P3)',
                     'coolwarm', 'ECC', vmin=-vmax_ecc, vmax=vmax_ecc)
    plot_spatial_map(axes[2, 2], ecc_data[:,:,2], 'i) ECC (P1-P2)',
                     'coolwarm', 'ECC', vmin=-vmax_ecc, vmax=vmax_ecc)

    fig.suptitle(f'EIVD Results for {var_name.upper()}', fontweight='bold', y=1.0)
    plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=1.0)
    plt.savefig(fig_path)
    print(f"  Saved EIVD plot: {fig_path}")
    plt.close(fig)


def plot_eli_index(eli_map, fig_path):
    """Plots the final ELI index map."""
    setup_publication_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5)) # Single panel
    
    # Center colormap on zero
    vmax_abs = np.nanmax(np.abs(eli_map))
    
    plot_spatial_map(ax, eli_map, 'Ecosystem Limitation Index (ELI)',
                     'coolwarm_r', 'ELI (Energy-Lim < 0 > Water-Lim)',
                     vmin=-vmax_abs, vmax=vmax_abs)
    
    plt.tight_layout(pad=0.5)
    plt.savefig(fig_path)
    print(f"  Saved ELI plot: {fig_path}")
    plt.close(fig)


def plot_timeseries_comparison(ts_data, ts_merged, weights, fig_path):
    """Plots the time series comparison from Example 6."""
    setup_publication_style()

    # Use a wider, shorter figure for time series
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.5)) 
    
    # Plot individual products with transparency
    ax.plot(ts_data[:, 0], label=f'Prod 1 (ERA5L) | w={weights[0]:.2f}',
            color='#0173B2', alpha=0.5, linewidth=0.75)
    ax.plot(ts_data[:, 1], label=f'Prod 2 (GLEAM) | w={weights[1]:.2f}',
            color='#029E73', alpha=0.5, linewidth=0.75)
    ax.plot(ts_data[:, 2], label=f'Prod 3 (GLDAS) | w={weights[2]:.2f}',
            color='#DE8F05', alpha=0.5, linewidth=0.75)
    
    # Plot merged product
    ax.plot(ts_merged, label='Merged (EIVD)', color='black', linewidth=1.2)
    
    ax.set_title('Time Series Comparison at Single Location (ETA)',
                 fontweight='bold', loc='left')
    ax.set_xlabel('Time Step (Month)')
    ax.set_ylabel('Variable Value (ETA)')
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3, linewidth=0.5) # Add grid for time series
    
    # Remove top and right spines for N/S style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout(pad=0.5)
    plt.savefig(fig_path)
    print(f"  Saved Time Series plot: {fig_path}")
    plt.close(fig)

# ============================================================================
# END VISUALIZATION FUNCTIONS
# ============================================================================


def example_1_dual_ivd(fig_dir):
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

    # Plot results for one variable ('eta')
    if 'eta' in results_ivd:
        plot_ivd_results(results_ivd['eta'], 'ETA',
                         fig_dir / 'ex1_ivd_results_eta.png')

    return results_ivd


def example_2_triple_eivd(fig_dir):
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

    # Plot results for one variable ('eta')
    if 'eta' in results_eivd:
        plot_eivd_results(results_eivd['eta'], 'ETA',
                          fig_dir / 'ex2_eivd_results_eta.png')

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


def example_4_calculate_eli(fig_dir):
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
    # We take the mean over time to get a single spatial map
    eli_components = {
        'soil_moisture': np.nanmean(
            (merged_results['nsma'] + merged_results['ssma']) / 2,
            axis=0  # Mean over time axis (0)
        ),
        'evapotranspiration': np.nanmean(merged_results['eta'], axis=0),
        'radiation': np.nanmean(merged_results['swa'], axis=0),
    }

    # Simple ELI formulation (example)
    # This creates a 2D (lat, lon) map
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
    
    # Plot the final ELI map
    plot_eli_index(eli, fig_dir / 'ex4_eli_index.png')

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
    output_dir = (Path(__file__).resolve().parent / 'eli_results')
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
    print("  - error_variance_product[1-3]: Error variances")
    print("  - rho2_product[1-3]: Data-truth correlations")
    print("  - weight_product[1-3]: Merging weights")
    print("  - merged: Merged product time series")
    print("  - error_cross_corr_[1-3]: Error cross-correlations")

    return output_file


def example_6_time_series_analysis(fig_dir):
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
    
    # Plot the time series
    plot_timeseries_comparison(tri_clean, merged, weights,
                               fig_dir / 'ex6_timeseries_comparison.png')

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
    
    # Setup publication style globally
    setup_publication_style()
    
    # Create directory for figures
    # Save figures to a folder under this script's directory
    FIG_DIR = (Path(__file__).resolve().parent / 'eli_figures')
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving figures to: {FIG_DIR.resolve()}")

    # Run examples
    print("\n\nPress Enter to run Example 1 (Dual IVD)...")
    # input()  # Uncomment for interactive mode
    results_1 = example_1_dual_ivd(FIG_DIR)

    print("\n\nPress Enter to run Example 2 (Triple EIVD)...")
    # input()
    results_2 = example_2_triple_eivd(FIG_DIR)

    print("\n\nPress Enter to run Example 3 (All Methods Comparison)...")
    # input()
    results_3 = example_3_all_methods()

    print("\n\nPress Enter to run Example 4 (Calculate ELI)...")
    # input()
    eli, merged = example_4_calculate_eli(FIG_DIR)

    print("\n\nPress Enter to run Example 5 (Export NetCDF)...")
    # input()
    output_file = example_5_export_netcdf()

    print("\n\nPress Enter to run Example 6 (Time Series Analysis)...")
    # input()
    ts_data, ts_merged, ts_weights = example_6_time_series_analysis(FIG_DIR)

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
    print(f"\nResults exported to: {FIG_DIR.resolve()} (Figures)")
    print(f"                   eli_results/ (NetCDF data)")
    print("="*80)
    
    print("\nShowing plots (if not in interactive mode)...")
    # plt.show() # Uncomment if you want plots to pop up


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    main()