"""
Example Usage of HCC-ET Framework
==================================

Demonstrates how to use the HCC-ET Framework for T/ET decomposition.

This example shows:
1. Basic usage with configuration file
2. Advanced usage with custom parameters
3. Individual phase execution
4. Result visualization and analysis

Author: [Your Name]
Date: 2025-11-09
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

# Import HCC-ET Framework
from hcc_et_framework import (
    HCCETFramework,
    load_config,
    create_default_config,
    data_loader,
    downscaler,
    physical_constraints,
    utils_hcc
)


def example_1_basic_usage():
    """
    Example 1: Basic Usage with Configuration File
    -----------------------------------------------
    Simplest way to use the framework - just provide a config file.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60 + "\n")

    # Create default config template (first time only)
    if not Path('my_config.yaml').exists():
        create_default_config('my_config.yaml')
        print("Created config template: my_config.yaml")
        print("Please edit this file with your data paths, then re-run.\n")
        return

    # Initialize framework
    framework = HCCETFramework(
        config_file='my_config.yaml',
        output_dir='./outputs_example1'
    )

    # Run full pipeline (all 5 phases)
    framework.run_full_pipeline()

    # Access results
    print("\nResults summary:")
    print(f"  T/ET ratio shape: {framework.fused_t_et_ratio.shape}")
    print(f"  T (transpiration) shape: {framework.T_final_1km.shape}")
    print(f"  E (evaporation) shape: {framework.E_final_1km.shape}")

    print("\nOutputs saved to: ./outputs_example1/")


def example_2_advanced_usage():
    """
    Example 2: Advanced Usage with Custom Parameters
    -------------------------------------------------
    More control over individual components.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Advanced Usage")
    print("=" * 60 + "\n")

    # Load configuration
    config = load_config('my_config.yaml')

    # Modify configuration on-the-fly
    config['fusion']['mode'] = 'eivd'  # Force EIVD mode
    config['fusion']['spatial_varying_weights'] = False  # Use global weights
    config['downscaling']['model_type'] = 'random_forest'

    # Initialize framework
    framework = HCCETFramework(
        config_file='my_config.yaml',  # Can still use base config
        output_dir='./outputs_example2'
    )

    # Override internal config
    framework.cfg = config

    # Run individual phases with custom logic
    print("Running Phase 1: Data Preparation...")
    framework.run_phase1_data_preparation()

    print("\nRunning Phase 2: Downscaling...")
    framework.run_phase2_downscaling()

    # Custom processing between phases
    print("\nCustom analysis of downscaled candidates...")
    candidates = framework.t_et_candidates_1km
    print(f"  Candidates shape: {candidates.shape}")
    print(f"  Mean T/ET by model:")
    for model in candidates.coords['model'].values:
        mean_val = candidates.sel(model=model).mean().values
        print(f"    {model:15s}: {mean_val:.3f}")

    print("\nRunning Phase 3: Fusion...")
    framework.run_phase3_fusion()

    print("\nRunning Phase 4: Physical Constraints...")
    framework.run_phase4_physical_constraints()

    print("\nRunning Phase 5: Temporal Hindcast...")
    framework.run_phase5_temporal_hindcast()

    print("\nAdvanced pipeline complete!")


def example_3_custom_fusion():
    """
    Example 3: Custom Fusion Workflow
    ----------------------------------
    Direct use of collocation methods without full framework.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Custom Fusion Workflow")
    print("=" * 60 + "\n")

    # This example uses synthetic data for demonstration
    print("Generating synthetic data...")

    # Synthetic "parent" T/ET ratios (3 products)
    n_time, n_lat, n_lon = 100, 50, 50
    true_signal = np.random.rand(n_time, n_lat, n_lon) * 0.7 + 0.2  # T/ET in [0.2, 0.9]

    # Add different noise to each product
    parent1 = true_signal + np.random.randn(n_time, n_lat, n_lon) * 0.05
    parent2 = true_signal + np.random.randn(n_time, n_lat, n_lon) * 0.08
    parent3 = true_signal + np.random.randn(n_time, n_lat, n_lon) * 0.10

    # Stack into xarray
    parent_data = xr.DataArray(
        np.stack([parent1, parent2, parent3], axis=-1),
        dims=['time', 'lat', 'lon', 'model'],
        coords={
            'time': np.arange(n_time),
            'lat': np.linspace(-90, 90, n_lat),
            'lon': np.linspace(-180, 180, n_lon),
            'model': ['GLEAM', 'PMLv2', 'GLDAS']
        }
    )

    # Synthetic predictors (1km = same resolution for demo)
    lst = np.random.randn(n_time, n_lat, n_lon) * 10 + 20
    ndvi = np.random.rand(n_time, n_lat, n_lon) * 0.8
    albedo = np.random.rand(n_time, n_lat, n_lon) * 0.3
    precip = np.random.rand(n_time, n_lat, n_lon) * 100

    predictors = xr.DataArray(
        np.stack([lst, ndvi, albedo, precip], axis=-1),
        dims=['time', 'lat', 'lon', 'variable'],
        coords={
            'time': np.arange(n_time),
            'lat': np.linspace(-90, 90, n_lat),
            'lon': np.linspace(-180, 180, n_lon),
            'variable': ['LST', 'NDVI', 'Albedo', 'Precip']
        }
    )

    # Step 1: Train downscaler
    print("\nStep 1: Training downscaler...")
    model = downscaler.train_downscaler(
        parent_data=parent_data,
        predictors_1km=predictors,
        model_type='random_forest',
        hyperparams={'n_estimators': 50, 'max_depth': 10, 'random_state': 42},
        cv_folds=3
    )

    # Step 2: Generate 1km candidates
    print("\nStep 2: Generating 1km candidates...")
    candidates = downscaler.predict(model, predictors)

    # Step 3: Apply EIVD fusion manually
    print("\nStep 3: Applying EIVD fusion...")
    from collocation import eivd

    # Stack data for EIVD
    data_stacked = candidates.stack(sample=('time', 'lat', 'lon')).values.T

    # Run EIVD
    EeeT, SNR, rho2, fMSE, L = eivd(data_stacked)

    print("\nEIVD Results:")
    print(f"  Error variances: {np.diag(EeeT)}")
    print(f"  Error cross-correlation (2-3): {EeeT[1,2]:.4f}")
    print(f"  SNR: {SNR}")
    print(f"  Data-truth correlations: {rho2}")

    # Compute fusion weights
    EeeT_inv = np.linalg.inv(EeeT)
    ones = np.ones(3)
    weights = EeeT_inv @ ones / (ones @ EeeT_inv @ ones)

    print(f"\nOptimal fusion weights:")
    for model, weight in zip(candidates.coords['model'].values, weights):
        print(f"  {model:15s}: {weight:.4f}")

    # Apply weights
    weights_xr = xr.DataArray(
        weights,
        dims=['model'],
        coords={'model': candidates.coords['model']}
    )

    fused = (candidates * weights_xr).sum(dim='model')

    print(f"\nFused T/ET ratio shape: {fused.shape}")
    print(f"  Mean: {fused.mean().values:.3f}")
    print(f"  Std: {fused.std().values:.3f}")

    # Compare with simple average
    simple_avg = candidates.mean(dim='model')
    improvement = np.sqrt(((true_signal - fused.values)**2).mean()) / \
                  np.sqrt(((true_signal - simple_avg.values)**2).mean())

    print(f"\nFusion improvement over simple average: {(1-improvement)*100:.1f}%")


def example_4_trend_analysis():
    """
    Example 4: Trend Analysis and Visualization
    --------------------------------------------
    Analyze trends from HCC-ET output.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Trend Analysis")
    print("=" * 60 + "\n")

    # Generate synthetic time series (1950-2025)
    print("Generating synthetic T/ET time series...")
    n_years = 75
    n_lat, n_lon = 50, 50

    # Add realistic trend: +0.002 per year (0.02 per decade)
    years = np.arange(n_years)
    trend = 0.002 * years[:, None, None]
    seasonal = 0.05 * np.sin(2 * np.pi * years[:, None, None] / 1)
    noise = np.random.randn(n_years, n_lat, n_lon) * 0.02
    base = 0.5

    t_et_ratio = base + trend + seasonal + noise

    # Convert to xarray
    t_et_xr = xr.DataArray(
        t_et_ratio,
        dims=['time', 'lat', 'lon'],
        coords={
            'time': np.arange(1950, 2025),
            'lat': np.linspace(-90, 90, n_lat),
            'lon': np.linspace(-180, 180, n_lon)
        },
        name='T_ET_ratio'
    )

    # Compute spatial trends
    print("\nComputing spatial trends...")
    slopes, pvals = utils_hcc.compute_spatial_trends(
        t_et_xr,
        dim='time',
        method='theil_sen'
    )

    # Convert to decadal trends
    decadal_trends = slopes * 10

    # Mask significant trends
    significant = decadal_trends.where(pvals < 0.05)

    print(f"\nTrend statistics:")
    print(f"  Mean trend: {decadal_trends.mean().values:.4f} per decade")
    print(f"  Std trend: {decadal_trends.std().values:.4f}")
    print(f"  % significant pixels: {100 * (pvals < 0.05).sum() / pvals.size:.1f}%")

    # Visualization
    print("\nCreating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Decadal trends
    im1 = axes[0].imshow(
        decadal_trends.values,
        cmap='RdBu_r',
        vmin=-0.05,
        vmax=0.05,
        extent=[-180, 180, -90, 90],
        aspect='auto'
    )
    axes[0].set_title('T/ET Trends (1950-2025, per decade)')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[0], label='Trend (per decade)')

    # Plot 2: Significance mask
    sig_mask = (pvals < 0.05).astype(int)
    im2 = axes[1].imshow(
        sig_mask,
        cmap='Greys',
        extent=[-180, 180, -90, 90],
        aspect='auto'
    )
    axes[1].set_title('Significant Trends (p < 0.05)')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')

    plt.tight_layout()
    plt.savefig('example4_trends.png', dpi=300, bbox_inches='tight')
    print("  Saved: example4_trends.png")

    # Time series for a sample pixel
    plt.figure(figsize=(10, 4))
    sample_ts = t_et_xr.isel(lat=25, lon=25)
    plt.plot(sample_ts.coords['time'], sample_ts.values, 'o-', alpha=0.6, label='T/ET ratio')

    # Fit trend line
    from scipy.stats import theilslopes
    result = theilslopes(sample_ts.values, sample_ts.coords['time'].values)
    trend_line = result.slope * sample_ts.coords['time'].values + result.intercept
    plt.plot(sample_ts.coords['time'], trend_line, 'r--', label=f'Trend: {result.slope*10:.4f}/decade')

    plt.xlabel('Year')
    plt.ylabel('T/ET Ratio')
    plt.title('Sample Pixel Time Series (Lat=25°, Lon=25°)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('example4_timeseries.png', dpi=300, bbox_inches='tight')
    print("  Saved: example4_timeseries.png")

    print("\nTrend analysis complete!")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("HCC-ET Framework: Example Usage")
    print("=" * 60)

    # Example 1: Basic usage (requires real data)
    # example_1_basic_usage()

    # Example 2: Advanced usage (requires real data)
    # example_2_advanced_usage()

    # Example 3: Custom fusion (uses synthetic data)
    example_3_custom_fusion()

    # Example 4: Trend analysis (uses synthetic data)
    example_4_trend_analysis()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
