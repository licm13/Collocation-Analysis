"""
Complex 3D Spatial Data Fusion Example
=======================================

This example demonstrates how to process real-world satellite-like data
with multiple dimensions (Time, Latitude, Longitude), handle missing values
(NaN from cloud masking), and perform optimal fusion using Triple Collocation.

场景说明：
- 模拟处理真实的卫星 NetCDF 数据（维度：Time, Latitude, Longitude）
- 使用 xarray 生成 3 个大小为 (100, 50, 50) 的 DataArray
- 在特定纬度带随机插入 NaN（模拟云层遮挡）
- 使用 collocation.tc 对每个像素的时间序列进行计算
- 基于 TC 计算出的误差方差，计算最优融合权重
- 将结果保存为 NetCDF 文件

Author: Claude (based on collocation package)
Date: 2024
"""

import os
import sys
import warnings

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xarray as xr

# Import TC method
from collocation import tc

# Configuration
MIN_VALID_SAMPLES = 50  # Minimum valid time points per pixel
CLOUD_FRACTION = 0.2    # Fraction of data masked as clouds


def generate_synthetic_satellite_data(
    n_time: int = 100,
    n_lat: int = 50,
    n_lon: int = 50,
    seed: int = 42
) -> tuple:
    """
    Generate synthetic satellite data mimicking real observations.

    Creates three data products with different error characteristics,
    plus a 'truth' field for validation.

    Parameters
    ----------
    n_time : int
        Number of time steps
    n_lat : int
        Number of latitude points
    n_lon : int
        Number of longitude points
    seed : int
        Random seed for reproducibility

    Returns
    -------
    ds1, ds2, ds3 : xr.DataArray
        Three synthetic satellite products
    truth : xr.DataArray
        True underlying signal (for validation)
    true_error_var : dict
        True error variances for each product
    """
    np.random.seed(seed)

    # Create coordinate arrays
    time = np.arange(n_time)
    lat = np.linspace(-25, 25, n_lat)  # Latitude range
    lon = np.linspace(100, 150, n_lon)  # Longitude range

    # Create a realistic "truth" field with:
    # - Spatial pattern (gradient + local features)
    # - Temporal variation (seasonal + random)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')

    # Base spatial pattern
    spatial_pattern = (
        0.5 * np.sin(2 * np.pi * lat_grid / 50) +  # Latitudinal gradient
        0.3 * np.cos(2 * np.pi * lon_grid / 50) +  # Longitudinal gradient
        0.2 * np.random.randn(n_lat, n_lon)         # Local features
    )

    # Temporal variation
    temporal_pattern = (
        np.sin(2 * np.pi * time / 20) +  # "Seasonal" cycle
        0.3 * np.random.randn(n_time)     # Random weather
    )

    # Combine into 3D truth field
    truth_data = np.zeros((n_time, n_lat, n_lon))
    for t in range(n_time):
        truth_data[t] = 10 + spatial_pattern + temporal_pattern[t]

    # Create truth DataArray
    truth = xr.DataArray(
        truth_data,
        dims=['time', 'lat', 'lon'],
        coords={'time': time, 'lat': lat, 'lon': lon},
        attrs={'long_name': 'True Signal', 'units': 'arbitrary'}
    )

    # Define error characteristics for each product
    error_std = {
        'product_1': 0.5,   # Low noise (best quality)
        'product_2': 1.0,   # Medium noise
        'product_3': 2.0,   # High noise (worst quality)
    }

    true_error_var = {k: v**2 for k, v in error_std.items()}

    # Generate observations = truth + noise
    ds1 = truth + error_std['product_1'] * xr.DataArray(
        np.random.randn(n_time, n_lat, n_lon),
        dims=['time', 'lat', 'lon'],
        coords={'time': time, 'lat': lat, 'lon': lon}
    )
    ds1.name = 'product_1'
    ds1.attrs = {'long_name': 'Satellite Product 1 (Low Noise)', 'units': 'arbitrary'}

    ds2 = truth + error_std['product_2'] * xr.DataArray(
        np.random.randn(n_time, n_lat, n_lon),
        dims=['time', 'lat', 'lon'],
        coords={'time': time, 'lat': lat, 'lon': lon}
    )
    ds2.name = 'product_2'
    ds2.attrs = {'long_name': 'Satellite Product 2 (Medium Noise)', 'units': 'arbitrary'}

    ds3 = truth + error_std['product_3'] * xr.DataArray(
        np.random.randn(n_time, n_lat, n_lon),
        dims=['time', 'lat', 'lon'],
        coords={'time': time, 'lat': lat, 'lon': lon}
    )
    ds3.name = 'product_3'
    ds3.attrs = {'long_name': 'Satellite Product 3 (High Noise)', 'units': 'arbitrary'}

    return ds1, ds2, ds3, truth, true_error_var


def add_cloud_mask(ds: xr.DataArray, cloud_band_lat: tuple = (-10, 10),
                   cloud_fraction: float = 0.2, seed: int = None) -> xr.DataArray:
    """
    Add NaN values to simulate cloud masking.

    Clouds preferentially affect a latitude band (e.g., tropics)
    and randomly mask a fraction of data points.

    Parameters
    ----------
    ds : xr.DataArray
        Input data array
    cloud_band_lat : tuple
        Latitude range with enhanced cloud cover
    cloud_fraction : float
        Fraction of data to mask in the cloud band

    Returns
    -------
    xr.DataArray
        Data with NaN values added
    """
    if seed is not None:
        np.random.seed(seed)

    masked = ds.copy()
    data = masked.values.copy()  # Work with numpy array

    lat = ds.coords['lat'].values

    for i, lat_val in enumerate(lat):
        # Higher cloud fraction in tropical band
        if cloud_band_lat[0] <= lat_val <= cloud_band_lat[1]:
            frac = cloud_fraction * 1.5  # More clouds in tropics
        else:
            frac = cloud_fraction * 0.5  # Fewer clouds outside

        # Random mask for this latitude
        n_mask = int(data.shape[0] * data.shape[2] * frac)
        mask_idx = np.random.choice(
            data.shape[0] * data.shape[2],
            size=n_mask,
            replace=False
        )
        t_idx, lon_idx = np.unravel_index(mask_idx, (data.shape[0], data.shape[2]))
        data[t_idx, i, lon_idx] = np.nan

    masked.values = data
    return masked


def apply_tc_per_pixel(ds1: xr.DataArray, ds2: xr.DataArray, ds3: xr.DataArray,
                       min_samples: int = 50) -> tuple:
    """
    Apply Triple Collocation to each pixel's time series.

    Handles NaN values by only using valid (non-NaN) time points.
    Pixels with fewer than min_samples valid points are skipped.

    Parameters
    ----------
    ds1, ds2, ds3 : xr.DataArray
        Three input data products, each (time, lat, lon)
    min_samples : int
        Minimum number of valid time points required

    Returns
    -------
    error_var : xr.Dataset
        Error variance maps for each product
    snr : xr.Dataset
        Signal-to-noise ratio maps
    weights : xr.Dataset
        Optimal fusion weights (inverse variance weighting)
    n_valid : xr.DataArray
        Number of valid samples per pixel
    """
    n_lat = len(ds1.coords['lat'])
    n_lon = len(ds1.coords['lon'])

    # Initialize output arrays
    error_var_1 = np.full((n_lat, n_lon), np.nan)
    error_var_2 = np.full((n_lat, n_lon), np.nan)
    error_var_3 = np.full((n_lat, n_lon), np.nan)
    snr_1 = np.full((n_lat, n_lon), np.nan)
    snr_2 = np.full((n_lat, n_lon), np.nan)
    snr_3 = np.full((n_lat, n_lon), np.nan)
    weight_1 = np.full((n_lat, n_lon), np.nan)
    weight_2 = np.full((n_lat, n_lon), np.nan)
    weight_3 = np.full((n_lat, n_lon), np.nan)
    n_valid_arr = np.zeros((n_lat, n_lon), dtype=int)

    # Get numpy arrays
    data1 = ds1.values
    data2 = ds2.values
    data3 = ds3.values

    # Process each pixel
    print(f"Processing {n_lat * n_lon} pixels...")
    n_skipped = 0

    for i in range(n_lat):
        for j in range(n_lon):
            # Extract time series for this pixel
            ts1 = data1[:, i, j]
            ts2 = data2[:, i, j]
            ts3 = data3[:, i, j]

            # Find valid (non-NaN) time points across all three products
            valid_mask = ~(np.isnan(ts1) | np.isnan(ts2) | np.isnan(ts3))
            n_valid = np.sum(valid_mask)
            n_valid_arr[i, j] = n_valid

            # Skip if insufficient valid samples
            if n_valid < min_samples:
                n_skipped += 1
                continue

            # Extract valid data
            ts1_valid = ts1[valid_mask]
            ts2_valid = ts2[valid_mask]
            ts3_valid = ts3[valid_mask]

            # Stack into TC input format
            tri_data = np.column_stack([ts1_valid, ts2_valid, ts3_valid])

            # Apply Triple Collocation
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    EeeT, SNR, rho2, fMSE = tc(tri_data)

                # Extract error variances (diagonal of EeeT)
                error_var_1[i, j] = EeeT[0, 0]
                error_var_2[i, j] = EeeT[1, 1]
                error_var_3[i, j] = EeeT[2, 2]

                # Store SNR
                snr_1[i, j] = SNR[0]
                snr_2[i, j] = SNR[1]
                snr_3[i, j] = SNR[2]

                # Calculate optimal weights (inverse variance weighting)
                # w_i = (1/σ²_i) / Σ(1/σ²_k)
                var_array = np.array([EeeT[0, 0], EeeT[1, 1], EeeT[2, 2]])

                # Handle negative or zero variances
                var_array = np.maximum(var_array, 1e-10)

                inv_var = 1.0 / var_array
                weights = inv_var / inv_var.sum()

                weight_1[i, j] = weights[0]
                weight_2[i, j] = weights[1]
                weight_3[i, j] = weights[2]

            except Exception as e:
                # Skip problematic pixels
                n_skipped += 1
                continue

    print(f"Completed. Skipped {n_skipped}/{n_lat * n_lon} pixels "
          f"(insufficient data or errors)")

    # Create output DataArrays
    coords = {'lat': ds1.coords['lat'], 'lon': ds1.coords['lon']}

    error_var = xr.Dataset({
        'error_var_1': xr.DataArray(error_var_1, dims=['lat', 'lon'], coords=coords,
                                    attrs={'long_name': 'Error Variance Product 1'}),
        'error_var_2': xr.DataArray(error_var_2, dims=['lat', 'lon'], coords=coords,
                                    attrs={'long_name': 'Error Variance Product 2'}),
        'error_var_3': xr.DataArray(error_var_3, dims=['lat', 'lon'], coords=coords,
                                    attrs={'long_name': 'Error Variance Product 3'}),
    })

    snr = xr.Dataset({
        'snr_1': xr.DataArray(snr_1, dims=['lat', 'lon'], coords=coords,
                              attrs={'long_name': 'SNR Product 1'}),
        'snr_2': xr.DataArray(snr_2, dims=['lat', 'lon'], coords=coords,
                              attrs={'long_name': 'SNR Product 2'}),
        'snr_3': xr.DataArray(snr_3, dims=['lat', 'lon'], coords=coords,
                              attrs={'long_name': 'SNR Product 3'}),
    })

    weights = xr.Dataset({
        'weight_1': xr.DataArray(weight_1, dims=['lat', 'lon'], coords=coords,
                                 attrs={'long_name': 'Fusion Weight Product 1'}),
        'weight_2': xr.DataArray(weight_2, dims=['lat', 'lon'], coords=coords,
                                 attrs={'long_name': 'Fusion Weight Product 2'}),
        'weight_3': xr.DataArray(weight_3, dims=['lat', 'lon'], coords=coords,
                                 attrs={'long_name': 'Fusion Weight Product 3'}),
    })

    n_valid_da = xr.DataArray(
        n_valid_arr, dims=['lat', 'lon'], coords=coords,
        attrs={'long_name': 'Number of Valid Samples'}
    )

    return error_var, snr, weights, n_valid_da


def create_fused_product(ds1: xr.DataArray, ds2: xr.DataArray, ds3: xr.DataArray,
                         weights: xr.Dataset) -> xr.DataArray:
    """
    Create an optimally fused product using TC-derived weights.

    Parameters
    ----------
    ds1, ds2, ds3 : xr.DataArray
        Input data products
    weights : xr.Dataset
        Fusion weights for each product

    Returns
    -------
    xr.DataArray
        Fused product
    """
    # Broadcast weights to time dimension
    w1 = weights['weight_1']
    w2 = weights['weight_2']
    w3 = weights['weight_3']

    # Weighted sum (handling NaN by using nansum-like logic)
    # Where weights are NaN (skipped pixels), use simple average
    fused = (ds1 * w1 + ds2 * w2 + ds3 * w3)

    # For pixels without valid weights, use simple average
    nan_weight_mask = np.isnan(w1.values)
    if nan_weight_mask.any():
        simple_avg = (ds1 + ds2 + ds3) / 3
        # Use xarray where for clean handling - broadcast mask to full data shape
        # Create mask DataArray with proper broadcasting
        mask_da = xr.DataArray(
            nan_weight_mask,
            dims=['lat', 'lon'],
            coords={'lat': ds1.coords['lat'], 'lon': ds1.coords['lon']}
        )
        fused = xr.where(mask_da, simple_avg, fused)

    # Ensure dimension order matches input
    fused = fused.transpose('time', 'lat', 'lon')

    fused.name = 'fused_product'
    fused.attrs = {
        'long_name': 'Optimally Fused Product (TC weights)',
        'units': 'arbitrary',
        'fusion_method': 'inverse_variance_weighting'
    }

    return fused


def evaluate_fusion(fused: xr.DataArray, truth: xr.DataArray,
                    ds1: xr.DataArray, ds2: xr.DataArray, ds3: xr.DataArray) -> dict:
    """
    Evaluate fusion performance against truth.

    Parameters
    ----------
    fused : xr.DataArray
        Fused product
    truth : xr.DataArray
        True signal
    ds1, ds2, ds3 : xr.DataArray
        Individual products

    Returns
    -------
    dict
        Performance metrics (RMSE for each product and fused)
    """
    def rmse(a, b):
        """Calculate RMSE ignoring NaN using xarray alignment."""
        # Use xarray's built-in alignment to handle any dimension ordering
        diff = (a - b)
        return float(np.sqrt((diff ** 2).mean(skipna=True)))

    metrics = {
        'rmse_product_1': rmse(ds1, truth),
        'rmse_product_2': rmse(ds2, truth),
        'rmse_product_3': rmse(ds3, truth),
        'rmse_fused': rmse(fused, truth),
    }

    return metrics


def main():
    """Main execution function."""
    print("=" * 70)
    print("Complex 3D Spatial Data Fusion Example")
    print("=" * 70)

    # Step 1: Generate synthetic satellite data
    print("\n📡 Step 1: Generating synthetic satellite data...")
    print("   Dimensions: Time=100, Lat=50, Lon=50")

    ds1, ds2, ds3, truth, true_error_var = generate_synthetic_satellite_data(
        n_time=100, n_lat=50, n_lon=50, seed=42
    )

    print(f"   True error variances: {true_error_var}")

    # Step 2: Add cloud masking (NaN values)
    print("\n☁️  Step 2: Adding cloud mask (NaN values)...")
    print(f"   Cloud fraction: ~{CLOUD_FRACTION*100:.0f}% (enhanced in tropical band)")

    ds1_masked = add_cloud_mask(ds1, cloud_fraction=CLOUD_FRACTION, seed=100)
    ds2_masked = add_cloud_mask(ds2, cloud_fraction=CLOUD_FRACTION, seed=200)
    ds3_masked = add_cloud_mask(ds3, cloud_fraction=CLOUD_FRACTION, seed=300)

    # Count NaN percentage
    nan_pct_1 = np.isnan(ds1_masked.values).mean() * 100
    nan_pct_2 = np.isnan(ds2_masked.values).mean() * 100
    nan_pct_3 = np.isnan(ds3_masked.values).mean() * 100
    print(f"   NaN percentages: P1={nan_pct_1:.1f}%, P2={nan_pct_2:.1f}%, P3={nan_pct_3:.1f}%")

    # Step 3: Apply Triple Collocation per pixel
    print("\n🔬 Step 3: Applying Triple Collocation per pixel...")
    print(f"   Minimum valid samples required: {MIN_VALID_SAMPLES}")

    error_var, snr, weights, n_valid = apply_tc_per_pixel(
        ds1_masked, ds2_masked, ds3_masked,
        min_samples=MIN_VALID_SAMPLES
    )

    # Report estimated vs true error variances
    print("\n📊 Estimated Error Variances (spatial mean):")
    for name, true_val in true_error_var.items():
        key = f"error_var_{name.split('_')[1]}"
        est_mean = float(error_var[key].mean(skipna=True))
        print(f"   {name}: True={true_val:.2f}, Estimated={est_mean:.2f}")

    # Step 4: Create fused product
    print("\n🔀 Step 4: Creating optimally fused product...")
    fused = create_fused_product(ds1_masked, ds2_masked, ds3_masked, weights)

    # Step 5: Evaluate performance
    print("\n📈 Step 5: Evaluating fusion performance...")
    metrics = evaluate_fusion(fused, truth, ds1_masked, ds2_masked, ds3_masked)

    print("\n   RMSE Results:")
    print(f"   - Product 1 (low noise):  {metrics['rmse_product_1']:.4f}")
    print(f"   - Product 2 (med noise):  {metrics['rmse_product_2']:.4f}")
    print(f"   - Product 3 (high noise): {metrics['rmse_product_3']:.4f}")
    print(f"   - Fused product:          {metrics['rmse_fused']:.4f} ✨")

    improvement = (1 - metrics['rmse_fused'] / metrics['rmse_product_1']) * 100
    print(f"\n   Improvement over best single product: {improvement:.1f}%")

    # Step 6: Save results to NetCDF
    print("\n💾 Step 6: Saving results to NetCDF...")

    # Combine all results into a single Dataset
    results = xr.Dataset({
        # Original products (one time slice for storage efficiency)
        'product_1': ds1_masked.isel(time=0),
        'product_2': ds2_masked.isel(time=0),
        'product_3': ds3_masked.isel(time=0),
        'truth': truth.isel(time=0),

        # Error variance maps
        'error_var_1': error_var['error_var_1'],
        'error_var_2': error_var['error_var_2'],
        'error_var_3': error_var['error_var_3'],

        # Fusion weights
        'weight_1': weights['weight_1'],
        'weight_2': weights['weight_2'],
        'weight_3': weights['weight_3'],

        # Fused product (one time slice)
        'fused': fused.isel(time=0),

        # Diagnostics
        'n_valid_samples': n_valid,
    })

    # Add global attributes
    results.attrs = {
        'title': 'Triple Collocation Spatial Fusion Results',
        'description': 'Error estimation and optimal fusion of three satellite products',
        'min_valid_samples': MIN_VALID_SAMPLES,
        'cloud_fraction': CLOUD_FRACTION,
        'rmse_product_1': metrics['rmse_product_1'],
        'rmse_product_2': metrics['rmse_product_2'],
        'rmse_product_3': metrics['rmse_product_3'],
        'rmse_fused': metrics['rmse_fused'],
    }

    # Save to current directory
    output_path = os.path.join(os.path.dirname(__file__), 'spatial_results.nc')
    results.to_netcdf(output_path)
    print(f"   Saved to: {output_path}")

    # Step 7: Create visualization (if matplotlib available)
    print("\n📊 Step 7: Creating visualization...")
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Error variance maps
        im1 = axes[0, 0].pcolormesh(
            results.lon, results.lat, results.error_var_1,
            cmap='viridis', shading='auto'
        )
        axes[0, 0].set_title('Error Variance: Product 1\n(True=0.25)')
        plt.colorbar(im1, ax=axes[0, 0])

        im2 = axes[0, 1].pcolormesh(
            results.lon, results.lat, results.error_var_2,
            cmap='viridis', shading='auto'
        )
        axes[0, 1].set_title('Error Variance: Product 2\n(True=1.00)')
        plt.colorbar(im2, ax=axes[0, 1])

        im3 = axes[0, 2].pcolormesh(
            results.lon, results.lat, results.error_var_3,
            cmap='viridis', shading='auto'
        )
        axes[0, 2].set_title('Error Variance: Product 3\n(True=4.00)')
        plt.colorbar(im3, ax=axes[0, 2])

        # Weight maps
        im4 = axes[1, 0].pcolormesh(
            results.lon, results.lat, results.weight_1,
            cmap='RdYlGn', vmin=0, vmax=1, shading='auto'
        )
        axes[1, 0].set_title('Fusion Weight: Product 1')
        plt.colorbar(im4, ax=axes[1, 0])

        im5 = axes[1, 1].pcolormesh(
            results.lon, results.lat, results.weight_2,
            cmap='RdYlGn', vmin=0, vmax=1, shading='auto'
        )
        axes[1, 1].set_title('Fusion Weight: Product 2')
        plt.colorbar(im5, ax=axes[1, 1])

        im6 = axes[1, 2].pcolormesh(
            results.lon, results.lat, results.weight_3,
            cmap='RdYlGn', vmin=0, vmax=1, shading='auto'
        )
        axes[1, 2].set_title('Fusion Weight: Product 3')
        plt.colorbar(im6, ax=axes[1, 2])

        for ax in axes.flat:
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')

        plt.suptitle('Triple Collocation: Error Estimation and Optimal Fusion Weights',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(os.path.dirname(__file__), 'figures',
                                'spatial_fusion_results.png')
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"   Figure saved to: {fig_path}")

        plt.close()

    except ImportError:
        print("   matplotlib not available, skipping visualization")

    print("\n" + "=" * 70)
    print("✅ Complex Spatial Fusion Example Complete!")
    print("=" * 70)

    return results, metrics


if __name__ == "__main__":
    results, metrics = main()
