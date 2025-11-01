"""
Spatial precipitation merging example.

Demonstrates ETCC application to gridded spatial data,
similar to the global-scale analysis in Wei et al. (2023).
"""

import numpy as np
import matplotlib.pyplot as plt
from etcc import (SpatialMerging, spatial_metrics, plot_spatial_metric,
                  generate_synthetic_data, compare_products)


def generate_spatial_data(nlat=20, nlon=30, ntime=365, seed=42):
    """
    Generate synthetic spatial precipitation data.
    
    Creates 3D arrays (lat, lon, time) with spatial patterns.
    """
    np.random.seed(seed)
    
    # Create spatial patterns
    lat = np.linspace(-30, 30, nlat)
    lon = np.linspace(-60, 60, nlon)
    
    # Initialize arrays
    shape = (nlat, nlon, ntime)
    truth = np.zeros(shape)
    x = np.zeros(shape)
    y = np.zeros(shape)
    z = np.zeros(shape)
    
    # Generate data for each grid cell
    for i in range(nlat):
        for j in range(nlon):
            # Add spatial variation (more precip near equator)
            spatial_factor = 1.0 + 0.5 * np.cos(lat[i] * np.pi / 30)
            
            # Generate time series for this grid cell
            t, tx, ty, tz = generate_synthetic_data(
                n_samples=ntime,
                correlation=0.6 + 0.2 * np.random.rand(),
                noise_level=0.2 + 0.15 * np.random.rand(),
                seed=seed + i * nlon + j
            )
            
            truth[i, j, :] = t * spatial_factor
            x[i, j, :] = tx * spatial_factor
            y[i, j, :] = ty * spatial_factor
            z[i, j, :] = tz * spatial_factor
    
    return lat, lon, truth, x, y, z


def main():
    print("=" * 70)
    print("ETCC Spatial Precipitation Merging Example")
    print("=" * 70 + "\n")
    
    # 1. Generate spatial data
    print("1. Generating spatial precipitation data...")
    nlat, nlon, ntime = 20, 30, 365
    lat, lon, truth, x, y, z = generate_spatial_data(nlat, nlon, ntime)
    print(f"   Grid size: {nlat} × {nlon}")
    print(f"   Time steps: {ntime}")
    print(f"   Latitude range: [{lat[0]:.1f}°, {lat[-1]:.1f}°]")
    print(f"   Longitude range: [{lon[0]:.1f}°, {lon[-1]:.1f}°]\n")
    
    # 2. Apply TC merging
    print("2. Applying TC merging to all grid cells...")
    tc_merger = SpatialMerging(method='tc')
    merged_tc = tc_merger.merge_gridded(x, y, z, axis=-1)
    print("   TC merging complete\n")
    
    # 3. Apply ETCC merging
    print("3. Applying ETCC merging to all grid cells...")
    etcc_merger = SpatialMerging(method='etcc', weight_increment=0.05)
    merged_etcc = etcc_merger.merge_gridded(x, y, z, axis=-1)
    print("   ETCC merging complete\n")
    
    # 4. Calculate spatial metrics
    print("4. Calculating spatial distribution of metrics...")
    
    # Calculate CC for each grid cell
    cc_x = spatial_metrics(x, truth, metric='cc', axis=-1)
    cc_y = spatial_metrics(y, truth, metric='cc', axis=-1)
    cc_z = spatial_metrics(z, truth, metric='cc', axis=-1)
    cc_tc = spatial_metrics(merged_tc, truth, metric='cc', axis=-1)
    cc_etcc = spatial_metrics(merged_etcc, truth, metric='cc', axis=-1)
    
    # Calculate RMSE for each grid cell
    rmse_x = spatial_metrics(x, truth, metric='rmse', axis=-1)
    rmse_y = spatial_metrics(y, truth, metric='rmse', axis=-1)
    rmse_z = spatial_metrics(z, truth, metric='rmse', axis=-1)
    rmse_tc = spatial_metrics(merged_tc, truth, metric='rmse', axis=-1)
    rmse_etcc = spatial_metrics(merged_etcc, truth, metric='rmse', axis=-1)
    
    # Print summary statistics
    print(f"\n   CC Statistics:")
    print(f"   Product X:    median={np.nanmedian(cc_x):.3f}, mean={np.nanmean(cc_x):.3f}")
    print(f"   Product Y:    median={np.nanmedian(cc_y):.3f}, mean={np.nanmean(cc_y):.3f}")
    print(f"   Product Z:    median={np.nanmedian(cc_z):.3f}, mean={np.nanmean(cc_z):.3f}")
    print(f"   TC Merged:    median={np.nanmedian(cc_tc):.3f}, mean={np.nanmean(cc_tc):.3f}")
    print(f"   ETCC Merged:  median={np.nanmedian(cc_etcc):.3f}, mean={np.nanmean(cc_etcc):.3f}")
    
    print(f"\n   RMSE Statistics (mm/day):")
    print(f"   Product X:    median={np.nanmedian(rmse_x):.3f}, mean={np.nanmean(rmse_x):.3f}")
    print(f"   Product Y:    median={np.nanmedian(rmse_y):.3f}, mean={np.nanmean(rmse_y):.3f}")
    print(f"   Product Z:    median={np.nanmedian(rmse_z):.3f}, mean={np.nanmean(rmse_z):.3f}")
    print(f"   TC Merged:    median={np.nanmedian(rmse_tc):.3f}, mean={np.nanmean(rmse_tc):.3f}")
    print(f"   ETCC Merged:  median={np.nanmedian(rmse_etcc):.3f}, mean={np.nanmean(rmse_etcc):.3f}\n")
    
    # 5. Visualizations
    print("5. Creating spatial visualizations...\n")
    
    # Create figure with multiple subplots (similar to Figure 1 in paper)
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    
    # CC maps
    products_cc = [
        ('Product X', cc_x),
        ('Product Y', cc_y),
        ('Product Z', cc_z),
        ('TC Merged', cc_tc),
        ('ETCC Merged', cc_etcc),
        ('ETCC - TC', cc_etcc - cc_tc)
    ]
    
    for idx, (name, data) in enumerate(products_cc):
        ax = axes[idx // 2, idx % 2]
        
        if idx == 5:  # Difference map
            im = ax.pcolormesh(lon, lat, data, cmap='RdBu_r', 
                             vmin=-0.1, vmax=0.1, shading='auto')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('CC Difference', fontsize=10)
        else:
            im = ax.pcolormesh(lon, lat, data, cmap='YlOrRd',
                             vmin=0.4, vmax=0.9, shading='auto')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('CC', fontsize=10)
        
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        ax.set_title(name, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('spatial_cc_maps.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # RMSE maps
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    products_rmse = [
        ('Product X', rmse_x),
        ('Product Y', rmse_y),
        ('Product Z', rmse_z),
        ('TC Merged', rmse_tc),
        ('ETCC Merged', rmse_etcc),
        ('TC - ETCC', rmse_tc - rmse_etcc)
    ]
    
    for idx, (name, data) in enumerate(products_rmse):
        ax = axes[idx]
        
        if idx == 5:  # Difference map
            im = ax.pcolormesh(lon, lat, data, cmap='RdBu',
                             vmin=-0.5, vmax=0.5, shading='auto')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('RMSE Difference (mm/day)', fontsize=10)
        else:
            im = ax.pcolormesh(lon, lat, data, cmap='YlOrRd_r',
                             vmin=0, vmax=3, shading='auto')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('RMSE (mm/day)', fontsize=10)
        
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        ax.set_title(name, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('spatial_rmse_maps.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate percentage of grid cells where ETCC improves over TC
    cc_improved = np.sum(cc_etcc > cc_tc) / (nlat * nlon) * 100
    rmse_improved = np.sum(rmse_etcc < rmse_tc) / (nlat * nlon) * 100
    
    print(f"6. Summary:")
    print(f"   Grid cells where ETCC improves CC over TC: {cc_improved:.1f}%")
    print(f"   Grid cells where ETCC improves RMSE over TC: {rmse_improved:.1f}%")
    print(f"   Average CC improvement: {np.nanmean(cc_etcc - cc_tc):.4f}")
    print(f"   Average RMSE improvement: {np.nanmean(rmse_tc - rmse_etcc):.4f} mm/day\n")
    
    print("=" * 70)
    print("Spatial analysis complete!")
    print("Figures saved: spatial_cc_maps.png, spatial_rmse_maps.png")
    print("=" * 70)


if __name__ == '__main__':
    main()
