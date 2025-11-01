"""
Utility functions for data processing and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import warnings


def generate_synthetic_data(n_samples: int = 1000, 
                           correlation: float = 0.7,
                           noise_level: float = 0.3,
                           seed: Optional[int] = None) -> Tuple[np.ndarray, ...]:
    """
    Generate synthetic precipitation data for testing.
    
    Creates three correlated products and a reference truth.
    
    Args:
        n_samples: Number of time steps
        correlation: Target correlation with truth
        noise_level: Amount of noise to add
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (truth, x, y, z) arrays
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate true precipitation (lognormal distribution)
    truth = np.random.lognormal(mean=1.0, sigma=0.8, size=n_samples)
    truth = np.maximum(truth, 0)  # Ensure non-negative
    
    # Generate correlated products with different error characteristics
    x = truth + np.random.normal(0, noise_level * np.mean(truth), n_samples)
    y = truth * 1.1 + np.random.normal(0, noise_level * 1.2 * np.mean(truth), n_samples)
    z = truth * 0.95 + np.random.normal(0, noise_level * 0.9 * np.mean(truth), n_samples)
    
    # Add some systematic errors
    x = x * (1 + 0.1 * np.sin(np.linspace(0, 4*np.pi, n_samples)))  # Seasonal bias
    y = y + 0.5  # Additive bias
    z = z * 1.05  # Multiplicative bias
    
    # Ensure non-negative
    x = np.maximum(x, 0)
    y = np.maximum(y, 0)
    z = np.maximum(z, 0)
    
    return truth, x, y, z


def load_precipitation_data(filepath: str, 
                           variable_name: str = 'precipitation') -> np.ndarray:
    """
    Load precipitation data from NetCDF file.
    
    Args:
        filepath: Path to NetCDF file
        variable_name: Name of precipitation variable
        
    Returns:
        Precipitation array
    """
    try:
        import netCDF4 as nc
    except ImportError:
        raise ImportError("netCDF4 required. Install with: pip install netCDF4")
    
    with nc.Dataset(filepath, 'r') as dataset:
        precip = dataset[variable_name][:]
        precip = np.array(precip)
    
    return precip


def save_to_netcdf(data: np.ndarray, 
                   filepath: str,
                   variable_name: str = 'precipitation',
                   lat: Optional[np.ndarray] = None,
                   lon: Optional[np.ndarray] = None,
                   time: Optional[np.ndarray] = None,
                   attributes: Optional[Dict] = None):
    """
    Save precipitation data to NetCDF file.
    
    Args:
        data: Precipitation array
        filepath: Output file path
        variable_name: Name for precipitation variable
        lat: Latitude array (optional)
        lon: Longitude array (optional)
        time: Time array (optional)
        attributes: Dictionary of attributes to add
    """
    try:
        import netCDF4 as nc
    except ImportError:
        raise ImportError("netCDF4 required. Install with: pip install netCDF4")
    
    with nc.Dataset(filepath, 'w', format='NETCDF4') as dataset:
        # Create dimensions
        if data.ndim == 3:
            nlat, nlon, ntime = data.shape
            dataset.createDimension('lat', nlat)
            dataset.createDimension('lon', nlon)
            dataset.createDimension('time', ntime)
            
            # Create variables
            if lat is not None:
                lat_var = dataset.createVariable('lat', 'f4', ('lat',))
                lat_var[:] = lat
                lat_var.units = 'degrees_north'
            
            if lon is not None:
                lon_var = dataset.createVariable('lon', 'f4', ('lon',))
                lon_var[:] = lon
                lon_var.units = 'degrees_east'
            
            if time is not None:
                time_var = dataset.createVariable('time', 'f8', ('time',))
                time_var[:] = time
                time_var.units = 'days since 1970-01-01'
            
            # Create precipitation variable
            precip_var = dataset.createVariable(variable_name, 'f4', 
                                               ('lat', 'lon', 'time'))
            precip_var[:] = data
            precip_var.units = 'mm/day'
            
        elif data.ndim == 1:
            ntime = len(data)
            dataset.createDimension('time', ntime)
            
            if time is not None:
                time_var = dataset.createVariable('time', 'f8', ('time',))
                time_var[:] = time
                time_var.units = 'days since 1970-01-01'
            
            precip_var = dataset.createVariable(variable_name, 'f4', ('time',))
            precip_var[:] = data
            precip_var.units = 'mm/day'
        
        # Add attributes
        if attributes is not None:
            for key, value in attributes.items():
                setattr(precip_var, key, value)
        
        # Global attributes
        dataset.title = 'ETCC Merged Precipitation'
        dataset.institution = 'Generated by ETCC Python package'
        dataset.source = 'Wei et al. (2023) GRL'


def plot_comparison(products: Dict[str, np.ndarray],
                   reference: np.ndarray,
                   time_index: Optional[np.ndarray] = None,
                   figsize: Tuple[int, int] = (12, 6),
                   save_path: Optional[str] = None):
    """
    Plot time series comparison of multiple products.
    
    Args:
        products: Dictionary of products {'name': array}
        reference: Reference precipitation
        time_index: Time index for x-axis (optional)
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=figsize)
    
    if time_index is None:
        time_index = np.arange(len(reference))
    
    # Plot reference
    plt.plot(time_index, reference, 'k-', linewidth=2, 
            label='Reference', alpha=0.7)
    
    # Plot products
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, (name, product) in enumerate(products.items()):
        plt.plot(time_index, product, color=colors[i % len(colors)],
                linewidth=1.5, label=name, alpha=0.7)
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Precipitation (mm/day)', fontsize=12)
    plt.title('Precipitation Product Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='best', frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_spatial_metric(metric_map: np.ndarray,
                       lat: np.ndarray,
                       lon: np.ndarray,
                       metric_name: str = 'CC',
                       vmin: Optional[float] = None,
                       vmax: Optional[float] = None,
                       cmap: str = 'RdYlBu_r',
                       figsize: Tuple[int, int] = (12, 6),
                       save_path: Optional[str] = None):
    """
    Plot spatial distribution of a metric.
    
    Args:
        metric_map: 2D array of metric values (lat, lon)
        lat: Latitude array
        lon: Longitude array
        metric_name: Name of metric for title
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
        cmap: Colormap name
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'rectilinear'})
    
    # Create meshgrid
    LON, LAT = np.meshgrid(lon, lat)
    
    # Plot
    im = ax.pcolormesh(LON, LAT, metric_map, cmap=cmap, 
                       vmin=vmin, vmax=vmax, shading='auto')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                       pad=0.05, aspect=40)
    cbar.set_label(metric_name, fontsize=12)
    
    # Labels
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Spatial Distribution of {metric_name}', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_scatter(predicted: np.ndarray,
                observed: np.ndarray,
                product_name: str = 'Merged',
                figsize: Tuple[int, int] = (8, 8),
                save_path: Optional[str] = None):
    """
    Plot scatter plot of predicted vs observed.
    
    Args:
        predicted: Predicted precipitation
        observed: Observed precipitation
        product_name: Name for title
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    from .evaluation import calculate_cc, calculate_rmse
    
    # Remove NaN values
    mask = ~(np.isnan(predicted) | np.isnan(observed))
    pred = predicted[mask]
    obs = observed[mask]
    
    # Calculate metrics
    cc = calculate_cc(pred, obs)
    rmse = calculate_rmse(pred, obs)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(obs, pred, alpha=0.5, s=10, c='#1f77b4', edgecolors='none')
    
    # 1:1 line
    max_val = max(np.max(obs), np.max(pred))
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='1:1 Line')
    
    # Best fit line
    z = np.polyfit(obs, pred, 1)
    p = np.poly1d(z)
    ax.plot(obs, p(obs), 'r-', linewidth=2, 
           label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
    
    # Labels and metrics
    ax.set_xlabel('Observed Precipitation (mm/day)', fontsize=12)
    ax.set_ylabel('Predicted Precipitation (mm/day)', fontsize=12)
    ax.set_title(f'{product_name}\nCC={cc:.3f}, RMSE={rmse:.3f} mm/day',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_weights_comparison(tc_weights: Dict[str, float],
                           etcc_weights: Dict[str, float],
                           product_names: List[str] = ['IMERG-E', 'SM2RAIN', 'ERA5'],
                           figsize: Tuple[int, int] = (10, 6),
                           save_path: Optional[str] = None):
    """
    Compare weights from TC and ETCC methods.
    
    Args:
        tc_weights: TC weights dictionary
        etcc_weights: ETCC weights dictionary
        product_names: Names of three products
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(product_names))
    width = 0.35
    
    tc_vals = [tc_weights['wx'], tc_weights['wy'], tc_weights['wz']]
    etcc_vals = [etcc_weights['wx'], etcc_weights['wy'], etcc_weights['wz']]
    
    bars1 = ax.bar(x - width/2, tc_vals, width, label='TC', 
                   color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, etcc_vals, width, label='ETCC',
                   color='#ff7f0e', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Product', fontsize=12)
    ax.set_ylabel('Weight', fontsize=12)
    ax.set_title('Comparison of Merging Weights: TC vs ETCC',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(product_names)
    ax.legend(frameon=True, fontsize=11)
    ax.set_ylim(0, max(max(tc_vals), max(etcc_vals)) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_boxplot_comparison(results: Dict[str, Dict[str, float]],
                              metric: str = 'cc',
                              figsize: Tuple[int, int] = (10, 6),
                              save_path: Optional[str] = None):
    """
    Create boxplot comparison similar to Figure 1 in the paper.
    
    Args:
        results: Results from spatial evaluation
        metric: Metric to plot ('cc' or 'rmse')
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    products = list(results.keys())
    values = [results[p][metric] for p in products]
    
    bp = ax.boxplot(values, labels=products, patch_artist=True,
                   widths=0.6, showmeans=True)
    
    # Color boxes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f'{metric.upper()} Distribution Across Grid Cells',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
