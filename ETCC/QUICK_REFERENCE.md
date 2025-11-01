# ETCC Quick Reference Guide

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## 1-Minute Start

```python
from etcc import ETCC
import numpy as np

# Your three precipitation products (1D time series)
x, y, z = ... # Load your data

# Merge with ETCC
etcc = ETCC()
merged = etcc.merge(x, y, z)

# Check weights
print(etcc.weights)  # {'wx': 0.35, 'wy': 0.30, 'wz': 0.35}
```

## Common Use Cases

### Case 1: Basic Merging

```python
from etcc import TripleCollocation, ETCC

# TC (minimizes RMSE)
tc = TripleCollocation()
merged_tc = tc.merge(x, y, z)

# ETCC (maximizes correlation)
etcc = ETCC(weight_increment=0.01)
merged_etcc = etcc.merge(x, y, z)
```

### Case 2: Evaluation

```python
from etcc import calculate_metrics, compare_products

# Single product
metrics = calculate_metrics(merged, reference)
print(f"CC: {metrics['cc']:.3f}, RMSE: {metrics['rmse']:.3f}")

# Compare multiple
products = {'TC': merged_tc, 'ETCC': merged_etcc, 'Original': x}
results = compare_products(products, reference)
print_comparison_table(results)
```

### Case 3: Spatial Data

```python
from etcc import SpatialMerging

# Data shape: (lat, lon, time)
x_grid = np.random.randn(50, 100, 365)
y_grid = np.random.randn(50, 100, 365)
z_grid = np.random.randn(50, 100, 365)

# Merge all grid cells
merger = SpatialMerging(method='etcc', weight_increment=0.05)
merged_grid = merger.merge_gridded(x_grid, y_grid, z_grid, axis=-1)
```

### Case 4: Spatial Metrics

```python
from etcc import spatial_metrics

# Calculate CC at each grid cell
cc_map = spatial_metrics(merged_grid, reference_grid, metric='cc', axis=-1)

# Shape: (lat, lon)
print(f"Mean CC: {np.mean(cc_map):.3f}")
```

### Case 5: Visualization

```python
from etcc import plot_comparison, plot_spatial_metric

# Time series
plot_comparison(
    products={'ETCC': merged, 'Original': x},
    reference=ref,
    save_path='comparison.png'
)

# Spatial map
plot_spatial_metric(
    metric_map=cc_map,
    lat=lat, lon=lon,
    metric_name='Correlation',
    save_path='cc_map.png'
)
```

## Parameter Tuning

### weight_increment
```python
# High precision (slow)
etcc = ETCC(weight_increment=0.01)  # ~5,151 evaluations

# Fast prototyping
etcc = ETCC(weight_increment=0.05)  # ~231 evaluations

# Very fast
etcc = ETCC(weight_increment=0.10)  # ~66 evaluations
```

### min_correlation
```python
# Default (from paper)
etcc = ETCC(min_correlation=0.01)

# Stricter (for high-quality data)
etcc = ETCC(min_correlation=0.05)
```

## Choosing Between TC and ETCC

| Use TC if: | Use ETCC if: |
|------------|--------------|
| Minimizing absolute errors is critical | Temporal correlation matters most |
| Working with very large datasets | Have moderate-sized dataset |
| Need fast computation | Can afford extra computation |
| Bias correction is primary goal | Capturing dynamics is priority |

## Common Pitfalls

### ❌ Don't Do This:
```python
# Using dependent products
x = satellite_a
y = satellite_a + noise  # NOT independent!
z = era5
```

### ✅ Do This Instead:
```python
# Use truly independent sources
x = imerg_early_run  # Satellite MW+IR
y = sm2rain_ascat    # Satellite soil moisture
z = era5             # Reanalysis model
```

### ❌ Don't Do This:
```python
# Too short time series
x = np.random.randn(10)  # Only 10 samples
y = np.random.randn(10)
z = np.random.randn(10)
```

### ✅ Do This Instead:
```python
# Adequate length
x = load_data()  # At least 100+ samples
y = load_data()
z = load_data()
```

## Quick Debugging

### Problem: Weights don't sum to 1
```python
# Check
print(sum(etcc.weights.values()))

# Should be: 1.0 ± 0.01
```

### Problem: Low correlation improvement
```python
# Check input correlations first
print(f"Input CCs: {np.corrcoef(x,ref)[0,1]:.3f}, "
      f"{np.corrcoef(y,ref)[0,1]:.3f}, "
      f"{np.corrcoef(z,ref)[0,1]:.3f}")

# If inputs already have high CC (>0.95), 
# improvement will be small
```

### Problem: Negative error variance
```python
# This happens when TC assumptions violated
# Check: Are products truly independent?
print(f"x-y correlation: {np.corrcoef(x,y)[0,1]:.3f}")
print(f"x-z correlation: {np.corrcoef(x,z)[0,1]:.3f}")
print(f"y-z correlation: {np.corrcoef(y,z)[0,1]:.3f}")

# All should be <0.9
```

## Performance Tips

### Speed up ETCC:
```python
# 1. Use larger increment
etcc = ETCC(weight_increment=0.05)

# 2. For spatial data, consider parallel processing
from joblib import Parallel, delayed

def merge_pixel(i):
    etcc = ETCC(weight_increment=0.05)
    return etcc.merge(x[i], y[i], z[i])

results = Parallel(n_jobs=-1)(delayed(merge_pixel)(i) 
                              for i in range(n_pixels))
```

### Memory optimization:
```python
# Process spatial data in chunks
chunk_size = 1000
for i in range(0, n_pixels, chunk_size):
    chunk = merged[i:i+chunk_size]
    # Process chunk
```

## Export Results

```python
# Save to NumPy
np.save('merged_precipitation.npy', merged)

# Save to NetCDF
from etcc import save_to_netcdf
save_to_netcdf(
    data=merged_grid,
    filepath='merged.nc',
    variable_name='precipitation',
    lat=lat, lon=lon, time=time,
    attributes={'method': 'ETCC', 'source': 'IMERG+SM2RAIN+ERA5'}
)
```

## Get Help

```python
# Check docstrings
help(ETCC)
help(TripleCollocation)
help(calculate_metrics)

# Print weights and stats
print(f"Weights: {etcc.weights}")
print(f"Max correlation: {etcc.max_correlation}")
print(f"Truth correlations: {etcc.correlation_with_truth}")
```

## Example Workflow

```python
# Complete analysis workflow
import numpy as np
from etcc import (ETCC, TripleCollocation, calculate_metrics, 
                  compare_products, plot_comparison)

# 1. Load data
x = np.load('imerg.npy')
y = np.load('sm2rain.npy')
z = np.load('era5.npy')
ref = np.load('reference.npy')

# 2. Merge with both methods
tc = TripleCollocation()
etcc = ETCC(weight_increment=0.01)

merged_tc = tc.merge(x, y, z)
merged_etcc = etcc.merge(x, y, z)

# 3. Evaluate
products = {
    'IMERG': x, 'SM2RAIN': y, 'ERA5': z,
    'TC': merged_tc, 'ETCC': merged_etcc
}
results = compare_products(products, ref)

# 4. Visualize
plot_comparison(products, ref, save_path='results.png')

# 5. Report
print(f"\nBest method: ETCC")
print(f"CC improvement: +{(results['ETCC']['cc'] - max(results['IMERG']['cc'], results['SM2RAIN']['cc'], results['ERA5']['cc']))*100:.1f}%")
```

## Further Reading

- `README.md` - General overview
- `METHODOLOGY.md` - Detailed theory
- `examples/` - Complete examples
- Paper: Wei et al. (2023) GRL, https://doi.org/10.1029/2023GL105120
