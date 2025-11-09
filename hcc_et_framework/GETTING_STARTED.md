# Getting Started with HCC-ET Framework

Welcome! This guide will help you get started with the HCC-ET Framework in **5 minutes**.

---

## Quick Start (3 Steps)

### 1. Create Configuration File

```bash
cd hcc_et_framework
cp config_template.yaml my_config.yaml
```

Edit `my_config.yaml` and update these critical paths:

```yaml
data:
  parent_dir: /your/path/to/parent_products  # GLEAM, PMLv2, GLDAS
  predictors_dir: /your/path/to/predictors_1km  # LST, NDVI, Albedo, Precip
  sapflux_dir: /your/path/to/sapfluxnet
  wb_et_dir: /your/path/to/water_balance
  total_et_dir: /your/path/to/total_et
```

### 2. Run the Framework

**Option A: Python Script**
```python
from hcc_et_framework import HCCETFramework

framework = HCCETFramework(
    config_file='my_config.yaml',
    output_dir='./outputs'
)

framework.run_full_pipeline()
```

**Option B: Command Line**
```bash
python -m hcc_et_framework.hcc_framework \
    --config my_config.yaml \
    --output-dir ./outputs \
    --phase all
```

### 3. Check Results

```bash
ls outputs/
# → t_et_ratio_1950_2025.nc
# → T_final_1km.nc
# → E_final_1km.nc
# → weights.nc
# → trends_by_landcover.csv
```

---

## Data Requirements

### Required Input Datasets

| Dataset | Description | Format | Example Products |
|---------|-------------|--------|------------------|
| **Parent T/ET** | Coarse-resolution T/ET ratios | NetCDF | GLEAM, PMLv2, GLDAS |
| **Predictors (1km)** | High-res predictors | NetCDF | MODIS LST, NDVI, Albedo; GPM Precip |
| **Total ET** | 1km total ET | NetCDF | CAMELE, GLEAM-1km |
| **SapFluxNet** | Site T observations | CSV/NetCDF | SapFluxNet database |
| **Water Balance** | Basin P-Q-dS | NetCDF | GRDC + GRACE |
| **Land Cover** | IGBP classification | NetCDF | MODIS MCD12Q1 |

### Data Preparation Tips

**1. Parent T/ET Ratios**
```python
# Expected structure:
# File: GLEAM_v3.7_T_ET_ratio.nc
# Variable: 'T_ET_ratio' or 'transp_fraction'
# Dims: (time, lat, lon)
# Range: [0, 1]
```

**2. 1km Predictors**
```python
# Expected structure:
# Files: LST_1km_2000_2025.nc, NDVI_1km_2000_2025.nc, ...
# Variables: 'LST', 'NDVI', 'Albedo', 'Precip'
# Dims: (time, lat, lon)
# Resolution: ~1km (0.01° or 0.008333°)
```

**3. SapFluxNet**
```python
# Expected structure (CSV):
# Columns: site_id, latitude, longitude, T_ET_ratio, PFT
# Or NetCDF: dims (lat, lon), values = T/ET where sites exist

# Convert CSV to 1km grid:
from hcc_et_framework import data_loader
sapflux_1km = data_loader.load_sapfluxnet_1km(
    data_dir='/path/to/sapflux',
    upscale_method='nearest'
)
```

---

## Running Individual Phases

Sometimes you want to run only specific phases:

```python
framework = HCCETFramework('my_config.yaml', './outputs')

# Phase 1: Load data
framework.run_phase1_data_preparation()

# Phase 2: Downscale to 1km
framework.run_phase2_downscaling()

# Phase 3: Fuse with EIVD/GLS
framework.run_phase3_fusion()

# Phase 4: Apply physical constraints
framework.run_phase4_physical_constraints()

# Phase 5: Hindcast and trends
framework.run_phase5_temporal_hindcast()
```

Or via command line:
```bash
# Run only Phase 3 (fusion)
python -m hcc_et_framework.hcc_framework --config my_config.yaml --phase 3
```

---

## Customizing Fusion Mode

The framework auto-selects the best fusion method based on the number of parent products:

| # Products | Auto-Selected Method | Manual Override |
|------------|----------------------|-----------------|
| 3 | EIVD | `mode: eivd` |
| 4 | EC | `mode: ec` |
| ≥3 | GLS | `mode: gls` |

**Force a specific method**:
```yaml
fusion:
  mode: eivd  # Force EIVD even if you have 4 products
```

**Key differences**:
- **EIVD**: Analytical solution, fast, handles ECC for 3 products
- **EC**: Tests all 6 combinations, handles ECC for 4 products
- **GLS**: General solution, uses TC to estimate covariance

---

## Troubleshooting

### Common Issues

**1. "No files found for GLEAM"**
```
Warning: No files found for GLEAM in /data/parent_products
```
**Fix**: Check that files match pattern `{product}_*_T_ET_ratio.nc`

**2. "Variable T_ET_ratio not found"**
```
Warning: Variable T_ET_ratio not found in GLEAM
```
**Fix**: Update `data_loader.py` line 50 to match your variable name, or add mapping:
```python
variables = {
    'GLEAM': 'your_variable_name',
    'PMLv2': 't_et_ratio',
}
```

**3. "Singular matrix in EIVD"**
```
Error: numpy.linalg.LinAlgError: Singular matrix
```
**Fix**: Products are too similar. Try:
- Use `mode: gls` instead of `mode: eivd`
- Check if products are truly independent
- Increase temporal resolution

**4. Memory error**
```
MemoryError
```
**Fix**: Reduce spatial domain or use global weights:
```yaml
fusion:
  spatial_varying_weights: false  # Use single weight vector
```

---

## Understanding the Output

### Output Files

**1. `t_et_ratio_1950_2025.nc`**
- Fused T/ET ratio time series (1km, 1950-2025)
- Use this for trend analysis

**2. `T_final_1km.nc` and `E_final_1km.nc`**
- Final transpiration and evaporation (1km)
- T = ET × (T/ET), E = ET × (1 - T/ET)

**3. `weights.nc`**
- Fusion weights for each parent product
- Helps understand which products contribute most

**4. `trends_by_landcover.csv`**
- Summary table of trends per land cover class
- Natural vs Managed ecosystem comparison

**5. `diagnostics.json`**
- Performance metrics (R², RMSE, KGE)
- Fusion diagnostics (effective N, entropy)

### Interpreting Results

**Fusion Weights**:
```python
import xarray as xr
weights = xr.open_dataarray('outputs/weights.nc')
print(weights)
# Output: GLEAM: 0.42, PMLv2: 0.31, GLDAS: 0.27
```
- Higher weight = lower error, higher contribution
- Weights sum to 1 (unbiased fusion)

**Trends**:
```python
import pandas as pd
trends = pd.read_csv('outputs/trends_by_landcover.csv')
print(trends)
#         landcover     slope    p_value
# 0        natural   +0.0052      0.001
# 1        managed   +0.0123      0.000
# 2          urban   -0.0021      0.312
```
- Positive slope = increasing T/ET (more productive)
- Negative slope = decreasing T/ET (less productive)
- p_value < 0.05 = statistically significant

---

## Next Steps

1. **Validate results**: Compare with eddy covariance sites
2. **Sensitivity analysis**: Test different fusion modes, hyperparameters
3. **Scientific interpretation**: Attribute trends to drivers (climate, land management)
4. **Visualization**: Create maps and time series plots

### Example Visualization

```python
import xarray as xr
import matplotlib.pyplot as plt

# Load results
t_et = xr.open_dataarray('outputs/t_et_ratio_1950_2025.nc')

# Compute trend
from hcc_et_framework import utils_hcc
slopes, pvals = utils_hcc.compute_spatial_trends(t_et)

# Plot
decadal_trends = slopes * 10  # Convert to per decade
decadal_trends.plot(cmap='RdBu_r', vmin=-0.05, vmax=0.05)
plt.title('T/ET Trends (1950-2025, per decade)')
plt.savefig('trends_map.png', dpi=300)
```

---

## Getting Help

- **Documentation**: See `README.md` for comprehensive guide
- **Examples**: Run `python example_usage.py` for demonstrations
- **Issues**: If you encounter problems, check:
  1. Data paths in config.yaml
  2. Variable names in NetCDF files
  3. File formats (NetCDF4 preferred)

---

## Citation

If you use this framework, please cite:

```bibtex
@article{yourname2025hcc,
  title={Widespread Reshaping of Productive vs Non-Productive Water Fluxes under Climate Change},
  author={Your Name et al.},
  journal={Nature Climate Change},
  year={2025}
}
```

---

**Happy analyzing!** 🌍💧
