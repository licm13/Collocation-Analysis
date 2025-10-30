# ELI (Ecosystem Limitation Index) Application

## Overview

This module provides a complete Python implementation of the ELI (Ecosystem Limitation Index) calculation framework, converted from the original MATLAB code in the `ELI Application/` folder.

The ELI quantifies the relative importance of water versus energy limitation in terrestrial ecosystems, based on the research paper:

**Reference:** "Widespread shift from ecosystem energy to water limitation with climate change"

## What was Converted

### Original MATLAB Files

1. **batch.m** - Batch processing script
2. **ELG38a_IVD.m** - Dual-product processing (ERA5L + GLEAM v3.8a, 1980-1999)
   - Uses IVD (Instrumental Variable Design) method
   - Processes 2 data products

3. **ELG21G38_EIVD.m** - Triple-product processing (ERA5L + GLEAM + GLDAS, 2000-2022)
   - Uses EIVD (Extended IVD) method
   - Processes 3 data products
   - Handles error cross-correlation

### Python Implementation

All functionality has been converted to Python in the `collocation/eli.py` module:

- **ELIProcessor**: Main class for processing ELI data
- **process_eli_data**: Convenience function for quick processing
- **calculate_eli_index**: Calculate ELI from merged results

## Variables Processed

The ELI framework processes the following variables:

| Variable | Description | Units | Data Sources |
|----------|-------------|-------|--------------|
| `nsma` | Near-surface soil moisture anomaly (0-10cm) | mm³/mm³ | ERA5L, GLEAM, GLDAS |
| `ssma` | Sub-surface soil moisture anomaly (10-100cm) | mm³/mm³ | ERA5L, GLEAM, GLDAS |
| `tvega` | Transpiration anomaly | mm/month | ERA5L, GLEAM, GLDAS |
| `eta` | Total evapotranspiration anomaly | mm/month | ERA5L, GLEAM, GLDAS |
| `swa` | Downward short-wave radiation flux anomaly | J/m² | ERA5L, GLDAS |

## Data Sources

### Supported Products

1. **ERA5-Land (ERA5L)**: ECMWF reanalysis, 0.25° resolution
2. **GLEAM v3.8a (G38a)**: Global Land Evaporation Amsterdam Model
3. **GLDAS v2.1 (G21)**: Global Land Data Assimilation System

## Collocation Methods

The Python implementation integrates ALL available collocation methods:

### 1. IVD (Instrumental Variable Design)
- **When to use**: 2 data products
- **Features**:
  - Optimal temporal offset selection
  - Error variance estimation
  - Optimal merging weights
- **Example**: ERA5L + GLEAM (1980-1999)

### 2. EIVD (Extended IVD)
- **When to use**: 3 data products
- **Features**:
  - All IVD features
  - **Error cross-correlation estimation**
  - Lag-1 temporal correlation
- **Example**: ERA5L + GLEAM + GLDAS (2000-2022)

### 3. TC (Triple Collocation)
- **When to use**: 3 data products (assumes independent errors)
- **Features**:
  - Classic TC method
  - SNR estimation
  - Faster than EIVD
- **Limitation**: Assumes zero error cross-correlation

### 4. Bayesian Triple Collocation (Optional)
- **When to use**: 3+ products, need uncertainty quantification
- **Features**:
  - Time-varying error estimates
  - Full posterior distributions
  - MCMC-based inference
- **Requirements**: PyMC3 library

## Installation

```bash
# Install the package
cd Collocation-Analysis
pip install -e .

# For Bayesian methods (optional)
pip install pymc3==3.11.5 theano-pymc
```

## Quick Start

### Example 1: Process Dual Products with IVD

```python
from collocation import ELIProcessor
import numpy as np

# Initialize processor
processor = ELIProcessor()

# Load your data (shape: n_time, n_lat, n_lon)
era5l_data = ...  # Load from NetCDF
gleam_data = ...  # Load from NetCDF

# Process with IVD
results = processor.process_dual_ivd(
    era5l_data,
    gleam_data,
    variable='eta'
)

# Access results
error_variance = results['error_variance']  # (n_lat, n_lon, 2)
rho2 = results['rho2']                      # Data-truth correlation
weights = results['weights']                 # Merging weights
merged = results['merged']                   # Merged product
```

### Example 2: Process Triple Products with EIVD

```python
# Load three products
era5l_data = ...
gleam_data = ...
gldas_data = ...

# Process with EIVD (handles error cross-correlation)
results = processor.process_triple_eivd(
    era5l_data,
    gleam_data,
    gldas_data,
    variable='eta'
)

# Access additional results
error_cross_corr = results['error_cross_correlation']  # (n_lat, n_lon, 3)
```

### Example 3: Compare All Methods

```python
# Apply ALL methods and compare
results_all = processor.process_triple_with_all_methods(
    era5l_data,
    gleam_data,
    gldas_data,
    variable='eta',
    use_bayesian=False  # Set True for Bayesian TC
)

# Access results from different methods
eivd_results = results_all['eivd']
tc_results = results_all['tc']
comparison = results_all['comparison']

# Get recommendations
for rec in comparison['recommendations']:
    print(rec)
```

### Example 4: Export to NetCDF

```python
# Save results to NetCDF
processor.save_to_netcdf(
    results,
    output_path='eli_eta_results.nc',
    variable='eta',
    data_source='ERA5L+GLEAM+GLDAS',
    metadata={
        'description': 'ELI analysis for evapotranspiration',
        'time_range': '2000-2022'
    }
)
```

## Complete Workflow Example

See `examples/eli_comprehensive_example.py` for a complete demonstration including:

1. Processing dual datasets (IVD)
2. Processing triple datasets (EIVD)
3. Comparing all methods
4. Calculating ELI indices
5. Time series analysis
6. Exporting results to NetCDF

Run the example:

```bash
cd Collocation-Analysis/examples
python eli_comprehensive_example.py
```

## Key Improvements Over MATLAB Code

### 1. **Unified Interface**
- Single `ELIProcessor` class handles all methods
- Consistent API across IVD, EIVD, TC, Bayesian TC

### 2. **Method Integration**
- All collocation methods available in one place
- Easy comparison between methods
- Automatic method selection based on data

### 3. **Modern Data Handling**
- Uses xarray for NetCDF I/O
- Better memory management
- Supports chunked processing for large datasets

### 4. **Enhanced Analysis**
- Built-in method comparison
- Automated recommendations
- Time series analysis tools

### 5. **Better Error Handling**
- Robust NaN handling
- Informative error messages
- Progress reporting

### 6. **Extensibility**
- Easy to add new methods
- Modular design
- Well-documented code

## Differences from MATLAB Code

### Spatial Coordinates
- **MATLAB**: Hardcoded for specific region (89.75°N to -60°S, global longitude)
- **Python**: Flexible coordinates, can be customized

### Data Format
- **MATLAB**: Direct NetCDF reading with hardcoded paths
- **Python**: Uses xarray, supports multiple formats

### Processing
- **MATLAB**: Grid-based loops
- **Python**: Same approach but with optional vectorization

### Methods Available
- **MATLAB**: IVD, EIVD only
- **Python**: IVD, EIVD, TC, Bayesian TC, and more

## Performance Considerations

For large spatial domains:

```python
# Process in chunks
from pathlib import Path
import numpy as np

# Define chunks
lat_chunks = [(0, 200), (200, 400), (400, 600)]

processor = ELIProcessor()

for lat_start, lat_end in lat_chunks:
    # Load data chunk
    data_chunk = load_data_chunk(lat_start, lat_end)

    # Process
    results = processor.process_triple_eivd(...)

    # Save chunk
    output_file = f'eli_results_lat_{lat_start}_{lat_end}.nc'
    processor.save_to_netcdf(results, output_file, ...)
```

## Citation

If you use this code, please cite:

1. **Original paper**: "Widespread shift from ecosystem energy to water limitation with climate change"

2. **Collocation methods**:
   - IVD: Dong et al. (2014)
   - EIVD: Dong et al. (2019)
   - TC: Stoffelen (1998), Scipal et al. (2008)
   - Bayesian TC: Zwieback et al. (2012)

## Support and Contribution

- **Issues**: Report bugs or request features on GitHub
- **Documentation**: See main `README.md` and docstrings
- **Examples**: Check `examples/` directory

## License

Same as the main Collocation-Analysis package.

## Contact

For questions about the ELI implementation:
- Original MATLAB code: licm_13@163.com
- Python conversion: See repository contributors

---

## Appendix: Method Selection Guide

### When to use IVD
- ✅ You have 2 data products
- ✅ Products have temporal correlation
- ✅ Need optimal merging weights

### When to use EIVD
- ✅ You have 3 data products
- ✅ Suspect error cross-correlation between products
- ✅ Products from similar sensors/models
- ✅ Need to quantify error correlation

### When to use TC
- ✅ You have 3 data products
- ✅ Confident errors are independent
- ✅ Need faster processing
- ✅ Large spatial domains

### When to use Bayesian TC
- ✅ You have 3+ products
- ✅ Need time-varying error estimates
- ✅ Want full uncertainty quantification
- ✅ Have computational resources for MCMC
- ⚠️ Computationally expensive

### Recommendation
**Start with EIVD** - it's the most general method that handles error cross-correlation. Compare with TC to assess whether error correlation matters for your data.

---

**Last updated**: 2025-10-30
**Version**: 1.2.0
