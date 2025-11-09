# HCC-ET Framework: Hierarchical Cross-Calibration for ET Decomposition

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready framework for quantifying global "productive" (T) vs "non-productive" (E) water flux reshaping under climate change and human land management (1950-2025, 1km resolution).

**Target Publication**: Nature Climate Change

---

## Table of Contents

- [Overview](#overview)
- [Scientific Innovation](#scientific-innovation)
- [Framework Architecture](#framework-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Citation](#citation)

---

## Overview

The HCC-ET Framework addresses a fundamental question in global change science:

> **How are "productive" transpiration (T) and "non-productive" evaporation (E) being systematically reshaped by climate change and human land management?**

### Key Features

- **Fuses T/ET ratio** (not total ET) using validated collocation methods
- **Handles Error Cross-Correlation (ECC)** induced by shared 1km predictors
- **Multi-scale physical constraints** (site-level SapFlux + basin-level Water Balance)
- **Production-ready**: Modular, documented, tested
- **Integrates seamlessly** with existing `collocation-analysis` library

### Why This Matters

- **T** (transpiration) = "productive" water flux → plant growth, carbon uptake
- **E** (evaporation) = "non-productive" water flux → soil/canopy evaporation
- **T/E partitioning** fundamentally shapes:
  - Ecosystem water use efficiency
  - Agricultural productivity
  - Climate feedbacks

---

## Scientific Innovation

### 1. Fusion Target: T/ET Ratio (Not Total ET)

**Traditional Approach** (wrong):
```
Fuse T₁, T₂, T₃  →  T_fused
Fuse E₁, E₂, E₃  →  E_fused
Problem: E = ET - T (residual), accumulates all errors!
```

**HCC-ET Approach** (correct):
```
Fuse (T/ET)₁, (T/ET)₂, (T/ET)₃  →  (T/ET)_fused
Compute: T = ET × (T/ET)_fused
         E = ET × (1 - (T/ET)_fused)
Benefit: Direct constraint on partitioning, not residual!
```

### 2. ECC-Aware Fusion: EIVD/GLS

**Why ECC Matters**:
- All 1km candidates use **SAME 1km predictors** (LST, NDVI, Albedo, Precip)
- Shared predictors → **correlated prediction errors**
- Traditional TC assumes **zero error correlation** → biased weights!

**HCC-ET Solution**:
- **EIVD** (Dong et al., 2019): Estimates error cross-correlation for 3 products
- **EC** (Gruber et al., 2016): Handles 4 products with full covariance
- **GLS via `fuse_fields`**: General approach for n ≥ 3 products

### 3. Multi-Scale Physical Constraints

| Constraint | Type | Scale | Source |
|------------|------|-------|--------|
| **SapFluxNet** | Soft | Site (~km²) | Sap flow + EC towers |
| **Water Balance** | Hard | Basin (10³-10⁶ km²) | P - Q - dS (GRDC + GRACE) |

**Why Both?**:
- **SapFlux**: High accuracy, sparse coverage → PFT-specific bias correction
- **Water Balance**: Lower accuracy, complete coverage → basin-scale closure

---

## Framework Architecture

```
hcc_et_framework/
├── hcc_framework.py          # Main orchestrator (5 phases)
├── data_loader.py            # Data I/O (parent products, predictors, constraints)
├── downscaler.py             # ML-based statistical downscaling (RF, GB)
├── physical_constraints.py   # SapFlux + WB constraints
├── config.py                 # Configuration management (YAML)
├── utils_hcc.py              # Utilities (trends, metrics, visualization)
├── __init__.py               # Package initialization
├── config_template.yaml      # Configuration template
└── README.md                 # This file
```

### Workflow (5 Phases)

```
Phase 1: Data Preparation
  ├─ Load parent T/ET ratios (GLEAM, PMLv2, GLDAS)
  ├─ Load 1km predictors (LST, NDVI, Albedo, Precip)
  └─ Load constraints (SapFlux, WB-ET, land cover)
        ↓
Phase 2: Statistical Downscaling
  ├─ Train RF models: parent T/ET → 1km T/ET candidates
  └─ Predict 1km candidates (shared predictors → ECC!)
        ↓
Phase 3: Cross-Calibration Fusion (EIVD/GLS)
  ├─ Estimate error covariance (with ECC!)
  ├─ Compute optimal weights
  ├─ Fuse candidates → T/ET ratio (1km)
  └─ Uncertainty quantification (Bayesian TC)
        ↓
Phase 4: Multi-Scale Physical Constraints
  ├─ Water Balance: Constrain total ET (basin scale)
  ├─ SapFluxNet: Constrain T/ET ratio (site scale)
  └─ Compute final T and E
        ↓
Phase 5: Temporal Hindcast & Trend Analysis
  ├─ Apply trained model to 1950-1980 predictors
  ├─ Merge with modern period (1981-2025)
  ├─ Compute Theil-Sen trends by land cover
  └─ Report: Natural vs Managed ecosystem trends
```

---

## Installation

### Prerequisites

- Python 3.7+
- Existing `collocation-analysis` library (installed at parent level)

### Install HCC-ET Framework

```bash
# Navigate to collocation-analysis repository
cd /path/to/Collocation-Analysis

# The framework is already in place at:
# /path/to/Collocation-Analysis/hcc_et_framework/

# Install additional dependencies (if needed)
pip install pyyaml scikit-learn scipy xarray netCDF4
```

### Verify Installation

```python
from hcc_et_framework import HCCETFramework, load_config
print("HCC-ET Framework installed successfully!")
```

---

## Quick Start

### 1. Create Configuration File

```bash
cd hcc_et_framework
cp config_template.yaml my_config.yaml
# Edit my_config.yaml with your data paths
```

### 2. Run Full Pipeline

```python
from hcc_et_framework import HCCETFramework

# Initialize framework
framework = HCCETFramework(
    config_file='my_config.yaml',
    output_dir='./outputs'
)

# Run all 5 phases
framework.run_full_pipeline()
```

### 3. Or Run Individual Phases

```python
# Run only Phase 3 (fusion)
framework.run_phase1_data_preparation()
framework.run_phase2_downscaling()
framework.run_phase3_fusion()
```

### 4. Command-Line Interface

```bash
# Run full pipeline
python -m hcc_et_framework.hcc_framework \
    --config my_config.yaml \
    --output-dir ./outputs \
    --phase all

# Run specific phase
python -m hcc_et_framework.hcc_framework \
    --config my_config.yaml \
    --phase 3
```

---

## Configuration

### Key Configuration Sections

#### 1. Data Paths

```yaml
data:
  parent_dir: /data/parent_products
  parent_products: [GLEAM, PMLv2, GLDAS]
  predictors_dir: /data/predictors_1km
  predictor_vars: [LST, NDVI, Albedo, Precip]
```

#### 2. Fusion Mode

```yaml
fusion:
  mode: auto  # auto-selects based on n_models
  # mode: eivd  # force EIVD for 3 products
  # mode: ec    # force EC for 4 products
  # mode: gls   # force GLS (general)

  shrinkage: lw  # Ledoit-Wolf shrinkage
  spatial_varying_weights: false  # true = pixel-wise (slow)
```

#### 3. Physical Constraints

```yaml
constraints:
  apply_wb_constraint: true
  wb_tolerance: 0.05  # 5% relative tolerance

  apply_sapflux_constraint: true
  sapflux_correction_strength: 0.5  # 0=none, 1=full
```

### Full Template

See [`config_template.yaml`](config_template.yaml) for complete configuration options.

---

## API Reference

### Main Classes

#### `HCCETFramework`

**Main orchestrator for HCC-ET pipeline.**

```python
class HCCETFramework:
    def __init__(self, config_file, output_dir='./outputs'):
        """Initialize framework with configuration."""

    def run_full_pipeline(self):
        """Execute all 5 phases sequentially."""

    def run_phase1_data_preparation(self):
        """Phase 1: Load all input data."""

    def run_phase2_downscaling(self):
        """Phase 2: Train ML models and generate 1km candidates."""

    def run_phase3_fusion(self):
        """Phase 3: Fuse candidates with EIVD/GLS."""

    def run_phase4_physical_constraints(self):
        """Phase 4: Apply SapFlux + WB constraints."""

    def run_phase5_temporal_hindcast(self):
        """Phase 5: Hindcast and trend analysis."""
```

**Example**:
```python
from hcc_et_framework import HCCETFramework

framework = HCCETFramework(
    config_file='config.yaml',
    output_dir='./outputs'
)

# Run all phases
framework.run_full_pipeline()

# Access results
T_final = framework.T_final_1km  # Transpiration (1km)
E_final = framework.E_final_1km  # Evaporation (1km)
```

---

### Data Loader

```python
from hcc_et_framework import data_loader

# Load parent T/ET ratios
parent_data = data_loader.load_parent_t_et_ratios(
    data_dir='/data/parent',
    products=['GLEAM', 'PMLv2', 'GLDAS'],
    time_range=('1981-01-01', '2025-12-31')
)

# Load 1km predictors
predictors = data_loader.load_predictors_1km(
    data_dir='/data/predictors',
    variables=['LST', 'NDVI', 'Albedo', 'Precip'],
    time_range=('1981-01-01', '2025-12-31')
)
```

---

### Downscaler

```python
from hcc_et_framework import downscaler

# Train downscaling models
model = downscaler.train_downscaler(
    parent_data=parent_data,
    predictors_1km=predictors,
    model_type='random_forest',
    cv_folds=5
)

# Generate 1km candidates
candidates = downscaler.predict(
    model=model,
    predictors_1km=predictors
)
```

---

### Physical Constraints

```python
from hcc_et_framework import physical_constraints

# Apply SapFluxNet constraint
sf_constraint = physical_constraints.SapfluxnetConstraint(sapflux_data)
t_et_corrected = sf_constraint.apply(
    t_et_ratio_fused=fused_ratio,
    total_et=total_et,
    pft_map=pft_map
)

# Apply Water Balance constraint
wb_constraint = physical_constraints.WaterBalanceConstraint(wb_et, basin_mask)
et_corrected = wb_constraint.apply(
    total_et_1km_fused=total_et,
    uncertainty=eivd_uncertainty
)
```

---

## Examples

### Example 1: Run Full Pipeline

```python
from hcc_et_framework import HCCETFramework

# Initialize
framework = HCCETFramework(
    config_file='config.yaml',
    output_dir='./outputs'
)

# Run all phases
framework.run_full_pipeline()

# Results saved to ./outputs/
# - t_et_ratio_1950_2025.nc
# - trends_by_landcover.csv
# - weights.nc
# - diagnostics.json
```

### Example 2: Custom Fusion Workflow

```python
from hcc_et_framework import data_loader, downscaler
from collocation import eivd
import xarray as xr

# Load data
parent = data_loader.load_parent_t_et_ratios(...)
predictors = data_loader.load_predictors_1km(...)

# Downscale
model = downscaler.train_downscaler(parent, predictors)
candidates = downscaler.predict(model, predictors)

# Apply EIVD manually
data_stacked = candidates.stack(sample=('time', 'lat', 'lon')).values.T
EeeT, SNR, rho2, fMSE, L = eivd(data_stacked)

print(f"Error cross-correlation (2-3): {EeeT[1,2]:.4f}")
print(f"SNR: {SNR}")

# Compute fusion weights
import numpy as np
EeeT_inv = np.linalg.inv(EeeT)
ones = np.ones(3)
weights = EeeT_inv @ ones / (ones @ EeeT_inv @ ones)

# Fuse
weights_xr = xr.DataArray(weights, dims=['model'], coords={'model': candidates.coords['model']})
fused = (candidates * weights_xr).sum(dim='model')
```

### Example 3: Trend Analysis

```python
from hcc_et_framework import utils_hcc
import xarray as xr

# Load fused T/ET ratio time series (1950-2025)
t_et_ratio = xr.open_dataarray('outputs/t_et_ratio_1950_2025.nc')

# Compute spatial trends
slopes, pvals = utils_hcc.compute_spatial_trends(
    t_et_ratio,
    dim='time',
    method='theil_sen'
)

# Convert to decadal trends
decadal_trends = slopes * 10  # per decade

# Mask significant trends (p < 0.05)
significant_trends = decadal_trends.where(pvals < 0.05)

# Plot
import matplotlib.pyplot as plt
significant_trends.plot(cmap='RdBu_r', vmin=-0.05, vmax=0.05)
plt.title('Significant T/ET Trends (1950-2025, per decade)')
plt.savefig('trends_map.png', dpi=300)
```

---

## Integration with Existing `collocation-analysis` Library

The HCC-ET Framework seamlessly integrates with your existing `collocation-analysis` library:

### Direct Use of Collocation Methods

```python
# Import from existing library
from collocation import eivd, ec, BayesianTC
from collocation.fusion.fuse import fuse_fields
from collocation.fusion.constraints import SumToOneConstraint, BoundsConstraint

# Use in HCC-ET workflow
EeeT, SNR, rho2, fMSE, L = eivd(candidates)  # EIVD fusion

# Or use high-level fusion API
result = fuse_fields(
    X=candidates,
    mode='gls',
    shrinkage='lw',
    return_weights=True,
    return_var=True
)
```

### Extending Constraints

HCC-ET extends `collocation.fusion.constraints.Constraint`:

```python
# HCC-ET constraints inherit from base class
from collocation.fusion.constraints import Constraint
from hcc_et_framework.physical_constraints import SapfluxnetConstraint

# New constraint class
class SapfluxnetConstraint(Constraint):
    def as_matrices(self):
        """Implement constraint matrix form."""
        return None, None  # Post-fusion constraint

    def apply(self, t_et_ratio_fused, total_et, pft_map):
        """Apply PFT-specific bias correction."""
        # ... implementation ...
```

---

## Output Files

After running the full pipeline, the following files are generated:

```
outputs/
├── t_et_ratio_1950_2025.nc        # Fused T/ET ratio time series
├── T_final_1km.nc                 # Final transpiration (1km)
├── E_final_1km.nc                 # Final evaporation (1km)
├── weights.nc                     # Fusion weights
├── trends_by_landcover.csv        # Trend summary table
├── diagnostics.json               # Performance metrics
├── downscaler_model.pkl           # Trained ML model
└── uncertainty_estimates.nc       # Bayesian TC uncertainty
```

---

## Performance Notes

### Computational Requirements

- **Phase 1** (Data Preparation): Fast (~minutes for global 1km)
- **Phase 2** (Downscaling): Moderate (~hours for RF training)
- **Phase 3** (Fusion):
  - Global weights: Fast (~seconds)
  - Pixel-wise weights: Slow (~hours for global 1km)
- **Phase 4** (Constraints): Moderate (~minutes)
- **Phase 5** (Hindcast): Fast (reuses trained model)

### Optimization Tips

1. **Use global weights** (set `spatial_varying_weights: false`)
2. **Subsample for Bayesian TC** (set `bayesian_sample_size: 10000`)
3. **Chunk large datasets** with Dask/xarray
4. **Parallelize downscaling** (set `n_jobs: -1` in RF config)

---

## Troubleshooting

### Common Issues

**1. Singular matrix in EIVD**
```
Error: numpy.linalg.LinAlgError: Singular matrix
```
**Solution**: Insufficient temporal variation or near-perfect correlation. Use GLS mode instead.

**2. Memory error with pixel-wise fusion**
```
Error: MemoryError
```
**Solution**: Set `spatial_varying_weights: false` or process in spatial chunks.

**3. Missing data files**
```
Warning: Data path does not exist: /data/...
```
**Solution**: Update paths in `config.yaml` to match your system.

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{yourname2025hcc,
  title={Widespread Reshaping of Productive vs Non-Productive Water Fluxes under Climate Change and Land Management},
  author={Your Name and Co-Authors},
  journal={Nature Climate Change},
  year={2025},
  volume={XX},
  pages={XXX--XXX},
  doi={10.1038/s41558-XXX-XXXX-X}
}
```

Also cite the underlying collocation methods:

```bibtex
@article{dong2019eivd,
  title={An instrument variable based algorithm for estimating cross-correlated hydrological remote sensing errors},
  author={Dong, Jianzhi and Crow, Wade T and Duan, Zheng and Wei, Lingna and Lu, Ying},
  journal={Journal of Hydrology},
  volume={581},
  pages={124385},
  year={2019},
  doi={10.1016/j.jhydrol.2019.124385}
}

@article{gruber2016ec,
  title={Estimating error cross-correlations in soil moisture data sets using extended collocation analysis},
  author={Gruber, Alexander and Su, Chun-Hsu and Crow, Wade T and Zwieback, Simon and Dorigo, Wouter A and Wagner, Wolfgang},
  journal={Journal of Geophysical Research: Atmospheres},
  volume={121},
  number={3},
  pages={1208--1219},
  year={2016},
  doi={10.1002/2015JD024027}
}
```

---

## License

MIT License - see parent repository for details.

---

## Contact

- **Author**: [Your Name]
- **Email**: [your.email@example.com]
- **GitHub**: [repository URL]

For bug reports and feature requests, please open an issue on GitHub.

---

## Acknowledgments

This framework builds on the excellent `collocation-analysis` library and integrates methods from:

- Dong et al. (2019) - EIVD
- Gruber et al. (2016) - Extended Collocation
- Wei et al. (2023) - ETCC
- Zwieback et al. (2012) - Bayesian TC

Special thanks to the SapFluxNet and GRDC communities for providing validation data.

---

**Version**: 1.0.0
**Last Updated**: 2025-11-09
