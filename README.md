# Collocation Analysis Package

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A comprehensive Python package for collocation-based error analysis of remote sensing and geophysical datasets. Converted from MATLAB and enhanced with modern Python features.

## Overview

Collocation analysis methods enable quantification of errors in multiple datasets without requiring ground truth. These techniques are widely used in remote sensing, climate science, and geophysical data validation.

## Architecture Overview

The repository is organised around a reusable `collocation` Python package and a collection of executable examples and regression tests. The modules map one-to-one to the statistical techniques implemented in the original MATLAB toolbox:

- **Core estimators (`collocation/`)**
  - `tc.py`, `ivd.py`, `ivs.py`, `eivd.py`, `ec.py`, `etcc_evaluation.py`, `etcc_spatial.py`: vector-based analytical solutions for 2-, 3- and 4-way collocation problems.
  - `etcc.py`: object-oriented implementations of the Wei et al. (2023) Triple Collocation merger (`TripleCollocation`) and the exhaustive-search correlation maximiser (`ETCC`).
  - `bayesian_tc.py`, `bayesian_tch.py`: optional PyMC3-based inference engines with full posterior diagnostics.
  - `simple_average.py`: deterministic fusion baselines used for benchmarking and quick quality control.
  - `fuse.py`, `covariance.py`, `utils.py`: shared helpers for bias estimation, covariance construction, and skill metrics (KGE, NSE, etc.). Optional xarray powered helpers expose availability flags (`XR_AVAILABLE`, `ELI_AVAILABLE`) so light-weight installs can gracefully skip the ecosystem modules.
- **Applied workflows (`collocation/eli.py`, `ELI Application/`)**
  - High-level processors that orchestrate collocation runs for ecosystem limitation studies including NetCDF I/O.
- **Examples (`examples/`)**
  - Reproducible scripts spanning introductory walkthroughs (`example_all_methods.py`), publication-grade comparisons (`comprehensive_comparison_TCH_included.py`), and the new `robust_collocation_scenarios.py` stress suite.
- **Tests (`tests/`)**
  - Unit and integration tests covering estimator algebra, covariance utilities, and multi-scenario regression checks (`test_method_workflows.py`).

This layered structure keeps numerical kernels small and testable while allowing end-to-end pipelines to be expressed as Python scripts that can double as tutorials.

## Features

### Implemented Methods

#### Classical Methods

- **IVD** (Information Vector Dual): 2-way collocation with temporal offset optimization
- **IVS** (Information Vector with Scaling): 2-way collocation with bootstrap uncertainty estimation
- **TC** (Triple Collocation): Classic 3-way collocation assuming independent errors
- **EIVD** (Extended IVD): 3-way collocation allowing error cross-correlation
- **EC** (Extended Collocation): 4-way quadruple collocation with optimal merging weights
- **ETCC** (Extended Triple Collocation for Correlation): 3-way collocation optimized for maximum correlation with truth
  - Maximizes correlation instead of minimizing error variance
  - Uses exhaustive weight search for optimal fusion
  - Based on Wei et al. (2023) for precipitation merging
  - Ideal for applications where correlation is more important than RMSE

#### Bayesian Methods

- **BTC** (Bayesian Triple Collocation): 3-way collocation with full Bayesian inference
  - Time-varying error structures
  - Complex heteroscedastic models
  - Full uncertainty quantification via MCMC
  - Non-constant calibration parameters
  - Requires PyMC3 (optional dependency)

- **BTCH** (Bayesian Three-Cornered Hat): 3-way collocation with Bayesian uncertainty
  - Constant error variances (homoscedastic)
  - Full Bayesian uncertainty quantification
  - Simpler and faster than BTC
  - Ideal when time-varying errors not expected
  - Requires PyMC3 (optional dependency)

- **BTCH_He2020** (He et al. 2020): Fast analytical BTCH method
  - Uses Three-Cornered Hat (TCH) for error variance point estimates
  - Applies analytical Bayesian weight formula (no MCMC)
  - Supports both uncorrelated and correlated error estimation
  - Fast, deterministic, minimal dependencies (NumPy only)
  - No uncertainty quantification (point estimates only)
  - Ideal when speed is critical and uncertainty not required

#### Application: Ecosystem Limitation Index (ELI)

- **NEW**: Complete Python implementation of ELI calculation framework
  - Converted from MATLAB code for ecosystem water/energy limitation analysis
  - Processes multiple climate data sources (ERA5-Land, GLEAM, GLDAS)
  - Applies all collocation methods (IVD, EIVD, TC, Bayesian TC)
  - Based on: "Widespread shift from ecosystem energy to water limitation with climate change"
  - Variables: soil moisture, evapotranspiration, transpiration, radiation
  - See [ELI_README.md](ELI_README.md) for detailed documentation
  - Example: `examples/eli_comprehensive_example.py`

### Key Capabilities

- Error variance estimation without ground truth
- Signal-to-noise ratio calculation
- Data-truth correlation estimation
- Optimal data fusion weights
- Error cross-correlation detection
- Bootstrap uncertainty quantification
- Comprehensive performance metrics (KGE, NSE, RMSE, MAE, etc.)

## Installation

### From source

```bash
git clone https://github.com/yourusername/Collocation-Analysis.git
cd Collocation-Analysis
pip install -e .
```

### Dependencies

#### Core Dependencies
- Python >= 3.7
- NumPy >= 1.18.0
- SciPy >= 1.4.0
- Matplotlib >= 3.1.0 (for examples)
- pytest >= 6.0.0 (for testing)

#### Optional Dependencies
- PyMC3 >= 3.11.0 (for Bayesian Triple Collocation)
- Theano >= 1.0.0 (backend for PyMC3)
- xarray >= 0.18 (for covariance post-processing, bias estimation, and the ELI workflow)

Install with Bayesian support:
```bash
pip install -e . "pymc3>=3.11.0" "theano-pymc"
```

## Quick Start

### Basic Example: Triple Collocation

```python
import numpy as np
from collocation import tc

# Three independent products measuring the same variable
product1 = np.random.randn(200) + true_signal
product2 = np.random.randn(200) + true_signal
product3 = np.random.randn(200) + true_signal

# Stack into matrix
data = np.column_stack([product1, product2, product3])

# Apply triple collocation
EeeT, SNR, rho2, fMSE = tc(data)

print("Error variances:", np.diag(EeeT))
print("Signal-to-Noise Ratios:", SNR)
print("Data-truth correlations:", rho2)
```

### IVD for Two Products

```python
from collocation import ivd

# Two products
dual = np.column_stack([product1, product2])

# Apply IVD
EeeT, rho2, weights = ivd(dual)

# Merge products using optimal weights
merged = weights[0] * product1 + weights[1] * product2
```

### Extended Collocation for Four Products

```python
from collocation import ec
from collocation.ec import select_best_combination

# Four products
quad = np.column_stack([product1, product2, product3, product4])

# Apply extended collocation
results = ec(quad)

# Select best combination
best = select_best_combination(results, criterion='min_fMSE')

# Get merged result
merged = best.weighted_result[0]
```

### ETCC for Maximum Correlation

```python
from collocation import ETCC, TripleCollocation

# Three precipitation products
precip1 = np.array([...])  # Product 1
precip2 = np.array([...])  # Product 2
precip3 = np.array([...])  # Product 3

# Traditional TC (minimizes error variance)
tc_method = TripleCollocation()
tc_merged = tc_method.merge(precip1, precip2, precip3)
print("TC weights:", tc_method.weights)
print("TC error variances:", tc_method.error_variances)

# ETCC (maximizes correlation with truth)
etcc_method = ETCC(weight_increment=0.01)
etcc_merged = etcc_method.merge(precip1, precip2, precip3)
print("ETCC weights:", etcc_method.weights)
print("ETCC max correlation:", etcc_method.max_correlation)
print("Correlations with truth:", etcc_method.correlation_with_truth)

# For gridded spatial data
from collocation import SpatialMerging

# Data shape: (lat, lon, time)
spatial_merger = SpatialMerging(method='etcc', weight_increment=0.01)
merged_grid = spatial_merger.merge_gridded(grid1, grid2, grid3, axis=-1)
```

### Bayesian Methods (Advanced)

#### Bayesian Triple Collocation (Time-Varying Errors)

```python
from collocation import BayesianTC, BAYESIAN_AVAILABLE

if BAYESIAN_AVAILABLE:
    # Three products (n_products, n_samples)
    data = np.array([product1, product2, product3])

    # Initialize and run Bayesian TC
    btc = BayesianTC(data)
    btc.run_inference(niter=2000, nadvi=200000)

    # Get results with full uncertainty quantification
    rmse_mean, rmse_std, rmse_quantiles = btc.get_error_estimates()

    # Get calibration parameters
    m_mean, l_mean = btc.get_calibration_parameters()

    # Print summary
    btc.summary()
else:
    print("Install PyMC3 for Bayesian methods: pip install pymc3==3.11.5 theano-pymc")
```

#### Bayesian Three-Cornered Hat (Constant Errors)

```python
from collocation import BayesianTCH, BAYESIAN_TCH_AVAILABLE

if BAYESIAN_TCH_AVAILABLE:
    # Three products (n_products, n_samples)
    data = np.array([product1, product2, product3])

    # Initialize and run Bayesian TCH (faster than BTC)
    btch = BayesianTCH(data)
    btch.run_inference(niter=2000, nadvi=50000)

    # Get results with uncertainty quantification
    rmse_mean, rmse_std, rmse_quantiles = btch.get_error_estimates()

    # Get SNR with uncertainty
    snr_mean, snr_std, snr_quantiles = btch.get_snr()

    # Get correlation with uncertainty
    rho2_mean, rho2_std, rho2_quantiles = btch.get_correlation()

    # Print summary (verbose=True for full details)
    btch.summary(verbose=True)

    # Plot posterior distributions
    fig, axes = btch.plot_posterior()
else:
    print("Install PyMC3 for Bayesian methods: pip install pymc3==3.11.5 theano-pymc")
```

#### BTCH He et al. 2020 (Fast Analytical Method)

```python
from collocation import BTCH_He2020, btch_he2020

# Option 1: Convenience function (similar to tc, ivd, etc.)
# Three products (n_samples, 3)
data = np.column_stack([product1, product2, product3])

# Basic usage (uncorrelated TCH)
error_variances, weights, fused = btch_he2020(data)
print("Error variances:", error_variances)
print("Weights:", weights)

# With correlated TCH
error_variances, weights, fused = btch_he2020(
    data,
    method='correlated',
    alpha=1e-2,  # Regularization parameter
    iters=500    # Optimization iterations
)

# Get full results
results = btch_he2020(data, return_full=True)
print("RMSE:", results['rmse'])
print("Error covariance:", results['error_covariance'])

# Option 2: Class-based API (more control)
btch = BTCH_He2020(data)

# Estimate variances
btch.estimate_variances(method='uncorrelated')

# Compute weights
btch.compute_weights()

# Fuse products
fused = btch.fuse()

# Print summary
btch.summary()

# Or run all steps at once
results = btch.run(method='correlated', alpha=1e-2, iters=500)
```

**Key Advantages:**
- Very fast (no MCMC, pure analytical)
- Minimal dependencies (NumPy only, no PyMC3)
- Deterministic results (no sampling variability)
- Can handle correlated errors (regularized TCH)
- Simple API consistent with other methods

**Limitations:**
- No uncertainty quantification (point estimates only)
- For full Bayesian inference, use BayesianTCH or BayesianTC instead

## Method Comparison

| Method | # Products | Error Correlation | Temporal Info | Uncertainty | Optimization | Best For |
|--------|-----------|-------------------|---------------|-------------|--------------|----------|
| **IVD** | 2 | No | Offset-based | No | Analytical | Fusing two products with optimal weights |
| **IVS** | 2 | No | Lag-1 + Bootstrap | Bootstrap | Analytical | Uncertainty quantification (2-way) |
| **TC** | 3 | Zero (assumed) | No | No | Min RMSE | Standard 3-way validation |
| **ETCC** | 3 | Zero (assumed) | No | No | Max Correlation | Precipitation/applications prioritizing correlation |
| **EIVD** | 3 | Yes (estimated) | Lag-1 | No | Analytical | Products with correlated errors |
| **EC** | 4 | Yes (estimated) | No | No | Analytical | Multi-sensor fusion |
| **BTC** | 3+ | Yes (estimated) | Time-varying | Full Bayesian | MCMC | Complex error structures, time-varying |
| **BTCH** | 3 | Zero (assumed) | No | Full Bayesian | MCMC | Constant errors with uncertainty quantification |
| **BTCH (He 2020)** | 3 | Yes (approx.) | No | No (point est.) | Analytical (TCH + weights) | Fast fusion without MCMC, correlated errors |

## Comprehensive Examples

### Example 1: All Classical Methods

Run the complete example demonstrating all classical methods:

```bash
cd examples
python example_all_methods.py
```

This will:
1. Generate synthetic data with known errors
2. Apply all classical collocation methods (IVD, IVS, TC, EIVD, EC)
3. Compare results with ground truth
4. Create visualization plots

### Example 2: Comprehensive Comparison (Publication-Quality)

Run the comprehensive comparison across multiple scenarios with Nature/Science quality figures:

```bash
cd examples
python comprehensive_comparison_TCH_included.py
```

This advanced example includes:
- **6 realistic scenarios**: Ideal, correlated errors, time-varying, biased, heavy-tailed, realistic
- **All methods compared**: IVD, IVS, TC, TCH, EIVD, EC, BTC, BTCH
- **Publication-quality figures**: Following Nature/Science journal standards
  - 300 DPI resolution
  - Colorblind-friendly palette
  - Proper font sizing (Arial/Helvetica, 7-9pt)
  - Single column (89mm) and double column (183mm) layouts
- **Comprehensive metrics**: RMSE, correlation, relative errors, distributions
- **Statistical comparison**: Performance across different challenging conditions

Output:
- Individual scenario comparison figures
- Overall performance comparison
- Detailed results table

### Example 3: Stress-testing heterogeneous error regimes

Run all supported scenarios from the command line:

```bash
python examples/robust_collocation_scenarios.py
```

The script synthesises four distinct sensor constellations (independent errors, correlated errors, bias-dominated products, and correlation-first precipitation merging) and prints a diagnostic report with weight choices, Kling-Gupta efficiency, and truth correlations.

## ELI Application

The Ecosystem Limitation Index (ELI) application provides a complete workflow for analyzing water vs. energy limitation in terrestrial ecosystems.

### Quick Start

```python
from collocation import ELIProcessor

# Initialize processor
processor = ELIProcessor()

# Process three data sources with EIVD
results = processor.process_triple_eivd(
    era5l_data,  # ERA5-Land reanalysis
    gleam_data,  # GLEAM evapotranspiration
    gldas_data,  # GLDAS land surface model
    variable='eta'
)

# Save results
processor.save_to_netcdf(
    results,
    'eli_eta_results.nc',
    variable='eta',
    data_source='ERA5L+GLEAM+GLDAS'
)
```

### Complete Examples

```bash
# Quick test (validates installation)
python examples/test_eli_quick.py

# Comprehensive example (all methods)
python examples/eli_comprehensive_example.py
```

### Documentation

See [ELI_README.md](ELI_README.md) for:
- Complete API documentation
- Data format specifications
- Method selection guide
- NetCDF I/O examples
- Performance optimization tips

## Testing

Run the test suite:

```bash
pytest tests/test_collocation.py -v
```

Run the integration-style workflow checks (includes ETCC, EIVD, averaging, and rescaling assertions):

```bash
pytest tests/test_method_workflows.py -v
```

Run tests with coverage:

```bash
pytest tests/test_collocation.py --cov=collocation --cov-report=html
```

Test ELI module:

```bash
python examples/test_eli_quick.py
```

## API Documentation

### IVD (Information Vector Dual)

```python
from collocation import ivd

EeeT, rho2, u = ivd(dual)
```

**Parameters:**
- `dual`: Input data (n, 2) - Two products

**Returns:**
- `EeeT`: Error covariance matrix (2, 2)
- `rho2`: Data-truth correlations (2,)
- `u`: Optimal merging weights (2,)

**Reference:**
> Dong, J., et al. (2014). Fusing active and passive remotely sensed soil moisture products using information vectors.

### IVS (Information Vector with Scaling)

```python
from collocation import ivs

RMSE, rho2 = ivs(X, N_boot=1000, column=1, M_A=1)
```

**Parameters:**
- `X`: Input data (n, 2)
- `N_boot`: Bootstrap samples (default: 1000)
- `column`: Column for lag-1 series (1 or 2)
- `M_A`: Error model (0=additive, 1=multiplicative)

**Returns:**
- `RMSE`: Error estimates (2,)
- `rho2`: Correlations (2,)

### TC (Triple Collocation)

```python
from collocation import tc

EeeT, SNR, rho2, fMSE = tc(tri)
```

**Parameters:**
- `tri`: Input data (n, 3) - Three products

**Returns:**
- `EeeT`: Error covariance matrix (3, 3)
- `SNR`: Signal-to-noise ratios (3,)
- `rho2`: Data-truth correlations (3,)
- `fMSE`: Fractional MSE (3,)

**Reference:**
> Stoffelen, A. (1998). Toward the true near-surface wind speed: Error modeling and calibration using triple collocation. JGR, 103(C4), 7755-7766.

### EIVD (Extended IVD)

```python
from collocation import eivd

EeeT, SNR, rho2, fMSE, L = eivd(tri)
```

**Parameters:**
- `tri`: Input data (n, 3)

**Returns:**
- `EeeT`: Error covariance matrix with cross-correlation (3, 3)
- `SNR`: Signal-to-noise ratios (3,)
- `rho2`: Data-truth correlations (3,)
- `fMSE`: Fractional MSE (3,)
- `L`: Lag-1 autocorrelations (3,)

**Note:** Products 2 and 3 can have non-zero error cross-correlation.

### EC (Extended Collocation)

```python
from collocation import ec

results = ec(qu)
```

**Parameters:**
- `qu`: Input data (n, 4) - Four products

**Returns:**
- `results`: List of ECResult objects (one per combination)

Each result contains:
- `EeeT`: Error covariance matrices
- `SNR`, `rho2`, `fMSE`: Quality metrics
- `re_weight`: Optimal merging weights
- `weighted_result`: Merged outputs

**Reference:**
> Gruber, A., et al. (2016). Estimating error cross-correlations in soil moisture data sets using extended collocation analysis. JGR: Atmospheres, 121(3), 1208-1219.

### ETCC (Extended Triple Collocation for Correlation)

```python
from collocation import ETCC, TripleCollocation, SpatialMerging

# Traditional TC (minimizes RMSE)
tc = TripleCollocation()
merged_tc = tc.merge(x, y, z)

# ETCC (maximizes correlation)
etcc = ETCC(weight_increment=0.01, min_correlation=0.01)
merged_etcc = etcc.merge(x, y, z)

# Spatial merging for gridded data
spatial = SpatialMerging(method='etcc', weight_increment=0.01)
merged_grid = spatial.merge_gridded(x_grid, y_grid, z_grid, axis=-1)
```

**Parameters (TripleCollocation):**
- No initialization parameters

**Parameters (ETCC):**
- `weight_increment`: Step size for weight search (default: 0.01)
  - Smaller values = finer search but slower
  - 0.01 gives ~5,151 weight combinations
- `min_correlation`: Minimum correlation threshold (default: 0.01)
  - Prevents numerical instabilities

**Parameters (SpatialMerging):**
- `method`: 'tc' or 'etcc' (default: 'etcc')
- `**kwargs`: Additional parameters passed to TC or ETCC

**Methods:**
```python
# For TripleCollocation
merged = tc.merge(x, y, z)
# Access results:
tc.weights              # {'wx': float, 'wy': float, 'wz': float}
tc.error_variances      # {'sigma2_x': float, 'sigma2_y': float, 'sigma2_z': float}

# For ETCC
merged = etcc.merge(x, y, z)
# Access results:
etcc.weights                   # {'wx': float, 'wy': float, 'wz': float}
etcc.max_correlation          # Maximum correlation achieved
etcc.correlation_with_truth   # {'rho_Rx': float, 'rho_Ry': float, 'rho_Rz': float}

# For SpatialMerging
merged_grid = spatial.merge_gridded(x, y, z, axis=-1)
```

**Input Format:**
- `x, y, z`: 1D arrays for point-wise merging
- For spatial: 3D arrays (lat, lon, time) or (time, lat, lon)

**Returns:**
- `merged`: Merged product with same shape as inputs

**Key Differences:**
- **TC**: Minimizes error variance (RMSE²), analytical solution
- **ETCC**: Maximizes correlation with truth, exhaustive search
- **Use TC when**: RMSE minimization is the goal
- **Use ETCC when**: Correlation maximization is more important (e.g., precipitation)

**Reference:**
> Wei, M., et al. (2023). Ground validation of GPM IMERG precipitation products over Iran. *Geophysical Research Letters*, 50(18).

### BTC (Bayesian Triple Collocation)

```python
from collocation import BayesianTC

# Initialize with data
btc = BayesianTC(data)  # data shape: (n_products, n_samples)

# Run inference
btc.run_inference(niter=2000, nadvi=200000, seed=123)

# Get results
rmse_mean, rmse_std, rmse_quantiles = btc.get_error_estimates()
m_mean, l_mean = btc.get_calibration_parameters()
```

### BTCH (Bayesian Three-Cornered Hat)

```python
from collocation import BayesianTCH

# Initialize with data
btch = BayesianTCH(data)  # data shape: (n_products, n_samples), requires 3 products

# Run inference (faster than BTC)
btch.run_inference(niter=2000, nadvi=50000, seed=123)

# Get results
rmse_mean, rmse_std, rmse_quantiles = btch.get_error_estimates()
snr_mean, snr_std, snr_quantiles = btch.get_snr()
rho2_mean, rho2_std, rho2_quantiles = btch.get_correlation()

# Print detailed summary
btch.summary(verbose=True)

# Plot posterior distributions
fig, axes = btch.plot_posterior()
```

**Parameters (BTC):**
- `data`: Input data (n_products, n_samples) - Three or more products
- `niter`: MCMC iterations (default: 2000)
- `nadvi`: ADVI iterations for initialization (default: 200000)
- `seed`: Random seed

**Returns:**
- `rmse_mean`: Posterior mean of RMSE for each product
- `rmse_std`: Posterior standard deviation of RMSE
- `rmse_quantiles`: 2.5%, 50%, 97.5% quantiles (credible intervals)
- `m_mean`: Additive bias estimates
- `l_mean`: Multiplicative bias estimates

**Advanced Options:**
```python
# Custom inference parameters
btc.setup_model(
    doft=4,                    # Degrees of freedom for priors
    priorfactor=1.0,           # Prior scaling
    thetaoffset=0.15,          # Offset for multiplicative bias
    thetamodel='beta',         # 'beta' or 'logistic'
    studenterrors=False        # Use Student-t errors
)
```

**Key Features:**
- Full Bayesian uncertainty quantification via MCMC
- Time-varying error variance estimation
- Non-constant calibration parameters
- Handles complex heteroscedastic structures
- Can incorporate explanatory variables for time-varying parameters

**Parameters (BTCH):**
- `data`: Input data (3, n_samples) - Exactly three products required
- `niter`: MCMC iterations (default: 2000)
- `nadvi`: ADVI iterations for initialization (default: 50000)
- `seed`: Random seed
- `nchains`: Number of MCMC chains (default: 2)

**Methods:**
- `get_error_estimates()`: Returns RMSE mean, std, and quantiles
- `get_snr()`: Returns SNR mean, std, and quantiles
- `get_correlation()`: Returns correlation mean, std, and quantiles
- `summary(verbose=True)`: Print detailed results
- `plot_posterior()`: Plot posterior distributions

**Key Differences from BTC:**
- BTCH assumes constant error variances (simpler model)
- BTC allows time-varying errors (more complex)
- BTCH is faster (~2-5x) with fewer ADVI iterations needed
- BTCH best for homoscedastic errors
- BTC best for heteroscedastic, time-varying errors

**Reference:**
> Zwieback, S., et al. (2012). Structural and statistical properties of the collocation technique for error characterization. Nonlin. Processes Geophys., 19, 69-80.

## Performance Metrics

The package includes comprehensive evaluation metrics:

```python
from collocation.utils import calculate_all_metrics

metrics = calculate_all_metrics(simulated, observed)
# Returns: r, KGE, NSE, PBIAS, RMSE, MAE
```

### Available Metrics

- **KGE** (Kling-Gupta Efficiency): Decomposes MSE into correlation, bias, and variability
- **NSE** (Nash-Sutcliffe Efficiency): Standard hydrological model performance
- **PBIAS** (Percent Bias): Systematic bias quantification
- **RMSE** (Root Mean Square Error): Overall error magnitude
- **MAE** (Mean Absolute Error): Average error magnitude
- **r** (Correlation): Linear association strength

## Applications

### Remote Sensing
- Soil moisture product validation
- Precipitation dataset comparison
- Ocean wind speed error estimation
- Land surface temperature analysis

### Climate Science
- Model-observation comparison
- Reanalysis dataset evaluation
- Multi-source data fusion

### Geophysics
- Multi-sensor integration
- Error budget estimation
- Data quality assessment

## Mathematical Background

### Triple Collocation Theory

For three products measuring the same geophysical variable:

```
X_i = α_i + β_i θ + ε_i
```

Where:
- `X_i`: Observed product i
- `θ`: True signal (unknown)
- `α_i, β_i`: Calibration parameters
- `ε_i`: Random error

Assumptions:
1. Errors are zero-mean: E[ε_i] = 0
2. Errors are uncorrelated with signal: E[ε_i θ] = 0
3. Errors are mutually independent: E[ε_i ε_j] = 0 (for TC)

The covariance structure provides 6 equations to solve for 3 error variances and 3 signal variances.

### Error Metrics

**Signal-to-Noise Ratio:**
```
SNR_i = (β_i² σ_θ²) / σ_ε_i²
```

**Data-Truth Correlation:**
```
ρ²_i = SNR_i / (1 + SNR_i)
```

**Fractional MSE:**
```
fMSE_i = 1 - ρ²_i = 1 / (1 + SNR_i)
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Citation

If you use this package in your research, please cite:

```bibtex
@software{collocation_analysis,
  title = {Collocation Analysis Package},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/Collocation-Analysis}
}
```

Also cite the relevant methodological papers (see API Documentation section).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original MATLAB implementation by licm_13@163.com
- Converted to Python with enhancements and comprehensive documentation
- Based on foundational work by Stoffelen (1998), Scipal et al. (2008), Dong et al. (2014, 2019), and Gruber et al. (2016)

## References

1. Stoffelen, A. (1998). Toward the true near-surface wind speed: Error modeling and calibration using triple collocation. *Journal of Geophysical Research*, 103(C4), 7755-7766.

2. Scipal, K., Holmes, T., De Jeu, R., Naeimi, V., & Wagner, W. (2008). A possible solution for the problem of estimating the error structure of global soil moisture data sets. *Geophysical Research Letters*, 35(24).

3. Dong, J., Crow, W. T., Reichle, R., & Liu, Q. (2014). Fusing active and passive remotely sensed soil moisture products using information vectors.

4. Dong, J., Crow, W. T., Duan, Z., Wei, L., & Lu, Y. (2019). An instrument variable based algorithm for estimating cross-correlated hydrological remote sensing errors. *Journal of Hydrology*, 581, 124385.

5. Gruber, A., Su, C. H., Zwieback, S., Crow, W., Dorigo, W., & Wagner, W. (2016). Recent advances in (soil moisture) triple collocation analysis. *International Journal of Applied Earth Observation and Geoinformation*, 45, 200-211.

6. Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling. *Journal of Hydrology*, 377(1-2), 80-91.

7. Wei, M., Qiao, B., Zhao, J., & Zuo, X. (2023). Ground validation of GPM IMERG precipitation products over Iran. *Geophysical Research Letters*, 50(18).

## Data Fusion Module

### Overview

The `collocation.fusion` subpackage provides a comprehensive framework for multi-source data fusion using optimal weighting strategies. Supports both simple inverse-variance weighting and advanced GLS/BLUE fusion with full error covariance.

**Key Features:**
- **Multiple fusion modes**: IVW, GLS/BLUE, constrained QP
- **Covariance estimation**: From MSE, TC/IVD, with shrinkage
- **Constraints**: Sum-to-one, bounds, physics (energy balance, ET ≤ PET)
- **Localization**: Moving windows, biome/climate partitioning
- **Robust estimation**: Huber loss, outlier detection
- **Uncertainty**: Analytical variance, bootstrap CI, diagnostics
- **CLI tools**: Command-line interface for ET fusion

### Fusion Algorithms

#### 1. Inverse Variance Weighting (IVW)

Simple diagonal fusion: **w_k ∝ 1/MSE_k**

```python
from collocation.fusion import solve_weights_ivw, estimate_mse

# Estimate MSE from data
mse = estimate_mse(y_pred, y_ref)  # shape: (n_models,)

# Compute IVW weights
weights = solve_weights_ivw(mse, sum_to_one=True)

# Fuse
x_fused = (y_pred * weights).sum(dim='model')
```

**When to use:**
- Errors are independent across models
- Simple, fast, transparent
- Good baseline for comparison

#### 2. Generalized Least Squares (GLS/BLUE)

Full covariance fusion: **w = Σ⁻¹1 / (1ᵀΣ⁻¹1)**

Minimizes fused variance: **Var(x̂) = (1ᵀΣ⁻¹1)⁻¹**

```python
from collocation.fusion import solve_weights_gls, build_sigma

# Build covariance matrix
Sigma = build_sigma(mse, cross_cov, shrinkage="lw")

# Solve for GLS weights
weights = solve_weights_gls(Sigma, sum_to_one=True)

# Variance of fused estimate
var_fused = 1.0 / (np.ones(K) @ np.linalg.inv(Sigma) @ np.ones(K))
```

**When to use:**
- Models have correlated errors
- Full covariance available (from TC, IVD, or sample estimate)
- Optimal BLUE estimator

#### 3. Constrained Quadratic Programming (QP)

Minimize variance with constraints:

**min_w** (1/2) wᵀΣw
**s.t.** 1ᵀw = 1, w_min ≤ w ≤ w_max, Aw ≤ b

```python
from collocation.fusion import solve_weights_qp

# Define constraints
A_eq = np.array([[1, 1, 1]])      # sum-to-one
b_eq = np.array([1.0])
bounds = (0.1, 0.9)               # prevent extreme weights

weights = solve_weights_qp(
    Sigma,
    A_eq=A_eq, b_eq=b_eq,
    bounds=bounds,
    solver="quadprog"
)
```

**When to use:**
- Need to enforce physical constraints
- Prevent degenerate weights
- Energy balance: Ec + Et = E
- Plausibility: 0 ≤ ET ≤ PET

### High-Level API: `fuse_fields`

End-to-end fusion orchestrator:

```python
from collocation.fusion import fuse_fields
import xarray as xr

# Multi-model data
X = xr.DataArray(
    data,  # shape: (time, lat, lon, model)
    dims=["time", "lat", "lon", "model"],
    coords={"model": ["ERA5", "GLEAM", "GLDAS"]}
)

# Reference for MSE estimation
X_ref = xr.DataArray(
    reference_data,  # shape: (time, lat, lon)
    dims=["time", "lat", "lon"]
)

# Fuse with GLS
result = fuse_fields(
    X, X_ref=X_ref,
    mode="gls",              # ivw, gls, or qp
    shrinkage="lw",          # Ledoit-Wolf shrinkage
    robust=False,            # Use Huber loss if True
    return_weights=True,
    return_var=True,
    return_diagnostics=True
)

# Access results
fused_ET = result["fused"]
weights = result["weights"]
variance = result["variance"]
diagnostics = result["diagnostics"]
```

### Covariance Estimation

#### From MSE

```python
from collocation.fusion import estimate_mse, build_sigma

# Estimate MSE per model
mse = estimate_mse(y_pred, y_ref, robust=False)

# Build diagonal covariance
Sigma_diag = build_sigma(mse, cross=None)

# Or estimate cross-covariance via Triple Collocation
from collocation.fusion import estimate_cross_covariance

cross_cov = estimate_cross_covariance(
    y_pred, y_ref, method="tc"  # or "ivd", "sample"
)
Sigma_full = build_sigma(mse, cross=cross_cov, shrinkage="lw")
```

#### Shrinkage Methods

Regularize ill-conditioned covariance:

| Method | Description | When to use |
|--------|-------------|-------------|
| **Ledoit-Wolf (lw)** | Shrink toward scaled identity | Small samples, default choice |
| **OAS** | Oracle Approximating Shrinkage | Similar to LW |
| **Ridge** | Diagonal loading Σ + λI | Explicit regularization |
| **none** | No shrinkage | Large samples, well-conditioned |

```python
Sigma = build_sigma(mse, shrinkage="lw", lam=0.5)
```

#### Covariance Tapering

Reduce spurious long-range correlations:

```python
from collocation.fusion import apply_covariance_tapering

# Compute distances
distance = compute_distance_matrix(lat, lon, metric="haversine")

# Apply Gaspari-Cohn taper
Sigma_tapered = apply_covariance_tapering(
    Sigma, distance,
    cutoff_radius=500,  # km
    taper_function="gaspari_cohn"
)
```

### Localization

Spatially/temporally varying weights:

#### Moving Windows

```python
from collocation.fusion import apply_moving_window

# Local MSE estimation
window_spec = {"time": 30, "space": 7}  # 30 days, 7×7 pixels
mse_local = estimate_mse(X, X_ref, window=window_spec)

# Fuse with local weights
result = fuse_fields(
    X, X_ref=X_ref,
    localization={"strategy": "window", "window": window_spec}
)
```

#### Biome Partitioning

```python
from collocation.fusion import partition_by_biome

# Separate weights per ecosystem
biome_map = xr.DataArray(biome_ids, dims=["lat", "lon"])
partitions = partition_by_biome(X, biome_map)

# Fuse each biome separately
for biome_id, X_biome in partitions.items():
    result = fuse_fields(X_biome, X_ref=X_ref)
```

### Constraints

#### Built-in Constraints

```python
from collocation.fusion import (
    SumToOneConstraint,
    BoundsConstraint,
    EnergyBalanceConstraint,
    ETConstraint,
)

# Sum-to-one
c1 = SumToOneConstraint(n_models=3)

# Bounds
c2 = BoundsConstraint(n_models=3, w_min=0.0, w_max=1.0)

# Energy balance: Ec + Et ≈ E
c3 = EnergyBalanceConstraint(
    ec_index=0, et_index=1, e_total_index=2, tolerance=0.1
)

# ET plausibility: 0 ≤ ET ≤ PET
c4 = ETConstraint(et_index=0, pet_values=PET)
```

#### Using Constraints in QP

```python
constraints = {
    "sum_to_one": True,
    "bounds": (0.0, 1.0),
}

result = fuse_fields(X, X_ref=X_ref, mode="qp", constraints=constraints)
```

### Robust Estimation

Handle outliers with Huber loss:

```python
from collocation.fusion import estimate_mse_robust, detect_outliers

# Robust MSE (downweights outliers)
mse_robust = estimate_mse_robust(y_pred, y_ref, loss="huber")

# Detect outliers
outlier_mask = detect_outliers(data, method="iqr")

# Robust fusion
result = fuse_fields(X, X_ref=X_ref, mode="ivw", robust=True)
```

### Uncertainty Quantification

#### Analytical Variance

```python
from collocation.fusion import propagate_variance

# Fused variance: Var(x̂) = wᵀΣw
var_fused = propagate_variance(Sigma, weights)
std_fused = np.sqrt(var_fused)
```

#### Bootstrap Confidence Intervals

```python
from collocation.fusion import bootstrap_uncertainty

ci = bootstrap_uncertainty(
    data, weights,
    n_boot=1000,
    alpha=0.05  # 95% CI
)

print(f"95% CI: [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")
```

#### Diagnostics

```python
from collocation.fusion import compute_diagnostics

diag = compute_diagnostics(weights, Sigma)

# Effective sample size (1 ≤ N_eff ≤ K)
print(f"Effective N: {diag['effective_n']:.2f}")

# Entropy (high = diverse, low = concentrated)
print(f"Entropy: {diag['entropy']:.3f}")

# Concentration (HHI index)
print(f"Concentration: {diag['concentration']:.3f}")
```

### Command-Line Interface

Fuse ET products from NetCDF files:

```bash
# Simple IVW
python scripts/fuse_et.py \
    --inputs ERA5.nc GLEAM.nc GLDAS.nc \
    --var ET \
    --mode ivw \
    --output fused_et.nc

# GLS with Ledoit-Wolf shrinkage
python scripts/fuse_et.py \
    --inputs *.nc \
    --var ET \
    --mode gls \
    --shrinkage lw \
    --mse-window time=30 \
    --output fused_et.nc \
    --weights weights.nc \
    --diagnostics

# Constrained QP with bounds
python scripts/fuse_et.py \
    --inputs *.nc \
    --var ET \
    --mode qp \
    --bounds 0.1 0.9 \
    --output fused_et.nc

# Robust fusion (outlier handling)
python scripts/fuse_et.py \
    --inputs *.nc \
    --var ET \
    --mode ivw \
    --robust \
    --output fused_et.nc
```

### Examples

Run comprehensive examples:

```bash
cd examples
python example_fusion.py
```

**Outputs:**
- Example 1: Simple IVW fusion
- Example 2: GLS with full covariance
- Example 3: Constrained QP
- Example 4: Robust fusion with outliers
- Diagnostic plots (600 dpi, journal-quality)

### Mathematical Background

#### IVW Weights

For independent errors with variances σ²_k:

**w_k = (1/σ²_k) / Σ_j(1/σ²_j)**

Fused variance: **Var(x̂) = 1 / Σ_k(1/σ²_k)**

#### GLS/BLUE Weights

For error covariance Σ:

**w = Σ⁻¹1 / (1ᵀΣ⁻¹1)**

Properties:
- Minimum variance unbiased estimator
- **Var(x̂) = (1ᵀΣ⁻¹1)⁻¹**
- Reduces to IVW if Σ is diagonal

#### Constrained QP

**min_w** (1/2) wᵀΣw - aᵀw
**s.t.** A_eq w = b_eq (equality)
       A_ineq w ≤ b_ineq (inequality)
       w_min ≤ w ≤ w_max (bounds)

Solved via quadprog, cvxopt, or OSQP.

### Performance Notes

- **IVW**: O(K) per pixel (diagonal only)
- **GLS**: O(K³) per pixel (Cholesky decomposition)
- **QP**: O(K³) per pixel + QP iterations
- **Localization**: Compute chunks in parallel with Dask

For large grids, use:
```python
result = fuse_fields(X.chunk({"time": 50, "lat": 20, "lon": 20}), ...)
```

### API Reference

#### Core Functions

- `fuse_fields()`: High-level fusion orchestrator
- `solve_weights_ivw()`: Inverse variance weights
- `solve_weights_gls()`: GLS/BLUE weights
- `solve_weights_qp()`: Constrained QP weights
- `estimate_mse()`: Mean squared error estimation
- `build_sigma()`: Covariance matrix construction
- `propagate_variance()`: Analytical variance propagation

#### Modules

- `fusion.weights`: Weight solvers
- `fusion.covariance`: Covariance estimation
- `fusion.constraints`: Constraint classes
- `fusion.robust`: Robust estimators
- `fusion.localization`: Spatial/temporal localization
- `fusion.uncertainty`: Uncertainty quantification
- `fusion.fuse`: High-level API

### Testing

Run fusion tests:

```bash
pytest tests/test_fusion.py -v
```

**Test coverage:**
- Synthetic data with known covariance
- IVW, GLS, QP solvers
- Constraint satisfaction
- Robust estimation with outliers
- Heteroscedastic errors
- Missing data handling
- Variance propagation

### References

1. **Ledoit, O., & Wolf, M. (2004).** A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411.

2. **Gaspari, G., & Cohn, S. E. (1999).** Construction of correlation functions in two and three dimensions. *Quarterly Journal of the Royal Meteorological Society*, 125(554), 723-757.

3. **Gruber, A., et al. (2016).** Recent advances in (soil moisture) triple collocation analysis. *International Journal of Applied Earth Observation and Geoinformation*, 45, 200-211.

4. **Yilmaz, M. T., & Crow, W. T. (2014).** Evaluation of assumptions in soil moisture triple collocation analysis. *Journal of Hydrometeorology*, 15(3), 1293-1302.

## Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Email: your.email@example.com

---

**Note:** This package is actively maintained and continuously improved. Check the repository for the latest updates.
