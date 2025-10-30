# Bayesian Triple Collocation Integration Guide

This guide explains the integration of Bayesian Triple Collocation methods and the comprehensive comparison framework.

## What's New

### 1. Bayesian Triple Collocation (BTC)

A new Bayesian approach to triple collocation that provides:
- **Full uncertainty quantification** through MCMC sampling
- **Time-varying error structures** (heteroscedastic models)
- **Credible intervals** for all estimates (95% CI)
- **Complex error modeling** with non-constant calibration parameters

**Location**: `collocation/bayesian_tc.py`

### 2. Comprehensive Comparison Example

A publication-ready comparison framework featuring:
- **6 challenging scenarios**: Ideal, correlated errors, time-varying, biased, heavy-tailed, realistic
- **All methods compared**: IVD, IVS, TC, EIVD, EC, BTC
- **Nature/Science quality figures**: 300 DPI, colorblind-friendly, proper typography
- **Statistical analysis**: RMSE, correlation, relative errors, distributions

**Location**: `examples/comprehensive_comparison.py`

## Installation

### Standard Installation (Classical methods only)

```bash
cd Collocation-Analysis
pip install numpy scipy matplotlib
```

### With Bayesian Support

```bash
cd Collocation-Analysis
pip install numpy scipy matplotlib
pip install "pymc3>=3.11.0" "theano-pymc"
```

**Note**: PyMC3 installation can be complex. See troubleshooting below if issues arise.

## Quick Start

### Classical Methods (No additional dependencies)

```python
import numpy as np
from collocation import ivd, ivs, tc, eivd, ec

# Generate synthetic data
truth = np.sin(np.linspace(0, 4*np.pi, 500)) * 0.15 + 0.2
product1 = truth + np.random.normal(0, 0.02, 500)
product2 = truth + np.random.normal(0, 0.03, 500)
product3 = truth + np.random.normal(0, 0.04, 500)

# Triple Collocation
tri = np.column_stack([product1, product2, product3])
EeeT, SNR, rho2, fMSE = tc(tri)
print("TC RMSE:", np.sqrt(np.diag(EeeT)))
```

### Bayesian Triple Collocation

```python
from collocation import BayesianTC, BAYESIAN_AVAILABLE

if BAYESIAN_AVAILABLE:
    # Prepare data (n_products, n_samples)
    data = np.array([product1, product2, product3])

    # Initialize and run inference
    btc = BayesianTC(data)
    btc.run_inference(niter=2000, nadvi=200000, seed=42)

    # Get results with uncertainty
    rmse_mean, rmse_std, rmse_quantiles = btc.get_error_estimates()

    print("BTC RMSE (mean ± std):")
    for i in range(3):
        print(f"  Product {i+1}: {rmse_mean[i]:.4f} ± {rmse_std[i]:.4f}")
        print(f"    95% CI: [{rmse_quantiles[i,0]:.4f}, {rmse_quantiles[i,2]:.4f}]")
else:
    print("PyMC3 not available. Install with: pip install pymc3==3.11.5 theano-pymc")
```

### Comprehensive Comparison

```bash
cd examples

# Quick test (fast, no figures)
python quick_comprehensive_test.py

# Full comparison with publication-quality figures
python comprehensive_comparison.py
```

**Output**:
- Individual scenario comparison figures (PNG, 300 DPI)
- Overall performance comparison across scenarios
- Detailed results table with all metrics

## Figure Examples

The comprehensive comparison generates publication-quality figures with:

### Layout
- **Top panel**: Time series showing all products and truth
- **Middle panels**: RMSE estimates and correlations by method
- **Bottom panels**: Error distributions and relative errors

### Style (Nature/Science standard)
- **Resolution**: 300 DPI (suitable for publication)
- **Colors**: Colorblind-friendly palette (Wong 2011)
- **Typography**: Arial/Helvetica, 7-9pt
- **Dimensions**:
  - Single column: 89mm (3.5 inches)
  - Double column: 183mm (7.2 inches)

## Scenarios Explained

### 1. Ideal Case
- Independent errors
- Constant variance
- Zero cross-correlation
- **Purpose**: Baseline performance

### 2. Correlated Errors
- Products 2-3 share common error source
- Tests methods' ability to handle error correlation
- **Challenge**: TC assumes independence (may fail)

### 3. Time-Varying Errors
- Heteroscedastic error structure
- Seasonal patterns in error variance
- **Challenge**: Classical methods assume constant variance

### 4. Systematic Biases
- Strong additive biases (up to 0.05)
- Strong multiplicative biases (0.8 to 1.2)
- **Challenge**: Tests calibration parameter estimation

### 5. Heavy-Tailed Errors
- Student-t errors (df=3) with outliers
- Non-Gaussian distributions
- **Challenge**: Methods assume Gaussian errors

### 6. Realistic Case
- Combined challenges from all above
- Most representative of real-world data
- **Best test**: Overall method robustness

## Performance Interpretation

### Relative Error Threshold
- **< 10%**: Excellent performance
- **10-20%**: Good performance
- **20-50%**: Acceptable for some applications
- **> 50%**: Poor performance

### When to Use Each Method

| Scenario | Recommended Method |
|----------|-------------------|
| 2 products only | IVD or IVS |
| Need uncertainty quantification | IVS or BTC |
| Standard 3-way, independent errors | TC |
| Suspected error correlation | EIVD or BTC |
| 4+ products available | EC |
| Time-varying errors | BTC |
| Complex error structures | BTC |
| Quick analysis | TC or EIVD |
| Full uncertainty needed | BTC |

## Computational Considerations

### Speed Comparison (500 samples)

| Method | Time | Memory |
|--------|------|--------|
| IVD | < 1 sec | Low |
| IVS | ~10 sec (100 bootstrap) | Low |
| TC | < 1 sec | Low |
| EIVD | ~1 sec | Low |
| EC | ~2 sec | Medium |
| BTC | ~5-10 min (2000 MCMC) | High |

### Recommendations
- **Exploratory analysis**: Use classical methods (TC, EIVD) first
- **Final analysis**: Use BTC for detailed uncertainty quantification
- **Large datasets**: Consider subsampling for BTC
- **Multiple scenarios**: Run BTC in parallel across scenarios

## Troubleshooting

### PyMC3 Installation Issues

**Problem**: PyMC3 installation fails

**Solutions**:
```bash
# Try specific versions
pip install pymc3==3.11.5 theano-pymc==1.1.2

# Or use conda
conda install -c conda-forge pymc3

# Check compatibility
python -c "import pymc3; print(pymc3.__version__)"
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'collocation'`

**Solution**:
```python
import sys
sys.path.insert(0, '/path/to/Collocation-Analysis')
from collocation import tc
```

### Memory Issues with BTC

**Problem**: Out of memory during MCMC

**Solutions**:
```python
# Reduce MCMC iterations
btc.run_inference(niter=1000, nadvi=50000)

# Use fewer chains
btc.run_inference(nchains=1)

# Subsample data
data_subset = data[:, ::2]  # Every 2nd sample
```

### Figure Display Issues

**Problem**: Figures don't display

**Solution**:
```python
import matplotlib
matplotlib.use('Agg')  # For headless environments
```

## Citation

If you use this package in your research, please cite:

```bibtex
@software{collocation_analysis_bayesian,
  title = {Collocation Analysis Package with Bayesian Methods},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/licm13/Collocation-Analysis}
}
```

**Key references**:
1. Stoffelen (1998) - Triple Collocation
2. Gruber et al. (2016) - Extended Collocation
3. Zwieback et al. (2012) - Bayesian Triple Collocation

## Support

For issues, questions, or contributions:
- **GitHub Issues**: https://github.com/licm13/Collocation-Analysis/issues
- **Documentation**: See README.md for detailed API reference

## Future Enhancements

Planned features:
- [ ] GPU acceleration for BTC
- [ ] Additional Bayesian models (e.g., spatial correlation)
- [ ] Interactive visualization dashboard
- [ ] Automated report generation
- [ ] Support for more than 4 products in classical methods

## Acknowledgments

- Original MATLAB implementation: licm_13@163.com
- BayesianTripleCollocation: Simon Zwieback (https://github.com/szwieback)
- Python conversion and integration: Claude Code

---

**Last updated**: 2024-10-30
**Version**: 1.1.0
