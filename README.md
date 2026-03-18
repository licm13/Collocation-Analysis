# Collocation Analysis Package

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![Version](https://img.shields.io/badge/version-2.0.0-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

A Python library for **collocation-based error analysis** of remote sensing and geophysical datasets — quantifying errors in multiple products *without ground truth*.

> For the full architecture guide (bilingual 中/EN), see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## What's New in v2.0

| Feature | Description |
|---------|-------------|
| **Sklearn-style API** | `TC().fit(data).metrics_` — uniform interface across all methods |
| **Smart Recommender** | `CollocationConsultant` diagnoses your data and suggests the best method |
| **Interactive Dashboard** | Plotly 3-panel HTML dashboard with time-series slider and stability heatmap |
| **ELI Pipeline** | One-click parallel ELI analysis with auto-generated HTML report |

---

## Installation

```bash
git clone https://github.com/licm13/Collocation-Analysis.git
cd Collocation-Analysis
pip install -e .

# Optional: interactive dashboard
pip install plotly

# Optional: xarray support
pip install xarray

# Optional: Bayesian methods
pip install "pymc3==3.11.5" theano-pymc
```

---

## Quick Start

### 1. Let the consultant choose your method

```python
import numpy as np
from collocation import CollocationConsultant

data = np.column_stack([product1, product2, product3])  # (n, 3)
report = CollocationConsultant(data, product_names=['ERA5', 'GLEAM', 'GLDAS']).consult()
print(report)
# ★ PRIMARY RECOMMENDATION  →  EIVD
#   Reason: High cross-correlation detected (GLEAM×GLDAS r=0.54)...
```

### 2. Fit with the sklearn-style API

```python
from collocation.estimators import TC, EIVD, IVD

# Fit and inspect
model = TC().fit(data)
print(model.metrics_['error_std'])   # [0.20, 0.31, 0.39]
print(model.summary())

# Compare methods in a loop
results = {name: cls().fit(data).metrics_
           for name, cls in [('TC', TC), ('EIVD', EIVD)]}
```

### 3. Interactive dashboard

```python
from collocation import InteractiveDashboard
from collocation.estimators import TC, EIVD
from collocation import tc

metrics = {'TC': TC().fit(data).metrics_, 'EIVD': EIVD().fit(data).metrics_}
InteractiveDashboard(data, metrics, method_fn=tc).save('dashboard.html')
```

### 4. ELI one-click pipeline

```python
from collocation import ELIPipeline

result = ELIPipeline(
    water  = np.column_stack([swvl1, gleam_sm, gldas_sm]),
    energy = np.column_stack([swd, era5_rn, gldas_rn]),
    vegetation = et_obs,
    product_names={
        'water':      ['ERA5-SM', 'GLEAM-SM', 'GLDAS-SM'],
        'energy':     ['ERA5-SW', 'ERA5-Rn',  'GLDAS-Rn'],
        'vegetation': ['GLEAM-ET'],
    },
).run()

print(result.summary())       # ELI ratio, per-method table
result.save('eli_report.html')
```

### 5. Classic functional API (unchanged)

```python
from collocation import tc, eivd, ivd, ec, btch_he2020

EeeT, SNR, rho2, fMSE = tc(data)                      # 3-way
EeeT, SNR, rho2, fMSE, L = eivd(data)                 # 3-way + cross-corr
EeeT, rho2, weights = ivd(np.column_stack([p1, p2]))   # 2-way
variances, weights, fused = btch_he2020(data)          # analytical BTCH
```

---

## Implemented Methods

### Classical

| Method | Products | Key Use Case |
|--------|----------|-------------|
| `IVD` | 2 | Active + passive sensor fusion |
| `IVS` | 2 | Two-product with bootstrap CI |
| `TC` / `TCH` | 3 | Standard triple collocation |
| `EIVD` | 3 | Correlated-error products |
| `EC` | 4 | Quadruple over-determination |
| `ETCC` | 3 | Correlation-optimised merging (precipitation) |
| `MTCH` | N≥3 | Multiplicative / positive-skew data |

### Bayesian

| Method | Uncertainty | Dependencies |
|--------|-------------|-------------|
| `BTCH_He2020` | None (analytical) | NumPy only |
| `BayesianTCH` | Full MCMC | PyMC3 |
| `BayesianTC` | Full MCMC + time-varying | PyMC3 |

---

## Module Map

```
collocation/
├── base.py           ← CollocationEstimator ABC
├── estimators.py     ← TC / EIVD / IVD / EC sklearn wrappers
├── consultant.py     ← CollocationConsultant smart recommender
├── plotting.py       ← plot_error_comparison, InteractiveDashboard
├── eli_pipeline.py   ← ELIPipeline, ELIResult
│
├── tc.py, eivd.py, ivd.py, ivs.py, ec.py   ← classical methods
├── etcc.py, mtch.py                          ← advanced methods
├── btch_he2020.py, bayesian_tc.py, bayesian_tch.py
│
├── utils.py, covariance.py, fuse.py, simple_average.py, eli.py
└── fusion/           ← IVW / GLS / QP weights, robust, localization
```

---

## Testing

```bash
pip install pytest
pytest tests/ -v

# New v2.0 tests only
pytest tests/test_upgrade.py -v   # 48 tests
```

---

## Dependencies

| Package | Role | Required? |
|---------|------|-----------|
| numpy ≥ 1.18 | Core numerics | **Yes** |
| scipy ≥ 1.4 | Statistics, linear algebra | **Yes** |
| matplotlib ≥ 3.1 | Static plots | Optional |
| plotly ≥ 5.0 | Interactive dashboard | Optional |
| xarray ≥ 0.18 | Native DataArray/Dataset input | Optional |
| pymc3 == 3.11.5 | Bayesian TC/TCH | Optional |

---

## Documentation

| File | Content |
|------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Full bilingual (EN/中) code architecture guide |
| [ELI_README.md](ELI_README.md) | Ecosystem Limitation Index application guide |
| [BAYESIAN_INTEGRATION_GUIDE.md](BAYESIAN_INTEGRATION_GUIDE.md) | PyMC3 setup and Bayesian workflow |
| [PERFORMANCE_SUMMARY.md](PERFORMANCE_SUMMARY.md) | Optimisation history (98% speedups) |
| [README_CN.md](README_CN.md) | 中文说明 |
| [CLAUDE.md](CLAUDE.md) | AI-assistant developer guide |

---

## Citation

If you use this package, please cite the underlying methods relevant to your analysis. Key references:

- **TC**: Stoffelen (1998), *JGR*, 103(C4)
- **EIVD**: Dong et al. (2019), *J. Hydrology*, 581
- **BTCH_He2020**: He et al. (2020), *HESS*
- **ETCC**: Wei et al. (2023)
- **ELI**: Dong et al. (2022), *Remote Sens. Environ.*

---

## License

MIT © Original MATLAB: licm_13@163.com | Python conversion & v2.0 upgrade: Claude
