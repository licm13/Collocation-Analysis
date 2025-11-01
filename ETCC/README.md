# Extended Triple Collocation for Precipitation Fusion (ETCC)

Python implementation of the Extended Triple Collocation method with maximized Correlation (ETCC) for merging multi-source precipitation products, based on Wei et al. (2023).

## Reference
Wei, L., Jiang, S., Ren, L., Yuan, S., Liu, Y., Yang, X., et al. (2023). An extended triple collocation method with maximized correlation for near global-land precipitation fusion. *Geophysical Research Letters*, 50, e2023GL105120. https://doi.org/10.1029/2023GL105120

## Overview

This repository implements two precipitation merging methods:

1. **Triple Collocation (TC)**: Minimizes error variance using RMSE-based weights
2. **Extended Triple Collocation for maximized Correlation (ETCC)**: Maximizes correlation with unknown truth using exhaustive search

## Key Features

- Complete implementation of TC and ETCC algorithms
- Evaluation metrics (Correlation Coefficient, RMSE)
- Spatial analysis and visualization tools
- Support for gridded precipitation data
- Comprehensive example notebooks

## Installation

```bash
git clone https://github.com/yourusername/etcc_precipitation.git
cd etcc_precipitation
pip install -r requirements.txt
```

## Quick Start

```python
from etcc.merging import TripleCollocation, ETCC
import numpy as np

# Load three independent precipitation products
x = np.load('data/product_x.npy')  # IMERG-E
y = np.load('data/product_y.npy')  # SM2RAIN-ASCAT
z = np.load('data/product_z.npy')  # ERA5

# Traditional TC merging
tc = TripleCollocation()
merged_tc = tc.merge(x, y, z)

# ETCC merging (maximized correlation)
etcc = ETCC(weight_increment=0.01)
merged_etcc = etcc.merge(x, y, z)

# Evaluate against reference data
from etcc.evaluation import calculate_metrics
reference = np.load('data/reference.npy')

metrics_tc = calculate_metrics(merged_tc, reference)
metrics_etcc = calculate_metrics(merged_etcc, reference)

print(f"TC - CC: {metrics_tc['cc']:.3f}, RMSE: {metrics_tc['rmse']:.3f}")
print(f"ETCC - CC: {metrics_etcc['cc']:.3f}, RMSE: {metrics_etcc['rmse']:.3f}")
```

## Repository Structure

```
etcc_precipitation/
├── etcc/
│   ├── __init__.py
│   ├── merging.py          # TC and ETCC implementations
│   ├── evaluation.py       # Metrics and validation
│   └── utils.py            # Helper functions
├── examples/
│   ├── basic_merging.py
│   ├── spatial_analysis.ipynb
│   └── comparison_tc_etcc.ipynb
├── tests/
│   ├── test_merging.py
│   └── test_evaluation.py
├── data/
│   └── sample_data.py      # Generate sample data
├── requirements.txt
├── setup.py
└── README.md
```

## Method Overview

### Triple Collocation (TC)

Minimizes error variance by:
1. Computing covariances between product pairs
2. Estimating RMSE for each product
3. Calculating optimal weights: wⱼ = (σᵢ²σₖ²) / (σₓ²σᵧ² + σₓ²σᵤ² + σᵧ²σᵤ²)

### ETCC Method

Maximizes correlation by:
1. Computing pairwise correlations (ρxy, ρxz, ρyz)
2. Estimating correlation with truth using extended TC
3. Building unique correlation function ρRM(wx,wy,wz)
4. Exhaustive search to find weights that maximize ρRM

## Applications

- Global precipitation merging
- Regional hydrometeorological studies
- Gauge-sparse area analysis
- Multi-source data fusion

## Citation

If you use this code, please cite:
```bibtex
@article{wei2023extended,
  title={An extended triple collocation method with maximized correlation for near global-land precipitation fusion},
  author={Wei, Linyong and Jiang, Shanhu and Ren, Liliang and Yuan, Shanshui and Liu, Yi and Yang, Xiaoli and Wang, Menghao and Zhang, Linqi and Yu, Huafei and Duan, Zheng},
  journal={Geophysical Research Letters},
  volume={50},
  number={24},
  pages={e2023GL105120},
  year={2023},
  publisher={Wiley Online Library}
}
```

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue.
