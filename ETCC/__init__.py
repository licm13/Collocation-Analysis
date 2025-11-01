"""
ETCC: Extended Triple Collocation for Precipitation Fusion

Python implementation of the Extended Triple Collocation method with 
maximized Correlation (ETCC) for merging multi-source precipitation products.

Reference:
Wei, L., Jiang, S., Ren, L., Yuan, S., Liu, Y., Yang, X., et al. (2023). 
An extended triple collocation method with maximized correlation for near 
global-land precipitation fusion. Geophysical Research Letters, 50, 
e2023GL105120. https://doi.org/10.1029/2023GL105120
"""

__version__ = '1.0.0'
__author__ = 'Based on Wei et al. (2023)'
__all__ = ['TripleCollocation', 'ETCC', 'SpatialMerging',
           'calculate_metrics', 'compare_products',
           'generate_synthetic_data', 'plot_comparison']

from .merging import TripleCollocation, ETCC, SpatialMerging
from .evaluation import (calculate_cc, calculate_rmse, calculate_mae,
                        calculate_bias, calculate_metrics, compare_products,
                        spatial_metrics, print_comparison_table,
                        statistical_significance, CrossValidation)
from .utils import (generate_synthetic_data, load_precipitation_data,
                   save_to_netcdf, plot_comparison, plot_spatial_metric,
                   plot_scatter, plot_weights_comparison,
                   create_boxplot_comparison)
