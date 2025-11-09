"""
HCC-ET Framework: Hierarchical Cross-Calibration for ET Decomposition
======================================================================

A production-ready framework for quantifying global "productive" (T) vs
"non-productive" (E) water flux reshaping under climate change and human
land management (1950-2025, 1km resolution).

Target Publication: Nature Climate Change

Key Features:
- Fuses T/ET ratio (transpiration efficiency) using EIVD/GLS
- Handles Error Cross-Correlation (ECC) from shared 1km predictors
- Multi-scale physical constraints (SapFlux + Water Balance)
- Statistical downscaling with Random Forest
- Comprehensive uncertainty quantification

Author: [Your Name]
Date: 2025-11-09
Version: 1.0.0
"""

__version__ = '1.0.0'
__author__ = '[Your Name]'
__email__ = '[your.email@example.com]'

# Import main classes and functions
from .hcc_framework import HCCETFramework, main
from .config import load_config, save_config, create_default_config
from . import data_loader
from . import downscaler
from . import physical_constraints
from . import utils_hcc

__all__ = [
    'HCCETFramework',
    'main',
    'load_config',
    'save_config',
    'create_default_config',
    'data_loader',
    'downscaler',
    'physical_constraints',
    'utils_hcc',
]
