"""
Collocation Analysis Package
=============================

A comprehensive Python package for collocation-based error analysis
of remote sensing and geophysical datasets.

This package provides implementations of various collocation methods:

Classical Methods:
- IVD: Information Vector Dual (2-way collocation)
- IVS: Information Vector with Scaling (2-way with bootstrap)
- TC: Triple Collocation (3-way, assumes zero error cross-correlation)
- TCH: Three-Cornered Hat (Classic, mathematically equivalent to TC)
- EIVD: Extended Information Vector Dual (3-way, allows error cross-correlation)
- EC: Extended Collocation (4-way quadruple collocation)
- ETCC: Extended Triple Collocation for maximized Correlation (3-way, optimizes correlation)
- MTCH: Multiplicative-error Three-Cornered Hat (N-way, multiplicative error model)

Simple Methods:
- SimpleAverage: Simple and weighted averaging for quick data fusion

Bayesian Methods:
- BTC: Bayesian Triple Collocation (3-way, time-varying errors, full uncertainty)
- BTCH: Bayesian Three-Cornered Hat (3-way, constant errors, full uncertainty)
- BTCH_He2020: BTCH method by He et al. (2020) (3-way, analytical weights, no MCMC)

Data Fusion Module (NEW):
- fusion: Comprehensive data fusion framework with IVW, GLS/BLUE, constrained QP
  Import via: from collocation.fusion import fuse_fields, solve_weights_gls, etc.
  See documentation for full API reference

Author: Converted from MATLAB by Claude
Original MATLAB code: licm_13@163.com
"""

from .ivd import ivd
from .ivs import ivs
from .tc import tc
from .eivd import eivd
from .ec import ec
from .utils import mse_judge, kge_objfun
XR_AVAILABLE = True

try:
    from .covariance import build_sigma_from_collocation
except ModuleNotFoundError:
    build_sigma_from_collocation = None
    XR_AVAILABLE = False

try:
    from .fuse import estimate_bias_from_collocation
except ModuleNotFoundError:
    estimate_bias_from_collocation = None
    XR_AVAILABLE = False

# Alias tc as tch (classic Three-Cornered Hat is mathematically TC)
from .tc import tc as tch

# ETCC methods
from .etcc import TripleCollocation, ETCC, SpatialMerging

# Simple averaging methods
from .simple_average import (
    simple_average,
    inverse_variance_weights,
    calculate_averaging_uncertainty,
    ensemble_statistics
)

# ELI (Ecosystem Limitation Index) module (optional xarray dependency)
try:
    from .eli import ELIProcessor, calculate_eli_index, process_eli_data
    ELI_AVAILABLE = True
except ModuleNotFoundError:
    ELIProcessor = None
    calculate_eli_index = None
    process_eli_data = None
    ELI_AVAILABLE = False

# Bayesian methods (optional, requires PyMC3)
try:
    from .bayesian_tc import BayesianTC, bayesian_tc, simulate_products
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    BayesianTC = None
    bayesian_tc = None
    simulate_products = None

# New Bayesian TCH (optional, requires PyMC3)
try:
    # This check ensures we don't fail if pymc3 wasn't imported
    # in bayesian_tch.py itself
    from .bayesian_tch import BayesianTCH, PYMC3_AVAILABLE as BTCH_PYMC3_OK
    if BTCH_PYMC3_OK:
        BAYESIAN_TCH_AVAILABLE = True
    else:
        BAYESIAN_TCH_AVAILABLE = False
        BayesianTCH = None
except ImportError:
    BAYESIAN_TCH_AVAILABLE = False
    BayesianTCH = None

# BTCH He et al. 2020 (no PyMC3 required, analytical method)
from .btch_he2020 import BTCH_He2020, btch_he2020

# MTCH – Multiplicative-error Three-Cornered Hat (Xu et al. 2019 / Ferreira et al. 2016)
from .mtch import mtch, MTCH


__version__ = '1.5.0'  # Incremented version for MTCH integration

__all__ = [
    # Classical methods
    'ivd',
    'ivs',
    'tc',
    'tch', # Added TCH alias
    'eivd',
    'ec',
    # ETCC methods
    'TripleCollocation',
    'ETCC',
    'SpatialMerging',
    # Simple methods
    'simple_average',
    'inverse_variance_weights',
    'calculate_averaging_uncertainty',
    'ensemble_statistics',
    # ELI application
    'ELIProcessor',
    'calculate_eli_index',
    'process_eli_data',
    'ELI_AVAILABLE',
    # Utilities
    'mse_judge',
    'kge_objfun',
    'build_sigma_from_collocation',
    'XR_AVAILABLE',
    'estimate_bias_from_collocation',
    # Bayesian methods
    'BayesianTC',
    'bayesian_tc',
    'simulate_products',
    'BAYESIAN_AVAILABLE',
    'BayesianTCH', # Added BTCH
    'BAYESIAN_TCH_AVAILABLE', # Added BTCH flag
    'BTCH_He2020', # BTCH He et al. 2020
    'btch_he2020', # Convenience function
    # MTCH – multiplicative-error TCH
    'mtch',
    'MTCH',
]