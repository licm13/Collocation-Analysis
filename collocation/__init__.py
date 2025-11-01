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
- EIVD: Extended Information Vector Dual (3-way, allows one error cross-correlation)
- ETCC: Extended Triple Collocation (3-way, allows full error cross-correlation)
- EC: Extended Collocation (4-way quadruple collocation)

Simple Methods:
- SimpleAverage: Simple and weighted averaging for quick data fusion

Bayesian Methods:
- BTC: Bayesian Triple Collocation (3-way, time-varying errors, full uncertainty)
- BTCH: Bayesian Three-Cornered Hat (3-way, constant errors, full uncertainty)

Author: Converted from MATLAB by Claude
Original MATLAB code: licm_13@163.com
"""

from .ivd import ivd
from .ivs import ivs
from .tc import tc
from .eivd import eivd
from .etcc import etcc, compare_methods as compare_tc_eivd_etcc
from .ec import ec
from .utils import mse_judge, kge_objfun

# Alias tc as tch (classic Three-Cornered Hat is mathematically TC)
from .tc import tc as tch

# Simple averaging methods
from .simple_average import (
    simple_average,
    inverse_variance_weights,
    calculate_averaging_uncertainty,
    ensemble_statistics
)

# ELI (Ecosystem Limitation Index) module
from .eli import ELIProcessor, calculate_eli_index, process_eli_data

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


__version__ = '1.4.0' # Incremented version for ETCC addition

__all__ = [
    # Classical methods
    'ivd',
    'ivs',
    'tc',
    'tch', # Added TCH alias
    'eivd',
    'etcc', # Added ETCC
    'compare_tc_eivd_etcc', # Added ETCC comparison
    'ec',
    # Simple methods
    'simple_average',
    'inverse_variance_weights',
    'calculate_averaging_uncertainty',
    'ensemble_statistics',
    # ELI application
    'ELIProcessor',
    'calculate_eli_index',
    'process_eli_data',
    # Utilities
    'mse_judge',
    'kge_objfun',
    # Bayesian methods
    'BayesianTC',
    'bayesian_tc',
    'simulate_products',
    'BAYESIAN_AVAILABLE',
    'BayesianTCH', # Added BTCH
    'BAYESIAN_TCH_AVAILABLE', # Added BTCH flag
]