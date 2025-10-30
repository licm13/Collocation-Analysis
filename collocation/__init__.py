"""
Collocation Analysis Package
=============================

A comprehensive Python package for collocation-based error analysis
of remote sensing and geophysical datasets.

This package provides implementations of various collocation methods:
- IVD: Information Vector Dual (2-way collocation)
- IVS: Information Vector with Scaling (2-way with bootstrap)
- TC: Triple Collocation (3-way, assumes zero error cross-correlation)
- EIVD: Extended Information Vector Dual (3-way, allows error cross-correlation)
- EC: Extended Collocation (4-way quadruple collocation)

Author: Converted from MATLAB by Claude
Original MATLAB code: licm_13@163.com
"""

from .ivd import ivd
from .ivs import ivs
from .tc import tc
from .eivd import eivd
from .ec import ec
from .utils import mse_judge, kge_objfun

__version__ = '1.0.0'

__all__ = [
    'ivd',
    'ivs',
    'tc',
    'eivd',
    'ec',
    'mse_judge',
    'kge_objfun',
]
