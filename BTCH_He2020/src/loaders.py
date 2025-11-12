
"""
Real data loader interfaces (stubs).

You can implement these to read real ET products from data/raw.
"""
from typing import Tuple, Dict, Any, Optional
import numpy as np

def load_real_products_timeseries() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        truth (T,) or None if unknown
        products (N,T)
        dates (T,) monthly strings 'YYYY-MM'
    Implementation: replace with actual reading logic from data/raw.
    """
    raise NotImplementedError("Implement reading of real ET product time series under data/raw/*.")

def load_real_products_grid() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        truth (T, Y, X) or None if unknown
        products (N, T, Y, X)
        dates (T,)
    Implementation: replace with actual reading logic (netCDF/GeoTIFF/etc.).
    """
    raise NotImplementedError("Implement reading of real ET product grids under data/raw/*.")
