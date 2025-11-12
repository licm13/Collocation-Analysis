
"""
Windowing utilities for time-varying BTCH weights.
"""
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from btch import tch_uncorrelated_variances, tch_correlated_regularized, integrate_btch

def _month_labels(dates: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    months = pd.to_datetime(dates).month
    labels = np.array([f"M{m:02d}" for m in months])
    unique_order = [f"M{m:02d}" for m in range(1,13)]
    return labels, unique_order

def _season_labels(dates: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    months = pd.to_datetime(dates).month
    def season(m):
        if m in [12,1,2]: return "DJF"
        if m in [3,4,5]: return "MAM"
        if m in [6,7,8]: return "JJA"
        return "SON"
    labels = np.array([season(m) for m in months])
    unique_order = ["DJF","MAM","JJA","SON"]
    return labels, unique_order

def _halfyear_labels(dates: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    months = pd.to_datetime(dates).month
    labels = np.array([ "H1" if m<=6 else "H2" for m in months ])
    unique_order = ["H1","H2"]
    return labels, unique_order

def assign_groups(dates: np.ndarray, scheme: str) -> Tuple[np.ndarray, List[str]]:
    scheme = scheme.lower()
    if scheme in ["monthly","month","mon"]:
        return _month_labels(dates)
    if scheme in ["seasonal","season","seas"]:
        return _season_labels(dates)
    if scheme in ["halfyear","half-year","half"]:
        return _halfyear_labels(dates)
    raise ValueError("Unknown scheme. Use monthly/seasonal/halfyear.")

def compute_windowed_btch(products: np.ndarray, dates: np.ndarray, scheme: str, mode: str="correlated"):
    """
    Compute time-varying BTCH weights per window and fuse series.
    Args:
        products: (N,T)
        dates: (T,) strings
        scheme: monthly|seasonal|halfyear
        mode: 'correlated' or 'uncorrelated'
    Returns:
        fused (T,)
        weights_by_group: dict[group] -> weights (N,)
        sigmas2_by_group: dict[group] -> variances (N,)
        labels: (T,) group labels
        order: list of groups in canonical order
    """
    N, T = products.shape
    labels, order = assign_groups(dates, scheme)
    weights_by_group = {}
    sigmas2_by_group = {}

    for g in order:
        idx = np.where(labels==g)[0]
        if idx.size < N+2:
            # too few samples; fallback to all-time uncorrelated as a stable default
            if mode=="correlated":
                mode_eff = "uncorrelated"
            else:
                mode_eff = mode
        else:
            mode_eff = mode

        X = products[:, idx]
        if mode_eff == "uncorrelated":
            diag_vars = tch_uncorrelated_variances(X)
            R = np.diag(diag_vars)
        else:
            R = tch_correlated_regularized(X, alpha=1e-2, lr=5e-3, iters=400, seed=0)

        fused_g, w, s2 = integrate_btch(X, R)
        weights_by_group[g] = w
        sigmas2_by_group[g] = s2

    # apply weights group-wise to full products
    fused = np.zeros(T)
    for t in range(T):
        g = labels[t]
        w = weights_by_group[g]
        fused[t] = w @ products[:, t]

    return fused, weights_by_group, sigmas2_by_group, labels, order
