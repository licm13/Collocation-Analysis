"""
Information Vector Dual (IVD) Method
=====================================

This module implements the IVD collocation analysis method for fusing
two independent data products (e.g., active and passive remote sensing).

The IVD method considers temporal offsets to ensure positive correlation
between instrument trends and prevent negative error correlation.

Reference:
    Dong, J., Crow, W. T., Reichle, R., & Liu, Q. (2014).
    Fusing active and passive remotely sensed soil moisture products
    using information vectors.
    https://www.researchgate.net/publication/228844200
"""

import numpy as np
from typing import Tuple, Optional


def ivd(dual: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate IVD results considering the choice of temporal offset.

    The IVD method finds an optimal temporal offset that maximizes the sum
    of autocorrelations (R_Ix + R_Jy), ensuring:
    - R_Ix * R_Jy > 0 (same trend between instruments)
    - Prevents negative correlation of random errors

    Parameters
    ----------
    dual : np.ndarray
        Input data array of shape (n, 2), where n >= 4
        Column 0: First data product (X)
        Column 1: Second data product (Y)

    Returns
    -------
    EeeT : np.ndarray
        Error covariance matrix of shape (2, 2)
        EeeT[0,0]: Error variance of product X
        EeeT[1,1]: Error variance of product Y

    rho2 : np.ndarray
        Data-truth correlation coefficients of shape (2,)
        rho2[0]: Correlation between X and truth
        rho2[1]: Correlation between Y and truth

    u : np.ndarray
        Linear-squared merging weights of shape (2,)
        u[0]: Weight for product X
        u[1]: Weight for product Y
        Can be used to merge products: merged = u[0]*X + u[1]*Y

    Raises
    ------
    ValueError
        If input data has wrong shape or contains NaN values

    Examples
    --------
    >>> import numpy as np
    >>> from collocation import ivd
    >>>
    >>> # Generate synthetic data
    >>> np.random.seed(42)
    >>> n = 100
    >>> true_signal = np.random.randn(n)
    >>> noise1 = 0.2 * np.random.randn(n)
    >>> noise2 = 0.3 * np.random.randn(n)
    >>> X = true_signal + noise1
    >>> Y = true_signal + noise2
    >>> dual = np.column_stack([X, Y])
    >>>
    >>> # Apply IVD
    >>> EeeT, rho2, u = ivd(dual)
    >>> print(f"Error variances: {np.diag(EeeT)}")
    >>> print(f"Correlations: {rho2}")
    >>> print(f"Weights: {u}")
    """
    # Check minimum sample size
    if dual.shape[0] < 4:
        EeeT = np.full((2, 2), np.nan)
        rho2 = np.full(2, np.nan)
        u = np.full(2, np.nan)
        return EeeT, rho2, u

    # Validate input data
    if dual.shape[1] != 2:
        raise ValueError('Error: Input data must be an N x 2 matrix')

    if np.any(np.isnan(dual)):
        raise ValueError('Error: Input data must not contain NaNs')

    X = dual[:, 0]
    Y = dual[:, 1]

    # Find optimal temporal offset
    # Optimization: Limit search to reasonable offset range (max 20% of data length)
    # and add early termination if correlation doesn't improve
    max_offset = min(len(X), max(4, len(X) // 5))
    sum_R = 0
    offset = 0
    no_improvement_count = 0
    max_no_improvement = 5  # Stop if no improvement in 5 consecutive iterations

    for i in range(1, max_offset):
        # Create lagged series
        tri_I = X[i:]
        tri_J = Y[i:]
        tri_X = X[:-i]
        tri_Y = Y[:-i]

        # Skip if series too short for meaningful correlation
        if len(tri_I) < 4:
            break

        # Calculate sum of correlation coefficients safely. When the
        # lag produces very short series, np.corrcoef may emit warnings
        # (degrees of freedom <= 0). Handle those cases and treat
        # non-finite correlations as zero for the purpose of choosing an
        # offset.
        def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
            if a.size < 2 or b.size < 2:
                return 0.0
            with np.errstate(invalid='ignore', divide='ignore'):
                c = np.corrcoef(a, b)
            try:
                val = float(c[0, 1])
            except Exception:
                val = 0.0
            if not np.isfinite(val):
                return 0.0
            return val

        corr_XI = _safe_corr(tri_X, tri_I)
        corr_YJ = _safe_corr(tri_Y, tri_J)
        judge = corr_XI + corr_YJ

        if judge > sum_R:
            sum_R = judge
            offset = i
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            # Early termination if no improvement for several iterations
            if no_improvement_count >= max_no_improvement:
                break

    # Generate Lag-offset series
    I = X[offset:]
    J = Y[offset:]
    X = X[:-offset] if offset > 0 else X
    Y = Y[:-offset] if offset > 0 else Y

    # Calculate IVD variances
    combin = np.column_stack([X, I, Y, J])
    # Protect against very small sample sizes or degenerate data which cause
    # numpy.cov to emit RuntimeWarnings (degrees of freedom <= 0, divide by zero).
    if combin.shape[0] < 2:
        # Not enough data to compute covariance reliably
        EeeT = np.full((2, 2), np.nan)
        rho2 = np.full(2, np.nan)
        u = np.full(2, np.nan)
        return EeeT, rho2, u

    with np.errstate(invalid='ignore', divide='ignore'):
        ExxT = np.cov(combin, rowvar=False)

    # If covariance produced non-finite values (e.g., due to constant columns),
    # return NaNs to signal insufficient variability instead of letting later
    # operations produce warnings or exceptions.
    if not np.all(np.isfinite(ExxT)):
        EeeT = np.full((2, 2), np.nan)
        rho2 = np.full(2, np.nan)
        u = np.full(2, np.nan)
        return EeeT, rho2, u

    # Initialize outputs
    u = np.zeros(2)
    rho2 = np.zeros(2)
    EeeT = np.zeros((2, 2))

    # Calculate scaling factor
    s_ivd = np.sqrt(ExxT[0, 1] / ExxT[2, 3])

    # Calculate error variances
    EeeT[0, 0] = ExxT[0, 0] - ExxT[0, 1] * s_ivd
    EeeT[1, 1] = ExxT[1, 1] - ExxT[1, 0] / s_ivd

    # Calculate merging weights
    u[0] = (EeeT[1, 1] * s_ivd) / (EeeT[0, 0] + EeeT[1, 1] * s_ivd)
    u[1] = (EeeT[0, 0] * s_ivd) / (EeeT[0, 0] + EeeT[1, 1] * s_ivd)

    # Calculate data-truth correlations
    rho2[0] = ExxT[0, 0] - ExxT[0, 2] * s_ivd
    rho2[1] = ExxT[1, 1] - ExxT[0, 2] / s_ivd

    return EeeT, rho2, u


def merge_products(X: np.ndarray, Y: np.ndarray,
                   weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Merge two data products using IVD-derived weights.

    Parameters
    ----------
    X : np.ndarray
        First data product
    Y : np.ndarray
        Second data product
    weights : np.ndarray, optional
        Merging weights. If None, will be calculated using IVD

    Returns
    -------
    merged : np.ndarray
        Merged data product

    Examples
    --------
    >>> X = np.array([1.0, 2.0, 3.0, 4.0])
    >>> Y = np.array([1.1, 2.1, 2.9, 4.2])
    >>> merged = merge_products(X, Y)
    """
    if weights is None:
        dual = np.column_stack([X, Y])
        _, _, weights = ivd(dual)

    merged = weights[0] * X + weights[1] * Y
    return merged
