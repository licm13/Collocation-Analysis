"""
Extended Information Vector Dual (EIVD) Method
===============================================

This module implements the EIVD collocation analysis method for
three independent data products, allowing for non-zero error
cross-correlation between products.

EIVD extends the triple collocation method by incorporating
temporal correlation information through lag-1 series, enabling
estimation of error cross-correlation.

References:
    - Dong, J., et al. (2019). An instrument variable based algorithm
      for estimating cross-correlated hydrological remote sensing errors.
      Journal of Hydrology, 581, 124385.
"""

import numpy as np
from typing import Tuple
import warnings


def eivd(tri: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate EIVD results with error cross-correlation estimation.

    EIVD allows for non-zero error cross-correlation between products,
    unlike traditional TC which assumes independent errors. This makes
    EIVD more suitable for products with potentially correlated errors
    (e.g., products from similar sensors or algorithms).

    Key difference from TC: Estimates error cross-correlation between
    products 2 and 3 (in the input ordering).

    Parameters
    ----------
    tri : np.ndarray
        Input data array of shape (n, 3), where n > 1
        Column 0: First data product (reference)
        Column 1: Second data product
        Column 2: Third data product

        Note: Only products 1 and 2 (columns 1&2) have non-zero
        error cross-correlation in the output

    Returns
    -------
    EeeT : np.ndarray
        Error covariance matrix of shape (3, 3)
        EeeT[i,i]: Error variance of product i
        EeeT[1,2] = EeeT[2,1]: Error cross-correlation between products 1 and 2
        EeeT[0,1] = EeeT[0,2] = 0 (zero by assumption)

    SNR : np.ndarray
        Signal-to-Noise Ratio of shape (3,)
        SNR[i] = (beta_i² * sigma_theta²) / sigma_e_i²

    rho2 : np.ndarray
        Data-truth correlation coefficients of shape (3,)
        rho2[i] = SNR[i] / (1 + SNR[i])

    fMSE : np.ndarray
        Fractional Mean Square Error of shape (3,)
        fMSE[i] = 1 - rho2[i]

    L : np.ndarray
        Lag-1 temporal autocorrelation of shape (3,)
        L[i] = E[X_i(t) * X_i(t-1)]
        Used for temporal correlation information

    Raises
    ------
    ValueError
        If input data does not have exactly 3 columns

    Examples
    --------
    >>> import numpy as np
    >>> from collocation import eivd
    >>>
    >>> # Generate synthetic data with correlated errors
    >>> np.random.seed(42)
    >>> n = 200
    >>> true_signal = np.random.randn(n)
    >>>
    >>> # Independent error for product 1
    >>> noise1 = 0.2 * np.random.randn(n)
    >>>
    >>> # Correlated errors for products 2 and 3
    >>> common_error = 0.1 * np.random.randn(n)
    >>> noise2 = common_error + 0.15 * np.random.randn(n)
    >>> noise3 = common_error + 0.2 * np.random.randn(n)
    >>>
    >>> X1 = true_signal + noise1
    >>> X2 = true_signal + noise2
    >>> X3 = true_signal + noise3
    >>> tri = np.column_stack([X1, X2, X3])
    >>>
    >>> # Apply EIVD
    >>> EeeT, SNR, rho2, fMSE, L = eivd(tri)
    >>> print(f"Error covariance matrix:\\n{EeeT}")
    >>> print(f"Error cross-correlation: {EeeT[1,2]}")
    >>> print(f"SNR: {SNR}")
    >>> print(f"Lag-1 autocorr: {L}")

    Notes
    -----
    The EIVD method solves an overdetermined linear system:
        y = Ax
    where:
        - y: Known variance combinations (10x1)
        - A: Equation matrix (10x8)
        - x: EIVD results (8x1) = [signal_variances; error_variances; error_covariance]

    The system includes:
        - 4 variance equations
        - 6 equations from lag-1 temporal correlation

    Solution via least squares:
        x = (A'A)^(-1) A'y

    Attention:
        - For input [x, y, z], only (y,z) have non-zero error cross-correlation
        - Temporal resolution affects lag-1 series quality
        - Method requires temporal correlation structure in data
    """
    # Validate input shape
    if tri.shape[1] != 3:
        raise ValueError('Error: Input data must be an N x 3 matrix')

    if tri.shape[0] < 2:
        warnings.warn("Insufficient data for EIVD analysis. Need at least 2 samples.", UserWarning)
        EeeT = np.zeros((3, 3))
        SNR = np.zeros(3)
        rho2 = np.zeros(3)
        fMSE = np.zeros(3)
        L = np.zeros(3)
        return EeeT, SNR, rho2, fMSE, L

    # Calculate error covariance matrix
    ExxT = np.cov(tri, rowvar=False)

    # Generate Lag-1 [temporal-resolution] series
    tri_lag = tri[1:, :]
    tri = tri[:-1, :]
    L = np.mean(tri * tri_lag, axis=0)

    # Build equation system for EIVD
    # More complex than TC: 10 equations, 8 unknowns
    # Includes temporal correlation information

    A = np.array([
        [1, 0, 0, 0, 1, 0, 0, 0],  # Var(X1)
        [0, 1, 0, 0, 0, 1, 0, 0],  # Var(X2)
        [0, 0, 1, 0, 0, 0, 1, 0],  # Var(X3)
        [0, 0, 0, 1, 0, 0, 0, 1],  # Cov(X2,X3) with error correlation
        [1, 0, 0, 0, 0, 0, 0, 0],  # From lag-1 correlations
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0]
    ])

    # Build right-hand side vector with temporal information
    y = np.array([
        ExxT[0, 0],  # Var(X1)
        ExxT[1, 1],  # Var(X2)
        ExxT[2, 2],  # Var(X3)
        ExxT[1, 2],  # Cov(X2,X3)
        ExxT[0, 1] * np.sqrt(L[0] / L[1]),  # Temporal correlation based
        ExxT[0, 2] * np.sqrt(L[0] / L[2]),
        ExxT[1, 0] * np.sqrt(L[1] / L[0]),
        ExxT[2, 0] * np.sqrt(L[2] / L[0]),
        ExxT[0, 1] * np.sqrt(L[2] / L[0]),
        ExxT[0, 2] * np.sqrt(L[1] / L[0])
    ])

    # Solve linear system via least squares
    x = np.linalg.lstsq(A, y, rcond=None)[0]

    # Assemble error variance matrix
    EeeT = np.diag(x[4:7])
    EeeT[1, 2] = x[7]  # Error cross-correlation between products 2 and 3
    EeeT[2, 1] = x[7]  # Symmetric

    # Calculate derived metrics
    SNR = x[:3] / x[4:7]
    rho2 = SNR / (1 + SNR)
    fMSE = 1 - rho2

    return EeeT, SNR, rho2, fMSE, L


def eivd_detect_correlation(tri: np.ndarray,
                           threshold: float = 0.01) -> dict:
    """
    Detect and quantify error cross-correlation using EIVD.

    Parameters
    ----------
    tri : np.ndarray
        Input data array of shape (n, 3)
    threshold : float, default=0.01
        Threshold for considering correlation significant

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'error_correlation': Error cross-correlation coefficient
        - 'is_significant': Whether correlation exceeds threshold
        - 'EeeT': Full error covariance matrix
        - 'normalized_correlation': Normalized correlation coefficient

    Examples
    --------
    >>> results = eivd_detect_correlation(tri, threshold=0.01)
    >>> if results['is_significant']:
    ...     print(f"Significant error correlation detected: {results['error_correlation']:.4f}")
    """
    EeeT, SNR, rho2, fMSE, L = eivd(tri)

    # Extract error cross-correlation
    error_corr = EeeT[1, 2]

    # Normalize by error standard deviations
    if EeeT[1, 1] > 0 and EeeT[2, 2] > 0:
        normalized_corr = error_corr / np.sqrt(EeeT[1, 1] * EeeT[2, 2])
    else:
        normalized_corr = 0.0

    return {
        'error_correlation': error_corr,
        'normalized_correlation': normalized_corr,
        'is_significant': abs(error_corr) > threshold,
        'EeeT': EeeT,
        'SNR': SNR,
        'rho2': rho2,
        'fMSE': fMSE,
        'lag1_autocorr': L
    }


def compare_tc_eivd(tri: np.ndarray) -> dict:
    """
    Compare TC and EIVD results to assess impact of error correlation.

    This function runs both TC and EIVD on the same data to quantify
    the difference caused by allowing error cross-correlation.

    Parameters
    ----------
    tri : np.ndarray
        Input data array of shape (n, 3)

    Returns
    -------
    comparison : dict
        Dictionary containing:
        - 'tc_results': TC analysis results
        - 'eivd_results': EIVD analysis results
        - 'error_corr': Error cross-correlation from EIVD
        - 'snr_diff': Difference in SNR estimates
        - 'recommendation': Which method to use

    Examples
    --------
    >>> comparison = compare_tc_eivd(tri)
    >>> print(comparison['recommendation'])
    >>> print(f"SNR difference: {comparison['snr_diff']}")
    """
    from .tc import tc

    # Run TC (assumes zero error correlation)
    tc_EeeT, tc_SNR, tc_rho2, tc_fMSE = tc(tri)

    # Run EIVD (allows error correlation)
    eivd_EeeT, eivd_SNR, eivd_rho2, eivd_fMSE, L = eivd(tri)

    # Calculate differences
    snr_diff = np.abs(tc_SNR - eivd_SNR)
    error_corr = eivd_EeeT[1, 2]

    # Recommendation based on error correlation magnitude
    if abs(error_corr) > 0.01 * np.sqrt(eivd_EeeT[1, 1] * eivd_EeeT[2, 2]):
        recommendation = "Use EIVD: Significant error cross-correlation detected"
    else:
        recommendation = "Either TC or EIVD: Error cross-correlation is negligible"

    return {
        'tc_results': {
            'EeeT': tc_EeeT,
            'SNR': tc_SNR,
            'rho2': tc_rho2,
            'fMSE': tc_fMSE
        },
        'eivd_results': {
            'EeeT': eivd_EeeT,
            'SNR': eivd_SNR,
            'rho2': eivd_rho2,
            'fMSE': eivd_fMSE,
            'lag1': L
        },
        'error_corr': error_corr,
        'snr_diff': snr_diff,
        'recommendation': recommendation
    }
