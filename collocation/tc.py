"""
Triple Collocation (TC) Method
===============================

This module implements the classic triple collocation analysis for
three independent data products.

Triple Collocation assumes zero error cross-correlation between products
and provides estimates of individual product errors without requiring
a ground truth reference.

References:
    - Stoffelen, A. (1998). Toward the true near-surface wind speed:
      Error modeling and calibration using triple collocation.
      Journal of Geophysical Research, 103(C4), 7755-7766.
    - Scipal, K., et al. (2008). A possible solution for the problem
      of estimating the error structure of global soil moisture data sets.
      Geophysical Research Letters, 35(24).
"""

import numpy as np
from typing import Tuple
import warnings


def tc(tri: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate classic triple collocation results.

    Triple collocation (TC) estimates error variances and signal-to-noise
    ratios for three independent data products without requiring ground truth.

    Key assumption: Zero error cross-correlation between products
    (i.e., errors are mutually independent).

    Parameters
    ----------
    tri : np.ndarray
        Input data array of shape (n, 3), where n >= 100
        Column 0: First data product
        Column 1: Second data product
        Column 2: Third data product

    Returns
    -------
    EeeT : np.ndarray
        Error covariance matrix of shape (3, 3), diagonal
        EeeT[i,i]: Error variance of product i
        Off-diagonal elements are zero (assumed)

    SNR : np.ndarray
        Signal-to-Noise Ratio of shape (3,)
        SNR[i] = (beta_i² * sigma_theta²) / sigma_e_i²
        Higher SNR indicates lower error relative to signal

    rho2 : np.ndarray
        Data-truth correlation coefficients of shape (3,)
        rho2[i] = SNR[i] / (1 + SNR[i])
        Values range from 0 to 1

    fMSE : np.ndarray
        Fractional Mean Square Error of shape (3,)
        fMSE[i] = 1 - rho2[i]
        Represents fractional error relative to total variance

    Raises
    ------
    ValueError
        If input data does not have exactly 3 columns

    Warnings
    --------
    If sample size is less than 100, results may be unstable

    Examples
    --------
    >>> import numpy as np
    >>> from collocation import tc
    >>>
    >>> # Generate synthetic data
    >>> np.random.seed(42)
    >>> n = 200
    >>> true_signal = np.random.randn(n)
    >>> noise1 = 0.2 * np.random.randn(n)
    >>> noise2 = 0.25 * np.random.randn(n)
    >>> noise3 = 0.3 * np.random.randn(n)
    >>> X1 = true_signal + noise1
    >>> X2 = true_signal + noise2
    >>> X3 = true_signal + noise3
    >>> tri = np.column_stack([X1, X2, X3])
    >>>
    >>> # Apply triple collocation
    >>> EeeT, SNR, rho2, fMSE = tc(tri)
    >>> print(f"Error variances: {np.diag(EeeT)}")
    >>> print(f"SNR: {SNR}")
    >>> print(f"Correlations: {rho2}")
    >>> print(f"Fractional MSE: {fMSE}")

    Notes
    -----
    The TC method solves the linear system:
        y = Ax
    where:
        - y: Known variance combinations (6x1)
        - A: Equation matrix (6x6)
        - x: TC results (6x1) = [SNR; error_variances]

    The solution is obtained via least squares:
        x = (A'A)^(-1) A'y

    Mathematical formulation:
        - Product i: X_i = alpha_i + beta_i * theta + e_i
        - theta: True signal (unknown)
        - e_i: Random error (zero mean, independent)
        - Covariance structure provides 6 equations to solve for
          3 SNR values and 3 error variances
    """
    # Check sample size
    if tri.shape[0] < 100:
        warnings.warn(
            f"Sample size ({tri.shape[0]}) is less than recommended minimum (100). "
            "Results may be unstable. Returning default values.",
            UserWarning
        )
        EeeT = 0.01 * np.eye(3)
        SNR = np.zeros(3)
        rho2 = np.zeros(3)
        fMSE = np.zeros(3)
        return EeeT, SNR, rho2, fMSE

    # Validate input shape
    if tri.shape[1] != 3:
        raise ValueError('Error: Input data must be an N x 3 matrix')

    # Calculate covariance matrix
    ExxT = np.cov(tri, rowvar=False)

    # Build equation system for triple collocation
    # Equations based on covariance structure:
    # Var(X_i) = beta_i² sigma_theta² + sigma_e_i²
    # Cov(X_i, X_j) = beta_i beta_j sigma_theta²

    A = np.array([
        [1, 0, 0, 1, 0, 0],  # Var(X1) = beta1²*sigma_theta² + sigma_e1²
        [0, 1, 0, 0, 1, 0],  # Var(X2) = beta2²*sigma_theta² + sigma_e2²
        [0, 0, 1, 0, 0, 1],  # Var(X3) = beta3²*sigma_theta² + sigma_e3²
        [1, 0, 0, 0, 0, 0],  # From cross-covariances
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ])

    # Build right-hand side vector
    y = np.array([
        ExxT[0, 0],  # Var(X1)
        ExxT[1, 1],  # Var(X2)
        ExxT[2, 2],  # Var(X3)
        ExxT[0, 1] * ExxT[0, 2] / ExxT[1, 2],  # beta1²*sigma_theta²
        ExxT[1, 0] * ExxT[1, 2] / ExxT[0, 2],  # beta2²*sigma_theta²
        ExxT[2, 0] * ExxT[2, 1] / ExxT[0, 1]   # beta3²*sigma_theta²
    ])

    # Solve linear system via least squares
    x = np.linalg.lstsq(A, y, rcond=None)[0]

    # Extract results
    # x[0:3] = beta_i² * sigma_theta² (signal variance)
    # x[3:6] = sigma_e_i² (error variance)

    # Error covariance matrix (diagonal)
    EeeT = np.diag(x[3:])

    # Signal-to-Noise Ratio
    SNR = x[:3] / x[3:]

    # Data-truth correlation coefficient
    rho2 = SNR / (1 + SNR)

    # Fractional Mean Square Error
    fMSE = 1 - rho2

    return EeeT, SNR, rho2, fMSE


def tc_with_rescaling(tri: np.ndarray,
                     reference_idx: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Triple collocation with data rescaling to a reference product.

    Parameters
    ----------
    tri : np.ndarray
        Input data array of shape (n, 3)
    reference_idx : int, default=0
        Index of reference product (0, 1, or 2)

    Returns
    -------
    EeeT : np.ndarray
        Error covariance matrix (3, 3)
    SNR : np.ndarray
        Signal-to-Noise Ratio (3,)
    rho2 : np.ndarray
        Data-truth correlation (3,)
    fMSE : np.ndarray
        Fractional MSE (3,)
    rescaled_data : np.ndarray
        Rescaled data array (n, 3)

    Examples
    --------
    >>> # Rescale products to first product
    >>> EeeT, SNR, rho2, fMSE, rescaled = tc_with_rescaling(tri, reference_idx=0)
    """
    if reference_idx not in [0, 1, 2]:
        raise ValueError('reference_idx must be 0, 1, or 2')

    # Calculate covariances for rescaling
    cov_matrix = np.cov(tri, rowvar=False)

    # Rescale to reference
    rescaled = tri.copy()
    ref = tri[:, reference_idx]
    ref_mean = np.mean(ref)
    ref_std = np.std(ref)

    for i in range(3):
        if i != reference_idx:
            # Linear rescaling: X_i' = a + b*X_i
            # Match mean and variance to reference
            data_mean = np.mean(tri[:, i])
            data_std = np.std(tri[:, i])

            # Calculate rescaling coefficients
            beta = ref_std / data_std
            alpha = ref_mean - beta * data_mean

            rescaled[:, i] = alpha + beta * tri[:, i]

    # Apply TC to rescaled data
    EeeT, SNR, rho2, fMSE = tc(rescaled)

    return EeeT, SNR, rho2, fMSE, rescaled


def rank_products(tri: np.ndarray) -> np.ndarray:
    """
    Rank three products by their quality using TC analysis.

    Parameters
    ----------
    tri : np.ndarray
        Input data array of shape (n, 3)

    Returns
    -------
    ranking : np.ndarray
        Indices of products sorted by quality (best to worst)
        Based on SNR (higher is better)

    Examples
    --------
    >>> ranking = rank_products(tri)
    >>> print(f"Best product: {ranking[0]}")
    >>> print(f"Quality order: {ranking}")
    """
    _, SNR, _, _ = tc(tri)
    ranking = np.argsort(SNR)[::-1]  # Sort descending (highest SNR first)
    return ranking
