"""
Information Vector with Scaling (IVS) Method
=============================================

This module implements the IVS collocation analysis method with bootstrap
resampling for robust uncertainty estimation.

The IVS method uses a lagged time series as an instrumental variable,
allowing error analysis with only two independent products. It supports
both multiplicative and additive error models.

Reference:
    Su, C. H., et al. (2014).
    A new instrumental variable based algorithm for estimating cross-correlated
    hydrological remote sensing errors.
    Journal of Hydrology, 520, 40-50.
"""

import numpy as np
from typing import Tuple, Literal
import warnings


def ivs(X: np.ndarray,
        N_boot: int = 1000,
        column: Literal[1, 2] = 1,
        M_A: Literal[0, 1] = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate IVS results with bootstrap resampling.

    The IVS method uses lag-1 autocorrelation as an instrumental variable
    to estimate error characteristics with only two data products.

    Parameters
    ----------
    X : np.ndarray
        Input data array of shape (n, 2), where n >= 4
        Column 0: First data product
        Column 1: Second data product

    N_boot : int, default=1000
        Number of bootstrap samples for uncertainty estimation

    column : {1, 2}, default=1
        Which column to use for generating Lag-1 series
        1: Use first column (X[:, 0])
        2: Use second column (X[:, 1])

    M_A : {0, 1}, default=1
        Error model type
        1: Multiplicative mode (uses log transformation)
        0: Additive mode (no transformation)

    Returns
    -------
    RMSE : np.ndarray
        Root Mean Square Error estimates of shape (2,)
        RMSE[0]: Error of first product
        RMSE[1]: Error of second product

    rho2 : np.ndarray
        Data-truth correlation coefficients of shape (2,)
        rho2[0]: Correlation between first product and truth
        rho2[1]: Correlation between second product and truth

    Raises
    ------
    ValueError
        If input data has wrong shape, contains NaN values,
        or parameters are invalid

    Examples
    --------
    >>> import numpy as np
    >>> from collocation import ivs
    >>>
    >>> # Generate synthetic data
    >>> np.random.seed(42)
    >>> n = 100
    >>> true_signal = np.abs(np.random.randn(n)) + 1.0
    >>> noise1 = 0.1 * np.random.randn(n)
    >>> noise2 = 0.15 * np.random.randn(n)
    >>> X1 = true_signal * (1 + noise1)  # Multiplicative error
    >>> X2 = true_signal * (1 + noise2)
    >>> X = np.column_stack([X1, X2])
    >>>
    >>> # Apply IVS with multiplicative mode
    >>> RMSE, rho2 = ivs(X, N_boot=500, column=1, M_A=1)
    >>> print(f"RMSE: {RMSE}")
    >>> print(f"Correlations: {rho2}")
    >>>
    >>> # For additive errors
    >>> X1_add = true_signal + noise1
    >>> X2_add = true_signal + noise2
    >>> X_add = np.column_stack([X1_add, X2_add])
    >>> RMSE_add, rho2_add = ivs(X_add, N_boot=500, column=1, M_A=0)

    Notes
    -----
    - The multiplicative mode (M_A=1) applies log transformation,
      making it suitable for variables with proportional errors
      (e.g., precipitation, soil moisture)
    - Negative values in multiplicative mode are replaced with 0.01
    - Bootstrap resampling provides robust uncertainty estimates
    - At least 4 samples are required for valid results
    """
    # Check minimum sample size
    if X.shape[0] < 4:
        return np.array([-9999, -9999]), np.array([-9999, -9999])

    # Validate input data
    if X.shape[1] != 2:
        raise ValueError('Error: Input data X must be an N x 2 matrix')

    if np.any(np.isnan(X)):
        raise ValueError('Error: Input data X must not contain NaNs')

    # Validate M_A parameter
    if M_A not in [0, 1]:
        raise ValueError('Error: M_A wrong input, 1 for multiplicative; 0 for additive')

    # Validate column parameter
    if column not in [1, 2]:
        raise ValueError('Error: column wrong input, 1 or 2')

    # Copy to avoid modifying input
    X = X.copy()

    # Apply transformation based on error model
    if M_A == 1:  # Multiplicative mode
        print('Multiplicative mode')
        # Deal with negative values
        X[X < 0] = 0.01
        # Log transformation
        X = np.log(X)
    else:  # Additive mode
        print('Additive mode')

    # Add Lag-1 series as instrumental variable
    if column == 1:
        I = X[1:, 0]
    else:
        I = X[1:, 1]

    # Remove last row and add instrumental variable
    X = X[:-1, :]
    X = np.column_stack([X, I])

    # Initialize outputs
    RMSE = np.zeros(2)
    rho2 = np.zeros(2)

    if X.shape[0] == 0:
        return np.array([-9999, -9999]), np.array([-9999, -9999])

    # Bootstrap resampling
    N_R = np.zeros((N_boot, 2))
    N_C = np.zeros((N_boot, 2))
    N_s_ivs = np.zeros(N_boot)

    for i in range(N_boot):
        # Bootstrap sample with replacement
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices, :]

        # Calculate covariance matrix
        Q = np.cov(X_boot, rowvar=False)

        # Calculate scaling ratio
        s_ivs = np.abs(Q[2, 0] / Q[2, 1])

        # Calculate R² (error variance components)
        R2 = np.zeros(2)
        R2[0] = Q[0, 0] - Q[0, 1] * s_ivs
        R2[1] = Q[1, 1] - Q[0, 1] / s_ivs

        # Handle negative R² values
        R2[R2 < 0] = 0.0001
        R = np.sqrt(R2)
        R[R < 1e-3] = 0.0001

        # Calculate correlation coefficients
        CC = np.zeros(2)
        CC[0] = Q[0, 1] * s_ivs / Q[0, 0]
        CC[1] = Q[0, 1] / (Q[0, 0] * s_ivs)

        # Handle invalid CC values
        CC[CC < 0] = 0.0001
        CC[CC > 1] = 0.9999

        N_R[i, :] = R
        N_C[i, :] = CC
        N_s_ivs[i] = s_ivs

    # Calculate mean values
    RMSE[0] = np.mean(N_R[:, 0])
    RMSE[1] = np.mean(N_R[:, 1])

    rho2[0] = np.mean(N_C[:, 0])
    rho2[1] = np.mean(N_C[:, 1])

    return RMSE, rho2


def ivs_with_uncertainty(X: np.ndarray,
                        N_boot: int = 1000,
                        column: Literal[1, 2] = 1,
                        M_A: Literal[0, 1] = 1,
                        confidence: float = 0.95) -> dict:
    """
    Calculate IVS results with bootstrap confidence intervals.

    Parameters
    ----------
    X : np.ndarray
        Input data array of shape (n, 2)
    N_boot : int, default=1000
        Number of bootstrap samples
    column : {1, 2}, default=1
        Column to use for Lag-1 series
    M_A : {0, 1}, default=1
        Error model (1: multiplicative, 0: additive)
    confidence : float, default=0.95
        Confidence level for intervals (e.g., 0.95 for 95% CI)

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'RMSE': Mean RMSE estimates
        - 'rho2': Mean correlation coefficients
        - 'RMSE_CI': Confidence intervals for RMSE
        - 'rho2_CI': Confidence intervals for rho2

    Examples
    --------
    >>> results = ivs_with_uncertainty(X, N_boot=1000, confidence=0.95)
    >>> print(f"RMSE: {results['RMSE']}")
    >>> print(f"95% CI: {results['RMSE_CI']}")
    """
    RMSE, rho2 = ivs(X, N_boot, column, M_A)

    # Calculate confidence intervals from bootstrap distribution
    alpha = (1 - confidence) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100

    # Run bootstrap again to get distribution
    X_copy = X.copy()
    if M_A == 1:
        X_copy[X_copy < 0] = 0.01
        X_copy = np.log(X_copy)

    if column == 1:
        I = X_copy[1:, 0]
    else:
        I = X_copy[1:, 1]

    X_copy = X_copy[:-1, :]
    X_copy = np.column_stack([X_copy, I])

    RMSE_dist = []
    rho2_dist = []

    for _ in range(N_boot):
        indices = np.random.choice(X_copy.shape[0], size=X_copy.shape[0], replace=True)
        X_boot = X_copy[indices, :]
        Q = np.cov(X_boot, rowvar=False)
        s_ivs = np.abs(Q[2, 0] / Q[2, 1])

        R2 = np.zeros(2)
        R2[0] = Q[0, 0] - Q[0, 1] * s_ivs
        R2[1] = Q[1, 1] - Q[0, 1] / s_ivs
        R2[R2 < 0] = 0.0001
        R = np.sqrt(R2)

        CC = np.zeros(2)
        CC[0] = Q[0, 1] * s_ivs / Q[0, 0]
        CC[1] = Q[0, 1] / (Q[0, 0] * s_ivs)
        CC = np.clip(CC, 0.0001, 0.9999)

        RMSE_dist.append(R)
        rho2_dist.append(CC)

    RMSE_dist = np.array(RMSE_dist)
    rho2_dist = np.array(rho2_dist)

    return {
        'RMSE': RMSE,
        'rho2': rho2,
        'RMSE_CI': np.percentile(RMSE_dist, [lower_percentile, upper_percentile], axis=0),
        'rho2_CI': np.percentile(rho2_dist, [lower_percentile, upper_percentile], axis=0)
    }
