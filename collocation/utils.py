"""
Utility Functions
=================

This module provides utility functions for data validation and
performance evaluation metrics.
"""

import numpy as np
from typing import Tuple, Optional


def mse_judge(data: float, omega: float) -> Tuple[float, float]:
    """
    Validate Mean Square Error data and weights.

    This function checks if data and weights are valid (not NaN, not negative).
    If invalid, both values are set to 0.

    Parameters
    ----------
    data : float
        Data value to check
    omega : float
        Weight or omega value to check

    Returns
    -------
    data : float
        Validated data value (0 if invalid)
    omega : float
        Validated omega value (0 if invalid)

    Examples
    --------
    >>> from collocation.utils import mse_judge
    >>>
    >>> # Valid inputs
    >>> data, omega = mse_judge(1.5, 0.8)
    >>> print(data, omega)  # Output: 1.5 0.8
    >>>
    >>> # Invalid data (NaN)
    >>> data, omega = mse_judge(np.nan, 0.8)
    >>> print(data, omega)  # Output: 0.0 0.0
    >>>
    >>> # Invalid data (negative)
    >>> data, omega = mse_judge(-1.5, 0.8)
    >>> print(data, omega)  # Output: 0.0 0.0
    >>>
    >>> # Invalid omega
    >>> data, omega = mse_judge(1.5, np.nan)
    >>> print(data, omega)  # Output: 0.0 0.0

    Notes
    -----
    This function is useful for ensuring numerical stability in
    calculations involving error metrics and weights.
    """
    if np.isnan(data):
        data = 0.0
        omega = 0.0
    elif data < 0:
        data = 0.0
        omega = 0.0
    elif np.isnan(omega):
        omega = 0.0
        data = 0.0
    elif omega < 0:
        omega = 0.0
        data = 0.0

    return data, omega


def kge_objfun(Rsim: np.ndarray, Robs: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Kling-Gupta Efficiency (KGE) objective function.

    KGE decomposes the Mean Squared Error and NSE performance criteria
    into three components: correlation, relative variability, and bias.

    Parameters
    ----------
    Rsim : np.ndarray
        Simulated/modeled data
    Robs : np.ndarray
        Observed/reference data

    Returns
    -------
    r : float
        Correlation coefficient between simulated and observed data
    KGE : float
        Kling-Gupta Efficiency value (lower is better)
        KGE = sqrt((r-1)² + (relvar-1)² + (bias-1)²)

    Raises
    ------
    ValueError
        If input arrays have different lengths

    Examples
    --------
    >>> import numpy as np
    >>> from collocation.utils import kge_objfun
    >>>
    >>> # Perfect match
    >>> obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> sim = obs.copy()
    >>> r, kge = kge_objfun(sim, obs)
    >>> print(f"r={r:.4f}, KGE={kge:.4f}")  # r=1.0, KGE≈0.0
    >>>
    >>> # With some error
    >>> sim = obs + 0.2 * np.random.randn(len(obs))
    >>> r, kge = kge_objfun(sim, obs)
    >>> print(f"r={r:.4f}, KGE={kge:.4f}")
    >>>
    >>> # Biased simulation
    >>> sim = obs * 1.2  # 20% bias
    >>> r, kge = kge_objfun(sim, obs)
    >>> print(f"r={r:.4f}, KGE={kge:.4f}")

    Notes
    -----
    The KGE metric decomposes into three components:
    1. Correlation (r): Linear correlation coefficient
    2. Relative variability (relvar): std(sim) / std(obs)
    3. Bias: mean(sim) / mean(obs)

    Perfect score: r=1, relvar=1, bias=1, resulting in KGE=0

    Reference:
        Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009).
        Decomposition of the mean squared error and NSE performance criteria:
        Implications for improving hydrological modelling.
        Journal of Hydrology, 377(1-2), 80-91.

    Alternative objective functions:
        - NSE (Nash-Sutcliffe Efficiency)
        - PBIAS (Percent Bias)
        - RMSE (Root Mean Square Error)
    """
    if len(Rsim) != len(Robs):
        raise ValueError("Simulated and observed arrays must have the same length")

    observed = Robs
    modelled = Rsim

    # Calculate statistics
    sdmodelled = np.std(modelled)
    sdobserved = np.std(observed)
    mmodelled = np.mean(modelled)
    mobserved = np.mean(observed)

    # Correlation coefficient
    r = np.corrcoef(observed, modelled)[0, 1]

    # Relative variability
    relvar = sdmodelled / sdobserved

    # Bias ratio
    bias = mmodelled / mobserved

    # KGE calculation
    KGE = np.sqrt(((r - 1) ** 2) + ((relvar - 1) ** 2) + ((bias - 1) ** 2))

    # Alternative weighted KGE (commented in original code)
    # KGE = np.sqrt(0.8 * ((r - 1) ** 2) + 0.2 * ((bias - 1) ** 2))

    return r, KGE


def nse(Rsim: np.ndarray, Robs: np.ndarray) -> float:
    """
    Calculate Nash-Sutcliffe Efficiency (NSE).

    Parameters
    ----------
    Rsim : np.ndarray
        Simulated/modeled data
    Robs : np.ndarray
        Observed/reference data

    Returns
    -------
    nse_value : float
        Nash-Sutcliffe Efficiency (-∞ to 1, 1 is perfect)

    Examples
    --------
    >>> nse_val = nse(simulated, observed)
    >>> print(f"NSE: {nse_val:.4f}")
    """
    numerator = np.sum((Robs - Rsim) ** 2)
    denominator = np.sum((Robs - np.mean(Robs)) ** 2)

    if denominator == 0:
        return np.nan

    nse_value = 1 - (numerator / denominator)
    return nse_value


def pbias(Rsim: np.ndarray, Robs: np.ndarray) -> float:
    """
    Calculate Percent Bias (PBIAS).

    Parameters
    ----------
    Rsim : np.ndarray
        Simulated/modeled data
    Robs : np.ndarray
        Observed/reference data

    Returns
    -------
    pbias_value : float
        Percent Bias (0 is perfect, positive means overestimation)

    Examples
    --------
    >>> bias = pbias(simulated, observed)
    >>> print(f"PBIAS: {bias:.2f}%")
    """
    numerator = np.sum(Robs - Rsim)
    denominator = np.sum(Robs)

    if denominator == 0:
        return np.nan

    pbias_value = 100 * numerator / denominator
    return pbias_value


def rmse(Rsim: np.ndarray, Robs: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error (RMSE).

    Parameters
    ----------
    Rsim : np.ndarray
        Simulated/modeled data
    Robs : np.ndarray
        Observed/reference data

    Returns
    -------
    rmse_value : float
        Root Mean Square Error (0 is perfect)

    Examples
    --------
    >>> error = rmse(simulated, observed)
    >>> print(f"RMSE: {error:.4f}")
    """
    rmse_value = np.sqrt(np.mean((Robs - Rsim) ** 2))
    return rmse_value


def mae(Rsim: np.ndarray, Robs: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    Parameters
    ----------
    Rsim : np.ndarray
        Simulated/modeled data
    Robs : np.ndarray
        Observed/reference data

    Returns
    -------
    mae_value : float
        Mean Absolute Error (0 is perfect)

    Examples
    --------
    >>> error = mae(simulated, observed)
    >>> print(f"MAE: {error:.4f}")
    """
    mae_value = np.mean(np.abs(Robs - Rsim))
    return mae_value


def correlation(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Calculate Pearson correlation coefficient.

    Parameters
    ----------
    X : np.ndarray
        First data array
    Y : np.ndarray
        Second data array

    Returns
    -------
    r : float
        Correlation coefficient (-1 to 1)

    Examples
    --------
    >>> r = correlation(data1, data2)
    >>> print(f"Correlation: {r:.4f}")
    """
    return np.corrcoef(X, Y)[0, 1]


def remove_outliers(data: np.ndarray,
                   n_std: float = 3.0,
                   method: str = 'zscore') -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outliers from data.

    Parameters
    ----------
    data : np.ndarray
        Input data array
    n_std : float, default=3.0
        Number of standard deviations for outlier detection
    method : str, default='zscore'
        Method for outlier detection: 'zscore' or 'iqr'

    Returns
    -------
    clean_data : np.ndarray
        Data with outliers removed
    mask : np.ndarray
        Boolean mask indicating which points were kept

    Examples
    --------
    >>> clean, mask = remove_outliers(data, n_std=3.0)
    >>> print(f"Removed {np.sum(~mask)} outliers")
    """
    if method == 'zscore':
        mean = np.mean(data)
        std = np.std(data)
        z_scores = np.abs((data - mean) / std)
        mask = z_scores < n_std

    elif method == 'iqr':
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - n_std * iqr
        upper_bound = q3 + n_std * iqr
        mask = (data >= lower_bound) & (data <= upper_bound)

    else:
        raise ValueError("method must be 'zscore' or 'iqr'")

    clean_data = data[mask]
    return clean_data, mask


def calculate_all_metrics(Rsim: np.ndarray, Robs: np.ndarray) -> dict:
    """
    Calculate all available performance metrics.

    Parameters
    ----------
    Rsim : np.ndarray
        Simulated/modeled data
    Robs : np.ndarray
        Observed/reference data

    Returns
    -------
    metrics : dict
        Dictionary containing all calculated metrics:
        - 'r': Correlation coefficient
        - 'KGE': Kling-Gupta Efficiency
        - 'NSE': Nash-Sutcliffe Efficiency
        - 'PBIAS': Percent Bias
        - 'RMSE': Root Mean Square Error
        - 'MAE': Mean Absolute Error

    Examples
    --------
    >>> metrics = calculate_all_metrics(simulated, observed)
    >>> for name, value in metrics.items():
    ...     print(f"{name}: {value:.4f}")
    """
    r, kge_val = kge_objfun(Rsim, Robs)
    nse_val = nse(Rsim, Robs)
    pbias_val = pbias(Rsim, Robs)
    rmse_val = rmse(Rsim, Robs)
    mae_val = mae(Rsim, Robs)

    return {
        'r': r,
        'KGE': kge_val,
        'NSE': nse_val,
        'PBIAS': pbias_val,
        'RMSE': rmse_val,
        'MAE': mae_val
    }
