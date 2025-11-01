"""
Robust Estimation for Data Fusion
===================================

Provides robust loss functions and outlier detection for MSE/covariance estimation:

1. **Huber loss**: Smooth combination of L2 (small errors) and L1 (large errors)
2. **MAE (L1)**: Less sensitive to outliers than MSE
3. **Outlier detection**: Statistical tests and isolation forests
4. **Robust covariance**: Downweight outliers in covariance estimation

Useful when data contains outliers, heavy tails, or quality flags.

Examples
--------
>>> import numpy as np
>>> from collocation.fusion import estimate_mse_robust, detect_outliers
>>>
>>> # Generate data with outliers
>>> errors = np.random.randn(100)
>>> errors[::10] = 10.0  # Add outliers
>>>
>>> # Robust MSE
>>> mse_robust = estimate_mse_robust(errors, np.zeros(100))
>>> mse_standard = np.mean(errors**2)
>>> print(mse_robust < mse_standard)  # True
>>>
>>> # Detect outliers
>>> outlier_mask = detect_outliers(errors, method="iqr")
>>> print(outlier_mask.sum())  # ~10
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from scipy import stats


def huber_loss(
    residual: np.ndarray,
    delta: float = 1.35,
) -> np.ndarray:
    """
    Huber loss function: smooth L1-L2 hybrid.

    L(r) = {
        0.5 * r^2           if |r| ≤ δ
        δ * (|r| - 0.5*δ)   if |r| > δ
    }

    Parameters
    ----------
    residual : np.ndarray
        Residual values (prediction - observation).
    delta : float, default=1.35
        Threshold for switching from quadratic to linear.
        Standard value: 1.35 * MAD (median absolute deviation).

    Returns
    -------
    np.ndarray
        Huber loss values.

    Examples
    --------
    >>> r = np.array([-2, -1, 0, 1, 2])
    >>> loss = huber_loss(r, delta=1.0)
    >>> loss
    array([1.5 , 0.5 , 0.  , 0.5 , 1.5 ])
    """
    abs_residual = np.abs(residual)
    loss = np.where(
        abs_residual <= delta,
        0.5 * residual**2,
        delta * (abs_residual - 0.5 * delta),
    )
    return loss


def huber_weight(
    residual: np.ndarray,
    delta: float = 1.35,
) -> np.ndarray:
    """
    Huber loss derivative → weight for iterative reweighted least squares.

    w(r) = {
        1            if |r| ≤ δ
        δ / |r|      if |r| > δ
    }

    Parameters
    ----------
    residual : np.ndarray
        Residual values.
    delta : float, default=1.35
        Huber threshold.

    Returns
    -------
    np.ndarray
        Weights in [0, 1].
    """
    abs_residual = np.abs(residual)
    weights = np.where(
        abs_residual <= delta,
        1.0,
        delta / np.maximum(abs_residual, 1e-10),  # Avoid division by zero
    )
    return weights


def estimate_mse_robust(
    y_pred: np.ndarray,
    y_ref: np.ndarray,
    loss: Literal["huber", "mae"] = "huber",
    delta: float | None = None,
    max_iter: int = 10,
    tol: float = 1e-4,
) -> float:
    """
    Robustly estimate MSE using Huber or MAE loss.

    Uses iterative reweighted least squares (IRLS) for Huber loss.

    Parameters
    ----------
    y_pred : np.ndarray
        Predictions.
    y_ref : np.ndarray
        Reference values.
    loss : {"huber", "mae"}, default="huber"
        Loss function:
        - "huber": Smooth L1-L2 hybrid (recommended)
        - "mae": Mean absolute error (L1)
    delta : float, optional
        Huber threshold. If None, uses 1.35 * MAD of residuals.
    max_iter : int, default=10
        Maximum IRLS iterations.
    tol : float, default=1e-4
        Convergence tolerance for weight change.

    Returns
    -------
    float
        Robust MSE estimate.

    Examples
    --------
    >>> y_pred = np.array([1, 2, 3, 10])  # Last value is outlier
    >>> y_ref = np.array([1.1, 2.1, 3.1, 3.5])
    >>> mse_robust = estimate_mse_robust(y_pred, y_ref, loss="huber")
    >>> mse_standard = np.mean((y_pred - y_ref)**2)
    >>> mse_robust < mse_standard
    True
    """
    y_pred = np.asarray(y_pred).ravel()
    y_ref = np.asarray(y_ref).ravel()

    if y_pred.shape != y_ref.shape:
        raise ValueError("y_pred and y_ref must have same shape")

    residual = y_pred - y_ref

    if loss == "mae":
        # Simple MAE
        return float(np.mean(np.abs(residual)))

    elif loss == "huber":
        # Iterative reweighted least squares
        if delta is None:
            # Use MAD-based threshold
            mad = np.median(np.abs(residual - np.median(residual)))
            delta = 1.35 * mad if mad > 0 else 1.0

        weights = np.ones_like(residual)

        for _ in range(max_iter):
            weights_old = weights.copy()

            # Compute Huber weights
            weights = huber_weight(residual, delta)

            # Weighted mean squared error
            wmse = np.average(residual**2, weights=weights)

            # Check convergence
            weight_change = np.mean(np.abs(weights - weights_old))
            if weight_change < tol:
                break

        return float(wmse)

    else:
        raise ValueError(f"Unknown loss: {loss}")


def detect_outliers(
    data: np.ndarray,
    method: Literal["iqr", "zscore", "mad", "isolation_forest"] = "iqr",
    threshold: float | None = None,
) -> np.ndarray:
    """
    Detect outliers using statistical tests.

    Parameters
    ----------
    data : np.ndarray
        Data array to check for outliers.
    method : {"iqr", "zscore", "mad", "isolation_forest"}, default="iqr"
        Detection method:
        - "iqr": Interquartile range (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
        - "zscore": Z-score > threshold (default 3.0)
        - "mad": Median absolute deviation > threshold (default 3.5)
        - "isolation_forest": Sklearn isolation forest (requires sklearn)
    threshold : float, optional
        Custom threshold for zscore/mad methods.

    Returns
    -------
    np.ndarray
        Boolean mask: True for outliers.

    Examples
    --------
    >>> data = np.array([1, 2, 2, 3, 3, 3, 4, 20])
    >>> outliers = detect_outliers(data, method="iqr")
    >>> np.where(outliers)[0]
    array([7])
    """
    data = np.asarray(data).ravel()

    if method == "iqr":
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (data < lower_bound) | (data > upper_bound)

    elif method == "zscore":
        if threshold is None:
            threshold = 3.0
        z_scores = np.abs(stats.zscore(data, nan_policy="omit"))
        outlier_mask = z_scores > threshold

    elif method == "mad":
        if threshold is None:
            threshold = 3.5
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad == 0:
            # All values identical
            outlier_mask = np.zeros(len(data), dtype=bool)
        else:
            modified_z_score = 0.6745 * (data - median) / mad
            outlier_mask = np.abs(modified_z_score) > threshold

    elif method == "isolation_forest":
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            raise ImportError(
                "isolation_forest method requires scikit-learn. "
                "Install with: pip install scikit-learn"
            )

        # Reshape for sklearn
        X = data.reshape(-1, 1)
        clf = IsolationForest(contamination=0.05, random_state=42)
        predictions = clf.fit_predict(X)
        outlier_mask = predictions == -1

    else:
        raise ValueError(f"Unknown method: {method}")

    return outlier_mask


def estimate_covariance_robust(
    X: np.ndarray,
    method: Literal["mcd", "oas", "ledoit_wolf"] = "mcd",
) -> np.ndarray:
    """
    Robustly estimate covariance matrix.

    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape (n_samples, n_features).
    method : {"mcd", "oas", "ledoit_wolf"}, default="mcd"
        Estimation method:
        - "mcd": Minimum Covariance Determinant (outlier-robust)
        - "oas": Oracle Approximating Shrinkage
        - "ledoit_wolf": Ledoit-Wolf shrinkage

    Returns
    -------
    np.ndarray
        Covariance matrix, shape (n_features, n_features).

    Notes
    -----
    Requires scikit-learn for MCD and OAS estimators.

    Examples
    --------
    >>> X = np.random.randn(100, 3)
    >>> X[::10] = 10.0  # Add outliers
    >>> Sigma_robust = estimate_covariance_robust(X, method="mcd")
    """
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional (n_samples, n_features)")

    try:
        from sklearn.covariance import (
            MinCovDet,
            OAS,
            LedoitWolf,
        )
    except ImportError:
        raise ImportError(
            "Robust covariance estimation requires scikit-learn. "
            "Install with: pip install scikit-learn"
        )

    if method == "mcd":
        estimator = MinCovDet().fit(X)
        cov = estimator.covariance_
    elif method == "oas":
        estimator = OAS().fit(X)
        cov = estimator.covariance_
    elif method == "ledoit_wolf":
        estimator = LedoitWolf().fit(X)
        cov = estimator.covariance_
    else:
        raise ValueError(f"Unknown method: {method}")

    return cov


def downweight_outliers(
    data: np.ndarray,
    outlier_mask: np.ndarray,
    downweight_factor: float = 0.1,
) -> np.ndarray:
    """
    Return weights that downweight detected outliers.

    Parameters
    ----------
    data : np.ndarray
        Data array.
    outlier_mask : np.ndarray
        Boolean mask: True for outliers.
    downweight_factor : float, default=0.1
        Weight for outliers (0=exclude, 1=no downweighting).

    Returns
    -------
    np.ndarray
        Weights in [downweight_factor, 1.0].

    Examples
    --------
    >>> data = np.array([1, 2, 3, 10])
    >>> outliers = np.array([False, False, False, True])
    >>> weights = downweight_outliers(data, outliers, downweight_factor=0.1)
    >>> weights
    array([1. , 1. , 1. , 0.1])
    """
    if downweight_factor < 0 or downweight_factor > 1:
        raise ValueError("downweight_factor must be in [0, 1]")

    weights = np.ones(len(data), dtype=float)
    weights[outlier_mask] = downweight_factor

    return weights


def robust_fusion_weights(
    mse: np.ndarray,
    data: np.ndarray,
    outlier_threshold: float = 3.0,
    downweight_factor: float = 0.1,
) -> np.ndarray:
    """
    Compute fusion weights with outlier detection and downweighting.

    Combines IVW with robust outlier handling.

    Parameters
    ----------
    mse : np.ndarray
        Mean squared errors per model, shape (K,).
    data : np.ndarray
        Original data for outlier detection, shape (n_samples, K).
    outlier_threshold : float, default=3.0
        Z-score threshold for outlier detection.
    downweight_factor : float, default=0.1
        Weight reduction for outlier-prone models.

    Returns
    -------
    np.ndarray
        Robust fusion weights, shape (K,).

    Examples
    --------
    >>> mse = np.array([0.5, 0.3, 0.8])
    >>> data = np.random.randn(100, 3)
    >>> data[:, 2] += np.random.choice([0, 10], size=100, p=[0.9, 0.1])
    >>> weights = robust_fusion_weights(mse, data)
    >>> # Model 2 (index 2) gets downweighted due to outliers
    """
    from .weights import solve_weights_ivw

    K = len(mse)

    if data.shape[1] != K:
        raise ValueError("data columns must match mse length")

    # Detect outliers per model
    outlier_counts = np.zeros(K)
    for k in range(K):
        outliers = detect_outliers(data[:, k], method="zscore", threshold=outlier_threshold)
        outlier_counts[k] = outliers.sum()

    # Downweight models with many outliers
    # Use outlier rate as downweight factor
    n_samples = data.shape[0]
    outlier_rates = outlier_counts / n_samples

    # Compute base IVW weights
    weights_ivw = solve_weights_ivw(mse, sum_to_one=True)

    # Apply outlier penalty
    outlier_penalty = 1.0 - outlier_rates * (1.0 - downweight_factor)
    weights_robust = weights_ivw * outlier_penalty

    # Renormalize
    weights_robust = weights_robust / weights_robust.sum()

    return weights_robust
