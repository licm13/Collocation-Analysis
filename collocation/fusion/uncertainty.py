"""
Uncertainty Quantification for Data Fusion
===========================================

Provides uncertainty estimates for fused products:

1. **Variance propagation**: Var(x̂) = w^T Σ w (analytical)
2. **Bootstrap CI**: Empirical confidence intervals
3. **Effective sample size**: Accounts for weight concentration
4. **Weight diagnostics**: Entropy, concentration metrics

Examples
--------
>>> import numpy as np
>>> from collocation.fusion import (
...     propagate_variance,
...     compute_effective_n,
...     bootstrap_uncertainty,
... )
>>>
>>> # Analytical variance
>>> Sigma = np.array([[0.5, 0.1], [0.1, 0.3]])
>>> weights = np.array([0.6, 0.4])
>>> var_fused = propagate_variance(Sigma, weights)
>>> print(f"Fused std: {np.sqrt(var_fused):.3f}")
>>>
>>> # Effective sample size
>>> n_eff = compute_effective_n(weights)
>>> print(f"Effective N: {n_eff:.2f}")  # ≤ 2
>>>
>>> # Bootstrap CI
>>> data = np.random.randn(100, 2)
>>> ci = bootstrap_uncertainty(data, weights, n_boot=1000, alpha=0.05)
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import xarray as xr


def propagate_variance(
    Sigma: np.ndarray | xr.DataArray,
    weights: np.ndarray | xr.DataArray,
) -> np.ndarray | xr.DataArray:
    """
    Analytically propagate variance through fusion: Var(x̂) = w^T Σ w.

    Parameters
    ----------
    Sigma : np.ndarray or xr.DataArray
        Error covariance matrix, shape (K, K) or (..., K, K).
    weights : np.ndarray or xr.DataArray
        Fusion weights, shape (K,) or (..., K).

    Returns
    -------
    np.ndarray or xr.DataArray
        Fused variance, scalar or shape (...,).

    Examples
    --------
    >>> Sigma = np.array([[0.5, 0.0], [0.0, 0.3]])
    >>> w = np.array([0.7, 0.3])
    >>> var = propagate_variance(Sigma, w)
    >>> np.sqrt(var)  # Fused standard deviation
    0.47...
    """
    if isinstance(Sigma, xr.DataArray) and isinstance(weights, xr.DataArray):
        # xarray version
        var = (weights * (Sigma @ weights)).sum(dim="model")
        return var

    elif isinstance(Sigma, np.ndarray) and isinstance(weights, np.ndarray):
        # numpy version
        if Sigma.ndim == 2:
            # Single matrix
            var = weights @ Sigma @ weights
        else:
            # Batched: (..., K, K)
            var = np.einsum("...i,...ij,...j->...", weights, Sigma, weights)
        return var

    else:
        raise TypeError("Sigma and weights must both be numpy arrays or xarray DataArrays")


def compute_effective_n(
    weights: np.ndarray | xr.DataArray,
    method: str = "kish",
) -> np.ndarray | xr.DataArray | float:
    """
    Compute effective sample size from fusion weights.

    Quantifies how concentrated weights are. Uniform weights → N_eff = K.
    One weight = 1, rest = 0 → N_eff = 1.

    Parameters
    ----------
    weights : np.ndarray or xr.DataArray
        Fusion weights, shape (K,) or (..., K). Must sum to 1.
    method : {"kish", "entropy"}, default="kish"
        Calculation method:
        - "kish": N_eff = (Σw)^2 / Σw^2 = 1 / Σw^2
        - "entropy": N_eff = exp(H) where H = -Σw log(w)

    Returns
    -------
    float or np.ndarray or xr.DataArray
        Effective sample size. Range: [1, K].

    Examples
    --------
    >>> # Uniform weights
    >>> w_uniform = np.array([0.25, 0.25, 0.25, 0.25])
    >>> compute_effective_n(w_uniform)
    4.0

    >>> # Concentrated weights
    >>> w_concentrated = np.array([0.9, 0.05, 0.03, 0.02])
    >>> compute_effective_n(w_concentrated)
    1.24...
    """
    if method == "kish":
        # Kish's effective sample size
        # N_eff = 1 / Σw_i^2
        if isinstance(weights, xr.DataArray):
            n_eff = 1.0 / (weights**2).sum(dim="model")
        else:
            n_eff = 1.0 / np.sum(weights**2, axis=-1)

    elif method == "entropy":
        # Entropy-based: N_eff = exp(H)
        # H = -Σ w_i log(w_i)
        # Avoid log(0)
        eps = 1e-15

        if isinstance(weights, xr.DataArray):
            w_safe = weights.where(weights > eps, eps)
            entropy = -(weights * np.log(w_safe)).sum(dim="model")
            n_eff = np.exp(entropy)
        else:
            w_safe = np.where(weights > eps, weights, eps)
            entropy = -np.sum(weights * np.log(w_safe), axis=-1)
            n_eff = np.exp(entropy)

    else:
        raise ValueError(f"Unknown method: {method}")

    return n_eff


def bootstrap_uncertainty(
    data: np.ndarray,
    weights: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> Dict[str, float]:
    """
    Estimate fusion uncertainty via bootstrap resampling.

    Parameters
    ----------
    data : np.ndarray
        Original data, shape (n_samples, n_models).
    weights : np.ndarray
        Fusion weights, shape (n_models,).
    n_boot : int, default=1000
        Number of bootstrap samples.
    alpha : float, default=0.05
        Significance level for confidence intervals (1-alpha coverage).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Bootstrap statistics:
        - "mean": Bootstrap mean of fused values
        - "std": Bootstrap standard deviation
        - "ci_lower": Lower CI bound
        - "ci_upper": Upper CI bound
        - "bias": Bootstrap bias estimate

    Examples
    --------
    >>> data = np.random.randn(100, 3)
    >>> weights = np.array([0.5, 0.3, 0.2])
    >>> ci = bootstrap_uncertainty(data, weights, n_boot=500)
    >>> print(f"95% CI: [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")
    """
    if data.ndim != 2:
        raise ValueError("data must be 2D (n_samples, n_models)")

    n_samples, n_models = data.shape

    if len(weights) != n_models:
        raise ValueError("weights length must match n_models")

    rng = np.random.default_rng(random_state)

    # Original fused estimate
    fused_orig = data @ weights

    # Bootstrap
    boot_estimates = np.zeros(n_boot)

    for i in range(n_boot):
        # Resample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        data_boot = data[indices]

        # Fuse
        fused_boot = data_boot @ weights

        # Store mean or another statistic
        boot_estimates[i] = fused_boot.mean()

    # Compute statistics
    boot_mean = np.mean(boot_estimates)
    boot_std = np.std(boot_estimates, ddof=1)

    # Confidence intervals (percentile method)
    ci_lower = np.percentile(boot_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(boot_estimates, 100 * (1 - alpha / 2))

    # Bias estimate
    bias = boot_mean - fused_orig.mean()

    return {
        "mean": float(boot_mean),
        "std": float(boot_std),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "bias": float(bias),
    }


def weight_entropy(
    weights: np.ndarray | xr.DataArray,
) -> np.ndarray | xr.DataArray | float:
    """
    Compute Shannon entropy of weight distribution.

    H = -Σ w_i log(w_i)

    High entropy → uniform weights (good diversity).
    Low entropy → concentrated weights (one model dominates).

    Parameters
    ----------
    weights : np.ndarray or xr.DataArray
        Fusion weights, shape (K,) or (..., K).

    Returns
    -------
    float or np.ndarray or xr.DataArray
        Entropy in nats. Range: [0, log(K)].

    Examples
    --------
    >>> w_uniform = np.array([0.25, 0.25, 0.25, 0.25])
    >>> entropy = weight_entropy(w_uniform)
    >>> np.exp(entropy)  # Perplexity = 4
    4.0

    >>> w_concentrated = np.array([1.0, 0.0, 0.0, 0.0])
    >>> weight_entropy(w_concentrated)
    0.0
    """
    eps = 1e-15

    if isinstance(weights, xr.DataArray):
        w_safe = weights.where(weights > eps, eps)
        H = -(weights * np.log(w_safe)).sum(dim="model")
    else:
        w_safe = np.where(weights > eps, weights, eps)
        H = -np.sum(weights * np.log(w_safe), axis=-1)

    return H


def weight_concentration(
    weights: np.ndarray | xr.DataArray,
) -> np.ndarray | xr.DataArray | float:
    """
    Compute weight concentration index (Herfindahl-Hirschman Index).

    HHI = Σ w_i^2

    High HHI → concentrated (one model dominates).
    Low HHI → dispersed (many models contribute).

    Parameters
    ----------
    weights : np.ndarray or xr.DataArray
        Fusion weights, shape (K,) or (..., K).

    Returns
    -------
    float or np.ndarray or xr.DataArray
        HHI. Range: [1/K, 1].

    Examples
    --------
    >>> w_uniform = np.array([0.25, 0.25, 0.25, 0.25])
    >>> weight_concentration(w_uniform)
    0.25

    >>> w_concentrated = np.array([1.0, 0.0, 0.0, 0.0])
    >>> weight_concentration(w_concentrated)
    1.0
    """
    if isinstance(weights, xr.DataArray):
        hhi = (weights**2).sum(dim="model")
    else:
        hhi = np.sum(weights**2, axis=-1)

    return hhi


def compute_diagnostics(
    weights: np.ndarray | xr.DataArray,
    Sigma: np.ndarray | xr.DataArray | None = None,
) -> Dict[str, np.ndarray | xr.DataArray | float]:
    """
    Compute comprehensive weight and uncertainty diagnostics.

    Parameters
    ----------
    weights : np.ndarray or xr.DataArray
        Fusion weights, shape (K,) or (..., K).
    Sigma : np.ndarray or xr.DataArray, optional
        Error covariance for variance propagation.

    Returns
    -------
    dict
        Diagnostics:
        - "effective_n": Effective sample size
        - "entropy": Shannon entropy
        - "concentration": HHI concentration index
        - "variance": Fused variance (if Sigma provided)
        - "std": Fused standard deviation (if Sigma provided)

    Examples
    --------
    >>> weights = np.array([0.6, 0.3, 0.1])
    >>> Sigma = np.diag([0.5, 0.3, 0.8])
    >>> diag = compute_diagnostics(weights, Sigma)
    >>> print(diag["effective_n"])
    2.17...
    """
    diagnostics = {}

    diagnostics["effective_n"] = compute_effective_n(weights)
    diagnostics["entropy"] = weight_entropy(weights)
    diagnostics["concentration"] = weight_concentration(weights)

    if Sigma is not None:
        var = propagate_variance(Sigma, weights)
        diagnostics["variance"] = var

        if isinstance(var, xr.DataArray):
            diagnostics["std"] = np.sqrt(var)
        else:
            diagnostics["std"] = np.sqrt(var)

    return diagnostics


def delta_method_variance(
    Sigma: np.ndarray,
    weights: np.ndarray,
    constraints: np.ndarray | None = None,
) -> float:
    """
    Approximate variance for constrained fusion via delta method.

    When constraints beyond sum-to-one are active, the simple formula
    Var(x̂) = w^T Σ w may not hold. Delta method provides first-order
    approximation.

    Parameters
    ----------
    Sigma : np.ndarray
        Error covariance, shape (K, K).
    weights : np.ndarray
        Optimal weights from constrained QP, shape (K,).
    constraints : np.ndarray, optional
        Active constraint matrix, shape (n_active, K).

    Returns
    -------
    float
        Approximate fused variance.

    Notes
    -----
    This is a simplified implementation. Full delta method would require
    derivatives of the QP solution w.r.t. data.

    For standard sum-to-one BLUE: Var(x̂) = (1^T Σ^{-1} 1)^{-1}.
    For general constraints: more complex, use bootstrap or approximation.
    """
    # For now, just use standard propagation
    # Full implementation would account for constraint curvature
    var = propagate_variance(Sigma, weights)

    if constraints is not None:
        # Apply correction factor (heuristic)
        # Number of active constraints reduces effective freedom
        n_active = constraints.shape[0] if constraints.ndim > 1 else 1
        correction = 1.0 + 0.1 * n_active  # Simple penalty
        var = var * correction

    return float(var)


def jackknife_variance(
    data: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Estimate fusion variance via jackknife resampling.

    Leave-one-out variance estimator.

    Parameters
    ----------
    data : np.ndarray
        Data matrix, shape (n_samples, n_models).
    weights : np.ndarray
        Fusion weights, shape (n_models,).

    Returns
    -------
    float
        Jackknife variance estimate.

    Examples
    --------
    >>> data = np.random.randn(100, 3)
    >>> weights = np.array([0.5, 0.3, 0.2])
    >>> var_jack = jackknife_variance(data, weights)
    """
    n_samples = data.shape[0]

    # Original estimate
    fused_full = (data @ weights).mean()

    # Jackknife estimates
    jack_estimates = np.zeros(n_samples)

    for i in range(n_samples):
        # Leave out sample i
        data_loo = np.delete(data, i, axis=0)
        fused_loo = (data_loo @ weights).mean()
        jack_estimates[i] = fused_loo

    # Jackknife variance
    # Var_jack = (n-1)/n * Σ(θ_i - θ̄)^2
    jack_mean = jack_estimates.mean()
    var_jack = (
        (n_samples - 1) / n_samples * np.sum((jack_estimates - jack_mean) ** 2)
    )

    return float(var_jack)
