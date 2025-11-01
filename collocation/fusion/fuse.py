"""
High-Level Data Fusion Orchestrator
====================================

Main entry point for data fusion workflows. Coordinates all sub-modules:
weights, covariance, constraints, localization, uncertainty.

Examples
--------
>>> import numpy as np
>>> import xarray as xr
>>> from collocation.fusion import fuse_fields
>>>
>>> # Generate synthetic multi-model data
>>> X = xr.DataArray(
...     np.random.randn(100, 50, 50, 3) + 273.15,  # Three temperature models
...     dims=["time", "lat", "lon", "model"],
...     coords={"model": ["ERA5", "MERRA2", "JRA55"]},
... )
>>>
>>> # Reference data for MSE estimation
>>> X_ref = xr.DataArray(
...     np.random.randn(100, 50, 50) + 273.15,
...     dims=["time", "lat", "lon"],
... )
>>>
>>> # Simple IVW fusion
>>> result = fuse_fields(
...     X, X_ref=X_ref, mode="ivw",
...     return_weights=True, return_var=True
... )
>>> print(result.keys())  # ['fused', 'weights', 'variance', 'diagnostics']
>>>
>>> # GLS fusion with constraints
>>> from collocation.fusion import BoundsConstraint
>>> constraints = {"bounds": (0.0, 1.0)}
>>> result = fuse_fields(
...     X, X_ref=X_ref, mode="gls",
...     constraints=constraints,
...     shrinkage="lw",
...     return_weights=True, return_var=True
... )
>>>
>>> # Constrained QP with physics constraints
>>> result = fuse_fields(
...     X, X_ref=X_ref, mode="qp",
...     constraints={
...         "sum_to_one": True,
...         "bounds": (0.0, 1.0),
...     },
... )
"""

from __future__ import annotations

from typing import Dict, Literal, Any

import numpy as np
import xarray as xr

from .weights import solve_weights_ivw, solve_weights_gls, solve_weights_qp
from .covariance import estimate_mse, estimate_cross_covariance, build_sigma
from .constraints import (
    SumToOneConstraint,
    BoundsConstraint,
    combine_constraints,
)
from .uncertainty import (
    propagate_variance,
    compute_diagnostics,
)
from .localization import localize_weights


def fuse_fields(
    X: xr.DataArray,
    Sigma: xr.DataArray | None = None,
    X_ref: xr.DataArray | None = None,
    mode: Literal["ivw", "gls", "qp"] = "gls",
    constraints: Dict[str, Any] | None = None,
    localization: Dict[str, Any] | None = None,
    shrinkage: str = "lw",
    lam: float | None = None,
    robust: bool = False,
    return_weights: bool = True,
    return_var: bool = True,
    return_diagnostics: bool = True,
    min_models: int = 2,
) -> Dict[str, xr.DataArray]:
    """
    Fuse multi-model fields using optimal weighting.

    High-level orchestrator that handles:
    1. MSE/covariance estimation (if Sigma not provided)
    2. Weight computation (IVW, GLS, or constrained QP)
    3. Fusion: x̂ = Σ w_k x_k
    4. Uncertainty quantification
    5. Diagnostics

    Parameters
    ----------
    X : xr.DataArray
        Multi-model data, shape (..., model). Must have 'model' dimension.
    Sigma : xr.DataArray, optional
        Error covariance matrix, shape (..., model, model_2).
        If None, estimated from X and X_ref.
    X_ref : xr.DataArray, optional
        Reference data for MSE estimation. Required if Sigma is None.
    mode : {"ivw", "gls", "qp"}, default="gls"
        Fusion mode:
        - "ivw": Inverse variance weighting (diagonal Σ)
        - "gls": Generalized least squares (full Σ)
        - "qp": Constrained quadratic programming
    constraints : dict, optional
        Constraint specification for mode="qp":
        - "sum_to_one": bool (default True)
        - "bounds": tuple (w_min, w_max)
        - "A_eq", "b_eq": equality constraints
        - "A_ineq", "b_ineq": inequality constraints
    localization : dict, optional
        Localization strategy:
        - "strategy": "window" | "biome" | "climate"
        - "window": window specification (e.g., {"time": 30, "space": 7})
        - "biome_map": xr.DataArray for biome partitioning
    shrinkage : str, default="lw"
        Covariance shrinkage method: "lw", "oas", "ridge", "none".
    lam : float, optional
        Shrinkage intensity (auto if None).
    robust : bool, default=False
        Use robust MSE estimation (Huber loss).
    return_weights : bool, default=True
        Include fusion weights in output.
    return_var : bool, default=True
        Include fused variance in output.
    return_diagnostics : bool, default=True
        Include diagnostic metrics in output.
    min_models : int, default=2
        Minimum number of models required. If fewer, returns NaN.

    Returns
    -------
    dict
        Fusion results:
        - "fused": xr.DataArray - Fused field
        - "weights": xr.DataArray - Fusion weights (if return_weights)
        - "variance": xr.DataArray - Fused variance (if return_var)
        - "std": xr.DataArray - Fused standard deviation (if return_var)
        - "diagnostics": dict - Weight/uncertainty diagnostics (if return_diagnostics)

    Examples
    --------
    >>> # Simple IVW fusion
    >>> result = fuse_fields(X, X_ref=X_ref, mode="ivw")
    >>> fused = result["fused"]
    >>>
    >>> # GLS with localization
    >>> result = fuse_fields(
    ...     X, X_ref=X_ref, mode="gls",
    ...     localization={"strategy": "window", "window": {"space": 7}},
    ... )
    >>>
    >>> # Constrained QP
    >>> result = fuse_fields(
    ...     X, X_ref=X_ref, mode="qp",
    ...     constraints={"sum_to_one": True, "bounds": (0.1, 0.9)},
    ... )

    Raises
    ------
    ValueError
        If X lacks 'model' dimension or insufficient models available.
    """
    # Validate inputs
    if "model" not in X.dims:
        raise ValueError("X must have 'model' dimension")

    n_models = len(X.coords["model"])
    if n_models < min_models:
        raise ValueError(f"At least {min_models} models required, got {n_models}")

    # Step 1: Estimate covariance if not provided
    if Sigma is None:
        if X_ref is None:
            raise ValueError("Must provide either Sigma or X_ref for MSE estimation")

        print(f"Estimating MSE from data (robust={robust})...")
        mse = estimate_mse(X, X_ref, robust=robust)

        if mode in {"gls", "qp"}:
            # Estimate full covariance for GLS/QP
            # For simplicity, use sample covariance
            # In production, could use TC/IVD if n_models >= 3
            print("Estimating cross-covariance...")
            if n_models >= 3:
                cross = estimate_cross_covariance(X, X_ref, method="tc")
            else:
                cross = estimate_cross_covariance(X, X_ref, method="sample")

            Sigma = build_sigma(mse, cross=cross, shrinkage=shrinkage, lam=lam)
        else:
            # IVW: diagonal covariance
            Sigma = build_sigma(mse, cross=None, shrinkage=shrinkage, lam=lam)

    # Step 2: Compute weights
    print(f"Computing fusion weights (mode={mode})...")

    if localization is not None:
        # Localized weights
        strategy = localization.get("strategy", "window")
        weights = localize_weights(X, Sigma, strategy=strategy, **localization)

    else:
        # Global weights
        weights = _compute_global_weights(
            X, Sigma, mode=mode, constraints=constraints
        )

    # Step 3: Fuse
    print("Fusing fields...")
    fused = (X * weights).sum(dim="model")
    fused.name = "fused"
    fused.attrs["fusion_mode"] = mode
    fused.attrs["n_models"] = n_models

    # Step 4: Build output dictionary
    output = {"fused": fused}

    if return_weights:
        weights.name = "weights"
        output["weights"] = weights

    if return_var:
        print("Propagating uncertainty...")
        variance = propagate_variance(Sigma, weights)
        variance.name = "variance"
        output["variance"] = variance

        std = np.sqrt(variance)
        std.name = "std"
        output["std"] = std

    if return_diagnostics:
        print("Computing diagnostics...")
        diag = compute_diagnostics(weights, Sigma if return_var else None)
        output["diagnostics"] = diag

    return output


def _compute_global_weights(
    X: xr.DataArray,
    Sigma: xr.DataArray,
    mode: str,
    constraints: Dict[str, Any] | None,
) -> xr.DataArray:
    """
    Compute global (non-localized) fusion weights.

    Handles scalar or spatially varying Sigma.
    """
    n_models = len(X.coords["model"])

    # Determine if Sigma is scalar or spatially varying
    if Sigma.ndim == 2:
        # Scalar covariance: single weight vector
        Sigma_np = Sigma.values

        if mode == "ivw":
            # IVW: use diagonal of Sigma
            mse = np.diag(Sigma_np)
            w_np = solve_weights_ivw(mse, sum_to_one=True)

        elif mode == "gls":
            # GLS
            w_np = solve_weights_gls(Sigma_np, sum_to_one=True)

        elif mode == "qp":
            # Constrained QP
            w_np = _solve_qp_with_constraints(Sigma_np, n_models, constraints)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Wrap in xarray
        weights = xr.DataArray(
            w_np,
            dims=["model"],
            coords={"model": X.coords["model"]},
        )

    else:
        # Spatially varying covariance: compute weights per pixel
        batch_shape = Sigma.shape[:-2]
        batch_dims = [d for d in Sigma.dims if d not in ["model", "model_2"]]

        # Extract numpy arrays
        Sigma_np = Sigma.values  # (..., K, K)
        weights_np = np.zeros((*batch_shape, n_models))

        # Solve per pixel
        for idx in np.ndindex(batch_shape):
            S = Sigma_np[idx]

            if mode == "ivw":
                mse = np.diag(S)
                weights_np[idx] = solve_weights_ivw(mse, sum_to_one=True)

            elif mode == "gls":
                weights_np[idx] = solve_weights_gls(S, sum_to_one=True)

            elif mode == "qp":
                weights_np[idx] = _solve_qp_with_constraints(S, n_models, constraints)

        # Wrap in xarray
        all_dims = (*batch_dims, "model")
        coords = {d: Sigma.coords[d] for d in batch_dims}
        coords["model"] = X.coords["model"]

        weights = xr.DataArray(
            weights_np,
            dims=all_dims,
            coords=coords,
        )

    return weights


def _solve_qp_with_constraints(
    Sigma: np.ndarray,
    n_models: int,
    constraints: Dict[str, Any] | None,
) -> np.ndarray:
    """
    Solve constrained QP with user-specified constraints.

    Parameters
    ----------
    Sigma : np.ndarray
        Covariance matrix, shape (K, K).
    n_models : int
        Number of models.
    constraints : dict
        Constraint specification.

    Returns
    -------
    np.ndarray
        Optimal weights.
    """
    if constraints is None:
        constraints = {}

    # Build constraint objects
    constraint_list = []

    # Sum-to-one (default True)
    if constraints.get("sum_to_one", True):
        constraint_list.append(SumToOneConstraint(n_models))

    # Bounds
    if "bounds" in constraints:
        w_min, w_max = constraints["bounds"]
        constraint_list.append(BoundsConstraint(n_models, w_min, w_max))

    # Custom equality constraints
    A_eq = constraints.get("A_eq")
    b_eq = constraints.get("b_eq")

    # Custom inequality constraints
    A_ineq = constraints.get("A_ineq")
    b_ineq = constraints.get("A_ineq")

    # Combine constraints
    if constraint_list:
        A_combined, b_combined = combine_constraints(constraint_list)
    else:
        A_combined, b_combined = None, None

    # Solve QP
    weights = solve_weights_qp(
        Sigma,
        A_eq=A_combined if A_eq is None else A_eq,
        b_eq=b_combined if b_eq is None else b_eq,
        A_ineq=A_ineq,
        b_ineq=b_ineq,
        solver="quadprog",
    )

    return weights


def fuse_with_missing_data(
    X: xr.DataArray,
    Sigma: xr.DataArray,
    min_models: int = 2,
) -> xr.DataArray:
    """
    Fuse fields with missing data.

    Computes weights on available models per pixel/timestep.

    Parameters
    ----------
    X : xr.DataArray
        Multi-model data with possible NaN values.
    Sigma : xr.DataArray
        Error covariance matrix.
    min_models : int, default=2
        Minimum models required for fusion. Otherwise returns NaN.

    Returns
    -------
    xr.DataArray
        Fused field with NaN where insufficient models.

    Examples
    --------
    >>> X = xr.DataArray(
    ...     np.random.randn(10, 3),
    ...     dims=["time", "model"],
    ... )
    >>> X[0, 2] = np.nan  # Missing data
    >>> Sigma = xr.DataArray(np.eye(3) * 0.5, dims=["model", "model_2"])
    >>> fused = fuse_with_missing_data(X, Sigma, min_models=2)
    """
    # Detect valid models per sample
    valid_mask = ~np.isnan(X)
    n_valid = valid_mask.sum(dim="model")

    # Mask for fusion
    can_fuse = n_valid >= min_models

    # For simplicity, use GLS with available models
    # Full implementation would subset Sigma appropriately

    # Quick version: set weights of missing models to zero
    from .weights import solve_weights_gls

    Sigma_np = Sigma.values  # (K, K)
    weights = solve_weights_gls(Sigma_np)

    # Apply weights
    weights_xr = xr.DataArray(
        weights,
        dims=["model"],
        coords={"model": X.coords["model"]},
    )

    # Zero out weights for missing models
    weights_masked = weights_xr.where(valid_mask, 0.0)

    # Renormalize per sample
    weight_sums = weights_masked.sum(dim="model")
    weights_normalized = weights_masked / weight_sums

    # Fuse
    fused = (X * weights_normalized).sum(dim="model")

    # Mask where insufficient models
    fused = fused.where(can_fuse, np.nan)

    return fused
