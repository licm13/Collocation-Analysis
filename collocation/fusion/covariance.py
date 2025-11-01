"""
Covariance Matrix Estimation for Data Fusion
=============================================

Builds error covariance matrices from:
1. Mean Squared Error (MSE) estimates
2. Triple Collocation / IVD statistics
3. Shrinkage estimators (Ledoit-Wolf, OAS)
4. Covariance tapering (Gaspari-Cohn)

Supports spatially/temporally varying covariance with moving-window estimation.

Examples
--------
>>> import numpy as np
>>> import xarray as xr
>>> from collocation.fusion import estimate_mse, build_sigma
>>>
>>> # Generate synthetic predictions and reference
>>> y_pred = xr.DataArray(
...     np.random.randn(100, 50, 50, 3),  # time, lat, lon, model
...     dims=["time", "lat", "lon", "model"],
... )
>>> y_ref = xr.DataArray(
...     np.random.randn(100, 50, 50),
...     dims=["time", "lat", "lon"],
... )
>>>
>>> # Estimate MSE
>>> mse = estimate_mse(y_pred, y_ref)
>>> print(mse.dims)  # ('lat', 'lon', 'model')
>>>
>>> # Build covariance with shrinkage
>>> Sigma = build_sigma(mse, shrinkage="lw")
>>> print(Sigma.shape)  # (50, 50, 3, 3)
"""

from __future__ import annotations

from typing import Dict, Literal

import numpy as np
import xarray as xr
from scipy import linalg


def estimate_mse(
    y_pred: xr.DataArray,
    y_ref: xr.DataArray,
    window: Dict[str, int | str] | None = None,
    robust: bool = False,
) -> xr.DataArray:
    """
    Estimate mean squared error per model.

    Supports moving-window estimation for spatially/temporally varying MSE.

    Parameters
    ----------
    y_pred : xr.DataArray
        Predictions with shape (..., model). Must include 'model' dimension.
    y_ref : xr.DataArray
        Reference data, same shape as y_pred but without 'model' dimension.
    window : dict, optional
        Moving window specification, e.g., {"time": 30, "space": 7}.
        Keys can be dimension names or "space" (for lat/lon).
    robust : bool, default=False
        Use robust MSE estimator (Huber loss). See `fusion.robust` module.

    Returns
    -------
    xr.DataArray
        MSE per model, dimensions match y_pred but with averaging dims removed.

    Examples
    --------
    >>> y_pred = xr.DataArray(
    ...     np.random.randn(100, 3),
    ...     dims=["time", "model"],
    ...     coords={"model": ["A", "B", "C"]},
    ... )
    >>> y_ref = xr.DataArray(np.random.randn(100), dims=["time"])
    >>> mse = estimate_mse(y_pred, y_ref)
    >>> mse.dims
    ('model',)
    """
    if "model" not in y_pred.dims:
        raise ValueError("y_pred must have 'model' dimension")

    # Broadcast y_ref to match y_pred
    y_ref_expanded = y_ref.expand_dims({"model": y_pred.coords["model"]})

    # Compute squared error
    se = (y_pred - y_ref_expanded) ** 2

    if window is not None:
        # Moving window estimation
        from .localization import apply_moving_window

        mse = apply_moving_window(se, window, reduction="mean")
    else:
        # Global average over sampling dimensions.
        # By default, average over the 'time' dimension if present to preserve
        # spatial variability (e.g., 'lat', 'lon'). Otherwise average over
        # all non-model dimensions.
        if "time" in se.dims:
            avg_dims = ["time"]
        else:
            avg_dims = [d for d in se.dims if d != "model"]
        if robust:
            # Use robust estimator from fusion.robust
            from .robust import estimate_mse_robust

            # Convert to numpy for robust estimation, then back to xarray
            mse_values = np.array(
                [
                    estimate_mse_robust(
                        se.sel(model=m).values.ravel(),
                        np.zeros_like(se.sel(model=m).values.ravel()),
                    )
                    for m in se.coords["model"].values
                ]
            )
            mse = xr.DataArray(
                mse_values,
                dims=["model"],
                coords={"model": se.coords["model"]},
            )
        else:
            mse = se.mean(dim=avg_dims)

    mse.name = "mse"
    mse.attrs.update(
        {
            "long_name": "Mean Squared Error",
            "units": f"({y_pred.attrs.get('units', '?')})^2",
            "robust": robust,
        }
    )

    return mse


def estimate_cross_covariance(
    y_pred: xr.DataArray,
    y_ref: xr.DataArray,
    method: Literal["sample", "tc", "ivd"] = "sample",
) -> xr.DataArray:
    """
    Estimate error cross-covariance between models.

    Parameters
    ----------
    y_pred : xr.DataArray
        Predictions, shape (..., model).
    y_ref : xr.DataArray
        Reference data.
    method : {"sample", "tc", "ivd"}, default="sample"
        - "sample": Sample covariance of errors
        - "tc": Triple collocation estimates (if ≥3 models)
        - "ivd": IVD-based estimates (if =2 models)

    Returns
    -------
    xr.DataArray
        Cross-covariance matrix, shape (..., model, model_2).

    Examples
    --------
    >>> y_pred = xr.DataArray(
    ...     np.random.randn(100, 3),
    ...     dims=["time", "model"],
    ... )
    >>> y_ref = xr.DataArray(np.random.randn(100), dims=["time"])
    >>> cross_cov = estimate_cross_covariance(y_pred, y_ref, method="sample")
    >>> cross_cov.shape
    (3, 3)
    """
    if "model" not in y_pred.dims:
        raise ValueError("y_pred must have 'model' dimension")

    n_models = len(y_pred.coords["model"])

    # Broadcast y_ref
    y_ref_expanded = y_ref.expand_dims({"model": y_pred.coords["model"]})

    # Compute errors
    errors = y_pred - y_ref_expanded

    if method == "sample":
        # Sample covariance
        # Stack all non-model dims into 'sample'
        stack_dims = [d for d in errors.dims if d != "model"]
        if stack_dims:
            errors_stacked = errors.stack(sample=stack_dims)
        else:
            errors_stacked = errors

        # Compute covariance: E[(e_i - mean_i)(e_j - mean_j)]
        errors_centered = errors_stacked - errors_stacked.mean(dim="sample")
        cov_matrix = (errors_centered @ errors_centered.T) / errors_stacked.sizes[
            "sample"
        ]

        # Rename dims to avoid conflicts
        cov_matrix = cov_matrix.rename({"model": "model", "model": "model_2"})

    elif method == "tc":
        if n_models < 3:
            raise ValueError("Triple collocation requires at least 3 models")

        # Use TC to estimate error covariances
        from ..tc import tc

        # Convert to numpy: (n_samples, n_models)
        stack_dims = [d for d in errors.dims if d != "model"]
        if stack_dims:
            errors_np = errors.stack(sample=stack_dims).transpose("sample", "model").values
        else:
            errors_np = errors.values.reshape(-1, n_models)

        # Apply TC
        EeeT, _, _, _ = tc(errors_np)

        # Convert back to xarray
        model_coords = y_pred.coords["model"].values
        cov_matrix = xr.DataArray(
            EeeT,
            dims=["model", "model_2"],
            coords={"model": model_coords, "model_2": model_coords},
        )

    elif method == "ivd":
        if n_models != 2:
            raise ValueError("IVD requires exactly 2 models")

        from ..ivd import ivd

        # Convert to numpy
        stack_dims = [d for d in errors.dims if d != "model"]
        if stack_dims:
            errors_np = errors.stack(sample=stack_dims).transpose("sample", "model").values
        else:
            errors_np = errors.values.reshape(-1, 2)

        # Apply IVD
        EeeT, _, _ = ivd(errors_np)

        # Convert back to xarray
        model_coords = y_pred.coords["model"].values
        cov_matrix = xr.DataArray(
            EeeT,
            dims=["model", "model_2"],
            coords={"model": model_coords, "model_2": model_coords},
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    cov_matrix.name = "error_covariance"
    cov_matrix.attrs["method"] = method

    return cov_matrix


def build_sigma(
    mse: xr.DataArray,
    cross: xr.DataArray | None = None,
    shrinkage: str = "lw",
    lam: float | None = None,
) -> xr.DataArray:
    """
    Build error covariance matrix Σ from MSE and optional cross-covariance.

    Applies shrinkage for regularization and ensures positive definiteness.

    Parameters
    ----------
    mse : xr.DataArray
        Diagonal error variances, shape (..., model).
    cross : xr.DataArray, optional
        Off-diagonal cross-covariances, shape (..., model, model_2).
        If None, assumes diagonal covariance (independent errors).
    shrinkage : {"lw", "oas", "ridge", "none"}, default="lw"
        Shrinkage method:
        - "lw": Ledoit-Wolf toward scaled identity
        - "oas": Oracle Approximating Shrinkage (treated as LW here)
        - "ridge": Diagonal loading Σ + λI
        - "none": No shrinkage
    lam : float, optional
        Shrinkage intensity (0=no shrinkage, 1=complete shrinkage to target).
        If None, uses automatic heuristic based on condition number.

    Returns
    -------
    xr.DataArray
        Symmetric positive semi-definite covariance, shape (..., model, model_2).

    Examples
    --------
    >>> mse = xr.DataArray([0.5, 0.3, 0.8], dims=["model"])
    >>> Sigma = build_sigma(mse, shrinkage="ridge", lam=0.01)
    >>> Sigma.shape
    (3, 3)
    """
    if "model" not in mse.dims:
        raise ValueError("mse must have 'model' dimension")

    model_coords = mse.coords["model"].values
    K = len(model_coords)

    if cross is None:
        # Diagonal covariance
        batch_shape = mse.shape[:-1]
        if batch_shape:
            # Spatially varying
            Sigma_data = np.zeros((*batch_shape, K, K), dtype=float)
            for idx in np.ndindex(batch_shape):
                Sigma_data[idx] = np.diag(mse.values[idx])
        else:
            Sigma_data = np.diag(mse.values)

        # Build xarray with proper dims
        if batch_shape:
            batch_dims = [d for d in mse.dims if d != "model"]
            all_dims = (*batch_dims, "model", "model_2")
            batch_coords = {d: mse.coords[d] for d in batch_dims}
        else:
            all_dims = ("model", "model_2")
            batch_coords = {}

        Sigma = xr.DataArray(
            Sigma_data,
            dims=all_dims,
            coords={
                **batch_coords,
                "model": model_coords,
                "model_2": model_coords,
            },
        )

    else:
        # Use provided cross-covariance. Construct a full Sigma by
        # broadcasting the provided cross matrix to any batch dims present
        # in `mse` and overwriting the diagonal with per-model MSE values.
        batch_dims = [d for d in mse.dims if d != "model"]
        batch_shape = mse.shape[:-1]

        # Use broadcast helper to align/reindex and broadcast
        from .broadcast import prepare_cross_and_mse_for_sigma

        cross_np, mse_np, batch_dims, batch_coords = prepare_cross_and_mse_for_sigma(
            cross, mse, model_coords
        )

        all_dims = (*batch_dims, "model", "model_2") if batch_dims else ("model", "model_2")

        Sigma = xr.DataArray(
            cross_np,
            dims=all_dims,
            coords={**batch_coords, "model": model_coords, "model_2": model_coords},
        )

        # overwrite diagonal entries with mse
        for k in range(K):
            Sigma.data[..., k, k] = mse_np[..., k]

    # Apply shrinkage
    Sigma_shrunk = _apply_shrinkage_xr(Sigma, method=shrinkage, lam=lam)

    Sigma_shrunk.name = "Sigma"
    Sigma_shrunk.attrs.update(
        {
            "long_name": "Error Covariance Matrix",
            "shrinkage": shrinkage,
            "shrinkage_lambda": lam if lam is not None else "auto",
        }
    )

    return Sigma_shrunk


def _apply_shrinkage_xr(
    Sigma: xr.DataArray,
    method: str = "lw",
    lam: float | None = None,
) -> xr.DataArray:
    """Apply shrinkage to xarray covariance matrix."""
    method = method.lower()

    if method in {"none", ""}:
        return Sigma

    # Extract numpy array
    Sigma_np = Sigma.values

    # Determine if batched
    if Sigma.ndim == 2:
        # Single matrix
        Sigma_shrunk_np = _apply_shrinkage_np(Sigma_np, method, lam)
    else:
        # Batched: (..., K, K)
        batch_shape = Sigma.shape[:-2]
        K = Sigma.shape[-1]
        Sigma_flat = Sigma_np.reshape(-1, K, K)
        Sigma_shrunk_flat = np.zeros_like(Sigma_flat)

        for i in range(Sigma_flat.shape[0]):
            Sigma_shrunk_flat[i] = _apply_shrinkage_np(Sigma_flat[i], method, lam)

        Sigma_shrunk_np = Sigma_shrunk_flat.reshape(*batch_shape, K, K)

    # Wrap back in xarray
    Sigma_shrunk = xr.DataArray(
        Sigma_shrunk_np,
        dims=Sigma.dims,
        coords=Sigma.coords,
        attrs=Sigma.attrs,
    )

    return Sigma_shrunk


def _apply_shrinkage_np(
    matrix: np.ndarray,
    method: str,
    lam: float | None,
) -> np.ndarray:
    """Apply shrinkage to a single covariance matrix (numpy)."""
    # Symmetrize
    sym_matrix = 0.5 * (matrix + matrix.T)
    n = sym_matrix.shape[0]

    if method == "ridge":
        # Ridge: Σ + λI
        if lam is None:
            lam_val = 1e-6 * np.trace(sym_matrix) / max(n, 1)
        else:
            lam_val = float(lam)
        return sym_matrix + lam_val * np.eye(n)

    # Ledoit-Wolf / OAS style shrinkage
    cond = np.linalg.cond(sym_matrix)

    if lam is not None:
        shrink_intensity = float(np.clip(lam, 0.0, 1.0))
    else:
        # Auto-heuristic based on condition number
        if not np.isfinite(cond):
            shrink_intensity = 1.0
        else:
            cond_threshold = 1e6
            if cond <= cond_threshold:
                shrink_intensity = 0.0
            else:
                shrink_intensity = np.clip((cond - cond_threshold) / cond, 0.0, 1.0)

    # Target: scaled identity
    mu = np.trace(sym_matrix) / max(n, 1)
    target = mu * np.eye(n)

    shrunk = (1.0 - shrink_intensity) * sym_matrix + shrink_intensity * target

    # Ensure positive definiteness
    eigvals = np.linalg.eigvalsh(shrunk)
    if np.any(eigvals <= 0):
        eps = max(1e-9, -eigvals.min() + 1e-9)
        shrunk = shrunk + eps * np.eye(n)

    return shrunk


def apply_covariance_tapering(
    Sigma: xr.DataArray,
    distance: xr.DataArray,
    cutoff_radius: float,
    taper_function: Literal["gaspari_cohn", "exponential"] = "gaspari_cohn",
) -> xr.DataArray:
    """
    Apply covariance tapering to reduce spurious long-range correlations.

    Multiplies covariance by a compactly-supported correlation function.

    Parameters
    ----------
    Sigma : xr.DataArray
        Covariance matrix, shape (..., model, model_2).
    distance : xr.DataArray
        Pairwise distances between models, shape (model, model_2).
        Can be spatial distance, climate zone dissimilarity, etc.
    cutoff_radius : float
        Correlation cutoff distance. Correlations beyond this are set to zero.
    taper_function : {"gaspari_cohn", "exponential"}, default="gaspari_cohn"
        Tapering kernel:
        - "gaspari_cohn": 5th-order piecewise polynomial (compact support)
        - "exponential": exp(-distance/cutoff)

    Returns
    -------
    xr.DataArray
        Tapered covariance matrix.

    Examples
    --------
    >>> Sigma = xr.DataArray(np.eye(3) * 0.5, dims=["model", "model_2"])
    >>> distance = xr.DataArray(
    ...     [[0, 10, 50], [10, 0, 40], [50, 40, 0]],
    ...     dims=["model", "model_2"],
    ... )
    >>> Sigma_tapered = apply_covariance_tapering(Sigma, distance, cutoff_radius=30)

    References
    ----------
    Gaspari, G., & Cohn, S. E. (1999). Construction of correlation functions in
    two and three dimensions. QJRMS, 125(554), 723-757.
    """
    if taper_function == "gaspari_cohn":
        taper = _gaspari_cohn_taper(distance.values, cutoff_radius)
    elif taper_function == "exponential":
        taper = np.exp(-distance.values / cutoff_radius)
    else:
        raise ValueError(f"Unknown taper function: {taper_function}")

    # Wrap in xarray
    taper_xr = xr.DataArray(taper, dims=distance.dims, coords=distance.coords)

    # Element-wise multiplication
    Sigma_tapered = Sigma * taper_xr

    # Symmetrize
    Sigma_tapered = 0.5 * (Sigma_tapered + Sigma_tapered.transpose(..., "model_2", "model"))

    Sigma_tapered.attrs = Sigma.attrs.copy()
    Sigma_tapered.attrs["taper_function"] = taper_function
    Sigma_tapered.attrs["cutoff_radius"] = cutoff_radius

    return Sigma_tapered


def _gaspari_cohn_taper(distance: np.ndarray, cutoff: float) -> np.ndarray:
    """
    Gaspari-Cohn 5th-order piecewise polynomial taper.

    Compact support: zero beyond cutoff distance.

    Parameters
    ----------
    distance : np.ndarray
        Distance array.
    cutoff : float
        Cutoff radius (compact support).

    Returns
    -------
    np.ndarray
        Taper values in [0, 1].
    """
    r = np.abs(distance) / cutoff
    taper = np.zeros_like(r)

    # Region 1: 0 <= r < 1
    mask1 = r < 1
    r1 = r[mask1]
    taper[mask1] = (
        1.0
        - (5.0 / 3.0) * r1**2
        + (5.0 / 8.0) * r1**3
        + (1.0 / 2.0) * r1**4
        - (1.0 / 4.0) * r1**5
    )

    # Region 2: 1 <= r < 2
    mask2 = (r >= 1) & (r < 2)
    r2 = r[mask2]
    taper[mask2] = (
        (4.0 - r2)
        - (5.0 / 3.0) * (2.0 - r2) ** 2
        + (5.0 / 8.0) * (2.0 - r2) ** 3
        + (1.0 / 2.0) * (2.0 - r2) ** 4
        - (1.0 / 4.0) * (2.0 - r2) ** 5
    )

    # Region 3: r >= 2 → taper = 0 (already initialized)

    return taper
