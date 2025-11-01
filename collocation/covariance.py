"""Utilities for constructing covariance matrices from collocation outputs.

This module provides helper routines for translating the outputs of the
collocation algorithms that live in :mod:`collocation` into covariance
matrices that can be consumed by downstream fusion algorithms.  The primary
entry point is :func:`build_sigma_from_collocation`, which inspects the
provided :class:`xarray.Dataset` and returns an :class:`xarray.DataArray`
representing the error covariance matrix.

The implementation is intentionally defensive – the historic MATLAB code that
this repository is based on produces several different naming conventions for
variance, covariance and bias fields.  The helper therefore accepts a wide
range of variable names and performs light sanity checks to guarantee that the
returned covariance matrix is symmetric positive semi-definite.  Optional
shrinkage is supported to stabilise nearly singular matrices.
"""

from __future__ import annotations

import logging
from typing import Iterable, Mapping

import numpy as np
import xarray as xr


LOGGER = logging.getLogger(__name__)


def _find_first_variable(ds: xr.Dataset, names: Iterable[str]) -> xr.DataArray | None:
    """Return the first data variable present in *ds* that matches ``names``.

    Parameters
    ----------
    ds:
        Input dataset produced by one of the collocation routines.
    names:
        Ordered list of potential variable names.

    Returns
    -------
    :class:`xarray.DataArray` or ``None``
        The matching data array, or ``None`` if none are present.
    """

    for name in names:
        if name in ds:
            return ds[name]
    return None


def _maybe_align_with_mapping(da: xr.DataArray, mapping: Mapping[str, object] | None) -> xr.DataArray:
    """Align *da* according to *mapping*.

    The ``mapping`` argument is intentionally flexible to support a variety of
    user supplied alignment hints:

    - ``{"model": ["m2", "m0", "m1"]}`` – reorders the coordinate values
      along the ``"model"`` dimension.
    - ``{"model": "product"}`` – renames the dimension ``"model"`` to
      ``"product"``.
    - ``{"model": {"A": "AMSR2"}}`` – renames individual coordinate labels.

    Parameters
    ----------
    da:
        DataArray to be aligned.
    mapping:
        Optional mapping specification.

    Returns
    -------
    :class:`xarray.DataArray`
        Aligned DataArray (``da`` is returned untouched if ``mapping`` is
        ``None``).
    """

    if not mapping:
        return da

    result = da
    for dim, instruction in mapping.items():
        if dim not in result.dims and not (
            isinstance(instruction, str) and instruction in result.dims
        ):
            # Nothing to do for this entry.
            continue

        if isinstance(instruction, str):
            # Rename dimension ``dim`` -> ``instruction`` if possible.
            if dim in result.dims:
                result = result.rename({dim: instruction})
            continue

        if isinstance(instruction, Mapping):
            # Rename coordinate values.
            existing_dim = dim if dim in result.dims else list(result.dims)[0]
            coord = result[existing_dim]
            new_labels = [instruction.get(value, value) for value in coord.values]
            result = result.assign_coords({existing_dim: new_labels})
            continue

        # Treat any iterable (lists, tuples, numpy arrays, DataArray indices)
        # as an explicit reindexing instruction.
        if isinstance(instruction, (list, tuple, np.ndarray, xr.DataArray)):
            key = dim if dim in result.dims else list(result.dims)[0]
            result = result.sel({key: list(instruction)})
            continue

        raise TypeError(
            "Unsupported mapping instruction for dimension %s: %r" % (dim, instruction)
        )

    return result


def _apply_shrinkage(matrix: np.ndarray, shrinkage: str, lam: float | None) -> np.ndarray:
    """Apply the requested shrinkage scheme to *matrix*.

    The implementation provides lightweight Ledoit-Wolf and OAS style shrinkage
    by shrinking the matrix towards a scaled identity.  The shrinkage intensity
    can be supplied explicitly via ``lam``; otherwise it is heuristically
    derived from the condition number of ``matrix``.
    """

    shrinkage = (shrinkage or "").lower()
    if shrinkage not in {"", "none", "lw", "oas", "ridge"}:
        raise ValueError(f"Unsupported shrinkage method: {shrinkage}")

    sym_matrix = 0.5 * (matrix + matrix.T)
    n = sym_matrix.shape[0]

    if shrinkage in {"", "none"}:
        return sym_matrix

    if shrinkage == "ridge":
        # Ridge regression style diagonal loading Σ + λI.
        lam_value = float(lam) if lam is not None else 1e-6 * np.trace(sym_matrix) / max(n, 1)
        return sym_matrix + lam_value * np.eye(n, dtype=sym_matrix.dtype)

    # Ledoit-Wolf / OAS style shrinkage towards a scaled identity matrix.
    # Without access to the raw residuals we fall back to a simple heuristic:
    # determine the shrinkage intensity from the condition number so that
    # severely ill-conditioned matrices are regularised more aggressively.
    cond = np.linalg.cond(sym_matrix)
    if lam is not None:
        shrink_intensity = float(np.clip(lam, 0.0, 1.0))
    else:
        # ``cond`` can be infinite if the matrix is singular.  Clip the value
        # to a reasonable range and compute a shrinkage factor that linearly
        # increases from 0 once the condition number exceeds 1e6.
        if not np.isfinite(cond):
            shrink_intensity = 1.0
        else:
            cond_threshold = 1e6
            if cond <= cond_threshold:
                shrink_intensity = 0.0
            else:
                shrink_intensity = np.clip((cond - cond_threshold) / cond, 0.0, 1.0)

    mu = np.trace(sym_matrix) / max(n, 1)
    target = mu * np.eye(n, dtype=sym_matrix.dtype)
    shrunk = (1.0 - shrink_intensity) * sym_matrix + shrink_intensity * target

    # Guard against numerical drift that could introduce negative eigenvalues.
    eigvals = np.linalg.eigvalsh(shrunk)
    if np.any(eigvals <= 0):
        eps = max(1e-9, -eigvals.min() + 1e-9)
        LOGGER.warning(
            "Covariance matrix not positive definite; applying %.3e diagonal loading.",
            eps,
        )
        shrunk = shrunk + eps * np.eye(n, dtype=shrunk.dtype)

    return shrunk


def build_sigma_from_collocation(
    tc: xr.Dataset,
    mapping: Mapping[str, object] | None = None,
    shrinkage: str = "lw",
    lam: float | None = None,
) -> xr.DataArray:
    """Construct an error covariance matrix from collocation results.

    Parameters
    ----------
    tc:
        :class:`xarray.Dataset` containing collocation diagnostics.  The
        function understands the following variable names (first match wins):

        ``cov`` | ``covariance`` | ``Sigma`` – full covariance matrices.

        ``var`` | ``variance`` | ``sigma2`` – per model error variances (used to
        construct a diagonal covariance matrix if cross terms are not
        available).

    mapping:
        Optional mapping that aligns the resulting covariance matrix with the
        ordering expected by downstream fusion components.  See
        :func:`_maybe_align_with_mapping` for the supported formats.

    shrinkage:
        Regularisation strategy applied to the covariance matrix.  Supported
        values are ``"lw"`` (Ledoit-Wolf style shrinkage), ``"oas"`` (treated as
        an alias of ``"lw"``) and ``"ridge"`` (diagonal loading ``Σ + λI``).

    lam:
        Optional shrinkage intensity.  For ``"ridge"`` the value is interpreted
        as the diagonal loading factor ``λ``.  For the other methods it controls
        the shrinkage intensity (``0`` – no shrinkage, ``1`` – complete shrinkage
        to the identity target).  When omitted a heuristic based on the
        condition number is used.

    Returns
    -------
    :class:`xarray.DataArray`
        Symmetric positive semi-definite covariance matrix aligned according to
        ``mapping`` (if provided).
    """

    if not isinstance(tc, xr.Dataset):
        raise TypeError("'tc' must be an xarray.Dataset instance")

    cov_da = _find_first_variable(tc, ["cov", "covariance", "Sigma", "sigma"])
    var_da = _find_first_variable(tc, ["var", "variance", "var_diag", "sigma2", "error_var"])

    if cov_da is None and var_da is None:
        raise ValueError("Dataset does not contain variance or covariance information")

    if cov_da is not None:
        if cov_da.ndim != 2:
            raise ValueError("Covariance array must be two-dimensional")
        if cov_da.shape[0] != cov_da.shape[1]:
            raise ValueError("Covariance matrix must be square")
        sigma = cov_da.astype(float)
    else:
        if var_da is None or var_da.ndim != 1:
            raise ValueError("Variance array must be one-dimensional when covariance is absent")
        dim = var_da.dims[0] if var_da.dims else "model"
        coords = {dim: var_da[dim] if dim in var_da.coords else np.arange(var_da.size)}
        diag_matrix = np.diag(var_da.astype(float).values)
        sigma = xr.DataArray(diag_matrix, coords=coords, dims=(dim, dim), name="Sigma")

    sigma = sigma.rename("Sigma")
    sigma = 0.5 * (sigma + sigma.transpose(..., sigma.dims[::-1]))

    shrunk = _apply_shrinkage(sigma.to_numpy(), shrinkage=shrinkage, lam=lam)
    sigma = xr.DataArray(shrunk, coords=sigma.coords, dims=sigma.dims, name="Sigma")

    sigma = _maybe_align_with_mapping(sigma, mapping)
    return sigma


