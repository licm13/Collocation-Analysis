"""Helpers to align and broadcast cross-covariance and mse arrays for Sigma construction.

This module provides a focused utility used by `build_sigma` to:
- align xarray `cross` arrays to have ('model','model_2') as their last two dims,
- reindex `cross` to match `mse` model labels when possible,
- determine a target batch shape and broadcast both `cross` and `mse` to that shape.

The function returns numpy arrays ready for diagonal replacement and the batch dims/coords
needed to wrap the result back into an xarray DataArray.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import xarray as xr


def prepare_cross_and_mse_for_sigma(
    cross, mse: xr.DataArray, model_coords: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...], dict]:
    """Align and broadcast `cross` and `mse` to a common batch shape.

    Parameters
    ----------
    cross : xr.DataArray | np.ndarray
        Cross-covariance provided by the user. Can be a 2D numpy array (K,K)
        or an xarray DataArray with batch dims and final dims ('model','model_2').
    mse : xr.DataArray
        MSE array with dims (..., 'model') or 1D over 'model'.
    model_coords : sequence
        Ordered list of model labels to use for reindexing/alignment.

    Returns
    -------
    cross_np : np.ndarray
        Broadcasted cross array with shape (*batch_shape, K, K).
    mse_np : np.ndarray
        Broadcasted mse array with shape (*batch_shape, K).
    batch_dims : tuple[str, ...]
        The names of batch dims used for the resulting Sigma (in order).
    batch_coords : dict
        Mapping from batch dim name to coordinates (xarray-friendly) used for Sigma.
    """
    K = len(model_coords)

    # Normalize cross to numpy and align labels if xarray
    if isinstance(cross, xr.DataArray):
        if not ("model" in cross.dims and "model_2" in cross.dims):
            raise ValueError("cross must contain 'model' and 'model_2' dimensions")

        batch_dims_cross = [d for d in cross.dims if d not in ("model", "model_2")]
        desired_order = tuple(batch_dims_cross) + ("model", "model_2")
        try:
            cross = cross.transpose(*desired_order)
        except Exception:
            pass

        try:
            cross = cross.reindex(model=model_coords, model_2=model_coords)
        except Exception:
            pass

        cross_np = cross.values
    else:
        cross_np = np.asarray(cross)

    cross_batch_shape = cross_np.shape[:-2] if cross_np.ndim >= 3 else ()
    mse_batch_shape = mse.shape[:-1]
    target_batch_shape = cross_batch_shape if cross_batch_shape else mse_batch_shape

    # Validate last two dims square when present
    if cross_np.ndim >= 2:
        if cross_np.shape[-2] != cross_np.shape[-1]:
            # allow reindexing behavior; caller can interpret NaNs
            # but raise only if absolutely inconsistent and no batch dims
            if not target_batch_shape:
                raise ValueError("cross must be square in its last two dimensions")

    # Broadcast cross to target batch shape
    if cross_np.ndim == 2 and target_batch_shape:
        cross_np = np.broadcast_to(cross_np, (*target_batch_shape, K, K)).copy()
    elif cross_np.ndim >= 3 and cross_np.shape[:-2] != tuple(target_batch_shape):
        cross_np = np.broadcast_to(cross_np, (*target_batch_shape, K, K)).copy()
    else:
        cross_np = cross_np.copy()

    # Broadcast mse to target batch shape
    mse_np = mse.values
    if target_batch_shape and mse_np.ndim == 1:
        mse_np = np.broadcast_to(mse_np, (*target_batch_shape, K)).copy()

    # Determine batch dims and coords to use for wrapping into xarray
    if target_batch_shape:
        if mse.shape[:-1]:
            batch_dims = tuple(d for d in mse.dims if d != "model")
            batch_coords = {d: mse.coords[d] for d in batch_dims}
        elif isinstance(cross, xr.DataArray):
            batch_dims = tuple(d for d in cross.dims if d not in ("model", "model_2"))
            batch_coords = {d: cross.coords[d] for d in batch_dims}
        else:
            batch_dims = tuple(f"dim_{i}" for i in range(len(target_batch_shape)))
            batch_coords = {d: np.arange(s) for d, s in zip(batch_dims, target_batch_shape)}
    else:
        batch_dims = tuple()
        batch_coords = {}

    return cross_np, mse_np, batch_dims, batch_coords
