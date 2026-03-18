"""
Base Estimator
==============

Abstract base class for all collocation estimators, providing a
scikit-learn-style interface: ``estimator.fit(data)`` then access
results via ``estimator.metrics_``.

Design goals
------------
- Uniform API across all collocation methods
- Drop-in compatibility with scikit-learn Pipelines
- xarray-aware: accepts DataArray / Dataset in addition to ndarray
- NaN-safe preprocessing built in

Author: Claude
Date: 2026-03-18
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

try:
    import xarray as xr
    XR_AVAILABLE = True
except ImportError:
    XR_AVAILABLE = False
    xr = None


class CollocationEstimator(ABC):
    """
    Abstract base class for all collocation estimators.

    Subclasses must implement ``_fit(data)`` which receives a cleaned
    ``np.ndarray`` and stores results into ``self.metrics_``.

    Parameters
    ----------
    dropna : bool, default=True
        Drop rows that contain any NaN / Inf value before fitting.
        If False, NaN propagation is left to the concrete estimator.
    min_samples : int, default=50
        Emit a ``UserWarning`` if the (post-cleaning) sample count is below
        this threshold.

    Attributes
    ----------
    metrics_ : dict
        Populated after ``fit()``.  Keys vary by estimator; always contains
        at least ``'n_samples'`` and ``'is_fitted'``.
    n_samples_ : int
        Number of samples used for fitting (after NaN removal).
    is_fitted_ : bool
        True after a successful ``fit()`` call.
    """

    def __init__(self, dropna: bool = True, min_samples: int = 50) -> None:
        self.dropna = dropna
        self.min_samples = min_samples
        self.metrics_: Dict[str, Any] = {}
        self.n_samples_: int = 0
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, data: Any) -> "CollocationEstimator":
        """
        Fit the estimator on *data*.

        Parameters
        ----------
        data : array-like, shape (n, k)
            Input data.  Accepts ``np.ndarray``, list-of-lists,
            ``xarray.DataArray`` (converted via ``.values``), and
            ``xarray.Dataset`` (stacked column-wise from data variables).

        Returns
        -------
        self : CollocationEstimator
            Fitted estimator (enables method chaining).
        """
        arr = self._coerce(data)
        arr = self._clean(arr)
        self.n_samples_ = arr.shape[0]
        self._fit(arr)
        self.metrics_["n_samples"] = self.n_samples_
        self.metrics_["is_fitted"] = True
        self.is_fitted_ = True
        return self

    def get_metrics(self) -> Dict[str, Any]:
        """Return a copy of the metrics dictionary."""
        self._check_fitted()
        return dict(self.metrics_)

    def summary(self) -> str:
        """Return a human-readable summary of fitted metrics."""
        self._check_fitted()
        lines = [f"{self.__class__.__name__} — {self.n_samples_} samples"]
        lines.append("-" * 48)
        for k, v in self.metrics_.items():
            if isinstance(v, np.ndarray):
                lines.append(f"  {k}: {np.array2string(v, precision=4, suppress_small=True)}")
            elif isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @abstractmethod
    def _fit(self, data: np.ndarray) -> None:
        """
        Core fitting logic; must populate ``self.metrics_``.

        Parameters
        ----------
        data : np.ndarray
            Clean, NaN-free array of shape (n, k).
        """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _coerce(self, data: Any) -> np.ndarray:
        """Convert heterogeneous input types to a 2-D ndarray."""
        if XR_AVAILABLE and isinstance(data, xr.Dataset):
            data = np.column_stack([data[v].values.ravel() for v in data.data_vars])
        elif XR_AVAILABLE and isinstance(data, xr.DataArray):
            data = data.values
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def _clean(self, arr: np.ndarray) -> np.ndarray:
        """Remove NaN / Inf rows if ``self.dropna`` is True."""
        if not self.dropna:
            return arr
        finite_mask = np.all(np.isfinite(arr), axis=1)
        n_dropped = np.sum(~finite_mask)
        if n_dropped > 0:
            warnings.warn(
                f"{self.__class__.__name__}: dropped {n_dropped} row(s) containing "
                "NaN or Inf values.",
                UserWarning,
                stacklevel=4,
            )
        arr = arr[finite_mask]
        if arr.shape[0] < self.min_samples:
            warnings.warn(
                f"{self.__class__.__name__}: only {arr.shape[0]} valid samples "
                f"(< recommended {self.min_samples}). Results may be unreliable.",
                UserWarning,
                stacklevel=4,
            )
        return arr

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted yet. Call fit() first."
            )
