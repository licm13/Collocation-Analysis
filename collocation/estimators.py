"""
Scikit-learn Style Estimators
==============================

Thin, sklearn-compatible wrappers around the core collocation functions,
exposing a unified ``fit / metrics_`` interface.

Usage
-----
::

    from collocation.estimators import TC, EIVD, IVD, EC

    model = TC().fit(data)          # data: (n, 3) array
    print(model.metrics_['SNR'])    # → array of SNRs
    print(model.summary())

    # Works inside a loop
    results = {}
    for name, est in [('TC', TC()), ('EIVD', EIVD())]:
        results[name] = est.fit(data).metrics_

Author: Claude
Date: 2026-03-18
"""

from __future__ import annotations

import numpy as np
from typing import Any

from .base import CollocationEstimator
from .tc import tc as _tc
from .eivd import eivd as _eivd
from .ivd import ivd as _ivd
from .ec import ec as _ec


# ---------------------------------------------------------------------------
# TC
# ---------------------------------------------------------------------------

class TC(CollocationEstimator):
    """
    Triple Collocation estimator (sklearn-style wrapper).

    Assumes zero error cross-correlation. Expects input shape ``(n, 3)``.

    Attributes (in ``metrics_`` after fit)
    ----------------------------------------
    EeeT : np.ndarray (3, 3)
        Error covariance matrix (diagonal).
    SNR : np.ndarray (3,)
        Signal-to-Noise Ratios.
    rho2 : np.ndarray (3,)
        Data-truth correlations.
    fMSE : np.ndarray (3,)
        Fractional Mean Square Errors.
    error_std : np.ndarray (3,)
        Error standard deviations (sqrt of diagonal of EeeT).
    """

    def __init__(self, dropna: bool = True, min_samples: int = 100) -> None:
        super().__init__(dropna=dropna, min_samples=min_samples)

    def _fit(self, data: np.ndarray) -> None:
        if data.shape[1] != 3:
            raise ValueError(f"TC requires exactly 3 columns, got {data.shape[1]}.")
        EeeT, SNR, rho2, fMSE = _tc(data)
        self.metrics_.update(
            EeeT=EeeT,
            SNR=SNR,
            rho2=rho2,
            fMSE=fMSE,
            error_std=np.sqrt(np.maximum(np.diag(EeeT), 0)),
        )


# ---------------------------------------------------------------------------
# EIVD
# ---------------------------------------------------------------------------

class EIVD(CollocationEstimator):
    """
    Extended Information Vector Dual estimator (sklearn-style wrapper).

    Allows non-zero error cross-correlation between products 2 & 3.
    Expects input shape ``(n, 3)``.

    Attributes (in ``metrics_`` after fit)
    ----------------------------------------
    EeeT : np.ndarray (3, 3)
        Full error covariance matrix (with off-diagonal term).
    SNR : np.ndarray (3,)
    rho2 : np.ndarray (3,)
    fMSE : np.ndarray (3,)
    L : np.ndarray (3,)
        Lag-1 temporal autocorrelations.
    cross_corr : float
        Error cross-correlation between products 1 and 2 (EeeT[1,2]).
    error_std : np.ndarray (3,)
    """

    def __init__(self, dropna: bool = True, min_samples: int = 50) -> None:
        super().__init__(dropna=dropna, min_samples=min_samples)

    def _fit(self, data: np.ndarray) -> None:
        if data.shape[1] != 3:
            raise ValueError(f"EIVD requires exactly 3 columns, got {data.shape[1]}.")
        EeeT, SNR, rho2, fMSE, L = _eivd(data)
        self.metrics_.update(
            EeeT=EeeT,
            SNR=SNR,
            rho2=rho2,
            fMSE=fMSE,
            L=L,
            cross_corr=float(EeeT[1, 2]),
            error_std=np.sqrt(np.maximum(np.diag(EeeT), 0)),
        )


# ---------------------------------------------------------------------------
# IVD
# ---------------------------------------------------------------------------

class IVD(CollocationEstimator):
    """
    Information Vector Dual estimator (sklearn-style wrapper).

    Two-product method. Expects input shape ``(n, 2)``.

    Attributes (in ``metrics_`` after fit)
    ----------------------------------------
    EeeT : np.ndarray (2, 2)
    rho2 : np.ndarray (2,)
    weights : np.ndarray (2,)
        Optimal linear merging weights.
    error_std : np.ndarray (2,)
    """

    def __init__(self, dropna: bool = True, min_samples: int = 50) -> None:
        super().__init__(dropna=dropna, min_samples=min_samples)

    def _fit(self, data: np.ndarray) -> None:
        if data.shape[1] != 2:
            raise ValueError(f"IVD requires exactly 2 columns, got {data.shape[1]}.")
        EeeT, rho2, u = _ivd(data)
        self.metrics_.update(
            EeeT=EeeT,
            rho2=rho2,
            weights=u,
            error_std=np.sqrt(np.maximum(np.diag(EeeT), 0)),
        )


# ---------------------------------------------------------------------------
# EC
# ---------------------------------------------------------------------------

class EC(CollocationEstimator):
    """
    Extended (Quadruple) Collocation estimator (sklearn-style wrapper).

    Four-product method. Expects input shape ``(n, 4)``.

    Attributes (in ``metrics_`` after fit)
    ----------------------------------------
    EeeT : np.ndarray (4, 4)
    SNR : np.ndarray (4,)
    rho2 : np.ndarray (4,)
    fMSE : np.ndarray (4,)
    error_std : np.ndarray (4,)
    """

    def __init__(self, dropna: bool = True, min_samples: int = 100) -> None:
        super().__init__(dropna=dropna, min_samples=min_samples)

    def _fit(self, data: np.ndarray) -> None:
        if data.shape[1] != 4:
            raise ValueError(f"EC requires exactly 4 columns, got {data.shape[1]}.")
        results = _ec(data)
        # ec() returns a list of ECResult objects (one per reference-pair combination).
        # Each ECResult contains nested lists: .EeeT[i], .SNR[i], .rho2[i], .fMSE[i]
        # for each rescaling variant. We aggregate: median across combinations × variants.
        rho2_all, snr_all, fmse_all, eet_diag_all = [], [], [], []
        for res in results:
            for i in range(len(res.rho2)):
                r2 = np.asarray(res.rho2[i])
                snr = np.asarray(res.SNR[i])
                fm = np.asarray(res.fMSE[i])
                ee = np.asarray(res.EeeT[i])
                if r2.shape == (4,) and np.all(np.isfinite(r2)):
                    rho2_all.append(np.clip(r2, 0, 1))
                    snr_all.append(snr)
                    fmse_all.append(fm)
                    eet_diag_all.append(np.diag(ee))

        if rho2_all:
            rho2 = np.nanmedian(rho2_all, axis=0)
            SNR = np.nanmedian(snr_all, axis=0)
            fMSE = np.nanmedian(fmse_all, axis=0)
            diag = np.nanmedian(eet_diag_all, axis=0)
            EeeT = np.diag(diag)
        else:
            rho2 = np.full(4, np.nan)
            SNR = np.full(4, np.nan)
            fMSE = np.full(4, np.nan)
            EeeT = np.full((4, 4), np.nan)

        self.metrics_.update(
            EeeT=EeeT,
            SNR=SNR,
            rho2=rho2,
            fMSE=fMSE,
            error_std=np.sqrt(np.maximum(np.diag(EeeT), 0)),
            raw_results=results,
        )
