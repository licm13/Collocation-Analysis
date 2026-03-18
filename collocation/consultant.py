"""
Collocation Consultant — Smart Method Recommender
==================================================

``CollocationConsultant`` ingests multiple co-located time series and
performs a lightweight diagnostic battery, then issues method recommendations
backed by quantitative evidence.

Diagnostics performed
---------------------
1. **Lag-1 autocorrelation** per product — flags heteroscedasticity /
   temporal persistence that benefits from Bayesian methods.
2. **Pairwise cross-correlation** — detects correlated errors between
   products; high correlation → recommend EIVD over TC.
3. **Variance stationarity** (rolling-window std ratio) — reveals
   time-varying error regimes → recommend BayesianTC.
4. **Normality test** (Shapiro-Wilk on residuals if n ≤ 5000, else
   D'Agostino–Pearson) — heavy tails → recommend robust fusion.
5. **Multiplicative-error check** (skewness of raw data) — positive-skew
   data → recommend MTCH.
6. **Sample-size check** — minimum for each method.

Usage
-----
::

    from collocation.consultant import CollocationConsultant
    import numpy as np

    data = np.column_stack([product1, product2, product3])
    c = CollocationConsultant(data)
    report = c.consult()
    print(report.text)              # Human-readable advice
    print(report.recommended)       # Primary recommended method name
    print(report.diagnostics)       # Raw diagnostic dict

Author: Claude
Date: 2026-03-18
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

try:
    import xarray as xr
    _XR = True
except ImportError:
    _XR = False


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConsultationReport:
    """
    Output of :meth:`CollocationConsultant.consult`.

    Attributes
    ----------
    recommended : str
        Primary recommended method (e.g. ``'EIVD'``, ``'BayesianTC'``, ``'TC'``).
    alternatives : list[str]
        Alternative methods worth considering, in priority order.
    text : str
        Human-readable narrative report.
    diagnostics : dict
        Raw numeric diagnostics (autocorrelations, cross-correlations, …).
    warnings : list[str]
        Data-quality warnings (short flags).
    """
    recommended: str
    alternatives: List[str]
    text: str
    diagnostics: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.text


# ---------------------------------------------------------------------------
# Thresholds (module-level constants for readability)
# ---------------------------------------------------------------------------

_CROSS_CORR_THRESHOLD   = 0.30   # |r| above this → correlated errors
_LAG1_AC_THRESHOLD      = 0.40   # autocorr above → strong persistence
_VAR_RATIO_THRESHOLD    = 2.50   # rolling-std ratio above → heteroscedastic
_SKEW_THRESHOLD         = 1.50   # absolute skewness above → multiplicative
_MIN_SAMPLES_TC         = 100
_MIN_SAMPLES_EIVD       = 50
_MIN_SAMPLES_BAYESIAN   = 200
_MIN_SAMPLES_MTCH       = 50
_NORMALITY_ALPHA        = 0.05


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CollocationConsultant:
    """
    Lightweight diagnostic advisor for collocation method selection.

    Parameters
    ----------
    data : array-like, shape (n, k) where k ∈ {2, 3, 4}
        Co-located time series.  Rows with NaN / Inf are silently dropped
        before diagnostics.
    product_names : list[str], optional
        Labels for each column.  Defaults to ``['Product 1', …]``.
    window_size : int, default=30
        Rolling window size (in samples) for variance-stationarity check.
    verbose : bool, default=False
        Print intermediate diagnostic values while running.

    Examples
    --------
    >>> import numpy as np
    >>> from collocation.consultant import CollocationConsultant
    >>> np.random.seed(0)
    >>> n = 300
    >>> theta = np.random.randn(n)
    >>> data = np.column_stack([
    ...     theta + 0.2 * np.random.randn(n),
    ...     theta + 0.3 * np.random.randn(n),
    ...     theta + 0.4 * np.random.randn(n),
    ... ])
    >>> report = CollocationConsultant(data).consult()
    >>> print(report.recommended)
    """

    def __init__(
        self,
        data: Any,
        product_names: Optional[List[str]] = None,
        window_size: int = 30,
        verbose: bool = False,
    ) -> None:
        self._raw = self._coerce(data)
        self._data = self._clean(self._raw)
        n, k = self._data.shape
        self.n = n
        self.k = k
        self.product_names = product_names or [f"Product {i+1}" for i in range(k)]
        self.window_size = window_size
        self.verbose = verbose

        if k < 2 or k > 4:
            raise ValueError(f"Expected 2–4 columns, got {k}.")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def consult(self) -> ConsultationReport:
        """
        Run all diagnostics and return a :class:`ConsultationReport`.
        """
        diag: Dict[str, Any] = {}

        # 1. Lag-1 autocorrelations
        lag1 = self._lag1_autocorrelation()
        diag["lag1_autocorr"] = lag1

        # 2. Pairwise cross-correlations
        xcorr = self._pairwise_crosscorr()
        diag["pairwise_crosscorr"] = xcorr

        # 3. Variance stationarity
        var_ratio = self._variance_ratio()
        diag["variance_ratio"] = var_ratio

        # 4. Normality of column residuals
        normality = self._normality_check()
        diag["normality_pvalue"] = normality

        # 5. Skewness (multiplicative-error indicator)
        skew = np.array([stats.skew(self._data[:, i]) for i in range(self.k)])
        diag["skewness"] = skew

        # 6. Sample size
        diag["n_samples"] = self.n

        if self.verbose:
            self._print_diag(diag)

        # ------------------------------------------------------------------
        # Decision logic
        # ------------------------------------------------------------------
        warns: List[str] = []
        evidence: List[Tuple[str, str]] = []   # (method, reason)

        # Sample-size gating
        if self.n < _MIN_SAMPLES_EIVD:
            warns.append(
                f"Only {self.n} samples — most methods require ≥{_MIN_SAMPLES_EIVD}."
            )

        # Correlated errors?
        high_xcorr_pairs = [
            (self.product_names[i], self.product_names[j], float(xcorr[i, j]))
            for i in range(self.k)
            for j in range(i + 1, self.k)
            if abs(xcorr[i, j]) > _CROSS_CORR_THRESHOLD
        ]
        if high_xcorr_pairs and self.k == 3:
            pair_strs = "; ".join(
                f"{a}×{b} (r={r:.2f})" for a, b, r in high_xcorr_pairs
            )
            evidence.append((
                "EIVD",
                f"High cross-correlation detected ({pair_strs}). "
                "TC assumes independent errors — EIVD explicitly models "
                "error co-variance and reduces bias.",
            ))

        # Heteroscedasticity / time-varying variance?
        max_ratio = float(np.nanmax(var_ratio))
        if max_ratio > _VAR_RATIO_THRESHOLD:
            idx = int(np.nanargmax(var_ratio))
            evidence.append((
                "BayesianTC",
                f"{self.product_names[idx]} shows time-varying variance "
                f"(rolling-std ratio ≈ {max_ratio:.1f}×). "
                "BayesianTC captures non-stationary errors via a state-space model.",
            ))

        # Strong temporal persistence?
        max_ac = float(np.max(np.abs(lag1)))
        if max_ac > _LAG1_AC_THRESHOLD:
            idx = int(np.argmax(np.abs(lag1)))
            evidence.append((
                "BayesianTC",
                f"{self.product_names[idx]} has strong lag-1 autocorrelation "
                f"(ρ₁ = {lag1[idx]:.2f}), suggesting serially correlated errors.",
            ))

        # Multiplicative / log-normal data?
        max_skew = float(np.max(np.abs(skew)))
        if max_skew > _SKEW_THRESHOLD and np.all(self._data > 0):
            evidence.append((
                "MTCH",
                f"Data exhibit high positive skewness (max |skew| = {max_skew:.2f}) "
                "and are strictly positive — multiplicative error model (MTCH) is "
                "more appropriate than additive-error methods.",
            ))

        # Non-normality?
        min_p = float(np.nanmin(normality))
        if min_p < _NORMALITY_ALPHA:
            idx = int(np.nanargmin(normality))
            warns.append(
                f"{self.product_names[idx]} residuals are non-normal "
                f"(p = {normality[idx]:.3f}); consider robust fusion weights."
            )

        # 4-product case
        if self.k == 4:
            evidence.append((
                "EC",
                "Four data products detected — Extended (Quadruple) Collocation "
                "provides additional over-determination and better SNR estimates.",
            ))

        # 2-product case
        if self.k == 2:
            evidence.append((
                "IVD",
                "Two data products detected — IVD is the appropriate 2-way method.",
            ))

        # Fallback: TC is always valid for 3 products without correlated errors
        if self.k == 3 and not high_xcorr_pairs:
            evidence.append((
                "TC",
                "No significant error cross-correlation detected. "
                "Classic Triple Collocation is appropriate.",
            ))

        # ------------------------------------------------------------------
        # Pick primary recommendation (first unique method from evidence)
        # ------------------------------------------------------------------
        seen: Dict[str, str] = {}
        for method, reason in evidence:
            if method not in seen:
                seen[method] = reason

        if seen:
            recommended = next(iter(seen))
            alternatives = [m for m in seen if m != recommended]
        else:
            recommended = "TC"
            alternatives = ["EIVD"]

        text = self._build_text(recommended, seen, warns, diag)

        return ConsultationReport(
            recommended=recommended,
            alternatives=alternatives,
            text=text,
            diagnostics=diag,
            warnings=warns,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _lag1_autocorrelation(self) -> np.ndarray:
        """Pearson lag-1 autocorrelation per column."""
        result = np.zeros(self.k)
        for i in range(self.k):
            x = self._data[:, i]
            if len(x) > 2:
                result[i] = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        return result

    def _pairwise_crosscorr(self) -> np.ndarray:
        """
        Estimate *error* cross-correlations using a row-mean truth proxy.

        Each product's error is approximated as:
            ε̂_i = X_i − mean(X, axis=1)
        where the row mean is a proxy for the unknown truth θ.  This
        removes the shared signal component so we measure co-variation
        of residual errors rather than of the products themselves.
        """
        truth_proxy = np.mean(self._data, axis=1, keepdims=True)
        errors = self._data - truth_proxy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = np.corrcoef(errors, rowvar=False)
        np.fill_diagonal(corr, 0.0)
        return corr

    def _variance_ratio(self) -> np.ndarray:
        """
        Rolling-window std ratio: max(rolling_std) / min(rolling_std) per column.

        Ratio >> 1 signals heteroscedasticity.
        """
        ratio = np.ones(self.k)
        w = min(self.window_size, self.n // 4)
        if w < 3:
            return ratio
        for i in range(self.k):
            x = self._data[:, i]
            stds = np.array([
                np.std(x[j: j + w], ddof=1)
                for j in range(0, len(x) - w + 1, w // 2)
            ])
            stds = stds[stds > 0]
            if len(stds) >= 2:
                ratio[i] = float(np.max(stds) / np.min(stds))
        return ratio

    def _normality_check(self) -> np.ndarray:
        """
        Return p-values of a normality test on column residuals.
        Shapiro-Wilk for n ≤ 5000, otherwise D'Agostino-Pearson.
        """
        pvals = np.ones(self.k)
        for i in range(self.k):
            x = self._data[:, i] - np.mean(self._data[:, i])
            try:
                if len(x) <= 5000:
                    _, p = stats.shapiro(x)
                else:
                    _, p = stats.normaltest(x)
                pvals[i] = p
            except Exception:
                pvals[i] = 1.0
        return pvals

    # ------------------------------------------------------------------
    # Report formatting
    # ------------------------------------------------------------------

    def _build_text(
        self,
        recommended: str,
        evidence: Dict[str, str],
        warns: List[str],
        diag: Dict[str, Any],
    ) -> str:
        lines = [
            "=" * 64,
            "  Collocation Method Recommendation Report",
            "=" * 64,
            f"  Samples : {self.n}",
            f"  Products: {', '.join(self.product_names)}",
            "",
            f"  ★ PRIMARY RECOMMENDATION  →  {recommended}",
        ]

        if evidence.get(recommended):
            lines.append(f"    Reason: {evidence[recommended]}")

        alts = [m for m in evidence if m != recommended]
        if alts:
            lines.append("")
            lines.append("  ▸ ALSO CONSIDER:")
            for m in alts:
                lines.append(f"    • {m}: {evidence[m]}")

        if warns:
            lines.append("")
            lines.append("  ⚠ WARNINGS:")
            for w in warns:
                lines.append(f"    ! {w}")

        # Diagnostic snapshot
        lines += [
            "",
            "  ── Diagnostic Snapshot ──────────────────────────────",
        ]
        lag1 = diag["lag1_autocorr"]
        xcorr = diag["pairwise_crosscorr"]
        vr = diag["variance_ratio"]
        skew = diag["skewness"]

        for i, name in enumerate(self.product_names):
            lines.append(
                f"    {name:<16} lag1ρ={lag1[i]:+.3f}  "
                f"var_ratio={vr[i]:.2f}  skew={skew[i]:+.2f}"
            )

        lines.append("")
        lines.append("  Cross-correlations (proxy):")
        for i in range(self.k):
            for j in range(i + 1, self.k):
                r = xcorr[i, j]
                flag = " ← HIGH" if abs(r) > _CROSS_CORR_THRESHOLD else ""
                lines.append(
                    f"    {self.product_names[i]} × {self.product_names[j]}: "
                    f"r = {r:+.3f}{flag}"
                )
        lines.append("=" * 64)
        return "\n".join(lines)

    def _print_diag(self, diag: Dict[str, Any]) -> None:
        print("[Consultant diagnostics]")
        for k, v in diag.items():
            if isinstance(v, np.ndarray):
                print(f"  {k}: {np.round(v, 4)}")
            else:
                print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce(data: Any) -> np.ndarray:
        if _XR:
            import xarray as xr
            if isinstance(data, xr.Dataset):
                data = np.column_stack([data[v].values.ravel() for v in data.data_vars])
            elif isinstance(data, xr.DataArray):
                data = data.values
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    @staticmethod
    def _clean(arr: np.ndarray) -> np.ndarray:
        mask = np.all(np.isfinite(arr), axis=1)
        return arr[mask]
