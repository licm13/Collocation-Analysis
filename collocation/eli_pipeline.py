"""
ELI One-Click Analysis Pipeline
=================================

A high-level pipeline that accepts three categories of ecosystem variables
(water, energy, vegetation response), automatically:

1. Aligns and pre-processes the arrays (NaN masking, anomaly computation).
2. Runs multiple collocation methods in parallel (TC, EIVD, IVS, BTCH_He2020).
3. Computes Ecosystem Limitation Index (ELI) from each method's outputs.
4. Renders a self-contained HTML diagnostic report.

Minimal usage
-------------
::

    from collocation.eli_pipeline import ELIPipeline
    import numpy as np

    pipe = ELIPipeline(
        water=np.column_stack([swvl1, gleam_w, gldas_w]),   # (n, k_w)
        energy=np.column_stack([swd, era5_rn, gldas_rn]),   # (n, k_e)
        vegetation=et_obs,                                   # (n,) or (n, k_v)
    )
    report = pipe.run()
    report.save("eli_report.html")
    print(report.summary())

Author: Claude
Date: 2026-03-18

References
----------
.. [1] Dong, J., et al. (2022). Ecosystem Limitation Index (ELI): A
       multi-source approach to quantifying terrestrial water and energy
       limitation. Remote Sensing of Environment.
"""

from __future__ import annotations

import html
import json
import textwrap
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# --- optional xarray support
try:
    import xarray as xr
    _XR = True
except ImportError:
    _XR = False

# --- collocation methods
from .tc import tc as _tc
from .eivd import eivd as _eivd
from .ivs import ivs as _ivs
from .btch_he2020 import btch_he2020 as _btch


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class MethodResult:
    """Results from one collocation method run on one variable category."""
    method: str
    category: str           # 'water' | 'energy' | 'vegetation'
    error_std: np.ndarray   # (k,)
    rho2: np.ndarray        # (k,)
    success: bool = True
    error_msg: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ELIResult:
    """
    Aggregated output of :meth:`ELIPipeline.run`.

    Attributes
    ----------
    eli_water : float
        ELI index for water limitation (mean ``1 - rho2`` over water products).
    eli_energy : float
        ELI index for energy limitation.
    eli_ratio : float
        eli_water / (eli_water + eli_energy) — fractional water limitation.
    method_results : dict[str, list[MethodResult]]
        Raw per-method results.
    html_path : str or None
        Path where the HTML report was saved (after calling ``save()``).
    """
    eli_water: float
    eli_energy: float
    eli_ratio: float
    method_results: Dict[str, List[MethodResult]]
    html_path: Optional[str] = None
    _html_content: str = field(default="", repr=False)

    def summary(self) -> str:
        lines = [
            "=" * 56,
            "  ELI Pipeline Summary",
            "=" * 56,
            f"  ELI (water)  : {self.eli_water:.4f}",
            f"  ELI (energy) : {self.eli_energy:.4f}",
            f"  ELI ratio    : {self.eli_ratio:.4f}  "
            f"({'water-limited' if self.eli_ratio > 0.5 else 'energy-limited'})",
            "",
            "  Per-method error std (water | energy):",
        ]
        for method, results in self.method_results.items():
            for r in results:
                if r.success:
                    std_str = np.array2string(r.error_std, precision=4)
                    lines.append(f"    {method:10s} [{r.category:10s}] → {std_str}")
                else:
                    lines.append(f"    {method:10s} [{r.category:10s}] → FAILED: {r.error_msg}")
        lines.append("=" * 56)
        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Save the HTML report to *path*."""
        if not self._html_content:
            warnings.warn("HTML report content is empty.", UserWarning)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self._html_content)
        self.html_path = path
        print(f"ELI report saved → {path}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ELIPipeline:
    """
    One-click ELI analysis pipeline.

    Parameters
    ----------
    water : array-like, shape (n,) or (n, k_w)
        Water-related variables (e.g. soil moisture products).
    energy : array-like, shape (n,) or (n, k_e)
        Energy-related variables (e.g. shortwave radiation, net radiation).
    vegetation : array-like, shape (n,) or (n, k_v)
        Vegetation response variables (e.g. ET, transpiration).
    product_names : dict, optional
        Labels for each category, e.g.::

            {'water': ['ERA5', 'GLEAM', 'GLDAS'],
             'energy': ['ERA5-SW', 'GLDAS-Rn'],
             'vegetation': ['ET']}
    methods : list[str], optional
        Subset of methods to run.  Options: ``'TC'``, ``'EIVD'``, ``'IVS'``,
        ``'BTCH'``.  Defaults to all available.
    compute_anomalies : bool, default=True
        Subtract temporal mean (column-wise) before analysis.
    n_bootstrap : int, default=500
        Bootstrap samples passed to IVS.
    max_workers : int, default=4
        Thread-pool size for parallel method execution.
    """

    _SUPPORTED = ("TC", "EIVD", "IVS", "BTCH")

    def __init__(
        self,
        water: Any,
        energy: Any,
        vegetation: Any,
        product_names: Optional[Dict[str, List[str]]] = None,
        methods: Optional[List[str]] = None,
        compute_anomalies: bool = True,
        n_bootstrap: int = 500,
        max_workers: int = 4,
    ) -> None:
        self.water = self._to_2d(water)
        self.energy = self._to_2d(energy)
        self.vegetation = self._to_2d(vegetation)
        self.product_names = product_names or {}
        self.methods = [m.upper() for m in (methods or list(self._SUPPORTED))]
        self.compute_anomalies = compute_anomalies
        self.n_bootstrap = n_bootstrap
        self.max_workers = max_workers

        # Validate and align
        n_vals = {self.water.shape[0], self.energy.shape[0], self.vegetation.shape[0]}
        if len(n_vals) > 1:
            raise ValueError(
                f"All inputs must have the same number of rows; "
                f"got water={self.water.shape[0]}, energy={self.energy.shape[0]}, "
                f"vegetation={self.vegetation.shape[0]}."
            )
        self.n = self.water.shape[0]

        # Default product names
        for cat, arr in [("water", self.water), ("energy", self.energy),
                         ("vegetation", self.vegetation)]:
            if cat not in self.product_names:
                self.product_names[cat] = [
                    f"{cat.capitalize()} P{i+1}" for i in range(arr.shape[1])
                ]

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> ELIResult:
        """
        Execute the full pipeline and return an :class:`ELIResult`.
        """
        # 1. Preprocess
        w = self._preprocess(self.water)
        e = self._preprocess(self.energy)
        v = self._preprocess(self.vegetation)

        # 2. Build analysis triplets: combine water+vegetation and energy+vegetation
        #    for 3-way methods; fall back to 2-way if only 2 products in a category.
        triplets: Dict[str, Tuple[np.ndarray, str]] = {}

        # water + first vegetation column
        wv = np.column_stack([w, v[:, :1]])  # (n, k_w + 1)
        ev = np.column_stack([e, v[:, :1]])  # (n, k_e + 1)
        triplets["water"] = (wv, "water")
        triplets["energy"] = (ev, "energy")

        # 3. Run methods in parallel
        all_results: Dict[str, List[MethodResult]] = {m: [] for m in self.methods}

        futures_map = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for cat, (arr, cat_name) in triplets.items():
                for method in self.methods:
                    fut = pool.submit(
                        self._run_single, method, arr, cat_name
                    )
                    futures_map[fut] = method

            for fut in as_completed(futures_map):
                method = futures_map[fut]
                result: MethodResult = fut.result()
                all_results[method].append(result)

        # 4. Compute ELI indices (average fMSE = 1 - rho2 across products & methods)
        eli_water = self._aggregate_eli(all_results, "water")
        eli_energy = self._aggregate_eli(all_results, "energy")
        denom = eli_water + eli_energy
        eli_ratio = eli_water / denom if denom > 0 else 0.5

        # 5. Build HTML report
        html_content = self._build_html(all_results, eli_water, eli_energy, eli_ratio)

        return ELIResult(
            eli_water=eli_water,
            eli_energy=eli_energy,
            eli_ratio=eli_ratio,
            method_results=all_results,
            _html_content=html_content,
        )

    # ------------------------------------------------------------------
    # Single-method runner
    # ------------------------------------------------------------------

    def _run_single(
        self, method: str, arr: np.ndarray, category: str
    ) -> MethodResult:
        """Execute one method on one array; catches all exceptions."""
        arr_clean = self._dropnan(arr)
        k = arr_clean.shape[1]

        try:
            if method == "TC":
                if k < 3:
                    raise ValueError(f"TC needs ≥3 columns, got {k}.")
                EeeT, SNR, rho2, fMSE = _tc(arr_clean[:, :3])
                return MethodResult(
                    method=method, category=category,
                    error_std=np.sqrt(np.maximum(np.diag(EeeT), 0)),
                    rho2=rho2, extra={"SNR": SNR, "fMSE": fMSE},
                )

            elif method == "EIVD":
                if k < 3:
                    raise ValueError(f"EIVD needs ≥3 columns, got {k}.")
                EeeT, SNR, rho2, fMSE, L = _eivd(arr_clean[:, :3])
                return MethodResult(
                    method=method, category=category,
                    error_std=np.sqrt(np.maximum(np.diag(EeeT), 0)),
                    rho2=rho2, extra={"SNR": SNR, "fMSE": fMSE, "L": L},
                )

            elif method == "IVS":
                if k < 2:
                    raise ValueError(f"IVS needs ≥2 columns, got {k}.")
                # IVS is a 2-way method; returns (RMSE, rho2)
                RMSE, rho2 = _ivs(arr_clean[:, :2], N_boot=self.n_bootstrap)
                return MethodResult(
                    method=method, category=category,
                    error_std=np.asarray(RMSE),
                    rho2=np.asarray(rho2),
                )

            elif method == "BTCH":
                if k < 3:
                    raise ValueError(f"BTCH needs ≥3 columns, got {k}.")
                # btch_he2020 returns (error_variances, weights, fused)
                error_variances, weights, _ = _btch(arr_clean[:, :3])
                error_std = np.sqrt(np.maximum(np.asarray(error_variances), 0))
                # BTCH doesn't directly output rho2; approximate from SNR heuristic
                rho2 = np.full(3, np.nan)
                return MethodResult(
                    method=method, category=category,
                    error_std=error_std, rho2=rho2,
                    extra={"weights": weights},
                )

            else:
                raise ValueError(f"Unknown method: {method}")

        except Exception as exc:
            return MethodResult(
                method=method, category=category,
                error_std=np.full(arr_clean.shape[1], np.nan),
                rho2=np.full(arr_clean.shape[1], np.nan),
                success=False,
                error_msg=str(exc),
            )

    # ------------------------------------------------------------------
    # ELI aggregation
    # ------------------------------------------------------------------

    def _aggregate_eli(
        self, all_results: Dict[str, List[MethodResult]], category: str
    ) -> float:
        """
        Mean ``1 - rho2`` across all successful method runs for *category*.
        """
        fmse_values = []
        for method_results in all_results.values():
            for r in method_results:
                if r.category == category and r.success:
                    valid = r.rho2[np.isfinite(r.rho2)]
                    if len(valid) > 0:
                        fmse_values.extend((1.0 - valid).tolist())
        return float(np.nanmean(fmse_values)) if fmse_values else 0.5

    # ------------------------------------------------------------------
    # HTML report builder
    # ------------------------------------------------------------------

    def _build_html(
        self,
        all_results: Dict[str, List[MethodResult]],
        eli_water: float,
        eli_energy: float,
        eli_ratio: float,
    ) -> str:
        limitation = "water-limited" if eli_ratio > 0.5 else "energy-limited"
        color_water = "#2166AC"
        color_energy = "#D6604D"

        # ── build method table rows ───────────────────────────────────────
        table_rows = ""
        for method, results in all_results.items():
            for r in results:
                status = "✓" if r.success else "✗"
                status_color = "#2e7d32" if r.success else "#c62828"
                if r.success:
                    std_str = ", ".join(f"{v:.4f}" for v in r.error_std)
                    rho_str = ", ".join(f"{v:.4f}" for v in r.rho2)
                else:
                    std_str = html.escape(r.error_msg[:60])
                    rho_str = "—"
                table_rows += f"""
                <tr>
                  <td>{html.escape(method)}</td>
                  <td>{html.escape(r.category)}</td>
                  <td style="color:{status_color}">{status}</td>
                  <td><code>{std_str}</code></td>
                  <td><code>{rho_str}</code></td>
                </tr>"""

        # ── ELI gauge bars ────────────────────────────────────────────────
        water_pct = round(eli_ratio * 100, 1)
        energy_pct = round((1 - eli_ratio) * 100, 1)

        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        report = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ELI Diagnostic Report</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: #f8f9fa; color: #212529; margin: 0; padding: 24px;
    }}
    h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 8px; }}
    h2 {{ color: #37474f; margin-top: 32px; font-size: 1.1rem; }}
    .meta {{ color: #607d8b; font-size: 0.85rem; margin-bottom: 24px; }}
    .card {{
      background: white; border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,.1);
      padding: 20px 24px; margin-bottom: 24px;
    }}
    .eli-summary {{
      display: grid; grid-template-columns: 1fr 1fr 1fr;
      gap: 16px; margin-bottom: 24px;
    }}
    .kpi {{
      background: white; border-radius: 8px;
      padding: 16px; text-align: center;
      box-shadow: 0 2px 8px rgba(0,0,0,.08);
    }}
    .kpi-val {{ font-size: 2rem; font-weight: 700; }}
    .kpi-lab {{ font-size: 0.8rem; color: #607d8b; margin-top: 4px; }}
    .gauge-wrap {{ margin: 12px 0; }}
    .gauge-label {{ font-size: 0.85rem; margin-bottom: 4px; }}
    .gauge-bar {{
      height: 22px; border-radius: 4px; display: flex;
      overflow: hidden; font-size: 0.78rem; font-weight: 600;
    }}
    .gauge-water {{
      background: {color_water}; color: white;
      display: flex; align-items: center; justify-content: center;
    }}
    .gauge-energy {{
      background: {color_energy}; color: white;
      display: flex; align-items: center; justify-content: center;
    }}
    table {{
      width: 100%; border-collapse: collapse; font-size: 0.88rem;
    }}
    th {{
      background: #1a237e; color: white;
      padding: 8px 12px; text-align: left;
    }}
    td {{ padding: 7px 12px; border-bottom: 1px solid #e0e0e0; }}
    tr:hover td {{ background: #f1f8e9; }}
    code {{ background: #f5f5f5; padding: 1px 4px; border-radius: 3px; }}
    .badge {{
      display: inline-block; padding: 3px 10px; border-radius: 12px;
      font-size: 0.78rem; font-weight: 600; color: white;
    }}
    .badge-water   {{ background: {color_water}; }}
    .badge-energy  {{ background: {color_energy}; }}
    footer {{
      text-align: center; font-size: 0.78rem;
      color: #9e9e9e; margin-top: 32px;
    }}
  </style>
</head>
<body>
  <h1>🌿 ELI Diagnostic Report</h1>
  <p class="meta">Generated: {timestamp} &nbsp;|&nbsp;
     Samples: {self.n} &nbsp;|&nbsp;
     Methods: {', '.join(self.methods)}</p>

  <!-- KPIs -->
  <div class="eli-summary">
    <div class="kpi">
      <div class="kpi-val" style="color:{color_water}">{eli_water:.4f}</div>
      <div class="kpi-lab">ELI Water</div>
    </div>
    <div class="kpi">
      <div class="kpi-val" style="color:{color_energy}">{eli_energy:.4f}</div>
      <div class="kpi-lab">ELI Energy</div>
    </div>
    <div class="kpi">
      <div class="kpi-val"
           style="color:{'#1565c0' if eli_ratio > 0.5 else '#b71c1c'}">
        {eli_ratio:.3f}
      </div>
      <div class="kpi-lab">
        ELI Ratio
        <span class="badge {'badge-water' if eli_ratio > 0.5 else 'badge-energy'}">
          {limitation}
        </span>
      </div>
    </div>
  </div>

  <!-- Gauge bar -->
  <div class="card">
    <h2>Water vs. Energy Limitation Balance</h2>
    <div class="gauge-wrap">
      <div class="gauge-label">
        Water {water_pct}% &nbsp;/&nbsp; Energy {energy_pct}%
      </div>
      <div class="gauge-bar">
        <div class="gauge-water" style="width:{water_pct}%">{water_pct}%</div>
        <div class="gauge-energy" style="width:{energy_pct}%">{energy_pct}%</div>
      </div>
    </div>
    <p style="font-size:0.85rem; color:#607d8b; margin-top:8px;">
      ELI ratio > 0.5 indicates water limitation;
      ratio &lt; 0.5 indicates energy limitation.
    </p>
  </div>

  <!-- Detailed results table -->
  <div class="card">
    <h2>Per-Method Results</h2>
    <table>
      <thead>
        <tr>
          <th>Method</th>
          <th>Category</th>
          <th>Status</th>
          <th>Error Std</th>
          <th>ρ²</th>
        </tr>
      </thead>
      <tbody>{table_rows}
      </tbody>
    </table>
  </div>

  <!-- Product names -->
  <div class="card">
    <h2>Input Configuration</h2>
    <table>
      <thead><tr><th>Category</th><th>Products</th></tr></thead>
      <tbody>
        <tr>
          <td>Water</td>
          <td>{', '.join(html.escape(p) for p in self.product_names.get('water', []))}</td>
        </tr>
        <tr>
          <td>Energy</td>
          <td>{', '.join(html.escape(p) for p in self.product_names.get('energy', []))}</td>
        </tr>
        <tr>
          <td>Vegetation</td>
          <td>{', '.join(html.escape(p) for p in self.product_names.get('vegetation', []))}</td>
        </tr>
      </tbody>
    </table>
  </div>

  <footer>
    Collocation Analysis Package &nbsp;·&nbsp;
    ELI Pipeline &nbsp;·&nbsp; {timestamp}
  </footer>
</body>
</html>"""
        return report

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_2d(arr: Any) -> np.ndarray:
        """Coerce input (xr.DataArray / xr.Dataset / ndarray / list) to 2-D float array."""
        if _XR:
            if isinstance(arr, xr.Dataset):
                arr = np.column_stack([arr[v].values.ravel() for v in arr.data_vars])
            elif isinstance(arr, xr.DataArray):
                arr = arr.values
        a = np.asarray(arr, dtype=float)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        return a

    @staticmethod
    def _preprocess(arr: np.ndarray) -> np.ndarray:
        """Subtract temporal mean (compute anomalies)."""
        col_mean = np.nanmean(arr, axis=0, keepdims=True)
        return arr - col_mean

    @staticmethod
    def _dropnan(arr: np.ndarray) -> np.ndarray:
        """Remove rows with any NaN or Inf."""
        mask = np.all(np.isfinite(arr), axis=1)
        return arr[mask]
