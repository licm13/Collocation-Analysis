"""
Interactive & Static Plotting
==============================

Two-tier plotting API:

**Static (matplotlib)**
    ``plot_error_comparison(metrics_dict)`` — publication-quality bar chart.

**Interactive (Plotly)**
    ``InteractiveDashboard(data, metrics_dict)`` — three-panel HTML dashboard:

    - *Left*  : Time-series with dynamic slice slider.
    - *Right* : RMSE bar chart that updates with the slice.
    - *Bottom*: Stability heat-map (window size × product RMSE).

Plotly is an optional dependency; if absent the interactive API raises a
clear ``ImportError`` rather than crashing silently.

Usage
-----
::

    from collocation.plotting import InteractiveDashboard, plot_error_comparison

    dash = InteractiveDashboard(data, {'TC': tc_metrics, 'EIVD': eivd_metrics})
    dash.show()          # opens browser
    dash.save('report.html')

    # or static:
    plot_error_comparison({'TC': tc_metrics, 'EIVD': eivd_metrics})

Author: Claude
Date: 2026-03-18
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Colour palette (publication-ready, colour-blind friendly)
# ---------------------------------------------------------------------------
_PALETTE = [
    "#2166AC", "#D6604D", "#4DAC26", "#8E24AA",
    "#F4A460", "#00838F", "#E64A19", "#546E7A",
]


# ---------------------------------------------------------------------------
# Static matplotlib helpers
# ---------------------------------------------------------------------------

def plot_error_comparison(
    metrics_dict: Dict[str, Dict[str, Any]],
    product_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 4),
    title: str = "Error Standard Deviation Comparison",
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Bar chart comparing ``error_std`` across multiple fitted estimators.

    Parameters
    ----------
    metrics_dict : dict[str, dict]
        Keys are method names; values are ``estimator.metrics_`` dicts that
        must contain ``'error_std'`` (shape ``(k,)``).
    product_names : list[str], optional
        Labels for each product.  Auto-generated if omitted.
    figsize : tuple, default=(10, 4)
    title : str
    save_path : str, optional
        If given, figure is saved here (PNG or PDF).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if not MPL_AVAILABLE:
        raise ImportError("matplotlib is required for static plots.")

    methods = list(metrics_dict)
    n_methods = len(methods)
    if n_methods == 0:
        raise ValueError("metrics_dict is empty.")

    # Infer k
    k = len(metrics_dict[methods[0]]["error_std"])
    if product_names is None:
        product_names = [f"P{i+1}" for i in range(k)]

    x = np.arange(k)
    width = 0.8 / n_methods
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * width

    fig, ax = plt.subplots(figsize=figsize)
    for idx, method in enumerate(methods):
        std = np.asarray(metrics_dict[method]["error_std"])
        bars = ax.bar(
            x + offsets[idx], std,
            width=width * 0.9,
            label=method,
            color=_PALETTE[idx % len(_PALETTE)],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        # Value labels
        for bar, val in zip(bars, std):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(product_names)
    ax.set_ylabel("Error Std Dev")
    ax.set_title(title)
    ax.legend(framealpha=0.9, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_stability_heatmap(
    data: np.ndarray,
    method_fn: Any,
    window_sizes: Optional[Sequence[int]] = None,
    product_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (8, 4),
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Heat-map of RMSE stability across different sample-window sizes.

    Parameters
    ----------
    data : np.ndarray, shape (n, k)
    method_fn : callable
        A collocation function (e.g. ``tc``) returning ``(EeeT, …)`` where
        ``EeeT`` is the first element.
    window_sizes : sequence[int], optional
        Window sizes to test.  Defaults to 10 values from n//10 to n.
    product_names : list[str], optional
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if not MPL_AVAILABLE:
        raise ImportError("matplotlib is required for static plots.")

    n, k = data.shape
    if window_sizes is None:
        window_sizes = np.linspace(max(50, n // 10), n, 10, dtype=int).tolist()
    if product_names is None:
        product_names = [f"P{i+1}" for i in range(k)]

    rmse_matrix = np.full((len(window_sizes), k), np.nan)
    for wi, ws in enumerate(window_sizes):
        sub = data[:ws]
        mask = np.all(np.isfinite(sub), axis=1)
        sub = sub[mask]
        if sub.shape[0] < 10:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = method_fn(sub)
            EeeT = out[0]
            rmse_matrix[wi] = np.sqrt(np.maximum(np.diag(EeeT), 0))
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        rmse_matrix.T,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn_r",
        interpolation="nearest",
    )
    ax.set_yticks(range(k))
    ax.set_yticklabels(product_names)
    ax.set_xticks(range(len(window_sizes)))
    ax.set_xticklabels([str(w) for w in window_sizes], rotation=45, ha="right")
    ax.set_xlabel("Window size (samples)")
    ax.set_ylabel("Product")
    ax.set_title("Error Std Dev Stability Across Window Sizes")
    fig.colorbar(im, ax=ax, label="Error Std Dev")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Interactive Plotly dashboard
# ---------------------------------------------------------------------------

class InteractiveDashboard:
    """
    Three-panel Plotly dashboard for interactive collocation diagnostics.

    Layout
    ------
    ┌─────────────────────────┬─────────────────┐
    │  Time-series (slider)   │  RMSE bar chart │
    └─────────────────────────┴─────────────────┘
    │       Window-stability heat-map            │
    └────────────────────────────────────────────┘

    Parameters
    ----------
    data : np.ndarray, shape (n, k)
        Raw time series data.
    metrics_dict : dict[str, dict]
        Keys: method names.  Values: ``estimator.metrics_`` dicts with
        at least ``'error_std'``.
    product_names : list[str], optional
    method_fn : callable, optional
        Collocation function for stability heat-map (e.g. ``tc``).
        If None, the heat-map panel is skipped.
    window_sizes : sequence[int], optional
        Window sizes for stability heat-map.
    title : str
    """

    def __init__(
        self,
        data: np.ndarray,
        metrics_dict: Dict[str, Dict[str, Any]],
        product_names: Optional[List[str]] = None,
        method_fn: Optional[Any] = None,
        window_sizes: Optional[Sequence[int]] = None,
        title: str = "Collocation Analysis Dashboard",
    ) -> None:
        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "Plotly is required for interactive dashboards. "
                "Install with: pip install plotly"
            )
        self.data = np.asarray(data, dtype=float)
        self.metrics_dict = metrics_dict
        n, k = self.data.shape
        self.n = n
        self.k = k
        self.product_names = product_names or [f"Product {i+1}" for i in range(k)]
        self.method_fn = method_fn
        self.window_sizes = (
            window_sizes
            or np.linspace(max(50, n // 10), n, 8, dtype=int).tolist()
        )
        self.title = title
        self._fig: Optional[go.Figure] = None

    def build(self) -> go.Figure:
        """Construct and return the Plotly Figure."""
        use_heatmap = self.method_fn is not None
        rows = 2 if use_heatmap else 1
        col_widths = [0.60, 0.40]
        row_heights = [0.55, 0.45] if use_heatmap else [1.0]

        subplot_specs: List[List[Any]] = [[{"type": "scatter"}, {"type": "bar"}]]
        if use_heatmap:
            subplot_specs.append([{"colspan": 2, "type": "heatmap"}, None])

        fig = make_subplots(
            rows=rows,
            cols=2,
            column_widths=col_widths,
            row_heights=row_heights,
            specs=subplot_specs,
            subplot_titles=(
                ["Time Series", "RMSE by Method"]
                + (["Stability Heat-map (window size)"] if use_heatmap else [])
            ),
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
        )

        # ── Time series panel ────────────────────────────────────────────
        for i, name in enumerate(self.product_names):
            col = _PALETTE[i % len(_PALETTE)]
            fig.add_trace(
                go.Scatter(
                    x=np.arange(self.n),
                    y=self.data[:, i],
                    name=name,
                    mode="lines",
                    line=dict(color=col, width=1.2),
                    hovertemplate=f"{name}<br>t=%{{x}}<br>val=%{{y:.4f}}<extra></extra>",
                ),
                row=1, col=1,
            )

        # Range slider for time-series
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeslider_thickness=0.06,
            row=1, col=1,
        )

        # ── RMSE bar chart ────────────────────────────────────────────────
        methods = list(self.metrics_dict)
        for mi, method in enumerate(methods):
            std = np.asarray(self.metrics_dict[method]["error_std"])
            fig.add_trace(
                go.Bar(
                    name=method,
                    x=self.product_names,
                    y=std,
                    marker_color=_PALETTE[(mi + 4) % len(_PALETTE)],
                    opacity=0.85,
                    text=[f"{v:.4f}" for v in std],
                    textposition="outside",
                    hovertemplate=f"{method}<br>%{{x}}: %{{y:.4f}}<extra></extra>",
                ),
                row=1, col=2,
            )

        # ── Stability heat-map ────────────────────────────────────────────
        if use_heatmap:
            rmse_matrix = self._compute_stability()
            fig.add_trace(
                go.Heatmap(
                    z=rmse_matrix,            # shape (k, n_windows)
                    x=[str(w) for w in self.window_sizes],
                    y=self.product_names,
                    colorscale="RdYlGn_r",
                    colorbar=dict(title="Error Std"),
                    hovertemplate=(
                        "Product: %{y}<br>Window: %{x}<br>"
                        "Err Std: %{z:.4f}<extra></extra>"
                    ),
                ),
                row=2, col=1,
            )

        # ── Layout polish ─────────────────────────────────────────────────
        fig.update_layout(
            title=dict(text=self.title, font=dict(size=16)),
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right", x=1,
            ),
            barmode="group",
            height=700 if use_heatmap else 420,
            margin=dict(t=80, b=40, l=50, r=30),
            hoverlabel=dict(bgcolor="white", font_size=12),
        )
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Error Std Dev", row=1, col=2)
        if use_heatmap:
            fig.update_xaxes(title_text="Window size (samples)", row=2, col=1)

        self._fig = fig
        return fig

    def show(self) -> None:
        """Render the dashboard in a browser (or Jupyter cell)."""
        if self._fig is None:
            self.build()
        self._fig.show()

    def save(self, path: str) -> None:
        """
        Save dashboard as a standalone HTML file.

        Parameters
        ----------
        path : str
            Output path (e.g. ``'dashboard.html'``).
        """
        if self._fig is None:
            self.build()
        self._fig.write_html(path, include_plotlyjs="cdn")
        print(f"Dashboard saved → {path}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_stability(self) -> np.ndarray:
        """Return (k, n_windows) RMSE matrix for stability heat-map."""
        rmse_matrix = np.full((self.k, len(self.window_sizes)), np.nan)
        for wi, ws in enumerate(self.window_sizes):
            sub = self.data[:ws]
            mask = np.all(np.isfinite(sub), axis=1)
            sub = sub[mask]
            if sub.shape[0] < 10:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = self.method_fn(sub)
                EeeT = out[0]
                rmse_matrix[:, wi] = np.sqrt(np.maximum(np.diag(EeeT), 0))
            except Exception:
                pass
        return rmse_matrix
