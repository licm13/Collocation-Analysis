"""Compare fusion performance under different covariance construction schemes.

This example synthesises collocation diagnostics for several distinct error
regimes and evaluates how the choice of covariance matrix influences the
resulting fused estimates.  The script produces a publication-grade diagnostic
figure (600 dpi) that contrasts inverse-variance weighting with full GLS style
covariances, Ledoit–Wolf shrinkage and ridge regularisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from collocation.covariance import build_sigma_from_collocation
from collocation.fuse import estimate_bias_from_collocation


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    description: str
    cov: np.ndarray
    biases: np.ndarray
    n_total: int
    n_colloc: int
    seed: int


MODEL_NAMES = ("A", "B", "C", "D")


def _simulate_collocation(config: ScenarioConfig) -> tuple[xr.Dataset, np.ndarray, np.ndarray]:
    """Generate synthetic collocation diagnostics for ``config``."""

    rng = np.random.default_rng(config.seed)
    truth = rng.normal(loc=0.0, scale=1.0, size=config.n_total)
    errors = rng.multivariate_normal(mean=np.zeros(len(MODEL_NAMES)), cov=config.cov, size=config.n_total)
    predictions = truth[:, None] + config.biases + errors

    residuals = predictions - truth[:, None]
    colloc_residuals = residuals[: config.n_colloc]
    bias_est = colloc_residuals.mean(axis=0)
    demeaned = colloc_residuals - bias_est
    cov_est = np.cov(demeaned, rowvar=False, ddof=1)
    var_est = np.diag(cov_est)

    coords = {"model": list(MODEL_NAMES)}
    dataset = xr.Dataset(
        {
            "cov": xr.DataArray(cov_est, dims=("model", "model"), coords=coords),
            "var": xr.DataArray(var_est, dims=("model",), coords=coords),
            "bias": xr.DataArray(bias_est, dims=("model",), coords=coords),
        }
    )
    return dataset, truth, predictions


def _solve_gls_weights(sigma: xr.DataArray) -> xr.DataArray:
    """Return GLS weights for ``sigma`` (guaranteed to sum to one)."""

    matrix = np.asarray(sigma.values, dtype=float)
    ones = np.ones(matrix.shape[0], dtype=float)
    sigma_inv = np.linalg.pinv(matrix)
    weights = sigma_inv @ ones
    denom = float(ones @ sigma_inv @ ones)
    if denom == 0.0:
        raise ZeroDivisionError("Degenerate covariance produced zero normalisation")
    weights = weights / denom
    return xr.DataArray(weights, coords={"model": sigma["model"].values}, dims=("model",), name="weights")


def _fuse_series(predictions: np.ndarray, weights: xr.DataArray, bias: xr.DataArray) -> np.ndarray:
    """Fuse ``predictions`` using ``weights`` after subtracting ``bias``."""

    corrected = predictions - bias.values
    return corrected @ weights.values


def _evaluate_methods(dataset: xr.Dataset, truth: np.ndarray, predictions: np.ndarray) -> dict:
    """Evaluate covariance strategies against ``truth``."""

    method_cfgs = {
        "full": {"drop_cov": False, "shrinkage": "", "lam": None},
        "shrunk": {"drop_cov": False, "shrinkage": "lw", "lam": 0.25},
        "diag": {"drop_cov": True, "shrinkage": "", "lam": None},
        "ridge": {"drop_cov": False, "shrinkage": "ridge", "lam": 0.15},
    }

    results: dict[str, dict[str, float | xr.DataArray]] = {}
    bias = estimate_bias_from_collocation(dataset)

    for method, cfg in method_cfgs.items():
        ds = dataset.drop_vars("cov") if cfg["drop_cov"] else dataset
        sigma = build_sigma_from_collocation(ds, shrinkage=cfg["shrinkage"], lam=cfg["lam"])
        weights = _solve_gls_weights(sigma)
        fused = _fuse_series(predictions, weights, bias)
        rmse = float(np.sqrt(np.mean((fused - truth) ** 2)))
        results[method] = {"rmse": rmse, "weights": weights, "sigma": sigma}

    return results


def _format_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def _plot_results(
    summary: Dict[str, Dict[str, Dict[str, float | xr.DataArray]]],
    configs: List[ScenarioConfig],
    output_path: Path,
) -> None:
    """Create a publication-grade summary figure and save it to ``output_path``."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.figsize": (8.6, 3.4),
            "axes.prop_cycle": plt.cycler(
                color=["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
            ),
        }
    )

    fig, axes = plt.subplots(1, 3, constrained_layout=True)
    methods = ["full", "shrunk", "diag", "ridge"]

    # Panel A – RMSE comparison across scenarios.
    ax_rmse = axes[0]
    for method in methods:
        rmse_values = [summary[cfg.name][method]["rmse"] for cfg in configs]
        ax_rmse.plot(
            range(len(configs)),
            rmse_values,
            marker="o",
            linewidth=1.5,
            label=method.upper(),
        )
    ax_rmse.set_xticks(range(len(configs)))
    ax_rmse.set_xticklabels([cfg.name for cfg in configs])
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_title("A  RMSE across regimes")
    ax_rmse.legend(frameon=False, fontsize=7)
    _format_axes(ax_rmse)

    # Panel B – weights for the correlated regime.
    ax_weights = axes[1]
    correlated_cfg = next(cfg for cfg in configs if cfg.name == "correlated")
    correlated_res = summary[correlated_cfg.name]
    bar_positions = np.arange(len(MODEL_NAMES))
    bar_width = 0.18
    for idx, method in enumerate(methods):
        offset = (idx - 1.5) * bar_width
        weights = correlated_res[method]["weights"].values
        ax_weights.bar(
            bar_positions + offset,
            weights,
            width=bar_width,
            label=method.upper(),
            alpha=0.85,
        )
    ax_weights.axhline(0.0, color="black", linewidth=0.6)
    ax_weights.set_xticks(bar_positions)
    ax_weights.set_xticklabels(MODEL_NAMES)
    ax_weights.set_ylabel("Weight")
    ax_weights.set_title("B  Weights (correlated)")
    _format_axes(ax_weights)

    # Panel C – covariance heatmap vs diagonal approximation.
    ax_heatmap = axes[2]
    sigma_full = correlated_res["full"]["sigma"].values
    sigma_diag = np.diag(np.diag(sigma_full))
    diff = sigma_full - sigma_diag
    im = ax_heatmap.imshow(diff, cmap="RdBu_r", vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
    ax_heatmap.set_xticks(range(len(MODEL_NAMES)))
    ax_heatmap.set_xticklabels(MODEL_NAMES)
    ax_heatmap.set_yticks(range(len(MODEL_NAMES)))
    ax_heatmap.set_yticklabels(MODEL_NAMES)
    ax_heatmap.set_title("C  Cross-covariance emphasis")
    _format_axes(ax_heatmap)
    cbar = fig.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("ΔΣ")

    fig.suptitle("Covariance strategies for multivariate fusion", y=1.02)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)


def run_covariance_comparison(output_path: str | Path | None = None) -> dict:
    """Run the comparison study and optionally save the diagnostic figure."""

    configs = [
        ScenarioConfig(
            name="independent",
            description="Low correlation, reliable variances",
            cov=np.diag([0.8, 1.0, 1.2, 1.1]),
            biases=np.array([0.3, -0.2, 0.1, -0.1]),
            n_total=400,
            n_colloc=400,
            seed=42,
        ),
        ScenarioConfig(
            name="correlated",
            description="Two highly correlated high-skill models",
            cov=np.array(
                [
                    [0.9, 0.78, 0.35, 0.2],
                    [0.78, 0.85, 0.32, 0.18],
                    [0.35, 0.32, 1.3, 0.25],
                    [0.2, 0.18, 0.25, 1.6],
                ]
            ),
            biases=np.array([0.25, -0.1, 0.05, -0.2]),
            n_total=400,
            n_colloc=400,
            seed=43,
        ),
        ScenarioConfig(
            name="small_sample",
            description="Strong correlations estimated from short sample",
            cov=np.array(
                [
                    [0.9, 0.82, 0.4, 0.25],
                    [0.82, 0.88, 0.38, 0.22],
                    [0.4, 0.38, 1.2, 0.3],
                    [0.25, 0.22, 0.3, 1.5],
                ]
            ),
            biases=np.array([0.2, -0.15, 0.12, -0.22]),
            n_total=400,
            n_colloc=48,
            seed=44,
        ),
    ]

    summary: Dict[str, Dict[str, Dict[str, float | xr.DataArray]]] = {}
    detailed_weights: Dict[str, Dict[str, np.ndarray]] = {}

    for config in configs:
        dataset, truth, predictions = _simulate_collocation(config)
        method_results = _evaluate_methods(dataset, truth, predictions)
        summary[config.name] = method_results
        detailed_weights[config.name] = {
            method: res["weights"].values for method, res in method_results.items()
        }

    figure_path = (
        Path(output_path)
        if output_path is not None
        else Path(__file__).with_name("covariance_method_comparison.png")
    )
    _plot_results(summary, configs, figure_path)

    metrics = {
        cfg.name: {
            method: float(values["rmse"])
            for method, values in summary[cfg.name].items()
        }
        for cfg in configs
    }

    return {
        "rmse": metrics,
        "weights": detailed_weights,
        "figure_path": figure_path,
    }


if __name__ == "__main__":
    output = run_covariance_comparison()
    print("Saved comparison figure to", output["figure_path"])
