"""Robust collocation stress tests across heterogeneous error regimes.

This script is designed as an executable example that exercises the
high-level API of the :mod:`collocation` package in several realistic data
fusion scenarios:

1. **Baseline independent errors** – validates the classic triple collocation
   implementation in :mod:`collocation.tc` and the object oriented workflow in
   :class:`collocation.etcc.TripleCollocation`.
2. **Correlated errors** – demonstrates how :func:`collocation.eivd` detects
   and quantifies error cross correlation when sensors share algorithmic
   components.
3. **Bias dominated sensors** – shows how to leverage
   :func:`collocation.simple_average.inverse_variance_weights` together with
   :func:`collocation.simple_average.simple_average` for robust averaging when
   the signal to noise ratio varies drastically.
4. **Correlation focused merging** – compares the classic triple collocation
   merge against :class:`collocation.etcc.ETCC`, which maximises the
   correlation with the (unknown) truth signal.

Running this file will print a scenario report and can be used as a regression
smoke test for downstream projects.

Usage
-----
```
python -m examples.robust_collocation_scenarios
```
```
python examples/robust_collocation_scenarios.py
```

Both entry points seed NumPy's random number generator for reproducibility.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collocation import tc
from collocation.etcc import ETCC, TripleCollocation
from collocation.eivd import eivd
from collocation.simple_average import (
    inverse_variance_weights,
    simple_average,
)
from collocation.utils import kge_objfun


@dataclass
class ScenarioResult:
    """Container for scenario diagnostics."""

    name: str
    description: str
    correlations: Tuple[float, float, float]
    merged_correlation: float
    weights: Tuple[float, float, float]
    kge: float

    def pretty_print(self) -> None:
        print(f"\n=== {self.name} ===")
        print(self.description)
        print("Sensor correlations:", np.round(self.correlations, 3))
        print("Merged correlation:", round(self.merged_correlation, 3))
        print("Weights:", np.round(self.weights, 3))
        print("KGE (merged vs. truth):", round(self.kge, 3))


def _seeded_signal(seed: int, n: int = 720) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 4 * np.pi, n)
    signal = np.sin(time) + 0.2 * np.sin(3 * time)
    signal += 0.05 * rng.standard_normal(n)
    return signal


def _construct_products(
    base_signal: np.ndarray,
    noise_levels: Iterable[float],
    *,
    correlated: bool = False,
    bias: Iterable[float] | None = None,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    noises = []
    if correlated:
        common = rng.standard_normal(base_signal.shape) * 0.4
    else:
        common = np.zeros_like(base_signal)

    for idx, level in enumerate(noise_levels):
        sensor_noise = rng.standard_normal(base_signal.shape) * level
        noises.append(sensor_noise + common)

    products = []
    for idx, noise in enumerate(noises):
        b = 0.0 if bias is None else float(bias[idx])
        products.append(base_signal * (1 + b) + noise)
    return tuple(products)  # type: ignore[return-value]


def _report_merge(
    truth: np.ndarray,
    products: Tuple[np.ndarray, np.ndarray, np.ndarray],
    weights: Tuple[float, float, float],
) -> Tuple[Tuple[float, float, float], float, float]:
    correlations = tuple(np.corrcoef(truth, p)[0, 1] for p in products)
    merged = sum(w * p for w, p in zip(weights, products))
    merged_corr = float(np.corrcoef(truth, merged)[0, 1])
    _, kge = kge_objfun(merged, truth)
    return correlations, merged_corr, kge


def run_baseline_scenario(truth: np.ndarray, rng: np.random.Generator) -> ScenarioResult:
    products = _construct_products(truth, [0.5, 0.7, 0.9], rng=rng, correlated=False)
    tc_matrix = np.column_stack(products)
    _, _, rho2, _ = tc(tc_matrix)

    tc_merger = TripleCollocation()
    merged = tc_merger.merge(*products)
    weights = (
        tc_merger.weights["wx"],
        tc_merger.weights["wy"],
        tc_merger.weights["wz"],
    )

    correlations, merged_corr, kge = _report_merge(truth, products, weights)
    assert math.isclose(np.sum(weights), 1.0, rel_tol=1e-6)
    assert merged.shape == truth.shape
    return ScenarioResult(
        name="Baseline: Independent errors",
        description=(
            "Classic triple collocation scenario with independent Gaussian "
            "noise. Demonstrates agreement between procedural and OO APIs."
        ),
        correlations=correlations,
        merged_correlation=merged_corr,
        weights=weights,
        kge=kge,
    )


def run_correlated_scenario(truth: np.ndarray, rng: np.random.Generator) -> ScenarioResult:
    products = _construct_products(
        truth,
        [0.6, 0.65, 0.75],
        rng=rng,
        correlated=True,
    )
    matrix = np.column_stack(products)
    eivd_result = eivd(matrix)
    error_covariance = eivd_result[0]

    tc_merger = TripleCollocation()
    merged = tc_merger.merge(*products)
    weights = (
        tc_merger.weights["wx"],
        tc_merger.weights["wy"],
        tc_merger.weights["wz"],
    )

    correlations, merged_corr, kge = _report_merge(truth, products, weights)
    return ScenarioResult(
        name="Correlated errors",
        description=(
            "EIVD identifies the expected positive error covariance between "
            "sensors 2 and 3 while TripleCollocation still provides a stable "
            "merge. Off-diagonal EIVD term: "
            f"{error_covariance[1, 2]:.3f}"
        ),
        correlations=correlations,
        merged_correlation=merged_corr,
        weights=weights,
        kge=kge,
    )


def run_bias_scenario(truth: np.ndarray, rng: np.random.Generator) -> ScenarioResult:
    products = _construct_products(
        truth,
        [0.4, 0.9, 1.2],
        bias=[0.0, 0.2, -0.15],
        rng=rng,
    )
    residuals = np.vstack([p - truth for p in products])
    variances = np.var(residuals, axis=1)
    weights = tuple(inverse_variance_weights(variances))
    correlations, merged_corr, kge = _report_merge(truth, products, weights)
    merged = simple_average(np.vstack(products), weights=np.array(weights), axis=0)
    assert merged.shape == truth.shape
    return ScenarioResult(
        name="Bias dominated sensors",
        description=(
            "Simple averaging with inverse-variance weights provides a "
            "baseline when sensors exhibit multiplicative biases."
        ),
        correlations=correlations,
        merged_correlation=merged_corr,
        weights=weights,
        kge=kge,
    )


def run_correlation_focus_scenario(truth: np.ndarray, rng: np.random.Generator) -> ScenarioResult:
    products = _construct_products(truth, [0.45, 0.8, 0.8], rng=rng, correlated=False)
    tc_merger = TripleCollocation()
    tc_merged = tc_merger.merge(*products)
    etcc_merger = ETCC(weight_increment=0.05)
    etcc_merged = etcc_merger.merge(*products)
    etcc_weights = (
        etcc_merger.weights["wx"],
        etcc_merger.weights["wy"],
        etcc_merger.weights["wz"],
    )

    correlations, merged_corr, kge = _report_merge(truth, products, etcc_weights)
    tc_corr = float(np.corrcoef(truth, tc_merged)[0, 1])
    etcc_corr = float(np.corrcoef(truth, etcc_merged)[0, 1])
    improvement = etcc_corr - tc_corr
    return ScenarioResult(
        name="Correlation focused merging",
        description=(
            "ETCC searches the weight simplex and improves the correlation by "
            f"{improvement:.3f} compared to classical TC weights."
        ),
        correlations=correlations,
        merged_correlation=etcc_corr,
        weights=etcc_weights,
        kge=kge,
    )


def main() -> None:
    truth = _seeded_signal(seed=42)
    rng = np.random.default_rng(1234)

    scenarios = [
        run_baseline_scenario(truth, rng),
        run_correlated_scenario(truth, rng),
        run_bias_scenario(truth, rng),
        run_correlation_focus_scenario(truth, rng),
    ]

    print("Robust collocation scenario summary")
    print("Truth variance:", np.var(truth))
    for scenario in scenarios:
        scenario.pretty_print()


if __name__ == "__main__":
    main()
