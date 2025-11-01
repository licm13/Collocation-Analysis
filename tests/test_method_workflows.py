"""Integration oriented tests that exercise multiple collocation components."""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collocation.etcc import ETCC, TripleCollocation
from collocation.eivd import eivd
from collocation.simple_average import inverse_variance_weights, simple_average
from collocation.tc import tc_with_rescaling
from collocation.utils import kge_objfun, mse_judge


def _generate_truth_and_products(seed: int = 123, n: int = 600):
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 6 * np.pi, n)
    truth = np.sin(time) + 0.2 * np.sin(2 * time) + 0.05 * rng.standard_normal(n)
    truth += 0.1 * np.cos(0.5 * time)
    noises = [0.35, 0.55, 0.9]
    products = tuple(truth + rng.standard_normal(n) * level for level in noises)
    return truth, products, noises


def test_triple_collocation_prefers_cleanest_sensor():
    truth, products, noise_levels = _generate_truth_and_products()
    merger = TripleCollocation()
    merged = merger.merge(*products)
    weights = np.array([
        merger.weights["wx"],
        merger.weights["wy"],
        merger.weights["wz"],
    ])

    assert merged.shape == truth.shape
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-6)

    cleanest_index = int(np.argmin(noise_levels))
    assert weights[cleanest_index] == weights.max()

    correlations = [np.corrcoef(truth, product)[0, 1] for product in products]
    merged_corr = np.corrcoef(truth, merged)[0, 1]
    assert merged_corr >= max(correlations) - 1e-3


def test_etcc_outperforms_classical_tc_in_correlation_metric():
    truth, products, _ = _generate_truth_and_products(seed=456)

    tc_merger = TripleCollocation()
    tc_result = tc_merger.merge(*products)
    tc_corr = np.corrcoef(truth, tc_result)[0, 1]

    etcc_merger = ETCC(weight_increment=0.05)
    etcc_result = etcc_merger.merge(*products)
    etcc_corr = np.corrcoef(truth, etcc_result)[0, 1]

    assert etcc_corr >= tc_corr - 1e-3
    assert math.isclose(
        etcc_merger.weights["wx"] + etcc_merger.weights["wy"] + etcc_merger.weights["wz"],
        1.0,
        rel_tol=1e-6,
    )


def test_eivd_detects_error_cross_correlation():
    rng = np.random.default_rng(789)
    n = 800
    truth = rng.normal(size=n)
    independent = truth + rng.normal(scale=0.4, size=n)
    shared_error = rng.normal(scale=0.6, size=n)
    correlated_a = truth + shared_error + rng.normal(scale=0.2, size=n)
    correlated_b = truth + shared_error + rng.normal(scale=0.25, size=n)

    matrix = np.column_stack([independent, correlated_a, correlated_b])
    eeeT, _, _, _, _ = eivd(matrix)

    assert eeeT[1, 2] > 0.0


def test_simple_average_with_inverse_variance_weights_matches_theory():
    truth, products, _ = _generate_truth_and_products(seed=321)
    residuals = np.vstack([p - truth for p in products])
    empirical_variances = np.var(residuals, axis=1)

    weights = inverse_variance_weights(empirical_variances)
    averaged = simple_average(np.vstack(products), weights=weights, axis=0)
    uniform = simple_average(np.vstack(products), axis=0)

    np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-6)
    assert averaged.shape == truth.shape

    _, kge_weighted = kge_objfun(averaged, truth)
    _, kge_uniform = kge_objfun(uniform, truth)
    assert kge_weighted <= kge_uniform + 1e-6

    data, omega = mse_judge(np.nan, 0.2)
    assert data == 0.0 and omega == 0.0


def test_tc_with_rescaling_preserves_reference_variance():
    truth, products, _ = _generate_truth_and_products(seed=654)
    tri_matrix = np.column_stack(products)
    _, _, _, _, rescaled = tc_with_rescaling(tri_matrix, reference_idx=1)
    ref_variance = np.var(products[1])
    np.testing.assert_allclose(np.var(rescaled[:, 1]), ref_variance, rtol=1e-6)
