"""
Tests for the MTCH (Multiplicative-error Three-Cornered Hat) module.

Tests cover:
- Basic functionality with 3 and 4 products
- Input validation (type, shape, positivity, NaN/Inf)
- Sample-size warning
- Accuracy against known ground-truth (relative error < 30%)
- Class-based API (MTCH)
- Edge cases (near-zero data, very small errors)
- Consistency: different reference products yield similar results
"""

import numpy as np
import pytest
import warnings

from collocation import mtch, MTCH
from collocation.mtch import MIN_SAMPLES, MIN_PRODUCTS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_data(n=300, n_products=3, log_stds=None, seed=42):
    """Generate multiplicative-error synthetic data."""
    rng = np.random.default_rng(seed)
    if log_stds is None:
        log_stds = [0.10, 0.15, 0.20][:n_products]
    log_stds = np.asarray(log_stds)

    true_signal = rng.gamma(shape=4.0, scale=2.0, size=n)  # Positive

    products = []
    for sigma in log_stds:
        e = rng.normal(0, sigma, size=n)
        products.append(true_signal * np.exp(e))

    data = np.column_stack(products)
    true_rmse = log_stds * true_signal.mean()
    return data, true_signal, true_rmse


@pytest.fixture
def sample_3way():
    """3-product synthetic multiplicative-error data."""
    return _make_data(n=300, n_products=3, log_stds=[0.10, 0.15, 0.20])


@pytest.fixture
def sample_4way():
    """4-product synthetic multiplicative-error data."""
    return _make_data(n=400, n_products=4, log_stds=[0.08, 0.12, 0.18, 0.25])


# ---------------------------------------------------------------------------
# TestMtchFunctional – functional API
# ---------------------------------------------------------------------------

class TestMtchFunctional:
    """Tests for the mtch() function."""

    def test_returns_array_of_correct_length(self, sample_3way):
        data, _, _ = sample_3way
        rmse = mtch(data)
        assert isinstance(rmse, np.ndarray)
        assert rmse.shape == (3,)

    def test_all_rmse_positive(self, sample_3way):
        data, _, _ = sample_3way
        rmse = mtch(data)
        assert np.all(rmse > 0), "All RMSE values must be positive"

    def test_accuracy_3way(self, sample_3way):
        """MTCH RMSE estimates within 30% of true values (3 products)."""
        data, _, true_rmse = sample_3way
        rmse_est = mtch(data)
        rel_errors = np.abs(rmse_est - true_rmse) / true_rmse
        assert np.all(rel_errors < 0.30), (
            f"Relative errors exceed 30%: {rel_errors}"
        )

    def test_accuracy_4way(self, sample_4way):
        """MTCH RMSE estimates within 35% of true values (4 products)."""
        data, _, true_rmse = sample_4way
        rmse_est = mtch(data)
        rel_errors = np.abs(rmse_est - true_rmse) / true_rmse
        assert np.all(rel_errors < 0.35), (
            f"Relative errors exceed 35%: {rel_errors}"
        )

    def test_ordering_preserved(self, sample_3way):
        """Better (smaller log-error) products should have smaller RMSE."""
        data, _, true_rmse = sample_3way
        rmse_est = mtch(data)
        # true_rmse is monotonically increasing (stds=[0.10, 0.15, 0.20])
        assert rmse_est[0] < rmse_est[1] < rmse_est[2], (
            "Expected RMSE ordering: product 0 < 1 < 2"
        )

    def test_reference_idx_consistency(self, sample_3way):
        """Changing reference_idx should give similar (not identical) results."""
        data, _, _ = sample_3way
        rmse_0 = mtch(data, reference_idx=0)
        rmse_1 = mtch(data, reference_idx=1)
        rmse_2 = mtch(data, reference_idx=2)
        # All within 20% of each other per product
        for i in range(3):
            ref_val = (rmse_0[i] + rmse_1[i] + rmse_2[i]) / 3
            for rmse in [rmse_0, rmse_1, rmse_2]:
                assert abs(rmse[i] - ref_val) / ref_val < 0.20, (
                    f"reference_idx variants disagree on product {i}: "
                    f"{rmse_0[i]:.4f}, {rmse_1[i]:.4f}, {rmse_2[i]:.4f}"
                )

    # --- Input validation ---

    def test_raises_on_non_array(self):
        with pytest.raises(TypeError, match="numpy array"):
            mtch([[1, 2, 3], [4, 5, 6]])

    def test_raises_on_1d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            mtch(np.random.rand(100))

    def test_raises_on_fewer_than_3_products(self):
        with pytest.raises(ValueError, match="at least"):
            mtch(np.random.rand(100, 2) + 0.1)

    def test_raises_on_non_positive_values(self):
        data, _, _ = _make_data()
        data[5, 0] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            mtch(data)

    def test_raises_on_negative_values(self):
        data, _, _ = _make_data()
        data[10, 1] = -1.0
        with pytest.raises(ValueError, match="strictly positive"):
            mtch(data)

    def test_raises_on_nan(self):
        data, _, _ = _make_data()
        data[3, 2] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            mtch(data)

    def test_raises_on_inf(self):
        data, _, _ = _make_data()
        data[7, 0] = np.inf
        with pytest.raises(ValueError, match="NaN"):
            mtch(data)

    def test_raises_on_invalid_reference_idx(self):
        data, _, _ = _make_data()
        with pytest.raises(ValueError, match="reference_idx"):
            mtch(data, reference_idx=10)

    def test_warns_on_small_sample(self):
        data, _, _ = _make_data(n=MIN_SAMPLES - 1)
        with pytest.warns(UserWarning, match="Sample size"):
            mtch(data)

    def test_no_warning_at_min_samples(self):
        data, _, _ = _make_data(n=MIN_SAMPLES)
        # Should not raise a UserWarning about sample size
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            try:
                mtch(data)
            except UserWarning as e:
                # Optimiser convergence warnings are acceptable; sample-size is not
                if "Sample size" in str(e):
                    pytest.fail(f"Unexpected sample-size warning at n={MIN_SAMPLES}")


# ---------------------------------------------------------------------------
# TestMtchClass – class-based API
# ---------------------------------------------------------------------------

class TestMtchClass:
    """Tests for the MTCH class."""

    def test_fit_returns_self(self, sample_3way):
        data, _, _ = sample_3way
        model = MTCH(data)
        result = model.fit()
        assert result is model

    def test_rmse_attribute_set_after_fit(self, sample_3way):
        data, _, _ = sample_3way
        model = MTCH(data)
        assert model.rmse is None
        model.fit()
        assert model.rmse is not None
        assert model.rmse.shape == (3,)

    def test_weights_attribute_set_after_compute_weights(self, sample_3way):
        data, _, _ = sample_3way
        model = MTCH(data)
        model.fit()
        assert model.weights is None
        model.compute_weights()
        assert model.weights is not None
        assert model.weights.shape == (3,)

    def test_weights_sum_to_one(self, sample_3way):
        data, _, _ = sample_3way
        model = MTCH(data)
        model.fit()
        model.compute_weights()
        assert abs(model.weights.sum() - 1.0) < 1e-9

    def test_fused_product_shape(self, sample_3way):
        data, _, _ = sample_3way
        n = data.shape[0]
        model = MTCH(data)
        model.fit()
        model.compute_weights()
        fused = model.fuse()
        assert fused.shape == (n,)

    def test_fused_product_positive(self, sample_3way):
        data, _, _ = sample_3way
        model = MTCH(data)
        model.fit()
        model.compute_weights()
        fused = model.fuse()
        assert np.all(fused > 0)

    def test_run_returns_dict(self, sample_3way):
        data, _, _ = sample_3way
        model = MTCH(data)
        results = model.run()
        assert isinstance(results, dict)
        for key in ("rmse", "log_variances", "weights", "fused"):
            assert key in results

    def test_compute_weights_raises_without_fit(self, sample_3way):
        data, _, _ = sample_3way
        model = MTCH(data)
        with pytest.raises(RuntimeError, match="fit"):
            model.compute_weights()

    def test_fuse_raises_without_weights(self, sample_3way):
        data, _, _ = sample_3way
        model = MTCH(data)
        model.fit()
        with pytest.raises(RuntimeError, match="compute_weights"):
            model.fuse()

    def test_class_vs_function_consistency(self, sample_3way):
        """Class-based and functional API must give identical results."""
        data, _, _ = sample_3way
        rmse_fn = mtch(data)

        model = MTCH(data)
        model.fit()
        np.testing.assert_array_equal(model.rmse, rmse_fn)

    def test_summary_prints_without_error(self, sample_3way, capsys):
        data, _, _ = sample_3way
        model = MTCH(data)
        model.run()
        model.summary()
        captured = capsys.readouterr()
        assert "RMSE" in captured.out

    def test_4_products(self, sample_4way):
        data, _, _ = sample_4way
        model = MTCH(data)
        results = model.run()
        assert results["rmse"].shape == (4,)
        assert results["weights"].shape == (4,)
        assert abs(results["weights"].sum() - 1.0) < 1e-9

    def test_invalid_data_raises(self):
        with pytest.raises(ValueError):
            MTCH(np.random.rand(100))  # 1-D

    def test_log_variances_consistent_with_rmse(self, sample_3way):
        """log_variances ≈ (RMSE / mean(data))²."""
        data, _, _ = sample_3way
        model = MTCH(data)
        model.fit()
        mean_data = np.mean(data, axis=0)
        expected_log_var = (model.rmse / mean_data) ** 2
        np.testing.assert_allclose(model.log_variances, expected_log_var, rtol=1e-6)


# ---------------------------------------------------------------------------
# TestMtchNumerical – numerical edge cases
# ---------------------------------------------------------------------------

class TestMtchNumerical:
    """Edge-case and numerical robustness tests."""

    def test_very_small_errors(self):
        """Very precise measurements: log-error stds = 0.02."""
        data, _, true_rmse = _make_data(
            n=500, n_products=3, log_stds=[0.02, 0.03, 0.04], seed=7
        )
        rmse = mtch(data)
        rel_errors = np.abs(rmse - true_rmse) / true_rmse
        # Allow more tolerance for very small errors (harder to estimate)
        assert np.all(rel_errors < 0.60), (
            f"Relative errors too large for small-error case: {rel_errors}"
        )
        assert np.all(rmse > 0)

    def test_large_number_of_products(self):
        """N=5 products should work without error."""
        data, _, _ = _make_data(
            n=500, n_products=5,
            log_stds=[0.08, 0.10, 0.12, 0.15, 0.20], seed=99
        )
        rmse = mtch(data)
        assert rmse.shape == (5,)
        assert np.all(rmse > 0)

    def test_output_finite(self, sample_3way):
        """All output values must be finite."""
        data, _, _ = sample_3way
        rmse = mtch(data)
        assert np.all(np.isfinite(rmse)), "RMSE contains NaN or Inf"

    def test_deterministic(self, sample_3way):
        """Two calls with identical data should return identical results."""
        data, _, _ = sample_3way
        rmse_1 = mtch(data)
        rmse_2 = mtch(data)
        np.testing.assert_array_equal(rmse_1, rmse_2)

    def test_scale_invariance(self):
        """
        Scaling all data by a constant factor k should scale RMSE by k.
        (RMSE is in original units and proportional to mean(Z).)
        """
        data, _, _ = _make_data(n=300, seed=11)
        k = 3.7
        rmse_original = mtch(data)
        rmse_scaled = mtch(data * k)
        np.testing.assert_allclose(rmse_scaled, rmse_original * k, rtol=0.01)
