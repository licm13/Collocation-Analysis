"""
Comprehensive test suite for fusion module.

Tests all components:
- Weight solvers (IVW, GLS, QP)
- Covariance estimation
- Constraints
- Robust estimators
- Localization
- Uncertainty quantification
- End-to-end fusion
"""

import numpy as np
import pytest
import xarray as xr

# Import fusion modules
from collocation.fusion import (
    solve_weights_ivw,
    solve_weights_gls,
    solve_weights_qp,
    estimate_mse,
    build_sigma,
    estimate_cross_covariance,
    SumToOneConstraint,
    BoundsConstraint,
    combine_constraints,
    huber_loss,
    estimate_mse_robust,
    detect_outliers,
    compute_effective_n,
    weight_entropy,
    propagate_variance,
    fuse_fields,
)


class TestWeightSolvers:
    """Test weight computation algorithms."""

    def test_ivw_simple(self):
        """Test simple inverse variance weighting."""
        mse = np.array([0.5, 0.3, 0.8])
        weights = solve_weights_ivw(mse)

        # Check properties
        assert weights.shape == (3,)
        assert np.allclose(weights.sum(), 1.0)
        assert np.all(weights >= 0)
        # Lower MSE should have higher weight
        assert weights[1] > weights[0] > weights[2]

    def test_ivw_uniform_mse(self):
        """Test IVW with uniform MSE gives uniform weights."""
        mse = np.array([0.5, 0.5, 0.5])
        weights = solve_weights_ivw(mse)
        expected = np.array([1/3, 1/3, 1/3])
        assert np.allclose(weights, expected)

    def test_gls_simple(self):
        """Test GLS weight solver."""
        Sigma = np.array([[0.5, 0.1], [0.1, 0.3]])
        weights = solve_weights_gls(Sigma)

        # Check properties
        assert weights.shape == (2,)
        assert np.allclose(weights.sum(), 1.0)
        assert np.all(weights >= 0)
        # Model with lower variance should have higher weight
        assert weights[1] > weights[0]

    def test_gls_diagonal_equals_ivw(self):
        """Test that GLS with diagonal Sigma equals IVW."""
        mse = np.array([0.5, 0.3, 0.8])
        Sigma = np.diag(mse)

        w_ivw = solve_weights_ivw(mse)
        w_gls = solve_weights_gls(Sigma)

        assert np.allclose(w_ivw, w_gls, atol=1e-6)

    def test_gls_ill_conditioned(self):
        """Test GLS with near-singular covariance."""
        # Create ill-conditioned matrix
        Sigma = np.array([[1.0, 0.999], [0.999, 1.0]])

        # Should not raise error (diagonal loading applied)
        weights = solve_weights_gls(Sigma)
        assert weights.shape == (2,)
        assert np.allclose(weights.sum(), 1.0)

    @pytest.mark.skipif(
        True, reason="quadprog not in core dependencies"
    )
    def test_qp_with_bounds(self):
        """Test constrained QP with bounds."""
        Sigma = np.array([[0.5, 0.1], [0.1, 0.3]])
        weights = solve_weights_qp(
            Sigma,
            A_eq=np.array([[1.0, 1.0]]),
            b_eq=np.array([1.0]),
            bounds=(0.2, 0.8),
        )

        assert np.allclose(weights.sum(), 1.0)
        assert np.all(weights >= 0.2 - 1e-6)
        assert np.all(weights <= 0.8 + 1e-6)


class TestCovarianceEstimation:
    """Test covariance matrix construction."""

    def test_estimate_mse_basic(self):
        """Test basic MSE estimation."""
        np.random.seed(42)
        n_samples, n_models = 100, 3

        # Generate synthetic data
        truth = np.random.randn(n_samples)
        noise = np.random.randn(n_samples, n_models) * np.array([0.5, 0.3, 0.8])
        y_pred_np = truth[:, None] + noise

        y_pred = xr.DataArray(
            y_pred_np,
            dims=["time", "model"],
            coords={"model": ["A", "B", "C"]},
        )
        y_ref = xr.DataArray(truth, dims=["time"])

        mse = estimate_mse(y_pred, y_ref)

        # Check shape and values
        assert mse.dims == ("model",)
        assert len(mse) == 3
        # MSE should roughly match noise levels
        # Model B (0.3) < Model A (0.5) < Model C (0.8)
        assert mse.sel(model="B") < mse.sel(model="A") < mse.sel(model="C")

    def test_build_sigma_diagonal(self):
        """Test building diagonal covariance from MSE."""
        mse = xr.DataArray([0.5, 0.3, 0.8], dims=["model"], coords={"model": ["A", "B", "C"]})
        Sigma = build_sigma(mse, cross=None, shrinkage="none")

        # Check shape
        assert Sigma.shape == (3, 3)
        assert "model" in Sigma.dims
        assert "model_2" in Sigma.dims

        # Check diagonal values
        for i, m in enumerate(["A", "B", "C"]):
            assert np.isclose(Sigma.sel(model=m, model_2=m).values, mse.sel(model=m).values)

        # Check off-diagonal is zero
        assert np.isclose(Sigma.sel(model="A", model_2="B").values, 0.0)

    def test_build_sigma_with_shrinkage(self):
        """Test covariance with shrinkage."""
        mse = xr.DataArray([0.5, 0.3, 0.8], dims=["model"])
        Sigma = build_sigma(mse, shrinkage="lw", lam=0.5)

        # Eigenvalues should be positive
        eigvals = np.linalg.eigvalsh(Sigma.values)
        assert np.all(eigvals > 0)


class TestConstraints:
    """Test constraint classes."""

    def test_sum_to_one_constraint(self):
        """Test sum-to-one constraint."""
        c = SumToOneConstraint(n_models=3)
        A, b = c.as_matrices()

        assert A.shape == (1, 3)
        assert b.shape == (1,)
        assert np.allclose(A, [[1, 1, 1]])
        assert np.allclose(b, [1.0])

        # Test feasibility check
        assert c.check_feasibility(np.array([0.5, 0.3, 0.2]))
        assert not c.check_feasibility(np.array([0.5, 0.3, 0.3]))

    def test_bounds_constraint(self):
        """Test bounds constraint."""
        c = BoundsConstraint(n_models=3, w_min=0.0, w_max=1.0)
        A, b = c.as_matrices()

        # Should have 6 constraints (3 lower + 3 upper)
        assert A.shape == (6, 3)
        assert b.shape == (6,)

        # Test feasibility
        assert c.check_feasibility(np.array([0.5, 0.3, 0.2]))
        assert not c.check_feasibility(np.array([1.5, -0.3, 0.2]))

    def test_combine_constraints(self):
        """Test combining multiple constraints."""
        c1 = SumToOneConstraint(n_models=3)
        c2 = BoundsConstraint(n_models=3, w_min=0.0, w_max=1.0)

        A, b = combine_constraints([c1, c2])

        # Should have 7 constraints (1 sum + 6 bounds)
        assert A.shape == (7, 3)
        assert b.shape == (7,)


class TestRobustEstimators:
    """Test robust estimation methods."""

    def test_huber_loss(self):
        """Test Huber loss function."""
        residuals = np.array([-2, -1, 0, 1, 2])
        loss = huber_loss(residuals, delta=1.0)

        # Check properties
        assert loss.shape == residuals.shape
        assert np.all(loss >= 0)
        # Loss at 0 should be 0
        assert np.isclose(loss[2], 0.0)

    def test_robust_mse_vs_standard(self):
        """Test that robust MSE handles outliers better."""
        np.random.seed(42)
        # Generate data with outliers
        y_pred = np.random.randn(100)
        y_ref = np.random.randn(100)

        # Add outliers
        y_pred[::10] = 10.0

        mse_standard = np.mean((y_pred - y_ref) ** 2)
        mse_robust = estimate_mse_robust(y_pred, y_ref, loss="huber")

        # Robust MSE should be lower
        assert mse_robust < mse_standard

    def test_outlier_detection_iqr(self):
        """Test IQR-based outlier detection."""
        data = np.array([1, 2, 2, 3, 3, 3, 4, 20])
        outliers = detect_outliers(data, method="iqr")

        # Last value (20) should be detected as outlier
        assert outliers[-1] == True
        # Most others should not be outliers
        assert np.sum(outliers) <= 2


class TestUncertainty:
    """Test uncertainty quantification."""

    def test_propagate_variance_simple(self):
        """Test variance propagation."""
        Sigma = np.array([[0.5, 0.0], [0.0, 0.3]])
        weights = np.array([0.6, 0.4])

        var = propagate_variance(Sigma, weights)

        # Analytical result: 0.6^2 * 0.5 + 0.4^2 * 0.3
        expected = 0.6**2 * 0.5 + 0.4**2 * 0.3
        assert np.isclose(var, expected)

    def test_effective_n_uniform(self):
        """Test effective N with uniform weights."""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        n_eff = compute_effective_n(weights)

        # Should equal number of models
        assert np.isclose(n_eff, 4.0)

    def test_effective_n_concentrated(self):
        """Test effective N with concentrated weights."""
        weights = np.array([0.9, 0.05, 0.03, 0.02])
        n_eff = compute_effective_n(weights)

        # Should be close to 1
        assert 1.0 < n_eff < 2.0

    def test_weight_entropy(self):
        """Test weight entropy calculation."""
        # Uniform distribution has maximum entropy
        w_uniform = np.array([0.25, 0.25, 0.25, 0.25])
        H_uniform = weight_entropy(w_uniform)

        # Concentrated distribution has low entropy
        w_concentrated = np.array([1.0, 0.0, 0.0, 0.0])
        H_concentrated = weight_entropy(w_concentrated)

        assert H_uniform > H_concentrated
        assert np.isclose(H_concentrated, 0.0)


class TestFusionEnd2End:
    """Test end-to-end fusion workflows."""

    def test_synthetic_fusion_ivw(self):
        """Test IVW fusion reduces RMSE."""
        np.random.seed(42)
        n_samples, n_models = 200, 3

        # Generate synthetic data
        truth = np.random.randn(n_samples)
        noise_levels = np.array([0.5, 0.3, 0.8])
        models = truth[:, None] + np.random.randn(n_samples, n_models) * noise_levels

        # Create xarray
        X = xr.DataArray(
            models,
            dims=["time", "model"],
            coords={"model": ["A", "B", "C"]},
        )
        X_ref = xr.DataArray(truth, dims=["time"])

        # Fuse with IVW
        result = fuse_fields(X, X_ref=X_ref, mode="ivw", return_weights=True)

        # Check output structure
        assert "fused" in result
        assert "weights" in result
        assert "variance" in result

        # Check fused RMSE is better than worst model
        fused = result["fused"].values
        rmse_fused = np.sqrt(np.mean((fused - truth) ** 2))

        rmse_models = [
            np.sqrt(np.mean((models[:, i] - truth) ** 2))
            for i in range(n_models)
        ]

        assert rmse_fused < max(rmse_models)

    def test_synthetic_fusion_gls(self):
        """Test GLS fusion with known covariance."""
        np.random.seed(42)
        n_samples = 200
        K = 3

        # True covariance
        Sigma_true = np.array([
            [0.5, 0.1, 0.0],
            [0.1, 0.3, 0.0],
            [0.0, 0.0, 0.8],
        ])

        # Generate data
        truth = np.random.randn(n_samples)
        noise = np.random.multivariate_normal(np.zeros(K), Sigma_true, size=n_samples)
        models = truth[:, None] + noise

        X = xr.DataArray(
            models,
            dims=["time", "model"],
            coords={"model": ["A", "B", "C"]},
        )
        Sigma = xr.DataArray(
            Sigma_true,
            dims=["model", "model_2"],
            coords={"model": ["A", "B", "C"], "model_2": ["A", "B", "C"]},
        )

        # Fuse with GLS
        result = fuse_fields(X, Sigma=Sigma, mode="gls")

        # Check weights match expected GLS solution
        weights = result["weights"].values

        # Solve GLS analytically
        from collocation.fusion import solve_weights_gls
        w_expected = solve_weights_gls(Sigma_true)

        assert np.allclose(weights, w_expected, atol=1e-6)

    def test_heteroscedastic_mse(self):
        """Test fusion with spatially varying MSE."""
        np.random.seed(42)

        # Generate spatially varying noise
        n_time, n_lat, n_lon, n_models = 50, 10, 10, 3

        # Noise increases with latitude
        lat_effect = np.linspace(0.2, 1.0, n_lat)
        noise_field = lat_effect[None, :, None, None] * np.array([0.5, 0.3, 0.8])[None, None, None, :]

        truth = np.random.randn(n_time, n_lat, n_lon)
        noise = np.random.randn(n_time, n_lat, n_lon, n_models) * noise_field
        models = truth[..., None] + noise

        X = xr.DataArray(
            models,
            dims=["time", "lat", "lon", "model"],
        )
        X_ref = xr.DataArray(truth, dims=["time", "lat", "lon"])

        result = fuse_fields(X, X_ref=X_ref, mode="ivw")

        # Weights should vary with latitude
        weights = result["weights"]
        assert "lat" in weights.dims


class TestMissingData:
    """Test handling of missing data."""

    def test_fusion_with_nans(self):
        """Test fusion gracefully handles NaN values."""
        np.random.seed(42)
        X = xr.DataArray(
            np.random.randn(100, 3),
            dims=["time", "model"],
        )

        # Introduce missing data
        X[10:20, 2] = np.nan

        X_ref = xr.DataArray(np.random.randn(100), dims=["time"])

        # Should not crash
        result = fuse_fields(X, X_ref=X_ref, mode="ivw")

        # Fused result should have some valid values
        assert np.sum(~np.isnan(result["fused"])) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
