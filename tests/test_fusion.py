import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
import importlib.util

# 添加绘图支持
try:
    import matplotlib
    matplotlib.use("Agg")  # 非交互式后端
    import matplotlib.pyplot as plt
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

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


def save_fusion_test_figure(fig, test_name, script_name="test_fusion"):
    """保存融合测试图片到figures文件夹"""
    if not HAS_MATPLOTLIB:
        return
    
    # 创建figures目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(script_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # 保存图片
    filename = f"{script_name}_{test_name}.png"
    filepath = os.path.join(fig_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Fusion test figure saved: {filepath}")


def plot_fusion_weights(weights, method_name, test_name):
    """绘制融合权重结果"""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f'{method_name} 权重测试结果 - {test_name}', fontsize=14, fontweight='bold')
    
    # 1. 权重条形图
    ax1 = axes[0, 0]
    if weights.ndim == 1:
        n_models = len(weights)
        x = np.arange(n_models)
        ax1.bar(x, weights, alpha=0.7, color='steelblue')
        ax1.set_title('融合权重')
        ax1.set_xlabel('模型')
        ax1.set_ylabel('权重')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'M{i+1}' for i in range(n_models)])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加权重数值标签
        for i, w in enumerate(weights):
            ax1.text(i, w + 0.01, f'{w:.3f}', ha='center', va='bottom')
    
    # 2. 权重分布
    ax2 = axes[0, 1]
    if weights.ndim == 1:
        ax2.pie(weights, labels=[f'M{i+1}' for i in range(len(weights))], 
                autopct='%1.1f%%', startangle=90)
        ax2.set_title('权重分布')
    
    # 3. 权重统计
    ax3 = axes[1, 0]
    if weights.ndim == 1:
        stats = {
            '和': np.sum(weights),
            '最大值': np.max(weights),
            '最小值': np.min(weights),
            '均值': np.mean(weights),
            '标准差': np.std(weights),
            '熵': weight_entropy(weights) if 'weight_entropy' in globals() else np.nan
        }
        y_pos = 0.9
        for key, value in stats.items():
            if np.isfinite(value):
                ax3.text(0.1, y_pos, f'{key}: {value:.4f}', transform=ax3.transAxes, fontsize=10)
            else:
                ax3.text(0.1, y_pos, f'{key}: N/A', transform=ax3.transAxes, fontsize=10)
            y_pos -= 0.12
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('权重统计')
    
    # 4. 方法信息
    ax4 = axes[1, 1]
    ax4.text(0.1, 0.9, f'方法: {method_name}', transform=ax4.transAxes, fontsize=12, fontweight='bold')
    ax4.text(0.1, 0.8, f'测试: {test_name}', transform=ax4.transAxes, fontsize=10)
    ax4.text(0.1, 0.6, f'模型数量: {len(weights) if weights.ndim == 1 else "N/A"}', 
             transform=ax4.transAxes, fontsize=10)
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    save_fusion_test_figure(fig, test_name)


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
        
        # 创建可视化
        if HAS_MATPLOTLIB:
            plot_fusion_weights(weights, 'IVW', 'simple_test')

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
        importlib.util.find_spec("quadprog") is None,
        reason="quadprog not installed; install optional dependency to run QP tests",
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


class TestVectorizedWeights:
    """Test vectorized weight computation performance and correctness."""

    def test_vectorized_ivw_matches_loop(self):
        """Test that vectorized IVW produces same results as loop version."""
        np.random.seed(42)
        n_lat, n_lon, n_models = 20, 20, 3

        # Create spatially varying covariance matrices
        variances = np.abs(np.random.randn(n_lat, n_lon, n_models)) + 0.1
        Sigma = np.zeros((n_lat, n_lon, n_models, n_models))
        for i in range(n_lat):
            for j in range(n_lon):
                Sigma[i, j] = np.diag(variances[i, j])

        # Import the vectorized function
        from collocation.fusion.fuse import _compute_ivw_weights_vectorized

        # Compute with vectorized version
        weights_vectorized = _compute_ivw_weights_vectorized(Sigma)

        # Compute with loop version
        weights_loop = np.zeros((n_lat, n_lon, n_models))
        for i in range(n_lat):
            for j in range(n_lon):
                mse = np.diag(Sigma[i, j])
                inv_var = 1.0 / mse
                weights_loop[i, j] = inv_var / inv_var.sum()

        # Should match
        assert np.allclose(weights_vectorized, weights_loop, rtol=1e-10)

    def test_vectorized_gls_matches_loop(self):
        """Test that vectorized GLS produces same results as loop version."""
        np.random.seed(42)
        n_lat, n_lon, n_models = 10, 10, 3

        # Create random positive definite covariance matrices
        Sigma = np.zeros((n_lat, n_lon, n_models, n_models))
        for i in range(n_lat):
            for j in range(n_lon):
                A = np.random.randn(n_models, n_models)
                Sigma[i, j] = A @ A.T + 0.1 * np.eye(n_models)

        # Import the vectorized function
        from collocation.fusion.fuse import _compute_gls_weights_vectorized

        # Compute with vectorized version
        weights_vectorized = _compute_gls_weights_vectorized(Sigma)

        # Compute with loop version using the existing solver
        from collocation.fusion.weights import solve_weights_gls
        weights_loop = np.zeros((n_lat, n_lon, n_models))
        for i in range(n_lat):
            for j in range(n_lon):
                weights_loop[i, j] = solve_weights_gls(Sigma[i, j], sum_to_one=True)

        # Should match (with some tolerance for numerical differences)
        assert np.allclose(weights_vectorized, weights_loop, rtol=1e-5, atol=1e-6)

    def test_vectorized_weights_sum_to_one(self):
        """Test that vectorized weights sum to 1."""
        np.random.seed(42)
        n_lat, n_lon, n_models = 15, 15, 4

        # Create random positive definite covariance matrices
        Sigma = np.zeros((n_lat, n_lon, n_models, n_models))
        for i in range(n_lat):
            for j in range(n_lon):
                A = np.random.randn(n_models, n_models)
                Sigma[i, j] = A @ A.T + 0.1 * np.eye(n_models)

        # Import vectorized functions
        from collocation.fusion.fuse import (
            _compute_ivw_weights_vectorized,
            _compute_gls_weights_vectorized
        )

        # IVW weights should sum to 1
        weights_ivw = _compute_ivw_weights_vectorized(Sigma)
        assert np.allclose(weights_ivw.sum(axis=-1), 1.0, rtol=1e-10)

        # GLS weights should sum to 1
        weights_gls = _compute_gls_weights_vectorized(Sigma)
        assert np.allclose(weights_gls.sum(axis=-1), 1.0, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
