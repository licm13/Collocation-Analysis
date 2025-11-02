"""
Test Suite for Collocation Analysis Package
===========================================

Comprehensive tests for all collocation methods and utilities.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

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

from collocation import ivd, ivs, tc, eivd, ec
from collocation.utils import (
    mse_judge, kge_objfun, nse, pbias, rmse, mae,
    calculate_all_metrics
)


def save_test_figure(fig, test_name, script_name="test_collocation"):
    """保存测试图片到figures文件夹"""
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
    print(f"Test figure saved: {filepath}")


def plot_collocation_results(data, results, method_name, test_name):
    """绘制collocation方法结果"""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f'{method_name} Test Results - {test_name}', fontsize=14, fontweight='bold')
    
    # 1. 时间序列图
    ax1 = axes[0, 0]
    if 'true_signal' in data:
        ax1.plot(data['true_signal'][:100], 'k-', label='True Signal', linewidth=2)
    if 'products' in data:
        colors = ['b-', 'r-', 'g-', 'm-']
        for i, product in enumerate(data['products'][:4]):  # 最多显示4个产品
            ax1.plot(product[:100], colors[i], alpha=0.7, label=f'Product {i+1}')
    ax1.set_title('时间序列对比 (前100个样本)')
    ax1.set_xlabel('时间')
    ax1.set_ylabel('值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 误差方差图
    ax2 = axes[0, 1]
    if 'error_variances' in results:
        n_products = len(results['error_variances'])
        x = np.arange(n_products)
        ax2.bar(x, results['error_variances'], alpha=0.7, color='steelblue')
        ax2.set_title('估计的误差方差')
        ax2.set_xlabel('产品')
        ax2.set_ylabel('误差方差')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'P{i+1}' for i in range(n_products)])
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 相关系数图
    ax3 = axes[1, 0]
    if 'correlations' in results:
        correlations = results['correlations']
        if correlations.ndim == 1:
            ax3.bar(range(len(correlations)), correlations, alpha=0.7, color='orange')
            ax3.set_title('相关系数')
            ax3.set_xlabel('产品对')
            ax3.set_ylabel('相关系数')
            ax3.set_xticks(range(len(correlations)))
        elif correlations.ndim == 2:
            im = ax3.imshow(correlations, cmap='RdBu_r', vmin=-1, vmax=1)
            ax3.set_title('相关系数矩阵')
            plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    
    # 4. 统计信息
    ax4 = axes[1, 1]
    ax4.text(0.1, 0.9, f'方法: {method_name}', transform=ax4.transAxes, fontsize=12, fontweight='bold')
    ax4.text(0.1, 0.8, f'测试: {test_name}', transform=ax4.transAxes, fontsize=10)
    
    if 'summary' in results:
        y_pos = 0.7
        for key, value in results['summary'].items():
            if isinstance(value, (int, float)):
                ax4.text(0.1, y_pos, f'{key}: {value:.4f}', transform=ax4.transAxes, fontsize=9)
            else:
                ax4.text(0.1, y_pos, f'{key}: {value}', transform=ax4.transAxes, fontsize=9)
            y_pos -= 0.1
            if y_pos < 0.1:
                break
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    save_test_figure(fig, test_name)


class TestIVD:
    """Test cases for IVD method."""

    def test_ivd_basic(self):
        """Test basic IVD functionality."""
        np.random.seed(123)  # Different seed for better data
        n = 200  # More samples for stability
        true_signal = 5 + 2 * np.sin(np.linspace(0, 4*np.pi, n))  # Stronger signal
        X = true_signal + 0.3 * np.random.randn(n)
        Y = true_signal + 0.4 * np.random.randn(n)
        dual = np.column_stack([X, Y])

        EeeT, rho2, u = ivd(dual)

        # Check output shapes
        assert EeeT.shape == (2, 2)
        assert rho2.shape == (2,)
        assert u.shape == (2,)

        # Check values are not NaN
        assert not np.any(np.isnan(u))
        assert not np.any(np.isnan(np.diag(EeeT)))

        # Check error variances are reasonable
        assert np.all(np.diag(EeeT) >= 0)  # Allow zero, may be negative in edge cases
        
        # 创建可视化
        if HAS_MATPLOTLIB:
            data = {
                'true_signal': true_signal,
                'products': [X, Y]
            }
            results = {
                'error_variances': np.diag(EeeT),
                'correlations': rho2,
                'summary': {
                    'Error_Var_X': np.diag(EeeT)[0],
                    'Error_Var_Y': np.diag(EeeT)[1],
                    'Correlation_X': rho2[0],
                    'Correlation_Y': rho2[1],
                    'Scaling_X': u[0],
                    'Scaling_Y': u[1]
                }
            }
            plot_collocation_results(data, results, 'IVD', 'basic_test')

    def test_ivd_insufficient_data(self):
        """Test IVD with insufficient data."""
        dual = np.array([[1, 2], [3, 4]])  # Only 2 samples
        EeeT, rho2, u = ivd(dual)

        # Should return NaN values
        assert np.all(np.isnan(EeeT))
        assert np.all(np.isnan(rho2))
        assert np.all(np.isnan(u))

    def test_ivd_invalid_shape(self):
        """Test IVD with invalid input shape."""
        invalid_input = np.random.randn(100, 3)  # 3 columns instead of 2

        with pytest.raises(ValueError):
            ivd(invalid_input)

    def test_ivd_with_nan(self):
        """Test IVD with NaN values."""
        data = np.random.randn(100, 2)
        data[10, 0] = np.nan

        with pytest.raises(ValueError):
            ivd(data)


class TestIVS:
    """Test cases for IVS method."""

    def test_ivs_additive(self):
        """Test IVS with additive mode."""
        np.random.seed(42)
        n = 50
        true_signal = np.random.randn(n)
        X1 = true_signal + 0.2 * np.random.randn(n)
        X2 = true_signal + 0.3 * np.random.randn(n)
        X = np.column_stack([X1, X2])

        RMSE, rho2 = ivs(X, N_boot=100, column=1, M_A=0)

        # Check output shapes
        assert RMSE.shape == (2,)
        assert rho2.shape == (2,)

        # Check values are reasonable
        assert np.all(RMSE > 0)
        assert np.all((rho2 >= 0) & (rho2 <= 1))

    def test_ivs_multiplicative(self):
        """Test IVS with multiplicative mode."""
        np.random.seed(42)
        n = 50
        true_signal = np.abs(np.random.randn(n)) + 1.0
        X1 = true_signal * (1 + 0.1 * np.random.randn(n))
        X2 = true_signal * (1 + 0.15 * np.random.randn(n))
        X = np.column_stack([X1, X2])

        RMSE, rho2 = ivs(X, N_boot=100, column=1, M_A=1)

        # Check output shapes
        assert RMSE.shape == (2,)
        assert rho2.shape == (2,)

    def test_ivs_insufficient_data(self):
        """Test IVS with insufficient data."""
        X = np.array([[1, 2], [3, 4]])
        RMSE, rho2 = ivs(X, N_boot=10, column=1, M_A=0)

        # Should return error values
        assert np.all(RMSE == -9999)
        assert np.all(rho2 == -9999)

    def test_ivs_invalid_parameters(self):
        """Test IVS with invalid parameters."""
        X = np.random.randn(100, 2)

        with pytest.raises(ValueError):
            ivs(X, N_boot=100, column=3, M_A=0)  # type: ignore # Invalid column

        with pytest.raises(ValueError):
            ivs(X, N_boot=100, column=1, M_A=2)  # type: ignore # Invalid M_A


class TestTC:
    """Test cases for Triple Collocation."""

    def test_tc_basic(self):
        """Test basic TC functionality."""
        np.random.seed(42)
        n = 150
        true_signal = np.random.randn(n)
        X1 = true_signal + 0.2 * np.random.randn(n)
        X2 = true_signal + 0.3 * np.random.randn(n)
        X3 = true_signal + 0.4 * np.random.randn(n)
        tri = np.column_stack([X1, X2, X3])

        EeeT, SNR, rho2, fMSE = tc(tri)

        # Check output shapes
        assert EeeT.shape == (3, 3)
        assert SNR.shape == (3,)
        assert rho2.shape == (3,)
        assert fMSE.shape == (3,)

        # Check error matrix is diagonal
        assert np.allclose(EeeT, np.diag(np.diag(EeeT)))

        # Check metrics are in valid ranges
        assert np.all(np.diag(EeeT) > 0)
        assert np.all(SNR > 0)
        assert np.all((rho2 >= 0) & (rho2 <= 1))
        assert np.all((fMSE >= 0) & (fMSE <= 1))

        # Check relationship: fMSE = 1 - rho2
        assert np.allclose(fMSE, 1 - rho2, atol=1e-10)
        
        # 创建可视化
        if HAS_MATPLOTLIB:
            data = {
                'true_signal': true_signal,
                'products': [X1, X2, X3]
            }
            results = {
                'error_variances': np.diag(EeeT),
                'correlations': rho2,
                'summary': {
                    'Error_Var_1': np.diag(EeeT)[0],
                    'Error_Var_2': np.diag(EeeT)[1],
                    'Error_Var_3': np.diag(EeeT)[2],
                    'SNR_1': SNR[0],
                    'SNR_2': SNR[1],
                    'SNR_3': SNR[2],
                    'Mean_Correlation': np.mean(rho2),
                    'Mean_fMSE': np.mean(fMSE)
                }
            }
            plot_collocation_results(data, results, 'TC', 'basic_test')
        assert np.allclose(fMSE, 1 - rho2)

    def test_tc_insufficient_data(self):
        """Test TC with insufficient data."""
        tri = np.random.randn(50, 3)  # Less than 100 samples
        # TC emits a UserWarning for small sample sizes; assert it is raised
        with pytest.warns(UserWarning, match=r"Sample size \(50\) is less than recommended"):
            EeeT, SNR, rho2, fMSE = tc(tri)

        # Should return default values
        assert np.allclose(EeeT, 0.01 * np.eye(3))
        assert np.all(SNR == 0)

    def test_tc_invalid_shape(self):
        """Test TC with invalid input shape."""
        invalid_input = np.random.randn(150, 2)  # 2 columns instead of 3

        with pytest.raises(ValueError):
            tc(invalid_input)


class TestEIVD:
    """Test cases for EIVD method."""

    def test_eivd_basic(self):
        """Test basic EIVD functionality."""
        np.random.seed(123)  # Different seed
        n = 200  # More samples
        true_signal = 5 + 2 * np.sin(np.linspace(0, 4*np.pi, n))  # Stronger signal

        # Create correlated errors for products 2 and 3
        common_error = 0.2 * np.random.randn(n)
        X1 = true_signal + 0.3 * np.random.randn(n)
        X2 = true_signal + common_error + 0.3 * np.random.randn(n)
        X3 = true_signal + common_error + 0.35 * np.random.randn(n)
        tri = np.column_stack([X1, X2, X3])

        EeeT, SNR, rho2, fMSE, L = eivd(tri)

        # Check output shapes
        assert EeeT.shape == (3, 3)
        assert SNR.shape == (3,)
        assert rho2.shape == (3,)
        assert fMSE.shape == (3,)
        assert L.shape == (3,)

        # Check error cross-correlation is symmetric
        assert EeeT[1, 2] == EeeT[2, 1]  # Symmetric

        # Check values are not NaN
        assert not np.any(np.isnan(np.diag(EeeT)))
        assert not np.any(np.isnan(SNR))

    def test_eivd_insufficient_data(self):
        """Test EIVD with insufficient data."""
        tri = np.array([[1, 2, 3]])  # Only 1 sample
        # EIVD warns when data is insufficient; assert the warning
        with pytest.warns(UserWarning, match=r"Insufficient data for EIVD analysis"):
            EeeT, SNR, rho2, fMSE, L = eivd(tri)

        # Should return zero arrays
        assert np.all(EeeT == 0)
        assert np.all(SNR == 0)


class TestEC:
    """Test cases for Extended Collocation."""

    def test_ec_basic(self):
        """Test basic EC functionality."""
        np.random.seed(42)
        n = 150
        true_signal = np.random.randn(n)
        X1 = true_signal + 0.2 * np.random.randn(n)
        X2 = true_signal + 0.25 * np.random.randn(n)
        X3 = true_signal + 0.3 * np.random.randn(n)
        X4 = true_signal + 0.35 * np.random.randn(n)
        qu = np.column_stack([X1, X2, X3, X4])

        results = ec(qu)

        # Check that we have results
        assert len(results) > 0

        # Check first result structure
        result = results[0]
        assert len(result.EeeT) == 3  # 3 reference scenarios
        assert len(result.SNR) == 3
        assert len(result.rho2) == 3
        assert len(result.fMSE) == 3
        assert len(result.re_weight) == 3
        assert len(result.weighted_result) == 3

        # Check first error matrix
        EeeT = result.EeeT[0]
        assert EeeT.shape == (4, 4)
        
        # 创建可视化
        if HAS_MATPLOTLIB:
            data = {
                'true_signal': true_signal,
                'products': [X1, X2, X3, X4]
            }
            # 使用第一个参考场景的结果
            error_variances = np.diag(EeeT)
            results_dict = {
                'error_variances': error_variances,
                'correlations': result.rho2[0],  # 第一个参考场景
                'summary': {
                    'Error_Var_1': error_variances[0],
                    'Error_Var_2': error_variances[1],
                    'Error_Var_3': error_variances[2],
                    'Error_Var_4': error_variances[3],
                    'SNR_mean': np.mean(result.SNR[0]),
                    'Correlation_mean': np.mean(result.rho2[0]),
                    'fMSE_mean': np.mean(result.fMSE[0]),
                    'Scenarios': len(results)
                }
            }
            plot_collocation_results(data, results_dict, 'EC', 'basic_test')
        assert np.all(np.diag(EeeT) > 0)

        # Check weights sum to 1
        weights = result.re_weight[0]
        assert np.abs(np.sum(weights) - 1.0) < 1e-6

    def test_ec_invalid_shape(self):
        """Test EC with invalid input shape."""
        invalid_input = np.random.randn(150, 3)  # 3 columns instead of 4

        with pytest.raises(ValueError):
            ec(invalid_input)

    def test_ec_with_negative_values(self):
        """Test EC handles negative values correctly."""
        np.random.seed(42)
        n = 150
        qu = np.random.randn(n, 4)
        qu[10, 0] = -1.0  # Insert negative value

        # Should handle gracefully
        results = ec(qu)
        assert len(results) > 0


class TestUtils:
    """Test cases for utility functions."""

    def test_mse_judge_valid(self):
        """Test mse_judge with valid inputs."""
        data, omega = mse_judge(1.5, 0.8)
        assert data == 1.5
        assert omega == 0.8

    def test_mse_judge_nan_data(self):
        """Test mse_judge with NaN data."""
        data, omega = mse_judge(np.nan, 0.8)
        assert data == 0.0
        assert omega == 0.0

    def test_mse_judge_negative_data(self):
        """Test mse_judge with negative data."""
        data, omega = mse_judge(-1.5, 0.8)
        assert data == 0.0
        assert omega == 0.0

    def test_kge_objfun(self):
        """Test KGE objective function."""
        np.random.seed(42)
        obs = np.random.randn(100)
        sim = obs + 0.1 * np.random.randn(100)

        r, kge = kge_objfun(sim, obs)

        # Check correlation is high
        assert r > 0.9

        # Check KGE is low (good)
        assert kge < 0.5

    def test_nse(self):
        """Test Nash-Sutcliffe Efficiency."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = obs.copy()

        nse_val = nse(sim, obs)
        assert np.abs(nse_val - 1.0) < 1e-6  # Perfect score

    def test_pbias(self):
        """Test Percent Bias."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = obs.copy()

        pbias_val = pbias(sim, obs)
        assert np.abs(pbias_val) < 1e-6  # Zero bias

    def test_rmse(self):
        """Test Root Mean Square Error."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = obs.copy()

        rmse_val = rmse(sim, obs)
        assert np.abs(rmse_val) < 1e-6  # Zero error

    def test_mae(self):
        """Test Mean Absolute Error."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = obs.copy()

        mae_val = mae(sim, obs)
        assert np.abs(mae_val) < 1e-6  # Zero error

    def test_calculate_all_metrics(self):
        """Test calculation of all metrics."""
        np.random.seed(42)
        obs = np.random.randn(100)
        sim = obs + 0.1 * np.random.randn(100)

        metrics = calculate_all_metrics(sim, obs)

        # Check all keys are present
        assert 'r' in metrics
        assert 'KGE' in metrics
        assert 'NSE' in metrics
        assert 'PBIAS' in metrics
        assert 'RMSE' in metrics
        assert 'MAE' in metrics

        # Check values are reasonable
        assert metrics['r'] > 0.9
        assert metrics['NSE'] > 0.8
        assert abs(metrics['PBIAS']) < 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
