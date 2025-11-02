"""Integration oriented tests that exercise multiple collocation components."""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

from collocation.etcc import ETCC, TripleCollocation
from collocation.eivd import eivd
from collocation.simple_average import inverse_variance_weights, simple_average
from collocation.tc import tc_with_rescaling
from collocation.utils import kge_objfun, mse_judge


def save_workflow_test_figure(fig, test_name, script_name="test_method_workflows"):
    """保存工作流测试图片到figures文件夹"""
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
    print(f"Workflow test figure saved: {filepath}")


def plot_workflow_comparison(truth, products, results, method_name, test_name):
    """绘制工作流对比结果"""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{method_name} 工作流测试 - {test_name}', fontsize=14, fontweight='bold')
    
    # 1. 时间序列对比
    ax1 = axes[0, 0]
    n_show = min(200, len(truth))  # 显示前200个点
    ax1.plot(truth[:n_show], 'k-', label='真实值', linewidth=2)
    colors = ['b-', 'r-', 'g-', 'm-', 'c-']
    for i, product in enumerate(products[:5]):  # 最多显示5个产品
        ax1.plot(product[:n_show], colors[i], alpha=0.7, label=f'产品 {i+1}')
    
    if 'merged' in results:
        ax1.plot(results['merged'][:n_show], 'orange', linewidth=2, alpha=0.8, label='融合结果')
    
    ax1.set_title('时间序列对比')
    ax1.set_xlabel('时间')
    ax1.set_ylabel('值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 误差统计
    ax2 = axes[0, 1]
    if 'error_stats' in results:
        stats = results['error_stats']
        methods = list(stats.keys())
        values = list(stats.values())
        
        x = np.arange(len(methods))
        ax2.bar(x, values, alpha=0.7, color='steelblue')
        ax2.set_title('误差统计 (RMSE)')
        ax2.set_xlabel('方法')
        ax2.set_ylabel('RMSE')
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, v in enumerate(values):
            ax2.text(i, v + max(values)*0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 3. 性能指标
    ax3 = axes[1, 0]
    if 'metrics' in results:
        metrics = results['metrics']
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        y_pos = np.arange(len(metric_names))
        ax3.barh(y_pos, metric_values, alpha=0.7, color='lightcoral')
        ax3.set_title('性能指标')
        ax3.set_xlabel('值')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(metric_names)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, v in enumerate(metric_values):
            ax3.text(v + max(metric_values)*0.01, i, f'{v:.3f}', va='center')
    
    # 4. 方法信息
    ax4 = axes[1, 1]
    ax4.text(0.1, 0.9, f'方法: {method_name}', transform=ax4.transAxes, fontsize=12, fontweight='bold')
    ax4.text(0.1, 0.8, f'测试: {test_name}', transform=ax4.transAxes, fontsize=10)
    ax4.text(0.1, 0.7, f'产品数: {len(products)}', transform=ax4.transAxes, fontsize=10)
    ax4.text(0.1, 0.6, f'样本数: {len(truth)}', transform=ax4.transAxes, fontsize=10)
    
    if 'summary' in results:
        y_pos = 0.5
        for key, value in results['summary'].items():
            if isinstance(value, (int, float)):
                ax4.text(0.1, y_pos, f'{key}: {value:.4f}', transform=ax4.transAxes, fontsize=9)
            else:
                ax4.text(0.1, y_pos, f'{key}: {value}', transform=ax4.transAxes, fontsize=9)
            y_pos -= 0.08
            if y_pos < 0.1:
                break
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    save_workflow_test_figure(fig, test_name)


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
    
    # 创建可视化
    if HAS_MATPLOTLIB:
        # 计算误差统计
        rmse_products = [np.sqrt(np.mean((truth - product)**2)) for product in products]
        rmse_merged = np.sqrt(np.mean((truth - merged)**2))
        
        error_stats = {
            f'产品_{i+1}': rmse for i, rmse in enumerate(rmse_products)
        }
        error_stats['融合结果'] = rmse_merged
        
        metrics = {
            'RMSE_改善': (min(rmse_products) - rmse_merged) / min(rmse_products),
            '最佳权重': weights.max(),
            '权重和': weights.sum(),
            '相关性改善': merged_corr - max(correlations)
        }
        
        results = {
            'merged': merged,
            'error_stats': error_stats,
            'metrics': metrics,
            'summary': {
                '最干净传感器': cleanest_index + 1,
                '融合相关性': merged_corr,
                '权重方差': np.var(weights)
            }
        }
        
        plot_workflow_comparison(truth, products, results, 'TripleCollocation', 'cleanest_sensor_test')


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
