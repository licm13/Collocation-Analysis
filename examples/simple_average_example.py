#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单平均方法示例
=================

演示如何使用简单平均方法融合多个数据产品。
这是最基础的数据融合方法，适用于快速初步分析。

作者: Collocation Analysis Team
日期: 2024-10-31
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_synthetic_data(n_samples=200, n_products=3, noise_levels=None, seed=42):
    """
    生成合成数据用于演示

    参数:
        n_samples: 样本数量
        n_products: 产品数量
        noise_levels: 每个产品的噪声水平 (标准差)
        seed: 随机种子

    返回:
        truth: 真实信号
        products: 各产品观测值列表
        noise_levels: 使用的噪声水平
    """
    np.random.seed(seed)

    # 生成真实信号 (正弦波 + 趋势)
    t = np.linspace(0, 4*np.pi, n_samples)
    truth = 10 + 5 * np.sin(t) + 0.5 * t

    # 默认噪声水平
    if noise_levels is None:
        noise_levels = [1.0, 1.5, 2.0][:n_products]

    # 生成观测产品
    products = []
    for noise_std in noise_levels:
        noise = np.random.normal(0, noise_std, n_samples)
        product = truth + noise
        products.append(product)

    return truth, products, noise_levels


def simple_average(products, weights=None):
    """
    计算多个产品的简单平均或加权平均

    参数:
        products: 产品列表，每个产品是一维数组
        weights: 权重列表，如果为None则使用等权重

    返回:
        averaged: 平均结果
    """
    products_array = np.array(products)

    if weights is None:
        # 简单算术平均
        averaged = np.mean(products_array, axis=0)
    else:
        # 加权平均
        weights = np.array(weights)
        weights = weights / weights.sum()  # 归一化权重
        averaged = np.sum(products_array * weights[:, np.newaxis], axis=0)

    return averaged


def calculate_metrics(estimated, truth):
    """
    计算评估指标

    参数:
        estimated: 估计值
        truth: 真实值

    返回:
        metrics: 指标字典
    """
    # RMSE
    rmse = np.sqrt(np.mean((estimated - truth)**2))

    # 相关系数
    correlation = np.corrcoef(estimated, truth)[0, 1]

    # R²
    ss_res = np.sum((truth - estimated)**2)
    ss_tot = np.sum((truth - np.mean(truth))**2)
    r2 = 1 - (ss_res / ss_tot)

    # MAE
    mae = np.mean(np.abs(estimated - truth))

    # 偏差
    bias = np.mean(estimated - truth)

    metrics = {
        'RMSE': rmse,
        'Correlation': correlation,
        'R2': r2,
        'MAE': mae,
        'Bias': bias
    }

    return metrics


def plot_results(truth, products, averaged, noise_levels, output_path=None):
    """
    绘制结果对比图

    参数:
        truth: 真实信号
        products: 产品列表
        averaged: 平均结果
        noise_levels: 噪声水平
        output_path: 输出路径
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 上图: 时间序列比较
    ax1 = axes[0]
    t = np.arange(len(truth))

    # 绘制真实信号
    ax1.plot(t, truth, 'k-', linewidth=2, label='真实信号', zorder=10)

    # 绘制各产品
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (product, noise) in enumerate(zip(products, noise_levels)):
        ax1.plot(t, product, color=colors[i], alpha=0.6,
                linewidth=1, label=f'产品{i+1} (σ={noise:.1f})')

    # 绘制平均结果
    ax1.plot(t, averaged, 'r-', linewidth=2, label='简单平均', zorder=9)

    ax1.set_xlabel('时间', fontsize=12)
    ax1.set_ylabel('值', fontsize=12)
    ax1.set_title('多产品简单平均融合', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 下图: 误差分布
    ax2 = axes[1]

    errors = []
    labels = []

    # 各产品误差
    for i, product in enumerate(products):
        error = product - truth
        errors.append(error)
        labels.append(f'产品{i+1}')

    # 平均结果误差
    avg_error = averaged - truth
    errors.append(avg_error)
    labels.append('简单平均')

    # 绘制箱线图
    bp = ax2.boxplot(errors, labels=labels, patch_artist=True)

    # 设置颜色
    for i, patch in enumerate(bp['boxes']):
        if i < len(products):
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.6)
        else:
            patch.set_facecolor('red')
            patch.set_alpha(0.8)

    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_ylabel('误差', fontsize=12)
    ax2.set_title('误差分布对比', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {output_path}")

    plt.show()


def main():
    """主函数"""

    print("=" * 80)
    print("简单平均方法示例")
    print("=" * 80)
    print()

    # 1. 生成合成数据
    print("1. 生成合成数据...")
    truth, products, noise_levels = generate_synthetic_data(
        n_samples=200,
        n_products=3,
        noise_levels=[1.0, 1.5, 2.0]
    )
    print(f"   - 样本数: {len(truth)}")
    print(f"   - 产品数: {len(products)}")
    print(f"   - 噪声水平: {noise_levels}")
    print()

    # 2. 简单平均
    print("2. 计算简单平均...")
    averaged = simple_average(products)
    print("   完成!")
    print()

    # 3. 计算各产品指标
    print("3. 评估各产品性能...")
    print("-" * 80)
    print(f"{'方法':<15} {'RMSE':<10} {'相关系数':<10} {'R²':<10} {'MAE':<10} {'偏差':<10}")
    print("-" * 80)

    all_metrics = []

    # 各产品
    for i, product in enumerate(products):
        metrics = calculate_metrics(product, truth)
        all_metrics.append(metrics)
        print(f"产品{i+1:<10} {metrics['RMSE']:<10.4f} {metrics['Correlation']:<10.4f} "
              f"{metrics['R2']:<10.4f} {metrics['MAE']:<10.4f} {metrics['Bias']:<10.4f}")

    # 平均结果
    avg_metrics = calculate_metrics(averaged, truth)
    all_metrics.append(avg_metrics)
    print(f"{'简单平均':<15} {avg_metrics['RMSE']:<10.4f} {avg_metrics['Correlation']:<10.4f} "
          f"{avg_metrics['R2']:<10.4f} {avg_metrics['MAE']:<10.4f} {avg_metrics['Bias']:<10.4f}")

    print("-" * 80)
    print()

    # 4. 改进分析
    print("4. 性能改进分析...")

    # 相对于最好的单个产品
    best_product_rmse = min(m['RMSE'] for m in all_metrics[:-1])
    improvement = (best_product_rmse - avg_metrics['RMSE']) / best_product_rmse * 100

    print(f"   - 最优单产品RMSE: {best_product_rmse:.4f}")
    print(f"   - 简单平均RMSE: {avg_metrics['RMSE']:.4f}")
    print(f"   - 相对改进: {improvement:.2f}%")
    print()

    # 5. 绘图
    print("5. 生成可视化图表...")
    # 保存到脚本当前路径下的 figures 文件夹
    fig_dir = Path(__file__).resolve().parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    output_path = fig_dir / "simple_average_example.png"
    plot_results(truth, products, averaged, noise_levels, output_path)
    print()

    # 6. 加权平均演示
    print("6. 加权平均演示（基于噪声水平的反比）...")

    # 使用噪声的倒数作为权重
    weights = [1.0 / n**2 for n in noise_levels]
    weighted_avg = simple_average(products, weights=weights)
    weighted_metrics = calculate_metrics(weighted_avg, truth)

    print(f"   - 权重: {[f'{w:.3f}' for w in np.array(weights)/sum(weights)]}")
    print(f"   - 加权平均RMSE: {weighted_metrics['RMSE']:.4f}")
    print(f"   - 简单平均RMSE: {avg_metrics['RMSE']:.4f}")
    improvement_weighted = (avg_metrics['RMSE'] - weighted_metrics['RMSE']) / avg_metrics['RMSE'] * 100
    print(f"   - 加权相对简单平均改进: {improvement_weighted:.2f}%")
    print()

    print("=" * 80)
    print("示例完成!")
    print("=" * 80)
    print()

    # 总结
    print("总结:")
    print("------")
    print("1. 简单平均是最基础的数据融合方法")
    print("2. 可以有效降低随机误差")
    print("3. 适用于快速初步分析")
    print("4. 不需要复杂的统计假设")
    print("5. 加权平均可以利用先验知识进一步改进")
    print()
    print("注意:")
    print("------")
    print("- 简单平均假设所有产品同等重要")
    print("- 无法估计误差结构")
    print("- 无法处理系统性偏差")
    print("- 对于更复杂的场景，建议使用IVD、TC、EIVD等方法")


if __name__ == "__main__":
    main()
