#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Average Method for Data Fusion
简单平均数据融合方法

This module implements the simplest data fusion approach - simple averaging.
While not as sophisticated as collocation methods, it provides a quick and
straightforward way to combine multiple data products.

本模块实现了最简单的数据融合方法 - 简单平均。
虽然不如交叉定标方法复杂，但它提供了一种快速直接的方式来组合多个数据产品。

Author: Collocation Analysis Team
Date: 2024-10-31
"""

import numpy as np
from typing import Optional, Tuple, List, Union


def simple_average(
    data: np.ndarray,
    weights: Optional[np.ndarray] = None,
    axis: int = 0
) -> np.ndarray:
    """
    Compute simple or weighted average of multiple data products.
    计算多个数据产品的简单平均或加权平均。

    This is the most basic data fusion method. It assumes all products are
    equally reliable (for unweighted) or have known relative reliabilities
    (for weighted). Unlike collocation methods, it does not estimate error
    characteristics.

    这是最基础的数据融合方法。它假设所有产品同等可靠（无权重）或具有已知的
    相对可靠性（加权）。与交叉定标方法不同，它不估计误差特性。

    Parameters / 参数
    ----------
    data : np.ndarray
        Input data array. Shape can be:
        - (n_samples, n_products): each column is a product
        - (n_products, n_samples): each row is a product (use axis=0)
        输入数据数组。形状可以是：
        - (n_samples, n_products): 每列是一个产品
        - (n_products, n_samples): 每行是一个产品 (使用 axis=0)

    weights : np.ndarray, optional
        Weights for each product. If None, equal weights are used.
        Will be normalized to sum to 1.
        每个产品的权重。如果为None，使用等权重。
        将被归一化为总和为1。

    axis : int, default=0
        Axis along which to average.
        - axis=0: average across first dimension (rows)
        - axis=1: average across second dimension (columns)
        沿哪个轴平均。
        - axis=0: 沿第一维（行）平均
        - axis=1: 沿第二维（列）平均

    Returns / 返回
    -------
    averaged : np.ndarray
        Averaged result.
        平均结果。

    Examples / 示例
    --------
    >>> # Simple average of 3 products
    >>> # 3个产品的简单平均
    >>> data = np.random.randn(100, 3)
    >>> result = simple_average(data, axis=1)

    >>> # Weighted average based on known quality
    >>> # 基于已知质量的加权平均
    >>> weights = np.array([0.5, 0.3, 0.2])
    >>> result = simple_average(data, weights=weights, axis=1)

    Notes / 注意
    -----
    Advantages:
    - Very simple and fast
    - No statistical assumptions needed
    - Reduces random errors
    - Easy to interpret

    优点：
    - 非常简单快速
    - 不需要统计假设
    - 降低随机误差
    - 易于解释

    Limitations:
    - Cannot estimate error characteristics
    - Cannot handle systematic biases
    - Assumes all products measure the same quantity
    - No uncertainty quantification

    局限性：
    - 无法估计误差特性
    - 无法处理系统性偏差
    - 假设所有产品测量相同的量
    - 没有不确定性量化
    """
    data = np.asarray(data)

    if weights is None:
        # Simple arithmetic average
        # 简单算术平均
        averaged = np.mean(data, axis=axis)
    else:
        # Weighted average
        # 加权平均
        weights = np.asarray(weights)

        # Normalize weights
        # 归一化权重
        weights = weights / np.sum(weights)

        # Reshape weights for broadcasting
        # 重塑权重以便广播
        if axis == 0:
            weights = weights[:, np.newaxis]
        elif axis == 1:
            weights = weights[np.newaxis, :]
        else:
            raise ValueError(f"axis must be 0 or 1, got {axis}")

        averaged = np.sum(data * weights, axis=axis)

    return averaged


def inverse_variance_weights(
    variances: np.ndarray
) -> np.ndarray:
    """
    Calculate optimal weights based on inverse variance.
    基于逆方差计算最优权重。

    For products with known error variances, the optimal weights
    that minimize the variance of the weighted average are
    inversely proportional to the variances.

    对于已知误差方差的产品，使加权平均方差最小化的最优权重
    与方差成反比。

    Parameters / 参数
    ----------
    variances : np.ndarray
        Error variances for each product.
        每个产品的误差方差。

    Returns / 返回
    -------
    weights : np.ndarray
        Normalized weights (sum to 1).
        归一化的权重（总和为1）。

    Examples / 示例
    --------
    >>> variances = np.array([1.0, 2.0, 4.0])
    >>> weights = inverse_variance_weights(variances)
    >>> print(weights)
    [0.571  0.286  0.143]

    Notes / 注意
    -----
    This assumes:
    - Independent errors across products
    - Known error variances
    - No systematic biases

    这假设：
    - 产品间误差独立
    - 已知误差方差
    - 无系统性偏差

    References / 参考文献
    ----------
    Optimal weighted averaging is a standard statistical technique.
    See any statistics textbook on combining independent estimates.

    最优加权平均是标准统计技术。
    参见任何关于组合独立估计的统计学教材。
    """
    variances = np.asarray(variances)

    if np.any(variances <= 0):
        raise ValueError("All variances must be positive")

    # Inverse variance weights
    # 逆方差权重
    weights = 1.0 / variances

    # Normalize
    # 归一化
    weights = weights / np.sum(weights)

    return weights


def calculate_averaging_uncertainty(
    variances: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Calculate the uncertainty (standard deviation) of the weighted average.
    计算加权平均的不确定性（标准差）。

    For independent errors with known variances, the variance of the
    weighted average can be calculated analytically.

    对于已知方差的独立误差，加权平均的方差可以解析计算。

    Parameters / 参数
    ----------
    variances : np.ndarray
        Error variances for each product.
        每个产品的误差方差。

    weights : np.ndarray, optional
        Weights for each product. If None, equal weights are assumed.
        每个产品的权重。如果为None，假设等权重。

    Returns / 返回
    -------
    std : float
        Standard deviation of the weighted average.
        加权平均的标准差。

    Examples / 示例
    --------
    >>> # Equal weight averaging
    >>> # 等权平均
    >>> variances = np.array([1.0, 1.0, 1.0])
    >>> std = calculate_averaging_uncertainty(variances)
    >>> print(f"Std of average: {std:.3f}")
    Std of average: 0.577

    >>> # Optimal weighted averaging
    >>> # 最优加权平均
    >>> weights = inverse_variance_weights(variances)
    >>> std = calculate_averaging_uncertainty(variances, weights)
    >>> print(f"Std of weighted average: {std:.3f}")

    Notes / 注意
    -----
    For equal weight averaging of N products with variance σ²:
        σ_avg = σ / √N

    对于N个方差为σ²的产品的等权平均：
        σ_avg = σ / √N

    For optimal weighted averaging:
        σ_avg = 1 / √(∑(1/σᵢ²))

    对于最优加权平均：
        σ_avg = 1 / √(∑(1/σᵢ²))
    """
    variances = np.asarray(variances)

    if weights is None:
        # Equal weights
        # 等权重
        n = len(variances)
        weights = np.ones(n) / n

    weights = np.asarray(weights)

    # Variance of weighted sum: Var(∑w_i X_i) = ∑w_i² Var(X_i)
    # (for independent X_i)
    # 加权和的方差: Var(∑w_i X_i) = ∑w_i² Var(X_i)
    # (对于独立的X_i)
    variance_avg = np.sum(weights**2 * variances)

    # Standard deviation
    # 标准差
    std = np.sqrt(variance_avg)

    return std


def ensemble_statistics(
    data: np.ndarray,
    axis: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate ensemble statistics for multiple products.
    计算多个产品的集合统计量。

    Computes mean, median, standard deviation, and range across products.
    计算产品间的均值、中位数、标准差和范围。

    Parameters / 参数
    ----------
    data : np.ndarray
        Input data array of shape (n_products, n_samples) or (n_samples, n_products).
        输入数据数组，形状为 (n_products, n_samples) 或 (n_samples, n_products)。

    axis : int, default=0
        Axis along which to compute statistics.
        沿哪个轴计算统计量。

    Returns / 返回
    -------
    mean : np.ndarray
        Mean across products.
        产品间均值。

    median : np.ndarray
        Median across products.
        产品间中位数。

    std : np.ndarray
        Standard deviation across products.
        产品间标准差。

    spread : np.ndarray
        Range (max - min) across products.
        产品间范围（最大值 - 最小值）。

    Examples / 示例
    --------
    >>> data = np.random.randn(3, 100)  # 3 products, 100 samples
    >>> mean, median, std, spread = ensemble_statistics(data, axis=0)
    >>> print(f"Mean std: {std.mean():.3f}")

    Notes / 注意
    -----
    These statistics provide a simple way to assess:
    - Central tendency (mean, median)
    - Disagreement between products (std, spread)
    - Potential outliers (large spread)

    这些统计量提供了一种简单的方式来评估：
    - 集中趋势（均值、中位数）
    - 产品间差异（标准差、范围）
    - 潜在异常值（大范围）
    """
    data = np.asarray(data)

    mean = np.mean(data, axis=axis)
    median = np.median(data, axis=axis)
    std = np.std(data, axis=axis)
    spread = np.ptp(data, axis=axis)  # peak-to-peak (max - min)

    return mean, median, std, spread


# Export public API
# 导出公共API
__all__ = [
    'simple_average',
    'inverse_variance_weights',
    'calculate_averaging_uncertainty',
    'ensemble_statistics'
]
