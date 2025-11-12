
# 附录：核心方法数学详解 (Appendix: Detailed Mathematical Methodology)

## 1. 记号 (Notation)
- `N`: 产品数 / Number of ET products  
- `M`: 时间样本数 / Number of time samples  
- `ET_i(t)`: 第 i 个产品在 t 时刻的 ET / i-th product ET at time t  
- `ε_i(t)`: 该产品误差 / error of product i  
- `R = cov(ε) ∈ ℝ^{N×N}`: 误差协方差矩阵 / error covariance
- `J ∈ ℝ^{(N-1)×N}`: 差分矩阵 / differencing matrix, s.t. `Y = J ET`

## 2. TCH 基本关系 (TCH Core Relation)
差分后 Y 的协方差为
\[
S = cov(Y) = J R J^{T}
\]
其中 R 未知，需要从 S 反推。

**无相关假设 (Uncorrelated TCH)**：若假定 `cov(ε_i, ε_j)=0 (i≠j)`，则
\[
Var(ET_i - ET_j) = Var(ε_i) + Var(ε_j)
\]
可基于所有成对差分方差的最小二乘估计 `σ_i^2 = Var(ε_i)`。

**相关情形 (Correlated TCH)**：R 带有非零对角外元素时，未知数多于方程。论文使用 Kuhn–Tucker 条件约束下的最优化来确定 R 的唯一解。为实用可运行，本仓库提供如下正则化近似：
\[
\min_{R\succeq 0} \ \|J R J^{T} - S\|_F^2 + \alpha \|\text{offdiag}(R)\|_F^2
\]
并采用特征值裁剪确保 `R` 的半正定。

## 3. BTCH 权重 (BTCH Weights)
在高斯似然假设下，最大似然的 `ET_t` 为加权和：
\[
ET_t = \sum_{i=1}^{N} w_i \, ET_i,\quad 
w_i = \frac{\prod_{k\neq i} \sigma_k^2}{\sum_{j=1}^{N} \prod_{k\neq j} \sigma_k^2}
\]
其中 `σ_i^2 = R_{ii}`。实现中用对数规避数值下溢。

## 4. 评价指标 (Metrics)
- Pearson 相关系数 / Correlation `R`
- 均方根误差 / RMSE
- 偏差 / Bias

