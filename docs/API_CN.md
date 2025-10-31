# Collocation Analysis API 中文文档

本文档提供Collocation Analysis工具包的详细中文API参考。

## 目录

1. [经典方法](#经典方法)
   - [IVD (信息向量对偶)](#ivd-信息向量对偶)
   - [IVS (带尺度的信息向量)](#ivs-带尺度的信息向量)
   - [TC (三路交叉定标)](#tc-三路交叉定标)
   - [EIVD (扩展IVD)](#eivd-扩展ivd)
   - [EC (扩展交叉定标)](#ec-扩展交叉定标)
2. [简单方法](#简单方法)
   - [SimpleAverage (简单平均)](#simpleaverage-简单平均)
3. [贝叶斯方法](#贝叶斯方法)
   - [BTC (贝叶斯三路交叉定标)](#btc-贝叶斯三路交叉定标)
4. [工具函数](#工具函数)
5. [性能指标](#性能指标)

---

## 经典方法

### IVD (信息向量对偶)

**函数签名:**
```python
collocation.ivd(dual)
```

**描述:**

IVD方法用于两个数据产品的交叉定标分析。该方法通过优化时间偏移，估计误差协方差矩阵和最优融合权重。

**参数:**

- **dual** : `np.ndarray`, shape (n, 2)
  - 输入数据矩阵，包含两个产品
  - 每列是一个产品，每行是一个时间点

**返回:**

- **EeeT** : `np.ndarray`, shape (2, 2)
  - 误差协方差矩阵
  - 对角线元素是各产品的误差方差

- **rho2** : `np.ndarray`, shape (2,)
  - 数据-真值相关性的平方（R²）
  - 值域: [0, 1]

- **u** : `np.ndarray`, shape (2,)
  - 最优融合权重
  - 总和为1

**示例:**

```python
import numpy as np
from collocation import ivd

# 两个产品数据
product1 = np.random.randn(200) + true_signal
product2 = np.random.randn(200) + true_signal
dual = np.column_stack([product1, product2])

# 应用IVD
EeeT, rho2, u = ivd(dual)

print("误差方差:", np.diag(EeeT))
print("数据-真值相关性:", rho2)
print("融合权重:", u)

# 使用最优权重融合
merged = u[0] * product1 + u[1] * product2
```

**参考文献:**

> Dong, J., Crow, W. T., Reichle, R., & Liu, Q. (2014). Fusing active and passive remotely sensed soil moisture products using information vectors.

**适用场景:**
- 只有两个数据产品
- 需要最优权重融合
- 产品间可能存在时间偏移

---

### IVS (带尺度的信息向量)

**函数签名:**
```python
collocation.ivs(X, N_boot=1000, column=1, M_A=1)
```

**描述:**

IVS方法用于两个数据产品的交叉定标，通过Bootstrap方法提供不确定性估计。

**参数:**

- **X** : `np.ndarray`, shape (n, 2)
  - 输入数据矩阵

- **N_boot** : `int`, 默认=1000
  - Bootstrap重采样次数

- **column** : `int`, 1或2
  - 用于计算滞后1自相关的列

- **M_A** : `int`, 0或1
  - 误差模型类型
  - 0: 加性误差模型
  - 1: 乘性误差模型

**返回:**

- **RMSE** : `np.ndarray`, shape (2,)
  - 各产品的均方根误差估计

- **rho2** : `np.ndarray`, shape (2,)
  - 数据-真值相关性的平方

**示例:**

```python
from collocation import ivs

# 应用IVS with Bootstrap
RMSE, rho2 = ivs(dual, N_boot=1000)

print("RMSE估计:", RMSE)
print("相关性:", rho2)
```

**参考文献:**

> Scipal, K., Holmes, T., De Jeu, R., Naeimi, V., & Wagner, W. (2008). A possible solution for the problem of estimating the error structure of global soil moisture data sets. Geophysical Research Letters, 35(24).

**适用场景:**
- 需要不确定性量化
- Bootstrap置信区间
- 加性或乘性误差模型

---

### TC (三路交叉定标)

**函数签名:**
```python
collocation.tc(tri)
```

**描述:**

经典的三路交叉定标方法，假设三个产品的误差相互独立。

**参数:**

- **tri** : `np.ndarray`, shape (n, 3)
  - 输入数据矩阵，包含三个产品

**返回:**

- **EeeT** : `np.ndarray`, shape (3, 3)
  - 误差协方差矩阵

- **SNR** : `np.ndarray`, shape (3,)
  - 信噪比

- **rho2** : `np.ndarray`, shape (3,)
  - 数据-真值相关性的平方

- **fMSE** : `np.ndarray`, shape (3,)
  - 分数均方误差 (1 - R²)

**示例:**

```python
from collocation import tc

# 三个产品
tri = np.column_stack([product1, product2, product3])

# 应用TC
EeeT, SNR, rho2, fMSE = tc(tri)

print("误差方差:", np.diag(EeeT))
print("信噪比:", SNR)
print("R²:", rho2)
```

**数学背景:**

对于三个产品测量同一真值:
```
X_i = α_i + β_i θ + ε_i
```

假设:
1. E[ε_i] = 0 (零均值误差)
2. E[ε_i θ] = 0 (误差与真值不相关)
3. E[ε_i ε_j] = 0, i≠j (误差独立)

**参考文献:**

> Stoffelen, A. (1998). Toward the true near-surface wind speed: Error modeling and calibration using triple collocation. Journal of Geophysical Research, 103(C4), 7755-7766.

**适用场景:**
- 有三个独立产品
- 假设误差独立
- 标准验证场景

---

### EIVD (扩展IVD)

**函数签名:**
```python
collocation.eivd(tri)
```

**描述:**

扩展IVD方法，允许产品2和产品3之间存在误差相关。

**参数:**

- **tri** : `np.ndarray`, shape (n, 3)
  - 输入数据矩阵

**返回:**

- **EeeT** : `np.ndarray`, shape (3, 3)
  - 误差协方差矩阵（非对角元素可能非零）

- **SNR** : `np.ndarray`, shape (3,)
  - 信噪比

- **rho2** : `np.ndarray`, shape (3,)
  - 数据-真值相关性的平方

- **fMSE** : `np.ndarray`, shape (3,)
  - 分数均方误差

- **L** : `np.ndarray`, shape (3,)
  - 滞后1自相关系数

**示例:**

```python
from collocation import eivd

# 应用EIVD
EeeT, SNR, rho2, fMSE, L = eivd(tri)

print("误差协方差矩阵:\n", EeeT)
print("误差相关系数:", EeeT[1,2] / np.sqrt(EeeT[1,1] * EeeT[2,2]))
```

**参考文献:**

> Dong, J., Crow, W. T., Duan, Z., Wei, L., & Lu, Y. (2019). An instrument variable based algorithm for estimating cross-correlated hydrological remote sensing errors. Journal of Hydrology, 581, 124385.

**适用场景:**
- 产品可能有相关误差
- 共同误差源
- 更真实的误差结构

---

### EC (扩展交叉定标)

**函数签名:**
```python
collocation.ec(qu)
```

**描述:**

四路交叉定标方法，可以处理四个产品并估计所有可能的三元组合。

**参数:**

- **qu** : `np.ndarray`, shape (n, 4)
  - 输入数据矩阵，包含四个产品

**返回:**

- **results** : `list` of `ECResult`
  - 每个三元组合的结果列表

**ECResult对象包含:**
- `EeeT`: 误差协方差矩阵
- `SNR`: 信噪比
- `rho2`: R²
- `fMSE`: 分数MSE
- `re_weight`: 最优融合权重
- `weighted_result`: 融合结果
- `combination`: 产品组合索引

**示例:**

```python
from collocation import ec
from collocation.ec import select_best_combination

# 四个产品
quad = np.column_stack([p1, p2, p3, p4])

# 应用EC
results = ec(quad)

print(f"找到 {len(results)} 个组合")

# 选择最佳组合
best = select_best_combination(results, criterion='min_fMSE')
print(f"最佳组合: {best.combination}")
print(f"融合权重: {best.re_weight}")

# 获取融合结果
merged = best.weighted_result[0]
```

**辅助函数:**

```python
# 按不同标准选择最佳组合
best = select_best_combination(results, criterion='min_fMSE')
# criterion 可选: 'min_fMSE', 'max_SNR', 'max_rho2'
```

**参考文献:**

> Gruber, A., Su, C. H., Zwieback, S., Crow, W., Dorigo, W., & Wagner, W. (2016). Recent advances in (soil moisture) triple collocation analysis. International Journal of Applied Earth Observation and Geoinformation, 45, 200-211.

**适用场景:**
- 有四个或更多产品
- 需要多传感器融合
- 探索不同产品组合

---

## 简单方法

### SimpleAverage (简单平均)

**函数签名:**
```python
collocation.simple_average(data, weights=None, axis=0)
```

**描述:**

计算多个数据产品的简单平均或加权平均。这是最基础的数据融合方法。

**参数:**

- **data** : `np.ndarray`
  - 输入数据数组
  - 形状可以是 (n_samples, n_products) 或 (n_products, n_samples)

- **weights** : `np.ndarray`, 可选
  - 各产品的权重
  - 如果为None，使用等权重
  - 将自动归一化

- **axis** : `int`, 默认=0
  - 沿哪个轴平均
  - 0: 沿第一维平均
  - 1: 沿第二维平均

**返回:**

- **averaged** : `np.ndarray`
  - 平均结果

**示例:**

```python
from collocation import simple_average

# 简单平均
data = np.column_stack([p1, p2, p3])
avg = simple_average(data, axis=1)

# 加权平均
weights = np.array([0.5, 0.3, 0.2])
wavg = simple_average(data, weights=weights, axis=1)
```

**辅助函数:**

#### inverse_variance_weights

```python
collocation.inverse_variance_weights(variances)
```

根据逆方差计算最优权重。

**参数:**
- **variances** : 各产品的误差方差

**返回:**
- **weights** : 归一化的最优权重

**示例:**
```python
from collocation import inverse_variance_weights

variances = np.array([1.0, 2.0, 4.0])
weights = inverse_variance_weights(variances)
print(weights)  # [0.571, 0.286, 0.143]
```

#### calculate_averaging_uncertainty

```python
collocation.calculate_averaging_uncertainty(variances, weights=None)
```

计算加权平均的不确定性。

**示例:**
```python
from collocation import calculate_averaging_uncertainty

std = calculate_averaging_uncertainty(variances, weights)
print(f"平均结果的标准差: {std:.3f}")
```

#### ensemble_statistics

```python
collocation.ensemble_statistics(data, axis=0)
```

计算集合统计量。

**返回:**
- **mean** : 均值
- **median** : 中位数
- **std** : 标准差
- **spread** : 范围 (max - min)

**示例:**
```python
from collocation import ensemble_statistics

mean, median, std, spread = ensemble_statistics(data, axis=0)
```

**适用场景:**
- 快速初步分析
- 不需要复杂统计
- 已知产品权重
- 基准方法

**优点:**
- 非常简单快速
- 不需要统计假设
- 降低随机误差
- 易于理解和解释

**局限性:**
- 无法估计误差特性
- 无法处理系统性偏差
- 没有不确定性量化
- 假设所有产品测量相同量

---

## 贝叶斯方法

### BTC (贝叶斯三路交叉定标)

**要求:** PyMC3 >= 3.11.0

**类签名:**
```python
collocation.BayesianTC(data)
```

**描述:**

贝叶斯三路交叉定标方法，提供完整的不确定性量化和时变误差估计。

**初始化参数:**

- **data** : `np.ndarray`, shape (n_products, n_samples)
  - 输入数据
  - 注意：与经典方法不同，这里每行是一个产品

**主要方法:**

#### setup_model

```python
btc.setup_model(
    doft=4,
    priorfactor=1.0,
    thetaoffset=0.15,
    thetamodel='beta',
    studenterrors=False
)
```

设置贝叶斯模型参数。

**参数:**
- **doft** : 先验自由度
- **priorfactor** : 先验缩放因子
- **thetaoffset** : 乘性偏差偏移
- **thetamodel** : 'beta' 或 'logistic'
- **studenterrors** : 是否使用Student-t误差

#### run_inference

```python
btc.run_inference(niter=2000, nadvi=200000, seed=None)
```

运行MCMC推断。

**参数:**
- **niter** : MCMC迭代次数
- **nadvi** : ADVI预热迭代次数
- **seed** : 随机种子

#### get_error_estimates

```python
rmse_mean, rmse_std, rmse_quantiles = btc.get_error_estimates()
```

获取误差估计。

**返回:**
- **rmse_mean** : RMSE后验均值
- **rmse_std** : RMSE后验标准差
- **rmse_quantiles** : 2.5%, 50%, 97.5% 分位数

#### get_calibration_parameters

```python
m_mean, l_mean = btc.get_calibration_parameters()
```

获取标定参数。

**返回:**
- **m_mean** : 加性偏差
- **l_mean** : 乘性偏差

**完整示例:**

```python
from collocation import BayesianTC, BAYESIAN_AVAILABLE

if BAYESIAN_AVAILABLE:
    # 准备数据 (每行是一个产品)
    data = np.array([product1, product2, product3])

    # 初始化
    btc = BayesianTC(data)

    # 设置模型
    btc.setup_model(doft=4, priorfactor=1.0)

    # 运行推断
    btc.run_inference(niter=2000, nadvi=200000, seed=42)

    # 获取结果
    rmse_mean, rmse_std, rmse_q = btc.get_error_estimates()
    m_mean, l_mean = btc.get_calibration_parameters()

    # 打印摘要
    btc.summary()

    # 打印结果
    for i in range(3):
        print(f"产品{i+1}:")
        print(f"  RMSE: {rmse_mean[i]:.3f} ± {rmse_std[i]:.3f}")
        print(f"  95% CI: [{rmse_q[0][i]:.3f}, {rmse_q[2][i]:.3f}]")
        print(f"  加性偏差: {m_mean[i]:.3f}")
        print(f"  乘性偏差: {l_mean[i]:.3f}")
else:
    print("请安装PyMC3: pip install pymc3==3.11.5 theano-pymc")
```

**参考文献:**

> Zwieback, S., Scipal, K., Dorigo, W., & Wagner, W. (2012). Structural and statistical properties of the collocation technique for error characterization. Nonlinear Processes in Geophysics, 19(1), 69-80.

**适用场景:**
- 需要完整不确定性量化
- 时变误差结构
- 复杂异方差模型
- 充足的计算资源

**优点:**
- 完整的后验分布
- 可信区间
- 时变参数
- 灵活的先验

**局限性:**
- 计算成本高
- 需要MCMC专业知识
- 收敛诊断复杂

---

## 工具函数

### mse_judge

```python
collocation.mse_judge(simulate, truth)
```

计算MSE和其他评估指标。

**参数:**
- **simulate** : 模拟/估计值
- **truth** : 真实值

**返回:** 评估指标字典

### kge_objfun

```python
collocation.kge_objfun(simulate, truth)
```

计算KGE (Kling-Gupta效率) 目标函数。

---

## 性能指标

### calculate_all_metrics

```python
from collocation.utils import calculate_all_metrics

metrics = calculate_all_metrics(simulated, observed)
```

**返回:**
- **r** : 相关系数
- **KGE** : Kling-Gupta效率
- **NSE** : Nash-Sutcliffe效率
- **PBIAS** : 百分比偏差
- **RMSE** : 均方根误差
- **MAE** : 平均绝对误差

**示例:**

```python
metrics = calculate_all_metrics(merged, truth)

print(f"相关系数: {metrics['r']:.3f}")
print(f"KGE: {metrics['KGE']:.3f}")
print(f"NSE: {metrics['NSE']:.3f}")
print(f"PBIAS: {metrics['PBIAS']:.2f}%")
print(f"RMSE: {metrics['RMSE']:.3f}")
print(f"MAE: {metrics['MAE']:.3f}")
```

**指标说明:**

- **r (相关系数)**:
  - 范围: [-1, 1]
  - 1表示完美正相关

- **KGE (Kling-Gupta效率)**:
  - 范围: (-∞, 1]
  - 1表示完美匹配
  - 综合考虑相关性、偏差和变异性

- **NSE (Nash-Sutcliffe效率)**:
  - 范围: (-∞, 1]
  - 1表示完美匹配
  - 水文模型标准指标

- **PBIAS (百分比偏差)**:
  - 范围: (-∞, ∞)
  - 0表示无偏差
  - 正值表示低估，负值表示高估

- **RMSE (均方根误差)**:
  - 范围: [0, ∞)
  - 0表示无误差
  - 对大误差敏感

- **MAE (平均绝对误差)**:
  - 范围: [0, ∞)
  - 0表示无误差
  - 对异常值更鲁棒

---

## 使用建议

### 方法选择指南

1. **只有2个产品:**
   - 使用 `ivd` 或 `ivs`
   - 需要不确定性 → `ivs`
   - 需要融合权重 → `ivd`
   - 快速分析 → `simple_average`

2. **有3个产品:**
   - 假设误差独立 → `tc`
   - 可能有误差相关 → `eivd`
   - 需要贝叶斯推断 → `BayesianTC`
   - 快速分析 → `simple_average`

3. **有4个或更多产品:**
   - 使用 `ec`
   - 或对所有三元组使用 `tc`/`eivd`
   - 快速分析 → `simple_average`

4. **特殊需求:**
   - 时变误差 → `BayesianTC`
   - 完整不确定性 → `BayesianTC` 或 `ivs`
   - 快速原型 → `simple_average`
   - 最简单 → `simple_average`

### 数据准备

所有方法都需要:
1. 数据去均值化（可选但推荐）
2. 时间对齐
3. 无缺失值
4. 合理的样本量（至少100个点）

### 结果解释

- **信噪比 (SNR)**: 越高越好，>10通常认为很好
- **R² (rho2)**: 越接近1越好，>0.8通常认为很好
- **分数MSE (fMSE)**: 越小越好，<0.2通常认为很好
- **误差方差**: 越小越好，与数据尺度相关

---

## 完整工作流程示例

```python
import numpy as np
from collocation import tc, eivd, ec, simple_average
from collocation.utils import calculate_all_metrics

# 1. 准备数据
data = np.column_stack([product1, product2, product3])

# 2. 去均值化（可选）
data_centered = data - data.mean(axis=0)

# 3. 应用TC
print("=== 三路交叉定标 ===")
EeeT, SNR, rho2, fMSE = tc(data_centered)
print(f"误差方差: {np.diag(EeeT)}")
print(f"SNR: {SNR}")
print(f"R²: {rho2}")

# 4. 应用EIVD（如果怀疑有误差相关）
print("\n=== 扩展IVD ===")
EeeT_e, SNR_e, rho2_e, fMSE_e, L = eivd(data_centered)
print(f"误差协方差矩阵:\n{EeeT_e}")

# 5. 简单平均作为基准
print("\n=== 简单平均 ===")
avg = simple_average(data, axis=1)

# 6. 与真值比较（如果有）
if truth is not None:
    for i, product in enumerate([product1, product2, product3]):
        metrics = calculate_all_metrics(product, truth)
        print(f"\n产品{i+1} 指标:")
        print(f"  R: {metrics['r']:.3f}")
        print(f"  RMSE: {metrics['RMSE']:.3f}")

    metrics_avg = calculate_all_metrics(avg, truth)
    print(f"\n简单平均 指标:")
    print(f"  R: {metrics_avg['r']:.3f}")
    print(f"  RMSE: {metrics_avg['RMSE']:.3f}")

# 7. 如果有4个产品
if have_4_products:
    print("\n=== 扩展交叉定标 ===")
    quad = np.column_stack([product1, product2, product3, product4])
    results = ec(quad)

    from collocation.ec import select_best_combination
    best = select_best_combination(results, criterion='min_fMSE')
    print(f"最佳组合: {best.combination}")
    print(f"融合权重: {best.re_weight}")
```

---

## 常见问题

**Q: 为什么我的误差方差是负数？**
A: 这通常意味着数据不满足交叉定标的假设（如误差相关性）。尝试使用EIVD或检查数据质量。

**Q: 我应该使用多少数据点？**
A: 至少100个独立观测。更多更好，特别是对于贝叶斯方法。

**Q: 数据需要去均值化吗？**
A: 推荐但不是必须的。去均值化可以提高数值稳定性。

**Q: 如何处理缺失值？**
A: 必须删除或插值。交叉定标方法需要完整的时间序列。

**Q: 简单平均和TC哪个更好？**
A: TC能提供误差估计和最优权重，但简单平均更快更稳定。对于初步分析，先用简单平均。

**Q: 贝叶斯方法值得额外的计算成本吗？**
A: 如果你需要完整的不确定性量化或怀疑时变误差，那么值得。否则经典方法通常足够。

---

## 联系与支持

如有问题或建议:
- GitHub Issues: [项目地址]
- Email: your.email@example.com
- 文档: [在线文档地址]

---

**最后更新:** 2024-10-31
**版本:** 1.2.0
