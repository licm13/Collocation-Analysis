# Collocation Analysis Package - 交叉定标分析工具包

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

[English](README.md) | 简体中文

一个全面的Python工具包，用于遥感和地球物理数据集的交叉定标误差分析。由MATLAB代码转换而来，并增强了现代Python特性。

## 概述

交叉定标分析方法能够在不需要地面真值的情况下量化多个数据集中的误差。这些技术广泛应用于遥感、气候科学和地球物理数据验证。

## 功能特性

### 已实现的方法

#### 经典方法

- **IVD** (Information Vector Dual，信息向量对偶): 2路交叉定标，优化时间偏移
- **IVS** (Information Vector with Scaling，带尺度的信息向量): 2路交叉定标，Bootstrap不确定性估计
- **TC** (Triple Collocation，三路交叉定标): 经典3路交叉定标，假设误差独立
- **EIVD** (Extended IVD，扩展IVD): 3路交叉定标，允许误差相关
- **EC** (Extended Collocation，扩展交叉定标): 4路四元交叉定标，提供最优融合权重
- **ETCC** (Extended Triple Collocation for Correlation，最大相关性扩展三路交叉定标): 3路交叉定标，优化与真值的相关性
  - 最大化相关性而非最小化误差方差
  - 使用穷举搜索寻找最优融合权重
  - 基于Wei et al. (2023)的降水融合方法
  - 适用于相关性比RMSE更重要的应用场景

#### 贝叶斯方法

- **BTC** (Bayesian Triple Collocation，贝叶斯三路交叉定标): 3路交叉定标，完整贝叶斯推断
  - 时变误差结构
  - 复杂异方差模型
  - 通过MCMC完整量化不确定性
  - 非常数标定参数
  - 需要PyMC3（可选依赖）

- **BTCH** (Bayesian Three-Cornered Hat，贝叶斯三角帽): 3路交叉定标，贝叶斯不确定性量化
  - 常数误差方差（同方差）
  - 完整贝叶斯不确定性量化
  - 比BTC更简单快速
  - 适用于不期望时变误差的场景
  - 需要PyMC3（可选依赖）

#### 简单平均方法

- **SimpleAverage** (简单平均): 多产品简单算术平均
  - 最基础的数据融合方法
  - 适用于快速初步分析
  - 不需要复杂统计假设
  - 配套ET产品数据集分析工具

#### 应用: 生态系统限制指数 (ELI)

- **新增**: 完整的Python实现ELI计算框架
  - 从MATLAB代码转换，用于生态系统水/能量限制分析
  - 处理多个气候数据源 (ERA5-Land, GLEAM, GLDAS)
  - 应用所有交叉定标方法 (IVD, EIVD, TC, Bayesian TC)
  - 基于论文: "Widespread shift from ecosystem energy to water limitation with climate change"
  - 变量: 土壤湿度、蒸散发、蒸腾、辐射
  - 详见 [ELI_README.md](ELI_README.md)
  - 示例: `examples/eli_comprehensive_example.py`

### 核心功能

- 无需地面真值的误差方差估计
- 信噪比计算
- 数据-真值相关性估计
- 最优数据融合权重
- 误差相关性检测
- Bootstrap不确定性量化
- 全面的性能指标 (KGE, NSE, RMSE, MAE等)

## 安装

### 从源代码安装

```bash
git clone https://github.com/yourusername/Collocation-Analysis.git
cd Collocation-Analysis
pip install -e .
```

### 依赖项

#### 核心依赖
- Python >= 3.7
- NumPy >= 1.18.0
- SciPy >= 1.4.0
- Matplotlib >= 3.1.0 (用于示例)
- pytest >= 6.0.0 (用于测试)

#### 可选依赖 (用于贝叶斯方法)
- PyMC3 >= 3.11.0 (用于贝叶斯三路交叉定标)
- Theano >= 1.0.0 (PyMC3的后端)

安装贝叶斯支持:
```bash
pip install -e . "pymc3>=3.11.0" "theano-pymc"
```

## 快速入门

### 基础示例: 三路交叉定标

```python
import numpy as np
from collocation import tc

# 三个测量同一变量的独立产品
product1 = np.random.randn(200) + true_signal
product2 = np.random.randn(200) + true_signal
product3 = np.random.randn(200) + true_signal

# 堆叠成矩阵
data = np.column_stack([product1, product2, product3])

# 应用三路交叉定标
EeeT, SNR, rho2, fMSE = tc(data)

print("误差方差:", np.diag(EeeT))
print("信噪比:", SNR)
print("数据-真值相关性:", rho2)
```

### 两个产品的IVD方法

```python
from collocation import ivd

# 两个产品
dual = np.column_stack([product1, product2])

# 应用IVD
EeeT, rho2, weights = ivd(dual)

# 使用最优权重融合产品
merged = weights[0] * product1 + weights[1] * product2
```

### 四个产品的扩展交叉定标

```python
from collocation import ec
from collocation.ec import select_best_combination

# 四个产品
quad = np.column_stack([product1, product2, product3, product4])

# 应用扩展交叉定标
results = ec(quad)

# 选择最佳组合
best = select_best_combination(results, criterion='min_fMSE')

# 获取融合结果
merged = best.weighted_result[0]
```

### ETCC - 最大化相关性

```python
from collocation import ETCC, TripleCollocation

# 三个降水产品
precip1 = np.array([...])  # 产品1
precip2 = np.array([...])  # 产品2
precip3 = np.array([...])  # 产品3

# 传统TC（最小化误差方差）
tc_method = TripleCollocation()
tc_merged = tc_method.merge(precip1, precip2, precip3)
print("TC权重:", tc_method.weights)
print("TC误差方差:", tc_method.error_variances)

# ETCC（最大化与真值的相关性）
etcc_method = ETCC(weight_increment=0.01)
etcc_merged = etcc_method.merge(precip1, precip2, precip3)
print("ETCC权重:", etcc_method.weights)
print("ETCC最大相关性:", etcc_method.max_correlation)
print("各产品与真值的相关性:", etcc_method.correlation_with_truth)

# 对于网格化空间数据
from collocation import SpatialMerging

# 数据形状: (lat, lon, time)
spatial_merger = SpatialMerging(method='etcc', weight_increment=0.01)
merged_grid = spatial_merger.merge_gridded(grid1, grid2, grid3, axis=-1)
```

### 简单平均方法

```python
import numpy as np

# 多个产品
products = [product1, product2, product3]

# 简单平均
merged = np.mean(products, axis=0)

# 或使用SimpleAverage工具进行ET产品分析
# 详见 SimpleAverage/README.md
```

### 贝叶斯方法 (高级)

#### 贝叶斯三路交叉定标 (时变误差)

```python
from collocation import BayesianTC, BAYESIAN_AVAILABLE

if BAYESIAN_AVAILABLE:
    # 三个产品 (n_products, n_samples)
    data = np.array([product1, product2, product3])

    # 初始化并运行贝叶斯TC
    btc = BayesianTC(data)
    btc.run_inference(niter=2000, nadvi=200000)

    # 获取完整不确定性量化的结果
    rmse_mean, rmse_std, rmse_quantiles = btc.get_error_estimates()

    # 获取标定参数
    m_mean, l_mean = btc.get_calibration_parameters()

    # 打印摘要
    btc.summary()
else:
    print("安装PyMC3以使用贝叶斯方法: pip install pymc3==3.11.5 theano-pymc")
```

#### 贝叶斯三角帽 (常数误差)

```python
from collocation import BayesianTCH, BAYESIAN_TCH_AVAILABLE

if BAYESIAN_TCH_AVAILABLE:
    # 三个产品 (n_products, n_samples)
    data = np.array([product1, product2, product3])

    # 初始化并运行贝叶斯TCH（比BTC更快）
    btch = BayesianTCH(data)
    btch.run_inference(niter=2000, nadvi=50000)

    # 获取不确定性量化的结果
    rmse_mean, rmse_std, rmse_quantiles = btch.get_error_estimates()

    # 获取带不确定性的信噪比
    snr_mean, snr_std, snr_quantiles = btch.get_snr()

    # 获取带不确定性的相关性
    rho2_mean, rho2_std, rho2_quantiles = btch.get_correlation()

    # 打印详细摘要
    btch.summary(verbose=True)

    # 绘制后验分布
    fig, axes = btch.plot_posterior()
else:
    print("安装PyMC3以使用贝叶斯方法: pip install pymc3==3.11.5 theano-pymc")
```

## 方法比较

| 方法 | 产品数 | 误差相关 | 时间信息 | 不确定性 | 优化目标 | 最适用于 |
|------|--------|---------|---------|---------|---------|---------|
| **IVD** | 2 | 否 | 基于偏移 | 否 | 解析解 | 两个产品的最优权重融合 |
| **IVS** | 2 | 否 | 滞后1 + Bootstrap | Bootstrap | 解析解 | 不确定性量化 (2路) |
| **TC** | 3 | 零 (假设) | 否 | 否 | 最小RMSE | 标准3路验证 |
| **ETCC** | 3 | 零 (假设) | 否 | 否 | 最大相关性 | 降水/优先考虑相关性的应用 |
| **EIVD** | 3 | 是 (估计) | 滞后1 | 否 | 解析解 | 有相关误差的产品 |
| **EC** | 4 | 是 (估计) | 否 | 否 | 解析解 | 多传感器融合 |
| **BTC** | 3+ | 是 (估计) | 时变 | 完整贝叶斯 | MCMC | 复杂误差结构，时变 |
| **BTCH** | 3 | 零 (假设) | 否 | 完整贝叶斯 | MCMC | 常数误差的不确定性量化 |
| **SimpleAverage** | 2+ | 不考虑 | 否 | 否 | 算术平均 | 快速初步分析 |

## 完整示例

### 示例1: 所有经典方法

运行演示所有经典方法的完整示例:

```bash
cd examples
python example_all_methods.py
```

这将:
1. 生成具有已知误差的合成数据
2. 应用所有经典交叉定标方法 (IVD, IVS, TC, EIVD, EC)
3. 与地面真值比较结果
4. 创建可视化图表

### 示例2: 综合比较 (出版质量)

运行跨多个场景的综合比较，生成Nature/Science质量的图表:

```bash
cd examples
python comprehensive_comparison_TCH_included.py
```

此高级示例包括:
- **6个真实场景**: 理想、相关误差、时变、有偏差、重尾、真实
- **所有方法比较**: IVD, IVS, TC, TCH, EIVD, EC, BTC, BTCH
- **出版质量图表**: 遵循Nature/Science期刊标准
  - 300 DPI分辨率
  - 色盲友好调色板
  - 适当的字体大小 (Arial/Helvetica, 7-9pt)
  - 单栏 (89mm) 和双栏 (183mm) 布局
- **综合指标**: RMSE, 相关性, 相对误差, 分布
- **统计比较**: 不同挑战性条件下的性能

输出:
- 各场景比较图
- 总体性能比较
- 详细结果表

### 示例3: SimpleAverage工具使用

```bash
cd SimpleAverage
python quick_scan.py  # 快速扫描ET产品数据集
python analyze_et_products.py  # 详细分析ET产品
```

详见 [SimpleAverage/README.md](SimpleAverage/README.md)

## SimpleAverage工具

SimpleAverage文件夹提供了ET（蒸散发）产品数据集分析工具，支持快速扫描和详细分析。

### 主要功能

- **快速扫描** (`quick_scan.py`): 快速查看文件夹结构和文件统计
- **详细分析** (`analyze_et_products.py`): 深入分析NetCDF文件元数据
  - 自动识别时空分辨率
  - ET类型分类（总ET、PET、分量等）
  - 生成CSV和Excel报告
  - 可视化进度条（支持tqdm）

### 使用方法

1. 修改数据路径:
```python
data_path = r"你的数据路径"
```

2. 运行分析:
```bash
python SimpleAverage/analyze_et_products.py
```

详细文档请参考 [SimpleAverage/README.md](SimpleAverage/README.md)

## ELI应用

生态系统限制指数 (ELI) 应用提供了完整的工作流程，用于分析陆地生态系统中的水与能量限制。

### 快速入门

```python
from collocation import ELIProcessor

# 初始化处理器
processor = ELIProcessor()

# 使用EIVD处理三个数据源
results = processor.process_triple_eivd(
    era5l_data,  # ERA5-Land再分析
    gleam_data,  # GLEAM蒸散发
    gldas_data,  # GLDAS陆面模型
    variable='eta'
)

# 保存结果
processor.save_to_netcdf(
    results,
    'eli_eta_results.nc',
    variable='eta',
    data_source='ERA5L+GLEAM+GLDAS'
)
```

### 完整示例

```bash
# 快速测试（验证安装）
python examples/test_eli_quick.py

# 综合示例（所有方法）
python examples/eli_comprehensive_example.py
```

### 文档

详见 [ELI_README.md](ELI_README.md):
- 完整API文档
- 数据格式规范
- 方法选择指南
- NetCDF I/O示例
- 性能优化技巧

## 测试

运行测试套件:

```bash
pytest tests/test_collocation.py -v
```

运行带覆盖率的测试:

```bash
pytest tests/test_collocation.py --cov=collocation --cov-report=html
```

测试ELI模块:

```bash
python examples/test_eli_quick.py
```

## API文档

### IVD (信息向量对偶)

```python
from collocation import ivd

EeeT, rho2, u = ivd(dual)
```

**参数:**
- `dual`: 输入数据 (n, 2) - 两个产品

**返回:**
- `EeeT`: 误差协方差矩阵 (2, 2)
- `rho2`: 数据-真值相关性 (2,)
- `u`: 最优融合权重 (2,)

**参考文献:**
> Dong, J., et al. (2014). Fusing active and passive remotely sensed soil moisture products using information vectors.

### IVS (带尺度的信息向量)

```python
from collocation import ivs

RMSE, rho2 = ivs(X, N_boot=1000, column=1, M_A=1)
```

**参数:**
- `X`: 输入数据 (n, 2)
- `N_boot`: Bootstrap样本数 (默认: 1000)
- `column`: 滞后1序列的列 (1或2)
- `M_A`: 误差模型 (0=加性, 1=乘性)

**返回:**
- `RMSE`: 误差估计 (2,)
- `rho2`: 相关性 (2,)

### TC (三路交叉定标)

```python
from collocation import tc

EeeT, SNR, rho2, fMSE = tc(tri)
```

**参数:**
- `tri`: 输入数据 (n, 3) - 三个产品

**返回:**
- `EeeT`: 误差协方差矩阵 (3, 3)
- `SNR`: 信噪比 (3,)
- `rho2`: 数据-真值相关性 (3,)
- `fMSE`: 分数MSE (3,)

**参考文献:**
> Stoffelen, A. (1998). Toward the true near-surface wind speed: Error modeling and calibration using triple collocation. JGR, 103(C4), 7755-7766.

### EIVD (扩展IVD)

```python
from collocation import eivd

EeeT, SNR, rho2, fMSE, L = eivd(tri)
```

**参数:**
- `tri`: 输入数据 (n, 3)

**返回:**
- `EeeT`: 带交叉相关的误差协方差矩阵 (3, 3)
- `SNR`: 信噪比 (3,)
- `rho2`: 数据-真值相关性 (3,)
- `fMSE`: 分数MSE (3,)
- `L`: 滞后1自相关 (3,)

**注意:** 产品2和3可以有非零误差交叉相关。

### EC (扩展交叉定标)

```python
from collocation import ec

results = ec(qu)
```

**参数:**
- `qu`: 输入数据 (n, 4) - 四个产品

**返回:**
- `results`: ECResult对象列表 (每个组合一个)

每个结果包含:
- `EeeT`: 误差协方差矩阵
- `SNR`, `rho2`, `fMSE`: 质量指标
- `re_weight`: 最优融合权重
- `weighted_result`: 融合输出

**参考文献:**
> Gruber, A., et al. (2016). Estimating error cross-correlations in soil moisture data sets using extended collocation analysis. JGR: Atmospheres, 121(3), 1208-1219.

### ETCC (最大相关性扩展三路交叉定标)

```python
from collocation import ETCC, TripleCollocation, SpatialMerging

# 传统TC（最小化RMSE）
tc = TripleCollocation()
merged_tc = tc.merge(x, y, z)

# ETCC（最大化相关性）
etcc = ETCC(weight_increment=0.01, min_correlation=0.01)
merged_etcc = etcc.merge(x, y, z)

# 网格化数据的空间融合
spatial = SpatialMerging(method='etcc', weight_increment=0.01)
merged_grid = spatial.merge_gridded(x_grid, y_grid, z_grid, axis=-1)
```

**参数 (TripleCollocation):**
- 无初始化参数

**参数 (ETCC):**
- `weight_increment`: 权重搜索步长（默认：0.01）
  - 值越小搜索越精细但速度越慢
  - 0.01对应约5,151个权重组合
- `min_correlation`: 最小相关性阈值（默认：0.01）
  - 防止数值不稳定

**参数 (SpatialMerging):**
- `method`: 'tc'或'etcc'（默认：'etcc'）
- `**kwargs`: 传递给TC或ETCC的额外参数

**方法:**
```python
# TripleCollocation
merged = tc.merge(x, y, z)
# 访问结果:
tc.weights              # {'wx': float, 'wy': float, 'wz': float}
tc.error_variances      # {'sigma2_x': float, 'sigma2_y': float, 'sigma2_z': float}

# ETCC
merged = etcc.merge(x, y, z)
# 访问结果:
etcc.weights                   # {'wx': float, 'wy': float, 'wz': float}
etcc.max_correlation          # 达到的最大相关性
etcc.correlation_with_truth   # {'rho_Rx': float, 'rho_Ry': float, 'rho_Rz': float}

# SpatialMerging
merged_grid = spatial.merge_gridded(x, y, z, axis=-1)
```

**输入格式:**
- `x, y, z`: 点式融合的1D数组
- 空间数据: 3D数组 (lat, lon, time)或(time, lat, lon)

**返回:**
- `merged`: 与输入形状相同的融合产品

**主要区别:**
- **TC**: 最小化误差方差（RMSE²），解析解
- **ETCC**: 最大化与真值的相关性，穷举搜索
- **使用TC的场景**: RMSE最小化是目标
- **使用ETCC的场景**: 相关性更重要（如降水应用）

**参考文献:**
> Wei, M., et al. (2023). Ground validation of GPM IMERG precipitation products over Iran. *Geophysical Research Letters*, 50(18).

### BTC (贝叶斯三路交叉定标)

```python
from collocation import BayesianTC

# 用数据初始化
btc = BayesianTC(data)  # data形状: (n_products, n_samples)

# 运行推断
btc.run_inference(niter=2000, nadvi=200000, seed=123)

# 获取结果
rmse_mean, rmse_std, rmse_quantiles = btc.get_error_estimates()
m_mean, l_mean = btc.get_calibration_parameters()
```

### BTCH (贝叶斯三角帽)

```python
from collocation import BayesianTCH

# 用数据初始化
btch = BayesianTCH(data)  # data形状: (n_products, n_samples)，需要恰好3个产品

# 运行推断（比BTC更快）
btch.run_inference(niter=2000, nadvi=50000, seed=123)

# 获取结果
rmse_mean, rmse_std, rmse_quantiles = btch.get_error_estimates()
snr_mean, snr_std, snr_quantiles = btch.get_snr()
rho2_mean, rho2_std, rho2_quantiles = btch.get_correlation()

# 打印详细摘要
btch.summary(verbose=True)

# 绘制后验分布
fig, axes = btch.plot_posterior()
```

**参数:**
- `data`: 输入数据 (n_products, n_samples) - 三个或更多产品
- `niter`: MCMC迭代次数 (默认: 2000)
- `nadvi`: ADVI初始化迭代次数 (默认: 200000)
- `seed`: 随机种子

**返回:**
- `rmse_mean`: 每个产品RMSE的后验均值
- `rmse_std`: RMSE的后验标准差
- `rmse_quantiles`: 2.5%, 50%, 97.5% 分位数 (可信区间)
- `m_mean`: 加性偏差估计
- `l_mean`: 乘性偏差估计

**高级选项:**
```python
# 自定义推断参数
btc.setup_model(
    doft=4,                    # 先验自由度
    priorfactor=1.0,           # 先验缩放
    thetaoffset=0.15,          # 乘性偏差偏移
    thetamodel='beta',         # 'beta' 或 'logistic'
    studenterrors=False        # 使用Student-t误差
)
```

**主要特性:**
- 通过MCMC完整贝叶斯不确定性量化
- 时变误差方差估计
- 非常数标定参数
- 处理复杂异方差结构
- 可以纳入时变参数的解释变量

**参数 (BTCH):**
- `data`: 输入数据 (3, n_samples) - 恰好需要3个产品
- `niter`: MCMC迭代次数 (默认: 2000)
- `nadvi`: ADVI初始化迭代次数 (默认: 50000)
- `seed`: 随机种子
- `nchains`: MCMC链数 (默认: 2)

**方法:**
- `get_error_estimates()`: 返回RMSE均值、标准差和分位数
- `get_snr()`: 返回SNR均值、标准差和分位数
- `get_correlation()`: 返回相关性均值、标准差和分位数
- `summary(verbose=True)`: 打印详细结果
- `plot_posterior()`: 绘制后验分布

**与BTC的主要区别:**
- BTCH假设常数误差方差（更简单的模型）
- BTC允许时变误差（更复杂）
- BTCH更快（约2-5倍），需要更少的ADVI迭代
- BTCH最适合同方差误差
- BTC最适合异方差、时变误差

**参考文献:**
> Zwieback, S., et al. (2012). Structural and statistical properties of the collocation technique for error characterization. Nonlin. Processes Geophys., 19, 69-80.

## 性能指标

该工具包包含全面的评估指标:

```python
from collocation.utils import calculate_all_metrics

metrics = calculate_all_metrics(simulated, observed)
# 返回: r, KGE, NSE, PBIAS, RMSE, MAE
```

### 可用指标

- **KGE** (Kling-Gupta效率): 将MSE分解为相关性、偏差和变异性
- **NSE** (Nash-Sutcliffe效率): 标准水文模型性能
- **PBIAS** (百分比偏差): 系统偏差量化
- **RMSE** (均方根误差): 总体误差大小
- **MAE** (平均绝对误差): 平均误差大小
- **r** (相关性): 线性关联强度

## 应用领域

### 遥感
- 土壤湿度产品验证
- 降水数据集比较
- 海洋风速误差估计
- 地表温度分析

### 气候科学
- 模型-观测比较
- 再分析数据集评估
- 多源数据融合

### 地球物理
- 多传感器集成
- 误差预算估计
- 数据质量评估

## 数学背景

### 三路交叉定标理论

对于测量同一地球物理变量的三个产品:

```
X_i = α_i + β_i θ + ε_i
```

其中:
- `X_i`: 观测产品i
- `θ`: 真实信号 (未知)
- `α_i, β_i`: 标定参数
- `ε_i`: 随机误差

假设:
1. 误差零均值: E[ε_i] = 0
2. 误差与信号不相关: E[ε_i θ] = 0
3. 误差相互独立: E[ε_i ε_j] = 0 (对于TC)

协方差结构提供6个方程来求解3个误差方差和3个信号方差。

### 误差指标

**信噪比:**
```
SNR_i = (β_i² σ_θ²) / σ_ε_i²
```

**数据-真值相关性:**
```
ρ²_i = SNR_i / (1 + SNR_i)
```

**分数MSE:**
```
fMSE_i = 1 - ρ²_i = 1 / (1 + SNR_i)
```

## 项目结构

```
Collocation-Analysis/
├── collocation/           # 核心模块
│   ├── __init__.py
│   ├── ivd.py            # IVD方法
│   ├── ivs.py            # IVS方法
│   ├── tc.py             # TC方法
│   ├── eivd.py           # EIVD方法
│   ├── ec.py             # EC方法
│   ├── bayesian_tc.py    # 贝叶斯TC方法
│   ├── eli.py            # ELI应用
│   └── utils.py          # 工具函数
├── SimpleAverage/         # 简单平均工具
│   ├── README.md
│   ├── analyze_et_products.py
│   ├── quick_scan.py
│   └── UPDATE_NOTES.md
├── examples/              # 示例代码
│   ├── example_all_methods.py
│   ├── comprehensive_comparison.py
│   ├── eli_comprehensive_example.py
│   └── test_eli_quick.py
├── tests/                 # 测试
│   └── test_collocation.py
├── README.md              # 英文文档
├── README_CN.md           # 中文文档 (本文件)
├── ELI_README.md          # ELI详细文档
└── requirements.txt       # 依赖项
```

## 贡献

欢迎贡献！请:

1. Fork仓库
2. 创建特性分支
3. 为新功能添加测试
4. 确保所有测试通过
5. 提交pull request

## 引用

如果你在研究中使用此工具包，请引用:

```bibtex
@software{collocation_analysis,
  title = {Collocation Analysis Package},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/Collocation-Analysis}
}
```

同时请引用相关的方法学论文（见API文档部分）。

## 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 致谢

- 原始MATLAB实现: licm_13@163.com
- 转换为Python并增强了功能和全面的文档
- 基于Stoffelen (1998), Scipal等 (2008), Dong等 (2014, 2019), 和Gruber等 (2016) 的基础性工作

## 参考文献

1. Stoffelen, A. (1998). Toward the true near-surface wind speed: Error modeling and calibration using triple collocation. *Journal of Geophysical Research*, 103(C4), 7755-7766.

2. Scipal, K., Holmes, T., De Jeu, R., Naeimi, V., & Wagner, W. (2008). A possible solution for the problem of estimating the error structure of global soil moisture data sets. *Geophysical Research Letters*, 35(24).

3. Dong, J., Crow, W. T., Reichle, R., & Liu, Q. (2014). Fusing active and passive remotely sensed soil moisture products using information vectors.

4. Dong, J., Crow, W. T., Duan, Z., Wei, L., & Lu, Y. (2019). An instrument variable based algorithm for estimating cross-correlated hydrological remote sensing errors. *Journal of Hydrology*, 581, 124385.

5. Gruber, A., Su, C. H., Zwieback, S., Crow, W., Dorigo, W., & Wagner, W. (2016). Recent advances in (soil moisture) triple collocation analysis. *International Journal of Applied Earth Observation and Geoinformation*, 45, 200-211.

6. Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling. *Journal of Hydrology*, 377(1-2), 80-91.

## 联系方式

如有问题、建议或反馈:
- 在GitHub上提交issue
- Email: your.email@example.com

---

**注意:** 本工具包正在积极维护和持续改进中。请查看仓库以获取最新更新。
