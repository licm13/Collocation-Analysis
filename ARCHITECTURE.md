# Collocation Analysis Package — Architecture & Developer Guide
# 代码库架构与开发者指南

> **Version / 版本**: 2.0.0
> **Last Updated / 最后更新**: 2026-03-18
> **Languages / 语言**: English · 中文

---

## Table of Contents / 目录

1. [Repository at a Glance / 仓库概览](#1-repository-at-a-glance--仓库概览)
2. [Package Layout / 包目录结构](#2-package-layout--包目录结构)
3. [Core Analytical Methods / 核心分析方法](#3-core-analytical-methods--核心分析方法)
4. [v2.0 Upgrade Modules / v2.0 新增模块详解](#4-v20-upgrade-modules--v20-新增模块详解)
   - [4.1 base.py — Abstract Estimator Base](#41-basepy--abstract-estimator-base--抽象基类)
   - [4.2 estimators.py — Sklearn-style API](#42-estimatorspy--sklearn-style-api--统一估计器接口)
   - [4.3 consultant.py — Smart Recommender](#43-consultantpy--smart-recommender--智能推荐引擎)
   - [4.4 plotting.py — Visualization Layer](#44-plottingpy--visualization-layer--可视化层)
   - [4.5 eli_pipeline.py — ELI One-Click Pipeline](#45-eli_pipelinepy--eli-one-click-pipeline--eli一键分析管线)
5. [Data Fusion Subpackage / 数据融合子包](#5-data-fusion-subpackage--数据融合子包)
6. [Design Principles / 设计原则](#6-design-principles--设计原则)
7. [Dependency Map / 依赖关系图](#7-dependency-map--依赖关系图)
8. [Quick Reference / 快速参考](#8-quick-reference--快速参考)

---

## 1. Repository at a Glance / 仓库概览

### English

The **Collocation Analysis Package** is a Python library for **error quantification of geophysical datasets without ground truth**. It implements a growing family of collocation methods—from the classic two-way IVD to full Bayesian MCMC—and wraps them in a modern, consistent API.

The package was converted from a MATLAB toolbox and has evolved through three major phases:

| Phase | Focus | Key Additions |
|-------|-------|---------------|
| v1.0 | Port from MATLAB | `tc`, `ivd`, `ivs`, `eivd`, `ec` |
| v1.5 | Advanced methods | Bayesian TC/TCH, BTCH_He2020, MTCH, data fusion |
| **v2.0** | **Developer experience** | **sklearn API, smart consultant, Plotly dashboard, ELI pipeline** |

### 中文

**Collocation Analysis Package** 是一个用于**无需地面真值即可量化地球物理数据集误差**的 Python 库。它实现了从经典两路 IVD 到全贝叶斯 MCMC 的一系列交叉定标方法，并将其封装在现代化、统一的 API 中。

该包从 MATLAB 工具箱转换而来，经历了三个主要发展阶段：

| 阶段 | 重点 | 主要新增内容 |
|------|------|------------|
| v1.0 | 从 MATLAB 移植 | `tc`、`ivd`、`ivs`、`eivd`、`ec` |
| v1.5 | 高级方法 | 贝叶斯 TC/TCH、BTCH_He2020、MTCH、数据融合 |
| **v2.0** | **开发者体验** | **sklearn API、智能推荐、Plotly 仪表盘、ELI 管线** |

---

## 2. Package Layout / 包目录结构

```
Collocation-Analysis/
│
├── collocation/                    ← Main Python package / 主包
│   ├── __init__.py                 ← Public exports, version = 2.0.0
│   │
│   ├── ── Classical Methods / 经典方法 ──────────────────────────────
│   ├── ivd.py                      ← Information Vector Dual (2-way)
│   ├── ivs.py                      ← IVD + bootstrap scaling (2-way)
│   ├── tc.py                       ← Triple Collocation (3-way)
│   ├── eivd.py                     ← Extended IVD, allows cross-corr (3-way)
│   ├── ec.py                       ← Extended / Quadruple Collocation (4-way)
│   ├── etcc.py                     ← ETCC: correlation-maximising TC (3-way)
│   ├── etcc_evaluation.py          ← ETCC evaluation helpers
│   ├── etcc_spatial.py             ← ETCC spatial aggregation
│   ├── mtch.py                     ← Multiplicative-error TCH (N-way)
│   │
│   ├── ── Bayesian Methods / 贝叶斯方法 ──────────────────────────────
│   ├── bayesian_tc.py              ← BayesianTC: time-varying errors (PyMC3)
│   ├── bayesian_tch.py             ← BayesianTCH: constant errors (PyMC3)
│   ├── btch_he2020.py              ← BTCH analytical weights (no MCMC)
│   │
│   ├── ── v2.0 New Modules / v2.0 新增模块 ──────────────────────────
│   ├── base.py                     ← CollocationEstimator abstract base
│   ├── estimators.py               ← TC / EIVD / IVD / EC sklearn wrappers
│   ├── consultant.py               ← CollocationConsultant smart recommender
│   ├── plotting.py                 ← Static (matplotlib) + interactive (Plotly)
│   ├── eli_pipeline.py             ← ELI one-click parallel analysis pipeline
│   │
│   ├── ── Utilities / 工具模块 ──────────────────────────────────────
│   ├── utils.py                    ← KGE, NSE, RMSE, MAE, mse_judge
│   ├── covariance.py               ← Covariance construction helpers
│   ├── fuse.py                     ← Bias estimation helpers
│   ├── simple_average.py           ← IVW averaging baselines
│   ├── eli.py                      ← ELI processor (xarray, legacy API)
│   │
│   └── fusion/                     ← Data fusion subpackage / 数据融合子包
│       ├── __init__.py
│       ├── weights.py              ← IVW / GLS / BLUE / QP solvers
│       ├── covariance.py           ← MSE estimation, shrinkage
│       ├── fuse.py                 ← High-level fusion orchestrator
│       ├── constraints.py          ← Physics / sum-to-one constraints
│       ├── uncertainty.py          ← Bootstrap, variance propagation
│       ├── robust.py               ← Huber loss, outlier detection
│       ├── localization.py         ← Moving-window, biome partitioning
│       └── broadcast.py            ← Broadcasting utilities
│
├── tests/
│   ├── conftest.py                 ← Fixtures, bilingual reporting
│   ├── test_collocation.py         ← Core method tests (545 lines)
│   ├── test_fusion.py              ← Fusion module tests
│   ├── test_method_workflows.py    ← Integration tests
│   ├── test_performance.py         ← Benchmark / regression tests
│   └── test_upgrade.py             ← v2.0 new-module tests (48 cases)
│
├── examples/                       ← Runnable demonstration scripts
├── docs/                           ← Extra documentation
├── scripts/                        ← CLI tools (fuse_et.py, etc.)
│
├── README.md                       ← Short user guide (English)
├── README_CN.md                    ← Short user guide (Chinese)
├── ARCHITECTURE.md                 ← This file / 本文件
├── CLAUDE.md                       ← AI-assistant developer guide
├── ELI_README.md                   ← ELI application guide
├── BAYESIAN_INTEGRATION_GUIDE.md   ← Bayesian setup guide
├── PERFORMANCE_SUMMARY.md          ← Optimisation history
├── setup.py                        ← Package metadata
├── requirements.txt                ← Core + optional dependencies
└── pytest.ini                      ← Test configuration
```

---

## 3. Core Analytical Methods / 核心分析方法

### Method Comparison Table / 方法对比表

| Method | Products | Error Cross-corr | Uncertainty | Speed | Best For |
|--------|----------|-----------------|-------------|-------|----------|
| `ivd` | 2 | No | No | ★★★★★ | Active+passive fusion |
| `ivs` | 2 | No | Bootstrap CI | ★★★★ | 2-product with uncertainty |
| `tc` | 3 | No (assumed=0) | No | ★★★★★ | Standard 3-way analysis |
| `eivd` | 3 | Yes (P2×P3) | No | ★★★★★ | Correlated-error products |
| `ec` | 4 | No | No | ★★★★ | 4-product over-determination |
| `etcc` | 3 | No | Exhaustive | ★★★ | Correlation-optimised merging |
| `mtch` | N≥3 | No | No | ★★★★ | Log-normal / positive data |
| `btch_he2020` | 3 | Optional | No | ★★★★★ | Fast Bayesian weighting |
| `BayesianTC` | 3 | No | Full MCMC | ★ | Time-varying errors |
| `BayesianTCH` | 3 | No | Full MCMC | ★★ | Constant errors, uncertainty |

| 方法 | 产品数 | 误差互相关 | 不确定性 | 速度 | 最适场景 |
|------|--------|----------|---------|------|---------|
| `ivd` | 2 | 无 | 无 | ★★★★★ | 主动+被动遥感融合 |
| `ivs` | 2 | 无 | 自举置信区间 | ★★★★ | 带不确定性的两路分析 |
| `tc` | 3 | 无（假设为0）| 无 | ★★★★★ | 标准三路交叉定标 |
| `eivd` | 3 | 有（P2×P3）| 无 | ★★★★★ | 误差相关的产品 |
| `ec` | 4 | 无 | 无 | ★★★★ | 四路过定系统 |
| `etcc` | 3 | 无 | 穷举搜索 | ★★★ | 相关性优化融合 |
| `mtch` | N≥3 | 无 | 无 | ★★★★ | 对数正态/正值数据 |
| `btch_he2020` | 3 | 可选 | 无 | ★★★★★ | 快速贝叶斯权重 |
| `BayesianTC` | 3 | 无 | 全MCMC | ★ | 时变误差 |
| `BayesianTCH` | 3 | 无 | 全MCMC | ★★ | 恒定误差+不确定性 |

### Mathematical Foundations / 数学基础

All classical methods assume the **linear error model**:

所有经典方法均假设**线性误差模型**：

```
X_i = α_i + β_i · θ + ε_i
```

where / 其中：
- `X_i` — observed product i / 观测产品 i
- `θ` — unknown true signal / 未知真实信号
- `α_i`, `β_i` — calibration parameters / 定标参数
- `ε_i` — zero-mean random error / 零均值随机误差，`E[ε_i] = 0`

**TC** solves for `σ²_εi` using the system of covariance equations.
**EIVD** extends TC by additionally solving for `E[ε_2 · ε_3]`.
**MTCH** operates in log-space to handle multiplicative models: `Z_i = Θ · ε_i`.

---

## 4. v2.0 Upgrade Modules / v2.0 新增模块详解

### 4.1 `base.py` — Abstract Estimator Base / 抽象基类

#### Purpose / 目的

Provides `CollocationEstimator`, the abstract base class that all sklearn-style wrappers inherit. Centralises NaN handling, xarray conversion, and the "not fitted" guard so concrete estimators stay minimal.

提供 `CollocationEstimator` 抽象基类，所有 sklearn 风格的封装器均继承自该类。集中处理 NaN 清理、xarray 转换和"未拟合"检查，使具体估计器保持简洁。

#### Key Design / 核心设计

```python
class CollocationEstimator(ABC):
    """
    fit(data) → self          # Chain-friendly
    metrics_  : dict          # All results live here
    summary() → str           # Human-readable report
    """

    def fit(self, data):
        arr = self._coerce(data)   # xr.DataArray / Dataset → ndarray
        arr = self._clean(arr)     # drop NaN/Inf rows, warn if few samples
        self._fit(arr)             # ← implemented by subclass
        self.metrics_['n_samples'] = self.n_samples_
        return self                # enables chaining

    @abstractmethod
    def _fit(self, data: np.ndarray) -> None:
        """Subclass fills self.metrics_ here."""
```

#### Input Coercion Pipeline / 输入转换管道

```
User Input
   │
   ├─ xr.Dataset  → column-stack data variables
   ├─ xr.DataArray → .values
   ├─ list / tuple → np.asarray
   └─ ndarray      → as-is
         │
         ▼
   2-D float ndarray
         │
         ▼
   Drop NaN/Inf rows  (if dropna=True)
         │
         ▼
   Warn if n < min_samples
         │
         ▼
   Pass to _fit()
```

---

### 4.2 `estimators.py` — Sklearn-style API / 统一估计器接口

#### Purpose / 目的

Thin wrappers that map the raw functional APIs (`tc`, `eivd`, `ivd`, `ec`) to the `CollocationEstimator` interface. Users interact with a consistent `fit / metrics_` pattern regardless of which method they choose.

将原始函数式 API（`tc`、`eivd`、`ivd`、`ec`）映射到 `CollocationEstimator` 接口的轻量封装器。无论选择哪种方法，用户都通过统一的 `fit / metrics_` 模式交互。

#### Available Estimators / 可用估计器

| Class | Wraps | Input shape | Key `metrics_` keys |
|-------|-------|------------|---------------------|
| `TCEstimator` | `tc()` | `(n, 3)` | `EeeT`, `SNR`, `rho2`, `fMSE`, `error_std` |
| `EIVDEstimator` | `eivd()` | `(n, 3)` | + `L`, `cross_corr` |
| `IVDEstimator` | `ivd()` | `(n, 2)` | `EeeT`, `rho2`, `weights`, `error_std` |
| `ECEstimator` | `ec()` | `(n, 4)` | Aggregated median across reference-pair combinations |

#### Usage Patterns / 使用模式

```python
from collocation.estimators import TC, EIVD, IVD

# 1. Basic fit-and-read
model = TC().fit(data)
print(model.metrics_['error_std'])    # array([0.20, 0.31, 0.39])
print(model.summary())

# 2. Method-comparison loop  / 方法对比循环
results = {}
for name, cls in [('TC', TC), ('EIVD', EIVD)]:
    results[name] = cls().fit(data).metrics_

# 3. xarray input  / xarray 输入
import xarray as xr
da = xr.DataArray(data, dims=['time', 'product'])
TC().fit(da).get_metrics()

# 4. Method chaining  / 链式调用
std = TC(min_samples=50).fit(data).metrics_['error_std']
```

#### EC Aggregation Note / EC 聚合说明

`ec()` returns 6 `ECResult` objects (one per reference-pair combination), each containing 3 rescaling variants. `ECEstimator` takes the **element-wise median** across all valid combinations×variants to produce stable scalar metrics.

`ec()` 返回 6 个 `ECResult` 对象（每个参考对组合一个），每个包含 3 个重新缩放变体。`ECEstimator` 对所有有效组合×变体取**逐元素中位数**，以产生稳定的标量指标。

---

### 4.3 `consultant.py` — Smart Recommender / 智能推荐引擎

#### Purpose / 目的

`CollocationConsultant` acts as an automated first-look analysis tool. Given raw time series, it runs five diagnostic tests and returns a `ConsultationReport` with a ranked recommendation, supporting evidence, and a formatted narrative.

`CollocationConsultant` 充当自动化初步分析工具。给定原始时间序列，它运行五项诊断测试，返回包含优先推荐、支持证据和格式化叙述的 `ConsultationReport`。

#### Diagnostic Battery / 诊断项目

```
Input data (n, k)
       │
       ├─ [1] Lag-1 Autocorrelation
       │       ρ₁ = Corr(X_t, X_{t-1})
       │       High ρ₁ → serial correlation → BayesianTC
       │
       ├─ [2] Error Cross-correlation  ← v2.0 improved
       │       truth_proxy = row_mean(data)
       │       ε̂_i = X_i − truth_proxy
       │       r_ij = Corr(ε̂_i, ε̂_j)
       │       |r_ij| > 0.30 → EIVD
       │
       ├─ [3] Variance Stationarity
       │       rolling_std ratio: max/min over non-overlapping windows
       │       ratio > 2.5 → heteroscedastic → BayesianTC
       │
       ├─ [4] Normality Test
       │       Shapiro-Wilk (n ≤ 5000) or D'Agostino-Pearson
       │       p < 0.05 → warns: consider robust fusion
       │
       └─ [5] Skewness / Multiplicative Check
               |skew| > 1.5 and all values > 0 → MTCH
```

#### Cross-correlation Improvement in v2.0 / v2.0 互相关估计改进

**Why row-mean? / 为何使用行均值？**

In collocation data, all products observe the same truth `θ`. Subtracting the **column mean** (as done naively) still leaves `θ` in the residuals, creating artifically high cross-correlations even when errors are independent. Subtracting the **row mean** (a proxy for `θ`) cancels the shared signal:

在交叉定标数据中，所有产品观测相同的真实值 `θ`。朴素地减去**列均值**后，`θ` 仍残留于残差中，即使误差独立也会产生虚高的互相关。减去**行均值**（`θ` 的代理）可消除共享信号：

```
Column-mean residual:  X_i - mean(X_i)  = β_i·θ' + ε_i   ← θ still present!
Row-mean residual:     X_i - mean_j(X_j) ≈ ε_i + scale      ← errors isolated ✓
```

#### Decision Logic / 决策逻辑

```python
# Priority order (first triggered wins primary slot):
evidence = []

if |r_ij| > 0.30 and k == 3:
    evidence.append(('EIVD', reason))

if max_var_ratio > 2.5 or max_lag1 > 0.40:
    evidence.append(('BayesianTC', reason))

if max_skewness > 1.5 and all_positive:
    evidence.append(('MTCH', reason))

if k == 4:
    evidence.append(('EC', reason))

if k == 2:
    evidence.append(('IVD', reason))

if k == 3 and not correlated:
    evidence.append(('TC', reason))   # fallback
```

#### ConsultationReport / 报告对象

```python
report = CollocationConsultant(data, product_names=['ERA5','GLEAM','GLDAS']).consult()

report.recommended          # 'EIVD'
report.alternatives         # ['BayesianTC']
report.text                 # Formatted narrative
report.diagnostics          # {'lag1_autocorr': [...], 'variance_ratio': [...], ...}
report.warnings             # ['Product 2 residuals are non-normal (p=0.012)']
str(report)                 # Same as report.text
```

**Sample report output / 示例报告输出：**

```
════════════════════════════════════════════════════════════════
  Collocation Method Recommendation Report
════════════════════════════════════════════════════════════════
  Samples : 300
  Products: ERA5, GLEAM, GLDAS

  ★ PRIMARY RECOMMENDATION  →  EIVD
    Reason: High cross-correlation detected (GLEAM×GLDAS (r=0.54)).
            TC assumes independent errors — EIVD explicitly models
            error co-variance and reduces bias.

  ▸ ALSO CONSIDER:
    • TC: No significant error cross-correlation detected.

  ── Diagnostic Snapshot ──────────────────────────────
    ERA5             lag1ρ=+0.121  var_ratio=1.02  skew=+0.14
    GLEAM            lag1ρ=+0.234  var_ratio=1.08  skew=-0.08
    GLDAS            lag1ρ=+0.189  var_ratio=1.11  skew=+0.21

  Cross-correlations (proxy):
    ERA5 × GLEAM: r = +0.12
    ERA5 × GLDAS: r = +0.09
    GLEAM × GLDAS: r = +0.54 ← HIGH
════════════════════════════════════════════════════════════════
```

---

### 4.4 `plotting.py` — Visualization Layer / 可视化层

#### Purpose / 目的

Two-tier plotting: lightweight **matplotlib** static charts for publications, and a full **Plotly** interactive HTML dashboard for exploratory analysis.

双层可视化：轻量级 **matplotlib** 静态图表用于出版，完整 **Plotly** 交互式 HTML 仪表盘用于探索性分析。

#### Static API (matplotlib) / 静态 API

```python
from collocation.plotting import plot_error_comparison, plot_stability_heatmap
from collocation import tc

# Bar chart comparing multiple methods / 多方法误差柱状图
fig = plot_error_comparison(
    {'TC': tc_metrics, 'EIVD': eivd_metrics},
    product_names=['ERA5', 'GLEAM', 'GLDAS'],
    title='Error Std Dev by Method',
    save_path='error_comparison.png',
)

# Stability heatmap: error vs window size / 稳定性热力图
fig = plot_stability_heatmap(
    data, tc,
    window_sizes=[100, 200, 300, 400],
    product_names=['ERA5', 'GLEAM', 'GLDAS'],
)
```

#### Interactive Dashboard (Plotly) / 交互式仪表盘

```python
from collocation.plotting import InteractiveDashboard

dash = InteractiveDashboard(
    data,
    {'TC': tc_metrics, 'EIVD': eivd_metrics},
    product_names=['ERA5', 'GLEAM', 'GLDAS'],
    method_fn=tc,          # enables stability heatmap panel
    window_sizes=[100, 200, 300],
)
dash.show()               # open in browser / 在浏览器中打开
dash.save('report.html')  # standalone HTML / 独立 HTML 文件
```

#### Dashboard Layout / 仪表盘布局

```
┌────────────────────────────────────┬────────────────────┐
│  Time Series (range slider)        │  RMSE Bar Chart    │
│                                    │  (grouped by       │
│  ~~~~~ ERA5                        │   method)          │
│  ─ ─ ─ GLEAM                       │                    │
│  ·····  GLDAS                      │  TC   ██           │
│                                    │  EIVD ██           │
│  [────────────────] ← drag slider  │                    │
└────────────────────────────────────┴────────────────────┘
│  Window-size Stability Heatmap                          │
│                                                         │
│  ERA5  │ 0.21 │ 0.20 │ 0.20 │ 0.19 │  (colour = RMSE) │
│  GLEAM │ 0.33 │ 0.31 │ 0.30 │ 0.30 │                   │
│  GLDAS │ 0.42 │ 0.40 │ 0.39 │ 0.39 │                   │
│        │  100 │  200 │  300 │  400 │ ← window size     │
└─────────────────────────────────────────────────────────┘
```

#### Graceful Degradation / 优雅降级

```python
from collocation.plotting import PLOTLY_AVAILABLE
if not PLOTLY_AVAILABLE:
    # Falls back to matplotlib automatically
    # InteractiveDashboard raises ImportError with install hint
    pass
```

---

### 4.5 `eli_pipeline.py` — ELI One-Click Pipeline / ELI一键分析管线

#### Purpose / 目的

`ELIPipeline` collapses an entire Ecosystem Limitation Index analysis—data alignment, multi-method collocation, ELI ratio computation, and HTML reporting—into a single `run()` call.

`ELIPipeline` 将整个生态系统限制指数分析（数据对齐、多方法交叉定标、ELI 比率计算和 HTML 报告生成）压缩为单次 `run()` 调用。

#### Conceptual Background / 概念背景

The **Ecosystem Limitation Index (ELI)** quantifies whether terrestrial ecosystem productivity is more limited by water availability or energy availability:

**生态系统限制指数（ELI）**量化陆地生态系统生产力受水分可用性还是能量可用性的限制程度：

```
ELI_water  = 1 - mean(ρ²_water_products)    # fractional error of water vars
ELI_energy = 1 - mean(ρ²_energy_products)   # fractional error of energy vars

ELI_ratio  = ELI_water / (ELI_water + ELI_energy)

ELI_ratio > 0.5  →  water-limited  (水分限制)
ELI_ratio < 0.5  →  energy-limited (能量限制)
```

#### Pipeline Architecture / 管线架构

```
ELIPipeline(water, energy, vegetation)
       │
       ├─ _to_2d()          xr / list → 2-D ndarray
       ├─ shape validation  check n_rows equality
       │
       ├─ run()
       │    ├─ _preprocess()  subtract temporal mean (anomalies)
       │    ├─ Build analysis triplets:
       │    │    water+veg[:, 0]  →  (n, k_w+1)
       │    │    energy+veg[:, 0] →  (n, k_e+1)
       │    │
       │    ├─ ThreadPoolExecutor (max_workers=4 by default)
       │    │    ├─ TC   on water triplet
       │    │    ├─ TC   on energy triplet
       │    │    ├─ EIVD on water triplet
       │    │    ├─ EIVD on energy triplet
       │    │    ├─ IVS  on water[:, :2]
       │    │    ├─ IVS  on energy[:, :2]
       │    │    ├─ BTCH on water triplet
       │    │    └─ BTCH on energy triplet
       │    │
       │    ├─ _aggregate_eli()  → nanmean(1 - rho2) per category
       │    └─ _build_html()     → standalone HTML string
       │
       └─ ELIResult(eli_water, eli_energy, eli_ratio, method_results, html)
```

#### Usage / 使用示例

```python
import numpy as np
from collocation.eli_pipeline import ELIPipeline

# Load or generate input data
water = np.column_stack([swvl1, gleam_soil, gldas_soil])    # (n, 3)
energy = np.column_stack([swd, era5_Rn, gldas_Rn])          # (n, 3)
veg = et_observations                                         # (n,)

# One-click analysis / 一键分析
pipe = ELIPipeline(
    water, energy, veg,
    product_names={
        'water':     ['ERA5-SM', 'GLEAM-SM', 'GLDAS-SM'],
        'energy':    ['ERA5-SW', 'ERA5-Rn',  'GLDAS-Rn'],
        'vegetation':['GLEAM-ET'],
    },
    methods=['TC', 'EIVD', 'IVS', 'BTCH'],
    n_bootstrap=500,
)

result = pipe.run()
print(result.summary())
result.save('eli_diagnostic_report.html')

# Access raw results
for method, method_results in result.method_results.items():
    for r in method_results:
        if r.success:
            print(f"{method} [{r.category}]: error_std = {r.error_std}")
```

#### HTML Report Structure / HTML 报告结构

```
┌─────────────────────────────────────────────┐
│  🌿 ELI Diagnostic Report                   │
│  Generated: 2026-03-18  Samples: 365        │
│                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────────┐ │
│  │  0.4231 │  │  0.3178 │  │    0.571    │ │
│  │ELI Water│  │ELI Enrg │  │ ELI Ratio   │ │
│  │         │  │         │  │water-limited│ │
│  └─────────┘  └─────────┘  └─────────────┘ │
│                                             │
│  Water ███████████████████░░░░░░░  57.1%    │
│  Energy ░░░░░░░░░░░░░░░░░░████████ 42.9%    │
│                                             │
│  ┌─ Per-Method Results ─────────────────┐   │
│  │ Method │ Category │ Status │ Err Std │   │
│  │ TC     │ water    │  ✓    │ [0.21,…]│   │
│  │ EIVD   │ water    │  ✓    │ [0.19,…]│   │
│  │ TC     │ energy   │  ✓    │ [0.15,…]│   │
│  └────────────────────────────────────-─┘   │
└─────────────────────────────────────────────┘
```

#### Error Handling / 错误处理

Each method runs in an isolated `try/except`. Failed methods are reported in the HTML table with a ✗ mark and the error message, but do not abort the pipeline. The ELI ratio is computed from whichever methods succeeded.

每个方法在独立的 `try/except` 中运行。失败的方法在 HTML 表格中以 ✗ 标记和错误信息报告，但不会中断管线。ELI 比率从成功的方法中计算。

---

## 5. Data Fusion Subpackage / 数据融合子包

The `collocation/fusion/` subpackage provides production-grade data fusion algorithms that operate on the error estimates produced by the core methods.

`collocation/fusion/` 子包提供生产级数据融合算法，基于核心方法产生的误差估计运行。

| Module | Responsibility |
|--------|---------------|
| `weights.py` | IVW (inverse-variance), GLS/BLUE, constrained QP solvers |
| `covariance.py` | MSE estimation, covariance shrinkage (Ledoit-Wolf) |
| `fuse.py` | High-level orchestrator: `fuse_fields(data, method='gls')` |
| `constraints.py` | `SumToOneConstraint`, `BoundsConstraint`, `EnergyBalanceConstraint` |
| `uncertainty.py` | Bootstrap CI, effective sample size, variance propagation |
| `robust.py` | Huber-weighted estimates, MAD outlier detection |
| `localization.py` | Moving-window fusion, biome-partitioned analysis |
| `broadcast.py` | Shape-broadcasting utilities for spatial fields |

```python
from collocation.fusion import fuse_fields, solve_weights_gls

weights = solve_weights_gls(error_covariance)
fused   = fuse_fields(data, method='gls', error_cov=error_covariance)
```

---

## 6. Design Principles / 设计原则

### English

1. **One method = one module**: Every collocation algorithm lives in its own file. Easy to read, test, and replace independently.

2. **Functional core, OO shell**: Raw algorithms are plain functions (e.g., `tc(data)`). The OO layer (`TCEstimator`) adds convenience without hiding the math.

3. **Optional dependencies, zero hard failure**: PyMC3, xarray, and Plotly are optional. Availability flags (`BAYESIAN_AVAILABLE`, `PLOTLY_AVAILABLE`) let users branch gracefully.

4. **NaN-first design**: All v2.0 entry points strip NaN/Inf rows before computation. Earth-science data is rarely clean.

5. **Type hints everywhere in new code**: `np.ndarray`, `Optional`, `Dict`, `Tuple` annotations on all new modules aid IDE support and catch interface errors early.

6. **Parallel-safe**: `ELIPipeline` uses `ThreadPoolExecutor`; each method call is stateless and safe to run concurrently.

7. **Test-driven**: 48 new tests in `test_upgrade.py` cover normal, edge, xarray, NaN, and wrong-shape cases for every new class.

### 中文

1. **一方法一模块**：每种交叉定标算法独占一个文件，便于独立阅读、测试和替换。

2. **函数式核心，面向对象外壳**：原始算法是纯函数（如 `tc(data)`），OO 层（`TCEstimator`）增加便利性而不掩盖数学本质。

3. **可选依赖，零硬失败**：PyMC3、xarray 和 Plotly 均为可选。可用性标志（`BAYESIAN_AVAILABLE`、`PLOTLY_AVAILABLE`）让用户优雅地分支处理。

4. **NaN 优先设计**：所有 v2.0 入口点在计算前剔除 NaN/Inf 行。地球科学数据鲜少干净。

5. **新代码全面使用类型注解**：所有新模块使用 `np.ndarray`、`Optional`、`Dict`、`Tuple` 注解，助力 IDE 支持并早期捕获接口错误。

6. **线程安全**：`ELIPipeline` 使用 `ThreadPoolExecutor`；每个方法调用无状态，可安全并发。

7. **测试驱动**：`test_upgrade.py` 中 48 个新测试覆盖每个新类的正常、边界、xarray、NaN 和错误形状情况。

---

## 7. Dependency Map / 依赖关系图

```
collocation/
  __init__.py
      │
      ├── base.py ──────────────────────────── (no intra-package deps)
      │
      ├── estimators.py ──── imports: base, tc, eivd, ivd, ec
      │
      ├── consultant.py ─────────────────────── (no intra-package deps)
      │                        optional: xarray
      │
      ├── plotting.py ───────────────────────── (no intra-package deps)
      │                        optional: matplotlib, plotly
      │
      ├── eli_pipeline.py ─── imports: tc, eivd, ivs, btch_he2020
      │                        optional: xarray
      │
      ├── tc.py, eivd.py, ivd.py, ivs.py, ec.py
      │       └── numpy, scipy
      │
      ├── bayesian_tc.py, bayesian_tch.py
      │       └── optional: pymc3, theano
      │
      ├── covariance.py, fuse.py
      │       └── optional: xarray
      │
      └── fusion/ ─────── numpy, scipy
                  optional: xarray (localization)
```

**Minimum install / 最小安装** (numpy + scipy only):
Core methods (`tc`, `ivd`, `eivd`, `ivs`, `ec`, `btch_he2020`, `mtch`) + all v2.0 classes except the Plotly dashboard.

**Full install / 完整安装**:

```bash
pip install -e .
pip install xarray plotly pymc3==3.11.5 theano-pymc
```

---

## 8. Quick Reference / 快速参考

### Import Cheat-Sheet / 导入速查

```python
# ── Classical functions / 经典函数 ────────────────────────────────
from collocation import tc, eivd, ivd, ivs, ec
from collocation import tch                        # alias for tc

# ── Sklearn-style estimators / 估计器类 ───────────────────────────
from collocation import TCEstimator, EIVDEstimator, IVDEstimator, ECEstimator
from collocation.estimators import TC, EIVD, IVD, EC  # same, shorter import

# ── Smart recommender / 智能推荐 ──────────────────────────────────
from collocation import CollocationConsultant, ConsultationReport

# ── Plotting / 绘图 ───────────────────────────────────────────────
from collocation import InteractiveDashboard, plot_error_comparison
from collocation import plot_stability_heatmap, PLOTLY_AVAILABLE

# ── ELI pipeline / ELI 管线 ───────────────────────────────────────
from collocation import ELIPipeline, ELIResult

# ── Advanced methods / 高级方法 ───────────────────────────────────
from collocation import BTCH_He2020, btch_he2020
from collocation import mtch, MTCH
from collocation import TripleCollocation, ETCC, SpatialMerging

# ── Bayesian (optional) / 贝叶斯（可选）──────────────────────────
from collocation import BayesianTC, BayesianTCH, BAYESIAN_AVAILABLE

# ── Data fusion subpackage / 数据融合子包 ────────────────────────
from collocation.fusion import fuse_fields, solve_weights_gls, solve_weights_ivw

# ── Utilities / 工具 ──────────────────────────────────────────────
from collocation.utils import kge_objfun, mse_judge
from collocation.covariance import build_sigma_from_collocation
```

### Workflow Decision Tree / 工作流决策树

```
How many products? / 有几个产品？
    │
    ├─ 2 → ivd() or IVD().fit()
    │
    ├─ 3 → Run CollocationConsultant first
    │         │
    │         ├─ Correlated errors?     → eivd() / EIVD().fit()
    │         ├─ Time-varying variance? → BayesianTC (if PyMC3 available)
    │         ├─ Multiplicative / skewed positive data? → mtch()
    │         └─ Independent errors?   → tc() / TC().fit()
    │
    └─ 4 → ec() / EC().fit()

Need uncertainty estimates? / 需要不确定性估计？
    ├─ Bootstrap CI → ivs()
    ├─ Full posterior → BayesianTC / BayesianTCH (requires PyMC3)
    └─ Analytical weights → btch_he2020()

Need interactive exploration? / 需要交互式探索？
    └─ InteractiveDashboard(data, metrics_dict).save('report.html')

Running ELI analysis? / 运行 ELI 分析？
    └─ ELIPipeline(water, energy, veg).run().save('eli.html')
```

### v2.0 New Classes at a Glance / v2.0 新类速览

| Class / 类 | Location | One-liner |
|-----------|----------|-----------|
| `CollocationEstimator` | `base.py` | Abstract base; handles NaN, xarray, not-fitted guard |
| `TC` | `estimators.py` | `tc()` wrapped as `fit / metrics_` |
| `EIVD` | `estimators.py` | `eivd()` wrapped, adds `cross_corr`, `L` |
| `IVD` | `estimators.py` | `ivd()` wrapped, adds `weights` |
| `EC` | `estimators.py` | `ec()` wrapped, aggregates across combinations |
| `CollocationConsultant` | `consultant.py` | 5-test diagnostic + narrative recommendation |
| `ConsultationReport` | `consultant.py` | Dataclass holding `recommended`, `diagnostics`, `text` |
| `InteractiveDashboard` | `plotting.py` | 3-panel Plotly dashboard; `.show()` / `.save()` |
| `ELIPipeline` | `eli_pipeline.py` | Parallel multi-method ELI analysis |
| `ELIResult` | `eli_pipeline.py` | Holds `eli_ratio`, per-method results, HTML content |

---

*This document is maintained alongside the source code. When adding new modules, update the Package Layout (§2), add a subsection in §4, and extend the Quick Reference (§8).*

*本文档与源代码同步维护。添加新模块时，请更新§2的包目录结构，在§4中添加小节，并扩展§8的快速参考。*
