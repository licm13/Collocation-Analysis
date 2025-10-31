# ET产品数据集分析工具使用说明

## 🎉 v1.1 更新

- ✨ **新增可视化进度条** - 使用tqdm显示实时分析进度
- 📁 **优化输出路径** - 所有结果文件自动保存到脚本所在目录
- 🔄 **智能降级** - 未安装tqdm时自动使用简单进度显示

## 功能概述
这个工具可以自动遍历指定路径下的所有ET（蒸散发）产品NetCDF文件，提取并整理以下信息：

- ✅ 时间分辨率（daily, monthly, 8-day等）
- ✅ 空间分辨率（度和公里）
- ✅ 时间跨度（起止日期）
- ✅ 空间范围（经纬度范围）
- ✅ 数据类型（总ET、PET、分量等）
- ✅ 主要变量名称
- ✅ 文件统计信息

## 安装依赖

在运行脚本前，请确保安装了必要的Python包：

```bash
# 必需的包
pip install xarray netCDF4 pandas numpy openpyxl

# 推荐安装（显示漂亮的进度条）
pip install tqdm
```

或者使用conda：

```bash
conda install xarray netCDF4 pandas numpy openpyxl tqdm -c conda-forge
```

## 使用方法

### 1. 修改数据路径

打开 `analyze_et_products.py` 或 `quick_scan.py`，找到以下代码行：

```python
data_path = r"Z:\Evaporation_Flux"
```

将其修改为你的实际数据路径。

### 2. 运行脚本

```bash
python analyze_et_products.py
```

### 3. 查看结果

**所有输出文件会自动保存在脚本所在的目录**，包括：

1. **ET_products_summary_[时间戳].csv** - 数据集汇总表
   - 包含每个数据集的概览信息
   - 易于快速了解所有数据集

2. **ET_products_detail_[时间戳].csv** - 详细信息表
   - 包含每个文件的详细元数据
   - 用于深入分析

3. **ET_products_report_[时间戳].xlsx** - Excel综合报告
   - 包含以上两个表格的整合版本
   - 方便在Excel中查看和筛选

## 进度显示

### 安装了tqdm（推荐）：
```
分析进度: 45%|████████████▌             | 123/275 [02:15<02:48, 0.90文件/s] 当前: ERA5_ET_2020.nc
```

### 未安装tqdm（简单模式）：
```
[123/275] (45%) 正在分析: ERA5_ET_2020.nc
```

## 输出示例

### 汇总表字段说明

| 字段 | 说明 |
|------|------|
| 数据集名称 | 文件夹名称（如 ERA5L, GLEAM等） |
| 文件数量 | 该数据集包含的NetCDF文件数 |
| ET类型 | 总ET / PET / 包含分量 等 |
| 时间分辨率 | daily / monthly / 8-day 等 |
| 空间分辨率 | 度数和公里数 |
| 空间范围 | 经纬度范围 |
| 主要变量 | 数据集中的变量名 |
| 时间跨度 | 数据的起止日期 |

## 常见问题

### Q1: 路径不存在错误
**A:** 确保数据路径正确，Windows路径需要使用原始字符串（r"路径"）或双反斜杠

### Q2: 读取NetCDF文件失败
**A:** 某些文件可能格式特殊，脚本会记录错误并继续处理其他文件

### Q3: 时间坐标识别失败
**A:** 脚本会尝试多种常见的时间坐标名称（time, t, Time等），如果都失败会标记为"N/A"

### Q4: 处理速度慢
**A:** 对于大量文件，处理可能需要一些时间。脚本会显示进度信息。

## 数据集类型识别规则

脚本会根据以下规则自动识别ET数据类型：

- **总ET**: 包含 ET, evapotranspiration, LE 等变量
- **PET**: 包含 PET, potential_evapotranspiration 等变量
- **蒸腾**: 包含 transpiration, ET_t 等变量
- **土壤蒸发**: 包含 soil_evaporation, Es 等变量
- **冠层截留**: 包含 interception, Ei 等变量

## 自定义修改

如果需要添加新的识别规则或修改分析逻辑，可以编辑：

- `identify_et_type()` - 修改ET类型识别规则
- `get_temporal_resolution()` - 修改时间分辨率判断
- `analyze_netcdf_file()` - 修改文件分析逻辑

## 技术支持

遇到问题可以检查：
1. Python版本是否 >= 3.7
2. 依赖包是否正确安装
3. NetCDF文件是否损坏
4. 是否有文件读取权限

## 更新日志

**v1.0** (2025-10-31)
- 初始版本
- 支持自动识别时空分辨率
- 支持ET类型分类
- 生成CSV和Excel报告
