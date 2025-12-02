# Collocation-Analysis Notebook Conversion Brief

本文档汇总了 `Collocation-Analysis` 代码库中与示例脚本和包导入相关的关键信息，并提供一份可直接发送给 Claude/Codex 的 Prompt，用于将现有的 Python 示例脚本转换为面向大一新生的 Jupyter Notebook 教程。

## 代码库要点摘要
- **项目目标**：提供多种基于配准（collocation）的误差估计与数据融合方法，涵盖经典 TC/IVD/EC、扩展 ETCC，以及贝叶斯版本等。【F:README.md†L1-L120】
- **核心包入口**：`collocation/__init__.py` 暴露了 IVD、TC、EIVD、EC、ETCC、简单平均以及贝叶斯接口，并通过 `__all__` 统一出口；其中还包含可选的 ELI 处理器和 BTCH 等接口。【F:collocation/__init__.py†L1-L86】【F:collocation/__init__.py†L88-L140】
- **示例脚本风格**：
  - `examples/simple_average_example.py`：全中文注释，流程化打印步骤，生成正弦叠加趋势的合成数据，演示简单/加权平均、指标计算与可视化（箱线图 + 时序）。【F:examples/simple_average_example.py†L1-L122】【F:examples/simple_average_example.py†L124-L214】
  - `examples/example_all_methods.py`：使用 `sys.path.insert` 导入包，演示 IVD/IVS/TC/EIVD/EC 的调用、比较与合成结果绘图；包含综合可视化保存逻辑。【F:examples/example_all_methods.py†L1-L120】【F:examples/example_all_methods.py†L122-L227】
  - `examples/advanced_demo.py`：以线性回归拟合为例做参数网格实验，使用 Pandas 汇总结果，批量生成图像并保存，展示了更“科研化”的输出路径与中文标题处理。【F:examples/advanced_demo.py†L1-L101】

## Notebook 编写建议
1. **结构化分段**：按“环境配置 → 数据生成 → 核心算法调用 → 结果评估 → 可视化 → 思考题”拆分单元格，避免将脚本逻辑塞进一个 Cell。
2. **路径与导入**：首个代码单元建议使用 `sys.path.append('..')`（假设 Notebook 存放在 `examples/`），避免脚本式的 `sys.path.insert` 混入后续演示。
3. **教学化增强**：
   - 在介绍 TC/IVD/平均等方法时，插入 LaTeX 公式（例如平均公式、TC 的误差方差推导）帮助初学者建立“代码 ↔ 数学”映射。
   - 每步前加中文 Markdown 说明，并在数据生成后展示 `head` 或前几行切片，让学生直观看到样本结构。
   - 在“实验”部分添加交互式参数（样本量、噪声水平）的小练习或思考题。
4. **可视化与输出**：移除 `plt.show()` 依赖，保留 inline 输出；保留现有示例中的文件保存逻辑作为可选步骤，提示学生在 Notebook 环境下路径相对位置。

## 可直接使用的 Prompt（复制给 Claude/Codex）
````markdown
# Role
你是一名精通 Python 数据科学且善于教学的助教，面向刚入学的大一学生。他们能读懂基础 Python，但对统计学和遥感中的配准（collocation）概念比较陌生。

# Context
我有一个名为 `Collocation-Analysis` 的 Python 代码库，用于遥感/地球物理数据的误差分析与融合。`examples/` 目录下有多个脚本（如 `simple_average_example.py`、`example_all_methods.py`、`advanced_demo.py`），需要转换为循序渐进的 Jupyter Notebook 教程，帮助初学者理解为什么“平均能降噪”、IVD/TC/EC 的差异，以及如何做参数实验。

# Task
根据我提供的脚本内容，将其重组为 Notebook：拆分为多个代码单元与中文 Markdown 讲解，保留核心逻辑并加入教学化可视化。

# Requirements
1) **结构分段**：按“导入与路径设置 → 数据生成 → 核心算法调用 → 指标计算 → 绘图 → 思考题”拆成多个 Cell。
2) **路径处理**：首个 Cell 使用 `sys.path.append('..')`（Notebook 位于 `examples/`）确保 `import collocation` 成功。
3) **数学讲解**：
   - 简单平均：说明等权与加权平均公式，以及噪声方差降低的直观解释。
   - TC/IVD/EC：用 LaTeX 写出关键假设和误差方差/相关系数公式，指出独立误差 vs 相关误差的区别。
4) **数据展示**：在生成数据后展示前几行/片段（如 `print(truth[:5])` 或 `df.head()`）。
5) **可视化**：使用 `%matplotlib inline`；移除 `plt.show()` 依赖。保留图像保存为可选步骤，并说明保存路径（相对 `examples/figures/...`）。
6) **互动练习**：在末尾添加 1-2 个“动手尝试”或“思考题”（例如调整样本量、噪声水平，观察 RMSE/相关系数变化）。
7) **语言**：所有 Markdown 与新增注释使用中文；代码保持原有变量含义。

# Output
以 Markdown 形式输出 Notebook 草案，包含清晰的章节标题、中文讲解与分段代码块，便于直接粘贴到 `.ipynb` 中。
````

将上述 Prompt 提供给大模型，即可获得适合初学者的 Notebook 版本。先从 `simple_average_example.py` 起步，验证风格与结构，再迁移到 `example_all_methods.py` 与 `advanced_demo.py`。
