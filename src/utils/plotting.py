"""
设计说明 (中文, 详尽)
--------------------
本模块提供仓库内统一的绘图配置与图像保存接口, 目标:
1. 统一设置 Matplotlib 的中文字体优先列表并处理负号显示 (axes.unicode_minus).
2. 在可能的系统上自动选择可用的中文字体 (优先级由候选列表决定), 尽量避免中文乱码.
3. 提供 ensure_fig_dir 与 save_fig 等工具函数, 所有示例/模块应复用本模块保存图片到统一的 figures/ 目录, DPI >= 300.
4. 减少各处重复设置字体的行为, 便于全局统一维护与变更.

设计说明 (English, brief)
-------------------------
Provide a central plotting utility to set Chinese fonts and save figures consistently across the repository.
"""

from __future__ import annotations

import os
from typing import Optional
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import font_manager

# 候选中文字体列表 (按优先级)
_CANDIDATE_CHINESE_FONTS = [
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",  # 兼容 macOS / 部分环境
]

# 全局一次性配置: 设置候选字体与负号显示 (若需要, 后续可在程序启动时再次调用 ensure_chinese_font)
matplotlib.rcParams["axes.unicode_minus"] = False
# 预置一个可覆盖的 sans-serif 列表; ensure_chinese_font 会尝试选择系统可用字体并将其放在首位
matplotlib.rcParams["font.sans-serif"] = _CANDIDATE_CHINESE_FONTS.copy()

def _available_system_font_names() -> set[str]:
    """
    返回系统中已注册的字体名称集合 (字体的 family/name 字符串).
    """
    return {f.name for f in font_manager.fontManager.ttflist}

def ensure_chinese_font(preferred: Optional[list[str]] = None) -> Optional[str]:
    """
    尝试选择一个在系统中已安装的中文字体并设置为 matplotlib 的首选 sans-serif.

    参数:
    - preferred: 可选的优先字体列表 (按优先级). 如果未提供, 将使用模块内置的候选列表.
    
    返回值:
    - 成功设置的字体名称 (字符串), 如果未找到则返回 None (此时使用 matplotlib 的默认字体, 可能导致中文乱码).
    
    使用说明 (中文):
    - 在脚本/程序入口处调用 ensure_chinese_font(), 以便在绘图前完成字体检测与设置.
    - 如果运行环境 (如 CI 或 无头服务器) 没有中文字体, 函数会返回 None; 请在 README 中提供安装字体的说明.
    
    Usage (English, brief):
    - Call at program startup to prefer a system-installed Chinese font, returning the chosen font name.
    """
    candidates = preferred if preferred is not None else _CANDIDATE_CHINESE_FONTS
    installed = _available_system_font_names()
    for name in candidates:
        if name in installed:
            matplotlib.rcParams["font.sans-serif"] = [name] + [f for f in candidates if f != name]
            return name
    # 未找到候选字体: 不改变 rcParams 中的列表 (保持原先候选列表作为 fallback)
    return None

def ensure_fig_dir(path: str = "figures") -> str:
    """
    确保输出目录存在并返回路径 (可被 tests/ examples 复用).
    """
    os.makedirs(path, exist_ok=True)
    return path

def save_fig(fig: Figure, filename: str, folder: str = "figures", dpi: int = 300) -> str:
    """
    将 Matplotlib Figure 保存到指定目录, 默认 DPI = 300, bbox_inches='tight'.

    参数:
    - fig: matplotlib.figure.Figure 实例
    - filename: 文件名, 例如 'plot.png' (可以包含子目录, 如 'subdir/plot.png', 函数会创建子目录)
    - folder: 根目录 (默认 'figures')
    - dpi: 保存分辨率 (默认 300)

    返回:
    - 最终保存的文件路径 (字符串)
    """
    outdir = ensure_fig_dir(folder)
    # 支持 filename 带子目录的情形
    fullpath = os.path.join(outdir, filename)
    os.makedirs(os.path.dirname(fullpath), exist_ok=True)
    fig.savefig(fullpath, dpi=dpi, bbox_inches="tight")
    return fullpath

# 在模块导入时尝试一次自动选择可用中文字体 (不阻塞执行)
try:
    _detected = ensure_chinese_font()
    if _detected:
        # 仅作信息性打印 (库中避免打印太多, 使用 logging 更合适; 此处尽量不破坏上游程序输出)
        pass
except Exception:
    # 在极端环境 (无字体模块或权限等) 下, 忽略字体设置错误, 保持稳定性
    pass
