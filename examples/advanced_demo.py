"""
Advanced demo（复杂示例，中文为主）
- 功能：
  * 使用合成数据做参数网格（不同噪声/样本量/系数）实验
  * 对比不同配置下的拟合误差（RMSE），输出对比图、误差分布图和结果表格（CSV）
  * 固定随机种子保证可复现
- 使用：python examples/advanced_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

from src.utils.plotting import save_fig, ensure_chinese_font

RANDOM_SEED = 2025
np.random.seed(RANDOM_SEED)

def _generate_data(n: int, coef: float, noise_std: float) -> tuple[np.ndarray, np.ndarray]:
    """
    生成简单的线性数据：y = coef * x + intercept + noise
    """
    x = np.linspace(0, 10, n)
    intercept = 1.0
    noise = np.random.normal(scale=noise_std, size=x.shape)
    y = coef * x + intercept + noise
    return x, y

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def main():
    font_name = ensure_chinese_font()
    if font_name:
        print(f"使用中文字体: {font_name}")

    # 获取脚本名称（不含扩展名）作为输出文件夹名
    script_name = Path(__file__).stem  # 'advanced_demo'
    script_dir = Path(__file__).parent  # examples 目录
    output_folder = script_dir / "figures" / script_name  # examples/figures/advanced_demo

    # 参数网格：不同噪声、不同行数、不同系数
    grid = {
        "n": [50, 200],
        "coef": [1.5, 2.5],
        "noise_std": [1.0, 4.0],
    }
    experiments = list(itertools.product(grid["n"], grid["coef"], grid["noise_std"]))

    results: list[Dict] = []

    # 为每个实验生成数据并评估
    for idx, (n, coef, noise_std) in enumerate(experiments):
        x, y = _generate_data(n=n, coef=coef, noise_std=noise_std)
        # 拟合
        coeffs = np.polyfit(x, y, deg=1)
        y_fit = np.polyval(coeffs, x)
        rmse_val = _rmse(y, y_fit)
        results.append(
            {
                "experiment_id": idx,
                "n": n,
                "coef_true": coef,
                "noise_std": noise_std,
                "rmse": float(rmse_val),
                "fit_slope": float(coeffs[0]),
                "fit_intercept": float(coeffs[1]),
            }
        )

        # 每个配置绘制误差分布图并保存
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(y - y_fit, bins=30, color="C2", alpha=0.7)
        ax.set_title(f"误差分布（n={n}, coef={coef}, noise={noise_std}）")
        ax.set_xlabel("误差 (y - y_pred)")
        ax.set_ylabel("频数")
        # 保存到 examples/figures/advanced_demo/ 下
        fig_path = output_folder / f"error_hist_{idx}.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 汇总结果并保存 CSV 与对比图
    df = pd.DataFrame(results)
    out_csv = output_folder / f"{script_name}_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"实验结果已保存到: {out_csv}")

    # 对比不同配置下的 RMSE（热力图 / 折线）
    pivot = df.pivot_table(index="n", columns="noise_std", values="rmse", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6, 4))
    pivot.plot(marker="o", ax=ax)
    ax.set_title("不同样本量与噪声下的 RMSE 对比")
    ax.set_xlabel("样本量 n")
    ax.set_ylabel("RMSE")
    ax.legend(title="noise_std")
    # 保存到 examples/figures/advanced_demo/ 下
    fig_path = output_folder / "rmse_comparison.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"高级示例运行完成，所有图与表格保存在 {output_folder}/ 目录。")


if __name__ == "__main__":
    main()
