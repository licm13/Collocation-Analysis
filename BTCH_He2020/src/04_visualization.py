
"""
生成图表（遵守约束：使用 matplotlib，单图单画布，不设置颜色）。
Generate figures (matplotlib only, single-plot per figure, no color settings).
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

INP = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "et_products_demo.npz")
RES = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables", "btch_results.json")

def main():
    data = np.load(INP)
    truth = data["truth"]           # (T,)
    products = data["products"]     # (N,T)

    with open(TAB, "r", encoding="utf-8") as f:
        res = json.load(f)

    # Recompute fused series for plotting
    # Uncorrelated
    w_unc = np.array(res["uncorrelated_TCH"]["weights"])
    fused_unc = w_unc @ products
    # Correlated
    w_cor = np.array(res["correlated_TCH_reg"]["weights"])
    fused_cor = w_cor @ products
    # EM
    fused_em = products.mean(axis=0)

    os.makedirs(RES, exist_ok=True)

    # 1) Time series
    plt.figure()
    plt.plot(truth, label="Truth")
    plt.plot(fused_em, label="EM")
    plt.plot(fused_unc, label="BTCH (Uncorr)")
    plt.plot(fused_cor, label="BTCH (Corr-Reg)")
    plt.legend()
    plt.title("Time Series (Demo)")
    plt.xlabel("Time (months)")
    plt.ylabel("ET (a.u.)")
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "timeseries_demo.png"), dpi=200)
    plt.close()

    # 2) Scatter truth vs fused (corr-reg)
    plt.figure()
    plt.scatter(truth, fused_cor, s=12)
    plt.plot([truth.min(), truth.max()], [truth.min(), truth.max()])
    plt.title("Truth vs BTCH (Corr-Reg)")
    plt.xlabel("Truth")
    plt.ylabel("Fused")
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "scatter_truth_btch_corr.png"), dpi=200)
    plt.close()

    # 3) Scatter truth vs EM
    plt.figure()
    plt.scatter(truth, fused_em, s=12)
    plt.plot([truth.min(), truth.max()], [truth.min(), truth.max()])
    plt.title("Truth vs EM")
    plt.xlabel("Truth")
    plt.ylabel("EM")
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "scatter_truth_em.png"), dpi=200)
    plt.close()

    print(f"Saved figures to {RES}")

if __name__ == "__main__":
    main()
