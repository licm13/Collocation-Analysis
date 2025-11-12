
"""
分析主流程 / Main analysis workflow

- 加载数据（示例为合成数据）
- 估计 TCH 协方差（无相关 + 正则相关两种）
- 计算 BTCH 权重与融合；与 EM 对比
- 输出评估指标与权重表
"""
import os, json
import numpy as np
from utils import corrcoef, rmse, bias
from btch import tch_uncorrelated_variances, tch_correlated_regularized, integrate_btch, integrate_em

INP = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "et_products_demo.npz")
OUT_JSON = os.path.join(os.path.dirname(__file__), "..", "results", "tables", "btch_results.json")

def main():
    data = np.load(INP)
    truth = data["truth"]           # (T,)
    products = data["products"]     # (N,T)
    N, T = products.shape

    # 1) Uncorrelated TCH (diagonal R)
    diag_vars = tch_uncorrelated_variances(products)
    R_unc = np.diag(diag_vars)
    fused_unc, w_unc, s2_unc = integrate_btch(products, R_unc)

    # 2) Regularized correlated TCH (approx.)
    R_cor = tch_correlated_regularized(products, alpha=1e-2, lr=5e-3, iters=600, seed=0)
    fused_cor, w_cor, s2_cor = integrate_btch(products, R_cor)

    # 3) Ensemble Mean
    fused_em = integrate_em(products)

    # Metrics
    results = {
        "uncorrelated_TCH": {
            "R": corrcoef(fused_unc, truth),
            "RMSE": rmse(fused_unc, truth),
            "Bias": bias(fused_unc, truth),
            "weights": w_unc.tolist(),
            "variances": s2_unc.tolist()
        },
        "correlated_TCH_reg": {
            "R": corrcoef(fused_cor, truth),
            "RMSE": rmse(fused_cor, truth),
            "Bias": bias(fused_cor, truth),
            "weights": w_cor.tolist(),
            "variances": s2_cor.tolist()
        },
        "EM": {
            "R": corrcoef(fused_em, truth),
            "RMSE": rmse(fused_em, truth),
            "Bias": bias(fused_em, truth)
        }
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("=== Summary (demo) ===")
    for k,v in results.items():
        print(k, "-> R={:.3f} RMSE={:.3f} Bias={:.3f}".format(v["R"], v["RMSE"], v["Bias"] if "Bias" in v else 0.0))

if __name__ == "__main__":
    main()
