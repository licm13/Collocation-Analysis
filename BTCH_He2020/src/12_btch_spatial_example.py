
"""
12_btch_spatial_example.py
Compute spatial BTCH (uncorrelated TCH) with static and monthly windowed weights.
Generates basic maps for EM vs BTCH performance.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from utils import rmse
from windowing import assign_groups
from btch import tch_uncorrelated_variances

INP = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "grid_et_demo.npz")
RES = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables", "spatial_btch_summary.json")

def solve_weights_uncorr(products_2d):
    """
    products_2d: (N, T)
    returns weights (N,)
    """
    diag_vars = tch_uncorrelated_variances(products_2d)
    # product-of-others weights
    sig2 = diag_vars
    log_s2 = np.log(sig2)
    total = log_s2.sum()
    logs = total - log_s2
    w = np.exp(logs - logs.max())
    w = w / w.sum()
    return w

def main():
    data = np.load(INP, allow_pickle=True)
    truth = data["truth"]        # (T, Y, X)
    products = data["products"]  # (N, T, Y, X)
    dates = data["dates"]
    N, T, Y, X = products.shape

    os.makedirs(RES, exist_ok=True)

    # 1) Static weights across all time (uncorrelated)
    fused_static = np.zeros((T, Y, X))
    weights_static = np.zeros((Y, X, N))

    for y in range(Y):
        for x in range(X):
            P = products[:, :, y, x]  # (N,T)
            w = solve_weights_uncorr(P)
            weights_static[y, x] = w
            fused_static[:, y, x] = w @ P

    # 2) Monthly windowed weights (uncorrelated) per pixel
    from pandas import to_datetime
    labels, order = assign_groups(dates, "monthly")
    fused_mon = np.zeros((T, Y, X))

    for y in range(Y):
        for x in range(X):
            P = products[:, :, y, x]
            # compute weights for each month group using samples across years
            w_by_g = {}
            for g in order:
                idx = np.where(labels==g)[0]
                if idx.size < N+2:
                    w_by_g[g] = weights_static[y, x]  # fallback
                else:
                    w_by_g[g] = solve_weights_uncorr(P[:, idx])

            # apply per time
            for t in range(T):
                g = labels[t]
                fused_mon[t, y, x] = w_by_g[g] @ P[:, t]

    # Baselines
    em = products.mean(axis=0)  # (T,Y,X)

    # Metrics (RMSE over time)
    rmse_em = np.sqrt(np.mean((em - truth)**2, axis=0))
    rmse_static = np.sqrt(np.mean((fused_static - truth)**2, axis=0))
    rmse_mon = np.sqrt(np.mean((fused_mon - truth)**2, axis=0))

    # Simple maps
    plt.figure()
    plt.imshow(truth.mean(axis=0))
    plt.title("Truth mean")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "grid_truth_mean.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.imshow(em.mean(axis=0))
    plt.title("EM mean")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "grid_em_mean.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.imshow(rmse_em)
    plt.title("RMSE(EM)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "grid_rmse_em.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.imshow(rmse_static)
    plt.title("RMSE(BTCH static weights)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "grid_rmse_btch_static.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.imshow(rmse_mon)
    plt.title("RMSE(BTCH monthly weights)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "grid_rmse_btch_monthly.png"), dpi=200)
    plt.close()

    # Summaries
    summary = {
        "RMSE_mean": {
            "EM": float(np.nanmean(rmse_em)),
            "BTCH_static": float(np.nanmean(rmse_static)),
            "BTCH_monthly": float(np.nanmean(rmse_mon))
        }
    }
    with open(TAB, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Spatial BTCH example completed. Figures saved to", RES)

if __name__ == "__main__":
    main()
