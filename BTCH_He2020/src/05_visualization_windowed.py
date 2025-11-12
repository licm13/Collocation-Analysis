
"""
05_visualization_windowed.py
Plots for time-window BTCH results.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

INP = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "et_products_demo.npz")
TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables", "windowed_btch.json")
RES = os.path.join(os.path.dirname(__file__), "..", "results", "figures")

def main():
    data = np.load(INP, allow_pickle=True)
    truth = data["truth"]
    products = data["products"]
    dates = data["dates"]

    with open(TAB, "r", encoding="utf-8") as f:
        res = json.load(f)

    os.makedirs(RES, exist_ok=True)

    # For each scheme, build fused series for correlated/uncorrelated and compare with EM
    em = products.mean(axis=0)

    for scheme in ["monthly","seasonal","halfyear"]:
        # Recompute fused series from saved weights (for transparency)
        for mode in ["uncorrelated","correlated"]:
            info = res[scheme][mode]
            groups = info["groups"]
            # Build label mapping from dates
            import pandas as pd
            months = pd.to_datetime(dates).month
            if scheme=="monthly":
                labels = np.array([f"M{m:02d}" for m in months])
            elif scheme=="seasonal":
                def season(m):
                    if m in [12,1,2]: return "DJF"
                    if m in [3,4,5]: return "MAM"
                    if m in [6,7,8]: return "JJA"
                    return "SON"
                labels = np.array([season(m) for m in months])
            else:
                labels = np.array(["H1" if m<=6 else "H2" for m in months])

            weights_by_group = {g: np.array(w) for g, w in info["weights_by_group"].items()}
            fused = np.zeros_like(em)
            for t in range(len(em)):
                g = labels[t]
                fused[t] = weights_by_group[g] @ products[:, t]

            # Plot time series
            plt.figure()
            plt.plot(truth, label="Truth")
            plt.plot(em, label="EM")
            plt.plot(fused, label=f"BTCH-{mode} ({scheme})")
            plt.legend()
            plt.title(f"Windowed BTCH ({scheme}, {mode})")
            plt.xlabel("Time (months)")
            plt.ylabel("ET (a.u.)")
            plt.tight_layout()
            plt.savefig(os.path.join(RES, f"windowed_{scheme}_{mode}_timeseries.png"), dpi=200)
            plt.close()

    print("Saved windowed BTCH figures to", RES)

if __name__ == "__main__":
    main()
