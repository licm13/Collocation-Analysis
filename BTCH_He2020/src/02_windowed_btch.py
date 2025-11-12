
"""
02_windowed_btch.py
Run time-window BTCH (monthly/seasonal/half-year) with correlated & uncorrelated modes.
"""
import os, json
import numpy as np
from utils import corrcoef, rmse, bias
from windowing import compute_windowed_btch
INP = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "et_products_demo.npz")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "tables", "windowed_btch.json")

def main():
    data = np.load(INP, allow_pickle=True)
    truth = data["truth"]
    products = data["products"]
    dates = data["dates"]

    out = {}
    for scheme in ["monthly", "seasonal", "halfyear"]:
        out[scheme] = {}
        for mode in ["uncorrelated","correlated"]:
            fused, w_by_g, s2_by_g, labels, order = compute_windowed_btch(products, dates, scheme, mode=mode)
            out[scheme][mode] = {
                "R": float(r := getattr(__import__('utils'), 'corrcoef')(fused, truth)),
                "RMSE": float(getattr(__import__('utils'), 'rmse')(fused, truth)),
                "Bias": float(getattr(__import__('utils'), 'bias')(fused, truth)),
                "groups": order,
                "weights_by_group": {g: w.tolist() for g, w in w_by_g.items()},
                "variances_by_group": {g: s2.tolist() for g, s2 in s2_by_g.items()}
            }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("Saved windowed BTCH results to", OUT)

if __name__ == "__main__":
    main()
