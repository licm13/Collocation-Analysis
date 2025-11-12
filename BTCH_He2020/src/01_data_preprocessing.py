
"""
数据预处理 / Data preprocessing (with monthly dates)

- 生成合成 ET 真值 + 多产品观测
- 含日期序列（按月）
"""
import numpy as np, os
from datetime import datetime
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "et_products_demo.npz")

def generate_synthetic_et(N=10, T=120, start="2000-01", seed=42):
    """
    生成合成“真值 ET”与多产品观测，并带有日期序列。
    Returns:
        truth (T,), products (N,T), dates (T,) as monthly strings 'YYYY-MM'
    """
    rng = np.random.default_rng(seed)
    dates = pd.period_range(start=start, periods=T, freq="M").astype(str).to_numpy()

    t = np.arange(T)
    truth = 3.0 + 1.2*np.sin(2*np.pi*t/12.0) + 0.005*(t - T/2.0)  # arbitrary units

    # Construct correlated errors with decreasing correlation by index distance
    sigmas = np.linspace(0.15, 0.45, N)  # std for each product
    base_corr = 0.4
    R = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            rho = (base_corr**abs(i-j))
            R[i,j] = rho * sigmas[i] * sigmas[j]

    # Sample T iid columns from N(0, R)
    L = np.linalg.cholesky(R + 1e-12*np.eye(N))
    eps = L @ rng.standard_normal(size=(N, T))

    products = truth[None, :] + eps
    return truth, products, dates

def main():
    truth, products, dates = generate_synthetic_et()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, truth=truth, products=products, dates=dates)
    print(f"Saved synthetic data with dates to {OUT}")

if __name__ == "__main__":
    main()
