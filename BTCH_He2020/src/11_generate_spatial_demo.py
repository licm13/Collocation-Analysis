
"""
11_generate_spatial_demo.py
Create a small synthetic grid ET dataset with N products and T months.
"""
import numpy as np, os
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "grid_et_demo.npz")

def generate_spatial_demo(N=6, T=120, ny=20, nx=30, start="2000-01", seed=123):
    rng = np.random.default_rng(seed)
    dates = pd.period_range(start=start, periods=T, freq="M").astype(str).to_numpy()

    t = np.arange(T)
    # base truth with seasonality
    truth_t = 3.0 + 1.0*np.sin(2*np.pi*t/12.0) + 0.003*(t - T/2.0)  # (T,)
    # spatial pattern
    yy, xx = np.meshgrid(np.linspace(-1,1,ny), np.linspace(-1,1,nx), indexing="ij")
    spatial = 0.6*yy + 0.4*xx + 0.3*yy*xx  # (ny,nx)

    truth = np.zeros((T, ny, nx))
    for k in range(T):
        truth[k] = truth_t[k] + spatial

    # product errors (correlated among products; pixel-wise iid)
    sigmas = np.linspace(0.12, 0.35, N)
    base_corr = 0.3
    R = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            rho = (base_corr**abs(i-j))
            R[i,j] = rho * sigmas[i] * sigmas[j]
    L = np.linalg.cholesky(R + 1e-10*np.eye(N))

    products = np.zeros((N, T, ny, nx))
    for k in range(T):
        # sample product noise for each pixel
        noise = L @ rng.standard_normal(size=(N, ny*nx))
        noise = noise.reshape(N, ny, nx)
        products[:, k] = truth[k][None, ...] + noise

    return truth, products, dates

def main():
    truth, products, dates = generate_spatial_demo()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, truth=truth, products=products, dates=dates)
    print("Saved spatial demo to", OUT)

if __name__ == "__main__":
    main()
