"""
Example: Multiplicative-error Three-Cornered Hat (MTCH)
========================================================

Demonstrates the MTCH method for estimating per-product RMSE under a
multiplicative error model.  MTCH is ideal for strictly positive geophysical
variables such as evapotranspiration (ET), precipitation, or soil moisture
fluxes, where relative (percentage) errors are more stationary than absolute
errors.

The example covers:
1. Generating synthetic multiplicative-error data with known ground truth
2. Running MTCH via the functional API (mtch)
3. Running MTCH via the class-based API (MTCH)
4. Comparing MTCH estimates against true RMSE
5. Comparing MTCH with the additive-error TC method
6. Plotting results

Reference:
    Xu, T., Guo, Z., Xia, Y., et al. (2019). Journal of Hydrology, 578, 124105.
    Ferreira, V.G., et al. (2016). J. Appl. Remote Sens, 10, 015015.

Author: Collocation-Analysis project
Date: 2025
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

from collocation import mtch, MTCH, tc

# Font configuration: choose installed fonts and ensure minus glyph support.
def _configure_plot_fonts() -> None:
    preferred_fonts = [
        "DejaVu Sans",
        "Arial",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    selected = [name for name in preferred_fonts if name in installed]

    if not selected:
        selected = ["DejaVu Sans"]

    existing = [name for name in mpl.rcParams.get("font.sans-serif", []) if name not in selected]
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = selected + existing
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["mathtext.fontset"] = "dejavusans"


_configure_plot_fonts()


# ============================================================================
# Synthetic data generation
# ============================================================================

def generate_multiplicative_data(
    n: int = 300,
    n_products: int = 3,
    true_mean: float = 5.0,
    true_std: float = 2.0,
    log_error_stds: tuple = (0.10, 0.15, 0.20),
    seed: int = 42,
) -> tuple:
    """
    Generate synthetic multiplicative-error data with known ground truth.

    Model:  Z_i = Theta * exp(e_i),  e_i ~ N(0, sigma_i^2)

    The true RMSE of product i (in original units) is approximately:
        RMSE_i ≈ sigma_i * mean(Theta)

    Parameters
    ----------
    n : int
        Number of time steps.
    n_products : int
        Number of products to simulate.
    true_mean : float
        Mean of the true signal (log-normal mean parameter mu).
    true_std : float
        Std dev used for the gamma distribution of the true signal.
    log_error_stds : tuple
        Per-product log-space error standard deviations (sigma_i).
    seed : int
        Random seed.

    Returns
    -------
    data : np.ndarray, shape (n, n_products)
    true_signal : np.ndarray, shape (n,)
    true_rmse : np.ndarray, shape (n_products,)
    log_error_stds : np.ndarray, shape (n_products,)
    """
    rng = np.random.default_rng(seed)

    # True signal: positive, log-normally distributed
    true_signal = rng.gamma(shape=(true_mean / true_std) ** 2,
                            scale=true_std ** 2 / true_mean, size=n)

    log_error_stds = np.asarray(log_error_stds[:n_products])

    # Simulate products: Z_i = Theta * exp(e_i)
    products = []
    for sigma_i in log_error_stds:
        e_i = rng.normal(0, sigma_i, size=n)
        z_i = true_signal * np.exp(e_i)
        products.append(z_i)

    data = np.column_stack(products)

    # True RMSE ≈ sigma_i * mean(Theta)
    true_rmse = log_error_stds * np.mean(true_signal)

    return data, true_signal, true_rmse, log_error_stds


# ============================================================================
# Demo 1: Functional API
# ============================================================================

def demo_functional_api():
    """Demonstrate the functional mtch() interface."""
    print("\n" + "=" * 70)
    print("DEMO 1: Functional API  –  mtch(data)")
    print("=" * 70)

    data, true_signal, true_rmse, log_stds = generate_multiplicative_data(
        n=300, log_error_stds=(0.10, 0.15, 0.20)
    )

    rmse_est = mtch(data)

    print(f"\n{'Product':<10} {'True RMSE':>12} {'MTCH RMSE':>12} {'Rel. Error':>12}")
    print("-" * 50)
    for i in range(data.shape[1]):
        rel_err = abs(rmse_est[i] - true_rmse[i]) / true_rmse[i]
        print(f"  {i:<8} {true_rmse[i]:>12.4f} {rmse_est[i]:>12.4f} {rel_err:>11.1%}")

    return data, true_rmse, rmse_est


# ============================================================================
# Demo 2: Class-based API
# ============================================================================

def demo_class_api():
    """Demonstrate the class-based MTCH interface."""
    print("\n" + "=" * 70)
    print("DEMO 2: Class-based API  –  MTCH(data).run()")
    print("=" * 70)

    data, true_signal, true_rmse, _ = generate_multiplicative_data(
        n=500, log_error_stds=(0.08, 0.12, 0.18, 0.25), n_products=4
    )

    model = MTCH(data)
    results = model.run()
    model.summary()

    print("True RMSE  :", true_rmse)
    print("MTCH RMSE  :", results["rmse"])

    return model


# ============================================================================
# Demo 3: MTCH vs TC comparison
# ============================================================================

def demo_mtch_vs_tc():
    """
    Compare MTCH and TC on multiplicative-error data.

    TC assumes additive errors; MTCH assumes multiplicative errors.
    This demo shows that MTCH gives better RMSE estimates when the true
    error model is multiplicative.
    """
    print("\n" + "=" * 70)
    print("DEMO 3: MTCH vs TC  (multiplicative-error scenario)")
    print("=" * 70)

    data, true_signal, true_rmse, _ = generate_multiplicative_data(
        n=400, log_error_stds=(0.10, 0.15, 0.20)
    )

    # MTCH estimates
    rmse_mtch = mtch(data)

    # TC estimates (error variances → RMSE)
    EeeT, SNR, rho2, fMSE = tc(data)
    rmse_tc = np.sqrt(np.diag(EeeT))

    print(f"\n{'Product':<10} {'True RMSE':>12} {'MTCH RMSE':>12} {'TC RMSE':>12}")
    print("-" * 50)
    for i in range(3):
        print(f"  {i:<8} {true_rmse[i]:>12.4f} {rmse_mtch[i]:>12.4f} {rmse_tc[i]:>12.4f}")

    return data, true_rmse, rmse_mtch, rmse_tc


# ============================================================================
# Demo 4: Sensitivity to sample size
# ============================================================================

def demo_sample_sensitivity():
    """
    Show how MTCH accuracy changes with sample size.
    """
    print("\n" + "=" * 70)
    print("DEMO 4: Sensitivity to sample size")
    print("=" * 70)

    sample_sizes = [50, 100, 200, 500, 1000]
    log_error_stds = np.array([0.10, 0.15, 0.20])

    results = {}
    for n in sample_sizes:
        data, _, true_rmse, _ = generate_multiplicative_data(
            n=n, log_error_stds=log_error_stds, seed=0
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rmse_est = mtch(data)
        rel_errors = np.abs(rmse_est - true_rmse) / true_rmse
        results[n] = rel_errors.mean()
        print(f"  n={n:5d}:  mean relative error = {rel_errors.mean():.2%}")

    return results


# ============================================================================
# Plotting
# ============================================================================

def make_plots(data, true_rmse, rmse_mtch, rmse_tc):
    """Generate comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("MTCH Method – Example Results", fontsize=13, fontweight="bold")

    # --- Left: RMSE bar comparison ---
    ax = axes[0]
    n_products = len(true_rmse)
    x = np.arange(n_products)
    width = 0.25

    ax.bar(x - width, true_rmse, width, label="True RMSE", color="#2196F3", alpha=0.85)
    ax.bar(x, rmse_mtch, width, label="MTCH estimate", color="#FF5722", alpha=0.85)
    ax.bar(x + width, rmse_tc, width, label="TC estimate", color="#4CAF50", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Product {i}" for i in range(n_products)])
    ax.set_ylabel("RMSE (original units)")
    ax.set_title("RMSE Comparison: True vs. Estimated")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # --- Right: Scatter of data products (first 2 columns) ---
    ax = axes[1]
    ax.scatter(data[:, 0], data[:, 1], s=8, alpha=0.4, color="#607D8B")
    lim_min = min(data[:, :2].min() * 0.9, 0.1)
    lim_max = data[:, :2].max() * 1.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=1.2, label="1:1 line")
    ax.set_xlabel("Product 0")
    ax.set_ylabel("Product 1")
    ax.set_title("Product 0 vs. Product 1 (log-scale scatter)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(script_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, "mtch_example.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {out_path}")

    return fig


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("MTCH – Multiplicative-error Three-Cornered Hat – Example")
    print("=" * 70)

    data, true_rmse, rmse_mtch = demo_functional_api()
    demo_class_api()
    data, true_rmse, rmse_mtch, rmse_tc = demo_mtch_vs_tc()
    demo_sample_sensitivity()

    fig = make_plots(data, true_rmse, rmse_mtch, rmse_tc)
    plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
