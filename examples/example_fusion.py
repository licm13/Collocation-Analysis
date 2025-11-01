"""
Comprehensive Fusion Examples
==============================

Demonstrates all fusion workflows:
1. Simple IVW fusion
2. GLS fusion with full covariance
3. Constrained QP fusion
4. Localized fusion
5. Robust fusion with outliers
6. Publication-quality diagnostic plots
"""

# Make repository root importable so this example can be run directly
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
# Path already imported above

# Import fusion modules
from collocation.fusion import (
    fuse_fields,
    solve_weights_ivw,
    solve_weights_gls,
    estimate_mse,
    build_sigma,
)


def generate_synthetic_et_data(
    n_time=365,
    n_lat=50,
    n_lon=50,
    n_models=3,
    spatial_heterogeneity=True,
):
    """
    Generate synthetic evapotranspiration data.

    Creates realistic multi-model ET fields with:
    - Seasonal cycles
    - Spatial patterns
    - Model-specific biases and noise levels
    - Optional outliers

    Parameters
    ----------
    n_time : int
        Number of timesteps (days).
    n_lat, n_lon : int
        Spatial dimensions.
    n_models : int
        Number of ET products.
    spatial_heterogeneity : bool
        Add spatially varying error.

    Returns
    -------
    X : xr.DataArray
        Multi-model ET predictions.
    X_truth : xr.DataArray
        True ET field.
    """
    print("Generating synthetic ET data...")

    # Time axis (days)
    time = np.arange(n_time)

    # Spatial coordinates
    lat = np.linspace(-45, 45, n_lat)
    lon = np.linspace(-120, 120, n_lon)

    # Create mesh
    time_mesh, lat_mesh, lon_mesh = np.meshgrid(time, lat, lon, indexing="ij")

    # True ET: seasonal cycle + spatial pattern
    # Base ET increases with abs(latitude) (more temperate)
    lat_effect = 1.0 + 2.0 * np.abs(lat_mesh) / 45.0
    # Seasonal cycle (max in summer day 180, min in winter day 0/365)
    seasonal = 1.0 + 0.5 * np.sin(2 * np.pi * (time_mesh - 90) / 365)
    # Combine
    ET_true = 2.0 + lat_effect * seasonal

    # Model-specific biases and noise levels
    model_names = [f"Model_{chr(65+i)}" for i in range(n_models)]
    biases = np.array([0.2, -0.1, 0.3])[:n_models]
    noise_base = np.array([0.5, 0.3, 0.8])[:n_models]

    # Generate model predictions
    models = np.zeros((n_time, n_lat, n_lon, n_models))

    for i in range(n_models):
        # Add bias
        bias = biases[i]

        # Noise level
        if spatial_heterogeneity:
            # Noise increases with latitude
            noise_spatial = noise_base[i] * (1.0 + np.abs(lat_mesh) / 90.0)
        else:
            noise_spatial = noise_base[i]

        # Generate noisy observations
        noise = np.random.randn(n_time, n_lat, n_lon) * noise_spatial
        models[:, :, :, i] = ET_true * (1 + bias) + noise

    # Wrap in xarray
    X = xr.DataArray(
        models,
        dims=["time", "lat", "lon", "model"],
        coords={
            "time": time,
            "lat": lat,
            "lon": lon,
            "model": model_names,
        },
        attrs={
            "long_name": "Evapotranspiration",
            "units": "mm/day",
        },
    )

    X_truth = xr.DataArray(
        ET_true,
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={
            "long_name": "True Evapotranspiration",
            "units": "mm/day",
        },
    )

    return X, X_truth


def example_1_simple_ivw():
    """Example 1: Simple inverse variance weighting."""
    print("\n" + "=" * 60)
    print("Example 1: Simple Inverse Variance Weighting (IVW)")
    print("=" * 60)

    # Generate data
    X, X_truth = generate_synthetic_et_data(
        n_time=365,
        n_lat=20,
        n_lon=20,
        n_models=3,
        spatial_heterogeneity=False,
    )

    # Use truth as reference for MSE estimation
    # (in practice, would use independent observations)
    result = fuse_fields(X, X_ref=X_truth, mode="ivw", return_weights=True)

    # Print results
    print("\nFusion weights:")
    for model in result["weights"].coords["model"].values:
        w = result["weights"].sel(model=model)
        # If weight is scalar, print directly; if spatial/time-varying, print a mean summary
        try:
            size = int(w.size)
        except Exception:
            size = None

        if size == 1:
            weight = float(w.values)
            print(f"  {model}: {weight:.3f}")
        else:
            mean_val = float(w.mean().values)
            other_dims = [d for d in w.dims if d != "model"]
            dim_note = ",".join(other_dims) if other_dims else "multi"
            print(f"  {model}: {mean_val:.3f} (mean over {dim_note})")

    # Compute RMSE
    rmse_fused = np.sqrt(np.mean((result["fused"] - X_truth) ** 2))
    print(f"\nFused RMSE: {rmse_fused:.4f} mm/day")

    # Individual model RMSEs
    print("\nIndividual model RMSEs:")
    for model in X.coords["model"].values:
        rmse = np.sqrt(np.mean((X.sel(model=model) - X_truth) ** 2))
        print(f"  {model}: {rmse:.4f} mm/day")

    return result, X, X_truth


def example_2_gls_fusion():
    """Example 2: GLS fusion with full error covariance."""
    print("\n" + "=" * 60)
    print("Example 2: GLS Fusion with Error Covariance")
    print("=" * 60)

    # Generate data with correlated errors
    X, X_truth = generate_synthetic_et_data(
        n_time=365,
        n_lat=20,
        n_lon=20,
        n_models=3,
    )

    # Fuse with GLS (estimates full covariance)
    result = fuse_fields(
        X,
        X_ref=X_truth,
        mode="gls",
        shrinkage="lw",
        return_weights=True,
        return_var=True,
    )

    print("\nGLS Fusion weights:")
    for model in result["weights"].coords["model"].values:
        w = result["weights"].sel(model=model)
        try:
            size = int(w.size)
        except Exception:
            size = None

        if size == 1:
            weight = float(w.values)
            print(f"  {model}: {weight:.3f}")
        else:
            mean_val = float(w.mean().values)
            other_dims = [d for d in w.dims if d != "model"]
            dim_note = ",".join(other_dims) if other_dims else "multi"
            print(f"  {model}: {mean_val:.3f} (mean over {dim_note})")

    # Variance reduction
    print("\nUncertainty:")
    fused_std = float(result["std"].mean())
    print(f"  Mean fused std: {fused_std:.4f} mm/day")

    # Diagnostics
    if "diagnostics" in result:
        diag = result["diagnostics"]
        print(f"\nDiagnostics:")
        def _fmt_stat(x, fmt="{:.2f}"):
            try:
                size = int(x.size)
            except Exception:
                size = None

            if size == 1:
                return fmt.format(float(x))
            else:
                # Return mean over whatever dimensions are present
                try:
                    mean_val = float(x.mean())
                except Exception:
                    mean_val = float(np.array(x).mean())
                dims = getattr(x, "dims", None)
                dim_note = ",".join(dims) if dims else "multi"
                return f"{fmt.format(mean_val)} (mean over {dim_note})"

        print(f"  Effective N: {_fmt_stat(diag['effective_n'], '{:.2f}')}")
        print(f"  Weight entropy: {_fmt_stat(diag['entropy'], '{:.3f}')}")
        print(f"  Concentration (HHI): {_fmt_stat(diag['concentration'], '{:.3f}')}")

    return result, X, X_truth


def example_3_constrained_qp():
    """Example 3: Constrained QP fusion."""
    print("\n" + "=" * 60)
    print("Example 3: Constrained QP Fusion")
    print("=" * 60)

    # This example would use quadprog for full QP solving
    # For now, demonstrate constraint setup
    print("NOTE: Full QP solver requires 'quadprog' package")
    print("Install with: pip install quadprog")
    print("\nExample constraints:")
    print("  - Sum to one: Σw = 1")
    print("  - Bounds: 0.1 ≤ w ≤ 0.9")
    print("  - ET constraint: 0 ≤ ET ≤ PET")

    # Generate data
    X, X_truth = generate_synthetic_et_data(n_time=100, n_lat=10, n_lon=10)

    # Define constraints
    constraints = {
        "sum_to_one": True,
        "bounds": (0.1, 0.9),  # Prevent extreme weights
    }

    # Note: QP solver needs quadprog installed
    # Fallback to GLS with bounds applied post-hoc
    print("\nUsing GLS with post-hoc bounds (approximation)...")
    result = fuse_fields(
        X,
        X_ref=X_truth,
        mode="gls",
        return_weights=True,
    )

    # Apply bounds manually
    weights = result["weights"].values
    weights_clipped = np.clip(weights, 0.1, 0.9)
    weights_clipped = weights_clipped / weights_clipped.sum()

    print("\nWeights (with bounds):")
    for i, model in enumerate(result["weights"].coords["model"].values):
        wval = weights_clipped[i]
        try:
            # scalar-like
            if np.asarray(wval).size == 1:
                print(f"  {model}: {float(wval):.3f}")
            else:
                mean_w = float(np.nanmean(wval))
                print(f"  {model}: {mean_w:.3f} (mean over spatial dims)")
        except Exception:
            print(f"  {model}: {wval}")

    return result, X, X_truth


def example_4_robust_fusion():
    """Example 4: Robust fusion with outliers."""
    print("\n" + "=" * 60)
    print("Example 4: Robust Fusion with Outliers")
    print("=" * 60)

    # Generate data
    X, X_truth = generate_synthetic_et_data(n_time=365, n_lat=20, n_lon=20)

    # Add random outliers to one model
    X_with_outliers = X.copy()
    outlier_mask = np.random.rand(*X.shape) < 0.05  # 5% outliers
    X_with_outliers = X_with_outliers.where(~outlier_mask, X + 10.0)

    # Standard fusion
    result_standard = fuse_fields(
        X_with_outliers,
        X_ref=X_truth,
        mode="ivw",
        robust=False,
    )

    # Robust fusion
    result_robust = fuse_fields(
        X_with_outliers,
        X_ref=X_truth,
        mode="ivw",
        robust=True,
    )

    # Compare RMSE
    rmse_standard = np.sqrt(np.mean((result_standard["fused"] - X_truth) ** 2))
    rmse_robust = np.sqrt(np.mean((result_robust["fused"] - X_truth) ** 2))

    print(f"\nStandard fusion RMSE: {rmse_standard:.4f} mm/day")
    print(f"Robust fusion RMSE: {rmse_robust:.4f} mm/day")
    print(f"Improvement: {(rmse_standard - rmse_robust) / rmse_standard * 100:.1f}%")

    return result_robust, X_with_outliers, X_truth


def create_diagnostic_plots(result, X, X_truth, output_dir="fusion_plots"):
    """Create publication-quality diagnostic plots."""
    print(f"\nCreating diagnostic plots in {output_dir}/...")
    Path(output_dir).mkdir(exist_ok=True)

    # Set journal-quality style
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "font.size": 9,
        "font.family": "serif",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
    })

    # Plot 1: Weight maps
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))
    for i, model in enumerate(result["weights"].coords["model"].values):
        w = result["weights"].sel(model=model)
        try:
            size = int(w.size)
        except Exception:
            size = None

        if "lat" in w.dims and "lon" in w.dims:
            # Spatial weights: reduce all non-spatial dims to produce a 2D map
            reduce_dims = [d for d in w.dims if d not in ("lat", "lon")]
            if reduce_dims:
                img = w.mean(dim=reduce_dims)
            else:
                img = w
            im = axes[i].imshow(img, cmap="viridis", vmin=0, vmax=1)
            axes[i].set_title(f"{model} weights")
        elif size == 1:
            # Scalar weight
            axes[i].bar([0], [float(w.values)], color="steelblue")
            axes[i].set_ylim(0, 1)
            axes[i].set_title(f"{model}")
        else:
            # Multi-dimensional but no lat dim: show mean across remaining dims
            mean_val = float(w.mean().values)
            other_dims = [d for d in w.dims if d != "model"]
            dim_note = ",".join(other_dims) if other_dims else "multi"
            axes[i].bar([0], [mean_val], color="steelblue")
            axes[i].set_ylim(0, 1)
            axes[i].set_title(f"{model} (mean over {dim_note})")
        axes[i].set_ylabel("Weight")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fusion_weights.png", dpi=600, bbox_inches="tight")
    plt.close()

    # Plot 2: Fused vs truth
    fig, ax = plt.subplots(figsize=(4, 4))
    fused_mean = result["fused"].mean(dim=["lat", "lon"])
    truth_mean = X_truth.mean(dim=["lat", "lon"])

    ax.scatter(truth_mean, fused_mean, alpha=0.5, s=10, color="steelblue")
    ax.plot([truth_mean.min(), truth_mean.max()],
            [truth_mean.min(), truth_mean.max()],
            "k--", lw=1, label="1:1 line")
    ax.set_xlabel("True ET (mm/day)")
    ax.set_ylabel("Fused ET (mm/day)")
    ax.set_title("Fused vs True ET")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fused_vs_truth.png", dpi=600, bbox_inches="tight")
    plt.close()

    # Plot 3: Time series comparison
    fig, ax = plt.subplots(figsize=(6, 3))

    # Plot individual models
    for model in X.coords["model"].values:
        model_mean = X.sel(model=model).mean(dim=["lat", "lon"])
        ax.plot(model_mean, alpha=0.5, lw=0.8, label=model)

    # Plot fused
    ax.plot(fused_mean, "k-", lw=1.5, label="Fused")

    # Plot truth
    ax.plot(truth_mean, "r--", lw=1, label="Truth")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("ET (mm/day)")
    ax.set_title("Time Series Comparison")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/timeseries_comparison.png", dpi=600, bbox_inches="tight")
    plt.close()

    print(f"Plots saved to {output_dir}/")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE DATA FUSION EXAMPLES")
    print("=" * 60)

    np.random.seed(42)  # Reproducibility

    # Run examples
    result1, X1, truth1 = example_1_simple_ivw()
    result2, X2, truth2 = example_2_gls_fusion()
    result3, X3, truth3 = example_3_constrained_qp()
    result4, X4, truth4 = example_4_robust_fusion()

    # Create plots for Example 2 (most complete)
    create_diagnostic_plots(result2, X2, truth2)

    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
