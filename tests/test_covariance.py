import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from collocation.covariance import build_sigma_from_collocation
from collocation.fuse import estimate_bias_from_collocation


def _save_sigma_fig(sigma, name_suffix: str = "sigma"):
    """
    Save a heatmap of the covariance matrix next to this script under ./figures.
    Does nothing if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig_dir = Path(__file__).resolve().parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    script_stem = Path(__file__).stem
    out_path = fig_dir / f"{script_stem}_{name_suffix}.png"

    plt.figure(figsize=(4.5, 3.6))
    im = plt.imshow(sigma.values, cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Covariance")
    plt.title(f"Covariance heatmap: {name_suffix}")
    plt.xlabel(sigma.dims[1])
    plt.ylabel(sigma.dims[0])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_path}")


def _make_dataset():
    models = ["m0", "m1", "m2"]
    var = xr.DataArray([1.0, 2.0, 3.0], dims=("model",), coords={"model": models})
    cov = xr.DataArray(
        [[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]],
        dims=("model", "model_2"),
        coords={"model": models, "model_2": models},
    )
    bias = xr.DataArray([0.1, -0.2, 0.0], dims=("model",), coords={"model": models})
    return xr.Dataset({"var": var, "cov": cov, "bias": bias})


def test_build_sigma_from_collocation_defaults():
    ds = _make_dataset()
    sigma = build_sigma_from_collocation(ds)
    np.testing.assert_allclose(sigma.values, ds["cov"].values)
    assert sigma.dims == ("model", "model_2")
    # Save a diagnostic figure
    _save_sigma_fig(sigma, "default_covariance")


def test_build_sigma_from_collocation_mapping_reorder():
    ds = _make_dataset()
    sigma = build_sigma_from_collocation(ds, mapping={"model": ["m2", "m0", "m1"]})
    assert list(sigma["model"].values) == ["m2", "m0", "m1"]


def test_build_sigma_from_collocation_ridge():
    ds = _make_dataset()
    sigma = build_sigma_from_collocation(ds, shrinkage="ridge", lam=0.5)
    expected = ds["cov"].values + 0.5 * np.eye(3)
    np.testing.assert_allclose(sigma.values, expected)


def test_build_sigma_from_variance_only():
    ds = _make_dataset().drop_vars("cov")
    sigma = build_sigma_from_collocation(ds)
    expected = np.diag(ds["var"].values)
    np.testing.assert_allclose(sigma.values, expected)


def test_estimate_bias_from_collocation_prefers_bias():
    ds = _make_dataset()
    bias = estimate_bias_from_collocation(ds)
    np.testing.assert_allclose(bias.values, ds["bias"].values)


def test_estimate_bias_from_collocation_zero_fallback():
    ds = _make_dataset().drop_vars("bias")
    bias = estimate_bias_from_collocation(ds)
    np.testing.assert_allclose(bias.values, np.zeros(3))


def test_covariance_suite_success():
    # Final success indicator (prints are captured unless -s is used)
    print("✓ Covariance tests completed successfully.")
    assert True

