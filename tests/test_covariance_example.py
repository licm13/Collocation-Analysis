import numpy as np
import os
import pytest
import sys

matplotlib = pytest.importorskip("matplotlib")
xr = pytest.importorskip("xarray")

from examples.covariance_method_comparison import run_covariance_comparison

# Ensure repository root is on sys.path so 'examples' and 'collocation' are importable.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_covariance_comparison_example(tmp_path):
    output_path = tmp_path / "covariance_methods.png"
    result = run_covariance_comparison(output_path=output_path)

    rmse = result["rmse"]

    # Full covariance should outperform diagonal in correlated regime.
    assert rmse["correlated"]["full"] < rmse["correlated"]["diag"]

    # Shrinkage stabilises the ill-conditioned sample.
    assert rmse["small_sample"]["shrunk"] < rmse["small_sample"]["full"]

    # Independent regime yields nearly identical performance for full and diagonal.
    np.testing.assert_allclose(
        rmse["independent"]["full"],
        rmse["independent"]["diag"],
        rtol=0,
        atol=0.02,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
