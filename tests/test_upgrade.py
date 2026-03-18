"""
Tests for v2.0 upgrade modules:
  - collocation.base       (CollocationEstimator)
  - collocation.estimators (TC, EIVD, IVD, EC)
  - collocation.consultant (CollocationConsultant)
  - collocation.plotting   (plot_error_comparison, InteractiveDashboard)
  - collocation.eli_pipeline (ELIPipeline)
"""
import warnings
import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def data3(rng):
    """3-way synthetic data with known errors."""
    n = 300
    theta = rng.standard_normal(n)
    d = np.column_stack([
        theta + 0.20 * rng.standard_normal(n),
        theta + 0.30 * rng.standard_normal(n),
        theta + 0.40 * rng.standard_normal(n),
    ])
    return d


@pytest.fixture(scope="module")
def data2(rng):
    n = 300
    theta = rng.standard_normal(n)
    return np.column_stack([
        theta + 0.20 * rng.standard_normal(n),
        theta + 0.35 * rng.standard_normal(n),
    ])


@pytest.fixture(scope="module")
def data4(rng):
    n = 300
    theta = rng.standard_normal(n)
    return np.column_stack([
        theta + 0.15 * rng.standard_normal(n),
        theta + 0.25 * rng.standard_normal(n),
        theta + 0.35 * rng.standard_normal(n),
        theta + 0.45 * rng.standard_normal(n),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Base estimator
# ─────────────────────────────────────────────────────────────────────────────

class TestCollocationEstimatorBase:
    def test_not_fitted_raises(self):
        from collocation.estimators import TC
        m = TC()
        with pytest.raises(RuntimeError, match="not fitted"):
            m.get_metrics()

    def test_fit_returns_self(self, data3):
        from collocation.estimators import TC
        m = TC()
        result = m.fit(data3)
        assert result is m

    def test_method_chaining(self, data3):
        from collocation.estimators import TC
        metrics = TC().fit(data3).get_metrics()
        assert isinstance(metrics, dict)

    def test_dropna(self, data3):
        from collocation.estimators import TC
        dirty = data3.copy()
        dirty[5, 1] = np.nan
        dirty[10, 2] = np.inf
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = TC().fit(dirty)
        assert m.n_samples_ == data3.shape[0] - 2

    def test_xarray_input(self, data3):
        pytest.importorskip("xarray")
        import xarray as xr
        from collocation.estimators import TC
        da = xr.DataArray(data3, dims=["time", "product"])
        m = TC().fit(da)
        assert m.is_fitted_

    def test_list_input(self, data3):
        from collocation.estimators import TC
        m = TC().fit(data3.tolist())
        assert m.is_fitted_

    def test_summary_string(self, data3):
        from collocation.estimators import TC
        s = TC().fit(data3).summary()
        assert "TC" in s
        assert "samples" in s.lower()


# ─────────────────────────────────────────────────────────────────────────────
# TC estimator
# ─────────────────────────────────────────────────────────────────────────────

class TestTCEstimator:
    def test_metrics_keys(self, data3):
        from collocation.estimators import TC
        m = TC().fit(data3).metrics_
        for key in ("EeeT", "SNR", "rho2", "fMSE", "error_std", "n_samples"):
            assert key in m

    def test_error_std_positive(self, data3):
        from collocation.estimators import TC
        std = TC().fit(data3).metrics_["error_std"]
        assert np.all(std >= 0)

    def test_rho2_range(self, data3):
        from collocation.estimators import TC
        r = TC().fit(data3).metrics_["rho2"]
        assert np.all(r >= 0) and np.all(r <= 1)

    def test_wrong_columns(self, data2):
        from collocation.estimators import TC
        with pytest.raises(ValueError, match="3 columns"):
            TC().fit(data2)

    def test_approximate_accuracy(self, data3):
        from collocation.estimators import TC
        std = TC().fit(data3).metrics_["error_std"]
        expected = np.array([0.20, 0.30, 0.40])
        # Allow 30% relative error due to finite-sample variance
        assert np.all(np.abs(std - expected) / expected < 0.30)


# ─────────────────────────────────────────────────────────────────────────────
# EIVD estimator
# ─────────────────────────────────────────────────────────────────────────────

class TestEIVDEstimator:
    def test_metrics_keys(self, data3):
        from collocation.estimators import EIVD
        m = EIVD().fit(data3).metrics_
        for key in ("EeeT", "SNR", "rho2", "fMSE", "L", "cross_corr", "error_std"):
            assert key in m

    def test_cross_corr_is_float(self, data3):
        from collocation.estimators import EIVD
        cc = EIVD().fit(data3).metrics_["cross_corr"]
        assert isinstance(cc, float)

    def test_L_shape(self, data3):
        from collocation.estimators import EIVD
        L = EIVD().fit(data3).metrics_["L"]
        assert L.shape == (3,)


# ─────────────────────────────────────────────────────────────────────────────
# IVD estimator
# ─────────────────────────────────────────────────────────────────────────────

class TestIVDEstimator:
    def test_metrics_keys(self, data2):
        from collocation.estimators import IVD
        m = IVD().fit(data2).metrics_
        for key in ("EeeT", "rho2", "weights", "error_std"):
            assert key in m

    def test_weights_positive(self, data2):
        from collocation.estimators import IVD
        w = IVD().fit(data2).metrics_["weights"]
        assert np.all(w >= 0)

    def test_wrong_columns(self, data3):
        from collocation.estimators import IVD
        with pytest.raises(ValueError, match="2 columns"):
            IVD().fit(data3)


# ─────────────────────────────────────────────────────────────────────────────
# EC estimator
# ─────────────────────────────────────────────────────────────────────────────

class TestECEstimator:
    def test_metrics_keys(self, data4):
        from collocation.estimators import EC
        m = EC().fit(data4).metrics_
        for key in ("EeeT", "SNR", "rho2", "fMSE", "error_std"):
            assert key in m

    def test_error_std_shape(self, data4):
        from collocation.estimators import EC
        std = EC().fit(data4).metrics_["error_std"]
        assert std.shape == (4,)

    def test_wrong_columns(self, data3):
        from collocation.estimators import EC
        with pytest.raises(ValueError, match="4 columns"):
            EC().fit(data3)


# ─────────────────────────────────────────────────────────────────────────────
# CollocationConsultant
# ─────────────────────────────────────────────────────────────────────────────

class TestCollocationConsultant:
    def test_basic_3way(self, data3):
        from collocation.consultant import CollocationConsultant
        report = CollocationConsultant(data3).consult()
        assert report.recommended in ("TC", "EIVD", "BayesianTC", "MTCH")
        assert isinstance(report.alternatives, list)
        assert isinstance(report.text, str)
        assert len(report.text) > 50

    def test_2way(self, data2):
        from collocation.consultant import CollocationConsultant
        report = CollocationConsultant(data2).consult()
        assert report.recommended == "IVD"

    def test_diagnostics_keys(self, data3):
        from collocation.consultant import CollocationConsultant
        diag = CollocationConsultant(data3).consult().diagnostics
        for key in ("lag1_autocorr", "pairwise_crosscorr", "variance_ratio",
                    "normality_pvalue", "skewness", "n_samples"):
            assert key in diag

    def test_correlated_products_recommends_eivd(self, rng):
        """Artificially inject correlated errors; expect EIVD recommendation."""
        from collocation.consultant import CollocationConsultant
        n = 300
        theta = rng.standard_normal(n)
        shared_err = 0.5 * rng.standard_normal(n)
        data = np.column_stack([
            theta + 0.10 * rng.standard_normal(n),
            theta + shared_err + 0.15 * rng.standard_normal(n),
            theta + shared_err + 0.15 * rng.standard_normal(n),
        ])
        report = CollocationConsultant(data).consult()
        assert report.recommended == "EIVD"

    def test_heteroscedastic_recommends_bayesian(self, rng):
        """Simulate time-varying variance → variance_ratio should exceed threshold."""
        from collocation.consultant import CollocationConsultant, _VAR_RATIO_THRESHOLD
        n = 600
        theta = rng.standard_normal(n)
        # 3-segment noise: quiet – loud – quiet (ratio ≈ 10×)
        noise_scale = np.ones(n) * 0.05
        noise_scale[n // 3: 2 * n // 3] = 1.5   # middle third is 30× louder
        data = np.column_stack([
            theta + noise_scale * rng.standard_normal(n),
            theta + noise_scale * rng.standard_normal(n),
            theta + noise_scale * rng.standard_normal(n),
        ])
        report = CollocationConsultant(data).consult()
        # Variance ratio must be elevated
        vr = report.diagnostics["variance_ratio"]
        assert np.max(vr) > _VAR_RATIO_THRESHOLD, (
            f"Expected variance_ratio > {_VAR_RATIO_THRESHOLD}, got {vr}"
        )
        # BayesianTC should appear in recommendations
        all_recs = [report.recommended] + report.alternatives
        assert "BayesianTC" in all_recs

    def test_report_str(self, data3):
        from collocation.consultant import CollocationConsultant, ConsultationReport
        report = CollocationConsultant(data3).consult()
        assert str(report) == report.text

    def test_invalid_columns(self, rng):
        from collocation.consultant import CollocationConsultant
        with pytest.raises(ValueError):
            CollocationConsultant(rng.standard_normal((100, 5)))

    def test_nan_rows_dropped(self, data3):
        from collocation.consultant import CollocationConsultant
        dirty = data3.copy()
        dirty[:5, 0] = np.nan
        # Should not raise
        report = CollocationConsultant(dirty).consult()
        assert report.diagnostics["n_samples"] == data3.shape[0] - 5

    def test_product_names_reflected(self, data3):
        from collocation.consultant import CollocationConsultant
        names = ["ERA5", "GLEAM", "GLDAS"]
        report = CollocationConsultant(data3, product_names=names).consult()
        assert "ERA5" in report.text

    def test_verbose_does_not_raise(self, data3, capsys):
        from collocation.consultant import CollocationConsultant
        CollocationConsultant(data3, verbose=True).consult()
        out = capsys.readouterr().out
        assert "lag1" in out


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

class TestPlotting:
    @pytest.fixture(autouse=True)
    def mpl_backend(self):
        import matplotlib
        matplotlib.use("Agg")
        yield

    def test_plot_error_comparison(self, data3):
        from collocation.estimators import TC, EIVD
        from collocation.plotting import plot_error_comparison
        tc_m = TC().fit(data3).metrics_
        ei_m = EIVD().fit(data3).metrics_
        fig = plot_error_comparison({"TC": tc_m, "EIVD": ei_m})
        import matplotlib.pyplot as plt
        assert fig is not None
        plt.close("all")

    def test_plot_stability_heatmap(self, data3):
        from collocation.plotting import plot_stability_heatmap
        from collocation import tc
        import matplotlib.pyplot as plt
        fig = plot_stability_heatmap(
            data3, tc,
            window_sizes=[100, 200, 300],
        )
        assert fig is not None
        plt.close("all")

    def test_interactive_dashboard_no_plotly(self, data3, monkeypatch):
        """When Plotly is absent, InteractiveDashboard should raise ImportError."""
        import collocation.plotting as cp
        original = cp.PLOTLY_AVAILABLE
        cp.PLOTLY_AVAILABLE = False
        try:
            from collocation.plotting import InteractiveDashboard
            from collocation.estimators import TC
            tc_m = TC().fit(data3).metrics_
            with pytest.raises(ImportError, match="Plotly"):
                InteractiveDashboard(data3, {"TC": tc_m})
        finally:
            cp.PLOTLY_AVAILABLE = original

    def test_interactive_dashboard_build(self, data3):
        """Build an interactive dashboard if Plotly is available."""
        pytest.importorskip("plotly")
        from collocation.plotting import InteractiveDashboard
        from collocation.estimators import TC, EIVD
        from collocation import tc as tc_fn
        tc_m = TC().fit(data3).metrics_
        ei_m = EIVD().fit(data3).metrics_
        dash = InteractiveDashboard(
            data3,
            {"TC": tc_m, "EIVD": ei_m},
            method_fn=tc_fn,
            window_sizes=[100, 200, 300],
        )
        fig = dash.build()
        assert fig is not None

    def test_dashboard_save_html(self, data3, tmp_path):
        pytest.importorskip("plotly")
        from collocation.plotting import InteractiveDashboard
        from collocation.estimators import TC
        tc_m = TC().fit(data3).metrics_
        out = str(tmp_path / "dashboard.html")
        InteractiveDashboard(data3, {"TC": tc_m}).save(out)
        import os
        assert os.path.exists(out)
        content = open(out).read()
        assert "plotly" in content.lower()


# ─────────────────────────────────────────────────────────────────────────────
# ELI Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestELIPipeline:
    @pytest.fixture(scope="class")
    def eli_data(self, rng):
        n = 300
        theta = rng.standard_normal(n)
        water = np.column_stack([
            theta + 0.20 * rng.standard_normal(n),
            theta + 0.25 * rng.standard_normal(n),
            theta + 0.30 * rng.standard_normal(n),
        ])
        energy = np.column_stack([
            theta + 0.15 * rng.standard_normal(n),
            theta + 0.20 * rng.standard_normal(n),
            theta + 0.25 * rng.standard_normal(n),
        ])
        veg = theta + 0.10 * rng.standard_normal(n)
        return water, energy, veg

    def test_run_returns_result(self, eli_data):
        from collocation.eli_pipeline import ELIPipeline
        water, energy, veg = eli_data
        pipe = ELIPipeline(water, energy, veg, methods=["TC"])
        result = pipe.run()
        assert hasattr(result, "eli_water")
        assert hasattr(result, "eli_energy")
        assert hasattr(result, "eli_ratio")

    def test_eli_ratio_in_range(self, eli_data):
        from collocation.eli_pipeline import ELIPipeline
        water, energy, veg = eli_data
        result = ELIPipeline(water, energy, veg, methods=["TC", "EIVD"]).run()
        assert 0.0 <= result.eli_ratio <= 1.0

    def test_html_content(self, eli_data):
        from collocation.eli_pipeline import ELIPipeline
        water, energy, veg = eli_data
        result = ELIPipeline(water, energy, veg, methods=["TC"]).run()
        html = result._html_content
        assert "<!DOCTYPE html>" in html
        assert "ELI" in html

    def test_save_html(self, eli_data, tmp_path):
        from collocation.eli_pipeline import ELIPipeline
        water, energy, veg = eli_data
        result = ELIPipeline(water, energy, veg, methods=["TC"]).run()
        path = str(tmp_path / "eli.html")
        result.save(path)
        import os
        assert os.path.exists(path)

    def test_summary_string(self, eli_data):
        from collocation.eli_pipeline import ELIPipeline
        water, energy, veg = eli_data
        result = ELIPipeline(water, energy, veg, methods=["TC"]).run()
        s = result.summary()
        assert "ELI" in s

    def test_mismatched_lengths_raise(self, rng):
        from collocation.eli_pipeline import ELIPipeline
        with pytest.raises(ValueError, match="same number of rows"):
            ELIPipeline(
                rng.standard_normal((100, 3)),
                rng.standard_normal((200, 3)),
                rng.standard_normal((100, 1)),
            )

    def test_nan_in_water_handled(self, eli_data, rng):
        from collocation.eli_pipeline import ELIPipeline
        water, energy, veg = eli_data
        dirty_water = water.copy()
        dirty_water[10:15, 0] = np.nan
        # Should not raise
        result = ELIPipeline(dirty_water, energy, veg, methods=["TC"]).run()
        assert result.eli_ratio is not None

    def test_1d_vegetation_input(self, rng):
        """1-D vegetation input should be accepted."""
        from collocation.eli_pipeline import ELIPipeline
        n = 200
        theta = rng.standard_normal(n)
        w = np.column_stack([theta + 0.2 * rng.standard_normal(n)] * 3)
        e = np.column_stack([theta + 0.2 * rng.standard_normal(n)] * 3)
        v = theta + 0.1 * rng.standard_normal(n)   # 1-D
        result = ELIPipeline(w, e, v, methods=["TC"]).run()
        assert result.eli_ratio is not None

    def test_xarray_input(self, rng):
        pytest.importorskip("xarray")
        import xarray as xr
        from collocation.eli_pipeline import ELIPipeline
        n = 200
        theta = rng.standard_normal(n)
        w = xr.DataArray(
            np.column_stack([theta + 0.2 * rng.standard_normal(n)] * 3),
            dims=["time", "product"]
        )
        e = np.column_stack([theta + 0.2 * rng.standard_normal(n)] * 3)
        v = theta + 0.1 * rng.standard_normal(n)
        result = ELIPipeline(w, e, v, methods=["TC"]).run()
        assert result.eli_ratio is not None

    def test_multiple_methods(self, eli_data):
        from collocation.eli_pipeline import ELIPipeline
        water, energy, veg = eli_data
        result = ELIPipeline(
            water, energy, veg,
            methods=["TC", "EIVD", "IVS"],
            n_bootstrap=100,
        ).run()
        # All methods should have produced results
        assert len(result.method_results) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Integration: pipeline → consultant → estimators
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    def test_consultant_then_fit(self, data3):
        """Use consultant recommendation to choose estimator, then fit."""
        from collocation.consultant import CollocationConsultant
        from collocation import estimators
        report = CollocationConsultant(data3).consult()
        estimator_map = {
            "TC": estimators.TC,
            "EIVD": estimators.EIVD,
        }
        cls = estimator_map.get(report.recommended, estimators.TC)
        model = cls().fit(data3)
        assert model.is_fitted_

    def test_loop_over_estimators(self, data3):
        """Simulate a method comparison loop."""
        from collocation.estimators import TC, EIVD
        results = {}
        for name, est in [("TC", TC()), ("EIVD", EIVD())]:
            results[name] = est.fit(data3).metrics_["error_std"]
        assert set(results) == {"TC", "EIVD"}
        for std in results.values():
            assert std.shape == (3,)
