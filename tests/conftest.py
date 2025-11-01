"""Pytest configuration helpers.

This file ensures the project root is on sys.path so tests can import the
local `collocation` package without installing the package into the env.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

# Read verbosity flag from environment. If TEST_VERBOSE is '0' or unset, keep default printing.
TEST_VERBOSE: Optional[str] = os.environ.get("TEST_VERBOSE", "1")


# Prepend repository root to sys.path so tests import local package
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)

if REPO_ROOT not in sys.path:
    # insert at 0 to take precedence over installed packages
    sys.path.insert(0, REPO_ROOT)


def pytest_configure(config):
    # optional: expose repo root to tests via config if needed
    config.repo_root = REPO_ROOT


def _color_text(text: str, color_code: str) -> str:
    """Wrap text with ANSI color codes. Works in most terminals."""
    RESET = "\x1b[0m"
    return f"{color_code}{text}{RESET}"


def pytest_runtest_logreport(report):
    """Pytest hook for per-test reporting.

    Prints a concise, colorized message for pass/fail/skip during the call phase.
    Controlled by the TEST_VERBOSE environment variable (set to '0' to silence).
    """
    try:
        if TEST_VERBOSE == "0":
            return

        # Only act on the test call phase
        if report.when != "call":
            return

        node = report.nodeid

        # ANSI color codes
        GREEN = "\x1b[32m"
        RED = "\x1b[31m"
        YELLOW = "\x1b[33m"
        CYAN = "\x1b[36m"
        RESET = "\x1b[0m"

        if report.outcome == "passed":
            msg = _color_text("✔ 测试通过", GREEN)
            print(f"{msg}: {CYAN}{node}{RESET}")
        elif report.outcome == "failed":
            msg = _color_text("✖ 测试失败", RED)
            print(f"{msg}: {CYAN}{node}{RESET}")
        elif report.outcome == "skipped":
            # Show reason if available
            reason = ""
            try:
                # Attempt to extract skip reason, fallback to longreprtext
                if hasattr(report, "longrepr") and isinstance(report.longrepr, tuple) and len(report.longrepr) > 2:
                    reason = str(report.longrepr[2])
                else:
                    reason = getattr(report, "longreprtext", "skipped")
            except Exception:
                reason = "skipped"
            msg = _color_text("⚠ 跳过", YELLOW)
            print(f"{msg}: {CYAN}{node}{RESET}  ({reason})")
    except Exception:
        # Never let reporting break tests
        pass


# Import local test fixtures to make them available project-wide
try:
    from .fixtures_covariance import *  # noqa: F401,F403
except Exception:
    # If fixtures file is missing or import fails, continue without failing tests
    pass


# --- Inline test fixtures (copied from fixtures_covariance.py) ---
import numpy as _np
import xarray as _xr
import pytest as _pytest


@_pytest.fixture
def models():
    return ["A", "B", "C"]


@_pytest.fixture
def lat():
    return [0.0, 1.0]


@_pytest.fixture
def lon():
    return [10.0, 11.0]


@_pytest.fixture
def mse_1d(models):
    return _xr.DataArray(_np.array([0.1, 0.2, 0.3]), dims=("model",), coords={"model": models})


@_pytest.fixture
def mse_batch(models, lat, lon):
    vals = _np.abs(_np.random.RandomState(0).randn(len(lat), len(lon), len(models))) + 0.1
    return _xr.DataArray(vals, dims=("lat", "lon", "model"), coords={"lat": lat, "lon": lon, "model": models})


@_pytest.fixture
def cross_2d(models):
    vals = _np.eye(len(models)) * 0.2
    return _xr.DataArray(vals, dims=("model", "model_2"), coords={"model": models, "model_2": models})


@_pytest.fixture
def cross_batch(models, lat, lon):
    vals = _np.zeros((len(lat), len(lon), len(models), len(models)))
    return _xr.DataArray(vals, dims=("lat", "lon", "model", "model_2"), coords={"lat": lat, "lon": lon, "model": models, "model_2": models})

