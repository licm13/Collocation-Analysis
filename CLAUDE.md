# CLAUDE.md - AI Assistant Guide for Collocation Analysis Package

> **Last Updated**: 2025-11-14
> **For**: AI assistants working on this codebase
> **Purpose**: Comprehensive guide to codebase structure, conventions, and development workflows

---

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Package Structure](#package-structure)
3. [Code Conventions and Patterns](#code-conventions-and-patterns)
4. [Development Workflow](#development-workflow)
5. [Testing Guidelines](#testing-guidelines)
6. [Documentation Standards](#documentation-standards)
7. [Common Tasks](#common-tasks)
8. [Architecture Decisions](#architecture-decisions)
9. [Performance Considerations](#performance-considerations)
10. [Git Workflow](#git-workflow)

---

## Repository Overview

### Purpose
Python package for **collocation-based error analysis** of remote sensing and geophysical datasets. Enables error quantification without ground truth using multiple independent measurements.

### Key Facts
- **Converted from**: MATLAB codebase (original author: licm_13@163.com)
- **Language**: Python 3.7+
- **Size**: ~26 Python modules in main package, ~6,161 total lines
- **Architecture**: Modular design with optional dependencies
- **Documentation**: Dual language (English + Chinese)
- **License**: MIT

### Primary Use Cases
1. Remote sensing product validation (soil moisture, precipitation, wind speed)
2. Multi-source data fusion with optimal weighting
3. Error budget estimation without ground truth
4. Climate model evaluation
5. Ecosystem limitation analysis (ELI application)

---

## Package Structure

### Directory Layout

```
Collocation-Analysis/
├── collocation/              # Main Python package
│   ├── __init__.py          # Package exports and availability flags
│   ├── tc.py                # Triple Collocation (3-way)
│   ├── ivd.py               # Information Vector Dual (2-way)
│   ├── ivs.py               # Information Vector with Scaling (2-way + bootstrap)
│   ├── eivd.py              # Extended IVD (3-way with error correlation)
│   ├── ec.py                # Extended Collocation (4-way)
│   ├── etcc.py              # Extended TC for Correlation (Wei et al. 2023)
│   ├── etcc_evaluation.py   # ETCC evaluation utilities
│   ├── etcc_spatial.py      # ETCC spatial operations
│   ├── bayesian_tc.py       # Bayesian TC (time-varying errors)
│   ├── bayesian_tch.py      # Bayesian Three-Cornered Hat (constant errors)
│   ├── btch_he2020.py       # Analytical BTCH (He et al. 2020)
│   ├── utils.py             # Performance metrics (KGE, NSE, RMSE, etc.)
│   ├── covariance.py        # Covariance estimation and construction
│   ├── fuse.py              # Bias estimation and fusion helpers
│   ├── simple_average.py    # Simple/weighted averaging methods
│   ├── eli.py               # Ecosystem Limitation Index processor
│   └── fusion/              # Data fusion subpackage (9 modules)
│       ├── weights.py       # IVW, GLS/BLUE, QP weight solvers
│       ├── covariance.py    # MSE/covariance estimation, shrinkage
│       ├── fuse.py          # High-level fusion orchestrator
│       ├── constraints.py   # Physics constraints
│       ├── uncertainty.py   # Variance propagation, bootstrap
│       ├── robust.py        # Robust estimators, outlier detection
│       ├── localization.py  # Moving window, biome partitioning
│       └── broadcast.py     # Broadcasting utilities
│
├── examples/                 # Runnable examples (20 files)
│   ├── example_all_methods.py              # Comprehensive demo
│   ├── comprehensive_comparison.py         # Publication-ready comparison
│   ├── etcc_example.py                     # ETCC methodology
│   ├── bayesian_tch_example.py             # Bayesian workflow
│   ├── eli_comprehensive_example.py        # ELI pipeline
│   ├── example_fusion.py                   # Fusion framework
│   ├── robust_collocation_scenarios.py     # Stress testing
│   └── figures/                            # Generated output plots
│
├── tests/                    # Test suite (15 files)
│   ├── conftest.py          # Pytest configuration, custom reporting
│   ├── test_collocation.py  # Core methods tests (545 lines)
│   ├── test_fusion.py       # Fusion module tests
│   ├── test_method_workflows.py  # Integration tests
│   ├── test_performance.py  # Performance benchmarks
│   └── figures/             # Test output plots
│
├── scripts/                  # Utility scripts
│   ├── fuse_et.py           # ET fusion CLI tool
│   └── apply_plotting_fixes.py  # Font configuration utilities
│
├── docs/                     # Additional documentation
│   ├── PERFORMANCE.md       # Optimization guide
│   └── API_CN.md            # Chinese API docs
│
├── BTCH_He2020/             # BTCH replication project
├── ETCC/                    # ETCC documentation and methodology
├── hcc_et_framework/        # HCC-ET framework
├── ELI Application/         # ELI MATLAB legacy code
├── SimpleAverage/           # Simple averaging documentation
│
├── README.md                # Main documentation (English)
├── README_CN.md             # Chinese documentation
├── ELI_README.md            # ELI application guide
├── BAYESIAN_INTEGRATION_GUIDE.md  # Bayesian setup guide
├── PERFORMANCE_SUMMARY.md   # Optimization achievements
├── setup.py                 # Package installation
├── requirements.txt         # Core dependencies
└── pytest.ini               # Test configuration
```

### Module Organization Principles

1. **One method = One module**: Each collocation method is self-contained
2. **Optional dependencies**: Graceful degradation (PyMC3, xarray)
3. **Subpackages for complexity**: Fusion module demonstrates scalable organization
4. **Separation of concerns**: Core algorithms, utilities, applications, examples, tests

---

## Code Conventions and Patterns

### 1. Module Structure Pattern

**Every module follows this template**:

```python
"""
Module Title
============

Multi-paragraph description of purpose and functionality.

This module implements X based on Y paper. Key features:
- Feature 1
- Feature 2

References
----------
.. [1] Author et al. (Year). Title. Journal, Volume(Issue), Pages.
       DOI: https://doi.org/...

Author: Original MATLAB by X, Python conversion by Y
Date: YYYY-MM-DD
"""

import numpy as np
from typing import Tuple, Optional, Dict, Union, List
import warnings
from scipy import linalg

# Constants
MIN_SAMPLES = 50
OFFSET_SEARCH_LIMIT = 20  # 10% of data length

def main_function(data: np.ndarray, param: int = 100) -> Tuple[np.ndarray, ...]:
    """
    Main implementation with detailed docstring.

    See documentation pattern below for full structure.
    """
    # Input validation
    if data.shape[0] < MIN_SAMPLES:
        warnings.warn(f"Sample size ({data.shape[0]}) less than recommended")
        return default_values

    # Core algorithm
    result = algorithm_implementation(data, param)

    # Return with consistent structure
    return result

# Helper functions (private with leading underscore if not exported)
def _helper_function(x):
    """Private helper function."""
    pass
```

### 2. Input Validation Pattern

**Always validate inputs comprehensively**:

```python
def method(data: np.ndarray, reference: Optional[np.ndarray] = None) -> Tuple[...]:
    """Method with robust input validation."""

    # 1. Check array type
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be numpy array, got {type(data)}")

    # 2. Check shape
    if data.ndim != 2:
        raise ValueError(f"data must be 2D array, got shape {data.shape}")

    if data.shape[1] != 3:
        raise ValueError(f"data must have 3 columns, got {data.shape[1]}")

    # 3. Check for NaN/Inf
    if np.any(np.isnan(data)):
        raise ValueError("data contains NaN values")

    if np.any(np.isinf(data)):
        raise ValueError("data contains infinite values")

    # 4. Check sample size with warning
    n_samples = data.shape[0]
    if n_samples < MIN_SAMPLES:
        warnings.warn(
            f"Sample size ({n_samples}) is less than recommended ({MIN_SAMPLES}). "
            f"Results may be unreliable.",
            UserWarning
        )

    # 5. Check value ranges if applicable
    if reference is not None and data.shape[0] != reference.shape[0]:
        raise ValueError(
            f"data and reference must have same length: "
            f"{data.shape[0]} != {reference.shape[0]}"
        )

    # Proceed with algorithm
    ...
```

### 3. Numerical Stability Pattern

**Protect against numerical issues**:

```python
def robust_calculation(matrix: np.ndarray) -> np.ndarray:
    """Calculation with numerical safeguards."""

    # Use error state context manager
    with np.errstate(invalid='ignore', divide='ignore'):
        result = potentially_unstable_operation(matrix)

    # Check for degenerate results
    if not np.all(np.isfinite(result)):
        # Return NaN or use fallback
        warnings.warn("Numerical instability detected, returning NaN")
        return np.full_like(result, np.nan)

    # Protect matrix inversion
    try:
        inv_matrix = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        # Use pseudoinverse as fallback
        warnings.warn("Matrix is singular, using pseudoinverse")
        inv_matrix = np.linalg.pinv(matrix)

    # Add small regularization if needed
    regularized = matrix + 1e-10 * np.eye(matrix.shape[0])

    return inv_matrix
```

### 4. Optional Dependency Pattern

**Handle optional dependencies gracefully**:

```python
# At module level
try:
    import pymc3 as pm
    import theano.tensor as tt
    PYMC3_AVAILABLE = True
except ImportError:
    PYMC3_AVAILABLE = False
    pm = None
    tt = None

try:
    import xarray as xr
    XR_AVAILABLE = True
except ImportError:
    XR_AVAILABLE = False
    xr = None

# Export availability flags
__all__ = ['method_name', 'PYMC3_AVAILABLE', 'XR_AVAILABLE']

# Check before using
def bayesian_method(data):
    """Method requiring PyMC3."""
    if not PYMC3_AVAILABLE:
        raise ImportError(
            "PyMC3 required for Bayesian methods. "
            "Install with: pip install pymc3==3.11.5 theano-pymc"
        )

    # Use PyMC3
    with pm.Model() as model:
        ...
```

### 5. Performance Optimization Pattern

**Optimize hot paths without sacrificing readability**:

```python
def optimized_search(data: np.ndarray, max_search: int = 100):
    """Optimized search with early termination."""

    # Pre-compute constants outside loops
    n = len(data)
    mean_data = np.mean(data)
    std_data = np.std(data)

    # Limit search space (e.g., 10-20% of data)
    search_limit = min(max_search, n // 5)

    # Early termination counter
    no_improvement = 0
    best_score = -np.inf

    for i in range(search_limit):
        score = compute_score(data, i, mean_data, std_data)

        if score > best_score:
            best_score = score
            no_improvement = 0
        else:
            no_improvement += 1

            # Early exit if no improvement for 5 iterations
            if no_improvement > 5:
                break

    return best_score

def vectorized_operation(x, y, z, weights):
    """Use vectorized NumPy operations."""
    # Good: Vectorized
    merged = weights[0] * x + weights[1] * y + weights[2] * z

    # Bad: Loop-based (avoid this)
    # merged = np.zeros_like(x)
    # for i in range(len(x)):
    #     merged[i] = weights[0] * x[i] + weights[1] * y[i] + weights[2] * z[i]

    return merged
```

### 6. Documentation Pattern

**Use comprehensive Google/NumPy style docstrings**:

```python
def collocation_method(
    data: np.ndarray,
    reference: Optional[np.ndarray] = None,
    bootstrap: int = 1000,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    One-line summary of the method.

    Extended description paragraph explaining the purpose, assumptions,
    and mathematical background. Can include multiple paragraphs.

    The method assumes:
    1. Errors are zero-mean
    2. Errors are uncorrelated with truth
    3. Errors are mutually independent

    Parameters
    ----------
    data : np.ndarray
        Input data array with shape (n_samples, n_products).
        Each column represents one data product.
    reference : np.ndarray, optional
        Reference data for validation, shape (n_samples,).
        If None, no validation is performed.
    bootstrap : int, default=1000
        Number of bootstrap samples for uncertainty estimation.
        Must be at least 100.
    alpha : float, default=0.05
        Significance level for confidence intervals (0 < alpha < 1).
        Default 0.05 gives 95% confidence intervals.

    Returns
    -------
    error_covariance : np.ndarray
        Error covariance matrix, shape (n_products, n_products).
        Diagonal elements are error variances.
    correlations : np.ndarray
        Data-truth correlations, shape (n_products,).
        Values range from 0 to 1.
    diagnostics : dict
        Dictionary containing:
        - 'snr': Signal-to-noise ratios
        - 'fmse': Fractional mean squared errors
        - 'ci_lower': Lower confidence bounds
        - 'ci_upper': Upper confidence bounds

    Raises
    ------
    ValueError
        If data has wrong shape or contains NaN/Inf values.
    TypeError
        If data is not a numpy array.

    Warnings
    --------
    UserWarning
        If sample size is below recommended minimum (50 samples).

    Examples
    --------
    Basic usage with three products:

    >>> import numpy as np
    >>> from collocation import collocation_method
    >>>
    >>> # Generate synthetic data
    >>> n = 200
    >>> true_signal = np.random.randn(n)
    >>> product1 = true_signal + 0.2 * np.random.randn(n)
    >>> product2 = true_signal + 0.3 * np.random.randn(n)
    >>> product3 = true_signal + 0.4 * np.random.randn(n)
    >>> data = np.column_stack([product1, product2, product3])
    >>>
    >>> # Apply method
    >>> error_cov, correlations, diagnostics = collocation_method(data)
    >>> print(f"Error variances: {np.diag(error_cov)}")
    >>> print(f"Correlations: {correlations}")

    With bootstrap uncertainty:

    >>> error_cov, corr, diag = collocation_method(data, bootstrap=2000)
    >>> print(f"95% CI: [{diag['ci_lower']}, {diag['ci_upper']}]")

    Notes
    -----
    Mathematical formulation:

    .. math::
        X_i = \\alpha_i + \\beta_i \\theta + \\epsilon_i

    where:
        - :math:`X_i`: Observed product i
        - :math:`\\theta`: True signal (unknown)
        - :math:`\\alpha_i, \\beta_i`: Calibration parameters
        - :math:`\\epsilon_i`: Random error with variance :math:`\\sigma_i^2`

    The covariance structure provides 6 equations for 3 products:

    .. math::
        \\text{Cov}(X_i, X_j) = \\beta_i \\beta_j \\sigma_\\theta^2 + \\delta_{ij} \\sigma_i^2

    See Also
    --------
    related_method : Brief description of relationship
    alternative_method : When to use this instead

    References
    ----------
    .. [1] Stoffelen, A. (1998). Toward the true near-surface wind speed.
           Journal of Geophysical Research, 103(C4), 7755-7766.
    .. [2] Gruber, A., et al. (2016). Recent advances in triple collocation.
           Int. J. Appl. Earth Obs. Geoinformation, 45, 200-211.
    """
    # Implementation
    pass
```

### 7. Inline Comments Style

```python
def complex_calculation(data):
    """Complex calculation with explanatory comments."""

    # ========================================================================
    # Step 1: Data Preprocessing
    # ========================================================================

    # Remove temporal mean (bias correction)
    # Following Equation 3 from Smith et al. (2020)
    data_centered = data - np.mean(data, axis=0)

    # Calculate covariance matrix using Bessel's correction
    # This is unbiased for normally distributed data
    cov_matrix = np.cov(data_centered, rowvar=False)

    # ========================================================================
    # Step 2: Construct Equation System
    # ========================================================================

    # Build coefficient matrix A for linear system Ax = b
    # where x contains the unknown error variances
    n_products = data.shape[1]
    n_equations = n_products * (n_products + 1) // 2  # Unique covariances
    A = np.zeros((n_equations, n_products))

    eq_idx = 0
    for i in range(n_products):
        for j in range(i, n_products):
            # Each covariance gives one equation
            A[eq_idx, i] = 1.0 if i == j else 0.0  # Error variance term
            eq_idx += 1

    # Solve system (details omitted for brevity)
    x = np.linalg.solve(A, b)  # x = [error_var_1, error_var_2, ...]

    return x
```

---

## Development Workflow

### Branch Strategy

1. **Main branch**: `main` - stable, production-ready code
2. **Feature branches**: `claude/<descriptive-name>-<session-id>`
   - Format: `claude/integrate-btch-he2020-package-011CV3GaYZeCiWF7EssdQhnA`
   - Descriptive name indicates the feature
   - Session ID ensures uniqueness
3. **Development flow**:
   - Create feature branch from main
   - Develop and test
   - Submit PR with clear description
   - Merge to main after review

### Commit Message Guidelines

**Format**: Clear, descriptive, action-oriented

```
# Good examples:
Integrate BTCH (He et al. 2020) into main collocation package
Add BTCH method replication project files
Optimize slow code: ETCC exhaustive search, EC rescaling, IVD offset search
Add comprehensive data fusion module with IVW, GLS/BLUE, and constrained QP
Fix numerical stability issue in TC when data is near-singular

# Structure for complex commits:
Add feature X with Y and Z

- Implement core algorithm in module.py
- Add tests covering edge cases A, B, C
- Update documentation with examples
- Add example script demonstrating usage

Closes #123
```

### Development Process

1. **Start with requirements**: Understand what needs to be implemented
2. **Review existing code**: Check similar modules for patterns
3. **Implement incrementally**:
   - Core algorithm first
   - Input validation
   - Error handling
   - Optimization (if needed)
4. **Add tests**: Unit tests covering normal and edge cases
5. **Add examples**: At least one runnable example
6. **Document**: Module docstring, function docstrings, README updates
7. **Performance check**: Profile if dealing with hot paths
8. **Submit PR**: Include description, testing notes, related issues

### Recent Development Patterns (from git history)

1. **BTCH He2020 Integration** (Latest):
   - Added new module `btch_he2020.py`
   - Created subdirectory with documentation
   - Integrated into main package `__init__.py`
   - Added tests and examples

2. **Performance Optimization Pass**:
   - Profiled slow methods (IVD offset search, ETCC exhaustive search)
   - Achieved 98% speedup through optimizations
   - Documented in `PERFORMANCE_SUMMARY.md`
   - Maintained API compatibility

3. **Fusion Module Addition**:
   - Created subpackage `collocation/fusion/`
   - Multiple modules with clear separation
   - High-level API in `fuse.py`
   - CLI tool in `scripts/fuse_et.py`

---

## Testing Guidelines

### Test Structure

**Location**: `/tests/` directory

**Organization**:
```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── test_collocation.py      # Core methods (IVD, IVS, TC, EIVD, EC)
├── test_fusion.py           # Fusion module tests
├── test_method_workflows.py # Integration tests
├── test_performance.py      # Performance benchmarks
├── test_covariance*.py      # Covariance utilities
└── figures/                 # Test output plots
```

### Test Patterns

#### 1. Class-Based Organization

```python
class TestMethodName:
    """Test cases for MethodName."""

    def test_basic_functionality(self):
        """Test basic operation with valid input."""
        data = np.random.randn(200, 3)
        result = method_name(data)

        # Assert shape
        assert result.shape == (3, 3)

        # Assert properties
        assert np.all(np.diag(result) > 0)  # Positive variances

    def test_invalid_input_shape(self):
        """Test error handling for invalid shape."""
        data = np.random.randn(200, 2)  # Wrong number of columns

        with pytest.raises(ValueError, match="must have 3 columns"):
            method_name(data)

    def test_insufficient_samples(self):
        """Test warning for insufficient data."""
        data = np.random.randn(10, 3)  # Too few samples

        with pytest.warns(UserWarning, match="Sample size.*less than"):
            result = method_name(data)

    def test_nan_handling(self):
        """Test error for NaN values."""
        data = np.random.randn(200, 3)
        data[5, 1] = np.nan

        with pytest.raises(ValueError, match="contains NaN"):
            method_name(data)

    def test_against_ground_truth(self):
        """Test accuracy with known ground truth."""
        # Generate synthetic data with known errors
        n = 500
        true_signal = np.random.randn(n)
        error_var = [0.1, 0.2, 0.3]

        product1 = true_signal + np.sqrt(error_var[0]) * np.random.randn(n)
        product2 = true_signal + np.sqrt(error_var[1]) * np.random.randn(n)
        product3 = true_signal + np.sqrt(error_var[2]) * np.random.randn(n)
        data = np.column_stack([product1, product2, product3])

        # Apply method
        error_cov, _, _, _ = method_name(data)
        estimated_var = np.diag(error_cov)

        # Check accuracy (allow 20% error due to sampling)
        for i in range(3):
            relative_error = abs(estimated_var[i] - error_var[i]) / error_var[i]
            assert relative_error < 0.20, f"Product {i}: {relative_error:.2%} error"
```

#### 2. Fixtures for Reusable Setup

```python
# In conftest.py
import pytest
import numpy as np

@pytest.fixture
def sample_data_3way():
    """Generate 3-way collocation test data."""
    np.random.seed(42)
    n = 200
    true_signal = np.random.randn(n)

    product1 = true_signal + 0.2 * np.random.randn(n)
    product2 = true_signal + 0.3 * np.random.randn(n)
    product3 = true_signal + 0.4 * np.random.randn(n)

    return np.column_stack([product1, product2, product3])

@pytest.fixture
def expected_error_variances():
    """Expected error variances for sample_data_3way."""
    return np.array([0.04, 0.09, 0.16])  # 0.2^2, 0.3^2, 0.4^2

# Use in tests
def test_with_fixture(sample_data_3way, expected_error_variances):
    """Test using fixtures."""
    result = method(sample_data_3way)
    assert np.allclose(result, expected_error_variances, rtol=0.2)
```

#### 3. Assertion Patterns

```python
# Shape assertions
assert result.shape == expected_shape
assert len(output) == n_products

# Value range assertions
assert np.all(variance > 0), "Variances must be positive"
assert np.all((correlation >= 0) & (correlation <= 1)), "Correlations in [0,1]"

# Numerical relationship assertions
assert np.allclose(fMSE, 1 - rho2, atol=1e-10), "fMSE = 1 - rho^2"
assert abs(np.sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"

# Approximate equality
assert np.allclose(estimated, expected, rtol=0.1, atol=1e-8)

# Matrix properties
assert np.allclose(matrix, matrix.T), "Matrix must be symmetric"
assert np.all(np.linalg.eigvals(matrix) > 0), "Matrix must be positive definite"
```

### Test Configuration

**File**: `pytest.ini`

```ini
[pytest]
testpaths = tests              # Only test top-level tests/
python_files = test_*.py       # Test files start with test_
python_classes = Test*         # Test classes start with Test
python_functions = test_*      # Test functions start with test_
filterwarnings = default       # Show all warnings
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_collocation.py -v

# Run specific test class
pytest tests/test_collocation.py::TestTC -v

# Run specific test
pytest tests/test_collocation.py::TestTC::test_tc_basic -v

# Run with coverage
pytest tests/ --cov=collocation --cov-report=html

# Run with output capture disabled (see prints)
pytest tests/ -v -s

# Run only fast tests (if markers used)
pytest tests/ -m "not slow"
```

### Test Quality Checklist

- [ ] Tests for normal operation
- [ ] Tests for edge cases (empty input, single sample, etc.)
- [ ] Tests for invalid input (wrong shape, NaN, wrong type)
- [ ] Tests with known ground truth
- [ ] Tests for numerical stability (near-singular matrices, etc.)
- [ ] Tests for optional dependencies (xarray, PyMC3)
- [ ] Performance regression tests (if optimizing)
- [ ] Visual output tests (generate plots for verification)

---

## Documentation Standards

### Documentation Hierarchy

1. **README.md**: User-facing overview, installation, quick start, examples
2. **Module docstrings**: Technical details, API reference
3. **Function docstrings**: Parameters, returns, examples
4. **Inline comments**: Algorithm explanations, mathematical context
5. **Guides**: Specialized documentation (ELI_README.md, BAYESIAN_INTEGRATION_GUIDE.md)

### Docstring Style: Google/NumPy Hybrid

**Required sections**:
- Summary (one-line)
- Extended description (multi-paragraph)
- Parameters (with types, shapes, ranges)
- Returns (with types, shapes, interpretation)
- Raises (exceptions and when)
- Examples (executable code)
- Notes (mathematical formulation, references)
- See Also (related functions)
- References (papers with DOI)

**Example**:

```python
def method(data: np.ndarray, param: int = 100) -> Tuple[np.ndarray, float]:
    """
    Brief one-line summary ending with period.

    Extended description with multiple paragraphs. Explain the purpose,
    when to use this method, and key assumptions.

    This method assumes:
    1. Assumption one
    2. Assumption two

    Parameters
    ----------
    data : np.ndarray
        Description including:
        - Shape: (n_samples, n_features)
        - Valid range: all finite values
        - Units: [specify if applicable]
    param : int, default=100
        Description with default value
        Must be positive integer

    Returns
    -------
    result : np.ndarray
        Description with shape (n_features, n_features)
        Interpretation: covariance matrix of errors
    score : float
        Description with valid range [0, 1]
        Higher is better

    Raises
    ------
    ValueError
        If data contains NaN or has wrong shape
    TypeError
        If data is not numpy array

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(200, 3)
    >>> result, score = method(data)
    >>> print(result.shape)
    (3, 3)

    Notes
    -----
    Mathematical formulation:

    .. math::
        X_i = \\alpha + \\beta \\theta + \\epsilon_i

    See Also
    --------
    related_function : Brief description

    References
    ----------
    .. [1] Author et al. (Year). Title. Journal.
    """
    pass
```

### README Structure

```markdown
# Package Name

Brief tagline

![Badges]

## Overview

High-level description (2-3 paragraphs)

## Features

- Feature 1: Description
- Feature 2: Description

## Installation

### From source
```bash
commands
```

### Dependencies
List with versions

## Quick Start

### Example 1: Basic Usage
```python
code
```

### Example 2: Advanced Usage
```python
code
```

## API Documentation

### Method Name

```python
from package import method
result = method(data)
```

**Parameters:**
- param1: description

**Returns:**
- result: description

**Reference:**
> Citation

## Testing

```bash
pytest tests/ -v
```

## Contributing

Guidelines for contributors

## Citation

BibTeX entry

## License

License information
```

### Chinese Documentation Support

- **Dual language**: Both English and Chinese documentation maintained
- **README_CN.md**: Full Chinese translation of README.md
- **Chinese comments**: Acceptable in code for clarity
- **Test output**: Chinese characters in test reporting (conftest.py)

```python
# Example of bilingual comments
def calculate_score(data):
    """Calculate performance score."""
    # 计算性能得分 (Calculate performance score)
    score = np.mean(data)
    return score
```

---

## Common Tasks

### Task 1: Add a New Collocation Method

1. **Create new module** in `collocation/`:
   ```bash
   touch collocation/new_method.py
   ```

2. **Implement module structure**:
   ```python
   """
   New Method Name
   ===============

   Description and references.
   """

   import numpy as np
   from typing import Tuple
   import warnings

   def new_method(data: np.ndarray) -> Tuple[np.ndarray, ...]:
       """Implementation with full docstring."""
       # Input validation
       # Algorithm
       # Return results
       pass
   ```

3. **Add to package** in `collocation/__init__.py`:
   ```python
   from .new_method import new_method
   __all__ = [..., 'new_method']
   ```

4. **Add tests** in `tests/test_collocation.py`:
   ```python
   class TestNewMethod:
       def test_basic(self):
           """Test basic functionality."""
           pass
   ```

5. **Add example** in `examples/`:
   ```bash
   touch examples/new_method_example.py
   ```

6. **Update documentation**:
   - Add section to README.md
   - Add to method comparison table
   - Update API documentation

### Task 2: Optimize Slow Code

1. **Profile to identify bottleneck**:
   ```python
   import cProfile
   import pstats

   profiler = cProfile.Profile()
   profiler.enable()

   # Run slow code
   result = slow_function(data)

   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(20)
   ```

2. **Common optimizations**:
   - Reduce search space (limit iterations)
   - Pre-compute constants outside loops
   - Vectorize operations
   - Use NumPy built-ins
   - Add early termination

3. **Benchmark before/after**:
   ```python
   import time

   # Before
   start = time.time()
   result_old = old_implementation(data)
   time_old = time.time() - start

   # After
   start = time.time()
   result_new = new_implementation(data)
   time_new = time.time() - start

   speedup = time_old / time_new
   print(f"Speedup: {speedup:.1f}x")

   # Verify results match
   assert np.allclose(result_old, result_new)
   ```

4. **Add performance test** to prevent regression:
   ```python
   # In tests/test_performance.py
   def test_method_performance():
       """Ensure method completes within time budget."""
       data = np.random.randn(1000, 3)

       start = time.time()
       result = method(data)
       elapsed = time.time() - start

       assert elapsed < 1.0, f"Method too slow: {elapsed:.2f}s"
   ```

5. **Document optimization** in PERFORMANCE_SUMMARY.md

### Task 3: Add Optional Dependency Feature

1. **Add import with fallback**:
   ```python
   # In module
   try:
       import optional_package as op
       OPTIONAL_AVAILABLE = True
   except ImportError:
       OPTIONAL_AVAILABLE = False
       op = None

   __all__ = ['feature', 'OPTIONAL_AVAILABLE']
   ```

2. **Check availability in functions**:
   ```python
   def feature_requiring_optional(data):
       """Feature requiring optional package."""
       if not OPTIONAL_AVAILABLE:
           raise ImportError(
               "This feature requires optional_package. "
               "Install with: pip install optional_package"
           )

       # Use optional package
       result = op.process(data)
       return result
   ```

3. **Add to setup.py**:
   ```python
   extras_require = {
       'optional': ['optional_package>=1.0.0'],
       'all': ['optional_package>=1.0.0', ...]
   }
   ```

4. **Document in README**:
   ```markdown
   ### Optional Dependencies

   - optional_package >= 1.0.0 (for feature X)

   Install with optional support:
   ```bash
   pip install -e .[optional]
   ```
   ```

5. **Add conditional tests**:
   ```python
   @pytest.mark.skipif(not OPTIONAL_AVAILABLE, reason="Requires optional_package")
   def test_optional_feature():
       """Test feature requiring optional dependency."""
       pass
   ```

### Task 4: Create a New Example

1. **Create example file**:
   ```python
   """
   Example: Descriptive Title
   ==========================

   Description of what this example demonstrates.

   Author: Name
   Date: YYYY-MM-DD
   """

   import sys, os
   sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

   import numpy as np
   import matplotlib.pyplot as plt
   from collocation import method1, method2

   # Font configuration for plots
   plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
   plt.rcParams['axes.unicode_minus'] = False

   def generate_data():
       """Generate synthetic test data."""
       n = 200
       true_signal = np.random.randn(n)
       # ...
       return data

   def demonstrate_method1():
       """Demonstrate method 1."""
       print("\n" + "=" * 70)
       print("METHOD 1")
       print("=" * 70)

       data = generate_data()
       result = method1(data)

       print(f"Result shape: {result.shape}")
       print(f"Result values: {result}")

   def main():
       """Run all demonstrations."""
       print("Example: Descriptive Title")
       print("=" * 70)

       demonstrate_method1()
       demonstrate_method2()

       # Generate plots
       fig, axes = plt.subplots(2, 2, figsize=(12, 10))
       # ... plotting code ...

       # Save figure
       script_dir = os.path.dirname(os.path.abspath(__file__))
       fig_dir = os.path.join(script_dir, "figures")
       os.makedirs(fig_dir, exist_ok=True)
       script_name = os.path.splitext(os.path.basename(__file__))[0]
       fig.savefig(
           os.path.join(fig_dir, f"{script_name}.png"),
           dpi=150,
           bbox_inches='tight'
       )

       print(f"\nFigure saved to {fig_dir}/{script_name}.png")
       plt.show()

   if __name__ == "__main__":
       main()
   ```

2. **Test example runs**:
   ```bash
   python examples/new_example.py
   ```

3. **Add to README** in Examples section

### Task 5: Debug Numerical Issues

1. **Add diagnostic output**:
   ```python
   def debug_method(data, verbose=True):
       """Method with diagnostic output."""
       if verbose:
           print(f"Input shape: {data.shape}")
           print(f"Input range: [{np.min(data):.3f}, {np.max(data):.3f}]")
           print(f"Input mean: {np.mean(data):.3f}")
           print(f"Input std: {np.std(data):.3f}")
           print(f"NaN count: {np.sum(np.isnan(data))}")

       # Check condition number
       cov = np.cov(data, rowvar=False)
       cond = np.linalg.cond(cov)
       if verbose:
           print(f"Covariance condition number: {cond:.2e}")

       if cond > 1e10:
           warnings.warn(f"Ill-conditioned covariance matrix (cond={cond:.2e})")

       # Proceed with calculation
       result = np.linalg.solve(A, b)

       if verbose:
           print(f"Result range: [{np.min(result):.3f}, {np.max(result):.3f}]")
           print(f"Finite values: {np.sum(np.isfinite(result))}/{result.size}")

       return result
   ```

2. **Use error state context**:
   ```python
   with np.errstate(all='warn'):
       result = calculation(data)
   ```

3. **Add regularization**:
   ```python
   # Add small diagonal loading
   regularized = matrix + 1e-6 * np.eye(matrix.shape[0])
   result = np.linalg.solve(regularized, b)
   ```

4. **Check intermediate results**:
   ```python
   assert np.all(np.isfinite(intermediate)), "Intermediate result has NaN/Inf"
   assert np.all(variance > 0), "Variance must be positive"
   ```

---

## Architecture Decisions

### Key Design Principles

1. **Modularity**: One method = one module
   - **Rationale**: Easy to understand, test, and maintain
   - **Impact**: Each file is self-contained and can be used independently

2. **Optional Dependencies**: Graceful degradation
   - **Rationale**: Core functionality works with minimal dependencies (NumPy, SciPy)
   - **Impact**: Users can install only what they need; flags indicate availability

3. **Consistent API**: All methods have similar interfaces
   - **Rationale**: Easy to learn and switch between methods
   - **Impact**: User code is more portable

   ```python
   # Similar pattern across all methods
   result1, metric1, ... = ivd(data)
   result2, metric2, ... = tc(data)
   result3, metric3, ... = eivd(data)
   ```

4. **Performance over purity**: Optimize hot paths
   - **Rationale**: Some operations (ETCC search, IVD offset) are very slow
   - **Impact**: Added early termination, search limits; 98% speedup achieved

5. **Type hints**: Modern Python typing
   - **Rationale**: Better IDE support, clearer interfaces, catch errors early
   - **Impact**: All new code uses type hints; gradual migration of old code

6. **Dual language support**: English + Chinese
   - **Rationale**: Original MATLAB code by Chinese author; large Chinese user base
   - **Impact**: Documentation maintained in both languages

7. **Test-driven**: Comprehensive test coverage
   - **Rationale**: Prevent regressions, especially after optimizations
   - **Impact**: High confidence in correctness; tests double as examples

8. **Subpackages for complexity**: Fusion module structure
   - **Rationale**: Data fusion is complex with many components
   - **Impact**: Clean separation of concerns; easier to maintain

### Trade-offs Made

1. **Optimization vs. readability**:
   - **Decision**: Optimize hot paths, document why
   - **Trade-off**: Some optimized code is less readable
   - **Mitigation**: Extensive comments explaining optimization

2. **Flexibility vs. simplicity**:
   - **Decision**: Provide both simple functions and advanced options
   - **Trade-off**: More parameters to document
   - **Mitigation**: Sensible defaults, optional parameters

3. **Compatibility vs. modernization**:
   - **Decision**: Python 3.7+ (for type hints, pathlib, etc.)
   - **Trade-off**: No Python 2 support
   - **Mitigation**: Python 2 is deprecated anyway

4. **Completeness vs. focus**:
   - **Decision**: Include many methods (IVD, IVS, TC, EIVD, EC, ETCC, BTC, BTCH)
   - **Trade-off**: Large package, many dependencies to maintain
   - **Mitigation**: Optional dependencies, modular structure

---

## Performance Considerations

### Profiling Strategy

1. **Use cProfile for hot path identification**:
   ```python
   python -m cProfile -s cumulative script.py > profile.txt
   ```

2. **Focus optimization on top 5 functions** (Pareto principle)

3. **Benchmark before/after** every optimization

### Common Performance Issues

1. **Unnecessary loops**:
   ```python
   # Slow
   result = np.zeros(n)
   for i in range(n):
       result[i] = weight * data[i]

   # Fast
   result = weight * data  # Vectorized
   ```

2. **Repeated calculations**:
   ```python
   # Slow
   for i in range(n):
       x = np.std(data)  # Recalculated every iteration
       result[i] = data[i] / x

   # Fast
   x = np.std(data)  # Calculate once
   result = data / x  # Vectorized
   ```

3. **Exhaustive search**:
   ```python
   # Slow: Search all combinations
   for w1 in np.arange(0, 1, 0.01):
       for w2 in np.arange(0, 1, 0.01):
           for w3 in np.arange(0, 1, 0.01):
               if w1 + w2 + w3 == 1:
                   score = evaluate(w1, w2, w3)

   # Fast: Limit search space, early termination
   max_iter = 5000
   no_improvement_limit = 10
   for i, (w1, w2, w3) in enumerate(weight_combinations):
       if i > max_iter:
           break
       if no_improvement_count > no_improvement_limit:
           break
       score = evaluate(w1, w2, w3)
   ```

4. **Large matrix operations**:
   ```python
   # Consider condition number before inversion
   cond = np.linalg.cond(matrix)
   if cond > 1e10:
       # Use pseudoinverse or regularization
       inv = np.linalg.pinv(matrix)
   else:
       inv = np.linalg.inv(matrix)
   ```

### Performance Benchmarks (from PERFORMANCE_SUMMARY.md)

Recent optimizations achieved:
- **IVD**: 98% speedup (8.3s → 0.17s)
- **ETCC**: ~96% speedup (search space reduction)
- **EC**: Improved rescaling efficiency

### Memory Considerations

1. **Avoid unnecessary copies**:
   ```python
   # Creates copy
   data_copy = data.copy()

   # In-place operation (if safe)
   data -= np.mean(data, axis=0)
   ```

2. **Use generators for large sequences**:
   ```python
   # Memory intensive
   combinations = [(w1, w2, w3) for w1 in range(100) for w2 in range(100) ...]

   # Memory efficient
   combinations = ((w1, w2, w3) for w1 in range(100) for w2 in range(100) ...)
   ```

3. **Clear large arrays when done**:
   ```python
   large_array = np.zeros((10000, 10000))
   # ... use array ...
   del large_array  # Free memory
   ```

---

## Git Workflow

### Creating Feature Branch

```bash
# Ensure you're on main and up to date
git checkout main
git pull origin main

# Create feature branch with descriptive name
git checkout -b claude/add-new-method-<session-id>
```

### Making Commits

```bash
# Stage changes
git add collocation/new_method.py
git add tests/test_new_method.py
git add examples/new_method_example.py

# Commit with clear message
git commit -m "Add NewMethod for X-way collocation

- Implement core algorithm in new_method.py
- Add comprehensive tests covering edge cases
- Include runnable example demonstrating usage
- Update README with API documentation

Closes #42"

# Push to remote
git push -u origin claude/add-new-method-<session-id>
```

### Pull Request Guidelines

**PR Title**: Clear and descriptive
```
Add NewMethod for X-way collocation with uncertainty estimation
```

**PR Description Template**:
```markdown
## Summary
Brief description of what this PR does.

## Changes
- Change 1: Description
- Change 2: Description
- Change 3: Description

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Added new tests for new functionality
- [ ] Manually tested with example script

## Documentation
- [ ] Docstrings added/updated
- [ ] README updated
- [ ] Example added

## Performance
- [ ] No performance regression (or improvement documented)

## Checklist
- [ ] Code follows project conventions
- [ ] All tests pass locally
- [ ] No merge conflicts with main

## Related Issues
Closes #42
Related to #38
```

### Code Review Checklist

**For reviewers**:
- [ ] Code follows conventions in this guide
- [ ] Tests are comprehensive (normal, edge, error cases)
- [ ] Documentation is clear and complete
- [ ] No performance regressions
- [ ] Error handling is appropriate
- [ ] Type hints are used
- [ ] Examples are runnable

---

## Quick Reference

### Package Imports

```python
# Core methods
from collocation import ivd, ivs, tc, eivd, ec

# ETCC methods
from collocation import ETCC, TripleCollocation, SpatialMerging

# Bayesian methods
from collocation import BayesianTC, BayesianTCH, BTCH_He2020
from collocation import BAYESIAN_AVAILABLE, BAYESIAN_TCH_AVAILABLE

# Utilities
from collocation.utils import calculate_all_metrics, kge_objfun
from collocation.covariance import build_sigma_from_collocation
from collocation.fuse import estimate_bias

# Fusion module
from collocation.fusion import (
    solve_weights_ivw,
    solve_weights_gls,
    solve_weights_qp,
    fuse_fields,
)

# ELI application
from collocation import ELIProcessor, ELI_AVAILABLE
```

### File Locations

```
Core methods:              collocation/{ivd,ivs,tc,eivd,ec}.py
ETCC:                      collocation/etcc.py
Bayesian:                  collocation/bayesian_{tc,tch}.py
BTCH analytical:           collocation/btch_he2020.py
Utilities:                 collocation/utils.py
Fusion:                    collocation/fusion/*.py
Tests:                     tests/test_*.py
Examples:                  examples/*_example.py
Documentation:             README.md, *_README.md, docs/*.md
Setup:                     setup.py, requirements.txt, pytest.ini
```

### Command Reference

```bash
# Install package
pip install -e .

# Install with optional dependencies
pip install -e .[bayesian]

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=collocation --cov-report=html

# Run specific test
pytest tests/test_collocation.py::TestTC::test_tc_basic -v

# Run example
python examples/example_all_methods.py

# Profile code
python -m cProfile -s cumulative script.py > profile.txt

# Git workflow
git checkout -b claude/feature-name-<session-id>
git add <files>
git commit -m "Clear message"
git push -u origin claude/feature-name-<session-id>
```

### Common Patterns Quick Copy

**Input validation**:
```python
if not isinstance(data, np.ndarray):
    raise TypeError(f"data must be numpy array, got {type(data)}")

if data.ndim != 2 or data.shape[1] != 3:
    raise ValueError(f"data must be (n, 3), got {data.shape}")

if np.any(np.isnan(data)):
    raise ValueError("data contains NaN values")

if data.shape[0] < MIN_SAMPLES:
    warnings.warn(f"Sample size {data.shape[0]} < {MIN_SAMPLES}")
```

**Optional dependency**:
```python
try:
    import optional_package
    OPTIONAL_AVAILABLE = True
except ImportError:
    OPTIONAL_AVAILABLE = False
    optional_package = None

__all__ = ['function', 'OPTIONAL_AVAILABLE']

def function():
    if not OPTIONAL_AVAILABLE:
        raise ImportError("Install with: pip install optional_package")
    # Use optional_package
```

**Test structure**:
```python
class TestMethod:
    def test_basic(self):
        """Test basic functionality."""
        data = np.random.randn(200, 3)
        result = method(data)
        assert result.shape == (3, 3)

    def test_invalid_input(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            method(invalid_data)
```

---

## Appendix: Key Files Summary

### Critical Files to Understand

1. **`collocation/__init__.py`**: Package exports, availability flags
2. **`collocation/tc.py`**: Canonical example of method structure
3. **`tests/test_collocation.py`**: Comprehensive test examples
4. **`examples/example_all_methods.py`**: Usage patterns
5. **`setup.py`**: Package configuration
6. **`README.md`**: User documentation

### Documentation Files

- **README.md**: Main user documentation
- **CLAUDE.md**: This file - AI assistant guide
- **ELI_README.md**: ELI application guide
- **BAYESIAN_INTEGRATION_GUIDE.md**: Bayesian methods setup
- **PERFORMANCE_SUMMARY.md**: Optimization achievements
- **README_CN.md**: Chinese translation

### Configuration Files

- **setup.py**: Package installation, dependencies
- **requirements.txt**: Core dependencies
- **pytest.ini**: Test configuration
- **.gitignore**: Git exclusions

---

## Questions or Issues?

**If you encounter unclear patterns or need clarification**:

1. **Check existing code**: Find similar functionality and follow the pattern
2. **Read tests**: Tests often clarify expected behavior
3. **Review examples**: Examples demonstrate practical usage
4. **Check documentation**: README and guides have extensive information
5. **Look at recent commits**: Git history shows evolution and reasoning

**For AI assistants working on this codebase**: This guide covers the essential patterns, conventions, and workflows. When in doubt, follow existing patterns closely and maintain consistency with the established style.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-14
**Maintainer**: Development team
**Feedback**: Submit issues or PRs to improve this guide
