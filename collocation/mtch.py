"""
Multiplicative-error-based Three-Cornered Hat (MTCH) Method
=============================================================

This module implements the MTCH method for estimating RMSE between three or
more data products and the (unknown) truth, using a multiplicative error model.

The key difference from classical TCH is the error model:
- **Classical TCH**: Additive errors  X_i = theta + e_i
- **MTCH**: Multiplicative errors  Z_i = Theta * epsilon_i  (equivalently, log(Z_i) = log(Theta) + log(epsilon_i))

By log-transforming the data, MTCH reduces to a TCH problem in log space and
then converts the estimated log-space standard deviations back to the original
scale by multiplying by the sample mean.  This makes MTCH particularly suited
to variables that are strictly positive and whose relative (percentage) errors
are more stationary than their absolute errors, e.g., evapotranspiration,
precipitation, or soil-moisture fluxes.

The optimization follows Ferreira et al. (2016): given the covariance matrix
S of the pairwise log-differences, the algorithm finds the last column
x = R[:,N-1] of the N×N error covariance matrix R by minimising an objective
function (Eq. 8 of Ferreira et al.) subject to a positive-semi-definite
constraint (Eq. 9 of Ferreira et al.).  The remaining elements of R are then
recovered from S analytically.

References
----------
.. [1] Li, X. (2015). Improving global evapotranspiration estimation by
       merging four products based on multiplicative error model and the
       three-cornered hat method. (Unpublished / conference proceedings)

.. [2] Xu, T., Guo, Z., Xia, Y., et al. (2019). Evaluation of twelve
       evapotranspiration products from machine learning, remote sensing
       and land surface models over conterminous United States.
       Journal of Hydrology, 578, 124105.
       https://doi.org/10.1016/j.jhydrol.2019.124105

.. [3] Ferreira, V.G., Montecino, H.D.C., Yakubu, C.I., Heck, B. (2016).
       Uncertainties of the Gravity Recovery and Climate Experiment
       time-variable gravity-field solutions based on three-cornered hat
       method. J. Appl. Remote Sens, 10, 015015.
       https://doi.org/10.1117/1.JRS.10.015015

Original MATLAB implementation: X. Li, katelisomeone@gmail.com (Nov 2025)
Python port: Collocation-Analysis project
"""

import numpy as np
import warnings
from typing import Optional, Tuple, Union, Dict

from scipy.optimize import minimize
from scipy.linalg import det


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

MIN_SAMPLES: int = 50
"""Minimum recommended number of time samples."""

MIN_PRODUCTS: int = 3
"""Minimum number of data products required."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _objective(r: np.ndarray, S: np.ndarray, n: int) -> float:
    """
    Objective function (Eq. 8, Ferreira et al. 2016).

    Parameters
    ----------
    r : np.ndarray
        Optimisation variable of length N = n+1.
        r[:n]  = R[0:n, N-1]  (cross-covariances with reference product)
        r[n]   = R[N-1, N-1]  (variance of reference product in log space)
    S : np.ndarray
        Covariance matrix of log-differences, shape (n, n).
    n : int
        N - 1 (number of non-reference products).

    Returns
    -------
    float
        Scalar objective value.
    """
    rN = r[n]  # R(N,N) in MATLAB notation (0-indexed: r[n])

    f = 0.0
    # Sum of squared cross-covariance terms
    for j in range(n):
        f += r[j] ** 2
        for k in range(j + 1, n):
            f += (S[j, k] - rN + r[j] + r[k]) ** 2

    det_S = det(S)
    if det_S <= 0:
        return np.inf
    return f / (det_S ** (2.0 / n))


def _constraint(r: np.ndarray, S: np.ndarray, S_inv: np.ndarray,
                 n: int, det_S: float) -> float:
    """
    Inequality constraint (Eq. 9, Ferreira et al. 2016).

    scipy's SLSQP requires the constraint in the form g(x) >= 0.
    MATLAB's fmincon uses c <= 0, so this function returns the negated
    MATLAB constraint, making it positive when feasible.

    Parameters
    ----------
    r : np.ndarray
        Optimisation variable, length N.
    S : np.ndarray
        Covariance matrix of log-differences, shape (n, n).
    S_inv : np.ndarray
        Pre-computed inverse of S, shape (n, n).
    n : int
        N - 1.
    det_S : float
        Pre-computed determinant of S.

    Returns
    -------
    float
        Positive when feasible (constraint satisfied).
    """
    fr = r[:n]       # First n elements
    rN = r[n]        # Last element R(N,N)
    u = np.ones(n)

    diff = fr - rN * u  # shape (n,)
    val = (rN - diff @ S_inv @ diff) / (det_S ** (1.0 / n))
    return val


def _reconstruct_R(x: np.ndarray, S: np.ndarray, N: int) -> np.ndarray:
    """
    Recover the full N×N error covariance matrix R from the optimised
    last column x and the difference covariance matrix S.

    The relationship is (for i,j in 0..N-2):
        S[i,j] = R[i,j] + R[N-1,N-1] - R[i,N-1] - R[j,N-1]
    Rearranging:
        R[i,j] = S[i,j] - R[N-1,N-1] + R[i,N-1] + R[j,N-1]

    Parameters
    ----------
    x : np.ndarray
        Optimised last column of R, shape (N,).
    S : np.ndarray
        Covariance matrix of log-differences, shape (N-1, N-1).
    N : int
        Total number of products.

    Returns
    -------
    R : np.ndarray
        Reconstructed error covariance matrix, shape (N, N).
    """
    n = N - 1
    R = np.zeros((N, N))

    # Fill last column and row with optimised x
    R[:, N - 1] = x
    R[N - 1, :] = x

    # Recover upper-triangle block (including diagonal) for indices 0..n-1
    R_NN = x[N - 1]  # R[N-1, N-1]
    for i in range(n):
        for j in range(i, n):
            R[i, j] = S[i, j] - R_NN + x[i] + x[j]
            R[j, i] = R[i, j]  # Symmetry

    return R


# ---------------------------------------------------------------------------
# Public API – functional interface
# ---------------------------------------------------------------------------

def mtch(
    data: np.ndarray,
    reference_idx: Optional[int] = None,
    tol: float = 2e-10,
    maxiter: int = 2000,
) -> np.ndarray:
    """
    Multiplicative-error Three-Cornered Hat (MTCH) error estimation.

    Estimates the RMSE of each data product relative to the unknown truth,
    using a multiplicative error model in log space.  Suitable for strictly
    positive geophysical variables (e.g. ET, precipitation, runoff).

    Parameters
    ----------
    data : np.ndarray
        Input data matrix, shape (n_samples, n_products).
        All values **must be strictly positive** (> 0) because a log
        transform is applied internally.
        n_samples >= 50 and n_products >= 3 are recommended.
    reference_idx : int, optional
        Index of the product used as the internal reference for forming
        pairwise log-differences.  Defaults to the **last** product
        (index n_products - 1), matching the original MATLAB behaviour.
        Changing the reference should not affect the results when the
        method converges, but may affect numerical conditioning.
    tol : float, default=2e-10
        Convergence tolerance passed to the SLSQP optimiser (both ``ftol``
        and ``eps``).
    maxiter : int, default=2000
        Maximum number of optimiser iterations.

    Returns
    -------
    rmse : np.ndarray
        Shape (n_products,).  RMSE of each product relative to truth,
        in the **original (non-log) units** of ``data``.
        Computed as ``std_log * mean(data, axis=0)`` where ``std_log`` is
        the per-product standard deviation of log-errors.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D, contains non-positive values, or has fewer
        than 3 columns.
    TypeError
        If ``data`` is not a numpy array.

    Warnings
    --------
    UserWarning
        If n_samples < MIN_SAMPLES (50): results may be unreliable.
    UserWarning
        If the optimiser does not converge.

    Examples
    --------
    Three products with multiplicative errors:

    >>> import numpy as np
    >>> from collocation import mtch
    >>>
    >>> np.random.seed(42)
    >>> n = 300
    >>> true_signal = 5 + np.random.exponential(scale=2, size=n)  # Positive
    >>> # Multiplicative noise: Z_i = Theta * exp(e_i)
    >>> z1 = true_signal * np.exp(0.10 * np.random.randn(n))
    >>> z2 = true_signal * np.exp(0.15 * np.random.randn(n))
    >>> z3 = true_signal * np.exp(0.20 * np.random.randn(n))
    >>> data = np.column_stack([z1, z2, z3])
    >>>
    >>> rmse = mtch(data)
    >>> print(f"RMSE estimates: {rmse}")

    Notes
    -----
    **Error model:**

    .. math::

        Z_i = \\Theta \\cdot \\epsilon_i, \\quad
        \\log Z_i = \\log \\Theta + \\log \\epsilon_i

    After the log transform the problem becomes an additive TCH in log space.
    The error covariance matrix R (in log space) is recovered via constrained
    optimisation.  The diagonal entries R[i,i] give the log-error variances,
    and the per-product RMSE in original units is approximated as:

    .. math::

        \\text{RMSE}_i \\approx \\sqrt{R_{ii}} \\cdot \\bar{Z}_i

    This first-order approximation is accurate when log-errors are small
    (i.e., relative errors are less than ~30 %).

    See Also
    --------
    MTCH : Class-based API.
    tc : Classical (additive-error) triple collocation.
    btch_he2020 : Bayesian Three-Cornered Hat.

    References
    ----------
    .. [1] Xu et al. (2019). J. Hydrology, 578, 124105.
    .. [2] Ferreira et al. (2016). J. Appl. Remote Sens, 10, 015015.
    """
    # ------------------------------------------------------------------
    # 1. Input validation
    # ------------------------------------------------------------------
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be a numpy array, got {type(data)}")

    if data.ndim != 2:
        raise ValueError(
            f"data must be 2-D (n_samples, n_products), got shape {data.shape}"
        )

    n_samples, N = data.shape

    if N < MIN_PRODUCTS:
        raise ValueError(
            f"data must have at least {MIN_PRODUCTS} columns (products), "
            f"got {N}."
        )

    if np.any(data <= 0):
        raise ValueError(
            "MTCH requires strictly positive data (all values > 0) because "
            "a log transform is applied.  Check for zero or negative values."
        )

    if np.any(~np.isfinite(data)):
        raise ValueError("data contains NaN or infinite values.")

    if n_samples < MIN_SAMPLES:
        warnings.warn(
            f"Sample size ({n_samples}) is less than the recommended minimum "
            f"({MIN_SAMPLES}).  Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # 2. Log transform and differencing
    # ------------------------------------------------------------------
    if reference_idx is None:
        reference_idx = N - 1  # Default: last product (MATLAB behaviour)

    if not (0 <= reference_idx < N):
        raise ValueError(
            f"reference_idx must be in [0, {N - 1}], got {reference_idx}."
        )

    X = np.log(data)  # (n_samples, N)

    # Build difference matrix Y: columns are X[:,i] - X[:,reference_idx]
    # for all i != reference_idx
    non_ref_cols = [i for i in range(N) if i != reference_idx]
    Y = X[:, non_ref_cols] - X[:, [reference_idx]]  # (n_samples, N-1)

    n = N - 1  # Number of difference series

    # Covariance of differences (N-1) x (N-1)
    if n == 1:
        S = np.atleast_2d(np.var(Y, ddof=1))
    else:
        S = np.cov(Y, rowvar=False)  # (n, n)

    # ------------------------------------------------------------------
    # 3. Initial conditions  [MATLAB: R(N,N) = (2*u*inv(S)*u')^(-1)]
    # ------------------------------------------------------------------
    u = np.ones(n)

    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        warnings.warn(
            "Covariance matrix S is singular; using pseudoinverse.",
            UserWarning,
            stacklevel=2,
        )
        S_inv = np.linalg.pinv(S)

    det_S = float(det(S))
    if det_S <= 0:
        warnings.warn(
            "Determinant of S is non-positive.  Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )
        det_S = max(det_S, 1e-300)

    R_NN_init = 1.0 / (2.0 * u @ S_inv @ u)

    # x0 = R[:, reference_idx] with the reference's variance initialised
    x0 = np.zeros(N)
    x0[N - 1] = R_NN_init  # Reference product is always placed last in x

    # ------------------------------------------------------------------
    # 4. Constrained optimisation
    # ------------------------------------------------------------------
    constraints = {
        "type": "ineq",
        "fun": _constraint,
        "args": (S, S_inv, n, det_S),
    }

    opt_result = minimize(
        _objective,
        x0,
        args=(S, n),
        method="SLSQP",
        constraints=constraints,
        options={
            "ftol": tol,
            "eps": tol,
            "maxiter": maxiter,
            "disp": False,
        },
    )

    if not opt_result.success:
        warnings.warn(
            f"MTCH optimiser did not converge: {opt_result.message}. "
            "Results may be inaccurate.",
            UserWarning,
            stacklevel=2,
        )

    x = opt_result.x  # Optimised R[:, reference_idx]

    # ------------------------------------------------------------------
    # 5. Reconstruct full R matrix in log-error space
    # ------------------------------------------------------------------
    R_log = _reconstruct_R(x, S, N)

    # Extract diagonal (log-error variances).
    # The reconstruction uses the internal ordering where the reference
    # product's column is placed last.  We now need to map back to the
    # original product ordering.
    log_variances_internal = np.diag(R_log)  # Length N, internal order

    # Map internal index → original product index
    # internal[0..n-1] = non_ref_cols[0..n-1], internal[N-1] = reference_idx
    internal_to_original = non_ref_cols + [reference_idx]
    log_variances = np.empty(N)
    for internal_i, orig_i in enumerate(internal_to_original):
        log_variances[orig_i] = log_variances_internal[internal_i]

    # Guard against tiny negative values from numerical noise
    log_variances = np.maximum(log_variances, 0.0)
    std_log = np.sqrt(log_variances)  # Per-product log-space std dev

    # ------------------------------------------------------------------
    # 6. Convert to original-scale RMSE
    # ------------------------------------------------------------------
    mean_Z = np.nanmean(data, axis=0)  # (N,)
    rmse = std_log * mean_Z

    return rmse


# ---------------------------------------------------------------------------
# Public API – class-based interface
# ---------------------------------------------------------------------------

class MTCH:
    """
    Multiplicative-error Three-Cornered Hat (MTCH) – class-based API.

    Provides a convenient object-oriented interface around :func:`mtch`,
    mirroring the style of :class:`~collocation.btch_he2020.BTCH_He2020`.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix, shape (n_samples, n_products).
        All values must be strictly positive.
    reference_idx : int, optional
        Reference product index for internal log-differencing.
        Defaults to the last product (index n_products - 1).

    Attributes
    ----------
    data : np.ndarray
        Copy of the input data.
    n_samples : int
        Number of time samples.
    n_products : int
        Number of data products.
    rmse : np.ndarray or None
        RMSE estimates after calling :meth:`fit`.
    log_variances : np.ndarray or None
        Log-space error variances after calling :meth:`fit`.
    weights : np.ndarray or None
        Inverse-RMSE fusion weights after calling :meth:`compute_weights`.
    fused_product : np.ndarray or None
        Weighted-average fused product after calling :meth:`fuse`.

    Examples
    --------
    >>> import numpy as np
    >>> from collocation import MTCH
    >>>
    >>> np.random.seed(0)
    >>> n, N = 200, 3
    >>> theta = 10 + np.random.exponential(2, n)
    >>> data = np.column_stack([
    ...     theta * np.exp(0.10 * np.random.randn(n)),
    ...     theta * np.exp(0.20 * np.random.randn(n)),
    ...     theta * np.exp(0.30 * np.random.randn(n)),
    ... ])
    >>>
    >>> model = MTCH(data)
    >>> model.fit()
    >>> model.compute_weights()
    >>> fused = model.fuse()
    >>> model.summary()
    """

    def __init__(
        self,
        data: np.ndarray,
        reference_idx: Optional[int] = None,
    ) -> None:
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError(
                f"data must be 2-D (n_samples, n_products), got {data.shape}"
            )
        self.data = data
        self.n_samples, self.n_products = data.shape
        self.reference_idx = reference_idx

        self.rmse: Optional[np.ndarray] = None
        self.log_variances: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.fused_product: Optional[np.ndarray] = None

    def fit(
        self,
        tol: float = 2e-10,
        maxiter: int = 2000,
    ) -> "MTCH":
        """
        Run MTCH optimisation and compute per-product RMSE.

        Parameters
        ----------
        tol : float, default=2e-10
            Optimiser convergence tolerance.
        maxiter : int, default=2000
            Maximum optimiser iterations.

        Returns
        -------
        self
            Returns self to allow method chaining.
        """
        self.rmse = mtch(
            self.data,
            reference_idx=self.reference_idx,
            tol=tol,
            maxiter=maxiter,
        )
        self.log_variances = (self.rmse / np.nanmean(self.data, axis=0)) ** 2
        return self

    def compute_weights(self) -> np.ndarray:
        """
        Compute inverse-RMSE fusion weights.

        Weight of product i is proportional to 1/RMSE_i² (inverse-variance
        weighting in the approximation that RMSE ≈ error std dev).

        Returns
        -------
        weights : np.ndarray
            Normalised weights summing to 1, shape (n_products,).

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if self.rmse is None:
            raise RuntimeError("Call fit() before compute_weights().")

        inv_var = 1.0 / np.maximum(self.rmse ** 2, 1e-300)
        self.weights = inv_var / inv_var.sum()
        return self.weights

    def fuse(self) -> np.ndarray:
        """
        Fuse products using the computed weights.

        Returns
        -------
        fused : np.ndarray
            Weighted average, shape (n_samples,).

        Raises
        ------
        RuntimeError
            If :meth:`compute_weights` has not been called yet.
        """
        if self.weights is None:
            raise RuntimeError("Call compute_weights() before fuse().")

        self.fused_product = self.data @ self.weights
        return self.fused_product

    def run(
        self,
        tol: float = 2e-10,
        maxiter: int = 2000,
    ) -> Dict[str, np.ndarray]:
        """
        Run the full MTCH pipeline: fit → weights → fuse.

        Parameters
        ----------
        tol : float, default=2e-10
            Optimiser tolerance.
        maxiter : int, default=2000
            Maximum optimiser iterations.

        Returns
        -------
        results : dict
            Keys: ``'rmse'``, ``'log_variances'``, ``'weights'``, ``'fused'``.
        """
        self.fit(tol=tol, maxiter=maxiter)
        self.compute_weights()
        self.fuse()
        return {
            "rmse": self.rmse,
            "log_variances": self.log_variances,
            "weights": self.weights,
            "fused": self.fused_product,
        }

    def summary(self) -> None:
        """Print a formatted summary of MTCH results."""
        if self.rmse is None:
            print("No results yet.  Call fit() first.")
            return

        print("\n" + "=" * 70)
        print("MTCH (Multiplicative-error Three-Cornered Hat) Results")
        print("=" * 70)
        print(f"Number of products : {self.n_products}")
        print(f"Number of samples  : {self.n_samples}")

        print("\nError Estimates:")
        print("-" * 70)
        for i in range(self.n_products):
            print(
                f"  Product {i}: RMSE = {self.rmse[i]:.6g}  "
                f"(log-space std = {np.sqrt(self.log_variances[i]):.6f})"
            )

        if self.weights is not None:
            print("\nFusion Weights (inverse-RMSE²):")
            print("-" * 70)
            for i in range(self.n_products):
                print(
                    f"  Product {i}: w = {self.weights[i]:.6f} "
                    f"({self.weights[i] * 100:.2f} %)"
                )
            print(f"\n  Weight sum = {self.weights.sum():.6f} (should be 1.0)")

        print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "mtch",
    "MTCH",
    "MIN_SAMPLES",
    "MIN_PRODUCTS",
]
