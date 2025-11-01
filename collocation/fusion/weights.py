"""
Weight Solvers for Data Fusion
================================

Implements optimal weight calculation for multi-source data fusion:

1. **Inverse Variance Weighting (IVW)**: w_k ∝ 1/MSE_k
2. **Generalized Least Squares (GLS/BLUE)**: w = Σ^{-1}1 / (1^T Σ^{-1}1)
3. **Constrained Quadratic Programming (QP)**: min w^T Σ w s.t. constraints

All solvers handle heteroscedastic errors, degeneracy, and numerical stability.

Examples
--------
>>> import numpy as np
>>> from collocation.fusion import solve_weights_ivw, solve_weights_gls
>>>
>>> # Simple IVW
>>> mse = np.array([0.5, 0.3, 0.8])
>>> w = solve_weights_ivw(mse)
>>> print(w)  # [0.35, 0.58, 0.07]
>>>
>>> # GLS with full covariance
>>> Sigma = np.array([[0.5, 0.1, 0], [0.1, 0.3, 0], [0, 0, 0.8]])
>>> w = solve_weights_gls(Sigma)
>>> print(w.sum())  # 1.0
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import linalg


def solve_weights_ivw(
    mse: np.ndarray,
    sum_to_one: bool = True,
    bounds: Tuple[float, float] | None = None,
    softmax_temp: float | None = None,
    min_weight: float = 0.0,
) -> np.ndarray:
    """
    Compute inverse variance weights: w_k ∝ 1/MSE_k.

    Handles heteroscedastic MSE fields. Optionally applies softmax or clipping
    to avoid weight degeneracy when one product is much better than others.

    Parameters
    ----------
    mse : np.ndarray
        Mean squared errors, shape (K,) or (..., K) for spatially varying MSE.
    sum_to_one : bool, default=True
        Normalize weights to sum to 1.
    bounds : tuple of (float, float), optional
        Global (w_min, w_max) bounds. Applied after normalization.
    softmax_temp : float, optional
        Temperature for softmax normalization: w_k = exp(-MSE_k/T).
        Higher values → more uniform weights. If None, uses direct 1/MSE.
    min_weight : float, default=0.0
        Minimum weight threshold. Values below this are set to zero and
        weights are renormalized.

    Returns
    -------
    np.ndarray
        Optimal weights, same shape as `mse`.

    Raises
    ------
    ValueError
        If all MSE values are non-positive or infinite.

    Examples
    --------
    >>> mse = np.array([0.5, 0.3, 0.8])
    >>> w = solve_weights_ivw(mse)
    >>> np.allclose(w.sum(), 1.0)
    True
    >>> w[1] > w[0] > w[2]  # Lower MSE → higher weight
    True
    """
    mse = np.asarray(mse, dtype=float)
    if mse.ndim == 0:
        raise ValueError("MSE must be at least 1-dimensional")

    # Check validity
    if not np.all(np.isfinite(mse)):
        raise ValueError("MSE contains non-finite values")
    if not np.all(mse > 0):
        raise ValueError("MSE must be strictly positive")

    # Compute weights
    if softmax_temp is not None:
        # Softmax approach: w_k = exp(-MSE_k / T)
        temp = float(softmax_temp)
        if temp <= 0:
            raise ValueError("softmax_temp must be positive")
        weights = np.exp(-mse / temp)
    else:
        # Direct inverse: w_k = 1 / MSE_k
        weights = 1.0 / mse

    # Apply minimum weight threshold
    if min_weight > 0:
        weights = np.where(weights < min_weight, 0.0, weights)

    # Normalize
    if sum_to_one:
        weight_sum = weights.sum(axis=-1, keepdims=True)
        # Handle edge case where all weights are zero
        weight_sum = np.where(weight_sum == 0, 1.0, weight_sum)
        weights = weights / weight_sum

    # Apply bounds if specified
    if bounds is not None:
        w_min, w_max = bounds
        weights = np.clip(weights, w_min, w_max)
        # Renormalize after clipping if sum_to_one
        if sum_to_one:
            weight_sum = weights.sum(axis=-1, keepdims=True)
            weight_sum = np.where(weight_sum == 0, 1.0, weight_sum)
            weights = weights / weight_sum

    return weights


def solve_weights_gls(
    Sigma: np.ndarray,
    sum_to_one: bool = True,
    bounds: Tuple[float, float] | None = None,
    diagonal_loading: float | None = None,
) -> np.ndarray:
    """
    Compute GLS/BLUE weights: w = Σ^{-1}1 / (1^T Σ^{-1}1).

    Uses Cholesky decomposition for numerical stability. Adds diagonal loading
    if the matrix is near-singular.

    Parameters
    ----------
    Sigma : np.ndarray
        Error covariance matrix, shape (K, K) or (..., K, K) for spatially
        varying covariance.
    sum_to_one : bool, default=True
        Enforce sum-to-one constraint (standard BLUE).
    bounds : tuple of (float, float), optional
        After computing GLS weights, clip to [w_min, w_max] and renormalize.
    diagonal_loading : float, optional
        Add λI to Sigma before inversion (ridge regularization).
        If None, automatically adds small loading if ill-conditioned.

    Returns
    -------
    np.ndarray
        GLS optimal weights, shape (K,) or (..., K).

    Raises
    ------
    ValueError
        If Sigma is not positive definite after loading.
    LinAlgError
        If Cholesky decomposition fails.

    Examples
    --------
    >>> Sigma = np.array([[0.5, 0.1], [0.1, 0.3]])
    >>> w = solve_weights_gls(Sigma)
    >>> np.allclose(w.sum(), 1.0)
    True

    Notes
    -----
    The BLUE solution minimizes variance: Var(x̂) = (1^T Σ^{-1} 1)^{-1}.
    For spatially varying Sigma, computes weights pixel-by-pixel.
    """
    Sigma = np.asarray(Sigma, dtype=float)

    # Handle scalar case
    if Sigma.ndim == 0:
        raise ValueError("Sigma must be at least 2-dimensional")

    # Extract last two dimensions for matrix operations
    if Sigma.ndim == 2:
        K = Sigma.shape[0]
        if Sigma.shape[1] != K:
            raise ValueError("Sigma must be square")
        batch_shape = ()
        Sigma_batch = Sigma[None, :, :]
    else:
        batch_shape = Sigma.shape[:-2]
        K = Sigma.shape[-1]
        if Sigma.shape[-2] != K:
            raise ValueError("Last two dimensions of Sigma must match")
        Sigma_batch = Sigma.reshape(-1, K, K)

    n_batch = Sigma_batch.shape[0]
    weights_batch = np.zeros((n_batch, K), dtype=float)

    # Vector of ones
    ones = np.ones(K, dtype=float)

    for i in range(n_batch):
        S = Sigma_batch[i]

        # Symmetrize
        S = 0.5 * (S + S.T)

        # Check condition number and add diagonal loading if needed
        if diagonal_loading is not None:
            lam = float(diagonal_loading)
        else:
            # Auto-detect: if condition number > 1e8, add small loading
            try:
                cond = np.linalg.cond(S)
                if cond > 1e8 or not np.isfinite(cond):
                    lam = 1e-6 * np.trace(S) / K
                else:
                    lam = 0.0
            except linalg.LinAlgError:
                lam = 1e-6 * np.trace(S) / K

        if lam > 0:
            S = S + lam * np.eye(K)

        # Solve via Cholesky: S = L L^T
        try:
            L = linalg.cholesky(S, lower=True)
        except linalg.LinAlgError:
            # Fallback: add more aggressive loading
            lam_fallback = 1e-4 * np.trace(S) / K
            S = S + lam_fallback * np.eye(K)
            try:
                L = linalg.cholesky(S, lower=True)
            except linalg.LinAlgError:
                raise ValueError(
                    f"Sigma is not positive definite even after loading (batch {i})"
                )

        # Solve L y = 1 → y = L^{-1} 1
        y = linalg.solve_triangular(L, ones, lower=True)

        if sum_to_one:
            # w = S^{-1} 1 / (1^T S^{-1} 1)
            # y = L^{-1} 1 → L^T w_unnorm = y → w_unnorm = (L^T)^{-1} y
            w_unnorm = linalg.solve_triangular(L.T, y, lower=False)
            norm_factor = w_unnorm.sum()
            if abs(norm_factor) < 1e-15:
                # Degenerate case: uniform weights
                weights_batch[i] = 1.0 / K
            else:
                weights_batch[i] = w_unnorm / norm_factor
        else:
            # Just w = S^{-1} 1
            weights_batch[i] = linalg.solve_triangular(L.T, y, lower=False)

    # Reshape back to original batch shape
    if batch_shape == ():
        weights = weights_batch[0]
    else:
        weights = weights_batch.reshape(*batch_shape, K)

    # Apply bounds if specified
    if bounds is not None:
        w_min, w_max = bounds
        weights = np.clip(weights, w_min, w_max)
        if sum_to_one:
            weight_sum = weights.sum(axis=-1, keepdims=True)
            weight_sum = np.where(weight_sum == 0, 1.0, weight_sum)
            weights = weights / weight_sum

    return weights


def solve_weights_qp(
    Sigma: np.ndarray,
    A_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    A_ineq: np.ndarray | None = None,
    b_ineq: np.ndarray | None = None,
    bounds: Tuple[float, float] | None = None,
    solver: str = "quadprog",
) -> np.ndarray:
    """
    Solve constrained quadratic program: min_w (1/2) w^T Σ w.

    Subject to:
        A_eq w = b_eq     (equality constraints)
        A_ineq w ≤ b_ineq (inequality constraints)
        w_min ≤ w ≤ w_max (box bounds)

    Uses quadprog or cvxopt backend (quadprog is lighter and sufficient for
    convex QP).

    Parameters
    ----------
    Sigma : np.ndarray
        Positive semi-definite covariance matrix, shape (K, K).
    A_eq : np.ndarray, optional
        Equality constraint matrix, shape (n_eq, K). Common: [[1,1,...,1]]
        for sum-to-one.
    b_eq : np.ndarray, optional
        Equality RHS, shape (n_eq,). For sum-to-one: [1.0].
    A_ineq : np.ndarray, optional
        Inequality constraint matrix, shape (n_ineq, K).
    b_ineq : np.ndarray, optional
        Inequality RHS, shape (n_ineq,).
    bounds : tuple of (float, float), optional
        Box constraints (w_min, w_max) applied to all weights.
    solver : str, default="quadprog"
        Backend solver. Options: "quadprog", "cvxopt", "osqp".

    Returns
    -------
    np.ndarray
        Optimal weights, shape (K,).

    Raises
    ------
    ImportError
        If required solver library is not installed.
    ValueError
        If problem is infeasible or unbounded.

    Examples
    --------
    >>> # Minimize variance with sum-to-one and bounds
    >>> Sigma = np.array([[0.5, 0.1], [0.1, 0.3]])
    >>> A_eq = np.array([[1.0, 1.0]])
    >>> b_eq = np.array([1.0])
    >>> w = solve_weights_qp(Sigma, A_eq, b_eq, bounds=(0.0, 1.0))
    >>> np.allclose(w.sum(), 1.0)
    True
    >>> np.all((w >= 0) & (w <= 1))
    True

    Notes
    -----
    For standard GLS (sum-to-one only), use `solve_weights_gls` for speed.
    Use QP when additional physics constraints are needed.
    """
    Sigma = np.asarray(Sigma, dtype=float)
    K = Sigma.shape[0]

    if Sigma.ndim != 2 or Sigma.shape[1] != K:
        raise ValueError("Sigma must be a square 2D matrix")

    # Symmetrize
    Sigma = 0.5 * (Sigma + Sigma.T)

    # Regularize if needed
    eigvals = np.linalg.eigvalsh(Sigma)
    if np.any(eigvals <= 0):
        eps = max(1e-9, -eigvals.min() + 1e-9)
        Sigma = Sigma + eps * np.eye(K)

    solver = solver.lower()

    if solver == "quadprog":
        try:
            import quadprog
        except ImportError:
            raise ImportError(
                "quadprog not installed. Install with: pip install quadprog"
            )

        # quadprog format: min 0.5 x^T G x - a^T x
        # s.t. C^T x >= b (note: >= not <=)
        # We have: min 0.5 w^T Σ w (a=0)
        # Equality: A_eq w = b_eq → add as pairs of inequalities
        # Inequality: A_ineq w <= b_ineq → convert to -A_ineq w >= -b_ineq

        G = Sigma
        a = np.zeros(K)

        # Build constraint matrix C and vector b
        # Format: [C^T] w >= [b]
        C_list = []
        b_list = []

        # Equality constraints as pairs
        if A_eq is not None and b_eq is not None:
            A_eq = np.atleast_2d(A_eq)
            b_eq = np.atleast_1d(b_eq)
            # A_eq w = b_eq → A_eq w >= b_eq AND -A_eq w >= -b_eq
            C_list.append(A_eq.T)
            b_list.append(b_eq)
            C_list.append(-A_eq.T)
            b_list.append(-b_eq)

        # Inequality constraints: A_ineq w <= b_ineq → -A_ineq w >= -b_ineq
        if A_ineq is not None and b_ineq is not None:
            A_ineq = np.atleast_2d(A_ineq)
            b_ineq = np.atleast_1d(b_ineq)
            C_list.append(-A_ineq.T)
            b_list.append(-b_ineq)

        # Box bounds: w_min <= w <= w_max
        if bounds is not None:
            w_min, w_max = bounds
            # w >= w_min
            C_list.append(np.eye(K))
            b_list.append(np.full(K, w_min))
            # w <= w_max → -w >= -w_max
            C_list.append(-np.eye(K))
            b_list.append(np.full(K, -w_max))

        if C_list:
            C = np.hstack(C_list)
            b = np.concatenate(b_list)
        else:
            # No constraints: unconstrained QP
            C = np.zeros((K, 0))
            b = np.zeros(0)

        try:
            # Note: quadprog uses meq parameter for number of equality constraints
            # We encoded equalities as pairs, so meq = 2 * n_eq
            n_eq = 0 if A_eq is None else A_eq.shape[0]
            meq = 2 * n_eq

            sol = quadprog.solve_qp(G, a, C, b, meq=meq)
            weights = sol[0]
        except Exception as e:
            raise ValueError(f"QP solver failed: {e}")

    elif solver in {"cvxopt", "osqp"}:
        raise NotImplementedError(
            f"Solver '{solver}' not yet implemented. Use 'quadprog' or install cvxopt/osqp."
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    return weights


def compute_fusion_variance(
    Sigma: np.ndarray,
    weights: np.ndarray,
) -> float | np.ndarray:
    """
    Compute variance of the fused estimate: Var(x̂) = w^T Σ w.

    Parameters
    ----------
    Sigma : np.ndarray
        Error covariance, shape (K, K) or (..., K, K).
    weights : np.ndarray
        Fusion weights, shape (K,) or (..., K).

    Returns
    -------
    float or np.ndarray
        Fused variance, scalar or shape (...,).

    Examples
    --------
    >>> Sigma = np.array([[0.5, 0.1], [0.1, 0.3]])
    >>> w = solve_weights_gls(Sigma)
    >>> var_fused = compute_fusion_variance(Sigma, w)
    >>> var_fused < min(np.diag(Sigma))  # Fusion reduces variance
    True
    """
    Sigma = np.asarray(Sigma, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Var(x̂) = w^T Σ w
    if Sigma.ndim == 2:
        # Single covariance matrix
        var = weights @ Sigma @ weights
    else:
        # Batched: (..., K, K) and (..., K)
        var = np.einsum("...i,...ij,...j->...", weights, Sigma, weights)

    return var
