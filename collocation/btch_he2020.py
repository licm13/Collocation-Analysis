"""
Bayesian Three-Cornered Hat (BTCH) - He et al. (2020) Implementation

This module implements the BTCH method from He et al. (2020), which combines
classical Three-Cornered Hat (TCH) error variance estimation with an analytical
Bayesian weighting formula.

**Key Differences from Other Bayesian Methods:**

1. **BTCH_He2020 (this module)**:
   - Uses TCH for point estimation of error variances
   - Applies analytical Bayesian weight formula (no MCMC)
   - Fast, deterministic, no sampling required
   - No uncertainty quantification (point estimates only)
   - Minimal dependencies (NumPy only)

2. **BayesianTCH (Zwieback et al.)**:
   - Uses MCMC (PyMC3) for full posterior distribution
   - Provides complete uncertainty quantification
   - Assumes constant error variances
   - Requires PyMC3/Theano dependencies

3. **BayesianTC (Zwieback et al.)**:
   - Uses MCMC (PyMC3) for full posterior distribution
   - Models time-varying error structures
   - More complex, computationally intensive
   - Full Bayesian inference with posteriors

**Method Overview:**

The BTCH method consists of two steps:

1. **Step 1 - TCH Error Variance Estimation:**
   - Classical uncorrelated TCH: Assumes independent errors
   - Regularized correlated TCH: Approximates correlated errors with PSD projection
   - Output: Point estimates of error variances (diagonal of R)

2. **Step 2 - Analytical Bayesian Weights:**
   - Applies the weight formula from maximizing joint likelihood:
     w_k = (∏_{i≠k} σ²_i) / (∑_j ∏_{i≠j} σ²_i)
   - Implemented in log-space for numerical stability
   - Output: Optimal fusion weights

**References:**
- He, Q., Wang, M., & Liu, K. (2020). A Bayesian Three-Cornered Hat (BTCH) method:
  improving the terrestrial evapotranspiration estimation. Hydrology and Earth System Sciences.

Author: Claude (Integration of He et al. 2020 replication into main package)
Date: 2025
"""

import numpy as np
import warnings
from typing import Tuple, Optional, Dict, Union


# ==============================================================================
# Core TCH and Weight Computation Functions
# ==============================================================================

def _build_J(N: int) -> np.ndarray:
    """
    Build differencing matrix J (N-1, N) that subtracts the N-th reference.

    Args:
        N: Number of products

    Returns:
        J matrix of shape (N-1, N)
    """
    J = np.zeros((N-1, N))
    for i in range(N-1):
        J[i, i] = 1.0
        J[i, -1] = -1.0
    return J


def _compute_S(et_products: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute covariance of differenced products S = cov(Y).

    Args:
        et_products: Shape (N, T) - N products, T time steps

    Returns:
        S: Covariance matrix (N-1, N-1)
        J: Differencing matrix (N-1, N)
    """
    N, T = et_products.shape
    J = _build_J(N)
    Y = J @ et_products  # (N-1, T)
    S = np.cov(Y)  # Variables in rows, ddof=1 by default
    return S, J


def _project_psd(S: np.ndarray) -> np.ndarray:
    """
    Project a symmetric matrix to the nearest PSD by clipping negative eigenvalues.

    Args:
        S: Symmetric matrix

    Returns:
        PSD projected matrix
    """
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-12, None)
    return V @ np.diag(w) @ V.T


def tch_uncorrelated(et_products: np.ndarray) -> np.ndarray:
    """
    Classical Three-Cornered Hat (uncorrelated errors) to estimate variances σ²_i.

    Uses least squares on Var(ET_i - ET_j) ≈ σ²_i + σ²_j assuming zero error
    cross-correlation.

    Args:
        et_products: Shape (N, T) - N products, T time steps

    Returns:
        variances: Shape (N,) - Error variance for each product
    """
    N, T = et_products.shape

    # Compute pairwise difference variances
    pairs = []
    b = []
    for i in range(N):
        for j in range(i+1, N):
            diff = et_products[i] - et_products[j]
            v = np.var(diff, ddof=1)
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            pairs.append(row)
            b.append(v)

    A = np.vstack(pairs)  # Shape (P, N)
    b = np.array(b)

    # Non-negative least squares with small ridge for stability
    ridge = 1e-8 * np.eye(N)
    x = np.linalg.lstsq(A.T @ A + ridge, A.T @ b, rcond=None)[0]
    x = np.maximum(x, 1e-12)  # No negative variances

    return x


def tch_correlated_regularized(
    et_products: np.ndarray,
    alpha: float = 1e-2,
    lr: float = 1e-2,
    iters: int = 500,
    seed: int = 0
) -> np.ndarray:
    """
    Regularized correlated TCH approximation (runnable version).

    Solves: min_R  ||J R J^T - S||²_F + α ||offdiag(R)||²_F  s.t. R ≽ 0

    Uses gradient descent with PSD projection and symmetry enforcement.

    Args:
        et_products: Shape (N, T)
        alpha: Off-diagonal penalty (default: 1e-2)
        lr: Learning rate (default: 1e-2)
        iters: Number of iterations (default: 500)
        seed: Random seed (default: 0)

    Returns:
        R: Estimated error covariance matrix (N, N)
    """
    rng = np.random.default_rng(seed)
    S, J = _compute_S(et_products)
    N, T = et_products.shape

    # Initialize with diagonal from uncorrelated solution
    diag_vars = tch_uncorrelated(et_products)
    R = np.diag(diag_vars)

    for k in range(iters):
        JRJ = J @ R @ J.T
        G_core = J.T @ (JRJ - S) @ J  # Gradient of ||JRJ - S||²_F

        # Off-diagonal penalty gradient
        off = R - np.diag(np.diag(R))
        G = 2.0 * G_core + 2.0 * alpha * off

        R = R - lr * G

        # Ensure symmetry and PSD
        R = 0.5 * (R + R.T)
        R = _project_psd(R)

    return R


def btch_weights_from_variances(variances: np.ndarray) -> np.ndarray:
    """
    Compute BTCH weights from error variances using the analytical formula.

    Formula from He et al. (2020) based on maximizing joint likelihood:
        w_k = ∏_{i≠k} σ²_i / ∑_j ∏_{i≠j} σ²_i

    Implemented in log domain for numerical stability.

    Args:
        variances: Shape (N,) - Error variances σ²_i

    Returns:
        weights: Shape (N,) - Normalized fusion weights

    Raises:
        ValueError: If any variance is non-positive
    """
    variances = np.asarray(variances).ravel()
    if np.any(variances <= 0):
        raise ValueError("All variances must be positive.")

    log_s2 = np.log(variances)
    N = variances.size

    # Precompute total sum of logs
    total = np.sum(log_s2)

    # Each weight uses total - log_s2[k]
    logs = np.array([total - log_s2[k] for k in range(N)])

    # Stabilize with max-normalization
    m = np.max(logs)
    w_unnorm = np.exp(logs - m)
    w = w_unnorm / np.sum(w_unnorm)

    return w


# ==============================================================================
# High-Level Class API
# ==============================================================================

class BTCH_He2020:
    """
    Bayesian Three-Cornered Hat method by He et al. (2020).

    This class provides a high-level interface for error variance estimation
    and data fusion using the BTCH method.

    **Workflow:**
    1. Initialize with data
    2. Estimate error variances using TCH (uncorrelated or correlated)
    3. Compute analytical Bayesian weights
    4. Fuse products using optimal weights

    **Example:**
    ```python
    from collocation import BTCH_He2020
    import numpy as np

    # Three products: shape (n_samples, n_products)
    data = np.column_stack([product1, product2, product3])

    # Initialize and run
    btch = BTCH_He2020(data)
    btch.estimate_variances(method='uncorrelated')
    btch.compute_weights()
    fused = btch.fuse()

    # Access results
    print("Error variances:", btch.error_variances)
    print("Weights:", btch.weights)
    print("RMSE:", btch.rmse)
    ```

    Parameters
    ----------
    data : ndarray
        Input data matrix. Can be:
        - (n_samples, n_products): Samples in rows, products in columns
        - (n_products, n_samples): Products in rows, samples in columns
        Will be converted to (n_products, n_samples) internally.
    transpose : bool, optional
        If True, transpose input data. Default: False (auto-detect)

    Attributes
    ----------
    data : ndarray
        Internal data storage (n_products, n_samples)
    n_products : int
        Number of products
    n_samples : int
        Number of time steps/samples
    error_covariance : ndarray
        Estimated error covariance matrix R (n_products, n_products)
    error_variances : ndarray
        Diagonal of R (n_products,)
    rmse : ndarray
        Root mean square errors sqrt(variances)
    weights : ndarray
        Fusion weights (n_products,)
    fused_product : ndarray
        Fused result (n_samples,)
    """

    def __init__(self, data: np.ndarray, transpose: bool = False):
        """Initialize BTCH with data."""
        data = np.asarray(data)

        if data.ndim != 2:
            raise ValueError("Data must be 2-dimensional (n_samples, n_products) or (n_products, n_samples)")

        # Auto-detect orientation: assume larger dimension is samples
        if not transpose and data.shape[0] > data.shape[1]:
            data = data.T

        if transpose:
            data = data.T

        self.data = data
        self.n_products, self.n_samples = data.shape

        if self.n_products != 3:
            warnings.warn(
                f"BTCH is designed for exactly 3 products, got {self.n_products}. "
                "Results may be unreliable.",
                UserWarning
            )

        # Initialize result containers
        self.error_covariance = None
        self.error_variances = None
        self.rmse = None
        self.weights = None
        self.fused_product = None
        self._tch_method = None

    def estimate_variances(
        self,
        method: str = 'uncorrelated',
        **kwargs
    ) -> np.ndarray:
        """
        Estimate error variances using Three-Cornered Hat.

        Parameters
        ----------
        method : {'uncorrelated', 'correlated'}, default='uncorrelated'
            TCH variant to use:
            - 'uncorrelated': Classical TCH assuming independent errors
            - 'correlated': Regularized TCH with approximate correlations
        **kwargs : dict
            Additional parameters for correlated TCH:
            - alpha : float, off-diagonal penalty (default: 1e-2)
            - lr : float, learning rate (default: 1e-2)
            - iters : int, iterations (default: 500)
            - seed : int, random seed (default: 0)

        Returns
        -------
        variances : ndarray
            Error variances (n_products,)
        """
        self._tch_method = method

        if method == 'uncorrelated':
            variances = tch_uncorrelated(self.data)
            self.error_covariance = np.diag(variances)
        elif method == 'correlated':
            self.error_covariance = tch_correlated_regularized(self.data, **kwargs)
            variances = np.diag(self.error_covariance).copy()
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose 'uncorrelated' or 'correlated'."
            )

        self.error_variances = variances
        self.rmse = np.sqrt(variances)

        return variances

    def compute_weights(self) -> np.ndarray:
        """
        Compute analytical Bayesian fusion weights.

        Must call estimate_variances() first.

        Returns
        -------
        weights : ndarray
            Fusion weights (n_products,)

        Raises
        ------
        RuntimeError
            If variances have not been estimated yet
        """
        if self.error_variances is None:
            raise RuntimeError("Must call estimate_variances() first.")

        self.weights = btch_weights_from_variances(self.error_variances)
        return self.weights

    def fuse(self) -> np.ndarray:
        """
        Fuse products using computed weights.

        Must call compute_weights() first.

        Returns
        -------
        fused : ndarray
            Fused product (n_samples,)

        Raises
        ------
        RuntimeError
            If weights have not been computed yet
        """
        if self.weights is None:
            raise RuntimeError("Must call compute_weights() first.")

        self.fused_product = self.weights @ self.data
        return self.fused_product

    def run(
        self,
        method: str = 'uncorrelated',
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Run complete BTCH workflow: estimate variances, compute weights, fuse.

        Parameters
        ----------
        method : str, default='uncorrelated'
            TCH method ('uncorrelated' or 'correlated')
        **kwargs : dict
            Additional parameters passed to estimate_variances()

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'fused': Fused product (n_samples,)
            - 'weights': Fusion weights (n_products,)
            - 'error_variances': Error variances (n_products,)
            - 'rmse': Root mean square errors (n_products,)
            - 'error_covariance': Full covariance matrix (n_products, n_products)
        """
        self.estimate_variances(method=method, **kwargs)
        self.compute_weights()
        self.fuse()

        return {
            'fused': self.fused_product,
            'weights': self.weights,
            'error_variances': self.error_variances,
            'rmse': self.rmse,
            'error_covariance': self.error_covariance
        }

    def summary(self):
        """Print summary of BTCH results."""
        if self.error_variances is None:
            print("No results yet. Run estimate_variances() first.")
            return

        print("\n" + "="*70)
        print("BTCH (He et al. 2020) Results")
        print("="*70)
        print(f"\nMethod: {self._tch_method or 'unknown'} TCH")
        print(f"Number of products: {self.n_products}")
        print(f"Number of samples: {self.n_samples}")

        print("\nError Estimates (Point Estimates):")
        print("-" * 70)
        for i in range(self.n_products):
            print(f"Product {i}: RMSE = {self.rmse[i]:.6f}, Variance = {self.error_variances[i]:.6f}")

        if self.weights is not None:
            print("\nFusion Weights:")
            print("-" * 70)
            for i in range(self.n_products):
                print(f"Product {i}: w = {self.weights[i]:.6f} ({self.weights[i]*100:.2f}%)")
            print(f"\nWeight sum: {self.weights.sum():.6f} (should be 1.0)")

        if self._tch_method == 'correlated' and self.error_covariance is not None:
            print("\nError Covariance Matrix:")
            print("-" * 70)
            print(self.error_covariance)

            # Show correlations
            std = np.sqrt(np.diag(self.error_covariance))
            corr = self.error_covariance / np.outer(std, std)
            print("\nError Correlation Matrix:")
            print(corr)

        print("="*70 + "\n")

        print("NOTE: This method provides POINT ESTIMATES only (no uncertainty).")
        print("For full Bayesian uncertainty quantification, use BayesianTCH or BayesianTC.")


# ==============================================================================
# Convenience Function (like tc, ivd, eivd, etc.)
# ==============================================================================

def btch_he2020(
    data: np.ndarray,
    method: str = 'uncorrelated',
    return_full: bool = False,
    **kwargs
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray],
           Dict[str, np.ndarray]]:
    """
    Bayesian Three-Cornered Hat (He et al. 2020) - Convenience function.

    Performs error variance estimation via TCH and computes analytical
    Bayesian fusion weights.

    Parameters
    ----------
    data : ndarray
        Input data (n_samples, 3) or (3, n_samples)
    method : {'uncorrelated', 'correlated'}, default='uncorrelated'
        TCH variant:
        - 'uncorrelated': Classical TCH (independent errors)
        - 'correlated': Regularized TCH (approximate correlations)
    return_full : bool, default=False
        If True, return full results dict. If False, return tuple.
    **kwargs : dict
        Additional parameters for correlated TCH (alpha, lr, iters, seed)

    Returns
    -------
    If return_full=False (default):
        error_variances : ndarray (n_products,)
            Error variance for each product
        weights : ndarray (n_products,)
            Fusion weights
        fused : ndarray (n_samples,)
            Fused product

    If return_full=True:
        results : dict
            Full results dictionary with all estimates

    Examples
    --------
    >>> # Basic usage
    >>> variances, weights, fused = btch_he2020(data)

    >>> # With correlated TCH
    >>> variances, weights, fused = btch_he2020(
    ...     data, method='correlated', alpha=1e-2, iters=500
    ... )

    >>> # Get full results
    >>> results = btch_he2020(data, return_full=True)
    >>> print(results['rmse'])

    See Also
    --------
    BTCH_He2020 : Class-based API with more control
    tc : Classical triple collocation
    BayesianTCH : MCMC-based Bayesian TCH with full uncertainty
    BayesianTC : MCMC-based Bayesian TC with time-varying errors
    """
    btch = BTCH_He2020(data)
    results = btch.run(method=method, **kwargs)

    if return_full:
        return results
    else:
        return results['error_variances'], results['weights'], results['fused']


# ==============================================================================
# Utility Functions
# ==============================================================================

def compare_tch_methods(
    data: np.ndarray,
    alpha: float = 1e-2,
    iters: int = 500
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compare uncorrelated vs correlated TCH methods.

    Parameters
    ----------
    data : ndarray
        Input data (n_samples, 3) or (3, n_samples)
    alpha : float, default=1e-2
        Regularization for correlated TCH
    iters : int, default=500
        Iterations for correlated TCH

    Returns
    -------
    comparison : dict
        Results from both methods with keys 'uncorrelated' and 'correlated'
    """
    btch = BTCH_He2020(data)

    # Uncorrelated
    results_uncorr = btch.run(method='uncorrelated')

    # Correlated
    results_corr = btch.run(method='correlated', alpha=alpha, iters=iters)

    return {
        'uncorrelated': results_uncorr,
        'correlated': results_corr
    }


__all__ = [
    'BTCH_He2020',
    'btch_he2020',
    'tch_uncorrelated',
    'tch_correlated_regularized',
    'btch_weights_from_variances',
    'compare_tch_methods',
]
