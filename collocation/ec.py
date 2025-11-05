"""
Extended Collocation (EC) / Quadruple Collocation Method
=========================================================

This module implements the Extended Collocation method for four
independent data products (quadruple collocation).

EC extends triple collocation to handle four or more products,
allowing for error cross-correlation estimation and providing
optimal merging weights for multi-source data fusion.

References:
    - Gruber, A., et al. (2016). Estimating error cross-correlations in
      soil moisture data sets using extended collocation analysis.
      Journal of Geophysical Research: Atmospheres, 121(3), 1208-1219.
    - Chen, F., et al. (2017). Error characterization of soil moisture
      satellite products: Retrieving error cross-correlation through
      extended quadruple collocation. IEEE TGRS, 55(8), 4564-4576.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
from dataclasses import dataclass


@dataclass
class ECResult:
    """
    Container for Extended Collocation results.

    Attributes
    ----------
    EeeT : List[np.ndarray]
        Error covariance matrices for each reference configuration
    SNR : List[np.ndarray]
        Signal-to-Noise Ratios for each reference configuration
    rho2 : List[np.ndarray]
        Data-truth correlations for each reference configuration
    fMSE : List[np.ndarray]
        Fractional MSE for each reference configuration
    re_weight : List[np.ndarray]
        Optimal merging weights for each reference configuration
    res_qu : np.ndarray
        Rescaled data for all reference configurations
    weighted_result : List[np.ndarray]
        Weighted merged results for each reference configuration
    combination : Tuple[int, int]
        The coordinate combination used (e.g., (0, 1))
    """
    EeeT: List[np.ndarray]
    SNR: List[np.ndarray]
    rho2: List[np.ndarray]
    fMSE: List[np.ndarray]
    re_weight: List[np.ndarray]
    res_qu: np.ndarray
    weighted_result: List[np.ndarray]
    combination: Tuple[int, int]


def ec(qu: np.ndarray) -> List[ECResult]:
    """
    Calculate Extended Collocation results for all possible combinations.

    The EC method analyzes four data products by testing all possible
    two-coordinate reference combinations (6 total), providing error
    estimates and optimal merging weights for each configuration.

    Parameters
    ----------
    qu : np.ndarray
        Input data array of shape (n, 4), where n >= 4
        Columns represent four independent data products

    Returns
    -------
    results : List[ECResult]
        List of ECResult objects, one for each of the 6 possible
        coordinate combinations. Each result contains:
        - Error covariance matrices (3 reference scenarios per combination)
        - SNR, rho2, fMSE metrics
        - Optimal merging weights
        - Rescaled data
        - Weighted merged results

    Raises
    ------
    ValueError
        If input does not have exactly 4 columns
    RuntimeError
        If insufficient valid data (< 4 rows after removing invalid values)

    Examples
    --------
    >>> import numpy as np
    >>> from collocation import ec
    >>>
    >>> # Generate synthetic data with 4 products
    >>> np.random.seed(42)
    >>> n = 150
    >>> true_signal = np.random.randn(n)
    >>>
    >>> # Four products with different error levels
    >>> X1 = true_signal + 0.2 * np.random.randn(n)
    >>> X2 = true_signal + 0.25 * np.random.randn(n)
    >>> X3 = true_signal + 0.3 * np.random.randn(n)
    >>> X4 = true_signal + 0.35 * np.random.randn(n)
    >>> qu = np.column_stack([X1, X2, X3, X4])
    >>>
    >>> # Apply Extended Collocation
    >>> results = ec(qu)
    >>>
    >>> # Examine first combination
    >>> result = results[0]
    >>> print(f"Combination: {result.combination}")
    >>> print(f"Number of reference scenarios: {len(result.EeeT)}")
    >>> print(f"Optimal weights (first scenario): {result.re_weight[0]}")
    >>> print(f"Error variances (first scenario): {np.diag(result.EeeT[0])}")
    >>>
    >>> # Get merged result from best scenario
    >>> merged = result.weighted_result[0]
    >>> print(f"Merged result length: {len(merged)}")

    Notes
    -----
    The EC method:
    1. Tests all 6 possible two-coordinate reference combinations:
       (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)

    2. For each combination:
       - Rescales remaining products to the reference pair
       - Runs 3 different reference scenarios
       - Solves overdetermined linear system (13 equations, 10 unknowns)
       - Calculates optimal weights for data fusion

    3. Handles invalid data (NaN, negative values) automatically

    4. Provides multiple merging options with different reference standards

    The linear system solved:
        y = Ax
    where:
        - A: 13x10 equation matrix
        - y: Known variance combinations (13x1)
        - x: [signal_variances; error_variances; error_covariance] (10x1)

    Weight calculation uses error covariance structure:
        w_i = minimize variance of weighted sum
        Subject to: sum(w_i) = 1
    """
    # Validate input
    if qu.shape[1] != 4:
        raise ValueError('Function is only valid for quadruple situation (4 columns required).')

    # Define all possible two-coordinate combinations
    combinations = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3), (2, 3)
    ]

    results = []

    # Loop through each combination
    for comb in combinations:
        ref_idx = list(comb)
        other_idx = [i for i in range(4) if i not in ref_idx]

        # Permute columns: reference pair first, then others
        perm_qu = qu[:, ref_idx + other_idx]

        # Identify and handle invalid rows (NaN or negative values)
        invalid_rows = np.any(perm_qu < 0, axis=1) | np.any(np.isnan(perm_qu), axis=1)
        valid_rows = ~invalid_rows
        cal_qu = perm_qu[valid_rows, :]

        # Check if sufficient valid data
        if cal_qu.shape[0] < 4:
            warnings.warn(
                f"Insufficient valid data for combination {comb}. "
                f"Only {cal_qu.shape[0]} valid rows (need >= 4). Skipping.",
                UserWarning
            )
            continue

        # Calculate unscaled covariance matrix
        ExxT_unres = np.cov(cal_qu, rowvar=False)

        # Pre-compute means for all products to avoid redundant calculations
        cal_qu_means = np.nanmean(cal_qu, axis=0)

        # Initialize rescaled data storage
        # Shape: (n_valid, 4, 4) for 4 reference scenarios (only use first 3)
        res_qu = np.zeros((cal_qu.shape[0], 4, 4))

        # Rescale data for each reference coordinate
        for i in range(4):
            if i == 0:
                # Reference: first product (index 0)
                res_qu[:, 0, i] = cal_qu[:, 0]
                res_qu[:, 1, i] = _rescale_product(
                    cal_qu[:, 1], cal_qu[:, 0], ExxT_unres, 0, 1, 2,
                    source_mean=cal_qu_means[1], target_mean=cal_qu_means[0]
                )
                res_qu[:, 2, i] = _rescale_product(
                    cal_qu[:, 2], cal_qu[:, 0], ExxT_unres, 0, 2, 1,
                    source_mean=cal_qu_means[2], target_mean=cal_qu_means[0]
                )
                res_qu[:, 3, i] = _rescale_product(
                    cal_qu[:, 3], cal_qu[:, 0], ExxT_unres, 0, 3, 1,
                    source_mean=cal_qu_means[3], target_mean=cal_qu_means[0]
                )
            elif i == 1:
                # Reference: second product (index 1)
                res_qu[:, 0, i] = _rescale_product(
                    cal_qu[:, 0], cal_qu[:, 1], ExxT_unres, 1, 0, 2,
                    source_mean=cal_qu_means[0], target_mean=cal_qu_means[1]
                )
                res_qu[:, 1, i] = cal_qu[:, 1]
                res_qu[:, 2, i] = _rescale_product(
                    cal_qu[:, 2], cal_qu[:, 1], ExxT_unres, 1, 2, 0,
                    source_mean=cal_qu_means[2], target_mean=cal_qu_means[1]
                )
                res_qu[:, 3, i] = _rescale_product(
                    cal_qu[:, 3], cal_qu[:, 1], ExxT_unres, 1, 3, 0,
                    source_mean=cal_qu_means[3], target_mean=cal_qu_means[1]
                )
            elif i == 2:
                # Reference: third product (index 2)
                res_qu[:, 0, i] = _rescale_product(
                    cal_qu[:, 0], cal_qu[:, 2], ExxT_unres, 2, 0, 1,
                    source_mean=cal_qu_means[0], target_mean=cal_qu_means[2]
                )
                res_qu[:, 1, i] = _rescale_product(
                    cal_qu[:, 1], cal_qu[:, 2], ExxT_unres, 2, 1, 0,
                    source_mean=cal_qu_means[1], target_mean=cal_qu_means[2]
                )
                res_qu[:, 2, i] = cal_qu[:, 2]
                res_qu[:, 3, i] = _rescale_product(
                    cal_qu[:, 3], cal_qu[:, 2], ExxT_unres, 2, 3, 0,
                    source_mean=cal_qu_means[3], target_mean=cal_qu_means[2]
                )
            else:  # i == 3
                # Reference: fourth product (index 3)
                res_qu[:, 0, i] = _rescale_product(
                    cal_qu[:, 0], cal_qu[:, 3], ExxT_unres, 3, 0, 1,
                    source_mean=cal_qu_means[0], target_mean=cal_qu_means[3]
                )
                res_qu[:, 1, i] = _rescale_product(
                    cal_qu[:, 1], cal_qu[:, 3], ExxT_unres, 3, 1, 0,
                    source_mean=cal_qu_means[1], target_mean=cal_qu_means[3]
                )
                res_qu[:, 2, i] = _rescale_product(
                    cal_qu[:, 2], cal_qu[:, 3], ExxT_unres, 3, 2, 0,
                    source_mean=cal_qu_means[2], target_mean=cal_qu_means[3]
                )
                res_qu[:, 3, i] = cal_qu[:, 3]

        # Calculate EC results for first 3 reference scenarios
        EeeT_list = []
        SNR_list = []
        rho2_list = []
        fMSE_list = []
        re_weight_list = []
        weighted_result_list = []

        for i in range(3):  # Only use first 3 reference scenarios
            # Calculate rescaled covariance matrix
            ExxT = np.cov(res_qu[:, :, i], rowvar=False)

            # Solve EC linear system
            x = _solve_ec_system(ExxT)

            # Assemble error covariance matrix
            EeeT = np.diag(x[5:9])
            EeeT[0, 1] = x[9]  # Error cross-correlation
            EeeT[1, 0] = x[9]

            # Calculate metrics
            SNR = x[:4] / x[5:9]
            rho2 = SNR / (1 + SNR)
            fMSE = 1 - rho2

            # Calculate optimal merging weights
            re_weight = _calculate_weights(EeeT)

            # Calculate weighted merged result
            weighted_result = _merge_with_weights(
                qu, res_qu[:, :, i], re_weight, invalid_rows
            )

            # Store results
            EeeT_list.append(EeeT)
            SNR_list.append(SNR)
            rho2_list.append(rho2)
            fMSE_list.append(fMSE)
            re_weight_list.append(re_weight)
            weighted_result_list.append(weighted_result)

        # Create result object
        result = ECResult(
            EeeT=EeeT_list,
            SNR=SNR_list,
            rho2=rho2_list,
            fMSE=fMSE_list,
            re_weight=re_weight_list,
            res_qu=res_qu,
            weighted_result=weighted_result_list,
            combination=comb
        )
        results.append(result)

    if len(results) == 0:
        raise RuntimeError("No valid results obtained from any combination. Check input data quality.")

    return results


def _rescale_product(source: np.ndarray,
                    target: np.ndarray,
                    cov: np.ndarray,
                    ref_idx: int,
                    src_idx: int,
                    helper_idx: int,
                    source_mean: Optional[float] = None,
                    target_mean: Optional[float] = None) -> np.ndarray:
    """
    Rescale a product to match the reference using covariance information.

    Parameters
    ----------
    source : np.ndarray
        Source product to rescale
    target : np.ndarray
        Target reference product
    cov : np.ndarray
        Covariance matrix
    ref_idx : int
        Index of reference product in covariance matrix
    src_idx : int
        Index of source product in covariance matrix
    helper_idx : int
        Index of helper product for scaling calculation
    source_mean : float, optional
        Pre-computed mean of source (avoids redundant calculation)
    target_mean : float, optional
        Pre-computed mean of target (avoids redundant calculation)

    Returns
    -------
    rescaled : np.ndarray
        Rescaled product
    """
    scaling = cov[ref_idx, helper_idx] / cov[src_idx, helper_idx]
    
    # Use pre-computed means if available to avoid redundant np.nanmean calls
    if source_mean is None:
        source_mean = np.nanmean(source)
    if target_mean is None:
        target_mean = np.nanmean(target)
    
    rescaled = scaling * (source - source_mean) + target_mean
    return rescaled


def _solve_ec_system(ExxT: np.ndarray) -> np.ndarray:
    """
    Solve the Extended Collocation linear system.

    Parameters
    ----------
    ExxT : np.ndarray
        Covariance matrix of rescaled data (4x4)

    Returns
    -------
    x : np.ndarray
        Solution vector (10x1) containing:
        [signal_var1, signal_var2, signal_var3, signal_var4,
         error_var1, error_var2, error_var3, error_var4,
         error_cov12, error_cov21]
    """
    # Build equation matrix (13x10)
    A = np.array([
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Var(X1)
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],  # Var(X2)
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # Var(X3)
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # Var(X4)
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],  # Cov(X1,X2)
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # From cross-covariances
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    ])

    # Build right-hand side vector (13x1)
    y = np.array([
        ExxT[0, 0],  # Var(X1)
        ExxT[1, 1],  # Var(X2)
        ExxT[2, 2],  # Var(X3)
        ExxT[3, 3],  # Var(X4)
        ExxT[0, 1],  # Cov(X1,X2)
        ExxT[0, 2] * ExxT[0, 3] / ExxT[2, 3],  # Signal covariances
        ExxT[1, 2] * ExxT[1, 3] / ExxT[2, 3],
        ExxT[0, 2] * ExxT[2, 3] / ExxT[0, 3],
        ExxT[1, 2] * ExxT[2, 3] / ExxT[1, 3],
        ExxT[0, 3] * ExxT[2, 3] / ExxT[0, 2],
        ExxT[1, 3] * ExxT[2, 3] / ExxT[1, 2],
        ExxT[0, 2] * ExxT[1, 3] / ExxT[2, 3],
        ExxT[0, 3] * ExxT[1, 2] / ExxT[2, 3]
    ])

    # Solve via least squares
    x = np.linalg.lstsq(A, y, rcond=None)[0]
    return x


def _calculate_weights(EeeT: np.ndarray) -> np.ndarray:
    """
    Calculate optimal merging weights from error covariance matrix.

    The weights minimize the variance of the weighted average while
    ensuring they sum to 1.

    Parameters
    ----------
    EeeT : np.ndarray
        Error covariance matrix (4x4)

    Returns
    -------
    weights : np.ndarray
        Optimal weights (4,)
    """
    var = np.diag(EeeT)
    covar = EeeT[0, 1]

    # Calculate normalization factor
    z = (var[0] + var[1] - 2 * covar) / (var[0] * var[1] - covar**2)
    z += 1 / var[2] + 1 / var[3]

    # Calculate individual weights
    weights = np.zeros(4)
    weights[0] = (var[1] - covar) / ((var[0] * var[1] - covar**2) * z)
    weights[1] = (var[0] - covar) / ((var[0] * var[1] - covar**2) * z)
    weights[2] = 1 / (var[2] * z)
    weights[3] = 1 / (var[3] * z)

    return weights


def _merge_with_weights(original: np.ndarray,
                       rescaled: np.ndarray,
                       weights: np.ndarray,
                       invalid_mask: np.ndarray) -> np.ndarray:
    """
    Merge data products using optimal weights.

    Parameters
    ----------
    original : np.ndarray
        Original data (n, 4)
    rescaled : np.ndarray
        Rescaled data for valid rows only
    weights : np.ndarray
        Merging weights (4,)
    invalid_mask : np.ndarray
        Boolean mask for invalid rows

    Returns
    -------
    merged : np.ndarray
        Merged result (n,) with NaN for invalid rows
    """
    n = original.shape[0]
    merged = np.full(n, np.nan)

    # Fill invalid rows with mean of original data
    if np.any(invalid_mask):
        merged[invalid_mask] = np.nanmean(original[invalid_mask, :], axis=1)

    # Fill valid rows with weighted average of rescaled data
    merged[~invalid_mask] = np.sum(rescaled * weights, axis=1)

    return merged


def select_best_combination(results: List[ECResult],
                           criterion: str = 'min_fMSE') -> ECResult:
    """
    Select the best EC result from all combinations.

    Parameters
    ----------
    results : List[ECResult]
        List of EC results from different combinations
    criterion : str, default='min_fMSE'
        Selection criterion:
        - 'min_fMSE': Minimum average fractional MSE
        - 'max_SNR': Maximum average SNR
        - 'min_error_var': Minimum average error variance

    Returns
    -------
    best_result : ECResult
        The best result according to criterion

    Examples
    --------
    >>> results = ec(qu)
    >>> best = select_best_combination(results, criterion='min_fMSE')
    >>> print(f"Best combination: {best.combination}")
    """
    if not results:
        raise ValueError("No results to select from")

    scores = []
    for result in results:
        if criterion == 'min_fMSE':
            score = np.mean([np.mean(fMSE) for fMSE in result.fMSE])
        elif criterion == 'max_SNR':
            score = -np.mean([np.mean(SNR) for SNR in result.SNR])  # Negative for argmin
        elif criterion == 'min_error_var':
            score = np.mean([np.mean(np.diag(EeeT)) for EeeT in result.EeeT])
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        scores.append(score)

    best_idx = np.argmin(scores)
    return results[best_idx]
