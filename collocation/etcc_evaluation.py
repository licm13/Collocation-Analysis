"""
Evaluation metrics and validation functions.

Implements common metrics for precipitation product assessment:
- Correlation Coefficient (CC)
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Bias
"""

import numpy as np
from typing import Dict, Optional, Tuple
import warnings


def calculate_cc(predicted: np.ndarray, observed: np.ndarray, 
                mask_zeros: bool = False) -> float:
    """
    Calculate Pearson Correlation Coefficient.
    
    CC range: [-1, 1], ideal value: 1
    
    Args:
        predicted: Predicted/merged precipitation
        observed: Reference/observed precipitation
        mask_zeros: If True, exclude zero values from both arrays
        
    Returns:
        Correlation coefficient
    """
    if mask_zeros:
        mask = (predicted > 0) & (observed > 0)
        predicted = predicted[mask]
        observed = observed[mask]
    
    if len(predicted) == 0:
        warnings.warn("No valid data for CC calculation")
        return np.nan
    
    # Remove NaN values
    mask = ~(np.isnan(predicted) | np.isnan(observed))
    predicted = predicted[mask]
    observed = observed[mask]
    
    if len(predicted) < 2:
        warnings.warn("Insufficient data for CC calculation")
        return np.nan
    
    # Calculate correlation
    cc = np.corrcoef(predicted, observed)[0, 1]
    
    return cc


def calculate_rmse(predicted: np.ndarray, observed: np.ndarray,
                  mask_zeros: bool = False) -> float:
    """
    Calculate Root Mean Square Error.
    
    RMSE range: [0, +∞], ideal value: 0
    
    Args:
        predicted: Predicted/merged precipitation
        observed: Reference/observed precipitation
        mask_zeros: If True, exclude zero values from both arrays
        
    Returns:
        RMSE value
    """
    if mask_zeros:
        mask = (predicted > 0) & (observed > 0)
        predicted = predicted[mask]
        observed = observed[mask]
    
    if len(predicted) == 0:
        warnings.warn("No valid data for RMSE calculation")
        return np.nan
    
    # Remove NaN values
    mask = ~(np.isnan(predicted) | np.isnan(observed))
    predicted = predicted[mask]
    observed = observed[mask]
    
    if len(predicted) == 0:
        warnings.warn("No valid data for RMSE calculation")
        return np.nan
    
    # Calculate RMSE
    mse = np.mean((predicted - observed) ** 2)
    rmse = np.sqrt(mse)
    
    return rmse


def calculate_mae(predicted: np.ndarray, observed: np.ndarray,
                 mask_zeros: bool = False) -> float:
    """
    Calculate Mean Absolute Error.
    
    MAE range: [0, +∞], ideal value: 0
    
    Args:
        predicted: Predicted/merged precipitation
        observed: Reference/observed precipitation
        mask_zeros: If True, exclude zero values from both arrays
        
    Returns:
        MAE value
    """
    if mask_zeros:
        mask = (predicted > 0) & (observed > 0)
        predicted = predicted[mask]
        observed = observed[mask]
    
    if len(predicted) == 0:
        warnings.warn("No valid data for MAE calculation")
        return np.nan
    
    # Remove NaN values
    mask = ~(np.isnan(predicted) | np.isnan(observed))
    predicted = predicted[mask]
    observed = observed[mask]
    
    if len(predicted) == 0:
        warnings.warn("No valid data for MAE calculation")
        return np.nan
    
    # Calculate MAE
    mae = np.mean(np.abs(predicted - observed))
    
    return mae


def calculate_bias(predicted: np.ndarray, observed: np.ndarray,
                  mask_zeros: bool = False) -> float:
    """
    Calculate mean bias.
    
    Bias = mean(predicted - observed)
    Positive bias: overestimation
    Negative bias: underestimation
    
    Args:
        predicted: Predicted/merged precipitation
        observed: Reference/observed precipitation
        mask_zeros: If True, exclude zero values from both arrays
        
    Returns:
        Mean bias
    """
    if mask_zeros:
        mask = (predicted > 0) & (observed > 0)
        predicted = predicted[mask]
        observed = observed[mask]
    
    if len(predicted) == 0:
        warnings.warn("No valid data for bias calculation")
        return np.nan
    
    # Remove NaN values
    mask = ~(np.isnan(predicted) | np.isnan(observed))
    predicted = predicted[mask]
    observed = observed[mask]
    
    if len(predicted) == 0:
        warnings.warn("No valid data for bias calculation")
        return np.nan
    
    # Calculate bias
    bias = np.mean(predicted - observed)
    
    return bias


def calculate_relative_bias(predicted: np.ndarray, observed: np.ndarray,
                           mask_zeros: bool = False) -> float:
    """
    Calculate relative bias (%).
    
    Relative bias = (mean(predicted) - mean(observed)) / mean(observed) * 100
    
    Args:
        predicted: Predicted/merged precipitation
        observed: Reference/observed precipitation
        mask_zeros: If True, exclude zero values from both arrays
        
    Returns:
        Relative bias in percentage
    """
    if mask_zeros:
        mask = (predicted > 0) & (observed > 0)
        predicted = predicted[mask]
        observed = observed[mask]
    
    if len(predicted) == 0:
        warnings.warn("No valid data for relative bias calculation")
        return np.nan
    
    # Remove NaN values
    mask = ~(np.isnan(predicted) | np.isnan(observed))
    predicted = predicted[mask]
    observed = observed[mask]
    
    if len(predicted) == 0 or np.mean(observed) == 0:
        warnings.warn("No valid data for relative bias calculation")
        return np.nan
    
    # Calculate relative bias
    rel_bias = (np.mean(predicted) - np.mean(observed)) / np.mean(observed) * 100
    
    return rel_bias


def calculate_metrics(predicted: np.ndarray, observed: np.ndarray,
                     mask_zeros: bool = False) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Args:
        predicted: Predicted/merged precipitation
        observed: Reference/observed precipitation
        mask_zeros: If True, exclude zero values from both arrays
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        'cc': calculate_cc(predicted, observed, mask_zeros),
        'rmse': calculate_rmse(predicted, observed, mask_zeros),
        'mae': calculate_mae(predicted, observed, mask_zeros),
        'bias': calculate_bias(predicted, observed, mask_zeros),
        'relative_bias': calculate_relative_bias(predicted, observed, mask_zeros)
    }
    
    return metrics


def spatial_metrics(predicted: np.ndarray, observed: np.ndarray,
                   metric: str = 'cc', axis: int = -1) -> np.ndarray:
    """
    Calculate spatial distribution of a metric.
    
    For gridded data (lat, lon, time), calculates metric at each grid cell.
    
    Args:
        predicted: Predicted precipitation (lat, lon, time)
        observed: Reference precipitation (lat, lon, time)
        metric: 'cc', 'rmse', 'mae', 'bias'
        axis: Time axis (default: -1)
        
    Returns:
        Spatial map of metric (lat, lon)
    """
    metric_funcs = {
        'cc': calculate_cc,
        'rmse': calculate_rmse,
        'mae': calculate_mae,
        'bias': calculate_bias
    }
    
    if metric not in metric_funcs:
        raise ValueError(f"Unknown metric: {metric}")
    
    func = metric_funcs[metric]
    shape = predicted.shape
    
    # Flatten spatial dimensions
    if axis == -1:
        n_time = shape[-1]
        pred_flat = predicted.reshape(-1, n_time)
        obs_flat = observed.reshape(-1, n_time)
        spatial_shape = shape[:-1]
    else:
        pred_flat = np.moveaxis(predicted, axis, -1).reshape(-1, shape[axis])
        obs_flat = np.moveaxis(observed, axis, -1).reshape(-1, shape[axis])
        spatial_shape = [s for i, s in enumerate(shape) if i != axis]
    
    # Calculate metric for each pixel
    n_pixels = pred_flat.shape[0]
    result = np.zeros(n_pixels)
    
    for i in range(n_pixels):
        result[i] = func(pred_flat[i], obs_flat[i])
    
    # Reshape to spatial dimensions
    result = result.reshape(spatial_shape)
    
    return result


def compare_products(products: Dict[str, np.ndarray], 
                    reference: np.ndarray,
                    mask_zeros: bool = False) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple precipitation products against reference.
    
    Useful for comparing TC vs ETCC vs original products.
    
    Args:
        products: Dictionary of products {'name': array}
        reference: Reference precipitation
        mask_zeros: If True, exclude zero values
        
    Returns:
        Dictionary of metrics for each product
    """
    results = {}
    
    for name, product in products.items():
        results[name] = calculate_metrics(product, reference, mask_zeros)
    
    return results


def print_comparison_table(results: Dict[str, Dict[str, float]]):
    """
    Print formatted comparison table of metrics.
    
    Args:
        results: Output from compare_products()
    """
    print("\n" + "="*70)
    print(f"{'Product':<20} {'CC':>10} {'RMSE':>10} {'MAE':>10} {'Bias':>10}")
    print("="*70)
    
    for product, metrics in results.items():
        print(f"{product:<20} "
              f"{metrics['cc']:>10.4f} "
              f"{metrics['rmse']:>10.4f} "
              f"{metrics['mae']:>10.4f} "
              f"{metrics['bias']:>10.4f}")
    
    print("="*70 + "\n")


def statistical_significance(corr1: float, corr2: float, n: int,
                            alpha: float = 0.05) -> Tuple[bool, float]:
    """
    Test if difference between two correlations is statistically significant.
    
    Uses Fisher's z-transformation.
    
    Args:
        corr1: First correlation coefficient
        corr2: Second correlation coefficient
        n: Sample size
        alpha: Significance level (default: 0.05)
        
    Returns:
        Tuple of (is_significant, p_value)
    """
    from scipy import stats
    
    # Fisher's z-transformation
    z1 = 0.5 * np.log((1 + corr1) / (1 - corr1))
    z2 = 0.5 * np.log((1 + corr2) / (1 - corr2))
    
    # Standard error
    se = np.sqrt(2 / (n - 3))
    
    # Test statistic
    z = (z1 - z2) / se
    
    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    is_significant = p_value < alpha
    
    return is_significant, p_value


class CrossValidation:
    """
    Cross-validation framework for merging methods.
    """
    
    def __init__(self, n_folds: int = 5):
        """
        Initialize cross-validation.
        
        Args:
            n_folds: Number of folds for cross-validation
        """
        self.n_folds = n_folds
        
    def temporal_cv(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                   reference: np.ndarray, method: str = 'etcc') -> Dict[str, float]:
        """
        Perform temporal cross-validation.
        
        Splits time series into folds, trains on n-1 folds, tests on 1 fold.
        
        Args:
            x, y, z: Input precipitation products (time series)
            reference: Reference precipitation
            method: 'tc' or 'etcc'
            
        Returns:
            Dictionary of averaged metrics across folds
        """
        from .merging import TripleCollocation, ETCC
        
        n = len(x)
        fold_size = n // self.n_folds
        
        all_metrics = []
        
        for fold in range(self.n_folds):
            # Define test indices
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < self.n_folds - 1 else n
            
            # Split data
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_start:test_end] = False
            
            x_train, x_test = x[train_mask], x[~train_mask]
            y_train, y_test = y[train_mask], y[~train_mask]
            z_train, z_test = z[train_mask], z[~train_mask]
            ref_test = reference[~train_mask]
            
            # Train and merge
            if method.lower() == 'tc':
                merger = TripleCollocation()
            else:
                merger = ETCC()
            
            # Fit on training data
            _ = merger.merge(x_train, y_train, z_train)
            
            # Use learned weights for test data
            merged_test = (merger.weights['wx'] * x_test +
                          merger.weights['wy'] * y_test +
                          merger.weights['wz'] * z_test)
            
            # Evaluate
            metrics = calculate_metrics(merged_test, ref_test)
            all_metrics.append(metrics)
        
        # Average metrics across folds
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if not np.isnan(m[key])]
            avg_metrics[key] = np.mean(values) if len(values) > 0 else np.nan
            avg_metrics[f'{key}_std'] = np.std(values) if len(values) > 0 else np.nan
        
        return avg_metrics
