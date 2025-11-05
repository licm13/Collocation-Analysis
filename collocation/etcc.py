"""
Core merging algorithms for Triple Collocation and ETCC methods.

This module implements:
1. Traditional Triple Collocation (TC) - minimizes error variance
2. Extended Triple Collocation for maximized Correlation (ETCC)

Reference: Wei et al. (2023), Geophysical Research Letters
"""

import numpy as np
from typing import Tuple, Dict, Optional
import warnings


class TripleCollocation:
    """
    Traditional Triple Collocation merging method.
    
    Minimizes error variance by calculating optimal weights based on 
    estimated RMSE of each product against unknown truth.
    
    Attributes:
        weights: Dictionary containing weights for each product
        error_variances: Estimated error variances (RMSE²) for each product
    """
    
    def __init__(self):
        self.weights = None
        self.error_variances = None
        
    def calculate_covariances(self, x: np.ndarray, y: np.ndarray, 
                            z: np.ndarray) -> Dict[str, float]:
        """
        Calculate covariances between product pairs.
        
        Equations 2-7 from Wei et al. (2023):
        Cxy = αx*αy*σR²
        Cxz = αx*αz*σR²
        Cyz = αy*αz*σR²
        Cxx = αx²*σR² + σx²
        Cyy = αy²*σR² + σy²
        Czz = αz²*σR² + σz²
        
        Args:
            x, y, z: Input precipitation products (1D arrays)
            
        Returns:
            Dictionary of covariances
        """
        covs = {
            'Cxy': np.cov(x, y)[0, 1],
            'Cxz': np.cov(x, z)[0, 1],
            'Cyz': np.cov(y, z)[0, 1],
            'Cxx': np.var(x),
            'Cyy': np.var(y),
            'Czz': np.var(z)
        }
        return covs
    
    def estimate_error_variances(self, covs: Dict[str, float]) -> Dict[str, float]:
        """
        Estimate error variance (RMSE²) for each product.
        
        Equations 8-10 from Wei et al. (2023):
        σx² = Cxx - (Cxy*Cxz)/Cyz
        σy² = Cyy - (Cxy*Cyz)/Cxz
        σz² = Czz - (Cxz*Cyz)/Cxy
        
        Args:
            covs: Dictionary of covariances
            
        Returns:
            Dictionary of error variances
        """
        sigma2_x = covs['Cxx'] - (covs['Cxy'] * covs['Cxz']) / covs['Cyz']
        sigma2_y = covs['Cyy'] - (covs['Cxy'] * covs['Cyz']) / covs['Cxz']
        sigma2_z = covs['Czz'] - (covs['Cxz'] * covs['Cyz']) / covs['Cxy']
        
        # Handle negative variances (should not occur theoretically)
        if sigma2_x < 0 or sigma2_y < 0 or sigma2_z < 0:
            warnings.warn("Negative error variance detected. Setting to small positive value.")
            sigma2_x = max(sigma2_x, 1e-6)
            sigma2_y = max(sigma2_y, 1e-6)
            sigma2_z = max(sigma2_z, 1e-6)
        
        return {'sigma2_x': sigma2_x, 'sigma2_y': sigma2_y, 'sigma2_z': sigma2_z}
    
    def calculate_weights(self, error_vars: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate optimal weights by minimizing error variance.
        
        Equations 15-17 from Wei et al. (2023):
        wx = (σy²*σz²) / (σx²*σy² + σx²*σz² + σy²*σz²)
        wy = (σx²*σz²) / (σx²*σy² + σx²*σz² + σy²*σz²)
        wz = (σx²*σy²) / (σx²*σy² + σx²*σz² + σy²*σz²)
        
        Args:
            error_vars: Dictionary of error variances
            
        Returns:
            Dictionary of weights (wx, wy, wz)
        """
        sx2 = error_vars['sigma2_x']
        sy2 = error_vars['sigma2_y']
        sz2 = error_vars['sigma2_z']
        
        denominator = sx2*sy2 + sx2*sz2 + sy2*sz2
        
        if denominator == 0:
            warnings.warn("Zero denominator in weight calculation. Using equal weights.")
            return {'wx': 1/3, 'wy': 1/3, 'wz': 1/3}
        
        wx = (sy2 * sz2) / denominator
        wy = (sx2 * sz2) / denominator
        wz = (sx2 * sy2) / denominator
        
        # Normalize to ensure sum = 1
        total = wx + wy + wz
        wx, wy, wz = wx/total, wy/total, wz/total
        
        return {'wx': wx, 'wy': wy, 'wz': wz}
    
    def merge(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Merge three precipitation products using TC method.
        
        Equation 11 from Wei et al. (2023):
        M = wx*x + wy*y + wz*z
        
        Args:
            x, y, z: Input precipitation products
            
        Returns:
            Merged precipitation product
        """
        # Calculate covariances
        covs = self.calculate_covariances(x, y, z)
        
        # Estimate error variances
        self.error_variances = self.estimate_error_variances(covs)
        
        # Calculate weights
        self.weights = self.calculate_weights(self.error_variances)
        
        # Merge
        merged = (self.weights['wx'] * x + 
                 self.weights['wy'] * y + 
                 self.weights['wz'] * z)
        
        return merged


class ETCC:
    """
    Extended Triple Collocation for maximized Correlation (ETCC).
    
    Maximizes correlation between merged product and unknown truth
    using exhaustive search over weight combinations.
    
    Attributes:
        weight_increment: Step size for weight search (default: 0.01)
        weights: Optimal weights found by maximization
        max_correlation: Maximum correlation achieved
        correlation_with_truth: Estimated correlations of inputs with truth
    """
    
    def __init__(self, weight_increment: float = 0.01, min_correlation: float = 0.01):
        """
        Initialize ETCC method.
        
        Args:
            weight_increment: Step size for exhaustive search (0.01 in paper)
            min_correlation: Minimum correlation threshold to avoid numerical issues
        """
        self.weight_increment = weight_increment
        self.min_correlation = min_correlation
        self.weights = None
        self.max_correlation = None
        self.correlation_with_truth = None
        
    def calculate_pairwise_correlations(self, x: np.ndarray, y: np.ndarray, 
                                       z: np.ndarray) -> Dict[str, float]:
        """
        Calculate correlation between each pair of products.
        
        Equations 18-20 from Wei et al. (2023):
        ρxy = Corr(x, y)
        ρxz = Corr(x, z)
        ρyz = Corr(y, z)
        
        Args:
            x, y, z: Input precipitation products
            
        Returns:
            Dictionary of pairwise correlations
        """
        rho_xy = np.corrcoef(x, y)[0, 1]
        rho_xz = np.corrcoef(x, z)[0, 1]
        rho_yz = np.corrcoef(y, z)[0, 1]
        
        # Apply minimum correlation threshold (from paper)
        rho_xy = max(rho_xy, self.min_correlation)
        rho_xz = max(rho_xz, self.min_correlation)
        rho_yz = max(rho_yz, self.min_correlation)
        
        return {'rho_xy': rho_xy, 'rho_xz': rho_xz, 'rho_yz': rho_yz}
    
    def estimate_correlation_with_truth(self, pairwise_corrs: Dict[str, float]) -> Dict[str, float]:
        """
        Estimate correlation of each product with unknown truth using extended TC.
        
        Equations 21-23 from Wei et al. (2023):
        ρRx = (ρxy * ρxz / ρyz)^(1/2)
        ρRy = (ρxy * ρyz / ρxz)^(1/2)
        ρRz = (ρxz * ρyz / ρxy)^(1/2)
        
        Args:
            pairwise_corrs: Dictionary of pairwise correlations
            
        Returns:
            Dictionary of correlations with truth
        """
        rho_xy = pairwise_corrs['rho_xy']
        rho_xz = pairwise_corrs['rho_xz']
        rho_yz = pairwise_corrs['rho_yz']
        
        rho_Rx = np.sqrt((rho_xy * rho_xz) / rho_yz)
        rho_Ry = np.sqrt((rho_xy * rho_yz) / rho_xz)
        rho_Rz = np.sqrt((rho_xz * rho_yz) / rho_xy)
        
        return {'rho_Rx': rho_Rx, 'rho_Ry': rho_Ry, 'rho_Rz': rho_Rz}
    
    def correlation_function(self, wx: float, wy: float, wz: float,
                           x: np.ndarray, y: np.ndarray, z: np.ndarray,
                           rho_truth: Dict[str, float],
                           variances: Dict[str, float]) -> float:
        """
        Calculate correlation between merged product and truth.
        
        Equation 30 from Wei et al. (2023):
        ρRM = (1/√CMM) * {wx*ρRx*√Cxx + wy*ρRy*√Cyy + wz*ρRz*√Czz}
        
        Args:
            wx, wy, wz: Weights for products x, y, z
            x, y, z: Input products
            rho_truth: Correlations with truth
            variances: Variances of input products
            
        Returns:
            Correlation between merged product and truth
        """
        # Calculate merged product
        M = wx * x + wy * y + wz * z
        
        # Variance of merged product
        C_MM = np.var(M)
        
        if C_MM == 0:
            return 0.0
        
        # Numerator from Equation 30
        numerator = (wx * rho_truth['rho_Rx'] * np.sqrt(variances['Cxx']) +
                    wy * rho_truth['rho_Ry'] * np.sqrt(variances['Cyy']) +
                    wz * rho_truth['rho_Rz'] * np.sqrt(variances['Czz']))
        
        # Correlation
        rho_RM = numerator / np.sqrt(C_MM)
        
        return rho_RM
    
    def exhaustive_search(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                         rho_truth: Dict[str, float],
                         variances: Dict[str, float]) -> Tuple[Dict[str, float], float]:
        """
        Exhaustive search to find weights that maximize correlation.
        
        Uses increment of 0.01 (default) to test all weight combinations
        where wx + wy + wz = 1 and wx, wy, wz ∈ [0, 1].
        
        Optimized version using vectorized operations to reduce loop overhead.
        
        Args:
            x, y, z: Input products
            rho_truth: Correlations with truth
            variances: Variances of input products
            
        Returns:
            Tuple of (optimal weights dict, maximum correlation)
        """
        step = self.weight_increment
        weights_range = np.arange(0, 1 + step, step)
        
        # Pre-compute constants for correlation calculation
        sqrt_Cxx = np.sqrt(variances['Cxx'])
        sqrt_Cyy = np.sqrt(variances['Cyy'])
        sqrt_Czz = np.sqrt(variances['Czz'])
        
        rho_Rx = rho_truth['rho_Rx']
        rho_Ry = rho_truth['rho_Ry']
        rho_Rz = rho_truth['rho_Rz']
        
        max_corr = -np.inf
        best_weights = None
        
        # Generate valid weight combinations more efficiently
        for wx in weights_range:
            for wy in weights_range:
                wz = 1 - wx - wy
                
                # Check if wz is valid
                if wz < -step or wz > 1 + step:
                    continue
                
                # Ensure weights sum to 1
                wz = max(0, min(1, wz))
                
                # Vectorized merged product calculation
                M = wx * x + wy * y + wz * z
                C_MM = np.var(M)
                
                if C_MM == 0:
                    continue
                
                # Vectorized numerator calculation
                numerator = (wx * rho_Rx * sqrt_Cxx +
                           wy * rho_Ry * sqrt_Cyy +
                           wz * rho_Rz * sqrt_Czz)
                
                # Correlation
                corr = numerator / np.sqrt(C_MM)
                
                # Update if better
                if corr > max_corr:
                    max_corr = corr
                    best_weights = {'wx': wx, 'wy': wy, 'wz': wz}
        
        return best_weights, max_corr
    
    def merge(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Merge three precipitation products using ETCC method.
        
        Args:
            x, y, z: Input precipitation products
            
        Returns:
            Merged precipitation product
        """
        # Calculate pairwise correlations
        pairwise_corrs = self.calculate_pairwise_correlations(x, y, z)
        
        # Estimate correlation with truth
        self.correlation_with_truth = self.estimate_correlation_with_truth(pairwise_corrs)
        
        # Get variances
        variances = {
            'Cxx': np.var(x),
            'Cyy': np.var(y),
            'Czz': np.var(z)
        }
        
        # Exhaustive search for optimal weights
        self.weights, self.max_correlation = self.exhaustive_search(
            x, y, z, self.correlation_with_truth, variances
        )
        
        # Merge using optimal weights
        merged = (self.weights['wx'] * x + 
                 self.weights['wy'] * y + 
                 self.weights['wz'] * z)
        
        return merged


class SpatialMerging:
    """
    Wrapper for applying TC/ETCC to gridded spatial data.
    
    Handles multi-dimensional arrays where last dimensions are space/time.
    """
    
    def __init__(self, method: str = 'etcc', **kwargs):
        """
        Initialize spatial merging.
        
        Args:
            method: 'tc' or 'etcc'
            **kwargs: Arguments passed to merging method
        """
        if method.lower() == 'tc':
            self.merger = TripleCollocation(**kwargs)
        elif method.lower() == 'etcc':
            self.merger = ETCC(**kwargs)
        else:
            raise ValueError("method must be 'tc' or 'etcc'")
        
        self.method = method
        self.spatial_weights = None
        
    def merge_gridded(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     axis: int = -1) -> np.ndarray:
        """
        Merge gridded precipitation data.
        
        Args:
            x, y, z: Input products with shape (lat, lon, time) or (time, lat, lon)
            axis: Time axis (default: -1 for last axis)
            
        Returns:
            Merged product with same shape as input
        """
        shape = x.shape
        
        # Flatten spatial dimensions
        if axis == -1:
            n_time = shape[-1]
            x_flat = x.reshape(-1, n_time)
            y_flat = y.reshape(-1, n_time)
            z_flat = z.reshape(-1, n_time)
        else:
            x_flat = np.moveaxis(x, axis, -1).reshape(-1, shape[axis])
            y_flat = np.moveaxis(y, axis, -1).reshape(-1, shape[axis])
            z_flat = np.moveaxis(z, axis, -1).reshape(-1, shape[axis])
        
        # Merge each grid cell
        n_pixels = x_flat.shape[0]
        merged_flat = np.zeros_like(x_flat)
        
        for i in range(n_pixels):
            try:
                merged_flat[i] = self.merger.merge(x_flat[i], y_flat[i], z_flat[i])
            except Exception as e:
                warnings.warn(f"Merging failed for pixel {i}: {e}. Using mean.")
                merged_flat[i] = (x_flat[i] + y_flat[i] + z_flat[i]) / 3
        
        # Reshape back to original
        if axis == -1:
            merged = merged_flat.reshape(shape)
        else:
            merged = merged_flat.reshape(shape)
            merged = np.moveaxis(merged, -1, axis)
        
        return merged
