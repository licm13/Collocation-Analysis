"""
Unit tests for evaluation metrics.
"""

import pytest
import numpy as np
from etcc import (calculate_cc, calculate_rmse, calculate_mae,
                  calculate_bias, calculate_metrics, compare_products,
                  spatial_metrics, generate_synthetic_data)


class TestMetrics:
    """Tests for individual metric functions."""
    
    def setup_method(self):
        """Generate test data."""
        self.truth, self.x, _, _ = generate_synthetic_data(n_samples=100, seed=42)
        
    def test_calculate_cc_perfect(self):
        """Test CC with perfect correlation."""
        cc = calculate_cc(self.truth, self.truth)
        assert np.isclose(cc, 1.0, atol=1e-6)
    
    def test_calculate_cc_range(self):
        """Test CC is in valid range."""
        cc = calculate_cc(self.x, self.truth)
        assert -1 <= cc <= 1
    
    def test_calculate_rmse_perfect(self):
        """Test RMSE with perfect match."""
        rmse = calculate_rmse(self.truth, self.truth)
        assert np.isclose(rmse, 0.0, atol=1e-6)
    
    def test_calculate_rmse_positive(self):
        """Test RMSE is non-negative."""
        rmse = calculate_rmse(self.x, self.truth)
        assert rmse >= 0
    
    def test_calculate_mae_perfect(self):
        """Test MAE with perfect match."""
        mae = calculate_mae(self.truth, self.truth)
        assert np.isclose(mae, 0.0, atol=1e-6)
    
    def test_calculate_mae_positive(self):
        """Test MAE is non-negative."""
        mae = calculate_mae(self.x, self.truth)
        assert mae >= 0
    
    def test_calculate_bias_perfect(self):
        """Test bias with perfect match."""
        bias = calculate_bias(self.truth, self.truth)
        assert np.isclose(bias, 0.0, atol=1e-6)
    
    def test_calculate_bias_overestimate(self):
        """Test bias with overestimation."""
        overestimate = self.truth * 1.5
        bias = calculate_bias(overestimate, self.truth)
        assert bias > 0
    
    def test_calculate_bias_underestimate(self):
        """Test bias with underestimation."""
        underestimate = self.truth * 0.5
        bias = calculate_bias(underestimate, self.truth)
        assert bias < 0
    
    def test_metrics_with_nan(self):
        """Test metrics handle NaN values."""
        x_with_nan = self.x.copy()
        x_with_nan[0:5] = np.nan
        
        cc = calculate_cc(x_with_nan, self.truth)
        assert not np.isnan(cc)
        
        rmse = calculate_rmse(x_with_nan, self.truth)
        assert not np.isnan(rmse)
    
    def test_metrics_with_zeros_masking(self):
        """Test metrics with zero masking option."""
        # Add some zeros
        x_with_zeros = self.x.copy()
        x_with_zeros[0:10] = 0
        truth_with_zeros = self.truth.copy()
        truth_with_zeros[0:10] = 0
        
        cc_with_mask = calculate_cc(x_with_zeros, truth_with_zeros, mask_zeros=True)
        cc_without_mask = calculate_cc(x_with_zeros, truth_with_zeros, mask_zeros=False)
        
        # Results may differ
        assert not np.isnan(cc_with_mask)
        assert not np.isnan(cc_without_mask)


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""
    
    def setup_method(self):
        """Generate test data."""
        self.truth, self.x, _, _ = generate_synthetic_data(n_samples=100, seed=42)
    
    def test_returns_dict(self):
        """Test that function returns dictionary."""
        metrics = calculate_metrics(self.x, self.truth)
        assert isinstance(metrics, dict)
    
    def test_contains_all_metrics(self):
        """Test that all expected metrics are present."""
        metrics = calculate_metrics(self.x, self.truth)
        expected_keys = ['cc', 'rmse', 'mae', 'bias', 'relative_bias']
        for key in expected_keys:
            assert key in metrics
    
    def test_all_metrics_valid(self):
        """Test that all metric values are valid numbers."""
        metrics = calculate_metrics(self.x, self.truth)
        for key, value in metrics.items():
            assert isinstance(value, (int, float))
            if key != 'relative_bias':  # relative_bias can be large
                assert not np.isinf(value)


class TestCompareProducts:
    """Tests for compare_products function."""
    
    def setup_method(self):
        """Generate test data."""
        self.truth, self.x, self.y, self.z = generate_synthetic_data(
            n_samples=100, seed=42
        )
    
    def test_returns_dict(self):
        """Test that function returns dictionary."""
        products = {'x': self.x, 'y': self.y, 'z': self.z}
        results = compare_products(products, self.truth)
        assert isinstance(results, dict)
    
    def test_contains_all_products(self):
        """Test that all products are in results."""
        products = {'x': self.x, 'y': self.y, 'z': self.z}
        results = compare_products(products, self.truth)
        for name in products.keys():
            assert name in results
    
    def test_each_product_has_metrics(self):
        """Test that each product has all metrics."""
        products = {'x': self.x, 'y': self.y}
        results = compare_products(products, self.truth)
        expected_keys = ['cc', 'rmse', 'mae', 'bias', 'relative_bias']
        for product_results in results.values():
            for key in expected_keys:
                assert key in product_results


class TestSpatialMetrics:
    """Tests for spatial metrics calculation."""
    
    def test_spatial_cc(self):
        """Test spatial CC calculation."""
        # Create 3D data (lat, lon, time)
        nlat, nlon, ntime = 5, 5, 100
        
        predicted = np.random.randn(nlat, nlon, ntime) + 5
        observed = predicted + np.random.randn(nlat, nlon, ntime) * 0.5
        
        cc_map = spatial_metrics(predicted, observed, metric='cc', axis=-1)
        
        assert cc_map.shape == (nlat, nlon)
        assert np.all((cc_map >= -1) & (cc_map <= 1) | np.isnan(cc_map))
    
    def test_spatial_rmse(self):
        """Test spatial RMSE calculation."""
        nlat, nlon, ntime = 5, 5, 100
        
        predicted = np.random.randn(nlat, nlon, ntime) + 5
        observed = predicted + np.random.randn(nlat, nlon, ntime) * 0.5
        
        rmse_map = spatial_metrics(predicted, observed, metric='rmse', axis=-1)
        
        assert rmse_map.shape == (nlat, nlon)
        assert np.all((rmse_map >= 0) | np.isnan(rmse_map))
    
    def test_spatial_mae(self):
        """Test spatial MAE calculation."""
        nlat, nlon, ntime = 5, 5, 100
        
        predicted = np.random.randn(nlat, nlon, ntime) + 5
        observed = predicted + np.random.randn(nlat, nlon, ntime) * 0.5
        
        mae_map = spatial_metrics(predicted, observed, metric='mae', axis=-1)
        
        assert mae_map.shape == (nlat, nlon)
        assert np.all((mae_map >= 0) | np.isnan(mae_map))
    
    def test_invalid_metric(self):
        """Test error handling for invalid metric."""
        nlat, nlon, ntime = 5, 5, 100
        
        predicted = np.random.randn(nlat, nlon, ntime)
        observed = np.random.randn(nlat, nlon, ntime)
        
        with pytest.raises(ValueError):
            spatial_metrics(predicted, observed, metric='invalid_metric')


class TestEdgeCases:
    """Tests for edge cases in metrics."""
    
    def test_empty_arrays(self):
        """Test metrics with empty arrays."""
        empty = np.array([])
        with pytest.warns(UserWarning):
            cc = calculate_cc(empty, empty)
        assert np.isnan(cc)
    
    def test_single_value(self):
        """Test metrics with single value."""
        single = np.array([5.0])
        with pytest.warns(UserWarning):
            cc = calculate_cc(single, single)
        assert np.isnan(cc)
    
    def test_all_nans(self):
        """Test metrics with all NaN values."""
        all_nan = np.full(100, np.nan)
        with pytest.warns(UserWarning):
            cc = calculate_cc(all_nan, all_nan)
        assert np.isnan(cc)
    
    def test_constant_arrays(self):
        """Test metrics with constant arrays."""
        const = np.ones(100) * 5
        # CC of constant with constant is undefined
        with pytest.warns(RuntimeWarning):
            cc = calculate_cc(const, const)
        # RMSE should be 0
        rmse = calculate_rmse(const, const)
        assert np.isclose(rmse, 0.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
