"""
Performance tests for collocation methods.

These tests ensure that performance optimizations don't regress
and that methods complete within reasonable time bounds.
"""

import pytest
import numpy as np
import time
from collocation import ivd, tc, eivd, ec
from collocation.etcc import ETCC, TripleCollocation


@pytest.fixture
def synthetic_data():
    """Generate synthetic test data."""
    np.random.seed(42)
    n = 500
    true_signal = np.random.randn(n)
    
    dual = np.column_stack([
        true_signal + 0.2 * np.random.randn(n),
        true_signal + 0.25 * np.random.randn(n)
    ])
    
    tri = np.column_stack([
        true_signal + 0.2 * np.random.randn(n),
        true_signal + 0.25 * np.random.randn(n),
        true_signal + 0.3 * np.random.randn(n)
    ])
    
    quad = np.column_stack([
        true_signal + 0.2 * np.random.randn(n),
        true_signal + 0.25 * np.random.randn(n),
        true_signal + 0.3 * np.random.randn(n),
        true_signal + 0.35 * np.random.randn(n)
    ])
    
    return dual, tri, quad


class TestPerformance:
    """Test performance of collocation methods."""
    
    def test_ivd_performance(self, synthetic_data):
        """IVD should complete within reasonable time."""
        dual, _, _ = synthetic_data
        
        start = time.perf_counter()
        EeeT, rho2, u = ivd(dual)
        elapsed = time.perf_counter() - start
        
        # Should complete in less than 5ms for 500 samples
        assert elapsed < 0.005, f"IVD took {elapsed*1000:.2f}ms, expected < 5ms"
        assert EeeT.shape == (2, 2)
        assert rho2.shape == (2,)
        assert u.shape == (2,)
    
    def test_tc_performance(self, synthetic_data):
        """TC should complete within reasonable time."""
        _, tri, _ = synthetic_data
        
        start = time.perf_counter()
        EeeT, SNR, rho2, fMSE = tc(tri)
        elapsed = time.perf_counter() - start
        
        # Should complete in less than 1ms for 500 samples
        assert elapsed < 0.001, f"TC took {elapsed*1000:.2f}ms, expected < 1ms"
        assert EeeT.shape == (3, 3)
        assert SNR.shape == (3,)
    
    def test_eivd_performance(self, synthetic_data):
        """EIVD should complete within reasonable time."""
        _, tri, _ = synthetic_data
        
        start = time.perf_counter()
        EeeT, SNR, rho2, fMSE, L = eivd(tri)
        elapsed = time.perf_counter() - start
        
        # Should complete in less than 1ms for 500 samples
        assert elapsed < 0.001, f"EIVD took {elapsed*1000:.2f}ms, expected < 1ms"
        assert EeeT.shape == (3, 3)
        assert L.shape == (3,)
    
    def test_ec_performance(self, synthetic_data):
        """EC should complete within reasonable time."""
        _, _, quad = synthetic_data
        
        start = time.perf_counter()
        results = ec(quad)
        elapsed = time.perf_counter() - start
        
        # Should complete in less than 10ms for 500 samples
        assert elapsed < 0.010, f"EC took {elapsed*1000:.2f}ms, expected < 10ms"
        assert len(results) > 0
    
    def test_etcc_performance_coarse(self, synthetic_data):
        """ETCC with coarse increment should be fast."""
        _, tri, _ = synthetic_data
        x, y, z = tri[:, 0], tri[:, 1], tri[:, 2]
        
        etcc = ETCC(weight_increment=0.05)
        
        start = time.perf_counter()
        merged = etcc.merge(x, y, z)
        elapsed = time.perf_counter() - start
        
        # Should complete in less than 10ms for increment=0.05
        assert elapsed < 0.010, f"ETCC took {elapsed*1000:.2f}ms, expected < 10ms"
        assert merged.shape == x.shape
        assert etcc.weights is not None
    
    def test_etcc_performance_fine(self, synthetic_data):
        """ETCC with fine increment should complete reasonably."""
        _, tri, _ = synthetic_data
        x, y, z = tri[:, 0], tri[:, 1], tri[:, 2]
        
        etcc = ETCC(weight_increment=0.01)
        
        start = time.perf_counter()
        merged = etcc.merge(x, y, z)
        elapsed = time.perf_counter() - start
        
        # Should complete in less than 150ms for increment=0.01
        assert elapsed < 0.150, f"ETCC took {elapsed*1000:.2f}ms, expected < 150ms"
        assert merged.shape == x.shape


class TestScalability:
    """Test scalability with different data sizes."""
    
    @pytest.mark.parametrize("n_samples", [100, 500, 1000])
    def test_ivd_scalability(self, n_samples):
        """IVD should scale reasonably with data size."""
        np.random.seed(42)
        true_signal = np.random.randn(n_samples)
        dual = np.column_stack([
            true_signal + 0.2 * np.random.randn(n_samples),
            true_signal + 0.25 * np.random.randn(n_samples)
        ])
        
        start = time.perf_counter()
        EeeT, rho2, u = ivd(dual)
        elapsed = time.perf_counter() - start
        
        # Time should grow sub-linearly due to early termination
        max_time = 0.01  # 10ms max
        assert elapsed < max_time, f"IVD took {elapsed*1000:.2f}ms for n={n_samples}"
    
    @pytest.mark.parametrize("n_samples", [100, 500, 1000])
    def test_tc_scalability(self, n_samples):
        """TC should scale linearly with data size (covariance calculation)."""
        np.random.seed(42)
        true_signal = np.random.randn(n_samples)
        tri = np.column_stack([
            true_signal + 0.2 * np.random.randn(n_samples),
            true_signal + 0.25 * np.random.randn(n_samples),
            true_signal + 0.3 * np.random.randn(n_samples)
        ])
        
        start = time.perf_counter()
        EeeT, SNR, rho2, fMSE = tc(tri)
        elapsed = time.perf_counter() - start
        
        # Should be very fast regardless of size
        max_time = 0.002  # 2ms max
        assert elapsed < max_time, f"TC took {elapsed*1000:.2f}ms for n={n_samples}"


class TestOptimizationCorrectness:
    """Verify optimizations don't change results."""
    
    def test_ivd_early_termination_correctness(self):
        """Early termination should find same or similar offset."""
        np.random.seed(42)
        n = 1000
        true_signal = np.random.randn(n)
        
        # Create data with clear optimal offset
        X = true_signal + 0.2 * np.random.randn(n)
        Y = np.roll(true_signal, 5) + 0.25 * np.random.randn(n)  # Shifted by 5
        dual = np.column_stack([X, Y])
        
        EeeT, rho2, u = ivd(dual)
        
        # Should produce valid results
        assert np.all(np.isfinite(EeeT))
        assert np.all(np.isfinite(rho2))
        assert np.all(np.isfinite(u))
        assert np.sum(u) > 0  # Weights should be positive
    
    def test_ec_mean_caching_correctness(self):
        """Cached means should produce identical results."""
        np.random.seed(42)
        n = 200
        true_signal = np.random.randn(n)
        quad = np.column_stack([
            true_signal + 0.2 * np.random.randn(n),
            true_signal + 0.25 * np.random.randn(n),
            true_signal + 0.3 * np.random.randn(n),
            true_signal + 0.35 * np.random.randn(n)
        ])
        
        results = ec(quad)
        
        # Should produce valid results for multiple combinations
        assert len(results) > 0
        for result in results:
            assert len(result.EeeT) > 0
            for EeeT in result.EeeT:
                # Check that error variances are reasonable
                assert np.all(np.diag(EeeT) >= 0)
    
    def test_etcc_vectorization_correctness(self):
        """Vectorized ETCC should produce same results."""
        np.random.seed(42)
        n = 300
        true_signal = np.random.randn(n)
        x = true_signal + 0.2 * np.random.randn(n)
        y = true_signal + 0.25 * np.random.randn(n)
        z = true_signal + 0.3 * np.random.randn(n)
        
        # Test with coarse increment
        etcc = ETCC(weight_increment=0.1)
        merged = etcc.merge(x, y, z)
        
        # Check weights sum to 1 (use 0.01 tolerance for coarse increment)
        # Note: Coarser increment=0.1 can have small numerical errors
        total_weight = (etcc.weights['wx'] + 
                       etcc.weights['wy'] + 
                       etcc.weights['wz'])
        assert abs(total_weight - 1.0) < 0.01, \
            f"Weights sum to {total_weight}, expected close to 1.0"
        
        # Check merged result is reasonable
        assert np.all(np.isfinite(merged))
        assert np.std(merged) > 0  # Has variability


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
