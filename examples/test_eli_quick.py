"""
Quick Test Script for ELI Module
=================================

This script performs a quick test of the ELI module to ensure
everything is working correctly.

Run this script to verify the installation:
    python test_eli_quick.py
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*70)
print(" "*20 + "ELI MODULE QUICK TEST")
print("="*70)

# Test 1: Import modules
print("\n[Test 1] Importing modules...")
try:
    from collocation import (
        ELIProcessor,
        process_eli_data,
        calculate_eli_index,
        ivd,
        eivd,
        tc
    )
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create synthetic data
print("\n[Test 2] Creating synthetic data...")
try:
    np.random.seed(42)
    n_time, n_lat, n_lon = 100, 10, 10

    # Generate simple synthetic data
    true_signal = np.random.randn(n_time, n_lat, n_lon)

    data1 = true_signal + 0.2 * np.random.randn(n_time, n_lat, n_lon)
    data2 = true_signal + 0.3 * np.random.randn(n_time, n_lat, n_lon)
    data3 = true_signal + 0.25 * np.random.randn(n_time, n_lat, n_lon)

    print(f"✓ Generated data with shape: {data1.shape}")
except Exception as e:
    print(f"✗ Data generation failed: {e}")
    sys.exit(1)

# Test 3: Test IVD method
print("\n[Test 3] Testing IVD method...")
try:
    processor = ELIProcessor()
    results_ivd = processor.process_dual_ivd(data1, data2, variable='test')

    assert 'error_variance' in results_ivd
    assert 'rho2' in results_ivd
    assert 'weights' in results_ivd
    assert 'merged' in results_ivd

    print("✓ IVD method works correctly")
    print(f"  - Error variance shape: {results_ivd['error_variance'].shape}")
    print(f"  - Merged product shape: {results_ivd['merged'].shape}")
except Exception as e:
    print(f"✗ IVD test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test EIVD method
print("\n[Test 4] Testing EIVD method...")
try:
    results_eivd = processor.process_triple_eivd(
        data1, data2, data3, variable='test'
    )

    assert 'error_variance' in results_eivd
    assert 'error_cross_correlation' in results_eivd
    assert 'merged' in results_eivd

    print("✓ EIVD method works correctly")
    print(f"  - Error cross-correlation shape: "
          f"{results_eivd['error_cross_correlation'].shape}")
except Exception as e:
    print(f"✗ EIVD test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test TC method
print("\n[Test 5] Testing TC method...")
try:
    results_tc = processor.process_triple_tc(
        data1, data2, data3, variable='test'
    )

    assert 'error_variance' in results_tc
    assert 'snr' in results_tc
    assert 'rho2' in results_tc

    print("✓ TC method works correctly")
    print(f"  - Mean SNR: {np.nanmean(results_tc['snr'], axis=(0,1))}")
except Exception as e:
    print(f"✗ TC test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test time series analysis
print("\n[Test 6] Testing time series analysis...")
try:
    # Extract time series at one location
    i, j = 5, 5
    ts1 = data1[:, i, j]
    ts2 = data2[:, i, j]
    ts3 = data3[:, i, j]

    tri = np.column_stack([ts1, ts2, ts3])

    # Apply IVD on dual
    dual = tri[:, :2]
    EeeT_ivd, rho2_ivd, u_ivd = ivd(dual)
    print("✓ IVD on time series works")
    print(f"  - Weights: {u_ivd}")

    # Apply EIVD
    EeeT_eivd, SNR_eivd, rho2_eivd, fMSE_eivd, L_eivd = eivd(tri)
    print("✓ EIVD on time series works")
    print(f"  - Error cross-corr: {EeeT_eivd[1,2]:.6f}")

    # Apply TC
    EeeT_tc, SNR_tc, rho2_tc, fMSE_tc = tc(tri)
    print("✓ TC on time series works")
    print(f"  - SNR: {SNR_tc}")

except Exception as e:
    print(f"✗ Time series test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test NetCDF export
print("\n[Test 7] Testing NetCDF export...")
try:
    output_dir = Path('test_output')
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / 'test_eli_results.nc'

    processor.save_to_netcdf(
        results_eivd,
        output_file,
        variable='test',
        data_source='synthetic',
        metadata={'test': 'quick_test'}
    )

    assert output_file.exists()
    print(f"✓ NetCDF export works")
    print(f"  - File created: {output_file}")

    # Clean up
    import shutil
    shutil.rmtree(output_dir)
    print("  - Cleaned up test files")

except Exception as e:
    print(f"✗ NetCDF export test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Compare methods
print("\n[Test 8] Testing method comparison...")
try:
    comparison = processor.compare_methods({
        'eivd': results_eivd,
        'tc': results_tc
    })

    assert 'methods_used' in comparison
    assert 'recommendations' in comparison

    print("✓ Method comparison works")
    print(f"  - Methods compared: {comparison['methods_used']}")
    if comparison['recommendations']:
        print(f"  - Recommendations: {comparison['recommendations'][0]}")

except Exception as e:
    print(f"✗ Method comparison test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*70)
print(" "*20 + "ALL TESTS PASSED ✓")
print("="*70)
print("\nThe ELI module is working correctly!")
print("\nNext steps:")
print("  1. Run full example: python eli_comprehensive_example.py")
print("  2. Process your own data using the ELIProcessor class")
print("  3. See ELI_README.md for detailed documentation")
print("="*70)
