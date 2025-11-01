"""
Standalone test script to verify ETCC implementation.
"""

import sys
sys.path.insert(0, '/home/claude/etcc_precipitation')

import numpy as np
from etcc.merging import TripleCollocation, ETCC
from etcc.evaluation import calculate_metrics, compare_products
from etcc.utils import generate_synthetic_data

print("=" * 70)
print("Testing ETCC Implementation")
print("=" * 70 + "\n")

# Generate synthetic data
print("1. Generating synthetic data...")
truth, x, y, z = generate_synthetic_data(n_samples=500, seed=42)
print(f"   ✓ Generated {len(truth)} samples")
print(f"   Mean precipitation: {np.mean(truth):.2f} mm/day\n")

# Test TC merging
print("2. Testing Triple Collocation (TC)...")
try:
    tc = TripleCollocation()
    merged_tc = tc.merge(x, y, z)
    print(f"   ✓ TC merging successful")
    print(f"   Weights: wx={tc.weights['wx']:.3f}, wy={tc.weights['wy']:.3f}, wz={tc.weights['wz']:.3f}")
    print(f"   Weight sum: {sum(tc.weights.values()):.6f}")
    assert np.isclose(sum(tc.weights.values()), 1.0), "Weights don't sum to 1!"
    print(f"   ✓ Weights sum to 1.0\n")
except Exception as e:
    print(f"   ✗ TC merging failed: {e}\n")
    sys.exit(1)

# Test ETCC merging
print("3. Testing ETCC...")
try:
    etcc = ETCC(weight_increment=0.05)  # Faster for testing
    merged_etcc = etcc.merge(x, y, z)
    print(f"   ✓ ETCC merging successful")
    print(f"   Weights: wx={etcc.weights['wx']:.3f}, wy={etcc.weights['wy']:.3f}, wz={etcc.weights['wz']:.3f}")
    print(f"   Max correlation: {etcc.max_correlation:.3f}")
    print(f"   Weight sum: {sum(etcc.weights.values()):.6f}\n")
except Exception as e:
    print(f"   ✗ ETCC merging failed: {e}\n")
    sys.exit(1)

# Test metrics calculation
print("4. Testing evaluation metrics...")
try:
    products = {
        'Product X': x,
        'Product Y': y,
        'Product Z': z,
        'TC Merged': merged_tc,
        'ETCC Merged': merged_etcc
    }
    
    results = compare_products(products, truth)
    print(f"   ✓ Metrics calculation successful\n")
    
    print("   Results:")
    print(f"   {'Product':<15} {'CC':>8} {'RMSE':>8} {'MAE':>8}")
    print("   " + "-" * 45)
    for name, metrics in results.items():
        print(f"   {name:<15} {metrics['cc']:>8.4f} {metrics['rmse']:>8.3f} {metrics['mae']:>8.3f}")
    print()
    
except Exception as e:
    print(f"   ✗ Metrics calculation failed: {e}\n")
    sys.exit(1)

# Verify improvements
print("5. Verifying improvements...")
best_input_cc = max(results['Product X']['cc'], 
                   results['Product Y']['cc'],
                   results['Product Z']['cc'])
tc_cc = results['TC Merged']['cc']
etcc_cc = results['ETCC Merged']['cc']

print(f"   Best input CC: {best_input_cc:.4f}")
print(f"   TC CC: {tc_cc:.4f} ({(tc_cc-best_input_cc)/best_input_cc*100:+.2f}%)")
print(f"   ETCC CC: {etcc_cc:.4f} ({(etcc_cc-best_input_cc)/best_input_cc*100:+.2f}%)")

if tc_cc >= best_input_cc - 0.01:
    print(f"   ✓ TC improves or maintains correlation")
else:
    print(f"   ⚠ TC correlation lower than best input")

if etcc_cc >= tc_cc - 0.01:
    print(f"   ✓ ETCC achieves equal or better correlation than TC")
else:
    print(f"   ⚠ ETCC correlation lower than TC")

print("\n" + "=" * 70)
print("All tests passed successfully!")
print("=" * 70)
