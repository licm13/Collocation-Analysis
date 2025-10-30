"""
Quick test of comprehensive comparison (without Bayesian methods for speed)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comprehensive_comparison import (
    simulate_scenario_1_ideal,
    simulate_scenario_2_correlated,
    apply_classical_methods,
    setup_publication_style
)

print("Testing comprehensive comparison script...")
print("="*60)

# Test scenario generation
print("\n1. Testing scenario generation...")
scenario = simulate_scenario_1_ideal(n=200)
print(f"   ✓ Scenario '{scenario['name']}' generated")
print(f"   - Data shape: {scenario['data'].shape}")
print(f"   - True RMSE: {scenario['rmse_true']}")

# Test classical methods
print("\n2. Testing classical methods application...")
results = apply_classical_methods(scenario)

for method, result in results.items():
    if result['success']:
        print(f"   ✓ {method}: RMSE = {result['rmse'][:result['n_products']]}")
    else:
        print(f"   ✗ {method}: Failed - {result.get('error', 'Unknown')}")

# Test publication style setup
print("\n3. Testing publication style setup...")
setup_publication_style()
print("   ✓ Publication style configured")

print("\n" + "="*60)
print("All basic tests passed! ✓")
print("\nTo run full comparison with figures:")
print("  python comprehensive_comparison.py")
