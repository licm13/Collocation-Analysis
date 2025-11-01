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

# Add plotting support
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

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

# --------------------------------------------------------------------
# Quick diagnostic figure: RMSE comparison for the classical methods
try:
    # collect RMSE per method (safe access)
    methods = []
    rmse_vals = []
    for method, result in results.items():
        methods.append(method)
        if isinstance(result, dict) and 'rmse' in result and result['rmse'] is not None:
            # try to take the first value or mean of array-like
            try:
                val = float(np.nanmean(result['rmse']))
            except Exception:
                val = np.nan
        else:
            val = np.nan
        rmse_vals.append(val)

    # prepare output directory next to this script
    fig_dir = Path(__file__).resolve().parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    script_stem = Path(__file__).stem
    out_file = fig_dir / f"{script_stem}_rmse_comparison.png"

    # simple bar plot
    plt.figure(figsize=(8, 4))
    x = np.arange(len(methods))
    plt.bar(x, rmse_vals, color='C0', alpha=0.8)
    plt.xticks(x, methods, rotation=45, ha='right')
    plt.ylabel("RMSE")
    plt.title("Quick RMSE Comparison (classical methods)")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved quick diagnostic figure: {out_file}")
except Exception as _err:
    print(f"\nFailed to create/save quick figure: {_err}")
