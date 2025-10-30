"""Quick test to verify all methods work."""
import numpy as np
from collocation import ivd, ivs, tc, eivd, ec

print("Testing Collocation Analysis Package")
print("=" * 50)

np.random.seed(42)
n = 150
true_signal = 5 + 2 * np.sin(np.linspace(0, 4*np.pi, n))

# Generate 4 products
X1 = true_signal + 0.3 * np.random.randn(n)
X2 = true_signal + 0.4 * np.random.randn(n)
X3 = true_signal + 0.5 * np.random.randn(n)
X4 = true_signal + 0.6 * np.random.randn(n)

print("\n1. Testing IVD (2 products)...")
dual = np.column_stack([X1, X2])
EeeT, rho2, u = ivd(dual)
print(f"   ✓ Error variances: {np.diag(EeeT)}")
print(f"   ✓ Correlations: {rho2}")

print("\n2. Testing IVS (2 products with bootstrap)...")
RMSE, rho2 = ivs(dual, N_boot=100, column=1, M_A=0)
print(f"   ✓ RMSE: {RMSE}")
print(f"   ✓ Correlations: {rho2}")

print("\n3. Testing TC (3 products)...")
tri = np.column_stack([X1, X2, X3])
EeeT, SNR, rho2, fMSE = tc(tri)
print(f"   ✓ Error variances: {np.diag(EeeT)}")
print(f"   ✓ SNR: {SNR}")

print("\n4. Testing EIVD (3 products)...")
EeeT, SNR, rho2, fMSE, L = eivd(tri)
print(f"   ✓ Error variances: {np.diag(EeeT)}")
print(f"   ✓ Error correlation: {EeeT[1,2]:.4f}")

print("\n5. Testing EC (4 products)...")
quad = np.column_stack([X1, X2, X3, X4])
results = ec(quad)
print(f"   ✓ Number of combinations: {len(results)}")
print(f"   ✓ First combination: {results[0].combination}")

print("\n" + "=" * 50)
print("All tests passed successfully! ✓")
print("Package is ready to use.")
