# ETCC Methodology Documentation

## Theoretical Background

### Triple Collocation (TC) Method

The Triple Collocation method was originally developed for error characterization in remote sensing (Stoffelen, 1998). It requires three independent measurements of the same variable.

#### Assumptions

1. **Independence**: Errors in the three products are independent
2. **Zero mean errors**: E[εⱼ] = 0 for j ∈ {x, y, z}
3. **Uncorrelated errors**: Errors are uncorrelated with the truth

#### Linear Model

Each measurement j relates to the truth R through:

```
j = αⱼ·R + Bⱼ + εⱼ
```

Where:
- αⱼ: Multiplicative calibration factor
- Bⱼ: Additive bias
- εⱼ: Random error

#### Error Variance Estimation

Using the covariance between product pairs:

```
Cxy = αx·αy·σ²R
Cxz = αx·αz·σ²R
Cyz = αy·αz·σ²R
```

We can solve for error variances without knowing R:

```
σ²x = Cxx - (Cxy·Cxz)/Cyz
σ²y = Cyy - (Cxy·Cyz)/Cxz
σ²z = Czz - (Cxz·Cyz)/Cxy
```

#### Optimal Weights

The TC method minimizes error variance to derive weights:

```
wx = (σ²y·σ²z) / (σ²x·σ²y + σ²x·σ²z + σ²y·σ²z)
wy = (σ²x·σ²z) / (σ²x·σ²y + σ²x·σ²z + σ²y·σ²z)
wz = (σ²x·σ²y) / (σ²x·σ²y + σ²x·σ²z + σ²y·σ²z)
```

These weights satisfy: wx + wy + wz = 1

### Extended Triple Collocation for maximized Correlation (ETCC)

ETCC extends TC by explicitly maximizing correlation rather than minimizing error variance.

#### Correlation with Truth

Using extended TC (McColl et al., 2014), we estimate correlation with truth:

```
ρRx = √(ρxy·ρxz / ρyz)
ρRy = √(ρxy·ρyz / ρxz)
ρRz = √(ρxz·ρyz / ρxy)
```

Where ρxy, ρxz, ρyz are the observable pairwise correlations.

#### Correlation Function

The key innovation of ETCC is the correlation function between merged product and truth:

```
ρRM = (1/√CMM) · {wx·ρRx·√Cxx + wy·ρRy·√Cyy + wz·ρRz·√Czz}
```

Where:
- CMM: Variance of merged product M = wx·x + wy·y + wz·z
- Cxx, Cyy, Czz: Variances of input products

#### Optimization

Instead of analytical differentiation (which is intractable due to CMM depending on weights), ETCC uses exhaustive search:

1. Generate all weight combinations with increment Δw (default 0.01)
2. For each combination satisfying wx + wy + wz = 1:
   - Calculate ρRM(wx, wy, wz)
   - Track maximum
3. Return weights that maximize ρRM

**Computational complexity**: O(1/Δw²) evaluations

For Δw = 0.01: approximately 5,151 combinations to evaluate

## Key Differences: TC vs ETCC

| Aspect | TC | ETCC |
|--------|----|----- |
| Objective | Minimize error variance | Maximize correlation |
| Metric | RMSE | Correlation coefficient |
| Weight derivation | Analytical (closed-form) | Numerical (exhaustive search) |
| Computation | Fast (direct calculation) | Slower (iterative search) |
| Theory | Well-established (1998) | Recent extension (2023) |
| Best for | Reducing absolute errors | Capturing relative dynamics |

## Implementation Details

### Numerical Considerations

1. **Minimum correlation threshold**: Set ρ ≥ 0.01 to avoid division by zero
2. **Weight normalization**: Ensure Σw = 1 through normalization
3. **Negative variance handling**: Set to small positive value if negative
4. **Missing data**: Remove NaN values before calculation

### Recommended Parameters

- **Weight increment**: 0.01 for final products, 0.05 for rapid prototyping
- **Minimum correlation**: 0.01 (from Wei et al., 2023)
- **Time series length**: Minimum 30 samples, recommended >100 for stability

### When to Use Each Method

**Use TC when**:
- Minimizing absolute errors (RMSE) is primary goal
- Computational efficiency is critical
- Working with large spatial grids
- Bias correction is paramount

**Use ETCC when**:
- Temporal correlation is more important than absolute accuracy
- Working with anomaly detection or trend analysis
- Products have different biases but similar dynamics
- Available computational resources permit

## Validation Strategy

### Reference Data Requirements

Ideal reference (gauge) data should:
1. Be independent from all three input products
2. Have high spatial/temporal coverage
3. Have known accuracy characteristics
4. Match the spatial scale of analysis

### Evaluation Metrics

**Primary metrics** (from paper):
- Correlation Coefficient (CC): Measures temporal agreement
- Root Mean Square Error (RMSE): Measures absolute differences

**Additional metrics**:
- Mean Absolute Error (MAE): Less sensitive to outliers
- Bias: Systematic over/underestimation
- Relative bias: Percentage difference

### Cross-Validation

For robust evaluation:
1. Split data temporally (not randomly to preserve structure)
2. Use k-fold cross-validation (k=5 recommended)
3. Report mean ± std across folds
4. Test statistical significance of differences

## Extensions and Applications

### Beyond Precipitation

ETCC has been successfully applied to:
- Soil moisture (Kim et al., 2015)
- Evapotranspiration (Baik et al., 2018)
- Snow depth (He et al., 2023)
- Terrestrial water storage (Yin & Park, 2021)

### Potential Improvements

1. **Adaptive weighting**: Vary weights by season or climate region
2. **Multi-scale fusion**: Apply at different temporal aggregations
3. **Machine learning integration**: Use ML to predict optimal weights
4. **Uncertainty quantification**: Provide confidence intervals for merged product
5. **Detection-based merging**: Separate rain/no-rain detection and magnitude estimation

## References

1. Wei, L., et al. (2023). An extended triple collocation method with maximized correlation for near global-land precipitation fusion. Geophysical Research Letters, 50, e2023GL105120.

2. Stoffelen, A. (1998). Toward the true near-surface wind speed: Error modeling and calibration using triple collocation. Journal of Geophysical Research, 103(C4), 7755-7766.

3. McColl, K. A., et al. (2014). Extended triple collocation: Estimating errors and correlation coefficients with respect to an unknown target. Geophysical Research Letters, 41(17), 6229-6236.

4. Gruber, A., et al. (2016). Recent advances in (soil moisture) triple collocation analysis. International Journal of Applied Earth Observation and Geoinformation, 45, 200-211.

5. Kim, S., et al. (2015). A framework for combining multiple soil moisture retrievals based on maximizing temporal correlation. Geophysical Research Letters, 42(16), 6662-6670.
