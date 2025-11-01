"""
Data Fusion Module
==================

Scientific data fusion framework supporting:
- MSE-only inverse-variance fusion
- Full error-covariance (GLS/BLUE) fusion
- Constrained optimization with physics constraints
- Localization and uncertainty quantification
- Robust estimation with outlier handling

Modules
-------
weights : Weight solvers (IVW, GLS/BLUE, constrained QP)
covariance : Covariance estimation (MSE, TC/IVD, shrinkage, tapering)
fuse : High-level fusion orchestrator
robust : Robust losses and outlier detection
constraints : Physics and bound constraints
uncertainty : Variance propagation and diagnostics
localization : Spatial/temporal localization

References
----------
.. [1] Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for
       large-dimensional covariance matrices. Journal of Multivariate Analysis.
.. [2] Gaspari, G., & Cohn, S. E. (1999). Construction of correlation functions
       in two and three dimensions. Quarterly Journal of the Royal Meteorological
       Society.
"""

from __future__ import annotations

__version__ = "1.0.0"

# Import main APIs
from .covariance import (
    estimate_mse,
    build_sigma,
    estimate_cross_covariance,
    apply_covariance_tapering,
)
from .weights import (
    solve_weights_ivw,
    solve_weights_gls,
    solve_weights_qp,
)
from .fuse import fuse_fields
from .constraints import (
    Constraint,
    SumToOneConstraint,
    BoundsConstraint,
    EnergyBalanceConstraint,
    ETConstraint,
    combine_constraints,
)
from .uncertainty import (
    propagate_variance,
    compute_effective_n,
    bootstrap_uncertainty,
    weight_entropy,
)
from .robust import (
    estimate_mse_robust,
    huber_loss,
    detect_outliers,
)
from .localization import (
    apply_moving_window,
    partition_by_biome,
)

__all__ = [
    # Covariance
    "estimate_mse",
    "build_sigma",
    "estimate_cross_covariance",
    "apply_covariance_tapering",
    # Weights
    "solve_weights_ivw",
    "solve_weights_gls",
    "solve_weights_qp",
    # Fusion
    "fuse_fields",
    # Constraints
    "Constraint",
    "SumToOneConstraint",
    "BoundsConstraint",
    "EnergyBalanceConstraint",
    "ETConstraint",
    # Uncertainty
    "propagate_variance",
    "compute_effective_n",
    "bootstrap_uncertainty",
    "weight_entropy",
    # Robust
    "estimate_mse_robust",
    "huber_loss",
    "detect_outliers",
    # Localization
    "apply_moving_window",
    "partition_by_biome",
]
