"""
Physics and Bound Constraints for Data Fusion
==============================================

Defines constraint classes for constrained optimization in data fusion:

1. **Sum-to-one**: Σw_k = 1 (standard unbiased fusion)
2. **Bounds**: w_min ≤ w_k ≤ w_max (prevent negative/extreme weights)
3. **Energy balance**: E_canopy + E_transpiration ≈ E_total
4. **ET constraint**: 0 ≤ ET ≤ PET (physical plausibility)

Constraints can be combined and used with QP solvers.

Examples
--------
>>> from collocation.fusion import (
...     SumToOneConstraint,
...     BoundsConstraint,
...     EnergyBalanceConstraint,
... )
>>>
>>> # Sum-to-one constraint
>>> c1 = SumToOneConstraint(n_models=3)
>>> A_eq, b_eq = c1.as_matrices()
>>> print(A_eq)  # [[1, 1, 1]]
>>> print(b_eq)  # [1.0]
>>>
>>> # Bounds constraint
>>> c2 = BoundsConstraint(n_models=3, w_min=0.0, w_max=1.0)
>>> A_ineq, b_ineq = c2.as_matrices()
>>>
>>> # Energy balance
>>> c3 = EnergyBalanceConstraint(
...     ec_index=0,
...     et_index=1,
...     e_total_index=2,
...     tolerance=0.1,
... )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import xarray as xr


class Constraint(ABC):
    """
    Abstract base class for fusion constraints.

    All constraints can be converted to matrix form for use in QP solvers.
    """

    @abstractmethod
    def as_matrices(self) -> Tuple[np.ndarray | None, np.ndarray | None]:
        """
        Convert constraint to matrix form for QP solvers.

        Returns
        -------
        A : np.ndarray or None
            Constraint matrix.
        b : np.ndarray or None
            Constraint RHS vector.

        Notes
        -----
        Equality constraints: A_eq w = b_eq
        Inequality constraints: A_ineq w ≤ b_ineq
        """
        pass

    @abstractmethod
    def check_feasibility(self, weights: np.ndarray) -> bool:
        """
        Check if weights satisfy the constraint.

        Parameters
        ----------
        weights : np.ndarray
            Weight vector to check.

        Returns
        -------
        bool
            True if constraint is satisfied.
        """
        pass

    def violation_magnitude(self, weights: np.ndarray) -> float:
        """
        Compute magnitude of constraint violation.

        Parameters
        ----------
        weights : np.ndarray
            Weight vector.

        Returns
        -------
        float
            Violation magnitude (0 if feasible).
        """
        A, b = self.as_matrices()
        if A is None or b is None:
            return 0.0

        residual = A @ weights - b
        return float(np.linalg.norm(residual))


class SumToOneConstraint(Constraint):
    """
    Sum-to-one constraint: Σw_k = 1.

    Standard constraint for unbiased fusion.

    Parameters
    ----------
    n_models : int
        Number of models (length of weight vector).

    Examples
    --------
    >>> c = SumToOneConstraint(n_models=3)
    >>> A, b = c.as_matrices()
    >>> print(A)
    [[1. 1. 1.]]
    >>> print(b)
    [1.]
    """

    def __init__(self, n_models: int):
        if n_models < 1:
            raise ValueError("n_models must be at least 1")
        self.n_models = n_models

    def as_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return equality constraint: [1, 1, ..., 1] w = 1."""
        A = np.ones((1, self.n_models), dtype=float)
        b = np.array([1.0], dtype=float)
        return A, b

    def check_feasibility(self, weights: np.ndarray, tol: float = 1e-6) -> bool:
        """Check if sum of weights equals 1."""
        return abs(weights.sum() - 1.0) < tol


class BoundsConstraint(Constraint):
    """
    Box bounds: w_min ≤ w_k ≤ w_max.

    Prevents negative weights or extreme concentration on single model.

    Parameters
    ----------
    n_models : int
        Number of models.
    w_min : float, default=0.0
        Lower bound for each weight.
    w_max : float, default=1.0
        Upper bound for each weight.

    Examples
    --------
    >>> c = BoundsConstraint(n_models=3, w_min=0.0, w_max=1.0)
    >>> A, b = c.as_matrices()
    >>> # Returns inequality: w >= w_min and w <= w_max
    """

    def __init__(self, n_models: int, w_min: float = 0.0, w_max: float = 1.0):
        if n_models < 1:
            raise ValueError("n_models must be at least 1")
        if w_min >= w_max:
            raise ValueError("w_min must be less than w_max")

        self.n_models = n_models
        self.w_min = w_min
        self.w_max = w_max

    def as_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return inequality constraints in form: A w ≤ b.

        Converts:
            w_min ≤ w ≤ w_max
        to:
            w ≥ w_min  →  -w ≤ -w_min
            w ≤ w_max  →   w ≤  w_max
        """
        K = self.n_models

        # w >= w_min  →  -I w <= -w_min
        A_lower = -np.eye(K)
        b_lower = np.full(K, -self.w_min)

        # w <= w_max  →  I w <= w_max
        A_upper = np.eye(K)
        b_upper = np.full(K, self.w_max)

        A = np.vstack([A_lower, A_upper])
        b = np.concatenate([b_lower, b_upper])

        return A, b

    def check_feasibility(self, weights: np.ndarray, tol: float = 1e-6) -> bool:
        """Check if all weights are within bounds."""
        return np.all(weights >= self.w_min - tol) and np.all(weights <= self.w_max + tol)


class EnergyBalanceConstraint(Constraint):
    """
    Energy balance constraint: E_canopy + E_transpiration ≈ E_total.

    For ecosystem studies where:
    - E_canopy (Ec): canopy evaporation
    - E_transpiration (Et): transpiration
    - E_total (E): total evapotranspiration

    Enforces: |Ec + Et - E| ≤ ε

    Parameters
    ----------
    ec_index : int
        Index of canopy evaporation model.
    et_index : int
        Index of transpiration model.
    e_total_index : int
        Index of total evapotranspiration model.
    tolerance : float, default=0.1
        Allowed deviation from energy balance.

    Notes
    -----
    This is a *soft* constraint encoded as inequality:
        (Ec + Et) - E ≤ ε
        E - (Ec + Et) ≤ ε
    """

    def __init__(
        self,
        ec_index: int,
        et_index: int,
        e_total_index: int,
        n_models: int | None = None,
        tolerance: float = 0.1,
    ):
        self.ec_index = ec_index
        self.et_index = et_index
        self.e_total_index = e_total_index
        self.tolerance = tolerance

        # Infer n_models if not provided
        if n_models is None:
            self.n_models = max(ec_index, et_index, e_total_index) + 1
        else:
            self.n_models = n_models

    def as_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode as inequality constraints.

        (w_ec * Ec + w_et * Et) - w_e * E <= tol
        w_e * E - (w_ec * Ec + w_et * Et) <= tol

        This is a linear constraint on fusion weights, assuming the
        energy balance check is applied *after* fusion.
        """
        # NOTE: This constraint is more naturally checked post-fusion
        # on the fused values rather than on the weights themselves.
        # We return None to indicate this is a post-hoc check.
        return None, None

    def check_feasibility(
        self,
        weights: np.ndarray,
        ec_values: np.ndarray | None = None,
        et_values: np.ndarray | None = None,
        e_total_values: np.ndarray | None = None,
    ) -> bool:
        """
        Check energy balance feasibility.

        Requires actual values of Ec, Et, E to check balance.

        Parameters
        ----------
        weights : np.ndarray
            Fusion weights.
        ec_values : np.ndarray
            Canopy evaporation values from each model.
        et_values : np.ndarray
            Transpiration values from each model.
        e_total_values : np.ndarray
            Total ET values from each model.

        Returns
        -------
        bool
            True if energy balance is satisfied within tolerance.
        """
        if ec_values is None or et_values is None or e_total_values is None:
            # Cannot check without values
            return True

        # Compute fused values
        ec_fused = weights[self.ec_index] * ec_values[self.ec_index]
        et_fused = weights[self.et_index] * et_values[self.et_index]
        e_fused = weights[self.e_total_index] * e_total_values[self.e_total_index]

        # Check balance
        balance_error = abs((ec_fused + et_fused) - e_fused)

        return balance_error <= self.tolerance


class ETConstraint(Constraint):
    """
    Physical plausibility constraint: 0 ≤ ET ≤ PET.

    Evapotranspiration must be non-negative and not exceed potential ET.

    Parameters
    ----------
    et_index : int
        Index of ET model in fusion.
    pet_values : np.ndarray or xr.DataArray
        Potential ET values (can be spatially/temporally varying).
    n_models : int, optional
        Total number of models.

    Notes
    -----
    This is a post-fusion constraint that checks the fused ET field.
    """

    def __init__(
        self,
        et_index: int,
        pet_values: np.ndarray | xr.DataArray,
        n_models: int | None = None,
    ):
        self.et_index = et_index
        self.pet_values = pet_values

        if n_models is None:
            self.n_models = et_index + 1
        else:
            self.n_models = n_models

    def as_matrices(self) -> Tuple[None, None]:
        """ET constraint is applied post-fusion, not on weights."""
        return None, None

    def check_feasibility(
        self,
        weights: np.ndarray,
        et_values: np.ndarray | xr.DataArray | None = None,
    ) -> bool:
        """
        Check if fused ET satisfies 0 ≤ ET ≤ PET.

        Parameters
        ----------
        weights : np.ndarray
            Fusion weights.
        et_values : np.ndarray or xr.DataArray
            ET values from each model.

        Returns
        -------
        bool
            True if constraint satisfied everywhere.
        """
        if et_values is None:
            return True

        # Compute fused ET
        if isinstance(et_values, xr.DataArray):
            et_fused = (weights * et_values).sum(dim="model")
        else:
            et_fused = weights[self.et_index] * et_values[self.et_index]

        # Check bounds
        non_negative = np.all(et_fused >= 0)
        below_pet = np.all(et_fused <= self.pet_values)

        return non_negative and below_pet


def combine_constraints(
    constraints: List[Constraint],
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """
    Combine multiple constraints into unified matrix form.

    Parameters
    ----------
    constraints : list of Constraint
        List of constraint objects.

    Returns
    -------
    A_combined : np.ndarray or None
        Combined constraint matrix.
    b_combined : np.ndarray or None
        Combined RHS vector.

    Examples
    --------
    >>> c1 = SumToOneConstraint(n_models=3)
    >>> c2 = BoundsConstraint(n_models=3, w_min=0.0, w_max=1.0)
    >>> A, b = combine_constraints([c1, c2])
    """
    A_list = []
    b_list = []

    for c in constraints:
        A, b = c.as_matrices()
        if A is not None and b is not None:
            A_list.append(A)
            b_list.append(b)

    if not A_list:
        return None, None

    A_combined = np.vstack(A_list)
    b_combined = np.concatenate(b_list)

    return A_combined, b_combined


def check_all_constraints(
    constraints: List[Constraint],
    weights: np.ndarray,
    **kwargs,
) -> Tuple[bool, List[bool]]:
    """
    Check if weights satisfy all constraints.

    Parameters
    ----------
    constraints : list of Constraint
        Constraints to check.
    weights : np.ndarray
        Weight vector.
    **kwargs
        Additional data needed for checking (e.g., ec_values, pet_values).

    Returns
    -------
    all_satisfied : bool
        True if all constraints satisfied.
    individual_results : list of bool
        Per-constraint satisfaction status.
    """
    results = []

    for c in constraints:
        # Try to check with provided kwargs
        try:
            satisfied = c.check_feasibility(weights, **kwargs)
        except TypeError:
            # Constraint doesn't need extra kwargs
            satisfied = c.check_feasibility(weights)

        results.append(satisfied)

    return all(results), results
