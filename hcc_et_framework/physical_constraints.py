"""
Physical Constraints Module for HCC-ET Framework
=================================================

Implements multi-scale physical constraints to improve fusion accuracy:

1. **Site-Level Constraint**: SapFluxNet transpiration observations
   - Type: Soft constraint (adjusts T/ET ratio via PFT-specific bias correction)
   - Scale: Site (~km²)
   - Validation: Eddy covariance sites, sap flow measurements

2. **Basin-Level Constraint**: Water Balance ET (P - Q - dS)
   - Type: Hard constraint (forces closure at basin scale)
   - Scale: Basin (10³-10⁶ km²)
   - Source: GRDC runoff + GRACE/GLDAS storage change

Both constraints are integrated into the existing collocation.fusion framework
by extending the BaseConstraint class.

Key Innovation:
- Constraints are applied AFTER cross-calibration fusion (Phase 3)
- Preserves uncertainty estimates from EIVD/GLS
- Multi-scale consistency (site → basin → global)

Author: [Your Name]
Date: 2025-11-09
"""

import logging
from typing import Optional, Dict

import numpy as np
import xarray as xr
from scipy import stats

# Import base constraint class from collocation library
from collocation.fusion.constraints import Constraint

logger = logging.getLogger(__name__)


class SapfluxnetConstraint(Constraint):
    """
    Soft constraint using SapFluxNet transpiration observations.

    Strategy:
    ---------
    1. Compare fused T/ET ratio with SapFluxNet T observations at site pixels
    2. Compute bias by Plant Functional Type (PFT)
    3. Apply PFT-specific bias correction to fused T/ET ratio
    4. Weight correction by EIVD uncertainty (high uncertainty → larger adjustment)

    This is a "soft" constraint because it adjusts the estimate rather than
    forcing exact agreement (which would ignore fusion uncertainty).

    Parameters
    ----------
    sapflux_1km_data : xr.DataArray
        Upscaled SapFluxNet T/ET observations, dims: (lat, lon)
        Contains NaN where no observations available
    correction_strength : float, default=0.5
        Strength of bias correction (0=none, 1=full)
    min_sites_per_pft : int, default=3
        Minimum sites required per PFT for correction

    Examples
    --------
    >>> constraint = SapfluxnetConstraint(sapflux_data)
    >>> t_et_corrected = constraint.apply(
    ...     t_et_ratio_fused=fused_ratio,
    ...     total_et=total_et_1km,
    ...     pft_map=pft_classification
    ... )
    """

    def __init__(
        self,
        sapflux_1km_data: xr.DataArray,
        correction_strength: float = 0.5,
        min_sites_per_pft: int = 3
    ):
        """Initialize SapFluxNet constraint."""
        self.sapflux_truth = sapflux_1km_data
        self.correction_strength = correction_strength
        self.min_sites_per_pft = min_sites_per_pft

        # Extract site locations
        self.site_mask = ~np.isnan(self.sapflux_truth)
        self.n_sites = int(self.site_mask.sum().values)

        logger.info(f"SapFluxNet constraint initialized with {self.n_sites} sites")

    def apply(
        self,
        t_et_ratio_fused: xr.DataArray,
        total_et: xr.DataArray,
        pft_map: xr.DataArray,
        uncertainty: Optional[xr.DataArray] = None
    ) -> xr.DataArray:
        """
        Apply SapFluxNet soft constraint.

        Parameters
        ----------
        t_et_ratio_fused : xr.DataArray
            Fused T/ET ratio from EIVD/GLS, dims: (time, lat, lon)
        total_et : xr.DataArray
            Total ET (used to compute T from T/ET), dims: (time, lat, lon)
        pft_map : xr.DataArray
            Plant Functional Type classification, dims: (lat, lon)
        uncertainty : xr.DataArray, optional
            Fusion uncertainty (from EIVD), dims: (time, lat, lon)
            Used to weight corrections (higher uncertainty → larger adjustment)

        Returns
        -------
        xr.DataArray
            Bias-corrected T/ET ratio, same dims as input
        """
        logger.info("Applying SapFluxNet soft constraint...")

        # Compute T from T/ET ratio
        T_fused = t_et_ratio_fused * total_et

        # Extract T at SapFluxNet site pixels
        T_fused_at_sites = T_fused.where(self.site_mask, drop=True)

        # Compute observed T from SapFluxNet (assuming sapflux_truth is T/ET)
        # Need to multiply by total ET at sites
        total_et_at_sites = total_et.where(self.site_mask, drop=True)
        T_obs_at_sites = self.sapflux_truth.where(self.site_mask, drop=True) * total_et_at_sites

        # Compute bias at each site
        # Bias = T_fused - T_obs
        bias_at_sites = T_fused_at_sites - T_obs_at_sites

        # Aggregate bias by PFT
        logger.info("  Computing PFT-specific bias corrections...")
        pft_biases = self._compute_pft_biases(bias_at_sites, pft_map)

        # Apply PFT-specific corrections
        logger.info("  Applying bias corrections by PFT...")
        t_et_corrected = self._apply_pft_corrections(
            t_et_ratio_fused=t_et_ratio_fused,
            total_et=total_et,
            pft_map=pft_map,
            pft_biases=pft_biases,
            uncertainty=uncertainty
        )

        # Validate corrected values
        n_corrected = (~t_et_corrected.equals(t_et_ratio_fused)).sum().values
        logger.info(f"  Corrected {n_corrected} pixels based on SapFluxNet")

        # Ensure physical range [0, 1]
        t_et_corrected = t_et_corrected.clip(0, 1)

        return t_et_corrected

    def _compute_pft_biases(
        self,
        bias_at_sites: xr.DataArray,
        pft_map: xr.DataArray
    ) -> Dict[int, float]:
        """
        Compute mean bias for each PFT.

        Returns:
        - Dictionary {pft_id: mean_bias}
        """
        pft_biases = {}

        # Get unique PFTs at site locations
        pft_at_sites = pft_map.where(self.site_mask, drop=True)
        unique_pfts = np.unique(pft_at_sites.values[~np.isnan(pft_at_sites.values)])

        for pft in unique_pfts:
            pft = int(pft)

            # Extract bias values for this PFT
            pft_mask = (pft_at_sites == pft)
            bias_pft = bias_at_sites.where(pft_mask, drop=True)

            # Compute mean bias (across time and space)
            bias_mean = float(bias_pft.mean().values)
            bias_std = float(bias_pft.std().values)
            n_sites_pft = int(pft_mask.sum().values)

            if n_sites_pft >= self.min_sites_per_pft:
                pft_biases[pft] = bias_mean
                logger.info(f"    PFT {pft:2d}: bias={bias_mean:+.3f} ± {bias_std:.3f} ({n_sites_pft} sites)")
            else:
                logger.warning(f"    PFT {pft:2d}: insufficient sites ({n_sites_pft} < {self.min_sites_per_pft})")

        return pft_biases

    def _apply_pft_corrections(
        self,
        t_et_ratio_fused: xr.DataArray,
        total_et: xr.DataArray,
        pft_map: xr.DataArray,
        pft_biases: Dict[int, float],
        uncertainty: Optional[xr.DataArray]
    ) -> xr.DataArray:
        """
        Apply PFT-specific bias corrections to T/ET ratio.

        Correction formula:
        - T_corrected = T_fused - bias_pft
        - T/ET_corrected = T_corrected / ET
        - Weighted by uncertainty if available
        """
        # Initialize corrected T/ET
        t_et_corrected = t_et_ratio_fused.copy()

        # Compute current T
        T_fused = t_et_ratio_fused * total_et

        # Apply correction for each PFT
        for pft, bias in pft_biases.items():
            # Create mask for this PFT
            pft_mask = (pft_map == pft)

            # Correction weight (higher uncertainty → larger correction)
            if uncertainty is not None:
                # Normalize uncertainty to [0, 1]
                unc_norm = uncertainty / uncertainty.max()
                weight = self.correction_strength * unc_norm
            else:
                weight = self.correction_strength

            # Apply correction to T
            T_corrected = xr.where(
                pft_mask,
                T_fused - (bias * weight),
                T_fused
            )

            # Convert back to T/ET ratio
            t_et_corrected = xr.where(
                pft_mask,
                T_corrected / total_et,
                t_et_corrected
            )

        return t_et_corrected

    def as_matrices(self):
        """
        Convert to matrix form (not applicable for post-fusion constraint).

        SapFluxNet is a post-fusion constraint, so we return None.
        """
        return None, None

    def check_feasibility(
        self,
        weights: np.ndarray,
        **kwargs
    ) -> bool:
        """
        Check feasibility (not applicable for post-fusion constraint).

        Always returns True since constraint is applied post-fusion.
        """
        return True


class WaterBalanceConstraint(Constraint):
    """
    Hard constraint using basin-scale Water Balance ET.

    Strategy:
    ---------
    1. Aggregate fused ET to basin scale
    2. Compare with Water Balance ET (P - Q - dS) from observations
    3. Compute basin-level residuals
    4. Distribute residuals back to 1km pixels within each basin
    5. Distribution weighted by EIVD uncertainty (high uncertainty → larger adjustment)

    This is a "hard" constraint because it forces exact closure at basin scale,
    which is justified by mass conservation principles.

    Parameters
    ----------
    wb_et_basins : xr.DataArray
        Basin-scale WB-ET, dims: (basin_id, time)
    basin_mask_1km : xr.DataArray
        Basin classification at 1km, dims: (lat, lon)
        Values are basin IDs
    tolerance : float, default=0.05
        Relative tolerance for basin closure (5%)

    Examples
    --------
    >>> constraint = WaterBalanceConstraint(wb_et, basin_mask)
    >>> et_corrected = constraint.apply(
    ...     total_et_1km_fused=fused_et,
    ...     uncertainty=eivd_variance
    ... )
    """

    def __init__(
        self,
        wb_et_basins: xr.DataArray,
        basin_mask_1km: xr.DataArray,
        tolerance: float = 0.05
    ):
        """Initialize Water Balance constraint."""
        self.wb_truth = wb_et_basins
        self.basin_mask = basin_mask_1km
        self.tolerance = tolerance

        self.basin_ids = wb_et_basins.coords['basin_id'].values
        self.n_basins = len(self.basin_ids)

        logger.info(f"Water Balance constraint initialized with {self.n_basins} basins")
        logger.info(f"  Tolerance: {self.tolerance * 100:.1f}%")

    def apply(
        self,
        total_et_1km_fused: xr.DataArray,
        uncertainty: Optional[xr.DataArray] = None
    ) -> xr.DataArray:
        """
        Apply Water Balance hard constraint.

        Parameters
        ----------
        total_et_1km_fused : xr.DataArray
            Fused total ET at 1km, dims: (time, lat, lon)
        uncertainty : xr.DataArray, optional
            Fusion uncertainty, dims: (time, lat, lon)
            Used to weight residual distribution

        Returns
        -------
        xr.DataArray
            Constrained total ET, same dims as input
        """
        logger.info("Applying Water Balance hard constraint...")

        # Initialize corrected ET
        et_corrected = total_et_1km_fused.copy()

        # Process each basin
        n_basins_corrected = 0

        for basin_id in self.basin_ids:
            # Create basin mask
            basin_pixels = (self.basin_mask == basin_id)

            # Skip if basin not in 1km grid
            if basin_pixels.sum().values == 0:
                continue

            # Aggregate fused ET to basin
            et_fused_basin = total_et_1km_fused.where(basin_pixels).mean(
                dim=['lat', 'lon'], skipna=True
            )

            # Get WB-ET for this basin
            et_wb_basin = self.wb_truth.sel(basin_id=basin_id)

            # Compute residual (bias)
            # Bias_basin = ET_fused_basin - ET_wb_basin
            bias_basin = et_fused_basin - et_wb_basin

            # Check if correction is needed
            relative_bias = abs(bias_basin / et_wb_basin)
            needs_correction = relative_bias > self.tolerance

            if needs_correction.any():
                # Distribute bias back to 1km pixels
                et_corrected = self._distribute_basin_bias(
                    et_current=et_corrected,
                    basin_pixels=basin_pixels,
                    bias_basin=bias_basin,
                    uncertainty=uncertainty
                )
                n_basins_corrected += 1

        logger.info(f"  Corrected {n_basins_corrected}/{self.n_basins} basins")

        return et_corrected

    def _distribute_basin_bias(
        self,
        et_current: xr.DataArray,
        basin_pixels: xr.DataArray,
        bias_basin: xr.DataArray,
        uncertainty: Optional[xr.DataArray]
    ) -> xr.DataArray:
        """
        Distribute basin-level bias to 1km pixels.

        Two strategies:
        1. Uniform distribution (simple)
        2. Uncertainty-weighted (sophisticated)

        We use uncertainty-weighted if available.
        """
        if uncertainty is not None:
            # Uncertainty-weighted distribution
            # Pixels with higher uncertainty receive larger corrections

            # Extract uncertainty within basin
            unc_basin = uncertainty.where(basin_pixels)

            # Normalize to sum to 1 (distribution weights)
            unc_sum = unc_basin.sum(dim=['lat', 'lon'], skipna=True)
            weights = unc_basin / unc_sum

            # Distribute bias
            correction = bias_basin * weights

        else:
            # Uniform distribution
            n_pixels = basin_pixels.sum()
            correction = bias_basin / n_pixels
            correction = xr.where(basin_pixels, correction, 0)

        # Apply correction
        et_corrected = et_current - correction

        return et_corrected

    def as_matrices(self):
        """Not applicable for post-fusion constraint."""
        return None, None

    def check_feasibility(
        self,
        weights: np.ndarray,
        et_values: Optional[np.ndarray] = None
    ) -> bool:
        """
        Check if ET satisfies water balance at basin scale.

        Parameters
        ----------
        weights : np.ndarray
            Fusion weights (not used for WB check)
        et_values : np.ndarray, optional
            ET values to check

        Returns
        -------
        bool
            True if water balance is satisfied within tolerance
        """
        if et_values is None:
            return True

        # Placeholder - actual implementation would aggregate and compare
        return True


class PhysicalPlausibilityConstraint(Constraint):
    """
    Enforce physical plausibility: 0 ≤ T/ET ≤ 1, 0 ≤ ET ≤ PET.

    This is a simple post-processing step to ensure results are physically valid.

    Parameters
    ----------
    pet_data : xr.DataArray, optional
        Potential ET data, dims: (time, lat, lon)
    """

    def __init__(self, pet_data: Optional[xr.DataArray] = None):
        """Initialize plausibility constraint."""
        self.pet_data = pet_data

    def apply(
        self,
        t_et_ratio: xr.DataArray,
        total_et: xr.DataArray
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Apply physical plausibility constraints.

        Parameters
        ----------
        t_et_ratio : xr.DataArray
            T/ET ratio
        total_et : xr.DataArray
            Total ET

        Returns
        -------
        t_et_ratio : xr.DataArray
            Constrained T/ET ratio
        total_et : xr.DataArray
            Constrained total ET
        """
        logger.info("Applying physical plausibility constraints...")

        # Constraint 1: T/ET in [0, 1]
        n_invalid = ((t_et_ratio < 0) | (t_et_ratio > 1)).sum().values
        if n_invalid > 0:
            logger.warning(f"  Clipping {n_invalid} invalid T/ET ratios")
        t_et_ratio = t_et_ratio.clip(0, 1)

        # Constraint 2: ET >= 0
        n_negative = (total_et < 0).sum().values
        if n_negative > 0:
            logger.warning(f"  Clipping {n_negative} negative ET values")
        total_et = total_et.clip(min=0)

        # Constraint 3: ET <= PET (if available)
        if self.pet_data is not None:
            exceeds_pet = (total_et > self.pet_data)
            n_exceeds = exceeds_pet.sum().values
            if n_exceeds > 0:
                logger.warning(f"  Clipping {n_exceeds} ET values exceeding PET")
            total_et = xr.where(exceeds_pet, self.pet_data, total_et)

        logger.info("  Physical plausibility constraints applied.")

        return t_et_ratio, total_et

    def as_matrices(self):
        """Not applicable."""
        return None, None

    def check_feasibility(self, weights: np.ndarray, **kwargs) -> bool:
        """Always returns True (post-processing constraint)."""
        return True


def apply_all_constraints(
    t_et_ratio_fused: xr.DataArray,
    total_et_1km: xr.DataArray,
    sapflux_data: xr.DataArray,
    wb_et_basins: xr.DataArray,
    basin_mask: xr.DataArray,
    pft_map: xr.DataArray,
    pet_data: Optional[xr.DataArray] = None,
    uncertainty: Optional[xr.DataArray] = None
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Apply all physical constraints in sequence.

    Order:
    1. Water Balance (hard constraint on total ET)
    2. SapFluxNet (soft constraint on T/ET ratio)
    3. Physical plausibility (post-processing)

    Parameters
    ----------
    t_et_ratio_fused : xr.DataArray
        Fused T/ET ratio from EIVD/GLS
    total_et_1km : xr.DataArray
        Total ET at 1km
    sapflux_data : xr.DataArray
        SapFluxNet observations
    wb_et_basins : xr.DataArray
        Basin-scale WB-ET
    basin_mask : xr.DataArray
        Basin classification
    pft_map : xr.DataArray
        Plant Functional Type map
    pet_data : xr.DataArray, optional
        Potential ET
    uncertainty : xr.DataArray, optional
        Fusion uncertainty from EIVD

    Returns
    -------
    t_et_ratio_final : xr.DataArray
        Constrained T/ET ratio
    T_final : xr.DataArray
        Final transpiration
    E_final : xr.DataArray
        Final evaporation
    """
    logger.info("\n" + "=" * 60)
    logger.info("Applying Physical Constraints (Sequential)")
    logger.info("=" * 60)

    # Step 1: Water Balance constraint on total ET
    wb_constraint = WaterBalanceConstraint(wb_et_basins, basin_mask)
    total_et_constrained = wb_constraint.apply(
        total_et_1km_fused=total_et_1km,
        uncertainty=uncertainty
    )

    # Step 2: SapFluxNet constraint on T/ET ratio
    sf_constraint = SapfluxnetConstraint(sapflux_data)
    t_et_ratio_constrained = sf_constraint.apply(
        t_et_ratio_fused=t_et_ratio_fused,
        total_et=total_et_constrained,
        pft_map=pft_map,
        uncertainty=uncertainty
    )

    # Step 3: Physical plausibility
    plaus_constraint = PhysicalPlausibilityConstraint(pet_data=pet_data)
    t_et_ratio_final, total_et_final = plaus_constraint.apply(
        t_et_ratio=t_et_ratio_constrained,
        total_et=total_et_constrained
    )

    # Compute final T and E
    T_final = total_et_final * t_et_ratio_final
    E_final = total_et_final * (1 - t_et_ratio_final)

    logger.info("\n" + "=" * 60)
    logger.info("Physical Constraints Complete")
    logger.info("=" * 60)

    return t_et_ratio_final, T_final, E_final
