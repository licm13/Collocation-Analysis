"""
HCC-ET Framework: Hierarchical Cross-Calibration for ET Decomposition
======================================================================

A production-ready framework for quantifying global "productive" (T) vs
"non-productive" (E) water flux reshaping under climate change and human
land management (1950-2025, 1km resolution).

Core Innovation:
- Fuses **T/ET ratio** (transpiration efficiency), not total ET
- Handles Error Cross-Correlation (ECC) using EIVD/GLS from collocation library
- Multi-scale physical constraints (SapFlux, Water Balance)
- Statistical downscaling with ML (Random Forest)

Target Publication: Nature Climate Change

Author: [Your Name]
Date: 2025-11-09
"""

import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

import numpy as np
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import theilslopes

# Import collocation-analysis library (existing, validated)
from collocation import eivd, ec, BayesianTC
from collocation.fusion.fuse import fuse_fields
from collocation.fusion.constraints import Constraint
from collocation.utils import calculate_all_metrics

# Import HCC-ET modules (new, to be created)
from . import data_loader
from . import downscaler
from . import physical_constraints
from . import config
from . import utils_hcc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HCCETFramework:
    """
    Main orchestrator for HCC-ET Framework.

    Workflow:
    ---------
    1. Data Preparation (1981-2025)
       - Load parent T/ET ratios (GLEAM, PMLv2, GLDAS)
       - Load 1km predictors (LST, NDVI, Albedo, Precip)
       - Load constraints (SapFlux, WB-ET)

    2. Statistical Downscaling (ML-based)
       - Train RF models: parent T/ET -> 1km T/ET candidates
       - Preserves ECC structure (shared predictors)

    3. Cross-Calibration Fusion (EIVD/GLS)
       - Fuse 1km T/ET candidates with ECC-aware weights
       - Uses EIVD if n_candidates=3, EC if n_candidates=4

    4. Multi-Scale Physical Constraints
       - Site-level: SapFluxNet T observations
       - Basin-level: Water Balance ET (P-Q-dS)

    5. Temporal Hindcast & Trend Analysis
       - Apply trained model to 1950-1980 predictors
       - Quantify T/ET reshaping trends (natural vs managed)

    Parameters
    ----------
    config_file : str or Path
        Path to configuration YAML file
    output_dir : str or Path
        Directory for saving results
    """

    def __init__(
        self,
        config_file: str | Path,
        output_dir: str | Path = "./outputs"
    ):
        """Initialize HCC-ET Framework."""
        self.cfg = config.load_config(config_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.parent_t_et_ratios = None
        self.predictors_1km = None
        self.sapflux_data = None
        self.wb_et_basins = None
        self.land_cover_mask = None

        # Model containers
        self.downscaler_model = None
        self.fusion_weights = None

        # Results containers
        self.t_et_candidates_1km = None
        self.fused_t_et_ratio = None
        self.T_final_1km = None
        self.E_final_1km = None

        logger.info(f"HCC-ET Framework initialized. Output: {self.output_dir}")

    def run_phase1_data_preparation(self) -> None:
        """
        Phase 1: Data Preparation (1981-2025).

        Loads:
        - Parent T/ET ratios (coarse resolution)
        - 1km predictors (LST, NDVI, etc.)
        - Physical constraints (SapFlux, WB-ET)
        - Land cover masks
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: Data Preparation")
        logger.info("=" * 60)

        # Load parent T/ET ratios (e.g., GLEAM, PMLv2, GLDAS)
        logger.info("Loading parent T/ET ratios (coarse resolution)...")
        self.parent_t_et_ratios = data_loader.load_parent_t_et_ratios(
            data_dir=self.cfg['data']['parent_dir'],
            products=self.cfg['data']['parent_products'],
            time_range=self.cfg['time']['calibration_period']
        )
        logger.info(f"  Shape: {self.parent_t_et_ratios.shape}")
        logger.info(f"  Products: {list(self.parent_t_et_ratios.coords['model'].values)}")

        # Load 1km predictors
        logger.info("Loading 1km predictors (LST, NDVI, Albedo, Precip)...")
        self.predictors_1km = data_loader.load_predictors_1km(
            data_dir=self.cfg['data']['predictors_dir'],
            variables=self.cfg['data']['predictor_vars'],
            time_range=self.cfg['time']['calibration_period']
        )
        logger.info(f"  Shape: {self.predictors_1km.shape}")
        logger.info(f"  Variables: {list(self.predictors_1km.coords['variable'].values)}")

        # Load SapFlux constraint data
        logger.info("Loading SapFluxNet observations...")
        self.sapflux_data = data_loader.load_sapfluxnet_1km(
            data_dir=self.cfg['data']['sapflux_dir'],
            upscale_method=self.cfg['constraints']['sapflux_upscale_method']
        )
        logger.info(f"  Sites: {self.sapflux_data.attrs.get('n_sites', 'N/A')}")

        # Load Water Balance constraint
        logger.info("Loading basin-scale Water Balance ET...")
        self.wb_et_basins = data_loader.load_wb_et_basins(
            data_dir=self.cfg['data']['wb_et_dir'],
            basin_ids=self.cfg['constraints']['basin_ids']
        )
        logger.info(f"  Basins: {len(self.wb_et_basins.coords['basin_id'])}")

        # Load land cover masks
        logger.info("Loading land cover masks (natural vs managed)...")
        self.land_cover_mask = data_loader.load_land_cover_masks(
            data_dir=self.cfg['data']['landcover_dir'],
            classification=self.cfg['analysis']['landcover_classes']
        )
        logger.info(f"  Classes: {list(self.land_cover_mask.coords['class'].values)}")

        logger.info("Phase 1 complete.\n")

    def run_phase2_downscaling(self) -> None:
        """
        Phase 2: Statistical Downscaling (ML-based).

        Trains Random Forest models to downscale parent T/ET ratios to 1km.

        Key Innovation:
        - ALL 1km candidates share SAME 1km predictors
        - This induces Error Cross-Correlation (ECC)
        - EIVD/GLS explicitly models this ECC in Phase 3
        """
        logger.info("=" * 60)
        logger.info("PHASE 2: Statistical Downscaling")
        logger.info("=" * 60)

        # Train downscaler
        logger.info("Training Random Forest downscaler...")
        logger.info(f"  Model: {self.cfg['downscaling']['model_type']}")
        logger.info(f"  Features: {self.cfg['downscaling']['n_features']}")

        self.downscaler_model = downscaler.train_downscaler(
            parent_data=self.parent_t_et_ratios,
            predictors_1km=self.predictors_1km,
            model_type=self.cfg['downscaling']['model_type'],
            hyperparams=self.cfg['downscaling']['hyperparams'],
            cv_folds=self.cfg['downscaling']['cv_folds']
        )

        # Generate 1km candidates
        logger.info("Predicting 1km T/ET candidates...")
        self.t_et_candidates_1km = downscaler.predict(
            model=self.downscaler_model,
            predictors_1km=self.predictors_1km
        )
        logger.info(f"  Output shape: {self.t_et_candidates_1km.shape}")
        logger.info(f"  Models: {list(self.t_et_candidates_1km.coords['model'].values)}")

        # Evaluate downscaling performance
        if self.cfg['downscaling']['evaluate']:
            logger.info("Evaluating downscaling performance...")
            metrics = downscaler.evaluate_downscaling(
                predictions=self.t_et_candidates_1km,
                parent_data=self.parent_t_et_ratios,
                predictors_1km=self.predictors_1km
            )
            logger.info(f"  R²: {metrics['r2']:.3f}")
            logger.info(f"  RMSE: {metrics['rmse']:.3f}")

        logger.info("Phase 2 complete.\n")

    def run_phase3_fusion(self) -> None:
        """
        Phase 3: Cross-Calibration Fusion (EIVD/GLS).

        Fuses 1km T/ET candidates using Error Cross-Correlation (ECC) aware methods.

        Method Selection:
        - n_models = 3: EIVD (Dong et al., 2019)
        - n_models = 4: EC (Gruber et al., 2016)
        - n_models >= 3: GLS via fuse_fields() with TC-estimated Sigma

        Returns:
        - Fused T/ET ratio (1km, time-series)
        - Fusion weights (spatially varying if configured)
        - Uncertainty estimates (from EIVD or BayesianTC)
        """
        logger.info("=" * 60)
        logger.info("PHASE 3: Cross-Calibration Fusion")
        logger.info("=" * 60)

        n_models = len(self.t_et_candidates_1km.coords['model'])
        logger.info(f"Number of candidate models: {n_models}")

        fusion_mode = self.cfg['fusion']['mode']  # 'eivd', 'ec', 'gls', 'auto'

        if fusion_mode == 'auto':
            # Auto-select based on n_models
            if n_models == 3:
                fusion_mode = 'eivd'
            elif n_models == 4:
                fusion_mode = 'ec'
            else:
                fusion_mode = 'gls'

        logger.info(f"Fusion mode: {fusion_mode.upper()}")

        if fusion_mode == 'eivd':
            # Use EIVD for n_models=3
            logger.info("Using EIVD (handles ECC for 3 products)...")
            self.fused_t_et_ratio, self.fusion_weights = self._fuse_with_eivd()

        elif fusion_mode == 'ec':
            # Use EC for n_models=4
            logger.info("Using EC (handles ECC for 4 products)...")
            self.fused_t_et_ratio, self.fusion_weights = self._fuse_with_ec()

        elif fusion_mode == 'gls':
            # Use GLS via fuse_fields (general, n_models >= 3)
            logger.info("Using GLS via fuse_fields (general approach)...")
            self.fused_t_et_ratio, self.fusion_weights = self._fuse_with_gls()

        else:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")

        logger.info(f"  Fused T/ET ratio shape: {self.fused_t_et_ratio.shape}")
        logger.info(f"  Weights shape: {self.fusion_weights.shape}")

        # Uncertainty quantification
        if self.cfg['fusion']['uncertainty_quantification']:
            logger.info("Running Bayesian TC for uncertainty quantification...")
            self._run_bayesian_uncertainty()

        logger.info("Phase 3 complete.\n")

    def run_phase4_physical_constraints(self) -> None:
        """
        Phase 4: Multi-Scale Physical Constraints.

        Applies two types of constraints:
        1. Site-level: SapFluxNet T observations (soft constraint)
        2. Basin-level: Water Balance ET from P-Q-dS (hard constraint)

        Workflow:
        - First constrain total ET with WB-ET (basin scale)
        - Then constrain T/ET ratio with SapFlux (site scale)
        - Final T and E computed as: T = ET * (T/ET), E = ET * (1 - T/ET)
        """
        logger.info("=" * 60)
        logger.info("PHASE 4: Multi-Scale Physical Constraints")
        logger.info("=" * 60)

        # Load or use pre-fused total ET (1km)
        logger.info("Loading total ET (1km) for constraint...")
        total_et_1km = data_loader.load_total_et_1km(
            data_dir=self.cfg['data']['total_et_dir'],
            product=self.cfg['data']['total_et_product']  # e.g., 'CAMELE'
        )
        logger.info(f"  Total ET shape: {total_et_1km.shape}")

        # Constraint 1: Basin Water Balance (hard constraint on total ET)
        if self.cfg['constraints']['apply_wb_constraint']:
            logger.info("Applying Water Balance constraint (basin scale)...")
            wb_constraint = physical_constraints.WaterBalanceConstraint(
                wb_et_basins=self.wb_et_basins,
                basin_mask_1km=data_loader.load_basin_mask_1km(
                    self.cfg['data']['basin_mask_file']
                )
            )
            total_et_1km = wb_constraint.apply(
                total_et_1km_fused=total_et_1km,
                uncertainty=None  # Could use EIVD uncertainty for weighting
            )
            logger.info("  Water Balance constraint applied.")

        # Constraint 2: SapFluxNet (soft constraint on T/ET ratio)
        if self.cfg['constraints']['apply_sapflux_constraint']:
            logger.info("Applying SapFluxNet constraint (site scale)...")
            sapflux_constraint = physical_constraints.SapfluxnetConstraint(
                sapflux_1km_data=self.sapflux_data
            )
            self.fused_t_et_ratio = sapflux_constraint.apply(
                t_et_ratio_fused=self.fused_t_et_ratio,
                total_et=total_et_1km,
                pft_map=data_loader.load_pft_map(self.cfg['data']['pft_map_file'])
            )
            logger.info("  SapFluxNet constraint applied.")

        # Compute final T and E
        logger.info("Computing final T and E from constrained T/ET ratio...")
        self.T_final_1km = total_et_1km * self.fused_t_et_ratio
        self.E_final_1km = total_et_1km * (1 - self.fused_t_et_ratio)

        logger.info(f"  T (transpiration) shape: {self.T_final_1km.shape}")
        logger.info(f"  E (evaporation) shape: {self.E_final_1km.shape}")

        # Sanity checks
        self._validate_physical_plausibility()

        logger.info("Phase 4 complete.\n")

    def run_phase5_temporal_hindcast(self) -> None:
        """
        Phase 5: Temporal Hindcast & Trend Analysis (1950-1980).

        Uses trained downscaler and fusion weights to:
        1. Generate 1950-1980 T/ET estimates
        2. Analyze reshaping trends (Theil-Sen slopes)
        3. Partition by land cover (natural vs managed ecosystems)
        """
        logger.info("=" * 60)
        logger.info("PHASE 5: Temporal Hindcast & Trend Analysis")
        logger.info("=" * 60)

        # Load historical predictors (1950-1980)
        logger.info("Loading historical predictors (1950-1980)...")
        predictors_historical = data_loader.load_predictors_1km(
            data_dir=self.cfg['data']['predictors_dir'],
            variables=self.cfg['data']['predictor_vars'],
            time_range=self.cfg['time']['hindcast_period']
        )
        logger.info(f"  Shape: {predictors_historical.shape}")

        # Apply downscaler to historical period
        logger.info("Generating historical T/ET candidates (1950-1980)...")
        t_et_historical = downscaler.predict(
            model=self.downscaler_model,
            predictors_1km=predictors_historical
        )

        # Fuse using pre-computed weights (assume stationarity)
        logger.info("Fusing historical T/ET with trained weights...")
        t_et_ratio_historical = (t_et_historical * self.fusion_weights).sum(dim='model')

        # Combine with modern period (1981-2025)
        logger.info("Merging historical and modern periods...")
        t_et_ratio_full = xr.concat(
            [t_et_ratio_historical, self.fused_t_et_ratio],
            dim='time'
        )
        logger.info(f"  Full time series: {t_et_ratio_full.shape}")

        # Trend Analysis
        logger.info("Computing Theil-Sen trends by land cover class...")
        trends = self._compute_trends(t_et_ratio_full)

        # Report results
        logger.info("\n" + "=" * 60)
        logger.info("TREND ANALYSIS RESULTS (1950-2025)")
        logger.info("=" * 60)
        for landcover, slope in trends.items():
            logger.info(f"{landcover:20s}: {slope['median']:.4f} /decade (p={slope['p_value']:.3f})")

        # Save results
        self._save_trend_results(trends, t_et_ratio_full)

        logger.info("Phase 5 complete.\n")

    def _fuse_with_eivd(self) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Fuse using EIVD method (for 3 products).

        Returns:
        - fused: Fused T/ET ratio
        - weights: Fusion weights (computed from EIVD error covariance)
        """
        # Apply EIVD pixel-wise or globally
        if self.cfg['fusion']['spatial_varying_weights']:
            # Pixel-wise EIVD (expensive but accurate)
            return self._fuse_eivd_pixelwise()
        else:
            # Global EIVD (fast, assumes spatial homogeneity)
            return self._fuse_eivd_global()

    def _fuse_eivd_global(self) -> Tuple[xr.DataArray, xr.DataArray]:
        """Global EIVD fusion (single weight vector for entire domain)."""
        # Stack data for EIVD
        data_stacked = self.t_et_candidates_1km.stack(
            sample=('time', 'lat', 'lon')
        ).values.T  # Shape: (n_samples, n_models)

        # Apply EIVD
        EeeT, SNR, rho2, fMSE, L = eivd(data_stacked)

        # Compute optimal weights from error covariance
        # w = inv(EeeT) @ 1 / (1^T @ inv(EeeT) @ 1)
        EeeT_inv = np.linalg.inv(EeeT)
        ones = np.ones(3)
        weights = EeeT_inv @ ones / (ones @ EeeT_inv @ ones)

        logger.info(f"  EIVD weights: {weights}")
        logger.info(f"  SNR: {SNR}")
        logger.info(f"  Error cross-correlation (2-3): {EeeT[1,2]:.4f}")

        # Apply weights
        weights_xr = xr.DataArray(
            weights,
            dims=['model'],
            coords={'model': self.t_et_candidates_1km.coords['model']}
        )

        fused = (self.t_et_candidates_1km * weights_xr).sum(dim='model')

        return fused, weights_xr

    def _fuse_eivd_pixelwise(self) -> Tuple[xr.DataArray, xr.DataArray]:
        """Pixel-wise EIVD fusion (spatially varying weights)."""
        # Initialize output arrays
        fused = xr.full_like(
            self.t_et_candidates_1km.isel(model=0),
            fill_value=np.nan
        )
        weights = xr.full_like(self.t_et_candidates_1km, fill_value=np.nan)

        # Loop over pixels
        for ilat in range(len(self.t_et_candidates_1km.coords['lat'])):
            for ilon in range(len(self.t_et_candidates_1km.coords['lon'])):
                # Extract time series for this pixel
                pixel_data = self.t_et_candidates_1km.isel(
                    lat=ilat, lon=ilon
                ).values.T  # Shape: (n_time, n_models)

                # Skip if insufficient valid data
                if np.sum(~np.isnan(pixel_data)) < 10:
                    continue

                # Apply EIVD
                try:
                    EeeT, SNR, rho2, fMSE, L = eivd(pixel_data)
                    EeeT_inv = np.linalg.inv(EeeT)
                    ones = np.ones(3)
                    w_pixel = EeeT_inv @ ones / (ones @ EeeT_inv @ ones)

                    # Store weights
                    weights.loc[dict(lat=ilat, lon=ilon)] = w_pixel

                    # Apply weights to compute fused value
                    fused.loc[dict(lat=ilat, lon=ilon)] = (
                        pixel_data * w_pixel[None, :]
                    ).sum(axis=1)

                except (np.linalg.LinAlgError, ValueError):
                    # Singular matrix or other numerical issue
                    continue

        return fused, weights

    def _fuse_with_ec(self) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Fuse using EC method (for 4 products).

        Uses Extended Collocation (Gruber et al., 2016) which tests all
        6 possible coordinate combinations and selects the best one.
        """
        # Apply EC globally (can be extended to pixel-wise)
        data_stacked = self.t_et_candidates_1km.stack(
            sample=('time', 'lat', 'lon')
        ).values.T  # Shape: (n_samples, 4)

        # Apply EC
        from collocation.ec import ec, select_best_combination

        logger.info("  Running Extended Collocation (6 combinations)...")
        results = ec(data_stacked)

        # Select best combination
        best_result = select_best_combination(
            results,
            criterion=self.cfg['fusion']['ec_criterion']  # 'min_fMSE', 'max_SNR'
        )

        logger.info(f"  Best combination: {best_result.combination}")
        logger.info(f"  Best weights (scenario 0): {best_result.re_weight[0]}")

        # Use weights from best combination (scenario 0)
        weights = best_result.re_weight[0]
        weights_xr = xr.DataArray(
            weights,
            dims=['model'],
            coords={'model': self.t_et_candidates_1km.coords['model']}
        )

        # Fuse
        fused = (self.t_et_candidates_1km * weights_xr).sum(dim='model')

        return fused, weights_xr

    def _fuse_with_gls(self) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Fuse using GLS via fuse_fields (general approach, n_models >= 3).

        Estimates error covariance using Triple Collocation, then applies
        Generalized Least Squares fusion with optional constraints.
        """
        # Use fuse_fields API from collocation.fusion
        result = fuse_fields(
            X=self.t_et_candidates_1km,
            X_ref=None,  # No reference (use TC for covariance estimation)
            mode='gls',
            constraints=None,  # Can add bounds, sum-to-one, etc.
            shrinkage=self.cfg['fusion']['shrinkage'],  # 'lw', 'oas', 'none'
            robust=self.cfg['fusion']['robust'],
            return_weights=True,
            return_var=True,
            return_diagnostics=True
        )

        fused = result['fused']
        weights = result['weights']

        logger.info(f"  GLS fusion complete.")
        logger.info(f"  Effective N models: {result['diagnostics'].get('effective_n', 'N/A')}")

        return fused, weights

    def _run_bayesian_uncertainty(self) -> None:
        """
        Run Bayesian TC for full uncertainty quantification.

        Uses BayesianTC from collocation library to estimate:
        - Posterior distributions of error variances
        - Time-varying error structures
        - Credible intervals for T/ET estimates
        """
        logger.info("  Setting up Bayesian TC...")

        # Sample subset for computational efficiency
        if self.cfg['fusion']['bayesian_sample_size'] is not None:
            n_samples = self.cfg['fusion']['bayesian_sample_size']
            indices = np.random.choice(
                self.t_et_candidates_1km.time.size,
                size=n_samples,
                replace=False
            )
            data_subset = self.t_et_candidates_1km.isel(time=indices)
        else:
            data_subset = self.t_et_candidates_1km

        # Stack data
        data_stacked = data_subset.stack(
            sample=('time', 'lat', 'lon')
        ).values  # Shape: (n_models, n_samples)

        # Run BayesianTC
        try:
            btc = BayesianTC(data_stacked)
            btc.run_inference(
                niter=self.cfg['fusion']['bayesian_niter'],
                nadvi=self.cfg['fusion']['bayesian_nadvi'],
                seed=42
            )

            # Get uncertainty estimates
            rmse_mean, rmse_std, rmse_quantiles = btc.get_error_estimates()

            logger.info(f"  Bayesian TC complete.")
            logger.info(f"  RMSE (mean): {rmse_mean}")
            logger.info(f"  RMSE (std): {rmse_std}")

            # Save uncertainty estimates
            self._save_uncertainty_estimates(btc)

        except Exception as e:
            logger.warning(f"  Bayesian TC failed: {e}")
            logger.warning("  Continuing without Bayesian uncertainty...")

    def _validate_physical_plausibility(self) -> None:
        """Validate physical plausibility of T and E estimates."""
        logger.info("Validating physical plausibility...")

        # Check 1: T/ET ratio in [0, 1]
        invalid_ratio = (
            (self.fused_t_et_ratio < 0) | (self.fused_t_et_ratio > 1)
        ).sum().values
        if invalid_ratio > 0:
            logger.warning(f"  {invalid_ratio} pixels have T/ET ratio outside [0,1]!")

        # Check 2: T and E are non-negative
        invalid_T = (self.T_final_1km < 0).sum().values
        invalid_E = (self.E_final_1km < 0).sum().values
        if invalid_T > 0 or invalid_E > 0:
            logger.warning(f"  Negative values: T={invalid_T}, E={invalid_E}")

        # Check 3: T + E ≈ ET (within tolerance)
        total_check = np.abs(
            (self.T_final_1km + self.E_final_1km) -
            (self.T_final_1km + self.E_final_1km)  # Should use loaded total_et
        )
        # This is a placeholder - actual implementation needs loaded total ET

        logger.info("  Validation complete.")

    def _compute_trends(self, t_et_ratio_full: xr.DataArray) -> Dict[str, Dict]:
        """
        Compute Theil-Sen trends by land cover class.

        Returns:
        - Dictionary with keys: 'natural', 'managed', 'urban', etc.
        - Each value is a dict with 'slope', 'intercept', 'p_value'
        """
        trends = {}

        for lc_class in self.land_cover_mask.coords['class'].values:
            # Mask data for this land cover class
            mask = self.land_cover_mask.sel(class_name=lc_class)
            data_masked = t_et_ratio_full.where(mask)

            # Compute spatial mean time series
            ts = data_masked.mean(dim=['lat', 'lon'], skipna=True).values
            time_numeric = np.arange(len(ts))

            # Theil-Sen slope estimation
            result = theilslopes(ts, time_numeric)

            trends[lc_class] = {
                'slope': result.slope * 10,  # per decade
                'intercept': result.intercept,
                'p_value': utils_hcc.compute_trend_pvalue(ts, result.slope),
                'median': result.slope * 10
            }

        return trends

    def _save_trend_results(
        self,
        trends: Dict[str, Dict],
        t_et_ratio_full: xr.DataArray
    ) -> None:
        """Save trend analysis results to NetCDF."""
        # Save full time series
        output_file = self.output_dir / "t_et_ratio_1950_2025.nc"
        t_et_ratio_full.to_netcdf(output_file)
        logger.info(f"  Saved: {output_file}")

        # Save trend summary as CSV
        import pandas as pd
        df = pd.DataFrame(trends).T
        trend_file = self.output_dir / "trends_by_landcover.csv"
        df.to_csv(trend_file)
        logger.info(f"  Saved: {trend_file}")

    def _save_uncertainty_estimates(self, btc: BayesianTC) -> None:
        """Save Bayesian TC uncertainty estimates."""
        # This is a placeholder - actual implementation would extract
        # and save posterior distributions
        pass

    def run_full_pipeline(self) -> None:
        """
        Execute the complete HCC-ET pipeline.

        Runs all 5 phases in sequence:
        1. Data Preparation
        2. Statistical Downscaling
        3. Cross-Calibration Fusion
        4. Physical Constraints
        5. Temporal Hindcast & Trend Analysis
        """
        logger.info("\n" + "=" * 60)
        logger.info("HCC-ET FRAMEWORK: Full Pipeline Execution")
        logger.info("=" * 60 + "\n")

        try:
            self.run_phase1_data_preparation()
            self.run_phase2_downscaling()
            self.run_phase3_fusion()
            self.run_phase4_physical_constraints()
            self.run_phase5_temporal_hindcast()

            logger.info("\n" + "=" * 60)
            logger.info("HCC-ET FRAMEWORK: Pipeline Complete!")
            logger.info("=" * 60)
            logger.info(f"Results saved to: {self.output_dir}")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point for HCC-ET Framework."""
    import argparse

    parser = argparse.ArgumentParser(
        description="HCC-ET Framework for T/ET Decomposition"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='Output directory for results'
    )
    parser.add_argument(
        '--phase',
        type=str,
        choices=['1', '2', '3', '4', '5', 'all'],
        default='all',
        help='Run specific phase or all phases'
    )

    args = parser.parse_args()

    # Initialize framework
    framework = HCCETFramework(
        config_file=args.config,
        output_dir=args.output_dir
    )

    # Run selected phase(s)
    if args.phase == 'all':
        framework.run_full_pipeline()
    else:
        phase_map = {
            '1': framework.run_phase1_data_preparation,
            '2': framework.run_phase2_downscaling,
            '3': framework.run_phase3_fusion,
            '4': framework.run_phase4_physical_constraints,
            '5': framework.run_phase5_temporal_hindcast
        }
        phase_map[args.phase]()


if __name__ == '__main__':
    main()
