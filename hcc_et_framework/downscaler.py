"""
Statistical Downscaling Module for HCC-ET Framework
====================================================

Handles machine learning-based downscaling of coarse-resolution parent
T/ET ratios to 1km resolution using high-resolution predictors.

Key Features:
- Multiple ML models: Random Forest, Gradient Boosting, Neural Networks
- Cross-validation for robust performance estimation
- Handles spatiotemporal variability
- Preserves Error Cross-Correlation (ECC) structure via shared predictors

IMPORTANT:
All 1km candidate products share the SAME 1km predictors (LST, NDVI, etc.).
This induces Error Cross-Correlation (ECC), which is handled in Phase 3
using EIVD/GLS fusion.

Author: [Your Name]
Date: 2025-11-09
"""

import logging
from typing import Dict, Optional, Tuple, Any

import numpy as np
import xarray as xr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

logger = logging.getLogger(__name__)


class DownscalerModel:
    """
    Container for trained downscaling models.

    Stores one model per parent product (e.g., GLEAM, PMLv2, GLDAS).
    """

    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize downscaler.

        Parameters
        ----------
        model_type : str
            Type of ML model: 'random_forest', 'gradient_boosting', 'ridge'
        """
        self.model_type = model_type
        self.models = {}  # {product_name: trained_model}
        self.feature_names = None
        self.cv_scores = {}

    def add_model(self, product_name: str, model: Any) -> None:
        """Add a trained model for a specific product."""
        self.models[product_name] = model

    def get_model(self, product_name: str) -> Any:
        """Get trained model for a specific product."""
        return self.models.get(product_name)

    def save(self, filepath: str) -> None:
        """Save trained models to disk."""
        joblib.dump(self, filepath)
        logger.info(f"Saved downscaler models to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'DownscalerModel':
        """Load trained models from disk."""
        model = joblib.load(filepath)
        logger.info(f"Loaded downscaler models from {filepath}")
        return model


def train_downscaler(
    parent_data: xr.DataArray,
    predictors_1km: xr.DataArray,
    model_type: str = 'random_forest',
    hyperparams: Optional[Dict[str, Any]] = None,
    cv_folds: int = 5,
    n_samples: Optional[int] = None
) -> DownscalerModel:
    """
    Train downscaling models for each parent product.

    Strategy:
    ---------
    1. Co-locate parent T/ET with 1km predictors (spatial aggregation)
    2. Train separate ML model for each parent product
    3. Cross-validate to estimate generalization performance
    4. Return trained models ready for 1km prediction

    Parameters
    ----------
    parent_data : xr.DataArray
        Coarse-resolution T/ET ratios, dims: (time, lat, lon, model)
    predictors_1km : xr.DataArray
        High-resolution predictors, dims: (time, lat, lon, variable)
    model_type : str
        ML model type: 'random_forest', 'gradient_boosting', 'ridge'
    hyperparams : dict, optional
        Model hyperparameters
    cv_folds : int
        Number of cross-validation folds
    n_samples : int, optional
        Max samples for training (for computational efficiency)

    Returns
    -------
    DownscalerModel
        Container with trained models for each parent product

    Examples
    --------
    >>> model = train_downscaler(
    ...     parent_data=parent_t_et,  # (time, lat, lon, 3)
    ...     predictors_1km=predictors,  # (time, lat, lon, 4)
    ...     model_type='random_forest',
    ...     cv_folds=5
    ... )
    """
    logger.info("Training downscaling models...")
    logger.info(f"  Model type: {model_type}")
    logger.info(f"  Parent products: {list(parent_data.coords['model'].values)}")
    logger.info(f"  Predictors: {list(predictors_1km.coords['variable'].values)}")

    # Initialize downscaler container
    downscaler = DownscalerModel(model_type=model_type)
    downscaler.feature_names = list(predictors_1km.coords['variable'].values)

    # Default hyperparameters
    if hyperparams is None:
        hyperparams = _get_default_hyperparams(model_type)

    # Prepare training data
    logger.info("Preparing training data (co-locating parent with predictors)...")
    X_train, y_train_dict = _prepare_training_data(
        parent_data=parent_data,
        predictors_1km=predictors_1km,
        n_samples=n_samples
    )

    logger.info(f"  Training samples: {X_train.shape[0]}")
    logger.info(f"  Features: {X_train.shape[1]}")

    # Train separate model for each parent product
    for product_name, y_train in y_train_dict.items():
        logger.info(f"\nTraining model for {product_name}...")

        # Initialize model
        model = _initialize_model(model_type, hyperparams)

        # Cross-validation
        if cv_folds > 1:
            logger.info(f"  Running {cv_folds}-fold cross-validation...")
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_folds,
                scoring='r2',
                n_jobs=-1
            )
            logger.info(f"  CV R² scores: {cv_scores}")
            logger.info(f"  Mean CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            downscaler.cv_scores[product_name] = cv_scores

        # Train final model on all data
        logger.info("  Training final model...")
        model.fit(X_train, y_train)

        # Evaluate on training data (sanity check)
        y_pred = model.predict(X_train)
        r2 = r2_score(y_train, y_pred)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        mae = mean_absolute_error(y_train, y_pred)

        logger.info(f"  Training R²: {r2:.3f}")
        logger.info(f"  Training RMSE: {rmse:.3f}")
        logger.info(f"  Training MAE: {mae:.3f}")

        # Feature importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            logger.info("  Feature importances:")
            for feat, imp in zip(downscaler.feature_names, importances):
                logger.info(f"    {feat:15s}: {imp:.3f}")

        # Store model
        downscaler.add_model(product_name, model)

    logger.info("\nDownscaling models trained successfully.")

    return downscaler


def predict(
    model: DownscalerModel,
    predictors_1km: xr.DataArray,
    chunk_size: Optional[int] = 1000
) -> xr.DataArray:
    """
    Generate 1km T/ET candidates using trained downscaler.

    Parameters
    ----------
    model : DownscalerModel
        Trained downscaling models
    predictors_1km : xr.DataArray
        High-resolution predictors, dims: (time, lat, lon, variable)
    chunk_size : int, optional
        Process data in chunks to manage memory

    Returns
    -------
    xr.DataArray
        1km T/ET candidates, dims: (time, lat, lon, model)

    Notes
    -----
    This function applies each trained model to the SAME predictors,
    which induces Error Cross-Correlation (ECC) in the outputs.
    """
    logger.info("Generating 1km T/ET candidates...")

    # Prepare predictor matrix
    X_pred = _prepare_prediction_data(predictors_1km)

    # Initialize output container
    n_samples = X_pred.shape[0]
    n_models = len(model.models)
    predictions = np.zeros((n_samples, n_models))

    # Predict for each model
    for i, (product_name, trained_model) in enumerate(model.models.items()):
        logger.info(f"  Predicting {product_name}...")

        # Predict in chunks to manage memory
        if chunk_size is not None and n_samples > chunk_size:
            y_pred = []
            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                chunk_pred = trained_model.predict(X_pred[start:end])
                y_pred.append(chunk_pred)
            predictions[:, i] = np.concatenate(y_pred)
        else:
            predictions[:, i] = trained_model.predict(X_pred)

        # Clip to physical range [0, 1]
        predictions[:, i] = np.clip(predictions[:, i], 0, 1)

    # Reshape to original spatial dimensions
    original_shape = predictors_1km.shape[:3]  # (time, lat, lon)
    predictions_reshaped = predictions.reshape((*original_shape, n_models))

    # Convert to xarray
    product_names = list(model.models.keys())
    predictions_xr = xr.DataArray(
        predictions_reshaped,
        dims=['time', 'lat', 'lon', 'model'],
        coords={
            'time': predictors_1km.coords['time'],
            'lat': predictors_1km.coords['lat'],
            'lon': predictors_1km.coords['lon'],
            'model': product_names
        },
        name='T_ET_ratio_1km'
    )

    predictions_xr.attrs['description'] = 'Downscaled T/ET ratios at 1km resolution'
    predictions_xr.attrs['source_models'] = product_names
    predictions_xr.attrs['downscaler_type'] = model.model_type

    logger.info(f"  Generated {n_models} candidate products")
    logger.info(f"  Output shape: {predictions_xr.shape}")

    return predictions_xr


def evaluate_downscaling(
    predictions: xr.DataArray,
    parent_data: xr.DataArray,
    predictors_1km: xr.DataArray
) -> Dict[str, float]:
    """
    Evaluate downscaling performance.

    Compares downscaled 1km predictions (aggregated to parent resolution)
    with original parent data.

    Parameters
    ----------
    predictions : xr.DataArray
        1km predictions, dims: (time, lat, lon, model)
    parent_data : xr.DataArray
        Original parent data, dims: (time, lat, lon, model)
    predictors_1km : xr.DataArray
        Predictors used for downscaling

    Returns
    -------
    dict
        Performance metrics: r2, rmse, mae, bias
    """
    logger.info("Evaluating downscaling performance...")

    # Aggregate 1km predictions to parent resolution
    predictions_agg = _aggregate_to_parent_grid(predictions, parent_data)

    # Compute metrics
    metrics = {}

    # Stack for easier computation
    pred_stacked = predictions_agg.stack(sample=('time', 'lat', 'lon', 'model')).values
    parent_stacked = parent_data.stack(sample=('time', 'lat', 'lon', 'model')).values

    # Remove NaN values
    valid_mask = ~(np.isnan(pred_stacked) | np.isnan(parent_stacked))
    pred_valid = pred_stacked[valid_mask]
    parent_valid = parent_stacked[valid_mask]

    # Compute metrics
    metrics['r2'] = r2_score(parent_valid, pred_valid)
    metrics['rmse'] = np.sqrt(mean_squared_error(parent_valid, pred_valid))
    metrics['mae'] = mean_absolute_error(parent_valid, pred_valid)
    metrics['bias'] = np.mean(pred_valid - parent_valid)
    metrics['correlation'] = np.corrcoef(pred_valid, parent_valid)[0, 1]

    logger.info("  Downscaling evaluation:")
    for metric, value in metrics.items():
        logger.info(f"    {metric:15s}: {value:.4f}")

    return metrics


def _get_default_hyperparams(model_type: str) -> Dict[str, Any]:
    """Get default hyperparameters for each model type."""
    if model_type == 'random_forest':
        return {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'random_state': 42
        }
    elif model_type == 'gradient_boosting':
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'random_state': 42
        }
    elif model_type == 'ridge':
        return {
            'alpha': 1.0,
            'random_state': 42
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _initialize_model(model_type: str, hyperparams: Dict[str, Any]) -> Any:
    """Initialize ML model with specified hyperparameters."""
    if model_type == 'random_forest':
        return RandomForestRegressor(**hyperparams)
    elif model_type == 'gradient_boosting':
        return GradientBoostingRegressor(**hyperparams)
    elif model_type == 'ridge':
        return Ridge(**hyperparams)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _prepare_training_data(
    parent_data: xr.DataArray,
    predictors_1km: xr.DataArray,
    n_samples: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Prepare training data by co-locating parent T/ET with 1km predictors.

    Strategy:
    - Aggregate 1km predictors to parent grid resolution
    - Create (X, y) pairs where X = aggregated predictors, y = parent T/ET
    - Optionally subsample for computational efficiency

    Returns:
    - X: Predictor matrix (n_samples, n_features)
    - y_dict: {product_name: target_vector (n_samples,)}
    """
    # Aggregate predictors to parent grid
    predictors_agg = _aggregate_to_parent_grid(predictors_1km, parent_data)

    # Stack data
    X = predictors_agg.stack(sample=('time', 'lat', 'lon')).values.T  # (n_samples, n_features)

    y_dict = {}
    for product in parent_data.coords['model'].values:
        y = parent_data.sel(model=product).stack(sample=('time', 'lat', 'lon')).values
        y_dict[str(product)] = y

    # Remove NaN values
    valid_mask = ~np.isnan(X).any(axis=1)
    for product in y_dict:
        valid_mask &= ~np.isnan(y_dict[product])

    X = X[valid_mask]
    for product in y_dict:
        y_dict[product] = y_dict[product][valid_mask]

    # Subsample if requested
    if n_samples is not None and X.shape[0] > n_samples:
        logger.info(f"  Subsampling {n_samples} from {X.shape[0]} available samples")
        indices = np.random.choice(X.shape[0], n_samples, replace=False)
        X = X[indices]
        for product in y_dict:
            y_dict[product] = y_dict[product][indices]

    return X, y_dict


def _prepare_prediction_data(predictors_1km: xr.DataArray) -> np.ndarray:
    """
    Prepare predictor matrix for prediction at 1km resolution.

    Returns:
    - X: Predictor matrix (n_pixels, n_features)
    """
    # Stack all spatial and temporal dimensions
    X = predictors_1km.stack(sample=('time', 'lat', 'lon')).values.T

    # Handle NaN values (simple forward-fill or mean imputation)
    # For now, just warn if NaNs present
    n_nan = np.isnan(X).sum()
    if n_nan > 0:
        logger.warning(f"  {n_nan} NaN values in predictors (will be filled with 0)")
        X = np.nan_to_num(X, nan=0.0)

    return X


def _aggregate_to_parent_grid(
    data_1km: xr.DataArray,
    parent_data: xr.DataArray
) -> xr.DataArray:
    """
    Aggregate 1km data to parent grid resolution.

    Uses spatial averaging (can be customized to other methods).
    """
    # Simple implementation: use xarray's interp_like or coarsen
    # For now, return a placeholder (actual implementation would use regridding)

    # This is a simplified version - actual implementation would use:
    # - xesmf for conservative regridding
    # - or xarray.coarsen for block averaging

    logger.info("  Aggregating 1km data to parent grid (placeholder)")

    # Placeholder: just return data as-is
    # In production, implement proper regridding:
    # data_agg = data_1km.interp_like(parent_data, method='linear')

    return data_1km


def save_predictions(
    predictions: xr.DataArray,
    output_file: str
) -> None:
    """
    Save 1km predictions to NetCDF file.

    Parameters
    ----------
    predictions : xr.DataArray
        Predictions to save
    output_file : str
        Output file path
    """
    logger.info(f"Saving predictions to {output_file}")

    # Add metadata
    predictions.attrs['creation_date'] = str(np.datetime64('today'))
    predictions.attrs['units'] = 'dimensionless'
    predictions.attrs['long_name'] = 'Transpiration efficiency (T/ET ratio)'

    # Save
    predictions.to_netcdf(output_file)

    logger.info(f"  Saved {output_file}")
