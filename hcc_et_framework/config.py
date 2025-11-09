"""
Configuration Management for HCC-ET Framework
==============================================

Handles loading and validation of framework configuration from YAML files.

Author: [Your Name]
Date: 2025-11-09
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

import yaml

logger = logging.getLogger(__name__)


# Default configuration template
DEFAULT_CONFIG = {
    # Data paths
    'data': {
        'parent_dir': '/data/parent_products',
        'parent_products': ['GLEAM', 'PMLv2', 'GLDAS'],
        'predictors_dir': '/data/predictors_1km',
        'predictor_vars': ['LST', 'NDVI', 'Albedo', 'Precip'],
        'sapflux_dir': '/data/sapfluxnet',
        'wb_et_dir': '/data/water_balance',
        'total_et_dir': '/data/total_et',
        'total_et_product': 'CAMELE',
        'landcover_dir': '/data/landcover',
        'basin_mask_file': '/data/masks/basin_mask_1km.nc',
        'pft_map_file': '/data/pft/pft_map_1km.nc',
    },

    # Time periods
    'time': {
        'calibration_period': ['1981-01-01', '2025-12-31'],
        'hindcast_period': ['1950-01-01', '1980-12-31'],
    },

    # Downscaling configuration
    'downscaling': {
        'model_type': 'random_forest',
        'hyperparams': {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'random_state': 42,
        },
        'cv_folds': 5,
        'n_features': 4,
        'evaluate': True,
    },

    # Fusion configuration
    'fusion': {
        'mode': 'auto',  # 'auto', 'eivd', 'ec', 'gls'
        'shrinkage': 'lw',  # 'lw', 'oas', 'ridge', 'none'
        'robust': False,
        'spatial_varying_weights': False,
        'ec_criterion': 'min_fMSE',  # 'min_fMSE', 'max_SNR'
        'uncertainty_quantification': True,
        'bayesian_sample_size': 10000,
        'bayesian_niter': 2000,
        'bayesian_nadvi': 200000,
    },

    # Physical constraints configuration
    'constraints': {
        'apply_wb_constraint': True,
        'apply_sapflux_constraint': True,
        'sapflux_upscale_method': 'nearest',
        'sapflux_correction_strength': 0.5,
        'wb_tolerance': 0.05,  # 5% relative tolerance
        'basin_ids': None,  # None = all basins
    },

    # Analysis configuration
    'analysis': {
        'landcover_classes': ['natural', 'managed', 'urban'],
        'trend_method': 'theil_sen',
        'significance_level': 0.05,
    },
}


def load_config(config_file: str | Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_file : str or Path
        Path to configuration YAML file

    Returns
    -------
    dict
        Configuration dictionary with all parameters

    Examples
    --------
    >>> config = load_config('config.yaml')
    >>> print(config['data']['parent_products'])
    ['GLEAM', 'PMLv2', 'GLDAS']
    """
    config_file = Path(config_file)

    if not config_file.exists():
        logger.warning(f"Config file not found: {config_file}")
        logger.warning("Using default configuration")
        return DEFAULT_CONFIG.copy()

    logger.info(f"Loading configuration from {config_file}")

    with open(config_file, 'r') as f:
        user_config = yaml.safe_load(f)

    # Merge with defaults (user config overrides defaults)
    config = _merge_configs(DEFAULT_CONFIG, user_config)

    # Validate configuration
    _validate_config(config)

    logger.info("Configuration loaded successfully")

    return config


def save_config(config: Dict[str, Any], output_file: str | Path) -> None:
    """
    Save configuration to YAML file.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    output_file : str or Path
        Output file path
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving configuration to {output_file}")

    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info("Configuration saved")


def create_default_config(output_file: str | Path) -> None:
    """
    Create a default configuration file for user customization.

    Parameters
    ----------
    output_file : str or Path
        Output file path

    Examples
    --------
    >>> create_default_config('config_template.yaml')
    """
    save_config(DEFAULT_CONFIG, output_file)
    logger.info(f"Default configuration template created: {output_file}")


def _merge_configs(default: Dict, user: Dict) -> Dict:
    """
    Recursively merge user config with defaults.

    User values override defaults, but missing keys use default values.
    """
    merged = default.copy()

    for key, value in user.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = _merge_configs(merged[key], value)
        else:
            # Override with user value
            merged[key] = value

    return merged


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration parameters.

    Checks:
    - Required keys are present
    - Data paths exist (warn if missing)
    - Parameter values are in valid ranges
    """
    logger.info("Validating configuration...")

    # Check required top-level keys
    required_keys = ['data', 'time', 'downscaling', 'fusion', 'constraints', 'analysis']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration section: {key}")

    # Validate data paths (warn if missing, but don't fail)
    data_config = config['data']
    path_keys = ['parent_dir', 'predictors_dir', 'sapflux_dir', 'wb_et_dir']

    for key in path_keys:
        if key in data_config:
            path = Path(data_config[key])
            if not path.exists():
                logger.warning(f"Data path does not exist: {path}")

    # Validate parent products
    if not data_config.get('parent_products'):
        raise ValueError("At least one parent product must be specified")

    # Validate fusion mode
    valid_modes = ['auto', 'eivd', 'ec', 'gls']
    if config['fusion']['mode'] not in valid_modes:
        raise ValueError(f"Invalid fusion mode: {config['fusion']['mode']}")

    # Validate downscaling model
    valid_models = ['random_forest', 'gradient_boosting', 'ridge']
    if config['downscaling']['model_type'] not in valid_models:
        raise ValueError(f"Invalid downscaling model: {config['downscaling']['model_type']}")

    # Validate constraint parameters
    wb_tol = config['constraints']['wb_tolerance']
    if not (0 < wb_tol < 1):
        raise ValueError(f"WB tolerance must be in (0, 1), got {wb_tol}")

    sapflux_strength = config['constraints']['sapflux_correction_strength']
    if not (0 <= sapflux_strength <= 1):
        raise ValueError(f"SapFlux correction strength must be in [0, 1], got {sapflux_strength}")

    logger.info("Configuration validation passed")


def print_config(config: Dict[str, Any]) -> None:
    """
    Print configuration in human-readable format.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    """
    print("\n" + "=" * 60)
    print("HCC-ET Framework Configuration")
    print("=" * 60)

    def _print_section(section_dict: Dict, indent: int = 0):
        """Recursively print configuration sections."""
        for key, value in section_dict.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                _print_section(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")

    _print_section(config)

    print("=" * 60 + "\n")


def get_nested_config(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested configuration value using dot-separated key path.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    key_path : str
        Dot-separated key path (e.g., 'data.parent_products')
    default : any, optional
        Default value if key not found

    Returns
    -------
    any
        Configuration value

    Examples
    --------
    >>> config = load_config('config.yaml')
    >>> products = get_nested_config(config, 'data.parent_products')
    >>> print(products)
    ['GLEAM', 'PMLv2', 'GLDAS']
    """
    keys = key_path.split('.')
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value
