#!/usr/bin/env python3
"""
Command-line tool for fusing evapotranspiration (ET) products.

Usage:
    python fuse_et.py --inputs file1.nc file2.nc file3.nc \\
                      --var ET \\
                      --mode gls \\
                      --shrinkage lw \\
                      --output fused_et.nc \\
                      --weights weights.nc \\
                      --diagnostics

Features:
- Multiple fusion modes: IVW, GLS, constrained QP
- Automatic MSE estimation with moving windows
- Covariance shrinkage (Ledoit-Wolf, OAS, Ridge)
- Constraint support (sum-to-one, bounds, physics)
- Robust estimation (outlier handling)
- Comprehensive diagnostics and uncertainty quantification
- NetCDF output with CF-compliant metadata

Examples:
    # Simple IVW fusion
    python fuse_et.py --inputs ERA5.nc GLEAM.nc GLDAS.nc --var ET --mode ivw

    # GLS with shrinkage
    python fuse_et.py --inputs ERA5.nc GLEAM.nc GLDAS.nc --var ET --mode gls \\
                      --shrinkage lw --mse-window time=30

    # Constrained QP with bounds
    python fuse_et.py --inputs *.nc --var ET --mode qp --bounds 0.1 0.9

    # Robust fusion (outlier handling)
    python fuse_et.py --inputs *.nc --var ET --mode ivw --robust
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import xarray as xr

# Import fusion modules
from collocation.fusion import fuse_fields


def parse_window_spec(window_str: str) -> dict:
    """
    Parse window specification string.

    Examples:
        "time=30" → {"time": 30}
        "space=7" → {"space": 7}
        "time=30,space=7" → {"time": 30, "space": 7}
    """
    window = {}
    for item in window_str.split(","):
        key, val = item.split("=")
        try:
            window[key.strip()] = int(val.strip())
        except ValueError:
            window[key.strip()] = val.strip()  # Keep as string (e.g., "30D")
    return window


def load_et_products(
    file_paths: List[str],
    var_name: str,
    model_dim: str = "model",
) -> xr.DataArray:
    """
    Load multiple ET products and stack into multi-model array.

    Parameters
    ----------
    file_paths : list of str
        Paths to NetCDF files.
    var_name : str
        Variable name to extract.
    model_dim : str
        Name for model dimension.

    Returns
    -------
    xr.DataArray
        Stacked multi-model data with 'model' dimension.
    """
    print(f"Loading {len(file_paths)} ET products...")

    datasets = []
    model_names = []

    for fpath in file_paths:
        ds = xr.open_dataset(fpath)

        if var_name not in ds:
            print(f"Warning: Variable '{var_name}' not found in {fpath}")
            continue

        da = ds[var_name]

        # Extract model name from filename
        model_name = Path(fpath).stem
        model_names.append(model_name)

        datasets.append(da)

    if not datasets:
        raise ValueError(f"No datasets loaded. Check variable name '{var_name}'.")

    # Stack along model dimension
    X = xr.concat(datasets, dim=model_dim)
    X = X.assign_coords({model_dim: model_names})

    print(f"Loaded {len(datasets)} products: {', '.join(model_names)}")
    print(f"Data shape: {dict(X.sizes)}")

    return X


def save_fusion_results(
    result: dict,
    output_path: str,
    weights_path: str | None = None,
    diagnostics_path: str | None = None,
):
    """
    Save fusion results to NetCDF files.

    Parameters
    ----------
    result : dict
        Fusion result dictionary from fuse_fields().
    output_path : str
        Path for fused output.
    weights_path : str, optional
        Path for weight file.
    diagnostics_path : str, optional
        Path for diagnostics file.
    """
    # Save fused field
    print(f"\nSaving fused ET to {output_path}...")
    fused_ds = result["fused"].to_dataset(name="ET_fused")

    if "variance" in result:
        fused_ds["ET_variance"] = result["variance"]
        fused_ds["ET_std"] = result["std"]

    # Add metadata
    fused_ds["ET_fused"].attrs.update({
        "long_name": "Fused Evapotranspiration",
        "units": "mm/day",
        "fusion_mode": result["fused"].attrs.get("fusion_mode", "unknown"),
    })

    fused_ds.to_netcdf(output_path)
    print(f"Saved: {output_path}")

    # Save weights
    if weights_path and "weights" in result:
        print(f"Saving weights to {weights_path}...")
        weights_ds = result["weights"].to_dataset(name="fusion_weights")

        if "diagnostics" in result:
            diag = result["diagnostics"]
            for key, val in diag.items():
                if isinstance(val, (xr.DataArray, xr.Dataset)):
                    weights_ds[f"diag_{key}"] = val

        weights_ds.to_netcdf(weights_path)
        print(f"Saved: {weights_path}")

    # Save diagnostics
    if diagnostics_path and "diagnostics" in result:
        print(f"Saving diagnostics to {diagnostics_path}...")
        diag_ds = xr.Dataset()

        for key, val in result["diagnostics"].items():
            if isinstance(val, (xr.DataArray, xr.Dataset)):
                diag_ds[key] = val
            elif isinstance(val, (int, float, np.number)):
                diag_ds[key] = xr.DataArray(val)

        diag_ds.to_netcdf(diagnostics_path)
        print(f"Saved: {diagnostics_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fuse multiple ET products using optimal weighting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input/output
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input NetCDF files (space-separated)",
    )
    parser.add_argument(
        "--var",
        default="ET",
        help="Variable name to fuse (default: ET)",
    )
    parser.add_argument(
        "--output",
        default="fused_et.nc",
        help="Output file for fused ET (default: fused_et.nc)",
    )
    parser.add_argument(
        "--weights",
        help="Output file for fusion weights (optional)",
    )
    parser.add_argument(
        "--diagnostics-file",
        help="Output file for diagnostics (optional)",
    )

    # Fusion mode
    parser.add_argument(
        "--mode",
        choices=["ivw", "gls", "qp"],
        default="gls",
        help="Fusion mode: ivw (inverse variance), gls (generalized least squares), "
        "qp (constrained quadratic programming). Default: gls",
    )

    # Covariance options
    parser.add_argument(
        "--shrinkage",
        choices=["none", "lw", "oas", "ridge"],
        default="lw",
        help="Covariance shrinkage method (default: lw)",
    )
    parser.add_argument(
        "--lam",
        type=float,
        help="Shrinkage intensity (0-1). Auto if not specified.",
    )
    parser.add_argument(
        "--mse-window",
        help="Moving window for MSE estimation (e.g., 'time=30' or 'space=7')",
    )

    # Constraints (for QP mode)
    parser.add_argument(
        "--bounds",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Weight bounds: --bounds 0.0 1.0",
    )

    # Robust options
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Use robust MSE estimation (Huber loss)",
    )

    # Output options
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Don't compute fusion weights",
    )
    parser.add_argument(
        "--no-uncertainty",
        action="store_true",
        help="Don't compute uncertainty estimates",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Include diagnostic metrics",
    )

    # Reference data (optional, for validation)
    parser.add_argument(
        "--reference",
        help="Reference dataset for MSE validation (optional)",
    )
    parser.add_argument(
        "--ref-var",
        default="ET",
        help="Reference variable name (default: ET)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.inputs:
        parser.error("Must provide at least one input file")

    # Print configuration
    print("=" * 60)
    print("ET FUSION TOOL")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Inputs: {len(args.inputs)} files")
    print(f"Variable: {args.var}")
    print(f"Shrinkage: {args.shrinkage}")
    if args.robust:
        print("Robust estimation: ENABLED")

    # Load data
    try:
        X = load_et_products(args.inputs, args.var)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        return 1

    # Load reference if provided
    X_ref = None
    if args.reference:
        print(f"\nLoading reference from {args.reference}...")
        try:
            ref_ds = xr.open_dataset(args.reference)
            X_ref = ref_ds[args.ref_var]
            print(f"Reference loaded: {dict(X_ref.sizes)}")
        except Exception as e:
            print(f"Warning: Could not load reference: {e}")

    # Parse constraints
    constraints = None
    if args.mode == "qp":
        constraints = {"sum_to_one": True}

        if args.bounds:
            w_min, w_max = args.bounds
            constraints["bounds"] = (w_min, w_max)
            print(f"Bounds: [{w_min}, {w_max}]")

    # Parse MSE window
    window = None
    if args.mse_window:
        window = parse_window_spec(args.mse_window)
        print(f"MSE window: {window}")

    # Run fusion
    print("\n" + "=" * 60)
    print("RUNNING FUSION")
    print("=" * 60)

    try:
        result = fuse_fields(
            X,
            X_ref=X_ref,
            mode=args.mode,
            constraints=constraints,
            shrinkage=args.shrinkage,
            lam=args.lam,
            robust=args.robust,
            return_weights=not args.no_weights,
            return_var=not args.no_uncertainty,
            return_diagnostics=args.diagnostics,
        )
    except Exception as e:
        print(f"Error during fusion: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # Print summary
    print("\n" + "=" * 60)
    print("FUSION COMPLETE")
    print("=" * 60)

    if "weights" in result:
        print("\nFusion weights:")
        for model in result["weights"].coords["model"].values:
            w = result["weights"].sel(model=model)
            if w.ndim == 0:
                # Scalar weight
                print(f"  {model}: {float(w.values):.4f}")
            else:
                # Spatial weights
                print(f"  {model}: mean={float(w.mean()):.4f}, "
                      f"std={float(w.std()):.4f}")

    if "diagnostics" in result:
        print("\nDiagnostics:")
        diag = result["diagnostics"]
        for key, val in diag.items():
            if isinstance(val, (xr.DataArray, xr.Dataset)):
                if val.ndim == 0:
                    print(f"  {key}: {float(val.values):.4f}")
                else:
                    print(f"  {key}: mean={float(val.mean()):.4f}")
            elif isinstance(val, (int, float, np.number)):
                print(f"  {key}: {float(val):.4f}")

    # Save results
    save_fusion_results(
        result,
        args.output,
        weights_path=args.weights,
        diagnostics_path=args.diagnostics_file,
    )

    print("\n" + "=" * 60)
    print("SUCCESS")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
