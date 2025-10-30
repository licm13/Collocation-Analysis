"""
Bayesian Triple Collocation

A Bayesian approach to triple collocation that estimates complex, non-constant error structures.
This module provides a simplified interface to the BayesianTripleCollocation package.

Based on the work by Simon Zwieback:
https://github.com/szwieback/BayesianTripleCollocation

References
----------
Zwieback, S., et al. (2012). Structural and statistical properties of the collocation technique
for error characterization. Nonlin. Processes Geophys., 19, 69-80.

Gruber, A., et al. (2016). Estimating error cross-correlations in soil moisture data sets
using extended collocation analysis. JGR: Atmospheres, 121(3), 1208-1219.
"""

import sys
import os
import numpy as np
import warnings

# Add BayesianTripleCollocation to path
btc_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'BayesianTripleCollocation-master')
if btc_path not in sys.path:
    sys.path.insert(0, btc_path)

try:
    import pymc3 as pm
    PYMC3_AVAILABLE = True
except ImportError:
    PYMC3_AVAILABLE = False
    warnings.warn(
        "PyMC3 not available. Bayesian TC methods will not work. "
        "Install with: pip install pymc3==3.11.5 theano-pymc",
        ImportWarning
    )


def check_dependencies():
    """Check if required dependencies are available."""
    if not PYMC3_AVAILABLE:
        raise ImportError(
            "PyMC3 is required for Bayesian Triple Collocation.\n"
            "Install with: pip install pymc3==3.11.5 theano-pymc"
        )


class BayesianTC:
    """
    Bayesian Triple Collocation Analysis

    This class provides a Bayesian approach to triple collocation that can handle:
    - Time-varying error variances
    - Complex error structures
    - Non-constant calibration parameters
    - Uncertainty quantification through MCMC

    Parameters
    ----------
    data : ndarray, shape (n_products, n_samples)
        Input data matrix where each row is a different product

    Attributes
    ----------
    n_products : int
        Number of products (sensors)
    n_samples : int
        Number of samples (time points)
    model : pymc3.Model
        The Bayesian model
    trace : pymc3.backends.base.MultiTrace
        MCMC samples from posterior distribution
    """

    def __init__(self, data):
        """Initialize Bayesian TC with data."""
        check_dependencies()

        self.data = np.asarray(data)
        if self.data.ndim != 2:
            raise ValueError("Data must be 2-dimensional (n_products, n_samples)")

        self.n_products, self.n_samples = self.data.shape

        if self.n_products < 3:
            raise ValueError("At least 3 products are required for triple collocation")

        self.model = None
        self.trace = None
        self.v_params = None
        self.tracevi = None

    def setup_model(self, **inference_params):
        """
        Setup the Bayesian model.

        Parameters
        ----------
        **inference_params : dict, optional
            Additional parameters for model setup:
            - doft : int, default=4
                Degrees of freedom for Student-t priors
            - priorfactor : float, default=1.0
                Prior scaling factor
            - thetaoffset : float, default=0.15
                Offset for multiplicative bias
            - thetamodel : str, default='beta'
                Distribution model for true signal ('beta' or 'logistic')
            - studenterrors : bool, default=False
                Use Student-t errors instead of Gaussian
        """
        from model_setup import model_setup
        from normalize_weights import normalize_weights

        # Prepare data
        visible = {'y': self.data}
        normalized_weights = normalize_weights(n=self.n_samples)

        # Setup model
        self.model = model_setup(
            visible=visible,
            normalized_weights=normalized_weights,
            inferenceparams=inference_params
        )

        return self.model

    def run_inference(self, niter=2000, nadvi=200000, ntraceadvi=1000,
                      seed=123, nchains=2, **model_kwargs):
        """
        Run Bayesian inference using ADVI + NUTS MCMC.

        Parameters
        ----------
        niter : int, default=2000
            Number of MCMC iterations per chain
        nadvi : int, default=200000
            Number of ADVI iterations
        ntraceadvi : int, default=1000
            Number of samples from variational posterior
        seed : int, default=123
            Random seed
        nchains : int, default=2
            Number of MCMC chains
        **model_kwargs : dict
            Additional parameters passed to setup_model

        Returns
        -------
        trace : pymc3.backends.base.MultiTrace
            MCMC samples from posterior
        """
        from model_inference import model_inference

        # Setup model if not already done
        if self.model is None:
            self.setup_model(**model_kwargs)

        # Run inference
        self.trace, self.v_params, self.tracevi = model_inference(
            self.model,
            niter=niter,
            nadvi=nadvi,
            ntraceadvi=ntraceadvi,
            seed=seed,
            nchains=nchains
        )

        return self.trace

    def get_error_estimates(self):
        """
        Get error (RMSE) estimates from posterior samples.

        Returns
        -------
        rmse_mean : ndarray, shape (n_products,)
            Posterior mean of RMSE for each product
        rmse_std : ndarray, shape (n_products,)
            Posterior standard deviation of RMSE
        rmse_quantiles : ndarray, shape (n_products, 3)
            2.5%, 50%, 97.5% quantiles of RMSE
        """
        if self.trace is None:
            raise ValueError("Must run inference first")

        sigmap_samples = self.trace.get_values('sigmap')

        rmse_mean = sigmap_samples.mean(axis=0)
        rmse_std = sigmap_samples.std(axis=0)
        rmse_quantiles = np.percentile(sigmap_samples, [2.5, 50, 97.5], axis=0).T

        return rmse_mean, rmse_std, rmse_quantiles

    def get_calibration_parameters(self):
        """
        Get calibration parameter estimates.

        Returns
        -------
        m_mean : ndarray, shape (n_products-1,)
            Additive bias (relative to reference product)
        l_mean : ndarray, shape (n_products-1,)
            Multiplicative bias (relative to reference product)
        """
        if self.trace is None:
            raise ValueError("Must run inference first")

        m_mean = self.trace.get_values('mest').mean(axis=0)
        l_mean = self.trace.get_values('lest').mean(axis=0)

        return m_mean, l_mean

    def get_posterior_samples(self, var_name):
        """
        Get posterior samples for a specific variable.

        Parameters
        ----------
        var_name : str
            Name of the variable (e.g., 'sigmap', 'mest', 'lest', 'theta')

        Returns
        -------
        samples : ndarray
            Posterior samples
        """
        if self.trace is None:
            raise ValueError("Must run inference first")

        return self.trace.get_values(var_name)

    def summary(self):
        """Print summary of inference results."""
        if self.trace is None:
            raise ValueError("Must run inference first")

        rmse_mean, rmse_std, rmse_quantiles = self.get_error_estimates()

        print("\n" + "="*60)
        print("Bayesian Triple Collocation Results")
        print("="*60)
        print(f"\nNumber of products: {self.n_products}")
        print(f"Number of samples: {self.n_samples}")
        print(f"\nRMSE Estimates (Posterior Mean ± Std):")
        print("-" * 60)

        for i in range(self.n_products):
            print(f"Product {i}: {rmse_mean[i]:.4f} ± {rmse_std[i]:.4f}")
            print(f"  95% CI: [{rmse_quantiles[i, 0]:.4f}, {rmse_quantiles[i, 2]:.4f}]")

        # Get calibration parameters
        m_mean, l_mean = self.get_calibration_parameters()

        print(f"\nCalibration Parameters (relative to reference product 0):")
        print("-" * 60)
        for i in range(self.n_products - 1):
            print(f"Product {i+1}:")
            print(f"  Additive bias (m): {m_mean[i]:.4f}")
            print(f"  Multiplicative bias (l): {l_mean[i]:.4f}")

        print("="*60 + "\n")


def bayesian_tc(data, niter=2000, nadvi=200000, seed=123, **kwargs):
    """
    Convenience function for Bayesian Triple Collocation.

    Parameters
    ----------
    data : ndarray, shape (n_products, n_samples)
        Input data matrix
    niter : int, default=2000
        Number of MCMC iterations
    nadvi : int, default=200000
        Number of ADVI iterations
    seed : int, default=123
        Random seed
    **kwargs : dict
        Additional parameters for model setup

    Returns
    -------
    btc : BayesianTC
        Fitted Bayesian TC object
    rmse : ndarray
        RMSE estimates for each product
    """
    btc = BayesianTC(data)
    btc.run_inference(niter=niter, nadvi=nadvi, seed=seed, **kwargs)
    rmse, _, _ = btc.get_error_estimates()

    return btc, rmse


def simulate_products(n=500, n_products=3, seed=123, **params):
    """
    Simulate synthetic products for testing.

    Parameters
    ----------
    n : int, default=500
        Number of samples
    n_products : int, default=3
        Number of products
    seed : int, default=123
        Random seed
    **params : dict
        Simulation parameters:
        - m : ndarray, additive biases
        - l : ndarray, multiplicative biases
        - sigmapsquared : ndarray, error variances
        - porosity : float, default=0.4
        - thetaoffset : float, default=0.15

    Returns
    -------
    data : ndarray, shape (n_products, n)
        Simulated product data
    truth : ndarray, shape (n,)
        True signal
    params : dict
        Simulation parameters used
    """
    from simulate_product import simulate_product

    # Default parameters
    default_params = {
        'm': np.array([0.0] + [0.03, -0.05][:n_products-1]),
        'l': np.array([1.0] + [1.1, 0.9][:n_products-1]),
        'sigmapsquared': np.array([0.02, 0.04, 0.05][:n_products])**2,
        'porosity': 0.4,
        'thetaoffset': 0.15,
        'soilmoisturemodel': 'uniform'
    }

    # Update with user parameters
    sim_params = {**default_params, **params}

    # Ensure arrays have correct length
    for key in ['m', 'l', 'sigmapsquared']:
        if len(sim_params[key]) != n_products:
            sim_params[key] = np.pad(
                sim_params[key][:n_products],
                (0, max(0, n_products - len(sim_params[key]))),
                mode='edge'
            )

    numpy_rng = np.random.RandomState(seed)
    visible, internal, normalized_weights = simulate_product(n, sim_params, numpy_rng=numpy_rng)

    return visible['y'], internal['theta'], internal['params']


if __name__ == "__main__":
    # Example usage
    print("Bayesian Triple Collocation Example")
    print("=" * 60)

    # Simulate data
    print("\nSimulating synthetic data...")
    data, truth, params = simulate_products(n=500, n_products=3, seed=42)

    print(f"Data shape: {data.shape}")
    print(f"True RMSE: {np.sqrt(params['sigmapsquared'])}")

    # Run Bayesian TC
    print("\nRunning Bayesian inference...")
    print("(This may take several minutes...)")

    btc = BayesianTC(data)
    btc.run_inference(niter=1000, nadvi=100000, seed=42)

    # Show results
    btc.summary()
