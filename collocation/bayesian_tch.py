"""
Bayesian Three-Cornered Hat (BTCH) Method
==========================================

This module implements a standard Bayesian Three-Cornered Hat (TCH) analysis.

This model is a simpler, non-time-varying Bayesian implementation, 
contrasting with the more complex, time-varying `BayesianTC` class.
It assumes a constant, unknown truth and Gaussian errors with 
unknown (constant) variances.

It is designed to be API-compatible with the existing BayesianTC class
for easy integration into comparison scripts.

References:
    - W. C. S. L. (2010). Bayesian Three-Cornered Hat. 
      NIST. (Internal report)
    - K. R. (2007). A Bayesian "three-cornered hat" method 
      for clocks and frequency standards.
"""

import numpy as np
import warnings

try:
    import pymc3 as pm
    import theano.tensor as tt
    PYMC3_AVAILABLE = True
except ImportError:
    PYMC3_AVAILABLE = False
    warnings.warn(
        "PyMC3 not available. Bayesian TCH method will not work. "
        "Install with: pip install pymc3==3.11.5 theano-pymc",
        ImportWarning
    )

def check_dependencies():
    """Check if required dependencies are available."""
    if not PYMC3_AVAILABLE:
        raise ImportError(
            "PyMC3 is required for Bayesian Three-Cornered Hat.\n"
            "Install with: pip install pymc3==3.11.5 theano-pymc"
        )

class BayesianTCH:
    """
    Standard Bayesian Three-Cornered Hat (TCH) Analysis

    This class provides a standard Bayesian approach to TCH, assuming
    constant error variances (homoscedastic).

    Parameters
    ----------
    data : ndarray, shape (3, n_samples)
        Input data matrix where each row is a different product.
        This standard model *requires exactly 3* products.

    Attributes
    ----------
    n_products : int
        Number of products (sensors), must be 3
    n_samples : int
        Number of samples (time points)
    model : pymc3.Model
        The Bayesian model
    trace : pymc3.backends.base.MultiTrace
        MCMC samples from posterior distribution
    """

    def __init__(self, data):
        """Initialize Bayesian TCH with data."""
        check_dependencies()

        self.data = np.asarray(data)
        if self.data.ndim != 2:
            raise ValueError("Data must be 2-dimensional (n_products, n_samples)")

        self.n_products, self.n_samples = self.data.shape

        if self.n_products != 3:
            raise ValueError(
                f"BayesianTCH requires exactly 3 products, but got {self.n_products}"
            )

        self.model = None
        self.trace = None

    def setup_model(self, **kwargs):
        """
        Setup the Bayesian model.
        
        Model:
            X_i = T + e_i
            e_i ~ Normal(0, sigma_i^2)
            T ~ Normal(mu_T, sigma_T^2) (unknown truth)
        
        Priors:
            sigma_i ~ HalfCauchy(beta=1)
            mu_T ~ Normal(data_mean, 100) (weakly informative)
            sigma_T ~ HalfCauchy(beta=1)
        """
        with pm.Model() as self.model:
            # Data
            X1_data = self.data[0, :]
            X2_data = self.data[1, :]
            X3_data = self.data[2, :]

            # Priors for error standard deviations
            sigma_e1 = pm.HalfCauchy('sigma_e1', beta=1)
            sigma_e2 = pm.HalfCauchy('sigma_e2', beta=1)
            sigma_e3 = pm.HalfCauchy('sigma_e3', beta=1)

            # Priors for the true signal T
            # Weakly informative prior on mean, centered at data mean
            mu_T = pm.Normal('mu_T', mu=np.mean(self.data), sd=100) 
            sigma_T = pm.HalfCauchy('sigma_T', beta=1)

            # Latent variable for true signal
            # shape = self.n_samples
            T = pm.Normal('T', mu=mu_T, sd=sigma_T, shape=self.n_samples) 

            # Likelihood
            X1 = pm.Normal('X1', mu=T, sd=sigma_e1, observed=X1_data)
            X2 = pm.Normal('X2', mu=T, sd=sigma_e2, observed=X2_data)
            X3 = pm.Normal('X3', mu=T, sd=sigma_e3, observed=X3_data)

            # Store RMSEs (which are just the sigmas in this model)
            pm.Deterministic('sigmap', tt.stack([sigma_e1, sigma_e2, sigma_e3]))

        return self.model

    def run_inference(self, niter=2000, nadvi=50000, ntraceadvi=1000,
                      seed=123, nchains=2, **model_kwargs):
        """
        Run Bayesian inference (ADVI + NUTS MCMC).

        Parameters are kept compatible with BayesianTC.
        """
        if self.model is None:
            self.setup_model(**model_kwargs)

        with self.model:
            # Initialize with ADVI for better NUTS sampling
            np.random.seed(seed)
            try:
                # Try to find a good starting point using ADVI
                v_params = pm.variational.advi(n=nadvi)
                start = pm.variational.sample_vp(v_params, draws=1)[0]
                
                # Run NUTS sampler
                self.trace = pm.sample(
                    niter, 
                    start=start, 
                    chains=nchains, 
                    random_seed=seed, 
                    cores=1, # Keep cores=1 for reproducibility/simplicity
                    progressbar=True
                )
            except Exception as e:
                # Fallback to standard sampling if ADVI fails
                warnings.warn(f"ADVI initialization failed ({e}). Falling back to default init.")
                self.trace = pm.sample(
                    niter, 
                    chains=nchains, 
                    random_seed=seed, 
                    cores=1, 
                    progressbar=True
                )
        
        return self.trace

    def get_error_estimates(self):
        """
        Get error (RMSE) estimates from posterior samples.
        In this model, RMSE = sigma_e.
        """
        if self.trace is None:
            raise ValueError("Must run inference first")

        sigmap_samples = self.trace.get_values('sigmap')

        rmse_mean = sigmap_samples.mean(axis=0)
        rmse_std = sigmap_samples.std(axis=0)
        rmse_quantiles = np.percentile(sigmap_samples, [2.5, 50, 97.5], axis=0).T

        return rmse_mean, rmse_std, rmse_quantiles

    def summary(self):
        """Print summary of inference results."""
        if self.trace is None:
            raise ValueError("Must run inference first")
        
        rmse_mean, rmse_std, rmse_quantiles = self.get_error_estimates()

        print("\n" + "="*60)
        print("Bayesian Three-Cornered Hat (BTCH) Results")
        print("="*60)
        print(f"\nRMSE Estimates (Posterior Mean ± Std):")
        print("-" * 60)

        for i in range(self.n_products):
            print(f"Product {i+1}: {rmse_mean[i]:.4f} ± {rmse_std[i]:.4f}")
            print(f"  95% CI: [{rmse_quantiles[i, 0]:.4f}, {rmse_quantiles[i, 2]:.4f}]")
        
        print("="*60 + "\n")