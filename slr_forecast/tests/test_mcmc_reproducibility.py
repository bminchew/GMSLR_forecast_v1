"""A fixed ``seed`` must reproduce the MCMC chain exactly.

emcee draws its stretch-move proposals from a sampler-local RandomState that
is initialized from NumPy's global state, so the fitters seed that stream
explicitly. These tests perturb the global state between two identical calls.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks'))

from bayesian_models import fit_bayesian_level, fit_bayesian_level_annual_correlated  # noqa: E402

YEARS = np.arange(2000.5, 2024.5)
T = 0.8 + 0.02 * (YEARS - 2000)
RATE = (0.6 * T + 0.3) * 1e-3
SIGMA = np.full_like(RATE, 0.2e-3)
H = np.cumsum(RATE) - RATE[0]


def _run_annual(global_seed):
    np.random.seed(global_seed)
    res = fit_bayesian_level_annual_correlated(
        H_obs=H, rate_obs=RATE, sigma_rate_obs=SIGMA, temperature=T, time=YEARS,
        order=1, fit_sigma_extra=False, n_samples=200, n_walkers=16, n_burnin=100,
        thin=1, seed=7, progress=False)
    return res.posterior_samples


def _run_level(global_seed):
    np.random.seed(global_seed)
    tau = YEARS - YEARS[0]
    res = fit_bayesian_level(
        H_obs=H, sigma_obs=np.sqrt(np.cumsum(SIGMA**2)),
        I2_obs=np.zeros_like(tau), I1_obs=np.cumsum(T), I0_obs=tau,
        n_samples=200, n_walkers=16, n_burnin=100, thin=1, seed=7, progress=False)
    return res.posterior_samples


def test_annual_correlated_fit_is_reproducible():
    assert np.array_equal(_run_annual(1), _run_annual(12345))


def test_level_fit_is_reproducible():
    assert np.array_equal(_run_level(1), _run_level(12345))
