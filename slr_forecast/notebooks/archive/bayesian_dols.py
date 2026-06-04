#!/usr/bin/env python3
"""
Bayesian Complement to DOLS Sea-Level Rate Models
====================================================

Six Bayesian models that complement the frequentist DOLS framework:

1. **Bayesian Static DOLS** — Same regression as calibrate_dols() but with
   posterior distributions instead of point estimates + HAC SEs.
2. **Bayesian DLM** — Time-varying coefficients via Gaussian random walk,
   providing a principled alternative to sliding-window DOLS.
3. **Hierarchical Multi-Dataset** — Partial pooling across GMSL datasets,
   replacing ad hoc ensemble statistics.
4. **Bayesian Rate-Space** — Fit rate(T) directly against observed kinematic
   rates with correlated errors and physically informed priors (α₀ ≥ 0).
   Avoids the collinearity of ∫T²/∫T in level-space DOLS.
5. **Bayesian Level-Space** — Self-consistent forward model integrating
   rate(T) to predict cumulative GMSL with time-varying uncertainty.
6. **Bayesian Rate-and-State** — Extends level-space with a state variable
   S(t) with relaxation time τ for lagged ocean/ice-sheet response.
   Reduces to model 5 as τ→0.

Backend: emcee (affine-invariant ensemble sampler) + arviz (diagnostics).
Design matrix construction replicates calibrate_dols() exactly.

Authors: Minchew 
"""

import sys
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.optimize import minimize
import emcee
import arviz as az
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=FutureWarning)

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from slr_analysis import (calibrate_dols, DOLSResult, KinematicsResult,
                          compute_kinematics, _to_month_start)


# =============================================================================
# DESIGN MATRIX BUILDER  (replicates calibrate_dols design matrix construction)
# =============================================================================


def build_dols_design_matrix(sea_level, temperature, gmsl_sigma=None,
                              saod=None, order=2, n_lags=2):
    """Build the DOLS design matrix without fitting.

    Replicates the exact construction from calibrate_dols() so that
    Bayesian models operate on the identical regressor matrix.

    Parameters
    ----------
    sea_level : pd.Series with DatetimeIndex
        Sea level observations (metres).
    temperature : pd.Series with DatetimeIndex
        Global mean surface temperature (°C).
    gmsl_sigma : pd.Series or None
        Observation uncertainties for WLS weighting.
    saod : pd.Series or None
        Stratospheric aerosol optical depth.
    order : int
        Polynomial order (1, 2, or 3).
    n_lags : int
        Number of leads/lags for ΔT (and ΔSAOD).

    Returns
    -------
    dict with keys:
        'X'          : np.ndarray (n_valid, n_cols+1)  — design matrix with constant
        'H'          : np.ndarray (n_valid,)  — sea level observations
        'sigma'      : np.ndarray or None  — observation uncertainties
        'time'       : np.ndarray (n_valid,)  — decimal year
        'n_phys'     : int  — number of physical parameters (order + 1)
        'idx_saod'   : int or None  — column index of ∫SAOD
        'n_lags'     : int
        'col_names'  : list of str  — column labels
        'valid_mask' : np.ndarray (bool)  — which rows survived NaN dropping
    """
    if order not in (1, 2, 3):
        raise ValueError(f"order must be 1, 2, or 3, got {order}")

    # ---- 1. Align series on common datetime index ----
    sl_ms = _to_month_start(sea_level)
    temp_ms = _to_month_start(temperature)
    saod_ms = _to_month_start(saod) if saod is not None else None

    common = sl_ms.index.intersection(temp_ms.index)
    if saod_ms is not None:
        common = common.intersection(saod_ms.index)
    common = common.sort_values()

    H = sl_ms.loc[common].values.astype(np.float64)
    T = temp_ms.loc[common].values.astype(np.float64)
    S = saod_ms.loc[common].values.astype(np.float64) if saod_ms is not None else None

    has_sigma = gmsl_sigma is not None
    if has_sigma:
        sig_ms = _to_month_start(gmsl_sigma)
        sigma = sig_ms.reindex(common).values.astype(np.float64)
        nan_sig = np.isnan(sigma)
        if nan_sig.any():
            sigma[nan_sig] = np.nanmedian(sigma)
    else:
        sigma = None

    # ---- 2. Decimal-year time vector ----
    time_years = np.array([
        t.year + (t.month - 1) / 12 + (t.day - 1) / 365.25
        for t in common
    ])
    n = len(H)
    dt_steps = np.diff(time_years)  # per-step dt for accurate integration

    # ---- 3. Trapezoidal integrals ∫Tᵏ ----
    integrals = []
    col_names = []
    for k in range(order, 0, -1):
        Tk = T ** k
        integral_Tk = np.zeros(n)
        for i in range(1, n):
            integral_Tk[i] = integral_Tk[i - 1] + 0.5 * (Tk[i] + Tk[i - 1]) * dt_steps[i - 1]
        integrals.append(integral_Tk)
        col_names.append(f"∫T^{k}")

    # ---- 4. Time trend ----
    time_trend = time_years - time_years[0]
    cols = list(integrals)
    cols.append(time_trend)
    col_names.append("trend")
    n_phys = len(cols)

    # ---- 5. Optional ∫SAOD ----
    if S is not None:
        integral_S = np.zeros(n)
        for i in range(1, n):
            integral_S[i] = integral_S[i - 1] + 0.5 * (S[i] + S[i - 1]) * dt_steps[i - 1]
        cols.append(integral_S)
        col_names.append("∫SAOD")
        idx_saod = len(cols) - 1
    else:
        idx_saod = None

    # ---- 6. ΔT leads/lags ----
    delta_T = np.diff(T, prepend=T[0])
    for lag in range(-n_lags, n_lags + 1):
        if lag < 0:
            shifted = np.concatenate([np.full(-lag, np.nan), delta_T[:lag]])
        elif lag > 0:
            shifted = np.concatenate([delta_T[lag:], np.full(lag, np.nan)])
        else:
            shifted = delta_T.copy()
        cols.append(shifted)
        col_names.append(f"ΔT_lag{lag}")

    # ---- 7. ΔSAOD leads/lags ----
    if S is not None:
        delta_S = np.diff(S, prepend=S[0])
        for lag in range(-n_lags, n_lags + 1):
            if lag < 0:
                shifted = np.concatenate([np.full(-lag, np.nan), delta_S[:lag]])
            elif lag > 0:
                shifted = np.concatenate([delta_S[lag:], np.full(lag, np.nan)])
            else:
                shifted = delta_S.copy()
            cols.append(shifted)
            col_names.append(f"ΔSAOD_lag{lag}")

    X_raw = np.column_stack(cols)

    # ---- 8. Drop NaN rows ----
    valid = ~np.any(np.isnan(X_raw), axis=1) & ~np.isnan(H)
    if has_sigma:
        valid = valid & ~np.isnan(sigma)

    X_v = X_raw[valid]
    H_v = H[valid]
    time_v = time_years[valid]
    sigma_v = sigma[valid] if has_sigma else None

    # Add constant (intercept) as last column
    X_v = np.column_stack([X_v, np.ones(X_v.shape[0])])
    col_names.append("const")

    return {
        'X': X_v,
        'H': H_v,
        'sigma': sigma_v,
        'time': time_v,
        'n_phys': n_phys,
        'idx_saod': idx_saod,
        'n_lags': n_lags,
        'col_names': col_names,
        'valid_mask': valid,
    }


def scale_design_matrix(dm_dict):
    """Center and scale design matrix columns for MCMC numerical stability.

    The constant column is NOT scaled. Physical and nuisance columns are
    standardized to zero mean, unit variance.

    Returns
    -------
    scaled_dict : dict — same structure as dm_dict but with scaled X
    scaler : dict with 'mean' and 'std' arrays for back-transformation
    """
    X = dm_dict['X'].copy()
    n_cols = X.shape[1]
    col_mean = np.zeros(n_cols)
    col_std = np.ones(n_cols)

    for j in range(n_cols - 1):  # skip last column (constant)
        col_mean[j] = X[:, j].mean()
        col_std[j] = X[:, j].std()
        if col_std[j] > 0:
            X[:, j] = (X[:, j] - col_mean[j]) / col_std[j]

    scaled_dict = dict(dm_dict)
    scaled_dict['X'] = X

    scaler = {'mean': col_mean, 'std': col_std}
    return scaled_dict, scaler


def unscale_coefficients(beta_scaled, scaler):
    """Transform coefficients from scaled to original (physical) space.

    Parameters
    ----------
    beta_scaled : np.ndarray (n_cols,) or (n_samples, n_cols)
    scaler : dict from scale_design_matrix

    Returns
    -------
    beta_original : same shape as beta_scaled
    """
    mean = scaler['mean']
    std = scaler['std']
    single = beta_scaled.ndim == 1
    if single:
        beta_scaled = beta_scaled[np.newaxis, :]

    beta_orig = beta_scaled.copy()
    n_cols = beta_scaled.shape[1]

    # For each non-constant column: β_orig_j = β_scaled_j / std_j
    # Constant absorbs the centering: β_orig_const += Σ (-mean_j/std_j) * β_scaled_j
    for j in range(n_cols - 1):
        if std[j] > 0:
            beta_orig[:, j] = beta_scaled[:, j] / std[j]
            beta_orig[:, -1] -= beta_scaled[:, j] * mean[j] / std[j]

    if single:
        return beta_orig[0]
    return beta_orig


# =============================================================================
# CONVERGENCE DIAGNOSTICS
# =============================================================================

def check_convergence(trace, quiet=False):
    """Compute and optionally print convergence diagnostics for an arviz trace.

    Parameters
    ----------
    trace : az.InferenceData
        Must have a posterior group.
    quiet : bool
        If True, suppress printing and only return the dict.

    Returns
    -------
    dict with keys:
        'rhat' : dict of {var_name: max R-hat across dimensions}
        'ess_bulk' : dict of {var_name: min bulk ESS}
        'ess_tail' : dict of {var_name: min tail ESS}
        'converged' : bool — True if all R-hat < 1.05 and all ESS > 100
        'warnings' : list of str
    """
    if not hasattr(trace, 'posterior') or len(trace.posterior.data_vars) == 0:
        return {'rhat': {}, 'ess_bulk': {}, 'ess_tail': {},
                'converged': True, 'warnings': ['Empty trace — no diagnostics']}

    rhat_ds = az.rhat(trace)
    ess_bulk_ds = az.ess(trace, method='bulk')
    ess_tail_ds = az.ess(trace, method='tail')

    rhat = {}
    ess_bulk = {}
    ess_tail = {}
    warn = []

    for var in trace.posterior.data_vars:
        rh = float(rhat_ds[var].max())
        eb = float(ess_bulk_ds[var].min())
        et = float(ess_tail_ds[var].min())
        rhat[var] = rh
        ess_bulk[var] = eb
        ess_tail[var] = et

        if rh > 1.05:
            warn.append(f"  {var}: R-hat = {rh:.3f} > 1.05 — chains have NOT converged")
        if eb < 100:
            warn.append(f"  {var}: bulk ESS = {eb:.0f} < 100 — increase samples or walkers")
        if et < 100:
            warn.append(f"  {var}: tail ESS = {et:.0f} < 100 — tail estimates unreliable")

    converged = all(v <= 1.05 for v in rhat.values()) and \
                all(v >= 100 for v in ess_bulk.values())

    if not quiet:
        print("  Convergence diagnostics:")
        for var in trace.posterior.data_vars:
            print(f"    {var}: R-hat={rhat[var]:.3f}  "
                  f"ESS_bulk={ess_bulk[var]:.0f}  ESS_tail={ess_tail[var]:.0f}")
        if warn:
            print("  WARNINGS:")
            for w in warn:
                print(w)
        else:
            print("  All diagnostics OK (R-hat < 1.05, ESS > 100)")

    return {'rhat': rhat, 'ess_bulk': ess_bulk, 'ess_tail': ess_tail,
            'converged': converged, 'warnings': warn}


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class BayesianDOLSResult:
    """Static Bayesian DOLS (analogous to DOLSResult)."""
    trace: az.InferenceData
    physical_coefficients: np.ndarray    # posterior mean [dα/dT, α₀, trend]
    physical_covariance: np.ndarray      # posterior covariance
    physical_hdi_94: np.ndarray          # (n_phys, 2) — 94% HDI
    r2: float
    loo: Optional[object] = None         # LOO-CV if computed
    design_matrix: Optional[dict] = None
    order: int = 2
    n_samples_posterior: int = 0
    sampler_diagnostics: Optional[dict] = None


@dataclass
class BayesianDLMResult:
    """Dynamic Linear Model with time-varying coefficients."""
    trace: az.InferenceData
    time: np.ndarray
    coefficients_mean: np.ndarray        # (n_time, n_phys)
    coefficients_hdi: np.ndarray         # (n_time, n_phys, 2)
    Q_posterior: Optional[np.ndarray] = None  # evolution noise posterior
    Q_fixed: Optional[float] = None
    loo: Optional[object] = None
    design_matrix: Optional[dict] = None
    sampler_diagnostics: Optional[dict] = None


@dataclass
class HierarchicalDOLSResult:
    """Hierarchical model across GMSL datasets."""
    trace: az.InferenceData
    population_mean: np.ndarray          # [dα/dT, α₀, trend]
    population_sd: np.ndarray
    dataset_coefficients: Dict[str, np.ndarray] = field(default_factory=dict)
    dataset_hdi: Dict[str, np.ndarray] = field(default_factory=dict)
    shrinkage_factors: Dict[str, float] = field(default_factory=dict)
    loo: Optional[object] = None
    sampler_diagnostics: Optional[dict] = None


@dataclass
class BayesianRateResult:
    """Bayesian rate-space calibration result.

    Fits rate(T) = dα/dT × T² + α₀ × T + trend directly against
    observed kinematic rates with correlated Gaussian errors and
    half-normal priors enforcing non-negative coefficients.

    Compatible with ``project_gmsl_ensemble()`` via
    ``physical_coefficients`` and ``physical_covariance``.
    """
    trace: az.InferenceData
    physical_coefficients: np.ndarray    # posterior mean [dα/dT, α₀, trend]
    physical_covariance: np.ndarray      # posterior covariance (3×3)
    physical_hdi_94: np.ndarray          # (3, 2) — 94% HDI
    posterior_samples: np.ndarray        # (n_samples, 3) — full chain
    r2: float                            # R² against observed rates
    corr_matrix: np.ndarray              # (n_obs, n_obs) correlation matrix
    corr_length: float                   # span_years used for correlation
    n_obs: int                           # number of rate observations
    n_eff: float                         # effective sample size
    time: np.ndarray                     # observation times (decimal year)
    temperature: np.ndarray              # matched temperature values
    rate_obs: np.ndarray                 # observed rates (mm/yr or m/yr)
    rate_se: np.ndarray                  # rate standard errors
    residuals: np.ndarray                # posterior-mean residuals
    order: int = 2
    sampler_diagnostics: Optional[dict] = None
    design_info: Optional[dict] = None


# =============================================================================
# MODEL 1: BAYESIAN STATIC DOLS
# =============================================================================

def _static_log_prior(theta, n_phys, has_saod, sigma_known):
    """Log prior for static DOLS model.

    Priors:
      Physical params (dα/dT, α₀, trend): N(0, 10²)
      Lead/lag nuisance: N(0, 1²)
      SAOD coefficient: N(0, 10²)
      Constant: N(0, 100²)
      sigma_obs (if estimated): HalfNormal(5) → log-uniform on log(sigma)
    """
    lp = 0.0
    idx = 0

    # Physical params
    for i in range(n_phys):
        lp += -0.5 * (theta[idx] / 10.0) ** 2
        idx += 1

    # SAOD
    if has_saod:
        lp += -0.5 * (theta[idx] / 10.0) ** 2
        idx += 1

    # Nuisance leads/lags
    n_nuisance = len(theta) - idx - 1 - (0 if sigma_known else 1)
    for i in range(n_nuisance):
        lp += -0.5 * (theta[idx] / 1.0) ** 2
        idx += 1

    # Constant
    lp += -0.5 * (theta[idx] / 100.0) ** 2
    idx += 1

    # sigma_obs (if estimated)
    if not sigma_known:
        log_sigma = theta[idx]
        sigma_obs = np.exp(log_sigma)
        if sigma_obs <= 0:
            return -np.inf
        # HalfNormal(5) on sigma → log-transform Jacobian
        lp += -0.5 * (sigma_obs / 5.0) ** 2 + log_sigma
        idx += 1

    return lp


def _static_log_likelihood(theta, X, H, sigma, n_phys, has_saod, sigma_known):
    """Log likelihood for static DOLS: H ~ N(X @ beta, sigma²)."""
    n_cols = X.shape[1]

    if sigma_known:
        beta = theta[:n_cols]
        sig = sigma
    else:
        beta = theta[:n_cols]
        sig = np.exp(theta[n_cols]) * np.ones(len(H))

    mu = X @ beta
    resid = H - mu
    return -0.5 * np.sum((resid / sig) ** 2 + 2 * np.log(sig))


def _static_log_prob(theta, X, H, sigma, n_phys, has_saod, sigma_known):
    lp = _static_log_prior(theta, n_phys, has_saod, sigma_known)
    if not np.isfinite(lp):
        return -np.inf
    ll = _static_log_likelihood(theta, X, H, sigma, n_phys, has_saod, sigma_known)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def fit_bayesian_dols(sea_level, temperature, gmsl_sigma=None, saod=None,
                       order=2, n_lags=2, n_samples=2000, n_walkers=32,
                       n_burnin=500, thin=1, progress=True):
    """Fit static Bayesian DOLS using emcee.

    Parameters
    ----------
    sea_level, temperature, gmsl_sigma, saod, order, n_lags
        Same as calibrate_dols().
    n_samples : int
        Number of posterior samples per walker (after burn-in).
    n_walkers : int
        Number of ensemble walkers (must be even, >= 2 * ndim).
    n_burnin : int
        Burn-in steps to discard.
    thin : int
        Thinning factor.
    progress : bool
        Show progress bar.

    Returns
    -------
    BayesianDOLSResult
    """
    # Build and scale design matrix
    dm = build_dols_design_matrix(sea_level, temperature, gmsl_sigma, saod,
                                   order, n_lags)
    dm_scaled, scaler = scale_design_matrix(dm)

    X = dm_scaled['X']
    H = dm_scaled['H']
    sigma = dm_scaled['sigma']
    n_phys = dm_scaled['n_phys']
    has_saod = dm_scaled['idx_saod'] is not None

    sigma_known = sigma is not None
    n_cols = X.shape[1]
    ndim = n_cols + (0 if sigma_known else 1)

    # Ensure enough walkers
    n_walkers = max(n_walkers, 2 * ndim + 2)
    if n_walkers % 2 != 0:
        n_walkers += 1

    # Initialize near OLS solution
    beta_ols = np.linalg.lstsq(X, H, rcond=None)[0]
    p0 = np.zeros((n_walkers, ndim))
    for i in range(n_walkers):
        p0[i, :n_cols] = beta_ols + 0.01 * np.random.randn(n_cols)
        if not sigma_known:
            resid = H - X @ beta_ols
            p0[i, n_cols] = np.log(np.std(resid)) + 0.1 * np.random.randn()

    # Run sampler
    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, _static_log_prob,
        args=(X, H, sigma, n_phys, has_saod, sigma_known)
    )

    # Burn-in
    state = sampler.run_mcmc(p0, n_burnin, progress=progress)
    sampler.reset()

    # Production
    sampler.run_mcmc(state, n_samples, thin_by=thin, progress=progress)
    flat_samples = sampler.get_chain(flat=True)  # (n_walkers * n_samples, ndim)

    # Unscale to physical space
    beta_samples = flat_samples[:, :n_cols]
    beta_phys = unscale_coefficients(beta_samples, scaler)

    # Extract physical coefficient posteriors
    phys_samples = beta_phys[:, :n_phys]  # (n_samples_total, n_phys)
    phys_mean = phys_samples.mean(axis=0)
    phys_cov = np.cov(phys_samples, rowvar=False)
    phys_hdi = np.array([
        az.hdi(phys_samples[:, k], hdi_prob=0.94) for k in range(n_phys)
    ])

    # R² (using posterior mean)
    beta_mean = beta_phys.mean(axis=0)
    mu_mean = dm['X'] @ beta_mean  # use unscaled X
    ss_res = np.sum((dm['H'] - mu_mean) ** 2)
    ss_tot = np.sum((dm['H'] - dm['H'].mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    # Build arviz InferenceData
    # Use get_chain(flat=False) to preserve walker structure for proper diagnostics
    chain_samples = sampler.get_chain(flat=False)  # (n_steps, n_walkers, ndim)
    n_chains_arviz = min(4, n_walkers)

    # Physical samples: unscale each walker independently
    phys_reshaped = np.zeros((n_chains_arviz, chain_samples.shape[0], n_phys))
    for c in range(n_chains_arviz):
        chain_beta = chain_samples[:, c, :n_cols]  # (n_steps, n_cols)
        chain_phys_unscaled = unscale_coefficients(chain_beta, scaler)
        phys_reshaped[c] = chain_phys_unscaled[:, :n_phys]

    # Name the physical parameters
    phys_names = dm['col_names'][:n_phys]
    var_dict = {name: phys_reshaped[:, :, k]
                for k, name in enumerate(phys_names)}

    trace = az.from_dict(var_dict)

    # Convergence diagnostics
    conv = check_convergence(trace, quiet=(not progress))

    # Sampler diagnostics
    diag = {
        'acceptance_fraction': sampler.acceptance_fraction.mean(),
        'n_walkers': n_walkers,
        'n_samples': n_samples,
        'n_burnin': n_burnin,
        'convergence': conv,
    }

    return BayesianDOLSResult(
        trace=trace,
        physical_coefficients=phys_mean,
        physical_covariance=phys_cov,
        physical_hdi_94=phys_hdi,
        r2=r2,
        design_matrix=dm,
        order=order,
        n_samples_posterior=flat_samples.shape[0],
        sampler_diagnostics=diag,
    )


# =============================================================================
# MODEL 2: BAYESIAN DLM  (Kalman filter + emcee for hyperparameters)
# =============================================================================

def _kalman_filter(Y, X_phys, X_nuisance, Q_diag, sigma_obs, beta_nuisance):
    """Kalman filter for TVP regression with known Q and nuisance params.

    State: θ(t) ∈ R^{n_phys}  (time-varying physical coefficients)
    Observation: Y(t) = X_phys(t,:) @ θ(t) + X_nuisance(t,:) @ β_nuisance + ε(t)
    Transition: θ(t) = θ(t-1) + η(t),  η ~ N(0, Q)
    Observation noise: ε(t) ~ N(0, R(t))  where R(t) = σ(t)²

    Parameters
    ----------
    Y : (n,)  — observations (adjusted for nuisance contribution)
    X_phys : (n, n_phys)  — time-varying regressors for physical params
    X_nuisance : (n, n_nuisance)  — regressors for static nuisance params
    Q_diag : (n_phys,)  — diagonal of transition covariance
    sigma_obs : (n,)  — observation standard deviations
    beta_nuisance : (n_nuisance,)  — static nuisance parameter values

    Returns
    -------
    log_lik : float  — log marginal likelihood
    theta_filt : (n, n_phys)  — filtered state means
    P_filt : (n, n_phys, n_phys)  — filtered state covariances
    """
    n, n_phys = X_phys.shape
    Q = np.diag(Q_diag)

    # Adjust observations for nuisance contribution
    Y_adj = Y - X_nuisance @ beta_nuisance

    # Initialize with diffuse prior
    theta = np.zeros(n_phys)
    P = np.eye(n_phys) * 100.0  # diffuse

    theta_filt = np.zeros((n, n_phys))
    P_filt = np.zeros((n, n_phys, n_phys))
    log_lik = 0.0

    for t in range(n):
        # Predict
        theta_pred = theta  # random walk
        P_pred = P + Q

        # Update
        H_t = X_phys[t, :]  # (n_phys,)  observation vector
        R_t = sigma_obs[t] ** 2
        y_pred = H_t @ theta_pred
        v_t = Y_adj[t] - y_pred  # innovation
        S_t = H_t @ P_pred @ H_t + R_t  # innovation variance

        if S_t <= 0:
            return -np.inf, theta_filt, P_filt

        K_t = P_pred @ H_t / S_t  # Kalman gain
        theta = theta_pred + K_t * v_t
        # Joseph form: guarantees positive semi-definiteness
        IKH = np.eye(n_phys) - np.outer(K_t, H_t)
        P = IKH @ P_pred @ IKH.T + R_t * np.outer(K_t, K_t)

        theta_filt[t] = theta
        P_filt[t] = P

        # Log-likelihood contribution
        log_lik += -0.5 * (np.log(2 * np.pi * S_t) + v_t ** 2 / S_t)

    return log_lik, theta_filt, P_filt


def _kalman_smoother(theta_filt, P_filt, Q_diag):
    """Rauch-Tung-Striebel smoother for the random walk model.

    Returns
    -------
    theta_smooth : (n, n_phys)
    P_smooth : (n, n_phys, n_phys)
    """
    n, n_phys = theta_filt.shape
    Q = np.diag(Q_diag)

    theta_smooth = theta_filt.copy()
    P_smooth = P_filt.copy()

    for t in range(n - 2, -1, -1):
        P_pred = P_filt[t] + Q
        try:
            J_t = P_filt[t] @ la.inv(P_pred)
        except la.LinAlgError:
            J_t = P_filt[t] @ la.pinv(P_pred)

        theta_smooth[t] = theta_filt[t] + J_t @ (theta_smooth[t + 1] - theta_filt[t])
        P_smooth[t] = P_filt[t] + J_t @ (P_smooth[t + 1] - P_pred) @ J_t.T

    return theta_smooth, P_smooth


def _dlm_log_prob(theta_hyper, Y, X_phys, X_nuisance, sigma_obs,
                   n_phys, n_nuisance):
    """Log posterior for DLM hyperparameters.

    theta_hyper = [log_Q_1, ..., log_Q_{n_phys}, beta_nuisance_1, ..., const]

    We sample log(Q) to enforce positivity.
    """
    # Unpack
    log_Q = theta_hyper[:n_phys]
    Q_diag = np.exp(log_Q)
    beta_nuisance = theta_hyper[n_phys:]

    # Check bounds
    if np.any(Q_diag > 100) or np.any(Q_diag < 1e-20):
        return -np.inf

    # Prior on log(Q): broad Normal
    lp = -0.5 * np.sum((log_Q + 5) ** 2 / 5.0 ** 2)  # prior centered at Q≈0.007

    # Prior on nuisance params: N(0, 1)
    n_nuis = len(beta_nuisance) - 1  # last is constant
    if n_nuis > 0:
        lp += -0.5 * np.sum(beta_nuisance[:n_nuis] ** 2)
    # Constant: N(0, 100)
    lp += -0.5 * (beta_nuisance[-1] / 100.0) ** 2

    if not np.isfinite(lp):
        return -np.inf

    # Kalman filter log-likelihood
    ll, _, _ = _kalman_filter(Y, X_phys, X_nuisance, Q_diag, sigma_obs,
                               beta_nuisance)
    if not np.isfinite(ll):
        return -np.inf

    return lp + ll


def fit_bayesian_dlm(sea_level, temperature, gmsl_sigma=None, saod=None,
                      order=2, n_lags=2, n_samples=2000, n_walkers=32,
                      n_burnin=1000, Q_fixed=None, thin=1, progress=True):
    """Fit Bayesian DLM with time-varying physical coefficients.

    The physical coefficients [dα/dT, α₀, trend] follow independent
    Gaussian random walks. The evolution noise Q is either estimated
    (fully Bayesian) or fixed.

    Parameters
    ----------
    Q_fixed : float or array-like or None
        If None, estimate Q from data (fully Bayesian).
        If scalar, use same Q for all physical params.
        If array, use Q_fixed[k] for each physical param.

    Returns
    -------
    BayesianDLMResult
    """
    # Build design matrix (no scaling — Kalman filter handles it)
    dm = build_dols_design_matrix(sea_level, temperature, gmsl_sigma, saod,
                                   order, n_lags)
    X = dm['X']
    H = dm['H']
    time = dm['time']
    n_phys = dm['n_phys']
    sigma = dm['sigma']

    n_obs, n_cols = X.shape

    # Use OLS residual std as default sigma if not provided
    if sigma is None:
        beta_ols = np.linalg.lstsq(X, H, rcond=None)[0]
        resid_std = np.std(H - X @ beta_ols)
        sigma = np.full(n_obs, resid_std)

    # Split X into physical and nuisance columns
    X_phys = X[:, :n_phys]
    X_nuisance = X[:, n_phys:]  # includes SAOD, leads/lags, constant
    n_nuisance = X_nuisance.shape[1]

    if Q_fixed is not None:
        # Fixed Q — just run Kalman filter + smoother, no MCMC for Q
        if np.isscalar(Q_fixed):
            Q_diag = np.full(n_phys, Q_fixed)
        else:
            Q_diag = np.asarray(Q_fixed)

        # Still need to estimate nuisance params — use OLS on residuals
        beta_ols = np.linalg.lstsq(X, H, rcond=None)[0]
        beta_nuisance = beta_ols[n_phys:]

        _, theta_filt, P_filt = _kalman_filter(
            H, X_phys, X_nuisance, Q_diag, sigma, beta_nuisance
        )
        theta_smooth, P_smooth = _kalman_smoother(theta_filt, P_filt, Q_diag)

        # HDI from smoother covariance
        coeff_hdi = np.zeros((n_obs, n_phys, 2))
        for t in range(n_obs):
            for k in range(n_phys):
                se = np.sqrt(P_smooth[t, k, k])
                coeff_hdi[t, k, 0] = theta_smooth[t, k] - 1.88 * se  # ~94% for normal
                coeff_hdi[t, k, 1] = theta_smooth[t, k] + 1.88 * se

        # Dummy trace
        trace = az.from_dict({})

        return BayesianDLMResult(
            trace=trace,
            time=time,
            coefficients_mean=theta_smooth,
            coefficients_hdi=coeff_hdi,
            Q_posterior=None,
            Q_fixed=Q_fixed,
            design_matrix=dm,
            sampler_diagnostics={'method': 'fixed_Q_kalman'},
        )

    # Fully Bayesian: sample Q and nuisance params with emcee
    ndim = n_phys + n_nuisance  # log_Q_1..n_phys, beta_nuisance
    n_walkers = max(n_walkers, 2 * ndim + 2)
    if n_walkers % 2 != 0:
        n_walkers += 1

    # Initialize
    beta_ols = np.linalg.lstsq(X, H, rcond=None)[0]
    p0 = np.zeros((n_walkers, ndim))
    for i in range(n_walkers):
        # log(Q) initialized near small values
        p0[i, :n_phys] = np.log(1e-4) + 0.5 * np.random.randn(n_phys)
        # Nuisance params near OLS
        p0[i, n_phys:] = beta_ols[n_phys:] + 0.01 * np.random.randn(n_nuisance)

    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, _dlm_log_prob,
        args=(H, X_phys, X_nuisance, sigma, n_phys, n_nuisance)
    )

    # Burn-in
    state = sampler.run_mcmc(p0, n_burnin, progress=progress)
    sampler.reset()

    # Production
    sampler.run_mcmc(state, n_samples, thin_by=thin, progress=progress)
    flat_samples = sampler.get_chain(flat=True)

    # Extract Q posteriors
    log_Q_samples = flat_samples[:, :n_phys]
    Q_samples = np.exp(log_Q_samples)
    Q_median = np.median(Q_samples, axis=0)

    # For each posterior sample, run Kalman smoother to get coefficient trajectories
    # (expensive — subsample for speed)
    n_sub = min(200, flat_samples.shape[0])
    idx_sub = np.random.choice(flat_samples.shape[0], n_sub, replace=False)

    all_theta = np.zeros((n_sub, n_obs, n_phys))
    for ii, idx in enumerate(idx_sub):
        Q_diag = np.exp(flat_samples[idx, :n_phys])
        beta_nuis = flat_samples[idx, n_phys:]
        _, theta_f, P_f = _kalman_filter(
            H, X_phys, X_nuisance, Q_diag, sigma, beta_nuis
        )
        theta_s, _ = _kalman_smoother(theta_f, P_f, Q_diag)
        all_theta[ii] = theta_s

    # Posterior summary
    coeff_mean = all_theta.mean(axis=0)  # (n_obs, n_phys)
    coeff_hdi = np.zeros((n_obs, n_phys, 2))
    for t in range(n_obs):
        for k in range(n_phys):
            coeff_hdi[t, k] = az.hdi(all_theta[:, t, k], hdi_prob=0.94)

    # Build arviz trace for Q — preserve walker structure
    chain_samples_dlm = sampler.get_chain(flat=False)  # (n_steps, n_walkers, ndim)
    n_chains_arviz = min(4, n_walkers)
    Q_chain = np.exp(chain_samples_dlm[:, :n_chains_arviz, :n_phys])  # (n_steps, n_chains, n_phys)
    Q_reshaped = Q_chain.transpose(1, 0, 2)  # (n_chains, n_steps, n_phys)
    phys_names = dm['col_names'][:n_phys]
    var_dict = {f"Q_{name}": Q_reshaped[:, :, k]
                for k, name in enumerate(phys_names)}
    trace = az.from_dict(var_dict)

    # Convergence diagnostics
    conv = check_convergence(trace, quiet=(not progress))

    diag = {
        'acceptance_fraction': sampler.acceptance_fraction.mean(),
        'n_walkers': n_walkers,
        'n_samples': n_samples,
        'n_burnin': n_burnin,
        'n_smoother_subsamples': n_sub,
        'convergence': conv,
    }

    return BayesianDLMResult(
        trace=trace,
        time=time,
        coefficients_mean=coeff_mean,
        coefficients_hdi=coeff_hdi,
        Q_posterior=Q_median,
        design_matrix=dm,
        sampler_diagnostics=diag,
    )


# =============================================================================
# MODEL 3: HIERARCHICAL MULTI-DATASET
# =============================================================================

def _hierarchical_log_prob(theta, datasets_info, n_phys):
    """Log posterior for hierarchical model.

    theta layout:
      [0 : n_phys]                    — population means μ_k
      [n_phys : 2*n_phys]             — log(σ_pop_k)
      [2*n_phys : ...]                — per-dataset: z_ik (non-centered) + nuisance + const
                                         + log_sigma_obs (if sigma not provided)

    For dataset i with n_nuisance_i nuisance params:
      z_i = theta[offset : offset + n_phys]
      nuis_i = theta[offset + n_phys : offset + n_phys + n_nuisance_i]
      [log_sigma_obs_i] = theta[offset + n_phys + n_nuisance_i] (if needed)
      θ_i = μ + σ_pop * z_i   (non-centered)
    """
    # Unpack population params
    mu_pop = theta[:n_phys]
    log_sigma_pop = theta[n_phys:2 * n_phys]
    sigma_pop = np.exp(log_sigma_pop)

    # Prior on population mean: N(0, 10)
    lp = -0.5 * np.sum((mu_pop / 10.0) ** 2)
    # Prior on log(sigma_pop): broad
    lp += -0.5 * np.sum((log_sigma_pop + 1) ** 2 / 3.0 ** 2)

    if not np.isfinite(lp):
        return -np.inf

    offset = 2 * n_phys

    for dinfo in datasets_info:
        X = dinfo['X']
        H = dinfo['H']
        sigma = dinfo['sigma']
        n_nuis = dinfo['n_nuisance']
        has_sigma = sigma is not None
        n_per_dataset = n_phys + n_nuis + (0 if has_sigma else 1)

        # Extract dataset-specific params
        z_i = theta[offset:offset + n_phys]
        nuis_i = theta[offset + n_phys:offset + n_phys + n_nuis]

        # Non-centered parameterization
        theta_i_phys = mu_pop + sigma_pop * z_i

        # Prior on z: N(0, 1) — standard normal
        lp += -0.5 * np.sum(z_i ** 2)
        # Prior on nuisance (all but last = leads/lags: N(0,1); last = const: N(0,100))
        if n_nuis > 1:
            lp += -0.5 * np.sum(nuis_i[:-1] ** 2)
        lp += -0.5 * (nuis_i[-1] / 100.0) ** 2

        # Likelihood for this dataset
        beta_full = np.concatenate([theta_i_phys, nuis_i])
        mu = X @ beta_full
        resid = H - mu

        if has_sigma:
            ll = -0.5 * np.sum((resid / sigma) ** 2 + 2 * np.log(sigma))
        else:
            # Sample sigma_obs for this dataset (HalfNormal(5) prior)
            log_sigma_obs = theta[offset + n_phys + n_nuis]
            sigma_obs = np.exp(log_sigma_obs)
            if sigma_obs <= 0 or sigma_obs > 1e3:
                return -np.inf
            # HalfNormal(5) on sigma → log-transform Jacobian
            lp += -0.5 * (sigma_obs / 5.0) ** 2 + log_sigma_obs
            ll = -0.5 * np.sum((resid / sigma_obs) ** 2) \
                 - len(resid) * np.log(sigma_obs)

        if not np.isfinite(ll):
            return -np.inf
        lp += ll

        offset += n_per_dataset

    return lp


def fit_hierarchical_dols(datasets, temperature, order=2, n_lags=2,
                           n_samples=2000, n_walkers=None,
                           n_burnin=1000, thin=1, progress=True):
    """Fit hierarchical DOLS across multiple GMSL datasets.

    Parameters
    ----------
    datasets : dict of {name: pd.Series} or {name: (pd.Series, pd.Series)}
        GMSL datasets. If tuple, second element is gmsl_sigma.
    temperature : pd.Series with DatetimeIndex
        Common GMST series.
    order, n_lags : int
        DOLS order and lead/lag specification.

    Returns
    -------
    HierarchicalDOLSResult
    """
    # Build design matrices for each dataset
    datasets_info = []
    dataset_names = []

    for name, data in datasets.items():
        if isinstance(data, tuple):
            sl, sl_sigma = data
        else:
            sl = data
            sl_sigma = None

        try:
            dm = build_dols_design_matrix(sl, temperature, sl_sigma, order=order,
                                           n_lags=n_lags)
        except Exception as e:
            print(f"  Skipping {name}: {e}")
            continue

        n_phys = dm['n_phys']
        n_cols = dm['X'].shape[1]
        n_nuisance = n_cols - n_phys

        datasets_info.append({
            'name': name,
            'X': dm['X'],
            'H': dm['H'],
            'sigma': dm['sigma'],
            'n_nuisance': n_nuisance,
            'dm': dm,
        })
        dataset_names.append(name)

    if len(datasets_info) < 2:
        raise ValueError("Need at least 2 datasets for hierarchical model")

    n_datasets = len(datasets_info)

    # Parameter count — add 1 for log_sigma_obs for datasets without sigma
    n_pop = 2 * n_phys  # mu + log_sigma
    n_per_dataset = [n_phys + d['n_nuisance'] + (0 if d['sigma'] is not None else 1)
                     for d in datasets_info]
    ndim = n_pop + sum(n_per_dataset)

    if n_walkers is None:
        n_walkers = max(2 * ndim + 2, 64)
    if n_walkers % 2 != 0:
        n_walkers += 1

    # Compute OLS solutions for data-informed initialization
    ols_phys = []
    ols_nuis = []
    for dinfo in datasets_info:
        beta_ols = np.linalg.lstsq(dinfo['X'], dinfo['H'], rcond=None)[0]
        ols_phys.append(beta_ols[:n_phys])
        ols_nuis.append(beta_ols[n_phys:])
    ols_phys = np.array(ols_phys)
    mu_ols = ols_phys.mean(axis=0)
    std_ols = ols_phys.std(axis=0)
    std_ols = np.maximum(std_ols, 1e-6)

    # Initialize
    p0 = np.zeros((n_walkers, ndim))
    for i in range(n_walkers):
        # Population mean near OLS cross-dataset average
        p0[i, :n_phys] = mu_ols + 0.1 * std_ols * np.random.randn(n_phys)
        # log(sigma_pop) near observed cross-dataset spread
        p0[i, n_phys:2 * n_phys] = np.log(std_ols + 1e-8) + 0.1 * np.random.randn(n_phys)

        offset = n_pop
        for di, dinfo in enumerate(datasets_info):
            n_per = n_per_dataset[di]
            n_nuis = dinfo['n_nuisance']
            # z_i ≈ (ols_i - mu) / sigma — near the non-centered OLS deviation
            z_init = (ols_phys[di] - mu_ols) / std_ols
            p0[i, offset:offset + n_phys] = z_init + 0.05 * np.random.randn(n_phys)
            # Nuisance near OLS
            p0[i, offset + n_phys:offset + n_phys + n_nuis] = (
                ols_nuis[di] + 0.01 * np.random.randn(len(ols_nuis[di]))
            )
            # log_sigma_obs for datasets without known sigma
            if dinfo['sigma'] is None:
                resid_ols = dinfo['H'] - dinfo['X'] @ np.concatenate([ols_phys[di], ols_nuis[di]])
                p0[i, offset + n_phys + n_nuis] = (
                    np.log(np.std(resid_ols)) + 0.1 * np.random.randn()
                )
            offset += n_per

    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, _hierarchical_log_prob,
        args=(datasets_info, n_phys)
    )

    # Burn-in
    state = sampler.run_mcmc(p0, n_burnin, progress=progress)
    sampler.reset()

    # Production
    sampler.run_mcmc(state, n_samples, thin_by=thin, progress=progress)
    flat_samples = sampler.get_chain(flat=True)

    # Extract population posteriors
    mu_samples = flat_samples[:, :n_phys]
    log_sigma_samples = flat_samples[:, n_phys:2 * n_phys]
    sigma_pop_samples = np.exp(log_sigma_samples)

    pop_mean = mu_samples.mean(axis=0)
    pop_sd = sigma_pop_samples.mean(axis=0)

    # Extract per-dataset coefficients
    dataset_coeffs = {}
    dataset_hdi_dict = {}
    shrinkage = {}

    offset = 2 * n_phys
    for di, dinfo in enumerate(datasets_info):
        name = dinfo['name']
        z_samples = flat_samples[:, offset:offset + n_phys]
        # Reconstruct physical coefficients
        theta_i = mu_samples + sigma_pop_samples * z_samples
        dataset_coeffs[name] = theta_i.mean(axis=0)
        dataset_hdi_dict[name] = np.array([
            az.hdi(theta_i[:, k], hdi_prob=0.94) for k in range(n_phys)
        ])

        # Shrinkage: 1 - posterior_var / prior_var
        post_var = theta_i.var(axis=0)
        prior_var = sigma_pop_samples.var(axis=0) + sigma_pop_samples.mean(axis=0) ** 2
        prior_var = np.maximum(prior_var, 1e-10)
        shrinkage[name] = float(np.mean(1.0 - post_var / prior_var))

        offset += n_per_dataset[di]

    # Build arviz trace — preserve walker structure
    chain_samples_hier = sampler.get_chain(flat=False)  # (n_steps, n_walkers, ndim)
    n_chains_arviz = min(4, n_walkers)
    mu_chain = chain_samples_hier[:, :n_chains_arviz, :n_phys]  # (n_steps, n_chains, n_phys)
    mu_reshaped = mu_chain.transpose(1, 0, 2)  # (n_chains, n_steps, n_phys)
    sigma_chain = np.exp(chain_samples_hier[:, :n_chains_arviz, n_phys:2*n_phys])
    sigma_reshaped = sigma_chain.transpose(1, 0, 2)
    phys_names = datasets_info[0]['dm']['col_names'][:n_phys]
    var_dict = {f"μ_{name}": mu_reshaped[:, :, k]
                for k, name in enumerate(phys_names)}
    var_dict["σ_pop"] = sigma_reshaped
    trace = az.from_dict(var_dict)

    # Convergence diagnostics
    conv = check_convergence(trace, quiet=(not progress))

    diag = {
        'acceptance_fraction': sampler.acceptance_fraction.mean(),
        'n_walkers': n_walkers,
        'n_samples': n_samples,
        'n_burnin': n_burnin,
        'n_datasets': n_datasets,
        'convergence': conv,
        'dataset_names': dataset_names,
    }

    return HierarchicalDOLSResult(
        trace=trace,
        population_mean=pop_mean,
        population_sd=pop_sd,
        dataset_coefficients=dataset_coeffs,
        dataset_hdi=dataset_hdi_dict,
        shrinkage_factors=shrinkage,
        sampler_diagnostics=diag,
    )


# =============================================================================
# MODEL 4: BAYESIAN RATE-SPACE CALIBRATION
# =============================================================================


def build_rate_correlation_matrix(
    time: np.ndarray,
    span_years: float,
    kernel: str = 'tricube',
    nugget: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """Build temporal correlation matrix for kinematic rate estimates.

    Adjacent rate estimates from ``compute_kinematics()`` share data
    through overlapping kernel windows, inducing positive correlation.
    This function constructs the correlation matrix from the known
    kernel bandwidth so that downstream Bayesian inference properly
    accounts for the reduced effective sample size.

    The correlation between rates at t_i and t_j is computed
    numerically as the normalised inner product of their kernel
    weight vectors:

        ρ_ij = Σ_k w_i(k) w_j(k) / sqrt(Σ_k w_i(k)² · Σ_k w_j(k)²)

    where w_i(k) = K((t_k − t_i) / h) is the kernel weight that data
    point k receives in the local regression centred at t_i, and h is
    the bandwidth (``span_years``).

    This gives the exact overlap-induced correlation for
    kernel-weighted regression and guarantees positive semi-
    definiteness (it is a Gram matrix).

    An optional ``nugget`` adds a fraction of independent variance to
    each observation, modelling sources of rate variability not
    captured by the smooth kernel model (e.g. interannual climate
    noise, volcanic/ENSO transients).  The final correlation is:

        R_final = (1 − nugget) × R_kernel + nugget × I

    Parameters
    ----------
    time : np.ndarray
        Observation times in decimal years, shape (n,).
    span_years : float
        Bandwidth of the local polynomial regression used in
        ``compute_kinematics()``.
    kernel : str, default 'tricube'
        Kernel function: 'tricube', 'gaussian', or 'epanechnikov'.
        Must match the kernel used in ``compute_kinematics()``.
    nugget : float, default 0.0
        Fraction of independent (white-noise) variance to add.
        0.0 = pure kernel-overlap correlation; 1.0 = fully independent.
        Typical values: 0.05–0.20.

    Returns
    -------
    R : np.ndarray, shape (n, n)
        Symmetric positive-definite correlation matrix (ones on diagonal).
    n_eff : float
        Effective number of independent observations: ``n² / sum(R)``.
    """
    kernels = {
        'tricube': lambda u: np.where(
            np.abs(u) <= 1, (1 - np.abs(u)**3)**3, 0.0),
        'gaussian': lambda u: np.exp(-0.5 * u**2),
        'epanechnikov': lambda u: np.where(
            np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0),
    }
    if kernel not in kernels:
        raise ValueError(
            f"Unknown kernel '{kernel}'. Options: {list(kernels.keys())}")

    K = kernels[kernel]
    n = len(time)
    h = span_years

    # Build a dense evaluation grid spanning the full data range
    # (monthly resolution is sufficient to resolve the kernel shape)
    t_min, t_max = time.min(), time.max()
    t_grid = np.arange(t_min - h, t_max + h, 1.0 / 12)  # monthly
    n_grid = len(t_grid)

    # Weight matrix W: W[i, k] = K((t_grid[k] − time[i]) / h)
    # Build efficiently column-by-column to limit memory
    W = np.zeros((n, n_grid))
    for i in range(n):
        u = (t_grid - time[i]) / h
        W[i] = K(u)

    # Correlation = normalised Gram matrix  W W^T / (||w_i|| ||w_j||)
    norms = np.sqrt(np.sum(W**2, axis=1))  # (n,)
    norms = np.maximum(norms, 1e-30)       # guard against zero
    W_normed = W / norms[:, None]
    R = W_normed @ W_normed.T

    # Ensure exact symmetry and unit diagonal (numerical hygiene)
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)

    # Apply nugget: R_final = (1 − nugget) × R_kernel + nugget × I
    if nugget > 0:
        R = (1.0 - nugget) * R + nugget * np.eye(n)

    # Effective sample size: n² / sum(R)
    n_eff = n**2 / R.sum()

    return R, n_eff


def _rate_log_prior(theta, prior_scales):
    """Half-normal prior enforcing non-negative coefficients.

    For each coefficient θ_k ≥ 0:
        log p(θ_k) = -½ (θ_k / σ_k)²   (unnormalized)

    Parameters
    ----------
    theta : np.ndarray, shape (n_params,)
    prior_scales : np.ndarray, shape (n_params,)
        Scale parameters for the half-normal distributions.
    """
    if np.any(theta < 0):
        return -np.inf
    return -0.5 * np.sum((theta / prior_scales)**2)


def _rate_log_likelihood(theta, X, rate, Sigma_inv, log_det_Sigma):
    """Correlated Gaussian log-likelihood for rate-space model.

    log L = -½ [r' Σ⁻¹ r + log|Σ| + n log(2π)]

    where r = rate_obs - X @ theta.
    """
    resid = rate - X @ theta
    return -0.5 * (resid @ Sigma_inv @ resid + log_det_Sigma)


def _rate_log_prob(theta, X, rate, Sigma_inv, log_det_Sigma, prior_scales):
    """Log-posterior for rate-space model."""
    lp = _rate_log_prior(theta, prior_scales)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _rate_log_likelihood(theta, X, rate, Sigma_inv,
                                     log_det_Sigma)


def fit_bayesian_rate_model(
    rate: np.ndarray,
    rate_se: np.ndarray,
    temperature: np.ndarray,
    time: np.ndarray,
    span_years: float = 30.0,
    kernel: str = 'tricube',
    nugget: float = 0.0,
    order: int = 2,
    fit_year_start: Optional[float] = None,
    n_samples: int = 3000,
    n_walkers: int = 32,
    n_burnin: int = 1000,
    thin: int = 1,
    prior_scale_a: float = 20.0,
    prior_scale_b: float = 20.0,
    prior_scale_c: float = 20.0,
    progress: bool = True,
    seed: Optional[int] = None,
) -> BayesianRateResult:
    """Bayesian rate-space calibration with correlated errors.

    Fits a polynomial rate–temperature relationship directly against
    observed kinematic rates, handling the serial correlation induced
    by overlapping kernel windows in ``compute_kinematics()``.

    Model
    -----
    For ``order=2`` (quadratic):

        rate_i = dα/dT × T_i² + α₀ × T_i + trend + ε_i

    where ε ~ N(0, Σ) with Σ = σ²_ols × R.  R is the kernel-overlap
    correlation matrix and σ²_ols is estimated from OLS residuals
    (Feasible GLS).  This properly separates the temporal correlation
    structure from the error scale, avoiding the pathologically tight
    likelihood that would result from using the tiny kinematic rate_se
    (which captures only measurement noise, not climate variability).

    Priors (physically informed):

    - dα/dT ≥ 0 : HalfNormal(σ = ``prior_scale_a``)
    - α₀ ≥ 0    : HalfNormal(σ = ``prior_scale_b``)
    - trend > 0  : HalfNormal(σ = ``prior_scale_c``)

    The non-negativity constraint encodes the physical expectation
    that sea-level sensitivity to temperature is non-negative over
    climate-relevant timescales.

    Parameters
    ----------
    rate : np.ndarray
        Observed kinematic rates, shape (n,).  Units: mm/yr or m/yr.
    rate_se : np.ndarray
        1-σ standard errors on rates, shape (n,).  Same units as rate.
    temperature : np.ndarray
        Temperature anomalies (°C), shape (n,), matched to rate times.
    time : np.ndarray
        Observation times in decimal years, shape (n,).
    span_years : float
        Bandwidth from ``compute_kinematics()``.
    kernel : str
        Kernel type used in ``compute_kinematics()``.
    nugget : float, default 0.0
        Fraction of independent variance added to the correlation
        matrix.  Regularises the near-singular kernel-overlap
        correlation.  Typical values: 0.05–0.20.
    order : int
        Polynomial order: 1 (linear) or 2 (quadratic).
    fit_year_start : float or None
        If not None, restrict fit to data from this year onward.
    n_samples : int
        Number of production MCMC steps per walker.
    n_walkers : int
        Number of emcee walkers (must be ≥ 2 × n_params).
    n_burnin : int
        Number of burn-in steps to discard.
    thin : int
        Thinning factor for the chain.
    prior_scale_a, prior_scale_b, prior_scale_c : float
        Scale parameters for the half-normal priors on [dα/dT, α₀, trend].
    progress : bool
        Show emcee progress bar.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    BayesianRateResult
        Result dataclass with ``physical_coefficients`` and
        ``physical_covariance`` compatible with
        ``project_gmsl_ensemble()``.

    Notes
    -----
    - The correlation matrix accounts for the overlap of kernel windows
      used in ``compute_kinematics()``.  This typically reduces the
      effective sample size from ~100 annual points to ~4–8 independent
      observations, appropriately widening the posterior.
    - The posterior may be asymmetric near zero due to the half-normal
      prior truncation.  Use ``posterior_samples`` for projections to
      preserve this non-Gaussianity.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")

    # ---- 1. Apply time mask ----
    mask = np.ones(len(rate), dtype=bool)
    if fit_year_start is not None:
        mask &= (time >= fit_year_start)
    # Remove NaN
    mask &= ~(np.isnan(rate) | np.isnan(rate_se) | np.isnan(temperature)
               | np.isnan(time))
    mask &= (rate_se > 0)

    t_fit = time[mask]
    r_fit = rate[mask]
    se_fit = rate_se[mask]
    T_fit = temperature[mask]
    n = len(t_fit)

    if progress:
        print(f"Bayesian rate-space fit: n={n} observations, "
              f"order={order}, span_years={span_years}")

    # ---- 2. Design matrix ----
    if order == 2:
        X = np.column_stack([T_fit**2, T_fit, np.ones(n)])
        ndim = 3
        prior_scales = np.array([prior_scale_a, prior_scale_b,
                                 prior_scale_c])
        param_names = ['dalpha_dT', 'alpha0', 'trend']
    else:  # order == 1
        X = np.column_stack([T_fit, np.ones(n)])
        ndim = 2
        prior_scales = np.array([prior_scale_b, prior_scale_c])
        param_names = ['alpha0', 'trend']

    # ---- 3. Correlation matrix + full covariance ----
    R, n_eff = build_rate_correlation_matrix(t_fit, span_years, kernel,
                                                nugget=nugget)

    # Estimate error scale from OLS residuals (Feasible GLS approach).
    # rate_se from kinematics captures only measurement uncertainty
    # (~0.04 mm/yr), while the actual model residuals are ~10× larger
    # due to climate variability (ENSO, volcanism, PDO) the polynomial
    # cannot capture.  Using rate_se directly in Σ creates an
    # impossibly tight likelihood.  Instead: Σ = σ²_ols × R.
    beta_ols = np.linalg.lstsq(X, r_fit, rcond=None)[0]
    resid_ols = r_fit - X @ beta_ols
    sigma2_ols = np.var(resid_ols, ddof=ndim)
    Sigma = sigma2_ols * R

    if progress:
        print(f"  Error scale (OLS residual σ): "
              f"{np.sqrt(sigma2_ols) * 1000:.3f} mm/yr  "
              f"(vs rate_se mean: {se_fit.mean() * 1000:.3f} mm/yr)")

    # Pre-compute inverse and log-determinant (one-time O(n³))
    try:
        L = la.cholesky(Sigma, lower=True)
        Sigma_inv = la.cho_solve((L, True), np.eye(n))
        log_det_Sigma = 2.0 * np.sum(np.log(np.diag(L)))
    except la.LinAlgError:
        # Fallback: eigenvalue clip for near-singular case
        eigvals, eigvecs = la.eigh(Sigma)
        floor = 1e-6 * eigvals.max()
        n_clipped = np.sum(eigvals < floor)
        eigvals = np.maximum(eigvals, floor)
        Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
        Sigma_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
        log_det_Sigma = np.sum(np.log(eigvals))
        if progress:
            print(f"  Warning: covariance matrix required eigenvalue "
                  f"clipping ({n_clipped} eigenvalues below "
                  f"{floor:.2e})")

    if progress:
        print(f"  Effective sample size: n_eff = {n_eff:.1f} "
              f"(from {n} annual observations)")

    # ---- 4. Initialize walkers ----
    rng = np.random.default_rng(seed)

    # Start from OLS solution (computed above), clamped to positive
    p0_center = np.maximum(beta_ols, 1e-6)

    # Small positive perturbation (scale = 10% of OLS magnitude)
    p0_scale = np.maximum(np.abs(p0_center) * 0.1, 1e-6)
    p0 = np.abs(p0_center[None, :] +
                p0_scale[None, :] * rng.standard_normal((n_walkers, ndim)))

    # ---- 5. Run emcee ----
    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, _rate_log_prob,
        args=(X, r_fit, Sigma_inv, log_det_Sigma, prior_scales),
    )
    sampler.run_mcmc(p0, n_burnin + n_samples, progress=progress)

    # ---- 6. Post-process ----
    flat_chain = sampler.get_chain(discard=n_burnin, thin=thin, flat=True)
    # shape: (n_walkers * n_samples / thin, ndim)

    phys_mean = flat_chain.mean(axis=0)
    phys_cov = np.cov(flat_chain, rowvar=False)
    phys_hdi = np.array([
        az.hdi(flat_chain[:, k], hdi_prob=0.94) for k in range(ndim)
    ])

    # R² (posterior mean)
    r_pred = X @ phys_mean
    ss_res = np.sum((r_fit - r_pred)**2)
    ss_tot = np.sum((r_fit - r_fit.mean())**2)
    r2 = 1.0 - ss_res / ss_tot

    # Residuals
    residuals = r_fit - r_pred

    # Build arviz InferenceData (preserve walker structure)
    chain_full = sampler.get_chain(discard=n_burnin, thin=thin,
                                   flat=False)
    # shape: (n_steps, n_walkers, ndim)
    n_chains_arviz = min(4, n_walkers)
    var_dict = {}
    for k, name in enumerate(param_names):
        # (n_chains, n_steps)
        var_dict[name] = chain_full[:, :n_chains_arviz, k].T

    trace = az.from_dict(var_dict)

    # Convergence diagnostics
    conv = check_convergence(trace, quiet=(not progress))

    diag = {
        'acceptance_fraction': sampler.acceptance_fraction.mean(),
        'n_walkers': n_walkers,
        'n_samples': n_samples,
        'n_burnin': n_burnin,
        'thin': thin,
        'convergence': conv,
    }

    if progress:
        print(f"  Posterior mean: "
              + ", ".join(f"{name}={phys_mean[k]:.3f}"
                          for k, name in enumerate(param_names)))
        print(f"  R² = {r2:.4f},  acceptance = "
              f"{diag['acceptance_fraction']:.2f}")

    return BayesianRateResult(
        trace=trace,
        physical_coefficients=phys_mean,
        physical_covariance=phys_cov,
        physical_hdi_94=phys_hdi,
        posterior_samples=flat_chain,
        r2=r2,
        corr_matrix=R,
        corr_length=span_years,
        n_obs=n,
        n_eff=n_eff,
        time=t_fit,
        temperature=T_fit,
        rate_obs=r_fit,
        rate_se=se_fit,
        residuals=residuals,
        order=order,
        sampler_diagnostics=diag,
        design_info={
            'fit_year_start': fit_year_start,
            'prior_scales': prior_scales,
            'param_names': param_names,
            'kernel': kernel,
        },
    )


def prepare_rate_data(
    sea_level: pd.Series,
    temperature: pd.Series,
    gmsl_sigma: Optional[pd.Series] = None,
    span_years: float = 30.0,
    kernel: str = 'tricube',
    units: str = 'm',
) -> dict:
    """Extract kinematic rates matched to temperature for rate-space fitting.

    Convenience function that bridges the pandas Series interface
    (used throughout the codebase) and the numpy array interface of
    ``fit_bayesian_rate_model()``.

    Steps:
    1. Align sea level and temperature on a common annual grid.
    2. Run ``compute_kinematics()`` to get rate and rate_se.
    3. Match rate estimates to annual-mean temperature.
    4. Return arrays ready for ``fit_bayesian_rate_model()``.

    Parameters
    ----------
    sea_level : pd.Series
        Sea level with DatetimeIndex (meters).
    temperature : pd.Series
        Temperature anomaly with DatetimeIndex (°C).
    gmsl_sigma : pd.Series or None
        1-σ uncertainty on sea level (meters).
    span_years : float
        Bandwidth for ``compute_kinematics()``.
    kernel : str
        Kernel for ``compute_kinematics()``.
    units : str
        Output units for rate: 'm' (m/yr) or 'mm' (mm/yr).

    Returns
    -------
    dict with keys:
        'rate'       : np.ndarray — kinematic rates
        'rate_se'    : np.ndarray — rate standard errors
        'temperature': np.ndarray — matched temperature anomalies
        'time'       : np.ndarray — decimal years
        'span_years' : float
        'kernel'     : str
        'units'      : str
    """
    # Normalise indices to month-start
    sl_ms = _to_month_start(sea_level)
    temp_ms = _to_month_start(temperature)

    # Annual mean temperature for matching
    temp_annual = temp_ms.resample('YE').mean()
    temp_years = temp_annual.index.year + 0.5
    temp_vals = temp_annual.values

    # Compute kinematics
    common = sl_ms.dropna().index
    dec_year = np.array([
        t.year + (t.month - 0.5) / 12 for t in common
    ])
    sl_vals = sl_ms.loc[common].values.astype(np.float64)

    if gmsl_sigma is not None:
        sig_ms = _to_month_start(gmsl_sigma)
        sig_vals = sig_ms.reindex(common).values.astype(np.float64)
        # Fill any NaN sigma with median
        med_sig = np.nanmedian(sig_vals)
        sig_vals = np.where(np.isnan(sig_vals), med_sig, sig_vals)
    else:
        sig_vals = np.ones_like(sl_vals) * np.std(sl_vals) * 0.01

    kin = compute_kinematics(
        time=dec_year, value=sl_vals, sigma=sig_vals,
        span_years=span_years, kernel=kernel,
    )

    df_kin = kin.to_dataframe()
    valid = ~np.isnan(df_kin['rate'])
    kin_years = df_kin.loc[valid, 'decimal_year'].values
    kin_rate = df_kin.loc[valid, 'rate'].values
    kin_se = df_kin.loc[valid, 'rate_se'].values

    # Subsample to annual: pick the rate estimate closest to mid-year
    # (kinematics are computed at monthly resolution; adjacent monthly
    # estimates are nearly identical and redundant)
    unique_years = np.unique(np.floor(kin_years).astype(int))
    ann_idx = []
    for uy in unique_years:
        mid = uy + 0.5
        mask = (kin_years >= uy) & (kin_years < uy + 1)
        if not np.any(mask):
            continue
        candidates = np.where(mask)[0]
        best = candidates[np.argmin(np.abs(kin_years[candidates] - mid))]
        ann_idx.append(best)
    ann_idx = np.array(ann_idx)

    kin_years = kin_years[ann_idx]
    kin_rate = kin_rate[ann_idx]
    kin_se = kin_se[ann_idx]

    # Match to annual temperature (±0.5 yr tolerance)
    matched_temp, matched_rate, matched_se, matched_time = [], [], [], []
    for yr, r, se in zip(kin_years, kin_rate, kin_se):
        idx = np.argmin(np.abs(temp_years - yr))
        if np.abs(temp_years[idx] - yr) <= 0.5:
            matched_temp.append(temp_vals[idx])
            matched_rate.append(r)
            matched_se.append(se)
            matched_time.append(yr)

    rate = np.array(matched_rate)
    rate_se = np.array(matched_se)
    temp_matched = np.array(matched_temp)
    time_matched = np.array(matched_time)

    # Unit conversion
    scale = 1000.0 if units == 'mm' else 1.0
    rate *= scale
    rate_se *= scale

    return {
        'rate': rate,
        'rate_se': rate_se,
        'temperature': temp_matched,
        'time': time_matched,
        'span_years': span_years,
        'kernel': kernel,
        'units': f'{units}/yr',
    }


# ====================================================================
# Model 5: Bayesian Level-Space Calibration
# ====================================================================


def calibrate_exponential_prior(
    prob_exceed: float = 0.10,
    threshold: float = 0.005,
) -> float:
    """Calibrate Exponential prior mean via a tail-probability constraint.

    For a non-negative parameter x with an Exponential(mean=μ) prior,
    this function solves for μ given a desired tail probability:

        P(x > threshold) = prob_exceed

    Since P(x > u) = exp(−u/μ) for x ~ Exponential(μ), solving gives:

        μ = −threshold / ln(prob_exceed)

    This is the standard penalised-complexity (PC) prior calibration:
    the prior penalises departure from x=0 (the simpler nested model)
    in proportion to the magnitude of x.

    Parameters
    ----------
    prob_exceed : float
        Desired prior probability that x exceeds threshold.
        Default 0.10 (10% chance of exceeding the threshold).
    threshold : float
        Threshold value for x (same units as the parameter).
        Default 0.005.

    Returns
    -------
    float
        Exponential mean μ (same units as threshold).

    Examples
    --------
    >>> calibrate_exponential_prior(0.10, 0.005)
    0.00217   # P(x > 0.005) = 10%
    >>> calibrate_exponential_prior(0.05, 0.005)
    0.00167   # P(x > 0.005) = 5%
    >>> calibrate_exponential_prior(0.10, 0.003)
    0.00130   # P(x > 0.003) = 10%
    """
    if prob_exceed <= 0 or prob_exceed >= 1:
        raise ValueError("prob_exceed must be in (0, 1)")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    return -threshold / np.log(prob_exceed)


# Backward-compatible alias
calibrate_exponential_prior_a = calibrate_exponential_prior


def prior_predictive_rate_check(
    prior_scale_a: float,
    prior_scale_b: float = 0.010,
    prior_c_mean: float = 0.002,
    prior_c_sigma: float = 0.005,
    T_values: Optional[np.ndarray] = None,
    n_draws: int = 50000,
    seed: Optional[int] = None,
) -> dict:
    """Draw from the joint prior and compute rate(T) = a·T² + b·T + c.

    Uses Exponential(mean=prior_scale_a) for a, HalfNormal(σ=prior_scale_b)
    for b, and Normal(prior_c_mean, prior_c_sigma) for c.

    Parameters
    ----------
    prior_scale_a : float
        Exponential mean for a (m/yr/°C²).
    prior_scale_b : float
        HalfNormal σ for b (m/yr/°C).
    prior_c_mean, prior_c_sigma : float
        Normal prior parameters for c (m/yr).
    T_values : array-like or None
        Temperature anomalies at which to evaluate rate.
        Default: [0.5, 0.8, 1.0, 2.0, 4.0] °C.
    n_draws : int
        Number of prior draws.
    seed : int or None
        Random seed.

    Returns
    -------
    dict
        Keys are T values; values are dicts with 'mean', 'median',
        'p5', 'p95', 'p99' of the prior predictive rate in m/yr.
    """
    if T_values is None:
        T_values = [0.5, 0.8, 1.0, 2.0, 4.0]

    rng = np.random.default_rng(seed)
    a = rng.exponential(prior_scale_a, n_draws)
    b = np.abs(rng.normal(0, prior_scale_b, n_draws))  # HalfNormal
    c = rng.normal(prior_c_mean, prior_c_sigma, n_draws)

    results = {}
    for T in T_values:
        rates = a * T**2 + b * T + c
        results[T] = {
            'mean': float(np.mean(rates)),
            'median': float(np.median(rates)),
            'p5': float(np.percentile(rates, 5)),
            'p95': float(np.percentile(rates, 95)),
            'p99': float(np.percentile(rates, 99)),
        }
    return results


@dataclass
class BayesianLevelResult:
    """Bayesian level-space calibration result.

    Fits the generative model:

        rate(t) = a·T(t)² + b·T(t) + c
        H(t)   = H₀ + ∫ rate(τ) dτ
        H_obs  ~ N(H(t), σ(t)²)

    directly against cumulative GMSL observations with time-varying
    observation uncertainty and physically informed priors.

    Compatible with ``project_gmsl_ensemble()`` via
    ``physical_coefficients`` and ``posterior_samples``.
    """
    trace: az.InferenceData
    physical_coefficients: np.ndarray    # posterior mean [a, b, c]
    physical_covariance: np.ndarray      # posterior covariance (3×3)
    physical_hdi_94: np.ndarray          # (3, 2) — 94% HDI
    posterior_samples: np.ndarray        # (n_samples, 3) — [a, b, c]
    H0_posterior: np.ndarray             # (n_samples,) — baseline level
    sigma_extra_posterior: np.ndarray    # (n_samples,) — learned noise
    r2: float                            # R² against observed GMSL
    residuals: np.ndarray                # posterior-mean residuals (m)
    time: np.ndarray                     # observation times (decimal year)
    H_obs: np.ndarray                    # observed GMSL (m)
    H_model_mean: np.ndarray             # posterior-mean predicted GMSL (m)
    sigma_obs: np.ndarray                # σ_obs(t) excl σ_extra (m)
    order: int = 2
    sampler_diagnostics: Optional[dict] = None
    design_info: Optional[dict] = None


def build_level_design_vectors(
    temperature_monthly: np.ndarray,
    time_monthly: np.ndarray,
    obs_times: np.ndarray,
) -> dict:
    """Build pre-computed design vectors for the level-space model.

    The forward model is linear in the parameters (a, b, c, H₀):

        H(t) = a·I₂(t) + b·I₁(t) + c·I₀(t) + H₀

    where I₂ = ∫T²dτ, I₁ = ∫Tdτ, I₀ = t − t₀ are computed once on the
    monthly temperature grid and then extracted at the annual observation
    times.  This makes each log-probability evaluation O(n_obs) rather
    than O(n_monthly).

    Parameters
    ----------
    temperature_monthly : np.ndarray
        Monthly temperature anomaly (°C), shape (n_monthly,).
    time_monthly : np.ndarray
        Monthly decimal years, shape (n_monthly,).
    obs_times : np.ndarray
        Annual observation times (decimal year), shape (n_obs,).

    Returns
    -------
    dict with keys:
        'I2_obs'  : np.ndarray (n_obs,) — ∫T² at observation times
        'I1_obs'  : np.ndarray (n_obs,) — ∫T at observation times
        'I0_obs'  : np.ndarray (n_obs,) — t − t₀ at observation times
        'I2_full' : np.ndarray (n_monthly,) — ∫T² on monthly grid
        'I1_full' : np.ndarray (n_monthly,) — ∫T on monthly grid
        'I0_full' : np.ndarray (n_monthly,) — t − t₀ on monthly grid
        'obs_idx' : np.ndarray (n_obs,) — indices into monthly grid
    """
    T = temperature_monthly
    t = time_monthly
    dt = np.diff(t)
    n = len(T)

    # Cumulative trapezoidal integrals starting from t[0]
    I2 = np.zeros(n)
    I1 = np.zeros(n)
    for i in range(n - 1):
        I2[i + 1] = I2[i] + 0.5 * (T[i]**2 + T[i + 1]**2) * dt[i]
        I1[i + 1] = I1[i] + 0.5 * (T[i] + T[i + 1]) * dt[i]
    I0 = t - t[0]

    # Match observation times to nearest monthly index
    obs_idx = np.array([
        np.argmin(np.abs(t - t_obs)) for t_obs in obs_times
    ])

    return {
        'I2_obs': I2[obs_idx],
        'I1_obs': I1[obs_idx],
        'I0_obs': I0[obs_idx],
        'I2_full': I2,
        'I1_full': I1,
        'I0_full': I0,
        'obs_idx': obs_idx,
        'time_monthly': t,
        'temperature_monthly': T,
    }


# ================================================================
# Budget closure constraint (two-pass approach)
# ================================================================

@dataclass
class BudgetTarget:
    """Budget-closure constraint for a single component.

    Derived from GMSL_obs - Σ_{j≠k} H_j at satellite-era times.
    Used as an additional likelihood term in Pass 2.
    """
    times: np.ndarray          # (n_budget,) decimal years
    level_obs: np.ndarray      # (n_budget,) GMSL - other components (meters)
    level_sigma: np.ndarray    # (n_budget,) combined uncertainty (meters)
    rate_accel: object         # SatelliteEraQuadraticResult for budget rate+accel
    component_name: str        # name of the target component


def compute_budget_target(
    gmsl_time: np.ndarray,
    gmsl_level: np.ndarray,
    gmsl_sigma: np.ndarray,
    component_models: dict,
    exclude: str,
    rate_accel_gmsl: 'SatelliteEraQuadraticResult' = None,
    component_rate_accels: dict = None,
) -> BudgetTarget:
    """Compute the budget-closure target for one component.

    For component k, the budget "observation" is:

        H_k_budget(t) = GMSL_obs(t) - Σ_{j≠k} H_j(t)

    where each H_j is the Pass 1 posterior-mean model evaluated at
    the budget times.

    Parameters
    ----------
    gmsl_time : (n,)
        Annual GMSL observation times (decimal year).
    gmsl_level : (n,)
        Observed GMSL (meters), rebased to project baseline.
    gmsl_sigma : (n,)
        1-σ GMSL uncertainty (meters).
    component_models : dict
        {name: (time_array, H_model_array, H_sigma_array)} for each
        fitted component.  H_model is the posterior-mean model
        prediction; H_sigma is the posterior predictive σ.
        Times need not match gmsl_time — linear interpolation is used.
    exclude : str
        Name of the component to exclude (the target).
    rate_accel_gmsl : SatelliteEraQuadraticResult or None
        Observed GMSL rate + acceleration (from satellite altimetry).
    component_rate_accels : dict or None
        {name: (rate, accel, rate_se, accel_se)} for each component.
        Used to compute the budget rate+accel for the excluded component.

    Returns
    -------
    BudgetTarget
    """
    from scipy.interpolate import interp1d

    # Sum all components except the excluded one at gmsl_time
    H_others = np.zeros_like(gmsl_level)
    sigma2_others = np.zeros_like(gmsl_level)

    for name, (t_comp, H_comp, sigma_comp) in component_models.items():
        if name == exclude:
            continue
        # Interpolate component to budget times
        f = interp1d(t_comp, H_comp, kind='linear',
                     bounds_error=False, fill_value=np.nan)
        H_interp = f(gmsl_time)

        # Interpolate uncertainty
        f_sig = interp1d(t_comp, sigma_comp, kind='linear',
                         bounds_error=False, fill_value=np.nan)
        sig_interp = f_sig(gmsl_time)

        H_others += np.nan_to_num(H_interp, nan=0.0)
        sigma2_others += np.nan_to_num(sig_interp**2, nan=0.0)

    # Budget target: what the excluded component "should" be
    level_obs = gmsl_level - H_others
    level_sigma = np.sqrt(gmsl_sigma**2 + sigma2_others)

    # Budget rate + acceleration (if provided)
    budget_ra = None
    if rate_accel_gmsl is not None and component_rate_accels is not None:
        rate_others = 0.0
        accel_others = 0.0
        rate_var_others = 0.0
        accel_var_others = 0.0
        for name, (r, a, r_se, a_se) in component_rate_accels.items():
            if name == exclude:
                continue
            rate_others += r
            accel_others += a
            rate_var_others += r_se**2
            accel_var_others += a_se**2

        budget_rate = rate_accel_gmsl.rate - rate_others
        budget_accel = rate_accel_gmsl.accel - accel_others

        # Propagate covariance: GMSL cov + other-component variances
        budget_ra_cov = rate_accel_gmsl.rate_accel_cov.copy()
        budget_ra_cov[0, 0] += rate_var_others
        budget_ra_cov[1, 1] += accel_var_others

        # Create a lightweight result-like object
        budget_ra = SatelliteEraQuadraticResult(
            rate=budget_rate,
            accel=budget_accel,
            rate_accel_cov=budget_ra_cov,
            rate_se=np.sqrt(budget_ra_cov[0, 0]),
            accel_se=np.sqrt(budget_ra_cov[1, 1]),
            eval_time=rate_accel_gmsl.eval_time,
            coefficients=np.zeros(3),
            cov_params=np.zeros((3, 3)),
            t_start=rate_accel_gmsl.t_start,
            t_end=rate_accel_gmsl.t_end,
            n_obs=0,
            r2=0.0,
            fit_result=None,
            fit_method='budget_residual',
        )

    return BudgetTarget(
        times=gmsl_time,
        level_obs=level_obs,
        level_sigma=level_sigma,
        rate_accel=budget_ra,
        component_name=exclude,
    )


def _budget_level_logp(
    H_model_at_budget: np.ndarray,
    budget: BudgetTarget,
) -> float:
    """Log-likelihood for the level-space budget constraint.

    Parameters
    ----------
    H_model_at_budget : (n_budget,)
        Model prediction for the target component at budget times.
    budget : BudgetTarget
        Contains level_obs and level_sigma.
    """
    resid = H_model_at_budget - budget.level_obs
    return -0.5 * np.sum(
        (resid / budget.level_sigma)**2
        + 2.0 * np.log(budget.level_sigma)
    )


@dataclass
class SatelliteEraQuadraticResult:
    """Result from a quadratic fit to a GMSL record.

    Provides end-of-record rate and acceleration with their joint
    2×2 covariance, suitable for use as an informational prior in
    the Bayesian level-space models.
    """
    rate: float                  # dH/dt at eval_time (m/yr)
    accel: float                 # d²H/dt² (m/yr²), constant for quadratic
    rate_accel_cov: np.ndarray   # (2, 2) covariance of (rate, accel)
    rate_se: float               # marginal SE on rate (m/yr)
    accel_se: float              # marginal SE on accel (m/yr²)
    eval_time: float             # time at which rate is evaluated (decimal yr)
    coefficients: np.ndarray     # (3,) [c0, c1, c2] from H = c0 + c1*dt + c2*dt²
    cov_params: np.ndarray       # (3, 3) parameter covariance
    t_start: float               # start of fitting window (decimal yr)
    t_end: float                 # end of fitting window (decimal yr)
    n_obs: int                   # number of observations used
    r2: float                    # R² of the fit
    fit_result: object           # statsmodels RegressionResultsWrapper
    fit_method: str = 'OLS_HAC'  # 'WLS' or 'OLS_HAC'


def fit_satellite_era_quadratic(
    time: np.ndarray,
    gmsl: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    t_start: float = 1993.0,
    t_end: float = None,
    eval_time: float = None,
    sigma_inflate: float = 1.0,
    meas_cov_path: Optional[str] = None,
    sigma_gia: float = 0.15e-3,
) -> SatelliteEraQuadraticResult:
    """Quadratic fit to a GMSL record with rigorous uncertainty.

    Fits H(t) = c₀ + c₁·(t − t₀) + c₂·(t − t₀)² where t₀ = t_start.

    Three fitting / uncertainty modes (in order of preference):

    1. **Full error budget** (when ``meas_cov_path`` is provided):
       OLS point estimates + three-component uncertainty following
       Hamlington et al. (2024):

       - **Measurement errors**: propagated from the Ablain et al.
         (2019) error variance-covariance matrix (10-day resolution
         NetCDF from doi:10.17882/58344).
       - **Serial correlation**: Maul & Martin (1993) lag-1
         autocorrelation inflation of formal OLS standard errors.
       - **GIA uncertainty**: constant rate uncertainty (no accel
         component); default 0.15 mm/yr (1σ) from Caron et al. (2018).

       All three are summed in quadrature (independent sources).

    2. **OLS + HAC** (when ``sigma`` is ``None`` and no
       ``meas_cov_path``): OLS with Newey-West standard errors.

    3. **WLS** (when ``sigma`` is provided): Weighted least squares
       with weights = 1/σ².

    The rate and acceleration at the evaluation time are::

        rate(t_eval) = c₁ + 2·c₂·(t_eval − t₀)
        accel        = 2·c₂

    with their 2×2 covariance propagated from the parameter
    covariance via the Jacobian.

    Parameters
    ----------
    time : np.ndarray, shape (n,)
        Observation times (decimal year).
    gmsl : np.ndarray, shape (n,)
        Observed GMSL (meters).
    sigma : np.ndarray or None
        1-σ observation uncertainty (meters).  If ``None``, uses
        OLS (with HAC or full error budget).
    t_start : float
        Start of fitting window (decimal year).  Default 1993.0.
    t_end : float or None
        End of fitting window.  None → end of record.
    eval_time : float or None
        Time at which to evaluate rate.  None → end of record.
    sigma_inflate : float
        Multiplicative inflation factor for the output covariance.
        Default 1.0 (no inflation).
    meas_cov_path : str or None
        Path to Ablain et al. (2019) error covariance NetCDF.
        When provided, enables the full three-component error budget
        (measurement + serial correlation + GIA).  The covariance
        matrix covers 1993–2018 at ~10-day cadence.  If the fitting
        window extends beyond the covariance matrix time span, the
        measurement error contribution is computed over the overlap
        period and scaled by the ratio of record lengths.
    sigma_gia : float
        1-σ GIA correction uncertainty on the rate (m/yr).
        Default 0.15e-3 (= 0.15 mm/yr), from Caron et al. (2018).
        Only used when ``meas_cov_path`` is provided.  GIA does not
        contribute to acceleration uncertainty.

    Returns
    -------
    SatelliteEraQuadraticResult

    References
    ----------
    Ablain et al. (2019), ESSD — measurement error covariance matrix.
    Caron et al. (2018), G-Cubed — GIA posterior ensemble.
    Hamlington et al. (2024), Commun. Earth Environ. — methodology.
    Maul & Martin (1993), GRL — serial correlation inflation.
    """
    # Truncate to fitting window
    if t_end is None:
        t_end = time[-1]
    mask = (time >= t_start) & (time <= t_end) & np.isfinite(gmsl)
    if sigma is not None:
        mask &= np.isfinite(sigma) & (sigma > 0)
    t_fit = time[mask]
    h_fit = gmsl[mask]

    if len(t_fit) < 5:
        raise ValueError(
            f"Only {len(t_fit)} observations in [{t_start}, {t_end}]; "
            f"need at least 5 for a quadratic fit.")

    if eval_time is None:
        eval_time = t_fit[-1]

    # Centre time on t_start for numerical stability
    dt = t_fit - t_start

    # Design matrix: [1, dt, dt²]
    X = np.column_stack([np.ones_like(dt), dt, dt**2])

    # --- Point estimates: always OLS when meas_cov_path is given ---
    if meas_cov_path is not None or sigma is None:
        model = sm.OLS(h_fit, X)
        fit_result = model.fit()
        fit_method = 'full_error_budget' if meas_cov_path else 'OLS_HAC'
    else:
        s_fit = sigma[mask]
        weights = 1.0 / s_fit**2
        model = sm.WLS(h_fit, X, weights=weights)
        fit_result = model.fit()
        fit_method = 'WLS'

    c0, c1, c2 = fit_result.params
    r2 = fit_result.rsquared

    # Rate and acceleration at eval_time
    dt_eval = eval_time - t_start
    rate = c1 + 2.0 * c2 * dt_eval
    accel = 2.0 * c2

    # Jacobian: d(rate, accel) / d(c0, c1, c2)
    J = np.array([
        [0.0, 1.0, 2.0 * dt_eval],
        [0.0, 0.0, 2.0],
    ])

    # --- Covariance estimation ---
    if meas_cov_path is not None:
        # Full three-component error budget
        import netCDF4 as nc4
        from datetime import datetime as _dt, timedelta as _td

        # (1) Measurement errors from Ablain et al. covariance
        with nc4.Dataset(meas_cov_path, 'r') as ds:
            days_1950 = np.array(ds.variables['time'][:])
            C_meas_full = np.array(ds.variables['covariance_matrix'][:])
        base = _dt(1950, 1, 1)
        t_cov = np.array([
            (base + _td(days=float(d))).year
            + ((base + _td(days=float(d))).timetuple().tm_yday - 0.5)
            / 365.25
            for d in days_1950
        ])

        # Build design matrix on the covariance grid (same t_start)
        mask_cov = (t_cov >= t_start) & (t_cov <= t_end)
        t_c = t_cov[mask_cov]
        idx_cov = np.where(mask_cov)[0]
        C_meas = C_meas_full[np.ix_(idx_cov, idx_cov)]
        dt_c = t_c - t_start
        X_c = np.column_stack([np.ones(len(dt_c)), dt_c, dt_c**2])
        XtX_c_inv = np.linalg.inv(X_c.T @ X_c)
        cov_beta_meas = XtX_c_inv @ (X_c.T @ C_meas @ X_c) @ XtX_c_inv

        # Scale if fitting window extends beyond covariance span.
        # Anisotropic scaling: for h = c0 + c1*t + c2*t², the uncertainty
        # on c1 (trend) scales as ~1/T and on c2 (acceleration) as ~1/T².
        # c0 (intercept at t=0) is insensitive to span extension.
        # Element-wise: S_ij = r^((p_i + p_j)/2) where r = T_cov/T_fit
        # and p = [0, 1, 2] are the T-dependence exponents for each param.
        t_cov_span = t_c[-1] - t_c[0]
        t_fit_span = t_fit[-1] - t_fit[0]
        if t_fit_span > t_cov_span * 1.05:
            r = t_cov_span / t_fit_span
            pwr = np.array([0.0, 1.0, 2.0])
            scale_mat = r ** ((pwr[:, None] + pwr[None, :]) / 2.0)
            cov_beta_meas *= scale_mat

        ra_cov_meas = J @ cov_beta_meas @ J.T

        # (2) Serial correlation (Maul & Martin 1993)
        resid = h_fit - X @ fit_result.params
        rho = np.corrcoef(resid[:-1], resid[1:])[0, 1]
        sigma2_resid = np.sum(resid**2) / (len(t_fit) - 3)
        XtX_inv = np.linalg.inv(X.T @ X)
        inflate = (1.0 + rho) / (1.0 - rho)
        cov_beta_serial = inflate * sigma2_resid * XtX_inv
        ra_cov_serial = J @ cov_beta_serial @ J.T

        # (3) GIA: constant rate, no acceleration
        ra_cov_gia = np.zeros((2, 2))
        ra_cov_gia[0, 0] = sigma_gia**2

        # Total: sum independent sources
        cov_params = cov_beta_meas + cov_beta_serial
        # (GIA only enters the rate-accel covariance, not param cov)
        rate_accel_cov = sigma_inflate**2 * (
            ra_cov_meas + ra_cov_serial + ra_cov_gia)

        fit_method = 'full_error_budget'

    elif sigma is None:
        # OLS + HAC fallback
        maxlags = int(np.ceil(np.sqrt(len(t_fit))))
        fit_result_hac = model.fit(
            cov_type='HAC', cov_kwds={'maxlags': maxlags})
        cov_params = np.array(fit_result_hac.cov_params())
        rate_accel_cov = sigma_inflate**2 * (J @ cov_params @ J.T)
        fit_method = 'OLS_HAC'

    else:
        # WLS
        cov_params = np.array(fit_result.cov_params())
        rate_accel_cov = sigma_inflate**2 * (J @ cov_params @ J.T)

    rate_se = np.sqrt(rate_accel_cov[0, 0])
    accel_se = np.sqrt(rate_accel_cov[1, 1])

    return SatelliteEraQuadraticResult(
        rate=rate,
        accel=accel,
        rate_accel_cov=rate_accel_cov,
        rate_se=rate_se,
        accel_se=accel_se,
        eval_time=eval_time,
        coefficients=np.array([c0, c1, c2]),
        cov_params=cov_params,
        t_start=t_start,
        t_end=t_end,
        n_obs=len(t_fit),
        r2=r2,
        fit_result=fit_result,
        fit_method=fit_method,
    )


def _rate_accel_prior_logp(
    rate_model: float,
    accel_model: float,
    rate_prior: SatelliteEraQuadraticResult,
    cov_inv: Optional[np.ndarray] = None,
) -> float:
    """Bivariate Gaussian log-prior penalty on (rate, accel).

    Parameters
    ----------
    rate_model : float
        Model-implied GMSL rate at end of record (m/yr).
    accel_model : float
        Model-implied GMSL acceleration (m/yr²).
    rate_prior : SatelliteEraQuadraticResult
        Contains observed rate, accel, and their 2×2 covariance.
    cov_inv : (2, 2) array or None
        Pre-computed inverse of ``rate_prior.rate_accel_cov``.
        If None, the inverse is computed on the fly (convenient but
        wasteful inside MCMC loops).

    Returns
    -------
    float
        Log-density of the bivariate Gaussian.
    """
    delta = np.array([rate_model - rate_prior.rate,
                      accel_model - rate_prior.accel])
    if cov_inv is None:
        cov_inv = np.linalg.inv(rate_prior.rate_accel_cov)
    return -0.5 * delta @ cov_inv @ delta


def _level_log_prior(theta, prior_scales, H0_prior_mean,
                     symmetric_a=False):
    """Log-prior for the Bayesian level-space model.

    Parameters: theta = [a, b, c, log_sigma_extra, H0]

    Priors:
        a (dα/dT)       : Exponential(mean = prior_scales[0]) — a ≥ 0  [default]
                           OR Normal(0, prior_scales[0]) if symmetric_a=True
                           PC prior: shrinks toward a=0 (order-1 model)
        b (α₀)          : HalfNormal(σ = prior_scales[1])     — b ≥ 0
        c (trend)        : Normal(prior_scales[2], prior_scales[3])
        σ_extra          : HalfCauchy(0, prior_scales[4])      — sampled as log(σ_extra)
        H₀              : Normal(H0_prior_mean, prior_scales[5])

    symmetric_a : bool
        If True, use Normal(0, prior_scales[0]) for a instead of
        Exponential.  This allows a < 0, appropriate for components
        whose rate–temperature sensitivity may saturate or reverse
        (e.g., glaciers approaching depletion).
    """
    a, b, c, log_sigma_extra, H0 = theta

    # Non-negativity hard bounds (b always ≥ 0; a ≥ 0 unless symmetric)
    if b < 0:
        return -np.inf
    if not symmetric_a and a < 0:
        return -np.inf

    sigma_extra = np.exp(log_sigma_extra)
    if sigma_extra < 1e-12:
        return -np.inf

    lp = 0.0
    if symmetric_a:
        # a ~ Normal(0, σ_a)  — allows a < 0 (saturation)
        lp += -0.5 * (a / prior_scales[0])**2
    else:
        # a ~ Exponential(mean = μ_a)  [PC prior — mode at a=0]
        # log p(a) = -a/μ + const  (const = -log(μ), dropped for MCMC)
        lp += -a / prior_scales[0]
    # b ~ HalfNormal(σ)
    lp += -0.5 * (b / prior_scales[1])**2
    # c ~ Normal(μ_c, σ_c)
    lp += -0.5 * ((c - prior_scales[2]) / prior_scales[3])**2
    # σ_extra ~ HalfCauchy(0, γ)  →  log-density + Jacobian
    gamma = prior_scales[4]
    lp += -np.log(1.0 + (sigma_extra / gamma)**2) + log_sigma_extra
    # H₀ ~ Normal
    lp += -0.5 * ((H0 - H0_prior_mean) / prior_scales[5])**2

    return lp


def _level_log_likelihood(theta, I2, I1, I0, H_obs, sigma_obs_fixed):
    """Gaussian log-likelihood for the level-space model.

    H_model = a·I2 + b·I1 + c·I0 + H0
    σ(t)²   = σ_obs_fixed(t)² + σ_extra²
    """
    a, b, c, log_sigma_extra, H0 = theta
    sigma_extra = np.exp(log_sigma_extra)

    H_model = a * I2 + b * I1 + c * I0 + H0
    sigma_total = np.sqrt(sigma_obs_fixed**2 + sigma_extra**2)
    resid = H_obs - H_model

    return -0.5 * np.sum(
        (resid / sigma_total)**2 + 2.0 * np.log(sigma_total)
    )


def _level_log_prob(theta, I2, I1, I0, H_obs, sigma_obs_fixed,
                    prior_scales, H0_prior_mean, symmetric_a=False,
                    rate_prior=None, T_end=0.0, dTdt_end=0.0,
                    rate_prior_cov_inv=None):
    """Log-posterior for the level-space model.

    Parameters
    ----------
    rate_prior : SatelliteEraQuadraticResult or None
        If provided, adds a bivariate Gaussian penalty on the
        model-implied (rate, accel) at end of record.
    T_end : float
        Temperature at end of record (°C anomaly).
        Only used when rate_prior is not None.
    dTdt_end : float
        Temperature trend at end of record (°C/yr).
        Only used when rate_prior is not None.
    """
    lp = _level_log_prior(theta, prior_scales, H0_prior_mean,
                          symmetric_a=symmetric_a)
    if not np.isfinite(lp):
        return -np.inf
    ll = _level_log_likelihood(theta, I2, I1, I0, H_obs, sigma_obs_fixed)
    if not np.isfinite(ll):
        return -np.inf

    # Satellite-era rate/accel prior
    if rate_prior is not None:
        a, b, c = theta[0], theta[1], theta[2]
        rate_model = a * T_end**2 + b * T_end + c
        accel_model = (2.0 * a * T_end + b) * dTdt_end
        lp_rate = _rate_accel_prior_logp(rate_model, accel_model,
                                         rate_prior, rate_prior_cov_inv)
        if not np.isfinite(lp_rate):
            return -np.inf
        return lp + ll + lp_rate

    return lp + ll


def fit_bayesian_level(
    H_obs: np.ndarray,
    sigma_obs: np.ndarray,
    I2_obs: np.ndarray,
    I1_obs: np.ndarray,
    I0_obs: np.ndarray,
    prior_scale_a: float = 0.010,
    prior_scale_b: float = 0.010,
    prior_c_mean: float = 0.002,
    prior_c_sigma: float = 0.005,
    prior_sigma_extra_scale: float = 0.005,
    prior_H0_sigma: float = 0.050,
    n_samples: int = 3000,
    n_walkers: int = 32,
    n_burnin: int = 1000,
    thin: int = 1,
    progress: bool = True,
    seed: Optional[int] = None,
    symmetric_a: bool = False,
    rate_prior: Optional['SatelliteEraQuadraticResult'] = None,
    temperature_monthly: Optional[np.ndarray] = None,
    time_monthly: Optional[np.ndarray] = None,
    obs_times: Optional[np.ndarray] = None,
) -> BayesianLevelResult:
    """Bayesian level-space calibration of the rate–temperature model.

    Fits the generative model directly against cumulative GMSL
    observations:

        rate(t)  = a·T(t)² + b·T(t) + c
        H(t)     = H₀ + ∫ rate(τ) dτ
        H_obs(t) ~ N(H(t), σ(t)²)

    where σ(t)² = σ_obs(t)² + σ_extra² combines the known time-varying
    observation uncertainty with a learned model-inadequacy term.

    The forward model is pre-linearised: ``H = a·I₂ + b·I₁ + c·I₀ + H₀``
    where the design vectors (I₂, I₁, I₀) are computed beforehand by
    ``build_level_design_vectors()``.

    Parameters
    ----------
    H_obs : np.ndarray, shape (n,)
        Observed cumulative GMSL (meters).
    sigma_obs : np.ndarray, shape (n,)
        Time-varying 1-σ observation uncertainty (meters),
        *excluding* σ_extra.
    I2_obs, I1_obs, I0_obs : np.ndarray, shape (n,)
        Pre-computed design vectors from ``build_level_design_vectors()``.
    prior_scale_a : float
        Exponential mean for a (dα/dT), in m/yr/°C².  Default 0.010.
        PC prior: mode at a=0, shrinks toward order-1 model.
        Use ``calibrate_exponential_prior_a()`` to set from a tail
        probability constraint.
        If ``symmetric_a=True``, this is the Normal σ instead.
    prior_scale_b : float
        HalfNormal σ for b (α₀), in m/yr/°C.  Default 0.010.
    prior_c_mean : float
        Normal prior mean for c (trend), in m/yr.  Default 0.002.
    prior_c_sigma : float
        Normal prior σ for c, in m/yr.  Default 0.005.
    prior_sigma_extra_scale : float
        HalfCauchy scale for σ_extra, in meters.  Default 0.005.
    prior_H0_sigma : float
        Normal prior σ for H₀, in meters.  Default 0.050.
    n_samples, n_walkers, n_burnin, thin : int
        MCMC sampler settings.
    progress : bool
        Show emcee progress bar.
    seed : int or None
        Random seed.
    symmetric_a : bool
        If True, use Normal(0, σ) prior on a instead of Exponential.
    rate_prior : SatelliteEraQuadraticResult or None
        If provided, adds a bivariate Gaussian penalty on the
        model-implied (rate, accel) at end of record.  Requires
        ``temperature_monthly`` and ``time_monthly`` to compute
        T_end and dT/dt at the evaluation epoch.
    temperature_monthly : np.ndarray or None
        Monthly temperature (°C), required when rate_prior is set.
    time_monthly : np.ndarray or None
        Monthly decimal years, required when rate_prior is set.
    obs_times : np.ndarray or None
        Observation times (decimal year), required when rate_prior is set.

    Returns
    -------
    BayesianLevelResult
    """
    n = len(H_obs)
    ndim = 5  # [a, b, c, log_sigma_extra, H0]

    prior_scales = np.array([
        prior_scale_a,          # [0] Exp mean for a
        prior_scale_b,          # [1] HN σ for b
        prior_c_mean,           # [2] Normal μ for c
        prior_c_sigma,          # [3] Normal σ for c
        prior_sigma_extra_scale,  # [4] HC γ for σ_extra
        prior_H0_sigma,         # [5] Normal σ for H₀
    ])
    H0_prior_mean = H_obs[0]   # centre H₀ prior on first observation

    # ---- Compute T_end and dT/dt for rate prior ----
    T_end_val = 0.0
    dTdt_end_val = 0.0
    if rate_prior is not None:
        if temperature_monthly is None or time_monthly is None:
            raise ValueError(
                "rate_prior requires temperature_monthly and time_monthly "
                "to compute T_end and dT/dt at the evaluation epoch.  "
                "Use the FULL monthly temperature record (not truncated "
                "to the observation period) to avoid boundary effects.")
        # Find temperature at the rate_prior evaluation time
        idx_eval = np.argmin(np.abs(time_monthly - rate_prior.eval_time))
        T_end_val = temperature_monthly[idx_eval]
        # Estimate dT/dt via linear fit over a ±5 yr window.
        # A wide window smooths interannual variability (ENSO) and
        # avoids boundary artefacts if the temperature record ends
        # near eval_time.
        dt_window = 5.0  # years
        mask_window = ((time_monthly >= rate_prior.eval_time - dt_window) &
                       (time_monthly <= rate_prior.eval_time + dt_window))
        if mask_window.sum() >= 12:
            t_w = time_monthly[mask_window]
            T_w = temperature_monthly[mask_window]
            dTdt_end_val = np.polyfit(t_w, T_w, 1)[0]
        else:
            # Narrow fallback (should not happen with full record)
            warnings.warn(
                f"Only {mask_window.sum()} months in ±{dt_window} yr "
                f"window around eval_time={rate_prior.eval_time:.1f}. "
                f"dT/dt estimate may be unreliable.  Pass the full "
                f"monthly temperature record, not the truncated one.")
            mask_narrow = (
                (time_monthly >= rate_prior.eval_time - 2.0) &
                (time_monthly <= rate_prior.eval_time + 2.0))
            if mask_narrow.sum() >= 3:
                dTdt_end_val = np.polyfit(
                    time_monthly[mask_narrow],
                    temperature_monthly[mask_narrow], 1)[0]

    if progress:
        print(f"Bayesian level-space fit: n={n} observations, ndim={ndim}")
        a_prior_str = (f"a~N(0,{prior_scale_a*1e3:.2f} mm/yr/°C²)"
                       if symmetric_a else
                       f"a~Exp(mean={prior_scale_a*1e3:.2f} mm/yr/°C²)")
        print(f"  Priors: {a_prior_str}, "
              f"b~HN({prior_scale_b*1e3:.1f} mm/yr/°C), "
              f"c~N({prior_c_mean*1e3:.1f}, {prior_c_sigma*1e3:.1f} mm/yr), "
              f"σ_extra~HC({prior_sigma_extra_scale*1e3:.1f} mm)")
        if rate_prior is not None:
            print(f"  Rate prior: rate={rate_prior.rate*1e3:.2f} "
                  f"± {rate_prior.rate_se*1e3:.2f} mm/yr, "
                  f"accel={rate_prior.accel*1e6:.2f} "
                  f"± {rate_prior.accel_se*1e6:.2f} μm/yr², "
                  f"T_end={T_end_val:.3f} °C, "
                  f"dT/dt={dTdt_end_val*1e3:.2f} m°C/yr")

    # ---- OLS initialization ----
    X = np.column_stack([I2_obs, I1_obs, I0_obs, np.ones(n)])
    beta_ols = np.linalg.lstsq(X, H_obs, rcond=None)[0]
    a0, b0, c0, H0_0 = beta_ols
    resid_ols = H_obs - X @ beta_ols
    sigma_extra_0 = np.std(resid_ols)

    if progress:
        print(f"  OLS init: a={a0*1e3:.3f}, b={b0*1e3:.3f}, "
              f"c={c0*1e3:.3f} mm/yr, σ_extra={sigma_extra_0*1e3:.2f} mm")

    # ---- Initialize walkers ----
    rng = np.random.default_rng(seed)
    p0_center = np.array([
        a0 if symmetric_a else max(a0, 1e-6),
        max(b0, 1e-6),
        c0,
        np.log(max(sigma_extra_0, 1e-6)),
        H0_0,
    ])
    p0_scale = np.array([
        max(abs(a0) * 0.1, 1e-6),
        max(abs(b0) * 0.1, 1e-6),
        max(abs(c0) * 0.1, 1e-6),
        0.2,
        max(abs(sigma_extra_0), 1e-4),
    ])
    p0 = p0_center[None, :] + p0_scale[None, :] * rng.standard_normal(
        (n_walkers, ndim))
    # Enforce b ≥ 0 always; a ≥ 0 only when not symmetric
    if not symmetric_a:
        p0[:, 0] = np.abs(p0[:, 0])
    p0[:, 1] = np.abs(p0[:, 1])

    # ---- Run emcee ----
    # Pre-compute covariance inverse once (avoids np.linalg.inv per step)
    _rp_cov_inv = (np.linalg.inv(rate_prior.rate_accel_cov)
                   if rate_prior is not None else None)
    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, _level_log_prob,
        args=(I2_obs, I1_obs, I0_obs, H_obs, sigma_obs,
              prior_scales, H0_prior_mean),
        kwargs={'symmetric_a': symmetric_a,
                'rate_prior': rate_prior,
                'T_end': T_end_val,
                'dTdt_end': dTdt_end_val,
                'rate_prior_cov_inv': _rp_cov_inv},
    )
    sampler.run_mcmc(p0, n_burnin + n_samples, progress=progress)

    # ---- Post-process ----
    flat_chain = sampler.get_chain(discard=n_burnin, thin=thin, flat=True)
    # flat_chain columns: [a, b, c, log_sigma_extra, H0]

    phys_samples = flat_chain[:, :3]       # [a, b, c]
    H0_samples = flat_chain[:, 4]
    sigma_extra_samples = np.exp(flat_chain[:, 3])

    phys_mean = phys_samples.mean(axis=0)
    phys_cov = np.cov(phys_samples, rowvar=False)
    phys_hdi = np.array([
        az.hdi(phys_samples[:, k], hdi_prob=0.94) for k in range(3)
    ])

    # R² (posterior mean)
    H_model_mean = (phys_mean[0] * I2_obs + phys_mean[1] * I1_obs
                    + phys_mean[2] * I0_obs + H0_samples.mean())
    resid = H_obs - H_model_mean
    r2 = 1.0 - np.sum(resid**2) / np.sum((H_obs - H_obs.mean())**2)

    # Observation times: I0_obs = t - t[0], relative to start.
    # Caller must track absolute times externally.

    # Build arviz InferenceData
    chain_full = sampler.get_chain(discard=n_burnin, thin=thin, flat=False)
    n_chains_arviz = min(4, n_walkers)
    param_names = ['dalpha_dT', 'alpha0', 'trend', 'log_sigma_extra', 'H0']
    var_dict = {}
    for k, name in enumerate(param_names):
        var_dict[name] = chain_full[:, :n_chains_arviz, k].T

    trace = az.from_dict(var_dict)
    conv = check_convergence(trace, quiet=(not progress))

    diag = {
        'acceptance_fraction': sampler.acceptance_fraction.mean(),
        'n_walkers': n_walkers,
        'n_samples': n_samples,
        'n_burnin': n_burnin,
        'thin': thin,
        'convergence': conv,
    }

    if progress:
        print(f"  Posterior mean: a={phys_mean[0]*1e3:.3f}, "
              f"b={phys_mean[1]*1e3:.3f}, c={phys_mean[2]*1e3:.3f} mm/yr")
        print(f"  σ_extra: median={np.median(sigma_extra_samples)*1e3:.2f} mm"
              f" [{np.percentile(sigma_extra_samples, 3)*1e3:.2f}, "
              f"{np.percentile(sigma_extra_samples, 97)*1e3:.2f}]")
        print(f"  R² = {r2:.4f},  acceptance = "
              f"{diag['acceptance_fraction']:.2f}")

    return BayesianLevelResult(
        trace=trace,
        physical_coefficients=phys_mean,
        physical_covariance=phys_cov,
        physical_hdi_94=phys_hdi,
        posterior_samples=phys_samples,
        H0_posterior=H0_samples,
        sigma_extra_posterior=sigma_extra_samples,
        r2=r2,
        residuals=resid,
        time=I0_obs,    # relative time; caller adds t_start
        H_obs=H_obs,
        H_model_mean=H_model_mean,
        sigma_obs=sigma_obs,
        order=2,
        sampler_diagnostics=diag,
        design_info={
            'prior_scales': prior_scales,
            'param_names': param_names,
            'H0_prior_mean': H0_prior_mean,
        },
    )


# ====================================================================
# Model 6: Bayesian Rate-and-State Level-Space Calibration
# ====================================================================

@dataclass
class BayesianStateLevelResult:
    """Bayesian rate-and-state level-space calibration result.

    Fits the generative model:

        dS/dt  = (T(t) - S(t)) / τ          (state variable ODE)
        rate(t) = a·T(t)² + b·T(t) + c + d·(S(t) - T(t))
        H(t)   = H₀ + ∫ rate(τ) dτ
        H_obs  ~ N(H(t), σ(t)²)

    The disequilibrium term d·(S−T) captures the lag between realised
    and equilibrium SLR rates from slow ocean/ice-sheet response.
    As τ→0, S→T and the model reduces to the instantaneous
    ``BayesianLevelResult`` (d·(S−T) → 0).

    Compatible with ``project_gmsl_state_ensemble()`` via
    ``physical_coefficients`` (4-vector) and ``posterior_samples``.
    """
    trace: az.InferenceData
    physical_coefficients: np.ndarray    # posterior mean [a, b, c, d]
    physical_covariance: np.ndarray      # posterior covariance (4×4)
    physical_hdi_94: np.ndarray          # (4, 2) — 94% HDI
    posterior_samples: np.ndarray        # (n_samples, 4) — [a, b, c, d]
    tau_posterior: np.ndarray            # (n_samples,) — relaxation time (yr)
    H0_posterior: np.ndarray             # (n_samples,) — baseline level
    sigma_extra_posterior: np.ndarray    # (n_samples,) — learned noise
    state_variable_mean: np.ndarray     # S(t) at posterior-mean τ (n_monthly,)
    r2: float                            # R² against observed GMSL
    residuals: np.ndarray                # posterior-mean residuals (m)
    time: np.ndarray                     # observation times (decimal year)
    H_obs: np.ndarray                    # observed GMSL (m)
    H_model_mean: np.ndarray             # posterior-mean predicted GMSL (m)
    sigma_obs: np.ndarray                # σ_obs(t) excl σ_extra (m)
    order: int = 2
    sampler_diagnostics: Optional[dict] = None
    design_info: Optional[dict] = None


def solve_state_ode(
    temperature: np.ndarray,
    time_years: np.ndarray,
    tau: float,
    S0: Optional[float] = None,
) -> np.ndarray:
    """Solve the state-variable ODE dS/dt = (T(t) - S(t)) / τ.

    Uses the analytical exponential solution for each time step,
    treating T as piecewise-constant (average of endpoints) over
    each interval:

        S(t+dt) = T_avg + (S(t) - T_avg) · exp(-dt/τ)

    This is exact for linear interpolation of T between grid points
    and avoids numerical ODE solver overhead.

    Parameters
    ----------
    temperature : np.ndarray, shape (n,)
        Temperature anomaly time series (°C).
    time_years : np.ndarray, shape (n,)
        Time in decimal years.
    tau : float
        Relaxation time in years.  Must be > 0.
        If tau < 0.01, returns T.copy() (τ→0 limit).
    S0 : float or None
        Initial condition for S.  If None, uses T[0]
        (equilibrium assumption at start of record).

    Returns
    -------
    np.ndarray, shape (n,)
        State variable S(t).
    """
    n = len(temperature)
    S = np.empty(n)

    # τ→0 limit: S tracks T instantaneously
    if tau < 0.01:
        return temperature.copy()

    # Initial condition
    S[0] = temperature[0] if S0 is None else S0

    # Step forward using analytical exponential integrator
    for i in range(n - 1):
        dt = time_years[i + 1] - time_years[i]
        T_avg = 0.5 * (temperature[i] + temperature[i + 1])
        decay = np.exp(-dt / tau)
        S[i + 1] = T_avg + (S[i] - T_avg) * decay

    return S


def build_state_level_design_vectors(
    temperature_monthly: np.ndarray,
    time_monthly: np.ndarray,
    obs_times: np.ndarray,
    tau: float,
) -> dict:
    """Build design vectors for the rate-and-state level-space model.

    Extends ``build_level_design_vectors()`` with the state-variable
    integral I_S = ∫(S - T) dτ, which depends on the relaxation time τ.

    For fixed τ, the forward model is linear in (a, b, c, d, H₀):

        H(t) = a·I₂(t) + b·I₁(t) + c·I₀(t) + d·I_S(t) + H₀

    Parameters
    ----------
    temperature_monthly : np.ndarray, shape (n_monthly,)
        Monthly temperature anomaly (°C).
    time_monthly : np.ndarray, shape (n_monthly,)
        Monthly decimal years.
    obs_times : np.ndarray, shape (n_obs,)
        Annual observation times (decimal year).
    tau : float
        Relaxation time in years for the state variable ODE.

    Returns
    -------
    dict with keys:
        'I2_obs', 'I1_obs', 'I0_obs', 'IS_obs' : (n_obs,) — design vectors
        'I2_full', 'I1_full', 'I0_full', 'IS_full' : (n_monthly,) — full grid
        'S_full' : (n_monthly,) — state variable on monthly grid
        'obs_idx' : (n_obs,) — indices into monthly grid
        'tau' : float
        'time_monthly', 'temperature_monthly' : input arrays
    """
    T = temperature_monthly
    t = time_monthly
    dt = np.diff(t)
    n = len(T)

    # Solve state ODE
    S = solve_state_ode(T, t, tau)

    # Disequilibrium: S - T
    diseq = S - T

    # Cumulative trapezoidal integrals from t[0]
    I2 = np.zeros(n)
    I1 = np.zeros(n)
    IS = np.zeros(n)
    for i in range(n - 1):
        I2[i + 1] = I2[i] + 0.5 * (T[i]**2 + T[i + 1]**2) * dt[i]
        I1[i + 1] = I1[i] + 0.5 * (T[i] + T[i + 1]) * dt[i]
        IS[i + 1] = IS[i] + 0.5 * (diseq[i] + diseq[i + 1]) * dt[i]
    I0 = t - t[0]

    # Match observation times to nearest monthly index
    obs_idx = np.array([
        np.argmin(np.abs(t - t_obs)) for t_obs in obs_times
    ])

    return {
        'I2_obs': I2[obs_idx],
        'I1_obs': I1[obs_idx],
        'I0_obs': I0[obs_idx],
        'IS_obs': IS[obs_idx],
        'I2_full': I2,
        'I1_full': I1,
        'I0_full': I0,
        'IS_full': IS,
        'S_full': S,
        'obs_idx': obs_idx,
        'tau': tau,
        'time_monthly': t,
        'temperature_monthly': T,
    }


def _state_level_log_prior(theta, prior_scales, H0_prior_mean):
    """Log-prior for the rate-and-state level-space model.

    Parameters
    ----------
    theta : array of length 7
        [a, b, c, d, log_tau, log_sigma_extra, H0]
    prior_scales : array of length 9
        [0] Exponential mean for a (dα/dT) — PC prior, mode at a=0
        [1] HN σ for b (α₀)
        [2] Normal μ for c
        [3] Normal σ for c
        [4] Exp mean for d (disequilibrium strength)
        [5] LogNormal μ for log(τ) (i.e., log of median τ)
        [6] LogNormal σ for log(τ)
        [7] HC γ for σ_extra
        [8] Normal σ for H₀
    H0_prior_mean : float
        Prior mean for H₀ (typically first observation).
    """
    a, b, c, d, log_tau, log_sigma_extra, H0 = theta

    # Hard bounds: a, b, d ≥ 0
    if a < 0 or b < 0 or d < 0:
        return -np.inf

    sigma_extra = np.exp(log_sigma_extra)
    if sigma_extra < 1e-12:
        return -np.inf

    tau = np.exp(log_tau)
    if tau < 0.1 or tau > 5000.0:
        return -np.inf

    lp = 0.0

    # a ~ Exponential(mean = μ_a)  [PC prior — mode at a=0]
    lp += -a / prior_scales[0]

    # b ~ HalfNormal(σ)
    lp += -0.5 * (b / prior_scales[1])**2

    # c ~ Normal(μ_c, σ_c)
    lp += -0.5 * ((c - prior_scales[2]) / prior_scales[3])**2

    # d ~ Exponential(mean = μ_d)  [PC prior — mode at d=0]
    lp += -d / prior_scales[4]

    # τ ~ LogNormal(μ_log, σ_log)  →  log(τ) ~ Normal(μ_log, σ_log)
    # Sampling in log_tau, so the Jacobian is implicit (we are sampling
    # the log directly, and the LogNormal density in log-space is just Normal)
    lp += -0.5 * ((log_tau - prior_scales[5]) / prior_scales[6])**2

    # σ_extra ~ HalfCauchy(0, γ)  →  log-density + Jacobian for log-sampling
    gamma = prior_scales[7]
    lp += -np.log(1.0 + (sigma_extra / gamma)**2) + log_sigma_extra

    # H₀ ~ Normal(H₀_prior_mean, σ_H0)
    lp += -0.5 * ((H0 - H0_prior_mean) / prior_scales[8])**2

    return lp


def _state_level_log_prob(
    theta,
    T_monthly, time_monthly, obs_idx,
    I2_obs, I1_obs, I0_obs,
    H_obs, sigma_obs_fixed,
    prior_scales, H0_prior_mean,
    rate_prior=None, idx_eval=None, dTdt_end=0.0,
    rate_prior_cov_inv=None,
):
    """Log-posterior for the rate-and-state level-space model.

    Since τ varies at each MCMC step, the state variable S(t) and
    its integral I_S must be recomputed each time.  The other design
    vectors (I₂, I₁, I₀) are independent of τ and passed as constants.

    Parameters
    ----------
    theta : array of length 7
        [a, b, c, d, log_tau, log_sigma_extra, H0]
    T_monthly : (n_monthly,)
        Monthly temperature.
    time_monthly : (n_monthly,)
        Monthly decimal years.
    obs_idx : (n_obs,)
        Indices mapping observation times to monthly grid.
    I2_obs, I1_obs, I0_obs : (n_obs,)
        Pre-computed design vectors (τ-independent).
    H_obs : (n_obs,)
        Observed GMSL (meters).
    sigma_obs_fixed : (n_obs,)
        Time-varying observation uncertainty (meters), excl σ_extra.
    prior_scales : (9,)
        Prior hyperparameters (see ``_state_level_log_prior``).
    H0_prior_mean : float
        Prior mean for H₀.
    rate_prior : SatelliteEraQuadraticResult or None
        If provided, adds a bivariate Gaussian penalty on the
        model-implied (rate, accel) at end of record.
    idx_eval : int or None
        Index into the monthly grid at the rate_prior evaluation time.
    dTdt_end : float
        Temperature trend at end of record (°C/yr).
    rate_prior_cov_inv : (2, 2) array or None
        Pre-computed inverse of ``rate_prior.rate_accel_cov``.
        Avoids redundant matrix inversion at every MCMC step.
    """
    # Prior
    lp = _state_level_log_prior(theta, prior_scales, H0_prior_mean)
    if not np.isfinite(lp):
        return -np.inf

    a, b, c, d, log_tau, log_sigma_extra, H0 = theta
    tau = np.exp(log_tau)
    sigma_extra = np.exp(log_sigma_extra)

    # Solve state ODE on monthly grid (cost: O(n_monthly))
    S = solve_state_ode(T_monthly, time_monthly, tau)
    diseq = S - T_monthly

    # Compute I_S = ∫(S - T) dτ on monthly grid, then extract at obs times
    n = len(T_monthly)
    dt_monthly = np.diff(time_monthly)
    IS = np.empty(n)
    IS[0] = 0.0
    IS[1:] = np.cumsum(0.5 * (diseq[:-1] + diseq[1:]) * dt_monthly)

    IS_obs = IS[obs_idx]

    # Forward model: H = a·I₂ + b·I₁ + c·I₀ + d·I_S + H₀
    H_model = a * I2_obs + b * I1_obs + c * I0_obs + d * IS_obs + H0

    # Likelihood
    sigma_total = np.sqrt(sigma_obs_fixed**2 + sigma_extra**2)
    resid = H_obs - H_model
    ll = -0.5 * np.sum((resid / sigma_total)**2 + 2.0 * np.log(sigma_total))

    if not np.isfinite(ll):
        return -np.inf

    # Satellite-era rate/accel prior
    if rate_prior is not None and idx_eval is not None:
        T_end = T_monthly[idx_eval]
        S_end = S[idx_eval]
        # rate(t) = a·T² + b·T + c + d·(S − T)
        rate_model = a * T_end**2 + b * T_end + c + d * (S_end - T_end)
        # d(rate)/dt = (2a·T + b)·dT/dt + d·(dS/dt − dT/dt)
        # where dS/dt = (T − S)/τ
        dSdt_end = (T_end - S_end) / tau
        accel_model = ((2.0 * a * T_end + b) * dTdt_end
                       + d * (dSdt_end - dTdt_end))
        lp_rate = _rate_accel_prior_logp(rate_model, accel_model,
                                         rate_prior, rate_prior_cov_inv)
        if not np.isfinite(lp_rate):
            return -np.inf
        return lp + ll + lp_rate

    return lp + ll


def fit_bayesian_state_level(
    H_obs: np.ndarray,
    sigma_obs: np.ndarray,
    I2_obs: np.ndarray,
    I1_obs: np.ndarray,
    I0_obs: np.ndarray,
    T_monthly: np.ndarray,
    time_monthly: np.ndarray,
    obs_idx: np.ndarray,
    prior_scale_a: float = 0.010,
    prior_scale_b: float = 0.010,
    prior_c_mean: float = 0.002,
    prior_c_sigma: float = 0.005,
    prior_scale_d: float = 0.005,
    prior_log_tau_mean: float = None,    # default: log(20)
    prior_log_tau_sigma: float = 0.7,
    prior_sigma_extra_scale: float = 0.005,
    prior_H0_sigma: float = 0.050,
    n_samples: int = 5000,
    n_walkers: int = 64,
    n_burnin: int = 3000,
    thin: int = 1,
    progress: bool = True,
    seed: Optional[int] = None,
    init_from_level: Optional['BayesianLevelResult'] = None,
    init_order: int = 2,
    rate_prior: Optional['SatelliteEraQuadraticResult'] = None,
) -> BayesianStateLevelResult:
    """Bayesian rate-and-state level-space calibration.

    Extends the instantaneous Bayesian level-space model with a state
    variable S(t) that relaxes toward temperature with timescale τ:

        dS/dt   = (T(t) − S(t)) / τ
        rate(t) = a·T² + b·T + c + d·(S − T)
        H(t)    = H₀ + ∫ rate dτ
        H_obs   ~ N(H(t), σ(t)²)

    As τ → 0, S → T and d·(S−T) → 0, recovering the instantaneous model.

    Parameters
    ----------
    H_obs : (n,) observed GMSL (meters)
    sigma_obs : (n,) time-varying σ (meters), excl σ_extra
    I2_obs, I1_obs, I0_obs : (n,) pre-computed design vectors
    T_monthly : (n_monthly,) monthly temperature
    time_monthly : (n_monthly,) monthly decimal years
    obs_idx : (n_obs,) indices mapping obs to monthly grid
    prior_scale_a : Exponential mean for a (m/yr/°C²); PC prior, mode at a=0
    prior_scale_b : HalfNormal σ for b (m/yr/°C)
    prior_c_mean, prior_c_sigma : Normal prior for c (m/yr)
    prior_scale_d : Exponential mean for d (m/yr/°C); PC prior, mode at d=0
    prior_log_tau_mean : LogNormal log-mean for τ (default log(20) ≈ 3.0)
    prior_log_tau_sigma : LogNormal log-σ for τ (default 0.7)
    prior_sigma_extra_scale : HalfCauchy γ for σ_extra (meters)
    prior_H0_sigma : Normal σ for H₀ (meters)
    n_samples, n_walkers, n_burnin, thin : MCMC settings
    progress : show emcee progress bar
    seed : random seed
    init_from_level : BayesianLevelResult, optional
        If provided, use the instantaneous model posterior for
        initialization of (a, b, c, σ_extra, H₀), with d near 0.
        This is STRONGLY recommended — the OLS initialization for
        the 5-regressor model with I_S is ill-conditioned and
        typically gives infeasible negative d and b.
    init_order : {1, 2}, default 2
        Order of the initialisation model.  When init_order=1 and
        init_from_level is provided, a₀ is overridden to ≈0 so that
        walkers start in the linear subspace.  This is consistent
        with the PC prior philosophy (start at the simpler model)
        and lets the MCMC decide whether a, d, or both are needed.
    rate_prior : SatelliteEraQuadraticResult or None
        If provided, adds a bivariate Gaussian penalty on the
        model-implied (rate, accel) at end of record.

    Returns
    -------
    BayesianStateLevelResult
    """
    if prior_log_tau_mean is None:
        prior_log_tau_mean = np.log(20.0)

    n = len(H_obs)
    ndim = 7  # [a, b, c, d, log_tau, log_sigma_extra, H0]

    prior_scales = np.array([
        prior_scale_a,            # [0] Exp mean for a
        prior_scale_b,            # [1] HN σ for b
        prior_c_mean,             # [2] Normal μ for c
        prior_c_sigma,            # [3] Normal σ for c
        prior_scale_d,            # [4] Exp mean for d
        prior_log_tau_mean,       # [5] LogNormal log-mean
        prior_log_tau_sigma,      # [6] LogNormal log-σ
        prior_sigma_extra_scale,  # [7] HC γ for σ_extra
        prior_H0_sigma,           # [8] Normal σ for H₀
    ])
    H0_prior_mean = H_obs[0]

    if progress:
        tau_median = np.exp(prior_log_tau_mean)
        tau_lo = np.exp(prior_log_tau_mean - 1.96 * prior_log_tau_sigma)
        tau_hi = np.exp(prior_log_tau_mean + 1.96 * prior_log_tau_sigma)
        print(f"Bayesian rate-and-state fit: n={n} obs, ndim={ndim}")
        print(f"  Priors: a~Exp(mean={prior_scale_a*1e3:.2f} mm/yr/°C²), "
              f"b~HN({prior_scale_b*1e3:.1f} mm/yr/°C), "
              f"c~N({prior_c_mean*1e3:.1f}, {prior_c_sigma*1e3:.1f} mm/yr)")
        print(f"  d~Exp(mean={prior_scale_d*1e3:.2f} mm/yr/°C), "
              f"τ~LogN(median={tau_median:.0f} yr, "
              f"95% CI [{tau_lo:.0f}, {tau_hi:.0f}] yr)")
        print(f"  σ_extra~HC({prior_sigma_extra_scale*1e3:.1f} mm)")

    # ---- Compute rate prior quantities ----
    # Note: idx_eval_rp indexes into T_monthly (the calibration-period
    # array used for the ODE solve).  dT/dt is estimated from a ±5 yr
    # window centred on eval_time using the SAME array; if this array
    # ends near eval_time, the user should extend T_monthly to cover
    # at least ±5 yr beyond eval_time, or pass a longer temperature
    # record.
    idx_eval_rp = None
    dTdt_end_rp = 0.0
    if rate_prior is not None:
        idx_eval_rp = int(np.argmin(
            np.abs(time_monthly - rate_prior.eval_time)))
        # dT/dt via linear fit over ±5 yr window
        dt_window = 5.0
        mask_w = ((time_monthly >= rate_prior.eval_time - dt_window) &
                  (time_monthly <= rate_prior.eval_time + dt_window))
        if mask_w.sum() >= 12:
            dTdt_end_rp = np.polyfit(
                time_monthly[mask_w], T_monthly[mask_w], 1)[0]
        else:
            warnings.warn(
                f"Only {mask_w.sum()} months in ±{dt_window} yr "
                f"window for dT/dt at {rate_prior.eval_time:.1f}. "
                f"Extend T_monthly beyond the calibration period.")
            mask_narrow = (
                (time_monthly >= rate_prior.eval_time - 2.0) &
                (time_monthly <= rate_prior.eval_time + 2.0))
            if mask_narrow.sum() >= 3:
                dTdt_end_rp = np.polyfit(
                    time_monthly[mask_narrow],
                    T_monthly[mask_narrow], 1)[0]
        if progress:
            print(f"  Rate prior: rate={rate_prior.rate*1e3:.2f} "
                  f"± {rate_prior.rate_se*1e3:.2f} mm/yr, "
                  f"accel={rate_prior.accel*1e6:.2f} "
                  f"± {rate_prior.accel_se*1e6:.2f} μm/yr², "
                  f"T_end={T_monthly[idx_eval_rp]:.3f} °C, "
                  f"dT/dt={dTdt_end_rp*1e3:.2f} m°C/yr")

    tau_init = np.exp(prior_log_tau_mean)

    # ---- Initialization strategy ----
    #
    # The 5-regressor OLS (I₂, I₁, I₀, I_S, 1) is severely ill-conditioned
    # because I_S is collinear with I₁ and I₀.  OLS typically yields negative
    # d and b, which violates the physical priors.
    #
    # Preferred strategy: initialize from a converged instantaneous model
    # (d=0 limit), then let the MCMC explore whether d > 0 improves the fit.

    if init_from_level is not None:
        # Use the instantaneous model posterior means
        a0 = init_from_level.physical_coefficients[0]
        b0 = init_from_level.physical_coefficients[1]
        c0 = init_from_level.physical_coefficients[2]
        d0 = 1e-4  # small positive — near the d=0 boundary
        H0_0 = init_from_level.H0_posterior.mean()
        sigma_extra_0 = np.median(init_from_level.sigma_extra_posterior)

        if init_order == 1:
            # Override a → near-zero: start in the linear subspace.
            # Consistent with PC prior (mode at a=0) and lets the
            # MCMC decide whether a or d (or both) explain nonlinearity.
            a0 = 1e-5
            init_method = 'from instantaneous model (order=1, a≈0)'
        else:
            init_method = 'from instantaneous model'

        if progress:
            print(f"  Init ({init_method}): a={a0*1e3:.3f}, "
                  f"b={b0*1e3:.3f}, c={c0*1e3:.3f}, d={d0*1e3:.3f} mm/yr, "
                  f"σ_extra={sigma_extra_0*1e3:.2f} mm")
    else:
        # OLS fallback with safeguards
        dv = build_state_level_design_vectors(
            T_monthly, time_monthly,
            time_monthly[obs_idx],
            tau_init,
        )
        X = np.column_stack([
            dv['I2_obs'], dv['I1_obs'], dv['I0_obs'],
            dv['IS_obs'], np.ones(n)
        ])
        beta_ols = np.linalg.lstsq(X, H_obs, rcond=None)[0]
        a0_ols, b0_ols, c0_ols, d0_ols, H0_ols = beta_ols
        resid_ols = H_obs - X @ beta_ols
        se_ols = max(np.std(resid_ols), 1e-4)

        if progress:
            print(f"  OLS (τ={tau_init:.0f}yr): a={a0_ols*1e3:.3f}, "
                  f"b={b0_ols*1e3:.3f}, c={c0_ols*1e3:.3f}, "
                  f"d={d0_ols*1e3:.3f} mm/yr")

        # If OLS gives infeasible negatives, fall back to 3-regressor OLS
        # (instantaneous model) + d=0
        if d0_ols < 0 or b0_ols < 0:
            X3 = np.column_stack([
                dv['I2_obs'], dv['I1_obs'], dv['I0_obs'], np.ones(n)
            ])
            beta3 = np.linalg.lstsq(X3, H_obs, rcond=None)[0]
            a0, b0, c0 = beta3[0], beta3[1], beta3[2]
            d0 = 1e-4
            H0_0 = beta3[3]
            resid3 = H_obs - X3 @ beta3
            sigma_extra_0 = max(np.std(resid3), 1e-4)
            init_method = 'from 3-regressor OLS (I_S collinearity fallback)'
            if progress:
                print(f"  OLS 5-reg infeasible (d<0 or b<0) → "
                      f"using 3-reg: a={a0*1e3:.3f}, b={b0*1e3:.3f}, "
                      f"c={c0*1e3:.3f}, d=0 mm/yr")
        else:
            a0, b0, c0, d0, H0_0 = beta_ols
            sigma_extra_0 = se_ols
            init_method = 'from 5-regressor OLS'

    # Enforce positivity for init values
    a0 = max(a0, 1e-5)
    b0 = max(b0, 1e-5)
    d0 = max(d0, 1e-5)
    sigma_extra_0 = max(sigma_extra_0, 1e-5)

    # ---- Initialize walkers in a tight ball ----
    rng = np.random.default_rng(seed)
    p0_center = np.array([
        a0,
        b0,
        c0,
        d0,
        np.log(tau_init),
        np.log(sigma_extra_0),
        H0_0,
    ])

    # Perturbation scales: small fraction of init values or prior scales
    # When init_order=1, use wider perturbation for a (starting near 0)
    # so walkers span a meaningful range from the start.
    a_pert = (prior_scale_a * 0.3 if init_order == 1
              else max(a0 * 0.05, prior_scale_a * 0.05))
    p0_scale = np.array([
        a_pert,
        max(b0 * 0.05, prior_scale_b * 0.05),
        max(abs(c0) * 0.05, prior_c_sigma * 0.05),
        max(d0 * 0.5, prior_scale_d * 0.02),  # wider for d since d≈0
        prior_log_tau_sigma * 0.15,
        0.1,
        max(sigma_extra_0 * 0.5, 0.001),
    ])

    p0 = p0_center[None, :] + p0_scale[None, :] * rng.standard_normal(
        (n_walkers, ndim))
    # Enforce a, b, d ≥ small positive
    p0[:, 0] = np.abs(p0[:, 0]) + 1e-6
    p0[:, 1] = np.abs(p0[:, 1]) + 1e-6
    p0[:, 3] = np.abs(p0[:, 3]) + 1e-7  # keep d near 0

    if progress:
        print(f"  Walker init: {init_method}, {n_walkers} walkers")

    # ---- Run emcee ----
    # Pre-compute covariance inverse once (avoids np.linalg.inv per step)
    _rp_cov_inv = (np.linalg.inv(rate_prior.rate_accel_cov)
                   if rate_prior is not None else None)
    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, _state_level_log_prob,
        args=(T_monthly, time_monthly, obs_idx,
              I2_obs, I1_obs, I0_obs,
              H_obs, sigma_obs,
              prior_scales, H0_prior_mean),
        kwargs={'rate_prior': rate_prior,
                'idx_eval': idx_eval_rp,
                'dTdt_end': dTdt_end_rp,
                'rate_prior_cov_inv': _rp_cov_inv},
    )
    sampler.run_mcmc(p0, n_burnin + n_samples, progress=progress)

    # ---- Post-process ----
    flat_chain = sampler.get_chain(discard=n_burnin, thin=thin, flat=True)
    # columns: [a, b, c, d, log_tau, log_sigma_extra, H0]

    phys_samples = flat_chain[:, :4]        # [a, b, c, d]
    tau_samples = np.exp(flat_chain[:, 4])
    sigma_extra_samples = np.exp(flat_chain[:, 5])
    H0_samples = flat_chain[:, 6]

    phys_mean = phys_samples.mean(axis=0)
    phys_cov = np.cov(phys_samples, rowvar=False)
    phys_hdi = np.array([
        az.hdi(phys_samples[:, k], hdi_prob=0.94) for k in range(4)
    ])

    # State variable at posterior-mean τ
    tau_mean = tau_samples.mean()
    S_mean = solve_state_ode(T_monthly, time_monthly, tau_mean)

    # Posterior-mean prediction
    dv_mean = build_state_level_design_vectors(
        T_monthly, time_monthly,
        time_monthly[obs_idx],
        tau_mean,
    )
    H_model_mean = (phys_mean[0] * dv_mean['I2_obs']
                    + phys_mean[1] * dv_mean['I1_obs']
                    + phys_mean[2] * dv_mean['I0_obs']
                    + phys_mean[3] * dv_mean['IS_obs']
                    + H0_samples.mean())
    resid = H_obs - H_model_mean
    r2 = 1.0 - np.sum(resid**2) / np.sum((H_obs - H_obs.mean())**2)

    # Build arviz InferenceData
    chain_full = sampler.get_chain(discard=n_burnin, thin=thin, flat=False)
    n_chains_arviz = min(4, n_walkers)
    param_names = ['dalpha_dT', 'alpha0', 'trend', 'd_diseq',
                   'log_tau', 'log_sigma_extra', 'H0']
    var_dict = {}
    for k, name in enumerate(param_names):
        var_dict[name] = chain_full[:, :n_chains_arviz, k].T

    trace = az.from_dict(var_dict)
    conv = check_convergence(trace, quiet=(not progress))

    diag = {
        'acceptance_fraction': sampler.acceptance_fraction.mean(),
        'n_walkers': n_walkers,
        'n_samples': n_samples,
        'n_burnin': n_burnin,
        'thin': thin,
        'convergence': conv,
    }

    if progress:
        print(f"\n  Posterior mean: a={phys_mean[0]*1e3:.3f}, "
              f"b={phys_mean[1]*1e3:.3f}, c={phys_mean[2]*1e3:.3f}, "
              f"d={phys_mean[3]*1e3:.3f} mm/yr")
        print(f"  τ: median={np.median(tau_samples):.1f} yr "
              f"[{np.percentile(tau_samples, 3):.1f}, "
              f"{np.percentile(tau_samples, 97):.1f}]")
        print(f"  σ_extra: median={np.median(sigma_extra_samples)*1e3:.2f} mm"
              f" [{np.percentile(sigma_extra_samples, 3)*1e3:.2f}, "
              f"{np.percentile(sigma_extra_samples, 97)*1e3:.2f}]")
        print(f"  R² = {r2:.4f},  acceptance = "
              f"{diag['acceptance_fraction']:.2f}")

    return BayesianStateLevelResult(
        trace=trace,
        physical_coefficients=phys_mean,
        physical_covariance=phys_cov,
        physical_hdi_94=phys_hdi,
        posterior_samples=phys_samples,
        tau_posterior=tau_samples,
        H0_posterior=H0_samples,
        sigma_extra_posterior=sigma_extra_samples,
        state_variable_mean=S_mean,
        r2=r2,
        residuals=resid,
        time=I0_obs,
        H_obs=H_obs,
        H_model_mean=H_model_mean,
        sigma_obs=sigma_obs,
        order=2,
        sampler_diagnostics=diag,
        design_info={
            'prior_scales': prior_scales,
            'param_names': param_names,
            'H0_prior_mean': H0_prior_mean,
            'tau_mean': tau_mean,
        },
    )


# ====================================================================
# Model 7: Physically-Motivated Thermosteric Model
# ====================================================================
#
# Physics:  The thermal expansion coefficient α increases with
# temperature (TEOS-10): α(T) ≈ α₀ + α₁·T.  Steric sea level is
# therefore quadratic in ocean temperature:
#
#     η(t) = a·S_u² + b_u·S_u + b_d·S_d + c·(t − t₀) + H₀
#
# where S_u and S_d are upper- and deep-ocean temperature states
# that lag GMST via cascade ODEs:
#
#     dS_u/dt = (T(t) − S_u) / τ_u     (upper ocean ← GMST)
#     dS_d/dt = (S_u − S_d)   / τ_d     (deep ocean ← upper ocean)
#
# As τ_u → 0 and τ_d → ∞, this reduces to the instantaneous model
# η = a·T² + b·T + c·t + H₀.


def solve_twolayer_ode(
    temperature: np.ndarray,
    time_years: np.ndarray,
    tau_u: float,
    tau_d: float = np.inf,
    Su0: Optional[float] = None,
    Sd0: Optional[float] = None,
) -> tuple:
    """Solve the two-layer cascade ODE for ocean temperature.

    The one-way-coupled system is:

        dS_u/dt = (T(t) − S_u) / τ_u
        dS_d/dt = (S_u  − S_d) / τ_d

    Solved analytically with the exponential integrator, treating the
    forcing as piecewise-linear over each time step.

    Parameters
    ----------
    temperature : (n,)  GMST anomaly time series (°C).
    time_years  : (n,)  Decimal years.
    tau_u : float  Upper-ocean relaxation time (years).
    tau_d : float  Deep-ocean relaxation time (years).
              If >= 1e6, the deep layer is frozen (single-layer model).
    Su0, Sd0 : Initial conditions.  Default = T[0] (equilibrium).

    Returns
    -------
    (S_u, S_d) : tuple of np.ndarray, each shape (n,).
    """
    n = len(temperature)
    T = temperature
    t = time_years
    S_u = np.empty(n)
    S_d = np.empty(n)

    # ── Initial conditions ──
    su0 = T[0] if Su0 is None else Su0
    sd0 = T[0] if Sd0 is None else Sd0
    S_u[0] = su0
    S_d[0] = sd0

    # ── τ_u → 0 limit: S_u tracks T instantaneously ──
    if tau_u < 0.01:
        S_u[:] = T.copy()
    else:
        for i in range(n - 1):
            dt = t[i + 1] - t[i]
            T_avg = 0.5 * (T[i] + T[i + 1])
            decay = np.exp(-dt / tau_u)
            S_u[i + 1] = T_avg + (S_u[i] - T_avg) * decay

    # ── Deep layer: driven by S_u ──
    if tau_d >= 1e6:
        # Single-layer limit: deep ocean frozen
        S_d[:] = sd0
    elif tau_d < 0.01:
        # τ_d → 0: deep layer tracks upper layer
        S_d[:] = S_u.copy()
    else:
        for i in range(n - 1):
            dt = t[i + 1] - t[i]
            Su_avg = 0.5 * (S_u[i] + S_u[i + 1])
            decay = np.exp(-dt / tau_d)
            S_d[i + 1] = Su_avg + (S_d[i] - Su_avg) * decay

    return S_u, S_d


@dataclass
class BayesianThermostericResult:
    """Physically-motivated thermosteric calibration result.

    Fits the generative model:

        dS_u/dt = (T − S_u) / τ_u
        dS_d/dt = (S_u − S_d) / τ_d          (two-layer only)
        η(t) = a·S_u² + b_u·S_u [+ b_d·S_d] + c·(t−t₀) + H₀
        η_obs ~ N(η, σ_obs² + σ_extra²)

    Physical interpretation:
        a   = (α₁/2)·h_eff   curvature from α(T) temperature dependence
        b_u = α₀·h_u          upper-ocean baseline expansion coefficient
        b_d = α₀·h_d          deep-ocean baseline expansion coefficient
        c   = non-thermal secular drift (halosteric, dynamics)

    As τ_u → 0, S_u → T and the model reduces to the instantaneous
    quadratic η = a·T² + b·T + c·t + H₀.

    When subsurface ocean T observations are provided, a second likelihood
    constrains S_u directly:

        T_subsurface(t) = κ·S_u(t) + δ + ε_ocean
    """
    trace: az.InferenceData
    physical_coefficients: np.ndarray    # [a, b, c] (1-layer) or [a, b_u, b_d, c] (2-layer)
    physical_covariance: np.ndarray
    physical_hdi_94: np.ndarray
    posterior_samples: np.ndarray        # (n_samples, n_phys)
    tau_u_posterior: np.ndarray          # (n_samples,)
    tau_d_posterior: Optional[np.ndarray]  # (n_samples,) or None
    H0_posterior: np.ndarray
    sigma_extra_posterior: np.ndarray
    S_u_mean: np.ndarray                 # S_u(t) at posterior-mean τ_u (n_monthly,)
    S_d_mean: Optional[np.ndarray]       # S_d(t) or None (single-layer)
    r2: float
    residuals: np.ndarray
    time: np.ndarray                     # relative observation times (yr)
    H_obs: np.ndarray
    H_model_mean: np.ndarray
    sigma_obs: np.ndarray
    n_layers: int = 1
    sampler_diagnostics: Optional[dict] = None
    design_info: Optional[dict] = None
    # ── Ocean T joint calibration (optional) ──
    kappa_posterior: Optional[np.ndarray] = None   # (n_samples,) scaling S_u → T_subsurface
    delta_posterior: Optional[np.ndarray] = None   # (n_samples,) offset
    sigma_ocean_posterior: Optional[np.ndarray] = None  # (n_samples,) extra ocean T noise
    T_ocean_obs: Optional[np.ndarray] = None       # observed subsurface T at ocean times
    T_ocean_model: Optional[np.ndarray] = None     # posterior-mean κ·S_u + δ at ocean times
    r2_ocean: Optional[float] = None               # R² for ocean T fit


def _thermosteric_log_prior(theta, prior_scales, H0_prior_mean, n_layers=1):
    """Log-prior for the physically-motivated thermosteric model.

    Single-layer (n_layers=1):
        theta = [a, b, c, log_tau_u, log_sigma_extra, H0]   (6 params)
        prior_scales = [mu_a, sigma_b, mu_c, sigma_c,
                        mu_log_tau_u, sigma_log_tau_u,
                        gamma_sigma, sigma_H0]               (8 elements)

    Two-layer (n_layers=2):
        theta = [a, b_u, b_d, c, log_tau_u, log_tau_d,
                 log_sigma_extra, H0]                        (8 params)
        prior_scales = [mu_a, sigma_bu, sigma_bd, mu_c, sigma_c,
                        mu_log_tau_u, sigma_log_tau_u,
                        mu_log_tau_d, sigma_log_tau_d,
                        gamma_sigma, sigma_H0]               (11 elements)
    """
    if n_layers == 1:
        a, b, c, log_tau_u, log_sigma_extra, H0 = theta
        if a < 0 or b < 0:
            return -np.inf
    else:
        a, b_u, b_d, c, log_tau_u, log_tau_d, log_sigma_extra, H0 = theta
        if a < 0 or b_u < 0 or b_d < 0:
            return -np.inf

    sigma_extra = np.exp(log_sigma_extra)
    if sigma_extra < 1e-12:
        return -np.inf

    tau_u = np.exp(log_tau_u)
    if tau_u < 0.1 or tau_u > 500.0:
        return -np.inf

    lp = 0.0

    if n_layers == 1:
        mu_a, sigma_b, mu_c, sigma_c = prior_scales[0:4]
        mu_lt_u, sigma_lt_u = prior_scales[4:6]
        gamma_sigma, sigma_H0 = prior_scales[6:8]

        # a ~ Exponential(mean = mu_a), PC prior
        lp += -a / mu_a
        # b ~ HalfNormal(sigma_b)
        lp += -0.5 * (b / sigma_b)**2
        # c ~ Normal(mu_c, sigma_c)
        lp += -0.5 * ((c - mu_c) / sigma_c)**2
        # tau_u ~ LogNormal  (log_tau_u ~ Normal)
        lp += -0.5 * ((log_tau_u - mu_lt_u) / sigma_lt_u)**2
    else:
        mu_a = prior_scales[0]
        sigma_bu, sigma_bd = prior_scales[1:3]
        mu_c, sigma_c = prior_scales[3:5]
        mu_lt_u, sigma_lt_u = prior_scales[5:7]
        mu_lt_d, sigma_lt_d = prior_scales[7:9]
        gamma_sigma, sigma_H0 = prior_scales[9:11]

        tau_d = np.exp(log_tau_d)
        if tau_d < 1.0 or tau_d > 5000.0:
            return -np.inf

        lp += -a / mu_a
        lp += -0.5 * (b_u / sigma_bu)**2
        lp += -0.5 * (b_d / sigma_bd)**2
        lp += -0.5 * ((c - mu_c) / sigma_c)**2
        lp += -0.5 * ((log_tau_u - mu_lt_u) / sigma_lt_u)**2
        lp += -0.5 * ((log_tau_d - mu_lt_d) / sigma_lt_d)**2

    # sigma_extra ~ HalfCauchy(0, gamma) + Jacobian
    lp += -np.log(1.0 + (sigma_extra / gamma_sigma)**2) + log_sigma_extra
    # H0 ~ Normal
    lp += -0.5 * ((H0 - H0_prior_mean) / sigma_H0)**2

    return lp


def _thermosteric_log_prob(
    theta,
    T_avg_ann, dt_ann, Su0, obs_idx_ann, n_ann,
    I0_obs,
    H_obs, sigma_obs_fixed,
    prior_scales, H0_prior_mean,
    n_layers=1,
    Sd0=None,
    ocean_obs=None, ocean_sigma=None, ocean_idx_ann=None,
    ocean_prior_scales=None,
):
    """Log-posterior for the physically-motivated thermosteric model.

    Optimised for MCMC: uses pre-computed annual T_avg and dt arrays
    with an inline ODE solve (~120 steps instead of ~1400 monthly).

    Forward model (level-space, no integration needed):
        η = a·S_u² + b·S_u [+ b_d·S_d] + c·I₀ + H₀

    When ocean_obs is provided, theta has 3 extra trailing elements
    [κ, δ, log_σ_ocean] and a second likelihood constrains S_u:
        T_subsurface(t) = κ·S_u(t) + δ
    """
    use_ocean = ocean_obs is not None
    n_base = 6 if n_layers == 1 else 8

    if use_ocean:
        theta_base = theta[:n_base]
        kappa, delta, log_sig_ocean = theta[n_base:]
        sigma_ocean_extra = np.exp(log_sig_ocean)
        if sigma_ocean_extra < 1e-12:
            return -np.inf
        if log_sig_ocean > 2 or log_sig_ocean < -20:
            return -np.inf
    else:
        theta_base = theta

    lp = _thermosteric_log_prior(theta_base, prior_scales, H0_prior_mean,
                                 n_layers)
    if not np.isfinite(lp):
        return -np.inf

    # ── Ocean T priors (κ, δ, σ_ocean) ──
    if use_ocean:
        # ocean_prior_scales = [mu_kappa, sigma_kappa, sigma_delta, gamma_ocean]
        mu_k, sig_k, sig_d, gamma_oc = ocean_prior_scales
        lp += -0.5 * ((kappa - mu_k) / sig_k) ** 2
        lp += -0.5 * (delta / sig_d) ** 2
        lp += (-np.log(1.0 + (sigma_ocean_extra / gamma_oc) ** 2)
               + log_sig_ocean)

    if n_layers == 1:
        a, b, c, log_tau_u, log_sigma_extra, H0 = theta_base
        tau_u = np.exp(log_tau_u)
        sigma_extra = np.exp(log_sigma_extra)

        # ── Inline ODE solve on annual grid ──
        Su = np.empty(n_ann)
        Su[0] = Su0
        if tau_u < 0.01:
            # instantaneous limit — S_u tracks temperature directly.
            # T_avg_ann[i] ≈ T at midpoint; best available in this scope.
            Su[1:] = T_avg_ann
        else:
            s = Su0
            inv_tau = 1.0 / tau_u
            for i in range(n_ann - 1):
                decay = np.exp(-dt_ann[i] * inv_tau)
                t_avg = T_avg_ann[i]
                s = t_avg + (s - t_avg) * decay
                Su[i + 1] = s

        Su_obs = Su[obs_idx_ann]
        H_model = a * Su_obs**2 + b * Su_obs + c * I0_obs + H0
    else:
        a, b_u, b_d, c, log_tau_u, log_tau_d, log_sigma_extra, H0 = theta_base
        tau_u = np.exp(log_tau_u)
        tau_d = np.exp(log_tau_d)
        sigma_extra = np.exp(log_sigma_extra)

        # ── S_u: inline ODE on annual grid ──
        Su = np.empty(n_ann)
        Su[0] = Su0
        if tau_u < 0.01:
            Su[1:] = T_avg_ann
        else:
            s = Su0
            inv_tau_u = 1.0 / tau_u
            for i in range(n_ann - 1):
                decay = np.exp(-dt_ann[i] * inv_tau_u)
                t_avg = T_avg_ann[i]
                s = t_avg + (s - t_avg) * decay
                Su[i + 1] = s

        # ── S_d: inline ODE on annual grid ──
        sd0 = Su0 if Sd0 is None else Sd0
        Sd = np.empty(n_ann)
        Sd[0] = sd0
        if tau_d >= 1e6:
            Sd[:] = sd0
        elif tau_d < 0.01:
            Sd[:] = Su
        else:
            s_d = sd0
            inv_tau_d = 1.0 / tau_d
            for i in range(n_ann - 1):
                su_avg = 0.5 * (Su[i] + Su[i + 1])
                decay = np.exp(-dt_ann[i] * inv_tau_d)
                s_d = su_avg + (s_d - su_avg) * decay
                Sd[i + 1] = s_d

        Su_obs = Su[obs_idx_ann]
        Sd_obs = Sd[obs_idx_ann]
        H_model = (a * Su_obs**2 + b_u * Su_obs
                    + b_d * Sd_obs + c * I0_obs + H0)

    sigma_total = np.sqrt(sigma_obs_fixed**2 + sigma_extra**2)
    resid = H_obs - H_model
    ll = -0.5 * np.sum(
        (resid / sigma_total)**2 + 2.0 * np.log(sigma_total)
    )
    if not np.isfinite(ll):
        return -np.inf

    # ── Ocean T likelihood (joint calibration) ──
    if use_ocean:
        Su_ocean = Su[ocean_idx_ann]
        T_model_ocean = kappa * Su_ocean + delta
        sig_oc = np.sqrt(ocean_sigma**2 + sigma_ocean_extra**2)
        resid_oc = ocean_obs - T_model_ocean
        ll_oc = -0.5 * np.sum(
            (resid_oc / sig_oc)**2 + 2.0 * np.log(sig_oc)
        )
        if not np.isfinite(ll_oc):
            return -np.inf
        ll += ll_oc

    return lp + ll


def fit_bayesian_thermosteric(
    H_obs: np.ndarray,
    sigma_obs: np.ndarray,
    T_monthly: np.ndarray,
    time_monthly: np.ndarray,
    obs_times: np.ndarray,
    n_layers: int = 1,
    # ── Priors ──
    prior_scale_a: float = 0.22,
    prior_scale_b: float = 0.15,
    prior_scale_b_d: float = 0.05,
    prior_c_mean: float = 0.0003,
    prior_c_sigma: float = 0.0005,
    prior_log_tau_u_mean: Optional[float] = None,
    prior_log_tau_u_sigma: float = 0.5,
    prior_log_tau_d_mean: Optional[float] = None,
    prior_log_tau_d_sigma: float = 0.5,
    prior_sigma_extra_scale: float = 0.003,
    prior_H0_sigma: float = 0.010,
    # ── Subsurface ocean T joint calibration (optional) ──
    T_ocean_obs: Optional[np.ndarray] = None,
    sigma_ocean_obs: Optional[np.ndarray] = None,
    time_ocean_obs: Optional[np.ndarray] = None,
    prior_kappa_mean: float = 0.5,
    prior_kappa_sigma: float = 0.5,
    prior_delta_sigma: float = 0.3,
    prior_sigma_ocean_scale: float = 0.1,
    # ── MCMC settings ──
    n_samples: int = 5000,
    n_walkers: int = 64,
    n_burnin: int = 3000,
    thin: int = 1,
    progress: bool = True,
    seed: Optional[int] = None,
) -> BayesianThermostericResult:
    """Physically-motivated Bayesian thermosteric calibration.

    Fits steric sea level as a quadratic function of effective ocean
    temperature (capturing the temperature dependence of the thermal
    expansion coefficient) with a cascade ODE for ocean thermal lag:

        dS_u/dt = (T − S_u) / τ_u
        dS_d/dt = (S_u − S_d) / τ_d          (two-layer only)
        η(t) = a·S_u² + b_u·S_u [+ b_d·S_d] + c·(t−t₀) + H₀

    As τ_u → 0, S_u → T and the model reduces to η = a·T² + b·T + c·t + H₀.

    Parameters
    ----------
    H_obs : (n,)  Observed steric sea level (meters), rebased.
    sigma_obs : (n,)  Time-varying σ (meters), excl σ_extra.
    T_monthly : (n_monthly,)  Monthly GMST anomaly (°C).
    time_monthly : (n_monthly,)  Monthly decimal years.
    obs_times : (n_obs,)  Annual observation times (decimal years).
    n_layers : {1, 2}  Number of ocean layers.
    prior_scale_a : Exponential mean for a (m/°C²); PC prior.
    prior_scale_b : HalfNormal σ for b_u (m/°C).
    prior_scale_b_d : HalfNormal σ for b_d (m/°C), two-layer only.
    prior_c_mean, prior_c_sigma : Normal prior for c (m/yr).
    prior_log_tau_u_mean : LogNormal log-mean for τ_u (default log(8)).
    prior_log_tau_u_sigma : LogNormal log-σ for τ_u.
    prior_log_tau_d_mean : LogNormal log-mean for τ_d (default log(150)).
    prior_log_tau_d_sigma : LogNormal log-σ for τ_d.
    prior_sigma_extra_scale : HalfCauchy γ (meters).
    prior_H0_sigma : Normal σ for H₀ (meters).

    Returns
    -------
    BayesianThermostericResult
    """
    if prior_log_tau_u_mean is None:
        prior_log_tau_u_mean = np.log(8.0)
    if prior_log_tau_d_mean is None:
        prior_log_tau_d_mean = np.log(150.0)

    n = len(H_obs)
    t_monthly = time_monthly

    # ── Monthly → annual aggregation for fast MCMC ──
    # The ODE at annual resolution is accurate for τ ≥ 1 yr and avoids
    # the 12× overhead of monthly stepping in the Python for-loop.
    year_floor = np.floor(t_monthly).astype(int)
    unique_years = np.unique(year_floor)
    T_annual = np.array([np.mean(T_monthly[year_floor == yr])
                         for yr in unique_years])
    time_annual = unique_years.astype(float) + 0.5  # mid-year
    n_ann = len(time_annual)
    dt_ann = np.diff(time_annual)
    T_avg_ann = 0.5 * (T_annual[:-1] + T_annual[1:])
    Su0_ann = T_annual[0]  # equilibrium initial condition

    # ── Observation indices (on annual grid) ──
    obs_idx_ann = np.array([
        np.argmin(np.abs(time_annual - t_obs)) for t_obs in obs_times
    ])
    # Also keep monthly obs_idx for post-processing
    obs_idx_monthly = np.array([
        np.argmin(np.abs(t_monthly - t_obs)) for t_obs in obs_times
    ])
    I0_obs = obs_times - obs_times[0]

    # ── Ocean T setup (optional joint calibration) ──
    use_ocean = T_ocean_obs is not None
    ocean_obs_ann = None
    ocean_sigma_ann = None
    ocean_idx_ann = None
    ocean_prior_scales = None

    if use_ocean:
        # Annualize ocean T observations
        oc_yr = np.floor(time_ocean_obs).astype(int)
        oc_unique = np.unique(oc_yr)
        ocean_T_ann = np.array([T_ocean_obs[oc_yr == y].mean()
                                for y in oc_unique])
        ocean_sig_ann = np.array([
            np.sqrt(np.mean(sigma_ocean_obs[oc_yr == y]**2))
            for y in oc_unique])
        ocean_time_ann = oc_unique.astype(float) + 0.5
        # Map to annual grid
        ocean_idx_ann = np.array([
            np.argmin(np.abs(time_annual - t)) for t in ocean_time_ann
        ])
        ocean_obs_ann = ocean_T_ann
        ocean_sigma_ann = ocean_sig_ann
        ocean_prior_scales = np.array([
            prior_kappa_mean, prior_kappa_sigma,
            prior_delta_sigma, prior_sigma_ocean_scale,
        ])

    if progress:
        print(f"  Annual grid: {n_ann} points ({time_annual[0]:.0f}–"
              f"{time_annual[-1]:.0f}), "
              f"monthly: {len(t_monthly)} points")
        if use_ocean:
            print(f"  Ocean T obs: {len(ocean_obs_ann)} annual pts "
                  f"({ocean_time_ann[0]:.0f}–{ocean_time_ann[-1]:.0f})")

    # ── Prior scales ──
    H0_prior_mean = H_obs[0]

    if n_layers == 1:
        ndim = 6
        prior_scales = np.array([
            prior_scale_a, prior_scale_b,
            prior_c_mean, prior_c_sigma,
            prior_log_tau_u_mean, prior_log_tau_u_sigma,
            prior_sigma_extra_scale, prior_H0_sigma,
        ])
        param_names = ['a_therm', 'b_therm', 'c_therm',
                        'log_tau_u', 'log_sigma_extra', 'H0']
    else:
        ndim = 8
        prior_scales = np.array([
            prior_scale_a, prior_scale_b, prior_scale_b_d,
            prior_c_mean, prior_c_sigma,
            prior_log_tau_u_mean, prior_log_tau_u_sigma,
            prior_log_tau_d_mean, prior_log_tau_d_sigma,
            prior_sigma_extra_scale, prior_H0_sigma,
        ])
        param_names = ['a_therm', 'b_u_therm', 'b_d_therm', 'c_therm',
                        'log_tau_u', 'log_tau_d', 'log_sigma_extra', 'H0']

    n_base = ndim
    if use_ocean:
        ndim += 3
        param_names = param_names + ['kappa', 'delta', 'log_sigma_ocean']

    # ── OLS initialization at fixed τ (using monthly for accuracy) ──
    tau_u_init = np.exp(prior_log_tau_u_mean)
    if n_layers == 2:
        tau_d_init = np.exp(prior_log_tau_d_mean)
    else:
        tau_d_init = np.inf

    S_u_init, S_d_init = solve_twolayer_ode(
        T_monthly, t_monthly, tau_u_init, tau_d_init
    )
    Su_obs = S_u_init[obs_idx_monthly]
    Sd_obs = S_d_init[obs_idx_monthly]

    if n_layers == 1:
        X_init = np.column_stack([Su_obs**2, Su_obs, I0_obs,
                                  np.ones(n)])
    else:
        X_init = np.column_stack([Su_obs**2, Su_obs, Sd_obs, I0_obs,
                                  np.ones(n)])

    # Weighted least squares
    W = np.diag(1.0 / sigma_obs**2)
    try:
        beta_ols = np.linalg.solve(X_init.T @ W @ X_init,
                                   X_init.T @ W @ H_obs)
    except np.linalg.LinAlgError:
        beta_ols = np.linalg.lstsq(X_init, H_obs, rcond=None)[0]

    resid_ols = H_obs - X_init @ beta_ols
    sigma_extra_init = max(np.std(resid_ols), 1e-4)

    if n_layers == 1:
        a_init = max(beta_ols[0], 1e-6)
        b_init = max(beta_ols[1], 1e-6)
        c_init = beta_ols[2]
        H0_init = beta_ols[3]

        theta0 = np.array([
            a_init, b_init, c_init,
            np.log(tau_u_init),
            np.log(sigma_extra_init),
            H0_init,
        ])
    else:
        a_init = max(beta_ols[0], 1e-6)
        bu_init = max(beta_ols[1], 1e-6)
        bd_init = max(beta_ols[2], 1e-6)
        c_init = beta_ols[3]
        H0_init = beta_ols[4]

        theta0 = np.array([
            a_init, bu_init, bd_init, c_init,
            np.log(tau_u_init), np.log(tau_d_init),
            np.log(sigma_extra_init),
            H0_init,
        ])

    # ── Ocean T OLS initialization: κ, δ from S_u vs ocean T ──
    if use_ocean:
        Su_at_ocean = S_u_init[np.array([
            np.argmin(np.abs(t_monthly - t)) for t in ocean_time_ann
        ])]
        X_oc = np.column_stack([Su_at_ocean, np.ones(len(ocean_obs_ann))])
        beta_oc = np.linalg.lstsq(X_oc, ocean_obs_ann, rcond=None)[0]
        kappa_init = beta_oc[0]
        delta_init = beta_oc[1]
        sig_oc_init = max(np.std(ocean_obs_ann - X_oc @ beta_oc), 0.01)
        theta0 = np.append(theta0, [kappa_init, delta_init,
                                     np.log(sig_oc_init)])

    if progress:
        print(f"  OLS init: a={a_init:.4f} m/°C², "
              f"b={'%.4f' % b_init if n_layers==1 else '%.4f' % bu_init} m/°C, "
              f"c={c_init*1e3:.3f} mm/yr, τ_u={tau_u_init:.1f} yr")
        if use_ocean:
            print(f"  OLS init (ocean): κ={kappa_init:.3f}, "
                  f"δ={delta_init:.4f} °C, "
                  f"σ_ocean={sig_oc_init:.3f} °C")

    # ── Walker initialization ──
    rng = np.random.default_rng(seed)
    pos = np.empty((n_walkers, ndim))
    for i in range(n_walkers):
        p = theta0 * (1.0 + 0.05 * rng.standard_normal(ndim))
        # Enforce positivity for a, b
        if n_layers == 1:
            p[0] = abs(p[0])  # a
            p[1] = abs(p[1])  # b
        else:
            p[0] = abs(p[0])  # a
            p[1] = abs(p[1])  # b_u
            p[2] = abs(p[2])  # b_d
        # Ocean params: κ can be any sign, δ any sign, log_σ_ocean perturbed
        pos[i] = p

    # ── MCMC (uses annual-resolution ODE for ~12× speedup) ──
    moves = [
        (emcee.moves.DESnookerMove(), 0.8),
        (emcee.moves.DEMove(),        0.2),
    ]
    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, _thermosteric_log_prob,
        args=(T_avg_ann, dt_ann, Su0_ann, obs_idx_ann, n_ann,
              I0_obs, H_obs, sigma_obs, prior_scales, H0_prior_mean),
        kwargs={
            'n_layers': n_layers,
            'ocean_obs': ocean_obs_ann,
            'ocean_sigma': ocean_sigma_ann,
            'ocean_idx_ann': ocean_idx_ann,
            'ocean_prior_scales': ocean_prior_scales,
        },
        moves=moves,
    )

    if progress:
        print(f"  Running emcee: {n_walkers} walkers, "
              f"{n_burnin} burn-in + {n_samples} production "
              f"({n_layers}-layer, {ndim} params)...")

    sampler.run_mcmc(pos, n_burnin + n_samples, progress=progress)

    # ── Extract chains ──
    flat = sampler.get_chain(discard=n_burnin, thin=thin, flat=True)

    if n_layers == 1:
        phys_samples = flat[:, :3]         # [a, b, c]
        tau_u_samples = np.exp(flat[:, 3])
        tau_d_samples = None
        sigma_extra_samples = np.exp(flat[:, 4])
        H0_samples = flat[:, 5]
    else:
        phys_samples = flat[:, :4]         # [a, b_u, b_d, c]
        tau_u_samples = np.exp(flat[:, 4])
        tau_d_samples = np.exp(flat[:, 5])
        sigma_extra_samples = np.exp(flat[:, 6])
        H0_samples = flat[:, 7]

    if use_ocean:
        kappa_samples = flat[:, n_base]
        delta_samples = flat[:, n_base + 1]
        sigma_ocean_samples = np.exp(flat[:, n_base + 2])
    else:
        kappa_samples = None
        delta_samples = None
        sigma_ocean_samples = None

    phys_mean = np.mean(phys_samples, axis=0)
    phys_cov = np.cov(phys_samples, rowvar=False)
    phys_hdi = np.column_stack([
        az.hdi(phys_samples[:, k], hdi_prob=0.94)
        for k in range(phys_samples.shape[1])
    ]).T

    # ── Posterior-mean prediction (full monthly resolution) ──
    tau_u_mean = np.median(tau_u_samples)
    tau_d_mean = np.median(tau_d_samples) if tau_d_samples is not None else np.inf
    S_u_mean, S_d_mean = solve_twolayer_ode(
        T_monthly, t_monthly, tau_u_mean, tau_d_mean
    )
    Su_obs_mean = S_u_mean[obs_idx_monthly]
    Sd_obs_mean = S_d_mean[obs_idx_monthly] if n_layers == 2 else None

    if n_layers == 1:
        H_model_mean = (phys_mean[0] * Su_obs_mean**2
                        + phys_mean[1] * Su_obs_mean
                        + phys_mean[2] * I0_obs
                        + np.mean(H0_samples))
    else:
        H_model_mean = (phys_mean[0] * Su_obs_mean**2
                        + phys_mean[1] * Su_obs_mean
                        + phys_mean[2] * Sd_obs_mean
                        + phys_mean[3] * I0_obs
                        + np.mean(H0_samples))

    resid = H_obs - H_model_mean
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((H_obs - np.mean(H_obs))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # ── Ocean T posterior-mean prediction ──
    T_ocean_model_mean = None
    r2_ocean = None
    if use_ocean:
        Su_at_oc_monthly = S_u_mean[np.array([
            np.argmin(np.abs(t_monthly - t)) for t in ocean_time_ann
        ])]
        kappa_m = np.mean(kappa_samples)
        delta_m = np.mean(delta_samples)
        T_ocean_model_mean = kappa_m * Su_at_oc_monthly + delta_m
        ss_res_oc = np.sum((ocean_obs_ann - T_ocean_model_mean)**2)
        ss_tot_oc = np.sum((ocean_obs_ann - np.mean(ocean_obs_ann))**2)
        r2_ocean = 1.0 - ss_res_oc / ss_tot_oc if ss_tot_oc > 0 else 0.0

    # ── arviz trace ──
    n_post = len(flat)
    n_chains_arviz = min(4, n_walkers)
    chain_full = sampler.get_chain(discard=n_burnin, thin=thin)
    var_dict = {}
    for k, name in enumerate(param_names):
        var_dict[name] = chain_full[:, :n_chains_arviz, k].T

    trace = az.from_dict(var_dict)
    conv = check_convergence(trace, quiet=(not progress))

    diag = {
        'acceptance_fraction': sampler.acceptance_fraction.mean(),
        'n_walkers': n_walkers,
        'n_samples': n_samples,
        'n_burnin': n_burnin,
        'thin': thin,
        'n_layers': n_layers,
        'convergence': conv,
    }

    if progress:
        if n_layers == 1:
            print(f"\n  Posterior mean: a={phys_mean[0]:.4f} m/°C², "
                  f"b={phys_mean[1]:.4f} m/°C, "
                  f"c={phys_mean[2]*1e3:.3f} mm/yr")
        else:
            print(f"\n  Posterior mean: a={phys_mean[0]:.4f} m/°C², "
                  f"b_u={phys_mean[1]:.4f} m/°C, "
                  f"b_d={phys_mean[2]:.4f} m/°C, "
                  f"c={phys_mean[3]*1e3:.3f} mm/yr")
        print(f"  τ_u: median={np.median(tau_u_samples):.1f} yr "
              f"[{np.percentile(tau_u_samples, 3):.1f}, "
              f"{np.percentile(tau_u_samples, 97):.1f}]")
        if tau_d_samples is not None:
            print(f"  τ_d: median={np.median(tau_d_samples):.0f} yr "
                  f"[{np.percentile(tau_d_samples, 3):.0f}, "
                  f"{np.percentile(tau_d_samples, 97):.0f}]")
        print(f"  σ_extra: median={np.median(sigma_extra_samples)*1e3:.2f} mm")
        print(f"  R² = {r2:.4f},  acceptance = "
              f"{diag['acceptance_fraction']:.2f}")
        # Physics check: a/b ratio
        b_col = 1 if n_layers == 1 else 1  # b or b_u
        ratio = phys_samples[:, 0] / np.maximum(phys_samples[:, b_col], 1e-8)
        print(f"  a/b ratio: median={np.median(ratio):.4f} °C⁻¹  "
              f"(TEOS-10 expectation ≈ 0.033 °C⁻¹)")
        if use_ocean:
            print(f"\n  Ocean T joint calibration:")
            print(f"    κ = {np.mean(kappa_samples):.3f} "
                  f"[{np.percentile(kappa_samples, 3):.3f}, "
                  f"{np.percentile(kappa_samples, 97):.3f}] "
                  f"°C_subsurface/°C_Su")
            print(f"    δ = {np.mean(delta_samples):.4f} "
                  f"[{np.percentile(delta_samples, 3):.4f}, "
                  f"{np.percentile(delta_samples, 97):.4f}] °C")
            print(f"    σ_ocean = {np.median(sigma_ocean_samples):.3f} °C")
            print(f"    R²_ocean = {r2_ocean:.4f}")

    return BayesianThermostericResult(
        trace=trace,
        physical_coefficients=phys_mean,
        physical_covariance=phys_cov,
        physical_hdi_94=phys_hdi,
        posterior_samples=phys_samples,
        tau_u_posterior=tau_u_samples,
        tau_d_posterior=tau_d_samples,
        H0_posterior=H0_samples,
        sigma_extra_posterior=sigma_extra_samples,
        S_u_mean=S_u_mean,
        S_d_mean=S_d_mean if n_layers == 2 else None,
        r2=r2,
        residuals=resid,
        time=I0_obs,
        H_obs=H_obs,
        H_model_mean=H_model_mean,
        sigma_obs=sigma_obs,
        n_layers=n_layers,
        sampler_diagnostics=diag,
        design_info={
            'prior_scales': prior_scales,
            'param_names': param_names,
            'H0_prior_mean': H0_prior_mean,
            'n_layers': n_layers,
        },
        kappa_posterior=kappa_samples,
        delta_posterior=delta_samples,
        sigma_ocean_posterior=sigma_ocean_samples,
        T_ocean_obs=ocean_obs_ann,
        T_ocean_model=T_ocean_model_mean,
        r2_ocean=r2_ocean,
    )


# ════════════════════════════════════════════════════════════════════
#  STAGE 2b-physical:  Physics-informed Greenland Ice Sheet model
# ════════════════════════════════════════════════════════════════════
#
#  Two-component model separating SMB and ice-dynamic discharge:
#
#      H_gris(t) = H_smb(t) + H_dyn(t) + c·(t−t₀) + H₀
#
#  SMB component (integral of rate):
#      rate_smb(t) = a_eff·T(t)² + b_eff·T(t)
#      H_smb(t) = a_eff·I₂(t) + b_eff·I₁(t)
#
#  where a_eff = β_smb·f²_arctic and b_eff = α_smb·f_arctic absorb
#  the polar amplification factor during calibration.
#
#  Discharge component (lagged response):
#      dD_eff/dt = (T(t) − D_eff) / τ_dyn       relaxation ODE
#      H_dyn(t) = γ · ∫₀ᵗ D_eff(τ) dτ           cumulative discharge
#
#  D_eff tracks GMST with lag τ_dyn (~10-50 yr), capturing both
#  atmospheric (meltwater lubrication, surface lowering) and oceanic
#  (submarine melt) forcing pathways.
#
#  Limiting behaviours:
#  - τ_dyn → 0: D_eff → T, H_dyn → γ·I₁, recovering polynomial
#  - γ → 0: pure SMB model, H = a_eff·I₂ + b_eff·I₁ + c·I₀ + H₀
#
#  Validation targets (Aschwanden et al. 2019, calibrated 2022):
#  - SSP1-2.6 ≈ RCP 2.6:  8 [4, 14] cm
#  - SSP5-8.5 ≈ RCP 8.5: 19 [9, 29] cm
#  - Discharge fraction: 8-45% of total


@dataclass
class BayesianGreenlandResult:
    """Physics-informed Greenland calibration result.

    Fits the two-component generative model:

        rate_smb = a_eff·T² + b_eff·T      (SMB, immediate)
        dD_eff/dt = (T − D_eff) / τ_dyn    (discharge ODE)
        H = a_eff·I₂ + b_eff·I₁ + γ·∫D_eff + c·I₀ + H₀
        H_obs ~ N(H, σ_obs² + σ_extra²)

    Physical interpretation:
        a_eff  = β_smb·f²_arctic     melt-elevation feedback (mm/yr/°C²)
        b_eff  = α_smb·f_arctic      SMB sensitivity × amplification
        γ      = discharge sensitivity (m/yr per °C-equivalent)
        τ_dyn  = ice-dynamic response time (outlet glaciers)
        c      = secular drift (background, GIA)
    """
    trace: az.InferenceData
    physical_coefficients: np.ndarray    # [a_eff, b_eff, γ, c]
    physical_covariance: np.ndarray
    physical_hdi_94: np.ndarray
    posterior_samples: np.ndarray         # (n_samples, 4)
    tau_dyn_posterior: np.ndarray          # (n_samples,)
    H0_posterior: np.ndarray
    sigma_extra_posterior: np.ndarray
    D_eff_mean: np.ndarray                # (n_monthly,) at posterior-median τ_dyn
    H_smb_mean: np.ndarray               # (n_obs,) SMB component at posterior mean
    H_dyn_mean: np.ndarray               # (n_obs,) discharge component at post. mean
    r2: float
    residuals: np.ndarray
    time: np.ndarray                      # observation times (decimal yr)
    H_obs: np.ndarray
    H_model_mean: np.ndarray
    sigma_obs: np.ndarray
    sampler_diagnostics: Optional[dict] = None
    design_info: Optional[dict] = None


def _greenland_log_prior(theta, prior_scales, H0_prior_mean):
    """Log-prior for the physics-informed Greenland model.

    Reparameterized: the sampler explores (b_total, f_dyn) instead
    of (b_eff, γ) to break the ridge degeneracy between SMB and
    discharge linear sensitivities.

    Parameters
    ----------
    theta : array_like, length 7
        [a_eff, b_total, f_dyn, log_τ_dyn, c, log_σ_extra, H₀]
        where b_eff = b_total × (1 − f_dyn), γ = b_total × f_dyn
    prior_scales : array_like, length 11
        [μ_a, σ_b_total, α_fdyn, β_fdyn, μ_logτ, σ_logτ,
         μ_c, σ_c, γ_σextra, σ_H0, _reserved]
    H0_prior_mean : float
    """
    a_eff, b_total, f_dyn, log_tau_dyn, c, log_sigma_extra, H0 = theta

    # Hard bounds
    if a_eff < 0 or b_total < 0:
        return -np.inf
    if f_dyn <= 0.0 or f_dyn >= 1.0:
        return -np.inf

    sigma_extra = np.exp(log_sigma_extra)
    if sigma_extra < 1e-8 or sigma_extra > 0.1:
        return -np.inf

    tau_dyn = np.exp(log_tau_dyn)
    if tau_dyn < 0.5 or tau_dyn > 500.0:
        return -np.inf

    mu_a          = prior_scales[0]
    sigma_b_total = prior_scales[1]
    alpha_fdyn    = prior_scales[2]
    beta_fdyn     = prior_scales[3]
    mu_lt, sigma_lt = prior_scales[4], prior_scales[5]
    mu_c, sigma_c   = prior_scales[6], prior_scales[7]
    gamma_sigma = prior_scales[8]
    sigma_H0    = prior_scales[9]

    lp = 0.0
    # a_eff ~ Exponential(mean = μ_a), PC prior
    lp += -a_eff / mu_a
    # b_total ~ HalfNormal(σ_b_total)
    lp += -0.5 * (b_total / sigma_b_total)**2
    # f_dyn ~ Beta(α, β)
    lp += (alpha_fdyn - 1.0) * np.log(f_dyn) + (beta_fdyn - 1.0) * np.log(1.0 - f_dyn)
    # τ_dyn ~ LogNormal  (log_tau_dyn ~ Normal)
    lp += -0.5 * ((log_tau_dyn - mu_lt) / sigma_lt)**2
    # c ~ Normal(μ_c, σ_c)
    lp += -0.5 * ((c - mu_c) / sigma_c)**2
    # σ_extra ~ HalfCauchy(0, γ_σ) + Jacobian for log-transform
    lp += -np.log(1.0 + (sigma_extra / gamma_sigma)**2) + log_sigma_extra
    # H₀ ~ Normal(H0_prior_mean, σ_H0)
    lp += -0.5 * ((H0 - H0_prior_mean) / sigma_H0)**2

    return lp


def _greenland_log_prob(
    theta,
    I2_obs, I1_obs, I0_obs,
    T_avg_ann, dt_ann, T0_ann, obs_idx_ann, n_ann,
    H_obs, sigma_obs_fixed,
    prior_scales, H0_prior_mean,
    rate_prior_mean=None, rate_prior_sigma=0.0, T_end=0.0,
    budget=None, budget_I2=None, budget_I1=None, budget_I0=None,
    budget_idx_ann=None,
):
    """Log-posterior for the physics-informed Greenland model.

    Reparameterized: theta = [a_eff, b_total, f_dyn, log_τ_dyn, c,
    log_σ_extra, H₀].  Physical parameters recovered as:
        b_eff = b_total × (1 − f_dyn)
        γ     = b_total × f_dyn

    Forward model:
        H_smb = a_eff·I₂ + b_eff·I₁
        dD/dt = (T − D) / τ_dyn → ∫D via trapezoidal rule
        H = H_smb + γ·∫D + c·I₀ + H₀

    Parameters
    ----------
    rate_prior_mean : float or None
        Observed Greenland SLR rate at end of record (m/yr).
        If None, no rate prior is applied.
    rate_prior_sigma : float
        1-σ uncertainty on the rate prior (m/yr).
    T_end : float
        Temperature anomaly at end of record (°C).  Only used when
        rate_prior_mean is not None.
    budget : BudgetTarget or None
        If provided, adds level-space and rate+accel budget closure
        constraints (Pass 2 of two-pass approach).
    budget_I2, budget_I1, budget_I0 : (n_budget,) or None
        Design vectors at budget times.
    budget_idx_ann : (n_budget,) int or None
        Indices into the annual grid for budget times.
    """
    lp = _greenland_log_prior(theta, prior_scales, H0_prior_mean)
    if not np.isfinite(lp):
        return -np.inf

    a_eff, b_total, f_dyn, log_tau_dyn, c, log_sigma_extra, H0 = theta
    tau_dyn = np.exp(log_tau_dyn)
    sigma_extra = np.exp(log_sigma_extra)

    # Recover physical parameters
    b_eff = b_total * (1.0 - f_dyn)
    gamma_dyn = b_total * f_dyn

    # ── SMB component (pre-computed design vectors, O(n_obs)) ──
    H_smb_obs = a_eff * I2_obs + b_eff * I1_obs

    # ── Discharge ODE + cumulative integral on annual grid ──
    d = T0_ann
    hdc = 0.0
    inv_tau = 1.0 / max(tau_dyn, 1e-6)
    H_dyn_cum = np.empty(n_ann)
    H_dyn_cum[0] = 0.0

    for i in range(n_ann - 1):
        decay = np.exp(-dt_ann[i] * inv_tau)
        d_new = T_avg_ann[i] + (d - T_avg_ann[i]) * decay
        hdc += gamma_dyn * 0.5 * (d + d_new) * dt_ann[i]
        H_dyn_cum[i + 1] = hdc
        d = d_new

    H_dyn_obs = H_dyn_cum[obs_idx_ann]

    # ── Total model ──
    H_model = H_smb_obs + H_dyn_obs + c * I0_obs + H0

    # ── Likelihood ──
    sigma_total = np.sqrt(sigma_obs_fixed**2 + sigma_extra**2)
    resid = H_obs - H_model
    ll = -0.5 * np.sum(
        (resid / sigma_total)**2 + 2.0 * np.log(sigma_total)
    )
    if not np.isfinite(ll):
        return -np.inf

    total = lp + ll

    # ── Rate prior: penalise model rate that deviates from observed ──
    if rate_prior_mean is not None and rate_prior_sigma > 0:
        # D_eff at end of annual grid is `d` after the ODE loop
        rate_model = a_eff * T_end**2 + b_eff * T_end + gamma_dyn * d + c
        lp_rate = -0.5 * ((rate_model - rate_prior_mean) / rate_prior_sigma)**2
        if not np.isfinite(lp_rate):
            return -np.inf
        total += lp_rate

    # ── Budget closure constraints (Pass 2) ──
    if budget is not None and budget_I2 is not None:
        # Level constraint: model at budget times vs budget target
        H_smb_budget = a_eff * budget_I2 + b_eff * budget_I1
        H_dyn_budget = H_dyn_cum[budget_idx_ann]
        H_budget_model = H_smb_budget + H_dyn_budget + c * budget_I0 + H0

        ll_budget = _budget_level_logp(H_budget_model, budget)
        if not np.isfinite(ll_budget):
            return -np.inf
        total += ll_budget

        # Rate + acceleration constraint (if available in budget)
        if budget.rate_accel is not None:
            rate_model_b = (a_eff * T_end**2 + b_eff * T_end
                            + gamma_dyn * d + c)
            # For acceleration, need dT/dt — approximate from budget
            # rate_accel. Use the bivariate penalty on rate only
            # (accel requires dT/dt which we store in the budget).
            lp_ra = _rate_accel_prior_logp(
                rate_model_b, 0.0, budget.rate_accel)
            if not np.isfinite(lp_ra):
                return -np.inf
            total += lp_ra

    return total


def _greenland_log_prob_fixc(
    theta6,
    I2_obs, I1_obs, I0_obs,
    T_avg_ann, dt_ann, T0_ann, obs_idx_ann, n_ann,
    H_obs, sigma_obs_fixed,
    prior_scales, H0_prior_mean,
    fix_c_val=0.0,
    rate_prior_mean=None, rate_prior_sigma=0.0, T_end=0.0,
    budget=None, budget_I2=None, budget_I1=None, budget_I0=None,
    budget_idx_ann=None,
):
    """Wrapper for fixed-c Greenland model.

    Maps 6-parameter theta (no c) into the 7-parameter log-posterior
    by inserting the fixed c value.

    theta6 = [a_eff, b_total, f_dyn, log_τ_dyn, log_σ_extra, H₀]
    """
    theta7 = np.array([
        theta6[0], theta6[1], theta6[2], theta6[3],
        fix_c_val, theta6[4], theta6[5],
    ])
    return _greenland_log_prob(
        theta7,
        I2_obs, I1_obs, I0_obs,
        T_avg_ann, dt_ann, T0_ann, obs_idx_ann, n_ann,
        H_obs, sigma_obs_fixed,
        prior_scales, H0_prior_mean,
        rate_prior_mean=rate_prior_mean,
        rate_prior_sigma=rate_prior_sigma,
        T_end=T_end,
        budget=budget,
        budget_I2=budget_I2,
        budget_I1=budget_I1,
        budget_I0=budget_I0,
        budget_idx_ann=budget_idx_ann,
    )


def fit_bayesian_greenland(
    H_obs: np.ndarray,
    sigma_obs: np.ndarray,
    I2_obs: np.ndarray,
    I1_obs: np.ndarray,
    I0_obs: np.ndarray,
    T_monthly: np.ndarray,
    time_monthly: np.ndarray,
    obs_times: np.ndarray,
    # ── Priors ──
    prior_scale_a: float = 0.00186,
    prior_scale_b_total: float = 0.004,
    prior_fdyn_alpha: float = 5.0,
    prior_fdyn_beta: float = 5.0,
    prior_log_tau_mean: Optional[float] = None,
    prior_log_tau_sigma: float = 0.6,
    prior_c_mean: float = 0.0002,
    prior_c_sigma: float = 0.00015,
    prior_sigma_extra_scale: float = 0.002,
    prior_H0_sigma: float = 0.005,
    # ── Rate prior ──
    rate_prior_mean: Optional[float] = None,
    rate_prior_sigma: float = 0.0,
    # ── Structural options ──
    fix_c: Optional[float] = None,
    T0_init: Optional[float] = None,
    # ── Budget closure (Pass 2) ──
    budget: Optional['BudgetTarget'] = None,
    # ── MCMC settings ──
    n_samples: int = 5000,
    n_walkers: int = 64,
    n_burnin: int = 3000,
    thin: int = 1,
    progress: bool = True,
    seed: Optional[int] = None,
) -> BayesianGreenlandResult:
    """Physics-informed Bayesian Greenland ice sheet calibration.

    Fits a two-component model separating surface mass balance (SMB)
    and ice-dynamic discharge:

        H_gris = H_smb + H_dyn + c·(t−t₀) + H₀

    SMB (instantaneous, quadratic in T):
        rate_smb = a_eff·T² + b_eff·T
        H_smb = a_eff·I₂ + b_eff·I₁

    Discharge (lagged, ODE):
        dD_eff/dt = (T − D_eff) / τ_dyn
        H_dyn = γ · ∫₀ᵗ D_eff(τ) dτ

    Reparameterized: the sampler explores (b_total, f_dyn) where
    b_total = b_eff + γ and f_dyn = γ / b_total, so that the
    well-constrained total linear sensitivity and the weakly-
    constrained discharge fraction are axis-aligned.  Physical
    parameters are recovered as b_eff = b_total·(1−f_dyn) and
    γ = b_total·f_dyn.

    Parameters
    ----------
    H_obs : (n,)  Observed Greenland SLR (meters), rebased.
    sigma_obs : (n,)  Time-varying σ (meters), excl σ_extra.
    I2_obs, I1_obs, I0_obs : (n,)  Pre-computed design vectors from
        ``build_level_design_vectors``.
    T_monthly : (n_monthly,)  Monthly GMST anomaly (°C).
    time_monthly : (n_monthly,)  Monthly decimal years.
    obs_times : (n,)  Annual observation times (decimal years).
    prior_scale_a : Exponential mean for a_eff (m/yr/°C²); PC prior.
    prior_scale_b_total : HalfNormal σ for b_total = b_eff + γ (m/yr/°C).
    prior_fdyn_alpha, prior_fdyn_beta : Beta(α, β) prior on f_dyn.
    prior_log_tau_mean : LogNormal log-mean for τ_dyn (default log(20)).
    prior_log_tau_sigma : LogNormal log-σ for τ_dyn.
    prior_c_mean, prior_c_sigma : Normal prior for c (m/yr).
    prior_sigma_extra_scale : HalfCauchy γ (meters).
    prior_H0_sigma : Normal σ for H₀ (meters).
    rate_prior_mean : float or None
        Observed Greenland SLR rate at end of record (m/yr).
        If None, no rate prior is applied.  Computed from Mankoff
        or GRACE data as -MB / 362500.
    rate_prior_sigma : float
        1-σ uncertainty on the rate prior (m/yr).
    fix_c : float or None
        If not None, fix the secular trend c to this value (m/yr)
        and remove it as a free parameter (ndim drops to 6).
        Useful to prevent c from absorbing temperature-driven signal.
        Typical GIA value: ~0.05e-3 m/yr.
    T0_init : float or None
        Initial condition for the discharge ODE (D_eff at t=0).
        If None, uses the mean of the first 30 years of annual
        temperature (pre-industrial equilibrium).  Set to 0.0 to
        initialise at the anomaly baseline.

    Returns
    -------
    BayesianGreenlandResult
    """
    if prior_log_tau_mean is None:
        prior_log_tau_mean = np.log(20.0)

    n = len(H_obs)

    # ── Monthly → annual aggregation for discharge ODE ──
    year_floor = np.floor(time_monthly).astype(int)
    unique_years = np.unique(year_floor)
    T_annual = np.array([np.mean(T_monthly[year_floor == yr])
                         for yr in unique_years])
    time_annual = unique_years.astype(float) + 0.5  # mid-year
    n_ann = len(time_annual)
    dt_ann = np.diff(time_annual)
    T_avg_ann = 0.5 * (T_annual[:-1] + T_annual[1:])
    # ODE initial condition: D_eff(t=0)
    if T0_init is not None:
        T0_ann = T0_init
    else:
        # Default: mean of first 30 years (pre-industrial equilibrium)
        n_spinup = min(30, n_ann)
        T0_ann = float(np.mean(T_annual[:n_spinup]))

    # Observation indices on annual grid
    obs_idx_ann = np.array([
        np.argmin(np.abs(time_annual - t_obs)) for t_obs in obs_times
    ])
    # Monthly indices for post-processing (full resolution)
    dt_monthly = np.diff(time_monthly)
    obs_idx_monthly = np.array([
        np.argmin(np.abs(time_monthly - t_obs)) for t_obs in obs_times
    ])

    if progress:
        print(f"  Annual grid: {n_ann} points ({time_annual[0]:.0f}–"
              f"{time_annual[-1]:.0f}), "
              f"monthly: {len(time_monthly)} points")

    # ── Prior scales (reparameterized) ──
    H0_prior_mean = H_obs[0]

    prior_scales = np.array([
        prior_scale_a,            # [0] Exponential mean for a_eff
        prior_scale_b_total,      # [1] HalfNormal σ for b_total
        prior_fdyn_alpha,         # [2] Beta α for f_dyn
        prior_fdyn_beta,          # [3] Beta β for f_dyn
        prior_log_tau_mean,       # [4] LogNormal log-mean for τ_dyn
        prior_log_tau_sigma,      # [5] LogNormal log-σ for τ_dyn
        prior_c_mean,             # [6] Normal mean for c
        prior_c_sigma,            # [7] Normal σ for c
        prior_sigma_extra_scale,  # [8] HalfCauchy γ for σ_extra
        prior_H0_sigma,           # [9] Normal σ for H₀
        0.0,                      # [10] reserved
    ])

    if fix_c is not None:
        ndim = 6  # [a_eff, b_total, f_dyn, log_τ_dyn, log_σ_extra, H₀]
        param_names = ['a_eff', 'b_total', 'f_dyn', 'log_tau_dyn',
                       'log_sigma_extra', 'H0']
    else:
        ndim = 7  # [a_eff, b_total, f_dyn, log_τ_dyn, c, log_σ_extra, H₀]
        param_names = ['a_eff', 'b_total', 'f_dyn', 'log_tau_dyn',
                       'c_gr', 'log_sigma_extra', 'H0']

    # ── End-of-record temperature for rate prior ──
    T_end_val = T_annual[-1]  # last annual-mean GMST anomaly

    if progress:
        if fix_c is not None:
            print(f"  Fixed c = {fix_c*1e3:.3f} mm/yr "
                  f"(removed as free parameter, ndim={ndim})")
        print(f"  ODE init D_eff(0) = {T0_ann:.4f} °C")
        if rate_prior_mean is not None:
            print(f"  Rate prior: {rate_prior_mean*1e3:.2f} "
                  f"± {rate_prior_sigma*1e3:.2f} mm/yr, "
                  f"T_end={T_end_val:.3f} °C")

    # ── OLS initialization at fixed τ_dyn ──
    tau_dyn_init = np.exp(prior_log_tau_mean)

    # Solve discharge ODE on monthly grid (once, for accuracy)
    D_eff_init, _ = solve_twolayer_ode(
        T_monthly, time_monthly, tau_dyn_init, np.inf, Su0=T0_ann
    )

    # Cumulative integral of D_eff (monthly grid)
    cum_D_monthly = np.zeros(len(T_monthly))
    for i in range(len(T_monthly) - 1):
        cum_D_monthly[i + 1] = (cum_D_monthly[i]
                                + 0.5 * (D_eff_init[i] + D_eff_init[i + 1])
                                * dt_monthly[i])
    cum_D_obs = cum_D_monthly[obs_idx_monthly]

    # OLS: H = a_eff·I₂ + b_eff·I₁ + γ·∫D + c·I₀ + H₀
    X_init = np.column_stack([I2_obs, I1_obs, cum_D_obs, I0_obs,
                              np.ones(n)])
    W = np.diag(1.0 / sigma_obs**2)
    try:
        beta_ols = np.linalg.solve(X_init.T @ W @ X_init,
                                   X_init.T @ W @ H_obs)
    except np.linalg.LinAlgError:
        beta_ols = np.linalg.lstsq(X_init, H_obs, rcond=None)[0]

    a_init = max(beta_ols[0], 1e-6)
    b_init = max(beta_ols[1], 1e-6)
    gamma_init = max(beta_ols[2], 1e-6)
    c_init = beta_ols[3]
    H0_init = beta_ols[4]

    # Convert OLS (b_eff, γ) → (b_total, f_dyn)
    b_total_init = b_init + gamma_init
    f_dyn_init = np.clip(gamma_init / b_total_init, 0.05, 0.95)

    resid_ols = H_obs - X_init @ beta_ols
    sigma_extra_init = max(np.std(resid_ols), 1e-4)

    if fix_c is not None:
        # 6 params: [a_eff, b_total, f_dyn, log_τ_dyn, log_σ_extra, H₀]
        theta0 = np.array([
            a_init, b_total_init, f_dyn_init,
            np.log(tau_dyn_init),
            np.log(sigma_extra_init),
            H0_init,
        ])
    else:
        # 7 params: [a_eff, b_total, f_dyn, log_τ_dyn, c, log_σ_extra, H₀]
        theta0 = np.array([
            a_init, b_total_init, f_dyn_init,
            np.log(tau_dyn_init),
            c_init,
            np.log(sigma_extra_init),
            H0_init,
        ])

    if progress:
        print(f"  OLS init: a_eff={a_init*1e3:.3f} mm/yr/°C², "
              f"b_total={b_total_init*1e3:.3f} mm/yr/°C "
              f"(b_eff={b_init*1e3:.3f}, γ={gamma_init*1e3:.3f}), "
              f"f_dyn={f_dyn_init:.3f}, "
              f"τ_dyn={tau_dyn_init:.0f} yr")

    # ── Walker initialization ──
    rng = np.random.default_rng(seed)
    pos = np.empty((n_walkers, ndim))
    if fix_c is not None:
        for i in range(n_walkers):
            p = theta0.copy()
            p[0] = abs(a_init * (1.0 + 0.05 * rng.standard_normal()))
            p[1] = abs(b_total_init * (1.0 + 0.05 * rng.standard_normal()))
            p[2] = np.clip(f_dyn_init + 0.02 * rng.standard_normal(),
                           0.05, 0.95)
            p[3] = theta0[3] + 0.05 * rng.standard_normal()
            p[4] = theta0[4] + 0.1 * rng.standard_normal()   # log_σ_extra
            p[5] = H0_init + abs(H0_init) * 0.05 * rng.standard_normal()
            pos[i] = p
    else:
        for i in range(n_walkers):
            p = theta0.copy()
            p[0] = abs(a_init * (1.0 + 0.05 * rng.standard_normal()))
            p[1] = abs(b_total_init * (1.0 + 0.05 * rng.standard_normal()))
            p[2] = np.clip(f_dyn_init + 0.02 * rng.standard_normal(),
                           0.05, 0.95)
            p[3] = theta0[3] + 0.05 * rng.standard_normal()
            p[4] = c_init + max(abs(c_init), 1e-5) * 0.05 * rng.standard_normal()
            p[5] = theta0[5] + 0.1 * rng.standard_normal()
            p[6] = H0_init + abs(H0_init) * 0.05 * rng.standard_normal()
            pos[i] = p

    # ── Budget closure design vectors (Pass 2) ──
    budget_kwargs = {}
    if budget is not None:
        # Build design vectors at budget times
        budget_design = build_level_design_vectors(
            temperature_monthly=T_monthly,
            time_monthly=time_monthly,
            obs_times=budget.times,
        )
        budget_idx = np.array([
            np.argmin(np.abs(time_annual - t_b))
            for t_b in budget.times
        ])
        budget_kwargs = {
            'budget': budget,
            'budget_I2': budget_design['I2_obs'],
            'budget_I1': budget_design['I1_obs'],
            'budget_I0': budget_design['I0_obs'],
            'budget_idx_ann': budget_idx,
        }
        if progress:
            print(f"  Budget constraint: {len(budget.times)} level points "
                  f"({budget.times[0]:.0f}–{budget.times[-1]:.0f})")
            if budget.rate_accel is not None:
                print(f"  Budget rate+accel: "
                      f"{budget.rate_accel.rate*1e3:.2f} "
                      f"± {budget.rate_accel.rate_se*1e3:.2f} mm/yr")

    # ── MCMC (annual-resolution ODE for discharge) ──
    moves = [
        (emcee.moves.DESnookerMove(), 0.8),
        (emcee.moves.DEMove(),        0.2),
    ]
    if fix_c is not None:
        log_prob_fn = _greenland_log_prob_fixc
        log_prob_kwargs = {
            'fix_c_val': fix_c,
            'rate_prior_mean': rate_prior_mean,
            'rate_prior_sigma': rate_prior_sigma,
            'T_end': T_end_val,
            **budget_kwargs,
        }
    else:
        log_prob_fn = _greenland_log_prob
        log_prob_kwargs = {
            'rate_prior_mean': rate_prior_mean,
            'rate_prior_sigma': rate_prior_sigma,
            'T_end': T_end_val,
            **budget_kwargs,
        }

    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_prob_fn,
        args=(I2_obs, I1_obs, I0_obs,
              T_avg_ann, dt_ann, T0_ann, obs_idx_ann, n_ann,
              H_obs, sigma_obs, prior_scales, H0_prior_mean),
        kwargs=log_prob_kwargs,
        moves=moves,
    )

    if progress:
        print(f"  Running emcee: {n_walkers} walkers, "
              f"{n_burnin} burn-in + {n_samples} production "
              f"({ndim} params)...")

    sampler.run_mcmc(pos, n_burnin + n_samples, progress=progress)

    # ── Extract chains ──
    flat = sampler.get_chain(discard=n_burnin, thin=thin, flat=True)

    # Recover physical parameters: b_eff, γ from (b_total, f_dyn)
    a_eff_samples = flat[:, 0]
    b_total_samples = flat[:, 1]
    f_dyn_samples = flat[:, 2]
    b_eff_samples = b_total_samples * (1.0 - f_dyn_samples)
    gamma_samples = b_total_samples * f_dyn_samples
    tau_dyn_samples = np.exp(flat[:, 3])

    if fix_c is not None:
        # theta6 = [a_eff, b_total, f_dyn, log_τ_dyn, log_σ_extra, H₀]
        c_samples = np.full(len(flat), fix_c)
        sigma_extra_samples = np.exp(flat[:, 4])
        H0_samples = flat[:, 5]
    else:
        # theta7 = [a_eff, b_total, f_dyn, log_τ_dyn, c, log_σ_extra, H₀]
        c_samples = flat[:, 4]
        sigma_extra_samples = np.exp(flat[:, 5])
        H0_samples = flat[:, 6]

    # physical = [a_eff, b_eff, γ, c]
    phys_samples = np.column_stack([
        a_eff_samples, b_eff_samples, gamma_samples, c_samples
    ])

    phys_mean = np.mean(phys_samples, axis=0)
    phys_cov = np.cov(phys_samples, rowvar=False)
    phys_hdi = np.column_stack([
        az.hdi(phys_samples[:, k], hdi_prob=0.94)
        for k in range(phys_samples.shape[1])
    ]).T

    # ── Posterior-mean prediction (full monthly resolution) ──
    tau_dyn_median = np.median(tau_dyn_samples)

    # D_eff at posterior-median τ_dyn (monthly)
    D_eff_mean, _ = solve_twolayer_ode(
        T_monthly, time_monthly, tau_dyn_median, np.inf, Su0=T0_ann
    )

    # Cumulative ∫D_eff (monthly)
    cum_D_mean = np.zeros(len(T_monthly))
    for i in range(len(T_monthly) - 1):
        cum_D_mean[i + 1] = (cum_D_mean[i]
                              + 0.5 * (D_eff_mean[i] + D_eff_mean[i + 1])
                              * dt_monthly[i])
    cum_D_obs_mean = cum_D_mean[obs_idx_monthly]

    # Component decomposition at posterior mean
    H_smb_mean_obs = phys_mean[0] * I2_obs + phys_mean[1] * I1_obs
    H_dyn_mean_obs = phys_mean[2] * cum_D_obs_mean
    H_model_mean = (H_smb_mean_obs + H_dyn_mean_obs
                    + phys_mean[3] * I0_obs + np.mean(H0_samples))

    resid = H_obs - H_model_mean
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((H_obs - np.mean(H_obs))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # ── arviz trace ──
    n_chains_arviz = min(4, n_walkers)
    chain_full = sampler.get_chain(discard=n_burnin, thin=thin)
    var_dict = {}
    for k, name in enumerate(param_names):
        var_dict[name] = chain_full[:, :n_chains_arviz, k].T

    trace = az.from_dict(var_dict)
    conv = check_convergence(trace, quiet=(not progress))

    diag = {
        'acceptance_fraction': sampler.acceptance_fraction.mean(),
        'n_walkers': n_walkers,
        'n_samples': n_samples,
        'n_burnin': n_burnin,
        'thin': thin,
        'convergence': conv,
    }

    if progress:
        print(f"\n  Posterior mean: a_eff={phys_mean[0]*1e3:.3f} mm/yr/°C², "
              f"b_eff={phys_mean[1]*1e3:.3f} mm/yr/°C")
        print(f"  γ={phys_mean[2]*1e3:.3f} mm/yr/°C, "
              f"c={phys_mean[3]*1e3:.3f} mm/yr")
        print(f"  b_total={np.mean(b_total_samples)*1e3:.3f} mm/yr/°C, "
              f"f_dyn={np.mean(f_dyn_samples):.3f} "
              f"[{np.percentile(f_dyn_samples, 3):.3f}, "
              f"{np.percentile(f_dyn_samples, 97):.3f}]")
        print(f"  τ_dyn: median={np.median(tau_dyn_samples):.1f} yr "
              f"[{np.percentile(tau_dyn_samples, 3):.1f}, "
              f"{np.percentile(tau_dyn_samples, 97):.1f}]")
        print(f"  σ_extra: median="
              f"{np.median(sigma_extra_samples)*1e3:.2f} mm")
        print(f"  R² = {r2:.4f},  acceptance = "
              f"{diag['acceptance_fraction']:.2f}")
        # Discharge fraction: change in each component over the obs period
        H_smb_change = H_smb_mean_obs[-1] - H_smb_mean_obs[0]
        H_dyn_change = H_dyn_mean_obs[-1] - H_dyn_mean_obs[0]
        H_trend_change = phys_mean[3] * (I0_obs[-1] - I0_obs[0])
        H_total_change = H_model_mean[-1] - H_model_mean[0]
        print(f"  Component changes ({obs_times[0]:.0f}–{obs_times[-1]:.0f}):")
        print(f"    SMB:       {H_smb_change*1e3:+.2f} mm")
        print(f"    Discharge: {H_dyn_change*1e3:+.2f} mm")
        print(f"    Trend:     {H_trend_change*1e3:+.2f} mm")
        print(f"    Total:     {H_total_change*1e3:+.2f} mm")
        H_phys_change = H_smb_change + H_dyn_change
        if abs(H_phys_change) > 1e-8:
            dyn_frac = H_dyn_change / H_phys_change
            print(f"  Discharge fraction of SMB+discharge: "
                  f"{dyn_frac:.1%}")

    return BayesianGreenlandResult(
        trace=trace,
        physical_coefficients=phys_mean,
        physical_covariance=phys_cov,
        physical_hdi_94=phys_hdi,
        posterior_samples=phys_samples,
        tau_dyn_posterior=tau_dyn_samples,
        H0_posterior=H0_samples,
        sigma_extra_posterior=sigma_extra_samples,
        D_eff_mean=D_eff_mean,
        H_smb_mean=H_smb_mean_obs,
        H_dyn_mean=H_dyn_mean_obs,
        r2=r2,
        residuals=resid,
        time=obs_times,
        H_obs=H_obs,
        H_model_mean=H_model_mean,
        sigma_obs=sigma_obs,
        sampler_diagnostics=diag,
        design_info={
            'prior_scales': prior_scales,
            'param_names': param_names,
            'H0_prior_mean': H0_prior_mean,
            'obs_idx_monthly': obs_idx_monthly,
            'tau_dyn_init': tau_dyn_init,
        },
    )


# =====================================================================
#  Joint Greenland SMB + Discharge model
# =====================================================================

@dataclass
class BayesianGreenlandJointResult:
    """Result from joint SMB + discharge Greenland calibration.

    Model:
        H_smb = a_smb·I₂_atm + b_smb·I₁_atm + H₀_smb
        H_dyn = γ_atm·I₁_atm + γ_ocean·∫D_eff dt + D₀·I₀ + H₀_dyn
        dD_eff/dt = (T_ocean − D_eff) / τ

    Fitted jointly to observed cumulative SMB and cumulative D from
    Mouginot (1972–2018) and Mankoff (1986–2023).
    """
    trace: az.InferenceData
    # SMB parameters
    a_smb_posterior: np.ndarray       # (n_samples,)
    b_smb_posterior: np.ndarray
    H0_smb_posterior: np.ndarray
    sigma_extra_smb_posterior: np.ndarray
    # Discharge parameters
    gamma_atm_posterior: np.ndarray
    gamma_ocean_posterior: np.ndarray
    tau_posterior: np.ndarray
    D0_posterior: np.ndarray
    H0_dyn_posterior: np.ndarray
    sigma_extra_dyn_posterior: np.ndarray
    # Model predictions at posterior mean
    H_smb_model: np.ndarray           # (n_smb,) predicted cumulative SMB
    H_dyn_model: np.ndarray           # (n_dyn,) predicted cumulative D
    D_eff_mean: np.ndarray            # (n_monthly,) ocean ODE state
    # Goodness of fit
    r2_smb: float
    r2_dyn: float
    r2_total: float                   # R² of H_smb + H_dyn vs observed total MB
    # Observations
    time_smb: np.ndarray
    time_dyn: np.ndarray
    H_smb_obs: np.ndarray
    H_dyn_obs: np.ndarray
    sigma_smb_obs: np.ndarray
    sigma_dyn_obs: np.ndarray
    sampler_diagnostics: Optional[dict] = None


def _greenland_joint_log_prob(
    theta, I2_smb, I1_smb, I0_smb, I1_dyn, I0_dyn,
    H_smb_obs, sigma_smb, H_dyn_obs, sigma_dyn,
    prior_scales,
    use_ocean=False,
    T_ocean_annual=None, dt_ocean=None, T0_ocean=0.0,
    obs_idx_ocean_smb=None, obs_idx_ocean_dyn=None, n_ocean=0,
    budget_rate_accel=None,
    T_atm_eval=0.0, dTdt_eval=0.0, ocean_eval_idx=None,
):
    """Log-posterior for the joint SMB + discharge Greenland model.

    Two modes:

    **Atmospheric only** (use_ocean=False, default):
        theta = [a_smb, b_smb, log_σ_smb, H0_smb,
                 γ_atm, D₀, log_σ_dyn, H0_dyn]  (8 params)
        H_dyn = γ_atm·I₁ + D₀·I₀ + H₀_dyn

    **With ocean ODE** (use_ocean=True):
        theta = [a_smb, b_smb, log_σ_smb, H0_smb,
                 γ_atm, γ_ocean, log_τ, D₀, log_σ_dyn, H0_dyn]  (10 params)
        H_dyn = γ_atm·I₁ + γ_ocean·∫D_eff + D₀·I₀ + H₀_dyn

    **Budget rate+acceleration constraint** (Pass 2):
        If budget_rate_accel is not None, adds a bivariate Gaussian penalty
        on (rate_model, accel_model) vs the satellite-era budget residual.
        rate = dH_smb/dt + dH_dyn/dt; accel = d²H/dt² at eval time.
        No level constraint (dominated by TWS/deep-ocean budget residual).
    """
    # ── Unpack theta ──
    if use_ocean:
        (a_smb, b_smb, log_sig_smb, H0_smb,
         gamma_atm, gamma_ocean, log_tau, D0,
         log_sig_dyn, H0_dyn) = theta
        if gamma_ocean < 0:
            return -np.inf
        if log_tau < -1 or log_tau > 7:
            return -np.inf
        tau = np.exp(log_tau)
    else:
        (a_smb, b_smb, log_sig_smb, H0_smb,
         gamma_atm, D0, log_sig_dyn, H0_dyn) = theta

    # Hard bounds
    if a_smb < 0 or gamma_atm < 0:
        return -np.inf
    if log_sig_smb > 0 or log_sig_smb < -20:
        return -np.inf
    if log_sig_dyn > 0 or log_sig_dyn < -20:
        return -np.inf

    sigma_extra_smb = np.exp(log_sig_smb)
    sigma_extra_dyn = np.exp(log_sig_dyn)

    # ── Priors ──
    # prior_scales layout:
    #  [0] scale_a_smb (Exp mean)
    #  [1] scale_b_smb (HN σ)
    #  [2] scale_gamma_atm (HN σ)
    #  [3] scale_gamma_ocean (HN σ)  — only used when use_ocean=True
    #  [4] log_tau_mean               — only used when use_ocean=True
    #  [5] log_tau_sigma              — only used when use_ocean=True
    #  [6] D0_sigma
    #  [7] sig_extra_smb_scale (HC γ)
    #  [8] sig_extra_dyn_scale (HC γ)
    #  [9] H0_smb_sigma
    # [10] H0_dyn_sigma

    lp = 0.0

    # a_smb ~ Exp(mean = scale)
    scale_a = prior_scales[0]
    lp += -a_smb / scale_a - np.log(scale_a)

    # b_smb ~ Normal(0, σ²)
    scale_b = prior_scales[1]
    lp += -0.5 * (b_smb / scale_b) ** 2

    # γ_atm ~ HalfNormal(σ)
    scale_ga = prior_scales[2]
    lp += -0.5 * (gamma_atm / scale_ga) ** 2

    if use_ocean:
        # γ_ocean ~ HalfNormal(σ)
        scale_go = prior_scales[3]
        lp += -0.5 * (gamma_ocean / scale_go) ** 2
        # τ ~ LogNormal(μ, σ)
        mu_lt = prior_scales[4]
        sig_lt = prior_scales[5]
        lp += -0.5 * ((log_tau - mu_lt) / sig_lt) ** 2

    # D₀ ~ Normal(0, σ)
    sig_D0 = prior_scales[6]
    lp += -0.5 * (D0 / sig_D0) ** 2

    # σ_extra ~ HalfCauchy(γ)
    gamma_hc_smb = prior_scales[7]
    lp += -np.log(1 + (sigma_extra_smb / gamma_hc_smb) ** 2) + log_sig_smb

    gamma_hc_dyn = prior_scales[8]
    lp += -np.log(1 + (sigma_extra_dyn / gamma_hc_dyn) ** 2) + log_sig_dyn

    # H₀ ~ Normal(H_obs[0], σ)
    sig_H0s = prior_scales[9]
    lp += -0.5 * ((H0_smb - H_smb_obs[0]) / sig_H0s) ** 2

    sig_H0d = prior_scales[10]
    lp += -0.5 * ((H0_dyn - H_dyn_obs[0]) / sig_H0d) ** 2

    # ── SMB forward model ──
    H_smb_pred = a_smb * I2_smb + b_smb * I1_smb + H0_smb
    var_smb = sigma_smb ** 2 + sigma_extra_smb ** 2
    resid_smb = H_smb_obs - H_smb_pred
    lp += -0.5 * np.sum(resid_smb ** 2 / var_smb + np.log(var_smb))

    # ── Discharge forward model ──
    if use_ocean:
        # Solve ODE: dD_eff/dt = (T_ocean - D_eff) / τ
        D_eff = np.empty(n_ocean)
        D_eff[0] = T0_ocean
        alpha = np.exp(-dt_ocean / tau)
        for i in range(n_ocean - 1):
            D_eff[i + 1] = (D_eff[i] * alpha[i]
                            + 0.5 * (T_ocean_annual[i] + T_ocean_annual[i + 1]) * (1.0 - alpha[i]))
        cum_D = np.zeros(n_ocean)
        for i in range(n_ocean - 1):
            cum_D[i + 1] = (cum_D[i]
                            + 0.5 * (D_eff[i] + D_eff[i + 1]) * dt_ocean[i])
        cum_D_dyn = cum_D[obs_idx_ocean_dyn]
        H_dyn_pred = (gamma_atm * I1_dyn + gamma_ocean * cum_D_dyn
                      + D0 * I0_dyn + H0_dyn)
    else:
        # Atmospheric only: H_dyn = γ_atm·I₁ + D₀·I₀ + H₀
        H_dyn_pred = gamma_atm * I1_dyn + D0 * I0_dyn + H0_dyn

    var_dyn = sigma_dyn ** 2 + sigma_extra_dyn ** 2
    resid_dyn = H_dyn_obs - H_dyn_pred
    lp += -0.5 * np.sum(resid_dyn ** 2 / var_dyn + np.log(var_dyn))

    # ── Budget rate + acceleration constraint (Pass 2) ──
    if budget_rate_accel is not None:
        # Rate: dH/dt = dH_smb/dt + dH_dyn/dt at eval time
        T = T_atm_eval
        rate_smb = a_smb * T ** 2 + b_smb * T
        if use_ocean:
            rate_dyn = gamma_atm * T + gamma_ocean * D_eff[ocean_eval_idx] + D0
        else:
            rate_dyn = gamma_atm * T + D0
        rate_total = rate_smb + rate_dyn

        # Acceleration: d²H/dt² at eval time
        dT = dTdt_eval
        accel_smb = (2.0 * a_smb * T + b_smb) * dT
        if use_ocean:
            dDeff_dt = ((T_ocean_annual[ocean_eval_idx]
                         - D_eff[ocean_eval_idx]) / tau)
            accel_dyn = gamma_atm * dT + gamma_ocean * dDeff_dt
        else:
            accel_dyn = gamma_atm * dT
        accel_total = accel_smb + accel_dyn

        lp_ra = _rate_accel_prior_logp(rate_total, accel_total,
                                       budget_rate_accel)
        if not np.isfinite(lp_ra):
            return -np.inf
        lp += lp_ra

    return lp


def fit_bayesian_greenland_joint(
    # ── SMB observations ──
    H_smb_obs: np.ndarray,
    sigma_smb_obs: np.ndarray,
    time_smb: np.ndarray,
    I2_smb: np.ndarray,
    I1_smb: np.ndarray,
    I0_smb: np.ndarray,
    # ── Discharge observations ──
    H_dyn_obs: np.ndarray,
    sigma_dyn_obs: np.ndarray,
    time_dyn: np.ndarray,
    I1_dyn: np.ndarray,
    I0_dyn: np.ndarray,
    # ── Ocean temperature forcing (optional) ──
    T_ocean_monthly: Optional[np.ndarray] = None,
    time_ocean_monthly: Optional[np.ndarray] = None,
    # ── Budget rate+accel constraint (Pass 2) ──
    budget_rate_accel: Optional['SatelliteEraQuadraticResult'] = None,
    T_atm_eval: float = 0.0,
    dTdt_eval: float = 0.0,
    # ── Priors ──
    prior_scale_a_smb: float = 0.002,
    prior_scale_b_smb: float = 0.004,
    prior_scale_gamma_atm: float = 0.002,
    prior_scale_gamma_ocean: float = 0.002,
    prior_log_tau_mean: float = None,
    prior_log_tau_sigma: float = 0.5,
    prior_D0_sigma: float = 0.0001,
    prior_sigma_extra_smb: float = 0.002,
    prior_sigma_extra_dyn: float = 0.002,
    prior_H0_smb_sigma: float = 0.005,
    prior_H0_dyn_sigma: float = 0.005,
    # ── MCMC settings ──
    n_samples: int = 10000,
    n_walkers: int = 64,
    n_burnin: int = 5000,
    thin: int = 1,
    progress: bool = True,
    seed: Optional[int] = None,
) -> BayesianGreenlandJointResult:
    """Joint Bayesian fit of Greenland SMB and discharge components.

    Two modes:

    **Atmospheric only** (T_ocean_monthly=None, default):
        H_smb = a_smb·I₂ + b_smb·I₁ + H₀_smb
        H_dyn = γ_atm·I₁ + D₀·I₀ + H₀_dyn
        8 parameters.

    **With ocean ODE** (T_ocean_monthly provided):
        H_smb = a_smb·I₂ + b_smb·I₁ + H₀_smb
        H_dyn = γ_atm·I₁ + γ_ocean·∫D_eff + D₀·I₀ + H₀_dyn
        dD_eff/dt = (T_ocean − D_eff) / τ
        10 parameters.

    Parameters
    ----------
    H_smb_obs, sigma_smb_obs, time_smb : (n_smb,)
        Observed cumulative SMB (m SLE, SLR convention) with σ and times.
    I2_smb, I1_smb, I0_smb : (n_smb,)
        Design vectors for SMB (from atmospheric T).
    H_dyn_obs, sigma_dyn_obs, time_dyn : (n_dyn,)
        Observed cumulative discharge (m SLE, SLR convention).
    I1_dyn, I0_dyn : (n_dyn,)
        Design vectors for discharge (from atmospheric T).
    T_ocean_monthly, time_ocean_monthly : (n,) or None
        Monthly subsurface ocean temperature (°C).
        If None, uses atmospheric-only discharge model.
    budget_rate_accel : SatelliteEraQuadraticResult or None
        Budget-closure rate and acceleration constraint for Pass 2.
        Contains observed Greenland rate, accel, and 2×2 covariance
        (GMSL minus all other components). No level constraint.
    T_atm_eval : float
        Greenland surface temperature (°C, relative to baseline) at the
        budget evaluation time. Used to compute model rate and accel.
    dTdt_eval : float
        Temperature trend (°C/yr) at the budget evaluation time.
        Used to compute model acceleration.
    """
    use_ocean = T_ocean_monthly is not None
    if prior_log_tau_mean is None:
        prior_log_tau_mean = np.log(20.0)

    n_smb = len(H_smb_obs)
    n_dyn = len(H_dyn_obs)

    # ── Ocean T setup (only when use_ocean=True) ──
    T_ocean_annual = None
    dt_ocean = None
    T0_ocean = 0.0
    obs_idx_ocean_smb = None
    obs_idx_ocean_dyn = None
    n_ocean = 0

    if use_ocean:
        year_floor = np.floor(time_ocean_monthly).astype(int)
        unique_years = np.unique(year_floor)
        T_ocean_annual = np.array([
            np.mean(T_ocean_monthly[year_floor == yr])
            for yr in unique_years
        ])
        time_ocean_annual = unique_years.astype(float) + 0.5
        n_ocean = len(time_ocean_annual)
        dt_ocean = np.diff(time_ocean_annual)

        n_spinup = min(10, n_ocean)
        T0_ocean = float(np.mean(T_ocean_annual[:n_spinup]))

        obs_idx_ocean_smb = np.array([
            np.argmin(np.abs(time_ocean_annual - t)) for t in time_smb
        ])
        obs_idx_ocean_dyn = np.array([
            np.argmin(np.abs(time_ocean_annual - t)) for t in time_dyn
        ])

    # ── Budget constraint setup ──
    ocean_eval_idx = None
    if budget_rate_accel is not None and use_ocean:
        ocean_eval_idx = int(np.argmin(
            np.abs(time_ocean_annual - budget_rate_accel.eval_time)))

    if progress:
        mode = "atmospheric + ocean" if use_ocean else "atmospheric only"
        print(f"  Mode: {mode}")
        print(f"  SMB obs: {n_smb} pts ({time_smb[0]:.0f}–{time_smb[-1]:.0f})")
        print(f"  Dyn obs: {n_dyn} pts ({time_dyn[0]:.0f}–{time_dyn[-1]:.0f})")
        if use_ocean:
            print(f"  Ocean T: {n_ocean} annual pts "
                  f"({time_ocean_annual[0]:.0f}–{time_ocean_annual[-1]:.0f})")
            print(f"  ODE init T_ocean(0) = {T0_ocean:.2f} °C")
        if budget_rate_accel is not None:
            M = 1e3
            print(f"  Budget constraint (Pass 2):")
            print(f"    Target rate: {budget_rate_accel.rate*M:.3f} "
                  f"± {budget_rate_accel.rate_se*M:.3f} mm/yr "
                  f"at {budget_rate_accel.eval_time:.1f}")
            print(f"    Target accel: {budget_rate_accel.accel*M:.4f} "
                  f"± {budget_rate_accel.accel_se*M:.4f} mm/yr²")
            print(f"    T_atm_eval = {T_atm_eval:.3f} °C, "
                  f"dT/dt = {dTdt_eval:.4f} °C/yr")

    # ── Prior scales array ──
    prior_scales = np.array([
        prior_scale_a_smb,       # [0]
        prior_scale_b_smb,       # [1]
        prior_scale_gamma_atm,   # [2]
        prior_scale_gamma_ocean, # [3]
        prior_log_tau_mean,      # [4]
        prior_log_tau_sigma,     # [5]
        prior_D0_sigma,          # [6]
        prior_sigma_extra_smb,   # [7]
        prior_sigma_extra_dyn,   # [8]
        prior_H0_smb_sigma,      # [9]
        prior_H0_dyn_sigma,      # [10]
        0.0,                     # [11] reserved
    ])

    if use_ocean:
        ndim = 10
        param_names = [
            'a_smb', 'b_smb', 'log_sigma_smb', 'H0_smb',
            'gamma_atm', 'gamma_ocean', 'log_tau', 'D0',
            'log_sigma_dyn', 'H0_dyn',
        ]
    else:
        ndim = 8
        param_names = [
            'a_smb', 'b_smb', 'log_sigma_smb', 'H0_smb',
            'gamma_atm', 'D0', 'log_sigma_dyn', 'H0_dyn',
        ]

    # ── OLS initialization ──
    # SMB: H_smb = a·I₂ + b·I₁ + H₀
    X_smb = np.column_stack([I2_smb, I1_smb, np.ones(n_smb)])
    W_smb = np.diag(1.0 / sigma_smb_obs ** 2)
    try:
        beta_smb = np.linalg.solve(X_smb.T @ W_smb @ X_smb,
                                   X_smb.T @ W_smb @ H_smb_obs)
    except np.linalg.LinAlgError:
        beta_smb = np.linalg.lstsq(X_smb, H_smb_obs, rcond=None)[0]

    if use_ocean:
        # Discharge ODE at prior median τ for initialization
        tau_init = np.exp(prior_log_tau_mean)
        D_eff_init = np.empty(n_ocean)
        D_eff_init[0] = T0_ocean
        alpha_init = np.exp(-dt_ocean / tau_init)
        for i in range(n_ocean - 1):
            D_eff_init[i + 1] = (D_eff_init[i] * alpha_init[i]
                                  + 0.5 * (T_ocean_annual[i] + T_ocean_annual[i + 1]) * (1.0 - alpha_init[i]))
        cum_D_init = np.zeros(n_ocean)
        for i in range(n_ocean - 1):
            cum_D_init[i + 1] = (cum_D_init[i]
                                  + 0.5 * (D_eff_init[i] + D_eff_init[i + 1])
                                  * dt_ocean[i])
        cum_D_dyn_init = cum_D_init[obs_idx_ocean_dyn]

        X_dyn = np.column_stack([I1_dyn, cum_D_dyn_init, I0_dyn,
                                  np.ones(n_dyn)])
    else:
        # Atmospheric only: H_dyn = γ_atm·I₁ + D₀·I₀ + H₀
        X_dyn = np.column_stack([I1_dyn, I0_dyn, np.ones(n_dyn)])

    W_dyn = np.diag(1.0 / sigma_dyn_obs ** 2)
    try:
        beta_dyn = np.linalg.solve(X_dyn.T @ W_dyn @ X_dyn,
                                   X_dyn.T @ W_dyn @ H_dyn_obs)
    except np.linalg.LinAlgError:
        beta_dyn = np.linalg.lstsq(X_dyn, H_dyn_obs, rcond=None)[0]

    a_init = max(beta_smb[0], 1e-6)
    b_init = beta_smb[1]
    H0s_init = beta_smb[2]

    ga_init = max(beta_dyn[0], 1e-6)
    if use_ocean:
        go_init = max(beta_dyn[1], 1e-6)
        D0_init = beta_dyn[2]
        H0d_init = beta_dyn[3]
        tau_init = np.exp(prior_log_tau_mean)
    else:
        go_init = 0.0
        D0_init = beta_dyn[1]
        H0d_init = beta_dyn[2]
        tau_init = 10.0  # unused

    resid_smb_init = H_smb_obs - X_smb @ beta_smb
    resid_dyn_init = H_dyn_obs - X_dyn @ beta_dyn
    sig_smb_init = max(np.std(resid_smb_init), 1e-5)
    sig_dyn_init = max(np.std(resid_dyn_init), 1e-5)

    if use_ocean:
        theta0 = np.array([
            a_init, b_init, np.log(sig_smb_init), H0s_init,
            ga_init, go_init, np.log(tau_init), D0_init,
            np.log(sig_dyn_init), H0d_init,
        ])
    else:
        theta0 = np.array([
            a_init, b_init, np.log(sig_smb_init), H0s_init,
            ga_init, D0_init,
            np.log(sig_dyn_init), H0d_init,
        ])

    if progress:
        M = 1e3
        print(f"  OLS init (SMB): a={a_init*M:.3f} mm/yr/°C², "
              f"b={b_init*M:.3f} mm/yr/°C")
        if use_ocean:
            print(f"  OLS init (Dyn): γ_atm={ga_init*M:.3f}, "
                  f"γ_ocean={go_init*M:.3f} mm/yr/°C, "
                  f"D₀={D0_init*M:.4f} mm/yr, τ={tau_init:.0f} yr")
        else:
            print(f"  OLS init (Dyn): γ_atm={ga_init*M:.3f} mm/yr/°C, "
                  f"D₀={D0_init*M:.4f} mm/yr")

    # ── Walker initialization ──
    rng = np.random.default_rng(seed)
    pos = np.empty((n_walkers, ndim))
    for i in range(n_walkers):
        p = theta0.copy()
        p[0] = abs(a_init * (1.0 + 0.1 * rng.standard_normal()))
        p[1] = b_init + abs(b_init + 1e-4) * 0.1 * rng.standard_normal()
        p[2] = theta0[2] + 0.1 * rng.standard_normal()
        p[3] = H0s_init + max(abs(H0s_init), 1e-4) * 0.05 * rng.standard_normal()
        p[4] = abs(ga_init * (1.0 + 0.1 * rng.standard_normal()))
        if use_ocean:
            p[5] = abs(go_init * (1.0 + 0.1 * rng.standard_normal()))
            p[6] = theta0[6] + 0.1 * rng.standard_normal()
            p[7] = D0_init + max(abs(D0_init), 1e-5) * 0.1 * rng.standard_normal()
            p[8] = theta0[8] + 0.1 * rng.standard_normal()
            p[9] = H0d_init + max(abs(H0d_init), 1e-4) * 0.05 * rng.standard_normal()
        else:
            p[5] = D0_init + max(abs(D0_init), 1e-5) * 0.1 * rng.standard_normal()
            p[6] = theta0[6] + 0.1 * rng.standard_normal()
            p[7] = H0d_init + max(abs(H0d_init), 1e-4) * 0.05 * rng.standard_normal()
        pos[i] = p

    # ── MCMC ──
    moves = [
        (emcee.moves.DESnookerMove(), 0.8),
        (emcee.moves.DEMove(),        0.2),
    ]

    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, _greenland_joint_log_prob,
        args=(I2_smb, I1_smb, I0_smb, I1_dyn, I0_dyn,
              H_smb_obs, sigma_smb_obs, H_dyn_obs, sigma_dyn_obs,
              prior_scales),
        kwargs={
            'use_ocean': use_ocean,
            'T_ocean_annual': T_ocean_annual,
            'dt_ocean': dt_ocean,
            'T0_ocean': T0_ocean,
            'obs_idx_ocean_smb': obs_idx_ocean_smb,
            'obs_idx_ocean_dyn': obs_idx_ocean_dyn,
            'n_ocean': n_ocean,
            'budget_rate_accel': budget_rate_accel,
            'T_atm_eval': T_atm_eval,
            'dTdt_eval': dTdt_eval,
            'ocean_eval_idx': ocean_eval_idx,
        },
        moves=moves,
    )

    if progress:
        print(f"  Running emcee: {n_walkers} walkers, "
              f"{n_burnin} burn-in + {n_samples} production "
              f"({ndim} params)...")

    sampler.run_mcmc(pos, n_burnin + n_samples, progress=progress)

    # ── Extract chains ──
    flat = sampler.get_chain(discard=n_burnin, thin=thin, flat=True)

    a_smb_s = flat[:, 0]
    b_smb_s = flat[:, 1]
    sig_smb_s = np.exp(flat[:, 2])
    H0s_s = flat[:, 3]
    ga_s = flat[:, 4]

    if use_ocean:
        go_s = flat[:, 5]
        tau_s = np.exp(flat[:, 6])
        D0_s = flat[:, 7]
        sig_dyn_s = np.exp(flat[:, 8])
        H0d_s = flat[:, 9]
    else:
        go_s = np.zeros(len(flat))
        tau_s = np.full(len(flat), np.nan)
        D0_s = flat[:, 5]
        sig_dyn_s = np.exp(flat[:, 6])
        H0d_s = flat[:, 7]

    # ── Posterior-mean predictions ──
    a_m, b_m = np.mean(a_smb_s), np.mean(b_smb_s)
    H0s_m = np.mean(H0s_s)
    ga_m = np.mean(ga_s)
    D0_m = np.mean(D0_s)
    H0d_m = np.mean(H0d_s)

    H_smb_pred = a_m * I2_smb + b_m * I1_smb + H0s_m

    D_eff_post = None
    if use_ocean:
        go_m = np.mean(go_s)
        tau_med = np.median(tau_s)
        D_eff_post = np.empty(n_ocean)
        D_eff_post[0] = T0_ocean
        alpha_post = np.exp(-dt_ocean / tau_med)
        for i in range(n_ocean - 1):
            D_eff_post[i + 1] = (D_eff_post[i] * alpha_post[i]
                                  + 0.5 * (T_ocean_annual[i] + T_ocean_annual[i + 1]) * (1.0 - alpha_post[i]))
        cum_D_post = np.zeros(n_ocean)
        for i in range(n_ocean - 1):
            cum_D_post[i + 1] = (cum_D_post[i]
                                  + 0.5 * (D_eff_post[i] + D_eff_post[i + 1])
                                  * dt_ocean[i])
        cum_D_dyn_post = cum_D_post[obs_idx_ocean_dyn]
        H_dyn_pred = (ga_m * I1_dyn + go_m * cum_D_dyn_post
                      + D0_m * I0_dyn + H0d_m)
    else:
        H_dyn_pred = ga_m * I1_dyn + D0_m * I0_dyn + H0d_m

    def _r2(obs, pred):
        ss_res = np.sum((obs - pred) ** 2)
        ss_tot = np.sum((obs - np.mean(obs)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    r2_smb = _r2(H_smb_obs, H_smb_pred)
    r2_dyn = _r2(H_dyn_obs, H_dyn_pred)

    # Total MB R² (using observations at overlapping times)
    # Find common times between SMB and D
    smb_set = set(np.round(time_smb, 1))
    dyn_set = set(np.round(time_dyn, 1))
    common_times = sorted(smb_set & dyn_set)
    if len(common_times) > 5:
        smb_idx = [i for i, t in enumerate(np.round(time_smb, 1))
                   if t in common_times]
        dyn_idx = [i for i, t in enumerate(np.round(time_dyn, 1))
                   if t in common_times]
        H_total_obs = H_smb_obs[smb_idx] + H_dyn_obs[dyn_idx]
        H_total_pred = H_smb_pred[smb_idx] + H_dyn_pred[dyn_idx]
        r2_total = _r2(H_total_obs, H_total_pred)
    else:
        r2_total = np.nan

    # ── Convergence diagnostics ──
    n_chains_arviz = min(4, n_walkers)
    chain_full = sampler.get_chain(discard=n_burnin, thin=thin)
    var_dict = {}
    for k, name in enumerate(param_names):
        var_dict[name] = chain_full[:, :n_chains_arviz, k].T
    trace = az.from_dict(var_dict)
    conv = check_convergence(trace, quiet=(not progress))

    diag = {
        'acceptance_fraction': sampler.acceptance_fraction.mean(),
        'n_walkers': n_walkers,
        'n_samples': n_samples,
        'n_burnin': n_burnin,
        'thin': thin,
        'convergence': conv,
    }

    if progress:
        M = 1e3
        print(f"\n  SMB posterior: a_smb={np.mean(a_smb_s)*M:.3f} "
              f"[{np.percentile(a_smb_s, 3)*M:.3f}, "
              f"{np.percentile(a_smb_s, 97)*M:.3f}] mm/yr/°C²")
        print(f"    b_smb={np.mean(b_smb_s)*M:.3f} "
              f"[{np.percentile(b_smb_s, 3)*M:.3f}, "
              f"{np.percentile(b_smb_s, 97)*M:.3f}] mm/yr/°C")
        print(f"    σ_extra_smb={np.median(sig_smb_s)*M:.2f} mm, "
              f"R²={r2_smb:.4f}")
        print(f"  Dyn posterior: γ_atm={np.mean(ga_s)*M:.3f} "
              f"[{np.percentile(ga_s, 3)*M:.3f}, "
              f"{np.percentile(ga_s, 97)*M:.3f}] mm/yr/°C")
        if use_ocean:
            print(f"    γ_ocean={np.mean(go_s)*M:.3f} "
                  f"[{np.percentile(go_s, 3)*M:.3f}, "
                  f"{np.percentile(go_s, 97)*M:.3f}] mm/yr/°C")
            print(f"    τ={np.median(tau_s):.1f} "
                  f"[{np.percentile(tau_s, 3):.1f}, "
                  f"{np.percentile(tau_s, 97):.1f}] yr")
        print(f"    D₀={np.mean(D0_s)*M:.4f} mm/yr, "
              f"σ_extra_dyn={np.median(sig_dyn_s)*M:.2f} mm")
        print(f"    R²_dyn={r2_dyn:.4f}")
        print(f"  Total MB: R²={r2_total:.4f}")
        print(f"  Acceptance: {diag['acceptance_fraction']:.2f}")

        # Component changes
        H_smb_change = H_smb_pred[-1] - H_smb_pred[0]
        H_dyn_change = H_dyn_pred[-1] - H_dyn_pred[0]
        print(f"\n  Component changes (model):")
        print(f"    SMB:       {H_smb_change*M:+.2f} mm")
        print(f"    Discharge: {H_dyn_change*M:+.2f} mm")
        print(f"    Total:     {(H_smb_change + H_dyn_change)*M:+.2f} mm")

        # Acceleration partition
        # Use linear regression of rate on time to get acceleration
        dt_smb = time_smb[-1] - time_smb[0]
        smb_rate_start = a_m * 2 * I2_smb[0] / max(I0_smb[0], 1e-6) + b_m
        smb_rate_end = a_m * 2 * I2_smb[-1] / max(I0_smb[-1], 1e-6) + b_m

    return BayesianGreenlandJointResult(
        trace=trace,
        a_smb_posterior=a_smb_s,
        b_smb_posterior=b_smb_s,
        H0_smb_posterior=H0s_s,
        sigma_extra_smb_posterior=sig_smb_s,
        gamma_atm_posterior=ga_s,
        gamma_ocean_posterior=go_s,
        tau_posterior=tau_s,
        D0_posterior=D0_s,
        H0_dyn_posterior=H0d_s,
        sigma_extra_dyn_posterior=sig_dyn_s,
        H_smb_model=H_smb_pred,
        H_dyn_model=H_dyn_pred,
        D_eff_mean=D_eff_post,
        r2_smb=r2_smb,
        r2_dyn=r2_dyn,
        r2_total=r2_total,
        time_smb=time_smb,
        time_dyn=time_dyn,
        H_smb_obs=H_smb_obs,
        H_dyn_obs=H_dyn_obs,
        sigma_smb_obs=sigma_smb_obs,
        sigma_dyn_obs=sigma_dyn_obs,
        sampler_diagnostics=diag,
    )


def prepare_greenland_components(
    mankoff_path: str,
    mouginot_df: Optional['pd.DataFrame'] = None,
    baseline_window: tuple = (1995, 2005),
    start_year: int = 1972,
    end_year: Optional[int] = None,
) -> dict:
    """Prepare Greenland SMB/D cumulative SLE from Mankoff and Mouginot.

    Reads the Mankoff MB_SMB_D_BMB_ann.csv and optionally merges with
    a Mouginot DataFrame (from ``read_mouginot2019_greenland``).  Both
    datasets provide independent SMB and discharge rate estimates; when
    both are supplied the observations are concatenated and sorted by
    time, giving two independent observations per overlap year.

    Parameters
    ----------
    mankoff_path : str
        Path to ``MB_SMB_D_BMB_ann.csv``.
    mouginot_df : DataFrame or None
        Output of ``read_mouginot2019_greenland()``.  Must contain
        columns ``decimal_year``, ``smb_rate``, ``smb_rate_sigma``,
        ``discharge_rate``, ``discharge_rate_sigma``, ``mb_rate``,
        ``mb_rate_sigma`` (all m/yr SLE, SLR convention).
        If None, uses Mankoff only.
    baseline_window : tuple
        (start, end) years for rebaseline.  Default (1995, 2005).
    start_year : int
        First year to include.  Default 1972 (Mouginot).
    end_year : int or None
        Last year.  None → end of record.

    Returns
    -------
    dict with keys:
        'time_smb', 'H_smb', 'sigma_smb' : SMB cumulative SLE (m)
        'time_dyn', 'H_dyn', 'sigma_dyn' : Discharge cumulative SLE (m)
        'time_mb', 'H_mb', 'sigma_mb'    : Total MB cumulative SLE (m)
        'sources'  : (n,) str array — 'mankoff' or 'mouginot' per obs
        'df' : raw Mankoff DataFrame (Gt/yr units)

    Notes
    -----
    Each dataset is cumulated independently (so baseline-rate subtraction
    and integration use each dataset's own time grid), then the resulting
    cumulative series are concatenated and sorted by time.
    """
    import pandas as pd

    try:
        from slr_forecast.config import GT_TO_M_SLE
    except ImportError:
        GT_TO_M_SLE = 1.0 / 362500.0

    # ── Helper: cumulate rates into level ──
    def _cumulate(rate, sigma, times, bl_window):
        """Subtract baseline mean rate, integrate anomalies, rebase level.

        Uncertainty is cumulated bidirectionally from the baseline
        midpoint so that it grows both forward and backward in time.
        """
        bl_mask = (times >= bl_window[0]) & (times <= bl_window[1])

        # Subtract baseline-period mean rate to get anomaly rates
        if bl_mask.sum() > 0:
            rate_ref = rate[bl_mask].mean()
        else:
            rate_ref = 0.0
        rate_anom = rate - rate_ref

        dt = np.diff(times, prepend=times[0] - 1)
        cum = np.cumsum(rate_anom * dt)

        # Rebase level: subtract value at baseline midpoint
        if bl_mask.sum() > 0:
            bl_mid_idx = np.argmin(np.abs(
                times - np.mean(bl_window)))
            cum = cum - cum[bl_mid_idx]

            # Bidirectional uncertainty: cumulate outward from baseline midpoint
            cum_sig = np.zeros(len(times))
            sig_dt = sigma * dt
            # Forward from baseline midpoint
            for i in range(bl_mid_idx + 1, len(times)):
                cum_sig[i] = np.sqrt(cum_sig[i - 1]**2 + sig_dt[i]**2)
            # Backward from baseline midpoint
            for i in range(bl_mid_idx - 1, -1, -1):
                cum_sig[i] = np.sqrt(cum_sig[i + 1]**2 + sig_dt[i + 1]**2)
            cum_sig = np.maximum(cum_sig, sigma.mean())
        else:
            cum_sig = np.sqrt(np.cumsum((sigma * dt) ** 2))

        return cum, cum_sig

    # ── Mankoff ──
    df = pd.read_csv(mankoff_path)
    if end_year is not None:
        df = df[(df['time'] >= start_year) & (df['time'] <= end_year)].copy()
    else:
        df = df[df['time'] >= start_year].copy()
    df = df.sort_values('time').reset_index(drop=True)
    man_years = df['time'].values.astype(float)

    # Rename to Mouginot-style columns (m/yr SLE, SLR convention)
    df['decimal_year'] = man_years
    df['smb_rate'] = -df['SMB'].values * GT_TO_M_SLE
    df['smb_rate_sigma'] = np.abs(df['SMB_err'].values) * GT_TO_M_SLE
    df['discharge_rate'] = df['D'].values * GT_TO_M_SLE
    df['discharge_rate_sigma'] = np.abs(df['D_err'].values) * GT_TO_M_SLE
    df['mb_rate'] = -df['MB'].values * GT_TO_M_SLE
    df['mb_rate_sigma'] = np.abs(df['MB_err'].values) * GT_TO_M_SLE

    man_smb_rate = df['smb_rate'].values
    man_smb_err = df['smb_rate_sigma'].values
    man_dyn_rate = df['discharge_rate'].values
    man_dyn_err = df['discharge_rate_sigma'].values
    man_mb_rate = df['mb_rate'].values
    man_mb_err = df['mb_rate_sigma'].values

    H_smb_man, sig_smb_man = _cumulate(man_smb_rate, man_smb_err,
                                        man_years, baseline_window)
    H_dyn_man, sig_dyn_man = _cumulate(man_dyn_rate, man_dyn_err,
                                        man_years, baseline_window)
    H_mb_man, sig_mb_man = _cumulate(man_mb_rate, man_mb_err,
                                      man_years, baseline_window)

    if mouginot_df is None:
        # Mankoff only — same output as before
        return {
            'time_smb': man_years, 'H_smb': H_smb_man,
            'sigma_smb': sig_smb_man,
            'time_dyn': man_years, 'H_dyn': H_dyn_man,
            'sigma_dyn': sig_dyn_man,
            'time_mb': man_years, 'H_mb': H_mb_man,
            'sigma_mb': sig_mb_man,
            'sources': np.full(len(man_years), 'mankoff'),
            'df': df,
        }

    # ── Mouginot ──
    mou = mouginot_df.copy()
    mou_years = mou['decimal_year'].values.astype(float)
    # Mouginot reader already outputs m/yr SLE, SLR convention
    mou_smb_rate = mou['smb_rate'].values
    mou_smb_err = mou['smb_rate_sigma'].values
    mou_dyn_rate = mou['discharge_rate'].values
    mou_dyn_err = mou['discharge_rate_sigma'].values
    mou_mb_rate = mou['mb_rate'].values
    mou_mb_err = mou['mb_rate_sigma'].values

    # Apply year bounds
    mask = mou_years >= start_year
    if end_year is not None:
        mask &= mou_years <= end_year
    mou_years = mou_years[mask]
    mou_smb_rate = mou_smb_rate[mask]
    mou_smb_err = mou_smb_err[mask]
    mou_dyn_rate = mou_dyn_rate[mask]
    mou_dyn_err = mou_dyn_err[mask]
    mou_mb_rate = mou_mb_rate[mask]
    mou_mb_err = mou_mb_err[mask]

    H_smb_mou, sig_smb_mou = _cumulate(mou_smb_rate, mou_smb_err,
                                        mou_years, baseline_window)
    H_dyn_mou, sig_dyn_mou = _cumulate(mou_dyn_rate, mou_dyn_err,
                                        mou_years, baseline_window)
    H_mb_mou, sig_mb_mou = _cumulate(mou_mb_rate, mou_mb_err,
                                      mou_years, baseline_window)

    # ── Merge: concatenate and sort by time ──
    def _merge_sorted(t1, h1, s1, t2, h2, s2):
        t = np.concatenate([t1, t2])
        h = np.concatenate([h1, h2])
        s = np.concatenate([s1, s2])
        idx = np.argsort(t)
        return t[idx], h[idx], s[idx]

    t_smb, H_smb, sig_smb = _merge_sorted(
        man_years, H_smb_man, sig_smb_man,
        mou_years, H_smb_mou, sig_smb_mou)
    t_dyn, H_dyn, sig_dyn = _merge_sorted(
        man_years, H_dyn_man, sig_dyn_man,
        mou_years, H_dyn_mou, sig_dyn_mou)
    t_mb, H_mb, sig_mb = _merge_sorted(
        man_years, H_mb_man, sig_mb_man,
        mou_years, H_mb_mou, sig_mb_mou)

    sources = np.concatenate([
        np.full(len(man_years), 'mankoff'),
        np.full(len(mou_years), 'mouginot'),
    ])
    sources = sources[np.argsort(np.concatenate([man_years, mou_years]))]

    return {
        'time_smb': t_smb, 'H_smb': H_smb, 'sigma_smb': sig_smb,
        'time_dyn': t_dyn, 'H_dyn': H_dyn, 'sigma_dyn': sig_dyn,
        'time_mb': t_mb, 'H_mb': H_mb, 'sigma_mb': sig_mb,
        'sources': sources,
        'df': df,
    }


def prepare_mankoff_components(
    mankoff_path: str,
    baseline_window: tuple = (1995, 2005),
    start_year: int = 1972,
    end_year: Optional[int] = None,
) -> dict:
    """Backward-compatible wrapper around prepare_greenland_components.

    Mankoff only (no Mouginot).  See prepare_greenland_components.
    """
    return prepare_greenland_components(
        mankoff_path=mankoff_path,
        mouginot_df=None,
        baseline_window=baseline_window,
        start_year=start_year,
        end_year=end_year,
    )


def prepare_mouginot_components(
    mouginot_df: 'pd.DataFrame',
    baseline_window: tuple = (1995, 2005),
    start_year: int = 1972,
    end_year: Optional[int] = None,
) -> dict:
    """Prepare Mouginot SMB/D as cumulative SLE for the joint model.

    Uses the Mouginot DataFrame from ``read_mouginot2019_greenland()``,
    which provides separate ``smb_rate``, ``discharge_rate`` columns
    (m/yr SLE, SLR convention) for 1972–2018.

    Parameters
    ----------
    mouginot_df : DataFrame
        From ``read_mouginot2019_greenland()``.
    baseline_window : tuple
    start_year, end_year : int

    Returns
    -------
    dict with same keys as prepare_greenland_components.
    """
    mou = mouginot_df.copy()
    years = mou['decimal_year'].values.astype(float)

    mask = years >= start_year
    if end_year is not None:
        mask &= years <= end_year
    years = years[mask]
    smb_rate = mou['smb_rate'].values[mask]
    smb_err = mou['smb_rate_sigma'].values[mask]
    dyn_rate = mou['discharge_rate'].values[mask]
    dyn_err = mou['discharge_rate_sigma'].values[mask]
    mb_rate = mou['mb_rate'].values[mask]
    mb_err = mou['mb_rate_sigma'].values[mask]

    def _cumulate(rate, sigma, times, bl_window):
        bl_mask = (times >= bl_window[0]) & (times <= bl_window[1])
        if bl_mask.sum() > 0:
            rate_ref = rate[bl_mask].mean()
        else:
            rate_ref = 0.0
        rate_anom = rate - rate_ref
        dt = np.diff(times, prepend=times[0] - 1)
        cum = np.cumsum(rate_anom * dt)
        if bl_mask.sum() > 0:
            bl_mid_idx = np.argmin(np.abs(times - np.mean(bl_window)))
            cum = cum - cum[bl_mid_idx]
            # Bidirectional uncertainty from baseline midpoint
            cum_sig = np.zeros(len(times))
            sig_dt = sigma * dt
            for i in range(bl_mid_idx + 1, len(times)):
                cum_sig[i] = np.sqrt(cum_sig[i - 1]**2 + sig_dt[i]**2)
            for i in range(bl_mid_idx - 1, -1, -1):
                cum_sig[i] = np.sqrt(cum_sig[i + 1]**2 + sig_dt[i + 1]**2)
            cum_sig = np.maximum(cum_sig, sigma.mean())
        else:
            cum_sig = np.sqrt(np.cumsum((sigma * dt) ** 2))
        return cum, cum_sig

    H_smb, sig_smb = _cumulate(smb_rate, smb_err, years, baseline_window)
    H_dyn, sig_dyn = _cumulate(dyn_rate, dyn_err, years, baseline_window)
    H_mb, sig_mb = _cumulate(mb_rate, mb_err, years, baseline_window)

    return {
        'time_smb': years, 'H_smb': H_smb, 'sigma_smb': sig_smb,
        'time_dyn': years, 'H_dyn': H_dyn, 'sigma_dyn': sig_dyn,
        'time_mb': years, 'H_mb': H_mb, 'sigma_mb': sig_mb,
        'sources': np.full(len(years), 'mouginot'),
        'df': mou,
    }


# =========================================================================
#  Greenland discharge-only model
# =========================================================================

@dataclass
class BayesianGreenlandDischargeResult:
    """Result from discharge-only Greenland calibration.

    Model:
        H_dyn = γ_atm·I₁ + γ_ocean·∫D_eff dt + D₀·I₀ + H₀_dyn
        dD_eff/dt = (T_ocean − D_eff) / τ

    SMB is treated as fixed observations (RACMO-derived), not fit.
    """
    trace: az.InferenceData
    # Discharge parameters
    gamma_atm_posterior: np.ndarray
    gamma_ocean_posterior: np.ndarray
    tau_posterior: np.ndarray
    D0_posterior: np.ndarray
    H0_dyn_posterior: np.ndarray
    sigma_extra_dyn_posterior: np.ndarray
    # Model predictions
    H_dyn_model: np.ndarray           # (n_dyn,) at obs times
    D_eff_mean: Optional[np.ndarray]  # ODE state or None
    # SMB pass-through (not fit)
    H_smb_obs: np.ndarray
    sigma_smb_obs: np.ndarray
    time_smb: np.ndarray
    # Fit quality
    r2_dyn: float
    r2_total: float                   # R² of (H_smb_obs + H_dyn_model) vs (H_smb_obs + H_dyn_obs)
    # Observations
    time_dyn: np.ndarray
    H_dyn_obs: np.ndarray
    sigma_dyn_obs: np.ndarray
    sampler_diagnostics: Optional[dict] = None


def _greenland_discharge_log_prob(
    theta, I1_dyn, I0_dyn,
    H_dyn_obs, sigma_dyn,
    prior_scales,
    T_ocean_annual=None, dt_ocean=None, T0_ocean=0.0,
    obs_idx_ocean_dyn=None, n_ocean=0,
):
    """Log-posterior for discharge-only Greenland model.

    theta = [γ_atm, γ_ocean, log_τ, D₀, log_σ_dyn, H₀_dyn]  (6 params)
    H_dyn = γ_atm·I₁ + γ_ocean·∫D_eff + D₀·I₀ + H₀_dyn

    prior_scales layout:
        [0] scale_gamma_atm (HN σ)
        [1] scale_gamma_ocean (HN σ)
        [2] log_tau_mean
        [3] log_tau_sigma
        [4] D0_sigma
        [5] sig_extra_dyn_scale (HC γ)
        [6] H0_dyn_sigma
    """
    gamma_atm, gamma_ocean, log_tau, D0, log_sig_dyn, H0_dyn = theta

    if gamma_ocean < 0:
        return -np.inf
    if log_tau < -1 or log_tau > 7:
        return -np.inf
    if gamma_atm < 0:
        return -np.inf
    if log_sig_dyn > 0 or log_sig_dyn < -20:
        return -np.inf

    tau = np.exp(log_tau)
    sigma_extra_dyn = np.exp(log_sig_dyn)

    # ── Priors ──
    lp = 0.0
    scale_ga = prior_scales[0]
    lp += -0.5 * (gamma_atm / scale_ga) ** 2

    scale_go = prior_scales[1]
    lp += -0.5 * (gamma_ocean / scale_go) ** 2

    mu_lt = prior_scales[2]
    sig_lt = prior_scales[3]
    lp += -0.5 * ((log_tau - mu_lt) / sig_lt) ** 2

    sig_D0 = prior_scales[4]
    lp += -0.5 * (D0 / sig_D0) ** 2

    gamma_hc = prior_scales[5]
    lp += -np.log(1 + (sigma_extra_dyn / gamma_hc) ** 2) + log_sig_dyn

    sig_H0 = prior_scales[6]
    lp += -0.5 * ((H0_dyn - H_dyn_obs[0]) / sig_H0) ** 2

    # ── Discharge forward model (ODE) ──
    D_eff = np.empty(n_ocean)
    D_eff[0] = T0_ocean
    alpha = np.exp(-dt_ocean / tau)
    for i in range(n_ocean - 1):
        D_eff[i + 1] = (D_eff[i] * alpha[i]
                        + 0.5 * (T_ocean_annual[i] + T_ocean_annual[i + 1]) * (1.0 - alpha[i]))
    cum_D = np.zeros(n_ocean)
    for i in range(n_ocean - 1):
        cum_D[i + 1] = (cum_D[i]
                        + 0.5 * (D_eff[i] + D_eff[i + 1]) * dt_ocean[i])
    cum_D_dyn = cum_D[obs_idx_ocean_dyn]
    H_dyn_pred = (gamma_atm * I1_dyn + gamma_ocean * cum_D_dyn
                  + D0 * I0_dyn + H0_dyn)

    var_dyn = sigma_dyn ** 2 + sigma_extra_dyn ** 2
    resid_dyn = H_dyn_obs - H_dyn_pred
    lp += -0.5 * np.sum(resid_dyn ** 2 / var_dyn + np.log(var_dyn))

    return lp


def fit_bayesian_greenland_discharge(
    # ── Discharge observations ──
    H_dyn_obs: np.ndarray,
    sigma_dyn_obs: np.ndarray,
    time_dyn: np.ndarray,
    I1_dyn: np.ndarray,
    I0_dyn: np.ndarray,
    # ── SMB observations (pass-through, not fit) ──
    H_smb_obs: np.ndarray,
    sigma_smb_obs: np.ndarray,
    time_smb: np.ndarray,
    # ── Ocean temperature forcing ──
    T_ocean_monthly: np.ndarray,
    time_ocean_monthly: np.ndarray,
    # ── Priors ──
    prior_scale_gamma_atm: float = 0.002,
    prior_scale_gamma_ocean: float = 0.002,
    prior_log_tau_mean: Optional[float] = None,
    prior_log_tau_sigma: float = 0.5,
    prior_D0_sigma: float = 0.0005,
    prior_sigma_extra_dyn: float = 0.002,
    prior_H0_dyn_sigma: float = 0.005,
    # ── MCMC settings ──
    n_samples: int = 10000,
    n_walkers: int = 64,
    n_burnin: int = 5000,
    thin: int = 1,
    progress: bool = True,
    seed: Optional[int] = None,
) -> BayesianGreenlandDischargeResult:
    """Discharge-only Bayesian fit for Greenland.

    Fits the ODE discharge model against observed cumulative discharge
    from Mouginot et al. (2019).  SMB is passed through as fixed
    observations (RACMO-derived) and is not fit to temperature.

    Parameters
    ----------
    H_dyn_obs, sigma_dyn_obs, time_dyn : (n_dyn,)
        Observed cumulative discharge (m SLE, SLR convention).
    I1_dyn, I0_dyn : (n_dyn,)
        Design vectors for discharge (from Greenland surface T).
    H_smb_obs, sigma_smb_obs, time_smb : (n_smb,)
        Observed cumulative SMB (not fit, passed through for total MB).
    T_ocean_monthly, time_ocean_monthly : (n,)
        Monthly subsurface ocean temperature (°C).
    """
    if prior_log_tau_mean is None:
        prior_log_tau_mean = np.log(10.0)

    n_dyn = len(H_dyn_obs)
    ndim = 6
    param_names = ['gamma_atm', 'gamma_ocean', 'log_tau', 'D0',
                    'log_sigma_dyn', 'H0_dyn']

    # ── Ocean T setup ──
    year_floor = np.floor(time_ocean_monthly).astype(int)
    unique_years = np.unique(year_floor)
    T_ocean_annual = np.array([
        np.mean(T_ocean_monthly[year_floor == yr])
        for yr in unique_years
    ])
    time_ocean_annual = unique_years.astype(float) + 0.5
    n_ocean = len(time_ocean_annual)
    dt_ocean = np.diff(time_ocean_annual)

    n_spinup = min(10, n_ocean)
    T0_ocean = float(np.mean(T_ocean_annual[:n_spinup]))

    obs_idx_ocean_dyn = np.array([
        np.argmin(np.abs(time_ocean_annual - t)) for t in time_dyn
    ])

    if progress:
        print(f"  Dyn obs: {n_dyn} pts ({time_dyn[0]:.0f}–{time_dyn[-1]:.0f})")
        print(f"  SMB obs: {len(H_smb_obs)} pts (fixed, not fit)")
        print(f"  Ocean T: {n_ocean} annual pts "
              f"({time_ocean_annual[0]:.0f}–{time_ocean_annual[-1]:.0f})")
        print(f"  ODE init T_ocean(0) = {T0_ocean:.2f} °C")

    # ── Prior scales ──
    prior_scales = np.array([
        prior_scale_gamma_atm,    # [0]
        prior_scale_gamma_ocean,  # [1]
        prior_log_tau_mean,       # [2]
        prior_log_tau_sigma,      # [3]
        prior_D0_sigma,           # [4]
        prior_sigma_extra_dyn,    # [5]
        prior_H0_dyn_sigma,       # [6]
    ])

    # ── OLS initialization ──
    tau_init = np.exp(prior_log_tau_mean)
    D_eff_init = np.empty(n_ocean)
    D_eff_init[0] = T0_ocean
    alpha_init = np.exp(-dt_ocean / tau_init)
    for i in range(n_ocean - 1):
        D_eff_init[i + 1] = (D_eff_init[i] * alpha_init[i]
                              + 0.5 * (T_ocean_annual[i] + T_ocean_annual[i + 1]) * (1.0 - alpha_init[i]))
    cum_D_init = np.zeros(n_ocean)
    for i in range(n_ocean - 1):
        cum_D_init[i + 1] = (cum_D_init[i]
                              + 0.5 * (D_eff_init[i] + D_eff_init[i + 1])
                              * dt_ocean[i])
    cum_D_dyn_init = cum_D_init[obs_idx_ocean_dyn]

    X_dyn = np.column_stack([I1_dyn, cum_D_dyn_init, I0_dyn,
                              np.ones(n_dyn)])
    W_dyn = np.diag(1.0 / sigma_dyn_obs ** 2)
    try:
        beta_dyn = np.linalg.solve(X_dyn.T @ W_dyn @ X_dyn,
                                   X_dyn.T @ W_dyn @ H_dyn_obs)
    except np.linalg.LinAlgError:
        beta_dyn = np.linalg.lstsq(X_dyn, H_dyn_obs, rcond=None)[0]

    ga_init = max(beta_dyn[0], 1e-6)
    go_init = max(beta_dyn[1], 1e-6)
    D0_init = beta_dyn[2]
    H0d_init = beta_dyn[3]

    resid_init = H_dyn_obs - X_dyn @ beta_dyn
    sig_dyn_init = max(np.std(resid_init), 1e-5)

    theta0 = np.array([
        ga_init, go_init, np.log(tau_init), D0_init,
        np.log(sig_dyn_init), H0d_init,
    ])

    if progress:
        M = 1e3
        print(f"  OLS init: γ_atm={ga_init*M:.3f}, "
              f"γ_ocean={go_init*M:.3f} mm/yr/°C, "
              f"D₀={D0_init*M:.4f} mm/yr, τ={tau_init:.0f} yr")

    # ── Walker initialization ──
    rng = np.random.default_rng(seed)
    pos = np.empty((n_walkers, ndim))
    for i in range(n_walkers):
        p = theta0.copy()
        p[0] = abs(ga_init * (1.0 + 0.1 * rng.standard_normal()))
        p[1] = abs(go_init * (1.0 + 0.1 * rng.standard_normal()))
        p[2] = theta0[2] + 0.1 * rng.standard_normal()
        p[3] = D0_init + max(abs(D0_init), 1e-5) * 0.1 * rng.standard_normal()
        p[4] = theta0[4] + 0.1 * rng.standard_normal()
        p[5] = H0d_init + max(abs(H0d_init), 1e-4) * 0.05 * rng.standard_normal()
        pos[i] = p

    # ── MCMC ──
    moves = [
        (emcee.moves.DESnookerMove(), 0.8),
        (emcee.moves.DEMove(),        0.2),
    ]
    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, _greenland_discharge_log_prob,
        args=(I1_dyn, I0_dyn, H_dyn_obs, sigma_dyn_obs, prior_scales),
        kwargs={
            'T_ocean_annual': T_ocean_annual,
            'dt_ocean': dt_ocean,
            'T0_ocean': T0_ocean,
            'obs_idx_ocean_dyn': obs_idx_ocean_dyn,
            'n_ocean': n_ocean,
        },
        moves=moves,
    )

    if progress:
        print(f"  Running emcee: {n_walkers} walkers, "
              f"{n_burnin} burn-in + {n_samples} production "
              f"({ndim} params)...")

    sampler.run_mcmc(pos, n_burnin + n_samples, progress=progress)

    # ── Extract chains ──
    flat = sampler.get_chain(discard=n_burnin, thin=thin, flat=True)
    ga_s = flat[:, 0]
    go_s = flat[:, 1]
    tau_s = np.exp(flat[:, 2])
    D0_s = flat[:, 3]
    sig_dyn_s = np.exp(flat[:, 4])
    H0d_s = flat[:, 5]

    # ── Posterior-mean predictions ──
    ga_m = np.mean(ga_s)
    go_m = np.mean(go_s)
    tau_med = np.median(tau_s)
    D0_m = np.mean(D0_s)
    H0d_m = np.mean(H0d_s)

    D_eff_post = np.empty(n_ocean)
    D_eff_post[0] = T0_ocean
    alpha_post = np.exp(-dt_ocean / tau_med)
    for i in range(n_ocean - 1):
        D_eff_post[i + 1] = (D_eff_post[i] * alpha_post[i]
                              + 0.5 * (T_ocean_annual[i] + T_ocean_annual[i + 1]) * (1.0 - alpha_post[i]))
    cum_D_post = np.zeros(n_ocean)
    for i in range(n_ocean - 1):
        cum_D_post[i + 1] = (cum_D_post[i]
                              + 0.5 * (D_eff_post[i] + D_eff_post[i + 1])
                              * dt_ocean[i])
    cum_D_dyn_post = cum_D_post[obs_idx_ocean_dyn]
    H_dyn_pred = (ga_m * I1_dyn + go_m * cum_D_dyn_post
                  + D0_m * I0_dyn + H0d_m)

    def _r2(obs, pred):
        ss_res = np.sum((obs - pred) ** 2)
        ss_tot = np.sum((obs - np.mean(obs)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    r2_dyn = _r2(H_dyn_obs, H_dyn_pred)

    # Total MB R² (SMB is fixed observations)
    smb_set = set(np.round(time_smb, 1))
    dyn_set = set(np.round(time_dyn, 1))
    common_times = sorted(smb_set & dyn_set)
    if len(common_times) > 5:
        smb_idx = [i for i, t in enumerate(np.round(time_smb, 1))
                   if t in common_times]
        dyn_idx = [i for i, t in enumerate(np.round(time_dyn, 1))
                   if t in common_times]
        H_total_obs = H_smb_obs[smb_idx] + H_dyn_obs[dyn_idx]
        H_total_pred = H_smb_obs[smb_idx] + H_dyn_pred[dyn_idx]
        r2_total = _r2(H_total_obs, H_total_pred)
    else:
        r2_total = np.nan

    # ── Convergence diagnostics ──
    n_chains_arviz = min(4, n_walkers)
    chain_full = sampler.get_chain(discard=n_burnin, thin=thin)
    var_dict = {}
    for k, name in enumerate(param_names):
        var_dict[name] = chain_full[:, :n_chains_arviz, k].T
    trace = az.from_dict(var_dict)
    conv = check_convergence(trace, quiet=(not progress))

    diag = {
        'acceptance_fraction': sampler.acceptance_fraction.mean(),
        'n_walkers': n_walkers,
        'n_samples': n_samples,
        'n_burnin': n_burnin,
        'thin': thin,
        'convergence': conv,
    }

    if progress:
        M = 1e3
        print(f"\n  Dyn posterior: γ_atm={np.mean(ga_s)*M:.3f} "
              f"[{np.percentile(ga_s, 3)*M:.3f}, "
              f"{np.percentile(ga_s, 97)*M:.3f}] mm/yr/°C")
        print(f"    γ_ocean={np.mean(go_s)*M:.3f} "
              f"[{np.percentile(go_s, 3)*M:.3f}, "
              f"{np.percentile(go_s, 97)*M:.3f}] mm/yr/°C")
        print(f"    τ={np.median(tau_s):.1f} "
              f"[{np.percentile(tau_s, 3):.1f}, "
              f"{np.percentile(tau_s, 97):.1f}] yr")
        print(f"    D₀={np.mean(D0_s)*M:.4f} mm/yr, "
              f"σ_extra_dyn={np.median(sig_dyn_s)*M:.2f} mm")
        print(f"    R²_dyn={r2_dyn:.4f}")
        print(f"  Total MB: R²={r2_total:.4f}")
        print(f"  Acceptance: {diag['acceptance_fraction']:.2f}")

    return BayesianGreenlandDischargeResult(
        trace=trace,
        gamma_atm_posterior=ga_s,
        gamma_ocean_posterior=go_s,
        tau_posterior=tau_s,
        D0_posterior=D0_s,
        H0_dyn_posterior=H0d_s,
        sigma_extra_dyn_posterior=sig_dyn_s,
        H_dyn_model=H_dyn_pred,
        D_eff_mean=D_eff_post,
        H_smb_obs=H_smb_obs,
        sigma_smb_obs=sigma_smb_obs,
        time_smb=time_smb,
        r2_dyn=r2_dyn,
        r2_total=r2_total,
        time_dyn=time_dyn,
        H_dyn_obs=H_dyn_obs,
        sigma_dyn_obs=sigma_dyn_obs,
        sampler_diagnostics=diag,
    )
