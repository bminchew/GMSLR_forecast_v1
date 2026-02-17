#!/usr/bin/env python3
"""
Bayesian Complement to DOLS Sea-Level Rate Models
====================================================

Three Bayesian models that complement the frequentist DOLS framework:

1. **Bayesian Static DOLS** — Same regression as calibrate_dols() but with
   posterior distributions instead of point estimates + HAC SEs.
2. **Bayesian DLM** — Time-varying coefficients via Gaussian random walk,
   providing a principled alternative to sliding-window DOLS.
3. **Hierarchical Multi-Dataset** — Partial pooling across GMSL datasets,
   replacing ad hoc ensemble statistics.

Backend: emcee (affine-invariant ensemble sampler) + arviz (diagnostics).
Design matrix construction replicates calibrate_dols() exactly.

Authors: Minchew research group, 2026
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

warnings.filterwarnings("ignore", category=FutureWarning)

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from slr_analysis import calibrate_dols, DOLSResult


# =============================================================================
# DESIGN MATRIX BUILDER  (replicates calibrate_dols lines 1023–1138)
# =============================================================================

def _to_month_start(s: pd.Series) -> pd.Series:
    """Snap datetime index to the first of each month."""
    new_idx = s.index.to_period('M').to_timestamp()
    out = s.copy()
    out.index = new_idx
    return out[~out.index.duplicated(keep='first')]


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
    dt = np.median(np.diff(time_years))

    # ---- 3. Trapezoidal integrals ∫Tᵏ ----
    integrals = []
    col_names = []
    for k in range(order, 0, -1):
        Tk = T ** k
        integral_Tk = np.zeros(n)
        for i in range(1, n):
            integral_Tk[i] = integral_Tk[i - 1] + 0.5 * (Tk[i] + Tk[i - 1]) * dt
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
            integral_S[i] = integral_S[i - 1] + 0.5 * (S[i] + S[i - 1]) * dt
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
    # Reshape flat_samples to (n_chains, n_draw, ndim) for arviz
    n_chains_arviz = min(4, n_walkers)
    chain_len = flat_samples.shape[0] // n_chains_arviz
    samples_reshaped = flat_samples[:n_chains_arviz * chain_len].reshape(
        n_chains_arviz, chain_len, ndim
    )

    # Physical samples reshaped for arviz
    phys_reshaped = np.zeros((n_chains_arviz, chain_len, n_phys))
    for c in range(n_chains_arviz):
        chain_beta = samples_reshaped[c, :, :n_cols]
        chain_phys_unscaled = unscale_coefficients(chain_beta, scaler)
        phys_reshaped[c] = chain_phys_unscaled[:, :n_phys]

    # Name the physical parameters
    phys_names = dm['col_names'][:n_phys]
    var_dict = {name: phys_reshaped[:, :, k]
                for k, name in enumerate(phys_names)}

    trace = az.from_dict(var_dict)

    # Sampler diagnostics
    diag = {
        'acceptance_fraction': sampler.acceptance_fraction.mean(),
        'n_walkers': n_walkers,
        'n_samples': n_samples,
        'n_burnin': n_burnin,
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
        P = P_pred - np.outer(K_t, K_t) * S_t  # Joseph form simplification

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

    # Build arviz trace for Q
    n_chains_arviz = min(4, n_walkers)
    chain_len = Q_samples.shape[0] // n_chains_arviz
    Q_reshaped = Q_samples[:n_chains_arviz * chain_len].reshape(
        n_chains_arviz, chain_len, n_phys
    )
    phys_names = dm['col_names'][:n_phys]
    var_dict = {f"Q_{name}": Q_reshaped[:, :, k]
                for k, name in enumerate(phys_names)}
    trace = az.from_dict(var_dict)

    diag = {
        'acceptance_fraction': sampler.acceptance_fraction.mean(),
        'n_walkers': n_walkers,
        'n_samples': n_samples,
        'n_burnin': n_burnin,
        'n_smoother_subsamples': n_sub,
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

    For dataset i with n_nuisance_i nuisance params:
      z_i = theta[offset : offset + n_phys]
      nuis_i = theta[offset + n_phys : offset + n_phys + n_nuisance_i]
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
        n_per_dataset = n_phys + n_nuis

        # Extract dataset-specific params
        z_i = theta[offset:offset + n_phys]
        nuis_i = theta[offset + n_phys:offset + n_per_dataset]
        offset += n_per_dataset

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

        if sigma is not None:
            ll = -0.5 * np.sum((resid / sigma) ** 2 + 2 * np.log(sigma))
        else:
            # Estimate sigma from residuals (uniform prior on log sigma)
            sig_est = np.std(resid)
            if sig_est <= 0:
                return -np.inf
            ll = -0.5 * len(resid) * np.log(2 * np.pi * sig_est ** 2) \
                 - 0.5 * np.sum(resid ** 2) / sig_est ** 2

        if not np.isfinite(ll):
            return -np.inf
        lp += ll

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

    # Parameter count
    n_pop = 2 * n_phys  # mu + log_sigma
    n_per_dataset = [n_phys + d['n_nuisance'] for d in datasets_info]
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
            # z_i ≈ (ols_i - mu) / sigma — near the non-centered OLS deviation
            z_init = (ols_phys[di] - mu_ols) / std_ols
            p0[i, offset:offset + n_phys] = z_init + 0.05 * np.random.randn(n_phys)
            # Nuisance near OLS
            p0[i, offset + n_phys:offset + n_per] = ols_nuis[di] + 0.01 * np.random.randn(len(ols_nuis[di]))
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

    # Build arviz trace
    n_chains_arviz = min(4, n_walkers)
    chain_len = flat_samples.shape[0] // n_chains_arviz
    mu_reshaped = mu_samples[:n_chains_arviz * chain_len].reshape(
        n_chains_arviz, chain_len, n_phys
    )
    phys_names = datasets_info[0]['dm']['col_names'][:n_phys]
    var_dict = {f"μ_{name}": mu_reshaped[:, :, k]
                for k, name in enumerate(phys_names)}
    var_dict["σ_pop"] = sigma_pop_samples[:n_chains_arviz * chain_len].reshape(
        n_chains_arviz, chain_len, n_phys
    )
    trace = az.from_dict(var_dict)

    diag = {
        'acceptance_fraction': sampler.acceptance_fraction.mean(),
        'n_walkers': n_walkers,
        'n_samples': n_samples,
        'n_burnin': n_burnin,
        'n_datasets': n_datasets,
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
