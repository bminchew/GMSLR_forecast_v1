"""
Bayesian Predictive Synthesis (BPS) trend constraint for GMSL projections.

Combines a non-parametric WLS trend forecast with the rate-and-state model
projection using BPS (McAlinn & West, 2019). The synthesis hyperparameters
are calibrated from leave-future-out cross-validation on the satellite-era
record. Their posterior uncertainty is propagated through the combined
predictive, producing credible intervals that account for uncertainty in
both the individual forecasts and the combination itself.

See companion document: near_term_trend_constraint_revised.tex

Depends on existing project infrastructure:
    - bayesian_dols.check_convergence()  — MCMC convergence diagnostics
    - slr_analysis.calibrate_dols()      — DOLS model fitting (for CV refit)
    - slr_projections.project_gmsl_state_ensemble() — model projections (for CV)
"""

import warnings
from dataclasses import dataclass, field
from typing import Optional

import arviz as az
import emcee
import numpy as np
import statsmodels.api as sm
from scipy import optimize, special, stats

try:
    from slr_forecast.config import Z_90
except ImportError:
    Z_90 = 1.645

# =============================================================================
# CONSTANTS
# =============================================================================

N_ENSEMBLE = 10_000
SATELLITE_START = 1993.0

# Option C exponential weight defaults
DEFAULT_TAU_0 = 18.2       # yr, from bayesian_ratestate_results.json
DEFAULT_DELTA_T0 = 1.0     # °C, temperature at which effective timescale halves


# =============================================================================
# SECTION 1: TEMPERATURE DEPARTURE
# =============================================================================

def compute_T_cal_max(T_cal: np.ndarray) -> float:
    """Maximum GMST in the calibration window."""
    return float(np.max(T_cal))


def compute_delta_T(
    T_scenario: np.ndarray,
    T_max_cal: float,
    overshoot_mode: bool = True,
) -> np.ndarray:
    """Temperature departure from the calibration domain.

    Parameters
    ----------
    T_scenario : array, shape (n_proj,)
        Scenario temperature trajectory for t > t_end.
    T_max_cal : float
        Maximum GMST during calibration (from compute_T_cal_max).
    overshoot_mode : bool
        If True (default), ΔT(t) = cummax(max(0, T(s) - T_max_cal)).
        If False, ΔT(t) = max(0, T(t) - T_max_cal).

    Returns
    -------
    delta_T : array, shape (n_proj,)
    """
    exceedance = np.maximum(0.0, T_scenario - T_max_cal)
    if overshoot_mode:
        return np.maximum.accumulate(exceedance)
    return exceedance


# =============================================================================
# SECTION 2: WLS TREND MODEL
# =============================================================================

def fit_wls_quadratic(
    years: np.ndarray,
    gmsl: np.ndarray,
    gmsl_uncertainty: Optional[np.ndarray] = None,
    satellite_start: float = SATELLITE_START,
    fix_scale: bool = True,
) -> dict:
    """Fit WLS quadratic H(t) = α + β·(t-t_ref) + γ·(t-t_ref)² to
    satellite-era GMSL.

    Parameters
    ----------
    years : array
        Decimal years of observations.
    gmsl : array
        GMSL observations (meters).
    gmsl_uncertainty : array or None
        Per-observation 1-σ uncertainties (meters). If None, OLS is used.
    satellite_start : float
        Start year for satellite era.
    fix_scale : bool
        If True, use cov_type='fixed scale' (trust observation uncertainties
        as absolute). If False, estimate residual variance.

    Returns
    -------
    dict with keys: 'params', 'cov_params', 't_ref', 'sigma_sq_est',
                    'wls_result', 'n_obs'
    """
    mask = years >= satellite_start
    t = years[mask]
    h = gmsl[mask]
    sigma = gmsl_uncertainty[mask] if gmsl_uncertainty is not None else None

    t_ref = 0.5 * (t[0] + t[-1])
    dt = t - t_ref

    # Design matrix: [1, dt, dt²]
    X = np.column_stack([np.ones_like(dt), dt, dt**2])

    if sigma is not None:
        weights = 1.0 / sigma**2
    else:
        weights = np.ones_like(h)

    model = sm.WLS(h, X, weights=weights)
    if fix_scale and sigma is not None:
        result = model.fit(cov_type='fixed scale')
    else:
        result = model.fit()

    # Estimated residual variance (diagnostic)
    resid = h - result.fittedvalues
    if sigma is not None:
        sigma_sq_est = float(np.sum((resid / sigma)**2) / (len(h) - 3))
    else:
        sigma_sq_est = float(np.sum(resid**2) / (len(h) - 3))

    return {
        'params': result.params,           # [α, β, γ]
        'cov_params': result.cov_params(),  # (3, 3)
        't_ref': t_ref,
        'sigma_sq_est': sigma_sq_est,
        'wls_result': result,
        'n_obs': len(h),
    }


def predict_at_t_end(
    wls_fit: dict,
    t_end: float,
    sigma_gia: float = 0.15e-3,
    sigma_ais_sys: float = 0.10e-3,
) -> dict:
    """Extract posterior on (H, r, r̈) at t_end via Jacobian propagation.

    The Jacobian maps (α, β, γ) → (H, r, r̈) at t_end:
        J = [[1, Δ, Δ²], [0, 1, 2Δ], [0, 0, 2]]
    where Δ = t_end - t_ref.

    Systematic inflations (GIA, WAIS subtraction) are rank-1:
        Σ_sys = σ² · v · v^T,  v = (Δ, 1, 0)

    Parameters
    ----------
    wls_fit : dict
        Output of fit_wls_quadratic().
    t_end : float
        End of calibration window (decimal year).
    sigma_gia : float
        GIA rate uncertainty in m/yr (default 0.15 mm/yr = 0.15e-3 m/yr).
    sigma_ais_sys : float
        WAIS subtraction rate systematic in m/yr (default 0.10 mm/yr).

    Returns
    -------
    dict with keys: 'H_mean', 'r_mean', 'rdot_mean', 'mean_3', 'cov_3x3'
    """
    params = wls_fit['params']
    cov_params = wls_fit['cov_params']
    t_ref = wls_fit['t_ref']

    delta = t_end - t_ref

    # Jacobian: (α, β, γ) → (H, r, r̈)
    J = np.array([
        [1.0, delta, delta**2],
        [0.0, 1.0,   2 * delta],
        [0.0, 0.0,   2.0],
    ])

    # Transform mean
    mean_3 = J @ params  # [H, r, r̈] at t_end

    # Transform covariance (WLS)
    cov_wls = J @ cov_params @ J.T

    # Rank-1 systematic inflations: v = (Δ, 1, 0)
    v = np.array([delta, 1.0, 0.0])
    cov_gia = sigma_gia**2 * np.outer(v, v)
    cov_ais = sigma_ais_sys**2 * np.outer(v, v)

    cov_total = cov_wls + cov_gia + cov_ais

    return {
        'H_mean': float(mean_3[0]),
        'r_mean': float(mean_3[1]),
        'rdot_mean': float(mean_3[2]),
        'mean_3': mean_3,
        'cov_3x3': cov_total,
    }


def sample_derivatives(
    prediction: dict,
    n_samples: int = N_ENSEMBLE,
    seed: Optional[int] = None,
) -> dict:
    """Draw (H, r, rdot) samples from MVN(mean, cov_3x3).

    Returns
    -------
    dict with keys: 'H', 'r', 'rdot', 'samples' (n_samples, 3)
    """
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(
        prediction['mean_3'], prediction['cov_3x3'], size=n_samples,
    )
    return {
        'H': samples[:, 0],
        'r': samples[:, 1],
        'rdot': samples[:, 2],
        'samples': samples,
    }


def bma_cubic_check(
    years: np.ndarray,
    gmsl: np.ndarray,
    gmsl_uncertainty: Optional[np.ndarray],
    t_end: float,
    satellite_start: float = SATELLITE_START,
) -> dict:
    """BMA quadratic-vs-cubic via BIC.

    Adapts the BIC computation pattern from polyfit_model_selection() in
    ipcc_emergent_sensitivity.py.

    P(M_quad | D) ≈ exp(-BIC_q/2) / [exp(-BIC_q/2) + exp(-BIC_c/2)]

    Returns
    -------
    dict with: 'bma_prob_cubic', 'bma_prob_quadratic', 'cubic_coeff',
               'cubic_coeff_se', 'scale_diagnostic_flag', 'sigma_sq_estimated'
    """
    mask = years >= satellite_start
    t = years[mask]
    h = gmsl[mask]
    sigma = gmsl_uncertainty[mask] if gmsl_uncertainty is not None else None
    n = len(t)

    t_ref = 0.5 * (t[0] + t[-1])
    dt = t - t_ref

    if sigma is not None:
        weights = 1.0 / sigma**2
    else:
        weights = np.ones(n)

    # Quadratic: H = α + β·dt + γ·dt²
    X_q = np.column_stack([np.ones(n), dt, dt**2])
    model_q = sm.WLS(h, X_q, weights=weights)
    res_q = model_q.fit()
    resid_q = h - res_q.fittedvalues
    if sigma is not None:
        ssr_q = float(np.sum((resid_q / sigma)**2))
    else:
        ssr_q = float(np.sum(resid_q**2))
    k_q = 3

    # Cubic: H = α + β·dt + γ·dt² + δ·dt³
    X_c = np.column_stack([np.ones(n), dt, dt**2, dt**3])
    model_c = sm.WLS(h, X_c, weights=weights)
    res_c = model_c.fit()
    resid_c = h - res_c.fittedvalues
    if sigma is not None:
        ssr_c = float(np.sum((resid_c / sigma)**2))
    else:
        ssr_c = float(np.sum(resid_c**2))
    k_c = 4

    # BIC: n·ln(SSR/n) + k·ln(n)
    bic_q = n * np.log(ssr_q / n) + k_q * np.log(n)
    bic_c = n * np.log(ssr_c / n) + k_c * np.log(n)

    # BMA posterior model probabilities via BIC approximation
    log_num_q = -0.5 * bic_q
    log_num_c = -0.5 * bic_c
    log_denom = np.logaddexp(log_num_q, log_num_c)
    prob_quad = np.exp(log_num_q - log_denom)
    prob_cubic = np.exp(log_num_c - log_denom)

    # Cubic coefficient and its SE
    cubic_coeff = float(res_c.params[3])
    cubic_se = float(res_c.bse[3]) if len(res_c.bse) > 3 else np.nan

    # Scale diagnostic
    sigma_sq_est = ssr_q / (n - k_q) if n > k_q else np.nan

    return {
        'bma_prob_quadratic': float(prob_quad),
        'bma_prob_cubic': float(prob_cubic),
        'cubic_coeff': cubic_coeff,
        'cubic_coeff_se': cubic_se,
        'bic_quadratic': float(bic_q),
        'bic_cubic': float(bic_c),
        'sigma_sq_estimated': float(sigma_sq_est),
        'scale_diagnostic_flag': sigma_sq_est > 2.0,
    }


# =============================================================================
# SECTION 3: TREND PROJECTION ENSEMBLE
# =============================================================================

def trend_projection_ensemble(
    t_proj: np.ndarray,
    t_end: float,
    derivative_samples: dict,
) -> np.ndarray:
    """Generate trend projection ensemble. Vectorised.

    H_trend^(s)(t) = H^(s) + r^(s)·dt + 0.5·rdot^(s)·dt²

    Parameters
    ----------
    t_proj : array, shape (n_proj,)
        Projection times.
    t_end : float
        End of calibration window.
    derivative_samples : dict
        Output of sample_derivatives(): keys 'H', 'r', 'rdot'.

    Returns
    -------
    H_ensemble : array, shape (n_samples, n_proj)
    """
    dt = t_proj - t_end  # (n_proj,)
    H = derivative_samples['H'][:, np.newaxis]       # (n_samples, 1)
    r = derivative_samples['r'][:, np.newaxis]
    rdot = derivative_samples['rdot'][:, np.newaxis]
    return H + r * dt + 0.5 * rdot * dt**2


# =============================================================================
# SECTION 4: BPS SYNTHESIS
# =============================================================================

def compute_synthesis_weight(
    dt: np.ndarray,
    delta_T: np.ndarray,
    phi_0: float,
    kappa_net: float,
    kappa_t: float,
) -> np.ndarray:
    """w(t) = σ(φ₀ - κ_net·ΔT²·dt² - κ_t·dt⁴). Vectorised."""
    phi = phi_0 - kappa_net * delta_T**2 * dt**2 - kappa_t * dt**4
    return special.expit(phi)


def synthesis_log_prior(theta: np.ndarray) -> float:
    """Log-prior for ψ = (φ₀, log κ_net, log κ_t).

    φ₀ ~ TruncN(2.5, 0.5², φ₀ > 0)
    κ_net ~ LogN(log(0.5), 0.6)  → log κ_net ~ N(log(0.5), 0.6²)
    κ_t ~ LogN(log(1e-5), 0.6)   → log κ_t ~ N(log(1e-5), 0.6²)

    Note: dt is in years. Prior medians calibrated so that:
    - κ_net·ΔT²·dt² ~ O(1) at dt=10 yr, ΔT=0.15 °C
    - κ_t·dt⁴ ~ O(1) at dt=20 yr
    """
    phi_0, log_kn, log_kt = theta

    if phi_0 <= 0:
        return -np.inf

    # φ₀: truncated normal (truncation at 0)
    lp_phi0 = stats.norm.logpdf(phi_0, loc=2.5, scale=0.5)

    # log κ_net: normal in log-space
    lp_kn = stats.norm.logpdf(log_kn, loc=np.log(0.5), scale=0.6)

    # log κ_t: normal in log-space
    lp_kt = stats.norm.logpdf(log_kt, loc=np.log(1e-5), scale=0.6)

    return lp_phi0 + lp_kn + lp_kt


def synthesis_log_likelihood(theta: np.ndarray, cv_data: list) -> float:
    """Log-sum-exp stable mixture log-likelihood over CV holdouts.

    Parameters
    ----------
    theta : (3,) — [φ₀, log κ_net, log κ_t]
    cv_data : list of dicts, each with keys:
        'dt' : array of lead times (years)
        'delta_T' : array of ΔT values at each holdout time
        'log_h_trend' : array of log trend densities at observed H
        'log_h_model' : array of log model densities at observed H

    Returns
    -------
    float : total log-likelihood
    """
    phi_0, log_kn, log_kt = theta
    kappa_net = np.exp(log_kn)
    kappa_t = np.exp(log_kt)

    total_ll = 0.0
    for fold in cv_data:
        dt = fold['dt']
        dT = fold['delta_T']
        log_ht = fold['log_h_trend']
        log_hm = fold['log_h_model']

        # Weight at each holdout observation
        phi = phi_0 - kappa_net * dT**2 * dt**2 - kappa_t * dt**4
        log_w = -np.logaddexp(0.0, -phi)       # log(σ(φ))
        log_1mw = -np.logaddexp(0.0, phi)       # log(1 - σ(φ))

        # log(w·h_trend + (1-w)·h_model) via log-sum-exp
        log_mix = np.logaddexp(log_w + log_ht, log_1mw + log_hm)
        total_ll += np.sum(log_mix)

    if not np.isfinite(total_ll):
        return -np.inf
    return total_ll


def synthesis_log_posterior(theta: np.ndarray, cv_data: list) -> float:
    """Log-posterior = log-prior + log-likelihood."""
    lp = synthesis_log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + synthesis_log_likelihood(theta, cv_data)


# --- Option C: Exponential decay weight functions ---

def compute_synthesis_weight_exp(
    dt: np.ndarray,
    delta_T: np.ndarray,
    tau_0: float,
    delta_T0: float,
) -> np.ndarray:
    """w(t) = exp(−dt / τ_eff) where τ_eff = τ₀ / (1 + (ΔT/ΔT₀)²).

    Parameters
    ----------
    dt : array, shape (n_proj,)
        Lead time in years (t − t_end).
    delta_T : array, shape (n_proj,)
        Temperature departure from calibration domain (°C).
    tau_0 : float
        Baseline relaxation timescale (years).
    delta_T0 : float
        Temperature scale parameter (°C): ΔT at which τ_eff = τ₀/2.

    Returns
    -------
    w : array, shape (n_proj,)
        Synthesis weight in [0, 1].
    """
    tau_eff = tau_0 / (1.0 + (delta_T / delta_T0)**2)
    # Guard against division by zero
    tau_eff = np.maximum(tau_eff, 1e-10)
    return np.exp(-dt / tau_eff)


def synthesis_log_prior_exp(theta: np.ndarray) -> float:
    """Log-prior for ψ = (log ΔT₀,).

    ΔT₀ ~ LogNormal(log(1.0), 0.5)  →  log ΔT₀ ~ N(0, 0.5²)

    This places the median at 1.0°C with 90% CI ≈ [0.44, 2.27] °C.
    """
    log_dT0 = theta[0]
    return stats.norm.logpdf(log_dT0, loc=0.0, scale=0.5)


def synthesis_log_likelihood_exp(
    theta: np.ndarray,
    cv_data: list,
    tau_0: float,
) -> float:
    """Log-likelihood for exponential weight form over CV holdouts.

    Parameters
    ----------
    theta : (1,) — [log ΔT₀]
    cv_data : list of dicts (same format as sigmoid version)
    tau_0 : float or array
        If array, marginalize by averaging likelihoods over a random
        subset of τ₀ samples (Monte Carlo integration).

    Returns
    -------
    float : total log-likelihood
    """
    log_dT0 = theta[0]
    delta_T0 = np.exp(log_dT0)

    if delta_T0 < 1e-6:
        return -np.inf

    tau_0_arr = np.atleast_1d(tau_0)
    n_tau = len(tau_0_arr)

    # If many τ₀ samples, subsample for speed
    if n_tau > 200:
        rng = np.random.default_rng(42)
        tau_0_arr = rng.choice(tau_0_arr, 200, replace=False)
        n_tau = 200

    # Compute log-likelihood for each tau_0 value
    ll_per_tau = np.zeros(n_tau)
    for ti, tau_val in enumerate(tau_0_arr):
        total_ll = 0.0
        for fold in cv_data:
            dt = fold['dt']
            dT = fold['delta_T']
            log_ht = fold['log_h_trend']
            log_hm = fold['log_h_model']

            # Effective timescale
            tau_eff = tau_val / (1.0 + (dT / delta_T0)**2)
            tau_eff = np.maximum(tau_eff, 1e-10)

            # Weight
            w = np.exp(-dt / tau_eff)
            w = np.clip(w, 1e-15, 1.0 - 1e-15)

            log_w = np.log(w)
            log_1mw = np.log(1.0 - w)

            # log(w·h_trend + (1-w)·h_model) via log-sum-exp
            log_mix = np.logaddexp(log_w + log_ht, log_1mw + log_hm)
            total_ll += np.sum(log_mix)

        ll_per_tau[ti] = total_ll

    # Marginalise over τ₀ samples via log-sum-exp
    if n_tau == 1:
        result = ll_per_tau[0]
    else:
        result = special.logsumexp(ll_per_tau) - np.log(n_tau)

    if not np.isfinite(result):
        return -np.inf
    return result


def synthesis_log_posterior_exp(
    theta: np.ndarray, cv_data: list, tau_0: float,
) -> float:
    """Log-posterior = log-prior + log-likelihood (exponential weight)."""
    lp = synthesis_log_prior_exp(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + synthesis_log_likelihood_exp(theta, cv_data, tau_0)


def make_model_refit_fn(
    full_sl, full_temp, full_time, full_sigma,
    historical_temperature, historical_time,
    tau_samples, baseline_year=2005.0,
    n_samples=500, order=2, n_lags=2,
):
    """Factory for model_refit_fn using existing calibrate_dols() +
    project_gmsl_state_ensemble().

    Parameters
    ----------
    full_sl : pd.Series
        Full GMSL time series (for refitting).
    full_temp : pd.Series
        Full temperature time series.
    full_time : np.ndarray
        Decimal years for the full record.
    full_sigma : pd.Series or None
        GMSL uncertainties.
    historical_temperature : np.ndarray
        Historical temperature for ODE spin-up.
    historical_time : np.ndarray
        Decimal years for historical temperature.
    tau_samples : np.ndarray
        Posterior samples of relaxation time τ (years).
    baseline_year : float
    n_samples : int
        MC samples per refit (500 for CV speed).
    order, n_lags : int
        DOLS parameters.

    Returns
    -------
    callable : t_h -> dict with 'H_predictive_mean', 'H_predictive_std',
               'time' (decimal years of holdout period)
    """
    # Import here to avoid circular imports at module level
    from slr_analysis import calibrate_dols
    from slr_projections import project_gmsl_state_ensemble

    import pandas as pd

    def refit_fn(t_h):
        # 1. Subset data to [start, t_h]
        mask_cal = full_time <= t_h
        sl_sub = full_sl[mask_cal]
        temp_sub = full_temp[mask_cal]
        sigma_sub = full_sigma[mask_cal] if full_sigma is not None else None

        # 2. Refit DOLS (fast, ~0.1s)
        result = calibrate_dols(
            sl_sub, temp_sub, gmsl_sigma=sigma_sub,
            order=order, n_lags=n_lags,
        )

        # 3. Build holdout temperature trajectory
        mask_holdout = full_time > t_h
        holdout_time = full_time[mask_holdout]
        holdout_temp = full_temp[mask_holdout]

        if len(holdout_time) == 0:
            return {
                'H_predictive_mean': np.array([]),
                'H_predictive_std': np.array([]),
                'time': np.array([]),
            }

        # Build DataFrame for project_gmsl_state_ensemble
        temp_df = pd.DataFrame({
            'temperature': holdout_temp.values if hasattr(holdout_temp, 'values') else holdout_temp,
            'decimal_year': holdout_time,
        })

        # Historical temperature for ODE spin-up (up to t_h)
        hist_mask = historical_time <= t_h
        hist_temp_sub = historical_temperature[hist_mask]
        hist_time_sub = historical_time[hist_mask]

        # 4. Project
        proj_result = project_gmsl_state_ensemble(
            coefficients=result.physical_coefficients,
            coefficients_cov=result.physical_covariance,
            tau_samples=tau_samples,
            temperature_projections={'holdout': temp_df},
            historical_temperature=hist_temp_sub,
            historical_time=hist_time_sub,
            n_samples=n_samples,
            baseline_year=baseline_year,
            temp_col='temperature',
        )

        # 5. Extract mean and std at holdout times
        scen = proj_result['scenarios']['holdout']
        gmsl_med = scen['gmsl'].values
        gmsl_upper = scen['gmsl_upper'].values
        gmsl_lower = scen['gmsl_lower'].values
        # Convert 5th-95th percentile range to std (Gaussian approx)
        gmsl_std = (gmsl_upper - gmsl_lower) / (2 * Z_90)

        return {
            'H_predictive_mean': gmsl_med,
            'H_predictive_std': gmsl_std,
            'time': holdout_time,
        }

    return refit_fn


class BPSSynthesis:
    """Orchestrates BPS: CV data, posterior sampling, weight computation."""

    def __init__(
        self,
        n_walkers: int = 16,
        n_warmup: int = 2000,
        n_samples: int = 5000,
        seed: Optional[int] = None,
        delta_T_power: int = 2,
        weight_mode: str = 'sigmoid',
        tau_0: float = DEFAULT_TAU_0,
        tau_0_samples: Optional[np.ndarray] = None,
        delta_T0: float = DEFAULT_DELTA_T0,
    ):
        """
        Parameters
        ----------
        n_walkers : int
            Number of emcee walkers.
        n_warmup : int
            Burn-in steps.
        n_samples : int
            Production steps per walker.
        seed : int or None
            Random seed.
        delta_T_power : int
            Power of ΔT in the weight process (2 = default, 4 = optimistic).
            Only used for weight_mode='sigmoid'.
        weight_mode : str
            Weight function form:
            - 'sigmoid': Original φ₀ − κ_net·ΔT²·dt² − κ_t·dt⁴ (3 params)
            - 'heuristic': Exponential w=exp(−dt/τ_eff) with fixed τ₀, ΔT₀
            - 'cv': Exponential with τ₀ fixed, ΔT₀ calibrated via MCMC
            - 'state': Exponential with τ₀ per-sample from posterior, ΔT₀ via MCMC
        tau_0 : float
            Baseline relaxation timescale in years (default from rate-and-state).
        tau_0_samples : array or None
            Per-sample τ₀ values from state model posterior. Required for 'state' mode.
        delta_T0 : float
            Temperature scale parameter in °C. For 'heuristic', used directly.
            For 'cv'/'state', used as initial value for MCMC.
        """
        self.n_walkers = n_walkers
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.seed = seed
        self.delta_T_power = delta_T_power
        self.weight_mode = weight_mode
        self.tau_0 = tau_0
        self.tau_0_samples = tau_0_samples
        self.delta_T0 = delta_T0

        if weight_mode == 'state' and tau_0_samples is None:
            raise ValueError("weight_mode='state' requires tau_0_samples")
        if weight_mode not in ('sigmoid', 'heuristic', 'cv', 'state'):
            raise ValueError(f"Unknown weight_mode: {weight_mode}")

        self.emcee_trace = None
        self.emcee_sampler = None
        self.dynesty_results = None
        self._posterior_samples = None  # sigmoid: (n, 3), exp: (n, 1)
        self._dynesty_posterior = None

    def build_cv_data(
        self,
        years: np.ndarray,
        gmsl: np.ndarray,
        gmsl_uncertainty: Optional[np.ndarray],
        T_obs: np.ndarray,
        model_refit_fn,
        holdout_endpoints: Optional[np.ndarray] = None,
        min_holdout: int = 5,
        satellite_start: float = SATELLITE_START,
    ) -> list:
        """Leave-future-out cross-validation.

        For each t_h:
        1. fit_wls_quadratic() on [satellite_start, t_h]
        2. Recompute T_cal_max(t_h) to avoid information leakage
        3. model_refit_fn(t_h) → model predictive
        4. Evaluate trend/model Gaussian log-densities at holdout obs

        Parameters
        ----------
        years : array
            Decimal years of GMSL observations.
        gmsl : array
            GMSL observations (meters, non-WAIS).
        gmsl_uncertainty : array or None
        T_obs : array
            Observed temperature (same time grid as years).
        model_refit_fn : callable
            t_h -> dict with 'H_predictive_mean', 'H_predictive_std', 'time'
        holdout_endpoints : array or None
            If None, uses [2003, 2005, ..., max(years) - min_holdout].
        min_holdout : int
            Minimum holdout years.
        satellite_start : float

        Returns
        -------
        list of dicts (one per holdout endpoint), each with:
            't_h', 'dt', 'delta_T', 'log_h_trend', 'log_h_model',
            'H_obs_holdout'
        """
        t_max = years[-1] if len(years) > 0 else 2020.0

        if holdout_endpoints is None:
            holdout_endpoints = np.arange(2003, t_max - min_holdout + 1, 2)

        cv_data = []
        for t_h in holdout_endpoints:
            # Subset for WLS fit
            mask_cal = (years >= satellite_start) & (years <= t_h)
            if mask_cal.sum() < 5:
                continue

            # 1. Fit WLS quadratic on [satellite_start, t_h]
            wls = fit_wls_quadratic(
                years, gmsl, gmsl_uncertainty,
                satellite_start=satellite_start,
            )
            # Re-fit with truncated data
            mask_all = years <= t_h
            wls = fit_wls_quadratic(
                years[mask_all], gmsl[mask_all],
                gmsl_uncertainty[mask_all] if gmsl_uncertainty is not None else None,
                satellite_start=satellite_start,
            )
            pred = predict_at_t_end(wls, t_h)

            # 2. Recompute T_cal_max using only data up to t_h
            T_cal_max_h = compute_T_cal_max(T_obs[years <= t_h])

            # 3. Model refit
            model_pred = model_refit_fn(t_h)

            # 4. Holdout observations
            mask_holdout = years > t_h
            t_holdout = years[mask_holdout]
            H_holdout = gmsl[mask_holdout]
            T_holdout = T_obs[mask_holdout]

            if len(t_holdout) == 0:
                continue

            # Lead times
            dt_holdout = t_holdout - t_h

            # ΔT at holdout times (relative to T_cal_max of truncated record)
            delta_T_holdout = compute_delta_T(
                T_holdout, T_cal_max_h,
                overshoot_mode=True,
            )
            # Apply ΔT power for alternative forcing forms
            if self.delta_T_power != 2:
                delta_T_holdout = delta_T_holdout ** (self.delta_T_power / 2)

            # Trend agent density at holdout observations
            # H_trend(t) = H + r·dt + 0.5·rdot·dt²
            # This is Gaussian with known mean and variance from cov_3x3
            trend_mean = (pred['H_mean']
                          + pred['r_mean'] * dt_holdout
                          + 0.5 * pred['rdot_mean'] * dt_holdout**2)
            # Variance: [1, dt, 0.5*dt²] @ cov_3x3 @ [1, dt, 0.5*dt²]^T
            cov = pred['cov_3x3']
            trend_var = np.array([
                _quadform_trend_var(cov, dti) for dti in dt_holdout
            ])
            trend_std = np.sqrt(np.maximum(trend_var, 1e-30))
            log_h_trend = stats.norm.logpdf(H_holdout, loc=trend_mean,
                                            scale=trend_std)

            # Model agent density at holdout observations
            # Interpolate model predictive to holdout times
            log_h_model = _compute_model_log_density(
                model_pred, t_holdout, H_holdout,
            )

            cv_data.append({
                't_h': t_h,
                'dt': dt_holdout,
                'delta_T': delta_T_holdout,
                'log_h_trend': log_h_trend,
                'log_h_model': log_h_model,
                'H_obs_holdout': H_holdout,
            })

        return cv_data

    def fit(self, cv_data: list, progress: bool = True) -> dict:
        """Run emcee sampling on synthesis hyperparameters.

        For weight_mode='sigmoid': ψ = (φ₀, log κ_net, log κ_t) — 3 params.
        For weight_mode='cv'/'state': ψ = (log ΔT₀) — 1 param.
        For weight_mode='heuristic': no MCMC (fixed ΔT₀).

        Returns
        -------
        dict with convergence info and acceptance fraction.
        """
        from bayesian_dols import check_convergence

        # --- Heuristic: no MCMC needed ---
        if self.weight_mode == 'heuristic':
            self._posterior_samples = np.full((1, 1), self.delta_T0)
            return {
                'acceptance_fraction': 1.0,
                'n_walkers': 0,
                'n_samples': 0,
                'n_warmup': 0,
                'convergence': {'converged': True, 'warnings': [],
                                'rhat': {}, 'ess_bulk': {}, 'ess_tail': {}},
                'weight_mode': 'heuristic',
            }

        # --- Exponential weight modes ('cv' or 'state') ---
        if self.weight_mode in ('cv', 'state'):
            tau_0_arg = (self.tau_0_samples
                         if self.weight_mode == 'state'
                         else self.tau_0)

            ndim = 1
            n_walkers = max(self.n_walkers, 2 * ndim + 2)
            if n_walkers % 2 != 0:
                n_walkers += 1

            rng = np.random.default_rng(self.seed)

            # MLE initialisation
            x0 = np.array([np.log(self.delta_T0)])
            try:
                opt = optimize.minimize(
                    lambda th: -synthesis_log_posterior_exp(th, cv_data, tau_0_arg),
                    x0, method='Nelder-Mead',
                    options={'maxiter': 2000},
                )
                if opt.success and np.isfinite(opt.fun):
                    x0 = opt.x
            except Exception:
                pass

            # Walker initialisation
            p0 = np.zeros((n_walkers, ndim))
            for i in range(n_walkers):
                p0[i] = x0 + 0.02 * rng.standard_normal(ndim)

            sampler = emcee.EnsembleSampler(
                n_walkers, ndim, synthesis_log_posterior_exp,
                args=(cv_data, tau_0_arg),
            )

            state = sampler.run_mcmc(p0, self.n_warmup, progress=progress)
            sampler.reset()
            sampler.run_mcmc(state, self.n_samples, progress=progress)
            self.emcee_sampler = sampler

            # Extract posterior in natural space: ΔT₀ = exp(log_ΔT₀)
            flat = sampler.get_chain(flat=True)  # (n_walkers * n_samples, 1)
            posterior = np.exp(flat)  # (n, 1): ΔT₀ in °C
            self._posterior_samples = posterior

            # Arviz trace
            chain_samples = sampler.get_chain(flat=False)  # (n_steps, n_walkers, 1)
            n_chains = min(4, n_walkers)
            var_dict = {
                'delta_T0': np.exp(chain_samples[:, :n_chains, 0]).T,
            }
            self.emcee_trace = az.from_dict(var_dict)

            conv = check_convergence(self.emcee_trace, quiet=(not progress))
            return {
                'acceptance_fraction': float(sampler.acceptance_fraction.mean()),
                'n_walkers': n_walkers,
                'n_samples': self.n_samples,
                'n_warmup': self.n_warmup,
                'convergence': conv,
                'weight_mode': self.weight_mode,
            }

        # --- Sigmoid weight mode (original 3-param) ---
        ndim = 3
        n_walkers = max(self.n_walkers, 2 * ndim + 2)
        if n_walkers % 2 != 0:
            n_walkers += 1

        rng = np.random.default_rng(self.seed)

        # Find MLE for walker initialization
        x0 = np.array([2.5, np.log(0.5), np.log(1e-5)])
        try:
            opt = optimize.minimize(
                lambda th: -synthesis_log_posterior(th, cv_data),
                x0, method='Nelder-Mead',
                options={'maxiter': 2000},
            )
            if opt.success and np.isfinite(opt.fun):
                x0 = opt.x
        except Exception:
            pass  # Fall back to prior median

        # Ensure φ₀ > 0
        x0[0] = max(x0[0], 0.1)

        # Initialize walkers near MLE
        p0 = np.zeros((n_walkers, ndim))
        for i in range(n_walkers):
            p0[i] = x0 + 0.01 * rng.standard_normal(ndim)
            p0[i, 0] = max(p0[i, 0], 0.01)  # φ₀ > 0

        # Run sampler
        sampler = emcee.EnsembleSampler(
            n_walkers, ndim, synthesis_log_posterior,
            args=(cv_data,),
        )

        # Burn-in
        state = sampler.run_mcmc(p0, self.n_warmup, progress=progress)
        sampler.reset()

        # Production
        sampler.run_mcmc(state, self.n_samples, progress=progress)
        self.emcee_sampler = sampler

        # Extract flat samples and transform to natural space
        flat = sampler.get_chain(flat=True)  # (n_walkers * n_samples, 3)
        posterior = np.column_stack([
            flat[:, 0],            # φ₀
            np.exp(flat[:, 1]),    # κ_net
            np.exp(flat[:, 2]),    # κ_t
        ])
        self._posterior_samples = posterior

        # Build arviz trace (preserve chain structure for diagnostics)
        chain_samples = sampler.get_chain(flat=False)  # (n_steps, n_walkers, 3)
        n_chains = min(4, n_walkers)
        var_dict = {
            'phi_0': chain_samples[:, :n_chains, 0].T,
            'kappa_net': np.exp(chain_samples[:, :n_chains, 1]).T,
            'kappa_t': np.exp(chain_samples[:, :n_chains, 2]).T,
        }
        self.emcee_trace = az.from_dict(var_dict)

        # Convergence diagnostics
        conv = check_convergence(self.emcee_trace, quiet=(not progress))

        diag = {
            'acceptance_fraction': float(sampler.acceptance_fraction.mean()),
            'n_walkers': n_walkers,
            'n_samples': self.n_samples,
            'n_warmup': self.n_warmup,
            'convergence': conv,
            'weight_mode': 'sigmoid',
        }
        return diag

    def fit_dynesty(self, cv_data: list, nlive: int = 250) -> dict:
        """Run dynesty nested sampling as cross-check.

        Returns
        -------
        dict with nested sampling results summary.
        """
        import dynesty

        # --- Heuristic: no nested sampling needed ---
        if self.weight_mode == 'heuristic':
            self._dynesty_posterior = self._posterior_samples
            return {'logz': 0.0, 'logz_err': 0.0, 'n_samples': 1}

        # --- Exponential weight modes ---
        if self.weight_mode in ('cv', 'state'):
            tau_0_arg = (self.tau_0_samples
                         if self.weight_mode == 'state'
                         else self.tau_0)

            ndim = 1

            def prior_transform_exp(u):
                # log ΔT₀ ~ N(0, 0.5²)
                log_dT0 = stats.norm.ppf(u[0], loc=0.0, scale=0.5)
                return np.array([log_dT0])

            def loglike_exp(theta):
                return synthesis_log_likelihood_exp(theta, cv_data, tau_0_arg)

            sampler = dynesty.NestedSampler(
                loglike_exp, prior_transform_exp, ndim, nlive=nlive,
                sample='rslice', bootstrap=0,
            )
            sampler.run_nested(print_progress=False, maxiter=50000)
            results = sampler.results
            self.dynesty_results = results

            from dynesty.utils import resample_equal
            weights = np.exp(results.logwt - results.logz[-1])
            samples = resample_equal(results.samples, weights)

            dynesty_posterior = np.exp(samples)  # (n, 1): ΔT₀
            self._dynesty_posterior = dynesty_posterior

            return {
                'logz': float(results.logz[-1]),
                'logz_err': float(results.logzerr[-1]),
                'n_samples': len(samples),
            }

        # --- Sigmoid mode ---
        ndim = 3

        def prior_transform(u):
            """Transform unit cube to prior."""
            phi_0 = stats.truncnorm.ppf(
                u[0], a=0, b=np.inf, loc=2.5, scale=0.5,
            )
            log_kn = stats.norm.ppf(u[1], loc=np.log(0.5), scale=0.6)
            log_kt = stats.norm.ppf(u[2], loc=np.log(1e-5), scale=0.6)
            return np.array([phi_0, log_kn, log_kt])

        def loglike(theta):
            return synthesis_log_likelihood(theta, cv_data)

        sampler = dynesty.NestedSampler(
            loglike, prior_transform, ndim, nlive=nlive,
            sample='rslice', bootstrap=0,
        )
        sampler.run_nested(print_progress=False, maxiter=50000)
        results = sampler.results
        self.dynesty_results = results

        # Extract weighted posterior samples
        from dynesty.utils import resample_equal
        weights = np.exp(results.logwt - results.logz[-1])
        samples = resample_equal(results.samples, weights)

        # Transform to natural space
        dynesty_posterior = np.column_stack([
            samples[:, 0],            # φ₀
            np.exp(samples[:, 1]),    # κ_net
            np.exp(samples[:, 2]),    # κ_t
        ])
        self._dynesty_posterior = dynesty_posterior

        return {
            'logz': float(results.logz[-1]),
            'logz_err': float(results.logzerr[-1]),
            'n_samples': len(samples),
        }

    def check_sampler_agreement(self, tolerance: float = 0.1) -> dict:
        """Compare emcee/dynesty marginal quantiles.

        Parameters
        ----------
        tolerance : float
            Max allowed difference in quantiles, normalised by IQR.

        Returns
        -------
        dict with 'agrees', 'max_diff', 'recommended_sampler',
                  'per_param' details.
        """
        if self._posterior_samples is None or self._dynesty_posterior is None:
            return {'agrees': False, 'max_diff': np.inf,
                    'recommended_sampler': 'dynesty',
                    'per_param': {}}

        if self.weight_mode == 'heuristic':
            return {'agrees': True, 'max_diff': 0.0,
                    'recommended_sampler': 'emcee',
                    'per_param': {}}

        if self.weight_mode in ('cv', 'state'):
            names = ['delta_T0']
        else:
            names = ['phi_0', 'kappa_net', 'kappa_t']

        per_param = {}
        max_diff = 0.0

        for i, name in enumerate(names):
            e_q = np.percentile(self._posterior_samples[:, i], [5, 25, 50, 75, 95])
            d_q = np.percentile(self._dynesty_posterior[:, i], [5, 25, 50, 75, 95])
            iqr = 0.5 * (e_q[3] - e_q[1] + d_q[3] - d_q[1])
            if iqr < 1e-15:
                iqr = 1.0
            diffs = np.abs(e_q - d_q) / iqr
            max_d = float(np.max(diffs))
            max_diff = max(max_diff, max_d)
            per_param[name] = {
                'emcee_quantiles': e_q,
                'dynesty_quantiles': d_q,
                'max_iqr_diff': max_d,
            }

        agrees = max_diff < tolerance
        return {
            'agrees': agrees,
            'max_diff': max_diff,
            'recommended_sampler': 'emcee' if agrees else 'dynesty',
            'per_param': per_param,
        }

    def get_posterior_samples(self) -> np.ndarray:
        """Return posterior samples, shape (n_total, 3): [φ₀, κ_net, κ_t]."""
        if self._posterior_samples is None:
            raise RuntimeError("Must call fit() first")
        return self._posterior_samples

    def _exp_weight_array(
        self, dt: np.ndarray, delta_T: np.ndarray,
        dT0_samples: np.ndarray,
    ) -> np.ndarray:
        """Compute exponential weight array, shape (n_samples, n_proj).

        For 'state' mode, τ₀ varies per sample; otherwise scalar.
        """
        n_samp = len(dT0_samples)
        dT0 = dT0_samples[:, 0:1]  # (n_samp, 1)

        if self.weight_mode == 'state' and self.tau_0_samples is not None:
            # Resample tau_0_samples to match dT0 sample count
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(self.tau_0_samples), n_samp, replace=True)
            tau_0_arr = self.tau_0_samples[idx][:, np.newaxis]  # (n_samp, 1)
        else:
            tau_0_arr = self.tau_0  # scalar

        tau_eff = tau_0_arr / (1.0 + (delta_T[np.newaxis, :] / dT0)**2)
        tau_eff = np.maximum(tau_eff, 1e-10)
        return np.exp(-dt[np.newaxis, :] / tau_eff)

    def posterior_mean_weight(
        self, dt: np.ndarray, delta_T: np.ndarray,
    ) -> np.ndarray:
        """w̄(t) = mean over posterior samples. Vectorised.

        Parameters
        ----------
        dt : array, shape (n_proj,)
            Lead times t - t_end.
        delta_T : array, shape (n_proj,)
            Temperature departure at each projection time.

        Returns
        -------
        w_bar : array, shape (n_proj,)
        """
        samples = self.get_posterior_samples()

        if self.weight_mode in ('heuristic', 'cv', 'state'):
            w = self._exp_weight_array(dt, delta_T, samples)
            return w.mean(axis=0)

        # Sigmoid mode
        phi = (samples[:, 0:1]
               - samples[:, 1:2] * (delta_T**2 * dt**2)[np.newaxis, :]
               - samples[:, 2:3] * (dt**4)[np.newaxis, :])
        w = special.expit(phi)   # (n_psi, n_proj)
        return w.mean(axis=0)    # (n_proj,)

    def posterior_weight_ensemble(
        self, dt: np.ndarray, delta_T: np.ndarray,
        n_samples: Optional[int] = None, seed: Optional[int] = None,
    ) -> np.ndarray:
        """Per-sample weight trajectories, shape (n_samples, n_proj).

        If n_samples differs from the posterior size, resample with replacement.
        """
        samples = self.get_posterior_samples()
        if n_samples is not None and n_samples != len(samples):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(samples), n_samples, replace=True)
            samples = samples[idx]

        if self.weight_mode in ('heuristic', 'cv', 'state'):
            return self._exp_weight_array(dt, delta_T, samples)

        # Sigmoid mode
        phi = (samples[:, 0:1]
               - samples[:, 1:2] * (delta_T**2 * dt**2)[np.newaxis, :]
               - samples[:, 2:3] * (dt**4)[np.newaxis, :])
        return special.expit(phi)

    def convergence_diagnostics(self) -> dict:
        """Delegate to check_convergence() from bayesian_dols.py."""
        from bayesian_dols import check_convergence

        if self.emcee_trace is None:
            raise RuntimeError("Must call fit() first")
        conv = check_convergence(self.emcee_trace, quiet=True)
        if self.emcee_sampler is not None:
            conv['acceptance_fraction'] = float(
                self.emcee_sampler.acceptance_fraction.mean()
            )
        return conv

    def holdout_influence_diagnostics(
        self, cv_data: list, ess_threshold: int = 100,
    ) -> dict:
        """Importance-weight pre-screen for leave-one-holdout-out.

        For each holdout fold j, compute importance weights from the
        full posterior by removing fold j's contribution to the likelihood.
        If the effective sample size (ESS) of the importance weights falls
        below ess_threshold, flag that fold for full MCMC refit.

        Returns
        -------
        dict with 'ess_per_fold', 'flagged_for_refit'
        """
        if self._posterior_samples is None:
            raise RuntimeError("Must call fit() first")

        if self.weight_mode == 'heuristic':
            # No posterior to diagnose
            return {
                'ess_per_fold': {fold['t_h']: np.inf for fold in cv_data},
                'flagged_for_refit': [],
            }

        samples = self.get_posterior_samples()
        n_post = len(samples)

        # Convert back to sampling space and choose likelihood function
        if self.weight_mode in ('cv', 'state'):
            theta_samples = np.log(samples)  # (n, 1): log(ΔT₀)
            tau_0_arg = (self.tau_0_samples
                         if self.weight_mode == 'state'
                         else self.tau_0)
            ll_fn = lambda th, folds: synthesis_log_likelihood_exp(
                th, folds, tau_0_arg)
        else:
            theta_samples = np.column_stack([
                samples[:, 0],
                np.log(samples[:, 1]),
                np.log(samples[:, 2]),
            ])
            ll_fn = lambda th, folds: synthesis_log_likelihood(th, folds)

        # Full log-likelihood per fold
        ll_per_fold = np.zeros((n_post, len(cv_data)))
        for j, fold in enumerate(cv_data):
            for s in range(n_post):
                ll_per_fold[s, j] = ll_fn(theta_samples[s], [fold])

        ess_per_fold = {}
        flagged = []
        for j, fold in enumerate(cv_data):
            # Importance weight: exp(-ll_j) (remove fold j)
            log_w = -ll_per_fold[:, j]
            log_w -= np.max(log_w)  # numerical stability
            w = np.exp(log_w)
            w /= w.sum()
            ess = 1.0 / np.sum(w**2)
            t_h = fold['t_h']
            ess_per_fold[t_h] = float(ess)
            if ess < ess_threshold:
                flagged.append(t_h)

        return {
            'ess_per_fold': ess_per_fold,
            'flagged_for_refit': flagged,
        }


# =============================================================================
# SECTION 5: COMBINATION AND WAIS ADD-BACK
# =============================================================================

def align_ensemble_sizes(
    *ensembles: np.ndarray,
    target_size: int = N_ENSEMBLE,
    seed: Optional[int] = None,
) -> tuple:
    """Resample each ensemble to target_size with replacement.

    Parameters
    ----------
    *ensembles : arrays of shape (n_i, n_proj)
    target_size : int
    seed : int or None

    Returns
    -------
    tuple of resampled arrays, each (target_size, n_proj)
    """
    rng = np.random.default_rng(seed)
    result = []
    for ens in ensembles:
        n = ens.shape[0]
        if n == target_size:
            result.append(ens)
        else:
            idx = rng.choice(n, target_size, replace=True)
            result.append(ens[idx])
    return tuple(result)


def compute_mixture_quantiles(
    H_trend_ens: np.ndarray,
    H_model_ens: np.ndarray,
    w_bar: np.ndarray,
    quantiles: Optional[np.ndarray] = None,
) -> dict:
    """Weighted quantiles of BPS mixture via weighted-pool method.

    At each time t, pool trend/model samples with weights
    w̄(t)/n_trend and (1-w̄(t))/n_model.

    Parameters
    ----------
    H_trend_ens : (n_trend, n_proj)
    H_model_ens : (n_model, n_proj)
    w_bar : (n_proj,) posterior mean weight
    quantiles : array of quantile levels (default [0.05, 0.17, 0.5, 0.83, 0.95])

    Returns
    -------
    dict with 'quantiles' (levels), 'values' (n_quantiles, n_proj)
    """
    if quantiles is None:
        quantiles = np.array([0.05, 0.17, 0.5, 0.83, 0.95])

    n_trend = H_trend_ens.shape[0]
    n_model = H_model_ens.shape[0]
    n_proj = H_trend_ens.shape[1]
    n_q = len(quantiles)
    result = np.zeros((n_q, n_proj))

    for j in range(n_proj):
        w = w_bar[j]
        # Pool samples
        pooled = np.concatenate([H_trend_ens[:, j], H_model_ens[:, j]])
        weights = np.concatenate([
            np.full(n_trend, w / n_trend),
            np.full(n_model, (1 - w) / n_model),
        ])

        # Sort by value
        order = np.argsort(pooled)
        pooled_sorted = pooled[order]
        weights_sorted = weights[order]

        # Cumulative weights
        cum_w = np.cumsum(weights_sorted)
        cum_w /= cum_w[-1]  # Normalise

        # Interpolate quantiles
        for qi, q in enumerate(quantiles):
            result[qi, j] = np.interp(q, cum_w, pooled_sorted)

    return {
        'quantiles': quantiles,
        'values': result,
    }


def generate_stochastic_transition_ensemble(
    H_trend_ens: np.ndarray,
    H_model_ens: np.ndarray,
    w_ensemble: np.ndarray,
    epsilon: float = 0.05,
    seed: Optional[int] = None,
) -> dict:
    """Temporally coherent trajectories via stochastic transition.

    For each sample s:
    1. Draw u^(s) ~ U(0,1)
    2. ω^(s)(t) = σ((w^(s)(t) - u^(s)) / ε)
    3. H^(s)(t) = ω·H_trend + (1-ω)·H_model

    Prerequisites: all inputs must have the same first dimension (n_samples).
    Use align_ensemble_sizes() first.

    Parameters
    ----------
    H_trend_ens : (n_samples, n_proj)
    H_model_ens : (n_samples, n_proj)
    w_ensemble : (n_samples, n_proj) — per-sample weight trajectories
    epsilon : float — smoothing parameter
    seed : int or None

    Returns
    -------
    dict with 'H_combined' (n_samples, n_proj), 'u_draws' (n_samples,),
              'transition_times_idx' (n_samples,) — index where ω crosses 0.5
    """
    n_samples = H_trend_ens.shape[0]
    n_proj = H_trend_ens.shape[1]
    rng = np.random.default_rng(seed)

    u = rng.uniform(size=n_samples)  # (n_samples,)

    # ω^(s)(t) = σ((w^(s)(t) - u^(s)) / ε)
    omega = special.expit(
        (w_ensemble - u[:, np.newaxis]) / epsilon
    )  # (n_samples, n_proj)

    H_combined = omega * H_trend_ens + (1 - omega) * H_model_ens

    # Find approximate transition time for each sample
    # (where ω crosses 0.5, i.e., w crosses u)
    transition_idx = np.full(n_samples, n_proj - 1, dtype=int)
    for s in range(n_samples):
        crossings = np.where(w_ensemble[s] < u[s])[0]
        if len(crossings) > 0:
            transition_idx[s] = crossings[0]

    return {
        'H_combined': H_combined,
        'u_draws': u,
        'transition_times_idx': transition_idx,
    }


def add_wais_component(
    H_comb_non_wais: np.ndarray,
    H_wais: np.ndarray,
) -> np.ndarray:
    """H_total = H_non_wais + H_wais."""
    return H_comb_non_wais + H_wais


def rate_smoothness_diagnostic(
    H_combined: np.ndarray,
    t_proj: np.ndarray,
) -> dict:
    """Finite-difference rates and discontinuity check.

    Returns
    -------
    dict with 'rates' (n_samples, n_proj-1), 'rate_jumps_median',
              'max_jump_fraction' (max jump / median rate spread)
    """
    dt = np.diff(t_proj)
    rates = np.diff(H_combined, axis=1) / dt[np.newaxis, :]

    # Rate jumps (second differences)
    rate_jumps = np.abs(np.diff(rates, axis=1))
    median_jump = float(np.median(rate_jumps))

    # Compare to ensemble rate spread
    rate_spread = np.std(rates, axis=0)
    median_spread = float(np.median(rate_spread))

    max_jump_frac = median_jump / median_spread if median_spread > 0 else np.inf

    return {
        'rates': rates,
        'rate_jumps_median': median_jump,
        'rate_spread_median': median_spread,
        'max_jump_fraction': max_jump_frac,
    }


# =============================================================================
# SECTION 6: CROSSOVER DIAGNOSTICS
# =============================================================================

def compute_crossover_diagnostics(
    bps: BPSSynthesis,
    t_proj: np.ndarray,
    t_end: float,
    delta_T_by_ssp: dict,
    T_rate_by_ssp: Optional[dict] = None,
) -> dict:
    """Compute crossover time posterior from BPS.

    Crossover is defined as w(t) = 0.5:
    - Sigmoid mode: φ(t) = 0  (since σ(0) = 0.5)
    - Exponential mode: dt/τ_eff = ln(2)

    Parameters
    ----------
    bps : BPSSynthesis
        Fitted BPS object.
    t_proj : array
        Projection times.
    t_end : float
        End of calibration window.
    delta_T_by_ssp : dict
        {ssp: delta_T array} for each SSP.
    T_rate_by_ssp : dict or None
        {ssp: dT/dt at t_end} for analytic approximation. None entries
        skip the analytic formula (overshoot SSPs).

    Returns
    -------
    dict with per-SSP crossover diagnostics.
    """
    samples = bps.get_posterior_samples()
    dt = t_proj - t_end
    results = {}
    use_exp = bps.weight_mode in ('heuristic', 'cv', 'state')

    for ssp, delta_T in delta_T_by_ssp.items():
        n_samples = len(samples)
        crossover_years = np.full(n_samples, np.nan)

        if use_exp:
            # Exponential mode: find where w(t) = 0.5
            # w(t) = exp(-dt/τ_eff), crossover at dt = τ_eff · ln(2)
            rng = np.random.default_rng(bps.seed)
            if bps.weight_mode == 'state' and bps.tau_0_samples is not None:
                tau0_draw = rng.choice(
                    bps.tau_0_samples, n_samples, replace=True)
            else:
                tau0_draw = np.full(n_samples, bps.tau_0)

            for s in range(n_samples):
                dT0_s = samples[s, 0]
                tau0_s = tau0_draw[s]

                # w(t) at each projection time
                tau_eff = tau0_s / (1.0 + (delta_T / dT0_s)**2)
                tau_eff = np.maximum(tau_eff, 1e-10)
                w_vals = np.exp(-dt / tau_eff)

                # Find where w crosses 0.5
                crossings = np.where(np.diff(np.sign(w_vals - 0.5)))[0]
                if len(crossings) > 0:
                    j = crossings[0]
                    dw = w_vals[j] - w_vals[j + 1]
                    if abs(dw) > 1e-15:
                        frac = (w_vals[j] - 0.5) / dw
                        crossover_years[s] = (
                            t_proj[j] + frac * (t_proj[j + 1] - t_proj[j]))
                    else:
                        crossover_years[s] = t_proj[j]
                elif w_vals[-1] > 0.5:
                    crossover_years[s] = np.inf
        else:
            # Sigmoid mode: find where φ(t) = 0
            for s in range(n_samples):
                phi_0 = samples[s, 0]
                kn = samples[s, 1]
                kt = samples[s, 2]

                phi_vals = phi_0 - kn * delta_T**2 * dt**2 - kt * dt**4

                crossings = np.where(np.diff(np.sign(phi_vals)))[0]
                if len(crossings) > 0:
                    j = crossings[0]
                    if phi_vals[j + 1] != phi_vals[j]:
                        frac = phi_vals[j] / (phi_vals[j] - phi_vals[j + 1])
                        crossover_years[s] = (
                            t_proj[j] + frac * (t_proj[j + 1] - t_proj[j]))
                    else:
                        crossover_years[s] = t_proj[j]
                elif phi_vals[-1] > 0:
                    crossover_years[s] = np.inf

        valid = np.isfinite(crossover_years)
        valid_years = crossover_years[valid]

        # Weight at 2050 and 2100
        w_at_2050 = bps.posterior_mean_weight(
            np.array([2050 - t_end]), np.interp([2050], t_proj, delta_T),
        )
        w_at_2100 = bps.posterior_mean_weight(
            np.array([2100 - t_end]), np.interp([2100], t_proj, delta_T),
        )

        # Analytic approximation
        analytic_dt_star = None
        if use_exp:
            # For exponential: crossover at dt ≈ τ₀·ln(2) when ΔT≈0
            med_dT0 = np.median(samples[:, 0])
            med_tau0 = bps.tau_0
            analytic_dt_star = float(med_tau0 * np.log(2))
        elif T_rate_by_ssp is not None and T_rate_by_ssp.get(ssp) is not None:
            T_rate = T_rate_by_ssp[ssp]
            med_phi0 = np.median(samples[:, 0])
            med_kn = np.median(samples[:, 1])
            if med_kn > 0 and T_rate > 0:
                dt_star = (med_phi0 / (med_kn * T_rate**2))**0.25
                analytic_dt_star = float(dt_star)

        # Determine driver
        if use_exp:
            med_dT = np.median(delta_T[-min(10, len(delta_T)):])
            driver = ('forcing_departure' if med_dT > 0.1
                      else 'exponential_decay')
        else:
            med_kn = np.median(samples[:, 1])
            med_kt = np.median(samples[:, 2])
            med_dT = np.median(delta_T[-min(10, len(delta_T)):])
            if med_dT > 0.1 and med_kn * med_dT**2 > med_kt * (30**2):
                driver = 'forcing_departure'
            else:
                driver = 'accel_persist'

        results[ssp] = {
            'crossover_years': crossover_years,
            'crossover_median': float(np.nanmedian(valid_years)) if len(valid_years) > 0 else np.nan,
            'crossover_5': float(np.nanpercentile(valid_years, 5)) if len(valid_years) > 0 else np.nan,
            'crossover_95': float(np.nanpercentile(valid_years, 95)) if len(valid_years) > 0 else np.nan,
            'w_bar_at_2050': float(w_at_2050[0]),
            'w_bar_at_2100': float(w_at_2100[0]),
            'driver': driver,
            'analytic_dt_star': analytic_dt_star,
            'fraction_no_crossover': float(1 - valid.mean()),
        }

    return results


# =============================================================================
# SECTION 7: FORCING-FORM SENSITIVITY
# =============================================================================

def run_forcing_form_sensitivity(
    years: np.ndarray,
    gmsl: np.ndarray,
    gmsl_uncertainty: Optional[np.ndarray],
    T_obs: np.ndarray,
    model_refit_fn,
    delta_T_by_ssp: dict,
    t_proj: np.ndarray,
    t_end: float,
    bps_kwargs: Optional[dict] = None,
) -> dict:
    """Run full BPS pipeline under default (ΔT²) and optimistic (ΔT⁴) forms.

    Requires two separate calibrations because κ_net has different units.

    Parameters
    ----------
    years, gmsl, gmsl_uncertainty, T_obs, model_refit_fn :
        Same as BPSSynthesis.build_cv_data().
    delta_T_by_ssp : dict
        {ssp: delta_T array} for crossover diagnostics.
    t_proj : array
        Projection times.
    t_end : float
    bps_kwargs : dict or None
        Kwargs passed to BPSSynthesis constructor.

    Returns
    -------
    dict with 'default' and 'optimistic' sub-dicts, each containing
    BPS diagnostics and crossover results.
    """
    kwargs = bps_kwargs or {}
    results = {}

    for form_name, dT_power in [('default', 2), ('optimistic', 4)]:
        bps = BPSSynthesis(delta_T_power=dT_power, **kwargs)
        cv_data = bps.build_cv_data(
            years, gmsl, gmsl_uncertainty, T_obs, model_refit_fn,
        )
        bps.fit(cv_data, progress=False)
        bps.fit_dynesty(cv_data)
        agreement = bps.check_sampler_agreement()

        dt = t_proj - t_end
        crossover = compute_crossover_diagnostics(
            bps, t_proj, t_end, delta_T_by_ssp,
        )

        results[form_name] = {
            'bps': bps,
            'cv_data': cv_data,
            'sampler_agreement': agreement,
            'crossover': crossover,
        }

    return results


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _quadform_trend_var(cov_3x3: np.ndarray, dt: float) -> float:
    """Variance of H_trend at lead time dt from (H, r, rdot) covariance.

    H_trend = H + r·dt + 0.5·rdot·dt²
    v = [1, dt, 0.5·dt²]
    Var = v^T @ cov @ v
    """
    v = np.array([1.0, dt, 0.5 * dt**2])
    return float(v @ cov_3x3 @ v)


def _compute_model_log_density(
    model_pred: dict,
    t_holdout: np.ndarray,
    H_holdout: np.ndarray,
) -> np.ndarray:
    """Compute Gaussian log-density of model agent at holdout observations.

    Uses the mean and std from the model refit to approximate the model
    predictive as Gaussian at each holdout time.
    """
    model_time = model_pred['time']
    model_mean = model_pred['H_predictive_mean']
    model_std = model_pred['H_predictive_std']

    if len(model_mean) == 0:
        return np.full(len(t_holdout), -50.0)

    # Interpolate model mean and std to holdout times
    mean_interp = np.interp(t_holdout, model_time, model_mean)
    std_interp = np.interp(t_holdout, model_time, model_std)
    std_interp = np.maximum(std_interp, 1e-6)  # floor

    return stats.norm.logpdf(H_holdout, loc=mean_interp, scale=std_interp)
