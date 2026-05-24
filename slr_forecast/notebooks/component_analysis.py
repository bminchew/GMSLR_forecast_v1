"""
Component decomposition analysis functions.

Extracted from component_decomposition.ipynb to keep notebooks free of
function definitions.  All functions operate in SLR convention
(positive = sea level rise) and SI-derived internal units (meters, °C, yr).
"""

import numpy as np
import pandas as pd
from bayesian_models import (
    build_level_design_vectors, fit_bayesian_level,
    calibrate_exponential_prior,
)

# ---------------------------------------------------------------------------
# Constants (re-exported from config when available; local fallbacks here
# so the module works standalone from the notebooks/ directory)
# ---------------------------------------------------------------------------
try:
    from slr_forecast.config import BASELINE_YEAR, N_SAMPLES
    from slr_forecast import M_TO_MM
except ImportError:
    BASELINE_YEAR = 2000.0
    M_TO_MM = 1000.0
    N_SAMPLES = 2000


# =========================================================================
# Data preparation helpers
# =========================================================================

def annualize_imbie(df, baseline_year=BASELINE_YEAR):
    """Resample monthly IMBIE to annual, rebase to *baseline_year*.

    Parameters
    ----------
    df : DataFrame
        Monthly IMBIE data with columns ``decimal_year``,
        ``cumulative_mass_balance``, ``cumulative_mass_balance_sigma``.
    baseline_year : float
        Year to rebase cumulative values to zero.

    Returns
    -------
    years : ndarray
        Annual decimal years.
    H_rebase : ndarray
        Cumulative SLR (m), rebased.
    sigma : ndarray
        Uncertainty (m, positive).
    """
    t = df['decimal_year'].values
    year_int = np.floor(t).astype(int)
    unique_years = np.unique(year_int)

    years = np.zeros(len(unique_years))
    H = np.zeros(len(unique_years))
    sigma = np.zeros(len(unique_years))

    for i, yr in enumerate(unique_years):
        mask = year_int == yr
        years[i] = yr + 0.5
        H[i] = df['cumulative_mass_balance'].values[mask][-1]
        sigma[i] = np.abs(df['cumulative_mass_balance_sigma'].values[mask][-1])

    # Data is already in meters from the reader (read_imbie_* converts
    # mm → m).  No further unit conversion needed.

    # Rebase
    bl_idx = np.argmin(np.abs(years - baseline_year))
    H_rebase = H - H[bl_idx]

    return years, H_rebase, sigma


def apply_sigma_taper(sigma, years, t_ref, f_max):
    """Inflate sigma for data before *t_ref* with a linear taper.

    σ_inflated(t) = σ(t) × [1 + (f_max − 1) × max(0, (t_ref − t)/(t_ref − t_start))]

    At t >= t_ref the original σ is unchanged.  At t_start: σ × f_max.

    Parameters
    ----------
    sigma : ndarray
        Observation uncertainties.
    years : ndarray
        Observation times (decimal years).
    t_ref : float
        Reference year (no inflation applied at or after this year).
    f_max : float
        Maximum inflation factor at the start of the record.

    Returns
    -------
    ndarray
        Inflated sigma array.
    """
    if f_max <= 1.0:
        return sigma.copy()
    t_start = years[0]
    if t_ref <= t_start:
        return sigma.copy()
    factor = np.ones_like(sigma, dtype=float)
    pre = years < t_ref
    factor[pre] = 1.0 + (f_max - 1.0) * (t_ref - years[pre]) / (t_ref - t_start)
    return sigma * factor


# =========================================================================
# Fitting helpers
# =========================================================================

def restrict_and_fit(name, H_full, sigma_full, years_full, obs_window,
                     temp_monthly, time_monthly, prior_kw,
                     baseline_year=BASELINE_YEAR, seed=200):
    """Restrict data to an observation window, rebuild design vectors, and fit.

    Parameters
    ----------
    name : str
        Component name (for logging).
    H_full, sigma_full, years_full : ndarray
        Full-record component data (meters).
    obs_window : tuple of (int, int)
        (start_year, end_year) inclusive.
    temp_monthly, time_monthly : ndarray
        Monthly temperature and corresponding decimal-year arrays.
    prior_kw : dict
        Keyword arguments for ``fit_bayesian_level`` priors.
    baseline_year : float
        Rebase year within the restricted window.
    seed : int
        Random seed.

    Returns
    -------
    result : BayesianLevelResult
    years_r : ndarray
    H_r : ndarray
    sigma_r : ndarray
    design_r : dict
    """
    yr_start, yr_end = obs_window
    mask = (years_full >= yr_start) & (years_full <= yr_end)
    years_r = years_full[mask]
    H_r = H_full[mask].copy()
    sigma_r = sigma_full[mask].copy()

    bl_idx_r = np.argmin(np.abs(years_r - baseline_year))
    H_r = H_r - H_r[bl_idx_r]

    design_r = build_level_design_vectors(
        temperature_monthly=temp_monthly,
        time_monthly=time_monthly,
        obs_times=years_r,
    )

    result = fit_bayesian_level(
        H_obs=H_r,
        sigma_obs=sigma_r,
        I2_obs=design_r['I2_obs'],
        I1_obs=design_r['I1_obs'],
        I0_obs=design_r['I0_obs'],
        n_samples=4000,
        n_walkers=32,
        n_burnin=2000,
        thin=2,
        seed=seed,
        **prior_kw,
    )
    return result, years_r, H_r, sigma_r, design_r


# =========================================================================
# Model evaluation and diagnostics
# =========================================================================

def model_ensemble_draws(result, I2, I1, I0, n_draw=500):
    """Draw *n_draw* random model predictions from a BayesianLevelResult posterior.

    Parameters
    ----------
    result : BayesianLevelResult
        Fitted result with ``posterior_samples`` (n, 3) and ``H0_posterior`` (n,).
    I2, I1, I0 : ndarray
        Design vectors at evaluation points.
    n_draw : int
        Number of draws.

    Returns
    -------
    H_ens : ndarray, shape (n_draw, len(I2))
        Ensemble of model predictions.
    """
    n_avail = len(result.posterior_samples)
    n_draw = min(n_draw, n_avail)
    idx = np.random.choice(n_avail, n_draw, replace=False)
    H_ens = np.zeros((n_draw, len(I2)))
    for i, j in enumerate(idx):
        a = result.posterior_samples[j, 0]
        b = result.posterior_samples[j, 1]
        c = result.posterior_samples[j, 2]
        H0 = result.H0_posterior[j]
        H_ens[i] = a * I2 + b * I1 + c * I0 + H0
    return H_ens


def eval_model_median(result, design, order=1):
    """Evaluate model at posterior median.

    Parameters
    ----------
    result : BayesianLevelResult
    design : dict with keys ``I2_obs``, ``I1_obs``, ``I0_obs``
    order : int
        If 1, force a=0 (linear model).

    Returns
    -------
    ndarray
        Model prediction at posterior-median parameters.
    """
    samples = result.posterior_samples
    a_med = np.median(samples[:, 0]) if order >= 2 else 0.0
    b_med = np.median(samples[:, 1])
    c_med = np.median(samples[:, 2])
    H0_med = np.median(result.H0_posterior)
    return (a_med * design['I2_obs'] + b_med * design['I1_obs']
            + c_med * design['I0_obs'] + H0_med)


def compute_component_rates(years, H, window=3):
    """Compute rate via centred differences with given half-window.

    Parameters
    ----------
    years : ndarray
    H : ndarray
        Cumulative values (meters).
    window : int
        Half-window for centred differences.

    Returns
    -------
    ndarray
        Rate array (m/yr), NaN at edges.
    """
    rates = np.full_like(H, np.nan)
    for i in range(window, len(H) - window):
        dt = years[i + window] - years[i - window]
        rates[i] = (H[i + window] - H[i - window]) / dt
    return rates


# =========================================================================
# Temperature scenario construction
# =========================================================================

def build_full_temperature_scenario(t_hist, t_ssp, offset):
    """Combine historical + SSP temperature, rebaseline to 1995-2005.

    Parameters
    ----------
    t_hist : DataFrame
        Historical temperature with ``decimal_year`` and ``temperature``.
    t_ssp : DataFrame
        SSP temperature projection with same columns.
    offset : float
        Baseline offset to subtract (°C).

    Returns
    -------
    DataFrame
        Combined temperature scenario.
    """
    hist_part = t_hist[t_hist['decimal_year'] < 2015].copy()
    combined = pd.concat([hist_part, t_ssp]).sort_index()
    combined = combined[~combined.index.duplicated(keep='last')]
    combined['temperature'] = combined['temperature'] - offset
    if 'temperature_lower' in combined.columns:
        combined['temperature_lower'] -= offset
        combined['temperature_upper'] -= offset
    return combined


# =========================================================================
# Variance decomposition
# =========================================================================

def compute_variance_fractions(ssp, components, proj_years, comp_proj,
                               n_samples=N_SAMPLES, normalise=True):
    """Compute variance fraction time series for given components.

    Parameters
    ----------
    ssp : str
        SSP name.
    components : list of str
        Component names to include.
    proj_years : ndarray
        Projection year array.
    comp_proj : dict
        Nested dict ``{ssp: {component: {'samples': ndarray}}}``
    n_samples : int
        Number of MC samples (used for array sizing).
    normalise : bool
        If True, rescale fractions to sum to 1 at each time step.

    Returns
    -------
    fracs : dict of ndarray
        Variance fractions per component.
    raw_sum : ndarray
        Raw sum of individual Var_i / Var_total (>1 if positive covariance).
    """
    total_samples = np.zeros((n_samples, len(proj_years)))
    for cname in components:
        if cname in comp_proj[ssp]:
            total_samples += comp_proj[ssp][cname]['samples']
    total_var = np.var(total_samples, axis=0)
    total_var_safe = np.maximum(total_var, 1e-30)

    raw_fracs = {}
    for cname in components:
        if cname in comp_proj[ssp]:
            cvar = np.var(comp_proj[ssp][cname]['samples'], axis=0)
            raw_fracs[cname] = cvar / total_var_safe
        else:
            raw_fracs[cname] = np.zeros(len(proj_years))

    raw_sum = sum(raw_fracs[c] for c in components)

    if normalise:
        raw_sum_safe = np.maximum(raw_sum, 1e-30)
        fracs = {c: raw_fracs[c] / raw_sum_safe for c in components}
    else:
        fracs = raw_fracs

    return fracs, raw_sum


# =========================================================================
# Surface-to-ocean temperature transfer function
# =========================================================================

def fit_ocean_transfer_function(T_surface_monthly, time_surface,
                                T_ocean_monthly, time_ocean,
                                lag_years=0, annual=True):
    """Calibrate a linear transfer function: T_ocean = α·T_surface + β.

    Aligns surface and ocean temperature on common annual (or monthly)
    time points, fits OLS with uncertainty, and returns parameters for
    projecting ocean temperature under SSP surface-T scenarios.

    Parameters
    ----------
    T_surface_monthly : ndarray
        Monthly surface temperature anomaly (°C), e.g. Greenland T or GMST.
    time_surface : ndarray
        Decimal-year times for surface temperature.
    T_ocean_monthly : ndarray
        Monthly subsurface ocean temperature anomaly (°C), e.g. EN4/Argo
        200–500 m around Greenland.
    time_ocean : ndarray
        Decimal-year times for ocean temperature.
    lag_years : float
        If > 0, ocean T is assumed to lag surface T by this many years.
        The surface T is shifted forward by ``lag_years`` before alignment.
    annual : bool
        If True (default), annualise both series before fitting to reduce
        noise.  If False, fit on monthly data.

    Returns
    -------
    dict with keys:
        'alpha'       : float — slope (°C_ocean / °C_surface)
        'beta'        : float — intercept (°C)
        'alpha_se'    : float — standard error on alpha
        'beta_se'     : float — standard error on beta
        'r2'          : float — R² of fit
        'r'           : float — Pearson correlation
        'n'           : int   — number of data points used
        'years'       : ndarray — common years used
        'T_surf_used' : ndarray — surface T values used
        'T_ocean_used': ndarray — ocean T values used
        'residual_std': float — std of residuals (°C)
        'lag_years'   : float — lag applied
    """
    import statsmodels.api as sm

    if annual:
        # Annualise surface T
        yr_surf = np.floor(time_surface).astype(int)
        unique_surf = np.unique(yr_surf)
        T_surf_ann = np.array([T_surface_monthly[yr_surf == y].mean()
                               for y in unique_surf])

        # Annualise ocean T
        yr_ocean = np.floor(time_ocean).astype(int)
        unique_ocean = np.unique(yr_ocean)
        T_ocean_ann = np.array([T_ocean_monthly[yr_ocean == y].mean()
                                for y in unique_ocean])

        t_surf = unique_surf.astype(float) + 0.5
        t_ocean = unique_ocean.astype(float) + 0.5
    else:
        T_surf_ann = T_surface_monthly
        t_surf = time_surface
        T_ocean_ann = T_ocean_monthly
        t_ocean = time_ocean

    # Apply lag: shift surface time forward
    t_surf_shifted = t_surf + lag_years

    # Find common years (within 0.6 yr tolerance for annual)
    tol = 0.6 if annual else 0.05
    common_idx_surf = []
    common_idx_ocean = []
    for i, ts in enumerate(t_surf_shifted):
        diffs = np.abs(t_ocean - ts)
        j = np.argmin(diffs)
        if diffs[j] < tol:
            common_idx_surf.append(i)
            common_idx_ocean.append(j)

    if len(common_idx_surf) < 5:
        raise ValueError(f'Only {len(common_idx_surf)} common points found '
                         f'(need >= 5). Check time ranges and lag.')

    T_s = T_surf_ann[common_idx_surf]
    T_o = T_ocean_ann[common_idx_ocean]
    years = t_ocean[common_idx_ocean]

    # Drop NaN
    valid = np.isfinite(T_s) & np.isfinite(T_o)
    T_s = T_s[valid]
    T_o = T_o[valid]
    years = years[valid]

    # OLS: T_ocean = alpha * T_surface + beta
    X = sm.add_constant(T_s)
    model = sm.OLS(T_o, X).fit()

    beta = model.params[0]
    alpha = model.params[1]
    beta_se = model.bse[0]
    alpha_se = model.bse[1]

    return {
        'alpha': float(alpha),
        'beta': float(beta),
        'alpha_se': float(alpha_se),
        'beta_se': float(beta_se),
        'r2': float(model.rsquared),
        'r': float(np.corrcoef(T_s, T_o)[0, 1]),
        'n': int(len(T_s)),
        'years': years,
        'T_surf_used': T_s,
        'T_ocean_used': T_o,
        'residual_std': float(np.std(model.resid)),
        'lag_years': float(lag_years),
    }


# =========================================================================
# Discharge delay model
# =========================================================================

def fit_discharge_delay_model(
    dyn_years, H_dyn, sigma_dyn,
    T_ocean_ann, T_ocean_years,
    delta_candidates,
    rate_window_yrs=10,
    rate_constraint_weight=1,
    n_samples=2000,
    seed=None,
):
    """Fit the Greenland discharge delay model by demeaned WLS.

    The model is:

        H_dyn(t) = gamma * integral(T_ocean(t' - delta), t0..t) + r0 * t + const

    fitted in demeaned form to eliminate the intercept.  The time delay
    ``delta`` is selected from ``delta_candidates`` by BIC, and the
    posterior is a BIC-weighted mixture of Gaussian draws from each
    candidate's WLS covariance.

    Parameters
    ----------
    dyn_years : ndarray
        Observation times (decimal year) for cumulative discharge.
    H_dyn : ndarray
        Cumulative discharge observations (meters SLE, SLR convention).
    sigma_dyn : ndarray
        1-sigma uncertainties on H_dyn (meters).
    T_ocean_ann : ndarray
        Annual subsurface ocean temperature anomaly (deg C).
    T_ocean_years : ndarray
        Mid-year times for T_ocean_ann.
    delta_candidates : array-like of float
        Candidate time delays (years) to evaluate.
    rate_window_yrs : float
        Number of years at the end of the record used to estimate the
        observed discharge rate (for the sanity-check rate constraint).
    rate_constraint_weight : float
        Weight multiplier for the rate constraint equation (1 = natural
        measurement uncertainty; >1 inflates the rate weight).
    n_samples : int
        Number of posterior MC samples.
    seed : int or None
        Random seed for posterior draws.

    Returns
    -------
    types.SimpleNamespace with attributes:
        gamma_posterior, r0_posterior, delta_posterior : ndarray (n_samples,)
        delta_best : float — BIC-selected delay
        gamma_best, r0_best : float — WLS point estimates at delta_best
        r2_dyn : float — rate-space R² at delta_best
        bic_best : float
        fit_results : dict — per-delta fit details
        H_mean_cal, int_T_mean_cal, t_mean_cal : float — demeaning constants
        r_obs_dyn, sigma_r_obs, t_end : float — end-of-record rate
        xcorr_lags, xcorr_r, peak_lag, peak_r : ndarray/float — cross-corr
        bic_weights : ndarray — normalised BIC weights
        DELTA_CANDIDATES : ndarray
    """
    from types import SimpleNamespace

    delta_candidates = np.asarray(delta_candidates, dtype=float)

    # ── End-of-record observed discharge rate ──
    rate_mask = dyn_years >= (dyn_years[-1] - rate_window_yrs)
    t_rate = dyn_years[rate_mask]
    H_rate = H_dyn[rate_mask]
    sig_rate = sigma_dyn[rate_mask]
    W_rate = 1.0 / sig_rate**2
    X_rate_fit = np.column_stack([np.ones(len(t_rate)), t_rate])
    XtWX_rate = X_rate_fit.T @ np.diag(W_rate) @ X_rate_fit
    XtWy_rate = X_rate_fit.T @ (W_rate * H_rate)
    beta_rate = np.linalg.solve(XtWX_rate, XtWy_rate)
    _, r_obs_dyn = beta_rate
    cov_rate = np.linalg.inv(XtWX_rate)
    sigma_r_obs = np.sqrt(cov_rate[1, 1])
    t_end = dyn_years[-1]

    # ── Cross-correlation: ocean T rate vs discharge rate ──
    dT_ocean = np.diff(T_ocean_ann) / np.diff(T_ocean_years)
    t_dT = 0.5 * (T_ocean_years[:-1] + T_ocean_years[1:])
    dH_dyn = np.diff(H_dyn) / np.diff(dyn_years)
    t_dH = 0.5 * (dyn_years[:-1] + dyn_years[1:])

    max_lag = 12
    xcorr_lags = np.arange(0, max_lag + 1)
    xcorr_r = np.zeros(len(xcorr_lags))
    for k, lag in enumerate(xcorr_lags):
        dT_shifted = np.interp(t_dH, t_dT + lag, dT_ocean,
                               left=np.nan, right=np.nan)
        valid = np.isfinite(dT_shifted)
        if valid.sum() > 3:
            xcorr_r[k] = np.corrcoef(dH_dyn[valid], dT_shifted[valid])[0, 1]
        else:
            xcorr_r[k] = np.nan

    peak_lag = xcorr_lags[np.nanargmax(xcorr_r)]
    peak_r = float(np.nanmax(xcorr_r))

    # ── Demeaned WLS fit + rate constraint for each candidate delta ──
    fit_results = {}
    for delta in delta_candidates:
        T_shifted = np.interp(dyn_years, T_ocean_years + delta, T_ocean_ann,
                              left=np.nan, right=np.nan)
        valid = np.isfinite(T_shifted)
        if valid.sum() < 4:
            continue

        yrs_v = dyn_years[valid]
        H_v = H_dyn[valid]
        sig_v = sigma_dyn[valid]
        T_v = T_shifted[valid]

        dt_v = np.diff(yrs_v, prepend=yrs_v[0] - 1)
        int_T = np.cumsum(T_v * dt_v)

        H_mean = H_v.mean()
        int_T_mean = int_T.mean()
        t_mean = yrs_v.mean()

        H_dm = H_v - H_mean
        int_T_dm = int_T - int_T_mean
        t_dm = yrs_v - t_mean

        # Level equations
        W_level = 1.0 / sig_v**2
        X_level = np.column_stack([int_T_dm, t_dm])

        # Rate constraint: gamma * T_ocean(t_end - delta) + r0 = r_obs
        T_ocean_at_end = np.interp(t_end - delta, T_ocean_years, T_ocean_ann)
        X_rate_eq = np.array([[T_ocean_at_end, 1.0]])
        y_rate_eq = np.array([r_obs_dyn])
        W_rate_eq = np.array([rate_constraint_weight / sigma_r_obs**2])

        # Combined WLS
        X = np.vstack([X_level, X_rate_eq])
        y = np.concatenate([H_dm, y_rate_eq])
        W = np.concatenate([W_level, W_rate_eq])

        XtWX = X.T @ np.diag(W) @ X
        XtWy = X.T @ (W * y)
        beta_hat = np.linalg.solve(XtWX, XtWy)
        gamma_fit, r0_fit = beta_hat
        cov_beta = np.linalg.inv(XtWX)

        # Residuals on level equations (for BIC likelihood)
        H_pred_dm = X_level @ beta_hat
        resid = H_dm - H_pred_dm

        # R² in rate space: how well does gamma*T_ocean(t-delta) + r0
        # explain the observed discharge rate?
        rate_obs = np.diff(H_v) / np.diff(yrs_v)
        rate_pred = gamma_fit * T_v[1:] + r0_fit
        rate_mean = rate_obs.mean()
        SS_res_rate = np.sum((rate_obs - rate_pred)**2)
        SS_tot_rate = np.sum((rate_obs - rate_mean)**2)
        r2_rate = 1.0 - SS_res_rate / SS_tot_rate

        r_model = gamma_fit * T_ocean_at_end + r0_fit

        n_valid = len(H_dm)
        log_lik = -0.5 * np.sum(W_level * resid**2 + np.log(sig_v**2))
        bic = 2 * np.log(n_valid) - 2 * log_lik

        fit_results[float(delta)] = {
            'gamma': gamma_fit, 'r0': r0_fit,
            'cov': cov_beta, 'r2_rate': r2_rate, 'bic': bic,
            'n_valid': n_valid,
            'H_mean': H_mean, 'int_T_mean': int_T_mean, 't_mean': t_mean,
            'H_pred_dm': H_pred_dm, 'H_dm': H_dm, 'resid': resid,
            'yrs_valid': yrs_v, 'valid_mask': valid,
            'r_model': r_model, 'r_obs': r_obs_dyn,
            'T_ocean_at_end': T_ocean_at_end,
        }

    # ── Select best delta by BIC ──
    bics = np.array([fit_results[d]['bic'] for d in delta_candidates
                     if d in fit_results])
    valid_deltas = np.array([d for d in delta_candidates if d in fit_results])
    bic_min = bics.min()
    bic_weights = np.exp(-0.5 * (bics - bic_min))
    bic_weights /= bic_weights.sum()

    delta_best = float(valid_deltas[np.argmin(bics)])
    best = fit_results[delta_best]

    # ── MC posterior: BIC-weighted Gaussian mixture ──
    rng_mc = np.random.default_rng(seed)
    n_per_delta = np.round(n_samples * bic_weights).astype(int)
    n_per_delta[np.argmax(bic_weights)] += n_samples - n_per_delta.sum()

    gamma_posterior = np.empty(n_samples)
    r0_posterior = np.empty(n_samples)
    delta_posterior = np.empty(n_samples)

    idx_start = 0
    for d, n_d in zip(valid_deltas, n_per_delta):
        if n_d <= 0:
            continue
        fr = fit_results[d]
        draws = rng_mc.multivariate_normal(
            [fr['gamma'], fr['r0']], fr['cov'], size=n_d)
        gamma_posterior[idx_start:idx_start + n_d] = draws[:, 0]
        r0_posterior[idx_start:idx_start + n_d] = draws[:, 1]
        delta_posterior[idx_start:idx_start + n_d] = d
        idx_start += n_d

    # Shuffle to avoid ordering artefacts
    shuffle_idx = rng_mc.permutation(n_samples)
    gamma_posterior = gamma_posterior[shuffle_idx]
    r0_posterior = r0_posterior[shuffle_idx]
    delta_posterior = delta_posterior[shuffle_idx]

    return SimpleNamespace(
        gamma_posterior=gamma_posterior,
        r0_posterior=r0_posterior,
        delta_posterior=delta_posterior,
        delta_best=delta_best,
        gamma_best=best['gamma'],
        r0_best=best['r0'],
        r2_dyn=best['r2_rate'],
        bic_best=best['bic'],
        fit_results=fit_results,
        H_mean_cal=best['H_mean'],
        int_T_mean_cal=best['int_T_mean'],
        t_mean_cal=best['t_mean'],
        r_obs_dyn=r_obs_dyn,
        sigma_r_obs=sigma_r_obs,
        t_end=t_end,
        xcorr_lags=xcorr_lags,
        xcorr_r=xcorr_r,
        peak_lag=peak_lag,
        peak_r=peak_r,
        bic_weights=bic_weights,
        DELTA_CANDIDATES=valid_deltas,
    )



def project_ocean_temperature(transfer, T_surface_proj, time_proj,
                              n_samples=0, rng=None):
    """Project subsurface ocean temperature from surface T using a transfer function.

    Parameters
    ----------
    transfer : dict
        As returned by ``fit_ocean_transfer_function``.
    T_surface_proj : ndarray
        Projected surface temperature anomaly (°C).
    time_proj : ndarray
        Decimal-year times for projections.
    n_samples : int
        If > 0, draw MC samples propagating parameter and residual uncertainty.
        Returns ``(T_ocean_median, T_ocean_samples)`` instead of a single array.
    rng : np.random.Generator or None
        Random number generator for MC samples.

    Returns
    -------
    ndarray or tuple
        If ``n_samples == 0``: T_ocean projection (ndarray).
        If ``n_samples > 0``: (T_ocean_median, T_ocean_samples) where
        ``T_ocean_samples`` has shape ``(n_samples, len(time_proj))``.
    """
    alpha = transfer['alpha']
    beta = transfer['beta']
    T_ocean = alpha * T_surface_proj + beta

    if n_samples <= 0:
        return T_ocean

    if rng is None:
        rng = np.random.default_rng()

    alpha_draws = rng.normal(alpha, transfer['alpha_se'], size=n_samples)
    beta_draws = rng.normal(beta, transfer['beta_se'], size=n_samples)
    resid_draws = rng.normal(0, transfer['residual_std'],
                             size=(n_samples, len(T_surface_proj)))

    T_ocean_samples = (alpha_draws[:, None] * T_surface_proj[None, :]
                       + beta_draws[:, None] + resid_draws)

    return np.median(T_ocean_samples, axis=0), T_ocean_samples
