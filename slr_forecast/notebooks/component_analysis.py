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
    from slr_forecast.config import BASELINE_YEAR, N_SAMPLES, PREIND_TO_1900_M
    from slr_forecast import M_TO_MM
except ImportError:
    BASELINE_YEAR = 2000.0
    M_TO_MM = 1000.0
    N_SAMPLES = 2000
    PREIND_TO_1900_M = 0.010


# =========================================================================
# Data preparation helpers
# =========================================================================

def preindustrial_to_baseline_m(h5_path, baseline_year=BASELINE_YEAR):
    """GMSL rise from the pre-industrial (1850-1900) mean to *baseline_year* (m).

    The observed rise from 1900 to *baseline_year* is the mean of the
    Frederikse et al. (2020) and Dangendorf et al. (2024) reconstructions
    (annual values, ``raw/df_frederikse`` and ``raw/df_dangendorf`` in the
    processed HDF5). No record starts before 1900, so the rise from the
    1850-1900 mean to 1900 is the literature estimate ``PREIND_TO_1900_M``.

    Returns
    -------
    total : float
        Rise from pre-industrial to *baseline_year* (m).
    parts : dict
        Each reconstruction's 1900 -> baseline rise, their mean, and the
        pre-1900 increment (m).
    """
    fr = pd.read_hdf(h5_path, 'raw/df_frederikse')
    dg = pd.read_hdf(h5_path, 'raw/df_dangendorf')
    rise = lambda t, v: float(np.interp(baseline_year, t, v) - np.interp(1900.0, t, v))
    parts = {'frederikse2020': rise(fr['year'].values.astype(float), fr['gmsl'].values),
             'dangendorf2024': rise(dg['decimal_year'].values, dg['gmsl'].values)}
    parts['observed_1900'] = float(np.mean([parts['frederikse2020'],
                                            parts['dangendorf2024']]))
    parts['preind_to_1900'] = PREIND_TO_1900_M
    return parts['observed_1900'] + PREIND_TO_1900_M, parts


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
        Annual times, Y + 1.0 for calendar year Y.  Each value is the
        December cumulative level, which includes December's increment
        (the cumulative column is the running sum of the monthly rates),
        so it is the level at the end of year Y.  The increment of year Y
        therefore ends at ``years`` and its midpoint is ``years - 0.5``,
        which is where an annual-mean covariate belongs.
    H_rebase : ndarray
        Cumulative SLR (m), rebased.  With integer ``baseline_year`` the
        zero falls exactly at that time (the end of the previous year).
    sigma : ndarray
        Uncertainty of the *rebased* level (m, non-negative), i.e. of
        ``H(t) - H(baseline_year)`` rather than of the published
        cumulative anomaly.  Exactly zero at the baseline year, which is
        a normalization rather than a measurement.
    """
    t = df['decimal_year'].values
    year_int = np.floor(t).astype(int)
    unique_years = np.unique(year_int)

    years = np.zeros(len(unique_years))
    H = np.zeros(len(unique_years))
    sigma = np.zeros(len(unique_years))

    for i, yr in enumerate(unique_years):
        mask = year_int == yr
        years[i] = yr + 1.0
        H[i] = df['cumulative_mass_balance'].values[mask][-1]
        sigma[i] = np.abs(df['cumulative_mass_balance_sigma'].values[mask][-1])

    # Data is already in meters from the reader (read_imbie_* converts
    # mm → m).  No further unit conversion needed.

    # Rebase
    bl_idx = np.argmin(np.abs(years - baseline_year))
    H_rebase = H - H[bl_idx]

    # Re-anchor sigma to the same epoch as the level.  IMBIE reports the
    # cumulative sigma accumulated from the start of *its* record, so it
    # grows monotonically in time and is smallest at the record start.
    # That is the uncertainty of H(t) - H(t_start), not of the rebased
    # H(t) - H(baseline_year) we return above.  For a cumulative sum of
    # independent annual increments the rebased marginal is
    # |Var(H_t) - Var(H_bl)|, which is V-shaped about the anchor.  This
    # is the form every consumer expects: _anchor_covariance (see
    # component_levelspace_robust_se.py) builds the correlation-aware
    # covariance from it and requires it to be PSD, and the error bars
    # drawn on the rebased record are its marginals.  Passing the raw
    # published sigma instead makes that covariance non-PSD and its
    # sandwich intervals narrower than the naive ones.
    sigma = np.sqrt(np.abs(sigma**2 - sigma[bl_idx]**2))

    return years, H_rebase, sigma


def imbie_annual_rates(df):
    """Calendar years and annual-mean mass-balance rate and 1-sigma (m/yr).

    ``df`` is a monthly IMBIE-3 frame from ``read_imbie3``.  Year Y's
    rate is the mean of its monthly rates, so rate * 1 yr is the increment
    of the cumulative record over year Y, whose end-of-year level
    ``annualize_imbie`` labels Y + 1.0.
    """
    t = df['decimal_year'].values
    yi = np.floor(t).astype(int)
    years = np.unique(yi)
    rate = np.array([df['mass_balance_rate'].values[yi == y].mean() for y in years])
    sigma = np.array([df['mass_balance_rate_sigma'].values[yi == y].mean() for y in years])
    return years, rate, sigma


def fit_imbie_time_model(H_obs, years, rate, sigma_rate, order=1,
                         baseline_year=BASELINE_YEAR, seed=700,
                         n_samples=6000, n_burnin=3000,
                         prior_scale_acc=5e-5, prior_scale_jerk=1e-5,
                         prior_rate_sigma=0.002, prior_sigma_extra_scale=0.0002,
                         prior_H0_sigma=0.005, ar1=False, progress=False):
    """Fit a rate polynomial in time to an IMBIE-3 cumulative record.

    order=1: rate(t) = v + a tau;  order=2 adds A tau^2 (jerk j = 2A),
    with tau = t - baseline_year.  The likelihood is that of the annual
    increments (``fit_bayesian_level_annual_correlated``): each increment
    carries the reported rate error plus a fitted sigma_extra, independent
    across years (``ar1=True`` adds a fitted lag-1 correlation).  Year Y's
    increment is evaluated at its midpoint, tau = Y + 0.5 - baseline_year.
    Priors are signed Normals: a ~ N(0, prior_scale_acc), A ~ N(0,
    prior_scale_jerk), v ~ N(0, prior_rate_sigma) (m, yr units).

    Parameters
    ----------
    H_obs : ndarray
        End-of-year cumulative levels (m) from ``annualize_imbie``.
    years, rate, sigma_rate : ndarray
        Calendar years and annual rates and 1-sigma (m/yr) from
        ``imbie_annual_rates``, aligned with ``H_obs``.

    Returns
    -------
    dict with 'fit' (BayesianLevelCorrelatedResult), 'years', 'tau',
    'order', 'baseline_year'.
    """
    from bayesian_models import fit_bayesian_level_annual_correlated
    years = np.asarray(years, dtype=int)
    rate = np.asarray(rate, dtype=float)
    tau = years + 0.5 - baseline_year
    fit = fit_bayesian_level_annual_correlated(
        H_obs=np.asarray(H_obs, dtype=float), rate_obs=rate,
        sigma_rate_obs=np.asarray(sigma_rate, dtype=float),
        temperature=tau, time=years + 1.0, order=order,
        fit_sigma_extra=True, fit_ar1=ar1, symmetric_a=True, symmetric_b=True,
        prior_scale_a=prior_scale_jerk, prior_scale_b=prior_scale_acc,
        prior_b_mean=0.0, prior_c_mean=0.0, prior_c_sigma=prior_rate_sigma,
        prior_H0_mean=float(H_obs[0] - rate[0]), prior_H0_sigma=prior_H0_sigma,
        prior_sigma_extra_scale=prior_sigma_extra_scale,
        n_samples=n_samples, n_walkers=32, n_burnin=n_burnin, thin=2,
        seed=seed, progress=progress)
    return {'fit': fit, 'years': years, 'tau': tau, 'order': order,
            'baseline_year': baseline_year}


def time_model_draws(tm):
    """Posterior draws (A, a, v, H0) of a ``fit_imbie_time_model`` result.

    H0 is the level at ``baseline_year``: the level before the first
    increment plus the modelled increments of the years before baseline.
    A is zero for order=1.
    """
    fit = tm['fit']
    A = fit.posterior_samples[:, 0]
    a = fit.posterior_samples[:, 1]
    v = fit.posterior_samples[:, 2]
    pre = tm['years'] < tm['baseline_year']
    tp = tm['tau'][pre]
    H0 = fit.H0_posterior + pre.sum() * v + tp.sum() * a + (tp**2).sum() * A
    return A, a, v, H0


def time_model_levels(tm, t):
    """Posterior draws of the level (m) at times ``t``, shape (n_draws, len(t))."""
    A, a, v, H0 = time_model_draws(tm)
    tau = np.asarray(t, dtype=float) - tm['baseline_year']
    return (H0[:, None] + v[:, None] * tau + 0.5 * a[:, None] * tau**2
            + A[:, None] * tau**3 / 3.0)


def time_model_bic(tm, sigma_rate):
    """BIC of a ``fit_imbie_time_model`` result at the posterior mean.

    -2 ln L uses the cumulative-record covariance at the posterior-mean
    sigma_extra (and phi when fitted).  k counts the rate coefficients, H0,
    sigma_extra and phi when fitted.
    """
    from bayesian_models import _level_correlated_cov
    fit = tm['fit']
    r = fit.residuals
    n = len(r)
    phi = fit.phi_posterior
    C = _level_correlated_cov(np.asarray(sigma_rate, dtype=float), np.ones(n),
                              fit.sigma_extra_posterior.mean(),
                              phi.mean() if phi is not None else 0.0)
    _, logdet = np.linalg.slogdet(C)
    m2ll = float(r @ np.linalg.solve(C, r) + logdet + n * np.log(2 * np.pi))
    k = (tm['order'] + 1) + 2 + (1 if phi is not None else 0)
    return {'m2lnL': m2ll, 'k': k, 'n': n, 'bic': m2ll + k * np.log(n)}


def anchor_to_observed(samples, proj_years, obs_years, obs_H, obs_sigma,
                       baseline_year=BASELINE_YEAR, seed=0):
    """Shift projection members so they pass through the observed level.

    Each member's model path M_k(t) (zero at ``baseline_year``) is offset by
    D_k = A_k - M_k(t_a), where t_a is the end of the observed record and
    A_k ~ N(obs_H[-1], obs_sigma[-1]) is the observed level there with its
    uncertainty (relative to the baseline, as returned by
    ``annualize_imbie``).  The offset ramps linearly from zero at the
    baseline to D_k at t_a and is constant after it, so members keep their
    modelled shape, the projection is conditioned on the observed record's
    end point (as the WAIS trajectories are), and the baseline and the
    pre-baseline hindcast are unchanged.  The draws A_k are shared across
    scenarios when the same ``seed`` is used.

    Returns the shifted samples (same shape) and the offsets D_k.
    """
    samples = np.asarray(samples, dtype=float)
    proj_years = np.asarray(proj_years, dtype=float)
    t_a = float(obs_years[-1])
    rng = np.random.default_rng(seed)
    A = float(obs_H[-1]) + rng.normal(0.0, float(obs_sigma[-1]), size=samples.shape[0])
    M_a = np.array([np.interp(t_a, proj_years, row) for row in samples])
    D = A - M_a
    w = np.clip((proj_years - baseline_year) / (t_a - baseline_year), 0.0, 1.0)
    return samples + D[:, None] * w[None, :], D


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
    """Combine historical + SSP temperature, subtract caller-provided offset.

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
                                lag_years=0, annual=True, hac_maxlags=None):
    """Calibrate a linear transfer function: T_ocean = α·T_surface + β.

    Aligns surface and ocean temperature on common annual (or monthly)
    time points, fits OLS with HAC (Newey–West) standard errors, and
    returns parameters for projecting ocean temperature under SSP
    surface-T scenarios.  The residuals are serially correlated, so the
    plain OLS covariance is replaced by the HAC covariance (as in
    ``slr_analysis.calibrate_dols``) and the residual lag-1
    autocorrelation is returned for AR(1) residual noise in projections.

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
    hac_maxlags : int, optional
        Maximum lags for the HAC covariance.  If None, uses the Newey–West
        rule floor(4·(n/100)^(2/9)) (3 for n = 52 annual values).

    Returns
    -------
    dict with keys:
        'alpha'       : float — slope (°C_ocean / °C_surface)
        'beta'        : float — intercept (°C)
        'alpha_se'    : float — HAC standard error on alpha
        'beta_se'     : float — HAC standard error on beta
        'cov'         : ndarray (2, 2) — HAC covariance of (alpha, beta),
                        ordered [alpha, beta]
        'corr'        : float — HAC correlation of alpha and beta
        'hac_maxlags' : int   — lags used in the HAC covariance
        'resid_ar1'   : float — lag-1 autocorrelation of the residuals
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

    # OLS: T_ocean = alpha * T_surface + beta, HAC (Newey–West) covariance
    if hac_maxlags is None:
        hac_maxlags = int(np.floor(4 * (len(T_s) / 100) ** (2 / 9)))
    X = sm.add_constant(T_s)
    model = sm.OLS(T_o, X).fit(cov_type='HAC',
                               cov_kwds={'maxlags': hac_maxlags})

    beta = model.params[0]
    alpha = model.params[1]
    beta_se = model.bse[0]
    alpha_se = model.bse[1]
    # statsmodels orders [const, slope]; reorder to [alpha, beta]
    cov = np.asarray(model.cov_params())[::-1, ::-1]
    resid = np.asarray(model.resid)
    resid_ar1 = np.corrcoef(resid[:-1], resid[1:])[0, 1]

    return {
        'alpha': float(alpha),
        'beta': float(beta),
        'alpha_se': float(alpha_se),
        'beta_se': float(beta_se),
        'cov': cov,
        'corr': float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])),
        'hac_maxlags': int(hac_maxlags),
        'resid_ar1': float(resid_ar1),
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
    sigma_rate=None,
):
    """Fit the Greenland discharge delay model.

    The model is:

        H_dyn(t) = gamma * integral(T_ocean(t' - delta), t0..t) + r0 * t

    The time delay ``delta`` is selected from ``delta_candidates`` by BIC,
    and the posterior is a BIC-weighted mixture of Gaussian draws from each
    candidate's covariance.

    Two estimators:

    * ``sigma_rate`` given (production): weighted least squares on the
      annual increments H_i - H_{i-1} = (gamma T_i + r0) dt_i, with weights
      1/(sigma_rate_i dt_i)^2.  The cumulative record is a running sum of
      annual rates whose errors are independent, so this is the
      generalized-least-squares fit to the cumulative record under its
      correlated error covariance (bayesian_models._cumulate).  The excess
      scatter is absorbed by a common error scale s^2: the covariance is
      s^2 (X'WX)^{-1} with s^2 = chi2/dof, and the BIC uses the profile
      likelihood with s^2 estimated (k = 3: gamma, r0, s^2).  There is no
      rate-constraint row: the end-of-record rate is a linear function of
      the same increments, so a separate row would count them twice.  The
      end-of-record rate is still computed and returned as a check
      (``r_obs_dyn`` against each fit's ``r_model``).
    * ``sigma_rate`` None (the estimator used before 2026-10-17): demeaned
      WLS on the cumulative levels with diagonal weights 1/sigma_dyn^2 plus
      the end-of-record rate-constraint row.  Its BIC treats the cumulative
      values as independent, which is not correct for a running sum; kept
      for comparison and tests.

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
    sigma_rate : ndarray or None
        1-sigma of the annual discharge rate (m SLE/yr), aligned with
        ``dyn_years``.  Selects the increment estimator (see above).

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
    # Common fit window: every candidate delta is fit to, and its BIC
    # evaluated on, the same years -- those the delayed forcing covers for
    # all candidates.  The likelihood includes -0.5*sum(log sigma^2), so
    # per-delta windows of different length (a shorter delay reaches
    # earlier years) make the BICs incomparable and favour short delays.
    # The level estimator drops points with sigma_dyn == 0 (the rebase
    # anchor of annualize_imbie, where 1/sigma^2 is undefined).  The
    # increment estimator weights by the rate errors, and the anchor level
    # is an ordinary point of the running sum, so it keeps it.
    if sigma_rate is not None:
        sig_ok = np.isfinite(np.asarray(sigma_rate, float)) & (np.asarray(sigma_rate, float) > 0)
    else:
        sig_ok = np.asarray(sigma_dyn) > 0
    common = sig_ok.copy()
    for delta in delta_candidates:
        common &= np.isfinite(np.interp(dyn_years, T_ocean_years + delta, T_ocean_ann,
                                        left=np.nan, right=np.nan))
    for delta in delta_candidates:
        T_shifted = np.interp(dyn_years, T_ocean_years + delta, T_ocean_ann,
                              left=np.nan, right=np.nan)
        # Drop the rebase anchor along with epochs the delayed forcing
        # does not cover.  annualize_imbie returns sigma == 0 at the
        # baseline year -- a normalization rather than a measurement --
        # and the 1/sigma^2 weights below are undefined there.  Same
        # convention as the keep masks in
        # component_levelspace_robust_se.  Records whose sigma comes
        # from bayesian_models._cumulate are floored above zero, so this
        # leaves the Mouginot and Mankoff fits unchanged.
        valid = common & np.isfinite(T_shifted) & sig_ok
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
        W_level = 1.0 / sig_v**2 if sigma_rate is None else None
        X_level = np.column_stack([int_T_dm, t_dm])
        T_ocean_at_end = np.interp(t_end - delta, T_ocean_years, T_ocean_ann)

        if sigma_rate is not None:
            # Annual increments, independent errors (GLS on the levels)
            idx_v = np.flatnonzero(valid)
            if not np.all(np.diff(idx_v) == 1):
                raise ValueError('increment fit needs a contiguous fit window')
            dt_i = np.diff(yrs_v)
            dH = np.diff(H_v)
            X_inc = np.column_stack([T_v[1:] * dt_i, dt_i])
            w_inc = 1.0 / (np.asarray(sigma_rate, float)[valid][1:] * dt_i) ** 2
            XtWX = X_inc.T @ (w_inc[:, None] * X_inc)
            beta_hat = np.linalg.solve(XtWX, X_inc.T @ (w_inc * dH))
            gamma_fit, r0_fit = beta_hat
            r_inc = dH - X_inc @ beta_hat
            n_valid = len(dH)
            chi2 = float(np.sum(w_inc * r_inc**2))
            dof = n_valid - 2
            scale = chi2 / dof
            cov_beta = np.linalg.inv(XtWX) * scale
            # Profile log-likelihood, common error scale at its ML value
            s2_ml = chi2 / n_valid
            log_lik = -0.5 * (n_valid + np.sum(np.log(s2_ml / w_inc))
                              + n_valid * np.log(2 * np.pi))
            bic = 3 * np.log(n_valid) - 2 * log_lik
        else:
            # Rate constraint: gamma * T_ocean(t_end - delta) + r0 = r_obs
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

        # Level residuals (BIC likelihood of the level estimator; plots)
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

        if sigma_rate is not None:
            # Same operator as r_obs_dyn (WLS slope of the levels over the
            # final window) applied to the modelled levels
            in_rate = rate_mask[valid]
            if in_rate.sum() != rate_mask.sum():
                raise ValueError('end-of-record window lies outside the fit window')
            r_model = float(np.linalg.solve(
                XtWX_rate, X_rate_fit.T @ (W_rate * H_pred_dm[in_rate]))[1])
        else:
            r_model = gamma_fit * T_ocean_at_end + r0_fit

        if sigma_rate is None:
            n_valid = len(H_dm)
            log_lik = -0.5 * np.sum(W_level * resid**2 + np.log(sig_v**2))
            bic = 2 * np.log(n_valid) - 2 * log_lik
            chi2 = float(np.sum(W_level * resid**2))
            dof = n_valid - 1 - 2
            scale = 1.0

        fit_results[float(delta)] = {
            'gamma': gamma_fit, 'r0': r0_fit,
            'cov': cov_beta, 'r2_rate': r2_rate, 'bic': bic,
            'n_valid': n_valid,
            'H_mean': H_mean, 'int_T_mean': int_T_mean, 't_mean': t_mean,
            'H_pred_dm': H_pred_dm, 'H_dm': H_dm, 'resid': resid,
            'yrs_valid': yrs_v, 'valid_mask': valid,
            'r_model': r_model, 'r_obs': r_obs_dyn,
            'T_ocean_at_end': T_ocean_at_end,
            'chi2': chi2, 'dof': dof, 'chi2_over_dof': chi2 / dof,
            'scale': scale,
            'estimator': 'increments' if sigma_rate is not None else 'level',
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
        If > 0, draw MC samples propagating parameter and residual uncertainty:
        (alpha, beta) jointly from the HAC covariance and the residual as a
        stationary AR(1) on the ``time_proj`` grid (assumed annual).
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

    ab_draws = rng.multivariate_normal([alpha, beta], transfer['cov'],
                                       size=n_samples)
    alpha_draws, beta_draws = ab_draws[:, 0], ab_draws[:, 1]
    # Stationary AR(1) residual: marginal SD = residual_std
    phi, sd = transfer['resid_ar1'], transfer['residual_std']
    z = rng.standard_normal((n_samples, len(T_surface_proj)))
    resid_draws = np.empty_like(z)
    resid_draws[:, 0] = sd * z[:, 0]
    for k in range(1, z.shape[1]):
        resid_draws[:, k] = (phi * resid_draws[:, k - 1]
                             + sd * np.sqrt(1 - phi**2) * z[:, k])

    T_ocean_samples = (alpha_draws[:, None] * T_surface_proj[None, :]
                       + beta_draws[:, None] + resid_draws)

    return np.median(T_ocean_samples, axis=0), T_ocean_samples
