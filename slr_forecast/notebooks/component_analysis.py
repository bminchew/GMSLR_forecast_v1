"""
Component decomposition analysis functions.

Extracted from component_decomposition.ipynb to keep notebooks free of
function definitions.  All functions operate in SLR convention
(positive = sea level rise) and SI-derived internal units (meters, °C, yr).
"""

import numpy as np
import pandas as pd
from bayesian_dols import (
    build_level_design_vectors, fit_bayesian_level,
    calibrate_exponential_prior,
)

# ---------------------------------------------------------------------------
# Constants (re-exported from config when available; local fallbacks here
# so the module works standalone from the notebooks/ directory)
# ---------------------------------------------------------------------------
try:
    from slr_forecast.config import BASELINE_YEAR, M_TO_MM, N_SAMPLES
except ImportError:
    BASELINE_YEAR = 2005.0
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
