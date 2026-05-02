"""
Component decomposition projection and export functions.

Extracted from component_decomposition.ipynb to keep notebooks free of
function definitions.  All functions operate in SLR convention
(positive = sea level rise) and SI-derived internal units (meters, °C, yr).
"""

import json
import os

import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.special import expit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
try:
    from slr_forecast.config import BASELINE_YEAR, M_TO_MM, N_SAMPLES, WAIS_ONSET_YEAR
except ImportError:
    BASELINE_YEAR = 2000.0
    M_TO_MM = 1000.0
    N_SAMPLES = 2000
    WAIS_ONSET_YEAR = 2010.0


# =========================================================================
# A4 WAIS deep-uncertainty framework
# =========================================================================

# ---------------------------------------------------------------------------
# A4 WAIS scenario parameters
#
# Each scenario specifies:
#   P        – mixture weight (must sum to 1)
#   low_mm   – 5th-percentile SLR at 2100 (mm, n=3 baseline)
#   high_mm  – 95th-percentile SLR at 2100 (mm, n=3 baseline)
#   alpha    – skew-normal shape parameter in log-space
#              alpha > 0: positive skew (heavy right tail, Robel et al. 2019)
#              alpha < 0: negative skew (mode toward upper range)
#              alpha = 0: symmetric (reduces to log-normal)
#   beta_loc – log-mean of trajectory exponent β (power-law time ramp)
#   beta_scale – log-std of trajectory exponent β
#              β = 1: linear (constant rate)
#              β > 1: accelerating (back-loaded, MISI dynamics)
#              Derived from grounding-line flux scaling q_g ∝ h_g^{n+1}
#              (Schoof 2007); on retrograde beds, cumulative loss grows
#              as (t − t₀)^β with β ≈ 1.5–2.5 for n = 3–4.
#   misi     – whether MISI is active (informational flag)
#
# S1: Status quo — current discharge, no instability
# S2: MISI — marine ice sheet instability with amplification from missing
#     model processes (calving, feedbacks); positive skew per Robel et al.
#     (2019) who showed MISI amplifies and skews uncertainty toward
#     worst-case outcomes. Merges former S2 (moderate MISI) and S3
#     (MISI + amplifiers) — there is no physically defensible case for
#     MISI proceeding without amplification from processes that models
#     omit (Martin et al., AGU Advances).
# S3: MISI + MICI — full instability cascade including marine ice cliff
#     instability; negative skew reflects skepticism that MICI operates
#     at maximum efficiency.
# ---------------------------------------------------------------------------

A4_SCENARIOS = {
    'S1_status_quo': {'P': 0.10, 'low_mm': 25,  'high_mm': 85,
                      'alpha': 0.0,
                      'beta_loc': 0.0, 'beta_scale': 0.0,
                      'misi': False},
    'S2_misi':       {'P': 0.80, 'low_mm': 150, 'high_mm': 1000,
                      'alpha': 4.0,
                      'beta_loc': np.log(1.8), 'beta_scale': 0.3,
                      'misi': True},
    'S3_misi_mici':  {'P': 0.10, 'low_mm': 600, 'high_mm': 1400,
                      'alpha': -3.0,
                      'beta_loc': np.log(2.2), 'beta_scale': 0.3,
                      'misi': True},
}

# A1 rheology correction (n=3 -> n=4): Martin et al. (in press)
RHEOLOGY_FACTOR_MEDIAN = 1.28
RHEOLOGY_FACTOR_SIGMA = 0.07

# Observed Glen's law exponent: Millstein, Minchew, & Pegler (2022)
N_OBS_MEAN = 4.1
N_OBS_SIGMA = 0.4
N_REF = 3         # reference exponent used by ISMIP6 and literature scenarios

# Rheology sensitivity: fractional increase in SLR per unit increase in n
# Martin et al. (2026): 21-35% for Δn=1 → r₀ ≈ 0.28
RHEOLOGY_SENSITIVITY = 0.28


def _sample_log_skewnormal(n, low, high, alpha, rng):
    """Draw positive samples from a skew-normal in log-space.

    Parameters
    ----------
    n : int
        Number of samples.
    low, high : float
        5th and 95th percentiles of the target distribution (linear space).
    alpha : float
        Skew-normal shape parameter in log-space.  alpha=0 gives log-normal.
    rng : numpy.random.Generator

    Returns
    -------
    samples : ndarray, shape (n,)
        Positive samples in the same units as *low* and *high*.

    Notes
    -----
    The skew-normal distribution SN(xi, omega, alpha) has CDF that depends
    on all three parameters jointly.  We solve for (xi, omega) such that
    quantile(0.05) = log(low) and quantile(0.95) = log(high) in log-space,
    using scipy's ppf.  For alpha=0 this recovers the log-normal case.
    """
    from scipy.stats import skewnorm

    log_lo = np.log(low)
    log_hi = np.log(high)

    # Quantiles of the standard skew-normal SN(0, 1, alpha)
    q05_std = skewnorm.ppf(0.05, alpha)
    q95_std = skewnorm.ppf(0.95, alpha)

    # Solve for location (xi) and scale (omega) so that
    #   xi + omega * q05_std = log(low)
    #   xi + omega * q95_std = log(high)
    omega = (log_hi - log_lo) / (q95_std - q05_std)
    xi = log_lo - omega * q05_std

    # Draw in log-space, exponentiate
    log_samples = skewnorm.rvs(alpha, loc=xi, scale=omega, size=n,
                               random_state=rng)
    return np.exp(log_samples)


def sample_a4_wais(n_samples, rng, year=2100, rheology_mode='A',
                   anchor_year=None, anchor_value_mm=None,
                   anchor_sigma_mm=None, obs_value_mm=None,
                   obs_sigma_mm=None):
    """Draw WAIS SLR samples (meters) from A4 scenario mixture at a given year.

    Returns array of shape (n_samples,) in **meters**.

    The projection is anchored to the IMBIE observed cumulative value at
    ``anchor_year``.  Before the anchor year, samples are drawn from
    N(obs_value, obs_sigma²) to reflect IMBIE measurement uncertainty.
    After the anchor year, the *remaining* A4 contribution is distributed
    with a power-law ramp, with the anchor uncertainty propagated:

        H(t) = H_anchor_i + H_remaining_i · ((t − t_a) / (2100 − t_a))^β

    where H_anchor_i ~ N(anchor_value, anchor_sigma²) and
    H_remaining_i = H_2100 − H_anchor_i.

    Parameters
    ----------
    n_samples : int
    rng : numpy.random.Generator
    year : float
    rheology_mode : {'A', 'B'}
        How the rheology correction (n=3 → n≈4) is applied:

        **Mode A** — Independent corrections (default).
          Scenario ranges and β priors are defined at n=3 (matching the
          literature). Two separate corrections are applied:
            1. Endpoint: R ~ N(1.28, 0.07²), truncated ≥ 1.
            2. Trajectory: β_corrected = β_{n=3} · (n_draw + 1) / (n_ref + 1),
               where n_draw ~ N(4.1, 0.4²).
          R and the β correction are drawn independently.

        **Mode B** — Unified n-driven corrections.
          A single draw of n per sample drives both endpoint and trajectory:
            1. n_draw ~ N(4.1, 0.4²), clipped ≥ n_ref.
            2. Endpoint: R(n) = 1 + r₀ · (n_draw − n_ref).
            3. Trajectory: β(n) = β_ref · (n_draw + 1) / (n_ref + 1).
          Correlates the endpoint and trajectory corrections through n.
    anchor_year : float or None
        Year at which the projection is anchored to observations.  If None,
        uses ``WAIS_ONSET_YEAR`` from config (default 2010).
    anchor_value_mm : float or None
        IMBIE observed cumulative WAIS SLR (mm, relative to BASELINE_YEAR)
        at ``anchor_year``.  If None, defaults to 0.0.
    anchor_sigma_mm : float or None
        IMBIE 1-sigma uncertainty (mm) at ``anchor_year``.  If None,
        defaults to 0.0 (deterministic anchor).
    obs_value_mm : float or None
        IMBIE observed cumulative (mm) at ``year``, used for pre-anchor
        years.  If None, uses ``anchor_value_mm``.
    obs_sigma_mm : float or None
        IMBIE 1-sigma uncertainty (mm) at ``year``, used for pre-anchor
        years.  If None, defaults to 0.0.
    """
    if anchor_year is None:
        anchor_year = WAIS_ONSET_YEAR
    if anchor_value_mm is None:
        anchor_value_mm = 0.0
    if anchor_sigma_mm is None:
        anchor_sigma_mm = 0.0

    # Before anchor year: return IMBIE value with measurement uncertainty
    if year <= anchor_year:
        val = obs_value_mm if obs_value_mm is not None else anchor_value_mm
        sig = obs_sigma_mm if obs_sigma_mm is not None else 0.0
        if sig > 0:
            return rng.normal(val, sig, size=n_samples) / M_TO_MM
        return np.full(n_samples, val / M_TO_MM)

    t_norm = (year - anchor_year) / (2100 - anchor_year)

    if rheology_mode not in ('A', 'B'):
        raise ValueError(f"rheology_mode must be 'A' or 'B', got {rheology_mode!r}")

    # Per-sample anchor with IMBIE uncertainty
    if anchor_sigma_mm > 0:
        anchor_draws = rng.normal(anchor_value_mm, anchor_sigma_mm,
                                  size=n_samples)
    else:
        anchor_draws = np.full(n_samples, anchor_value_mm)

    samples = np.zeros(n_samples)
    scenario_names = list(A4_SCENARIOS.keys())
    probs = np.array([A4_SCENARIOS[s]['P'] for s in scenario_names])

    scenario_idx = rng.choice(len(scenario_names), size=n_samples, p=probs)

    # ── Pre-draw n for Mode B (shared across scenarios within each sample) ──
    if rheology_mode == 'B':
        n_draw_all = rng.normal(N_OBS_MEAN, N_OBS_SIGMA, size=n_samples)
        n_draw_all = np.maximum(n_draw_all, N_REF)  # n ≥ n_ref

    # Spawn independent child RNGs per scenario
    child_rngs = rng.spawn(len(scenario_names))

    for i, sname in enumerate(scenario_names):
        mask = scenario_idx == i
        n_s = mask.sum()
        if n_s == 0:
            continue
        s = A4_SCENARIOS[sname]
        crng = child_rngs[i]

        # ── Endpoint: draw H_2100 from skew-normal (n=3 ranges) ──
        base = _sample_log_skewnormal(
            n_s, s['low_mm'], s['high_mm'], s['alpha'], crng,
        )

        if rheology_mode == 'A':
            # ── Mode A: independent endpoint and trajectory corrections ──
            rheo = crng.normal(RHEOLOGY_FACTOR_MEDIAN, RHEOLOGY_FACTOR_SIGMA,
                               size=n_s)
            rheo = np.maximum(rheo, 1.0)
            base *= rheo

            # Trajectory exponent: draw β at n=3, then correct for n≈4
            if s['beta_scale'] > 0:
                beta_n3 = crng.lognormal(s['beta_loc'], s['beta_scale'],
                                         size=n_s)
                n_draw = crng.normal(N_OBS_MEAN, N_OBS_SIGMA, size=n_s)
                n_draw = np.maximum(n_draw, N_REF)
                beta = beta_n3 * (n_draw + 1) / (N_REF + 1)
            else:
                beta = np.ones(n_s)  # S1: linear ramp, no correction

        else:  # Mode B
            n_draw = n_draw_all[mask]
            rheo = 1.0 + RHEOLOGY_SENSITIVITY * (n_draw - N_REF)
            rheo = np.maximum(rheo, 1.0)
            base *= rheo

            if s['beta_scale'] > 0:
                beta_ref = crng.lognormal(s['beta_loc'], s['beta_scale'],
                                          size=n_s)
                beta = beta_ref * (n_draw + 1) / (N_REF + 1)
            else:
                beta = np.ones(n_s)

        # Remaining contribution after anchor: H_2100 - H_anchor_i
        anchor_i = anchor_draws[mask]
        h_remaining = np.maximum(base - anchor_i, 0.0)
        samples[mask] = anchor_i + h_remaining * (t_norm ** beta)

    return samples / M_TO_MM  # meters


def sample_a4_wais_endpoint(n_samples, rng, rheology_mode='A',
                             scenario_overrides=None):
    """Draw WAIS SLR endpoint samples (meters) at 2100 from the A4 mixture.

    This is a lightweight wrapper for sensitivity analyses that need to
    perturb scenario parameters without trajectory or anchor logic.  Each
    scenario's endpoint is sampled from the log-skew-normal, multiplied by
    the rheology correction, and returned directly.

    Parameters
    ----------
    n_samples : int
    rng : numpy.random.Generator
    rheology_mode : {'A', 'B'}
    scenario_overrides : dict or None
        Per-scenario parameter overrides.  Keys are scenario names
        (e.g. 'S2_misi'); values are dicts that can override any of
        'P', 'low_mm', 'high_mm', 'alpha'.  Missing keys use defaults
        from A4_SCENARIOS.  You can also pass a top-level key 'weights'
        mapping scenario names to new probabilities (must sum to 1).

    Returns
    -------
    samples_m : ndarray, shape (n_samples,)
        Endpoint samples in meters at 2100.
    """
    overrides = scenario_overrides or {}

    scenario_names = list(A4_SCENARIOS.keys())

    # Build effective parameters per scenario
    eff = {}
    for sname in scenario_names:
        base = dict(A4_SCENARIOS[sname])
        if sname in overrides:
            base.update(overrides[sname])
        eff[sname] = base

    # Allow top-level weight override
    if 'weights' in overrides:
        for sname, w in overrides['weights'].items():
            eff[sname]['P'] = w

    probs = np.array([eff[s]['P'] for s in scenario_names])
    probs = probs / probs.sum()  # ensure normalization

    scenario_idx = rng.choice(len(scenario_names), size=n_samples, p=probs)

    # Pre-draw n for Mode B
    if rheology_mode == 'B':
        n_draw_all = rng.normal(N_OBS_MEAN, N_OBS_SIGMA, size=n_samples)
        n_draw_all = np.maximum(n_draw_all, N_REF)

    # Spawn independent child RNGs per scenario so that changing n_s in
    # one scenario (e.g. via weight perturbation) does not shift the RNG
    # state for subsequent scenarios.
    child_rngs = rng.spawn(len(scenario_names))

    samples = np.zeros(n_samples)
    for i, sname in enumerate(scenario_names):
        mask = scenario_idx == i
        n_s = mask.sum()
        if n_s == 0:
            continue
        s = eff[sname]
        crng = child_rngs[i]

        base = _sample_log_skewnormal(
            n_s, s['low_mm'], s['high_mm'], s['alpha'], crng,
        )

        if rheology_mode == 'A':
            rheo = crng.normal(RHEOLOGY_FACTOR_MEDIAN, RHEOLOGY_FACTOR_SIGMA,
                               size=n_s)
            rheo = np.maximum(rheo, 1.0)
            base *= rheo
        else:
            n_draw = n_draw_all[mask]
            rheo = 1.0 + RHEOLOGY_SENSITIVITY * (n_draw - N_REF)
            rheo = np.maximum(rheo, 1.0)
            base *= rheo

        samples[mask] = base

    return samples / M_TO_MM  # meters


def sample_a4_wais_trajectories(n_samples, rng, years, rheology_mode='A',
                                 anchor_year=None, anchor_value_mm=None,
                                 anchor_sigma_mm=None,
                                 obs_years=None, obs_values_mm=None,
                                 obs_sigmas_mm=None):
    """Draw coherent WAIS SLR trajectories from the A4 scenario mixture.

    Unlike ``sample_a4_wais`` (which draws independently at each year),
    this function draws scenario assignments, endpoint values, rheology
    factors, and trajectory exponents **once** per sample, then evaluates
    the power-law ramp deterministically across all years.  This ensures
    that individual trajectories are smooth and internally consistent.

    Parameters
    ----------
    n_samples : int
    rng : numpy.random.Generator
    years : array-like
        Projection years (e.g. 1950–2150).
    rheology_mode : {'A', 'B'}
    anchor_year : float or None
        Defaults to WAIS_ONSET_YEAR.
    anchor_value_mm : float or None
        IMBIE cumulative WAIS SLR (mm) at anchor_year.
    anchor_sigma_mm : float or None
        IMBIE 1-sigma (mm) at anchor_year.
    obs_years, obs_values_mm, obs_sigmas_mm : array-like or None
        IMBIE time series for pre-anchor interpolation (years, mm, mm).
        If provided, pre-anchor samples are drawn from N(obs(t), sigma(t)²).

    Returns
    -------
    samples_m : ndarray, shape (n_samples, len(years))
        Trajectories in meters relative to BASELINE_YEAR.
    params : dict
        Per-sample drawn parameters: 'scenario_idx', 'h2100_mm',
        'beta', 'anchor_mm'.
    """
    from scipy.interpolate import interp1d

    years = np.asarray(years, dtype=float)
    n_years = len(years)

    if anchor_year is None:
        anchor_year = WAIS_ONSET_YEAR
    if anchor_value_mm is None:
        anchor_value_mm = 0.0
    if anchor_sigma_mm is None:
        anchor_sigma_mm = 0.0

    # ── Build obs interpolators for pre-anchor years ──
    if obs_years is not None and obs_values_mm is not None:
        obs_interp = interp1d(obs_years, obs_values_mm,
                              kind='linear', bounds_error=False, fill_value=0.0)
        if obs_sigmas_mm is not None:
            sig_interp = interp1d(obs_years, obs_sigmas_mm,
                                  kind='linear', bounds_error=False, fill_value=0.0)
        else:
            sig_interp = lambda t: 0.0
    else:
        obs_interp = lambda t: anchor_value_mm
        sig_interp = lambda t: 0.0

    # ── Draw all per-sample parameters once ──
    scenario_names = list(A4_SCENARIOS.keys())
    probs = np.array([A4_SCENARIOS[s]['P'] for s in scenario_names])
    scenario_idx = rng.choice(len(scenario_names), size=n_samples, p=probs)

    # Per-sample anchor with IMBIE uncertainty
    if anchor_sigma_mm > 0:
        anchor_draws = rng.normal(anchor_value_mm, anchor_sigma_mm,
                                  size=n_samples)
    else:
        anchor_draws = np.full(n_samples, anchor_value_mm)

    # Pre-draw n for Mode B
    if rheology_mode == 'B':
        n_draw_all = rng.normal(N_OBS_MEAN, N_OBS_SIGMA, size=n_samples)
        n_draw_all = np.maximum(n_draw_all, N_REF)

    h2100 = np.zeros(n_samples)
    beta_arr = np.zeros(n_samples)

    # Spawn independent child RNGs per scenario
    child_rngs = rng.spawn(len(scenario_names))

    for i, sname in enumerate(scenario_names):
        mask = scenario_idx == i
        n_s = mask.sum()
        if n_s == 0:
            continue
        s = A4_SCENARIOS[sname]
        crng = child_rngs[i]

        # Endpoint: draw H_2100 from skew-normal (n=3 ranges)
        base = _sample_log_skewnormal(
            n_s, s['low_mm'], s['high_mm'], s['alpha'], crng,
        )

        if rheology_mode == 'A':
            rheo = crng.normal(RHEOLOGY_FACTOR_MEDIAN, RHEOLOGY_FACTOR_SIGMA,
                               size=n_s)
            rheo = np.maximum(rheo, 1.0)
            base *= rheo

            if s['beta_scale'] > 0:
                beta_n3 = crng.lognormal(s['beta_loc'], s['beta_scale'],
                                         size=n_s)
                n_draw = crng.normal(N_OBS_MEAN, N_OBS_SIGMA, size=n_s)
                n_draw = np.maximum(n_draw, N_REF)
                beta_arr[mask] = beta_n3 * (n_draw + 1) / (N_REF + 1)
            else:
                beta_arr[mask] = 1.0
        else:  # Mode B
            n_draw = n_draw_all[mask]
            rheo = 1.0 + RHEOLOGY_SENSITIVITY * (n_draw - N_REF)
            rheo = np.maximum(rheo, 1.0)
            base *= rheo

            if s['beta_scale'] > 0:
                beta_ref = crng.lognormal(s['beta_loc'], s['beta_scale'],
                                          size=n_s)
                beta_arr[mask] = beta_ref * (n_draw + 1) / (N_REF + 1)
            else:
                beta_arr[mask] = 1.0

        h2100[mask] = base

    # ── Compute trajectories across all years ──
    samples_mm = np.zeros((n_samples, n_years))
    t_denom = 2100.0 - anchor_year

    for j, yr in enumerate(years):
        if yr < (obs_years[0] if obs_years is not None else anchor_year):
            # Before observations: zero
            continue
        elif yr <= anchor_year:
            # Pre-anchor: IMBIE observation with measurement uncertainty
            val = float(obs_interp(yr))
            sig = float(sig_interp(yr))
            if sig > 0:
                samples_mm[:, j] = rng.normal(val, sig, size=n_samples)
            else:
                samples_mm[:, j] = val
        else:
            # Post-anchor: deterministic power-law ramp from drawn parameters
            t_norm = (yr - anchor_year) / t_denom
            h_remaining = np.maximum(h2100 - anchor_draws, 0.0)
            samples_mm[:, j] = anchor_draws + h_remaining * (t_norm ** beta_arr)

    params = {
        'scenario_idx': scenario_idx,
        'h2100_mm': h2100,
        'beta': beta_arr,
        'anchor_mm': anchor_draws,
    }
    return samples_mm / M_TO_MM, params


# =========================================================================
# Glacier volume cap
# =========================================================================

def apply_glacier_volume_cap(samples, v_total=0.32):
    """Cap cumulative glacier mass loss at the total glacier ice volume.

    Glaciers are a finite reservoir (~0.32 m SLE from the Randolph Glacier
    Inventory / Farinotti et al. 2019).  Under sustained warming, cumulative
    mass loss cannot exceed this volume.  This function clamps each MC
    sample so that cumulative loss never exceeds ``v_total``.

    Parameters
    ----------
    samples : ndarray, shape (n_samples, n_times)
        Cumulative glacier SLR contribution in meters (positive = SLR).
        Values are expected to be relative to a baseline year where
        cumulative loss is near zero.
    v_total : float
        Total glacier volume in meters SLE (default 0.32 m).

    Returns
    -------
    ndarray, same shape as ``samples``
        Capped samples.  For each MC draw, the cumulative trajectory is
        clamped at ``v_total`` from the first time it would exceed that
        value onward.
    """
    capped = samples.copy()
    capped[capped > v_total] = v_total
    return capped


# =========================================================================
# Greenland joint-model projections
# =========================================================================

def project_greenland_joint_ensemble(
    result_joint,
    ocean_transfer,
    proj_monthly_temps,
    proj_monthly_times,
    gr_temp_monthly,
    gr_time_monthly,
    T_ocean_monthly,
    time_ocean_monthly,
    projection_times,
    baseline_year=2005.0,
    n_samples=2000,
    AA=2.58,
    seed=None,
):
    """Project Greenland SLR from the joint SMB + discharge model.

    Uses separate SMB and discharge posteriors from ``result_joint``,
    with ocean temperature projected via a surface-to-ocean transfer
    function for the discharge ODE.

    Parameters
    ----------
    result_joint : BayesianGreenlandJointResult
        Joint model fit result.
    ocean_transfer : dict
        From ``fit_ocean_transfer_function``.
    proj_monthly_temps : dict
        ``{ssp_name: ndarray}`` — monthly GMST for each SSP.
    proj_monthly_times : dict
        ``{ssp_name: ndarray}`` — monthly decimal years.
    gr_temp_monthly : ndarray
        Historical monthly Greenland surface T (for design vectors).
    gr_time_monthly : ndarray
        Decimal years for Greenland T.
    T_ocean_monthly : ndarray
        Historical monthly ocean T (for ODE spin-up).
    time_ocean_monthly : ndarray
        Decimal years for ocean T.
    projection_times : ndarray
        Annual times at which to evaluate projections.
    baseline_year : float
    n_samples : int
    AA : float
        Arctic amplification factor (GMST → Greenland surface T).
    seed : int or None

    Returns
    -------
    dict
        ``{ssp: {'samples': (n_samples, n_times), 'median': ...,
        'p5': ..., 'p17': ..., 'p83': ..., 'p95': ...,
        'smb_median': ..., 'dyn_median': ...}}``
    """
    from bayesian_dols import build_level_design_vectors, solve_twolayer_ode

    rng = np.random.default_rng(seed)

    # Draw posterior indices
    n_post = len(result_joint.a_smb_posterior)
    mc_idx = rng.choice(n_post, size=n_samples, replace=n_samples > n_post)

    # Extract posteriors at drawn indices
    a_smb = result_joint.a_smb_posterior[mc_idx]
    b_smb = result_joint.b_smb_posterior[mc_idx]
    H0_smb = result_joint.H0_smb_posterior[mc_idx]
    gamma_atm = result_joint.gamma_atm_posterior[mc_idx]
    gamma_ocean = result_joint.gamma_ocean_posterior[mc_idx]
    tau = result_joint.tau_posterior[mc_idx]
    D0 = result_joint.D0_posterior[mc_idx]
    H0_dyn = result_joint.H0_dyn_posterior[mc_idx]

    # Transfer function draws (propagate parameter uncertainty)
    alpha_draws = rng.normal(ocean_transfer['alpha'],
                             ocean_transfer['alpha_se'], size=n_samples)
    beta_draws = rng.normal(ocean_transfer['beta'],
                            ocean_transfer['beta_se'], size=n_samples)

    projections = {}

    for ssp_name in proj_monthly_temps:
        T_gmst_mon = proj_monthly_temps[ssp_name]
        t_mon = proj_monthly_times[ssp_name]

        # Build Greenland surface T = AA × GMST (monthly)
        T_gr_proj = T_gmst_mon * AA

        # Splice: historical Greenland T + projected Greenland T
        ssp_start = t_mon[0]
        hist_mask_gr = gr_time_monthly < ssp_start
        # Limit historical to where we have data
        gr_hist_t = gr_time_monthly[hist_mask_gr]
        gr_hist_T = gr_temp_monthly[hist_mask_gr]

        # For SSP period, use AA × GMST
        ssp_mask = t_mon >= ssp_start
        t_full_gr = np.concatenate([gr_hist_t, t_mon[ssp_mask]])
        T_full_gr = np.concatenate([gr_hist_T, T_gr_proj[ssp_mask]])

        # Build design vectors for Greenland surface T
        dv = build_level_design_vectors(
            temperature_monthly=T_full_gr,
            time_monthly=t_full_gr,
            obs_times=projection_times,
        )
        I2_proj = dv['I2_obs']
        I1_proj = dv['I1_obs']
        I0_proj = dv['I0_obs']

        # Build projected ocean T: splice historical + transfer(AA × GMST)
        # Historical ocean T for ODE spin-up
        hist_mask_oc = time_ocean_monthly < ssp_start
        oc_hist_t = time_ocean_monthly[hist_mask_oc]
        oc_hist_T = T_ocean_monthly[hist_mask_oc]

        # MC ensemble
        ens_total = np.zeros((n_samples, len(projection_times)))
        ens_smb = np.zeros((n_samples, len(projection_times)))
        ens_dyn = np.zeros((n_samples, len(projection_times)))

        for i in range(n_samples):
            # SMB: H_smb = a·I2 + b·I1 + H0_smb
            H_smb_i = a_smb[i] * I2_proj + b_smb[i] * I1_proj + H0_smb[i]

            # Discharge: build per-draw ocean T using transfer function
            T_ocean_proj_i = alpha_draws[i] * T_gr_proj[ssp_mask] + beta_draws[i]
            t_full_oc = np.concatenate([oc_hist_t, t_mon[ssp_mask]])
            T_full_oc = np.concatenate([oc_hist_T, T_ocean_proj_i])

            # Solve ODE: dD_eff/dt = (T_ocean - D_eff) / tau
            D_eff_i, _ = solve_twolayer_ode(T_full_oc, t_full_oc,
                                             tau[i], np.inf)

            # Cumulative discharge at projection times
            # ∫D_eff dt via trapezoidal rule, then interpolate
            dt = np.diff(t_full_oc)
            D_eff_mid = 0.5 * (D_eff_i[:-1] + D_eff_i[1:])
            cum_D = np.concatenate([[0], np.cumsum(D_eff_mid * dt)])
            cum_D_proj = np.interp(projection_times, t_full_oc, cum_D)

            # Also need I1 for gamma_atm (atmospheric T sensitivity of discharge)
            H_dyn_i = (gamma_atm[i] * I1_proj
                       + gamma_ocean[i] * cum_D_proj
                       + D0[i] * I0_proj + H0_dyn[i])

            H_total_i = H_smb_i + H_dyn_i

            # Rebase to baseline_year
            bl_idx = np.argmin(np.abs(projection_times - baseline_year))
            ens_smb[i] = H_smb_i - H_smb_i[bl_idx]
            ens_dyn[i] = H_dyn_i - H_dyn_i[bl_idx]
            ens_total[i] = H_total_i - H_total_i[bl_idx]

        projections[ssp_name] = {
            'samples': ens_total,
            'median': np.median(ens_total, axis=0),
            'p5': np.percentile(ens_total, 5, axis=0),
            'p17': np.percentile(ens_total, 17, axis=0),
            'p83': np.percentile(ens_total, 83, axis=0),
            'p95': np.percentile(ens_total, 95, axis=0),
            'smb_median': np.median(ens_smb, axis=0),
            'dyn_median': np.median(ens_dyn, axis=0),
        }

    return projections


# =========================================================================
# IPCC component readers
# =========================================================================

def read_ipcc_component_nc(conf_base, conf_level, ssp_code, component):
    """Read IPCC AR6 confidence-level NetCDF for a single component.

    Parameters
    ----------
    conf_base : str
        Base directory for confidence output files.
    conf_level : str
        E.g. ``'medium_confidence'``.
    ssp_code : str
        E.g. ``'ssp245'``.
    component : str
        E.g. ``'oceandynamics'``, ``'glaciers'``, ``'GIS'``, ``'AIS'``,
        ``'landwaterstorage'``, ``'total'``.

    Returns
    -------
    dict or None
        ``{'years': ndarray, 'quantiles': ndarray, 'slc': ndarray}``
        where ``slc`` is in mm.  Returns ``None`` if file not found.
    """
    fname = f'{component}_{ssp_code}_{conf_level}_values.nc'
    fpath = os.path.join(conf_base, conf_level, ssp_code, fname)
    if not os.path.exists(fpath):
        return None
    ds = nc.Dataset(fpath, 'r')
    data = {
        'years': ds.variables['years'][:].data.copy(),
        'quantiles': ds.variables['quantiles'][:].data.copy(),
        'slc': np.squeeze(ds.variables['sea_level_change'][:].data.copy()),  # mm
    }
    ds.close()
    return data


def ipcc_extract(data, quantiles_target=(0.05, 0.5, 0.95)):
    """Extract specific quantile lines from IPCC data.

    Parameters
    ----------
    data : dict
        As returned by ``read_ipcc_component_nc``.
    quantiles_target : tuple of float
        Target quantile values.

    Returns
    -------
    dict
        ``{'years': ndarray, 'q05': ndarray, 'q50': ndarray, 'q95': ndarray}``
        with keys named ``q{int(qt*100):02d}``.
    """
    out = {'years': data['years']}
    for qt in quantiles_target:
        idx = np.argmin(np.abs(data['quantiles'] - qt))
        out[f'q{int(qt * 100):02d}'] = data['slc'][idx]
    return out


# =========================================================================
# Projection statistics helpers
# =========================================================================

def get_our_stats(comp_projections, proj_years, ssp, component_key,
                  year=2100, n_samples=N_SAMPLES):
    """Get median [5, 95] in mm for a component at *year*.

    Parameters
    ----------
    comp_projections : dict
        ``{ssp: {component: {'samples': ndarray}}}``
    proj_years : ndarray
    ssp, component_key : str
    year : int
    n_samples : int

    Returns
    -------
    tuple of (p5, median, p95) or None
    """
    idx_yr = np.argmin(np.abs(proj_years - year))
    if component_key == 'AIS':
        samples = np.zeros((n_samples, len(proj_years)))
        for cname in ['EAIS', 'Peninsula', 'WAIS']:
            if cname in comp_projections[ssp]:
                samples += comp_projections[ssp][cname]['samples']
        s = samples[:, idx_yr] * M_TO_MM
    elif component_key in comp_projections[ssp]:
        s = comp_projections[ssp][component_key]['samples'][:, idx_yr] * M_TO_MM
    else:
        return None
    return np.percentile(s, 5), np.median(s), np.percentile(s, 95)


def get_ipcc_stats(ipcc_components, ssp, ipcc_key, year=2100):
    """Get median [5, 95] in mm from IPCC at *year*.

    Parameters
    ----------
    ipcc_components : dict
        ``{ssp: {component: data_dict}}``
    ssp, ipcc_key : str
    year : int

    Returns
    -------
    tuple of (p5, median, p95) or None
    """
    if ssp not in ipcc_components or ipcc_key not in ipcc_components[ssp]:
        return None
    data = ipcc_components[ssp][ipcc_key]
    yr_idx = np.argmin(np.abs(data['years'] - year))
    if np.abs(data['years'][yr_idx] - year) > 5:
        return None
    q05_idx = np.argmin(np.abs(data['quantiles'] - 0.05))
    q50_idx = np.argmin(np.abs(data['quantiles'] - 0.50))
    q95_idx = np.argmin(np.abs(data['quantiles'] - 0.95))
    return (data['slc'][q05_idx, yr_idx],
            data['slc'][q50_idx, yr_idx],
            data['slc'][q95_idx, yr_idx])


# =========================================================================
# JSON export
# =========================================================================

def safe_float(x):
    """Convert numpy scalar to Python float, handling NaN."""
    val = float(x)
    return val if np.isfinite(val) else None


def stats_dict(samples_mm):
    """Compute summary statistics dict from samples in mm."""
    return {
        'median': safe_float(np.median(samples_mm)),
        'p05': safe_float(np.percentile(samples_mm, 5)),
        'p17': safe_float(np.percentile(samples_mm, 17)),
        'p83': safe_float(np.percentile(samples_mm, 83)),
        'p95': safe_float(np.percentile(samples_mm, 95)),
        'mean': safe_float(np.mean(samples_mm)),
        'std': safe_float(np.std(samples_mm)),
    }


def export_results_json(export_dict, filepath):
    """Write results dictionary to JSON.

    Parameters
    ----------
    export_dict : dict
        Nested dictionary of results.
    filepath : str
        Output path.
    """
    with open(filepath, 'w') as f:
        json.dump(export_dict, f, indent=2)
    fsize = os.path.getsize(filepath) / 1024
    print(f'Exported: {filepath}  ({fsize:.1f} KB)')
    for section in export_dict:
        content = export_dict[section]
        if isinstance(content, dict):
            print(f'  {section}: {len(content)} entries')
        else:
            print(f'  {section}: {type(content).__name__}')


# =========================================================================
# ISMIP6 regional readers
# =========================================================================

OCEAN_AREA_M2 = 3.625e14  # standard ocean surface area (m²)

# ISMIP6 experiment → SSP mapping (Seroussi et al. 2020, Table 1)
ISMIP6_EXP_SSP = {
    'exp05': 'CMIP6-median', 'exp06': 'CMIP6-median',
    'exp09': 'SSP1-1.9', 'exp10': 'SSP1-2.6',
    'exp11': 'SSP2-4.5', 'exp12': 'SSP3-7.0', 'exp13': 'SSP5-8.5',
}


def read_ismip6_regional(
    ismip6_base,
    region,
    experiments=None,
    use_ctrl_anomaly=True,
):
    """Read ISMIP6 Antarctica regional ivaf and convert to SLE.

    Parameters
    ----------
    ismip6_base : str
        Path to ``ComputedScalarsPaper/`` directory.
    region : {1, 2, 3}
        1 = West Antarctica, 2 = East Antarctica, 3 = Peninsula.
    experiments : list of str or None
        Experiment names to read (e.g. ``['exp05', 'exp13']``).
        If None, reads all experiments in ISMIP6_EXP_SSP.
    use_ctrl_anomaly : bool
        If True (default), read the ``_minus_ctrl_proj_`` files
        (anomaly from control). If False, read raw ivaf and subtract
        the first time step.

    Returns
    -------
    dict
        ``{(group, model, exp): {'time': ndarray, 'sle_m': ndarray,
        'ssp': str}}``
        where ``sle_m`` is in meters (positive = sea level rise).
    """
    if experiments is None:
        experiments = list(ISMIP6_EXP_SSP.keys())

    region_var = f'ivaf_region_{region}'
    results = {}

    for group_name in sorted(os.listdir(ismip6_base)):
        group_path = os.path.join(ismip6_base, group_name)
        if not os.path.isdir(group_path):
            continue
        for model_name in sorted(os.listdir(group_path)):
            model_path = os.path.join(group_path, model_name)
            if not os.path.isdir(model_path):
                continue
            for exp in experiments:
                exp_path = os.path.join(model_path, exp)
                if not os.path.isdir(exp_path):
                    continue

                if use_ctrl_anomaly:
                    prefix = 'computed_ivaf_minus_ctrl_proj_AIS'
                else:
                    prefix = 'computed_ivaf_AIS'

                fname = f'{prefix}_{group_name}_{model_name}_{exp}.nc'
                fpath = os.path.join(exp_path, fname)
                if not os.path.exists(fpath):
                    continue

                try:
                    ds = nc.Dataset(fpath, 'r')
                    time = ds.variables['time'][:].data.copy()
                    ivaf_region = ds.variables[region_var][:].data.copy()
                    rhoi = float(ds.variables['rhoi'][:])
                    rhow = float(ds.variables['rhow'][:])
                    ds.close()
                except Exception:
                    continue

                # SLE = -delta_ivaf * rhoi / (ocean_area * rhow)
                # For ctrl anomaly files, ivaf_region is already the delta
                sle_m = -ivaf_region * rhoi / (OCEAN_AREA_M2 * rhow)

                ssp = ISMIP6_EXP_SSP.get(exp, exp)
                results[(group_name, model_name, exp)] = {
                    'time': time,
                    'sle_m': sle_m,
                    'ssp': ssp,
                }

    return results


def ismip6_ensemble_stats(ismip6_data, experiments=None, baseline_year=2015.0):
    """Compute ensemble median and spread from ISMIP6 regional data.

    Parameters
    ----------
    ismip6_data : dict
        As returned by ``read_ismip6_regional``.
    experiments : list of str or None
        Filter to specific experiments.  If None, use all.
    baseline_year : float
        Rebase all trajectories to this year.

    Returns
    -------
    dict
        ``{'time': ndarray, 'median': ndarray, 'p5': ndarray,
        'p95': ndarray, 'p17': ndarray, 'p83': ndarray,
        'n_models': int, 'labels': list}``
        All values in meters.
    """
    trajectories = []
    labels = []
    common_time = None

    for key, val in ismip6_data.items():
        if experiments is not None and key[2] not in experiments:
            continue
        t = val['time']
        sle = val['sle_m']
        if common_time is None:
            common_time = t
        # Interpolate onto common time grid
        sle_interp = np.interp(common_time, t, sle)
        # Rebase
        bl_idx = np.argmin(np.abs(common_time - baseline_year))
        sle_interp -= sle_interp[bl_idx]
        trajectories.append(sle_interp)
        labels.append(f'{key[0]}/{key[1]}')

    if len(trajectories) == 0:
        return None

    ens = np.array(trajectories)
    return {
        'time': common_time,
        'median': np.median(ens, axis=0),
        'p5': np.percentile(ens, 5, axis=0),
        'p17': np.percentile(ens, 17, axis=0),
        'p83': np.percentile(ens, 83, axis=0),
        'p95': np.percentile(ens, 95, axis=0),
        'n_models': len(trajectories),
        'labels': labels,
    }


# ---------------------------------------------------------------------------
# Rate-space blending
# ---------------------------------------------------------------------------

def blend_rate_space(proj_years, comp_samples, sq_rate_samples, sq_level_samples_rb,
                     sq_time, t_origin, h_origin, t_center, tau_blend):
    """Blend quadratic and component-sum rates, integrate to level.

    Parameters
    ----------
    proj_years : ndarray (T,)
        Full projection time axis.
    comp_samples : ndarray (N, T)
        Component-sum level samples (meters, rel. to baseline).
    sq_rate_samples : ndarray (N, T_sq)
        Quadratic rate samples (m/yr) on sq_time grid.
    sq_level_samples_rb : ndarray (N, T_sq)
        Quadratic level samples (meters, rel. to baseline) on sq_time grid.
    sq_time : ndarray (T_sq,)
        Time axis for quadratic samples.
    t_origin : float
        Forecast origin (end of obs record).
    h_origin : float
        Observed GMSL at t_origin (meters, rel. to baseline).
    t_center : float
        Centre of sigmoid transition.
    tau_blend : float
        Width of sigmoid transition (years).

    Returns
    -------
    forecast_samples : ndarray (N, T_forecast)
        Blended level forecast (meters, rel. to baseline).
    forecast_years : ndarray (T_forecast,)
        Time axis for the forecast (from t_origin onward).
    w_t : ndarray (T_forecast,)
        Sigmoid weight at each forecast year (1 = pure quadratic).
    """
    n_samples = comp_samples.shape[0]

    # Forecast grid: from origin onward (annual steps matching proj_years)
    fmask = proj_years >= t_origin
    f_years = proj_years[fmask]
    n_t = len(f_years)

    # Sigmoid weight: w=1 (quadratic) early, w=0 (component) late
    w_t = 1.0 - expit((f_years - t_center) / tau_blend)

    # Component-sum rate: central difference on annual grid
    dt_proj = np.diff(proj_years)
    comp_rate_all = np.diff(comp_samples, axis=1) / dt_proj[None, :]
    # Rate at midpoints; shift to full-year grid via averaging neighbours
    comp_rate_full = np.zeros_like(comp_samples)
    comp_rate_full[:, 1:-1] = 0.5 * (comp_rate_all[:, :-1] + comp_rate_all[:, 1:])
    comp_rate_full[:, 0] = comp_rate_all[:, 0]
    comp_rate_full[:, -1] = comp_rate_all[:, -1]

    # Restrict to forecast window
    comp_rate_f = comp_rate_full[:, fmask]

    # Interpolate quadratic rate onto forecast grid
    sq_rate_f = np.zeros((n_samples, n_t))
    for k in range(n_samples):
        sq_rate_f[k] = np.interp(f_years, sq_time, sq_rate_samples[k])

    # Blended rate (sample-by-sample)
    blended_rate = w_t[None, :] * sq_rate_f + (1.0 - w_t[None, :]) * comp_rate_f

    # Integrate from h_origin via cumulative trapezoidal rule
    dt_f = np.diff(f_years)
    forecast_samples = np.zeros((n_samples, n_t))
    forecast_samples[:, 0] = h_origin
    for j in range(1, n_t):
        forecast_samples[:, j] = (forecast_samples[:, j - 1]
                                  + 0.5 * (blended_rate[:, j - 1] + blended_rate[:, j])
                                  * dt_f[j - 1])

    return forecast_samples, f_years, w_t
