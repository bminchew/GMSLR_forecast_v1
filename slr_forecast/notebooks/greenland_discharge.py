"""Greenland discharge: calibration and projection, as functions.

Used by component_greenland.ipynb for the production fit and projections
(§2b, §4) and for the sensitivity tests in §6, which repeat both with an
alternative Arctic amplification factor or ocean-temperature layer. Keeping
one implementation means the sensitivity tests run exactly the production
code with one input changed.

Since 2026-10-17 the delay model is fitted to the annual increments of the
cumulative discharge record (fit_discharge_delay_model with sigma_rate),
which replaces the level fit with diagonal weights, the end-of-record rate
row and the sandwich covariance of component_greenland_robust_se.

The surface-to-ocean transfer is the low-pass form
T_ocean = g·EMA_tau(T_surface) + c (fit_ocean_lowpass_transfer): the
subsurface ocean follows the slow part of surface warming, so the annual
regression it replaces was attenuated by surface interannual variability.
"""

import numpy as np

from component_analysis import (ema_filter, fit_ocean_lowpass_transfer,
                                fit_discharge_delay_model)


def annualize(t_monthly, x_monthly):
    """Calendar-year means; returns (values, mid-year times)."""
    yr = np.floor(t_monthly).astype(int)
    uy = np.unique(yr)
    return np.array([x_monthly[yr == y].mean() for y in uy]), uy.astype(float) + 0.5


def full_years_only(t_monthly, x_monthly):
    """Drop the months of any calendar year with fewer than 12 values.

    EN4 ends mid-year (2021-06); an annual mean of a partial year carries
    part of the seasonal cycle and is not comparable with full-year means.
    """
    t_monthly = np.asarray(t_monthly, float)
    yr = np.floor(t_monthly).astype(int)
    uy, cnt = np.unique(yr, return_counts=True)
    keep = np.isin(yr, uy[cnt >= 12])
    return t_monthly[keep], np.asarray(x_monthly)[keep]


def calibrate_greenland_discharge(gr_temp_monthly, gr_time_monthly,
                                  T_ocean_monthly, time_ocean_monthly,
                                  mou_comp, delta_candidates,
                                  rate_window_yrs, n_samples, seed=2026,
                                  tau_grid=None):
    """Ocean transfer function and delay-model fit with posterior draws.

    The delay model is fitted to the annual increments of the Mouginot
    cumulative discharge with the reported annual rate errors
    (fit_discharge_delay_model with sigma_rate), which is the
    correlation-aware fit to the cumulative record. Each delta's covariance
    includes the excess-scatter scale chi2/dof, and the BIC weights use the
    same estimated scale.

    Returns
    -------
    ocean_transfer : dict
        fit_ocean_lowpass_transfer output
        (T_ocean = g EMA_tau(T_surface) + c; 'alpha', 'beta' at tau_best).
    result_discharge : SimpleNamespace
        fit_discharge_delay_model output with gamma/r0/delta posteriors.
    T_ocean_ann, T_ocean_years : ndarray
        Annual ocean temperature used in the fit.
    """
    kw = {} if tau_grid is None else {'tau_grid': tau_grid}
    ocean_transfer = fit_ocean_lowpass_transfer(
        T_surface_monthly=gr_temp_monthly, time_surface=gr_time_monthly,
        T_ocean_monthly=T_ocean_monthly, time_ocean=time_ocean_monthly, **kw)

    T_ocean_ann, T_ocean_years = annualize(time_ocean_monthly, T_ocean_monthly)

    result_discharge = fit_discharge_delay_model(
        dyn_years=mou_comp['time_dyn'], H_dyn=mou_comp['H_dyn'],
        sigma_dyn=mou_comp['sigma_dyn'],
        T_ocean_ann=T_ocean_ann, T_ocean_years=T_ocean_years,
        delta_candidates=delta_candidates,
        rate_window_yrs=rate_window_yrs,
        n_samples=n_samples, seed=seed,
        sigma_rate=discharge_rate_sigma(mou_comp))
    return ocean_transfer, result_discharge, T_ocean_ann, T_ocean_years


def discharge_rate_sigma(mou_comp, years=None):
    """Reported 1-sigma of the annual discharge rate (m SLE/yr) at *years*
    (default: mou_comp['time_dyn'])."""
    df = mou_comp['df']
    yrs_df = df['decimal_year'].values.astype(float)
    years = mou_comp['time_dyn'] if years is None else years
    idx = np.searchsorted(yrs_df, np.asarray(years, float))
    assert np.allclose(yrs_df[idx], years)
    return df['discharge_rate_sigma'].values[idx].astype(float)


def project_greenland_discharge(ssps, proj_years, proj_monthly_temps,
                                proj_monthly_times, proj_monthly_offsets,
                                temp_monthly, temp_time_monthly,
                                T_ocean_monthly, time_ocean_monthly,
                                result_discharge, alpha_draws, beta_draws,
                                AA_draws, oc_noise, oc_noise_years,
                                oc_noise_pre, oc_noise_pre_years,
                                baseline_year, sample_warming_paths,
                                aa_scale=None, gr_temp_monthly=None,
                                gr_time_monthly=None, tau_draws=None):
    """Discharge ensemble per SSP (metres SLE, rebased to *baseline_year*).

    For each member: draw (gamma, r0, delta) from the discharge posterior and
    (g, c, tau, AA) from their draws (alpha_draws = g, beta_draws = c); build
    one annual surface-T series (observed surface T before the scenario
    branch, then the member's GMST path x AA), filter it with EMA_tau and
    map it to ocean T as g EMA + c; use EN4 observations in the years EN4
    covers and the mapped series in all other years; add the transfer AR(1)
    residual on the years not covered by EN4; delay by delta, integrate, and
    un-demean with the calibration means. The filter runs through the EN4
    years, so its state at the end of EN4 reflects the observed surface
    history. tau_draws = None means tau = 0 (no filter).

    Surface T before the scenario branch: if gr_temp_monthly/gr_time_monthly
    (observed Greenland regional T, the series the transfer function is
    fitted to) are given, they are used for every month they cover, and
    GMST x AA only before the regional record. If they are None, GMST x AA
    is used throughout.

    aa_scale : callable or None
        Multiplicative scale on AA for GMST x AA years before the regional
        record (the historical ramp); None means no ramp.

    Returns {ssp: ndarray (n_samples, len(proj_years))}.
    """
    n_samples = len(alpha_draws)
    out = {}
    for ssp in ssps:
        unique_yrs, T_oc = project_greenland_ocean_temperature(
            proj_monthly_temps[ssp], proj_monthly_times[ssp], proj_monthly_offsets[ssp],
            temp_monthly, temp_time_monthly, T_ocean_monthly, time_ocean_monthly,
            alpha_draws, beta_draws, AA_draws, oc_noise, oc_noise_years,
            oc_noise_pre, oc_noise_pre_years, sample_warming_paths,
            aa_scale=aa_scale, gr_temp_monthly=gr_temp_monthly,
            gr_time_monthly=gr_time_monthly, tau_draws=tau_draws)
        t_oc_ann_proj = unique_yrs.astype(float) + 0.5

        ens_dyn = np.zeros((n_samples, len(proj_years)))
        dt_proj = np.diff(proj_years, prepend=proj_years[0] - 1)
        bl_idx = np.argmin(np.abs(proj_years - baseline_year))

        for i in range(n_samples):
            delta_i = result_discharge.delta_posterior[i]
            gamma_i = result_discharge.gamma_posterior[i]
            r0_i = result_discharge.r0_posterior[i]
            T_oc_ann_proj = T_oc[i]

            # Delay: T_delayed(t) = T_ocean(t - delta); zero before the ocean
            # series starts, and the last annual value held flat after it ends
            # (the scenario forcing ends in 2099, as GMST is held flat for SMB)
            T_delayed = np.interp(proj_years, t_oc_ann_proj + delta_i, T_oc_ann_proj,
                                  left=0.0, right=T_oc_ann_proj[-1])
            int_T_delayed = np.cumsum(T_delayed * dt_proj)

            # Un-demean: H = gamma (int T - mean_cal) + r0 (t - t_mean_cal) + H_mean_cal
            H_dyn_i = (gamma_i * (int_T_delayed - result_discharge.int_T_mean_cal)
                       + r0_i * (proj_years - result_discharge.t_mean_cal)
                       + result_discharge.H_mean_cal)
            ens_dyn[i] = H_dyn_i - H_dyn_i[bl_idx]
        out[ssp] = ens_dyn
    return out


def project_greenland_ocean_temperature(T_mon, t_mon, T_off_mon,
                                       temp_monthly, temp_time_monthly,
                                       T_ocean_monthly, time_ocean_monthly,
                                       alpha_draws, beta_draws, AA_draws,
                                       oc_noise, oc_noise_years,
                                       oc_noise_pre, oc_noise_pre_years,
                                       sample_warming_paths, aa_scale=None,
                                       gr_temp_monthly=None, gr_time_monthly=None,
                                       tau_draws=None):
    """Annual ocean temperature (200-500 m, EN4 box) per member for one SSP.

    This is the ocean forcing of project_greenland_discharge (see there for
    the construction and the arguments); T_mon, t_mon and T_off_mon are one
    SSP's entries of proj_monthly_temps, proj_monthly_times and
    proj_monthly_offsets.

    Returns (years, T_oc): calendar years (int) and an
    (n_samples, len(years)) array of annual means in degC, on the baseline
    of T_ocean_monthly.
    """
    n_samples = len(alpha_draws)
    tau_draws = np.zeros(n_samples) if tau_draws is None else np.asarray(tau_draws, float)

    ssp_start_idx = min(np.searchsorted(t_mon, temp_time_monthly[-1]), len(t_mon) - 1)
    ssp_start = t_mon[ssp_start_idx]
    ssp_mask = t_mon >= ssp_start

    # EN4 annual observations before the scenario branch (full years)
    if np.any(np.unique(np.floor(time_ocean_monthly).astype(int), return_counts=True)[1] < 12):
        raise ValueError('EN4 input must hold full calendar years (full_years_only)')
    hist_mask_oc = time_ocean_monthly < ssp_start
    oc_hist_t = time_ocean_monthly[hist_mask_oc]
    oc_hist_T = T_ocean_monthly[hist_mask_oc]
    en4_years = np.unique(np.floor(oc_hist_t).astype(int))
    en4_ann = np.array([oc_hist_T[np.floor(oc_hist_t).astype(int) == y].mean()
                        for y in en4_years])

    # Observed surface T before the branch: Greenland regional T where it
    # exists, GMST x AA (member-dependent) before it
    use_gr = gr_temp_monthly is not None
    if use_gr:
        gr_t = np.asarray(gr_time_monthly, float)
        gr_T = np.asarray(gr_temp_monthly, float)
        obs_mask = gr_t < ssp_start
        obs_t, obs_Ts = gr_t[obs_mask], gr_T[obs_mask]
        pre_mask = temp_time_monthly < gr_t.min()
    else:
        obs_t = obs_Ts = np.array([])
        pre_mask = temp_time_monthly < ssp_start
    pre_t = temp_time_monthly[pre_mask]
    pre_T_gmst = temp_monthly[pre_mask]
    pre_aa_scale = (np.ones_like(pre_t) if aa_scale is None else aa_scale(pre_t))
    proj_t = t_mon[ssp_mask]

    # Calendar-year index of every month of the surface series
    t_surf = np.concatenate([pre_t, obs_t, proj_t])
    yr_floor = np.floor(t_surf).astype(int)
    unique_yrs, yr_idx = np.unique(yr_floor, return_inverse=True)
    n_per_yr = np.bincount(yr_idx)
    is_en4 = np.isin(unique_yrs, en4_years)
    en4_vals = en4_ann[np.searchsorted(en4_years, unique_yrs[is_en4])]
    # Residual noise never goes on an observed EN4 year
    fut = (unique_yrs >= np.floor(ssp_start)) & ~is_en4
    pre = (unique_yrs < np.floor(ssp_start)) & ~is_en4

    T_oc = np.zeros((n_samples, len(unique_yrs)))
    for i in range(n_samples):
        # GMST for member i: median path plus its AR6 warming percentile path
        T_gmst_i = T_mon[ssp_mask] + (T_off_mon[i, ssp_mask] if sample_warming_paths else 0.0)
        Ts_mon = np.concatenate([pre_T_gmst * AA_draws[i] * pre_aa_scale, obs_Ts,
                                 T_gmst_i * AA_draws[i]])
        Ts_ann = np.bincount(yr_idx, weights=Ts_mon) / n_per_yr

        # Low-pass transfer to ocean T; EN4 observations where they exist
        T_oc_i = alpha_draws[i] * ema_filter(Ts_ann, tau_draws[i]) + beta_draws[i]
        T_oc_i[is_en4] = en4_vals
        # Transfer residual: projected years, and (separate path) the
        # other years EN4 does not cover
        T_oc_i[fut] += oc_noise[i, unique_yrs[fut] - oc_noise_years[0]]
        T_oc_i[pre] += oc_noise_pre[i, unique_yrs[pre] - oc_noise_pre_years[0]]
        T_oc[i] = T_oc_i
    return unique_yrs, T_oc
