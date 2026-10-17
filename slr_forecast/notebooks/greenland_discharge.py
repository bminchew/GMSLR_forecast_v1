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
"""

import numpy as np

from component_analysis import fit_ocean_transfer_function, fit_discharge_delay_model


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
                                  rate_window_yrs, n_samples, seed=2026):
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
        fit_ocean_transfer_function output (T_ocean = alpha T_surface + beta).
    result_discharge : SimpleNamespace
        fit_discharge_delay_model output with gamma/r0/delta posteriors.
    T_ocean_ann, T_ocean_years : ndarray
        Annual ocean temperature used in the fit.
    """
    ocean_transfer = fit_ocean_transfer_function(
        T_surface_monthly=gr_temp_monthly, time_surface=gr_time_monthly,
        T_ocean_monthly=T_ocean_monthly, time_ocean=time_ocean_monthly,
        lag_years=0, annual=True)

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
                                gr_time_monthly=None):
    """Discharge ensemble per SSP (metres SLE, rebased to *baseline_year*).

    For each member: draw (gamma, r0, delta) from the discharge posterior and
    (alpha, beta, AA) from their draws; build ocean T from EN4 observations,
    the transfer function applied to surface T in the years EN4 does not
    cover, and the projected branch (member's GMST path x AA through the
    transfer function); add the transfer-function AR(1) residual on the
    years not covered by EN4; delay by delta, integrate, and un-demean with
    the calibration means.

    Surface T outside EN4: if gr_temp_monthly/gr_time_monthly (observed
    Greenland regional T, the series the transfer function is fitted to)
    are given, they are used for every month before the scenario branch
    that EN4 does not cover (before EN4 starts and between its end and the
    first projected month), and GMST x AA only before the regional record.
    If they are None, the pre-EN4 years use GMST x AA and the years between
    the end of EN4 and the scenario branch have no ocean T (as before).

    aa_scale : callable or None
        Multiplicative scale on AA for pre-EN4 GMST x AA years (the
        historical ramp); None means no ramp.

    Returns {ssp: ndarray (n_samples, len(proj_years))}.
    """
    n_samples = len(alpha_draws)
    out = {}
    for ssp in ssps:
        T_mon = proj_monthly_temps[ssp]
        t_mon = proj_monthly_times[ssp]
        T_off_mon = proj_monthly_offsets[ssp]

        ssp_start_idx = min(np.searchsorted(t_mon, temp_time_monthly[-1]), len(t_mon) - 1)
        ssp_start = t_mon[ssp_start_idx]
        ssp_mask = t_mon >= ssp_start

        # Historical ocean T for splicing (EN4 observations)
        hist_mask_oc = time_ocean_monthly < ssp_start
        oc_hist_t = time_ocean_monthly[hist_mask_oc]
        oc_hist_T = T_ocean_monthly[hist_mask_oc]

        # Pre-EN4 ocean T from GMST x AA through the transfer function
        # (only before the regional record when gr_temp_monthly is given)
        en4_start = time_ocean_monthly.min()
        en4_end = time_ocean_monthly.max()
        use_gr = gr_temp_monthly is not None
        if use_gr:
            gr_t = np.asarray(gr_time_monthly, float)
            gr_T = np.asarray(gr_temp_monthly, float)
            fill_mask = (((gr_t < en4_start) | (gr_t > en4_end))
                         & (gr_t < ssp_start))
            fill_t, fill_Ts = gr_t[fill_mask], gr_T[fill_mask]
            pre_en4_mask = temp_time_monthly < min(gr_t.min(), en4_start)
            en4_years = np.unique(np.floor(oc_hist_t).astype(int))
        else:
            fill_t = fill_Ts = np.array([])
            pre_en4_mask = temp_time_monthly < en4_start
        pre_en4_t = temp_time_monthly[pre_en4_mask]
        pre_en4_T_gmst = temp_monthly[pre_en4_mask]
        pre_en4_aa_scale = (np.ones_like(pre_en4_t) if aa_scale is None
                            else aa_scale(pre_en4_t))

        ens_dyn = np.zeros((n_samples, len(proj_years)))
        dt_proj = np.diff(proj_years, prepend=proj_years[0] - 1)
        bl_idx = np.argmin(np.abs(proj_years - baseline_year))

        for i in range(n_samples):
            delta_i = result_discharge.delta_posterior[i]
            gamma_i = result_discharge.gamma_posterior[i]
            r0_i = result_discharge.r0_posterior[i]

            # GMST for member i: median path plus its AR6 warming percentile path
            T_gmst_i = T_mon[ssp_mask] + (T_off_mon[i, ssp_mask] if sample_warming_paths else 0.0)
            T_ocean_proj = alpha_draws[i] * (T_gmst_i * AA_draws[i]) + beta_draws[i]
            T_ocean_pre_en4 = (alpha_draws[i] * (pre_en4_T_gmst * AA_draws[i] * pre_en4_aa_scale)
                               + beta_draws[i])
            T_ocean_fill = alpha_draws[i] * fill_Ts + beta_draws[i]
            t_oc_full = np.concatenate([pre_en4_t, fill_t, oc_hist_t, t_mon[ssp_mask]])
            T_oc_full = np.concatenate([T_ocean_pre_en4, T_ocean_fill, oc_hist_T,
                                        T_ocean_proj])

            # Annualize the full ocean T series
            yr_floor = np.floor(t_oc_full).astype(int)
            unique_yrs = np.unique(yr_floor)
            T_oc_ann_proj = np.array([T_oc_full[yr_floor == yr].mean() for yr in unique_yrs])
            t_oc_ann_proj = unique_yrs.astype(float) + 0.5
            # Transfer-function residual on the projected years (before the delay)
            fut = unique_yrs >= np.floor(ssp_start)
            T_oc_ann_proj[fut] += oc_noise[i, unique_yrs[fut] - oc_noise_years[0]]
            # Same residual on the other non-EN4 years (separate path)
            if use_gr:
                pre = (unique_yrs < np.floor(ssp_start)) & ~np.isin(unique_yrs, en4_years)
            else:
                pre = unique_yrs < np.floor(en4_start)
            T_oc_ann_proj[pre] += oc_noise_pre[i, unique_yrs[pre] - oc_noise_pre_years[0]]

            # Delay: T_delayed(t) = T_ocean(t - delta); zero before ocean data
            T_delayed = np.interp(proj_years, t_oc_ann_proj + delta_i, T_oc_ann_proj,
                                  left=np.nan, right=np.nan)
            T_delayed[np.isnan(T_delayed)] = 0.0
            int_T_delayed = np.cumsum(T_delayed * dt_proj)

            # Un-demean: H = gamma (int T - mean_cal) + r0 (t - t_mean_cal) + H_mean_cal
            H_dyn_i = (gamma_i * (int_T_delayed - result_discharge.int_T_mean_cal)
                       + r0_i * (proj_years - result_discharge.t_mean_cal)
                       + result_discharge.H_mean_cal)
            ens_dyn[i] = H_dyn_i - H_dyn_i[bl_idx]
        out[ssp] = ens_dyn
    return out
