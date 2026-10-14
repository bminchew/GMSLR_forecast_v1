"""Greenland discharge: calibration and projection, as functions.

Used by component_greenland.ipynb for the production fit and projections
(§2b, §4) and for the sensitivity tests in §6, which repeat both with an
alternative Arctic amplification factor or ocean-temperature layer. Keeping
one implementation means the sensitivity tests run exactly the production
code with one input changed.

The code was moved here from the notebook cells unchanged; the production
outputs are identical to those of the inline version.
"""

import numpy as np

from component_analysis import fit_ocean_transfer_function, fit_discharge_delay_model
from component_greenland_robust_se import robust_delay_intervals


def annualize(t_monthly, x_monthly):
    """Calendar-year means; returns (values, mid-year times)."""
    yr = np.floor(t_monthly).astype(int)
    uy = np.unique(yr)
    return np.array([x_monthly[yr == y].mean() for y in uy]), uy.astype(float) + 0.5


def calibrate_greenland_discharge(gr_temp_monthly, gr_time_monthly,
                                  T_ocean_monthly, time_ocean_monthly,
                                  mou_comp, delta_candidates,
                                  rate_window_yrs, rate_constraint_weight,
                                  n_samples, seed=2026, verbose=True):
    """Ocean transfer function, delay-model fit and robust posterior draws.

    Returns
    -------
    ocean_transfer : dict
        fit_ocean_transfer_function output (T_ocean = alpha T_surface + beta).
    result_discharge : DischargeDelayResult
        fit_discharge_delay_model output with gamma/r0/delta posteriors
        redrawn from the correlation-aware covariance (robust_delay_intervals),
        preserving the per-delta point estimates and BIC weights.
    robust_discharge : dict
        robust_delay_intervals output.
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
        rate_constraint_weight=rate_constraint_weight,
        n_samples=n_samples, seed=seed)

    # Correlation-aware parameter covariance: the calibration target is a
    # cumulative record, so the least-squares covariance is too narrow.
    # Point estimates are unchanged; only the posterior spread widens.
    robust_discharge = robust_delay_intervals(
        result_discharge,
        mou_comp['time_dyn'], mou_comp['H_dyn'], mou_comp['sigma_dyn'],
        T_ocean_ann, T_ocean_years, mou_comp=mou_comp,
        rate_window_yrs=rate_window_yrs,
        rate_constraint_weight=rate_constraint_weight,
        scale='auto', verbose=verbose)

    # Redraw the posterior from the corrected covariance, preserving the
    # per-delta point estimates and the BIC mixture weights.
    rng = np.random.default_rng(seed)
    w = np.asarray(result_discharge.bic_weights, float)
    w = w / w.sum()
    deltas = np.asarray(result_discharge.DELTA_CANDIDATES, float)
    n_per = np.round(n_samples * w).astype(int)
    n_per[np.argmax(w)] += n_samples - n_per.sum()
    g = np.empty(n_samples); r = np.empty(n_samples); d = np.empty(n_samples)
    i0 = 0
    for dl, nd in zip(deltas, n_per):
        if nd <= 0:
            continue
        pd_ = robust_discharge['per_delta'][float(dl)]
        draws = rng.multivariate_normal([pd_['gamma'], pd_['r0']], pd_['cov_robust'], size=nd)
        g[i0:i0 + nd] = draws[:, 0]
        r[i0:i0 + nd] = draws[:, 1]
        d[i0:i0 + nd] = dl
        i0 += nd
    sh = rng.permutation(n_samples)
    result_discharge.gamma_posterior = g[sh]
    result_discharge.r0_posterior = r[sh]
    result_discharge.delta_posterior = d[sh]
    return ocean_transfer, result_discharge, robust_discharge, T_ocean_ann, T_ocean_years


def project_greenland_discharge(ssps, proj_years, proj_monthly_temps,
                                proj_monthly_times, proj_monthly_offsets,
                                temp_monthly, temp_time_monthly,
                                T_ocean_monthly, time_ocean_monthly,
                                result_discharge, alpha_draws, beta_draws,
                                AA_draws, oc_noise, oc_noise_years,
                                oc_noise_pre, oc_noise_pre_years,
                                baseline_year, sample_warming_paths,
                                aa_scale=None):
    """Discharge ensemble per SSP (metres SLE, rebased to *baseline_year*).

    For each member: draw (gamma, r0, delta) from the discharge posterior and
    (alpha, beta, AA) from their draws; build ocean T from the pre-EN4 proxy
    (GMST x AA through the transfer function), EN4 observations, and the
    projected branch (member's GMST path x AA through the transfer function);
    add the transfer-function AR(1) residual on proxy and projected years;
    delay by delta, integrate, and un-demean with the calibration means.

    aa_scale : callable or None
        Multiplicative scale on AA for pre-EN4 years (the historical ramp);
        None means no ramp.

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
        en4_start = time_ocean_monthly.min()
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
            t_oc_full = np.concatenate([pre_en4_t, oc_hist_t, t_mon[ssp_mask]])
            T_oc_full = np.concatenate([T_ocean_pre_en4, oc_hist_T, T_ocean_proj])

            # Annualize the full ocean T series
            yr_floor = np.floor(t_oc_full).astype(int)
            unique_yrs = np.unique(yr_floor)
            T_oc_ann_proj = np.array([T_oc_full[yr_floor == yr].mean() for yr in unique_yrs])
            t_oc_ann_proj = unique_yrs.astype(float) + 0.5
            # Transfer-function residual on the projected years (before the delay)
            fut = unique_yrs >= np.floor(ssp_start)
            T_oc_ann_proj[fut] += oc_noise[i, unique_yrs[fut] - oc_noise_years[0]]
            # Same residual on the pre-EN4 proxy years (separate path)
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
