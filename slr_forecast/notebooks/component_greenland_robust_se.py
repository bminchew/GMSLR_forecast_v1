"""
Correlation-aware standard errors for the Greenland discharge delay model.

This does NOT refit anything.  It takes the published estimator exactly as
it is -- demeaned weighted least squares on the cumulative discharge record
with the end-of-record rate constraint, ``component_analysis.
fit_discharge_delay_model`` -- and recomputes only the parameter
covariance.

Why
---
The published covariance is ``(X^T W X)^{-1}`` with ``W = diag(1/sigma^2)``.
That formula is correct only if the observations are independent.  They are
not: the calibration target is a cumulative sum of annual rate errors, so
adjacent values share nearly all of their accumulated error.  Treating them
as independent overstates the effective sample size and yields intervals
that are too narrow -- in a coverage study over synthetic records built
through the real ``bayesian_models._cumulate``, the nominal 90% intervals
contained the truth about 35% of the time.

Weighted least squares with the wrong weights is inefficient but still
unbiased, so the published point estimates stand.  Only the uncertainty
arithmetic needs correcting, which is what this module does:

    Cov(beta) = (X'WX)^{-1} X'W  Omega  W X (X'WX)^{-1}

the sandwich form, with ``Omega`` the true covariance of the fitted
quantities rather than the assumed diagonal.

Omega is built from the actual generative structure of the record.  The
observations are ``y = [M H ; c'H]`` -- the demeaned cumulative series
stacked with the rate-constraint row -- so with ``L = [M ; c']``,

    Omega = L Sigma L'      Sigma = A diag(v) A'

where ``A`` is the signed increment-path operator of the bidirectional
cumulation in ``_cumulate`` and ``v`` the per-step variance increments from
the reported annual rate uncertainties.  Two consequences fall out
automatically: the demeaning is handled (it is in ``M``), and the rate
constraint is correctly recognised as a deterministic function of data
already in the fit (it is in ``c'``) rather than as independent
information.

Error scale
-----------
The reported annual uncertainties are known to understate the true
observational errors, more so for the older part of the record.  The
generalised chi-square of the fit measures this, and ``scale='auto'``
(the default) inflates ``Omega`` by chi2/dof accordingly.  Pass
``scale=1.0`` to report intervals conditional on the formal errors being
correct.

Units follow ``component_analysis``: meters SLE, degC, yr, SLR convention.
"""

import numpy as np

try:
    from slr_forecast import M_TO_MM
except ImportError:
    M_TO_MM = 1000.0

Z90 = 1.6448536269514722          # two-sided 90% normal quantile


def _increment_covariance(yrs, sigma_rate_valid, dt_valid, baseline_window):
    """True covariance of the cumulative record on the fitted subset.

    Mirrors ``bayesian_models._cumulate``: variance accumulates outward
    from the baseline midpoint in both directions, so two observations on
    opposite sides of the anchor share no increments, and two on the same
    side share every increment up to the nearer one.  The anchor itself is
    deterministic (the series is rebased there) and so has zero variance.
    """
    n = len(yrs)
    anchor = int(np.argmin(np.abs(yrs - float(np.mean(baseline_window)))))
    sig_dt_sq = (sigma_rate_valid * dt_valid) ** 2

    A = np.zeros((n, n))
    v = np.zeros(n)
    for i in range(anchor + 1, n):
        v[i] = sig_dt_sq[i]
        A[i, anchor + 1:i + 1] = 1.0
    for i in range(anchor - 1, -1, -1):
        v[i] = sig_dt_sq[i + 1]
        A[i, i:anchor] = -1.0

    return A @ np.diag(v) @ A.T, anchor


def robust_delay_intervals(
    result,
    dyn_years, H_dyn, sigma_dyn,
    T_ocean_ann, T_ocean_years,
    sigma_rate=None, mou_comp=None,
    rate_window_yrs=10,
    rate_constraint_weight=1,
    baseline_window=(1995, 2005),
    scale='auto',
    conf=0.90,
    verbose=True,
):
    """Recompute the delay-model parameter intervals, without refitting.

    Parameters
    ----------
    result : SimpleNamespace
        Output of ``component_analysis.fit_discharge_delay_model``.  Its
        point estimates are used as-is and are never modified.
    dyn_years, H_dyn, sigma_dyn : ndarray
        The same arrays passed to the fit.
    T_ocean_ann, T_ocean_years : ndarray
        The same ocean forcing passed to the fit.
    sigma_rate : ndarray or None
        Reported 1-sigma on the annual discharge rate (m SLE/yr).  Taken
        from ``mou_comp['df']['discharge_rate_sigma']`` when not given.
    mou_comp : dict or None
        Output of ``prepare_mouginot_components``.
    rate_window_yrs, rate_constraint_weight, baseline_window
        Must match the settings used for the fit.
    scale : {'auto', float}
        Variance inflation applied to Omega.  ``'auto'`` estimates it as
        the generalised chi2/dof of the fit, which absorbs the known
        understatement of the formal observational errors.
    conf : float
        Interval level.  Default 0.90.
    verbose : bool

    Returns
    -------
    dict
        Per-delta and mixture-level results, with ``gamma``/``r0`` point
        estimates unchanged from ``result`` and both the published and
        corrected intervals for comparison.
    """
    from scipy import stats

    if sigma_rate is None:
        if mou_comp is None:
            raise ValueError('supply sigma_rate or mou_comp')
        df = mou_comp['df']
        yrs_df = df['decimal_year'].values.astype(float)
        idx = np.searchsorted(yrs_df, np.asarray(dyn_years, dtype=float))
        sigma_rate = df['discharge_rate_sigma'].values[idx]
    sigma_rate = np.asarray(sigma_rate, dtype=float)

    dyn_years = np.asarray(dyn_years, dtype=float)
    H_dyn = np.asarray(H_dyn, dtype=float)
    sigma_dyn = np.asarray(sigma_dyn, dtype=float)
    z = stats.norm.ppf(0.5 + conf / 2.0)

    # Rate-constraint row as a linear functional of H: r_obs = c' H
    rate_mask = dyn_years >= (dyn_years[-1] - rate_window_yrs)
    t_r = dyn_years[rate_mask]
    W_r = 1.0 / sigma_dyn[rate_mask] ** 2
    A_r = np.column_stack([np.ones(len(t_r)), t_r])
    pinv_r = np.linalg.inv(A_r.T @ np.diag(W_r) @ A_r) @ A_r.T @ np.diag(W_r)
    c_full = np.zeros(len(dyn_years))
    c_full[rate_mask] = pinv_r[1]                    # slope row
    sigma_r_obs = float(result.sigma_r_obs)
    t_end = float(dyn_years[-1])

    per_delta = {}
    for delta, fr in result.fit_results.items():
        valid = fr['valid_mask']
        yrs_v = dyn_years[valid]
        n = len(yrs_v)
        sig_v = sigma_dyn[valid]
        dt_v = np.diff(yrs_v, prepend=yrs_v[0] - 1)

        Sigma, anchor = _increment_covariance(
            yrs_v, sigma_rate[valid], dt_v, baseline_window)

        # Design exactly as the published fit builds it
        T_shift = np.interp(dyn_years, T_ocean_years + delta, T_ocean_ann,
                            left=np.nan, right=np.nan)[valid]
        int_T = np.cumsum(T_shift * dt_v)
        X_lvl = np.column_stack([int_T - int_T.mean(), yrs_v - yrs_v.mean()])
        T_at_end = float(np.interp(t_end - delta, T_ocean_years, T_ocean_ann))
        X = np.vstack([X_lvl, [[T_at_end, 1.0]]])

        w = np.concatenate([1.0 / sig_v ** 2,
                            [rate_constraint_weight / sigma_r_obs ** 2]])
        XtWX = X.T @ (w[:, None] * X)
        XtWX_inv = np.linalg.inv(XtWX)

        # Omega = L Sigma L', L = [M ; c'] on the fitted subset
        M = np.eye(n) - np.ones((n, n)) / n
        c_v = c_full[valid]
        L = np.vstack([M, c_v[None, :]])
        Omega = L @ Sigma @ L.T

        # Error-scale calibration from the generalised chi-square
        beta_pub = np.array([fr['gamma'], fr['r0']])
        resid_lvl = fr['H_dm'] - X_lvl @ beta_pub
        Sig_dm = M @ Sigma @ M.T
        Sig_dm_inv = np.linalg.pinv(Sig_dm, rcond=1e-10)
        dof = n - 1 - 2
        chi2 = float(resid_lvl @ Sig_dm_inv @ resid_lvl)
        s2 = chi2 / dof if scale == 'auto' else float(scale)

        bread = XtWX_inv @ X.T @ np.diag(w)
        cov_robust = s2 * (bread @ Omega @ bread.T)

        per_delta[float(delta)] = {
            'gamma': fr['gamma'], 'r0': fr['r0'],
            'se_published': np.sqrt(np.diag(fr['cov'])),
            'se_robust': np.sqrt(np.diag(cov_robust)),
            'cov_robust': cov_robust,
            'chi2_over_dof': chi2 / dof, 'dof': dof, 'scale_applied': s2,
            'p_gof': float(stats.chi2.sf(chi2, dof)),
            'n': n, 'anchor_year': float(yrs_v[anchor]),
        }

    # Mixture over delta, using the published BIC weights
    deltas = np.array(sorted(per_delta.keys()))
    wts = np.asarray(result.bic_weights, dtype=float)
    wts = wts / wts.sum()
    g_mean = sum(w * per_delta[d]['gamma'] for w, d in zip(wts, deltas))
    r_mean = sum(w * per_delta[d]['r0'] for w, d in zip(wts, deltas))
    g_var = sum(w * (per_delta[d]['se_robust'][0] ** 2
                     + (per_delta[d]['gamma'] - g_mean) ** 2)
                for w, d in zip(wts, deltas))
    r_var = sum(w * (per_delta[d]['se_robust'][1] ** 2
                     + (per_delta[d]['r0'] - r_mean) ** 2)
                for w, d in zip(wts, deltas))

    best = per_delta[float(result.delta_best)]
    out = {
        'per_delta': per_delta,
        'delta_best': float(result.delta_best),
        'gamma': float(result.gamma_best), 'r0': float(result.r0_best),
        'se_robust': (float(np.sqrt(g_var)), float(np.sqrt(r_var))),
        'se_published': tuple(best['se_published']),
        'ci_robust': (
            (result.gamma_best - z * np.sqrt(g_var),
             result.gamma_best + z * np.sqrt(g_var)),
            (result.r0_best - z * np.sqrt(r_var),
             result.r0_best + z * np.sqrt(r_var))),
        'chi2_over_dof': best['chi2_over_dof'],
        'p_gof': best['p_gof'],
        'scale_applied': best['scale_applied'],
        'inflation': tuple(np.array(
            [np.sqrt(g_var), np.sqrt(r_var)]) / best['se_published']),
        'conf': conf,
    }

    if verbose:
        g, r = result.gamma_best * M_TO_MM, result.r0_best * M_TO_MM
        gs, rs = out['se_robust'][0] * M_TO_MM, out['se_robust'][1] * M_TO_MM
        gp, rp = (best['se_published'][0] * M_TO_MM,
                  best['se_published'][1] * M_TO_MM)
        pct = int(round(conf * 100))
        print(f'Correlation-aware intervals (delta = {out["delta_best"]:.0f} yr, '
              f'point estimates unchanged)')
        print(f'  goodness of fit: chi2/dof = {best["chi2_over_dof"]:.3f} '
              f'(dof {best["dof"]}, p = {best["p_gof"]:.4f}), '
              f'variance scale applied = {best["scale_applied"]:.3f}')
        print(f'  gamma = {g:.4f} mm/yr/degC')
        print(f'      published {pct}% CI  [{g - z * gp:.4f}, {g + z * gp:.4f}]')
        print(f'      corrected {pct}% CI  [{g - z * gs:.4f}, {g + z * gs:.4f}]'
              f'   (x{out["inflation"][0]:.2f})')
        print(f'  r_0   = {r:.4f} mm/yr')
        print(f'      published {pct}% CI  [{r - z * rp:.4f}, {r + z * rp:.4f}]')
        print(f'      corrected {pct}% CI  [{r - z * rs:.4f}, {r + z * rs:.4f}]'
              f'   (x{out["inflation"][1]:.2f})')

    return out
