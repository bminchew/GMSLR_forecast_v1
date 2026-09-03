"""
Correlation-aware standard errors for cumulative (integrated) component
records.

``robust_level_intervals`` / ``robust_curve_band`` cover the level-space
Bayesian model (``bayesian_models.fit_bayesian_level``), used to calibrate
glaciers, Antarctic Peninsula, and EAIS. They do NOT refit anything: they
recompute only the parameter covariance for the published linear fit
(``a`` forced to zero), leaving the point estimates (posterior
median/MAP) as reported.

``robust_quadratic_level_fit`` / ``robust_quadratic_curve_band`` cover the
WAIS descriptive quadratic (WLS, no Bayesian model/priors -- WAIS has no
calibrated rate-temperature relationship), using the same
correlation-aware covariance machinery.

Why
---
``fit_bayesian_level``'s likelihood is a per-point independent Gaussian sum
(``bayesian_models._level_log_likelihood``), even though ``H_obs`` is a
cumulative (integrated) record: adjacent values share nearly all of their
accumulated error.  This is the same issue documented for the Greenland
discharge fit in ``component_greenland_robust_se.py`` (that fix does not
cover this estimator).

Because the MCMC posterior for these components is materially shaped by
its priors (particularly the H0 prior, centered on the record's first
observation -- see ``fit_bayesian_level``'s ``H0_prior_mean = H_obs[0]``),
a naive frequentist comparison that ignores the priors is not
apples-to-apples: it lets H0 float freely, which is degenerate with `b`
(observed correlation ~0.8 for glaciers) and manufactures an artificially
wide "naive" reference baseline. To get a fair comparison we build a
MAP-equivalent generalized least squares estimator that includes the
actual Gaussian priors on b, c, H0 as pseudo-observations (independent of
the data and of each other), then apply the sandwich correction

    Cov(beta) = (X'WX)^{-1} X'W Omega W X (X'WX)^{-1}

with Omega block-diagonal: the correlation-aware covariance of the
cumulative data block (see ``_anchor_covariance``), and the (uncorrelated)
prior variances for the pseudo-observation block.

Validity
--------
This is a linearized (Gaussian) approximation to a model whose actual
priors are HalfNormal(b >= 0) and Exponential(a >= 0) with a hard
non-negativity bound. It is valid when the posterior sits comfortably
away from that boundary -- checked automatically via
``map_vs_posterior_ratio`` in the returned dict; treat results with a
ratio outside roughly [0.8, 1.25] as unreliable (the boundary constraint
is doing real work and a closed-form Gaussian approximation cannot
capture it -- EAIS's `b` is an example: its posterior median sits at
less than one posterior-sigma from zero).

Units follow ``component_analysis``: meters SLE, degC, yr, SLR convention.
"""

import numpy as np

try:
    from slr_forecast import M_TO_MM
except ImportError:
    M_TO_MM = 1000.0

Z90 = 1.6448536269514722          # two-sided 90% normal quantile


def _anchor_covariance(years, sigma_obs, baseline_year):
    """True covariance of a cumulative record's rebased values.

    Variance accumulates outward from the baseline (rebase) year in both
    directions: two observations on the same side of the anchor share
    every increment up to the nearer one; two on opposite sides share
    none. Built directly from the reported (rebased) marginal sigma --
    algebraically equivalent to ``component_greenland_robust_se.
    _increment_covariance`` (which instead reconstructs it from raw
    per-step rate uncertainties), since for a cumulative sum of
    independent increments, sigma_rebased(t)^2 = |Var(H_t) - Var(H_bl)|
    is exactly this covariance's diagonal, and shared history along the
    cumulation determines the off-diagonal.

    Points with sigma_obs == 0 (the rebase anchor itself, a
    normalization rather than a measurement) must be excluded before
    calling this -- never regularize that row with jitter.
    """
    if np.any(sigma_obs <= 0):
        raise ValueError('drop zero/negative-sigma rows before calling '
                          '_anchor_covariance (deterministic anchor point)')
    n = len(years)
    anchor = int(np.argmin(np.abs(years - baseline_year)))
    Sigma = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            k = min(i, j) if (i >= anchor and j >= anchor) else max(i, j)
            Sigma[i, j] = sigma_obs[k] ** 2
    return Sigma, anchor


def _psd_sqrt_diag(cov):
    """sqrt(diag(cov)), clipping numerically-negative eigenvalues to zero.

    The sandwich covariance is mathematically PSD; near-collinear design
    columns (e.g. `c` vs the H0 pseudo-row) can leave tiny negative
    eigenvalues from floating-point cancellation. Clip rather than
    silently propagate NaN from sqrt of a negative diagonal entry.
    """
    eigval, eigvec = np.linalg.eigh(cov)
    eigval_clipped = np.clip(eigval, 0.0, None)
    cov_psd = (eigvec * eigval_clipped) @ eigvec.T
    return np.sqrt(np.diag(cov_psd)), cov_psd


def robust_level_intervals(
    years, H_obs, sigma_obs,
    I1_obs, I0_obs,
    prior_scale_b, prior_c_mean, prior_c_sigma, prior_H0_sigma,
    baseline_year,
    posterior_samples=None, H0_posterior=None,
    conf=0.90,
    verbose=True,
):
    """Recompute (b, c) parameter intervals for a linear level-space fit
    (``a`` forced to zero), without refitting.

    Parameters
    ----------
    years, H_obs, sigma_obs : ndarray
        The same arrays passed to ``fit_bayesian_level`` (meters, decimal
        year). Rows with ``sigma_obs == 0`` (the deterministic rebase
        anchor) are dropped automatically.
    I1_obs, I0_obs : ndarray
        Design vectors from ``build_level_design_vectors`` at ``years``.
    prior_scale_b, prior_c_mean, prior_c_sigma, prior_H0_sigma : float
        Same prior hyperparameters passed to ``fit_bayesian_level``.
        ``H0``'s prior mean is fixed by that function to ``H_obs[0]``
        (the first, pre-filtering observation) -- pass the untouched
        array.
    baseline_year : float
        Year the record was rebased to (``BASELINE_YEAR``).
    posterior_samples, H0_posterior : ndarray or None
        Published MCMC output, for the point estimate and the
        MAP-vs-posterior validity check. Optional but strongly
        recommended.
    conf : float
        Interval level. Default 0.90.

    Returns
    -------
    dict with beta_map (b, c, H0), se_naive, se_robust, ci_robust,
    map_vs_posterior_ratio (b, c), and inflation (se_robust/se_published
    when posterior_samples given).
    """
    from scipy import stats

    years = np.asarray(years, dtype=float)
    H_obs_full = np.asarray(H_obs, dtype=float)
    sigma_obs_full = np.asarray(sigma_obs, dtype=float)
    H0_prior_mean = H_obs_full[0]     # matches fit_bayesian_level exactly

    keep = sigma_obs_full > 0
    yrs_k = years[keep]
    H_k = H_obs_full[keep]
    sig_k = sigma_obs_full[keep]
    I1_k = np.asarray(I1_obs, dtype=float)[keep]
    I0_k = np.asarray(I0_obs, dtype=float)[keep]
    n = len(yrs_k)

    X_data = np.column_stack([I1_k, I0_k, np.ones(n)])
    w_data = 1.0 / sig_k ** 2

    # Prior pseudo-rows -- Gaussian approximation of HalfNormal(b) valid
    # away from the b=0 boundary (see module docstring / validity check).
    X_prior = np.eye(3)
    y_prior = np.array([0.0, prior_c_mean, H0_prior_mean])
    w_prior = np.array([1.0 / prior_scale_b ** 2,
                         1.0 / prior_c_sigma ** 2,
                         1.0 / prior_H0_sigma ** 2])

    X = np.vstack([X_data, X_prior])
    y = np.concatenate([H_k, y_prior])
    w = np.concatenate([w_data, w_prior])
    W = np.diag(w)

    XtWX_inv = np.linalg.inv(X.T @ W @ X)
    beta_map = XtWX_inv @ (X.T @ W @ y)
    se_naive, _ = _psd_sqrt_diag(XtWX_inv)

    Sigma_data, anchor = _anchor_covariance(yrs_k, sig_k, baseline_year)
    Sigma_full = np.zeros((n + 3, n + 3))
    Sigma_full[:n, :n] = Sigma_data
    Sigma_full[n:, n:] = np.diag(1.0 / w_prior)

    bread = XtWX_inv @ X.T @ W
    cov_robust = bread @ Sigma_full @ bread.T
    se_robust, cov_robust_psd = _psd_sqrt_diag(cov_robust)

    z = stats.norm.ppf(0.5 + conf / 2.0)
    out = {
        'beta_map': {'b': beta_map[0], 'c': beta_map[1], 'H0': beta_map[2]},
        'beta_map_vec': beta_map,               # (b, c, H0), for curve propagation
        'cov_robust': cov_robust_psd,           # full 3x3 (b, c, H0) covariance
        'se_naive': {'b': se_naive[0], 'c': se_naive[1], 'H0': se_naive[2]},
        'se_robust': {'b': se_robust[0], 'c': se_robust[1], 'H0': se_robust[2]},
        'ci_robust': {
            'b': (beta_map[0] - z * se_robust[0], beta_map[0] + z * se_robust[0]),
            'c': (beta_map[1] - z * se_robust[1], beta_map[1] + z * se_robust[1]),
        },
        'inflation_vs_naive': {'b': se_robust[0] / se_naive[0],
                                'c': se_robust[1] / se_naive[1]},
        'anchor_year': float(yrs_k[anchor]),
        'n': n,
        'conf': conf,
    }

    if posterior_samples is not None:
        post_med_b = float(np.median(posterior_samples[:, 1]))
        post_med_c = float(np.median(posterior_samples[:, 2]))
        post_std_b = float(np.std(posterior_samples[:, 1]))
        post_std_c = float(np.std(posterior_samples[:, 2]))
        out['posterior_median'] = {'b': post_med_b, 'c': post_med_c}
        out['posterior_std'] = {'b': post_std_b, 'c': post_std_c}
        out['map_vs_posterior_ratio'] = {
            'b': beta_map[0] / post_med_b if post_med_b else np.nan,
            'c': beta_map[1] / post_med_c if post_med_c else np.nan,
        }
        out['inflation_vs_published'] = {
            'b': se_robust[0] / post_std_b,
            'c': se_robust[1] / post_std_c,
        }

    if verbose:
        b, c = beta_map[0] * M_TO_MM, beta_map[1] * M_TO_MM
        bn, cn = se_naive[0] * M_TO_MM, se_naive[1] * M_TO_MM
        br, cr = se_robust[0] * M_TO_MM, se_robust[1] * M_TO_MM
        pct = int(round(conf * 100))
        print(f'Correlation-aware level-space intervals (n={n}, anchor year='
              f'{out["anchor_year"]:.1f}, point estimate = MAP, not refit)')
        if 'map_vs_posterior_ratio' in out:
            rb = out['map_vs_posterior_ratio']['b']
            rc = out['map_vs_posterior_ratio']['c']
            flag_b = '' if 0.8 <= rb <= 1.25 else '  ** MAP/posterior mismatch -- Gaussian approx unreliable (boundary effect?) **'
            flag_c = '' if 0.8 <= rc <= 1.25 else '  ** MAP/posterior mismatch -- Gaussian approx unreliable **'
            print(f'  MAP/posterior-median ratio: b={rb:.2f}{flag_b}  c={rc:.2f}{flag_c}')
        print(f'  b = {b:.4f} mm/yr/degC')
        print(f'      naive (independent-error) {pct}% halfwidth  = {z*bn:.4f} mm')
        print(f'      robust (correlation-aware) {pct}% halfwidth = {z*br:.4f} mm'
              f'   (x{out["inflation_vs_naive"]["b"]:.2f} vs naive)')
        if 'inflation_vs_published' in out:
            print(f'      published MCMC std = {out["posterior_std"]["b"]*M_TO_MM:.4f} mm'
                  f'   (robust is x{out["inflation_vs_published"]["b"]:.2f} of published)')
        print(f'  c = {c:.4f} mm/yr')
        print(f'      naive (independent-error) {pct}% halfwidth  = {z*cn:.4f} mm')
        print(f'      robust (correlation-aware) {pct}% halfwidth = {z*cr:.4f} mm'
              f'   (x{out["inflation_vs_naive"]["c"]:.2f} vs naive)')
        if 'inflation_vs_published' in out:
            print(f'      published MCMC std = {out["posterior_std"]["c"]*M_TO_MM:.4f} mm'
                  f'   (robust is x{out["inflation_vs_published"]["c"]:.2f} of published)')

    return out


def robust_quadratic_level_fit(years, H_obs, sigma_obs, baseline_year, conf=0.90):
    """WLS quadratic-in-level fit for WAIS, with correlation-aware CI.

    WAIS has no calibrated rate-temperature model (see main text); this is
    a purely descriptive quadratic fit to the cumulative record, with a
    free intercept H0 (offset at ``baseline_year``, not forced to zero):

        H(t) = 0.5*m*(t - t_bl)^2 + c0*(t - t_bl) + H0

    This matches how the level-space Bayesian models handle the same
    baseline year for glaciers/Peninsula/EAIS (``fit_bayesian_level``
    always carries a free H0), rather than treating H(t_bl) as
    deterministically known -- H0 has its own estimation uncertainty
    here too, so the 90% band does not collapse to zero width at
    ``baseline_year``.

    Point estimate is ordinary WLS with per-point diagonal weights
    (1/sigma_obs**2) -- standard practice, and adequate for the point
    estimate since it is unbiased regardless of the error correlation
    structure. The CI is not: ``H_obs`` is a cumulative (integrated)
    record, so adjacent points share nearly all of their accumulated
    error, and naive diagonal weights understate the parameter
    uncertainty (see module docstring / ``feedback_cumulative_covariance``
    memory). We instead build the sandwich covariance

        Cov(beta) = (X'WX)^{-1} X'W Sigma W X (X'WX)^{-1}

    with ``Sigma`` the true correlation-aware covariance of the rebased
    cumulative record (``_anchor_covariance``) -- i.e. the covariance
    that integrating (assumed independent) annual rates into level space
    actually produces. This mirrors ``robust_level_intervals`` above but
    without the Bayesian priors (plain WLS, not MAP), matching how
    glaciers/Peninsula/EAIS handle the same coherent-error issue.

    Parameters
    ----------
    years, H_obs, sigma_obs : ndarray
        Rebased cumulative record (meters SLE, decimal year).
    baseline_year : float
        Year the record was rebased to (``BASELINE_YEAR``).
    conf : float
        Interval level. Default 0.90.

    Returns
    -------
    dict with beta_map (m, c0, H0), cov_robust (3x3), se_naive, se_robust,
    ci_robust, inflation_vs_naive, anchor_year, n, conf.
    """
    from scipy import stats

    years = np.asarray(years, dtype=float)
    H_obs = np.asarray(H_obs, dtype=float)
    sigma_obs = np.asarray(sigma_obs, dtype=float)

    keep = sigma_obs > 0
    yrs_k = years[keep]
    H_k = H_obs[keep]
    sig_k = sigma_obs[keep]
    n = len(yrs_k)

    X = np.column_stack([0.5 * (yrs_k - baseline_year) ** 2,
                          yrs_k - baseline_year,
                          np.ones(n)])
    w = 1.0 / sig_k ** 2
    W = np.diag(w)

    XtWX_inv = np.linalg.inv(X.T @ W @ X)
    beta_map = XtWX_inv @ (X.T @ W @ H_k)
    se_naive, _ = _psd_sqrt_diag(XtWX_inv)

    Sigma_data, anchor = _anchor_covariance(yrs_k, sig_k, baseline_year)
    bread = XtWX_inv @ X.T @ W
    cov_robust = bread @ Sigma_data @ bread.T
    se_robust, cov_robust_psd = _psd_sqrt_diag(cov_robust)

    z = stats.norm.ppf(0.5 + conf / 2.0)
    out = {
        'beta_map': {'m': beta_map[0], 'c0': beta_map[1], 'H0': beta_map[2]},
        'beta_map_vec': beta_map,               # (m, c0, H0), for curve propagation
        'cov_robust': cov_robust_psd,            # full 3x3 (m, c0, H0) covariance
        'se_naive': {'m': se_naive[0], 'c0': se_naive[1], 'H0': se_naive[2]},
        'se_robust': {'m': se_robust[0], 'c0': se_robust[1], 'H0': se_robust[2]},
        'ci_robust': {
            'm': (beta_map[0] - z * se_robust[0], beta_map[0] + z * se_robust[0]),
            'c0': (beta_map[1] - z * se_robust[1], beta_map[1] + z * se_robust[1]),
            'H0': (beta_map[2] - z * se_robust[2], beta_map[2] + z * se_robust[2]),
        },
        'inflation_vs_naive': {'m': se_robust[0] / se_naive[0],
                                'c0': se_robust[1] / se_naive[1],
                                'H0': se_robust[2] / se_naive[2]},
        'anchor_year': float(yrs_k[anchor]),
        'baseline_year': float(baseline_year),
        'n': n,
        'conf': conf,
    }
    return out


def robust_quadratic_curve_band(out, t_grid, conf=0.90):
    """Corrected uncertainty band for H(t) = 0.5*m*(t-t_bl)^2 + c0*(t-t_bl) + H0.

    Propagates the full (m, c0, H0) robust covariance from
    ``robust_quadratic_level_fit`` through the quadratic design at
    arbitrary evaluation points via the delta method.

    Parameters
    ----------
    out : dict
        Return value of ``robust_quadratic_level_fit``.
    t_grid : ndarray
        Evaluation points (years).
    conf : float
        Interval level. Default 0.90.

    Returns
    -------
    H_fit, H_lo, H_hi : ndarray
    """
    from scipy import stats
    z = stats.norm.ppf(0.5 + conf / 2.0)
    beta = out['beta_map_vec']
    cov = out['cov_robust']
    t_grid = np.asarray(t_grid, dtype=float)
    bl = out['baseline_year']
    design = np.column_stack([0.5 * (t_grid - bl) ** 2, t_grid - bl, np.ones_like(t_grid)])
    H_fit = design @ beta
    var = np.einsum('ij,jk,ik->i', design, cov, design)
    se = np.sqrt(np.clip(var, 0.0, None))
    return H_fit, H_fit - z * se, H_fit + z * se


def robust_curve_band(out, I1, I0, conf=0.90):
    """Corrected uncertainty band for H(t) = b*I1(t) + c*I0(t) + H0.

    Propagates the full (b, c, H0) robust covariance from
    ``robust_level_intervals`` through the design vectors at arbitrary
    evaluation points via the delta method, for use as the plotted 90%
    band in place of the (too-narrow) independent-error MCMC posterior
    spread. The point estimates and central curve are unchanged --
    only the band width uses the corrected covariance.

    Parameters
    ----------
    out : dict
        Return value of ``robust_level_intervals`` (needs ``beta_map_vec``
        and ``cov_robust``).
    I1, I0 : ndarray
        Design vectors at the evaluation points (e.g. the plotting grid).
    conf : float
        Interval level. Default 0.90.

    Returns
    -------
    H_fit, H_lo, H_hi : ndarray
    """
    from scipy import stats
    z = stats.norm.ppf(0.5 + conf / 2.0)
    beta = out['beta_map_vec']
    cov = out['cov_robust']
    design = np.column_stack([I1, I0, np.ones_like(I1)])
    H_fit = design @ beta
    var = np.einsum('ij,jk,ik->i', design, cov, design)
    se = np.sqrt(np.clip(var, 0.0, None))
    return H_fit, H_fit - z * se, H_fit + z * se
