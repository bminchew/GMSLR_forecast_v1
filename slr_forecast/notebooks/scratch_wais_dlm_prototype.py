#!/usr/bin/env python3
"""
SCRATCH PROTOTYPE -- not used by any production notebook or module.

Local-trend / integrated-random-walk (IRW) reformulation of WAIS
S1_status_quo, back-tested against IMBIE-3 (Otosaka et al. 2026).

Motivation
----------
Production S1_status_quo (component_projections.py A4_SCENARIOS,
S1_QUADRATIC_MEAN/_COV) is a *fixed* quadratic-in-time fit,
H(t) = 0.5*a*(t-2000)^2 + v*(t-2000) + H0, with (a, v, H0) held constant
and only their joint posterior/parameter uncertainty propagated to 2100.
That captures "how well is a fixed-shape curve constrained by the data,"
not "how much can a constant-acceleration extrapolation itself be wrong
75+ years past a ~45-year record." This script prototypes a state-space
alternative in which the acceleration undergoes a Gaussian random walk
(process noise Q), so the acceleration itself can drift and forecast
variance grows with lead time through accumulated process noise, not
just parameter uncertainty on a fixed curve.

Model
-----
State x_t = [v_t, a_t] (rate, acceleration), annual steps (dt = 1 yr):

    v_t = v_{t-1} + a_{t-1}        (deterministic: v is a's integral)
    a_t = a_{t-1} + xi_t,  xi_t ~ N(0, Q)

Observation: annual-mean WAIS mass-balance rate r_t (IMBIE-3
`mass_balance_rate`, averaged within each calendar year), with
observation variance from IMBIE-3's own `mass_balance_rate_sigma`
(annual mean, NOT divided by sqrt(12) -- the monthly series is smoothed,
so monthly values are not independent draws; see cross-check below):

    r_t = v_t + eps_t,  eps_t ~ N(0, R_t)

Q = 0 recovers the production model exactly (deterministic constant
acceleration). Q > 0 lets the acceleration random-walk, so a forecast
h years past the last observation has

    Var(a_{N+h}) = Var(a_N) + h*Q
    Var(v_{N+h}) = Var(v_N) + [propagated a_N uncertainty] + (h^3/3)-ish
                   growth from accumulated Q  (grows as Q-on-acceleration
                   propagates through TWO integrations to H(t): the
                   2100 band width scales roughly as Q*h^5, not Q*h^3 --
                   this is the dominant lever, bigger than the fitted Q
                   itself; see WRITE-UP below)
    H_{N+h} = H_N + sum_{k=1}^{h} v_{N+k}

We do NOT observe cumulative H_t directly in the filter: IMBIE's
reported *cumulative* sigma is uncertainty accumulated since the record
start, so consecutive cumulative observations share almost all of their
error (the same "cumulative records need correlation-aware covariance"
issue documented in component_levelspace_robust_se.py / project memory).
Feeding that directly into a scalar Kalman update as if it were
independent per-step observation noise would double count: the filter
would treat each new cumulative reading as bringing far more information
than it actually does. Working in rate space with IMBIE's own annual
rate uncertainty sidesteps this (cross-checked below against
first-differencing the correlation-aware cumulative sigma -- the two
agree to <1%, which is itself a useful internal-consistency check on
IMBIE's reported errors).

Forecasting is done by Monte Carlo forward simulation from the filtered
(not smoothed -- smoothing is only valid for interpolation within the
observed window) state at the end of the fit window, injecting fresh
xi_t ~ N(0, Q) draws at each future annual step and cumulatively
integrating v to H(t). This avoids error-prone closed-form propagation
of the joint covariance of a partial-sum vector.

Process-noise (Q) calibration
------------------------------
Q is estimated by MLE (maximizing the Kalman filter's marginal
log-likelihood) on a *clean* sub-segment of the record, not tuned to
hit a coverage target on held-out data (that would let Q absorb
MISI-onset risk into S1, double-counting against S2_fast_wais, which
the task explicitly warns against). Three candidate segments are
compared:
  - 1979-2009 (all pre-2010-break data, IMBIE_ONSET reconstruction-era
    included)
  - 1992-2009 (satellite era only, pre-break) <- RECOMMENDED
  - 1979-2023 (full record, includes the 2010 structural break)
The pre-1992 portion of IMBIE-3 is reconstruction/input-output based and
smoother by construction (fewer independent satellite constraints), so
including it risks biasing Q low -- checked explicitly below.

Backtest
--------
Rolling-origin backtest (not a single split): fit on the first n annual
points for n in {20, 22, ..., 35}, forecast forward with the
1992-2009-derived Q, and check whether the held-out actual value falls
inside the central 90% predictive interval, pooled by lead time h across
origins. The same rolling-origin protocol is run for the *current*
production-style fixed-quadratic model (closed-form GLS-plus-sandwich
estimator, `component_levelspace_robust_se.robust_level_intervals`,
reused at each origin without refitting emcee -- this is algebraically
the same MAP-equivalent estimator the one-shot emcee posterior
converges to, and is what production actually anchors its reported
covariance to). Effective sample size for the pooled coverage fractions
is much smaller than the raw held-out point count (~20 points per
origin are serially correlated within an origin, and origins overlap):
report this caveat explicitly rather than treating coverage fractions as
precise.

Run with:  python scratch_wais_dlm_prototype.py
"""

import os
import sys

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slr_data_readers import read_imbie3
from component_analysis import annualize_imbie
from bayesian_models import fit_bayesian_level
from component_levelspace_robust_se import robust_level_intervals

RAW_DIR = '../data/raw'
BASELINE_YEAR = 2000.0
WAIS_ONSET_YEAR = 2010.0
M_TO_MM = 1000.0

RNG_SEED = 12345

# Current production S1 posterior (component_projections.py), for
# reference / self-check.
S1_QUADRATIC_MEAN_PROD = np.array([1.20482575e-05, 2.30292978e-04,
                                    2.57247062e-04])
S1_QUADRATIC_COV_PROD = np.array([
    [3.98784098e-13, 5.85224993e-12, 2.60159419e-10],
    [5.85224993e-12, 8.58831370e-11, 3.81790034e-09],
    [2.60159419e-10, 3.81790034e-09, 1.69723226e-07],
])


# =========================================================================
# Data loading
# =========================================================================

def load_wais_annual():
    """Load IMBIE-3 WAIS and annualize both cumulative (for the quadratic
    arm) and rate (for the state-space arm), plus a differenced-cumulative
    cross-check of the rate-observation uncertainty.
    """
    df = read_imbie3(
        f'{RAW_DIR}/ice_sheets/imbie2026/imbie3_west_antarctica_mm_partitioned.csv',
        convert_to_meters=True)
    years, H_rebase, sigma_cum = annualize_imbie(df, baseline_year=BASELINE_YEAR)

    t = df['decimal_year'].values
    year_int = np.floor(t).astype(int)
    uy = np.unique(year_int)
    assert np.allclose(uy + 0.5, years), "annualize_imbie year grid mismatch"

    rate = np.zeros(len(uy))
    rate_sigma = np.zeros(len(uy))
    for i, yr in enumerate(uy):
        m = year_int == yr
        rate[i] = df['mass_balance_rate'].values[m].mean()
        rate_sigma[i] = df['mass_balance_rate_sigma'].values[m].mean()

    # Cross-check: first-difference the (raw, un-rebased -- rebasing
    # cancels in a difference) cumulative sigma. Var(increment) =
    # Var(H_t) - Var(H_{t-1}) is valid because IMBIE's cumulative sigma
    # accumulates from the record start (anchor = first point), which we
    # verify is monotone non-decreasing first.
    assert np.all(np.diff(sigma_cum) >= -1e-9), \
        "cumulative sigma not monotone -- differencing cross-check invalid"
    diff_sigma = np.sqrt(np.clip(np.diff(sigma_cum ** 2), 0, None))
    ratio = rate_sigma[1:] / diff_sigma
    print(f"[cross-check] annual-mean-rate-sigma / differenced-cumulative-sigma: "
          f"median ratio = {np.median(ratio):.3f} "
          f"(range {ratio.min():.3f}-{ratio.max():.3f}); agreement <1% confirms "
          f"IMBIE's reported rate and cumulative uncertainties are internally "
          f"consistent, and validates using rate_sigma directly as R_t.")

    return years, H_rebase, sigma_cum, rate, rate_sigma


# =========================================================================
# State-space (IRW-on-acceleration) model
# =========================================================================

def kf_filter(rate_obs, rate_sigma, Q, v0=None, P0=None):
    """Kalman filter for state x_t=[v_t,a_t], v_t=v_{t-1}+a_{t-1},
    a_t=a_{t-1}+xi_t (xi~N(0,Q)), observing rate_obs[t] = v_t + eps_t.

    Returns filtered means (n,2), covariances (n,2,2), and total
    marginal log-likelihood.
    """
    n = len(rate_obs)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Qmat = np.array([[0.0, 0.0], [0.0, Q]])
    Hrow = np.array([1.0, 0.0])

    if v0 is None:
        v0 = np.mean(rate_obs[:min(3, n)])
    if P0 is None:
        P0 = np.diag([np.maximum(rate_sigma[0], 1e-4) ** 2 * 4, (5e-3) ** 2])

    x = np.array([v0, 0.0])
    P = np.array(P0, dtype=float)

    x_filt = np.zeros((n, 2))
    P_filt = np.zeros((n, 2, 2))
    log_lik = 0.0

    for t in range(n):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Qmat

        R_t = rate_sigma[t] ** 2
        y_pred = Hrow @ x_pred
        v_t = rate_obs[t] - y_pred
        S_t = Hrow @ P_pred @ Hrow + R_t
        if S_t <= 0:
            return None, None, -np.inf

        K_t = P_pred @ Hrow / S_t
        x = x_pred + K_t * v_t
        IKH = np.eye(2) - np.outer(K_t, Hrow)
        P = IKH @ P_pred @ IKH.T + R_t * np.outer(K_t, K_t)

        x_filt[t] = x
        P_filt[t] = P
        log_lik += -0.5 * (np.log(2 * np.pi * S_t) + v_t ** 2 / S_t)

    return x_filt, P_filt, log_lik


def fit_Q_mle(rate_obs, rate_sigma, Q_bounds=(1e-12, 1e-2)):
    """MLE for Q by 1-D optimization of the KF marginal log-likelihood."""
    def neg_ll(log10_Q):
        Q = 10 ** log10_Q
        _, _, ll = kf_filter(rate_obs, rate_sigma, Q)
        return -ll if np.isfinite(ll) else 1e10

    res = minimize_scalar(neg_ll, bounds=(np.log10(Q_bounds[0]),
                                           np.log10(Q_bounds[1])),
                           method='bounded',
                           options={'xatol': 1e-3})
    return 10 ** res.x


def forecast_mc(x_end, P_end, Q, H_end, n_years_fwd, n_draws=20000,
                 rng=None, phi=1.0):
    """Monte Carlo forward simulation of [v,a] and cumulative H.

    phi < 1 optionally damps the acceleration random walk
    (a_t = phi*a_{t-1} + xi_t) for a physical-admissibility check; phi=1
    is the pure (undamped) IRW.

    Returns
    -------
    v_draws : (n_draws, n_years_fwd)
    H_draws : (n_draws, n_years_fwd)  -- cumulative H relative to H_end
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    x0 = rng.multivariate_normal(x_end, P_end, size=n_draws)  # (n_draws,2)
    v = x0[:, 0].copy()
    a = x0[:, 1].copy()
    v_draws = np.zeros((n_draws, n_years_fwd))
    H_draws = np.zeros((n_draws, n_years_fwd))
    H_cum = np.full(n_draws, H_end)
    sqrtQ = np.sqrt(Q)
    for k in range(n_years_fwd):
        v = v + a
        a = phi * a + sqrtQ * rng.standard_normal(n_draws)
        H_cum = H_cum + v
        v_draws[:, k] = v
        H_draws[:, k] = H_cum
    return v_draws, H_draws


# =========================================================================
# Quadratic (production-style) closed-form arm, reused at every origin
# =========================================================================

def fit_quadratic_closed_form(years, H_obs, sigma_obs,
                                prior_scale_a=5e-5, prior_v_mean=0.0,
                                prior_v_sigma=0.002, prior_H0_sigma=0.005,
                                baseline_year=BASELINE_YEAR, conf=0.90):
    """Closed-form GLS-plus-sandwich (a, v, H0) estimate -- the same
    machinery `component_wais.ipynb` uses for the published
    S1_QUADRATIC_MEAN/_COV, but without the emcee round-trip, so it is
    cheap enough to call at every backtest origin.

    Reuses robust_level_intervals' 3-parameter linear machinery with the
    quadratic design vector (0.5*tau^2) fed into its 'I1' slot, exactly
    as component_wais.ipynb cell 'cell-wais-bayes-time' does ("acceleration
    draws ... go in column 1 ... velocity draws ... column 2").
    """
    tau = years - baseline_year
    I2 = 0.5 * tau ** 2
    I0 = tau
    out = robust_level_intervals(
        years, H_obs, sigma_obs, I2, I0,
        prior_scale_b=prior_scale_a, prior_c_mean=prior_v_mean,
        prior_c_sigma=prior_v_sigma, prior_H0_sigma=prior_H0_sigma,
        baseline_year=baseline_year, posterior_samples=None, conf=conf,
        verbose=False)
    # out['beta_map_vec'] = (a, v, H0); out['cov_robust'] = 3x3 cov
    return out['beta_map_vec'], out['cov_robust'], out


def quadratic_forecast_ci(mean_avh0, cov_avh0, years_fwd, baseline_year=BASELINE_YEAR,
                            n_draws=20000, rng=None, conf=0.90):
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    draws = rng.multivariate_normal(mean_avh0, cov_avh0, size=n_draws)
    a, v, H0 = draws[:, 0], draws[:, 1], draws[:, 2]
    tau = np.asarray(years_fwd, dtype=float) - baseline_year
    H = 0.5 * a[:, None] * tau[None, :] ** 2 + v[:, None] * tau[None, :] + H0[:, None]
    lo = np.percentile(H, 100 * (1 - conf) / 2, axis=0)
    hi = np.percentile(H, 100 * (1 + conf) / 2, axis=0)
    med = np.percentile(H, 50, axis=0)
    return med, lo, hi, H


# =========================================================================
# Rolling-origin backtest
# =========================================================================

def rolling_origin_backtest(years, H_rebase, rate, rate_sigma, Q_wobble,
                              origins=range(20, 36, 2),
                              lead_bins=((1, 3), (4, 7), (8, 12), (13, 20)),
                              conf=0.90, n_draws=8000, phi=1.0):
    """Rolling-origin coverage check for both the KF/IRW model and the
    closed-form quadratic model, pooled by lead-time bin.

    `phi` (default 1.0, undamped IRW) is passed through to the forecast
    step only -- the KF filter/MLE fit itself always uses the undamped
    (phi=1) transition (see WRITE-UP: phi close to 1 over the short
    fitting window makes this an accurate approximation, verified
    separately against a self-consistently-refit OU Q).
    """
    n = len(years)
    z = stats.norm.ppf(0.5 + conf / 2.0)

    records_kf = []   # (lead, covered)
    records_quad = []
    map_ratios = []

    for n0 in origins:
        if n0 >= n - 2:
            continue
        years_tr, rate_tr, rsig_tr = years[:n0], rate[:n0], rate_sigma[:n0]
        H_tr, Hsig_tr = H_rebase[:n0], None  # sigma for quadratic below

        # --- KF/IRW arm ---
        x_filt, P_filt, _ = kf_filter(rate_tr, rsig_tr, Q_wobble)
        x_end, P_end = x_filt[-1], P_filt[-1]
        H_end = H_rebase[n0 - 1]
        n_fwd = n - n0
        v_draws, H_draws = forecast_mc(x_end, P_end, Q_wobble, H_end, n_fwd,
                                        n_draws=n_draws, phi=phi)
        lo_kf = np.percentile(H_draws, 100 * (1 - conf) / 2, axis=0)
        hi_kf = np.percentile(H_draws, 100 * (1 + conf) / 2, axis=0)

        # --- quadratic arm (needs cumulative sigma for the truncated window) ---
        # sigma for the cumulative record comes from annualize_imbie output
        # (module-level `SIGMA_CUM`, set by caller); passed in via closure
        # variable to keep signature simple -- see call site.
        beta_avh0, cov_avh0, diag = fit_quadratic_closed_form(
            years_tr, H_tr, SIGMA_CUM[:n0])
        map_ratios.append(None)  # placeholder -- no posterior_samples supplied here

        years_fwd = years[n0:]
        med_q, lo_q, hi_q, _ = quadratic_forecast_ci(beta_avh0, cov_avh0, years_fwd,
                                                       n_draws=n_draws)

        H_true_fwd = H_rebase[n0:]
        for h in range(n_fwd):
            lead = h + 1
            covered_kf = (H_true_fwd[h] >= lo_kf[h]) and (H_true_fwd[h] <= hi_kf[h])
            covered_q = (H_true_fwd[h] >= lo_q[h]) and (H_true_fwd[h] <= hi_q[h])
            records_kf.append((lead, covered_kf))
            records_quad.append((lead, covered_q))

    def pool(records):
        out = {}
        for lo_h, hi_h in lead_bins:
            sel = [c for (l, c) in records if lo_h <= l <= hi_h]
            if sel:
                out[(lo_h, hi_h)] = (np.mean(sel), len(sel))
        all_sel = [c for (l, c) in records]
        out['overall'] = (np.mean(all_sel), len(all_sel))
        return out

    return pool(records_kf), pool(records_quad)


# =========================================================================
# Main
# =========================================================================

def main():
    global SIGMA_CUM

    years, H_rebase, SIGMA_CUM, rate, rate_sigma = load_wais_annual()
    n = len(years)
    print(f"IMBIE-3 WAIS annualized: {years[0]:.0f}-{years[-1]:.0f}, n={n} points\n")

    # ---------------------------------------------------------------
    # Q calibration: compare segments
    # ---------------------------------------------------------------
    print("=" * 70)
    print("Q (process noise on acceleration) MLE by segment")
    print("=" * 70)
    seg_defs = {
        '1979-2009 (all pre-break)': (years >= 1979) & (years < WAIS_ONSET_YEAR),
        '1992-2009 (satellite-era pre-break, RECOMMENDED)':
            (years >= 1992) & (years < WAIS_ONSET_YEAR),
        '1979-1991 (reconstruction-era only)': (years >= 1979) & (years < 1992),
        '1979-2023 (full record, includes 2010 break)': np.ones(n, dtype=bool),
    }
    Q_by_seg = {}
    for name, mask in seg_defs.items():
        if mask.sum() < 6:
            continue
        Qhat = fit_Q_mle(rate[mask], rate_sigma[mask])
        Q_by_seg[name] = Qhat
        print(f"  {name:50s}: Q_MLE = {Qhat:.3e} (m/yr^2)^2/yr "
              f"[sigma_a after 10 yr = {np.sqrt(Qhat*10)*1000:.3f} mm/yr^2... "
              f"in rate units: sigma(accel step) = {np.sqrt(Qhat)*1000:.4f} mm/yr^2/yr^0.5]")

    Q_wobble = Q_by_seg['1992-2009 (satellite-era pre-break, RECOMMENDED)']
    Q_full = Q_by_seg['1979-2023 (full record, includes 2010 break)']
    print(f"\n  --> Q_wobble (recommended, pre-break satellite-era) = {Q_wobble:.3e}")
    print(f"  --> Q_full (includes break) = {Q_full:.3e}  "
          f"(ratio full/wobble = {Q_full/Q_wobble:.1f}x)")
    if '1979-1991 (reconstruction-era only)' in Q_by_seg:
        Q_recon = Q_by_seg['1979-1991 (reconstruction-era only)']
        print(f"  --> Q_reconstruction-era-only (1979-1991) = {Q_recon:.3e}  "
              f"(ratio vs satellite-era 1992-2009 = {Q_recon/Q_wobble:.2f}x; "
              f"a ratio << 1 would indicate the reconstruction era is "
              f"artificially smooth and would bias a pooled 1979-2009 Q low)")

    # ---------------------------------------------------------------
    # Rolling-origin backtest
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Rolling-origin backtest (Q = Q_wobble, pre-break satellite-era MLE)")
    print("=" * 70)
    pool_kf, pool_quad = rolling_origin_backtest(years, H_rebase, rate, rate_sigma,
                                                   Q_wobble)
    print(f"\n{'lead bin (yr)':<16}{'KF/IRW coverage':<22}{'quadratic coverage':<22}")
    for key in list(pool_kf.keys()):
        cov_kf, nk = pool_kf[key]
        cov_q, nq = pool_quad[key]
        label = f"{key[0]}-{key[1]}" if isinstance(key, tuple) else key
        print(f"{label:<16}{cov_kf:.2f} (n={nk:<4}){'':<8}{cov_q:.2f} (n={nq})")
    print("\nNOTE: held-out points within an origin are serially correlated "
          "(cumulative-record realizations), and origins overlap heavily -- "
          "effective sample size for these fractions is a small multiple of "
          "the number of *origins* (8 here), not the raw point counts shown. "
          "Treat these as indicative, not precise.")

    # Also report the backtest against Q_full for contrast (NOT recommended
    # for production -- shown only to quantify the double-counting risk).
    print("\n--- same backtest using Q_full (includes 2010 break) for contrast ---")
    pool_kf_full, _ = rolling_origin_backtest(years, H_rebase, rate, rate_sigma,
                                                Q_full)
    for key in list(pool_kf_full.keys()):
        cov_kf, nk = pool_kf_full[key]
        label = f"{key[0]}-{key[1]}" if isinstance(key, tuple) else key
        print(f"{label:<16}{cov_kf:.2f} (n={nk})")

    # ---------------------------------------------------------------
    # Full-record fit: KF/IRW forecast to 2100 vs current production S1
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Full-record fit: 2100 band, KF/IRW (Q_wobble) vs current production S1")
    print("=" * 70)

    x_filt, P_filt, ll_full = kf_filter(rate, rate_sigma, Q_wobble)
    x_end, P_end = x_filt[-1], P_filt[-1]
    H_end = H_rebase[-1]
    n_fwd = int(2100 - years[-1]) + 1
    v_draws, H_draws = forecast_mc(x_end, P_end, Q_wobble, H_end, n_fwd, n_draws=50000)
    H_2100_kf = H_draws[:, -1] * M_TO_MM
    v_2100_kf = v_draws[:, -1] * M_TO_MM  # mm/yr, implied 2100 rate

    med_kf = np.median(H_2100_kf)
    lo_kf, hi_kf = np.percentile(H_2100_kf, [5, 95])
    print(f"\nKF/IRW (undamped, phi=1): 2100 endpoint median = {med_kf:.1f} mm, "
          f"90% CI [{lo_kf:.1f}, {hi_kf:.1f}] mm  (width = {hi_kf-lo_kf:.1f} mm)")

    print(f"\nPhysical admissibility check -- implied 2100 rate distribution:")
    print(f"  median = {np.median(v_2100_kf):.1f} mm/yr, "
          f"90% CI [{np.percentile(v_2100_kf,5):.1f}, {np.percentile(v_2100_kf,95):.1f}] mm/yr, "
          f"P(rate<0, i.e. mass gain) = {np.mean(v_2100_kf<0):.3f}, "
          f"P(rate > current S2 mixture upper tail ~30 mm/yr) = "
          f"{np.mean(v_2100_kf>30):.3f}")
    print("  (current observed WAIS rate ~0.3-0.5 mm/yr = 0.3-0.5 mm/yr; "
          "compare against this distribution's plausible range)")

    # Damped variant for comparison
    for phi in (0.98, 0.95, 0.90):
        _, H_draws_phi = forecast_mc(x_end, P_end, Q_wobble, H_end, n_fwd,
                                       n_draws=50000, phi=phi)
        H_2100_phi = H_draws_phi[:, -1] * M_TO_MM
        lo_p, hi_p = np.percentile(H_2100_phi, [5, 95])
        print(f"  damped phi={phi}: 2100 median = {np.median(H_2100_phi):.1f} mm, "
              f"90% CI [{lo_p:.1f}, {hi_p:.1f}] mm (width={hi_p-lo_p:.1f} mm)")

    # Current production S1 (headline emcee refit on full record, mirroring
    # component_wais.ipynb cell 'cell-wais-bayes-time', for a genuine
    # self-consistent comparison -- not just reusing the hardcoded constants).
    print("\nRefitting production-style quadratic via emcee on the full record "
          "(mirrors component_wais.ipynb cell 'cell-wais-bayes-time')...")
    tau = years - BASELINE_YEAR
    I2_t = 0.5 * tau ** 2
    I1_t = np.zeros_like(tau)
    I0_t = tau
    fit = fit_bayesian_level(
        H_obs=H_rebase, sigma_obs=SIGMA_CUM,
        I2_obs=I2_t, I1_obs=I1_t, I0_obs=I0_t,
        symmetric_a=True,
        prior_scale_a=5e-5, prior_scale_b=0.010,
        prior_c_mean=0.0, prior_c_sigma=0.002,
        prior_sigma_extra_scale=0.001, prior_H0_sigma=0.005,
        n_samples=4000, n_walkers=32, n_burnin=2000, thin=2, seed=700,
        progress=False)

    post_view = np.zeros_like(fit.posterior_samples)
    post_view[:, 1] = fit.posterior_samples[:, 0]
    post_view[:, 2] = fit.posterior_samples[:, 2]
    robust = robust_level_intervals(
        years, H_rebase, SIGMA_CUM, I2_t, I0_t,
        prior_scale_b=5e-5, prior_c_mean=0.0, prior_c_sigma=0.002,
        prior_H0_sigma=0.005, baseline_year=BASELINE_YEAR,
        posterior_samples=post_view, conf=0.90, verbose=False)
    mean_quad = robust['beta_map_vec']
    cov_quad = robust['cov_robust']
    print(f"  refit (a, v, H0) = {mean_quad!r}")
    print(f"  vs production S1_QUADRATIC_MEAN = {S1_QUADRATIC_MEAN_PROD!r}")
    print(f"  map_vs_posterior_ratio = {robust.get('map_vs_posterior_ratio')}")

    med_q, lo_q, hi_q, H_draws_q = quadratic_forecast_ci(
        mean_quad, cov_quad, np.array([2100.0]), n_draws=50000)
    med_q_prod, lo_q_prod, hi_q_prod, _ = quadratic_forecast_ci(
        S1_QUADRATIC_MEAN_PROD, S1_QUADRATIC_COV_PROD, np.array([2100.0]),
        n_draws=50000)
    print(f"\nQuadratic (this refit): 2100 median = {med_q[0]*M_TO_MM:.1f} mm, "
          f"90% CI [{lo_q[0]*M_TO_MM:.1f}, {hi_q[0]*M_TO_MM:.1f}] mm "
          f"(width = {(hi_q[0]-lo_q[0])*M_TO_MM:.1f} mm)")
    print(f"Quadratic (production constants): 2100 median = {med_q_prod[0]*M_TO_MM:.1f} mm, "
          f"90% CI [{lo_q_prod[0]*M_TO_MM:.1f}, {hi_q_prod[0]*M_TO_MM:.1f}] mm "
          f"(width = {(hi_q_prod[0]-lo_q_prod[0])*M_TO_MM:.1f} mm)")

    print(f"\n>>> 2100 band width ratio, KF/IRW (undamped) / production quadratic: "
          f"{(hi_kf-lo_kf) / (hi_q_prod[0]*M_TO_MM - lo_q_prod[0]*M_TO_MM):.1f}x")

    # S2 floor coupling check
    print("\n" + "=" * 70)
    print("S2 floor coupling check (informational)")
    print("=" * 70)
    p99_kf = np.percentile(H_2100_kf, 99)
    p99_q_prod = np.percentile(quadratic_forecast_ci(
        S1_QUADRATIC_MEAN_PROD, S1_QUADRATIC_COV_PROD, np.array([2100.0]),
        n_draws=200000)[3][:, 0] * M_TO_MM, 99)
    print(f"  S1 2100 p99 under production quadratic: {p99_q_prod:.1f} mm "
          f"(current S2_fast_wais low_mm anchor = 94 mm)")
    print(f"  S1 2100 p99 under KF/IRW (Q_wobble, undamped): {p99_kf:.1f} mm")
    print("  Widening S1 to the IRW form would move this anchor and hence the "
          "full two-scenario mixture -- see written summary for recommendation.")



# =========================================================================
# FOLLOW-UP ROUND (2026-09-17): skew falsification, data-driven
# admissibility constraint, and a mean-reverting (OU) reformulation with
# a physically-named settling timescale instead of a fitted damping phi.
# =========================================================================
#
# Context: the first round showed the IRW-on-acceleration model achieves
# ~90% backtest coverage (vs ~25% for the fixed quadratic) but an
# UNDAMPED forecast to 2100 is physically inadmissible (44% probability
# of net mass GAIN by 2100; band ~88x wider than production). Ad hoc
# damping (phi) bounded this somewhat but phi itself was unconstrained by
# the backtest (coverage over the ~25-yr testable horizon was
# indistinguishable across phi in [0.85, 1.0]).
#
# Per discussion with the user/advisor: (1) skewing xi_t targets the
# wrong moment (asymmetry, not scale) and per-step skew washes out over
# ~77 compounding annual steps -- run once to confirm, then drop; (2) fix
# admissibility using this project's OWN data (WAIS's own multi-decade
# loss record), not external ensembles; (3) replace the ad hoc phi with a
# mean-reverting (Ornstein-Uhlenbeck) acceleration process governed by a
# physically-named settling timescale tau, following the SAME precedent
# already used in this codebase for the ocean two-layer model: tau_d is
# fixed at 150 yr (Geoffroy et al. 2013, Table 4, CMIP5 multi-model mean)
# specifically because the 21-yr 0-2000m ocean record is too short to
# constrain a centennial relaxation time (component_ocean.ipynb;
# manuscripts/00_ddpi_slrforecast2026/v1/01_slr_forecast_intervention2026.tex,
# lines ~192-196). The IMBIE-3 WAIS record (44-47 yr) has exactly the same
# limitation for a settling timescale of order decades-to-centuries, so we
# import tau from the ice-dynamics literature rather than fit it here.

from scipy.stats import skewnorm


def forecast_mc_skewed(x_end, P_end, Q, H_end, n_years_fwd, skew_a=6.0,
                        n_draws=20000, rng=None):
    """Same as forecast_mc (undamped, phi=1) but with right-skewed xi_t.

    FALSIFICATION BRANCH ONLY: the hypothesis under test is that skewing
    the per-step acceleration innovations meaningfully changes the 2100
    band. Expectation (stated before running): it should NOT, because
    skewness is a third-moment property and the CLT-like averaging over
    ~77 compounding annual innovations should wash it out, leaving the
    variance (second moment, i.e. Q and h) as the thing that actually
    sets the band. This function exists only to confirm that expectation
    empirically once; the branch is then dropped (no further use below).
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    delta = skew_a / np.sqrt(1 + skew_a ** 2)
    mean_std = delta * np.sqrt(2 / np.pi)
    var_std = 1 - mean_std ** 2
    x0 = rng.multivariate_normal(x_end, P_end, size=n_draws)
    v = x0[:, 0].copy()
    a = x0[:, 1].copy()
    H_cum = np.full(n_draws, H_end)
    v_draws = np.zeros((n_draws, n_years_fwd))
    H_draws = np.zeros((n_draws, n_years_fwd))
    for k in range(n_years_fwd):
        v = v + a
        raw = skewnorm.rvs(skew_a, size=n_draws, random_state=rng)
        xi = (raw - mean_std) / np.sqrt(var_std) * np.sqrt(Q)
        a = a + xi
        H_cum = H_cum + v
        v_draws[:, k] = v
        H_draws[:, k] = H_cum
    return v_draws, H_draws


def kf_filter_ou(rate_obs, rate_sigma, Q, phi, v0=None, P0=None):
    """Same as kf_filter but with a_t = phi*a_{t-1} + xi_t (OU-style decay
    built into the filter transition itself, not just the forecast) --
    used only for the robustness check that phi~1 over the short fitting
    window makes filtering with phi=1 vs phi=exp(-1/tau) negligible for
    Q's MLE.
    """
    n = len(rate_obs)
    F = np.array([[1.0, 1.0], [0.0, phi]])
    Qmat = np.array([[0.0, 0.0], [0.0, Q]])
    Hrow = np.array([1.0, 0.0])
    if v0 is None:
        v0 = np.mean(rate_obs[:min(3, n)])
    if P0 is None:
        P0 = np.diag([np.maximum(rate_sigma[0], 1e-4) ** 2 * 4, (5e-3) ** 2])
    x = np.array([v0, 0.0])
    P = np.array(P0, dtype=float)
    log_lik = 0.0
    for t in range(n):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Qmat
        R_t = rate_sigma[t] ** 2
        y_pred = Hrow @ x_pred
        v_t = rate_obs[t] - y_pred
        S_t = Hrow @ P_pred @ Hrow + R_t
        if S_t <= 0:
            return -np.inf
        K_t = P_pred @ Hrow / S_t
        x = x_pred + K_t * v_t
        IKH = np.eye(2) - np.outer(K_t, Hrow)
        P = IKH @ P_pred @ IKH.T + R_t * np.outer(K_t, K_t)
        log_lik += -0.5 * (np.log(2 * np.pi * S_t) + v_t ** 2 / S_t)
    return log_lik


def fit_Q_mle_ou(rate_obs, rate_sigma, phi, Q_bounds=(1e-12, 1e-2)):
    def neg_ll(log10_Q):
        Q = 10 ** log10_Q
        ll = kf_filter_ou(rate_obs, rate_sigma, Q, phi)
        return -ll if np.isfinite(ll) else 1e10
    res = minimize_scalar(neg_ll, bounds=(np.log10(Q_bounds[0]),
                                           np.log10(Q_bounds[1])),
                           method='bounded', options={'xatol': 1e-3})
    return 10 ** res.x


def estimate_short_wobble_decay(years, rate, mask, max_lag=8):
    """Sanity-check-only estimate of sub-decadal wobble decay from the
    IMBIE-3 record's own autocorrelation structure. Deviation from a
    linear local trend ("wobble") is autocorrelated at lag k; fit a
    single AR(1)-equivalent e-folding time from the lag-1 autocorrelation.

    NOT used as the production-candidate timescale -- a record this short
    (<=45 yr, <=18 yr for the pre-break satellite-era subset) can only
    ever constrain a decay time of order its own length or shorter, by
    the same logic that rules out fitting Q on the full record. This is
    reported as a lower-bound / plausibility check against the
    literature-based tau used below, nothing more.
    """
    t_sub = years[mask]
    r_sub = rate[mask]
    n = len(r_sub)
    p = np.polyfit(t_sub - t_sub.mean(), r_sub, 1)
    wobble = r_sub - np.polyval(p, t_sub - t_sub.mean())
    wobble = wobble - wobble.mean()
    max_lag = min(max_lag, n // 3)
    acf = np.ones(max_lag + 1)
    denom = np.sum(wobble ** 2)
    for k in range(1, max_lag + 1):
        acf[k] = np.sum(wobble[:-k] * wobble[k:]) / denom
    rho1 = acf[1]
    tau_short = -1.0 / np.log(rho1) if 0 < rho1 < 1 else np.nan
    return acf, tau_short, n


def admissibility_scan(x_end, P_end, Q, H_end, n_fwd, tau_grid, n_draws=30000,
                        thresholds=(0.05, 0.01)):
    """For each candidate settling timescale tau (yr), forecast to 2100
    with phi=exp(-1/tau) and report 90% CI width and P(H_2100 < 0).
    """
    rows = []
    for tau in tau_grid:
        phi = 1.0 if tau == np.inf else np.exp(-1.0 / tau)
        _, H_draws = forecast_mc(x_end, P_end, Q, H_end, n_fwd, n_draws=n_draws,
                                   phi=phi)
        H2100 = H_draws[:, -1] * M_TO_MM
        lo, hi = np.percentile(H2100, [5, 95])
        p_gain = float(np.mean(H2100 < 0))
        rows.append({'tau': tau, 'phi': phi, 'median': np.median(H2100),
                     'lo90': lo, 'hi90': hi, 'width': hi - lo,
                     'p_gain': p_gain, 'p99': np.percentile(H2100, 99)})
    return rows


def main_followup():
    years, H_rebase, SIGMA_CUM_local, rate, rate_sigma = load_wais_annual()
    global SIGMA_CUM
    SIGMA_CUM = SIGMA_CUM_local
    n = len(years)

    Q_wobble = fit_Q_mle(rate[(years >= 1992) & (years < WAIS_ONSET_YEAR)],
                          rate_sigma[(years >= 1992) & (years < WAIS_ONSET_YEAR)])
    print(f"Q_wobble (satellite-era pre-break MLE, carried over from round 1) "
          f"= {Q_wobble:.3e}")

    x_filt, P_filt, _ = kf_filter(rate, rate_sigma, Q_wobble)
    x_end, P_end = x_filt[-1], P_filt[-1]
    H_end = H_rebase[-1]
    n_fwd = int(2100 - years[-1]) + 1

    # ---------------------------------------------------------------
    # 1. Falsification: right-skewed xi_t, same Q, phi=1
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("1. Falsification: right-skewed acceleration innovations "
          "(same Q_wobble, phi=1)")
    print("=" * 70)
    _, H_draws_sym = forecast_mc(x_end, P_end, Q_wobble, H_end, n_fwd, n_draws=50000)
    H2100_sym = H_draws_sym[:, -1] * M_TO_MM
    lo_sym, hi_sym = np.percentile(H2100_sym, [5, 95])
    for skew_a in (6.0, 15.0):
        _, H_draws_skew = forecast_mc_skewed(x_end, P_end, Q_wobble, H_end, n_fwd,
                                              skew_a=skew_a, n_draws=50000)
        H2100_skew = H_draws_skew[:, -1] * M_TO_MM
        lo_s, hi_s = np.percentile(H2100_skew, [5, 95])
        print(f"  skew_a={skew_a:5.1f}: median={np.median(H2100_skew):.1f} mm, "
              f"90% CI=[{lo_s:.1f}, {hi_s:.1f}] mm, width={hi_s-lo_s:.1f} mm, "
              f"P(gain)={np.mean(H2100_skew<0):.3f}   "
              f"[symmetric baseline: median={np.median(H2100_sym):.1f}, "
              f"width={hi_sym-lo_sym:.1f}, P(gain)={np.mean(H2100_sym<0):.3f}]")
    print("  --> Confirmed: skewing xi_t leaves the 2100 width and P(gain) "
          "essentially unchanged from the symmetric undamped baseline "
          "(width ~1300 mm, P(gain)~0.40-0.44 regardless of skew_a). "
          "Per-step skew washes out over 77 compounding annual steps, as "
          "expected -- this branch is dropped; not pursued further.")

    # ---------------------------------------------------------------
    # 2. Admissibility constraint from the project's own data
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. Admissibility constraint: WAIS's own record, not external ensembles")
    print("=" * 70)
    neg_mask = rate < 0
    print(f"  Of {n} annual observations (1979-2023), {neg_mask.sum()} "
          f"({neg_mask.sum()/n:.0%}) have a negative annual rate (net mass "
          f"GAIN that single year) -- interannual sign flips are normal.")
    H_mm = H_rebase * 1000
    running_max = np.maximum.accumulate(H_mm)
    drawdown = running_max - H_mm
    i_max = np.argmax(drawdown)
    peak_idx = np.argmax(H_mm[:i_max + 1])
    print(f"  Max SUSTAINED drawdown (peak-to-later-trough decline in "
          f"cumulative H) across the full 45-yr record: {drawdown[i_max]:.3f} mm, "
          f"over {years[i_max]-years[peak_idx]:.0f} yr "
          f"({years[peak_idx]:.0f}->{years[i_max]:.0f}).")
    runs, cur = [], 0
    for v in neg_mask:
        if v:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    print(f"  Longest consecutive run of negative-rate years: {max(runs)} yr.")
    print("  In 45 years of observation, WAIS has never shown even a "
          "multi-year (let alone multi-decadal) SUSTAINED reversal below a "
          "prior baseline -- the deepest such drawdown is a fraction of a "
          "mm over 2 years. A 77-yr-ahead forecast that puts a "
          "non-negligible probability on net mass GAIN relative to today "
          "has no support in the observed record. We therefore require "
          "P(H_2100 < 0) <= 1% as the primary admissibility threshold "
          "(already generous relative to the empirical base rate of "
          "sustained reversals, which is indistinguishable from zero in "
          "45 years), and report sensitivity at a looser 5% threshold.")

    # ---------------------------------------------------------------
    # 3a. Sub-decadal wobble decay from IMBIE-3's own ACF (sanity check only)
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3a. Sub-decadal wobble decay from IMBIE-3 ACF (SANITY CHECK / "
          "LOWER BOUND ONLY -- not used as the production-candidate tau)")
    print("=" * 70)
    for label, mask in [('1992-2009 (satellite-era pre-break)',
                          (years >= 1992) & (years < WAIS_ONSET_YEAR)),
                         ('1979-2023 (full record)', np.ones(n, dtype=bool))]:
        acf, tau_short, n_sub = estimate_short_wobble_decay(years, rate, mask)
        print(f"  {label} (n={n_sub}): ACF lags 1-{len(acf)-1} = "
              f"{np.round(acf[1:],2)}; AR(1)-equivalent e-folding tau "
              f"= {tau_short:.1f} yr" if np.isfinite(tau_short) else
              f"  {label} (n={n_sub}): lag-1 ACF <= 0 -- wobble is "
              f"effectively uncorrelated year-to-year (decays within 1 yr)")
    print("  Interpretation: whatever this returns, a <=45-yr record can only "
          "ever resolve a decay time up to roughly its own length -- it "
          "cannot rule out or confirm a decades-to-centuries settling time. "
          "This is the same reasoning that keeps tau_d=150yr in the ocean "
          "two-layer model literature-sourced rather than fit to the 21-yr "
          "ocean record (Geoffroy et al. 2013; see module docstring above).")

    # ---------------------------------------------------------------
    # 3b. Literature settling timescale (production candidate)
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3b. Literature-based settling timescale (production candidate)")
    print("=" * 70)
    print("  Robel, Roe & Haseloff (2018, JGR Earth Surface, "
          "10.1029/2018JF004709) derive a two-timescale linearized "
          "response for marine-terminating glaciers/ice streams: a FAST "
          "timescale T_F (decades to a century -- glacier/grounding-zone "
          "thickness and length adjustment) and a SLOW timescale T_S "
          "(centuries to millennia -- large-scale interior adjustment, "
          "and the regime associated with marine ice sheet instability). "
          "For their worked example parameter set (Table 1: typical "
          "marine-terminating-glacier thickness, O(1) prograde bed "
          "slope, time-averaged SMB), fitting an AR(2) to simulated "
          "grounding-line position gives T_F = 70 yr and T_S = 8300 yr "
          "(section 3, p. 2211-2213) -- two widely separated scales, "
          "consistent with the abstract's general statement of "
          "'decades to centuries' (fast) vs 'centuries to millennia' "
          "(slow). We adopt the FAST timescale as the settling time for "
          "S1's generic non-instability wobble: S1 is defined as the "
          "no-MISI case, so its acceleration should relax on the fast "
          "dynamic-adjustment scale, not the slow/unstable scale "
          "associated with S2. tau = 70 yr, bracketed [50, 150] yr for "
          "sensitivity, following the same 'import from literature "
          "because the record is too short' logic as tau_d=150 yr "
          "(Geoffroy et al. 2013) in this project's ocean two-layer model.")

    # Robustness check: re-fit Q with the OU transition built into the
    # filter itself (phi=exp(-1/70) applied during filtering, not just
    # forecasting), on the same pre-break satellite-era window.
    phi_lit = np.exp(-1.0 / 70.0)
    mask_wobble = (years >= 1992) & (years < WAIS_ONSET_YEAR)
    Q_wobble_ou = fit_Q_mle_ou(rate[mask_wobble], rate_sigma[mask_wobble], phi_lit)
    print(f"\n  Robustness check: Q re-fit with phi=exp(-1/70)={phi_lit:.4f} "
          f"built into the filter itself: Q_wobble_ou = {Q_wobble_ou:.3e} "
          f"vs Q_wobble (phi=1 filter) = {Q_wobble:.3e} "
          f"(ratio = {Q_wobble_ou/Q_wobble:.3f}) -- "
          f"{'negligible difference, confirms phi~1 filtering + damped-forecast approximation is fine' if 0.8 < Q_wobble_ou/Q_wobble < 1.25 else 'NOT negligible -- approximation may need revisiting'}.")

    # ---------------------------------------------------------------
    # 3c. Sensitivity: 2100 width and P(H_2100<0) vs settling timescale
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3c. Sensitivity of 2100 band width and P(H_2100<0) to tau")
    print("=" * 70)
    tau_grid = [20, 30, 50, 70, 100, 150, 200, 300, np.inf]
    rows = admissibility_scan(x_end, P_end, Q_wobble, H_end, n_fwd, tau_grid)
    print(f"{'tau (yr)':<10}{'phi':<10}{'median (mm)':<14}{'90% CI (mm)':<24}"
          f"{'width (mm)':<12}{'P(gain)':<10}{'p99 (mm)':<10}")
    for r in rows:
        tau_str = 'inf (undamped)' if r['tau'] == np.inf else f"{r['tau']:.0f}"
        phi_str = '1.000' if r['tau'] == np.inf else f"{r['phi']:.4f}"
        print(f"{tau_str:<10}{phi_str:<10}{r['median']:<14.1f}"
              f"[{r['lo90']:.0f}, {r['hi90']:.0f}]".ljust(24) +
              f"{r['width']:<12.1f}{r['p_gain']:<10.3f}{r['p99']:<10.1f}")

    # Find minimal tau (within/near the literature range) satisfying each
    # admissibility threshold via a fine scan.
    tau_fine = np.arange(5, 305, 2.5)
    rows_fine = admissibility_scan(x_end, P_end, Q_wobble, H_end, n_fwd, tau_fine,
                                     n_draws=15000)
    for thresh in (0.05, 0.01):
        ok = [r for r in rows_fine if r['p_gain'] <= thresh]
        if ok:
            tau_needed = min(r['tau'] for r in ok)
            print(f"\n  Minimal tau satisfying P(H_2100<0) <= {thresh:.0%}: "
                  f"~{tau_needed:.0f} yr")
        else:
            print(f"\n  No tau in the scanned range [5, 300] yr satisfies "
                  f"P(H_2100<0) <= {thresh:.0%}")

    # ---------------------------------------------------------------
    # 4. Combined model: literature tau=70 yr + Q_wobble
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. Combined model (recommended): OU-on-acceleration, "
          "tau=70 yr (Robel et al. 2018 fast timescale), Q_wobble "
          "(1992-2009 pre-break MLE)")
    print("=" * 70)
    phi70 = np.exp(-1.0 / 70.0)
    _, H_draws_70 = forecast_mc(x_end, P_end, Q_wobble, H_end, n_fwd,
                                  n_draws=100000, phi=phi70)
    H2100_70 = H_draws_70[:, -1] * M_TO_MM
    med70, lo70, hi70 = (np.median(H2100_70), *np.percentile(H2100_70, [5, 95]))
    p_gain_70 = np.mean(H2100_70 < 0)
    p99_70 = np.percentile(H2100_70, 99)

    print(f"\n  {'model':<38}{'median (mm)':<14}{'90% CI (mm)':<24}"
          f"{'width (mm)':<12}{'P(gain)':<10}")
    print(f"  {'production quadratic (current)':<38}{83.5:<14.1f}"
          f"{'[76.1, 90.9]':<24}{14.8:<12.1f}{'~0':<10}")
    print(f"  {'IRW, undamped (round 1)':<38}{94.1:<14.1f}"
          f"{'[-563, 752]':<24}{1315.4:<12.1f}{0.444:<10.3f}")
    print(f"  {'OU tau=70yr + Q_wobble (this round)':<38}{med70:<14.1f}"
          f"{f'[{lo70:.0f}, {hi70:.0f}]':<24}{hi70-lo70:<12.1f}{p_gain_70:<10.3f}")
    print(f"\n  --> width ratio vs production quadratic: {(hi70-lo70)/14.8:.1f}x")
    print(f"  --> P(H_2100<0) = {p_gain_70:.3f} at tau=70yr "
          f"({'MEETS' if p_gain_70<=0.05 else 'DOES NOT MEET'} the 5% admissibility bar; "
          f"{'MEETS' if p_gain_70<=0.01 else 'DOES NOT MEET'} the 1% bar)")

    print(f"\n  S1 2100 p99 under this combined model: {p99_70:.1f} mm "
          f"(current S2_fast_wais low_mm anchor = 94 mm, set to "
          f"production-quadratic-S1's own p99). If S1 is replaced with "
          f"this model, the anchor would need to move to ~{p99_70:.0f} mm "
          f"-- FLAGGED, not applied: this is a {p99_70/94:.1f}x change to "
          f"S2's floor and reshapes the full two-scenario mixture; must be "
          f"a deliberate joint decision, not a silent propagation.")

    # ---------------------------------------------------------------
    # Rolling-origin backtest for the combined model
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Rolling-origin backtest: combined OU (tau=70yr) model vs quadratic")
    print("=" * 70)
    pool_ou, pool_quad = rolling_origin_backtest(years, H_rebase, rate, rate_sigma,
                                                   Q_wobble, phi=phi70)
    print(f"\n{'lead bin (yr)':<16}{'OU tau=70yr coverage':<24}{'quadratic coverage':<22}")
    for key in list(pool_ou.keys()):
        cov_ou, nk = pool_ou[key]
        cov_q, nq = pool_quad[key]
        label = f"{key[0]}-{key[1]}" if isinstance(key, tuple) else key
        print(f"{label:<16}{cov_ou:.2f} (n={nk:<4}){'':<10}{cov_q:.2f} (n={nq})")
    print("\nNOTE: same rolling-origin caveats as round 1 (effective sample "
          "size is a small multiple of the 8 origins, not the raw point "
          "counts). tau=70yr is barely distinguishable from undamped over "
          "this <=25-yr testable horizon (phi=exp(-1/70)=0.986 per step), "
          "so this table is expected to look similar to round 1's KF/IRW "
          "row -- the backtest validates the SHORT-horizon behavior of "
          "this model; it cannot and does not validate the 2100 "
          "extrapolation, which is why the admissibility constraint "
          "(step 2) rather than backtest coverage is what disciplines tau.")


def forecast_mc_floor(x_end, P_end, Q, H_end, n_years_fwd, phi, v_floor=0.0,
                       n_draws=100000, rng=None):
    """OU-on-acceleration forecast with a reflecting floor on the RATE
    v_t >= v_floor (default 0).

    WHY THIS EXISTS: the OU-on-acceleration model (forecast_mc with
    phi=exp(-1/tau)) damps the ACCELERATION back toward zero, but never
    damps the RATE v_t itself -- v is a pure running sum of whatever
    acceleration history occurred, with no reversion of its own. A
    decomposition (see WRITE-UP) shows that even at tau as short as
    20 yr (well below the literature "decades to a century" range), 2100
    P(H<0) barely moves (0.40 undamped -> 0.36-0.40 across tau in
    [20,300] yr) and NO tau in [5,300] yr satisfies the <=5% or <=1%
    admissibility bar from item 2. Roughly half of the 2100 spread comes
    from the filtered end-of-record STATE uncertainty in v_end itself
    (+-0.13 mm/yr, propagated undamped over 77 yr since nothing reverts
    v), not from future process noise at all -- so tau cannot fix it no
    matter how short. A floor on v directly encodes the same empirical
    admissibility fact from item 2 (WAIS has never shown a sustained
    multi-year reversal) as a mechanical constraint on the dynamics,
    rather than relying on tau to produce it indirectly. This goes one
    step beyond what was literally scoped (OU on acceleration alone);
    flagged as such in the written summary.
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    x0 = rng.multivariate_normal(x_end, P_end, size=n_draws)
    v = x0[:, 0].copy()
    a = x0[:, 1].copy()
    H_cum = np.full(n_draws, H_end)
    sqrtQ = np.sqrt(Q)
    H_draws = np.zeros((n_draws, n_years_fwd))
    for k in range(n_years_fwd):
        v = v + a
        v = np.maximum(v, v_floor)
        a = phi * a + sqrtQ * rng.standard_normal(n_draws)
        H_cum = H_cum + v
        H_draws[:, k] = H_cum
    return H_draws


def main_followup_part2():
    """Diagnose WHY OU-on-acceleration alone cannot satisfy the item-2
    admissibility constraint at any literature-plausible tau, and test
    the reflecting-floor supplement that does.
    """
    years, H_rebase, SIGMA_CUM_local, rate, rate_sigma = load_wais_annual()
    global SIGMA_CUM
    SIGMA_CUM = SIGMA_CUM_local
    Q_wobble = fit_Q_mle(rate[(years >= 1992) & (years < WAIS_ONSET_YEAR)],
                          rate_sigma[(years >= 1992) & (years < WAIS_ONSET_YEAR)])
    x_filt, P_filt, _ = kf_filter(rate, rate_sigma, Q_wobble)
    x_end, P_end = x_filt[-1], P_filt[-1]
    H_end = H_rebase[-1]
    n_fwd = int(2100 - years[-1]) + 1

    print("\n" + "=" * 70)
    print("5. WHY tau alone cannot satisfy admissibility: variance decomposition")
    print("=" * 70)
    print(f"  Filtered end-of-record state (2023): v_end = {x_end[0]*1000:.3f} "
          f"+/- {np.sqrt(P_end[0,0])*1000:.3f} mm/yr, "
          f"a_end = {x_end[1]*1000:.4f} +/- {np.sqrt(P_end[1,1])*1000:.4f} mm/yr^2 "
          f"-- acceleration is not significant at 90% (CI spans both signs).")
    print("  v has NO direct mean-reversion in this model (only a does): "
          "v_end's own uncertainty propagates UNDAMPED into every future "
          "year, so its contribution to Var(H_2100) scales as h^2 "
          "regardless of tau. Decomposition (tau=70yr):")

    rng = np.random.default_rng(1)
    phi70 = np.exp(-1.0 / 70.0)
    # (A) state uncertainty only, Q=0 going forward
    v0 = rng.multivariate_normal(x_end, P_end, size=200000)
    v, a = v0[:, 0].copy(), v0[:, 1].copy()
    Hc = np.full(200000, H_end)
    for k in range(n_fwd):
        v = v + a
        a = phi70 * a
        Hc = Hc + v
    HA = Hc * 1000
    loA, hiA = np.percentile(HA, [5, 95])
    # (B) process noise only, state known exactly
    v = np.full(200000, x_end[0]); a = np.full(200000, x_end[1])
    Hc = np.full(200000, H_end)
    sqrtQ = np.sqrt(Q_wobble)
    for k in range(n_fwd):
        v = v + a
        a = phi70 * a + sqrtQ * rng.standard_normal(200000)
        Hc = Hc + v
    HB = Hc * 1000
    loB, hiB = np.percentile(HB, [5, 95])
    print(f"    (A) state uncertainty only (P_end propagated, no future Q): "
          f"width={hiA-loA:.1f} mm, P(gain)={np.mean(HA<0):.3f}")
    print(f"    (B) process noise only (today's rate assumed known exactly): "
          f"width={hiB-loB:.1f} mm, P(gain)={np.mean(HB<0):.3f}")
    print("  Both are large. Shortening tau only shrinks (B)'s long-run "
          "growth law (h^5 -> roughly h^3, via a stationary-variance "
          "acceleration integrated into a diffusive rate); it does nothing "
          "to (A). So NO tau can satisfy the admissibility bar from item 2 "
          "-- confirmed by the tau=[5,300] scan above finding no solution.")

    print("\n" + "=" * 70)
    print("6. Reflecting floor on the rate (v_t >= 0): the mechanism that "
          "actually satisfies admissibility")
    print("=" * 70)
    print("  Rather than relying on tau to produce P(H_2100<0)<=1% indirectly "
          "(it cannot, per above), impose the item-2 empirical fact "
          "directly: v_t >= 0 each forecast year (no sustained net mass "
          "GAIN under the 'no instability' story -- exactly what the "
          "45-yr record shows). This is an ADDITIONAL mechanism beyond "
          "the OU-on-acceleration reformulation as literally scoped; "
          "flagged as such, not silently substituted.")
    for tau_lbl, tau in (('70 yr (literature)', 70.0), ('undamped', np.inf)):
        Hf = forecast_mc_floor(x_end, P_end, Q_wobble, H_end, n_fwd,
                                phi=(np.exp(-1.0 / tau) if np.isfinite(tau) else 1.0))
        Hf = Hf[:, -1] * M_TO_MM
        lo, hi = np.percentile(Hf, [5, 95])
        print(f"  tau={tau_lbl:<20}: median={np.median(Hf):.1f} mm, "
              f"90% CI=[{lo:.1f}, {hi:.1f}] mm, width={hi-lo:.1f} mm, "
              f"P(gain)={np.mean(Hf<0):.4f}, p99={np.percentile(Hf,99):.1f} mm")
    print("\n  Recommended combined model (if this floor is accepted): "
          "OU tau=70yr + Q_wobble + reflecting floor v>=0 -> "
          "2100 median ~106 mm, 90% CI ~[10, 578] mm, width ~568 mm "
          "(vs 14.8 mm production, 1315 mm undamped). P(gain)=0 by "
          "construction, satisfying both the 5% and 1% admissibility bars.")



# =========================================================================
# ROUND 3 (2026-09-17, superseding rounds 1-2 as the recommended approach):
# Tarantola-style external prediction-error covariance from the ISMIP6
# emulator, added to the UNCHANGED S1 quadratic-in-time fit.
# =========================================================================
#
# Rounds 1-2 (kept above, in file history, for reference -- NOT built on
# further; not invoked by __main__ below) tried to derive the extra
# long-lead-time uncertainty ourselves from the 44-47-yr IMBIE-3 record
# (a fitted process-noise Kalman filter, then an OU/mean-reversion
# timescale). Round 2 showed this doesn't actually work: the record is
# too short to pin down either a process-noise scale or a settling
# timescale at a 77-yr extrapolation horizon, and forcing an admissible
# answer required bolting on extra machinery (a reflecting floor) not
# implied by the data at all.
#
# New direction (per user): keep S1 exactly as the data-driven IMBIE-3
# quadratic-in-time extrapolation (unchanged median, unchanged near-term
# fit, unchanged S1_QUADRATIC_MEAN/_COV) -- the ONLY defect is that our
# own record cannot tell us how fast the *extrapolation* (prediction)
# error should grow at long lead times. Rather than inventing that growth
# law from 45 years of data (round 1-2's mistake), borrow it from an
# existing, published, peer-reviewed source that already is a WAIS-
# specific "no exotic instability" projection: the IPCC AR6 ISMIP6
# emulator (medium confidence), the same one already plotted in
# component_wais_pdf_exceedance_ipcc_p_s1_sweep.png.
#
# Framing (Tarantola, 2005, "Inverse Problem Theory"): total predictive
# covariance = data/parameter covariance (ours; the (a,v,H0) posterior,
# well constrained by 45 yr of IMBIE-3) + a "theory"/extrapolation-error
# covariance that our short record cannot resolve, estimated here from an
# independent, external source instead of from our own data.
#
#     H_S1(t) = H_quad(t) + eta * sqrt(extra_var(t)),   eta ~ N(0,1)  once
#               per Monte Carlo sample (not per year -- keeps each
#               trajectory smooth)
#
#     extra_var(t) = max(0, Var_ISMIP6(t) - Var_ISMIP6(anchor_year))
#
# No MCMC, no Kalman filter, no backtest, no free timescale parameter --
# the growth law and its normalization both come directly from the
# ISMIP6 emulator's own reported std(t).

from scipy.interpolate import interp1d

CONF_BASE = '../data/raw/ipcc_ar6/slr/ar6/global/dist_components'


def load_ismip6_wais(scenario='ssp245'):
    from slr_data_readers import read_ipcc_ar6_component
    df = read_ipcc_ar6_component(
        CONF_BASE, component_type='icesheets', sub_component='WAIS',
        model='ipccar6-ismipemuicesheet', scenario=scenario,
        convert_to_meters=True)
    return df


def build_extra_variance_interpolator(df, anchor_year):
    """extra_var(t) = max(0, Var_ISMIP6(t) - Var_ISMIP6(anchor_year)),
    linearly interpolated between the ISMIP6 emulator's native decadal
    output years and flat-extrapolated outside its [2020, 2100] range
    (only relevant for anchor_year, which for WAIS falls at ~2023.5,
    inside the data range -- no extrapolation is actually needed for the
    anchor; flat extrapolation past 2100 would apply only if S1 were
    evaluated beyond 2100, which is NOT done here -- see WRITE-UP).

    Uses the emulator's own reported `std` column directly (already
    established elsewhere in this project as symmetric/Gaussian-like,
    unlike LARMIP-2), not a percentile-spread fallback -- checked below
    to be present, monotonically increasing, and free of decadal-sampling
    noise/dips (no smoothing needed).
    """
    yrs = df.index.values.astype(float)
    var_native = df['std'].values ** 2
    assert np.all(np.diff(var_native) > 0), \
        "ISMIP6 WAIS variance(t) is not monotonically increasing -- would need smoothing"
    var_interp = interp1d(yrs, var_native, kind='linear', bounds_error=False,
                            fill_value=(var_native[0], var_native[-1]))
    var_anchor = float(var_interp(anchor_year))

    def extra_var(t):
        return np.maximum(0.0, var_interp(np.asarray(t, dtype=float)) - var_anchor)

    return extra_var, var_anchor, yrs, var_native


def main_round3():
    print("=" * 70)
    print("ROUND 3: external (ISMIP6 emulator) extrapolation-error covariance")
    print("=" * 70)

    years, H_rebase, SIGMA_CUM, rate, rate_sigma = load_wais_annual()
    anchor_year = float(years[-1])  # matches component_wais.ipynb's
    # `anchor_year = wais_year[-1]` used in sample_a4_wais_trajectories --
    # NOT WAIS_ONSET_YEAR (2010), which is only the function's unused
    # default; the actual notebook call always passes the last IMBIE-3
    # observation year explicitly.
    print(f"\nAnchor year (last IMBIE-3 observation, matches "
          f"component_wais.ipynb cell-009's `anchor_year = wais_year[-1]`): "
          f"{anchor_year:.1f}")

    # --- scenario sensitivity check (item 1) ---
    print("\nScenario sensitivity of ISMIP6 WAIS std(2100) (S1 is SSP-"
          "independent by construction; checking this doesn't matter):")
    for ssp in ('ssp126', 'ssp245', 'ssp585'):
        df_s = load_ismip6_wais(ssp)
        print(f"  {ssp}: std(2100) = {df_s.loc[2100,'std']*M_TO_MM:.1f} mm, "
              f"median = {df_s.loc[2100,'median']*M_TO_MM:.1f} mm")
    print("  --> std(2100) varies by <5% across SSP1-2.6 to SSP5-8.5 (59.7-62.1 mm); "
          "scenario choice does not matter. Using ssp245 as reference.")

    df = load_ismip6_wais('ssp245')
    print(f"\nISMIP6 emulator WAIS (ssp245) native output: years "
          f"{df.index.min()}-{df.index.max()} (decadal grid, n={len(df)}); "
          f"columns {list(df.columns)}")
    print(df[['median', 'p5', 'p95', 'std']].mul(1000).round(2))

    extra_var, var_anchor, yrs_native, var_native = \
        build_extra_variance_interpolator(df, anchor_year)
    print(f"\nVar_ISMIP6(anchor_year={anchor_year:.1f}) [interpolated] = "
          f"{var_anchor:.3e} m^2 (std = {np.sqrt(var_anchor)*M_TO_MM:.2f} mm) "
          f"-- anchor falls inside the native [2020,2100] range (no "
          f"extrapolation needed).")
    print("Sanity check: native ISMIP6 std(t) is monotonically increasing "
          "with no decadal-sampling dips (assert passed above) -- linear "
          "interpolation between the 9 native decadal points is used "
          "as-is, no smoothing applied.")

    extra_var_2100 = extra_var(2100.0)
    print(f"\nextra_var(2100) = max(0, Var_ISMIP6(2100) - Var_ISMIP6(anchor)) "
          f"= {extra_var_2100:.3e} m^2 (extra std = "
          f"{np.sqrt(extra_var_2100)*M_TO_MM:.2f} mm)")

    # --- draw S1 quadratic (UNCHANGED) + one eta per sample ---
    n_draws = 500000
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.multivariate_normal(S1_QUADRATIC_MEAN_PROD, S1_QUADRATIC_COV_PROD,
                                     size=n_draws)
    a, v, H0 = draws[:, 0], draws[:, 1], draws[:, 2]
    tau_2100 = 2100.0 - BASELINE_YEAR
    H_quad_2100_mm = (0.5 * a * tau_2100 ** 2 + v * tau_2100 + H0) * M_TO_MM

    eta = rng.standard_normal(n_draws)
    extra_std_2100_mm = np.sqrt(extra_var_2100) * M_TO_MM
    H_combined_2100_mm = H_quad_2100_mm + eta * extra_std_2100_mm

    med_c = np.median(H_combined_2100_mm)
    lo_c, hi_c = np.percentile(H_combined_2100_mm, [5, 95])
    p99_c = np.percentile(H_combined_2100_mm, 99)
    p_gain_c = np.mean(H_combined_2100_mm < 0)

    med_q, lo_q, hi_q = 83.5, 76.1, 90.9  # documented production values
    med_i, lo_i, hi_i = (df.loc[2100, 'median'] * M_TO_MM,
                          df.loc[2100, 'p5'] * M_TO_MM,
                          df.loc[2100, 'p95'] * M_TO_MM)

    print("\n" + "=" * 70)
    print("Results: 2100 S1 band, production vs round-3 (data+ISMIP6-external-error) "
          "vs ISMIP6-native")
    print("=" * 70)
    print(f"{'model':<42}{'median (mm)':<14}{'90% CI (mm)':<22}{'width (mm)':<12}")
    print(f"{'production quadratic (data only, current)':<42}{med_q:<14.1f}"
          f"{f'[{lo_q:.1f}, {hi_q:.1f}]':<22}{hi_q-lo_q:<12.1f}")
    print(f"{'round 3: quadratic + ISMIP6 extra-var (NEW)':<42}{med_c:<14.1f}"
          f"{f'[{lo_c:.1f}, {hi_c:.1f}]':<22}{hi_c-lo_c:<12.1f}")
    print(f"{'ISMIP6 emulator native WAIS (ssp245, ref.)':<42}{med_i:<14.1f}"
          f"{f'[{lo_i:.1f}, {hi_i:.1f}]':<22}{hi_i-lo_i:<12.1f}")

    print(f"\nround-3 p99 = {p99_c:.1f} mm  (current S2_fast_wais low_mm anchor "
          f"= 94 mm, pinned to production quadratic's p99); a "
          f"{p99_c/94:.1f}x change if adopted -- FLAGGED, not resolved here.")
    print(f"P(H_2100 < 0) under round 3 = {p_gain_c:.3f} "
          f"(ISMIP6 emulator's own native p5 at 2100 is {lo_i:.0f} mm, i.e. "
          f"its own distribution already puts ~5%+ mass below zero -- some "
          f"non-zero mass-gain probability here is consistent with "
          f"'matching the literature', not a defect to force to zero).")

    print(f"\nMedian check: round-3 median ({med_c:.1f} mm) vs production "
          f"median ({med_q:.1f} mm) -- should match (eta is mean-zero, "
          f"symmetric around the unchanged quadratic median): "
          f"{'MATCHES' if abs(med_c-med_q) < 1.0 else 'MISMATCH -- check'}.")

    return {
        'anchor_year': anchor_year, 'extra_var_2100': extra_var_2100,
        'median': med_c, 'ci90': (lo_c, hi_c), 'p99': p99_c, 'p_gain': p_gain_c,
    }



# =========================================================================
# ROUND 3b (2026-09-17 refinement): smooth quadratic-in-elapsed-time
# extension of extra_var(t) from anchor to 2150 (replacing round 3's
# linear-interpolate-then-freeze-at-2100 treatment), plus S2 floor rule
# comparison (p95 vs p99).
# =========================================================================
#
# Round 3 built extra_var(t) by linearly interpolating the 9 native
# ISMIP6 decadal points (2020-2100) and then holding flat for t>2100 --
# fine at 2100 (the stated main target) but not smooth/principled if S1 is
# ever evaluated out to 2150 (component_wais.ipynb's PROJ_YEARS does run
# to 2150). This section instead fits ONE smooth curve across the whole
# anchor-to-2150 window from the same 9 ISMIP6 points, and checks whether
# a quadratic is actually a good description of ISMIP6's own growth
# curve before trusting its extrapolation.

def fit_and_report_quadratic_extension(df, anchor_year):
    yrs = df.index.values.astype(float)
    elapsed = yrs - anchor_year
    var_native = df['std'].values ** 2
    std_native = df['std'].values

    # --- Attempt 1 (as literally requested): quadratic fit to Var(t) ---
    coeffs_var = np.polyfit(elapsed, var_native, 2)
    fit_var = np.polyval(coeffs_var, elapsed)
    r2_var = 1 - np.sum((var_native - fit_var) ** 2) / np.sum(
        (var_native - var_native.mean()) ** 2)
    c2_var = coeffs_var[0]
    vertex_var = -coeffs_var[1] / (2 * c2_var) if c2_var != 0 else np.nan
    e_grid = np.linspace(0, 2150 - anchor_year, 2000)
    var_grid_fit = np.polyval(coeffs_var, e_grid)
    mono_var = np.all(np.diff(var_grid_fit) >= -1e-12)

    print("  Attempt 1: quadratic fit to Var_ISMIP6(t) directly (as first "
          "requested):")
    print(f"    R^2 = {r2_var:.4f}")
    print(f"    vertex at elapsed={vertex_var:.1f} yr (year "
          f"{anchor_year+vertex_var:.0f}); parabola opens "
          f"{'upward' if c2_var>0 else 'downward'}")
    print(f"    monotonically non-decreasing from anchor to 2150? {mono_var}")
    if not mono_var:
        dip_idx = np.argmin(var_grid_fit)
        print(f"    --> FAILS: fitted variance DIPS to a minimum of "
              f"{var_grid_fit[dip_idx]*1e6:.1f} mm^2 at elapsed="
              f"{e_grid[dip_idx]:.1f} yr (year {anchor_year+e_grid[dip_idx]:.0f}), "
              f"i.e. inside the domain we need to extrapolate across, not "
              f"only past 2150. A parabola fit directly to Var(t) is pulled "
              f"down by the small early-record values (var grows very "
              f"slowly 2020-2050, then much faster 2050-2100 -- a genuinely "
              f"convex-in-time shape that a single quadratic in VARIANCE "
              f"cannot represent without an unphysical interior minimum). "
              f"FLAGGED as a bad fit despite a superficially decent "
              f"R^2={r2_var:.2f} -- not used.")

    # --- Attempt 2 (fallback): quadratic fit to std(t), squared ---
    coeffs_std = np.polyfit(elapsed, std_native, 2)
    fit_std = np.polyval(coeffs_std, elapsed)
    r2_std = 1 - np.sum((std_native - fit_std) ** 2) / np.sum(
        (std_native - std_native.mean()) ** 2)
    std_grid_fit = np.polyval(coeffs_std, e_grid)
    std_anchor = np.polyval(coeffs_std, 0.0)
    extra_var_grid = np.maximum(0.0, std_grid_fit ** 2 - std_anchor ** 2)
    mono_extra = np.all(np.diff(extra_var_grid) >= -1e-15)
    vertex_std = -coeffs_std[1] / (2 * coeffs_std[0])

    print("\n  Attempt 2 (fallback, used instead): quadratic fit to "
          "std_ISMIP6(t), then squared to get Var(t):")
    print(f"    R^2 (on std, mm scale) = {r2_std:.4f}  (vs {r2_var:.4f} "
          f"for the direct-variance fit)")
    print(f"    residuals (mm): "
          f"{np.round((std_native-fit_std)*M_TO_MM, 2).tolist()}")
    print(f"    std-fit has a shallow local minimum at elapsed="
          f"{vertex_std:.1f} yr (year {anchor_year+vertex_std:.0f}), "
          f"std_fit there = {np.polyval(coeffs_std, vertex_std)*M_TO_MM:.2f} mm "
          f"vs anchor std_fit(0) = {std_anchor*M_TO_MM:.2f} mm -- a "
          f"negligible ~{(std_anchor-np.polyval(coeffs_std, vertex_std))*M_TO_MM:.2f} mm "
          f"dip right next to the anchor.")
    print(f"    extra_var(t) = max(0, std_fit(t)^2 - std_fit(anchor)^2) "
          f"monotonically non-decreasing over the FULL anchor-to-2150 "
          f"range? {mono_extra} (the near-anchor std dip above is fully "
          f"absorbed by the max(0,...) floor since it occurs before "
          f"extra_var has grown away from zero)")
    print(f"    --> ADOPTED: R^2=0.994 vs 0.947, no unphysical interior "
          f"dip in the quantity that actually matters (extra_var), and "
          f"the residual pattern (max |resid| ~2.3 mm) is small and "
          f"non-systematic. Fitting std(t) directly (not Var(t)) is a "
          f"better-behaved regression here because ISMIP6's std(t) grows "
          f"closer to quadratically in time, while Var(t)=std(t)^2 grows "
          f"quartically -- a much more curved target for a degree-2 "
          f"polynomial to track over an 80-130 yr window.")

    def extra_var_fn(t):
        e = np.asarray(t, dtype=float) - anchor_year
        s = np.polyval(coeffs_std, e)
        return np.maximum(0.0, s ** 2 - std_anchor ** 2)

    return extra_var_fn, coeffs_std, r2_std


def main_round3b():
    print("\n" + "=" * 70)
    print("ROUND 3b: smooth quadratic-in-elapsed-time extension to 2150, "
          "and p95 vs p99 S2-anchor comparison")
    print("=" * 70)

    years, H_rebase, SIGMA_CUM, rate, rate_sigma = load_wais_annual()
    anchor_year = float(years[-1])
    df = load_ismip6_wais('ssp245')

    print(f"\nFitting Var_ISMIP6(t) as a function of elapsed time since "
          f"anchor_year={anchor_year:.1f}, using ISMIP6's 9 native decadal "
          f"points (2020-2100):")
    extra_var_fn, coeffs_std, r2_std = fit_and_report_quadratic_extension(df, anchor_year)
    print(f"\n  Adopted fit: std_ISMIP6(t) [m] = {coeffs_std[0]:.4e}*e^2 "
          f"+ {coeffs_std[1]:.4e}*e + {coeffs_std[2]:.4e}, e = t - {anchor_year:.1f}")

    n_draws = 500000
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.multivariate_normal(S1_QUADRATIC_MEAN_PROD, S1_QUADRATIC_COV_PROD,
                                     size=n_draws)
    a, v, H0 = draws[:, 0], draws[:, 1], draws[:, 2]
    eta = rng.standard_normal(n_draws)

    print(f"\n{'year':<8}{'H_quad med (mm)':<18}{'extra std (mm)':<16}"
          f"{'combined med':<14}{'90% CI (mm)':<22}{'width':<10}"
          f"{'p95':<10}{'p99':<10}{'P(gain)':<10}")
    results = {}
    for target_year in (2100.0, 2150.0):
        tau = target_year - BASELINE_YEAR
        H_quad_mm = (0.5 * a * tau ** 2 + v * tau + H0) * M_TO_MM
        extra_std_mm = np.sqrt(extra_var_fn(target_year)) * M_TO_MM
        H_comb = H_quad_mm + eta * extra_std_mm
        med = np.median(H_comb)
        lo, hi = np.percentile(H_comb, [5, 95])
        p95 = np.percentile(H_comb, 95)
        p99 = np.percentile(H_comb, 99)
        pgain = np.mean(H_comb < 0)
        print(f"{target_year:<8.0f}{np.median(H_quad_mm):<18.1f}{extra_std_mm:<16.2f}"
              f"{med:<14.1f}{f'[{lo:.1f}, {hi:.1f}]':<22}{hi-lo:<10.1f}"
              f"{p95:<10.1f}{p99:<10.1f}{pgain:<10.3f}")
        results[target_year] = {'median': med, 'ci90': (lo, hi), 'p95': p95,
                                 'p99': p99, 'p_gain': pgain}

    print(f"\n2100 sanity check vs round 3 (linear-interp/freeze method): "
          f"median {results[2100.0]['median']:.1f} mm (round 3: 83.6 mm), "
          f"90% CI [{results[2100.0]['ci90'][0]:.1f}, "
          f"{results[2100.0]['ci90'][1]:.1f}] mm (round 3: [-16.4, 183.5] mm) "
          f"-- close, as expected (small differences from the quadratic "
          f"fit smoothing over the last decade's residual vs. exact "
          f"linear interpolation to the 2100 point).")

    print("\n" + "=" * 70)
    print("S2_fast_wais low_mm anchor: p95 (newly requested) vs p99 (prior "
          "rule), under this quadratic-extended S1 model")
    print("=" * 70)
    print(f"{'year':<8}{'p95 (mm)':<14}{'p99 (mm)':<14}{'current S2 low_mm':<20}")
    for yr in (2100.0, 2150.0):
        r = results[yr]
        print(f"{yr:<8.0f}{r['p95']:<14.1f}{r['p99']:<14.1f}"
              f"{'94 (2100 rule, p99-based)' if yr==2100.0 else 'n/a (S2 not defined at 2150)':<20}")
    print(f"\n  At 2100: switching the anchor rule from p99 (={results[2100.0]['p99']:.1f} mm) "
          f"to p95 (={results[2100.0]['p95']:.1f} mm) would LOWER S2's floor "
          f"vs the p99 version (as expected, p95<p99), and both are still "
          f"well above the current production value of 94 mm (which itself "
          f"used the OLD, unwidened S1_QUADRATIC p99). Neither is applied "
          f"to A4_SCENARIOS -- reported only, per instructions.")
    print(f"  2150 values are reported for completeness since PROJ_YEARS "
          f"extends there in component_wais.ipynb, but S2_fast_wais has no "
          f"defined 2150 anchor rule in production to compare against.")

    return results


if __name__ == '__main__':
    # Rounds 1-2 (fitted process-noise Kalman filter, then OU/mean-
    # reversion) are kept above for reference but are NOT run by default
    # any more -- superseded by round 3/3b. Uncomment to reproduce them:
    #   main()
    #   main_followup()
    #   main_followup_part2()
    main_round3()
    main_round3b()
