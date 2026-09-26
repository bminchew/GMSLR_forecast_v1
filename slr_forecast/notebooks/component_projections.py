"""
Component decomposition projection and export functions.

Extracted from component_decomposition.ipynb to keep notebooks free of
function definitions.  All functions operate in SLR convention
(positive = sea level rise) and SI-derived internal units (meters, °C, yr).
"""

import json
import os
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.special import expit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
try:
    from slr_forecast.config import BASELINE_YEAR, N_SAMPLES, WAIS_ONSET_YEAR
    from slr_forecast import M_TO_MM
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
# S1: Status quo — current discharge, no instability. Sampled directly
#     from a Bayesian quadratic-in-time fit to the observed IMBIE WAIS
#     record (component_wais.ipynb cell 11) rather than the skew-normal
#     'low_mm'/'high_mm'/'alpha' parametrization used for S2: since S1 has
#     no MISI by construction, the most direct "nothing new happens"
#     baseline is the naive statistical continuation of what has actually
#     been observed, requiring no assumption about future ocean-warming
#     magnitude or melt-discharge sensitivity coefficients. See
#     get_s1_quadratic() and _sample_s1_quadratic_mm() below, and
#     manuscripts/00_ddpi_slrforecast2026/a4_scenario_justification.md §3
#     for the derivation this replaced and its reconciliation with the
#     new approach. `low_mm`/`high_mm`/`alpha`/`beta_loc`/`beta_scale` are
#     not defined for S1 and are not read by the sampling functions below.
# S2: Fast WAIS (MISI + MICI) — marine ice sheet instability, with or
#     without cascading ice-cliff failure. Merges the former three-scenario
#     split (S1/S2/S3) into two: there is no physically defensible case for
#     MISI proceeding without amplification from processes that models omit
#     (Martin et al., AGU Advances), so "moderate MISI" and "MISI+MICI" are
#     treated as one probability-weighted regime rather than two competing
#     branches. Its 95th-percentile bound (high_mm=1000 mm) is a round
#     number chosen so the full two-scenario *mixture's* own p95 at 2100
#     (last checked against the p83-based low_mm rule below; see the
#     validation cell in component_wais.ipynb for the current value) sits at or
#     below the IPCC AR6 low-confidence AIS storyline's p95 under SSP5-8.5
#     (Fox-Kemper et al. 2021: 1309 mm) -- rather than pinning S2's
#     within-scenario p95 to that storyline directly, which pushed the
#     mixture's own p95 to ~1.6 m, meaningfully above AR6 low confidence
#     with no additional data to defend the excess. 1 m remains consistent
#     with independent cross-checks: the Amundsen Sea Embayment's ice
#     volume above flotation plus spillover to neighboring basins
#     (~1.1 m; Morlighem et al. 2020), Bamber et al. (2019)'s WAIS 95th
#     percentile at +5C after rheology correction (~1.19 m), and DeConto &
#     Pollard (2016)'s most aggressive MICI projection (0.64-1.14 m total
#     Antarctic). Its 5th-percentile bound
#     (low_mm=84 mm) is pinned to the 50th percentile (median) of
#     S1_status_quo's 2100 endpoint distribution. NOTE this is a
#     qualitatively different anchor than the earlier percentile choices
#     below: at the median, S2's floor sits in the *middle* of S1's own
#     distribution rather than above its upper tail -- about half of
#     S1's possible outcomes exceed S2's floor. This means S2 is no
#     longer strictly "worse than any plausible no-instability outcome";
#     its low end now overlaps the bulk of S1's own spread. History
#     (2026-09-17, all same day): 94 mm (99th percentile, pre-ISMIP6-
#     widening) -> 180 mm (95th percentile, when the ISMIP6
#     extrapolation-error widening below was added, since S1's 99th
#     percentile was no longer a tight, sample-efficient anchor once S1
#     carried a real tail) -> 139 mm (83rd percentile, closer to the
#     independent basin-by-basin estimate) -> 84 mm (50th percentile, on
#     request). See _sample_s1_quadratic_mm() and S1_ISMIP6_STD_COEFFS
#     above for the current S1 distribution this is computed from. To
#     regenerate: draw a large sample (e.g. N=2,000,000) from
#     _sample_s1_quadratic_mm(N, rng, [2100.0]) or sample_a4_wais_endpoint
#     (N, rng, scenario_overrides={'weights': {'S1_status_quo': 1.0,
#     'S2_fast_wais': 0.0}}) and read off the median (mm). `alpha=3`
#     gives the endpoint distribution a
#     right skew, reflecting the grounding-line flux nonlinearity that
#     amplifies uncertainty toward greater ice loss once MISI is
#     triggered (Robel et al. 2019); alpha only reshapes the distribution
#     between its fixed 5th/95th percentile bounds (which do not depend
#     on alpha), and the mixture is insensitive to its precise value
#     (median shifts by at most 0.13 m across alpha in [0, 4] -- see the
#     skewness sensitivity cell in component_wais.ipynb).
# ---------------------------------------------------------------------------

A4_SCENARIOS = {
    'S1_status_quo': {'P': 0.10, 'misi': False},
    # low_mm regenerated 2026-09-17: 50th percentile (median) of
    # S1_status_quo's 2100 endpoint distribution (~84 mm) INCLUDING the
    # ISMIP6 extrapolation-error widening (S1_ISMIP6_STD_COEFFS above).
    # History: 94 mm (p99, pre-ISMIP6-widening) -> 180 mm (p95) -> 139 mm
    # (p83) -> 84 mm (p50/median, on request) -- all same day. NOTE: at
    # the median, S2's floor no longer sits above S1's tail; see the
    # paragraph above for the full derivation and this rule's caveat.
    # high_mm and beta_loc set 2026-09-18 as clean rounded values: 1300 mm
    # is the IPCC AR6 low-confidence AIS p95 under SSP5-8.5 (1309 mm)
    # rounded; median beta 2 is a clean value (previous effective value was
    # 2.35 = 1.84 x rheology factor 1.275; 2.25 and 2.5 also tried 2026-09-18).
    # Both are adopted values describing the
    # qualitative shape suggested by the literature, not derived from physics.
    # Was high_mm=1000, beta_loc=log(1.84) with an n-driven rheology rescaling.
    # low_mm is DERIVED, not stored: set_s1_quadratic() fills it from the
    # installed fit via s1_median_mm(), so the S2 floor follows the S1
    # median automatically instead of drifting out of step with it when
    # the fit is regenerated.  None here means "no fit installed yet".
    'S2_fast_wais':  {'P': 0.90, 'low_mm': None, 'high_mm': 1300,
                      'alpha': 3.0,
                      'beta_loc': np.log(2.0), 'beta_scale': 0.3,
                      'misi': True},
}

# ---------------------------------------------------------------------------
# S1_status_quo: direct posterior sampling (quadratic-in-time fit)
#
# S1 has no MISI by construction, so unlike S2 -- whose range comes from a
# forward physical/ISMIP6-adjacent scaling chain subject to the same n=3
# rheology bias as the rest of ISMIP6 -- S1 is grounded directly in the
# observed IMBIE-3 WAIS record via a Bayesian quadratic-in-time fit
# (kinematics form -- position under constant acceleration):
#     rate(t) = a*(t-2000) + v,   H(t) = 0.5*a*(t-2000)^2 + v*(t-2000) + H0
# fit to IMBIE-3 1979-2023 (Otosaka et al. 2026; component_wais.ipynb
# cell 11, bayesian_models.fit_bayesian_level, with a signed Normal prior on the
# acceleration a since a purely time-based fit has no directional
# physical constraint -- WAIS's own record shows both acceleration and
# deceleration). The naive continuation of this fit is the most direct
# "nothing new happens" baseline available: it needs no assumption about
# future ocean-warming magnitude or melt-discharge sensitivity, unlike the
# physical scaling chain it replaces (documented, and reconciled against
# this approach, in
# manuscripts/00_ddpi_slrforecast2026/a4_scenario_justification.md §3).
#
# The S1 quadratic is (a, v, H0) in (m/yr^2, m/yr, m), using the
# CORRELATION-AWARE covariance from component_levelspace_robust_se
# (robust_level_intervals + robust_curve_band) -- the raw MCMC posterior
# treats each point of this cumulative record as independent and
# understates uncertainty, the same issue corrected for the other
# components' level-space fits.
#
# These are NOT hardcoded.  They are inherited from the fit that produces
# them, by one of two routes:
#
#   set_s1_quadratic(mean, cov)   component_wais.ipynb calls this in the
#                                 fit cell, immediately after
#                                 robust_level_intervals returns.  This
#                                 route is required, not merely
#                                 convenient: that notebook's projection
#                                 cells run BEFORE its save_wais cell, so
#                                 within a single run there is no stored
#                                 copy for them to read.  Do not remove
#                                 the setter in favour of the loader.
#
#   load_s1_quadratic()           every other consumer (results_figures,
#                                 the tests, and anything else calling the
#                                 A4 samplers without refitting) reads
#                                 s1_quadratic_fit.json, which the WAIS
#                                 notebook writes beside this module.
#                                 get_s1_quadratic() calls this lazily on
#                                 first use, so callers normally do
#                                 nothing.
#
# The fit lives in a small tracked JSON file rather than in
# component_results.h5 deliberately: the h5 is gitignored bulk output, so
# putting it there would make a fresh clone unable to run the WAIS tests,
# and would hide changes to the fit from review.  At ~600 bytes this file
# belongs in version control, where a refit shows up as a readable diff.
#
# There is deliberately no hardcoded fallback.  A stale copy of this fit
# is exactly the failure this indirection exists to prevent: it is
# invisible, it survives a clean re-run, and it silently decouples the
# projections from the record they claim to be fitted to.  If neither
# route has supplied values, get_s1_quadratic() raises.
_S1_QUADRATIC = None    # (mean, cov, provenance) once set


def set_s1_quadratic(mean, cov, provenance='set_s1_quadratic()'):
    """Install the fitted S1 quadratic (a, v, H0) mean and covariance.

    Parameters
    ----------
    mean : array-like, shape (3,)
        ``(a, v, H0)`` in (m/yr^2, m/yr, m) -- ``beta_map_vec`` from
        ``component_levelspace_robust_se.robust_level_intervals``.
    cov : array-like, shape (3, 3)
        The correlation-aware covariance for the same three parameters
        (``cov_robust`` from the same call).
    provenance : str
        Free-text note on where the values came from, echoed by
        ``describe_s1_quadratic()`` so a stale read is visible.
    """
    global _S1_QUADRATIC
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if mean.shape != (3,):
        raise ValueError(f'S1 quadratic mean must have shape (3,), got {mean.shape}')
    if cov.shape != (3, 3):
        raise ValueError(f'S1 quadratic cov must have shape (3, 3), got {cov.shape}')
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-20):
        raise ValueError('S1 quadratic cov is not symmetric')
    eigmin = np.linalg.eigvalsh(cov).min()
    if eigmin < -1e-12 * max(np.abs(cov).max(), 1e-30):
        raise ValueError(
            f'S1 quadratic cov is not positive semidefinite (min eigenvalue '
            f'{eigmin:.3e}).  This is the signature of a covariance built '
            f'from a cumulative sigma that was not re-anchored to the rebase '
            f'epoch -- see component_analysis.annualize_imbie.')
    _S1_QUADRATIC = (mean, cov, provenance)
    # Keep the derived S2 floor in step with the fit it is defined from.
    A4_SCENARIOS['S2_fast_wais']['low_mm'] = s1_median_mm()
    return mean, cov


S2_LOW_MM_RULE_YEAR = 2100.0


def s1_median_mm(year=S2_LOW_MM_RULE_YEAR):
    """Median S1_status_quo endpoint at *year*, in mm above BASELINE_YEAR.

    Closed form rather than Monte Carlo: H(t) is linear in (a, v, H0) and
    the ISMIP6 extrapolation-error term added by _sample_s1_quadratic_mm
    is zero-mean, so the median is just the fit evaluated at its mean.
    That makes this exact and seed-free, where sampling it would add
    noise and a seed dependence to a scenario parameter.
    """
    mean = _S1_QUADRATIC[0]
    tau = float(year) - BASELINE_YEAR
    return (0.5 * mean[0] * tau ** 2 + mean[1] * tau + mean[2]) * M_TO_MM


S1_QUADRATIC_PATH = Path(__file__).resolve().parent / 's1_quadratic_fit.json'


def save_s1_quadratic(mean, cov, path=None, provenance='', fitted_at=None):
    """Write the fitted S1 quadratic to the tracked JSON, and install it.

    Called by component_wais.ipynb after the fit.  Keep the resulting
    file in version control: it is the single source of truth for the
    S1_status_quo shape, and a refit should be visible in review.
    """
    from datetime import datetime, timezone
    mean, cov = set_s1_quadratic(mean, cov, provenance=provenance or 'fresh fit')
    path = Path(path) if path is not None else S1_QUADRATIC_PATH
    payload = {
        'mean': mean.tolist(),
        'cov': cov.tolist(),
        'order': ['a', 'v', 'H0'],
        'units': {'a': 'm/yr^2', 'v': 'm/yr', 'H0': 'm'},
        'provenance': provenance,
        'fitted_at': fitted_at or datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2) + '\n')
    return path


def load_s1_quadratic(path=None):
    """Load and install the S1 quadratic from the tracked JSON."""
    path = Path(path) if path is not None else S1_QUADRATIC_PATH
    if not path.exists():
        raise FileNotFoundError(
            f'{path} does not exist.  Run component_wais.ipynb, whose fit '
            f'cell writes it via save_s1_quadratic().')
    payload = json.loads(path.read_text())
    return set_s1_quadratic(
        payload['mean'], payload['cov'],
        provenance=f"{path.name} ({payload.get('provenance', 'no provenance')}"
                   f", fitted {payload.get('fitted_at', 'unknown')})")


def get_s1_quadratic():
    """Return the ``(mean, cov)`` in force, loading from the store if needed."""
    if _S1_QUADRATIC is None:
        try:
            load_s1_quadratic()
        except (OSError, KeyError, ValueError) as exc:
            raise RuntimeError(
                'The S1 quadratic (a, v, H0) fit has not been supplied.  '
                'Either call set_s1_quadratic(mean, cov) after fitting (what '
                'component_wais.ipynb does), or run component_wais.ipynb so '
                f'{S1_QUADRATIC_PATH.name} is written for load_s1_quadratic() '
                f'to read.  '
                f'Load attempt failed with: {exc}') from exc
    return _S1_QUADRATIC[0], _S1_QUADRATIC[1]


def describe_s1_quadratic():
    """One-line provenance of the S1 quadratic currently in force."""
    if _S1_QUADRATIC is None:
        return 'S1 quadratic: not set'
    mean, _, prov = _S1_QUADRATIC
    return (f'S1 quadratic: a={mean[0] * 1e3:.4f} mm/yr^2, '
            f'v={mean[1] * 1e3:.4f} mm/yr  [{prov}]')

# ---------------------------------------------------------------------------
# S1_status_quo: external extrapolation-error covariance (ISMIP6 emulator)
#
# The S1 quadratic above captures how well IMBIE-3's 45-yr record
# constrains a *fixed* constant-acceleration curve -- i.e. parameter
# uncertainty for an assumed-correct model shape. It does NOT capture the
# risk that a strictly constant-acceleration extrapolation is itself a
# poor description of non-MISI WAIS dynamics 75+ years past the
# calibration window (2150 is ~1.9x the record length past the anchor).
# A 44-47-yr record cannot inform how fast that extrapolation error
# should grow with lead time -- the same reasoning already used elsewhere
# in this project to import, rather than fit, a long-horizon parameter
# from too-short a record (component_ocean.ipynb fixes the deep-ocean
# relaxation time tau_d=150 yr at the Geoffroy et al. (2013) CMIP5
# multi-model mean rather than fit it to the 21-yr ocean record).
#
# Framing (Tarantola, 2005, "Inverse Problem Theory"): total predictive
# covariance = data/parameter covariance (the S1 quadratic cov, well
# constrained by IMBIE-3) + a "theory"/extrapolation-error covariance
# that our own short record cannot resolve, estimated here from an
# independent, external, peer-reviewed source instead: the IPCC AR6
# ISMIP6 emulator's WAIS-specific projection (medium confidence; already
# a "no exotic instability" ice-sheet-model ensemble, and the same one
# shown in component_wais_pdf_exceedance_ipcc_p_s1_sweep.png).
#
# _sample_s1_quadratic_mm() adds ONE eta ~ N(0,1) per Monte Carlo sample
# (not per year, so each trajectory stays smooth) times sqrt(extra_var(t)):
#     H_S1(t) = H_quad(t) + eta * sqrt(extra_var(t))
#     extra_var(t) = max(0, std_ISMIP6(t)^2 - std_ISMIP6(anchor_year)^2)
# By construction extra_var(anchor_year) = 0 exactly (regardless of what
# `anchor_year` is passed at call time), so S1 stays anchored to the
# observed IMBIE value with no added spread there, and only widens for
# years after the anchor.
#
# std_ISMIP6(t) is a quadratic fit (elapsed time since the fit anchor)
# to the 9 native decadal points (2020-2100) of
# data/raw/ipcc_ar6/slr/ar6/global/dist_components/
# icesheets-ipccar6-ismipemuicesheet-ssp245_WAIS_globalsl.nc, read via
# slr_data_readers.read_ipcc_ar6_component(component_dir=..., component_type=
# 'icesheets', sub_component='WAIS', model='ipccar6-ismipemuicesheet',
# scenario='ssp245', convert_to_meters=True). ssp245 used as reference:
# WAIS's ISMIP6 std(2100) varies <5% across SSP1-2.6 to SSP5-8.5
# (59.7-62.1 mm), consistent with S1 being SSP-independent by
# construction.
#
# The fit is on std(t), NOT Var(t)=std(t)^2, directly: a quadratic fit to
# Var(t) itself has a spurious interior minimum (var dips to a negative
# value around 2042, R^2=0.947 despite the bad shape) because ISMIP6's
# variance grows slowly 2020-2050 then much faster after, a shape a
# single quadratic in VARIANCE cannot track without an unphysical dip.
# Fitting std(t) (which grows closer to quadratically in time) and
# squaring afterward is well-behaved (R^2=0.994, residuals <=2.3 mm, and
# the resulting extra_var(t) is exactly monotonically non-decreasing from
# the anchor year out to 2150 -- verified in
# notebooks/scratch_wais_dlm_prototype.py, round 3b).
#
# S1_ISMIP6_FIT_ANCHOR_YEAR is the anchor year (last IMBIE-3 observation,
# wais_year[-1]) AT THE TIME THIS FIT WAS PERFORMED -- it is where the
# elapsed-time variable `e` in the polynomial is centered, not
# necessarily identical to whatever `anchor_year` argument
# _sample_s1_quadratic_mm() receives at call time (extra_var(anchor_year)
# is always exactly 0 regardless, since both std(anchor_year) terms use
# the same fixed curve). To regenerate after IMBIE-3 or the ISMIP6
# emulator file are updated: rerun the fit in
# notebooks/scratch_wais_dlm_prototype.py's build_extra_variance_interpolator
# / fit_and_report_quadratic_extension() (round 3b), or equivalently:
#   df = read_ipcc_ar6_component(CONF_BASE, 'icesheets', 'WAIS',
#            'ipccar6-ismipemuicesheet', 'ssp245', convert_to_meters=True)
#   e = df.index.values - anchor_year   # anchor_year = wais_year[-1]
#   coeffs = np.polyfit(e, df['std'].values, 2)
S1_ISMIP6_FIT_ANCHOR_YEAR = 2023.5
S1_ISMIP6_STD_COEFFS = np.array([1.00993537e-05, -5.14603524e-05,
                                  3.27977164e-03])  # meters; std(t) = c2*e^2+c1*e+c0


def _s1_ismip6_std_m(years):
    """std_ISMIP6(t) [m], quadratic in elapsed time since
    S1_ISMIP6_FIT_ANCHOR_YEAR (see S1_ISMIP6_STD_COEFFS comment above)."""
    e = np.asarray(years, dtype=float) - S1_ISMIP6_FIT_ANCHOR_YEAR
    c2, c1, c0 = S1_ISMIP6_STD_COEFFS
    return c2 * e ** 2 + c1 * e + c0


def _s1_ismip6_extra_var_m2(years, anchor_year):
    """extra_var(t) = max(0, std_ISMIP6(t)^2 - std_ISMIP6(anchor_year)^2),
    the ISMIP6-external extrapolation-error variance added to S1 beyond
    `anchor_year` (see S1_ISMIP6_STD_COEFFS comment block above). Exactly
    0 at t=anchor_year by construction, for any anchor_year.
    """
    std_t = _s1_ismip6_std_m(years)
    std_anchor = _s1_ismip6_std_m(anchor_year)
    return np.maximum(0.0, std_t ** 2 - std_anchor ** 2)


def _sample_s1_quadratic_mm(n_samples, rng, years, anchor_year=None):
    """Direct-posterior S1_status_quo draws (mm): the quadratic-in-time
    fit to observed IMBIE WAIS mass balance (see get_s1_quadratic())
    PLUS an external ISMIP6-emulator-derived extrapolation-error term
    (S1_ISMIP6_STD_COEFFS) that grows the spread at long lead times
    beyond what IMBIE-3's 45-yr record alone can constrain -- see the
    S1_ISMIP6_STD_COEFFS comment block above for the full derivation and
    Tarantola (2005) framing. No rheology correction is applied to either
    term: that correction addresses ice-sheet-model (n=3 vs n≈4)
    structural bias, which does not apply to a statistical fit of real
    satellite-observed mass balance or to an independent projection
    ensemble's own reported uncertainty.

    This is the single shared implementation used by both
    sample_a4_wais_endpoint() and sample_a4_wais_trajectories() so the
    two sampling paths stay consistent.

    Parameters
    ----------
    n_samples : int
    rng : numpy.random.Generator
    years : array-like
        Query years.
    anchor_year : float or None
        Year at which the ISMIP6 extra-variance term is exactly zero
        (S1 stays anchored to the observed IMBIE value there with no
        added spread). Defaults to S1_ISMIP6_FIT_ANCHOR_YEAR when not
        given (used by sample_a4_wais_endpoint(), which has no separate
        anchor-splicing logic); sample_a4_wais_trajectories() passes its
        own dynamic anchor_year (the last IMBIE-3 observation year at
        call time) so the extra term vanishes exactly at the same point
        the trajectory itself splices onto the observed record.

    Returns
    -------
    ndarray, shape (n_samples, len(years))
        H(year) in mm, relative to BASELINE_YEAR.
    """
    if anchor_year is None:
        anchor_year = S1_ISMIP6_FIT_ANCHOR_YEAR
    _s1_mean, _s1_cov = get_s1_quadratic()
    draws = rng.multivariate_normal(_s1_mean, _s1_cov,
                                     size=n_samples)
    a, v, H0 = draws[:, 0], draws[:, 1], draws[:, 2]
    tau = np.asarray(years, dtype=float) - BASELINE_YEAR
    H_m = (0.5 * a[:, None] * tau[None, :] ** 2
           + v[:, None] * tau[None, :] + H0[:, None])

    extra_var_m2 = _s1_ismip6_extra_var_m2(years, anchor_year)  # shape (len(years),)
    eta = rng.standard_normal(n_samples)  # ONE draw per sample -- smooth trajectories
    H_m = H_m + eta[:, None] * np.sqrt(extra_var_m2)[None, :]

    return H_m * M_TO_MM


# A1 rheology correction (n=3 -> n=4): Martin et al. (in press)
RHEOLOGY_FACTOR_MEDIAN = 1.28
RHEOLOGY_FACTOR_SIGMA = 0.07

# Glen's law exponent. Rheology sensitivity deprecated (2026-09-18): n is held
# constant at N_REF, so Mode B gives R(n) = 1 and beta(n) = beta_ref exactly --
# S2's endpoint bounds and beta_loc in A4_SCENARIOS are used as stated.
# Previously n ~ N(4.1, 0.4^2) (Millstein, Minchew, & Pegler 2022).
N_OBS_MEAN = 3.0
N_OBS_SIGMA = 0.0
N_REF = 3         # reference exponent used by ISMIP6 and literature scenarios

# Rheology sensitivity: fractional increase in SLR per unit increase in n
# Martin et al. (2026): 21-35% for Δn=1 → r₀ ≈ 0.28
RHEOLOGY_SENSITIVITY = 0.28

# S2_fast_wais near-term blend: same sigmoid window (T_CENTER, TAU_BLEND)
# as component_forecast.ipynb's aggregate rate-space blend
# (blend_rate_space below), reused here rather than refit so the paper
# applies one blending rule everywhere instead of a second free parameter
# pair to justify.
WAIS_S2_BLEND_T_CENTER = 2035.0
WAIS_S2_BLEND_TAU = 5.0


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
    # Ensure the S1 fit is installed before any scenario dispatch: the S2
    # branch reads A4_SCENARIOS['S2_fast_wais']['low_mm'], which
    # set_s1_quadratic() derives, and an S2-only weighting would otherwise
    # never reach the S1 branch that would have triggered the load.
    get_s1_quadratic()
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
        crng = child_rngs[i]

        if sname == 'S1_status_quo':
            # Direct posterior sampling (quadratic-in-time fit), spliced
            # onto the shared anchor -- see A4_SCENARIOS comment block.
            # No rheology correction, no power-law ramp. anchor_year is
            # passed through so the ISMIP6 extra-variance term (see
            # S1_ISMIP6_STD_COEFFS) is exactly zero at this function's own
            # splice anchor, not S1_ISMIP6_FIT_ANCHOR_YEAR's default.
            h_model = _sample_s1_quadratic_mm(n_s, crng, [anchor_year, year],
                                               anchor_year=anchor_year)
            anchor_i = anchor_draws[mask]
            samples[mask] = anchor_i + (h_model[:, 1] - h_model[:, 0])
            continue

        s = A4_SCENARIOS[sname]

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
    perturb scenario parameters without trajectory or anchor logic.
    S2_fast_wais's endpoint is sampled from the log-skew-normal, multiplied
    by the rheology correction. S1_status_quo is instead sampled directly
    from its quadratic-in-time posterior (see get_s1_quadratic() and
    _sample_s1_quadratic_mm above) with no rheology correction -- see the
    A4_SCENARIOS comment block for why.

    Parameters
    ----------
    n_samples : int
    rng : numpy.random.Generator
    rheology_mode : {'A', 'B'}
    scenario_overrides : dict or None
        Per-scenario parameter overrides.  Keys are scenario names
        (e.g. 'S2_fast_wais'); values are dicts that can override any of
        'P', 'low_mm', 'high_mm', 'alpha'.  Missing keys use defaults
        from A4_SCENARIOS.  You can also pass a top-level key 'weights'
        mapping scenario names to new probabilities (must sum to 1).
        S1_status_quo has no 'low_mm'/'high_mm'/'alpha' to override --
        its distribution is fixed by the installed S1 quadratic fit.

    Returns
    -------
    samples_m : ndarray, shape (n_samples,)
        Endpoint samples in meters at 2100.
    """
    # Ensure the S1 fit is installed before any scenario dispatch: the S2
    # branch reads A4_SCENARIOS['S2_fast_wais']['low_mm'], which
    # set_s1_quadratic() derives, and an S2-only weighting would otherwise
    # never reach the S1 branch that would have triggered the load.
    get_s1_quadratic()
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
        crng = child_rngs[i]

        if sname == 'S1_status_quo':
            # No anchor-splicing logic in this lightweight endpoint
            # wrapper, so anchor_year is left at its default
            # (S1_ISMIP6_FIT_ANCHOR_YEAR) -- see
            # _sample_s1_quadratic_mm's docstring.
            samples[mask] = _sample_s1_quadratic_mm(n_s, crng, [2100.0])[:, 0]
            continue

        s = eff[sname]
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

    S1_status_quo is the exception: it has no MISI by construction, so its
    post-anchor shape is not the power-law ramp but the (a, v, H0)
    quadratic-in-time posterior drawn once per S1 sample (see
    get_s1_quadratic() above), spliced onto the same shared
    ``anchor_draws`` every scenario uses so trajectories stay continuous
    at ``anchor_year``: samples_mm = anchor_draws + (H_model(t) -
    H_model(anchor_year)). No rheology correction is applied to S1.

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
    # Ensure the S1 fit is installed before any scenario dispatch: the S2
    # branch reads A4_SCENARIOS['S2_fast_wais']['low_mm'], which
    # set_s1_quadratic() derives, and an S2-only weighting would otherwise
    # never reach the S1 branch that would have triggered the load.
    get_s1_quadratic()
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

    # S1_status_quo's post-anchor curve, precomputed per-sample across
    # `years` (plus anchor_year, to splice) from its own quadratic
    # posterior draw. Populated only for S1-masked rows below; unused
    # elsewhere.
    s1_idx = scenario_names.index('S1_status_quo')
    s1_anchor_model_mm = np.zeros(n_samples)
    s1_curve_mm = np.zeros((n_samples, n_years))

    # Spawn independent child RNGs per scenario
    child_rngs = rng.spawn(len(scenario_names))

    for i, sname in enumerate(scenario_names):
        mask = scenario_idx == i
        n_s = mask.sum()
        if n_s == 0:
            continue
        crng = child_rngs[i]

        if sname == 'S1_status_quo':
            eval_years = np.concatenate([[anchor_year, 2100.0], years])
            # anchor_year passed through so the ISMIP6 extra-variance term
            # (S1_ISMIP6_STD_COEFFS) is exactly zero at this trajectory's
            # own splice anchor -- see _sample_s1_quadratic_mm docstring.
            h_model = _sample_s1_quadratic_mm(n_s, crng, eval_years,
                                               anchor_year=anchor_year)
            s1_anchor_model_mm[mask] = h_model[:, 0]
            h2100[mask] = anchor_draws[mask] + (h_model[:, 1] - h_model[:, 0])
            s1_curve_mm[mask, :] = h_model[:, 2:]
            continue

        s = A4_SCENARIOS[sname]

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

            # S1_status_quo: override with its own quadratic-in-time shape
            # (spliced onto the shared anchor value) instead of the
            # power-law ramp above, which does not apply to S1.
            s1_mask = scenario_idx == s1_idx
            samples_mm[s1_mask, j] = (anchor_draws[s1_mask]
                                       + (s1_curve_mm[s1_mask, j]
                                          - s1_anchor_model_mm[s1_mask]))

    # ── S2_fast_wais: rate-space blend of the power-law path with the
    # IMBIE quadratic (naive "nothing new happens" continuation), so the
    # near-term rate is data-anchored instead of forced to zero at the
    # anchor year (the artifact of a pure power-law ramp with beta > 1).
    # Uses the same blend_rate_space() machinery and sigmoid window as
    # component_forecast.ipynb's aggregate blend, for one consistent
    # blending rule across the paper. The blended path is then rescaled
    # so H(2100) still equals the independently-drawn h2100 exactly: the
    # blend reshapes *how* the trajectory gets to 2100, not the assessed
    # endpoint distribution (which stays the S1-p99-pinned/mixture-p95-
    # capped 94-1000 mm skew-normal used by sample_a4_wais_endpoint()
    # elsewhere). S1 is
    # untouched -- it already *is* the quadratic.
    beta_eff_2035 = np.full(n_samples, np.nan)
    beta_eff_2050 = np.full(n_samples, np.nan)
    s2_idx = scenario_names.index('S2_fast_wais')
    s2_mask = scenario_idx == s2_idx
    n_s2 = int(s2_mask.sum())
    fmask = years >= anchor_year

    if n_s2 > 0 and np.any(fmask):
        quad_rng = child_rngs[s2_idx].spawn(1)[0]
        quad_draws = quad_rng.multivariate_normal(
            *get_s1_quadratic(), size=n_s2)
        a_q, v_q, H0_q = quad_draws[:, 0], quad_draws[:, 1], quad_draws[:, 2]
        tau_q = years - BASELINE_YEAR
        # quad_level_mm is passed only because blend_rate_space's signature
        # takes it (sq_level_samples_rb); the function never reads it --
        # only quad_rate_mm and the target's own level (target_mm) drive
        # the blend. Kept for signature parity / potential future use.
        quad_level_mm = (0.5 * a_q[:, None] * tau_q[None, :] ** 2
                          + v_q[:, None] * tau_q[None, :]
                          + H0_q[:, None]) * M_TO_MM
        quad_rate_mm = (a_q[:, None] * tau_q[None, :] + v_q[:, None]) * M_TO_MM

        target_mm = samples_mm[s2_mask, :]  # pre-blend: obs + power-law ramp
        anchor_s2 = anchor_draws[s2_mask]
        h2100_s2 = h2100[s2_mask]

        blended_mm, f_years, _ = blend_rate_space(
            years, target_mm, quad_rate_mm, quad_level_mm, years,
            anchor_year, anchor_s2,
            WAIS_S2_BLEND_T_CENTER, WAIS_S2_BLEND_TAU,
        )

        # Pin H(2100) exactly to h2100_s2 (see docstring above).
        i2100 = int(np.argmin(np.abs(f_years - 2100.0)))
        denom = blended_mm[:, i2100] - anchor_s2
        safe = np.abs(denom) > 1e-9
        scale = np.zeros(n_s2)
        scale[safe] = (h2100_s2[safe] - anchor_s2[safe]) / denom[safe]
        rescaled_mm = (anchor_s2[:, None]
                        + (blended_mm - anchor_s2[:, None]) * scale[:, None])

        samples_mm[np.ix_(s2_mask, fmask)] = rescaled_mm

        # Effective exponent of the blended+rescaled path -- a reportable
        # number for the text, replacing beta_ref=1.84 as "the" trajectory
        # shape (beta_arr above is now only the shape of the pre-blend
        # target, not the realized path). Evaluated at two points:
        # WAIS_S2_BLEND_T_CENTER itself (2035, inside the transition --
        # where the blend actually differs from the target) and 2050
        # (past the transition, where it necessarily reconverges toward
        # beta_arr and mainly serves as a sanity check on that
        # reconvergence). Undefined (NaN) wherever the blended level at
        # that year is at or below the anchor -- do not fill with a
        # clipped/placeholder value, since log(remaining fraction) is only
        # meaningful for a positive remaining fraction.
        def _beta_eff_at(target_year):
            i = int(np.argmin(np.abs(f_years - target_year)))
            tau_i = (f_years[i] - anchor_year) / (2100.0 - anchor_year)
            remain_frac = (rescaled_mm[:, i] - anchor_s2) / (h2100_s2 - anchor_s2)
            out = np.full(n_s2, np.nan)
            pos = remain_frac > 0
            out[pos] = np.log(remain_frac[pos]) / np.log(tau_i)
            return out

        beta_eff_2035[s2_mask] = _beta_eff_at(WAIS_S2_BLEND_T_CENTER)
        beta_eff_2050[s2_mask] = _beta_eff_at(2050.0)

    params = {
        'scenario_idx': scenario_idx,
        'h2100_mm': h2100,
        'beta': beta_arr,
        'beta_eff_2035': beta_eff_2035,
        'beta_eff_2050': beta_eff_2050,
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
    baseline_year=2000.0,
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
    from bayesian_models import build_level_design_vectors, solve_twolayer_ode

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


def read_ipcc_workflow_samples(workflow_base, wf_id, ssp_code, fname, year=2100):
    """Read raw Monte Carlo samples for a single IPCC AR6 FACTS workflow.

    Unlike the confidence-level p-box files (``read_ipcc_component_nc``),
    these ``full_sample_workflows`` files hold one constituent model's own
    Monte Carlo ensemble (e.g. the ISMIP6 emulator or LARMIP-2 alone) and
    so represent a genuine single-model distribution, not an envelope
    across models.

    Parameters
    ----------
    workflow_base : str
        Base directory for ``full_sample_workflows``.
    wf_id : str
        FACTS workflow identifier, e.g. ``'wf_1e'`` (ISMIP6 emulator),
        ``'wf_2e'`` (LARMIP-2), ``'wf_3e'`` (DeConto et al. 2021 MICI),
        ``'wf_4'`` (Bamber et al. 2019 SEJ).
    ssp_code : str
        E.g. ``'ssp585'``.
    fname : str
        Workflow-specific filename, e.g.
        ``'icesheets-dp20-icesheet-ssp585_AIS_globalsl.nc'``.
    year : int
        Target year to extract (nearest available year is used).

    Returns
    -------
    ndarray or None
        1-D array of samples in mm at the requested year. ``None`` if the
        file is not found.
    """
    fpath = os.path.join(workflow_base, wf_id, ssp_code, fname)
    if not os.path.exists(fpath):
        return None
    ds = nc.Dataset(fpath, 'r')
    years = ds.variables['years'][:]
    yr_idx = np.argmin(np.abs(years - year))
    samples = np.array(ds.variables['sea_level_change'][:, yr_idx, 0], dtype=float)
    ds.close()
    return samples


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

# ISMIP6 Antarctica core experiments (Seroussi et al. 2020, Table 1). All are
# forced by CMIP5 AOGCMs under RCP2.6 or RCP8.5; none is an SSP scenario.
# melt: ocean-melt parameterization ('open' or 'standard'); sensitivity: gamma0
# percentile of the standard parameterization ('PIGL' = Pine Island calibration);
# collapse: ice-shelf collapse imposed.
ISMIP6_EXPERIMENTS = {
    'exp01': dict(aogcm='NorESM1-M', scenario='RCP8.5', melt='open', sensitivity='medium', collapse=False),
    'exp02': dict(aogcm='MIROC-ESM-CHEM', scenario='RCP8.5', melt='open', sensitivity='medium', collapse=False),
    'exp03': dict(aogcm='NorESM1-M', scenario='RCP2.6', melt='open', sensitivity='medium', collapse=False),
    'exp04': dict(aogcm='CCSM4', scenario='RCP8.5', melt='open', sensitivity='medium', collapse=False),
    'exp05': dict(aogcm='NorESM1-M', scenario='RCP8.5', melt='standard', sensitivity='medium', collapse=False),
    'exp06': dict(aogcm='MIROC-ESM-CHEM', scenario='RCP8.5', melt='standard', sensitivity='medium', collapse=False),
    'exp07': dict(aogcm='NorESM1-M', scenario='RCP2.6', melt='standard', sensitivity='medium', collapse=False),
    'exp08': dict(aogcm='CCSM4', scenario='RCP8.5', melt='standard', sensitivity='medium', collapse=False),
    'exp09': dict(aogcm='NorESM1-M', scenario='RCP8.5', melt='standard', sensitivity='high', collapse=False),
    'exp10': dict(aogcm='NorESM1-M', scenario='RCP8.5', melt='standard', sensitivity='low', collapse=False),
    'exp11': dict(aogcm='CCSM4', scenario='RCP8.5', melt='open', sensitivity='medium', collapse=True),
    'exp12': dict(aogcm='CCSM4', scenario='RCP8.5', melt='standard', sensitivity='medium', collapse=True),
    'exp13': dict(aogcm='NorESM1-M', scenario='RCP8.5', melt='standard', sensitivity='PIGL', collapse=False),
}

# Closest ISMIP6 analogs to our SSP projections: the standard-melt,
# medium-sensitivity runs without ice-shelf collapse, grouped by forcing
# scenario. ISMIP6 has no core experiments for intermediate scenarios, so
# SSP2-4.5 and SSP3-7.0 have no analog.
ISMIP6_SSP_ANALOG = {
    'SSP1-2.6': ['exp07'],                    # RCP2.6
    'SSP5-8.5': ['exp05', 'exp06', 'exp08'],  # RCP8.5
}

# Default experiment set for read_ismip6_regional: RCP8.5 runs spanning the
# medium, high, low and PIGL melt sensitivities and the ice-shelf-collapse runs.
ISMIP6_DEFAULT_EXPERIMENTS = ['exp05', 'exp06', 'exp09', 'exp10', 'exp11', 'exp12', 'exp13']


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
        If None, reads ISMIP6_DEFAULT_EXPERIMENTS.
    use_ctrl_anomaly : bool
        If True (default), read the ``_minus_ctrl_proj_`` files
        (anomaly from control). If False, read raw ivaf and subtract
        the first time step.

    Returns
    -------
    dict
        ``{(group, model, exp): {'time': ndarray, 'sle_m': ndarray,
        'scenario': str, 'label': str}}``
        where ``sle_m`` is in meters (positive = sea level rise),
        ``scenario`` is the CMIP5 forcing scenario (e.g. 'RCP8.5') and
        ``label`` is '<AOGCM> <scenario>'.
    """
    if experiments is None:
        experiments = list(ISMIP6_DEFAULT_EXPERIMENTS)

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

                info = ISMIP6_EXPERIMENTS.get(exp, {})
                scenario = info.get('scenario', exp)
                results[(group_name, model_name, exp)] = {
                    'time': time,
                    'sle_m': sle_m,
                    'scenario': scenario,
                    'label': f"{info.get('aogcm', exp)} {scenario}",
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
