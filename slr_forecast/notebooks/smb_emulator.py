"""Greenland SMB statistical emulator of MAR v3.12 driven by several CMIP6 GCMs.

Replaces the literature-sensitivity SMB model (GREENLAND_SMB in
smb_projections.py) for Greenland.  Training data are the annual SMB of
MARv3.12 forced by CMIP6 GCMs (PROTECT ensemble; also used by Boberg et al.
2026), integrated over the contiguous ice sheet only (peripheral glaciers are
in the glacier component), with each GCM's own global-mean temperature.

For each GCM, all of its runs are fitted jointly (history 1980-2014 plus each
available SSP 2015-2100, each a separate AR(1) segment):

    dM(t) = M + b1 x(t) + b2 x(t)^2 + eps(t),    eps ~ AR(1),

where dM is the SMB anomaly (Gt/yr) relative to the history run's 1995-2005
mean and x the GCM's GMST anomaly relative to 1995-2005, smoothed with an
11-yr centred mean.  The posterior is analytic: Prais-Winsten whitening for
each AR(1) coefficient on a grid, a normal-inverse-gamma prior, and the grid
mixed by marginal likelihood.

In projections SMB = M0 + b1 x + b2 x^2, with x the GMST anomaly and M0 the
observed 1995-2005 SMB.  The ensemble is an equal-weight mixture across GCMs:
each Monte Carlo member uses one GCM and one posterior draw of (b1, b2).  No
scaling or tuning to the observed SMB is applied; the spread between regional
climate models is not sampled (MAR only).

Data files (produced by scripts/extract_mar_protect_smb.py and
scripts/gcm_greenland_temperature.py):
  data/raw/ice_sheets/greenland/mar_protect/mar_protect_integrated_smb.csv
  data/raw/ice_sheets/greenland/mar_protect/gcm_temperature.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

try:
    from slr_forecast.config import GT_TO_M_SLE
except ImportError:
    GT_TO_M_SLE = 1.0 / 362500.0

_ROOT = Path(__file__).resolve().parents[1]
MAR_DIR = _ROOT / 'data/raw/ice_sheets/greenland/mar_protect'
MAR_CSV = MAR_DIR / 'mar_protect_integrated_smb.csv'
TEMP_CSV = MAR_DIR / 'gcm_temperature.csv'

BASE = (1995, 2005)
SMOOTH_YRS = 11                                  # centred running mean of GMST
HIST_YEARS = (1980, 2014)
SSP_YEARS = (2015, 2100)
SCENARIOS = ('ssp126', 'ssp245', 'ssp585')

# GCM name in gcm_temperature.csv -> run prefix in the MAR CSV
GCM_DIR = {'CESM2': 'CESM2-CMIP6', 'MPI-ESM1-2-HR': 'MPI-ESM1-2-HR',
           'NorESM2-MM': 'NorESM2', 'UKESM1-0-LL': 'UKESM1-0-LL-CMIP6',
           'CNRM-CM6-1': 'CNRM-CM6', 'CNRM-ESM2-1': 'CNRM-ESM2',
           'IPSL-CM6A-LR': 'IPSL-CM6A-LR'}
# GCMs in the ensemble.  Listed explicitly so the ensemble does not change
# when new MAR runs are added to the CSV.
GCMS = ('CESM2', 'MPI-ESM1-2-HR', 'NorESM2-MM', 'UKESM1-0-LL',
        'CNRM-CM6-1', 'CNRM-ESM2-1', 'IPSL-CM6A-LR')

RHO_GRID = np.linspace(-0.6, 0.8, 29)
SIGMA0 = 100.0                                   # Gt/yr, prior-typical noise
A0, B0 = 2.0, SIGMA0**2                          # inverse-gamma prior on sigma^2
PRIOR_SD = np.array([2000.0, 1000.0, 1000.0])    # (M, b1, b2) prior sd at SIGMA0


def load_data():
    """Return (mar, temp) DataFrames read from MAR_CSV and TEMP_CSV."""
    mar = pd.read_csv(MAR_CSV).drop_duplicates(['run', 'year'])
    temp = pd.read_csv(TEMP_CSV)
    return mar, temp


def gmst_driver(temp, gcm, scen):
    """GCM GMST anomaly rel. 1995-2005, 11-yr centred mean (pd.Series by year)."""
    t = temp[(temp.gcm == gcm) & temp.experiment.isin(['historical', scen])]
    t = t.set_index('year').sort_index()
    t = t[~t.index.duplicated(keep='last')]['gmst_K']
    s = t - t.loc[BASE[0]:BASE[1]].mean()
    return s.rolling(SMOOTH_YRS, center=True, min_periods=SMOOTH_YRS // 2 + 1).mean()


def smb_series(mar, gcm, scen):
    """MAR history + scenario SMB anomaly rel. the history run's 1995-2005 mean."""
    d = GCM_DIR[gcm]
    h = mar[mar.run == f'{d}-histo'].set_index('year')['sheet']
    s = mar[mar.run == f'{d}-{scen}'].set_index('year')['sheet']
    return pd.concat([h, s]).sort_index() - h.loc[BASE[0]:BASE[1]].mean()


def training_segments(gcm, mar, temp):
    """[(name, y, x)] for the history run and each complete SSP run of a GCM."""
    d = GCM_DIR[gcm]
    n_ssp = SSP_YEARS[1] - SSP_YEARS[0] + 1
    scens = [s for s in SCENARIOS if (mar.run == f'{d}-{s}').sum() == n_ssp]
    if not scens:
        raise ValueError(f'{gcm}: no complete SSP run in {MAR_CSV.name}')
    x = {s: gmst_driver(temp, gcm, s) for s in scens}
    segs = [('history', smb_series(mar, gcm, scens[0]).loc[HIST_YEARS[0]:HIST_YEARS[1]],
             x[scens[0]])]
    segs += [(s, smb_series(mar, gcm, s).loc[SSP_YEARS[0]:SSP_YEARS[1]], x[s]) for s in scens]
    for name, y, xs in segs:
        if xs.reindex(y.index).isna().any():
            raise ValueError(f'{gcm} {name}: GMST missing for some SMB years')
    return segs


def _prais_winsten(Z, rho, starts):
    """Prais-Winsten transform applied separately to each segment."""
    Zs = np.empty_like(Z, dtype=float)
    bounds = list(starts) + [len(Z)]
    for s0, s1 in zip(bounds[:-1], bounds[1:]):
        Zs[s0] = np.sqrt(1.0 - rho**2) * Z[s0]
        Zs[s0 + 1:s1] = Z[s0 + 1:s1] - rho * Z[s0:s1 - 1]
    return Zs


def fit_quadratic_ar1(x, y, n_draws, rng, starts=(0,)):
    """Analytic posterior for y = M + b1 x + b2 x^2 + AR(1) noise.

    `starts` gives the index where each AR(1) segment begins (one shared
    intercept and slope across segments).  Returns a dict with 'beta'
    (n_draws, 3) = (M, b1, b2), 'sigma', 'rho' draws and the log evidence.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    X = np.column_stack([np.ones_like(x), x, x**2])
    V0 = np.diag((PRIOR_SD / SIGMA0) ** 2)
    V0i = np.linalg.inv(V0)
    ld0 = np.linalg.slogdet(V0)[1]
    post, lz = [], []
    for rho in RHO_GRID:
        Xs, ys = _prais_winsten(X, rho, starts), _prais_winsten(y, rho, starts)
        Vn = np.linalg.inv(V0i + Xs.T @ Xs)
        mn = Vn @ (Xs.T @ ys)
        an = A0 + 0.5 * len(y)
        bn = B0 + 0.5 * (ys @ ys - mn @ np.linalg.solve(Vn, mn))
        post.append((mn, Vn, an, bn))
        lz.append(-0.5 * len(y) * np.log(2 * np.pi)
                  + 0.5 * (np.linalg.slogdet(Vn)[1] - ld0)
                  + A0 * np.log(B0) - an * np.log(bn) + gammaln(an) - gammaln(A0)
                  + 0.5 * len(starts) * np.log(1 - rho**2))
    lz = np.array(lz)
    w = np.exp(lz - lz.max()); w /= w.sum()
    k = rng.choice(len(RHO_GRID), size=n_draws, p=w)
    beta = np.empty((n_draws, 3)); sig = np.empty(n_draws)
    for j in np.unique(k):
        idx = np.where(k == j)[0]
        mn, Vn, an, bn = post[j]
        s2 = bn / rng.gamma(an, 1.0, size=len(idx))
        L = np.linalg.cholesky(Vn)
        beta[idx] = mn + np.sqrt(s2)[:, None] * (rng.standard_normal((len(idx), 3)) @ L.T)
        sig[idx] = np.sqrt(s2)
    return {'beta': beta, 'sigma': sig, 'rho': RHO_GRID[k],
            'log_evidence': logsumexp(lz) - np.log(len(RHO_GRID))}


def fit_gcm(gcm, n_draws, rng, mar=None, temp=None):
    """Fit one GCM's MAR runs jointly; adds 'segments' and 'x_range' to the fit."""
    if mar is None or temp is None:
        mar, temp = load_data()
    segs = training_segments(gcm, mar, temp)
    starts = np.cumsum([0] + [len(s[1]) for s in segs[:-1]])
    y = np.concatenate([s[1].values for s in segs])
    x = np.concatenate([xs.loc[ys.index].values for _, ys, xs in segs])
    fit = fit_quadratic_ar1(x, y, n_draws, rng, starts=starts)
    fit['segments'] = [name for name, _, _ in segs]
    fit['x_range'] = (float(x.min()), float(x.max()))
    return fit


def fit_smb_emulator(n_samples, seed=600, gcms=GCMS):
    """Fit each GCM and assign one GCM and one posterior draw to each Monte
    Carlo member (GCMs alternate, so the weights are equal).

    Returns dict: 'b1', 'b2' (n_samples,), 'gcm' (n_samples,) labels, and
    'fits' {gcm: fit dict} for diagnostics.
    """
    rng = np.random.default_rng(seed)
    mar, temp = load_data()
    fits = {g: fit_gcm(g, n_samples, rng, mar, temp) for g in gcms}
    gcm = np.array([gcms[i % len(gcms)] for i in range(n_samples)])
    b1 = np.empty(n_samples); b2 = np.empty(n_samples)
    for g in gcms:
        idx = np.where(gcm == g)[0]
        b1[idx] = fits[g]['beta'][idx, 1]
        b2[idx] = fits[g]['beta'][idx, 2]
    return {'b1': b1, 'b2': b2, 'gcm': gcm, 'fits': fits}


def _centred_mean(T, n=SMOOTH_YRS):
    """Centred running mean that shortens the window at the ends."""
    h = n // 2
    c = np.concatenate([[0.0], np.cumsum(T)])
    k = np.arange(len(T))
    lo, hi = np.maximum(k - h, 0), np.minimum(k + h + 1, len(T))
    return (c[hi] - c[lo]) / (hi - lo)


def smooth_observed(T, years, through):
    """11-yr centred mean of T for years <= through (the observed record,
    matching the training driver); later years are returned unchanged."""
    T = np.asarray(T, dtype=float)
    return np.where(np.asarray(years) <= through, _centred_mean(T), T)


def project_smb_emulator(emulator, T_proj, time_proj, M0, T_offsets=None,
                         baseline_year=None, smooth_through=None):
    """Project Greenland SMB as a cumulative sea-level contribution.

    Parameters
    ----------
    emulator : dict from fit_smb_emulator
    T_proj : {ssp: ndarray} annual-mean GMST anomaly rel. 1995-2005 on time_proj
    time_proj : ndarray of years
    M0 : float or (n_samples,) array, observed 1995-2005 SMB (Gt/yr,
        mass-gain convention); an array gives one anchor per member
    T_offsets : {ssp: (n_samples, n_times)} per-member warming-path offsets,
        zero over the observed record; added after smoothing
    baseline_year : rebase cumulative values to zero at this year
    smooth_through : last year of the observed GMST record.  T_proj is
        replaced by its 11-yr centred mean up to this year, matching the
        training driver; later years (smooth AR6 paths) are used as given.

    Returns {ssp: {'samples' (m SLE), 'median', 'p5', 'p17', 'p83', 'p95',
    'rate_median' (m SLE/yr)}} -- the same structure as project_smb_ensemble.
    """
    b1, b2 = emulator['b1'][:, None], emulator['b2'][:, None]
    M0 = np.asarray(M0, dtype=float)
    if M0.ndim == 1:
        M0 = M0[:, None]
    time_proj = np.asarray(time_proj, dtype=float)
    dt = np.diff(time_proj, prepend=time_proj[0] - 1.0)
    out = {}
    for ssp, T in T_proj.items():
        T = np.asarray(T, dtype=float)
        if smooth_through is not None:
            T = smooth_observed(T, time_proj, smooth_through)
        x = T[None, :]
        if T_offsets is not None and ssp in T_offsets:
            x = x + T_offsets[ssp]
        smb = M0 + b1 * x + b2 * x**2                     # Gt/yr, mass gain
        slr_rate = -smb * GT_TO_M_SLE                      # m SLE/yr
        cum = np.cumsum(slr_rate * dt[None, :], axis=1)
        if baseline_year is not None:
            k = np.argmin(np.abs(time_proj - baseline_year))
            cum = cum - cum[:, k:k + 1]
        out[ssp] = {
            'samples': cum,
            'median': np.median(cum, axis=0),
            'p5': np.percentile(cum, 5, axis=0),
            'p17': np.percentile(cum, 17, axis=0),
            'p83': np.percentile(cum, 83, axis=0),
            'p95': np.percentile(cum, 95, axis=0),
            'rate_median': np.median(slr_rate, axis=0),
        }
    return out


class EmulatorSummary:
    """Attributes stored in component_results.h5 under smb_sensitivity."""

    def __init__(self, emulator, M0, M0_sigma=None):
        gcms = list(emulator['fits'])
        self.reference = ('Statistical emulator of MARv3.12 driven by CMIP6 GCMs '
                          '(PROTECT ensemble): ' + ', '.join(gcms))
        self.temperature_frame = 'driving GCM GMST anomaly rel. 1995-2005, 11-yr mean'
        self.SMB_0 = float(np.mean(M0))
        self.extra_attrs = {'gcms': ','.join(gcms)}
        if M0_sigma is not None:
            self.extra_attrs['SMB_0_sigma'] = float(M0_sigma)
        for g, f in emulator['fits'].items():
            self.extra_attrs[f'{g}_b1_median'] = float(np.median(f['beta'][:, 1]))
            self.extra_attrs[f'{g}_b2_median'] = float(np.median(f['beta'][:, 2]))
            self.extra_attrs[f'{g}_segments'] = ','.join(f['segments'])
