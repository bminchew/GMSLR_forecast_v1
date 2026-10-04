"""Greenland SMB emulator fitted to regional climate model projections.

Replaces the literature-sensitivity SMB model (GREENLAND_SMB in
smb_projections.py) for Greenland.  Training data are the annual SMB of
RACMO2.3p2 and MAR v3.12 forced by one CESM2 historical + SSP5-8.5 simulation
(Noel et al. 2021, GRL, doi:10.1029/2020GL090471; Glaude et al. 2024, GRL,
doi:10.1029/2024GL111902), integrated over the contiguous ice sheet only
(Promicemask == 3; peripheral glaciers are in the glacier component).

For each regional model,

    dM(t) = b1 x(t) + b2 x(t)^2 + M + eps(t),    eps ~ AR(1),

where dM is the SMB anomaly (Gt/yr) and x the CESM2 ensemble-mean Greenland
near-surface temperature anomaly (K), both relative to 1995-2005.  The
posterior is analytic: Prais-Winsten whitening for each AR(1) coefficient on a
grid, a normal-inverse-gamma prior, and the grid mixed by marginal likelihood.

In projections x = AA * dGMST, with the Arctic-amplification draw shared with
the discharge pathway, and SMB = M0 + b1 x + b2 x^2 with M0 the observed
1995-2005 SMB.  Each Monte Carlo member uses one regional model (equal weight)
and one posterior draw of (b1, b2).

HIRHAM5 is excluded: forced by CESM2 its 1995-2005 SMB is 39 Gt/yr against
350 Gt/yr forced by reanalysis (Mankoff et al. 2021).

Data files (produced by scripts/integrate_glaude2024_smb.py and downloaded
from Zenodo 10.5281/zenodo.4289959):
  data/raw/ice_sheets/greenland/glaude2024/glaude2024_integrated_smb.csv
  data/raw/ice_sheets/greenland/noel2021/TGrIS_CESM2_*.dat
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

try:
    from slr_forecast.config import GT_TO_M_SLE
except ImportError:
    GT_TO_M_SLE = 1.0 / 362500.0

_ROOT = Path(__file__).resolve().parents[1]
NOEL_DIR = _ROOT / 'data/raw/ice_sheets/greenland/noel2021'
GLAUDE_CSV = _ROOT / 'data/raw/ice_sheets/greenland/glaude2024/glaude2024_integrated_smb.csv'

BASE = (1995, 2005)
RCMS = ('RACMO', 'MAR')
RHO_GRID = np.linspace(-0.9, 0.9, 181)
SIGMA0 = 100.0                                   # Gt/yr, prior-typical noise
A0, B0 = 2.0, SIGMA0**2                          # inverse-gamma prior on sigma^2
PRIOR_SD = np.array([2000.0, 1000.0, 500.0])     # (M, b1, b2) prior sd at SIGMA0

_HIST = ['HIST-parent'] + [f'HIST-{i}' for i in range(1, 12)]
_PROJ = ['SSP5-8.5-parent', 'SSP5-8.5-1', 'SSP5-8.5-2', 'SSP3-7.0-1', 'SSP3-7.0-2',
         'SSP2-4.5-1', 'SSP2-4.5-2', 'SSP2-4.5-3', 'SSP1-2.6-1', 'SSP1-2.6-2']


def _read_dat(fname, names):
    df = pd.read_csv(NOEL_DIR / fname, sep=r'\s+', skiprows=1, header=None)
    df = df.iloc[:, :len(names) + 1]
    df.columns = ['year'] + names
    return df.set_index('year').astype(float)


def _anom(s):
    return s - s.loc[BASE[0]:BASE[1]].mean()


def training_data():
    """Return x (CESM2 ensemble-mean T_GrIS anomaly, K) and per-RCM SMB
    anomalies (Gt/yr), both as pd.Series indexed by year."""
    T_h = _read_dat('TGrIS_CESM2_Historical_1950-2014.dat', _HIST)
    T_p = _read_dat('TGrIS_CESM2_Projections_2015-2099.dat', _PROJ)
    ens = pd.concat([T_h.mean(axis=1),
                     T_p[['SSP5-8.5-parent', 'SSP5-8.5-1', 'SSP5-8.5-2']].mean(axis=1)])
    x = _anom(ens)
    g = pd.read_csv(GLAUDE_CSV, index_col=0)
    y = {m: _anom(g[f'{m}_sheet'].dropna()) for m in ('RACMO', 'MAR', 'HIRHAM')}
    return x, y


def _prais_winsten(Z, rho):
    Zs = np.empty_like(Z, dtype=float)
    Zs[0] = np.sqrt(1.0 - rho**2) * Z[0]
    Zs[1:] = Z[1:] - rho * Z[:-1]
    return Zs


def fit_quadratic_ar1(x, y, n_draws, rng):
    """Analytic posterior for y = M + b1 x + b2 x^2 + AR(1) noise.

    Returns a dict with 'beta' (n_draws, 3) = (M, b1, b2), 'sigma', 'rho'
    draws and the log evidence.
    """
    X = np.column_stack([np.ones_like(x), x, x**2])
    V0 = np.diag((PRIOR_SD / SIGMA0) ** 2)
    V0i = np.linalg.inv(V0)
    post, logml = [], []
    for rho in RHO_GRID:
        Xs, ys = _prais_winsten(X, rho), _prais_winsten(y, rho)
        Vn = np.linalg.inv(V0i + Xs.T @ Xs)
        mn = Vn @ (Xs.T @ ys)
        an = A0 + 0.5 * len(y)
        bn = B0 + 0.5 * (ys @ ys - mn @ np.linalg.solve(Vn, mn))
        lml = (-0.5 * len(y) * np.log(2 * np.pi)
               + 0.5 * (np.linalg.slogdet(Vn)[1] - np.linalg.slogdet(V0)[1])
               + A0 * np.log(B0) - an * np.log(bn) + gammaln(an) - gammaln(A0)
               + 0.5 * np.log(1 - rho**2))
        post.append((mn, Vn, an, bn))
        logml.append(lml)
    logml = np.array(logml)
    w = np.exp(logml - logml.max()); w /= w.sum()
    k = rng.choice(len(RHO_GRID), size=n_draws, p=w)
    beta = np.empty((n_draws, 3)); sig = np.empty(n_draws)
    for j in np.unique(k):
        idx = np.where(k == j)[0]
        mn, Vn, an, bn = post[j]
        s2 = bn / rng.gamma(an, 1.0, size=len(idx))
        L = np.linalg.cholesky(Vn)
        beta[idx] = mn + np.sqrt(s2)[:, None] * (rng.standard_normal((len(idx), 3)) @ L.T)
        sig[idx] = np.sqrt(s2)
    log_evidence = logml.max() + np.log(np.mean(np.exp(logml - logml.max())))
    return {'beta': beta, 'sigma': sig, 'rho': RHO_GRID[k], 'log_evidence': log_evidence}


def fit_smb_emulator(n_samples, seed=600, rcms=RCMS):
    """Fit each regional model and assign one model and one posterior draw to
    each Monte Carlo member (models alternate, so the weights are equal).

    Returns dict: 'b1', 'b2' (n_samples,), 'rcm' (n_samples,) labels, and
    'fits' {rcm: fit dict} for diagnostics.
    """
    rng = np.random.default_rng(seed)
    x, y = training_data()
    fits = {}
    for m in rcms:
        yrs = y[m].index.intersection(x.index)
        fits[m] = fit_quadratic_ar1(x.loc[yrs].values, y[m].loc[yrs].values,
                                    n_samples, rng)
        fits[m]['years'] = (int(yrs.min()), int(yrs.max()))
    rcm = np.array([rcms[i % len(rcms)] for i in range(n_samples)])
    b1 = np.empty(n_samples); b2 = np.empty(n_samples)
    for m in rcms:
        idx = np.where(rcm == m)[0]
        b1[idx] = fits[m]['beta'][idx, 1]
        b2[idx] = fits[m]['beta'][idx, 2]
    return {'b1': b1, 'b2': b2, 'rcm': rcm, 'fits': fits}


def project_smb_emulator(emulator, T_proj, time_proj, aa_draws, M0,
                         T_offsets=None, baseline_year=None):
    """Project Greenland SMB as a cumulative sea-level contribution.

    Parameters
    ----------
    emulator : dict from fit_smb_emulator
    T_proj : {ssp: ndarray} annual GMST anomaly rel. 1995-2005 on time_proj
    time_proj : ndarray of years
    aa_draws : (n_samples,) Arctic-amplification draws shared with discharge
    M0 : float, observed 1995-2005 SMB (Gt/yr, mass-gain convention)
    T_offsets : {ssp: (n_samples, n_times)} per-member warming-path offsets
    baseline_year : rebase cumulative values to zero at this year

    Returns {ssp: {'samples' (m SLE), 'median', 'p5', 'p17', 'p83', 'p95',
    'rate_median' (m SLE/yr)}} -- the same structure as project_smb_ensemble.
    """
    b1, b2 = emulator['b1'][:, None], emulator['b2'][:, None]
    aa = np.asarray(aa_draws)[:, None]
    dt = np.diff(time_proj, prepend=time_proj[0] - 1.0)
    out = {}
    for ssp, T in T_proj.items():
        dT = np.asarray(T)[None, :]
        if T_offsets is not None and ssp in T_offsets:
            dT = dT + T_offsets[ssp]
        x = aa * dT
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

    def __init__(self, emulator, M0):
        self.reference = ('Emulator of RACMO2.3p2 and MAR v3.12 forced by CESM2 '
                          'SSP5-8.5 (Noel et al. 2021; Glaude et al. 2024)')
        self.temperature_frame = 'Greenland T = AA x GMST'
        self.SMB_0 = float(M0)
        self.extra_attrs = {}
        for m, f in emulator['fits'].items():
            self.extra_attrs[f'{m}_b1_median'] = float(np.median(f['beta'][:, 1]))
            self.extra_attrs[f'{m}_b2_median'] = float(np.median(f['beta'][:, 2]))
