"""Sensitivity of the Greenland SMB emulator to noise that grows with warming.

The production emulator (notebooks/smb_emulator.py) assumes AR(1) residuals
with a constant innovation sd.  In the MAR runs the innovation sd grows with
the GMST anomaly x, so the hot, noisy years carry more weight than they
should.  Here the innovations are given sd sigma * (1 + c x), with c on a grid
mixed by marginal likelihood, and the integrated SMB anomaly is compared with
the production fit for each GCM and for the equal-weight mixture.

SMB anomaly integral: 2000-2100, driven by the AR6 median GMST paths (as in
component_greenland.ipynb), relative to M0; mm SLE, + = more sea-level rise.

Usage:  python scripts/hetero_sensitivity_mar.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
import smb_emulator as S

H5 = ROOT / 'data/processed/slr_processed_data.h5'
SSPS = {'SSP1-2.6': 'SSP1_2_6', 'SSP2-4.5': 'SSP2_4_5',
        'SSP3-7.0': 'SSP3_7_0', 'SSP5-8.5': 'SSP5_8_5'}
C_GRID = np.linspace(0.0, 1.5, 16)
N = 2000
GT_PER_MM = 362.5


def fit_hetero(x, y, starts, n, rng):
    """Like S.fit_quadratic_ar1, but innovations have sd sigma * (1 + c x)."""
    X = np.column_stack([np.ones_like(x), x, x**2])
    V0 = np.diag((S.PRIOR_SD / S.SIGMA0) ** 2); V0i = np.linalg.inv(V0)
    ld0 = np.linalg.slogdet(V0)[1]
    post, lz, cs = [], [], []
    for c in C_GRID:
        w = 1.0 + c * x
        if np.any(w <= 0):
            continue
        for rho in S.RHO_GRID:
            Xs = S._prais_winsten(X, rho, starts) / w[:, None]
            ys = S._prais_winsten(y, rho, starts) / w
            Vn = np.linalg.inv(V0i + Xs.T @ Xs); mn = Vn @ (Xs.T @ ys)
            an = S.A0 + 0.5 * len(y)
            bn = S.B0 + 0.5 * (ys @ ys - mn @ np.linalg.solve(Vn, mn))
            post.append((mn, Vn, an, bn)); cs.append(c)
            lz.append(-0.5 * len(y) * np.log(2 * np.pi)
                      + 0.5 * (np.linalg.slogdet(Vn)[1] - ld0)
                      + S.A0 * np.log(S.B0) - an * np.log(bn) + gammaln(an) - gammaln(S.A0)
                      + 0.5 * len(starts) * np.log(1 - rho**2) - np.log(w).sum())
    lz = np.array(lz); p = np.exp(lz - logsumexp(lz))
    k = rng.choice(len(lz), n, p=p)
    beta = np.empty((n, 3))
    for j in np.unique(k):
        idx = np.where(k == j)[0]; mn, Vn, an, bn = post[j]
        s2 = bn / rng.gamma(an, 1.0, len(idx))
        beta[idx] = mn + np.sqrt(s2)[:, None] * (rng.standard_normal((len(idx), 3))
                                                  @ np.linalg.cholesky(Vn).T)
    return beta, np.array(cs)[k]


def ar6_drivers():
    """AR6 median GMST anomaly rel. 1995-2005 on 2000-2100, as in the notebook."""
    be = pd.read_hdf(H5, 'harmonized/df_berkeley_h')
    tt = (be.index.year + (be.index.month - 0.5) / 12).values
    tm = be['temperature'].values
    hist = pd.read_hdf(H5, 'projections/temp/Historical')
    hb = hist.loc[(hist.decimal_year >= 1995) & (hist.decimal_year <= 2005), 'temperature'].mean()
    off = hb - tm[(tt >= 1995) & (tt < 2006)].mean()
    obs_end = int(np.floor(tt[-1]))
    t_obs = pd.Series(tm).groupby(np.floor(tt).astype(int)).mean()
    years = np.arange(1850, 2101)
    out = {}
    for ssp, key in SSPS.items():
        d = pd.read_hdf(H5, f'projections/temp/{key}')
        c = pd.concat([hist[hist.decimal_year < 2015], d])
        T = np.interp(years, c.decimal_year, c.temperature - off)
        o = years <= obs_end
        T[o] = t_obs.reindex(years[o]).values
        out[ssp] = S.smooth_observed(T, years, obs_end)[years >= 2000]
    return out


def integral_mm(beta, x):
    """Median SMB anomaly integral (mm SLE, + = more SLR) for draws beta."""
    smb = beta[:, [1]] * x[None, :] + beta[:, [2]] * x[None, :]**2
    return -smb.sum(axis=1) / GT_PER_MM


def main():
    mar, temp = S.load_data()
    drv = ar6_drivers()
    rng = np.random.default_rng(11)
    base, het = {}, {}
    print('Per GCM: b1/b2 medians (constant -> growing noise), posterior median c, '
          'and the integrated 2000-2100 SMB anomaly (mm SLE) under each SSP')
    for g in S.GCMS:
        segs = S.training_segments(g, mar, temp)
        starts = np.cumsum([0] + [len(s[1]) for s in segs[:-1]])
        y = np.concatenate([s[1].values for s in segs])
        x = np.concatenate([xs.loc[ys.index].values for _, ys, xs in segs])
        base[g] = S.fit_quadratic_ar1(x, y, N, rng, starts=starts)['beta']
        het[g], c = fit_hetero(x, y, starts, N, rng)
        mb, mh = np.median(base[g], 0), np.median(het[g], 0)
        row = '  '.join(f'{s} {np.median(integral_mm(base[g], drv[s])):5.0f}->'
                        f'{np.median(integral_mm(het[g], drv[s])):5.0f}' for s in SSPS)
        print(f'{g:14s} b1 {mb[1]:5.0f}->{mh[1]:5.0f}  b2 {mb[2]:6.1f}->{mh[2]:6.1f}  '
              f'c {np.median(c):.2f}  | {row}')
    print('\nEqual-weight mixture, integrated 2000-2100 SMB anomaly, median [5th, 95th] (mm SLE):')
    for s in SSPS:
        a = np.concatenate([integral_mm(base[g][:N // len(S.GCMS)], drv[s]) for g in S.GCMS])
        b = np.concatenate([integral_mm(het[g][:N // len(S.GCMS)], drv[s]) for g in S.GCMS])
        print(f'  {s}: constant noise {np.median(a):5.1f} [{np.percentile(a, 5):.1f}, '
              f'{np.percentile(a, 95):.1f}]   growing noise {np.median(b):5.1f} '
              f'[{np.percentile(b, 5):.1f}, {np.percentile(b, 95):.1f}]   '
              f'difference {np.median(b) - np.median(a):+.1f}')


if __name__ == '__main__':
    main()
