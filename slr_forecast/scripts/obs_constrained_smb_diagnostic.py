"""Diagnostic for option A: take the SMB response shape from the GCM-driven MAR
emulators and its strength from observed SMB.

Two ways to let observations set the strength (they differ at high warming):
  (k) amplitude scaling:    SMB = M + k * f_g(x)
  (r) temperature scaling:  SMB = M + f_g(r x)   (curvature scales as r^2)
with f_g(x) = b1 x + b2 x^2 the posterior-median MAR emulator of GCM g (GMST
quadratic, all scenarios, no lag), and x the observed GMST anomaly rel.
1995-2005 (11-yr centred mean).  Residuals AR(1); (k) analytic, (r) on a grid.

Reported:
  1. k and r per GCM for fit windows 1972-2012, 1972-2018 (Mouginot) and
     1986-2024 (Mankoff 3-RCM mean, not independent of the Greenland checks)
  2. out-of-sample: fit 1972-1999, predict 2000-2018 decade means
  3. plausibility: CESM2 chain under SSP5-8.5, 2080-2099 SMB anomaly, scaled,
     vs the MAR-CESM2 run and the RCM range of Glaude et al. (2024)
  4. SMB anomaly 2000-2100 on AR6 median GMST paths, each scaling
  5. shape near x = 0: each chain's own 1980-89 to 2015-24 SMB change vs its fit

Usage:  python scripts/obs_constrained_smb_diagnostic.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'notebooks')); sys.path.insert(0, str(ROOT / 'src'))
import test_scenario_transfer_mar as T
import cross_gcm_test_mar as C
import smb_emulator as S
from slr_forecast.readers.ice_sheets import read_mouginot2019_greenland
from bayesian_models import prepare_mouginot_components

GCMS = ['CESM2', 'MPI-ESM1-2-HR', 'NorESM2-MM', 'UKESM1-0-LL']
RHO = np.linspace(-0.5, 0.8, 27)
R_GRID = np.arange(0.5, 8.01, 0.05)
GT_PER_MM = 362.5


def ar1_nig(X, y, m0, sd0, sigma0=100.0, a0=2.0):
    """log evidence and posterior mean of beta for y = X beta + AR(1), NIG prior."""
    b0 = sigma0**2
    V0 = np.diag(np.asarray(sd0, float)**2) / sigma0**2; V0i = np.linalg.inv(V0)
    m0 = np.asarray(m0, float)
    lz, means = [], []
    for rho in RHO:
        Xs = np.vstack([np.sqrt(1 - rho**2) * X[:1], X[1:] - rho * X[:-1]])
        ys = np.concatenate([[np.sqrt(1 - rho**2) * y[0]], y[1:] - rho * y[:-1]])
        Vn = np.linalg.inv(V0i + Xs.T @ Xs); mn = Vn @ (V0i @ m0 + Xs.T @ ys)
        an = a0 + 0.5 * len(y)
        bn = b0 + 0.5 * (ys @ ys + m0 @ V0i @ m0 - mn @ np.linalg.solve(Vn, mn))
        lz.append(-0.5 * len(y) * np.log(2 * np.pi) + 0.5 * (np.linalg.slogdet(Vn)[1] - np.linalg.slogdet(V0)[1])
                  + a0 * np.log(b0) - an * np.log(bn) + gammaln(an) - gammaln(a0) + 0.5 * np.log(1 - rho**2))
        means.append((mn, Vn, an, bn))
    lz = np.array(lz); w = np.exp(lz - lz.max()); w /= w.sum()
    return logsumexp(lz) - np.log(len(RHO)), means, w


def k_posterior(f, y):
    X = np.column_stack([np.ones_like(f), f])
    _, means, w = ar1_nig(X, y, [0, 1], [2000, 5])
    # mixture over rho: mean and 5-95% via normal approx per rho (Student-t ignored)
    draws = []
    rng = np.random.default_rng(0)
    for (mn, Vn, an, bn), wi in zip(means, w):
        n = int(round(wi * 4000))
        if n == 0: continue
        s2 = bn / rng.gamma(an, 1.0, n)
        draws.append(mn[1] + np.sqrt(s2 * Vn[1, 1]) * rng.standard_normal(n))
    d = np.concatenate(draws)
    return np.percentile(d, [5, 50, 95])


def r_posterior(b1, b2, x, y):
    lz = []
    for r in R_GRID:
        f = b1 * r * x + b2 * (r * x)**2
        lz.append(ar1_nig(np.ones((len(y), 1)), y - f, [0], [2000])[0])
    lz = np.array(lz); p = np.exp(lz - logsumexp(lz)); c = np.cumsum(p)
    return np.interp([0.05, 0.5, 0.95], c, R_GRID)


def main():
    mar = pd.read_csv(T.MAR).drop_duplicates(['run', 'year']); temp = pd.read_csv(T.TEMP)
    rng = np.random.default_rng(5)
    be = pd.read_hdf(ROOT / 'data/processed/slr_processed_data.h5', 'raw/df_berkeley')
    g = be['temperature'].groupby(be.index.year).mean(); g = g - g.loc[1995:2005].mean()
    xs = g.rolling(11, center=True, min_periods=6).mean()
    mou = prepare_mouginot_components(
        read_mouginot2019_greenland(str(ROOT / 'data/raw/ice_sheets/greenland/mouginot2019_data.xlsx')),
        baseline_window=(1995, 2005))
    obs = pd.Series(-mou['df']['smb_rate'].values / S.GT_TO_M_SLE, index=mou['df']['decimal_year'].values.astype(int))
    M0 = obs.loc[1995:2005].mean()
    mank = pd.read_csv(ROOT / 'data/raw/ice_sheets/greenland/mankoff/MB_SMB_D_BMB_ann.csv').set_index('time')['SMB']
    mank = mank - mank.loc[1995:2005].mean() + M0     # same baseline level as Mouginot

    shapes = {}
    for gcm in GCMS:
        segs = C.segments(mar, temp, gcm)
        st = np.cumsum([0] + [len(s[1]) for s in segs[:-1]])
        y = np.concatenate([s[1].values for s in segs])
        b, *_ = C.posterior_draws(C.design(segs, 1), y, st, 2000, rng)
        shapes[gcm] = (np.median(b[:, 1]), np.median(b[:, 2]), segs)

    print('1. Calibration of strength on observed SMB  (k: amplitude; r: temperature scale)  median [5th, 95th]')
    windows = [('Mouginot 1972-2012', obs, 1972, 2012), ('Mouginot 1972-2018', obs, 1972, 2018),
               ('Mankoff 1986-2023', mank, 1986, 2023)]
    kr = {}
    for gcm, (b1, b2, _) in shapes.items():
        for lab, ser, a, bb in windows:
            yrs = np.arange(a, bb + 1); x = xs.loc[yrs].values; y = ser.loc[yrs].values - M0
            k = k_posterior(b1 * x + b2 * x**2, y); r = r_posterior(b1, b2, x, y)
            kr[(gcm, lab)] = (k, r)
            print(f'  {gcm:14s} {lab:20s} k = {k[1]:5.1f} [{k[0]:5.1f}, {k[2]:5.1f}]   r = {r[1]:4.2f} [{r[0]:4.2f}, {r[2]:4.2f}]')

    print('\n2. Out of sample: fit on Mouginot 1972-1999, predict 2000-2018 decade means (Gt/yr)')
    print(f'  observed: 2000-09 {obs.loc[2000:2009].mean():.0f}, 2010-18 {obs.loc[2010:2018].mean():.0f}')
    yrs_fit = np.arange(1972, 2000)
    for gcm, (b1, b2, _) in shapes.items():
        x = xs.loc[yrs_fit].values; y = obs.loc[yrs_fit].values - M0
        k = k_posterior(b1 * x + b2 * x**2, y)[1]; r = r_posterior(b1, b2, x, y)[1]
        out = []
        for a, bb in [(2000, 2009), (2010, 2018)]:
            xx = xs.loc[a:bb].values
            pk = M0 + k * (b1 * xx + b2 * xx**2); pr = M0 + b1 * r * xx + b2 * (r * xx)**2
            out.append(f'{a}s k {pk.mean():.0f} / r {pr.mean():.0f}')
        print(f'  {gcm:14s} (k={k:.1f}, r={r:.2f}): ' + ';  '.join(out))

    print('\n3. Plausibility: CESM2 chain, SSP5-8.5, 2080-2099 mean SMB anomaly rel. 1995-2005 (Gt/yr)')
    b1, b2, segs = shapes['CESM2']
    xc = T.drivers(temp, 'CESM2', 'ssp585')['GMST'].loc[2080:2099].values
    mar_c = T.smb_series(mar, 'CESM2-CMIP6', 'ssp585').loc[2080:2099].mean()
    print(f'  MAR-CESM2 (r11) itself: {mar_c:.0f};  emulator unscaled: {np.mean(b1 * xc + b2 * xc**2):.0f}')
    k, r = kr[('CESM2', 'Mouginot 1972-2018')][0][1], kr[('CESM2', 'Mouginot 1972-2018')][1][1]
    print(f'  scaled with Mouginot 1972-2018 calibration: k-scaled {k * np.mean(b1 * xc + b2 * xc**2):.0f}, '
          f'r-scaled {np.mean(b1 * r * xc + b2 * (r * xc)**2):.0f}')
    print('  RCM range for CESM2 SSP5-8.5 (Glaude et al. 2024, 2099 anomaly rel. 1980-1999): -1331 to -2105')

    print('\n4. SMB anomaly 2000-2100 (mm SLE) on AR6 median GMST, Mouginot 1972-2018 calibration, median over GCMs')
    hist = pd.read_hdf(ROOT / 'data/processed/slr_processed_data.h5', 'projections/temp/Historical')
    years = np.arange(1995, 2101); paths = {}
    for s_, key in [('2C', 'SSP1_2_6'), ('3C', 'SSP2_4_5'), ('4C', 'SSP3_7_0')]:
        d = pd.read_hdf(ROOT / 'data/processed/slr_processed_data.h5', f'projections/temp/{key}')
        c = pd.concat([hist[hist.decimal_year < 2015], d])
        gm = np.interp(years + 0.5, c.decimal_year.values, c.temperature.values)
        paths[s_] = gm - gm[(years >= 1995) & (years <= 2005)].mean()
    sel = (years >= 2000) & (years <= 2100)
    for lab in ['unscaled', 'k', 'r']:
        vals = {s_: [] for s_ in paths}
        for gcm, (b1, b2, _) in shapes.items():
            k, r = kr[(gcm, 'Mouginot 1972-2018')][0][1], kr[(gcm, 'Mouginot 1972-2018')][1][1]
            for s_, p in paths.items():
                x = p[sel]
                f = {'unscaled': b1 * x + b2 * x**2, 'k': k * (b1 * x + b2 * x**2),
                     'r': b1 * r * x + b2 * (r * x)**2}[lab]
                vals[s_].append(-f.sum() / GT_PER_MM)
        print(f'  {lab:9s} ' + '  '.join(f'{s_}: {np.median(v):5.0f} (GCM range {min(v):.0f} to {max(v):.0f})'
                                       for s_, v in vals.items()))
    print('  pipeline emulator (AA x GMST): 80 / 122 / 167; earlier parametric: 77 / 119 / 165')

    print('\n5. Shape near x = 0: each chain\'s own SMB change 1980-89 -> 2015-24 (SSP2-4.5 continuation) vs its quadratic')
    for gcm, (b1, b2, segs) in shapes.items():
        gd = C.GDIR[gcm]
        s = T.smb_series(mar, gd, 'ssp245'); xg = T.drivers(temp, gcm, 'ssp245')['GMST']
        d_obs = s.loc[2015:2024].mean() - s.loc[1980:1989].mean()
        f = lambda xx: b1 * xx + b2 * xx**2
        d_fit = f(xg.loc[2015:2024].values).mean() - f(xg.loc[1980:1989].values).mean()
        print(f'  {gcm:14s} MAR {d_obs:+5.0f}  quadratic {d_fit:+5.0f} Gt/yr  '
              f'(GMST change {xg.loc[2015:2024].mean() - xg.loc[1980:1989].mean():.2f} K)')


if __name__ == '__main__':
    main()
