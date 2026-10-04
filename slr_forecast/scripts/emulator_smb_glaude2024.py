"""Greenland SMB emulator fitted separately to RACMO, MAR and HIRHAM (Glaude et
al. 2024), all forced by the same CESM2 SSP5-8.5 run.

The three RCM series are the ice-sheet-only integrals from
integrate_glaude2024_smb.py (HIRHAM with its margin gaps filled).  The
high-pass annual SMB of all three correlates at 0.92-1.00 with the Noel et al.
(2021) RACMO run, so they share its forcing weather and Noel's CESM2 T_GrIS
is the regressor for each.  Anomalies are relative to each series' own
1995-2005 mean, and the fit is the quadratic of emulator_smb_noel2021.py
(ensemble-mean T_GrIS regressor, analytic AR(1) posterior).

Structural spread enters as an equal-weight mixture of the three posteriors.
Each run sees only SSP5-8.5, so path dependence cannot be tested here.

Usage:  python scripts/emulator_smb_glaude2024.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prototype_smb_emulator_mankoff as P
import emulator_smb_noel2021 as E

CSV = P.ROOT / 'data/raw/ice_sheets/greenland/glaude2024/glaude2024_integrated_smb.csv'
RCMS = ['RACMO', 'MAR', 'HIRHAM']
RATIOS = [1.6, 2.2, 2.5]


def main():
    rng = np.random.default_rng(P.SEED)
    P.PRIOR_SD.update({'T': 1000.0, 'T2': 500.0, 'T3': 200.0})
    racmo, T_h, T_p, S_h, S_p = E.load()
    ens = pd.concat([T_h.mean(axis=1),
                     T_p[['SSP5-8.5-parent', 'SSP5-8.5-1', 'SSP5-8.5-2']].mean(axis=1)])
    x_all = E.anom(ens)
    g = pd.read_csv(CSV, index_col=0)

    print('Units: b1 Gt/yr/K, b2 Gt/yr/K^2 of Greenland T; anomalies rel. 1995-2005. '
          'Medians [5th, 95th].\n')
    fits = {}
    for m in RCMS:
        s = g[f'{m}_sheet'].dropna()
        yrs = s.index.intersection(x_all.index)
        y = E.anom(s).loc[yrs].values
        x = x_all.loc[yrs].values
        out = []
        for form in ['linear', 'quadratic', 'cubic']:
            X, names = E.design(x, form)
            fits[(m, form)] = P.fit(X, y, names, rng)
        r = fits[(m, 'quadratic')]
        lz = {f: fits[(m, f)]['log_evidence'] for f in ['linear', 'quadratic', 'cubic']}
        X, _ = E.design(x, 'quadratic')
        res = y - X @ np.median(r['beta'], 0)
        print(f'{m:6s} {yrs.min()}-{yrs.max()} (n={len(yrs)}), 1995-2005 SMB {s.loc[1995:2005].mean():.0f} Gt/yr')
        print(f'   quadratic: b1 = {P.fmt(r["beta"][:, 1])}  b2 = {P.fmt(r["beta"][:, 2], 1)}'
              f'  rho = {P.fmt(r["rho"], 2)}  sigma = {P.fmt(r["sigma"])}')
        print(f'   dlogZ quadratic-linear = {lz["quadratic"] - lz["linear"]:+.1f}, '
              f'cubic-quadratic = {lz["cubic"] - lz["quadratic"]:+.1f}')
        print('   residual sd by x: ' + ', '.join(
            f'[{a},{b}) K: {res[(x >= a) & (x < b)].std():.0f}'
            for a, b in [(-2, 1), (1, 3), (3, 5), (5, 9)]))
        for a, b in [(2080, 2099)]:
            xm = x_all.loc[a:b].values
            pred = (r['beta'][:, [0]] + r['beta'][:, [1]] * xm + r['beta'][:, [2]] * xm**2).mean(1)
            print(f'   {a}-{b} mean anomaly: data {E.anom(s).loc[a:b].mean():.0f}, fit {np.median(pred):.0f}')
        print()

    # Our frame on AR6 median GMST paths
    hist = pd.read_hdf(P.H5_PATH, 'projections/temp/Historical')
    yrs = np.arange(1995, 2101)
    paths = {}
    for s_, k in [('SSP1-2.6', 'SSP1_2_6'), ('SSP2-4.5', 'SSP2_4_5'), ('SSP3-7.0', 'SSP3_7_0')]:
        d = pd.read_hdf(P.H5_PATH, f'projections/temp/{k}')
        c = pd.concat([hist[hist.decimal_year < 2015], d])
        gm = np.interp(yrs + 0.5, c.decimal_year.values, c.temperature.values)
        paths[s_] = gm - gm[(yrs >= 1995) & (yrs <= 2005)].mean()
    i = (yrs >= 2000) & (yrs <= 2100)

    print('── Our frame: C_T = b1 r, C_T2 = b2 r^2 (Gt/yr/degC GMST); '
          'SMB anomaly 2000-2100 (mm) at 2 / 3 / 4 degC ──')
    for rr in RATIOS:
        mix_b = []
        for m in RCMS:
            B = fits[(m, 'quadratic')]['beta'][::3]
            mix_b.append(B)
            mm = [-(B[:, 1:2] * (rr * g_[i]) + B[:, 2:3] * (rr * g_[i])**2).sum(1) / E.GT_PER_MM
                  for g_ in paths.values()]
            print(f'   r={rr} {m:6s} C_T = {P.fmt(B[:, 1] * rr)}, C_T2 = {P.fmt(B[:, 2] * rr**2, 1)};  '
                  + ' / '.join(f'{np.median(v):.0f}' for v in mm))
        B = np.concatenate(mix_b)
        mm = [-(B[:, 1:2] * (rr * g_[i]) + B[:, 2:3] * (rr * g_[i])**2).sum(1) / E.GT_PER_MM
              for g_ in paths.values()]
        print(f'   r={rr} {"mixture":6s} C_T = {P.fmt(B[:, 1] * rr)}, C_T2 = {P.fmt(B[:, 2] * rr**2, 1)};  '
              + ' / '.join(P.fmt(v) for v in mm))
        print()
    print('   adopted (-300, -50): 77 / 119 / 165 mm')


if __name__ == '__main__':
    main()
