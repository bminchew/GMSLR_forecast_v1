"""Greenland SMB emulator fitted to the RACMO2.3p2-CESM2 run of Noel et al. (2021).

Training data: annual ice-sheet-integrated SMB of the RACMO2.3p2 run forced by
CESM2 (historical branch 'HIST-parent' 1950-2014, then 'SSP5-8.5-parent'
2015-2099), from Zenodo 10.5281/zenodo.4289959.  The forcing run is not in
the CMIP6 archive, so the regressor is CESM2's Greenland near-surface
temperature T_GrIS (Noel et al. 2021, Text S1), not GMST.  Projections then
map GMST to T_GrIS with a ratio r shared with the discharge pathway.

Both SMB and T_GrIS are anomalies relative to each run's own 1995-2005 mean,
which removes the CESM2 biases in level (SMB 295 vs ~368 Gt/yr observed) and
in historical warming, and makes r a trend ratio.

Model:  SMB'(t) = b1 x(t) + b2 x(t)^2 [+ b3 x^3] + M + eps,  eps ~ AR(1)
fitted with the analytic machinery of prototype_smb_emulator_mankoff.py.

Regressors x:
  'ensemble' -- CESM2 ensemble-mean T_GrIS (12 historical members, then the
                3 SSP5-8.5 members), i.e. the forced signal.  Deviations of
                the parent run's own weather from it are Berkson-type and
                do not bias a linear slope.
  'annual'   -- the parent run's own annual T_GrIS (classical error;
                attenuates the slope by roughly its 7% noise share).

Withheld test: the reconstructed CESM2 SMB (RACMO-calibrated by Noel et al.)
for SSP1-2.6, SSP2-4.5 and SSP3-7.0 members, predicted from each member's own
T_GrIS.  These are CESM2 SMB rescaled to RACMO, not RACMO physics.

Usage:  python scripts/emulator_smb_noel2021.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prototype_smb_emulator_mankoff as P

D = P.ROOT / 'data/raw/ice_sheets/greenland/noel2021'
BASE = (1995, 2005)
GT_PER_MM = 362.5
RATIOS = [1.6, 2.2, 2.5]


def read_multi(fname, names):
    df = pd.read_csv(D / fname, sep=r'\s+', skiprows=1, header=None)
    df = df.iloc[:, :len(names) + 1]
    df.columns = ['year'] + names
    return df.set_index('year').astype(float)


HIST = ['HIST-parent'] + [f'HIST-{i}' for i in range(1, 12)]
PROJ = ['SSP5-8.5-parent', 'SSP5-8.5-1', 'SSP5-8.5-2', 'SSP3-7.0-1', 'SSP3-7.0-2',
        'SSP2-4.5-1', 'SSP2-4.5-2', 'SSP2-4.5-3', 'SSP1-2.6-1', 'SSP1-2.6-2']


def load():
    racmo = pd.concat([
        pd.read_csv(D / f, sep=r'\s+', skiprows=1, header=None, usecols=[0, 1, 2]).set_index(0)
        for f in ['SMB_TGrIS_RACMO2.3p2-CESM2_Historical_1950-2014.dat',
                  'SMB_TGrIS_RACMO2.3p2-CESM2_SSP5-8.5_2015-2099.dat']])
    racmo.columns = ['SMB', 'T']
    T_h = read_multi('TGrIS_CESM2_Historical_1950-2014.dat', HIST)
    T_p = read_multi('TGrIS_CESM2_Projections_2015-2099.dat', PROJ)
    S_h = read_multi('Reconstructed_SMB_CESM2_Historical_1950-2014.dat', HIST)
    S_p = read_multi('Reconstructed_SMB_CESM2_Projections_2015-2099.dat', PROJ)
    return racmo, T_h, T_p, S_h, S_p


def anom(s):
    return s - s.loc[BASE[0]:BASE[1]].mean()


def design(x, form):
    cols = [np.ones_like(x), x]
    names = ['const', 'T']
    if form in ('quadratic', 'cubic'):
        cols.append(x**2); names.append('T2')
    if form == 'cubic':
        cols.append(x**3); names.append('T3')
    return np.column_stack(cols), names


def main():
    rng = np.random.default_rng(P.SEED)
    P.PRIOR_SD.update({'T': 1000.0, 'T2': 500.0, 'T3': 200.0})
    racmo, T_h, T_p, S_h, S_p = load()
    years = racmo.index.values.astype(int)

    # Forced signal: ensemble-mean T_GrIS along the parent's pathway
    ens = pd.concat([T_h.mean(axis=1),
                     T_p[['SSP5-8.5-parent', 'SSP5-8.5-1', 'SSP5-8.5-2']].mean(axis=1)])
    regs = {'ensemble': anom(ens).loc[years].values,
            'annual': anom(racmo['T']).loc[years].values}
    y = anom(racmo['SMB']).values
    print(f'RACMO2.3p2-CESM2 parent run, {years[0]}-{years[-1]} (n = {len(years)}); '
          f'anomalies rel. {BASE[0]}-{BASE[1]} '
          f'(SMB {racmo["SMB"].loc[BASE[0]:BASE[1]].mean():.0f} Gt/yr, '
          f'T_GrIS {racmo["T"].loc[BASE[0]:BASE[1]].mean() - 254.07:+.2f} K vs PI)')
    print(f'x range (ensemble regressor): {regs["ensemble"].min():.2f} to {regs["ensemble"].max():.2f} K')
    print('Units: b1 Gt/yr/K, b2 Gt/yr/K^2 (K of Greenland T). Medians [5th, 95th].\n')

    fits = {}
    for rname, x in regs.items():
        print(f'── regressor: {rname} ' + '─' * 40)
        for form in ['linear', 'quadratic', 'cubic']:
            X, names = design(x, form)
            r = P.fit(X, y, names, rng)
            fits[(rname, form)] = r
            line = (f'   {form:9s} b1 = {P.fmt(r["beta"][:, 1])}'
                    + (f'  b2 = {P.fmt(r["beta"][:, 2], 1)}' if form != 'linear' else '')
                    + (f'  b3 = {P.fmt(r["beta"][:, 3], 2)}' if form == 'cubic' else '')
                    + f'  M = {P.fmt(r["beta"][:, 0])}  rho = {P.fmt(r["rho"], 2)}'
                      f'  sigma = {P.fmt(r["sigma"])}  logZ = {r["log_evidence"]:.1f}  BIC = {r["bic"]:.1f}')
            print(line)
        lz = {f: fits[(rname, f)]['log_evidence'] for f in ['linear', 'quadratic', 'cubic']}
        print(f'   dlogZ quadratic-linear = {lz["quadratic"] - lz["linear"]:+.1f}, '
              f'cubic-quadratic = {lz["cubic"] - lz["quadratic"]:+.1f}')
        # residual sd by warming range, quadratic model
        X, _ = design(x, 'quadratic')
        res = y - X @ np.median(fits[(rname, 'quadratic')]['beta'], 0)
        bins = [(-1, 1), (1, 3), (3, 5), (5, 9)]
        print('   quadratic residual sd by x range: ' + ', '.join(
            f'[{a},{b}) K: {res[(x >= a) & (x < b)].std():.0f} (n={((x >= a) & (x < b)).sum()})'
            for a, b in bins))
        print()

    # Withheld scenario transfer on reconstructed CESM2 members
    print('── Withheld test: reconstructed CESM2 SMB (RACMO-calibrated), quadratic, ensemble-regressor fit ──')
    r = fits[('ensemble', 'quadratic')]
    idx = rng.choice(len(r['sigma']), 20000, replace=False)
    B, s, rho = r['beta'][idx], r['sigma'][idx], r['rho'][idx]
    T_base = T_h.loc[BASE[0]:BASE[1]].mean().mean()
    S_base = S_h.loc[BASE[0]:BASE[1]].mean().mean()
    print('   (baselines: historical-ensemble 1995-2005 means; decade means over 2050-2059 and 2090-2099)')
    for m in ['SSP1-2.6-1', 'SSP1-2.6-2', 'SSP2-4.5-1', 'SSP2-4.5-2', 'SSP2-4.5-3',
              'SSP3-7.0-1', 'SSP3-7.0-2', 'SSP5-8.5-1', 'SSP5-8.5-2']:
        out = []
        for a, b in [(2050, 2059), (2090, 2099)]:
            xm = (T_p[m].loc[a:b] - T_base).values
            obs = (S_p[m].loc[a:b] - S_base).mean()
            mu = (B[:, [0]] + B[:, [1]] * xm + B[:, [2]] * xm**2).mean(1)
            # predictive for a 10-yr mean: AR(1) noise averaged over 10 years
            n = len(xm)
            k = np.arange(n)
            var10 = (s**2 / (1 - rho**2)) * np.array(
                [np.sum(rr ** np.abs(k[:, None] - k[None, :])) for rr in rho]) / n**2
            pred = mu + np.sqrt(var10) * rng.standard_normal(len(mu))
            lo, hi = np.percentile(pred, [5, 95])
            out.append(f'{a}s: x={xm.mean():.1f} obs {obs:.0f} pred {np.median(mu):.0f} [{lo:.0f}, {hi:.0f}]'
                       + ('' if lo <= obs <= hi else ' *outside*'))
        print(f'   {m:11s} ' + ';  '.join(out))
    print()

    # Conversion to our GMST frame and SMB anomaly 2000-2100 on AR6 medians
    print('── In our frame: C_T = b1 r, C_T2 = b2 r^2 (Gt/yr/degC GMST, rel. 1995-2005) ──')
    H = P.H5_PATH
    hist = pd.read_hdf(H, 'projections/temp/Historical')
    yrs = np.arange(1995, 2101)
    paths = {}
    for s_, k in [('SSP1-2.6', 'SSP1_2_6'), ('SSP2-4.5', 'SSP2_4_5'), ('SSP3-7.0', 'SSP3_7_0')]:
        d = pd.read_hdf(H, f'projections/temp/{k}')
        c = pd.concat([hist[hist.decimal_year < 2015], d])
        g = np.interp(yrs + 0.5, c.decimal_year.values, c.temperature.values)
        paths[s_] = g - g[(yrs >= 1995) & (yrs <= 2005)].mean()
    i = (yrs >= 2000) & (yrs <= 2100)
    for rname in ['ensemble', 'annual']:
        for form in ['linear', 'quadratic']:
            B = fits[(rname, form)]['beta']
            b1 = B[:, 1]
            b2 = B[:, 2] if form == 'quadratic' else np.zeros_like(b1)
            for rr in RATIOS:
                row = []
                for s_, g in paths.items():
                    x = rr * g[i]
                    mm = -(b1[:, None] * x + b2[:, None] * x**2).sum(1) / GT_PER_MM
                    row.append(f'{s_} {P.fmt(mm)}')
                print(f'   {rname:8s} {form:9s} r={rr}: C_T = {P.fmt(b1 * rr)}, '
                      f'C_T2 = {P.fmt(b2 * rr**2, 1)};  SMB anomaly 2000-2100 (mm): ' + ', '.join(row))
    print('   adopted (-300, -50): 77 / 119 / 165 mm (same convention)')


if __name__ == '__main__':
    main()
