"""Scenario-transfer test of the SMB emulator within one GCM-RCM chain.

Train on MARv3.12 driven by a CMIP6 GCM under SSP5-8.5 (plus its 1980-2014
history), then predict the same chain under SSP1-2.6 and SSP2-4.5, which the
fit never sees.  Three choices of driver are compared:

  GMST   -- the GCM's own global-mean temperature
  T_GrIS -- the GCM's Greenland temperature (Noel et al. 2021 definition)
  +lag   -- either driver passed through an exponential filter with time
            scale tau, selected on the training run by marginal likelihood

Drivers are anomalies relative to 1995-2005, smoothed with an 11-yr centred
running mean (single-member GCM runs carry internal variability), and the
lag filter starts in 1950.  SMB is the ice-sheet-only annual total
(extract_mar_protect_smb.py); the emulator is quadratic with AR(1) residuals
(smb_emulator.fit_quadratic_ar1).

Usage:  python scripts/test_scenario_transfer_mar.py [--gcm CESM2]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
import smb_emulator as S

MAR = ROOT / 'data/raw/ice_sheets/greenland/mar_protect/mar_protect_integrated_smb.csv'
TEMP = ROOT / 'data/raw/ice_sheets/greenland/mar_protect/gcm_temperature.csv'
TAUS = [0, 5, 10, 20, 30, 50]
GT_PER_MM = 362.5
BASE = (1995, 2005)


def drivers(temp, gcm, scen):
    t = temp[(temp.gcm == gcm) & temp.experiment.isin(['historical', scen])]
    t = t.set_index('year').sort_index()
    t = t[~t.index.duplicated(keep='last')]
    out = {}
    for col, name in [('gmst_K', 'GMST'), ('tgris_K', 'T_GrIS')]:
        s = t[col] - t[col].loc[BASE[0]:BASE[1]].mean()
        out[name] = s.rolling(11, center=True, min_periods=6).mean()
    return out


def lag(x, tau):
    if tau == 0:
        return x.copy()
    y = np.empty(len(x)); y[0] = x.iloc[0]
    for k in range(1, len(x)):
        y[k] = y[k - 1] + (x.iloc[k] - y[k - 1]) / tau
    return pd.Series(y, index=x.index)


def smb_series(mar, gcm_dir, scen):
    h = mar[mar.run == f'{gcm_dir}-histo'].set_index('year')['sheet']
    s = mar[mar.run == f'{gcm_dir}-{scen}'].set_index('year')['sheet']
    y = pd.concat([h, s]).sort_index()
    return y - h.loc[BASE[0]:BASE[1]].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gcm', default='CESM2')
    ap.add_argument('--extra', nargs='*', default=[],
                    help='extra driver+lag curves, e.g. GMST:20 GMST:10; drawn dash-dot, '
                         'dotted, then dash-dot-dot in the order given')
    args = ap.parse_args()
    gcm_dir = {'CESM2': 'CESM2-CMIP6', 'MPI-ESM1-2-HR': 'MPI-ESM1-2-HR',
               'NorESM2-MM': 'NorESM2', 'UKESM1-0-LL': 'UKESM1-0-LL-CMIP6'}[args.gcm]
    mar = pd.read_csv(MAR)
    mar = mar.drop_duplicates(['run', 'year'])
    temp = pd.read_csv(TEMP)
    scens = [s for s in ['ssp126', 'ssp245'] if (mar.run == f'{gcm_dir}-{s}').sum() == 86]
    rng = np.random.default_rng(1)
    extra = {}
    for e in args.extra:
        f, t = e.split(':'); extra.setdefault(f, set()).add(int(t))
    extra_styles = ['-.', ':', (0, (3, 1, 1, 1, 1, 1))]
    extra_labels = {f"{e.split(':')[0]} + lag {int(e.split(':')[1])} yr": extra_styles[k % 3]
                    for k, e in enumerate(args.extra)}

    y_tr = smb_series(mar, gcm_dir, 'ssp585')
    if len(y_tr) < 121:
        raise SystemExit(f'training run incomplete ({len(y_tr)} years)')
    print(f'{args.gcm}: training on SSP5-8.5 1980-2100 (n={len(y_tr)}), testing {scens}')
    print('SMB anomalies rel. 1995-2005 (Gt/yr); misses are actual minus predicted '
          '(negative = more loss than predicted)\n')

    results = {}
    for frame in ['GMST', 'T_GrIS']:
        d_tr = drivers(temp, args.gcm, 'ssp585')[frame]
        evid = {}
        fits = {}
        for tau in TAUS:
            x = lag(d_tr, tau).loc[y_tr.index].values
            r = S.fit_quadratic_ar1(x, y_tr.values, 4000, rng)
            evid[tau], fits[tau] = r['log_evidence'], r
        tau_best = max(evid, key=evid.get)
        print(f'── {frame}: training log evidence by tau: '
              + ', '.join(f'{t}:{evid[t] - evid[0]:+.1f}' for t in TAUS)
              + f'  -> selected tau = {tau_best} yr')
        for tau in sorted({0, tau_best} | extra.get(frame, set())):
            B = fits[tau]['beta']
            label = f'{frame}' + (f' + lag {tau} yr' if tau else ' (no lag)')
            for scen in scens:
                d = drivers(temp, args.gcm, scen)[frame]
                y = smb_series(mar, gcm_dir, scen)
                x = lag(d, tau).loc[y.index]
                pred = pd.Series(np.median(B[:, [0]] + B[:, [1]] * x.values + B[:, [2]] * x.values**2, 0),
                                 index=y.index)
                miss = y - pred
                cum_mm = -miss.loc[2015:2100].sum() / GT_PER_MM
                results[(label, scen)] = (y, pred)
                print(f'   {label:24s} {scen}: miss 2041-2060 {miss.loc[2041:2060].mean():+5.0f}, '
                      f'2081-2100 {miss.loc[2081:2100].mean():+5.0f} Gt/yr;  '
                      f'2015-2100 integrated {cum_mm:+5.1f} mm SLE '
                      f'({"emulator underpredicts SLR" if cum_mm > 0 else "emulator overpredicts SLR"})')
        # diagnostic only: which tau would have predicted the withheld runs best
        sse = {}
        for tau in TAUS:
            B = fits[tau]['beta']; s2 = 0
            for scen in scens:
                d = drivers(temp, args.gcm, scen)[frame]; y = smb_series(mar, gcm_dir, scen)
                x = lag(d, tau).loc[y.index].values
                p = np.median(B[:, [0]] + B[:, [1]] * x + B[:, [2]] * x**2, 0)
                s2 += np.sum(((y - p).loc[2015:2100].rolling(11, center=True, min_periods=6).mean())**2)
            sse[tau] = s2
        print(f'   (diagnostic, not a test: tau with smallest decadal miss on withheld runs = '
              f'{min(sse, key=sse.get)} yr)\n')

    # figure
    labels = sorted({k[0] for k in results}, key=lambda s: (s.split()[0], 'lag' in s))
    fig, axes = plt.subplots(1, len(scens), figsize=(7 * len(scens), 5), squeeze=False)
    colors = {'GMST (no lag)': '#2a78d6', 'T_GrIS (no lag)': '#eb6834'}
    for ax, scen in zip(axes[0], scens):
        y = results[(labels[0], scen)][0]
        ax.plot(y.index, y.values, color='0.75', lw=0.8, label='MAR annual')
        ax.plot(y.index, y.rolling(11, center=True, min_periods=6).mean(), 'k', lw=2, label='MAR 11-yr mean')
        for lab in labels:
            p = results[(lab, scen)][1]
            c = colors.get(lab, '#2a78d6' if lab.startswith('GMST') else '#eb6834')
            ls = '-' if 'no lag' in lab else extra_labels.get(lab, '--')
            ax.plot(p.index, p.values, color=c, lw=2, ls=ls, label=lab)
        ax.axhline(0, color='0.6', lw=0.6); ax.axvline(2015, color='0.6', lw=0.6, ls=':')
        ax.set_title(f'MAR-{args.gcm} {scen.upper()} (withheld)'); ax.set_xlabel('Year')
        ax.set_ylabel('SMB anomaly rel. 1995–2005 (Gt yr$^{-1}$)'); ax.grid(True, alpha=0.2)
    axes[0][0].legend(fontsize=9)
    fig.tight_layout()
    suffix = ''.join(f'_{e.replace(":", "lag")}' for e in args.extra)
    out = ROOT / f'figures/preview_mar_transfer_{args.gcm}{suffix}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'saved {out}')


if __name__ == '__main__':
    main()
