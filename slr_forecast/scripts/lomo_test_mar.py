"""Leave-one-model-out test of the GMST-driven SMB statistical emulator.

For each GCM in turn, fit the no-lag quadratic emulator to every other GCM's
MAR runs (each GCM fitted jointly over its history and all its scenarios),
form the equal-weight mixture of those posteriors (the planned production
ensemble), and predict the held-out GCM's MAR runs from that GCM's own GMST.

Reported per held-out run: actual minus predicted 20-yr means (2041-2060,
2081-2100) and whether they fall inside the mixture's 90% predictive range
(coefficients, spread between GCMs and AR(1) weather noise), plus the
integrated 2015-2100 miss in mm SLE (+ = emulator underpredicts SLR).

Usage:  python scripts/lomo_test_mar.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import test_scenario_transfer_mar as T
import cross_gcm_test_mar as C

GCMS = ['CESM2', 'MPI-ESM1-2-HR', 'NorESM2-MM', 'UKESM1-0-LL',
        'CNRM-CM6-1', 'CNRM-ESM2-1', 'IPSL-CM6A-LR']
N_PER = 2000
GT_PER_MM = 362.5


def mean_noise_sd(sig, rho, n):
    lags = np.arange(1, n)
    s = n + 2 * ((n - lags)[None, :] * rho[:, None] ** lags[None, :]).sum(1)
    return np.sqrt(sig**2 / (1 - rho**2) * s / n**2)


def main():
    mar = pd.read_csv(T.MAR).drop_duplicates(['run', 'year'])
    temp = pd.read_csv(T.TEMP)
    rng = np.random.default_rng(3)
    segs = {g: C.segments(mar, temp, g) for g in GCMS}
    fits = {}
    for g in GCMS:
        starts = np.cumsum([0] + [len(s[1]) for s in segs[g][:-1]])
        y = np.concatenate([s[1].values for s in segs[g]])
        b, s, r, _ = C.posterior_draws(C.design(segs[g], 1), y, starts, N_PER, rng)
        fits[g] = (b, s, r)
        print(f'{g:14s} fit on {[s[0] for s in segs[g]]}: b1 = {np.median(b[:, 1]):.0f} '
              f'Gt/yr/K, b2 = {np.median(b[:, 2]):.1f} Gt/yr/K^2')
    print()
    ncol = 3
    fig, axes = plt.subplots(len(GCMS), ncol, figsize=(17, 4 * len(GCMS)), squeeze=False)
    rows = []
    for i, test in enumerate(GCMS):
        train = [g for g in GCMS if g != test]
        B = np.vstack([fits[g][0] for g in train])
        SG = np.concatenate([fits[g][1] for g in train])
        RH = np.concatenate([fits[g][2] for g in train])
        for name, ys, xs in segs[test][1:]:
            x = xs.loc[ys.index].values
            pr = B[:, [0]] + B[:, [1]] * x[None, :] + B[:, [2]] * x[None, :]**2
            med = pd.Series(np.median(pr, 0), index=ys.index)
            cum = -(ys - med).loc[2015:2100].sum() / GT_PER_MM
            row = {'held out': test, 'run': name, 'mm': cum}
            for a, b in [(2041, 2060), (2081, 2100)]:
                sel = (ys.index >= a) & (ys.index <= b)
                draw = pr[:, sel].mean(1) + mean_noise_sd(SG, RH, int(sel.sum())) * rng.standard_normal(len(SG))
                lo, mid, hi = np.percentile(draw, [5, 50, 95]); obs = ys[sel].mean()
                row[f'{a}s'] = f'{obs - mid:+.0f} [{lo - mid:+.0f}, {hi - mid:+.0f}]' + ('' if lo <= obs <= hi else ' *')
            rows.append(row)
            j = ['ssp126', 'ssp245', 'ssp585'].index(name)
            ax = axes[i][j]
            full = T.smb_series(mar, C.GDIR[test], name)
            ax.plot(full.index, full.values, color='0.8', lw=0.8)
            ax.plot(full.index, full.rolling(11, center=True, min_periods=6).mean(), 'k', lw=2,
                    label=f'MAR-{test} (held out)')
            for g, c in zip(train, ['#2a78d6', '#eb6834', '#1baf7a', '#e87ba4', '#4a3aa7']):
                bg = fits[g][0]
                ax.plot(ys.index, np.median(bg[:, [0]] + bg[:, [1]] * x + bg[:, [2]] * x**2, 0),
                        color=c, lw=1, alpha=0.8, label=f'emulator fit on {g}')
            ax.fill_between(ys.index, *np.percentile(pr, [5, 95], 0), color='0.5', alpha=0.2, lw=0,
                            label='mixture 90% (coefficients + spread between GCMs)')
            ax.plot(ys.index, med, color='0.2', lw=2, ls='--', label='mixture median')
            ax.set_title(f'{test} {name.upper()} — integrated miss {cum:+.1f} mm')
            ax.axhline(0, color='0.6', lw=0.6); ax.grid(True, alpha=0.2)
        for j, name in enumerate(['ssp126', 'ssp245', 'ssp585']):
            if name not in [s[0] for s in segs[test][1:]]:
                axes[i][j].axis('off')
        axes[i][0].set_ylabel('SMB anomaly (Gt yr$^{-1}$)')
        axes[i][['ssp126', 'ssp245', 'ssp585'].index(segs[test][1][0])].legend(fontsize=7, loc='lower left')
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print('Held-out runs: actual minus mixture median of the 20-yr mean, Gt/yr [90% range]; '
          '* = outside; mm = integrated 2015-2100 miss (+ = underpredicts SLR)')
    print(df.to_string(index=False, float_format=lambda v: f'{v:+.1f}'))
    fig.tight_layout()
    out = ROOT / 'figures/preview_mar_lomo.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'\nsaved {out}')


if __name__ == '__main__':
    main()
