"""MAR and RACMO projections of Greenland SMB to 2100 against the MAR emulator.

For each of the 13 complete MAR scenario runs (7 CMIP6 GCMs; SSP1-2.6,
SSP2-4.5, SSP5-8.5), the cumulative SMB sea-level contribution over
2015-2100 simulated by MAR is compared with the emulator's prediction for
that run, driven by the same GCM's GMST. The prediction is out of sample:
the emulator is fitted to the other six GCMs only (leave-one-GCM-out, as in
supp_greenland_smb_lomo.png), so this figure shows the time evolution behind
that figure's 2015-2100 totals.

Conventions follow plot_supp_greenland_smb_emulator.py: SMB is the anomaly
relative to the run's own 1995-2005 mean, converted to mm SLE (positive =
sea-level rise) at 362.5 Gt per mm, and cumulated from 2015. The emulator
band is 90% and includes the coefficient posterior, the spread between the
six training GCMs, AR(1) weather noise, and the noise of the 11-yr anchor
mean, simulated year by year so that the band is valid at every year.

The last panel adds the one RACMO projection available, RACMO2.3p2 forced by
the CESM2 branch run under SSP5-8.5 (Noel et al. 2021; ice-sheet-only
integral of the Glaude et al. 2024 1-km grid, which matches Noel's annual
totals), with MAR forced by the same run and HIRHAM forced by CESM2 r1i1p1f1
SSP5-8.5 (both Glaude et al. 2024; ice-sheet only, HIRHAM margin gaps filled)
for reference.
That branch run's GMST is not archived, so the emulator there is driven by the
CMIP6 CESM2 SSP5-8.5 GMST, as in the CESM2 SSP5-8.5 MAR panel, and fitted to
the six GCMs other than CESM2. RACMO ends in 2099.

Usage:  python scripts/plot_greenland_smb_mar_vs_emulator.py
Output: figures/greenland_smb_mar_vs_emulator.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import plot_supp_greenland_smb_emulator as P   # sets paths, style, and S

S, GT, N = P.S, P.GT, P.N
OUT = ROOT / 'figures' / 'greenland_smb_mar_vs_emulator.png'
SCENS = ['ssp126', 'ssp245', 'ssp585']
GLAUDE = ROOT / 'data/raw/ice_sheets/greenland/glaude2024/glaude2024_integrated_smb.csv'


def emulator_paths(B, SG, RH, x, rng):
    """Cumulative SLR (mm) for each posterior/GCM draw, with AR(1) weather and
    the anchor-mean noise; x is the driver over the run's years."""
    n, nyr = len(B), len(x)
    forced = B[:, [1]] * x + B[:, [2]] * x**2                       # Gt/yr anomaly
    e = np.empty((n, nyr))
    e[:, 0] = SG / np.sqrt(1 - RH**2) * rng.standard_normal(n)
    for k in range(1, nyr):
        e[:, k] = RH * e[:, k - 1] + SG * rng.standard_normal(n)
    anchor = np.sqrt(P.sum_ar1_var(SG, RH, 11)) / 11 * rng.standard_normal(n)
    return np.cumsum(-(forced + e + anchor[:, None]) / GT, axis=1)


def main():
    mar, temp = S.load_data()
    emu = S.fit_smb_emulator(N, seed=600)
    rng = np.random.default_rng(11)

    runs = []
    for test in S.GCMS:
        train = [g for g in S.GCMS if g != test]
        B = np.vstack([emu['fits'][g]['beta'] for g in train])
        SG = np.concatenate([emu['fits'][g]['sigma'] for g in train])
        RH = np.concatenate([emu['fits'][g]['rho'] for g in train])
        for name, ys, xs in S.training_segments(test, mar, temp):
            if name == 'history':
                continue
            yrs = ys.index.values
            x = xs.loc[yrs].values
            mar_cum = np.cumsum(-ys.values / GT)
            em = emulator_paths(B, SG, RH, x, rng)
            runs.append(dict(gcm=test, scen=name, years=yrs, mar=mar_cum,
                             med=np.median(em, 0), lo=np.percentile(em, 5, 0),
                             hi=np.percentile(em, 95, 0)))

    runs.sort(key=lambda r: (SCENS.index(r['scen']), S.GCMS.index(r['gcm'])))

    # RACMO (and MAR) forced by the CESM2 branch run, SSP5-8.5
    g = pd.read_csv(GLAUDE, index_col=0)
    yrs = np.arange(2015, 2100)
    def cum(col):
        s = g[col]
        return np.cumsum(-(s.loc[yrs].values - s.loc[1995:2005].mean()) / GT)
    train = [x for x in S.GCMS if x != 'CESM2']
    B = np.vstack([emu['fits'][x]['beta'] for x in train])
    SG = np.concatenate([emu['fits'][x]['sigma'] for x in train])
    RH = np.concatenate([emu['fits'][x]['rho'] for x in train])
    xs = [seg[2] for seg in S.training_segments('CESM2', mar, temp) if seg[0] == 'ssp585'][0]
    em = emulator_paths(B, SG, RH, xs.loc[yrs].values, rng)
    racmo = dict(gcm='CESM2 branch run', scen='ssp585', years=yrs, mar=cum('RACMO_sheet'),
                 mar_same=cum('MAR_sheet'), hirham=cum('HIRHAM_sheet'), med=np.median(em, 0),
                 lo=np.percentile(em, 5, 0), hi=np.percentile(em, 95, 0), rcm='RACMO')
    runs.append(racmo)
    plot(runs)


# Print geometry: the supplement is 12-pt article with 1-in margins on letter
# paper, so \textwidth is 6.5 in. The supplement includes this figure at
# width=0.8\textwidth (5.2 in), so it is drawn at exactly that width and saved
# without bbox cropping (scale 1.0 in print), so the
# font sizes below are the printed sizes: 8-9 pt against 12-pt body text.
FIG_W, FIG_H = 0.8 * 6.5, 6.0    # in; matches width=0.8\textwidth
FS_TICK, FS_LABEL, FS_TITLE, FS_LEGEND = 8, 9, 8, 8
MAR_C, RACMO_C, HIRHAM_C = '#1a9850', '#7b3294', '#e66101'   # validated with dataviz


def plot(runs):
    plt.rcParams.update({'font.size': FS_LABEL, 'axes.titlesize': FS_TITLE,
                         'axes.labelsize': FS_LABEL, 'xtick.labelsize': FS_TICK,
                         'ytick.labelsize': FS_TICK, 'legend.fontsize': FS_LEGEND,
                         'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                         'ytick.major.width': 0.6, 'xtick.major.size': 2.5,
                         'ytick.major.size': 2.5, 'lines.linewidth': 1.2})
    ncol, nrow = 3, 5
    fig, axes = plt.subplots(nrow, ncol, figsize=(FIG_W, FIG_H), sharex=True,
                             layout='constrained')
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.04, hspace=0.04)
    axes = axes.ravel()
    ylim = {'ssp126': 65, 'ssp245': 120, 'ssp585': 230}
    for k, (ax, r) in enumerate(zip(axes, runs)):
        c = P.SEG_COLOR[r['scen']]
        ax.fill_between(r['years'], r['lo'], r['hi'], color=c, alpha=0.2, lw=0)
        ax.plot(r['years'], r['med'], color=c, lw=1.2, ls='--')
        if r.get('rcm') == 'RACMO':
            ax.plot(r['years'], r['mar_same'], color=MAR_C, lw=1.2, ls=':')
            ax.plot(r['years'], r['hirham'], color=HIRHAM_C, lw=1.2, ls='-.')
            ax.plot(r['years'], r['mar'], color=RACMO_C, lw=1.2)
            title = f'({chr(97 + k)}) CESM2, three RCMs'
        else:
            ax.plot(r['years'], r['mar'], color=MAR_C, lw=1.2)
            title = f"({chr(97 + k)}) {r['gcm']}"
        ax.set_title(title, loc='left', pad=2)
        ax.text(0.04, 0.95, P.SCEN_LABEL[r['scen']], transform=ax.transAxes, va='top',
                fontsize=FS_TICK, color='0.25')
        ax.axhline(0, color='0.7', lw=0.5)
        ax.set_xlim(2015, 2100)
        ax.set_ylim(-5 if r['scen'] == 'ssp126' else -8, ylim[r['scen']])
        ax.set_xticks([2020, 2060, 2100])
        ax.grid(alpha=0.2, lw=0.4)
        ax.tick_params(pad=1.5)
        rcm = r.get('rcm', 'MAR')
        print(f"{rcm:5s} {r['gcm']:16s} {P.SCEN_LABEL[r['scen']]}: {r['mar'][-1]:6.1f}  "
              f"emulator {r['med'][-1]:6.1f} [{r['lo'][-1]:6.1f}, {r['hi'][-1]:6.1f}]  "
              f"miss {r['mar'][-1] - r['med'][-1]:+5.1f} mm")
    axes[len(runs) - ncol].xaxis.set_tick_params(labelbottom=True)   # column above legend
    leg = axes[len(runs)]
    leg.axis('off')
    h = [plt.Line2D([], [], color=MAR_C, lw=1.2),
         plt.Line2D([], [], color='0.35', lw=1.2, ls='--'),
         plt.Rectangle((0, 0), 1, 1, color='0.35', alpha=0.25, lw=0),
         plt.Line2D([], [], color=RACMO_C, lw=1.2),
         plt.Line2D([], [], color=MAR_C, lw=1.2, ls=':'),
         plt.Line2D([], [], color=HIRHAM_C, lw=1.2, ls='-.')]
    fig.legend(h, ['MAR', 'Emulator median', 'Emulator 90%', 'RACMO (n)',
                   'MAR, same run (n)', 'HIRHAM (n)'], loc='outside lower center', ncol=3,
               frameon=False, handlelength=2.0, columnspacing=1.2, labelspacing=0.3)
    fig.supylabel('SMB contribution since 2015 (mm SLE)', fontsize=FS_LABEL)
    for ax in (axes[9], axes[10]):                    # panels (j) and (k)
        ax.xaxis.set_tick_params(labelbottom=True)
    for ax in (axes[9], axes[10], axes[len(runs) - ncol], axes[len(runs) - 2],
               axes[len(runs) - 1]):
        ax.set_xlabel('Year', fontsize=FS_LABEL, labelpad=1)
    fig.savefig(OUT, dpi=300)
    fig.savefig(OUT.with_suffix('.pdf'))
    plt.close(fig)
    print(f'saved {OUT}')


if __name__ == '__main__':
    main()
