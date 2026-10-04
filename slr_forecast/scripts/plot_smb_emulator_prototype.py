"""Plot the per-RCM SMB emulator prototype: annual SMB vs GMST with linear and
quadratic fits, for HIRHAM, MAR and RACMO (Mankoff MB_region.nc, 1986-2024).

Error bars and bands:
  - y bars: Mankoff stated SMB uncertainty (15%, systematic; not in likelihood)
  - x bars: Berkeley Earth annual GMST uncertainty (mean monthly 90% half-width
    / sqrt(12), i.e. monthly errors treated as independent)
  - dark band: 90% credible band of the fitted mean SMB
  - dashed lines: 90% posterior predictive band (adds AR(1) noise with
    marginal sd sigma / sqrt(1 - rho^2))
HIRHAM is shown with the HARMONIE step (2017-08-31) removed, the step being
identified from differences against MAR and RACMO; its uncorrected
2017-2024 values are drawn as open circles.

Usage:  python scripts/plot_smb_emulator_prototype.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prototype_smb_emulator_mankoff as P

OUT = P.ROOT / 'figures/smb_emulator_prototype_mankoff.png'
C_LIN, C_QUAD, C_REF = '#2a78d6', '#eb6834', '#5f5e5a'
N_BAND = 20000


def annual_err():
    ds = xr.open_dataset(P.NC_PATH)
    v = [f'{r}_err' for r in P.RCMS[:3]]
    df = ds[v].to_dataframe().loc[f'{P.YEAR0}':f'{P.YEAR1}-12-31']
    return df.groupby(df.index.year).sum()


def gmst_err():
    be = pd.read_hdf(P.H5_PATH, '/raw/df_berkeley')
    return be['temperature_unc'].groupby(be.index.year).mean() / np.sqrt(12)


def bands(r, form, tg):
    idx = np.random.default_rng(1).choice(len(r['sigma']), N_BAND, replace=False)
    b, s, rho = r['beta'][idx], r['sigma'][idx], r['rho'][idx]
    G = [np.ones_like(tg), tg] + ([tg**2] if form == 'quadratic' else [])
    mu = b @ np.vstack(G)                                  # (N_BAND, n_grid)
    # one noise draw per sample, shared across the grid: each grid point's
    # marginal is exact and the band edges are smooth
    z = np.random.default_rng(2).standard_normal(len(s))
    pred = mu + (s / np.sqrt(1 - rho**2) * z)[:, None]
    return (np.median(mu, 0), *np.percentile(mu, [5, 95], 0),
            *np.percentile(pred, [5, 95], 0))


def main():
    rng = np.random.default_rng(P.SEED)
    smb, T = P.load_smb(), P.load_gmst()
    years = np.arange(P.YEAR0, P.YEAR1 + 1)
    smb, err, Terr = smb.loc[years], annual_err().loc[years], gmst_err().loc[years]
    dT = T.loc[years].values

    # HIRHAM step from differences against MAR and RACMO
    step = np.where(years >= 2018, 1.0, 0.0)
    step[years == 2017] = 4.0 / 12.0
    P.PRIOR_SD['step'] = 500.0
    Xd = np.column_stack([np.ones(len(years)), step, dT])
    s = np.mean([np.median(P.fit(Xd, (smb['SMB_HIRHAM'] - smb[v]).values,
                                 ['const', 'step', 'T'], rng)['beta'][:, 1])
                 for v in ['SMB_MAR', 'SMB_RACMO']])
    y_all = {'SMB_HIRHAM': smb['SMB_HIRHAM'].values - s * step,
             'SMB_MAR': smb['SMB_MAR'].values, 'SMB_RACMO': smb['SMB_RACMO'].values}

    tg = np.linspace(dT.min() - 0.05, dT.max() + 0.05, 200)
    cmap = LinearSegmentedColormap.from_list('yr', ['#c9c8c2', '#0b0b0b'])
    norm = Normalize(P.YEAR0, P.YEAR1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True,
                             gridspec_kw={'wspace': 0.06})
    for ax, v in zip(axes, ['SMB_HIRHAM', 'SMB_MAR', 'SMB_RACMO']):
        y = y_all[v]
        txt = []
        for form, c in [('linear', C_LIN), ('quadratic', C_QUAD)]:
            X, names = P.design(T, years, form)
            r = P.fit(X, y, names, rng)
            med, lo, hi, plo, phi = bands(r, form, tg)
            ax.fill_between(tg, lo, hi, color=c, alpha=0.22, lw=0)
            ax.plot(tg, med, color=c, lw=2)
            ax.plot(tg, plo, color=c, lw=1, ls='--')
            ax.plot(tg, phi, color=c, lw=1, ls='--')
            m, l5, h95 = P.q(r['beta'][:, 1])
            line = f'{form}: $C_T$ = {m:.0f} [{l5:.0f}, {h95:.0f}]'
            if form == 'quadratic':
                m2, l2, h2 = P.q(r['beta'][:, 2])
                line += f'\n           $C_{{T^2}}$ = {m2:.0f} [{l2:.0f}, {h2:.0f}]'
            txt.append(line)

        # adopted parametric model, C_T and C_T2 zeroed below baseline
        tp = np.clip(tg, 0, None)
        ax.plot(tg, 380 - 300 * tp - 50 * tp**2, color=C_REF, lw=1.5, ls=':')

        ax.errorbar(dT, y, xerr=Terr.values, yerr=err[f'{v}_err'].values,
                    fmt='none', ecolor='#9a9994', elinewidth=0.8, zorder=2)
        ax.scatter(dT, y, c=years, cmap=cmap, norm=norm, s=40,
                   edgecolor='white', linewidth=0.8, zorder=3)
        if v == 'SMB_HIRHAM':
            k = years >= 2017
            ax.scatter(dT[k], smb[v].values[k], facecolor='none',
                       edgecolor='#52514e', s=40, linewidth=0.9, zorder=3)
            ax.text(0.02, 0.03, f'HARMONIE step ({s:.0f} Gt/yr) removed;\n'
                    'open circles: uncorrected 2017–2024', transform=ax.transAxes,
                    fontsize=9, color='#52514e', va='bottom')

        ax.set_title(P.LABEL[v], fontsize=13)
        ax.text(0.98, 0.97, '\n'.join(txt), transform=ax.transAxes, ha='right',
                va='top', fontsize=9, color='#0b0b0b',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.85))
        ax.axhline(0, color='#c9c8c2', lw=0.6, zorder=0)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('GMST anomaly rel. 1995–2005 (°C, Berkeley Earth)')
    axes[0].set_ylabel('Surface mass balance (Gt yr$^{-1}$)')
    axes[0].set_ylim(-150, 750)

    handles = [Line2D([], [], color=C_LIN, lw=2, label='linear fit (median)'),
               Line2D([], [], color=C_QUAD, lw=2, label='quadratic fit (median)'),
               Patch(color='#888888', alpha=0.3, label='90% credible band, mean'),
               Line2D([], [], color='#555555', lw=1, ls='--', label='90% predictive band'),
               Line2D([], [], color=C_REF, lw=1.5, ls=':', label='adopted model (−300, −50)')]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False,
               fontsize=10, bbox_to_anchor=(0.45, -0.04))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                      fraction=0.015, pad=0.01)
    cb.set_label('Year')
    fig.suptitle('Greenland SMB vs GMST, per-RCM Mankoff series, 1986–2024 '
                 '(C_T in Gt yr$^{-1}$ °C$^{-1}$, C_T2 in Gt yr$^{-1}$ °C$^{-2}$; '
                 'medians [5th, 95th])', fontsize=11, y=0.99)
    fig.savefig(OUT, dpi=150, bbox_inches='tight')
    print(f'saved {OUT}')


if __name__ == '__main__':
    main()
