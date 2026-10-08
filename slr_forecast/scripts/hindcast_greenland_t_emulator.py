"""Diagnostic: does observed Greenland temperature close the SMB hindcast gap?

The production SMB emulator (smb_emulator.py) is driven by GMST. Driven by
observed GMST it under-predicts the post-2000 SMB decline in Mouginot et al.
(2019). This script asks whether the gap lies in the SMB response or in how
global warming maps onto Greenland's climate. Each GCM's MAR runs are refit
with the GCM's own Greenland near-surface temperature (annual, land cells
59-85N, 72-12W; gcm_temperature.csv, column tgris_K) as the predictor, and
the emulator is then driven by observed Berkeley Earth Greenland temperature
over the same box. If the gap closes, the SMB response is consistent with
the observations and the misfit is in GMST -> Greenland T.

Drivers compared (all anomalies rel. 1995-2005, quadratic + AR(1), each GCM
fitted on all of its runs jointly, equal-weight mixture of 7 GCMs, M0 =
Mouginot 1995-2005 mean SMB, no weather noise, no elevation feedback):
  A  GMST, 11-yr centred mean       (the production emulator)
  B  Greenland T, 11-yr centred mean
  C  Greenland T, annual (unsmoothed)
  D  Greenland summer (JJA) T, 11-yr centred mean
  E  Greenland summer (JJA) T, annual (unsmoothed)
GCM JJA temperature comes from gcm_greenland_jja_temperature.py.

Diagnostic only: nothing here feeds the pipeline.

Usage:  python scripts/hindcast_greenland_t_emulator.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks')); sys.path.insert(0, str(ROOT / 'src'))
import smb_emulator as S
from slr_data_readers import read_berkeley_earth_gridded
from slr_forecast.readers.ice_sheets import read_mouginot2019_greenland
from bayesian_models import prepare_mouginot_components

N = 2000
GT_PER_MM = 362.5
OUT_FIG = ROOT / 'figures/diagnostics/hindcast_greenland_t_emulator.png'
VARIANTS = {'A': ('gmst_K', True, 'GMST, 11-yr mean (production)'),
            'B': ('tgris_K', True, 'Greenland T, 11-yr mean'),
            'C': ('tgris_K', False, 'Greenland T, annual'),
            'D': ('tgris_jja_K', True, 'Greenland JJA T, 11-yr mean'),
            'E': ('tgris_jja_K', False, 'Greenland JJA T, annual')}
PERIODS = [(1972, 1985), (1986, 1999), (2000, 2009), (2010, 2018)]


def driver(temp, gcm, scen, col, smooth):
    """GCM temperature anomaly rel. 1995-2005 (pd.Series by year)."""
    t = temp[(temp.gcm == gcm) & temp.experiment.isin(['historical', scen])]
    t = t.set_index('year').sort_index()
    t = t[~t.index.duplicated(keep='last')][col]
    s = t - t.loc[S.BASE[0]:S.BASE[1]].mean()
    if smooth:
        s = s.rolling(S.SMOOTH_YRS, center=True, min_periods=S.SMOOTH_YRS // 2 + 1).mean()
    return s


def fit_variant(mar, temp, col, smooth, rng):
    """Per-GCM posterior draws of (b1, b2) and the residual sd, for one driver."""
    fits = {}
    n_ssp = S.SSP_YEARS[1] - S.SSP_YEARS[0] + 1
    for gcm in S.GCMS:
        d = S.GCM_DIR[gcm]
        scens = [s for s in S.SCENARIOS if (mar.run == f'{d}-{s}').sum() == n_ssp]
        segs = [(S.smb_series(mar, gcm, scens[0]).loc[S.HIST_YEARS[0]:S.HIST_YEARS[1]],
                 driver(temp, gcm, scens[0], col, smooth))]
        segs += [(S.smb_series(mar, gcm, s).loc[S.SSP_YEARS[0]:S.SSP_YEARS[1]],
                  driver(temp, gcm, s, col, smooth)) for s in scens]
        starts = np.cumsum([0] + [len(y) for y, _ in segs[:-1]])
        y = np.concatenate([y.values for y, _ in segs])
        x = np.concatenate([xs.loc[y.index].values for y, xs in segs])
        f = S.fit_quadratic_ar1(x, y, N, rng, starts=starts)
        fits[gcm] = {'b1': f['beta'][:, 1], 'b2': f['beta'][:, 2],
                     'sigma': np.median(f['sigma']), 'rho': np.median(f['rho']),
                     'x_range': (x.min(), x.max())}
    gi = rng.integers(len(S.GCMS), size=N)
    k = rng.integers(N, size=N)
    b1 = np.array([fits[S.GCMS[g]]['b1'][j] for g, j in zip(gi, k)])
    b2 = np.array([fits[S.GCMS[g]]['b2'][j] for g, j in zip(gi, k)])
    return fits, b1, b2


def main():
    rng = np.random.default_rng(5)
    mar, temp = S.load_data()
    jja = pd.read_csv(S.MAR_DIR / 'gcm_temperature_jja.csv')
    temp = temp.merge(jja, on=['gcm', 'member', 'experiment', 'year'], how='left')

    # Observed drivers, annual, rel. 1995-2005
    be = pd.read_hdf(ROOT / 'data/processed/slr_processed_data.h5', 'harmonized/df_berkeley_h')
    g = be['temperature'].groupby(be.index.year).mean()
    gr = read_berkeley_earth_gridded(
        str(ROOT / 'data/raw/gmst/berkEarth_Global_TAVG_Gridded_1deg.nc'),
        lat_bounds=(59.0, 85.0), lon_bounds=(-72.0, -12.0))
    gr_yr = np.floor(gr['decimal_year']).astype(int)
    gr_mon = np.round((gr['decimal_year'] - gr_yr) * 12 + 0.5).astype(int)
    gr_jja = gr['temperature'][gr_mon.isin([6, 7, 8]).values].groupby(gr_yr[gr_mon.isin([6, 7, 8]).values]).agg(['mean', 'count'])
    gr_jja = gr_jja.loc[gr_jja['count'] == 3, 'mean']
    gr = gr['temperature'].groupby(gr_yr).agg(['mean', 'count'])
    gr = gr.loc[gr['count'] == 12, 'mean']
    years = np.arange(1960, min(g.index.max(), gr.index.max()) + 1)
    obs_end = years[-1]
    obs = {'gmst_K': g - g.loc[1995:2005].mean(), 'tgris_K': gr - gr.loc[1995:2005].mean(),
           'tgris_jja_K': gr_jja - gr_jja.loc[1995:2005].mean()}

    mou = prepare_mouginot_components(
        read_mouginot2019_greenland(str(ROOT / 'data/raw/ice_sheets/greenland/mouginot2019_data.xlsx')),
        baseline_window=(1995, 2005))
    md = mou['df']
    smb_obs = pd.Series(-md['smb_rate'].values / S.GT_TO_M_SLE,
                        index=md['decimal_year'].values.astype(int))     # Gt/yr, mass gain
    M0 = smb_obs.loc[1995:2005].mean()

    print(f'Observed Greenland T (Berkeley, 59-85N 72-12W) {years[0]}-{obs_end}; '
          f'M0 = {M0:.0f} Gt/yr')
    print(f'  Greenland T change 1972-85 -> 2010-18: '
          f'{obs["tgris_K"].loc[2010:2018].mean() - obs["tgris_K"].loc[1972:1985].mean():+.2f} K; '
          f'JJA {obs["tgris_jja_K"].loc[2010:2018].mean() - obs["tgris_jja_K"].loc[1972:1985].mean():+.2f} K; '
          f'GMST {obs["gmst_K"].loc[2010:2018].mean() - obs["gmst_K"].loc[1972:1985].mean():+.2f} K')

    res = {}
    for key, (col, smooth, label) in VARIANTS.items():
        fits, b1, b2 = fit_variant(mar, temp, col, smooth, rng)
        x = obs[col].reindex(years).values
        if smooth:
            x = S.smooth_observed(x, years, obs_end)
        rate = M0 + b1[:, None] * x + b2[:, None] * x**2               # Gt/yr
        res[key] = pd.DataFrame(rate.T, index=years)
        print(f'\n{key}: {label}')
        for gcm, f in fits.items():
            print(f'  {gcm:14s} b1 {np.median(f["b1"]):6.0f}  b2 {np.median(f["b2"]):6.1f} Gt/yr/K^n  '
                  f'resid sd {f["sigma"]:4.0f} Gt/yr  rho {f["rho"]:.2f}  '
                  f'x {f["x_range"][0]:.2f} to {f["x_range"][1]:.2f} K')

    # ── Summary against Mouginot ──
    yrs_o = np.arange(1972, 2019)
    print('\nMean SMB (Gt/yr), emulator median [5-95%] vs Mouginot')
    print('  period      Mouginot   ' + '   '.join(f'{k:>18s}' for k in VARIANTS))
    for a, b in PERIODS:
        row = f'  {a}-{b}  {smb_obs.loc[a:b].mean():8.0f}   '
        for k in VARIANTS:
            m = res[k].loc[a:b].mean(axis=0)
            row += f'   {np.median(m):5.0f} [{np.percentile(m, 5):4.0f},{np.percentile(m, 95):4.0f}]'
        print(row)
    print('\nCumulative SMB sea-level contribution 2001-2018 (mm), and annual stats 1972-2018')
    cum_obs = -smb_obs.loc[2001:2018].sum() / GT_PER_MM
    print(f'  Mouginot: {cum_obs:.1f} mm')
    for k in VARIANTS:
        cum = -res[k].loc[2001:2018].sum(axis=0) / GT_PER_MM
        med = res[k].loc[yrs_o].median(axis=1)
        r = np.corrcoef(med, smb_obs.loc[yrs_o])[0, 1]
        rmse = np.sqrt(np.mean((med - smb_obs.loc[yrs_o])**2))
        print(f'  {k}: {np.median(cum):5.1f} [{np.percentile(cum, 5):5.1f}, {np.percentile(cum, 95):5.1f}] mm;  '
              f'annual r = {r:.2f}, RMSE = {rmse:.0f} Gt/yr')

    # ── Figure ──
    colors = {'A': '0.3', 'B': '#0072B2', 'C': '#D55E00', 'D': '#009E73', 'E': '#CC79A7'}
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax = axes[0]
    ax.plot(smb_obs.index, smb_obs.values, 'o', color='#E69F00', ms=5, label='Mouginot et al., 2019')
    for k, (_, _, label) in VARIANTS.items():
        d = res[k].loc[1972:obs_end]
        ax.plot(d.index, d.median(axis=1), color=colors[k], lw=2, label=f'{k}: {label}')
        ax.fill_between(d.index, d.quantile(0.05, axis=1), d.quantile(0.95, axis=1),
                        color=colors[k], alpha=0.12, lw=0)
    ax.set_ylabel('SMB (Gt/yr)')
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    def cum(r):                       # mm SLE, zero at end of 2000
        c = (-r / GT_PER_MM).cumsum()
        return c - c.loc[2000]
    co = cum(smb_obs.loc[1972:2018])
    ax.plot(co.index, co.values, 'o', color='#E69F00', ms=5, label='Mouginot et al., 2019')
    for k in VARIANTS:
        d = res[k].loc[1972:obs_end].apply(cum, axis=0)
        ax.plot(d.index, d.median(axis=1), color=colors[k], lw=2, label=k)
        ax.fill_between(d.index, d.quantile(0.05, axis=1), d.quantile(0.95, axis=1),
                        color=colors[k], alpha=0.12, lw=0)
    ax.axhline(0, color='0.5', lw=0.5)
    ax.set_ylabel('Cumulative SMB contribution\n(mm SLE, rel. 2000)')
    ax.set_xlabel('Year')
    ax.grid(True, alpha=0.2)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
    print(f'\nSaved {OUT_FIG.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
