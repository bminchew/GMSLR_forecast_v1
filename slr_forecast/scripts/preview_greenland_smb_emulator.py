"""Preview figures: Greenland projections with the RCM-emulator SMB in place of
the adopted parametric SMB (C_T = -300 +/- 80, C_T2 = -50 +/- 30).

Nothing in the pipeline is changed.  The new SMB ensemble is built here and
paired, member by member, with the discharge samples stored in
component_results.h5, using the same per-member Arctic-amplification draw the
notebook used for discharge (default_rng(700), third block of N normals) and
the same AR6 warming-path offsets.

New SMB, member i:
    SMB_i(t) = M0 + b1 x + b2 x^2,   x = AA_i * dGMST_i(t)   (rel. 1995-2005)
with (b1, b2) a posterior draw from the RACMO or MAR quadratic emulator
(alternating members, equal weight; emulator_smb_glaude2024.py), calendar
anchoring, and M0 the Mouginot 1995-2005 mean SMB.  As in the notebook, the
observed Mouginot SMB is spliced in through 2018 and the projection is shifted
to meet it.

Figures (figures/preview_smb_emulator_*.png):
  timeseries  SMB, discharge and total Greenland, 2000-2100, new vs adopted
  pdf2100     total Greenland at 2100 per scenario, new vs adopted
  hindcast    unspliced emulator driven by observed GMST vs observed SMB
  fits        per-RCM emulator fits vs CESM2 Greenland temperature

Usage:  python scripts/preview_greenland_smb_emulator.py
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'notebooks'))
sys.path.insert(0, str(ROOT / 'notebooks' / 'arete_mpl'))
sys.path.insert(0, str(ROOT / 'src'))
import prototype_smb_emulator_mankoff as P
import emulator_smb_noel2021 as E
from warming_paths import warming_z, offsets_on_grid
from component_plotting import SSP_COLORS
from slr_forecast.readers.ice_sheets import read_mouginot2019_greenland
from bayesian_models import prepare_mouginot_components

try:
    import arete_mpl
    arete_mpl.use('poster')
except Exception:
    pass

H5 = ROOT / 'data/processed/slr_processed_data.h5'
RES = ROOT / 'data/processed/component_results.h5'
GLAUDE = ROOT / 'data/raw/ice_sheets/greenland/glaude2024/glaude2024_integrated_smb.csv'
FIG = ROOT / 'figures'
N = 2000
AA, AA_SIGMA = 2.5, 0.5
GT_PER_MM = 362.5
SSPS = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
SSP_KEY = {'SSP1-2.6': 'SSP1_2_6', 'SSP2-4.5': 'SSP2_4_5',
           'SSP3-7.0': 'SSP3_7_0', 'SSP5-8.5': 'SSP5_8_5'}
LABEL = {'SSP1-2.6': '2 °C (SSP1-2.6)', 'SSP2-4.5': '3 °C (SSP2-4.5)',
         'SSP3-7.0': '4 °C (SSP3-7.0)', 'SSP5-8.5': 'SSP5-8.5'}
OLD = dict(C_T=-300.0, C_T_sigma=80.0, C_T2=-50.0, C_T2_sigma=30.0, SMB_0=380.0)


def emulator_draws(rng):
    """Quadratic (M, b1, b2) posterior draws for RACMO and MAR."""
    P.PRIOR_SD.update({'T': 1000.0, 'T2': 500.0})
    racmo, T_h, T_p, S_h, S_p = E.load()
    ens = pd.concat([T_h.mean(axis=1),
                     T_p[['SSP5-8.5-parent', 'SSP5-8.5-1', 'SSP5-8.5-2']].mean(axis=1)])
    xa = E.anom(ens)
    g = pd.read_csv(GLAUDE, index_col=0)
    out, data = {}, {}
    for m in ['RACMO', 'MAR', 'HIRHAM']:
        s = g[f'{m}_sheet'].dropna()
        yrs = s.index.intersection(xa.index)
        x, y = xa.loc[yrs].values, E.anom(s).loc[yrs].values
        X, names = E.design(x, 'quadratic')
        r = P.fit(X, y, names, rng)
        out[m] = r['beta'][rng.choice(len(r['sigma']), N, replace=False)]
        data[m] = (x, y)
    return out, data


def gmst_paths(grid):
    """Observed Berkeley Earth GMST spliced to CMIP6 SSP medians, rel. 1995-2005,
    on an annual grid, plus per-member AR6 warming-path offsets."""
    be = pd.read_hdf(H5, 'harmonized/df_berkeley_h')
    t_be = (be.index.year + (be.index.month - 0.5) / 12.0).values
    T_be = be['temperature'].values
    T_be = T_be - T_be[(t_be >= 1995) & (t_be < 2006)].mean()
    hist = pd.read_hdf(H5, 'projections/temp/Historical')
    hb = hist.loc[(hist.decimal_year >= 1995) & (hist.decimal_year <= 2005), 'temperature'].mean()
    z = warming_z(N)
    paths, offs = {}, {}
    for ssp in SSPS:
        d = pd.read_hdf(H5, f'projections/temp/{SSP_KEY[ssp]}')
        c = pd.concat([hist[hist.decimal_year < 2015], d]).sort_values('decimal_year')
        t_c, T_c = c.decimal_year.values, c.temperature.values - hb
        t = np.concatenate([t_be, t_c[t_c > t_be[-1]]])
        T = np.concatenate([T_be, T_c[t_c > t_be[-1]]])
        yr = np.floor(t).astype(int)
        ann = pd.Series(T).groupby(yr).mean()
        paths[ssp] = np.interp(grid, ann.index.values.astype(float), ann.values)
        offs[ssp] = offsets_on_grid(d, z, grid + 0.5, zero_through=t_be[-1])
    return paths, offs


def cumulative_sle_m(rate_gt, grid, base=2000):
    cum = -np.cumsum(rate_gt, axis=-1) / GT_PER_MM / 1000.0
    return cum - cum[..., [np.argmin(np.abs(grid - base))]]


def splice_obs(samples, grid, obs_t, obs_H):
    obs = np.interp(grid, obs_t, obs_H, left=np.nan, right=np.nan)
    mask = ~np.isnan(obs)
    k = np.max(np.where(mask)[0])
    out = samples.copy()
    out[:, mask] = obs[mask]
    fut = grid > grid[k]
    out[:, fut] = samples[:, fut] + (obs[k] - samples[:, [k]])
    return out


def q(a, axis=0):
    return np.percentile(a, [5, 50, 95], axis=axis)


def main():
    rng = np.random.default_rng(20261003)
    with h5py.File(RES, 'r') as f:
        grid = f['proj_years'][:]
        old_tot = {s: f[f'greenland/projections/{s}/samples'][:] for s in SSPS}
        old_smb = {s: f[f'greenland/projections_smb/{s}/samples'][:] for s in SSPS}
        dis = {s: f[f'greenland/projections_discharge/{s}/samples'][:] for s in SSPS}

    rng_dyn = np.random.default_rng(700)          # notebook's discharge stream
    rng_dyn.normal(0, 1, N); rng_dyn.normal(0, 1, N)
    aa = rng_dyn.normal(AA, AA_SIGMA, N)

    B, fitdata = emulator_draws(rng)
    which = np.where(np.arange(N) % 2 == 0, 'RACMO', 'MAR')
    b = np.where((which == 'RACMO')[:, None], B['RACMO'], B['MAR'])

    mou = prepare_mouginot_components(
        read_mouginot2019_greenland(str(ROOT / 'data/raw/ice_sheets/greenland/mouginot2019_data.xlsx')),
        baseline_window=(1995, 2005))
    t_obs, H_obs = mou['time_smb'], mou['H_smb']
    H_obs = H_obs - np.interp(2000.0, t_obs, H_obs)
    rate_obs = -np.gradient(H_obs * 1000 * GT_PER_MM, t_obs)       # Gt/yr, mass-gain
    M0 = rate_obs[(t_obs >= 1995) & (t_obs < 2006)].mean()
    print(f'M0 (Mouginot 1995-2005 mean SMB) = {M0:.0f} Gt/yr')

    paths, offs = gmst_paths(grid)
    new_smb, new_raw, new_tot = {}, {}, {}
    for s in SSPS:
        dT = paths[s][None, :] + offs[s]
        x = aa[:, None] * dT
        rate = M0 + b[:, 1:2] * x + b[:, 2:3] * x**2
        raw = cumulative_sle_m(rate, grid)
        new_raw[s] = raw
        new_smb[s] = splice_obs(raw, grid, t_obs, H_obs)
        new_tot[s] = new_smb[s] + dis[s]

    i2100 = np.argmin(np.abs(grid - 2100))
    print('2100 (mm), median [5th, 95th]:')
    for s in SSPS:
        for lab, a in [('SMB new', new_smb[s]), ('SMB adopted', old_smb[s]),
                       ('total new', new_tot[s]), ('total adopted', old_tot[s])]:
            lo, md, hi = q(a[:, i2100] * 1000)
            print(f'  {s} {lab:13s} {md:5.0f} [{lo:.0f}, {hi:.0f}]')

    # ── Figure 1: time series ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    for ax, (title, new, old) in zip(axes, [
            ('SMB', new_smb, old_smb), ('Discharge (unchanged)', dis, dis),
            ('Total Greenland', new_tot, old_tot)]):
        for s in SSPS:
            lo, md, hi = q(new[s] * 1000)
            c = SSP_COLORS.get(s, None)
            ax.fill_between(grid, lo, hi, color=c, alpha=0.15, lw=0)
            ax.plot(grid, md, color=c, lw=2, label=LABEL[s])
            if old is not new:
                ax.plot(grid, np.median(old[s] * 1000, 0), color=c, lw=2.2, ls=':', zorder=5)
        ax.axhline(0, color='0.6', lw=0.6)
        w = (grid >= 2000) & (grid <= 2100)
        lo_all = min(np.percentile(new[s_] * 1000, 5, 0)[w].min() for s_ in SSPS)
        hi_all = max(np.percentile(new[s_] * 1000, 95, 0)[w].max() for s_ in SSPS)
        pad = 0.05 * (hi_all - lo_all)
        ax.set_ylim(lo_all - pad, hi_all + pad)
        ax.set_xlim(2000, 2100); ax.set_title(title); ax.set_xlabel('Year')
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel('Sea-level contribution since 2000 (mm)')
    axes[0].plot([], [], 'k-', lw=2, label='emulator (median, 90% band)')
    axes[0].plot([], [], 'k:', lw=2.2, label='current model (median)')
    axes[0].legend(fontsize=10, loc='upper left')
    fig.tight_layout()
    fig.savefig(FIG / 'preview_smb_emulator_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Figure 2: 2100 distributions ──
    ar6 = {'SSP1-2.6': 55, 'SSP2-4.5': 83, 'SSP3-7.0': 114}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    xs = np.linspace(-50, 450, 600)
    for ax, s in zip(axes, SSPS[:3]):
        for lab, a, ls, alpha in [('emulator SMB', new_tot[s], '-', 0.25),
                                  ('adopted SMB', old_tot[s], '--', 0.0)]:
            v = a[:, i2100] * 1000
            kd = gaussian_kde(v)(xs)
            ax.plot(xs, kd, color=SSP_COLORS.get(s), ls=ls, lw=2,
                    label=f'{lab}: {np.median(v):.0f} [{np.percentile(v, 5):.0f}, {np.percentile(v, 95):.0f}] mm')
            if alpha:
                ax.fill_between(xs, kd, color=SSP_COLORS.get(s), alpha=alpha, lw=0)
        ax.axvline(ar6[s], color='0.4', lw=1.2, ls=':', label=f'AR6 median {ar6[s]} mm')
        ax.set_title(f'Total Greenland at 2100, {LABEL[s]}')
        ax.set_xlabel('mm since 2000'); ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
        ax.set_xlim(-50, 450)
    axes[0].set_ylabel('Probability density')
    fig.tight_layout()
    fig.savefig(FIG / 'preview_smb_emulator_pdf2100.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Figure 3: hindcast check (unspliced, observed GMST through 2024) ──
    s = 'SSP2-4.5'
    sel = (grid >= 1960) & (grid <= 2024)
    dT = paths[s]
    x = aa[:, None] * dT[None, :]
    rate_new = M0 + b[:, 1:2] * x + b[:, 2:3] * x**2
    c_t = np.random.default_rng(600).normal(OLD['C_T'], OLD['C_T_sigma'], N)
    c_t2 = np.random.default_rng(600).normal(OLD['C_T2'], OLD['C_T2_sigma'], N)
    dTp = np.clip(dT, 0, None)[None, :]
    rate_old = OLD['SMB_0'] + c_t[:, None] * dTp + c_t2[:, None] * dTp**2
    mank = pd.read_csv(ROOT / 'data/raw/ice_sheets/greenland/mankoff/MB_SMB_D_BMB_ann.csv').set_index('time')['SMB']
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    ax = axes[0]
    for lab, rate, col in [('emulator (RACMO+MAR)', rate_new, '#2a78d6'),
                           ('adopted parametric', rate_old, '#eb6834')]:
        lo, md, hi = q(rate[:, sel], 0)
        ax.fill_between(grid[sel], lo, hi, color=col, alpha=0.2, lw=0)
        ax.plot(grid[sel], md, color=col, lw=2, label=lab)
    ax.plot(t_obs, rate_obs, 'k.-', lw=1, ms=4, label='Mouginot (RACMO, reanalysis)')
    ax.plot(mank.loc[1986:2024].index + 0.5, mank.loc[1986:2024].values, '.-', color='0.5',
            lw=1, ms=4, label='Mankoff 3-RCM mean (1986+)')
    ax.set_ylabel('SMB (Gt yr$^{-1}$)'); ax.set_xlabel('Year'); ax.set_title('Annual SMB, driven by observed GMST')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
    ax = axes[1]
    for lab, rate, col in [('emulator (RACMO+MAR)', rate_new, '#2a78d6'),
                           ('adopted parametric', rate_old, '#eb6834')]:
        cum = cumulative_sle_m(rate, grid) * 1000
        lo, md, hi = q(cum[:, sel], 0)
        ax.fill_between(grid[sel], lo, hi, color=col, alpha=0.2, lw=0)
        ax.plot(grid[sel], md, color=col, lw=2, label=lab)
    ax.plot(t_obs, H_obs * 1000, 'k.-', lw=1, ms=4, label='Mouginot cumulative')
    mk = -np.cumsum(mank.loc[1972:2024].values) / GT_PER_MM
    mk_t = mank.loc[1972:2024].index.values + 0.5
    ax.plot(mk_t, mk - np.interp(2000.5, mk_t, mk), '.-', color='0.5', lw=1, ms=4,
            label='Mankoff cumulative (pre-1986 reconstruction)')
    ax.axhline(0, color='0.6', lw=0.6)
    ax.set_ylabel('Cumulative SMB, mm SLE rel. 2000'); ax.set_xlabel('Year')
    ax.set_title('Cumulative, rel. 2000 (not spliced)'); ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / 'preview_smb_emulator_hindcast.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Figure 4: emulator fits ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True)
    xg = np.linspace(-2.3, 8, 200)
    for ax, m, col in zip(axes, ['RACMO', 'MAR', 'HIRHAM'], ['#2a78d6', '#eb6834', '#5f5e5a']):
        x, y = fitdata[m]
        ax.plot(x, y, '.', color='0.45', ms=5, label='annual SMB anomaly')
        mu = B[m][:, [0]] + B[m][:, [1]] * xg + B[m][:, [2]] * xg**2
        lo, md, hi = q(mu, 0)
        ax.fill_between(xg, lo, hi, color=col, alpha=0.3, lw=0)
        ax.plot(xg, md, color=col, lw=2, label='quadratic fit (median, 90% band)')
        bm = np.median(B[m], 0)
        ax.set_title(f'{m}-CESM2 SSP5-8.5'
                     + ('' if m != 'HIRHAM' else ' (sensitivity only)'))
        ax.text(0.03, 0.05, f'b1 = {bm[1]:.0f} Gt/yr/K\nb2 = {bm[2]:.1f} Gt/yr/K²',
                transform=ax.transAxes, fontsize=11)
        ax.axhline(0, color='0.6', lw=0.6); ax.axvline(0, color='0.6', lw=0.6)
        ax.set_xlabel('CESM2 Greenland T anomaly rel. 1995–2005 (K)'); ax.grid(True, alpha=0.2)
    axes[0].set_ylabel('SMB anomaly rel. 1995–2005 (Gt yr$^{-1}$)')
    axes[0].legend(fontsize=10, loc='upper right')
    fig.tight_layout()
    fig.savefig(FIG / 'preview_smb_emulator_fits.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('saved figures/preview_smb_emulator_{timeseries,pdf2100,hindcast,fits}.png')


if __name__ == '__main__':
    main()
