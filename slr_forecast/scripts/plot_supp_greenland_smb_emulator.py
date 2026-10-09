"""Supplementary figures for the Greenland SMB statistical emulator.

  supp_greenland_smb_emulator_fit.png   MAR SMB vs GCM GMST, per-GCM fits, and
                                        the joint (a_gris, b_gris) posteriors
  supp_greenland_smb_lomo.png           leave-one-GCM-out test, 2015-2100
  supp_greenland_smb_hindcast.png       emulator driven by observed GMST vs
                                        observed SMB, 1972-2024

Coefficients follow the paper's convention: sea-level contribution in mm SLE,
positive for sea-level rise, with a the quadratic and b the linear term,

    SMB contribution = c_gris + b_gris dT + a_gris dT^2,

so b_gris = -b1 / 362.5 and a_gris = -b2 / 362.5 for the module's mass-gain
coefficients (b1, b2) in Gt/yr.  T (dT in the module) is the GMST anomaly relative to
1995-2005 (11-yr centred mean).  The member draws (seed 600) and the anchor
draws (seed 602) are those of component_greenland.ipynb.

Usage:  python scripts/plot_supp_greenland_smb_emulator.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
sys.path.insert(0, str(ROOT / 'notebooks' / 'arete_mpl'))
sys.path.insert(0, str(ROOT / 'src'))
import arete_mpl
import smb_emulator as S
from component_plotting import SSP_COLORS

arete_mpl.use('poster')

FIG = ROOT / 'figures'
H5 = ROOT / 'data/processed/slr_processed_data.h5'
GT = 362.5                                    # Gt per mm SLE
N = 2000                                      # members, as in the notebook
M0_SIGMA = 57.0                               # Gt/yr, as in the notebook
SCEN_LABEL = {'ssp126': 'SSP1-2.6', 'SSP1-2.6': 'SSP1-2.6', 'ssp245': 'SSP2-4.5',
              'ssp585': 'SSP5-8.5'}
SEG_COLOR = {'history': '0.55', 'ssp126': SSP_COLORS['SSP1-2.6'],
             'ssp245': SSP_COLORS['SSP2-4.5'], 'ssp585': SSP_COLORS['SSP5-8.5']}
GCM_COLOR = dict(zip(S.GCMS, ['#1b9e77', '#d95f02', '#7570b3', '#e7298a',
                              '#66a61e', '#e6ab02', '#a6761d']))


def to_slr(gt):
    """Gt/yr of mass gain -> mm SLE/yr of sea-level rise."""
    return -np.asarray(gt) / GT


def segments_xy(g, mar, temp):
    segs = S.training_segments(g, mar, temp)
    return [(n, xs.loc[ys.index].values, ys.values) for n, ys, xs in segs]


# ── Print geometry (Figs. S6, S7) ────────────────────────────────────────
# The supplement is 12-pt article, 1-in margins, letter paper: \textwidth =
# 6.5 in. Figures are drawn at that width, saved without bbox cropping, and
# included at width=\textwidth (scale 1.0), so these are the printed sizes,
# within 2 pt of the 12-pt body text.
PRINT_W = 6.5
FS_TICK, FS_LABEL = 10, 11
SCEN_MARKER = {'history': 'o', 'ssp126': 'v', 'ssp245': 's', 'ssp585': 'o'}


def print_style():
    plt.rcParams.update({'font.size': FS_TICK, 'axes.titlesize': FS_TICK,
                         'axes.labelsize': FS_LABEL, 'xtick.labelsize': FS_TICK,
                         'ytick.labelsize': FS_TICK, 'legend.fontsize': FS_TICK,
                         'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                         'ytick.major.width': 0.6, 'xtick.major.size': 2.5,
                         'ytick.major.size': 2.5, 'lines.linewidth': 1.2})


def save_print(fig, stem):
    for ext, kw in (('png', dict(dpi=300)), ('pdf', {})):
        out = FIG / f'{stem}.{ext}'
        fig.savefig(out, **kw)
    plt.close(fig)
    print(f'saved {FIG / stem}.png/.pdf')


def ellipse90(ax, x, y, **kw):
    """90% ellipse of a bivariate normal fitted to the draws (chi2_2 = 4.605)."""
    from matplotlib.patches import Ellipse
    cov = np.cov(x, y)
    val, vec = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vec[1, 1], vec[0, 1]))
    w, h = 2 * np.sqrt(4.605 * val[::-1])
    ax.add_patch(Ellipse((np.median(x), np.median(y)), w, h, angle=ang, fill=False, **kw))
    major = vec[:, 1] * np.sqrt(4.605 * val[1])
    return np.median(x), np.median(y), major


# ── Figure 1: fits and coefficients ──────────────────────────────────────
def fig_fit(emu, mar, temp):
    print_style()
    fig, axes = plt.subplots(3, 3, figsize=(PRINT_W, 6.6), layout='constrained')
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.05, hspace=0.05)
    axes = axes.ravel()
    xg = np.linspace(-0.8, 6.7, 200)
    for k, g in enumerate(S.GCMS):
        ax = axes[k]
        for n, x, y in segments_xy(g, mar, temp):
            ax.scatter(x, to_slr(y), s=5, color=SEG_COLOR[n], marker=SCEN_MARKER[n],
                       alpha=0.75, lw=0)
        f = emu['fits'][g]
        lo, hi = f['x_range']
        xs = xg[(xg >= lo) & (xg <= hi)]
        B = f['beta']
        curve = to_slr(B[:, [0]] + B[:, [1]] * xs + B[:, [2]] * xs**2)
        ax.fill_between(xs, *np.percentile(curve, [5, 95], 0), color='k', alpha=0.25, lw=0)
        ax.plot(xs, np.median(curve, 0), 'k', lw=1.2)
        ax.set_title(f'({chr(97 + k)}) {g}', loc='left', pad=2)
        ax.axhline(0, color='0.7', lw=0.5)
        ax.set_xlim(-0.9, 6.8); ax.set_ylim(-1.2, 7.0)
        ax.set_xticks([0, 2, 4, 6]); ax.set_yticks([0, 2, 4, 6])
        ax.tick_params(pad=1.5)
        if k % 3:
            ax.tick_params(labelleft=False)
        if k < 4:                                   # panels with a GCM panel below
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r'$T$ ($^\circ$C)')
        if k % 3 == 0:
            ax.set_ylabel('SMB contribution\n(mm SLE yr$^{-1}$)')

    ax = axes[7]
    cents = {}
    for k, g in enumerate(S.GCMS):
        B = emu['fits'][g]['beta']
        cx, cy, _ = ellipse90(ax, to_slr(B[:, 1]), to_slr(B[:, 2]), color='0.15', lw=0.8)
        cents[g] = (k, cx, cy)
    # labels in a column right of the ellipses, ordered by a_gris, with leader lines
    order = sorted(S.GCMS, key=lambda g: -cents[g][2])
    for i, g in enumerate(order):
        k, cx, cy = cents[g]
        ly = 0.200 - i * 0.025
        ax.annotate(f'({chr(97 + k)})', (cx, cy), xytext=(0.53, ly), textcoords='data',
                    ha='left', va='center', fontsize=FS_TICK,
                    arrowprops=dict(arrowstyle='-', color='0.55', lw=0.5,
                                    shrinkA=1, shrinkB=0))
    ax.set_title('(h) Posteriors, 90%', loc='left', pad=2)
    ax.set_xlabel(r'$b_{gris}$ (mm SLE yr$^{-1}$ $^\circ$C$^{-1}$)')
    ax.set_ylabel(r'$a_{gris}$ (mm SLE yr$^{-1}$ $^\circ$C$^{-2}$)')
    ax.set_xlim(-0.3, 0.68); ax.set_ylim(0.0, 0.22)
    ax.set_xticks([-0.2, 0, 0.2, 0.4]); ax.set_yticks([0, 0.1, 0.2])
    ax.tick_params(pad=1.5)

    leg = axes[8]; leg.axis('off')
    h = [plt.Line2D([], [], ls='', marker=SCEN_MARKER[n], color=SEG_COLOR[n], ms=4)
         for n in ['history', 'ssp126', 'ssp245', 'ssp585']]
    h += [plt.Line2D([], [], color='k', lw=1.2),
          plt.Rectangle((0, 0), 1, 1, color='k', alpha=0.25, lw=0)]
    leg.legend(h, ['History', 'SSP1-2.6', 'SSP2-4.5', 'SSP5-8.5', 'Fit, median', 'Fit, 90%'],
               loc='center', frameon=False, borderaxespad=0, labelspacing=0.4)
    save_print(fig, 'supp_greenland_smb_emulator_fit')




# ── Figure 2: leave-one-GCM-out ──────────────────────────────────────────
def sum_ar1_var(sig, rho, n):
    """Variance of the sum of n consecutive AR(1) values."""
    lags = np.arange(1, n)
    return sig**2 / (1 - rho**2) * (n + 2 * ((n - lags)[None, :] * rho[:, None] ** lags[None, :]).sum(1))


def lomo(emu, mar, temp, rng):
    rows = []
    for test in S.GCMS:
        train = [g for g in S.GCMS if g != test]
        B = np.vstack([emu['fits'][g]['beta'] for g in train])
        SG = np.concatenate([emu['fits'][g]['sigma'] for g in train])
        RH = np.concatenate([emu['fits'][g]['rho'] for g in train])
        for n, x, y in segments_xy(test, mar, temp):
            if n == 'history':
                continue
            nyr = len(y)
            pred = (B[:, [1]] * x + B[:, [2]] * x**2).sum(1)          # Gt, anomaly
            # the held-out run is anchored to its own 1995-2005 mean, which
            # carries the weather noise of an 11-yr mean; plus AR(1) noise
            anchor_sd = np.sqrt(sum_ar1_var(SG, RH, 11)) / 11
            noise_sd = np.sqrt(sum_ar1_var(SG, RH, nyr) + (nyr * anchor_sd)**2)
            draw = pred + noise_sd * rng.standard_normal(len(pred))
            p_slr = -draw / GT                                          # mm SLE
            a_slr = -y.sum() / GT
            lo, med, hi = np.percentile(p_slr, [5, 50, 95])
            rows.append(dict(gcm=test, scen=n, actual=a_slr, pred=med, lo=lo, hi=hi,
                             miss=a_slr - med, inside=lo <= a_slr <= hi))
    return pd.DataFrame(rows)


def fig_lomo(df):
    print_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(PRINT_W, 3.9), layout='constrained',
                                 gridspec_kw={'width_ratios': [1, 1.05]})
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.06)
    for _, r in df.iterrows():
        c = SSP_COLORS[SCEN_LABEL[r.scen]]
        a1.errorbar(r.pred, r.actual, xerr=[[r.pred - r.lo], [r.hi - r.pred]], ls='',
                    marker=SCEN_MARKER[r.scen], color=c, ms=4, elinewidth=0.9, capsize=0)
    lim = [0, df[['actual', 'hi']].max().max() * 1.05]
    a1.plot(lim, lim, color='0.5', lw=0.8, ls='--')
    a1.set_xlim(lim); a1.set_ylim(lim); a1.set_aspect('equal')
    a1.set_xlabel('Emulator prediction (mm SLE)')
    a1.set_ylabel('MAR (mm SLE)')
    for s, n in [('SSP1-2.6', 'ssp126'), ('SSP2-4.5', 'ssp245'), ('SSP5-8.5', 'ssp585')]:
        a1.plot([], [], ls='', marker=SCEN_MARKER[n], color=SSP_COLORS[s], ms=4, label=s)
    a1.legend(frameon=False, loc='lower right', borderaxespad=0.2, handletextpad=0.2)

    ys = np.arange(len(df))[::-1]
    for y, (_, r) in zip(ys, df.iterrows()):
        c = SSP_COLORS[SCEN_LABEL[r.scen]]
        a2.errorbar(r.miss, y, xerr=[[r.pred - r.lo], [r.hi - r.pred]], ls='',
                    marker=SCEN_MARKER[r.scen], color=c, ms=4, elinewidth=0.9, capsize=0)
    a2.axvline(0, color='0.5', lw=0.8, ls='--')
    a2.set_yticks(ys)
    a2.set_yticklabels([f'{r.gcm} {SCEN_LABEL[r.scen]}' for _, r in df.iterrows()])
    a2.set_ylim(-0.7, len(df) - 0.3)
    a2.set_xlabel('MAR minus emulator (mm SLE)')
    for ax, lab in [(a1, '(a)'), (a2, '(b)')]:
        ax.set_title(lab, loc='left', pad=2)
        ax.tick_params(pad=1.5)
    save_print(fig, 'supp_greenland_smb_lomo')


# ── Figure 3: observed-GMST hindcast ─────────────────────────────────────
def observed_driver():
    """Calendar-year Berkeley Earth GMST rel. 1995-2005, 11-yr centred mean
    through 2024, with 2025-2029 from the SSP2-4.5 AR6 path (as in
    component_greenland.ipynb)."""
    be = pd.read_hdf(H5, 'harmonized/df_berkeley_h')
    tt = (be.index.year + (be.index.month - 0.5) / 12).values
    tm = be['temperature'].values
    hist = pd.read_hdf(H5, 'projections/temp/Historical')
    ssp = pd.read_hdf(H5, 'projections/temp/SSP2_4_5')
    off = (hist.loc[(hist.decimal_year >= 1995) & (hist.decimal_year <= 2005), 'temperature'].mean()
           - tm[(tt >= 1995) & (tt < 2006)].mean())
    obs_end = int(np.floor(tt[-1]))
    years = np.arange(1850, 2101)
    c = pd.concat([hist[hist.decimal_year < 2015], ssp])
    T = np.interp(years, c.decimal_year, c.temperature - off)
    o = years <= obs_end
    T[o] = pd.Series(tm).groupby(np.floor(tt).astype(int)).mean().reindex(years[o]).values
    return pd.Series(S.smooth_observed(T, years, obs_end), index=years)


def fig_hindcast(emu, rng):
    from slr_forecast.readers.ice_sheets import read_mouginot2019_greenland
    mou = read_mouginot2019_greenland(str(ROOT / 'data/raw/ice_sheets/greenland/mouginot2019_data.xlsx'))
    import bayesian_models as BM
    mc = BM.prepare_mouginot_components(mou, baseline_window=(1995, 2005))
    md = mc['df']
    mou_smb = pd.Series(-md['smb_rate'].values / S.GT_TO_M_SLE,
                        index=np.floor(md['decimal_year'].values).astype(int))
    M0 = float(mou_smb.loc[1995:2005].mean())
    mank = pd.read_csv(ROOT / 'data/raw/ice_sheets/greenland/mankoff/MB_SMB_D_BMB_ann.csv')
    mank = mank.set_index(mank['time'].astype(str).str[:4].astype(int))['SMB'].loc[1986:]

    x = observed_driver()
    years = np.arange(1972, int(mank.index.max()) + 1)
    xv = x.loc[years].values
    M0_draws = np.random.default_rng(602).normal(M0, M0_SIGMA, N)
    forced = M0_draws[:, None] + emu['b1'][:, None] * xv + emu['b2'][:, None] * xv**2
    sig = np.empty(N); rho = np.empty(N)
    for g in S.GCMS:
        idx = np.where(emu['gcm'] == g)[0]
        sig[idx] = emu['fits'][g]['sigma'][idx]; rho[idx] = emu['fits'][g]['rho'][idx]
    e = np.zeros((N, len(years)))
    e[:, 0] = sig / np.sqrt(1 - rho**2) * rng.standard_normal(N)
    for k in range(1, len(years)):
        e[:, k] = rho * e[:, k - 1] + sig * rng.standard_normal(N)
    weather = forced + e

    print_style()
    fig, ax = plt.subplots(figsize=(PRINT_W, 3.4), layout='constrained')
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03)
    ax.fill_between(years, *np.percentile(to_slr(weather), [5, 95], 0), color='#2E8B57',
                    alpha=0.12, lw=0, label='Emulator with weather noise, 90%')
    ax.fill_between(years, *np.percentile(to_slr(forced), [5, 95], 0), color='#2E8B57',
                    alpha=0.35, lw=0, label='Emulator forced response, 90%')
    ax.plot(years, np.median(to_slr(forced), 0), color='#2E8B57', lw=1.2,
            label='Emulator forced response, median')
    ax.plot(mou_smb.index, to_slr(mou_smb.values), 'o', color='k', ms=3,
            label='Mouginot et al. (2019), sets $c_{gris}$')
    ax.plot(mank.index, to_slr(mank.values), 's', mfc='white', mec='0.25', mew=0.8, ms=3.5,
            label='Mankoff et al. (2021), withheld')
    ax.set_xlim(1970, years[-1] + 1)
    ax.set_ylabel('SMB contribution (mm SLE yr$^{-1}$)')
    ax.set_xlabel('Year')
    ax.tick_params(pad=1.5)
    sec = ax.secondary_yaxis('right', functions=(lambda v: -v * GT, lambda v: -v / GT))
    sec.set_ylabel('SMB (Gt yr$^{-1}$)')
    sec.tick_params(pad=1.5)
    ax.legend(frameon=False, loc='upper left', ncol=2, borderaxespad=0.3,
              columnspacing=1.0, handletextpad=0.4, labelspacing=0.3)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.38 * (hi - lo))
    save_print(fig, 'supp_greenland_smb_hindcast')

    # numbers for the text
    f_med = pd.Series(np.median(forced, 0), index=years)
    print(f'M0 = {M0:.1f} Gt/yr')
    for a, b in [(1972, 1985), (1986, 1999), (2000, 2009), (2010, 2018), (2019, int(years[-1]))]:
        mo = mou_smb.loc[a:b].mean() if a <= 2018 else np.nan
        mk = mank.loc[max(a, 1986):b].mean() if b >= 1986 else np.nan
        print(f'  {a}-{b}: emulator {f_med.loc[a:b].mean():5.0f}  Mouginot {mo:5.0f}  '
              f'Mankoff {mk:5.0f} Gt/yr')
    wl, wh = np.percentile(weather, [5, 95], 0)
    for name, s in [('Mouginot', mou_smb.loc[1972:2018]), ('Mankoff', mank)]:
        yy = s.index.values; k = np.searchsorted(years, yy)
        out_ = (s.values < wl[k]) | (s.values > wh[k])
        print(f'  {name}: {out_.sum()} of {len(yy)} years outside the weather band: '
              f'{list(yy[out_])}')


def main():
    mar, temp = S.load_data()
    emu = S.fit_smb_emulator(N, seed=600)
    print('Coefficients, median [5th, 95th] (mm SLE yr^-1 degC^-n, + = sea-level rise):')
    for g in S.GCMS:
        B = emu['fits'][g]['beta']
        bq = np.percentile(to_slr(B[:, 1]), [50, 5, 95]); aq = np.percentile(to_slr(B[:, 2]), [50, 5, 95])
        print(f'  {g:14s} b_gris {bq[0]:.3f} [{bq[1]:.3f}, {bq[2]:.3f}]   '
              f'a_gris {aq[0]:.3f} [{aq[1]:.3f}, {aq[2]:.3f}]')
    fig_fit(emu, mar, temp)
    df = lomo(emu, mar, temp, np.random.default_rng(5))
    fig_lomo(df)
    print('Leave-one-GCM-out, integrated 2015-2100 (mm SLE):')
    print(df.round(1).to_string(index=False))
    print(f'  {(~df.inside).sum()} of {len(df)} outside the 90% range; misses '
          f'{df.miss.min():+.1f} to {df.miss.max():+.1f} mm; '
          f'median absolute miss {df.miss.abs().median():.1f} mm')
    fig_hindcast(emu, np.random.default_rng(7))


if __name__ == '__main__':
    main()
