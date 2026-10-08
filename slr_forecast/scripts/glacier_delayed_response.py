"""Diagnostic: a delayed-response (one-stage) glacier model against observations.

The production glacier model is rate = b T + c, fitted to GlaMBIE
2000-2023. Against Zemp et al. (2019) its sensitivity is not stationary:
fitted to 1962-1989 alone b is about -0.2 mm/yr/K, to 1990-2016 about 1.1
(3.5 sigma apart; glacier_zemp_diagnostic.py). A glacier that responds to
warming with a delay behaves this way. This script tests the standard
one-stage linear response (Johannesson et al. 1989; Christian et al. 2018;
Roe et al. 2021):

    rate(t) = b (T(t) - T_eq(t)),      dT_eq/dt = (T - T_eq) / tau,

with T_eq the temperature the glaciers are adjusted to and T_eq = T_eq0 in
1850. Discretised annually, T_eq(t) = phi T_eq(t-1) + (1 - phi) T(t) with
phi = exp(-1/tau), so for fixed tau

    rate = b X1(t) + kappa X2(t),   X1 = T - F0,   X2 = phi^(t - 1850),
    kappa = -b T_eq0,

where F0 is the filter started from zero. The rate is linear in (b, kappa),
and as tau -> infinity X1 -> T and X2 -> 1, i.e. the production model
b T + c. Under steady warming the rate tends to b tau dT/dt; once warming
stops it decays to zero over tau.

Data (rates, mm/yr SLE, all 19 RGI regions):
  (a) Zemp et al. (2019) 1962-1999 + GlaMBIE 2000-2023
  (b) Zemp et al. (2019) 1962-2016 + GlaMBIE 2000-2023
Zemp's geodetic error is fully correlated across years, so it also absorbs
a constant offset between the datasets (in (b) it absorbs the +0.18 mm/yr
overlap difference; the price is using 2000-2016 twice). Zemp's 95%
intervals are converted to 1 sigma (slr_data_readers.read_zemp2019_global).
Hydrological-year regressors use 0.25 X(Y-1) + 0.75 X(Y).

Inference: for each tau on a grid (3 to 3000 yr, plus the linear limit),
(b, kappa) have Gaussian priors b ~ N(0.61, 2.0) (the production prior
location; not truncated here) and kappa ~ N(0, 5) mm/yr, so the marginal
likelihood p(y | tau) is analytic. Tau priors: lognormal with 90% range
20-200 yr (Zekollari et al. 2020: Alpine mean 50 +/- 28 yr; Roe et al.
2021: 10-400 yr, large ice masses at the long end), and log-uniform.

Reported: likelihood profile in tau; posterior of tau and the b-tau trade-
off; the early/late stationarity check refitted with the delayed model;
the 1900-1950 hindcast under two assumptions about the 1850 imbalance
(kappa free, or glaciers in balance with 1850-1879); and glacier sea-level
rise 2000-2100 on the AR6 median GMST paths, per tau, with Rounce et al.
(2023) as an external plausibility check.

Omissions: no area or volume shrinkage (b is constant, so late high-
warming loss is overstated), no volume cap, a one-stage rather than three-
stage response, GMST as the glacier predictor as in production, median
warming paths only. Nothing here feeds the pipeline.

Usage:  python scripts/glacier_delayed_response.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
from slr_data_readers import read_zemp2019_global

H5 = ROOT / 'data/processed/slr_processed_data.h5'
ZEMP = ROOT / 'data/raw/glaciers/zemp2019/Zemp_etal_results_global.csv'
GLAMBIE = (ROOT / 'data/raw/glaciers/glambie/GlaMBIE_Data_DOI_10.5904_wgms-glambie-2024-07/'
           'glambie_results_20240716/calendar_years/0_global.csv')
OUT_FIG = ROOT / 'figures/diagnostics/glacier_delayed_response.png'
GT_PER_MM = 362.5
T0 = 1850
TAUS = np.r_[np.geomspace(3.0, 3000.0, 60), np.inf]
PRIOR_MEAN = np.array([0.61, 0.0])           # b (mm/yr/K), kappa (mm/yr)
PRIOR_SD = np.array([2.0, 5.0])
N_DRAW = 4000
SSPS = {'SSP1-2.6': 'SSP1_2_6', 'SSP2-4.5': 'SSP2_4_5', 'SSP3-7.0': 'SSP3_7_0', 'SSP5-8.5': 'SSP5_8_5'}
PROD_2100 = {'SSP1-2.6': 122, 'SSP2-4.5': 149, 'SSP3-7.0': 175, 'SSP5-8.5': 199}   # production medians


def temperatures():
    """Annual GMST rel. 1995-2005: observed (Berkeley) through the last full
    year, then each SSP's AR6 median, rebased as in component_greenland,
    held flat after 2099. Returns {ssp: pd.Series 1850-2100}, observed series."""
    be = pd.read_hdf(H5, 'harmonized/df_berkeley_h')
    obs = be['temperature'].groupby(be.index.year).agg(['mean', 'count'])
    obs = obs.loc[obs['count'] == 12, 'mean']
    hist = pd.read_hdf(H5, 'projections/temp/Historical')
    tt = be.index.year + (be.index.month - 0.5) / 12.0
    # CMIP6 -> Berkeley Earth offset over 1995-2005, as in component_greenland
    off = (hist.loc[(hist['decimal_year'] >= 1995) & (hist['decimal_year'] <= 2005), 'temperature'].mean()
           - be['temperature'].values[(tt >= 1995) & (tt < 2006)].mean())
    base = obs.loc[1995:2005].mean()
    obs = obs - base
    out = {}
    years = np.arange(T0, 2101)
    for ssp, key in SSPS.items():
        d = pd.read_hdf(H5, f'projections/temp/{key}')
        s = (d.groupby(np.floor(d['decimal_year']).astype(int))['temperature'].mean() - off - base)
        T = pd.Series(np.nan, index=years)
        T.loc[obs.index.min():obs.index.max()] = obs.reindex(np.arange(obs.index.min(), obs.index.max() + 1)).values
        fut = years > obs.index.max()
        T[fut] = s.reindex(years[fut]).values
        T = T.ffill()
        out[ssp] = T
    return out, obs


def regressors(T, tau):
    """X1 = T - F0 and X2 = phi^(t - T0) on the years of T (from T0)."""
    if np.isinf(tau):
        return T.values.copy(), np.ones(len(T))
    phi = np.exp(-1.0 / tau)
    F = np.empty(len(T)); F[0] = 0.0
    for i in range(1, len(T)):
        F[i] = phi * F[i - 1] + (1 - phi) * T.values[i]
    return T.values - F, phi ** (T.index.values - T0)


def design(T, tau, zy, gy):
    X1, X2 = regressors(T, tau)
    s1, s2 = pd.Series(X1, index=T.index), pd.Series(X2, index=T.index)
    hyd = lambda s: 0.25 * s.reindex(zy - 1).values + 0.75 * s.reindex(zy).values
    return np.vstack([np.column_stack([hyd(s1), hyd(s2)]),
                      np.column_stack([s1.reindex(gy).values, s2.reindex(gy).values])])


def data(variant):
    z = read_zemp2019_global(str(ZEMP)) * 1000.0
    z = z.loc[1962:1999] if variant == 'a' else z.loc[1962:2016]
    g = pd.read_csv(GLAMBIE)
    gy = g['start_dates'].astype(int).values
    y = np.r_[z['rate'].values, -g['combined_gt'].values / GT_PER_MM]
    nz = len(z)
    C = np.zeros((len(y), len(y)))
    C[:nz, :nz] = np.diag(z['sigma_ind'].values ** 2) + np.outer(z['sigma_cor'].values, z['sigma_cor'].values)
    C[nz:, nz:] = np.diag((g['combined_gt_errors'].values / GT_PER_MM) ** 2)
    return y, C, z.index.values, gy, nz


def posterior(X, y, C):
    """Gaussian prior on (b, kappa): posterior mean/cov and log marginal likelihood."""
    P0 = np.diag(PRIOR_SD ** 2)
    S = C + X @ P0 @ X.T
    Si = np.linalg.inv(S)
    r = y - X @ PRIOR_MEAN
    logml = -0.5 * (r @ Si @ r + np.linalg.slogdet(2 * np.pi * S)[1])
    Ci = np.linalg.inv(C)
    V = np.linalg.inv(np.linalg.inv(P0) + X.T @ Ci @ X)
    m = V @ (np.linalg.inv(P0) @ PRIOR_MEAN + X.T @ Ci @ y)
    return m, V, logml


def tau_prior(kind):
    lt = np.log(np.where(np.isinf(TAUS), 1e9, TAUS))
    if kind == 'lognormal 20-200':
        mu, s = np.log(np.sqrt(20 * 200)), np.log(10.0) / (2 * 1.645)
        lp = -0.5 * ((lt - mu) / s) ** 2
    else:                                    # log-uniform over the finite grid, linear limit included
        lp = np.zeros_like(lt)
    w = np.r_[np.gradient(np.log(TAUS[:-1])), np.mean(np.gradient(np.log(TAUS[:-1])))]
    return lp + np.log(w)


def q(v, w=None):
    v = np.asarray(v)
    if w is None:
        return np.percentile(v, [50, 5, 95])
    o = np.argsort(v); c = np.cumsum(w[o]); c /= c[-1]
    return np.interp([0.5, 0.05, 0.95], c, v[o])


def fmt(t):
    return f'{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]'


def ftau(x):
    return 'linear' if np.isinf(x) else f'{x:.0f}'


def main():
    rng = np.random.default_rng(620)
    Tssp, Tobs = temperatures()
    T = Tssp['SSP2-4.5']                     # identical to observations before the projections
    print(f'GMST rel. 1995-2005: observed {Tobs.index.min()}-{Tobs.index.max()}; '
          f'AR6 median paths after, flat after 2099')

    res = {}
    for variant in ('a', 'b'):
        y, C, zy, gy, nz = data(variant)
        fits = [posterior(design(T, tau, zy, gy), y, C) for tau in TAUS]
        logml = np.array([f[2] for f in fits])
        res[variant] = (fits, logml, y, C, zy, gy, nz)
        lab = {'a': 'Zemp 1962-1999 + GlaMBIE 2000-2023', 'b': 'Zemp 1962-2016 + GlaMBIE 2000-2023'}[variant]
        print(f'\n=== Data ({variant}) {lab}: {len(y)} rates ===')
        sel = [3, 10, 20, 30, 50, 100, 200, 500, 1000, 3000]
        idx = [np.argmin(np.abs(TAUS[:-1] - s)) for s in sel] + [len(TAUS) - 1]
        print('  log marginal likelihood relative to the linear model (tau = inf):')
        print('  ' + ', '.join(f'{ftau(TAUS[i])} {logml[i] - logml[-1]:+.1f}' for i in idx))
        ib = np.argmax(logml)
        print(f'  maximum at tau = {ftau(TAUS[ib])} yr ({logml[ib] - logml[-1]:+.1f}); '
              f'b there {fits[ib][0][0]:.2f} ± {np.sqrt(fits[ib][1][0, 0]):.2f} mm/yr/K')
        for kind in ('lognormal 20-200', 'log-uniform'):
            lw = logml + tau_prior(kind)
            w = np.exp(lw - lw.max()); w /= w.sum()
            tq = q(np.where(np.isinf(TAUS), 1e5, TAUS), w)
            bm = np.array([f[0][0] for f in fits])
            print(f'  tau prior {kind:17s}: tau {ftau(tq[0])} [{ftau(tq[1])}, {ftau(tq[2])}] yr '
                  f'(1e5 = linear); P(linear) {w[-1]:.2f}; b {fmt(q(bm, w))}')
        print('  b-tau trade-off (posterior mean b, and b*tau, by tau):')
        print('  ' + ', '.join(f'tau {ftau(TAUS[i])}: b {fits[i][0][0]:.2f}'
                               + (f' (b tau {fits[i][0][0] * TAUS[i]:.0f} mm/K)' if np.isfinite(TAUS[i]) else '')
                               for i in idx[1::2]))

    # ── Stationarity: Zemp windows refitted at fixed tau ──
    print('\n=== Stationarity: b fitted to Zemp 1962-1989 vs 1990-2016, at fixed tau ===')
    z = read_zemp2019_global(str(ZEMP)) * 1000.0
    for tau in (10.0, 20.0, 30.0, 50.0, 100.0, np.inf):
        bs = []
        for a_, b_ in ((1962, 1989), (1990, 2016)):
            zz = z.loc[a_:b_]
            X = design(T, tau, zz.index.values, np.array([], dtype=int))
            Cz = np.diag(zz['sigma_ind'].values ** 2) + np.outer(zz['sigma_cor'].values, zz['sigma_cor'].values)
            m, V, _ = posterior(X, zz['rate'].values, Cz)
            bs.append((m[0], np.sqrt(V[0, 0])))
        dz = (bs[1][0] - bs[0][0]) / np.hypot(bs[0][1], bs[1][1])
        print(f'  tau {ftau(tau):>6s}: early b {bs[0][0]:5.2f} ± {bs[0][1]:.2f}, late b {bs[1][0]:5.2f} ± {bs[1][1]:.2f}'
              f'  ({dz:+.1f} sigma)')

    # ── Draws from the full posterior (variant a, lognormal prior) ──
    def draws(variant, kind, tau_fixed=None):
        fits, logml, *_ = res[variant]
        if tau_fixed is None:
            lw = logml + tau_prior(kind)
            w = np.exp(lw - lw.max()); w /= w.sum()
            k = rng.choice(len(TAUS), size=N_DRAW, p=w)
        else:
            i_t = (len(TAUS) - 1 if np.isinf(tau_fixed)
                   else int(np.argmin(np.abs(np.where(np.isinf(TAUS), 1e9, TAUS) - tau_fixed))))
            k = np.full(N_DRAW, i_t)
        bk = np.empty((N_DRAW, 2))
        for j in np.unique(k):
            ii = np.where(k == j)[0]
            bk[ii] = rng.multivariate_normal(fits[j][0], fits[j][1], size=len(ii))
        return TAUS[k], bk

    def trajectories(T, tau, bk):
        out = np.empty((len(tau), len(T)))
        for t_ in np.unique(tau):
            ii = np.where(tau == t_)[0]
            X1, X2 = regressors(T, t_)
            out[ii] = bk[ii, :1] * X1[None, :] + bk[ii, 1:] * X2[None, :]
        return pd.DataFrame(out, columns=T.index)

    cum = lambda R, y0, y1: R.loc[:, y0 + 1:y1].sum(axis=1)       # mm, years y0+1..y1

    # ── Hindcast ──
    print('\n=== Hindcast (data a): cumulative glacier contribution relative to 2000, mm ===')
    tau_a, bk_a = draws('a', 'lognormal 20-200')
    R = trajectories(T, tau_a, bk_a)
    lin_tau, lin_bk = draws('a', None, tau_fixed=np.inf)
    Rl = trajectories(T, lin_tau, lin_bk)
    # Balance with 1850-1879: T_eq0 = mean T 1850-1879, so kappa = -b T_eq0.
    Teq0 = T.loc[1850:1879].mean()
    bk_bal = bk_a.copy(); bk_bal[:, 1] = -bk_bal[:, 0] * Teq0
    Rb = trajectories(T, tau_a, bk_bal)
    for y_ in (1900, 1920, 1950, 1962):
        print(f'  {y_}: linear {-np.median(cum(Rl, y_, 2000)):6.1f};  delayed, kappa free '
              f'{-np.median(cum(R, y_, 2000)):6.1f} [{-np.percentile(cum(R, y_, 2000), 95):6.1f}, '
              f'{-np.percentile(cum(R, y_, 2000), 5):6.1f}];  delayed, in balance with 1850-1879 '
              f'{-np.median(cum(Rb, y_, 2000)):6.1f}')
    print('  Frederikse (excl. RGI 5 and 19; model reconstruction before 1961): 1900 -72.7, 1920 -55.2, 1950 -26.3')
    print('  (the data cannot tell the two 1850 assumptions apart; kappa only matters for short tau before 1962)')

    # ── Projections ──
    print('\n=== Glacier sea-level rise 2000-2100 on AR6 median GMST paths (mm; data a) ===')
    print('  model                          ' + '  '.join(f'{s:>18s}' for s in SSPS))
    rows = [('linear (as production)', lin_tau, lin_bk)]
    tq = np.percentile(np.where(np.isinf(tau_a), 1e5, tau_a), [5, 50, 95])
    for lab, tf in ((f'delayed, tau {tq[0]:.0f} (5th pct)', tq[0]), (f'delayed, tau {tq[1]:.0f} (median)', tq[1]),
                    (f'delayed, tau {tq[2]:.0f} (95th pct)', tq[2])):
        rows.append((lab,) + draws('a', None, tau_fixed=tf))
    rows.append(('delayed, tau marginalised', tau_a, bk_a))
    rows.append(('delayed, log-uniform tau prior',) + draws('a', 'log-uniform'))
    proj = {}
    for lab, tt, bb in rows:
        line = f'  {lab:31s}'
        for ssp, Ts in Tssp.items():
            Rs = trajectories(Ts, tt, bb)
            c = cum(Rs, 2000, 2100)
            proj[(lab, ssp)] = Rs
            line += f'  {np.median(c):5.0f} [{np.percentile(c, 5):4.0f}, {np.percentile(c, 95):4.0f}]'
        print(line)
    print('  production (level-space fit, volume cap): ' + ', '.join(f'{s} {v}' for s, v in PROD_2100.items()))
    print('  Rounce et al. (2023), 2015-2100, ±95%: SSP1-2.6 98±38, SSP5-8.5 166±83 mm; '
          'end-of-century rate 0.70±0.45 (+1.5 °C) to 2.23±1.08 mm/yr (+4 °C)')
    print('\n  Rate in 2100 (mm/yr), median:')
    for lab, *_ in rows:
        print(f'  {lab:31s}' + ''.join(f'  {s} {np.median(proj[(lab, s)][2100]):.2f}' for s in SSPS))

    # ── Figure ──
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    ax = axes[0]
    for variant, col in (('a', 'k'), ('b', '#0072B2')):
        fits, logml, *_ = res[variant]
        ax.plot(np.where(np.isinf(TAUS), 6000, TAUS), logml - logml[-1], 'o-', ms=3, color=col,
                label=f'data ({variant})')
    ax.set_xscale('log'); ax.axhline(0, color='0.5', lw=0.5)
    ax.set_xlabel('tau (yr); rightmost point = linear model'); ax.set_ylabel('log marginal likelihood vs linear')
    ax.set_title('(a) Evidence for a finite response time', fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.2)

    ax = axes[1]
    y, C, zy, gy, nz = res['a'][2:]
    zz = read_zemp2019_global(str(ZEMP)).loc[1962:1999] * 1000
    ax.errorbar(zy, y[:nz], yerr=np.hypot(zz['sigma_ind'], zz['sigma_cor']), fmt='o', ms=3, lw=0.5,
                color='#0072B2', alpha=0.6, label='Zemp et al. (2019)')
    ax.plot(gy + 0.5, y[nz:], 's', ms=3, color='#E69F00', label='GlaMBIE')
    yrs = np.arange(1900, 2024)
    for Rm, col, lab in ((Rl, 'k', 'linear'), (R, '#CC79A7', 'delayed (kappa free)'),
                         (Rb, '#009E73', 'delayed (balance 1850-1879)')):
        ax.plot(yrs, Rm[yrs].median(axis=0), color=col, lw=1.6, label=lab)
    ax.set_xlim(1900, 2024); ax.set_ylabel('mm/yr SLE')
    ax.set_title('(b) Rates, data (a), lognormal tau prior', fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.2)

    ax = axes[2]
    yrs = np.arange(2000, 2101)
    for ssp, col in (('SSP1-2.6', '#1b9e77'), ('SSP5-8.5', '#d95f02')):
        for lab, ls in (('linear (as production)', '-'), ('delayed, tau marginalised', '--')):
            Rs = proj[(lab, ssp)]
            ax.plot(yrs, Rs[yrs].median(axis=0), color=col, ls=ls, lw=1.6, label=f'{ssp}, {lab}')
    ax.set_ylabel('mm/yr SLE'); ax.set_title('(c) Projected rates (median)', fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.2)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
    print(f'\nSaved {OUT_FIG.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
