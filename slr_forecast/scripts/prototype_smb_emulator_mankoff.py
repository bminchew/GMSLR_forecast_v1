"""Prototype of the Greenland SMB emulator fit, on per-RCM Mankoff SMB.

Fits, separately for each RCM (HIRHAM/HARMONIE, MAR, RACMO) and for their
mean,

    SMB(t) = M + C_T * dT(t) [+ C_T2 * dT(t)^2 | + C_cum * cumT(t)] + eps(t),
    eps ~ AR(1) with coefficient rho and innovation sd sigma,

where dT is Berkeley Earth annual GMST relative to 1995-2005 and SMB is in
Gt/yr (mass-gain convention).  Window: 1986-2024, the years in which the
Mankoff per-RCM series are RCM output (before 1986 all three are the same
adjusted Kjeldsen et al. 2015 reconstruction) and Berkeley Earth is complete.

The posterior is analytic.  For each rho on a grid, the Prais-Winsten
transform (first observation kept) whitens the AR(1) errors; a
normal-inverse-gamma prior then gives a closed-form posterior for
(beta, sigma^2) and a closed-form marginal likelihood p(y | rho).  The rho
grid is mixed by that marginal likelihood (uniform prior on rho), the same
construction as the BIC mixture over the discharge delay.

The RCMs share reanalysis forcing, so their residuals are correlated; each is
fit on its own and the three posteriors are mixed with equal weight.  The
historical spread between RCMs is not the structural spread under strong
warming (Glaude et al. 2024), and the 1986-2024 temperature range cannot
identify curvature.  This is a prototype of the machinery, not the emulator.

Usage:  python scripts/prototype_smb_emulator_mankoff.py
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[1]
NC_PATH = ROOT / 'data/raw/ice_sheets/greenland/mankoff/MB_region.nc'
H5_PATH = ROOT / 'data/processed/slr_processed_data.h5'

YEAR0, YEAR1 = 1986, 2024
BASE = (1995, 2005)
RCMS = ['SMB_HIRHAM', 'SMB_MAR', 'SMB_RACMO', 'SMB']
LABEL = {'SMB_HIRHAM': 'HIRHAM', 'SMB_MAR': 'MAR', 'SMB_RACMO': 'RACMO',
         'SMB': '3-RCM mean'}
RHO_GRID = np.linspace(-0.9, 0.9, 181)
N_DRAW = 200_000
SEED = 20261002

# Normal-inverse-gamma prior: beta | sigma^2 ~ N(0, sigma^2 V0),
# sigma^2 ~ IG(A0, B0).  V0 is set so the prior sd of each coefficient is
# PRIOR_SD at the prior-typical sigma SIGMA0 -- weak relative to the data.
SIGMA0 = 100.0                      # Gt/yr
A0, B0 = 2.0, SIGMA0**2             # prior mean of sigma^2 = B0/(A0-1)
PRIOR_SD = {'const': 2000.0, 'T': 1000.0, 'T2': 1000.0, 'cum': 100.0}

C_T_ADOPTED = -300.0                # current smb_projections.py value


# ── Data ──

def load_smb():
    ds = xr.open_dataset(NC_PATH)
    df = ds[RCMS].to_dataframe().loc[f'{YEAR0}':f'{YEAR1}-12-31']
    counts = df.groupby(df.index.year).count()
    ndays = pd.Series(df.index.year).value_counts().sort_index()
    if not (counts.values == ndays.values[:, None]).all():
        raise ValueError('incomplete daily coverage in the fit window')
    return df.groupby(df.index.year).sum()          # Gt/d summed -> Gt/yr


def load_gmst():
    be = pd.read_hdf(H5_PATH, '/raw/df_berkeley')
    T = be['temperature'].groupby(be.index.year).mean()
    T = T - T.loc[BASE[0]:BASE[1]].mean()
    if T.index.max() < YEAR1:
        raise ValueError('Berkeley Earth ends before the fit window')
    return T


# ── Analytic AR(1) Bayesian regression ──

def prais_winsten(Z, rho):
    Zs = np.empty_like(Z, dtype=float)
    Zs[0] = np.sqrt(1.0 - rho**2) * Z[0]
    Zs[1:] = Z[1:] - rho * Z[:-1]
    return Zs


def nig_posterior(X, y, V0):
    V0i = np.linalg.inv(V0)
    Vn = np.linalg.inv(V0i + X.T @ X)
    mn = Vn @ (X.T @ y)
    an = A0 + 0.5 * len(y)
    bn = B0 + 0.5 * (y @ y - mn @ np.linalg.solve(Vn, mn))
    logml = (-0.5 * len(y) * np.log(2 * np.pi)
             + 0.5 * (np.linalg.slogdet(Vn)[1] - np.linalg.slogdet(V0)[1])
             + A0 * np.log(B0) - an * np.log(bn)
             + gammaln(an) - gammaln(A0))
    return mn, Vn, an, bn, logml


def max_loglik(X, y):
    """Profile max log-likelihood over (beta, sigma, rho) for BIC."""
    best = -np.inf
    for rho in RHO_GRID:
        Xs, ys = prais_winsten(X, rho), prais_winsten(y, rho)
        beta = np.linalg.lstsq(Xs, ys, rcond=None)[0]
        s2 = np.mean((ys - Xs @ beta) ** 2)
        ll = (-0.5 * len(y) * (np.log(2 * np.pi * s2) + 1)
              + 0.5 * np.log(1 - rho**2))
        best = max(best, ll)
    return best


def fit(X, y, names, rng):
    V0 = np.diag([PRIOR_SD[n] ** 2 / SIGMA0**2 for n in names])
    post = []
    for rho in RHO_GRID:
        mn, Vn, an, bn, logml = nig_posterior(
            prais_winsten(X, rho), prais_winsten(y, rho), V0)
        post.append((mn, Vn, an, bn, logml + 0.5 * np.log(1 - rho**2)))
    logml = np.array([p[4] for p in post])
    w = np.exp(logml - logml.max())
    w /= w.sum()

    # Draw from the mixture over rho: sigma^2 ~ IG(an, bn), beta ~ N(mn, s2 Vn)
    k = rng.choice(len(RHO_GRID), size=N_DRAW, p=w)
    beta = np.empty((N_DRAW, X.shape[1]))
    sig = np.empty(N_DRAW)
    for j in np.unique(k):
        idx = np.where(k == j)[0]
        mn, Vn, an, bn, _ = post[j]
        s2 = bn / rng.gamma(an, 1.0, size=len(idx))
        L = np.linalg.cholesky(Vn)
        beta[idx] = mn + np.sqrt(s2)[:, None] * (rng.standard_normal((len(idx), len(mn))) @ L.T)
        sig[idx] = np.sqrt(s2)
    lmx = logml.max()
    log_evidence = lmx + np.log(np.sum(np.exp(logml - lmx)) / len(RHO_GRID))
    bic = -2 * max_loglik(X, y) + (X.shape[1] + 2) * np.log(len(y))
    return dict(beta=beta, sigma=sig, rho=RHO_GRID[k], w=w,
                log_evidence=log_evidence, bic=bic, names=names)


def q(a):
    return np.median(a), *np.percentile(a, [5, 95])


def fmt(a, nd=0):
    m, lo, hi = q(a)
    return f'{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]'


# ── Main ──

def design(T, years, form):
    dT = T.loc[years].values
    cols, names = [np.ones_like(dT), dT], ['const', 'T']
    if form == 'quadratic':
        cols.append(dT**2); names.append('T2')
    if form == 'cumulative':
        # cumulative warming since 1950, relative to its 1995-2005 mean
        cum = T.loc[1950:].cumsum()
        cum = cum - cum.loc[BASE[0]:BASE[1]].mean()
        cols.append(cum.loc[years].values); names.append('cum')
    return np.column_stack(cols), names


def main():
    rng = np.random.default_rng(SEED)
    smb, T = load_smb(), load_gmst()
    years = np.arange(YEAR0, YEAR1 + 1)
    smb = smb.loc[years]
    print(f'Window {YEAR0}-{YEAR1} (n = {len(years)}); dT range '
          f'{T.loc[years].min():.2f} to {T.loc[years].max():.2f} degC '
          f'(Berkeley Earth GMST rel. {BASE[0]}-{BASE[1]})')
    print(f'Units: SMB Gt/yr, C_T Gt/yr/degC GMST, C_T2 Gt/yr/degC^2. '
          f'Medians with 90% intervals [5th, 95th].\n')

    results = {}
    for form in ['linear', 'quadratic', 'cumulative']:
        X, names = design(T, years, form)
        print(f'── {form} model ' + '─' * 50)
        if form != 'linear':
            print(f'   design condition number (standardized): '
                  f'{np.linalg.cond((X[:, 1:] - X[:, 1:].mean(0)) / X[:, 1:].std(0)):.1f}')
        for v in RCMS:
            r = fit(X, smb[v].values, names, rng)
            results[(form, v)] = r
            line = (f'   {LABEL[v]:11s} C_T = {fmt(r["beta"][:, 1])}'
                    f'  M = {fmt(r["beta"][:, 0])}'
                    f'  rho = {fmt(r["rho"], 2)}  sigma = {fmt(r["sigma"])}'
                    f'  logZ = {r["log_evidence"]:.1f}  BIC = {r["bic"]:.1f}')
            if form == 'quadratic':
                c2 = r['beta'][:, 2]
                line += (f'\n               C_T2 = {fmt(c2)}  P(C_T2 > 0) = {np.mean(c2 > 0):.2f}'
                         f'  corr(C_T, C_T2) = {np.corrcoef(r["beta"][:, 1], c2)[0, 1]:.2f}')
            if form == 'cumulative':
                cc = r['beta'][:, 2]
                line += (f'\n               C_cum = {fmt(cc, 1)} Gt/yr per degC-yr'
                         f'  corr(C_T, C_cum) = {np.corrcoef(r["beta"][:, 1], cc)[0, 1]:.2f}')
            print(line)
        print()

    # Model comparison: log Bayes factors relative to linear
    print('── Model comparison (relative to linear; positive favors the alternative) ──')
    for v in RCMS:
        lin = results[('linear', v)]
        s = [f'{f}: dlogZ = {results[(f, v)]["log_evidence"] - lin["log_evidence"]:+.1f}, '
             f'dBIC = {results[(f, v)]["bic"] - lin["bic"]:+.1f}' for f in ['quadratic', 'cumulative']]
        print(f'   {LABEL[v]:11s} ' + ';  '.join(s))
    print()

    # Equal-weight mixture across the three RCMs (linear model)
    mix = np.concatenate([results[('linear', v)]['beta'][::3, 1] for v in RCMS[:3]])
    print('── Equal-weight 3-RCM mixture, linear model ──')
    print(f'   C_T = {fmt(mix)}   P(C_T < {C_T_ADOPTED:.0f}) = {np.mean(mix < C_T_ADOPTED):.3f}'
          f'   (adopted: {C_T_ADOPTED:.0f} +/- 80)\n')

    # Residual correlation between RCMs (linear model, posterior-median beta)
    X, _ = design(T, years, 'linear')
    res = pd.DataFrame({LABEL[v]: smb[v].values - X @ np.median(results[('linear', v)]['beta'], 0)
                        for v in RCMS[:3]}, index=years)
    print('── Residual correlation between RCMs (linear model) ──')
    print(res.corr().round(2).to_string(), '\n')

    # Robustness 1: drop 2012
    print('── Robustness: drop 2012 (linear model) ──')
    keep = years != 2012
    for v in RCMS:
        r = fit(X[keep], smb[v].values[keep], ['const', 'T'], rng)
        print(f'   {LABEL[v]:11s} C_T = {fmt(r["beta"][:, 1])}')
    print()

    # Robustness 2: 11-yr centered smoothed GMST as regressor (SMB unsmoothed)
    Ts = T.rolling(11, center=True).mean().dropna()
    yrs_s = years[np.isin(years, Ts.index)]
    Xs = np.column_stack([np.ones(len(yrs_s)), Ts.loc[yrs_s].values])
    print(f'── Robustness: 11-yr centered GMST regressor, {yrs_s[0]}-{yrs_s[-1]} (linear model) ──')
    for v in RCMS:
        r = fit(Xs, smb.loc[yrs_s, v].values, ['const', 'T'], rng)
        print(f'   {LABEL[v]:11s} C_T = {fmt(r["beta"][:, 1])}')
    # same shortened window with annual GMST, to separate window from smoothing
    Xa = np.column_stack([np.ones(len(yrs_s)), T.loc[yrs_s].values])
    print(f'   (annual GMST over the same {yrs_s[0]}-{yrs_s[-1]} window:)')
    for v in RCMS:
        r = fit(Xa, smb.loc[yrs_s, v].values, ['const', 'T'], rng)
        print(f'   {LABEL[v]:11s} C_T = {fmt(r["beta"][:, 1])}')
    print()

    # Robustness 3: HIRHAM -> HARMONIE switch on 2017-08-31
    d = (smb['SMB_HIRHAM'] - smb['SMB_MAR'])
    pre, post = d.loc[:2016], d.loc[2018:]
    se = np.sqrt(pre.var() / len(pre) + post.var() / len(post))
    print('── HIRHAM -> HARMONIE switch (2017-08-31): HIRHAM minus MAR ──')
    print(f'   1986-2016 mean {pre.mean():.0f}, 2018-{YEAR1} mean {post.mean():.0f} Gt/yr; '
          f'step {post.mean() - pre.mean():+.0f} +/- {se:.0f} (1 s.e., white-noise)')
    d2 = (smb['SMB_HIRHAM'] - smb['SMB_RACMO'])
    se2 = np.sqrt(d2.loc[:2016].var() / len(pre) + d2.loc[2018:].var() / len(post))
    print(f'   HIRHAM minus RACMO step {d2.loc[2018:].mean() - d2.loc[:2016].mean():+.0f} +/- {se2:.0f}')
    print()

    # HIRHAM refit with a step at the switch: 0 before 2017, 4/12 in 2017
    # (HARMONIE from September), 1 from 2018
    step = np.where(years >= 2018, 1.0, 0.0)
    step[years == 2017] = 4.0 / 12.0
    PRIOR_SD['step'] = 500.0
    print('── HIRHAM with a HARMONIE step term ──')
    hir_step = {}
    for form in ['linear', 'quadratic', 'cumulative']:
        X, names = design(T, years, form)
        r = fit(np.column_stack([X, step]), smb['SMB_HIRHAM'].values, names + ['step'], rng)
        hir_step[form] = r
        line = (f'   {form:10s} C_T = {fmt(r["beta"][:, 1])}  step = {fmt(r["beta"][:, -1])}'
                f'  rho = {fmt(r["rho"], 2)}  logZ = {r["log_evidence"]:.1f}  BIC = {r["bic"]:.1f}')
        if form == 'quadratic':
            line += f'  C_T2 = {fmt(r["beta"][:, 2])}'
        if form == 'cumulative':
            line += f'  C_cum = {fmt(r["beta"][:, 2], 1)}'
        print(line)
    lin = hir_step['linear']['log_evidence']
    print(f'   vs linear with step: quadratic dlogZ = {hir_step["quadratic"]["log_evidence"] - lin:+.1f}, '
          f'cumulative dlogZ = {hir_step["cumulative"]["log_evidence"] - lin:+.1f}')
    print(f'   linear with vs without step: dlogZ = '
          f'{lin - results[("linear", "SMB_HIRHAM")]["log_evidence"]:+.1f}')
    mix_s = np.concatenate([hir_step['linear']['beta'][::3, 1]]
                           + [results[('linear', v)]['beta'][::3, 1] for v in RCMS[1:3]])
    print(f'   Equal-weight mixture with HIRHAM step: C_T = {fmt(mix_s)}'
          f'   P(C_T < {C_T_ADOPTED:.0f}) = {np.mean(mix_s < C_T_ADOPTED):.3f}')
    print('   (within one series the step is confounded with the 2018-2024 SMB recovery)')
    print()

    # The step identified from differences between RCMs, where the shared
    # reanalysis weather cancels; HIRHAM is then corrected and refit.
    print('── HIRHAM step identified from differences between RCMs ──')
    steps = []
    for v in ['SMB_MAR', 'SMB_RACMO']:
        Xd = np.column_stack([np.ones(len(years)), step, T.loc[years].values])
        r = fit(Xd, (smb['SMB_HIRHAM'] - smb[v]).values, ['const', 'step', 'T'], rng)
        steps.append(np.median(r['beta'][:, 1]))
        print(f'   HIRHAM minus {LABEL[v]:5s} step = {fmt(r["beta"][:, 1])}, '
              f'difference in C_T = {fmt(r["beta"][:, 2])}')
    s = np.mean(steps)
    X, names = design(T, years, 'linear')
    r = fit(X, smb['SMB_HIRHAM'].values - s * step, names, rng)
    print(f'   HIRHAM with the mean step ({s:.0f} Gt/yr) removed: C_T = {fmt(r["beta"][:, 1])}')
    mix_d = np.concatenate([r['beta'][::3, 1]]
                           + [results[('linear', v)]['beta'][::3, 1] for v in RCMS[1:3]])
    print(f'   Equal-weight mixture: C_T = {fmt(mix_d)}'
          f'   P(C_T < {C_T_ADOPTED:.0f}) = {np.mean(mix_d < C_T_ADOPTED):.3f}')


if __name__ == '__main__':
    main()
