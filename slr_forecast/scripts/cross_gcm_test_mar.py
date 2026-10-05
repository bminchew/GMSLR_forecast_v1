"""Cross-GCM test of the GMST-driven SMB emulator.

Fit the emulator jointly to all scenarios of one GCM's MAR runs (history
1980-2014 plus each SSP, separate AR(1) segments), then predict another GCM's
MAR runs from that GCM's own GMST.  The test GCM is never seen in the fit.

Models: (0) no lag, SMB = M + b1 x + b2 x^2; (a) shared lag, x replaced by a
first-order relaxation of x with time scale tau, tau integrated over its
posterior on the training GCM.  x is the GCM's GMST anomaly rel. 1995-2005
(11-yr centred mean); SMB anomalies are rel. each run's own 1995-2005.

Usage:  python scripts/cross_gcm_test_mar.py --train CESM2 --test MPI-ESM1-2-HR
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gammaln, logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import test_scenario_transfer_mar as T
import joint_lag_fit_mar as J

GDIR = {'CESM2': 'CESM2-CMIP6', 'MPI-ESM1-2-HR': 'MPI-ESM1-2-HR',
        'NorESM2-MM': 'NorESM2', 'UKESM1-0-LL': 'UKESM1-0-LL-CMIP6',
        'CNRM-CM6-1': 'CNRM-CM6', 'CNRM-ESM2-1': 'CNRM-ESM2', 'IPSL-CM6A-LR': 'IPSL-CM6A-LR'}
N_DRAW = 4000


def segments(mar, temp, gcm):
    gdir = GDIR[gcm]
    scens = [s for s in ['ssp126', 'ssp245', 'ssp585'] if (mar.run == f'{gdir}-{s}').sum() == 86]
    x_full = {s: T.drivers(temp, gcm, s)['GMST'] for s in scens}
    segs = [('history', T.smb_series(mar, gdir, scens[0]).loc[1980:2014], x_full[scens[0]])]
    segs += [(s, T.smb_series(mar, gdir, s).loc[2015:2100], x_full[s]) for s in scens]
    return segs


def design(segs, tau):
    X = []
    for _, ys, xs in segs:
        x = T.lag(xs, tau).loc[ys.index].values if tau > 1 else xs.loc[ys.index].values
        X.append(np.column_stack([np.ones_like(x), x, x**2]))
    return np.vstack(X)


def posterior_draws(X, y, starts, n, rng):
    """NIG posterior draws of (M, b1, b2), sigma, rho, plus log evidence."""
    p = X.shape[1]
    V0 = np.diag([2000.0**2] + [J.PRIOR_SD**2] * (p - 1)) / J.SIGMA0**2
    V0i = np.linalg.inv(V0); ld0 = np.linalg.slogdet(V0)[1]
    post, lz = [], []
    for rho in J.RHO_GRID:
        Xs, ys = J.pw(X, rho, starts), J.pw(y, rho, starts)
        Vn = np.linalg.inv(V0i + Xs.T @ Xs); mn = Vn @ (Xs.T @ ys)
        an = J.A0 + 0.5 * len(y); bn = J.B0 + 0.5 * (ys @ ys - mn @ np.linalg.solve(Vn, mn))
        post.append((mn, Vn, an, bn))
        lz.append(-0.5 * len(y) * np.log(2 * np.pi) + 0.5 * (np.linalg.slogdet(Vn)[1] - ld0)
                  + J.A0 * np.log(J.B0) - an * np.log(bn) + gammaln(an) - gammaln(J.A0)
                  + 0.5 * len(starts) * np.log(1 - rho**2))
    lz = np.array(lz); w = np.exp(lz - lz.max()); w /= w.sum()
    k = rng.choice(len(J.RHO_GRID), n, p=w)
    beta = np.empty((n, p)); sig = np.empty(n)
    for j in np.unique(k):
        idx = np.where(k == j)[0]; mn, Vn, an, bn = post[j]
        s2 = bn / rng.gamma(an, 1.0, len(idx))
        beta[idx] = mn + np.sqrt(s2)[:, None] * (rng.standard_normal((len(idx), p)) @ np.linalg.cholesky(Vn).T)
        sig[idx] = np.sqrt(s2)
    return beta, sig, J.RHO_GRID[k], logsumexp(lz) - np.log(len(J.RHO_GRID))


def fit(segs, rng):
    starts = np.cumsum([0] + [len(s[1]) for s in segs[:-1]])
    y = np.concatenate([s[1].values for s in segs])
    out = {'no lag': posterior_draws(design(segs, 1), y, starts, N_DRAW, rng)[:3] + (None,)}
    # shared tau: mixture over the tau grid weighted by evidence
    per = {t: posterior_draws(design(segs, t), y, starts, N_DRAW, rng) for t in J.TAU_GRID}
    lz = np.array([per[t][3] for t in J.TAU_GRID]); w = np.exp(lz - logsumexp(lz))
    pick = rng.choice(len(J.TAU_GRID), N_DRAW, p=w)
    taus = np.array(J.TAU_GRID)[pick]
    out['shared lag'] = (np.array([per[t][0][i] for i, t in enumerate(taus)]),
                         np.array([per[t][1][i] for i, t in enumerate(taus)]),
                         np.array([per[t][2][i] for i, t in enumerate(taus)]), taus)
    return out


def predict(model, xs, years):
    beta, sig, rho, taus = model
    if taus is None:
        x = np.tile(xs.loc[years].values, (len(beta), 1))
    else:
        cache = {t: (T.lag(xs, t) if t > 1 else xs).loc[years].values for t in np.unique(taus)}
        x = np.array([cache[t] for t in taus])
    return beta[:, [0]] + beta[:, [1]] * x + beta[:, [2]] * x**2


def run(train, test, mar, temp, rng, ax_row=None):
    tr, te = segments(mar, temp, train), segments(mar, temp, test)
    models = fit(tr, rng)
    tau_w = pd.Series(models['shared lag'][3]).value_counts(normalize=True).sort_index()
    print(f'\n=== train {train} (all scenarios) -> predict {test} ===')
    print('  shared-lag posterior on training GCM: ' + ', '.join(f'{t}:{p:.2f}' for t, p in tau_w.items()))
    print('  misses = actual minus predicted, Gt/yr (negative = more loss than predicted); '
          'integrated 2015-2100 in mm SLE (+ = emulator underpredicts SLR);\n'
          '  [5th, 95th] of the predicted 20-yr mean includes coefficient and AR(1) noise uncertainty')
    for name, ys, xs in te[1:]:
        for lab, m in models.items():
            pr = predict(m, xs, ys.index)
            med = pd.Series(np.median(pr, 0), index=ys.index)
            miss = ys - med
            parts = []
            for a, b in [(2041, 2060), (2081, 2100)]:
                sel = (ys.index >= a) & (ys.index <= b)
                n = int(sel.sum()); lags = np.arange(1, n)
                # variance of an n-yr mean of AR(1) noise with marginal variance sigma^2/(1-rho^2)
                ssum = n + 2 * ((n - lags)[None, :] * m[2][:, None] ** lags[None, :]).sum(1)
                var = m[1]**2 / (1 - m[2]**2) * ssum / n**2
                draw = pr[:, sel].mean(1) + np.sqrt(var) * rng.standard_normal(len(var))
                lo, hi = np.percentile(draw, [5, 95]); obs = ys[sel].mean()
                flag = '' if lo <= obs <= hi else ' *outside*'
                parts.append(f'{a}-{b} {obs - np.median(draw):+5.0f} [{lo - np.median(draw):+.0f}, {hi - np.median(draw):+.0f}]{flag}')
            cum = -miss.loc[2015:2100].sum() / J.GT_PER_MM
            print(f'  {name} {lab:10s}: ' + ';  '.join(parts) + f';  integrated {cum:+5.1f} mm')
            if ax_row is not None:
                ax = ax_row[[s[0] for s in te[1:]].index(name)]
                if lab == 'no lag':
                    full = T.smb_series(mar, GDIR[test], name)
                    ax.plot(full.index, full.values, color='0.8', lw=0.8)
                    ax.plot(full.index, full.rolling(11, center=True, min_periods=6).mean(), 'k', lw=2,
                            label=f'MAR-{test}' if name == te[1][0] else None)
                c = '#2a78d6' if lab == 'no lag' else '#eb6834'
                ax.fill_between(ys.index, *np.percentile(pr, [5, 95], 0), color=c, alpha=0.15, lw=0)
                ax.plot(ys.index, med, color=c, lw=2, label=f'{lab} (fit on {train})' if name == te[1][0] else None)
                ax.set_title(f'{test} {name.upper()}'); ax.axhline(0, color='0.6', lw=0.6); ax.grid(True, alpha=0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='CESM2'); ap.add_argument('--test', default='MPI-ESM1-2-HR')
    args = ap.parse_args()
    mar = pd.read_csv(T.MAR).drop_duplicates(['run', 'year']); temp = pd.read_csv(T.TEMP)
    rng = np.random.default_rng(2)
    pairs = [(args.train, args.test), (args.test, args.train)]
    ncol = max(len(segments(mar, temp, g)) - 1 for g in (args.train, args.test))
    fig, axes = plt.subplots(2, ncol, figsize=(6 * ncol, 9), squeeze=False)
    for row, (tr, te) in enumerate(pairs):
        run(tr, te, mar, temp, rng, axes[row])
        axes[row][0].legend(fontsize=9, loc='lower left')
        axes[row][0].set_ylabel('SMB anomaly rel. 1995–2005 (Gt yr$^{-1}$)')
    fig.tight_layout()
    out = ROOT / f'figures/preview_mar_crossgcm_{args.train}_{args.test}.png'
    fig.savefig(out, dpi=140, bbox_inches='tight'); print(f'\nsaved {out}')


if __name__ == '__main__':
    main()
