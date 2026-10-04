"""Joint fit of MAR SMB across all scenarios of one GCM, comparing lag models.

Driver x: the GCM's GMST anomaly (rel. 1995-2005, 11-yr centred mean).
Lagged driver x_tau: first-order relaxation dx_tau/dt = (x - x_tau)/tau,
integrated from 1950.  SMB anomalies (rel. 1995-2005) from
extract_mar_protect_smb.py.

Models
  (a)  shared tau:        SMB = M + b1 x_tau + b2 x_tau^2
  (b1) fast + slow:       SMB = M + a1 x + b1 x_tau + b2 x_tau^2
  (b2) fast + slow:       SMB = M + a1 x + a2 x^2 + c1 x_tau
  (c)  tau per scenario:  as (a), with a separate tau for the history and for
                          each scenario (rate-dependent relaxation)
  (0)  no lag:            SMB = M + b1 x + b2 x^2

Likelihood: Gaussian AR(1) residuals, with the history and each scenario as
separate segments (Prais-Winsten per segment), normal-inverse-gamma prior on
the coefficients and noise, analytic marginal likelihood for each (rho, tau)
combination.  tau values are integrated over a uniform grid prior, so models
with more tau parameters pay for them in the evidence.

Usage:  python scripts/joint_lag_fit_mar.py [--gcm CESM2]
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import test_scenario_transfer_mar as T

TAU_GRID = [1, 2, 5, 7, 10, 15, 20, 30, 50]          # years (1 = essentially no lag)
RHO_GRID = np.linspace(-0.6, 0.8, 29)
SIGMA0 = 100.0
A0, B0 = 2.0, SIGMA0**2
PRIOR_SD = 1000.0                                      # Gt/yr per K (or per K^2), weak
GT_PER_MM = 362.5


def pw(Z, rho, starts):
    """Prais-Winsten transform applied separately to each segment."""
    Zs = np.empty_like(Z, dtype=float)
    bounds = list(starts) + [len(Z)]
    for s0, s1 in zip(bounds[:-1], bounds[1:]):
        Zs[s0] = np.sqrt(1 - rho**2) * Z[s0]
        Zs[s0 + 1:s1] = Z[s0 + 1:s1] - rho * Z[s0:s1 - 1]
    return Zs


def log_evidence(X, y, starts):
    """log p(y | X) integrated over sigma (NIG) and rho (uniform grid)."""
    p = X.shape[1]
    V0 = np.diag([2000.0**2] + [PRIOR_SD**2] * (p - 1)) / SIGMA0**2
    V0i = np.linalg.inv(V0); ld0 = np.linalg.slogdet(V0)[1]
    lz = []
    for rho in RHO_GRID:
        Xs, ys = pw(X, rho, starts), pw(y, rho, starts)
        Vn = np.linalg.inv(V0i + Xs.T @ Xs)
        mn = Vn @ (Xs.T @ ys)
        an = A0 + 0.5 * len(y)
        bn = B0 + 0.5 * (ys @ ys - mn @ np.linalg.solve(Vn, mn))
        lz.append(-0.5 * len(y) * np.log(2 * np.pi) + 0.5 * (np.linalg.slogdet(Vn)[1] - ld0)
                  + A0 * np.log(B0) - an * np.log(bn) + gammaln(an) - gammaln(A0)
                  + 0.5 * len(starts) * np.log(1 - rho**2))
    lz = np.array(lz)
    return logsumexp(lz) - np.log(len(RHO_GRID)), RHO_GRID[np.argmax(lz)]


def point_fit(X, y, starts, rho):
    Xs, ys = pw(X, rho, starts), pw(y, rho, starts)
    return np.linalg.lstsq(Xs, ys, rcond=None)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gcm', default='CESM2')
    args = ap.parse_args()
    gdir = {'CESM2': 'CESM2-CMIP6', 'MPI-ESM1-2-HR': 'MPI-ESM1-2-HR',
            'NorESM2-MM': 'NorESM2', 'UKESM1-0-LL': 'UKESM1-0-LL-CMIP6'}[args.gcm]
    mar = pd.read_csv(T.MAR).drop_duplicates(['run', 'year'])
    temp = pd.read_csv(T.TEMP)
    scens = [s for s in ['ssp126', 'ssp245', 'ssp585'] if (mar.run == f'{gdir}-{s}').sum() == 86]

    # Segments: history 1980-2014 once, then each scenario 2015-2100.
    segs = []
    x_full = {s: T.drivers(temp, args.gcm, s)['GMST'] for s in scens}
    y_hist = T.smb_series(mar, gdir, scens[0]).loc[1980:2014]
    segs.append(('history', y_hist, x_full[scens[0]]))
    for s in scens:
        segs.append((s, T.smb_series(mar, gdir, s).loc[2015:2100], x_full[s]))
    starts = np.cumsum([0] + [len(sg[1]) for sg in segs[:-1]])
    y = np.concatenate([sg[1].values for sg in segs])
    print(f'{args.gcm}: joint fit over history 1980-2014 + {scens} (n = {len(y)})\n')

    def design(form, taus):
        rows = []
        for (name, ys, xs), tau in zip(segs, taus):
            x = xs.loc[ys.index].values
            xl = T.lag(xs, tau).loc[ys.index].values
            one = np.ones_like(x)
            cols = {'0': [one, x, x**2], 'a': [one, xl, xl**2], 'c': [one, xl, xl**2],
                    'b1': [one, x, xl, xl**2], 'b2': [one, x, x**2, xl]}[form]
            rows.append(np.column_stack(cols))
        return np.vstack(rows)

    nseg = len(segs)
    models = {}
    lz0, _ = log_evidence(design('0', [1] * nseg), y, starts)
    models['(0) no lag'] = {'logZ': lz0, 'best': None}
    for form, label in [('a', '(a) shared tau'), ('b1', '(b1) fast linear + slow quadratic'),
                        ('b2', '(b2) fast quadratic + slow linear')]:
        lz = {t: log_evidence(design(form, [t] * nseg), y, starts)[0] for t in TAU_GRID}
        arr = np.array(list(lz.values()))
        post = np.exp(arr - logsumexp(arr))
        models[label] = {'logZ': logsumexp(arr) - np.log(len(TAU_GRID)),
                         'best': max(lz, key=lz.get),
                         'post': dict(zip(TAU_GRID, post))}
    # (c): separate tau for history and each scenario (grid product)
    combos, lzs = [], []
    for taus in itertools.product(TAU_GRID, repeat=nseg):
        combos.append(taus); lzs.append(log_evidence(design('c', list(taus)), y, starts)[0])
    lzs = np.array(lzs)
    pc = np.exp(lzs - logsumexp(lzs))
    marg = {name: {t: pc[[c[k] == t for c in combos]].sum() for t in TAU_GRID}
            for k, (name, _, _) in enumerate(segs)}
    models['(c) tau per segment'] = {'logZ': logsumexp(lzs) - np.log(len(combos)),
                                     'best': combos[int(np.argmax(lzs))], 'marg': marg}

    ref = models['(a) shared tau']['logZ']
    print('Model                                log evidence   vs (a)   best tau')
    for k, v in models.items():
        print(f'  {k:36s} {v["logZ"]:10.1f}   {v["logZ"] - ref:+6.1f}   {v["best"]}')
    print()
    for k in ['(a) shared tau', '(b1) fast linear + slow quadratic', '(b2) fast quadratic + slow linear']:
        p = models[k]['post']
        lo_hi = [t for t in TAU_GRID if p[t] > 0.05]
        print(f'  {k}: posterior over tau ' + ', '.join(f'{t}:{p[t]:.2f}' for t in TAU_GRID)
              + f'   (tau with >5%: {lo_hi})')
    print('  (c) marginal posterior of tau by segment:')
    for name, m in models['(c) tau per segment']['marg'].items():
        mean = sum(t * w for t, w in m.items())
        print(f'     {name:8s} mean {mean:5.1f} yr; ' + ', '.join(f'{t}:{w:.2f}' for t, w in m.items() if w > 0.02))
    print()

    # Fit quality per segment for each model at its best tau (decadal-mean residual RMS)
    print('Per-segment residuals at the best tau (RMS of 11-yr mean residual, Gt/yr; '
          'integrated 2015-2100 residual in mm SLE, + = model underpredicts SLR):')
    for k, form in [('(0) no lag', '0'), ('(a) shared tau', 'a'),
                    ('(b1) fast linear + slow quadratic', 'b1'),
                    ('(b2) fast quadratic + slow linear', 'b2'), ('(c) tau per segment', 'c')]:
        best = models[k]['best']
        taus = [1] * nseg if best is None else (list(best) if isinstance(best, tuple) else [best] * nseg)
        X = design(form, taus)
        _, rho = log_evidence(X, y, starts)
        beta = point_fit(X, y, starts, rho)
        res = y - X @ beta
        parts = []
        for (name, ys, _), s0, n in zip(segs, starts, [len(sg[1]) for sg in segs]):
            r = pd.Series(res[s0:s0 + n], index=ys.index)
            rms = np.sqrt(np.nanmean(r.rolling(11, center=True, min_periods=6).mean() ** 2))
            cum = '' if name == 'history' else f', {-r.sum() / GT_PER_MM:+.1f} mm'
            parts.append(f'{name} {rms:.0f}{cum}')
        print(f'  {k:36s} ' + ';  '.join(parts))


if __name__ == '__main__':
    main()
