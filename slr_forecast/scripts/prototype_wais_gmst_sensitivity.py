"""Prototype fit of a WAIS mass-loss sensitivity to GMST, on IMBIE-3.

Data: IMBIE-3 West Antarctica (Otosaka et al. 2026), annual rates of total
mass balance and of its dynamics part, in mm/yr SLE (SLR-positive, flipped
on read by read_imbie3), and Berkeley Earth annual GMST relative to
1995-2014.  The file holds annual rates repeated monthly, so each year is one
observation (annual mean rate, annual mean 1-sigma).

Rate models, for r(t) the annual mass-loss rate:

    const   r = a
    trend   r = a + d (t - 2000)
    lagT    r = a + b T(t - L),          L = 0..LAG_MAX yr
    intT    r = a + c I(t),  I(t) = sum_{1850}^{t} [T(t') - T_PI]   (degC yr)

T_PI is the Berkeley 1850-1900 mean, fixed.  In intT the baseline is a
physical parameter (shifting it adds c T_PI t, which a cannot absorb);
freeing it would make intT identical to intT + trend.  In lagT the baseline
is absorbed by a.

Errors: r = X beta + e,  Cov(e) = diag(sigma_IMBIE^2) + s^2 R(rho),
R_ij = rho^|i-j|, i.e. the reported IMBIE uncertainty plus an AR(1)
structural term.  Full GLS through a Cholesky factor on a (rho, s) grid.
Within a model, beta | rho, s is Gaussian (flat prior) and the grid is mixed
by its flat-prior marginal likelihood, so the posterior is an analytic
mixture of normals.  Across models (and across lags inside lagT) the weights
are BIC weights, k = p + 2, the same construction as the Greenland
discharge-delay mixture.  Each family (const, trend, intT, lagT) has prior
1/4 and each lag 1/16 within lagT, so the lagT family BIC is
-2 logsumexp(-BIC_L / 2) + 2 ln 16.  Reduced chi-square is computed at the
maximum-likelihood cell against the IMBIE sigmas alone.

Checks: dynamics refit without 2022-2023 (the IMBIE dynamics series is
total minus RCM SMB, so the 2022 SMB anomaly passes into it); projections
repeated with the SSP medians shifted so their 2015-2024 mean equals the
Berkeley 2015-2024 mean.

Two fit windows: 1979-2023 (pre-1992 is input-output only) and 1992-2023
(multi-technique).  Holdout: fit through 2009, predict 2010-2023 rates and
the 2010-2023 cumulative change.  Predictive AR(1) noise starts from its
stationary distribution (not conditioned on the last residual).

Projection: cumulative WAIS contribution from 2000.0 to 2100.0 = observed
IMBIE-3 sum of annual rates over 2000-2023, plus modelled rates for
2024-2099.  Projected GMST is observed Berkeley through 2024 and AR6 SSP
median paths (1850-1900 frame, shifted by the AR6 Historical 1995-2014 mean)
from 2025, with the pipeline's shared within-SSP warming percentile paths
(warming_paths.offsets_on_grid, anchored at WARMING_ANCHOR_YEAR).

The trend model (the rate-space form of the manuscript's S1 quadratic) is
projected the same way and is SSP-independent.  For reference, S1 itself is
evaluated from notebooks/s1_quadratic_fit.json: closed-form median and
parameter-only 90% interval, and the 90% interval including the ISMIP6
extrapolation-error term (component_projections._sample_s1_quadratic_mm,
own seed).

Sensitivities are reported in mm/yr/degC and in Gt/yr/degC using 360 Gt/mm,
the file's own conversion.

Writes figures/prototype_wais_gmst_sensitivity.png and prints tables.
Writes nothing to component_results.h5.

Usage:  python scripts/prototype_wais_gmst_sensitivity.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
sys.path.insert(0, str(ROOT / 'src'))

from slr_data_readers import read_imbie3          # noqa: E402
from warming_paths import warming_z, offsets_on_grid  # noqa: E402
from slr_forecast.config import WARMING_ANCHOR_YEAR   # noqa: E402
import component_projections as cp                    # noqa: E402

IMBIE_PATH = ROOT / 'data/raw/ice_sheets/imbie2026/imbie3_west_antarctica_mm_partitioned.csv'
H5_PATH = ROOT / 'data/processed/slr_processed_data.h5'
FIG_PATH = ROOT / 'figures/prototype_wais_gmst_sensitivity.png'

GT_PER_MM = 360.0                       # Otosaka et al. (2026) conversion
T_BASE = (1995, 2014)                   # GMST frame for fitting
PI = (1850, 1900)                       # T_PI for the integrated model
LAG_MAX = 15
WINDOWS = {'1979-2023': (1979, 2023), '1992-2023': (1992, 2023)}
DYN_CHECK = {'1979-2021': (1979, 2021), '1992-2021': (1992, 2021)}  # drop 2022-23
HOLDOUT_SPLIT = 2009                    # fit <= 2009, predict 2010-2023
RHO_GRID = np.linspace(-0.5, 0.95, 59)
S_GRID = np.concatenate([[0.0], np.geomspace(1e-4, 0.5, 60)])   # mm/yr
SSPS = ['SSP1_1_9', 'SSP1_2_6', 'SSP2_4_5', 'SSP3_7_0', 'SSP5_8_5']
SSP_LABEL = {k: k.replace('_', '-', 1).replace('_', '.') for k in SSPS}
SSP_COLOR = dict(zip(SSPS, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']))
N_DRAW = 2000
SEED = 20261003
PROJ_YEARS = np.arange(2024, 2100)      # rates summed to reach 2100.0


# ── Data ──────────────────────────────────────────────────────────────────

def load_imbie():
    d = read_imbie3(str(IMBIE_PATH), convert_to_meters=False)
    cols = ['mass_balance_rate', 'mass_balance_rate_sigma',
            'dynamics_rate', 'dynamics_rate_sigma',
            'cumulative_mass_balance', 'cumulative_mass_balance_sigma']
    ann = d[cols].groupby(d.index.year).mean()
    end = d[['cumulative_mass_balance', 'cumulative_mass_balance_sigma']]
    end = end.groupby(d.index.year).last()       # December value
    ann['cum_end'] = end['cumulative_mass_balance']
    ann['cum_end_sigma'] = end['cumulative_mass_balance_sigma']
    ann.index.name = 'year'
    return ann


def load_gmst():
    b = pd.read_hdf(H5_PATH, 'harmonized/df_berkeley_h')
    T = b['temperature'].groupby(b.index.year).mean()
    T = T - T.loc[T_BASE[0]:T_BASE[1]].mean()
    return T


def load_ssps(T_obs=None):
    """AR6 SSP GMST in the 1995-2014 frame.

    With T_obs, each SSP is instead shifted so its 2015-2024 median mean
    equals the observed 2015-2024 mean (splice sensitivity).
    """
    hist = pd.read_hdf(H5_PATH, 'projections/temp/Historical').set_index('decimal_year')
    shift = hist['temperature'].loc[T_BASE[0]:T_BASE[1]].mean()
    out = {}
    for k in SSPS:
        df = pd.read_hdf(H5_PATH, f'projections/temp/{k}').reset_index(drop=True)
        sh = shift
        if T_obs is not None:
            m = (df['decimal_year'] >= 2015) & (df['decimal_year'] <= 2024)
            sh = df.loc[m, 'temperature'].mean() - T_obs.loc[2015:2024].mean()
        for c in ['temperature', 'temperature_lower', 'temperature_upper']:
            df[c] = df[c] - sh
        out[k] = df
    return out, shift


# ── Design matrices ───────────────────────────────────────────────────────

def integrated_T(T):
    T_pi = T.loc[PI[0]:PI[1]].mean()
    return (T - T_pi).cumsum()


def design(model, years, T, lag=0):
    years = np.asarray(years)
    one = np.ones(len(years))
    if model == 'const':
        return one[:, None]
    if model == 'trend':
        return np.column_stack([one, years - 2000.0])
    if model == 'lagT':
        return np.column_stack([one, T.reindex(years - lag).values])
    if model == 'intT':
        return np.column_stack([one, integrated_T(T).reindex(years).values])
    raise ValueError(model)


# ── GLS on the (rho, s) grid ──────────────────────────────────────────────

def ar1_corr(n, rho):
    i = np.arange(n)
    return rho ** np.abs(i[:, None] - i[None, :])


def gls_grid(y, X, sig):
    """GLS on the (rho, s) grid.

    Returns per-cell beta_hat, cov, log max-likelihood, and log flat-prior
    marginal likelihood (beta integrated out).
    """
    n, p = X.shape
    D = np.diag(sig ** 2)
    cells = []
    for rho in RHO_GRID:
        R = ar1_corr(n, rho)
        for s in S_GRID:
            C = D + s ** 2 * R
            L = np.linalg.cholesky(C)
            Xw = np.linalg.solve(L, X)
            yw = np.linalg.solve(L, y)
            A = Xw.T @ Xw
            bh = np.linalg.solve(A, Xw.T @ yw)
            res = yw - Xw @ bh
            logdetC = 2 * np.log(np.diag(L)).sum()
            q = res @ res
            ll = -0.5 * (n * np.log(2 * np.pi) + logdetC + q)
            _, logdetA = np.linalg.slogdet(A)
            lml = ll + 0.5 * p * np.log(2 * np.pi) - 0.5 * logdetA
            cells.append((rho, s, bh, np.linalg.inv(A), ll, lml))
    return cells


def fit(model, years, y, sig, T, lag=0):
    X = design(model, years, T, lag)
    cells = gls_grid(y, X, sig)
    ll = np.array([c[4] for c in cells])
    lml = np.array([c[5] for c in cells])
    w = np.exp(lml - lml.max()); w /= w.sum()
    k = X.shape[1] + 2
    n = len(y)
    best = cells[int(np.argmax(ll))]
    chi2 = np.sum(((y - X @ best[2]) / sig) ** 2) / (n - X.shape[1])
    return dict(model=model, lag=lag, cells=cells, w=w, k=k, n=n,
                llmax=ll.max(), bic=k * np.log(n) - 2 * ll.max(),
                rho_ml=best[0], s_ml=best[1], chi2=chi2)


def posterior_moments(f):
    """Mean and sd of each beta under the grid mixture."""
    B = np.array([c[2] for c in f['cells']])
    V = np.array([np.diag(c[3]) for c in f['cells']])
    m = f['w'] @ B
    v = f['w'] @ (V + B ** 2) - m ** 2
    return m, np.sqrt(v)


def fit_family(years, y, sig, T):
    """All models; lagT for every lag.  Returns list of fits with BIC weights."""
    fits = [fit('const', years, y, sig, T), fit('trend', years, y, sig, T),
            fit('intT', years, y, sig, T)]
    fits += [fit('lagT', years, y, sig, T, L) for L in range(LAG_MAX + 1)]
    bic = np.array([f['bic'] for f in fits])
    wb = np.exp(-0.5 * (bic - bic.min())); wb /= wb.sum()
    for f, wi in zip(fits, wb):
        f['w_BIC'] = wi
    return fits


def family_weights(fits):
    """Family BICs and weights; lagT pooled over lags with prior 1/16 each."""
    fam = {}
    for name in ('const', 'trend', 'intT'):
        fam[name] = next(f for f in fits if f['model'] == name)['bic']
    b = np.array([f['bic'] for f in fits if f['model'] == 'lagT'])
    fam['lagT'] = (-2 * (np.log(np.exp(-0.5 * (b - b.min())).sum()) - 0.5 * b.min())
                   + 2 * np.log(len(b)))
    v = np.array(list(fam.values()))
    w = np.exp(-0.5 * (v - v.min())); w /= w.sum()
    return dict(zip(fam, v)), dict(zip(fam, w))


def lag_mixture(fits):
    lag = [f for f in fits if f['model'] == 'lagT']
    bic = np.array([f['bic'] for f in lag])
    w = np.exp(-0.5 * (bic - bic.min())); w /= w.sum()
    return lag, w


# ── Predictive sampling ───────────────────────────────────────────────────

def draw_params(f, n, rng):
    idx = rng.choice(len(f['cells']), size=n, p=f['w'])
    out_b, out_rho, out_s = [], [], []
    for i in idx:
        rho, s, bh, V, *_ = f['cells'][i]
        out_b.append(rng.multivariate_normal(bh, V))
        out_rho.append(rho); out_s.append(s)
    return np.array(out_b), np.array(out_rho), np.array(out_s)


def ar1_paths(rho, s, m, rng):
    """Stationary AR(1) paths, shape (n, m)."""
    n = len(rho)
    e = np.empty((n, m))
    e[:, 0] = s * rng.standard_normal(n)
    inn = s * np.sqrt(1 - rho ** 2)
    for t in range(1, m):
        e[:, t] = rho * e[:, t - 1] + inn * rng.standard_normal(n)
    return e


def predict_rates(fits_or_mix, years, T_paths, rng, n=N_DRAW):
    """Predictive rate draws (n, len(years)).

    fits_or_mix: a single fit dict, or (list_of_lag_fits, weights).
    T_paths: callable(model, lag, member_index_array) -> X rows (n, m, p).
    """
    if isinstance(fits_or_mix, dict):
        members = [(fits_or_mix, np.arange(n))]
    else:
        lf, w = fits_or_mix
        pick = rng.choice(len(lf), size=n, p=w)
        members = [(lf[j], np.where(pick == j)[0]) for j in np.unique(pick)]
    out = np.empty((n, len(years)))
    for f, ids in members:
        b, rho, s = draw_params(f, len(ids), rng)
        X = T_paths(f['model'], f['lag'], ids)          # (len(ids), m, p)
        mean = np.einsum('imp,ip->im', X, b)
        out[ids] = mean + ar1_paths(rho, s, len(years), rng)
    return out


# ── Projection GMST ───────────────────────────────────────────────────────

def projection_T(T_obs, df_ssp, z):
    """Per-member annual GMST series 1850-2099 (1995-2014 frame), (n, n_yr)."""
    yrs_obs = T_obs.index.values
    last_obs = int(min(yrs_obs.max(), WARMING_ANCHOR_YEAR))
    fut = np.arange(last_obs + 1, PROJ_YEARS[-1] + 1)
    med = np.interp(fut, df_ssp['decimal_year'].values, df_ssp['temperature'].values)
    off = offsets_on_grid(df_ssp, z, fut.astype(float))
    years = np.concatenate([np.arange(PI[0], last_obs + 1), fut])
    obs = T_obs.loc[PI[0]:last_obs].values
    paths = np.concatenate([np.tile(obs, (len(z), 1)), med[None, :] + off], axis=1)
    return years, paths


def make_T_rows(years_all, paths, years_out):
    """Callable for predict_rates on member-specific GMST paths."""
    T_pi = paths[:, (years_all >= PI[0]) & (years_all <= PI[1])].mean(axis=1, keepdims=True)
    I = np.cumsum(paths - T_pi, axis=1)
    pos = {y: i for i, y in enumerate(years_all)}

    def rows(model, lag, ids):
        m = len(years_out)
        one = np.ones((len(ids), m))
        if model == 'const':
            return one[..., None]
        if model == 'trend':
            return np.stack([one, np.tile(years_out - 2000.0, (len(ids), 1))], -1)
        if model == 'lagT':
            cols = [pos[y - lag] for y in years_out]
            return np.stack([one, paths[ids][:, cols]], -1)
        if model == 'intT':
            cols = [pos[y] for y in years_out]
            return np.stack([one, I[ids][:, cols]], -1)
    return rows


def obs_T_rows(T, years_out):
    def rows(model, lag, ids):
        X = design(model, years_out, T, lag)
        return np.broadcast_to(X, (len(ids),) + X.shape)
    return rows


# ── Main ──────────────────────────────────────────────────────────────────

def print_tables(fits):
    fb, fw = family_weights(fits)
    bmin = min(fb.values())
    print(f'{"family":<8}{"BIC":>9}{"dBIC":>7}{"weight":>8}')
    for name in sorted(fb, key=fb.get):
        print(f'{name:<8}{fb[name]:>9.2f}{fb[name] - bmin:>7.2f}{fw[name]:>8.3f}')
    lf, wl = lag_mixture(fits)
    order = np.argsort([f['bic'] for f in lf])
    show = [f for f in fits if f['model'] != 'lagT'] + [lf[i] for i in order[:3]]
    if 0 not in [lf[i]['lag'] for i in order[:3]]:
        show.append(lf[0])
    print(f'{"model":<8}{"lag":>4}{"k":>3}{"BIC":>9}{"w|lagT":>8}{"chi2_r":>8}'
          f'{"rho_ML":>8}{"s_ML":>7}   slope (mean +/- sd)')
    for f in sorted(show, key=lambda f: f['bic']):
        m, sd = posterior_moments(f)
        slope = '' if f['model'] == 'const' else f'{m[1]:+.4f} +/- {sd[1]:.4f}'
        wcond = f'{wl[f["lag"]]:8.3f}' if f['model'] == 'lagT' else f'{"":8}'
        print(f'{f["model"]:<8}{f["lag"]:>4}{f["k"]:>3}{f["bic"]:>9.2f}{wcond}'
              f'{f["chi2"]:>8.2f}{f["rho_ml"]:>8.2f}{f["s_ml"]:>7.4f}   {slope}')


def pct(a, q=(5, 50, 95)):
    return np.percentile(a, q)


def main():
    rng = np.random.default_rng(SEED)
    ann = load_imbie()
    T = load_gmst()
    ssp, shift = load_ssps()
    print(f'GMST frame: Berkeley annual rebased to {T_BASE[0]}-{T_BASE[1]}; '
          f'AR6 SSPs shifted by {shift:.3f} degC (Historical {T_BASE[0]}-{T_BASE[1]} mean).')
    print(f'Berkeley 2024 = {T.loc[2024]:.3f} degC; '
          f'SSP2-4.5 median 2025 = '
          f'{np.interp(2025, ssp["SSP2_4_5"].decimal_year, ssp["SSP2_4_5"].temperature):.3f} degC')

    series = {'total': ('mass_balance_rate', 'mass_balance_rate_sigma'),
              'dynamics': ('dynamics_rate', 'dynamics_rate_sigma')}
    results = {}

    # ── Fits and BIC tables
    for sname, (yc, sc) in series.items():
        for wname, (y0, y1) in WINDOWS.items():
            sub = ann.loc[y0:y1]
            yrs, y, sig = sub.index.values, sub[yc].values, sub[sc].values
            fits = fit_family(yrs, y, sig, T)
            results[(sname, wname)] = dict(fits=fits, years=yrs, y=y, sig=sig)
            lf, wl = lag_mixture(fits)
            print(f'\n=== {sname}, {wname} (n = {len(y)}) ===')
            print_tables(fits)
            # Lag-mixture sensitivity
            draws = np.concatenate([
                draw_params(f, int(round(N_DRAW * wi)), rng)[0][:, 1]
                for f, wi in zip(lf, wl) if round(N_DRAW * wi) > 0])
            q = pct(draws)
            print(f'lagT BIC-mixture over L=0..{LAG_MAX}: lag weights (conditional on lagT) '
                  + ', '.join(f'L{f["lag"]}:{wi:.2f}' for f, wi in zip(lf, wl) if wi > 0.02))
            print(f'  b = {q[1]:.3f} [{q[0]:.3f}, {q[2]:.3f}] mm/yr/degC  '
                  f'= {q[1]*GT_PER_MM:.0f} [{q[0]*GT_PER_MM:.0f}, {q[2]*GT_PER_MM:.0f}] Gt/yr/degC')
            fi = next(f for f in fits if f['model'] == 'intT')
            di = draw_params(fi, N_DRAW, rng)[0][:, 1]
            qi = pct(di)
            print(f'intT: c = {qi[1]*1e3:.2f} [{qi[0]*1e3:.2f}, {qi[2]*1e3:.2f}] '
                  f'x 1e-3 mm/yr per degC yr of integrated warming')
            results[(sname, wname)]['b_draws'] = draws

    # ── Dynamics without 2022-2023
    for wname, (y0, y1) in DYN_CHECK.items():
        sub = ann.loc[y0:y1]
        fits = fit_family(sub.index.values, sub['dynamics_rate'].values,
                          sub['dynamics_rate_sigma'].values, T)
        lf, wl = lag_mixture(fits)
        print(f'\n=== dynamics check, {wname} (n = {len(sub)}) ===')
        print_tables(fits)
        print('lag weights (conditional on lagT) '
              + ', '.join(f'L{f["lag"]}:{wi:.2f}' for f, wi in zip(lf, wl) if wi > 0.02))

    # ── Holdout: fit <= 2009, predict 2010-2023 (total mass balance)
    print('\n=== Holdout: fit through 2009, predict 2010-2023 (total) ===')
    hy = np.arange(HOLDOUT_SPLIT + 1, 2024)
    r_obs = ann.loc[hy, 'mass_balance_rate'].values
    cum_obs = ann.loc[2023, 'cum_end'] - ann.loc[HOLDOUT_SPLIT, 'cum_end']
    cum_sig = np.sqrt(ann.loc[2023, 'cum_end_sigma'] ** 2
                      - ann.loc[HOLDOUT_SPLIT, 'cum_end_sigma'] ** 2)
    print(f'observed 2010-2023 cumulative = {cum_obs:.2f} +/- {cum_sig:.2f} mm '
          f'(sigma: root-difference of the file\'s cumulative sigmas); '
          f'mean rate {r_obs.mean():.3f} mm/yr')
    holdout = {}
    for wname, (y0, _) in WINDOWS.items():
        sub = ann.loc[y0:HOLDOUT_SPLIT]
        fits = fit_family(sub.index.values, sub['mass_balance_rate'].values,
                          sub['mass_balance_rate_sigma'].values, T)
        lf, wl = lag_mixture(fits)
        rows = obs_T_rows(T, hy)
        cand = {'const': next(f for f in fits if f['model'] == 'const'),
                'trend': next(f for f in fits if f['model'] == 'trend'),
                'intT': next(f for f in fits if f['model'] == 'intT'),
                'lagT mix': (lf, wl)}
        print(f'-- fit window {y0}-{HOLDOUT_SPLIT}')
        for name, fm in cand.items():
            pr = predict_rates(fm, hy, rows, rng)
            cum = pr.sum(axis=1)
            q = pct(cum)
            z = (cum_obs - cum.mean()) / np.sqrt(cum.var() + cum_sig ** 2)
            print(f'   {name:<9} predicted {q[1]:6.2f} [{q[0]:6.2f}, {q[2]:6.2f}] mm   '
                  f'z = {z:+.2f}')
            holdout[(wname, name)] = pr
    # ── Projections (total, 1992-2023 window, lagT mixture and intT)
    print('\n=== Projected WAIS contribution 2000.0 -> 2100.0 (total, mm) ===')
    obs_2000_2023 = ann.loc[2000:2023, 'mass_balance_rate'].sum()
    print(f'observed 2000-2023 = {obs_2000_2023:.2f} mm (sum of IMBIE-3 annual rates)')
    z = warming_z(N_DRAW)
    proj = {}
    for wname in WINDOWS:
        fits = results[('total', wname)]['fits']
        lf, wl = lag_mixture(fits)
        fi = next(f for f in fits if f['model'] == 'intT')
        for k in SSPS:
            yrs_all, paths = projection_T(T, ssp[k], z)
            rows = make_T_rows(yrs_all, paths, PROJ_YEARS)
            for name, fm in (('lagT mix', (lf, wl)), ('intT', fi)):
                pr = predict_rates(fm, PROJ_YEARS, rows, rng)
                tot = obs_2000_2023 + pr.sum(axis=1)
                proj[(wname, name, k)] = (tot, pr)
        ftr = next(f for f in fits if f['model'] == 'trend')
        pr = predict_rates(ftr, PROJ_YEARS, obs_T_rows(T, PROJ_YEARS), rng)
        proj[(wname, 'trend')] = (obs_2000_2023 + pr.sum(axis=1), pr)
        print(f'-- fit window {wname}')
        a = pct(proj[(wname, 'trend')][0])
        r = pct(proj[(wname, 'trend')][1][:, -1])
        print(f'   {"trend":<10}{a[1]:8.1f} [{a[0]:6.1f},{a[2]:6.1f}]   (any SSP)'
              f'   rate 2099 {r[1]:.2f} [{r[0]:.2f},{r[2]:.2f}]')
        print(f'   {"SSP":<10}{"lagT mix":>26}{"intT":>26}{"rate 2099 (lagT)":>22}')
        for k in SSPS:
            a = pct(proj[(wname, 'lagT mix', k)][0])
            b = pct(proj[(wname, 'intT', k)][0])
            r = pct(proj[(wname, 'lagT mix', k)][1][:, -1])
            print(f'   {SSP_LABEL[k]:<10}{a[1]:8.1f} [{a[0]:6.1f},{a[2]:6.1f}]'
                  f'{b[1]:10.1f} [{b[0]:6.1f},{b[2]:6.1f}]'
                  f'{r[1]:8.2f} [{r[0]:.2f},{r[2]:.2f}]')

    # ── S1 reference (manuscript quadratic)
    s1 = s1_reference()
    print(f'\nS1 reference (s1_quadratic_fit.json): 2100 median {s1["median"]:.1f} mm; '
          f'parameter-only {s1["param"][0]:.1f}-{s1["param"][1]:.1f} mm; '
          f'with ISMIP6 term {s1["full"][0]:.1f}-{s1["full"][1]:.1f} mm (90%); '
          f'rate 2099.5 {s1["rate"]:.2f} mm/yr')

    # ── Splice sensitivity: SSPs matched to Berkeley over 2015-2024
    ssp_m, _ = load_ssps(T_obs=T)
    print('\n=== Splice sensitivity (1992-2023 fit): SSP medians matched to '
          'Berkeley 2015-2024 mean; 2100 medians, mm ===')
    print(f'   {"SSP":<10}{"shift":>7}{"lagT base":>11}{"lagT match":>12}'
          f'{"intT base":>11}{"intT match":>12}')
    fits = results[('total', '1992-2023')]['fits']
    lf, wl = lag_mixture(fits)
    fi = next(f for f in fits if f['model'] == 'intT')
    for k in SSPS:
        d_sh = (ssp_m[k]['temperature'] - ssp[k]['temperature']).iloc[0]
        yrs_all, paths = projection_T(T, ssp_m[k], z)
        rows = make_T_rows(yrs_all, paths, PROJ_YEARS)
        med = []
        for name, fm in (('lagT mix', (lf, wl)), ('intT', fi)):
            pr = predict_rates(fm, PROJ_YEARS, rows, rng)
            med.append(np.median(obs_2000_2023 + pr.sum(axis=1)))
        print(f'   {SSP_LABEL[k]:<10}{d_sh:>+7.3f}'
              f'{np.median(proj[("1992-2023", "lagT mix", k)][0]):>11.1f}{med[0]:>12.1f}'
              f'{np.median(proj[("1992-2023", "intT", k)][0]):>11.1f}{med[1]:>12.1f}')

    plot(ann, T, results, holdout, proj, hy, cum_obs, cum_sig, s1)


def s1_reference(year=2100.0, n=200_000):
    """S1 quadratic at *year* (mm above 2000): median, parameter-only and
    ISMIP6-widened 90% intervals, and the rate at year - 0.5."""
    cp.load_s1_quadratic()
    m, c = cp.get_s1_quadratic()
    tau = year - cp.BASELINE_YEAR
    J = np.array([0.5 * tau ** 2, tau, 1.0])
    sd = np.sqrt(J @ c @ J) * 1e3
    med = cp.s1_median_mm(year)
    full = cp._sample_s1_quadratic_mm(n, np.random.default_rng(SEED + 1),
                                      np.array([year]))[:, 0]
    return dict(median=med, param=(med - 1.645 * sd, med + 1.645 * sd),
                full=tuple(np.percentile(full, [5, 95])),
                rate=(m[1] + m[0] * (tau - 0.5)) * 1e3)


# ── Figure ────────────────────────────────────────────────────────────────

def plot(ann, T, results, holdout, proj, hy, cum_obs, cum_sig, s1):
    ink, ink2, grid = '#0b0b0b', '#52514e', '#e4e3df'
    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': ink2,
                         'axes.labelcolor': ink, 'xtick.color': ink2,
                         'ytick.color': ink2, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)

    # (a) rate vs GMST with lagT best fit, 1992-2023
    a = ax[0, 0]
    res = results[('total', '1992-2023')]
    lf, wl = lag_mixture(res['fits'])
    best = lf[int(np.argmax(wl))]
    Tl = T.reindex(res['years'] - best['lag']).values
    a.errorbar(Tl, res['y'], yerr=res['sig'], fmt='o', ms=3.5, color='#2a78d6',
               ecolor='#86b6ef', elinewidth=1, capsize=0, label='IMBIE-3 1992–2023')
    pre = ann.loc[1979:1991]
    a.errorbar(T.reindex(pre.index.values - best['lag']).values,
               pre['mass_balance_rate'], yerr=pre['mass_balance_rate_sigma'],
               fmt='o', ms=3.5, mfc='white', color='#52514e', ecolor='#c3c2b7',
               elinewidth=1, capsize=0, label='IMBIE-3 1979–1991')
    m, _ = posterior_moments(best)
    tt = np.linspace(np.nanmin(Tl) - 0.3, np.nanmax(Tl) + 0.05, 50)
    a.plot(tt, m[0] + m[1] * tt, color=ink, lw=1.5,
           label=f'lagT, L = {best["lag"]} yr (1992–2023 fit)')
    a.set_xlabel(f'GMST (L = {best["lag"]} yr earlier), °C rel. 1995–2014')
    a.set_ylabel('WAIS mass loss rate (mm/yr SLE)')
    a.legend(frameon=False, fontsize=7)
    a.grid(color=grid, lw=0.5)
    a.set_title('(a) Rate vs lagged GMST', loc='left', fontsize=9)

    # (b) dBIC vs lag, both windows
    b = ax[0, 1]
    for (wname, col) in zip(WINDOWS, ['#2a78d6', '#eb6834']):
        fits = results[('total', wname)]['fits']
        bmin = min(f['bic'] for f in fits)
        lf, _ = lag_mixture(fits)
        b.plot([f['lag'] for f in lf], [f['bic'] - bmin for f in lf], '-o',
               color=col, lw=2, ms=4, label=f'lagT, {wname}')
        for name, ls in (('trend', '--'), ('intT', ':'), ('const', '-.')):
            f = next(f for f in fits if f['model'] == name)
            b.axhline(f['bic'] - bmin, color=col, ls=ls, lw=1)
    for name, ls in (('trend', '--'), ('intT', ':'), ('const', '-.')):
        b.plot([], [], color=ink2, ls=ls, lw=1, label=name)
    b.set_xlabel('Lag L (yr)')
    b.set_ylabel('ΔBIC relative to best model in window')
    b.set_ylim(-0.5, 11)
    b.legend(frameon=False, fontsize=7, ncol=3, loc='upper center')
    b.grid(color=grid, lw=0.5)
    b.set_title('(b) Model comparison, total mass balance', loc='left', fontsize=9)

    # (c) holdout cumulative predictions
    c = ax[1, 0]
    names = ['const', 'trend', 'intT', 'lagT mix']
    for j, (wname, col) in enumerate(zip(WINDOWS, ['#2a78d6', '#eb6834'])):
        for i, name in enumerate(names):
            cum = holdout[(wname, name)].sum(axis=1)
            q = pct(cum)
            yv = i + (j - 0.5) * 0.3
            c.plot([q[0], q[2]], [yv, yv], color=col, lw=2)
            c.plot(q[1], yv, 'o', color=col, ms=5,
                   label=f'fit {WINDOWS[wname][0]}–2009' if i == 0 else None)
    c.axvspan(cum_obs - 1.645 * cum_sig, cum_obs + 1.645 * cum_sig, color='#e4e3df')
    c.axvline(cum_obs, color=ink, lw=1.5, label='observed, ±1.645σ band')
    c.set_yticks(range(len(names)), names)
    c.set_xlabel('WAIS contribution 2010–2023 (mm)')
    c.set_ylim(-0.6, len(names) + 0.9)
    c.legend(frameon=False, fontsize=6.5, loc='upper left')
    c.grid(color=grid, lw=0.5, axis='x')
    c.set_title('(c) Holdout 2010–2023 (median, 90%)', loc='left', fontsize=9)

    # (d) 2100 contribution by SSP, lagT mixture, 1992-2023 window
    d = ax[1, 1]
    for i, k in enumerate(SSPS):
        tot = proj[('1992-2023', 'lagT mix', k)][0]
        q = pct(tot)
        d.plot([q[0], q[2]], [i, i], color=SSP_COLOR[k], lw=2.5)
        d.plot(q[1], i, 'o', color=SSP_COLOR[k], ms=6)
        ti = pct(proj[('1992-2023', 'intT', k)][0])
        d.plot([ti[0], ti[2]], [i - 0.25, i - 0.25], color=SSP_COLOR[k], lw=1, ls=':')
        d.plot(ti[1], i - 0.25, 'o', mfc='white', color=SSP_COLOR[k], ms=4)
    q = pct(proj[('1992-2023', 'trend')][0])
    iy = len(SSPS)
    d.plot([q[0], q[2]], [iy, iy], color=ink2, lw=2.5)
    d.plot(q[1], iy, 's', color=ink2, ms=5)
    d.axvspan(*s1['param'], color='#e4e3df', zorder=0)
    d.axvline(s1['median'], color=ink, lw=1.2, zorder=1)
    for x in s1['full']:
        d.axvline(x, color=ink, lw=0.8, ls='--', zorder=1)
    d.plot([], [], '-o', color=ink2, lw=2.5, ms=5, label='lagT mixture')
    d.plot([], [], ':o', mfc='white', color=ink2, lw=1, ms=4, label='intT')
    d.plot([], [], color=ink, lw=1.2, label='S1 median, parameter 90%')
    d.plot([], [], color=ink, lw=0.8, ls='--', label='S1 90% with ISMIP6 term')
    d.set_yticks(range(len(SSPS) + 1), [SSP_LABEL[k] for k in SSPS] + ['trend (any SSP)'])
    d.set_ylim(-0.6, len(SSPS) + 1.9)
    d.set_xlabel('WAIS contribution 2000–2100 (mm)')
    d.legend(frameon=False, fontsize=6.5, loc='upper left', ncol=2)
    d.grid(color=grid, lw=0.5, axis='x')
    d.set_title('(d) 2000–2100, 1992–2023 fit (median, 90%)', loc='left', fontsize=9)

    fig.savefig(FIG_PATH, dpi=200)
    print(f'\nWrote {FIG_PATH.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
