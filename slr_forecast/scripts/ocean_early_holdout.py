"""Diagnostic: early-record holdout for the two-layer thermosteric model.

The production fit (component_ocean.ipynb, cell 5) calibrates the Geoffroy
et al. (2013) two-layer model on NOAA 0-700 m (1955.5-2024.5) and NOAA
700-2000 m (0-2000 m minus 0-700 m, 2005.5-2024.5) jointly; points after the
end of the GMST forcing (2024.5) are dropped. Here the 0-700 m record before
1990.5 is withheld: the model is refitted on 0-700 m 1990.5-2024.5 plus the
full 700-2000 m record, and predicts the withheld 1955.5-1989.5 values from
observed Berkeley Earth GMST and the volcanic forcing. Same split as the
Greenland discharge early holdout (component_greenland.ipynb,
HOLDOUT_SPLIT = 1990).

The upper layer S_u is driven by GMST and the volcanic term, and the deep
layer by S_u, so the 0-700 m prediction depends only on (a, b_u, tau_u, m_v).
The 700-2000 m record starts in 2005, so keeping all of it does not leak
withheld data.

Likelihood (separate sigma_extra per record, AR(1) with one shared rho),
priors, volcanic term, Li et al. (2022) x1.14 correction, 5% residual
structural error, demeaning and MCMC settings are copied from the
production cell, except that tau_d is fixed at TAU_D_FIXED here while
production samples it from the CMIP5 prior, so the full-record b_d, rho and
sigma_extra_2 below are not the production posterior (no linear drift term since 2026-10-16; the first run of
this script, with the drift term, found it traded off against b_u). Both the
production (full-record) fit and the holdout fit are run here with identical
code, so the parameter comparison between them is like for like. As in production, the 5%
structural term is independent per point (not a correlated scale error), and
both layers start in equilibrium with the 1850 GMST anomaly. In the holdout
fit the AR(1) lags are the year differences, so the gap left by the withheld
years is accounted for.

Because each record is demeaned over its own fit window, the comparison
in the withheld window uses the fit's own offset: model and data are both
referenced to their means over the 1990.5-2024.5 fit years.

Reported:
  - posterior parameters, full vs holdout
  - withheld window: OLS rate 1955.5-1989.5 (model vs NOAA), mean level of
    the withheld window relative to the fit window and to the 1995-2005
    mean (the paper's 2000 baseline, used in the supplement), reduced chi^2 of the
    35 withheld points against the predictive distribution (obs sigma +
    sigma_extra + parametric spread), and 90% coverage
  - GMST change over the withheld window
Observed intervals use the NOAA formal errors plus the 5% residual
structural term, treated as independent between years.

Nothing here feeds the pipeline. Results go to the 'ocean_early_holdout'
group in data/processed/component_results.h5.

Usage:  python scripts/ocean_early_holdout.py               # fit, save, plot
        python scripts/ocean_early_holdout.py --plot-only   # redraw figure from HDF5
        python scripts/ocean_early_holdout.py --from-posterior  # metrics, HDF5 and
                                                  # figure from stored posteriors
"""

import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import emcee

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from slr_forecast.config import (SEEDS, Z_90, KAPPA_EOS, VOLC_ERF_CSV,
                                 THERMO_VOLC_KV, THERMO_VOLC_M_SD)
from slr_forecast.readers.forcing import read_volcanic_erf
from scipy.linalg import cho_factor, cho_solve

H5_DATA = ROOT / 'data/processed/slr_processed_data.h5'
H5_OUT = ROOT / 'data/processed/component_results.h5'
OUT_FIG = ROOT / 'figures/component_ocean_early_holdout.png'
M_TO_MM = 1000.0

HOLDOUT_SPLIT = 1990.5            # first fit year of 0-700 m in the holdout
TAU_D_FIXED = 150.0               # yr. Production samples tau_d from a CMIP5 prior;
                                  # held fixed here because the 0-700 m prediction
                                  # does not depend on it and the likelihood is flat in it
STRUCTURAL_FRAC = 0.14            # Li et al. (2022)
RESIDUAL_STRUCTURAL = 0.05
N_WALKERS, N_BURNIN, N_PRODUCTION = 64, 6000, 30000   # as production
if os.environ.get('OCEAN_HOLDOUT_QUICK'):             # smoke test only
    N_BURNIN, N_PRODUCTION = 200, 400
N_DRAW = 4000                     # posterior draws for prediction

# ── Data (as component_ocean.ipynb cell 3) ──
df_temp = pd.read_hdf(H5_DATA, key='harmonized/df_berkeley_h')
temp_monthly = df_temp['temperature'].values
t_monthly = (df_temp.index.year + (df_temp.index.month - 0.5) / 12.0).values
yr_int = np.floor(t_monthly).astype(int)
unique_yrs = np.unique(yr_int)
T_annual = np.array([temp_monthly[yr_int == y].mean() for y in unique_yrs])
T_annual_years = unique_yrs + 0.5

d700 = pd.read_hdf(H5_DATA, 'harmonized/df_noaa_tsl_0_700m')
d700 = d700[d700['decimal_year'].values <= T_annual_years[-1]]    # forcing coverage, as production (OC-2)
noaa_year = d700['decimal_year'].values
_mm = d700['tsl_mm'].values
_bl = (noaa_year >= 1995.0) & (noaa_year <= 2006.0)
noaa_tsl_m = (_mm - _mm[_bl].mean()) / M_TO_MM
noaa_tsl_se_m = d700['tsl_se_mm'].values / M_TO_MM

d2000 = pd.read_hdf(H5_DATA, 'harmonized/df_noaa_tsl_0_2000m')
d2000 = d2000[d2000['decimal_year'].values <= T_annual_years[-1]]
noaa_deep_year = d2000['decimal_year'].values
noaa_deep_m = (d2000['tsl_mm'].values - d2000['tsl_mm'].values[0]) / M_TO_MM
noaa_deep_se_m = d2000['tsl_se_mm'].values / M_TO_MM

# ── Production preprocessing (cell 5) ──
noaa_700_corr = noaa_tsl_m * (1.0 + STRUCTURAL_FRAC)
noaa_2000_corr = noaa_deep_m * (1.0 + STRUCTURAL_FRAC)
sig700 = np.sqrt(noaa_tsl_se_m**2 + (RESIDUAL_STRUCTURAL * np.abs(noaa_700_corr))**2)
sig2000 = np.sqrt(noaa_deep_se_m**2 + (RESIDUAL_STRUCTURAL * np.abs(noaa_2000_corr))**2)

# 700-2000 m: 0-2000 m minus 0-700 m in the same years, as production
_k700 = np.array([np.argmin(np.abs(noaa_year - y)) for y in noaa_deep_year])
assert np.allclose(noaa_year[_k700], noaa_deep_year)
noaa_mid_corr = noaa_2000_corr - noaa_700_corr[_k700]
sig_mid = np.sqrt(noaa_deep_se_m**2 + noaa_tsl_se_m[_k700]**2
                  + (RESIDUAL_STRUCTURAL * np.abs(noaa_mid_corr - noaa_mid_corr[0]))**2)
lag_mid = np.abs(np.subtract.outer(noaa_deep_year, noaa_deep_year))

idx700_all = np.array([np.argmin(np.abs(T_annual_years - y)) for y in noaa_year])
idx2000 = np.array([np.argmin(np.abs(T_annual_years - y)) for y in noaa_deep_year])

dt_ann = np.diff(T_annual_years)
T_avg_ann = 0.5 * (T_annual[:-1] + T_annual[1:])
n_ann = len(T_annual)
# Volcanic ERF, annual means at mid-year (production fit cell)
Fv_annual = read_volcanic_erf(VOLC_ERF_CSV, T_annual_years)
Fv_avg_ann = 0.5 * (Fv_annual[:-1] + Fv_annual[1:])


def _ll_ar1(r, sig, sx, rho, lag):
    """Gaussian log-likelihood with covariance diag(sig^2) + sx^2 rho^lag."""
    cf = cho_factor(np.diag(sig**2) + sx**2 * rho**lag, lower=True)
    return -0.5 * (r @ cho_solve(cf, r) + 2 * np.sum(np.log(np.diag(cf[0]))))


def solve_layers(tau_u, kv=0.0):
    """Two-layer cascade, as the production log-prob. tau_u and the volcanic
    coefficient kv = m_v THERMO_VOLC_KV / b_u may be arrays."""
    tau_u = np.atleast_1d(tau_u)
    kv = np.broadcast_to(np.asarray(kv, float), tau_u.shape)
    Su = np.empty((n_ann, tau_u.size))
    Su[0] = T_annual[0]
    for i in range(n_ann - 1):
        decay = np.exp(-dt_ann[i] / tau_u)
        target = T_avg_ann[i] + kv * tau_u * Fv_avg_ann[i]
        Su[i + 1] = target + (Su[i] - target) * decay
    Sd = np.empty_like(Su)
    Sd[0] = T_annual[0]
    decay_d = np.exp(-dt_ann / TAU_D_FIXED)
    for i in range(n_ann - 1):
        su_avg = 0.5 * (Su[i] + Su[i + 1])
        Sd[i + 1] = su_avg + (Sd[i] - su_avg) * decay_d[i]
    return Su, Sd


def solve_layers_scalar(tau_u, kv):
    """Scalar-tau_u copy of the production ODE loop (fast path for MCMC)."""
    Su = np.empty(n_ann)
    Su[0] = T_annual[0]
    target = T_avg_ann + kv * tau_u * Fv_avg_ann
    for i in range(n_ann - 1):
        Su[i + 1] = target[i] + (Su[i] - target[i]) * np.exp(-dt_ann[i] / tau_u)
    Sd = np.empty(n_ann)
    Sd[0] = T_annual[0]
    inv_d = 1.0 / TAU_D_FIXED
    for i in range(n_ann - 1):
        su_avg = 0.5 * (Su[i] + Su[i + 1])
        decay = np.exp(-dt_ann[i] * inv_d)
        Sd[i + 1] = su_avg + (Sd[i] - su_avg) * decay
    return Su, Sd


class LogProb:
    """Production log-probability restricted to a subset of 0-700 m years."""

    def __init__(self, mask700):
        self.idx700 = idx700_all[mask700]
        y = noaa_700_corr[mask700]
        self.y700_dm = y - y.mean()
        self.sig700 = sig700[mask700]
        self.lag700 = np.abs(np.subtract.outer(noaa_year[mask700], noaa_year[mask700]))
        self.ymid_dm = noaa_mid_corr - noaa_mid_corr.mean()

    def __call__(self, theta):
        b_u, b_d, log_tau_u, log_sigma_extra, rho, log_sigma_extra_2, m_v = theta
        a = KAPPA_EOS * b_u**2          # fixed by the equation of state (D5)
        if b_u < 0 or b_d < 0 or m_v <= 0:
            return -np.inf
        tau_u = np.exp(log_tau_u)
        if tau_u < 0.1 or tau_u > 500:
            return -np.inf
        if not (-0.99 < rho < 0.99) or log_sigma_extra_2 > np.log(0.01):
            return -np.inf
        Su, Sd = solve_layers_scalar(tau_u, m_v * THERMO_VOLC_KV / b_u)

        s = Su[self.idx700]
        H = a * s**2 + b_u * s
        r700 = self.y700_dm - (H - H.mean())
        H = b_d * Sd[idx2000]
        rmid = self.ymid_dm - (H - H.mean())

        lp = -0.5 * ((log_tau_u - np.log(8)) / 0.5)**2
        lp += -0.5 * ((m_v - 1.0) / THERMO_VOLC_M_SD)**2
        lp += log_sigma_extra_2          # flat in sigma_extra_2 (Jacobian)
        return (_ll_ar1(r700, self.sig700, np.exp(log_sigma_extra), rho, self.lag700)
                + _ll_ar1(rmid, sig_mid, np.exp(log_sigma_extra_2), rho, lag_mid) + lp)


def initial_theta(mask700):
    """OLS initialization, as production."""
    Su, Sd = solve_layers(8.0)
    Su, Sd = Su[:, 0], Sd[:, 0]
    s = Su[idx700_all[mask700]]
    X = np.column_stack([s**2, s])
    y = noaa_700_corr[mask700]
    b700 = np.linalg.lstsq(X - X.mean(0), y - y.mean(), rcond=None)[0]
    s = Su[idx2000]
    X = np.column_stack([s**2, s, Sd[idx2000]])
    b2000 = np.linalg.lstsq(X - X.mean(0), noaa_2000_corr - noaa_2000_corr.mean(),
                            rcond=None)[0]
    return np.array([max(b700[1], 1e-6), max(b2000[2], 1e-6),
                     np.log(8.0), np.log(0.001), 0.3, np.log(0.001), 1.0])


def run_fit(label):
    mask = (noaa_year >= HOLDOUT_SPLIT - 0.01) if label == 'holdout' else np.ones_like(noaa_year, bool)
    logp = LogProb(mask)
    theta0 = initial_theta(mask)
    rng = np.random.default_rng(42)
    ndim = len(theta0)
    pos0 = theta0 + 1e-4 * rng.standard_normal((N_WALKERS, ndim))
    pos0[:, :2] = np.abs(pos0[:, :2])
    np.random.seed(SEEDS['bayesian'])
    t0 = time.time()
    sampler = emcee.EnsembleSampler(N_WALKERS, ndim, logp)
    state = sampler.run_mcmc(pos0, N_BURNIN)
    sampler.reset()
    sampler.run_mcmc(state, N_PRODUCTION)
    chain = sampler.get_chain()
    flat = sampler.get_chain(flat=True)
    # Derived a as column 0, so flat[:, 0..4] = (a, b_u, b_d, log_tau_u,
    # log_sigma_extra) as before D5; then rho, log_sigma_extra_2, m_v.
    flat = np.column_stack([KAPPA_EOS * flat[:, 0]**2, flat])
    tau_int = sampler.get_autocorr_time(quiet=True)
    n, m, _ = chain.shape
    W = chain.var(axis=0, ddof=1).mean(axis=0)
    B = n * chain.mean(axis=0).var(axis=0, ddof=1)
    r_hat = np.sqrt(((n - 1) / n * W + B / n) / W)
    return dict(label=label, flat=flat, n_fit700=int(mask.sum()),
                acc=float(np.mean(sampler.acceptance_fraction)),
                tau_int=tau_int, ess=N_WALKERS * N_PRODUCTION / tau_int,
                r_hat=r_hat, minutes=(time.time() - t0) / 60)


def predict(flat, rng, ref=None):
    """Upper-layer model at every 0-700 m year, referenced to the mean over
    `ref` (default: the 1990.5-2024.5 fit years, the reference of the
    withheld-window metrics)."""
    draws = flat[rng.choice(flat.shape[0], N_DRAW, replace=False)]
    a, b_u = draws[:, 0], draws[:, 1]
    tau_u, sig_x = np.exp(draws[:, 3]), np.exp(draws[:, 4])
    Su, _ = solve_layers(tau_u, draws[:, 7] * THERMO_VOLC_KV / b_u)
    s = Su[idx700_all]                                   # (n_obs, N_DRAW)
    H = a * s**2 + b_u * s
    if ref is None:
        ref = noaa_year >= HOLDOUT_SPLIT - 0.01
    return H - H[ref].mean(0), sig_x


def withheld_metrics(H, sig_x, rng):
    ref = noaa_year >= HOLDOUT_SPLIT - 0.01
    wh = ~ref
    y = (noaa_700_corr - noaa_700_corr[ref].mean())
    yrs = noaa_year[wh]
    # predictive distribution of each withheld point
    noise = rng.standard_normal(H[wh].shape) * np.sqrt(sig700[wh, None]**2 + sig_x[None]**2)
    pred = H[wh] + noise
    lo, hi = np.percentile(pred, [5, 95], axis=1)
    cover = np.mean((y[wh] >= lo) & (y[wh] <= hi))
    var_pred = H[wh].var(1) + sig700[wh]**2 + np.median(sig_x)**2
    chi2n = np.mean((y[wh] - np.median(H[wh], 1))**2 / var_pred)
    # OLS rate over the withheld window
    X = yrs - yrs.mean()
    rate_mod = (X @ (H[wh] - H[wh].mean(0))) / (X @ X)
    rate_obs = (X @ (y[wh] - y[wh].mean())) / (X @ X)
    rate_obs_se = np.sqrt(np.sum((X / (X @ X))**2 * sig700[wh]**2))
    # mean level of the withheld window relative to the fit window
    lev_mod = H[wh].mean(0)
    lev_obs = y[wh].mean()
    lev_obs_se = np.sqrt(np.sum(sig700[wh]**2) / wh.sum()**2
                         + np.sum(sig700[ref]**2) / ref.sum()**2)
    # the same relative to the 1995-2005 mean (the paper's 2000 baseline)
    lev2000_mod = (H[wh] - H[_bl].mean(0)).mean(0)
    lev2000_obs = y[wh].mean() - y[_bl].mean()
    lev2000_obs_se = np.sqrt(np.sum(sig700[wh]**2) / wh.sum()**2
                             + np.sum(sig700[_bl]**2) / _bl.sum()**2)
    return dict(cover=cover, chi2n=chi2n, n=int(wh.sum()),
                rate_mod=rate_mod, rate_obs=rate_obs, rate_obs_se=rate_obs_se,
                lev_mod=lev_mod, lev_obs=lev_obs, lev_obs_se=lev_obs_se,
                lev2000_mod=lev2000_mod, lev2000_obs=lev2000_obs,
                lev2000_obs_se=lev2000_obs_se,
                lo=lo, hi=hi, yrs=yrs, y=y)


def q(x):
    m, lo, hi = np.median(x), *np.percentile(x, [5, 95])
    return m, lo, hi


def main():
    print(f'Running full-record and holdout fits in parallel '
          f'({N_WALKERS} walkers, {N_BURNIN}+{N_PRODUCTION} steps each)...', flush=True)
    with Pool(2) as pool:
        full, hold = pool.map(run_fit, ['full', 'holdout'])

    print('\n' + '=' * 78)
    print('OCEAN 0-700 m EARLY-RECORD HOLDOUT (two-layer model)')
    print('=' * 78)
    for r in (full, hold):
        print(f"{r['label']:8s}: n(0-700 m) = {r['n_fit700']}, acceptance {r['acc']:.3f}, "
              f"min ESS {r['ess'].min():.0f}, max R_hat {r['r_hat'].max():.4f}, "
              f"{r['minutes']:.1f} min")
    report_and_save(full, hold)


def report_and_save(full, hold):
    """Parameter table, withheld-window metrics, HDF5 output and figure.
    Needs only each fit's label and flat posterior."""
    rng = np.random.default_rng(SEEDS['tests'])
    names = ['a (mm/C^2)', 'b_u (mm/C)', 'b_d (mm/C)', 'tau_u (yr)', 'sigma_extra (mm)',
             'rho', 'sigma_extra_2 (mm)', 'm_v']
    tf = ([lambda v: v * M_TO_MM] * 3 + [np.exp, lambda v: np.exp(v) * M_TO_MM]
          + [lambda v: v, lambda v: np.exp(v) * M_TO_MM, lambda v: v])
    print(f"\n{'parameter':18s}{'full record':>30s}{'holdout (0-700 m from 1990.5)':>34s}")
    for i, (nm, f) in enumerate(zip(names, tf)):
        a, b = q(f(full['flat'][:, i])), q(f(hold['flat'][:, i]))
        print(f'{nm:18s}{a[0]:12.3f} [{a[1]:7.3f}, {a[2]:7.3f}]'
              f'{b[0]:14.3f} [{b[1]:7.3f}, {b[2]:7.3f}]')

    T_dif = (T_annual[(T_annual_years >= 1985) & (T_annual_years < 1990)].mean()
             - T_annual[(T_annual_years >= 1955) & (T_annual_years < 1960)].mean())
    msk = (T_annual_years >= 1955) & (T_annual_years < 1990)
    T_ols = np.polyfit(T_annual_years[msk], T_annual[msk], 1)[0] * 35
    print(f'\nGMST change over the withheld window: {T_dif:.2f} C (1985-89 minus 1955-59 means); '
          f'OLS trend x 35 yr = {T_ols:.2f} C')

    out = {}
    for r in (full, hold):
        H, sx = predict(r['flat'], rng)
        mt = withheld_metrics(H, sx, rng)
        out[r['label']] = (H, sx, mt)
        tag = 'in sample' if r['label'] == 'full' else 'WITHHELD'
        print(f"\n{r['label']} fit, 1955.5-1989.5 ({tag}, {mt['n']} pts):")
        m = q(mt['rate_mod'] * M_TO_MM)
        print(f"  rate      model {m[0]:.3f} [{m[1]:.3f}, {m[2]:.3f}] mm/yr   "
              f"NOAA {mt['rate_obs']*M_TO_MM:.3f} +/- {1.645*mt['rate_obs_se']*M_TO_MM:.3f} (90%)")
        m = q(mt['lev_mod'] * M_TO_MM)
        print(f"  mean level rel. to 1990.5-2024.5 mean: model {m[0]:.2f} [{m[1]:.2f}, {m[2]:.2f}] mm   "
              f"NOAA {mt['lev_obs']*M_TO_MM:.2f} +/- {1.645*mt['lev_obs_se']*M_TO_MM:.2f} (90%)")
        m = q(mt['lev2000_mod'] * M_TO_MM)
        print(f"  mean level rel. to 1995-2005 mean:     model {m[0]:.2f} [{m[1]:.2f}, {m[2]:.2f}] mm   "
              f"NOAA {mt['lev2000_obs']*M_TO_MM:.2f} +/- {1.645*mt['lev2000_obs_se']*M_TO_MM:.2f} (90%)")
        print(f"  reduced chi^2 = {mt['chi2n']:.2f};  90% predictive coverage = "
              f"{mt['cover']*100:.0f}% of points")

    # ── Save ──
    h5_out = Path(os.environ.get('OCEAN_HOLDOUT_H5', H5_OUT))
    fig_out = Path(os.environ.get('OCEAN_HOLDOUT_FIG', OUT_FIG))
    with h5py.File(h5_out, 'a') as f:
        if 'ocean_early_holdout' in f:
            del f['ocean_early_holdout']
        g = f.create_group('ocean_early_holdout')
        g.attrs['description'] = ('Two-layer thermosteric fit with NOAA 0-700 m before 1990.5 '
                                  'withheld; scripts/ocean_early_holdout.py. Diagnostic only.')
        g.attrs['holdout_split'] = HOLDOUT_SPLIT
        for r in (full, hold):
            gg = g.create_group(r['label'])
            # Random rows, not a fixed stride: the flat chain is step-major
            # (row = step * n_walkers + walker), so a stride that is a
            # multiple of the walker count keeps only a few walkers (OC-3)
            _rng_thin = np.random.default_rng(4242)
            _n_keep = min(20000, r['flat'].shape[0])
            thin = r['flat'][np.sort(_rng_thin.choice(r['flat'].shape[0], _n_keep, replace=False))]
            gg.create_dataset('posterior_flat', data=thin, compression='gzip')
            gg.attrs['params'] = ('a, b_u, b_d (m units), log_tau_u, log_sigma_extra, rho, '
                                  'log_sigma_extra_2, m_v')
            H, sx, mt = out[r['label']]
            gg.create_dataset('rate_model_mm_yr', data=mt['rate_mod'] * M_TO_MM)
            gg.create_dataset('level_model_mm', data=mt['lev_mod'] * M_TO_MM)
            gg.attrs['rate_obs_mm_yr'] = mt['rate_obs'] * M_TO_MM
            gg.attrs['rate_obs_se_mm_yr'] = mt['rate_obs_se'] * M_TO_MM
            gg.attrs['level_obs_mm'] = mt['lev_obs'] * M_TO_MM
            gg.attrs['level_obs_se_mm'] = mt['lev_obs_se'] * M_TO_MM
            gg.create_dataset('level2000_model_mm', data=mt['lev2000_mod'] * M_TO_MM)
            gg.attrs['level2000_obs_mm'] = mt['lev2000_obs'] * M_TO_MM
            gg.attrs['level2000_obs_se_mm'] = mt['lev2000_obs_se'] * M_TO_MM
            gg.attrs['chi2n_withheld'] = mt['chi2n']
            gg.attrs['coverage90_withheld'] = mt['cover']
            gg.create_dataset('model_median_mm', data=np.median(H, 1) * M_TO_MM)
            gg.create_dataset('model_p05_mm', data=np.percentile(H, 5, 1) * M_TO_MM)
            gg.create_dataset('model_p95_mm', data=np.percentile(H, 95, 1) * M_TO_MM)
        g.create_dataset('noaa_year', data=noaa_year)
        g.create_dataset('noaa_700_ref_mm', data=out['holdout'][2]['y'] * M_TO_MM)
        g.create_dataset('sigma_700_total_mm', data=sig700 * M_TO_MM)

    plot_figure(h5_out, fig_out)


def plot_figure(h5_path=H5_OUT, fig_out=OUT_FIG):
    """Supplement figure from the stored results (no refit).

    (a) NOAA 0-700 m with the full-record and holdout fits, model and data
    each referenced to their 1995-2005 mean (the paper's 2000 baseline, as in
    the production fit figure); (b) the same relative to the full-record
    median. Bands are 90% credible intervals of the model (no observation
    noise), recomputed from the stored posteriors; error bars are 90%
    intervals of the total point error.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT / 'notebooks'))
    sys.path.insert(0, str(ROOT / 'notebooks' / 'arete_mpl'))
    import arete_mpl
    from component_plotting import COMP_COLORS
    arete_mpl.use('poster')
    INK, FULL_C, HOLD_C = '#333333', COMP_COLORS['Thermosteric'], '#036C9A'

    with h5py.File(h5_path, 'r') as f:
        g = f['ocean_early_holdout']
        yrs, y, sy = g['noaa_year'][:], g['noaa_700_ref_mm'][:], g['sigma_700_total_mm'][:]
        flats = {k: g[k]['posterior_flat'][:] for k in ('full', 'holdout')}
    y = y - y[_bl].mean()
    fit = {}
    for k, fl in flats.items():
        H, _ = predict(fl, np.random.default_rng(SEEDS['tests']), ref=_bl)
        H = H * M_TO_MM
        fit[k] = (np.median(H, 1), np.percentile(H, 5, 1), np.percentile(H, 95, 1))
    ref = fit['full'][0]
    labels = {'full': 'Fit to 1955–2024', 'holdout': 'Fit to 1990–2024 only'}
    colors = {'full': FULL_C, 'holdout': HOLD_C}

    with plt.rc_context({'font.size': 9, 'axes.linewidth': 0.6, 'axes.labelsize': 9,
                         'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5}):
        fig, ax = plt.subplots(2, 1, figsize=(6.5, 4.8), sharex=True,
                               gridspec_kw={'height_ratios': [2.4, 1], 'hspace': 0.08})
        for a, off in ((ax[0], 0.0), (ax[1], ref)):
            a.axvspan(yrs[0] - 1.5, HOLDOUT_SPLIT - 0.5, color='0.93', lw=0, zorder=0,
                      label='Withheld from the 1990–2024 fit' if a is ax[0] else None)
            for k in ('full', 'holdout'):
                med, lo, hi = fit[k]
                a.fill_between(yrs, lo - off, hi - off, color=colors[k], alpha=0.18, lw=0, zorder=2)
                if k == 'holdout':
                    # dashed where the fit extrapolates into the withheld years
                    wh = yrs < HOLDOUT_SPLIT
                    i0 = int(np.argmax(~wh))
                    a.plot(yrs[:i0 + 1], (med - off)[:i0 + 1], color=colors[k], lw=1.6,
                           ls='--', zorder=5)
                    a.plot(yrs[i0:], (med - off)[i0:], color=colors[k], lw=1.6, zorder=5,
                           label=labels[k] if a is ax[0] else None)
                else:
                    a.plot(yrs, med - off, color=colors[k], lw=1.6, zorder=5,
                           label=labels[k] if a is ax[0] else None)
            a.errorbar(yrs, y - off, yerr=Z_90 * sy, fmt='o', ms=2.6, mfc='white', mew=0.7,
                       color=INK, ecolor=INK, elinewidth=0.6, capsize=0, alpha=0.85, lw=0,
                       zorder=6, label='NOAA 0–700 m' if a is ax[0] else None)
            a.grid(True, alpha=0.18, lw=0.5)
            a.set_axisbelow(True)
            for sp in ('top', 'right'):
                a.spines[sp].set_visible(False)
        ax[1].axhline(0, color=FULL_C, ls='--', lw=0.9, zorder=4)
        ax[0].set_xlim(yrs[0] - 1.5, yrs[-1] + 1.5)
        ax[0].set_ylabel('Thermosteric sea level (mm)')
        ax[1].set_ylabel('Relative to full\ntime-series fit (mm)')
        ax[1].set_xlabel('Year')
        ax[0].legend(frameon=False, fontsize=7.2, loc='lower right', handlelength=1.6)
        for a, lab in ((ax[0], '(a)'), (ax[1], '(b)')):
            a.text(0.02, 0.95 if a is ax[0] else 0.93, lab, transform=a.transAxes,
                   va='top', fontweight='bold')
        fig_out = Path(fig_out)
        fig_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_out, dpi=300, bbox_inches='tight')
        plt.close(fig)
    print(f'Figure: {fig_out}')


if __name__ == '__main__':
    if '--from-posterior' in sys.argv:
        # recompute metrics, HDF5 output and figure from the stored posteriors
        with h5py.File(Path(os.environ.get('OCEAN_HOLDOUT_H5', H5_OUT)), 'r') as f:
            g = f['ocean_early_holdout']
            full, hold = ({'label': k, 'flat': g[k]['posterior_flat'][:]}
                          for k in ('full', 'holdout'))
        report_and_save(full, hold)
    elif '--plot-only' in sys.argv:
        plot_figure(Path(os.environ.get('OCEAN_HOLDOUT_H5', H5_OUT)),
                    Path(os.environ.get('OCEAN_HOLDOUT_FIG', OUT_FIG)))
    else:
        main()
