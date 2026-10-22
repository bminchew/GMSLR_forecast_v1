"""Early holdout of the Greenland surface-to-ocean transfer function.

The production transfer, T_ocean = g EMA_tau(T_surface) + c with AR(1)
residuals, is fitted to EN4 200-500 m over 1970-2020
(component_analysis.fit_ocean_lowpass_transfer, tau grid 1-20 yr, as in
component_greenland.ipynb cell 8). Here that same fit predicts the withheld
EN4 years before 1970 from the observed Greenland surface temperature alone,
and the prediction is compared with EN4.

Periods. EN4 before 1970 is an objective analysis that relaxes toward the
1971-2000 climatology where there are no profiles, so the test is only as
good as the observational support (scripts/en4_greenland_profile_counts.py,
data/raw/ocean_temp/en4/derived/README.md). 1958-1969 is the main test: every
year has at least 100 QC-passed profiles in the box and a mean EN4
observation weight of 0.45-0.68, comparable to 1970-1999 (0.62-0.66).
1962-1969 (weights 0.61-0.68) is the stricter subset. 1950-1957 (weights
0.31-0.49) is a lower-confidence extension. Where data are sparse the
analysis is pulled toward the 1971-2000 climatology, which sits below zero on
the 1995-2005 baseline used here, not toward zero.

Prediction. 200,000 posterior draws of (tau, g, c) (draw_lowpass_transfer,
seed 0, as cell 8), each mapped through g EMA_tau(T_surface) + c, with the
filter run over the whole surface record from 1850 as in production. Each
draw adds a stationary AR(1) residual path with that draw's phi and marginal
SD (seed 1). The residual is not conditioned on the fitted 1970 residual,
which makes the predictive interval slightly wide for the years next to
1970. EN4 analysis error before 1970 is not added; it is larger than in the
fit years, so the test is, if anything, strict.

Reported for each period: the observed and predicted period mean with the
90% predictive interval and its percentile; per-year 90% coverage; and the
reduced chi-square of the annual residuals against the predictive
covariance. The spread of the predicted period mean across the tau grid
shows how far the period can discriminate tau. A refit on 1958-2020 shows
how far the withheld years would move the fit.

Inputs: Berkeley Earth gridded Greenland surface T (cell 3), and the EN4 box
series in data/raw/ocean_temp/en4/derived/en4_greenland_box_series_monthly.csv
(production definition and baseline, written by
scripts/en4_greenland_profile_counts.py; checked here against the
production h5 key over 1970-2020, to the CSV rounding of 1e-5 degC). Not part of run_pipeline.sh, because the
pipeline does not download the EN4 analyses before 1970.

Run from the repository root:  python scripts/greenland_ocean_early_holdout.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
sys.path.insert(0, str(ROOT / 'src'))

from slr_data_readers import read_berkeley_earth_gridded  # noqa: E402
from component_analysis import (draw_lowpass_transfer, ema_filter,  # noqa: E402
                                fit_ocean_lowpass_transfer)
from greenland_discharge import full_years_only  # noqa: E402
from smb_emulator import ar1_noise  # noqa: E402

H5_PATH = ROOT / 'data' / 'processed' / 'slr_processed_data.h5'
EN4_CSV = ROOT / 'data' / 'raw' / 'ocean_temp' / 'en4' / 'derived' / 'en4_greenland_box_series_monthly.csv'
EN4_ANNUAL = ROOT / 'data' / 'raw' / 'ocean_temp' / 'en4' / 'derived' / 'en4_greenland_box_series_annual.csv'
GRIDDED_PATH = ROOT / 'data' / 'raw' / 'gmst' / 'berkEarth_Global_TAVG_Gridded_1deg.nc'
GREENLAND_MASK = ROOT / 'data' / 'processed' / 'greenland_mask_berkeley_1deg.csv'

TAU_GRID = np.arange(1, 21)        # component_greenland.ipynb cell 8
FIT = (1970, 2020)                 # production fit years (full years of EN4)
PERIODS = {'1958-1969 (main test)': (1958, 1969),
           '1962-1969 (observation weight >= 0.6)': (1962, 1969),
           '1950-1957 (sparse, lower confidence)': (1950, 1957)}
N_DRAWS = 200_000


def surface():
    """Greenland regional surface T, monthly, rel. 1995-2005 (cell 3)."""
    df = read_berkeley_earth_gridded(str(GRIDDED_PATH), cell_mask=str(GREENLAND_MASK))
    x, t = df['temperature'].values, df['decimal_year'].values
    x = x - np.nanmean(x[(t >= 1995.0) & (t < 2006.0)])
    ok = np.isfinite(x)
    return t[ok], x[ok]


def en4():
    """EN4 box series, monthly, production definition and baseline. Checked
    against the production h5 key over the fit years."""
    e = pd.read_csv(EN4_CSV)
    t = (e['year'] + (e['month'] - 0.5) / 12.0).values
    x = e['temperature_anom_degC'].values
    p = pd.read_hdf(H5_PATH, key='harmonized/df_en4_greenland_200_500m')
    tp, xp = full_years_only(p['decimal_year'].values, p['temperature'].values)
    xp = xp - np.nanmean(xp[(tp >= 1995.0) & (tp < 2006.0)])
    f = (t >= FIT[0]) & (t < FIT[1] + 1)
    fp = (tp >= FIT[0]) & (tp < FIT[1] + 1)
    assert np.allclose(t[f], tp[fp]) and np.allclose(x[f], xp[fp], atol=1e-5), \
        'EN4 CSV differs from the production h5 series over the fit years'
    return t, x


def annual(t, x, years):
    yr = np.floor(t).astype(int)
    return np.array([x[yr == y].mean() for y in years])


def main():
    ts, xs = surface()
    to, xo = en4()
    fit_mask = (to >= FIT[0]) & (to < FIT[1] + 1)
    tr = fit_ocean_lowpass_transfer(xs, ts, xo[fit_mask], to[fit_mask], tau_grid=TAU_GRID)
    sy, sa = tr['surf_years'], tr['surf_ann']
    print(f'Fit {FIT[0]}-{FIT[1]} (n = {tr["n"]}): tau_best {tr["tau_best"]:.0f} yr, '
          f'g = {tr["alpha"]:.3f} +/- {tr["alpha_se"]:.3f}, c = {tr["beta"]:.3f} degC, '
          f'residual SD {tr["residual_std"]:.3f} degC, phi {tr["resid_ar1"]:.2f}')

    d = draw_lowpass_transfer(tr, N_DRAWS, np.random.default_rng(0))
    taus = np.unique(d['tau'])
    E = {t: ema_filter(sa, t) for t in taus}
    y0 = min(p[0] for p in PERIODS.values())
    years = np.arange(y0, FIT[0])
    idx = years - int(sy[0])
    pred = np.empty((N_DRAWS, len(years)))
    for t in taus:
        k = d['tau'] == t
        pred[k] = d['g'][k, None] * E[t][idx][None, :] + d['c'][k, None]
    pred += ar1_noise(d['sd'] * np.sqrt(1 - d['phi'] ** 2), d['phi'], len(years),
                      np.random.default_rng(1))
    obs = annual(to, xo, years)
    w_obs = pd.read_csv(EN4_ANNUAL).set_index('year')['obs_weight_mean'].reindex(years).values

    for name, (a, b) in PERIODS.items():
        s = (years >= a) & (years <= b)
        P, o = pred[:, s], obs[s]
        m_obs, m_pred = o.mean(), P.mean(axis=1)
        lo, med, hi = np.percentile(m_pred, [5, 50, 95])
        plo, phi_ = np.percentile(P, [5, 95], axis=0)
        r = o - P.mean(axis=0)
        C = np.cov(P, rowvar=False)
        chi2 = float(r @ np.linalg.solve(C, r))
        # Predicted period mean (no residual) at each tau, with its own (g, c)
        by_tau = [tr['g_by_tau'][i] * E_[idx][s].mean() + tr['c_by_tau'][i]
                  for i, E_ in enumerate(ema_filter(sa, t) for t in tr['tau_grid'])]
        print(f'\n{name}, {s.sum()} years, mean EN4 observation weight '
              f'{w_obs[s].min():.2f}-{w_obs[s].max():.2f}:')
        print(f'  period mean: EN4 {m_obs:+.3f} degC; predicted {med:+.3f} [{lo:+.3f}, {hi:+.3f}] degC; '
              f'EN4 at percentile {100 * np.mean(m_pred < m_obs):.0f} of the prediction')
        print(f'  years inside the 90% predictive interval: {np.sum((o >= plo) & (o <= phi_))} of {s.sum()}')
        print(f'  reduced chi-square against the predictive covariance: {chi2 / s.sum():.2f} '
              f'(chi2 {chi2:.1f}, {s.sum()} years)')
        print(f'  predicted period mean across tau = {tr["tau_grid"][0]:.0f}-{tr["tau_grid"][-1]:.0f} yr '
              f'(each at its fitted g, c): {min(by_tau):+.3f} to {max(by_tau):+.3f} degC')
    print('\n  year  EN4     predicted median [5th, 95th]   obs weight')
    for j, y in enumerate(years):
        p5, p50, p95 = np.percentile(pred[:, j], [5, 50, 95])
        print(f'  {y}  {obs[j]:+.3f}  {p50:+.3f} [{p5:+.3f}, {p95:+.3f}]   {w_obs[j]:.2f}')

    # How far the withheld years would move the fit
    m = (to >= 1958) & (to < FIT[1] + 1)
    tr2 = fit_ocean_lowpass_transfer(xs, ts, xo[m], to[m], tau_grid=TAU_GRID)
    for lab, t_ in (('1970-2020', tr), ('1958-2020', tr2)):
        dd = draw_lowpass_transfer(t_, N_DRAWS, np.random.default_rng(0))
        q = lambda v: np.percentile(v, [50, 5, 95])
        print(f'Fit {lab}: tau {q(dd["tau"])[0]:.0f} [{q(dd["tau"])[1]:.0f}, {q(dd["tau"])[2]:.0f}] yr, '
              f'g {q(dd["g"])[0]:.3f} [{q(dd["g"])[1]:.3f}, {q(dd["g"])[2]:.3f}], '
              f'c {q(dd["c"])[0]:+.3f} [{q(dd["c"])[1]:+.3f}, {q(dd["c"])[2]:+.3f}] degC')


if __name__ == '__main__':
    main()
