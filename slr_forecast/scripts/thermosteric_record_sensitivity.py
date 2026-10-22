"""Calibration-record sensitivity of the two-layer thermosteric model.

Refits the production thermosteric model (notebooks/component_ocean.ipynb)
with each of three observational records in place of NOAA (Levitus et al.
2012), and projects the five SSPs to 2100.  The production model is unchanged;
this script feeds only the supplementary record-sensitivity figure and its
numbers.

Records
-------
  noaa  NOAA yearly 0-700 m (1955-) and 0-2000 m (2005-) thermosteric sea level
  en4   EN4.2.2 analyses, TEOS-10 thermosteric height, salinity held at the EN4
        1971-2000 climatology (data/processed/en4_thermosteric_annual.csv, from
        scripts/en4_thermosteric.py)
  iap   IAP v4.2 heat content (data/processed/iap_v42_ohc_global_monthly.csv, from
        scripts/iap_ohc_global.py), converted to height with EN4's own
        thermosteric-to-heat ratio over 2005-2024 for each layer.  This assumes
        the IAP warming has the EN4 spatial pattern.

EN4 and IAP give no global-mean standard error; their fits use the NOAA yearly
SE of the same year.  Each record is fitted unscaled (x1.00) and with the
production Li et al. (2022) scaling (x1.14), from 1955 (0-700 m) and 2005
(0-2000 m), as in production.

The likelihood, priors, volcanic term, sampler settings, seeds and projection
loop are transcribed from component_ocean.ipynb cells 3, 5 and 9.  The
log-probability is vectorized over walkers and checked against a scalar
transcription; the NOAA x1.14 fit reproduces the production posterior and
projections (checked in the collect step).

Usage
-----
  python scripts/thermosteric_record_sensitivity.py all        # 6 fits in parallel, then collect
  python scripts/thermosteric_record_sensitivity.py fit --record en4 --scale 1.14
  python scripts/thermosteric_record_sensitivity.py collect

Outputs: data/processed/thermosteric_record_fits/<label>.json and
data/processed/thermosteric_record_sensitivity.json.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'notebooks'))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import emcee  # noqa: E402
from scipy.linalg import cho_factor, cho_solve  # noqa: E402

from slr_forecast.config import (BASELINE_YEAR, BASELINE_WINDOW, SEEDS, KAPPA_EOS,  # noqa: E402
                                 VOLC_ERF_CSV, THERMO_VOLC_KV, THERMO_VOLC_M_SD)
from slr_forecast.readers.forcing import read_noaa_thermosteric_yearly, read_volcanic_erf  # noqa: E402
from bayesian_models import solve_twolayer_ode  # noqa: E402

H5_PATH = str(ROOT / 'data/processed/slr_processed_data.h5')
H5_COMP = ROOT / 'data/processed/component_results.h5'
RAW_DIR = ROOT / 'data/raw'
EN4_CSV = ROOT / 'data/processed/en4_thermosteric_annual.csv'
IAP_CSV = ROOT / 'data/processed/iap_v42_ohc_global_monthly.csv'
FIT_DIR = ROOT / 'data/processed/thermosteric_record_fits'
OUT_JSON = ROOT / 'data/processed/thermosteric_record_sensitivity.json'
CONV_WINDOW = (2005, 2024)
TREND_WINDOWS = ((1955, 2012), (1970, 2005), (1993, 2020), (2005, 2024), (2013, 2024))
FITS = [('noaa', 1.14), ('noaa', 1.0), ('en4', 1.0), ('en4', 1.14), ('iap', 1.0), ('iap', 1.14)]
M_TO_MM = 1000.0
STRUCTURAL_FRAC = 0.14
RESIDUAL_STRUCTURAL = 0.05
RATE_BELOW_2000 = 0.117e-3
SIGMA_BELOW_2000 = 0.037e-3

# ── GMST forcing (cell 3) ──
df_temp = pd.read_hdf(H5_PATH, key='harmonized/df_berkeley_h')
temp_monthly = df_temp['temperature'].values
temp_time_monthly = (df_temp.index.year + (df_temp.index.month - 0.5) / 12.0).values
yr_int = np.floor(temp_time_monthly).astype(int)
unique_yrs = np.unique(yr_int)
T_annual = np.array([temp_monthly[yr_int == y].mean() for y in unique_yrs])
T_annual_years = unique_yrs + 0.5
dt_ann = np.diff(T_annual_years)
T_avg_ann = 0.5 * (T_annual[:-1] + T_annual[1:])
n_ann = len(T_annual)
Fv_annual = read_volcanic_erf(VOLC_ERF_CSV, T_annual_years)
Fv_avg_ann = 0.5 * (Fv_annual[:-1] + Fv_annual[1:])

# CMIP5 tau_d prior (cell 5)
_G13_C0 = np.array([53, 90, 71, 69, 99, 69, 127, 105, 126, 82, 317, 95, 145, 71, 64, 105.])
_G13_GAMMA = np.array([0.67, 0.53, 0.59, 0.93, 0.50, 0.88, 0.76, 0.90, 1.16, 0.55,
                       0.65, 0.59, 0.76, 0.72, 0.66, 0.88])
_log_taud_cmip5 = np.log(_G13_C0 / _G13_GAMMA)
TAU_D_LOG_MU = _log_taud_cmip5.mean()
TAU_D_LOG_SIGMA = _log_taud_cmip5.std(ddof=1)
TAU_D_INIT = float(np.exp(TAU_D_LOG_MU))


def _trend(t, y):
    """OLS trend with a 90% interval; AR(1)-adjusted effective sample size."""
    X = np.c_[np.ones_like(t), t]
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ c
    r1 = np.corrcoef(r[:-1], r[1:])[0, 1]
    n = len(t)
    neff = max(3.0, n * (1 - r1) / (1 + r1))
    se = np.sqrt(r @ r / (n - 2) / ((t - t.mean())**2).sum()) * np.sqrt(n / neff)
    return float(c[1]), float(1.645 * se)


def load_en4():
    """EN4 annual series (scripts/en4_thermosteric.py annual)."""
    return pd.read_csv(EN4_CSV, index_col='year')


def mm_per_zj():
    """EN4's own thermosteric-to-heat ratio by layer (2005-2024 trend ratio)."""
    e = load_en4().loc[CONV_WINDOW[0]:CONV_WINDOW[1]]
    t = e.index.values.astype(float)
    return {H: _trend(t, e[f'h{H}_mm'].values)[0] / _trend(t, e[f'ohc{H}_zj'].values)[0]
            for H in (700, 2000)}


def load_iap():
    """IAP v4.2 annual means (complete years) converted to mm with mm_per_zj()."""
    d = pd.read_csv(IAP_CSV)
    d['year'] = d['yyyymm'] // 100
    n = d.groupby('year').size()
    a = d.groupby('year')[['ohc700_zj', 'ohc2000_zj']].mean()[n == 12]
    k = mm_per_zj()
    return pd.DataFrame({'h700_mm': a['ohc700_zj'] * k[700], 'h2000_mm': a['ohc2000_zj'] * k[2000],
                         'ohc700_zj': a['ohc700_zj'], 'ohc2000_zj': a['ohc2000_zj']})


def load_noaa():
    d7 = read_noaa_thermosteric_yearly(f'{RAW_DIR}/steric/noaa_tsl_yearly_0-700m.dat')
    d2 = read_noaa_thermosteric_yearly(f'{RAW_DIR}/steric/noaa_tsl_yearly_0-2000m.dat')
    return (d7['decimal_year'].values, d7['tsl_mm'].values, d7['tsl_se_mm'].values,
            d2['decimal_year'].values, d2['tsl_mm'].values, d2['tsl_se_mm'].values)


def build_data(record, scale, start700, se_mode='noaa'):
    """Return the arrays the production cell 5 builds, for the chosen record."""
    ny, n7, n7se, nyd, n2, n2se = load_noaa()
    if record == 'noaa':
        y7, v7, s7, y2, v2, s2 = ny, n7, n7se, nyd, n2, n2se
    else:
        e = load_en4() if record == 'en4' else load_iap()
        y7 = e.index.values + 0.5
        v7 = e['h700_mm'].values
        y2 = y7.copy()
        v2 = e['h2000_mm'].values
        keep2 = y2 >= 2005.5
        y2, v2 = y2[keep2], v2[keep2]
        # formal errors: NOAA yearly SE in the same year (EN4 and IAP give no global-mean SE)
        s7 = np.interp(y7, ny, n7se, left=np.nan)
        s2 = np.interp(y2, nyd, n2se)
        if se_mode == 'zero':
            s7 = np.zeros_like(v7)
            s2 = np.zeros_like(v2)
    # forcing coverage (OC-2) and start year
    k = (y7 <= T_annual_years[-1]) & (y7 >= start700 + 0.5 - 1e-6) & np.isfinite(v7)
    y7, v7, s7 = y7[k], v7[k], s7[k]
    k = y2 <= T_annual_years[-1]
    y2, v2, s2 = y2[k], v2[k], s2[k]
    if np.any(~np.isfinite(s7)):
        raise SystemExit('missing formal SE for some 0-700 m years')
    bl = (y7 >= BASELINE_WINDOW[0]) & (y7 < BASELINE_WINDOW[1] + 1)
    m7 = (v7 - v7[bl].mean()) / M_TO_MM
    m2 = (v2 - v2[0]) / M_TO_MM
    s7 = s7 / M_TO_MM
    s2 = s2 / M_TO_MM
    c7 = m7 * scale
    c2 = m2 * scale
    sig7 = np.sqrt(s7**2 + (RESIDUAL_STRUCTURAL * np.abs(c7))**2)
    sig2 = np.sqrt(s2**2 + (RESIDUAL_STRUCTURAL * np.abs(c2))**2)
    k700 = np.array([np.argmin(np.abs(y7 - y)) for y in y2])
    assert np.allclose(y7[k700], y2)
    cmid = c2 - c7[k700]
    sigmid = np.sqrt(s2**2 + s7[k700]**2 + (RESIDUAL_STRUCTURAL * np.abs(cmid))**2)
    assert np.allclose(np.diff(y7), 1) and np.allclose(np.diff(y2), 1)
    D = dict(y7=y7, y2=y2, c7=c7, c2=c2, cmid=cmid, sig7=sig7, sig2=sig2, sigmid=sigmid,
             s7=s7, s2=s2,
             c7_dm=c7 - c7.mean(), c2_dm=c2 - c2.mean(), cmid_dm=cmid - cmid.mean(),
             idx7=np.array([np.argmin(np.abs(T_annual_years - y)) for y in y7]),
             idx2=np.array([np.argmin(np.abs(T_annual_years - y)) for y in y2]))
    D['lag7'] = np.abs(np.subtract.outer(np.arange(len(y7)), np.arange(len(y7))))
    D['lag2'] = np.abs(np.subtract.outer(np.arange(len(y2)), np.arange(len(y2))))
    return D


def _ll_ar1(r, sig, sx, rho, lag):
    cf = cho_factor(np.diag(sig**2) + sx**2 * rho**lag, lower=True)
    return -0.5 * (r @ cho_solve(cf, r) + 2 * np.sum(np.log(np.diag(cf[0]))))


def logprob_scalar(theta, D):
    """Verbatim transcription of twolayer_log_prob (cell 5)."""
    b_u, b_d, log_tau_u, log_sigma_extra, log_tau_d = theta[:5]
    rho, log_sigma_extra_2, m_v = theta[5:]
    if b_u < 0 or b_d < 0:
        return -np.inf
    tau_u, tau_d = np.exp(log_tau_u), np.exp(log_tau_d)
    if not (0.1 < tau_u < 500) or not (10 < tau_d < 5000):
        return -np.inf
    a = KAPPA_EOS * b_u**2
    sigma_extra = np.exp(log_sigma_extra)
    lp = -0.5 * ((log_tau_u - np.log(8)) / 0.5)**2
    lp += -0.5 * ((log_tau_d - TAU_D_LOG_MU) / TAU_D_LOG_SIGMA)**2
    if m_v <= 0:
        return -np.inf
    lp += -0.5 * ((m_v - 1.0) / THERMO_VOLC_M_SD)**2
    Su = np.empty(n_ann)
    Su[0] = T_annual[0]
    T_target = T_avg_ann + m_v * THERMO_VOLC_KV / b_u * tau_u * Fv_avg_ann
    for i in range(n_ann - 1):
        Su[i + 1] = T_target[i] + (Su[i] - T_target[i]) * np.exp(-dt_ann[i] / tau_u)
    Sd = np.empty(n_ann)
    Sd[0] = T_annual[0]
    for i in range(n_ann - 1):
        su_avg = 0.5 * (Su[i] + Su[i + 1])
        Sd[i + 1] = su_avg + (Sd[i] - su_avg) * np.exp(-dt_ann[i] / tau_d)
    s = Su[D['idx7']]
    H_upper = a * s**2 + b_u * s
    resid_700 = D['c7_dm'] - (H_upper - H_upper.mean())
    if not (-0.99 < rho < 0.99) or log_sigma_extra_2 > np.log(0.01):
        return -np.inf
    lp += log_sigma_extra_2
    H_mid = b_d * Sd[D['idx2']]
    resid_mid = D['cmid_dm'] - (H_mid - H_mid.mean())
    return (_ll_ar1(resid_700, D['sig7'], sigma_extra, rho, D['lag7'])
            + _ll_ar1(resid_mid, D['sigmid'], np.exp(log_sigma_extra_2), rho, D['lag2']) + lp)


def _ll_ar1_batch(R, sig, sx, rho, lag):
    """R (n, N); sx, rho (n,). Same Gaussian log-likelihood, batched."""
    C = np.diag(sig**2)[None] + sx[:, None, None]**2 * rho[:, None, None]**lag[None]
    L = np.linalg.cholesky(C)
    z = np.linalg.solve(L, R[..., None])[..., 0]
    return -0.5 * (np.sum(z**2, axis=1)
                   + 2 * np.sum(np.log(np.diagonal(L, axis1=1, axis2=2)), axis=1))


def make_logprob_vec(D):
    def lpv(TH):
        TH = np.atleast_2d(TH)
        n = TH.shape[0]
        b_u, b_d, ltu, lse, ltd, rho, lse2, m_v = TH.T
        ok = ((b_u >= 0) & (b_d >= 0) & (np.exp(ltu) > 0.1) & (np.exp(ltu) < 500)
              & (np.exp(ltd) > 10) & (np.exp(ltd) < 5000) & (m_v > 0)
              & (rho > -0.99) & (rho < 0.99) & (lse2 <= np.log(0.01)))
        out = np.full(n, -np.inf)
        if not ok.any():
            return out
        b_u, b_d, ltu, lse, ltd, rho, lse2, m_v = TH[ok].T
        tau_u, tau_d = np.exp(ltu), np.exp(ltd)
        a = KAPPA_EOS * b_u**2
        lp = -0.5 * ((ltu - np.log(8)) / 0.5)**2
        lp += -0.5 * ((ltd - TAU_D_LOG_MU) / TAU_D_LOG_SIGMA)**2
        lp += -0.5 * ((m_v - 1.0) / THERMO_VOLC_M_SD)**2
        m = ok.sum()
        Su = np.empty((n_ann, m))
        Su[0] = T_annual[0]
        kv = m_v * THERMO_VOLC_KV / b_u * tau_u
        for i in range(n_ann - 1):
            tgt = T_avg_ann[i] + kv * Fv_avg_ann[i]
            Su[i + 1] = tgt + (Su[i] - tgt) * np.exp(-dt_ann[i] / tau_u)
        Sd = np.empty((n_ann, m))
        Sd[0] = T_annual[0]
        for i in range(n_ann - 1):
            sa = 0.5 * (Su[i] + Su[i + 1])
            Sd[i + 1] = sa + (Sd[i] - sa) * np.exp(-dt_ann[i] / tau_d)
        s = Su[D['idx7']].T                       # (m, N7)
        H = a[:, None] * s**2 + b_u[:, None] * s
        r7 = D['c7_dm'][None] - (H - H.mean(1, keepdims=True))
        lp = lp + lse2
        Hm = b_d[:, None] * Sd[D['idx2']].T
        rm = D['cmid_dm'][None] - (Hm - Hm.mean(1, keepdims=True))
        out[ok] = (_ll_ar1_batch(r7, D['sig7'], np.exp(lse), rho, D['lag7'])
                   + _ll_ar1_batch(rm, D['sigmid'], np.exp(lse2), rho, D['lag2']) + lp)
        return out
    return lpv


def theta_init(D):
    Su_i, Sd_i = solve_twolayer_ode(T_annual, T_annual_years, 8.0, TAU_D_INIT)
    s = Su_i[D['idx7']]
    X = np.column_stack([s**2, s]); X = X - X.mean(0)
    b7 = np.linalg.lstsq(X, D['c7_dm'], rcond=None)[0]
    s = Su_i[D['idx2']]
    X = np.column_stack([s**2, s, Sd_i[D['idx2']]]); X = X - X.mean(0)
    b2 = np.linalg.lstsq(X, D['c2_dm'], rcond=None)[0]
    return np.array([max(b7[1], 1e-6), max(b2[2], 1e-6), np.log(8.0), np.log(0.001),
                     np.log(TAU_D_INIT), 0.3, np.log(0.001), 1.0])


def run_mcmc(D, n_burn, n_prod, n_walkers=64):
    lpv = make_logprob_vec(D)
    theta0 = theta_init(D)
    # equivalence check: vectorized vs scalar transcription
    rng_c = np.random.default_rng(7)
    TH = theta0 + np.array([5e-3, 2e-2, .3, 1., .3, .3, 1., .2]) * rng_c.standard_normal((40, 8))
    TH[:, 3] = np.log(0.002) + 0.3 * rng_c.standard_normal(40)
    TH[:, 6] = np.log(0.0005) + 0.3 * rng_c.standard_normal(40)
    TH[:, 5] = np.clip(TH[:, 5], -0.9, 0.9)
    TH[:, 7] = np.abs(TH[:, 7])
    v = lpv(TH)
    s = np.array([logprob_scalar(t, D) for t in TH])
    fin = np.isfinite(s)
    assert np.array_equal(fin, np.isfinite(v))
    maxdiff = float(np.max(np.abs(v[fin] - s[fin]))) if fin.any() else 0.0
    assert maxdiff < 1e-6, maxdiff
    rng = np.random.default_rng(42)
    ndim = len(theta0)
    pos0 = theta0 + 1e-4 * rng.standard_normal((n_walkers, ndim))
    pos0[:, :2] = np.abs(pos0[:, :2])
    np.random.seed(SEEDS['bayesian'])
    sampler = emcee.EnsembleSampler(n_walkers, ndim, lpv, vectorize=True)
    t0 = time.time()
    state = sampler.run_mcmc(pos0, n_burn)
    sampler.reset()
    sampler.run_mcmc(state, n_prod)
    flat = sampler.get_chain(flat=True)
    flat = np.column_stack([KAPPA_EOS * flat[:, 0]**2, flat])
    tau_int = sampler.get_autocorr_time(quiet=True)
    chain = sampler.get_chain()
    n, mw, _ = chain.shape
    W = chain.var(axis=0, ddof=1).mean(axis=0)
    B = n * chain.mean(axis=0).var(axis=0, ddof=1)
    r_hat = np.sqrt(((n - 1) / n * W + B / n) / W)
    diag = dict(acc=float(np.mean(sampler.acceptance_fraction)), tau_int=tau_int.tolist(),
                ess=(n_walkers * n_prod / tau_int).tolist(), r_hat=r_hat.tolist(),
                minutes=(time.time() - t0) / 60, vec_scalar_maxdiff=maxdiff,
                n_burn=n_burn, n_prod=n_prod, n_walkers=n_walkers)
    return flat, diag


def project(flat, D):
    """Cell 9, projection loop, same random streams."""
    from component_io import PROJ_YEARS, PROJ_SSPS, N_SAMPLES
    from warming_paths import warming_z, ssp_row_offsets, SAMPLE_WARMING_PATHS
    from ar6_gmst import ar6_to_berkeley_offset
    SSP_H5_MAP = {'SSP1-2.6': 'SSP1_2_6', 'SSP2-4.5': 'SSP2_4_5', 'SSP3-7.0': 'SSP3_7_0',
                  'SSP5-8.5': 'SSP5_8_5', 'SSP1-1.9': 'SSP1_1_9'}
    N_MC = N_SAMPLES
    WARMING_Z = warming_z(N_MC)
    T_OFFSET = ar6_to_berkeley_offset(H5_PATH)
    rng_proj = np.random.default_rng(42)
    idx_thin = rng_proj.choice(len(flat), size=N_MC, replace=False)
    mc_a = flat[idx_thin, 0]; mc_bu = flat[idx_thin, 1]; mc_bd = flat[idx_thin, 2]
    mc_tau_u = np.exp(flat[idx_thin, 3]); mc_tau_d = np.exp(flat[idx_thin, 5])
    mc_m_v = flat[idx_thin, 8]
    res = {}
    for ssp in PROJ_SSPS:
        df_ssp = pd.read_hdf(H5_PATH, f'projections/temp/{SSP_H5_MAP[ssp]}')
        ssp_years = df_ssp['decimal_year'].values + 0.5
        ssp_T = df_ssp['temperature'].values - T_OFFSET
        ssp_T_sigma = df_ssp['temperature_sigma'].values
        dT_paths = ssp_row_offsets(df_ssp, WARMING_Z)
        future_mask = ssp_years > T_annual_years[-1]
        ext_years = np.arange(ssp_years[-1] + 1, PROJ_YEARS[-1] + 1)
        T_proj = np.concatenate([T_annual, ssp_T[future_mask], np.full(len(ext_years), ssp_T[-1])])
        years_proj = np.concatenate([T_annual_years, ssp_years[future_mask], ext_years])
        dT_future = np.concatenate([dT_paths[:, future_mask],
                                    np.repeat(dT_paths[:, -1:], len(ext_years), axis=1)], axis=1)
        n_ssp_future = int(future_mask.sum())
        Fv_proj = read_volcanic_erf(VOLC_ERF_CSV, years_proj)
        full = np.zeros((N_MC, len(PROJ_YEARS)))
        upper = np.zeros_like(full)
        for k in range(N_MC):
            T_k = T_proj.copy()
            n_hist = len(T_annual)
            _T_noise = rng_proj.normal(0, ssp_T_sigma[future_mask])
            if SAMPLE_WARMING_PATHS:
                T_k[n_hist:] += dT_future[k]
            else:
                T_k[n_hist:n_hist + n_ssp_future] += _T_noise
            T_target_k = T_k + mc_m_v[k] * THERMO_VOLC_KV / mc_bu[k] * mc_tau_u[k] * Fv_proj
            Su_k, Sd_k = solve_twolayer_ode(T_target_k, years_proj, mc_tau_u[k], mc_tau_d[k])
            H_upper_k = mc_a[k] * Su_k**2 + mc_bu[k] * Su_k
            H_total_k = H_upper_k + mc_bd[k] * Sd_k
            rate_deep_k = RATE_BELOW_2000 + rng_proj.normal(0, SIGMA_BELOW_2000)
            H_full_k = H_total_k + rate_deep_k * (years_proj - BASELINE_YEAR)
            H_full_k -= np.interp(BASELINE_YEAR, years_proj, H_full_k)
            H_upper_k -= np.interp(BASELINE_YEAR, years_proj, H_upper_k)
            full[k] = np.interp(PROJ_YEARS, years_proj, H_full_k)
            upper[k] = np.interp(PROJ_YEARS, years_proj, H_upper_k)
        res[ssp] = (full, upper)
    return PROJ_YEARS, res


def summarize(flat, diag, py, res):
    q = lambda x: [float(np.median(x)), *map(float, np.percentile(x, [5, 95]))]
    P = dict(a_mm_C2=q(flat[:, 0] * 1e3), b_u_mm_C=q(flat[:, 1] * 1e3),
             b_d_mm_C=q(flat[:, 2] * 1e3), tau_u_yr=q(np.exp(flat[:, 3])),
             sigma_extra_mm=q(np.exp(flat[:, 4]) * 1e3), tau_d_yr=q(np.exp(flat[:, 5])),
             rho=q(flat[:, 6]), sigma_extra_2_mm=q(np.exp(flat[:, 7]) * 1e3), m_v=q(flat[:, 8]))
    i21 = int(np.where(py == 2100)[0][0])
    proj = {s: q(f[:, i21] * 1e3) for s, (f, u) in res.items()}
    f = res['SSP2-4.5'][0] * 1e3
    j = {y: int(np.where(py == y)[0][0]) for y in (1950, 1970, 1993, 2000, 2005, 2024, 2025)}
    rates = {'1950-2000': q((f[:, j[2000]] - f[:, j[1950]]) / 50),
             '1970-2000': q((f[:, j[2000]] - f[:, j[1970]]) / 30),
             '1993-2025': q((f[:, j[2025]] - f[:, j[1993]]) / 32),
             '2005-2024': q((f[:, j[2024]] - f[:, j[2005]]) / 19)}
    return dict(params=P, proj2100=proj, hindcast_rates_full_depth=rates, diag=diag)



def label(record, scale):
    return f'{record}_x{round(scale * 100):03d}'


def cmd_fit(a):
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    lab = label(a.record, a.scale)
    D = build_data(a.record, a.scale, 1955)
    print(f'{lab}: 0-700 m {D["y7"][0]}-{D["y7"][-1]} ({len(D["y7"])}), '
          f'700-2000 m {D["y2"][0]}-{D["y2"][-1]} ({len(D["y2"])})', flush=True)
    flat, diag = run_mcmc(D, a.n_burn, a.n_prod)
    print(f'{lab}: MCMC done in {diag["minutes"]:.1f} min', flush=True)
    py, res = project(flat, D)
    S = summarize(flat, diag, py, res)
    S['config'] = dict(record=a.record, scale=a.scale, n_burn=a.n_burn, n_prod=a.n_prod)
    S['data'] = dict(y7=[float(D['y7'][0]), float(D['y7'][-1])],
                     y2=[float(D['y2'][0]), float(D['y2'][-1])])
    (FIT_DIR / f'{lab}.json').write_text(json.dumps(S, indent=1))


def production():
    """Production posterior and 2100 projections from component_results.h5 (read-only)."""
    q = lambda x: [float(np.median(x)), *map(float, np.percentile(x, [5, 95]))]
    with h5py.File(H5_COMP, 'r') as f:
        pg = f['ocean_posteriors']
        P = dict(a_mm_C2=q(pg['a_posterior'][:] * 1e3), b_u_mm_C=q(pg['b_u_posterior'][:] * 1e3),
                 b_d_mm_C=q(pg['b_d_posterior'][:] * 1e3), tau_u_yr=q(pg['tau_u_posterior'][:]),
                 sigma_extra_mm=q(pg['sigma_extra_posterior'][:] * 1e3),
                 tau_d_yr=q(pg['tau_d_posterior'][:]), rho=q(pg['rho_posterior'][:]),
                 sigma_extra_2_mm=q(pg['sigma_extra_2_posterior'][:] * 1e3),
                 m_v=q(pg['m_v_posterior'][:]))
        py = f['proj_years'][:]
        i21 = int(np.where(py == 2100)[0][0])
        proj = {s: q(f['ocean/projections'][s]['samples'][:, i21] * 1e3)
                for s in f['ocean/projections']}
    return dict(params=P, proj2100=proj)


def record_series():
    """Unscaled annual series of the three records (mm), each rebased to its own
    1995-2005 (0-700 m) or 2005-2009 (0-2000 m) mean."""
    ny, n7, _, nyd, n2, _ = load_noaa()
    S = {'noaa': (pd.Series(n7, index=np.floor(ny).astype(int)),
                  pd.Series(n2, index=np.floor(nyd).astype(int))),
         'en4': (load_en4()['h700_mm'], load_en4()['h2000_mm']),
         'iap': (load_iap()['h700_mm'], load_iap()['h2000_mm'])}
    out = {}
    for r, (s7, s2) in S.items():
        s7 = s7 - s7.loc[1995:2005].mean()
        s2 = s2.loc[2003:] - s2.loc[2005:2009].mean()
        out[r] = {'years_700': s7.index.tolist(), 'h700_mm': s7.round(3).tolist(),
                  'years_2000': s2.index.tolist(), 'h2000_mm': s2.round(3).tolist()}
    return out, S


def cmd_collect(a):
    series, S = record_series()
    trends = {}
    for H, li in ((700, 0), (2000, 1)):
        for w in TREND_WINDOWS:
            if H == 2000 and w[0] < 2005:
                continue
            row = {}
            for r in S:
                s = S[r][li].loc[w[0]:w[1]].dropna()
                row[r] = _trend(s.index.values.astype(float), s.values)
            for r in ('en4', 'iap'):
                d = (S[r][li] - S['noaa'][li]).loc[w[0]:w[1]].dropna()
                row[f'{r}_minus_noaa'] = _trend(d.index.values.astype(float), d.values)
                row[f'{r}_over_noaa'] = row[r][0] / row['noaa'][0]
            trends[f'0-{H}m_{w[0]}-{w[1]}'] = row
    e, i = load_en4(), load_iap()
    ohc = {}
    for H in (700, 2000):
        for w in TREND_WINDOWS:
            d = (i[f'ohc{H}_zj'] - e[f'ohc{H}_zj']).loc[w[0]:w[1]].dropna()
            ohc[f'0-{H}m_{w[0]}-{w[1]}'] = {
                'en4': _trend(*_ti(e[f'ohc{H}_zj'], w)), 'iap': _trend(*_ti(i[f'ohc{H}_zj'], w)),
                'iap_minus_en4': _trend(d.index.values.astype(float), d.values)}
    fits = {}
    for rec, sc in FITS:
        p = FIT_DIR / f'{label(rec, sc)}.json'
        if p.exists():
            fits[label(rec, sc)] = json.loads(p.read_text())
    prod = production()
    if 'noaa_x114' in fits:
        rep = fits['noaa_x114']
        dev = {s: abs(rep['proj2100'][s][0] - prod['proj2100'][s][0]) for s in prod['proj2100']}
        print('reproduction of production, |2100 median difference| (mm):',
              {s: round(v, 1) for s, v in dev.items()})
        assert max(dev.values()) < 3.0, 'NOAA x1.14 refit does not reproduce production'
    # rise across the 2001-2003 arrival of Argo coverage, 0-700 m (mm)
    step = {r: float(S[r][0].loc[2003] - S[r][0].loc[2001]) for r in S}
    out = dict(production=prod, fits=fits, trends_mm_per_yr=trends, ohc_trends_zj_per_yr=ohc,
               step_2001_2003_700m_mm=step,
               mm_per_zj=mm_per_zj(), conv_window=list(CONV_WINDOW), series=series,
               note='90% intervals; trends are OLS with AR(1)-adjusted effective sample size')
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f'wrote {OUT_JSON}')


def _ti(s, w):
    s = s.loc[w[0]:w[1]].dropna()
    return s.index.values.astype(float), s.values


def cmd_all(a):
    procs = [subprocess.Popen([sys.executable, __file__, 'fit', '--record', r, '--scale', str(s),
                               '--n-burn', str(a.n_burn), '--n-prod', str(a.n_prod)])
             for r, s in FITS]
    if any(p.wait() for p in procs):
        raise SystemExit('a fit failed')
    cmd_collect(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('fit', 'all', 'collect'):
        s = sub.add_parser(name)
        s.add_argument('--n-burn', type=int, default=6000)
        s.add_argument('--n-prod', type=int, default=30000)
        if name == 'fit':
            s.add_argument('--record', choices=['noaa', 'en4', 'iap'], required=True)
            s.add_argument('--scale', type=float, default=STRUCTURAL_FRAC + 1)
    a = ap.parse_args()
    {'fit': cmd_fit, 'all': cmd_all, 'collect': cmd_collect}[a.cmd](a)


if __name__ == '__main__':
    main()
