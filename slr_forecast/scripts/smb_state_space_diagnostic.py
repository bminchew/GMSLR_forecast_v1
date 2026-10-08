"""Stage 1 diagnostic: a slowly drifting error state for the Greenland SMB emulator.

The MAR multi-GCM emulator (smb_emulator.py), driven by observed GMST, does
not reproduce the multidecadal drift in observed SMB (Mouginot et al. 2019:
about +80 Gt/yr above the emulator in 1972-85, about -107 Gt/yr below it in
2010-18). This script treats that drift as a latent state and asks what the
data say about it. Nothing here feeds the pipeline.

Model, for emulator member k (GCM and posterior draw of b1, b2), in Gt/yr:

    r_t = SMB_obs,t - f_k(x_t) = M + delta_t + eps_t
    delta_t = phi delta_{t-1} + eta_t,      phi = exp(-1/tau)

f_k(x) = b1 x + b2 x^2 is the member's emulator anomaly at the observed GMST
anomaly x (rel. 1995-2005, 11-yr centred mean), M a constant level (diffuse
prior), delta a stationary AR(1) state with sd sigma_d, and eps weather plus
the reported observation error: Var(eps_t) = sigma_e^2 + s_t^2. The state
[M, delta] is estimated with a Kalman filter.

Steps
  1. Hyperparameters (tau, sigma_d, sigma_e) on a grid, from the Kalman
     marginal likelihood of the residual about the median emulator (pooled
     over members), with priors: tau lognormal, 90% range 10-50 yr (and a
     wide log-uniform 3-150 yr for comparison); sigma_d half-normal, scale
     150 Gt/yr; sigma_e flat on 20-250 Gt/yr (weak, so the data set it).
  2. Each member draws (tau, sigma_d, sigma_e) from that grid posterior and
     is filtered through its own residual; (M, delta_2018) are drawn jointly
     from the filtered mean and covariance.
  3. Effect at 2100 relative to the production projection: after the 2018
     splice the production rate is M0_k + f_k + noise, with M0_k ~ N(M0, 57).
     With the state it is M_k + f_k + delta_t + noise, so the cumulative SMB
     sea-level contribution over 2019-2100 changes by
         -(82 (M_k - M0_k) + sum_{2019}^{2100} delta_t) / 362.5  mm,
     split below into its permanent (M) and decaying (delta) parts.
  4. Out of sample: fit on 1972-2004 (GMST smoothed through 2004 only),
     forecast 2005-2018, next to the production approach (constant M0 with
     sd 57 and the MAR AR(1) weather noise).
  5. Data checks: Mankoff et al. (2021) three-RCM mean SMB, 1986-2023 (it
     includes RACMO, so it is only a partial check), and the GRACE-minus-
     discharge implied SMB, 2003-2018 (biennial; the two rows per year, one
     per discharge record, are averaged; 2020 is dropped as a partial year).
     Each is compared with Mouginot on the same years.
  7. Forecast check after the record: from the filtered 2018 state, forecast
     2019-2023 SMB and compare with Mankoff (three-RCM mean) and IMBIE-3
     (Otosaka et al. 2026, SMB anomaly; its SMB partition also comes from
     RCMs), neither used in the fit. Each is put on Mouginot's level by its
     mean offset over 2010-2018 (and, as a check, over the full overlap).
     Compared with the production approach (M0 +/- 57 and MAR AR(1) noise).
  6. Option 2: add c x to the observation (c constant, prior N(0, s_c^2)),
     for s_c = 100 and 300 Gt/yr/K. c x and a slow delta are nearly
     collinear over a monotonic 47-yr warming, so both widths are reported.

Simplifications: hyperparameters are pooled on the median-emulator residual
and reused for every member; the elevation feedback is left out of f_k
(near zero before 2018) and its interaction with the state is ignored in
step 3; the weather noise is unchanged from production.

Usage:  python scripts/smb_state_space_diagnostic.py
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks')); sys.path.insert(0, str(ROOT / 'src'))
import smb_emulator as S
from slr_forecast.readers.ice_sheets import read_mouginot2019_greenland
from bayesian_models import prepare_mouginot_components
from smb_state import TAU, SD_E, log_prior, kalman, grid_posterior

N = 2000
GT_PER_MM = 362.5
SPLICE, END = 2018, 2100
M0_SIGMA = 57.0
OUT_FIG = ROOT / 'figures/diagnostics/smb_state_space.png'
SSPS = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']


def q(v, w=None, p=(50, 5, 95)):
    if w is None:
        return np.percentile(v, p)
    o = np.argsort(v); c = np.cumsum(w[o]); c /= c[-1]
    return np.interp(np.array(p) / 100, c, v[o])


def fmt(v, w=None, f='.0f'):
    m, lo, hi = q(v, w)
    return f'{m:{f}} [{lo:{f}}, {hi:{f}}]'


def main():
    rng = np.random.default_rng(610)

    # ── Observations ──
    mou = prepare_mouginot_components(
        read_mouginot2019_greenland(str(ROOT / 'data/raw/ice_sheets/greenland/mouginot2019_data.xlsx')),
        baseline_window=(1995, 2005))
    md = mou['df']
    yrs_m = md['decimal_year'].values.astype(int)
    smb_m = -md['smb_rate'].values / S.GT_TO_M_SLE
    sig_m = md['smb_rate_sigma'].values / S.GT_TO_M_SLE
    M0 = smb_m[(yrs_m >= 1995) & (yrs_m <= 2005)].mean()

    mk = pd.read_csv(ROOT / 'data/raw/ice_sheets/greenland/mankoff/MB_SMB_D_BMB_ann.csv')
    mk = mk[(mk.time >= 1986) & (mk.time <= 2023)]
    gd = pd.read_csv(ROOT / 'data/processed/greenland_implied_smb.csv')
    gd['yr'] = np.floor(gd['year']).astype(int)
    gd = gd[(gd.yr >= 2003) & (gd.yr <= 2018)].groupby('yr').agg(rate=('implied_smb_rate_slr', 'mean'),
                                             sig=('implied_smb_rate_sigma_slr', 'mean'))
    gd_smb = -gd['rate'] / S.GT_TO_M_SLE
    gd_sig = gd['sig'] / S.GT_TO_M_SLE

    # ── Driver: observed GMST rel. 1995-2005, calendar-year mean, 11-yr centred ──
    be = pd.read_hdf(ROOT / 'data/processed/slr_processed_data.h5', 'harmonized/df_berkeley_h')
    tm = be['temperature'].values
    tt = be.index.year + (be.index.month - 0.5) / 12.0
    g = pd.Series(tm).groupby(np.floor(tt).astype(int)).mean()
    obs_end = int(np.floor(tt[-1]))
    def x_on(years, through):
        yy = np.arange(g.index[0], through + 1)
        xs = S.smooth_observed(g.reindex(yy).values, yy, through)
        return pd.Series(xs, index=yy).reindex(years).values

    # ── Emulator members, as in production (seeds 600, 602) ──
    emu = S.fit_smb_emulator(N, seed=600)
    b1, b2 = emu['b1'], emu['b2']
    M0_draws = np.random.default_rng(602).normal(M0, M0_SIGMA, N)
    f = lambda x: b1[:, None] * x[None, :] + b2[:, None] * x[None, :] ** 2      # (N, T)

    # ── MAR reference for sigma_e: residual sd, full fit and history only ──
    mar, temp = S.load_data()
    sd_full, sd_hist = [], []
    for gcm, fit in emu['fits'].items():
        bm = np.median(fit['beta'], axis=0)
        segs = S.training_segments(gcm, mar, temp)
        res = [ys.values - (bm[0] + bm[1] * xs.loc[ys.index].values + bm[2] * xs.loc[ys.index].values ** 2)
               for _, ys, xs in segs]
        sd_full.append(np.std(np.concatenate(res))); sd_hist.append(np.std(res[0]))
    print(f'MAR residual sd about each GCM fit: full {np.min(sd_full):.0f}-{np.max(sd_full):.0f}, '
          f'history 1980-2014 only {np.min(sd_hist):.0f}-{np.max(sd_hist):.0f} Gt/yr')
    print(f'Mouginot reported SMB error {sig_m.mean():.0f} Gt/yr; M0 = {M0:.0f} Gt/yr')

    # ── 1. Hyperparameters from the median-emulator residual ──
    x_m = x_on(yrs_m, obs_end)
    fm = np.median(f(x_m), axis=0)
    r_med = smb_m - fm
    print(f'\nResidual about the median emulator: sd {r_med.std():.0f} Gt/yr, '
          f'lag-1 {np.corrcoef(r_med[:-1], r_med[1:])[0, 1]:.2f}, '
          f'1972-85 {r_med[yrs_m <= 1985].mean() - r_med[(yrs_m >= 1995) & (yrs_m <= 2005)].mean():+.0f}, '
          f'2010-18 {r_med[yrs_m >= 2010].mean() - r_med[(yrs_m >= 1995) & (yrs_m <= 2005)].mean():+.0f} '
          f'Gt/yr rel. 1995-2005')
    post = {}
    for prior in ('narrow', 'wide'):
        gp = grid_posterior(r_med, sig_m, prior)
        post[prior] = gp
        w = gp['w']
        # likelihood ratio against no state (sigma_d = 0), best sigma_e each
        ll0 = gp['ll'][gp['sd_d'] == 0].max(); ll1 = gp['ll'].max()
        print(f'\nGrid posterior, tau prior {prior}:')
        print(f'  tau {fmt(gp["tau"], w, ".1f")} yr; sigma_d {fmt(gp["sd_d"], w)} Gt/yr; '
              f'sigma_e {fmt(gp["sd_e"], w)} Gt/yr')
        print(f'  max log-likelihood with state {ll1:.1f}, without (sigma_d = 0) {ll0:.1f}; '
              f'P(sigma_d < 20) = {w[gp["sd_d"] < 20].sum():.3f}')

    # ── 2. Per-member filtering (narrow prior) ──
    gp = post['narrow']
    pick = rng.choice(len(gp['w']), size=N, p=gp['w'])
    th = {k: gp[k][pick] for k in ('tau', 'sd_d', 'sd_e')}
    phi = np.exp(-1 / th['tau'])
    r_k = smb_m[None, :] - f(x_m)
    _, a, P = kalman(r_k, sig_m, phi, th['sd_d'], th['sd_e'])
    L = np.linalg.cholesky(P + 1e-9 * np.eye(2))
    draw = a + np.einsum('bij,bj->bi', L, rng.standard_normal((N, 2)))
    M_k, d18 = draw[:, 0], draw[:, 1]
    corr = P[:, 0, 1] / np.sqrt(P[:, 0, 0] * P[:, 1, 1])
    print('\nFiltered state at 2018 (members, narrow prior):')
    print(f'  M        {fmt(a[:, 0])} Gt/yr (filtered means); drawn {fmt(M_k)}; '
          f'filtered sd median {np.median(np.sqrt(P[:, 0, 0])):.0f}')
    print(f'  delta_18 {fmt(a[:, 1])} Gt/yr (filtered means); drawn {fmt(d18)}; '
          f'filtered sd median {np.median(np.sqrt(P[:, 1, 1])):.0f}')
    print(f'  corr(M, delta_18) median {np.median(corr):.2f}; '
          f'M + delta_18 {fmt(M_k + d18)} Gt/yr; production M0 {fmt(M0_draws)} Gt/yr')

    # ── 3. Effect at 2100 ──
    n_f = END - SPLICE
    d_path = np.empty((N, n_f)); d = d18.copy()
    q_sd = th['sd_d'] * np.sqrt(1 - phi ** 2)
    for j in range(n_f):
        d = phi * d + q_sd * rng.standard_normal(N)
        d_path[:, j] = d
    perm = -n_f * (M_k - M0_draws) / GT_PER_MM          # mm SLE, + = more SLR
    decay = -d_path.sum(axis=1) / GT_PER_MM
    dmm = perm + decay
    print(f'\nChange in cumulative SMB contribution 2019-2100 vs production (mm SLE, + = more SLR):')
    print(f'  total {fmt(dmm, f=".1f")}; permanent (M - M0) {fmt(perm, f=".1f")}; '
          f'decaying (delta) {fmt(decay, f=".1f")}')
    # sign check by hand for one member
    k = 0
    print(f'  check member 0: M {M_k[k]:.0f}, M0 {M0_draws[k]:.0f}, sum delta {d_path[k].sum():.0f} Gt -> '
          f'{-(n_f * (M_k[k] - M0_draws[k]) + d_path[k].sum()) / GT_PER_MM:.2f} mm (= {dmm[k]:.2f})')
    i2100 = END - 1950
    prod, new = {}, {}
    with h5py.File(ROOT / 'data/processed/component_results.h5', 'r') as h:
        for ssp in SSPS:
            s = h[f'greenland/projections_smb/{ssp}/samples'][:, i2100] * 1000.0
            prod[ssp], new[ssp] = s, s + dmm
    print('  SMB contribution at 2100, production -> with state (mm SLE, median [5, 95]):')
    for ssp in SSPS:
        print(f'    {ssp}: {fmt(prod[ssp])} -> {fmt(new[ssp])}  '
              f'(5-95 width {np.ptp(q(prod[ssp], p=(5, 95))):.0f} -> {np.ptp(q(new[ssp], p=(5, 95))):.0f})')

    # ── 4. Out of sample: fit 1972-2004, forecast 2005-2018 ──
    HO = 2004
    fit_m = yrs_m <= HO
    x_fit = x_on(yrs_m[fit_m], HO)
    x_fc = x_on(yrs_m, obs_end)
    gph = grid_posterior(smb_m[fit_m] - np.median(f(x_fit), axis=0), sig_m[fit_m], 'narrow')
    pk = rng.choice(len(gph['w']), size=N, p=gph['w'])
    tau_h, sdd_h, sde_h = gph['tau'][pk], gph['sd_d'][pk], gph['sd_e'][pk]
    phi_h = np.exp(-1 / tau_h)
    _, ah, Ph = kalman(smb_m[None, fit_m] - f(x_fit), sig_m[fit_m], phi_h, sdd_h, sde_h)
    Lh = np.linalg.cholesky(Ph + 1e-9 * np.eye(2))
    dr = ah + np.einsum('bij,bj->bi', Lh, rng.standard_normal((N, 2)))
    print(f'  holdout filtered state at {HO}: M {fmt(ah[:, 0])}, delta {fmt(ah[:, 1])} Gt/yr (filtered means); '
          f'M0 (1995-{HO}) {smb_m[(yrs_m >= 1995) & (yrs_m <= HO)].mean():.0f}')
    fc_yrs = yrs_m[~fit_m]
    fk = f(x_fc[~fit_m])
    d = dr[:, 1].copy(); st = np.empty_like(fk)
    for j in range(len(fc_yrs)):
        d = phi_h * d + sdd_h * np.sqrt(1 - phi_h ** 2) * rng.standard_normal(N)
        st[:, j] = dr[:, 0] + fk[:, j] + d + sde_h * rng.standard_normal(N)
    M0_h = smb_m[(yrs_m >= 1995) & (yrs_m <= HO)].mean()
    noise = S.ar1_noise(emu['sigma'], emu['rho'], len(fc_yrs), np.random.default_rng(611))
    pr = np.random.default_rng(612).normal(M0_h, M0_SIGMA, N)[:, None] + fk + noise
    obs_fc = smb_m[~fit_m]
    cum = lambda s: s.sum(axis=-1) / GT_PER_MM            # mm of mass gain over 2005-2018
    print(f'\nOut of sample 2005-2018 (fit on 1972-{HO}): tau {fmt(gph["tau"], gph["w"], ".1f")} yr, '
          f'sigma_d {fmt(gph["sd_d"], gph["w"])}, sigma_e {fmt(gph["sd_e"], gph["w"])} Gt/yr')
    print(f'  mean SMB: observed {obs_fc.mean():.0f}; state {fmt(st.mean(1))}; production {fmt(pr.mean(1))} Gt/yr')
    print(f'  cumulative SMB 2005-2018 (mm SLE of mass gain): observed {cum(obs_fc):.1f}; '
          f'state {fmt(cum(st), f=".1f")}; production {fmt(cum(pr), f=".1f")}')
    for lab, s in (('state', st), ('production', pr)):
        lo, hi = np.percentile(s, [5, 95], axis=0)
        print(f'  annual coverage of 90% band, {lab}: {np.mean((obs_fc >= lo) & (obs_fc <= hi)):.0%}; '
              f'percentile of observed cumulative {np.mean(cum(s) < cum(obs_fc)):.0%}')

    # ── 5. Data checks ──
    print('\nData checks (median-emulator residual, narrow prior):')
    yk = mk['time'].values.astype(int)
    r_mk = mk['SMB'].values - np.median(f(x_on(yk, obs_end)), axis=0)
    gk = grid_posterior(r_mk, mk['SMB_err'].values, 'narrow')
    print(f'  Mankoff 3-RCM 1986-2023: tau {fmt(gk["tau"], gk["w"], ".1f")} yr, '
          f'sigma_d {fmt(gk["sd_d"], gk["w"])}, sigma_e {fmt(gk["sd_e"], gk["w"])} Gt/yr; '
          f'P(sigma_d < 20) = {gk["w"][gk["sd_d"] < 20].sum():.3f}; max log-lik with/without state '
          f'{gk["ll"].max():.1f} / {gk["ll"][gk["sd_d"] == 0].max():.1f}')
    yg = gd.index.values
    r_gd = gd_smb.values - np.median(f(x_on(yg, obs_end)), axis=0)
    mou_s = pd.Series(smb_m, index=yrs_m)
    for lab, ser in (('Mankoff 3-RCM', pd.Series(mk['SMB'].values, index=yk)),
                     ('GRACE - D', gd_smb)):
        yy = ser.index.intersection(mou_s.index)
        dif = ser.loc[yy] - mou_s.loc[yy]
        per = lambda a_, b_: dif[(yy >= a_) & (yy <= b_)].mean()
        print(f'  {lab} minus Mouginot, same years: 1986-2002 {per(1986, 2002):+4.0f}, '
              f'2003-2009 {per(2003, 2009):+4.0f}, 2010-2018 {per(2010, 2018):+4.0f} Gt/yr '
              f'({len(yy)} years)')

    # ── 6. Option 2: add c x ──
    print('\nOption 2, state [M, c, delta] (median-emulator residual, narrow tau prior):')
    for s_c in (100.0, 300.0):
        go = grid_posterior(r_med, sig_m, 'narrow', H=x_m, prior_c=s_c, sd_e=SD_E[::2])
        w = go['w']
        c_m = go['a'][:, 1]; c_s = np.sqrt(go['P'][:, 1, 1])
        cdraw = c_m + c_s * rng.standard_normal(len(w))
        cs = cdraw[rng.choice(len(w), 4000, p=w)]
        dl = go['a'][:, 2][rng.choice(len(w), 4000, p=w)]
        print(f'  s_c = {s_c:.0f}: c {fmt(cs)} Gt/yr/K; delta_2018 filtered mean {fmt(dl)} Gt/yr '
              f'(without c: {fmt(a[:, 1])}); tau {fmt(go["tau"], w, ".1f")} yr; '
              f'c x at +1/+2/+3 K: {np.median(cs):.0f}/{2 * np.median(cs):.0f}/{3 * np.median(cs):.0f} Gt/yr')

    # ── 7. Forecast 2019-2023 from the 2018 state ──
    fy = np.arange(2019, 2024)
    xf = x_on(fy, obs_end)
    ff = f(xf)
    d = d18.copy(); st_f = np.empty_like(ff)
    for j in range(len(fy)):
        d = phi * d + th['sd_d'] * np.sqrt(1 - phi ** 2) * rng.standard_normal(N)
        st_f[:, j] = M_k + ff[:, j] + d + th['sd_e'] * rng.standard_normal(N)
    nz = S.ar1_noise(emu['sigma'], emu['rho'], len(fy), np.random.default_rng(613))
    pr_f = M0_draws[:, None] + ff + nz
    im = pd.read_csv(ROOT / 'data/raw/ice_sheets/imbie2026/imbie3_greenland_mm_partitioned.csv',
                     comment='#', parse_dates=['Date'])
    im = im.groupby(im['Date'].dt.year)['Surface mass balance anomaly (mm/yr)'].mean() * GT_PER_MM
    mou_s = pd.Series(smb_m, index=yrs_m)
    def crps(ens, y):
        e = np.sort(ens); n_ = len(e)
        return np.mean(np.abs(e - y)) - np.sum((2 * np.arange(1, n_ + 1) - n_ - 1) * e) / n_ ** 2
    print('\nForecast 2019-2023 from the 2018 state (5-yr mean SMB, Gt/yr):')
    print(f'  state model {fmt(st_f.mean(1))}; production {fmt(pr_f.mean(1))}; '
          f'emulator forced response (median) {np.median((M0_draws[:, None] + ff).mean(1)):.0f}')
    for lab, ser in (('Mankoff 3-RCM', pd.Series(mk['SMB'].values, index=yk)), ('IMBIE-3', im)):
        for win in ((2010, 2018), (1986, 2018)):
            yy = ser.index.intersection(mou_s.index)
            yy = yy[(yy >= win[0]) & (yy <= win[1])]
            off = (ser.loc[yy] - mou_s.loc[yy]).mean()
            obs5 = ser.loc[2019:2023].mean() - off
            row = f'  {lab:13s} aligned {win[0]}-{win[1]} (offset {off:+4.0f}): observed {obs5:4.0f};'
            for nm, ens in (('state', st_f.mean(1)), ('production', pr_f.mean(1))):
                row += (f'  {nm} pct {np.mean(ens < obs5):4.0%}, |median err| '
                        f'{abs(np.median(ens) - obs5):3.0f}, CRPS {crps(ens, obs5):3.0f}')
            print(row)
        print(f'  {lab:13s} annual 2019-2023: '
              + ', '.join(f'{ser.loc[y]:.0f}' for y in fy) + ' (unaligned)')

    # ── Figure ──
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    tbest = post['narrow']
    ib = np.argmax(tbest['w'])
    _, _, _, fmn, fsd, smn, ssd = kalman(r_med, sig_m, np.exp(-1 / tbest['tau'][ib]),
                                         tbest['sd_d'][ib], tbest['sd_e'][ib], return_path=True)
    Mb = kalman(r_med, sig_m, np.exp(-1 / tbest['tau'][ib]), tbest['sd_d'][ib], tbest['sd_e'][ib])[1][0, 0]
    ax.plot(yrs_m, r_med - Mb, 'o', color='#E69F00', ms=4, label='Mouginot residual - M')
    ax.plot(yk, r_mk - Mb, 's', color='#0072B2', ms=3, alpha=0.6, label='Mankoff 3-RCM residual - M')
    ax.plot(yg, r_gd - Mb, '^', color='#009E73', ms=5, alpha=0.8, label='GRACE - D residual - M')
    ax.plot(yrs_m, smn[0], 'k-', lw=2, label=r'smoothed $\delta$, 90% band (posterior-mode $\theta$ only)')
    ax.fill_between(yrs_m, smn[0] - 1.645 * ssd[0], smn[0] + 1.645 * ssd[0], color='k', alpha=0.15)
    ax.axhline(0, color='0.5', lw=0.5)
    ax.set_ylabel('Gt/yr'); ax.set_title(f'(a) Drift state, tau = {tbest["tau"][ib]:.0f} yr, '
                                          f'sigma_d = {tbest["sd_d"][ib]:.0f}, sigma_e = {tbest["sd_e"][ib]:.0f}',
                                          fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    ax = axes[0, 1]
    for lab, s, col in (('state model', st, '#CC79A7'), ('production (M0 +/- 57)', pr, '0.4')):
        c_ = np.cumsum(s, axis=1) / GT_PER_MM
        lo, md_, hi = np.percentile(c_, [5, 50, 95], axis=0)
        ax.plot(fc_yrs, md_, color=col, lw=2, label=lab)
        ax.fill_between(fc_yrs, lo, hi, color=col, alpha=0.2)
    ax.plot(fc_yrs, np.cumsum(obs_fc) / GT_PER_MM, 'o', color='#E69F00', label='Mouginot (withheld)')
    ax.set_ylabel('Cumulative SMB mass gain since 2004 (mm SLE)')
    ax.set_title('(b) Out of sample: fit 1972-2004, forecast 2005-2018', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    ax = axes[1, 0]
    for prior, col in (('narrow', 'k'), ('wide', '#0072B2')):
        gq = post[prior]
        mt = pd.Series(gq['w']).groupby(gq['tau']).sum()
        ax.plot(mt.index, mt.values / np.gradient(np.log(mt.index)), color=col, lw=2,
                label=f'posterior, {prior} prior')
        lp = np.exp(log_prior(TAU, np.zeros_like(TAU), prior))
        ax.plot(TAU, lp / np.trapezoid(lp, np.log(TAU)) * 1.0, color=col, ls=':', lw=1.2,
                label=f'prior, {prior}')
    ax.set_xscale('log'); ax.set_xlabel('tau (yr)'); ax.set_ylabel('density in log tau')
    ax.set_title('(c) Timescale of the drift', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    ax = axes[1, 1]
    for i, ssp in enumerate(SSPS):
        for j, (lab, s, col) in enumerate((('production', prod[ssp], '0.4'), ('with state', new[ssp], '#CC79A7'))):
            lo, m, hi = np.percentile(s, [5, 50, 95])
            xp = i + (j - 0.5) * 0.25
            ax.plot([xp, xp], [lo, hi], color=col, lw=6, alpha=0.5, solid_capstyle='butt',
                    label=lab if i == 0 else None)
            ax.plot(xp, m, 'o', color=col)
    ax.set_xticks(range(len(SSPS))); ax.set_xticklabels(SSPS)
    ax.set_ylabel('SMB contribution 2000-2100 (mm SLE)')
    ax.set_title('(d) 2100 SMB, median and 5-95%', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.2, axis='y')
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
    print(f'\nSaved {OUT_FIG.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
