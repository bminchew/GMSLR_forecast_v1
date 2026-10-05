"""Four-panel figure explaining the GMST-driven SMB emulator diagnostics.

(a) Greenland vs global warming, 1970-2024: observations and the four GCMs
    (historical + SSP2-4.5), each with its OLS slope (the warming ratio).
(b) SMB 1972-2024: observations, and MAR driven by each GCM (11-yr means),
    i.e. what the full GCM -> MAR chain itself does.
(c) The same MAR runs vs their fitted emulator (quadratic in the GCM's own
    GMST): does the fitted curve reproduce the chain's own recent decline?
(d) The emulator driven by observed GMST (r = 1) vs observed SMB.

Usage:  python scripts/plot_smb_frame_diagnostics.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
for p in ('scripts', 'notebooks', 'src'):
    sys.path.insert(0, str(ROOT / p))
import test_scenario_transfer_mar as T
import cross_gcm_test_mar as C
import smb_emulator as S
from slr_data_readers import read_berkeley_earth_gridded
from slr_forecast.readers.ice_sheets import read_mouginot2019_greenland
from bayesian_models import prepare_mouginot_components

GCMS = ['CESM2', 'MPI-ESM1-2-HR', 'NorESM2-MM', 'UKESM1-0-LL']
COL = dict(zip(GCMS, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']))


def sm(s):
    return s.rolling(11, center=True, min_periods=6).mean()


def main():
    rng = np.random.default_rng(7)
    mar = pd.read_csv(T.MAR).drop_duplicates(['run', 'year']); temp = pd.read_csv(T.TEMP)
    be = pd.read_hdf(ROOT / 'data/processed/slr_processed_data.h5', 'raw/df_berkeley')
    G = be['temperature'].groupby(be.index.year).mean(); G = G - G.loc[1995:2005].mean()
    gr = read_berkeley_earth_gridded(str(ROOT / 'data/raw/gmst/berkEarth_Global_TAVG_Gridded_1deg.nc'))
    Gr = gr['temperature'].groupby(gr.index.year).mean(); Gr = Gr - Gr.loc[1995:2005].mean()
    mou = prepare_mouginot_components(
        read_mouginot2019_greenland(str(ROOT / 'data/raw/ice_sheets/greenland/mouginot2019_data.xlsx')),
        baseline_window=(1995, 2005))
    obs = pd.Series(-mou['df']['smb_rate'].values / S.GT_TO_M_SLE, index=mou['df']['decimal_year'].values.astype(int))
    M0 = obs.loc[1995:2005].mean()
    mank = pd.read_csv(ROOT / 'data/raw/ice_sheets/greenland/mankoff/MB_SMB_D_BMB_ann.csv').set_index('time')['SMB']
    mank = mank.loc[1986:2023]; mank = mank - mank.loc[1995:2005].mean()

    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    yrs = np.arange(1970, 2025)

    # (a) warming ratio
    a = ax[0, 0]
    s = np.polyfit(G.loc[yrs], Gr.loc[yrs], 1)
    a.scatter(G.loc[yrs], Gr.loc[yrs], s=14, color='k', alpha=0.6)
    xx = np.linspace(-0.8, 1.0, 10)
    a.plot(xx, np.polyval(s, xx), 'k', lw=2.5, label=f'observed (Berkeley Earth): slope {s[0]:.2f}')
    for g in GCMS:
        t = temp[(temp.gcm == g) & temp.experiment.isin(['historical', 'ssp245'])].drop_duplicates('year').set_index('year').sort_index()
        gm = t.gmst_K - t.gmst_K.loc[1995:2005].mean(); tg = t.tgris_K - t.tgris_K.loc[1995:2005].mean()
        sg = np.polyfit(gm.loc[yrs], tg.loc[yrs], 1)
        a.plot(xx, np.polyval(sg, xx), color=COL[g], lw=2, label=f'{g}: slope {sg[0]:.2f}')
    a.set_xlabel('Global mean temperature anomaly (K)'); a.set_ylabel('Greenland temperature anomaly (K)')
    a.set_title('(a) Greenland vs global warming, 1970–2024\nModels and observations warm Greenland at similar rates')
    a.legend(fontsize=9, loc='upper left'); a.grid(True, alpha=0.2)

    # fits
    fits = {}
    for g in GCMS:
        segs = C.segments(mar, temp, g); st = np.cumsum([0] + [len(x[1]) for x in segs[:-1]])
        y = np.concatenate([x[1].values for x in segs])
        b, *_ = C.posterior_draws(C.design(segs, 1), y, st, 2000, rng)
        fits[g] = np.median(b, 0)

    # (b) chain itself vs observations
    a = ax[0, 1]
    a.plot(obs.index, obs - M0, color='0.75', lw=0.8)
    a.plot(obs.index, sm(obs - M0), 'k', lw=3, label='observed SMB (Mouginot), 11-yr mean')
    a.plot(mank.index, sm(mank), 'k', lw=1.5, ls='--', label='observed SMB (Mankoff 3-RCM), 11-yr mean')
    for g in GCMS:
        ssp = 'ssp245'
        ser = T.smb_series(mar, C.GDIR[g], ssp).loc[1980:2024]
        a.plot(ser.index, sm(ser), color=COL[g], lw=2, label=f'MAR driven by {g}')
    a.axhline(0, color='0.6', lw=0.6); a.set_xlim(1972, 2024)
    a.set_ylabel('SMB anomaly rel. 1995–2005 (Gt yr$^{-1}$)')
    a.set_title('(b) The full model chain (GCM → MAR) vs observations\nEach GCM has its own weather, so only the trend is comparable')
    a.legend(fontsize=8, loc='lower left'); a.grid(True, alpha=0.2)

    # (c) chain vs its own emulator
    a = ax[1, 0]
    for g in GCMS:
        ser = T.smb_series(mar, C.GDIR[g], 'ssp245').loc[1980:2024]
        xg = T.drivers(temp, g, 'ssp245')['GMST'].loc[ser.index]
        M, b1, b2 = fits[g]
        a.plot(ser.index, sm(ser), color=COL[g], lw=2)
        a.plot(ser.index, M + b1 * xg + b2 * xg**2, color=COL[g], lw=2, ls='--')
    a.plot([], [], 'k', lw=2, label='MAR run, 11-yr mean'); a.plot([], [], 'k--', lw=2, label='fitted emulator curve (same GCM)')
    a.axhline(0, color='0.6', lw=0.6); a.set_xlim(1980, 2024)
    a.set_ylabel('SMB anomaly rel. 1995–2005 (Gt yr$^{-1}$)')
    a.set_title('(c) Each MAR run (solid) vs its fitted emulator curve (dashed), 1980–2024')
    a.legend(fontsize=9, loc='lower left'); a.grid(True, alpha=0.2)

    # (d) emulator driven by observed GMST vs observations
    a = ax[1, 1]
    xo = sm(G).loc[1960:2024]
    a.plot(obs.index, sm(obs - M0), 'k', lw=3, label='observed SMB (Mouginot), 11-yr mean')
    a.plot(mank.index, sm(mank), 'k', lw=1.5, ls='--', label='observed SMB (Mankoff 3-RCM), 11-yr mean')
    for g in GCMS:
        M, b1, b2 = fits[g]
        a.plot(xo.index, b1 * xo + b2 * xo**2, color=COL[g], lw=2, label=f'emulator of MAR–{g}, observed GMST')
    a.axhline(0, color='0.6', lw=0.6); a.set_xlim(1972, 2024)
    a.set_ylabel('SMB anomaly rel. 1995–2005 (Gt yr$^{-1}$)')
    a.set_title('(d) Emulators driven by observed global temperature (no scaling)\nThey decline far less than observed')
    a.legend(fontsize=8, loc='lower left'); a.grid(True, alpha=0.2)
    for aa in ax.flat[1:]:
        aa.set_ylim(-250, 150)
    fig.tight_layout()
    out = ROOT / 'figures/preview_smb_frame_diagnostics.png'
    fig.savefig(out, dpi=130, bbox_inches='tight'); print(f'saved {out}')


if __name__ == '__main__':
    main()
