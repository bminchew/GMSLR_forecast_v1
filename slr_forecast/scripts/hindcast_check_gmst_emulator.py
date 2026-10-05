"""Hindcast check of the GMST-driven SMB statistical emulator against observed SMB.

The four tier-1 MAR emulators (CESM2, MPI-ESM1-2-HR, NorESM2-MM, UKESM1-0-LL;
quadratic in GMST, fitted on all scenarios, no lag) are driven by observed
Berkeley Earth GMST (anomaly rel. 1995-2005, 11-yr centred mean, as in the
fits) with SMB = M0 + b1 x + b2 x^2, M0 = Mouginot 1995-2005 mean SMB.
Their equal-weight mixture is compared with:
  - observed SMB: Mouginot et al. (2019) 1972-2018 and the Mankoff 3-RCM
    mean (RCM output from 1986; the earlier part is the adjusted Kjeldsen
    reconstruction, a loose check only);
  - the emulator now in the pipeline (Greenland T = AA x GMST, RACMO + MAR
    forced by CESM2, AA ~ N(2.5, 0.5));
  - the earlier parametric model (C_T = -300, C_T2 = -50, zeroed below baseline).
None of the emulators sees the observed SMB.

Usage:  python scripts/hindcast_check_gmst_emulator.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'notebooks')); sys.path.insert(0, str(ROOT / 'src'))
import test_scenario_transfer_mar as T
import cross_gcm_test_mar as C
import smb_emulator as S
from slr_forecast.readers.ice_sheets import read_mouginot2019_greenland
from bayesian_models import prepare_mouginot_components

GCMS = ['CESM2', 'MPI-ESM1-2-HR', 'NorESM2-MM', 'UKESM1-0-LL']
N = 2000
GT_PER_MM = 362.5


def main():
    rng = np.random.default_rng(4)
    mar = pd.read_csv(T.MAR).drop_duplicates(['run', 'year']); temp = pd.read_csv(T.TEMP)

    # observed GMST rel. 1995-2005, annual, 11-yr centred mean
    be = pd.read_hdf(ROOT / 'data/processed/slr_processed_data.h5', 'raw/df_berkeley')
    g = be['temperature'].groupby(be.index.year).mean()
    g = g - g.loc[1995:2005].mean()
    xs = g.rolling(11, center=True, min_periods=6).mean()
    years = np.arange(1900, 2025)
    x = xs.loc[years].values

    mou = prepare_mouginot_components(
        read_mouginot2019_greenland(str(ROOT / 'data/raw/ice_sheets/greenland/mouginot2019_data.xlsx')),
        baseline_window=(1995, 2005))
    md = mou['df']
    obs = pd.Series(-md['smb_rate'].values / S.GT_TO_M_SLE, index=md['decimal_year'].values.astype(int))
    M0 = obs.loc[1995:2005].mean()
    mank = pd.read_csv(ROOT / 'data/raw/ice_sheets/greenland/mankoff/MB_SMB_D_BMB_ann.csv').set_index('time')['SMB']

    # GMST emulators (mixture)
    rates = {}
    for gcm in GCMS:
        segs = C.segments(mar, temp, gcm)
        starts = np.cumsum([0] + [len(s[1]) for s in segs[:-1]])
        y = np.concatenate([s[1].values for s in segs])
        b, *_ = C.posterior_draws(C.design(segs, 1), y, starts, N // len(GCMS), rng)
        rates[gcm] = M0 + b[:, [1]] * x + b[:, [2]] * x**2
    mix = np.vstack(list(rates.values()))

    # pipeline emulator (Greenland T = AA x GMST), observed GMST without smoothing as in the notebook
    emu = S.fit_smb_emulator(N, seed=600)
    aa = np.random.default_rng(700).normal(2.5, 0.5, N)
    xr = aa[:, None] * g.loc[years].values[None, :]
    cur = M0 + emu['b1'][:, None] * xr + emu['b2'][:, None] * xr**2
    # earlier parametric model
    dTp = np.clip(g.loc[years].values, 0, None)[None, :]
    ct = np.random.default_rng(600).normal(-300, 80, N); ct2 = np.random.default_rng(601).normal(-50, 30, N)
    old = 380 + ct[:, None] * dTp + ct2[:, None] * dTp**2

    def cum(rate, y0=2000):
        c = -np.cumsum(rate, axis=-1) / GT_PER_MM
        return c - c[..., [list(years).index(y0)]]

    variants = [('GMST emulator, 4 GCMs (new)', mix, '#2a78d6'),
                ('pipeline emulator (AA x GMST)', cur, '#eb6834'),
                ('earlier parametric (-300, -50)', old, '#5f5e5a')]
    print(f'M0 = {M0:.0f} Gt/yr.  Mean SMB (Gt/yr), median of each ensemble:')
    windows = [(1900, 1949), (1950, 1971), (1972, 1985), (1986, 1999), (2000, 2009), (2010, 2018), (2019, 2024)]
    hdr = f'  {"":34s}' + ''.join(f'{a}-{str(b)[2:]:>2s}'.rjust(10) for a, b in windows)
    print(hdr)
    def row(lab, s):
        return f'  {lab:34s}' + ''.join(f'{s.loc[a:b].mean():10.0f}' if s.loc[a:b].size else ' ' * 10 for a, b in windows)
    print(row('Mouginot (RACMO, obs)', obs))
    print(row('Mankoff 3-RCM (pre-1986 = recon)', mank))
    for lab, r, _ in variants:
        print(row(lab, pd.Series(np.median(r, 0), index=years)))
    for gcm in GCMS:
        print(row(f'   {gcm} alone', pd.Series(np.median(rates[gcm], 0), index=years)))

    print('\nCumulative SMB sea-level contribution rel. 2000 (mm SLE), median [5th, 95th]:')
    oc = -obs.cumsum() / GT_PER_MM; oc = oc - oc.loc[2000]
    mkc = -mank.loc[1986:2024].cumsum() / GT_PER_MM; mkc = mkc - mkc.loc[2000]
    for y in (1972, 2018, 2024):
        line = f'  {y}: Mouginot {oc.get(y, np.nan):+6.1f}  Mankoff {mkc.get(y, np.nan):+6.1f}'
        for lab, r, _ in variants:
            c = cum(r)[:, list(years).index(y)]
            line += f' | {lab.split(",")[0].split(" (")[0]}: {np.median(c):+6.1f} [{np.percentile(c, 5):+.1f}, {np.percentile(c, 95):+.1f}]'
        print(line)

    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))
    ax = axes[0]
    for lab, r, c in variants:
        lo, md_, hi = np.percentile(r, [5, 50, 95], 0)
        ax.fill_between(years, lo, hi, color=c, alpha=0.15, lw=0); ax.plot(years, md_, color=c, lw=2, label=lab)
    ax.plot(obs.index, obs.values, 'k.-', lw=1, ms=4, label='Mouginot (RACMO, reanalysis)')
    ax.plot(mank.loc[1900:2024].index, mank.loc[1900:2024].values, color='0.55', lw=1,
            label='Mankoff (pre-1986: Kjeldsen reconstruction)')
    ax.set_xlim(1900, 2024); ax.set_ylabel('SMB (Gt yr$^{-1}$)'); ax.set_title('Annual SMB driven by observed GMST')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    ax = axes[1]
    sel = years >= 1960
    for lab, r, c in variants:
        cc = cum(r)
        lo, md_, hi = np.percentile(cc, [5, 50, 95], 0)
        ax.fill_between(years[sel], lo[sel], hi[sel], color=c, alpha=0.15, lw=0)
        ax.plot(years[sel], md_[sel], color=c, lw=2, label=lab)
    ax.plot(oc.index, oc.values, 'k.-', lw=1, ms=4, label='Mouginot')
    ax.plot(mkc.index, mkc.values, color='0.55', lw=1, label='Mankoff 3-RCM')
    ax.axhline(0, color='0.6', lw=0.6)
    ax.set_ylabel('Cumulative SMB, mm SLE rel. 2000'); ax.set_title('Cumulative (not spliced to observations)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out = ROOT / 'figures/preview_hindcast_gmst_emulator.png'
    fig.savefig(out, dpi=140, bbox_inches='tight'); print(f'\nsaved {out}')


if __name__ == '__main__':
    main()
