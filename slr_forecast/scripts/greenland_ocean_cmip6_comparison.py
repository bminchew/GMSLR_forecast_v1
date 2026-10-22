"""Projected Greenland ocean temperature compared with CMIP6 (oceanographer
referee, Major concern 4c).

Ours: the production ensemble of annual 200-500 m ocean temperature in the
EN4 Greenland box, the forcing of the discharge model. It is rebuilt here with
the production code (greenland_discharge.project_greenland_ocean_temperature),
the production inputs (component_greenland.ipynb cells 3 and 17) and the
production seeds (700 for the transfer and AA draws, 701 and 702 for the
transfer residual), from the transfer fit stored in component_results.h5
(greenland/ocean_transfer). As a check, the same inputs passed through
project_greenland_discharge reproduce the stored discharge ensemble
(greenland/projections_discharge) for every SSP.

CMIP6: one member per model, window mean minus the model's own historical
1995-2005 mean, in the same box and layer. Two layer means are reported:
thetao_levmean_degC (the unweighted level mean of the production EN4 reader,
so the definition matched to ours) and thetao_degC (volume-weighted). The
CMIP6 file is written by scripts/cmip6_greenland_thetao.py under the
gitignored data/raw/ocean_temp/cmip6_greenland/; this script is not part of
run_pipeline.sh because the pipeline does not download it.

Baseline 1995-2005 (calendar years) on both sides. Our scenario forcing ends
partway through 2099, so the window runs from 2081 to our last full year of
forcing (2098), and the same years are used for CMIP6.

The CMIP6 historical and SSP runs are separate files, and a few models step
at the 2014/2015 junction. For each model and SSP the junction step is the
2015-2019 mean minus the 2010-2014 mean, compared with the same difference
of consecutive 5-year means over the historical record (z = departure from
their mean in units of their SD). Models with |z| > JUNCTION_Z_MAX in any
SSP are listed, and the CMIP6 summaries are printed with and without them.
Intervals are 5th-95th percentiles (across members for ours, across models
for CMIP6). The ratio is the window warming divided by the GMST warming over
the same years and baseline.

Run from the repository root:  python scripts/greenland_ocean_cmip6_comparison.py
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
sys.path.insert(0, str(ROOT / 'src'))

from slr_data_readers import read_berkeley_earth_gridded  # noqa: E402
from component_analysis import draw_lowpass_transfer  # noqa: E402
from greenland_discharge import (full_years_only, project_greenland_discharge,  # noqa: E402
                                 project_greenland_ocean_temperature)
from smb_emulator import ar1_noise  # noqa: E402
from ar6_gmst import midyear, ar6_to_berkeley_offset  # noqa: E402
from warming_paths import warming_z, offsets_on_grid, SAMPLE_WARMING_PATHS  # noqa: E402
from slr_forecast.config import BASELINE_YEAR, N_SAMPLES  # noqa: E402

H5_PATH = ROOT / 'data' / 'processed' / 'slr_processed_data.h5'
RESULTS_H5 = ROOT / 'data' / 'processed' / 'component_results.h5'
CMIP6_CSV = ROOT / 'data' / 'raw' / 'ocean_temp' / 'cmip6_greenland' / 'cmip6_thetao_greenland_200_500m.csv'
GRIDDED_PATH = ROOT / 'data' / 'raw' / 'gmst' / 'berkEarth_Global_TAVG_Gridded_1deg.nc'
GREENLAND_MASK = ROOT / 'data' / 'processed' / 'greenland_mask_berkeley_1deg.csv'

AA, AA_SIGMA = 2.5, 0.5          # component_greenland.ipynb cell 1
SSPS = {'SSP1-2.6': ('SSP1_2_6', 'ssp126'), 'SSP2-4.5': ('SSP2_4_5', 'ssp245'),
        'SSP5-8.5': ('SSP5_8_5', 'ssp585')}
CHECK_SSPS = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'SSP1-1.9']
SSP_H5 = {'SSP1-2.6': 'SSP1_2_6', 'SSP2-4.5': 'SSP2_4_5', 'SSP3-7.0': 'SSP3_7_0',
          'SSP5-8.5': 'SSP5_8_5', 'SSP1-1.9': 'SSP1_1_9'}
PROJ_YEARS = np.arange(1900, 2151, dtype=float)
STORED_FIRST_YEAR = 1950          # the stored projections start in 1950
BL = (1995, 2005)
WIN = (2081, 2100)
JUNCTION_Z_MAX = 3.0


def q(x):
    return np.percentile(x, [50, 5, 95])


def fmt(x):
    m, lo, hi = q(x)
    return f'{m:.2f} [{lo:.2f}, {hi:.2f}]'


def production_inputs():
    """The inputs of component_greenland.ipynb cells 3 and 17."""
    df_gr = read_berkeley_earth_gridded(str(GRIDDED_PATH), cell_mask=str(GREENLAND_MASK))
    gr_raw, gr_t_raw = df_gr['temperature'].values, df_gr['decimal_year'].values
    bl = (gr_t_raw >= 1995.0) & (gr_t_raw < 2006.0)
    gr_T = gr_raw - np.nanmean(gr_raw[bl])
    ok = np.isfinite(gr_T)
    gr_T, gr_t = gr_T[ok], gr_t_raw[ok]

    df_temp = pd.read_hdf(H5_PATH, key='harmonized/df_berkeley_h')
    temp_monthly = df_temp['temperature'].values
    temp_time_monthly = (df_temp.index.year + (df_temp.index.month - 0.5) / 12.0).values

    df_en4 = pd.read_hdf(H5_PATH, key='harmonized/df_en4_greenland_200_500m')
    t_oc, T_oc = full_years_only(df_en4['decimal_year'].values, df_en4['temperature'].values)
    T_oc = T_oc - np.nanmean(T_oc[(t_oc >= 1995.0) & (t_oc < 2006.0)])

    df_hist = midyear(pd.read_hdf(H5_PATH, key='projections/temp/Historical'))
    offset_gmst = ar6_to_berkeley_offset(str(H5_PATH))
    wz = warming_z(N_SAMPLES)
    temps, times, offs = {}, {}, {}
    for ssp, key in SSP_H5.items():
        df_ssp = midyear(pd.read_hdf(H5_PATH, key=f'projections/temp/{key}'))
        comb = pd.concat([df_hist[df_hist['decimal_year'] < 2015].copy(), df_ssp]).sort_index()
        comb = comb[~comb.index.duplicated(keep='last')]
        comb['temperature'] = comb['temperature'] - offset_gmst
        t_mon = np.arange(comb['decimal_year'].min(), comb['decimal_year'].max(), 1 / 12)
        T_mon = np.interp(t_mon, comb['decimal_year'].values, comb['temperature'].values)
        temps[ssp] = np.concatenate([temp_monthly, T_mon[t_mon > temp_time_monthly[-1]]])
        times[ssp] = np.concatenate([temp_time_monthly, t_mon[t_mon > temp_time_monthly[-1]]])
        offs[ssp] = offsets_on_grid(df_ssp, wz, times[ssp], zero_through=temp_time_monthly[-1])
    return dict(gr_T=gr_T, gr_t=gr_t, temp_monthly=temp_monthly,
                temp_time_monthly=temp_time_monthly, T_oc=T_oc, t_oc=t_oc,
                temps=temps, times=times, offs=offs)


def production_draws(transfer, temp_time_monthly):
    """Transfer, AA and residual draws in the order of cell 17."""
    rng = np.random.default_rng(700)
    tr = draw_lowpass_transfer(transfer, N_SAMPLES, rng)
    AA_draws = rng.normal(AA, AA_SIGMA, size=N_SAMPLES)
    noise_years = np.arange(int(np.floor(temp_time_monthly[-1])), int(PROJ_YEARS[-1]) + 1)
    sd = tr['sd'] * np.sqrt(1 - tr['phi'] ** 2)
    noise = ar1_noise(sd, tr['phi'], len(noise_years), np.random.default_rng(701))
    pre_years = np.arange(int(np.floor(temp_time_monthly[0])), noise_years[0])
    noise_pre = ar1_noise(sd, tr['phi'], len(pre_years), np.random.default_rng(702))
    return tr, AA_draws, noise, noise_years, noise_pre, pre_years


def cmip6_changes(var, win):
    """Per model: window-mean change from the model's own historical 1995-2005
    mean, its ratio to the GMST change, and the junction step."""
    df = pd.read_csv(CMIP6_CSV).drop_duplicates(['source_id', 'member_id', 'experiment_id', 'year'])
    df = df[np.isfinite(df[var])]
    rows = []
    for (m, mem), d in df.groupby(['source_id', 'member_id']):
        h = d[d.experiment_id == 'historical'].set_index('year')
        if h.empty or not set(range(BL[0], BL[1] + 1)) <= set(h.index):
            continue
        T0 = h.loc[BL[0]:BL[1], var].mean()
        G0 = h.loc[BL[0]:BL[1], 'gmst_K'].mean()
        # Differences of consecutive 5-year means within the historical run
        hv = h[var].reindex(range(h.index.min(), 2015))
        d5 = np.array([hv.loc[y:y + 4].mean() - hv.loc[y - 5:y - 1].mean()
                       for y in range(hv.index.min() + 5, 2011)])
        d5 = d5[np.isfinite(d5)]
        for exp in d.experiment_id.unique():
            if exp == 'historical':
                continue
            w = d[d.experiment_id == exp].set_index('year').loc[win[0]:win[1]]
            if len(w) < 15:
                continue
            dT, dG = w[var].mean() - T0, w['gmst_K'].mean() - G0
            s = d[d.experiment_id == exp].set_index('year')
            step = s.loc[2015:2019, var].mean() - h.loc[2010:2014, var].mean()
            z = (step - d5.mean()) / d5.std(ddof=1)
            rows.append(dict(source_id=m, experiment_id=exp, dT=dT, ratio=dT / dG,
                             step=step, z=z))
    return pd.DataFrame(rows)


def main():
    inp = production_inputs()
    with h5py.File(RESULTS_H5, 'r') as hf:
        g = hf['greenland/ocean_transfer']
        transfer = {k: g[k][:] for k in ('tau_grid', 'tau_weights', 'g_by_tau', 'c_by_tau',
                                         'cov_by_tau', 'phi_by_tau', 'sd_by_tau')}
        pd_ = hf['greenland/posteriors/discharge']
        result_discharge = type('Posterior', (), dict(
            delta_posterior=pd_['delta_posterior'][:], gamma_posterior=pd_['gamma_posterior'][:],
            r0_posterior=pd_['r0_posterior'][:], **{k: float(pd_.attrs[k]) for k in
                                                     ('int_T_mean_cal', 't_mean_cal', 'H_mean_cal')}))
        stored = {s: hf[f'greenland/projections_discharge/{s}/samples'][:] for s in CHECK_SSPS}
    tr, AA_draws, noise, noise_years, noise_pre, pre_years = production_draws(
        transfer, inp['temp_time_monthly'])
    common = dict(temp_monthly=inp['temp_monthly'], temp_time_monthly=inp['temp_time_monthly'],
                  T_ocean_monthly=inp['T_oc'], time_ocean_monthly=inp['t_oc'],
                  alpha_draws=tr['g'], beta_draws=tr['c'], AA_draws=AA_draws,
                  oc_noise=noise, oc_noise_years=noise_years,
                  oc_noise_pre=noise_pre, oc_noise_pre_years=pre_years,
                  sample_warming_paths=SAMPLE_WARMING_PATHS,
                  gr_temp_monthly=inp['gr_T'], gr_time_monthly=inp['gr_t'],
                  tau_draws=tr['tau'])

    # Check: these inputs and draws reproduce the stored production discharge
    dyn = project_greenland_discharge(CHECK_SSPS, PROJ_YEARS, inp['temps'], inp['times'],
                                      inp['offs'], result_discharge=result_discharge,
                                      baseline_year=BASELINE_YEAR, **common)
    i0 = int(np.argmin(np.abs(PROJ_YEARS - STORED_FIRST_YEAR)))
    for s in CHECK_SSPS:
        assert np.allclose(dyn[s][:, i0:], stored[s], rtol=0, atol=1e-12), \
            f'{s}: rebuilt discharge differs from greenland/projections_discharge (rerun greenland?)'
    print(f'Check: rebuilt discharge matches the stored ensemble for {len(CHECK_SSPS)} SSPs '
          f'({N_SAMPLES} members)')

    # Window: WIN[0] to the last full calendar year of our forcing, both sides
    last_full = min(max(y for y, n in zip(*np.unique(np.floor(inp['times'][s]).astype(int),
                                                     return_counts=True)) if n == 12)
                    for s in SSPS)
    win = (WIN[0], min(WIN[1], last_full))
    print(f'\nGreenland-box ocean warming at 200-500 m, {win[0]}-{win[1]} mean relative to '
          f'{BL[0]}-{BL[1]} (degC): median [5th, 95th]')
    print('  ratio = warming / GMST warming over the same years')
    cm = {v: cmip6_changes(v, win) for v in ('thetao_levmean_degC', 'thetao_degC')}
    flagged = {v: sorted(c.source_id[np.abs(c.z) > JUNCTION_Z_MAX].unique()) for v, c in cm.items()}
    print(f'  CMIP6 junction steps 2015-19 minus 2010-14 with |z| > {JUNCTION_Z_MAX:.0f}:')
    for v, c in cm.items():
        f = c[np.abs(c.z) > JUNCTION_Z_MAX].sort_values(['source_id', 'experiment_id'])
        print(f'    {v}: ' + ('; '.join(f'{r.source_id} {r.experiment_id} {r.step:+.2f} degC '
                                         f'(z {r.z:+.1f})' for r in f.itertuples()) or 'none'))
    for ssp, (key, exp) in SSPS.items():
        yrs, T = project_greenland_ocean_temperature(
            inp['temps'][ssp], inp['times'][ssp], inp['offs'][ssp], **common)
        t, G = inp['times'][ssp], inp['temps'][ssp][None, :] + inp['offs'][ssp]
        yr = np.floor(t).astype(int)
        w = (yrs >= win[0]) & (yrs <= win[1])
        b = (yrs >= BL[0]) & (yrs <= BL[1])
        dT = T[:, w].mean(axis=1) - T[:, b].mean(axis=1)
        # Member GMST on the same calendar years (cell 17 construction)
        Gw = G[:, np.isin(yr, yrs[w])].mean(axis=1)
        Gb = G[:, np.isin(yr, yrs[b])].mean(axis=1)
        ratio = dT / (Gw - Gb)
        print(f'  {ssp}:')
        print(f'    ours ({N_SAMPLES} members): {fmt(dT)} degC; '
              f'ratio {fmt(ratio)}')
        p5, p95 = np.percentile(dT, [5, 95])
        for v, lab in (('thetao_levmean_degC', 'level mean, as EN4'),
                       ('thetao_degC', 'volume-weighted')):
            for excl in (False, True):
                c = cm[v][cm[v].experiment_id == exp]
                if excl:
                    c = c[~c.source_id.isin(flagged[v])]
                r = c.ratio.values[np.isfinite(c.ratio.values)]
                print(f'    CMIP6 {lab}{", without junction steps" if excl else ""} '
                      f'({len(c)} models): {fmt(c.dT.values)} degC; '
                      f'ratio {fmt(r)} ({len(r)} models); '
                      f'below our 5th {np.mean(c.dT < p5):.2f}, above our 95th {np.mean(c.dT > p95):.2f}')


if __name__ == '__main__':
    main()
