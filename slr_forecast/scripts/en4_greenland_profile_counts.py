"""Observational support of the EN4 Greenland T_ocean box, 1900-2025.

Purpose
-------
The Greenland discharge model is forced by EN4.2.2 (g10) objective-analysis
temperature averaged over a box around Greenland at 200-500 m. The analyses
exist from 1900, but Good et al. (2013) state that they relax toward the
1971-2000 climatology (persistence factor 0.9 per month) where no
observations are available. This script measures how many observed
profiles inform the box in each year and month, and builds the box series
with the analysis uncertainty and observation weights, so the usability of
pre-1970 EN4 can be judged. It is a diagnostic; nothing in the pipeline
reads its output.

Box (matches production, slr_forecast.readers.ocean_temp.read_en4_regional)
---------------------------------------------------------------------------
Analyses: 1 degree cells whose centres lie in 58-80 N and 285-355 E
(75-5 W), inclusive; the EN4 levels whose centres lie in 200-500 m (six
levels, 207-447 m, layer bounds 195.5-490.9 m); equal weight per level
(nanmean), then a cos(lat)-weighted mean over columns with a valid value.
EN4 cell centres are at integer degrees (lat -83..89, lon 1..360).

Profiles are assigned to the box by two rules, both reported:
  * position rule: 58 <= LATITUDE <= 80 and 285 <= LONGITUDE mod 360 <= 355;
  * cell-centre rule: the profile lies in a 1 degree cell whose centre is in
    the box, i.e. round(LATITUDE) in 58..80 and round(LONGITUDE) mod 360 in
    285..355 (rounding half up). This covers 57.5-80.5 N, 284.5-355.5 E.
The 200-500 m depth window for profiles uses DEPH_CORRECTED (the
bias-corrected depth) in [200, 500] m inclusive.

QC rule (EN4 profile file format page, metoffice.gov.uk/hadobs/en4/
en4-0-2-profile-file-format.html; flag values are '1' accept, '4' reject,
'0' no data / no QC). A profile counts as QC-passed at 200-500 m when
  * POSITION_QC != '4' (a '4' rejects the whole profile), and
  * PROFILE_POTM_QC != '4' (a '4' rejects the whole temperature profile), and
  * at least one level with DEPH_CORRECTED in [200, 500] m has a
    non-missing POTM_CORRECTED and POTM_CORRECTED_QC == '1'.
A second count additionally excludes profiles with bit 2 of
QC_FLAGS_PROFILES set (the documentation: "there are other profile(s)
within 0.2 degrees of latitude and longitude and 1 hour that appear to be of
higher quality", i.e. probable duplicates). Whether EN4 withholds those from
the analysis is not stated in the format page, so both counts are given.

Instrument type: PROJECT_NAME (source; for WODxx records the next three
characters are the WOD instrument code, e.g. OSD bottle/ocean-station
data, MBT, XBT, CTD, PFL) and WMO_INST_TYPE (simplified WMO table 1770:
401 XBT/MBT, 741 TESAC, 831 profiling floats).

Subcommands
-----------
  counts  --profiles-dir DIR --out-dir DIR
      Reads EN.4.2.2.profiles.g10.YYYY.zip (read in memory, not extracted).
      Writes en4_greenland_box_profile_counts.csv (annual),
      en4_greenland_box_profile_counts_monthly.csv and
      en4_greenland_box_profile_instruments.csv.
  series  --analyses-dir DIR [DIR ...] --out-dir DIR [--h5 FILE]
      Reads EN.4.2.2.analyses.g10.YYYY.zip from the given directories (a later
      directory overrides an earlier one for the same year), builds the box
      temperature with read_en4_regional itself (anomaly from the mean of the
      months 1995.0 <= t < 2006.0), adds the box uncertainty and observation
      weights with the same masks, checks the result against the production
      h5 series if --h5 is given, and writes en4_greenland_box_series_monthly.csv
      and en4_greenland_box_series_annual.csv.
  figure  --derived-dir DIR --out FILE
      Four-panel diagnostic figure.

Run from the repository root, e.g.
  python -I scripts/en4_greenland_profile_counts.py counts \
      --profiles-dir data/raw/ocean_temp/en4/profiles \
      --out-dir data/raw/ocean_temp/en4/derived

Reference: Good, S. A., M. J. Martin and N. A. Rayner (2013), J. Geophys.
Res. Oceans, 118, 6704-6716, doi:10.1002/2013JC009067.
"""

import argparse
import collections
import os
import re
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

LAT_BOUNDS = (58.0, 80.0)
LON_BOUNDS_E = (285.0, 355.0)
DEPTH_BOUNDS = (200.0, 500.0)
REF_PERIOD = (1995.0, 2006.0)
FILL = 99999.0

# land columns give all-NaN slices in nanmean; expected
warnings.filterwarnings('ignore', message='Mean of empty slice')


# ---------------------------------------------------------------- profiles

def _chars(arr):
    """Char array (N, L) already read from netCDF -> list of python str."""
    import netCDF4 as nc
    arr = np.ma.filled(arr, b' ') if np.ma.isMaskedArray(arr) else arr
    return [str(s) for s in nc.chartostring(np.asarray(arr, dtype='S1'))]


def _flag(arr):
    """Char flag array (already row-selected) -> array of single-char str ('' -> '0')."""
    a = np.ma.filled(arr, b'0') if np.ma.isMaskedArray(arr) else np.asarray(arr)
    a = np.char.decode(a.astype('S1'), 'ascii')
    a[a == ''] = '0'
    return a


def _rows(ds, name, sel):
    """Read a whole variable, then select profiles in numpy.

    netCDF4 integer-array indexing issues one read per index, which is slow
    for files with ~10^5 profiles; reading the full variable from the
    in-memory file is fast.
    """
    return ds[name][:][sel]


def instrument_label(project):
    """Source:instrument label from PROJECT_NAME."""
    p = project.strip()
    m = re.match(r'^(WOD\d\d)([A-Z]{3})', p)
    if m:
        return f'WOD:{m.group(2)}'
    if p.startswith('ARGO'):
        return 'ARGO'
    m = re.match(r'^(GTSPP|ASBO)\s*([A-Z0-9]{2})', p)
    if m:
        return f'{m.group(1)}:{m.group(2)}'
    return p.split(' ')[0] if p else 'unknown'


def count_month(ds):
    """Per-profile in-box and QC information for one monthly profile file."""
    lat = np.ma.filled(ds['LATITUDE'][:].astype(float), np.nan)
    lon = np.ma.filled(ds['LONGITUDE'][:].astype(float), np.nan)
    lat[np.abs(lat) > 90] = np.nan
    lon[np.abs(lon) > 360] = np.nan
    lon360 = np.mod(lon, 360.0)
    in_pos = ((lat >= LAT_BOUNDS[0]) & (lat <= LAT_BOUNDS[1])
              & (lon360 >= LON_BOUNDS_E[0]) & (lon360 <= LON_BOUNDS_E[1]))
    clat = np.floor(lat + 0.5)
    clon = np.mod(np.floor(lon + 0.5), 360.0)
    clon[clon == 0] = 360.0          # EN4 lon centres run 1..360
    in_cell = ((clat >= LAT_BOUNDS[0]) & (clat <= LAT_BOUNDS[1])
               & (clon >= LON_BOUNDS_E[0]) & (clon <= LON_BOUNDS_E[1]))
    sel = np.where(in_pos | in_cell)[0]
    out = {'in_pos': in_pos[sel], 'in_cell': in_cell[sel],
           'clat': clat[sel], 'clon': clon[sel]}
    if sel.size == 0:
        out.update(has_200_500=np.zeros(0, bool), qc=np.zeros(0, bool),
                   dup=np.zeros(0, bool), inst=[], wmo=[])
        return out
    deph = np.ma.filled(_rows(ds, 'DEPH_CORRECTED', sel).astype(float), np.nan)
    potm = np.ma.filled(_rows(ds, 'POTM_CORRECTED', sel).astype(float), np.nan)
    deph[deph >= FILL] = np.nan
    potm[potm >= FILL] = np.nan
    lev_qc = _flag(_rows(ds, 'POTM_CORRECTED_QC', sel))
    pos_qc = _flag(_rows(ds, 'POSITION_QC', sel))
    prof_qc = _flag(_rows(ds, 'PROFILE_POTM_QC', sel))
    qcp = np.ma.filled(_rows(ds, 'QC_FLAGS_PROFILES', sel), 0).astype(np.int64)
    in_band = (deph >= DEPTH_BOUNDS[0]) & (deph <= DEPTH_BOUNDS[1]) & np.isfinite(potm)
    has_band = in_band.any(axis=1)
    lev_ok = in_band & (lev_qc == '1')
    qc = (pos_qc != '4') & (prof_qc != '4') & lev_ok.any(axis=1)
    dup = (qcp >> 2) & 1 == 1
    proj = _chars(_rows(ds, 'PROJECT_NAME', sel))
    wmo = [w.strip() for w in _chars(_rows(ds, 'WMO_INST_TYPE', sel))]
    out.update(has_200_500=has_band, qc=qc, dup=dup,
               inst=[instrument_label(p) for p in proj], wmo=wmo)
    return out


def cmd_counts(args):
    import netCDF4 as nc
    pdir = Path(args.profiles_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zips = sorted(pdir.glob('EN.4.2.2.profiles.g10.*.zip'))
    monthly, inst_rows = [], []
    ann_cells_pos, ann_cells_any = {}, {}
    for z in zips:
        year = int(z.stem.split('.')[-1])
        inst_c = collections.Counter()
        cells_year = set()
        with zipfile.ZipFile(z) as zf:
            names = sorted(n for n in zf.namelist() if n.endswith('.nc'))
            for name in names:
                month = int(name[-5:-3])
                ds = nc.Dataset('mem', memory=zf.read(name))
                r = count_month(ds)
                ds.close()
                ip, ic, qc, dup = r['in_pos'], r['in_cell'], r['qc'], r['dup']
                hb = r['has_200_500']
                cells_m = {(a, b) for a, b, q, c in
                           zip(r['clat'], r['clon'], qc, ic) if q and c}
                cells_year |= cells_m
                monthly.append({
                    'year': year, 'month': month,
                    'n_box_pos': int(ip.sum()),
                    'n_box_cell': int(ic.sum()),
                    'n_box_pos_200_500m': int((ip & hb).sum()),
                    'n_qc_pos': int((ip & qc).sum()),
                    'n_qc_cell': int((ic & qc).sum()),
                    'n_qc_pos_nodup': int((ip & qc & ~dup).sum()),
                    'n_qc_cell_nodup': int((ic & qc & ~dup).sum()),
                    'n_cells_qc': len(cells_m),
                })
                for lab, w, p, q in zip(r['inst'], r['wmo'], ip, qc):
                    if p and q:
                        inst_c[(lab, w)] += 1
        ann_cells_any[year] = len(cells_year)
        for (lab, w), n in sorted(inst_c.items()):
            inst_rows.append({'year': year, 'instrument': lab,
                              'wmo_inst_type': w, 'n_qc_pos': n})
        print(f'{year}: n_qc_pos {sum(m["n_qc_pos"] for m in monthly if m["year"] == year)}',
              flush=True)
    dfm = pd.DataFrame(monthly).sort_values(['year', 'month'])
    cols = [c for c in dfm.columns if c.startswith('n_') and c != 'n_cells_qc']
    dfa = dfm.groupby('year')[cols].sum()
    dfa['n_months_with_qc_pos'] = dfm.groupby('year')['n_qc_pos'].apply(lambda s: int((s > 0).sum()))
    dfa['n_cells_qc_year'] = pd.Series(ann_cells_any)
    dfa['n_months_in_file'] = dfm.groupby('year').size()
    dfa = dfa.reset_index()
    dfa.to_csv(out_dir / 'en4_greenland_box_profile_counts.csv', index=False)
    dfm.to_csv(out_dir / 'en4_greenland_box_profile_counts_monthly.csv', index=False)
    pd.DataFrame(inst_rows).to_csv(out_dir / 'en4_greenland_box_profile_instruments.csv',
                                   index=False)
    print(f'wrote {len(dfa)} years, {len(dfm)} months to {out_dir}')


# ---------------------------------------------------------------- analyses

def _box_masks(ds):
    lat = ds['lat'][:]
    lon = ds['lon'][:]
    depth = ds['depth'][:]
    return ((lat >= LAT_BOUNDS[0]) & (lat <= LAT_BOUNDS[1]),
            (lon >= LON_BOUNDS_E[0]) & (lon <= LON_BOUNDS_E[1]),
            (depth >= DEPTH_BOUNDS[0]) & (depth <= DEPTH_BOUNDS[1]),
            np.cos(np.radians(np.asarray(lat[(lat >= LAT_BOUNDS[0]) & (lat <= LAT_BOUNDS[1])]))))


def _box_field(ds, var, masks):
    lm, om, dm, cw = masks
    x = ds[var][0, :, :, :][dm, :, :][:, lm, :][:, :, om]
    x = np.ma.filled(x.astype(float), np.nan)
    return x, cw


def box_extras(zip_paths):
    """Uncertainty and observation weights over the production box, monthly.

    Aggregation: per column the unweighted mean over the box levels, then the
    cos(lat)-weighted mean over columns with a valid temperature (the same
    rule as the temperature). Also the box maximum of the per-column mean
    uncertainty, the fraction of box cells (column x level) with observation
    weight > 0.1, and the raw temperature mean in degC as a check against
    read_en4_regional.
    """
    import netCDF4 as nc
    rows, masks = [], None
    for z in zip_paths:
        with zipfile.ZipFile(z) as zf:
            for name in sorted(n for n in zf.namelist() if n.endswith('.nc')):
                yyyymm = name.replace('.nc', '').split('.')[-1]
                ds = nc.Dataset('mem', memory=zf.read(name))
                if masks is None:
                    masks = _box_masks(ds)
                T, cw = _box_field(ds, 'temperature', masks)
                U, _ = _box_field(ds, 'temperature_uncertainty', masks)
                W, _ = _box_field(ds, 'temperature_observation_weights', masks)
                ds.close()
                Tc = np.nanmean(T, axis=0) - 273.15
                valid = np.isfinite(Tc)
                wt = np.where(valid, cw[:, None], 0.0)
                Uc = np.nanmean(np.where(np.isfinite(T), U, np.nan), axis=0)
                Wc = np.nanmean(np.where(np.isfinite(T), W, np.nan), axis=0)
                wv = np.isfinite(T)
                rows.append({
                    'year': int(yyyymm[:4]), 'month': int(yyyymm[4:6]),
                    'temperature_abs_degC': np.nansum(Tc * wt) / wt.sum(),
                    'uncertainty_mean_degC': np.nansum(np.where(valid, Uc, 0) * wt) / wt.sum(),
                    'uncertainty_max_degC': float(np.nanmax(Uc)),
                    'obs_weight_mean': np.nansum(np.where(valid, Wc, 0) * wt) / wt.sum(),
                    'frac_cells_obs_weight_gt_0p1': float((W[wv] > 0.1).mean()),
                })
    return pd.DataFrame(rows)


def cmd_series(args):
    from slr_forecast.readers.ocean_temp import read_en4_regional
    by_year = {}
    for d in args.analyses_dir:
        for z in sorted(Path(d).glob('EN.4.2.2.analyses.g10.*.zip')):
            by_year[int(z.stem.split('.')[-1])] = z.resolve()
    years = sorted(by_year)
    print(f'analyses: {years[0]}-{years[-1]} ({len(years)} files); '
          f'sources: {sorted({str(p.parent) for p in by_year.values()})}')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for y, p in by_year.items():
            os.symlink(p, Path(tmp) / p.name)
        df = read_en4_regional(tmp, lat_bounds=LAT_BOUNDS, lon_bounds=(-75.0, -5.0),
                               depth_bounds=DEPTH_BOUNDS, reference_period=REF_PERIOD)
    ref_mean = df.attrs['reference_mean_degC']
    ex = box_extras([by_year[y] for y in years])
    df = df.reset_index()
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    m = df.merge(ex, on=['year', 'month'], how='left', validate='one_to_one')
    chk = np.nanmax(np.abs(m['temperature_abs_degC'] - ref_mean - m['temperature']))
    print(f'reference mean (1995.0<=t<2006.0) {ref_mean:.4f} degC; '
          f'max |extras T - reader T| {chk:.2e} degC')
    if not chk < 1e-5:
        raise SystemExit('box_extras does not reproduce read_en4_regional')
    if args.h5:
        prod = pd.read_hdf(args.h5, 'harmonized/df_en4_greenland_200_500m')
        a = m.set_index('time')['temperature']
        b = prod['temperature']
        common = a.index.intersection(b.index)
        da = a[common] - a[common].mean()
        db = b[common] - b[common].mean()
        diff = float(np.nanmax(np.abs(da - db)))
        print(f'vs production h5 over {len(common)} common months '
              f'({common.min():%Y-%m} to {common.max():%Y-%m}), demeaned: '
              f'max |diff| {diff:.2e} degC; offset of means '
              f'{float(a[common].mean() - b[common].mean()):+.2e} degC')
        if not diff < 1e-6:
            raise SystemExit('box series does not reproduce the production series')
    m = m.rename(columns={'temperature': 'temperature_anom_degC',
                          'temperature_sigma': 'spatial_sd_degC'})
    keep = ['year', 'month', 'decimal_year', 'temperature_anom_degC',
            'temperature_abs_degC', 'spatial_sd_degC', 'n_valid',
            'uncertainty_mean_degC', 'uncertainty_max_degC', 'obs_weight_mean',
            'frac_cells_obs_weight_gt_0p1']
    m[keep].to_csv(out_dir / 'en4_greenland_box_series_monthly.csv', index=False,
                   float_format='%.5f')
    g = m.groupby('year')
    ann = g[keep[3:]].mean()
    ann['n_months'] = g.size()
    ann = ann[ann['n_months'] == 12].drop(columns='n_valid')
    ann.reset_index().to_csv(out_dir / 'en4_greenland_box_series_annual.csv', index=False,
                             float_format='%.5f')
    print(f'wrote {len(m)} months, {len(ann)} full years to {out_dir}')


# ---------------------------------------------------------------- figure

def cmd_figure(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = Path(args.derived_dir)
    ann = pd.read_csv(d / 'en4_greenland_box_series_annual.csv')
    cnt = pd.read_csv(d / 'en4_greenland_box_profile_counts.csv')
    ink, muted = '#1f2a37', '#8a94a3'
    fig, ax = plt.subplots(4, 1, figsize=(8, 10), sharex=True,
                           gridspec_kw={'hspace': 0.3})
    a = ax[0]
    a.axhline(0, color=muted, lw=0.8)
    a.plot(ann['year'], ann['temperature_anom_degC'], color=ink, lw=1.6)
    a.axvline(1970, color=muted, ls='--', lw=0.8)
    a.text(1970.5, 0.98, 'local EN4 began 1970', transform=a.get_xaxis_transform(),
           va='top', fontsize=8, color=muted)
    a.set_ylabel('Anomaly (°C)\nrel. 1995–2005')
    a.set_title('(a) EN4.2.2 g10 box temperature, 58–80°N, 75–5°W, 200–500 m',
                loc='left', fontsize=10)
    a = ax[1]
    n = cnt.set_index('year')['n_qc_pos']
    pos = n[n > 0]
    a.bar(pos.index, pos - 0.7, width=0.8, color=ink, bottom=0.7)   # top at n
    zero = n[n == 0].index
    a.plot(zero, np.full(len(zero), 0.9), 'x', color='#b4423a', ms=4,
           label='zero profiles')
    a.set_yscale('log')
    a.set_ylim(0.7, None)
    for lev in (10, 50, 100):
        a.axhline(lev, color=muted, lw=0.6, ls=':')
    a.set_ylabel('Profiles per year')
    a.set_title('(b) QC-passed profiles with a 200–500 m temperature in the box',
                loc='left', fontsize=10)
    if len(zero):
        a.legend(frameon=False, fontsize=8, loc='upper left')
    a = ax[2]
    a.plot(ann['year'], ann['uncertainty_mean_degC'], color=ink, lw=1.6)
    a.set_ylabel('Mean cell σ (°C)')
    a.set_title('(c) EN4 analysis uncertainty, box mean of the cell values',
                loc='left', fontsize=10)
    a = ax[3]
    a.plot(ann['year'], ann['obs_weight_mean'], color=ink, lw=1.6)
    a.set_ylabel('Obs. weight')
    a.set_ylim(0, None)
    a.set_title('(d) EN4 temperature observation weight, box mean (0 = background only)',
                loc='left', fontsize=10)
    a.set_xlabel('Year')
    for a in ax:
        a.grid(axis='y', color='#e5e7eb', lw=0.6)
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'wrote {args.out}')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('counts')
    c.add_argument('--profiles-dir', required=True)
    c.add_argument('--out-dir', required=True)
    s = sub.add_parser('series')
    s.add_argument('--analyses-dir', required=True, nargs='+')
    s.add_argument('--out-dir', required=True)
    s.add_argument('--h5')
    f = sub.add_parser('figure')
    f.add_argument('--derived-dir', required=True)
    f.add_argument('--out', required=True)
    args = p.parse_args()
    {'counts': cmd_counts, 'series': cmd_series, 'figure': cmd_figure}[args.cmd](args)


if __name__ == '__main__':
    main()
