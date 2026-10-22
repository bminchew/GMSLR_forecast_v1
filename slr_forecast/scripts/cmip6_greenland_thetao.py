"""CMIP6 200-500 m ocean temperature around Greenland, in the EN4 T_ocean box.

Purpose: compare the projected Greenland ocean forcing (EN4 box mean, projected
through the Greenland-T transfer function and Arctic amplification) with the
CMIP6 ensemble (referee report, oceanographer, Major concern 4c).

Region and layer follow the production EN4 series
(slr_forecast.readers.ocean_temp.read_en4_regional): grid cells whose centres
lie in 58-80 N, 75-5 W (285-355 E, inclusive), 200-500 m. Two layer means are
written for every model month, then averaged to calendar years (full years
only):

  thetao_degC          volume-weighted: cell area x thickness of each level
                       inside 200-500 m (partial-cell overlap of lev_bnds with
                       [200, 500] m), over every valid (ocean) cell.
  thetao_levmean_degC  EN4-style: per column, unweighted mean of the levels
                       whose centres lie in [200, 500] m (nanmean, so shallow
                       columns with one valid level count), then an area-
                       weighted mean over columns. This mimics the production
                       EN4 reader, which takes an unweighted level mean.

Data access. The model/member choice, cell area (Ofx areacello) and GMST
(Amon tas) come from the Pangeo CMIP6 cloud catalog, read as zarr over HTTPS,
as in gcm_greenland_temperature.py. thetao itself is read from ESGF THREDDS
OPeNDAP servers with server-side index subsetting (files listed by the DKRZ
federated index, CEDA/DKRZ replicas; if those fail or do not cover the years,
also by the US ESGF2 index, ORNL/UCAR replicas; if no OPeNDAP copy works, the
Pangeo zarr store is read whole, up to PANGEO_MAX_GB). The Pangeo
thetao zarr stores are chunked as whole global 3-D fields (e.g. CESM2:
8 x 60 x 384 x 320 per chunk, ~114 MB), so a box subset still transfers the
whole field (~50-100 GB per model for 1950-2100), while OPeNDAP returns only
the box (~0.2-1 MB per month).

GMST: area-weighted (cos lat) global annual mean of Amon tas from Pangeo, as in
gcm_greenland_temperature.py; taken from data/raw/ice_sheets/greenland/
mar_protect/gcm_temperature.csv where that file already has the run.

Output: data/raw/ocean_temp/cmip6_greenland/cmip6_thetao_greenland_200_500m.csv
  columns: source_id, member_id, experiment_id, year, thetao_degC,
           thetao_levmean_degC, gmst_K, grid_label
  Runs already in the file are skipped and new runs are appended. Each run is
  extracted by a worker subprocess (wall-clock timeout) that writes a fragment
  and a provenance JSON under fragments/; failures are logged in
  extraction_log.csv.

Usage (from the repository root):
  python scripts/cmip6_greenland_thetao.py                 # all models
  python scripts/cmip6_greenland_thetao.py --models CESM2  # subset
  python scripts/cmip6_greenland_thetao.py --plan-only     # print the plan
  python scripts/cmip6_greenland_thetao.py --rebuild       # CSV from fragments/
"""

import argparse
import concurrent.futures as cf
import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
CATALOG = 'https://storage.googleapis.com/cmip6/pangeo-cmip6.csv'
ESGF_SEARCH = 'https://esgf-data.dkrz.de/esg-search/search'
ESGF_SEARCH_US = 'https://esgf-node.ornl.gov/esgf-1-5-bridge'
NODE_PREF = ['esgf.ceda.ac.uk', 'esgf3.dkrz.de', 'esgf-node.ornl.gov', 'esgf-data.ucar.edu',
             'eagle.alcf.anl.gov']
OUT_DIR = ROOT / 'data/raw/ocean_temp/cmip6_greenland'
OUT = OUT_DIR / 'cmip6_thetao_greenland_200_500m.csv'
FRAG = OUT_DIR / 'fragments'
LOG = OUT_DIR / 'extraction_log.csv'
GCM_T = ROOT / 'data/raw/ice_sheets/greenland/mar_protect/gcm_temperature.csv'

LAT_BOUNDS = (58.0, 80.0)
LON_BOUNDS = (285.0, 355.0)          # 75 W - 5 W in 0-360 E
DEPTH = (200.0, 500.0)
YEARS = (1950, 2100)
PANGEO_MAX_GB = 40     # uncompressed whole-field transfer cap for the zarr fallback
EXPS = ['historical', 'ssp126', 'ssp245', 'ssp585']

# MAR PROTECT drivers and their members (gcm_greenland_temperature.py RUNS)
PROTECT = [('CESM2', 'r11i1p1f1'), ('MPI-ESM1-2-HR', 'r1i1p1f1'),
           ('NorESM2-MM', 'r1i1p1f1'), ('UKESM1-0-LL', 'r1i1p1f2'),
           ('CNRM-CM6-1', 'r1i1p1f2'), ('CNRM-ESM2-1', 'r1i1p1f2'),
           ('IPSL-CM6A-LR', 'r1i1p1f1')]
# NorESM2 'gn' ocean output is on isopycnal (rho) layers; 'gr' is on z levels
GRID_PREF = {'NorESM2-MM': 'gr', 'NorESM2-LM': 'gr'}
COLS = ['source_id', 'member_id', 'experiment_id', 'year', 'thetao_degC',
        'thetao_levmean_degC', 'gmst_K', 'grid_label']


def https(zstore):
    return zstore.replace('gs://', 'https://storage.googleapis.com/')


def load_catalog():
    """Pangeo catalog rows this script uses (Omon thetao, Ofx areacello, Amon
    tas). The filtered rows are cached in OUT_DIR on first use, with the
    access date, as the provenance record of the catalog state."""
    cache = OUT_DIR / 'pangeo_catalog_subset.csv'
    if cache.exists():
        return pd.read_csv(cache)
    cat = pd.read_csv(CATALOG, usecols=['source_id', 'experiment_id', 'member_id', 'table_id',
                                        'variable_id', 'grid_label', 'zstore', 'version'])
    keep = (((cat.table_id == 'Omon') & (cat.variable_id == 'thetao')
             & cat.experiment_id.isin(EXPS))
            | ((cat.table_id == 'Ofx') & (cat.variable_id == 'areacello'))
            | ((cat.table_id == 'Amon') & (cat.variable_id == 'tas')
               & cat.experiment_id.isin(EXPS)))
    cat = cat[keep].copy()
    cat['accessed'] = time.strftime('%Y-%m-%d')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cat.to_csv(cache, index=False)
    return cat


# ---------------------------------------------------------------- planning

def plan_runs(cat, only=None):
    """[(source_id, member_id, grid_label, [experiments])], PROTECT models first,
    plus a list of (source_id, reason) for skipped models."""
    th = cat[(cat.table_id == 'Omon') & (cat.variable_id == 'thetao')
             & cat.experiment_id.isin(EXPS)]
    plan, skipped = [], []
    order = [m for m, _ in PROTECT] + sorted(set(th.source_id) - {m for m, _ in PROTECT})
    protect_mem = dict(PROTECT)
    for model in order:
        if only and model not in only:
            continue
        t = th[th.source_id == model]
        if t.empty:
            skipped.append((model, 'no Omon thetao in Pangeo catalog'))
            continue
        avail = t.groupby(['member_id', 'grid_label']).experiment_id.apply(set)

        def ok(exps):
            return 'historical' in exps and bool(exps & {'ssp245', 'ssp585'})

        cands = [(m, g) for (m, g), e in avail.items() if ok(e)]
        if not cands:
            skipped.append((model, 'no member with thetao historical and ssp245 or ssp585 '
                                   f'(experiments: {sorted(set(t.experiment_id))})'))
            continue
        pref = [protect_mem.get(model), 'r1i1p1f1']
        pick = None
        for p in pref:
            for m, g in sorted(cands, key=lambda x: (x[1] != GRID_PREF.get(model, 'gn'), x[0])):
                if m == p:
                    pick = (m, g)
                    break
            if pick:
                break
        if pick is None:
            # most experiments, then lowest realization index
            def key(mg):
                r = re.match(r'r(\d+)i(\d+)p(\d+)f(\d+)', mg[0])
                return (-len(avail[mg]), tuple(int(x) for x in r.groups()) if r else (999,))
            pick = sorted(cands, key=key)[0]
        exps = [e for e in EXPS if e in avail[pick]]
        plan.append((model, pick[0], pick[1], exps))
    return plan, skipped


# ---------------------------------------------------------------- ESGF

def _esgf_query(index, q, timeout):
    url = index + '?' + urllib.parse.urlencode(q)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)['response']['docs']


def _covers(files, exp):
    # SSP runs need 2015-2099: some archives end in December 2099
    # (CAMS-CSM1-0), and the comparison window needs 15 of 2081-2100
    lo, hi = (YEARS[0], 2014) if exp == 'historical' else (2015, YEARS[1] - 1)
    yrs = set()
    for f in files:
        yrs.update(range(f['y0'], f['y1'] + 1))
    return all(y in yrs for y in range(lo, hi + 1))


def esgf_files(model, member, exp, grid, force_us=False):
    """OPeNDAP URLs per file (all replicas), from the DKRZ federated index (CEDA,
    DKRZ replicas); if those do not cover the years, also from the US ESGF2
    index (ORNL bridge: ORNL, UCAR, ALCF nodes)."""
    q = dict(project='CMIP6', source_id=model, experiment_id=exp, member_id=member,
             variable_id='thetao', table_id='Omon', grid_label=grid, type='File',
             format='application/solr+json', limit=1000, distrib='true', latest='true')
    docs = _esgf_query(ESGF_SEARCH, q, 120)
    files = _collect(docs)
    if force_us or not _covers(files, exp):
        for attempt in range(2):
            try:
                docs += _esgf_query(ESGF_SEARCH_US, q, 900)
                break
            except Exception:  # noqa: BLE001
                time.sleep(30)
        files = _collect(docs)
    return files


def _collect(docs):
    files = {}
    for d in docs:
        title = d['title']
        m = re.search(r'_(\d{6})-(\d{6})\.nc$', title)
        if not m:
            continue
        y0, y1 = int(m.group(1)[:4]), int(m.group(2)[:4])
        if y1 < YEARS[0] or y0 > YEARS[1]:
            continue
        dods = [u.split('|')[0] for u in d['url'] if u.endswith('|OPENDAP')]
        if not dods:
            continue
        u = dods[0]
        u = u[:-5] if u.endswith('.html') else u
        ver = str(d.get('version', ''))
        f = files.setdefault(title, {'y0': y0, 'y1': y1, 'urls': {}, 'version': ver})
        f['urls'][d['data_node']] = u
        f['dataset_id'] = d.get('dataset_id')
    out = []
    for title, f in sorted(files.items(), key=lambda kv: kv[1]['y0']):
        nodes = sorted(f['urls'], key=lambda n: NODE_PREF.index(n) if n in NODE_PREF else 99)
        out.append({'title': title, 'y0': f['y0'], 'y1': f['y1'], 'version': f['version'],
                    'dataset_id': f.get('dataset_id'),
                    'urls': [f['urls'][n] for n in nodes]})
    return out


def find_var(ds, names):
    for n in names:
        if n in ds.variables:
            return n
    return None


def open_dods(url):
    return xr.open_dataset(url, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
                           mask_and_scale=True)


def index_ranges(idx):
    """Contiguous ranges covering sorted unique indices, split at the largest gap
    when the set wraps around a grid seam."""
    u = np.unique(idx)
    if len(u) == 0:
        return []
    gaps = np.diff(u)
    if len(u) > 1 and gaps.max() > 1 and (u[-1] - u[0] + 1) > 1.5 * len(u):
        k = int(np.argmax(gaps))
        return [(int(u[0]), int(u[k]) + 1), (int(u[k + 1]), int(u[-1]) + 1)]
    return [(int(u[0]), int(u[-1]) + 1)]


def setup_grid(ds, model, grid, cat):
    """Box indices, level weights and cell areas from the first file."""
    tv = ds['thetao']
    if tv.ndim != 4:
        raise RuntimeError(f'thetao dims {tv.dims}: unstructured grid, not handled')
    tdim, zdim, ydim, xdim = tv.dims
    latn = find_var(ds, ['latitude', 'lat', 'nav_lat'])
    lonn = find_var(ds, ['longitude', 'lon', 'nav_lon'])
    lat = ds[latn].values
    lon = ds[lonn].values
    if lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    lon = lon % 360
    box = ((lat >= LAT_BOUNDS[0]) & (lat <= LAT_BOUNDS[1])
           & (lon >= LON_BOUNDS[0]) & (lon <= LON_BOUNDS[1]))
    jj, ii = np.where(box)
    jr = index_ranges(jj)
    ir = index_ranges(ii)
    if len(jr) != 1:
        raise RuntimeError(f'box rows not contiguous: {jr}')

    if zdim in ('rho', 'sigma', 'density'):
        raise RuntimeError(f'vertical coordinate {zdim} is not depth')
    z = ds[zdim]
    zunits = str(z.attrs.get('units', '')).lower()
    zb_name = z.attrs.get('bounds') or find_var(ds, [f'{zdim}_bnds', f'{zdim}_bounds', 'lev_bnds',
                                                     'olevel_bounds', 'olevel_bnds'])
    zc = z.values.astype(float)
    if zb_name and zb_name in ds.variables:
        zb = ds[zb_name].values.astype(float)
        if zb.ndim == 3:
            zb = zb[0]
        zb_src = zb_name
    else:
        mid = 0.5 * (zc[1:] + zc[:-1])
        zb = np.column_stack([np.r_[0.0, mid], np.r_[mid, zc[-1] + (zc[-1] - mid[-1])]])
        zb_src = 'midpoints'
    scale = 0.01 if ('cm' in zunits or 'centi' in zunits or zc.max() > 20000) else 1.0
    # bounds can carry their own units (CESM2: lev in cm, lev_bnds in m); scale
    # them so their midpoints match the level centres
    zb = np.sort(zb, axis=1)
    bscale = scale
    if zb_src != 'midpoints':
        r = np.nanmedian(zb.mean(axis=1)) / np.nanmedian(zc)
        bscale = scale if abs(np.log10(r)) < 1 else scale / r
        bscale = 1.0 if abs(bscale - 1) < 0.2 else (0.01 if abs(bscale - 0.01) < 0.002 else bscale)
    zc, zb = zc * scale, zb * bscale
    dz = np.clip(np.minimum(zb[:, 1], DEPTH[1]) - np.maximum(zb[:, 0], DEPTH[0]), 0, None)
    centre_in = (zc >= DEPTH[0]) & (zc <= DEPTH[1])
    k = np.where((dz > 0) | centre_in)[0]
    k0, k1 = int(k.min()), int(k.max()) + 1

    # cell area: Pangeo Ofx areacello on the same grid, else cos(lat)
    area_src = 'cos(lat) approximation'
    area = np.cos(np.deg2rad(lat))
    # same grid label first, then any label whose horizontal shape matches
    # (NorESM2 'gr' z-level output shares the 'gn' horizontal grid)
    a = cat[(cat.source_id == model) & (cat.table_id == 'Ofx')
            & (cat.variable_id == 'areacello')].copy()
    a['own'] = a.grid_label == grid
    a = a.sort_values(['own', 'version'])
    for _, row in a[::-1].iterrows():
        try:
            ac = xr.open_zarr(https(row.zstore), consolidated=True)['areacello'].values.astype(float)
            if ac.shape == lat.shape:
                area = ac
                area_src = f'areacello {https(row.zstore)}'
                if row.grid_label != grid:
                    area_src += f' (grid_label {row.grid_label}, same shape)'
                break
        except Exception:
            continue

    # areacello is NaN (or a fill value) on land; zero weight there
    area = np.where(np.isfinite(area) & (np.abs(area) < 1e15), area, 0.0)
    g = dict(dims=(tdim, zdim, ydim, xdim), jr=jr[0], ir=ir, k=(k0, k1),
             dz=dz[k0:k1], centre_in=centre_in[k0:k1], zc=zc[k0:k1],
             area=[area[jr[0][0]:jr[0][1], a0:a1] for a0, a1 in ir],
             box=[box[jr[0][0]:jr[0][1], a0:a1] for a0, a1 in ir],
             meta=dict(lat_var=latn, lon_var=lonn, lev_dim=zdim, lev_units=zunits,
                       lev_scale_to_m=scale, lev_bounds=zb_src, lev_bounds_scale_to_m=bscale, area=area_src,
                       levels_m=[round(float(x), 2) for x in zc[k0:k1]],
                       level_overlap_m=[round(float(x), 2) for x in dz[k0:k1]],
                       j_range=list(jr[0]), i_ranges=[list(r) for r in ir],
                       n_box_columns=int(box.sum())))
    return g


def box_means(T, g):
    """T: list over i-ranges of (time, lev, y, x) arrays in degC. Returns the two
    layer means per time step."""
    num_a = den_a = num_b = den_b = 0.0
    dz = g['dz'][None, :, None, None]
    cin = g['centre_in']
    for Ti, area, box in zip(T, g['area'], g['box']):
        Ti = np.where(np.abs(Ti) > 1e10, np.nan, Ti)
        Ti = np.where(box[None, None], Ti, np.nan)
        valid = np.isfinite(Ti)
        w = area[None, None] * dz * valid
        num_a = num_a + np.nansum(np.where(valid, Ti, 0.0) * w, axis=(1, 2, 3))
        den_a = den_a + w.sum(axis=(1, 2, 3))
        with np.errstate(invalid='ignore'):
            col = np.nanmean(Ti[:, cin], axis=1)              # (time, y, x)
        cv = np.isfinite(col)
        wc = area[None] * cv
        num_b = num_b + np.nansum(np.where(cv, col, 0.0) * wc, axis=(1, 2))
        den_b = den_b + wc.sum(axis=(1, 2))
    return num_a / den_a, num_b / den_b, den_a


ZERO_FRAC_MAX = 0.01   # exact zeros among valid box values in one chunk
JUMP_MAX = 2.0         # degC, change of the chunk-mean level mean from the previous chunk


def check_chunk(parts, g, model, year0, prev):
    """Reject a chunk that looks like a bad server return, so read_file retries
    it (then the next replica). Three screens: a share of exact zeros among
    the valid box values (a zero-filled or partly filled return); a jump of
    the chunk-mean level mean from the previous chunk of the same file
    (HadGEM3-GC31-MM returned ~0.8 degC against ~4.7 degC for the first
    24-month chunk of three files); and monthly box means identical to the
    previous chunk's (HadGEM3-GC31-MM ssp585 once returned 2030-2031 again
    for 2032-2033). prev is (chunk mean, monthly level means) or None."""
    v = np.concatenate([p[np.isfinite(p) & (np.abs(p) < 1e10) & box[None, None]]
                        for p, box in zip(parts, g['box'])])
    if v.size and np.mean(v == 0.0) > ZERO_FRAC_MAX:
        raise ValueError(f'{model} {year0}: {np.mean(v == 0.0):.1%} exact zeros in the box')
    if prev is not None:
        mb = box_means(parts, g)[1]
        if abs(np.nanmean(mb) - prev[0]) > JUMP_MAX:
            raise ValueError(f'{model} {year0}: chunk mean {np.nanmean(mb):.2f} degC '
                             f'after {prev[0]:.2f} degC')
        if len(mb) == len(prev[1]) and np.array_equal(mb, prev[1]):
            raise ValueError(f'{model} {year0}: chunk repeats the previous chunk')


def process_ds(ds, g_holder, model, grid, cat, step=24):
    """Monthly (year, month, volume-weighted, level-mean) box means of 1950-2100
    for one opened dataset (an OPeNDAP file or a Pangeo zarr store)."""
    if g_holder.get('g') is None:
        g_holder['g'] = setup_grid(ds, model, grid, cat)
    g = g_holder['g']
    tdim, zdim, ydim, xdim = g['dims']
    yrs = np.array([t.year for t in ds[tdim].values])
    mons = np.array([t.month for t in ds[tdim].values])
    sel = np.where((yrs >= YEARS[0]) & (yrs <= YEARS[1]))[0]
    units = str(ds['thetao'].attrs.get('units', '')).strip()
    rows = []
    prev = None
    for s0 in range(0, len(sel), step):
        ts = sel[s0:s0 + step]
        parts = []
        for a0, a1 in g['ir']:
            x = ds['thetao'].isel({tdim: slice(int(ts[0]), int(ts[-1]) + 1),
                                   zdim: slice(*g['k']),
                                   ydim: slice(*g['jr']),
                                   xdim: slice(a0, a1)}).values.astype(float)
            parts.append(x)
        med = np.nanmedian(parts[0])
        if units.upper().startswith('K') or med > 100:
            parts = [p - 273.15 for p in parts]
        check_chunk(parts, g, model, yrs[ts[0]], prev)
        ma, mb, _ = box_means(parts, g)
        prev = (np.nanmean(mb), mb)
        for t, a, b in zip(ts, ma, mb):
            rows.append((int(yrs[t]), int(mons[t]), a, b))
    return rows, units


def read_file(f, g_holder, model, grid, cat, log):
    last = None
    for url in f['urls']:
        for attempt in range(3):
            try:
                ds = open_dods(url)
                rows, units = process_ds(ds, g_holder, model, grid, cat)
                ds.close()
                log.append({'file': f['title'], 'url': url, 'thetao_units': units,
                            'n_months': len(rows)})
                return rows
            except Exception as e:  # noqa: BLE001
                last = f'{type(e).__name__}: {e}'
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f'{f["title"]}: {last}')


def read_pangeo(cat, model, member, exp, grid, g_holder, log):
    """Fallback when ESGF OPeNDAP does not cover the run: read the Pangeo zarr
    store (whole-field chunks are transferred; slow but complete)."""
    c = cat[(cat.source_id == model) & (cat.experiment_id == exp) & (cat.member_id == member)
            & (cat.table_id == 'Omon') & (cat.variable_id == 'thetao')
            & (cat.grid_label == grid)].sort_values('version')
    if c.empty:
        raise RuntimeError('no OPeNDAP thetao on ESGF and no Pangeo zarr store')
    zs = https(c.iloc[-1].zstore)
    ds = xr.open_zarr(zs, consolidated=True,
                      decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    chunk = ds['thetao'].encoding.get('chunks', (12,))[0] or 12
    tv = ds['thetao']
    n_mon = 12 * (min(YEARS[1], 2014 if exp == 'historical' else YEARS[1])
                  - (YEARS[0] if exp == 'historical' else 2015) + 1)
    gb = n_mon * np.prod(tv.shape[1:]) * 4 / 1e9
    if gb > PANGEO_MAX_GB:
        raise RuntimeError(f'Pangeo fallback skipped: whole-field transfer ~{gb:.0f} GB '
                           f'(> {PANGEO_MAX_GB} GB)')
    rows, units = process_ds(ds, g_holder, model, grid, cat, step=max(int(chunk), 12))
    log.append({'file': 'pangeo zarr', 'url': zs, 'thetao_units': units, 'n_months': len(rows)})
    return rows


def gmst_from_pangeo(cat, model, member, exp):
    c = cat[(cat.source_id == model) & (cat.experiment_id == exp) & (cat.member_id == member)
            & (cat.table_id == 'Amon') & (cat.variable_id == 'tas')].sort_values('version')
    if c.empty:
        return None, 'no Amon tas in Pangeo'
    zs = c.iloc[-1].zstore
    ds = xr.open_zarr(https(zs), consolidated=True,
                      decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    tas = ds['tas']
    yrs = np.array([t.year for t in tas['time'].values])
    keep = np.where((yrs >= YEARS[0]) & (yrs <= YEARS[1]))[0]
    tas = tas.isel(time=keep)
    w = np.cos(np.deg2rad(ds['lat']))
    g = tas.weighted(w).mean(['lat', 'lon']).values
    y = yrs[keep]
    s = pd.Series(g).groupby(y).agg(['mean', 'count'])
    s = s[s['count'] == 12]['mean']
    return s, https(zs)


def worker(model, member, grid, exp):
    cat = load_catalog()
    FRAG.mkdir(parents=True, exist_ok=True)
    tag = f'{model}_{member}_{exp}'
    g_holder, log, monthly = {}, [], []
    errors = []
    files = []
    for force_us in (False, True):
        try:
            files = esgf_files(model, member, exp, grid, force_us=force_us)
            if not (files and _covers(files, exp)):
                raise RuntimeError('ESGF OPeNDAP files do not cover the years')
            monthly = []
            for f in files:
                monthly += read_file(f, g_holder, model, grid, cat, log)
            break
        except Exception as e:  # noqa: BLE001
            errors.append(f'ESGF (US index {force_us}): {type(e).__name__}: {e}'[:400])
            monthly, log = [], []
            if 'not handled' in str(e) or 'is not depth' in str(e):
                raise
    if not monthly:
        files = []
        g_holder = {}    # the zarr store's grid, not one from a failed OPeNDAP file
        monthly = read_pangeo(cat, model, member, exp, grid, g_holder, log)
    m = pd.DataFrame(monthly, columns=['year', 'month', 'a', 'b'])
    n_dup = int(m.duplicated(['year', 'month']).sum())
    m = m.drop_duplicates(['year', 'month']).sort_values(['year', 'month'])
    # Full years only: 12 months with finite values of both layer means
    ok = np.isfinite(m['a']) & np.isfinite(m['b'])
    cnt = m[ok].groupby('year').month.count().reindex(m.year.unique(), fill_value=0)
    ann = m[ok].groupby('year')[['a', 'b']].mean().loc[sorted(cnt[cnt == 12].index)]
    lo, hi = (YEARS[0], 2014) if exp == 'historical' else (2015, YEARS[1])
    dropped = sorted(int(y) for y in set(range(lo, hi + 1)) - set(ann.index))

    gm_src = None
    gm = None
    if GCM_T.exists():
        old = pd.read_csv(GCM_T)
        o = old[(old.gcm == model) & (old.member == member) & (old.experiment == exp)]
        if not o.empty:
            gm = o.set_index('year')['gmst_K']
            gm_src = str(GCM_T.relative_to(ROOT))
    if gm is None:
        try:
            gm, gm_src = gmst_from_pangeo(cat, model, member, exp)
        except Exception as e:  # noqa: BLE001
            gm, gm_src = None, f'tas failed: {type(e).__name__}: {e}'
    df = pd.DataFrame({'source_id': model, 'member_id': member, 'experiment_id': exp,
                       'year': ann.index.astype(int), 'thetao_degC': ann['a'].values,
                       'thetao_levmean_degC': ann['b'].values,
                       'gmst_K': (gm.reindex(ann.index).values if gm is not None
                                  else np.nan),
                       'grid_label': grid})
    meta = dict(source_id=model, member_id=member, experiment_id=exp, grid_label=grid,
                esgf_search=ESGF_SEARCH, files=[{k: f[k] for k in ('title', 'version',
                                                                  'dataset_id', 'urls')}
                                                for f in files],
                read_log=log, esgf_errors=errors, gmst_source=gm_src, years=[int(ann.index.min()),
                                                        int(ann.index.max())],
                years_missing=dropped, duplicate_month_stamps=n_dup,
                **g_holder['g']['meta'])
    df.to_csv(FRAG / f'{tag}.csv', index=False, float_format='%.5f')
    (FRAG / f'{tag}.json').write_text(json.dumps(meta, indent=1, default=str))
    print(f'{tag}: {len(df)} years {meta["years"]}, {meta["n_box_columns"]} columns, '
          f'levels {meta["levels_m"]}, area: {meta["area"][:40]}', flush=True)
    if dropped or n_dup:
        print(f'{tag}: years missing or incomplete {dropped}; '
              f'{n_dup} duplicate month stamps dropped', flush=True)


# ---------------------------------------------------------------- driver

def append_fragment(tag):
    df = pd.read_csv(FRAG / f'{tag}.csv')
    df[COLS].to_csv(OUT, mode='a', header=not OUT.exists(), index=False, float_format='%.5f')


def log_event(row):
    pd.DataFrame([row]).to_csv(LOG, mode='a', header=not LOG.exists(), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', nargs=4, metavar=('MODEL', 'MEMBER', 'GRID', 'EXP'))
    ap.add_argument('--models', nargs='*')
    ap.add_argument('--plan-only', action='store_true')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--timeout', type=int, default=5400)
    ap.add_argument('--rebuild', action='store_true',
                    help='rewrite the output CSV from fragments/ (drops duplicate rows)')
    args = ap.parse_args()
    if args.worker:
        worker(*args.worker)
        return
    if args.rebuild:
        parts = [pd.read_csv(f) for f in sorted(FRAG.glob('*.csv'))]
        df = pd.concat(parts).drop_duplicates(['source_id', 'member_id', 'experiment_id', 'year'])
        df = df.sort_values(['source_id', 'experiment_id', 'year'])
        df[COLS].to_csv(OUT, index=False, float_format='%.5f')
        print(f'rebuilt {OUT.name}: {len(df)} rows, {df.source_id.nunique()} models')
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cat = load_catalog()
    plan, skipped = plan_runs(cat, args.models)
    pd.DataFrame([{'source_id': m, 'member_id': mem, 'grid_label': g, 'experiments': ' '.join(e)}
                  for m, mem, g, e in plan]).to_csv(OUT_DIR / 'model_plan.csv', index=False)
    if not args.models:
        pd.DataFrame(skipped, columns=['source_id', 'reason']).to_csv(
            OUT_DIR / 'models_skipped_catalog.csv', index=False)
    for p in plan:
        print(p)
    print(f'{len(plan)} models planned, {len(skipped)} skipped at catalog stage')
    if args.plan_only:
        return

    have = set()
    if OUT.exists():
        o = pd.read_csv(OUT)
        have = set(zip(o.source_id, o.member_id, o.experiment_id))
    jobs = [(m, mem, g, e) for m, mem, g, exps in plan for e in exps
            if (m, mem, e) not in have]

    def run(job):
        m, mem, g, e = job
        tag = f'{m}_{mem}_{e}'
        if (FRAG / f'{tag}.csv').exists():
            return job, 'fragment', ''
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, __file__, '--worker', m, mem, g, e],
                               capture_output=True, text=True, timeout=args.timeout)
            ok = r.returncode == 0 and (FRAG / f'{tag}.csv').exists()
            msg = (r.stdout.strip().splitlines() or [''])[-1] if ok else \
                (r.stderr.strip().splitlines() or ['?'])[-1]
            return job, 'ok' if ok else 'failed', f'{msg} ({time.time() - t0:.0f} s)'
        except subprocess.TimeoutExpired:
            return job, 'failed', f'timeout after {args.timeout} s'

    with cf.ThreadPoolExecutor(args.jobs) as ex:
        for job, status, msg in ex.map(run, jobs):
            m, mem, g, e = job
            tag = f'{m}_{mem}_{e}'
            if status in ('ok', 'fragment'):
                append_fragment(tag)
            log_event({'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'source_id': m,
                       'member_id': mem, 'grid_label': g, 'experiment_id': e,
                       'status': status, 'message': msg})
            print(f'[{status}] {tag} {msg}', flush=True)


if __name__ == '__main__':
    main()
