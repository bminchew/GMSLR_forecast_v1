"""CMIP6 historical global-mean thermosteric sea level (zostoga) and GMST.

Purpose: compare CMIP6 historical thermosteric sea level with the observed
records (NOAA, EN4, IAP) and with the production two-layer thermosteric model,
both as trends and as rise per degree of each model's own GMST change.

Data access (follows scripts/cmip6_greenland_thetao.py and
scripts/gcm_greenland_temperature.py):
  zostoga (Omon, a scalar series) for historical, ssp245 and piControl, read
  from the Pangeo CMIP6 cloud catalog (zarr over HTTPS); where Pangeo lacks a
  run, the netCDF files are downloaded from ESGF (DKRZ federated index,
  HTTPServer links) into --download-dir.
  GMST: area-weighted (cos lat) global annual mean of Amon tas from Pangeo, as
  in gcm_greenland_temperature.py; reused from data/raw/ocean_temp/
  cmip6_greenland/cmip6_thetao_greenland_200_500m.csv where that file already
  has the same model and member (computed the same way).

Member choice: r1i1p1f1 when it has historical zostoga, otherwise the lowest
variant label; members with ssp245 are preferred. The piControl member is the
historical run's parent_variant_label; if that piControl is not available the
next member is tried.

Drift: the historical branch point is branch_time_in_parent (in
parent_time_units, decoded with the piControl calendar), located on the
piControl time axis. Linear and quadratic fits to piControl annual zostoga
over the segment parallel to 1850-2024 are evaluated at the parallel years
and subtracted (linear: the corrected series; quadratic: a check). If the
branch point or parallel segment lies outside the piControl run, the fit uses
the whole piControl run and the model is flagged. GMST is not drift corrected.

Subcommands:
  fetch    build the per-model annual CSV (--out-csv)
  analyze  trends, per-degree rise, summary JSON (--summary-json), figure (--figure)

Usage (from the repository root; -I because downloaded files are read):
  python -I scripts/cmip6_zostoga.py fetch --download-dir data/raw/steric/cmip6_zostoga \
      --out-csv data/processed/cmip6_zostoga_global.csv
  python -I scripts/cmip6_zostoga.py analyze --out-csv data/processed/cmip6_zostoga_global.csv \
      --download-dir data/raw/steric/cmip6_zostoga \
      --summary-json data/processed/cmip6_zostoga_summary.json --figure DIAGNOSTIC.png
The summary JSON is read by notebooks/results_figures.ipynb (supplementary Fig. S12c,d)
and scripts/manuscript_numbers.py. analyze needs fetch_provenance.json and
zelinka_ecs.json from the download directory.
"""

import argparse
import concurrent.futures as cf
import json
import re
import sys
import time
import traceback
import urllib.parse
import urllib.request
from pathlib import Path

import cftime
import h5py
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'notebooks')]
CATALOG = 'https://storage.googleapis.com/cmip6/pangeo-cmip6.csv'
ESGF_SEARCH = 'https://esgf-data.dkrz.de/esg-search/search'
NODE_PREF = ['esgf.ceda.ac.uk', 'esgf3.dkrz.de', 'esgf-data1.llnl.gov', 'esgf-node.ornl.gov',
             'esgf-data.ucar.edu', 'eagle.alcf.anl.gov']
GMST_CSV = ROOT / 'data/raw/ocean_temp/cmip6_greenland/cmip6_thetao_greenland_200_500m.csv'
H5_COMP = ROOT / 'data/processed/component_results.h5'
H5_PROC = ROOT / 'data/processed/slr_processed_data.h5'
OBS_JSON = ROOT / 'data/processed/thermosteric_record_sensitivity.json'
AR6_DIR = ROOT / 'data/raw/ipcc_ar6/slr/ar6'
ECS_URL = ('https://raw.githubusercontent.com/mzelinka/cmip56_forcing_feedback_ecs/'
           'master/cmip56_forcing_feedback_ecs.json')

EXPS = ['historical', 'ssp245', 'piControl']
HIST_SPAN = (1850, 2024)          # segment of piControl parallel to historical + ssp245
OUT_YEARS = (1950, 2100)
TAS_YEARS = (1950, 2100)
PREF_MEMBER = 'r1i1p1f1'
DECODE = xr.coders.CFDatetimeCoder(use_cftime=True)


def log(*a):
    print(*a, flush=True)


def https(zstore):
    return zstore.replace('gs://', 'https://storage.googleapis.com/')


def member_key(m):
    r = re.match(r'r(\d+)i(\d+)p(\d+)f(\d+)', m)
    return (m != PREF_MEMBER,) + (tuple(int(x) for x in r.groups()) if r else (999,))


# ---------------------------------------------------------------- catalogs

def load_pangeo(dl):
    """Pangeo catalog rows for zostoga (historical/ssp245/piControl) and Amon tas,
    cached in the download directory with the access date."""
    cache = dl / 'pangeo_catalog_zostoga_tas.csv'
    if cache.exists():
        return pd.read_csv(cache)
    full = dl / 'pangeo-cmip6.csv'
    src = full if full.exists() else CATALOG
    cat = pd.read_csv(src, usecols=['source_id', 'experiment_id', 'member_id', 'table_id',
                                    'variable_id', 'grid_label', 'zstore', 'version'])
    keep = (cat.experiment_id.isin(EXPS)
            & (((cat.table_id == 'Omon') & (cat.variable_id == 'zostoga'))
               | ((cat.table_id == 'Amon') & (cat.variable_id == 'tas'))))
    cat = cat[keep].copy()
    cat['accessed'] = time.strftime('%Y-%m-%d')
    cat.to_csv(cache, index=False)
    return cat


def esgf_query(q, timeout=180):
    url = ESGF_SEARCH + '?' + urllib.parse.urlencode(q)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.load(r)['response']['docs']
        except Exception:  # noqa: BLE001
            if attempt == 2:
                raise
            time.sleep(10 * (attempt + 1))


def esgf_members(dl):
    """{(model, experiment): set(members)} for zostoga on ESGF, cached."""
    cache = dl / 'esgf_zostoga_datasets.json'
    if cache.exists():
        raw = json.loads(cache.read_text())
    else:
        raw = []
        for exp in EXPS:
            q = dict(project='CMIP6', variable_id='zostoga', table_id='Omon', experiment_id=exp,
                     type='Dataset', latest='true', distrib='true', limit=5000,
                     fields='source_id,member_id,experiment_id,version,data_node',
                     format='application/solr+json')
            for d in esgf_query(q):
                raw.append([d['source_id'][0], d['experiment_id'][0], d['member_id'][0]])
        cache.write_text(json.dumps({'accessed': time.strftime('%Y-%m-%d'), 'rows': raw}))
        raw = {'rows': raw}
    out = {}
    for m, e, mem in raw['rows']:
        out.setdefault((m, e), set()).add(mem)
    return out


def esgf_download(model, exp, member, dl):
    """Download the zostoga files of one run (latest version, preferred node).
    Returns the list of local paths."""
    q = dict(project='CMIP6', source_id=model, experiment_id=exp, member_id=member,
             variable_id='zostoga', table_id='Omon', type='File', latest='true',
             distrib='true', limit=2000, format='application/solr+json')
    docs = esgf_query(q)
    files = {}
    for d in docs:
        http = [u.split('|')[0] for u in d['url'] if u.endswith('|HTTPServer')]
        if not http:
            continue
        ver = str(d.get('version', ''))
        files.setdefault(ver, {}).setdefault(d['title'], {})[d['data_node']] = http[0]
    if not files:
        raise RuntimeError(f'ESGF: no zostoga files for {model} {exp} {member}')
    ver = sorted(files)[-1]
    out_dir = dl / 'esgf' / model / exp / member / f'v{ver}'
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for title, nodes in sorted(files[ver].items()):
        p = out_dir / title
        if not p.exists() or p.stat().st_size == 0:
            order = sorted(nodes, key=lambda n: NODE_PREF.index(n) if n in NODE_PREF else 99)
            last = None
            for n in order:
                try:
                    tmp = p.with_suffix('.part')
                    with urllib.request.urlopen(nodes[n], timeout=300) as r, open(tmp, 'wb') as f:
                        f.write(r.read())
                    tmp.rename(p)
                    last = None
                    break
                except Exception as e:  # noqa: BLE001
                    last = f'{n}: {type(e).__name__}: {e}'
            if last:
                raise RuntimeError(f'ESGF download failed {title}: {last}')
        paths.append(p)
    return paths, f'ESGF v{ver}'


# ---------------------------------------------------------------- zostoga

def open_zostoga(model, exp, member, pcat, dl, force_esgf=False):
    """(dataset, source string) for one run, Pangeo first, then ESGF."""
    c = pcat[(pcat.source_id == model) & (pcat.experiment_id == exp) & (pcat.member_id == member)
             & (pcat.variable_id == 'zostoga')]
    if not c.empty and not force_esgf:
        row = c.sort_values('version').iloc[-1]
        try:
            ds = xr.open_zarr(https(row.zstore), consolidated=True, decode_times=DECODE)
            return ds, f'Pangeo {https(row.zstore)}'
        except Exception as e:  # noqa: BLE001
            log(f'  Pangeo failed {model} {exp} {member}: {e}; trying ESGF')
    paths, src = esgf_download(model, exp, member, dl)
    ds = xr.open_mfdataset(paths, decode_times=DECODE, combine='by_coords',
                           data_vars='minimal', coords='minimal', compat='override')
    return ds, src


def annual(ds):
    """Annual mean zostoga (m) over complete calendar years."""
    z = ds['zostoga']
    extra = [d for d in z.dims if d != 'time']
    if extra:
        z = z.squeeze(extra, drop=True) if all(z.sizes[d] == 1 for d in extra) else z.isel(
            {d: 0 for d in extra})
    t = z['time'].values
    yrs = np.array([x.year for x in t])
    v = z.values.astype(float)
    v = np.where(np.abs(v) > 1e10, np.nan, v)
    s = pd.Series(v, index=yrs)
    n = s.groupby(level=0).count()
    a = s.groupby(level=0).mean()
    return a[n >= 12], t


def branch_year(hist_ds, pi_times):
    """Decimal piControl year of the historical branch point."""
    a = hist_ds.attrs
    bt = float(re.match(r'\s*[-+]?[0-9.]+(?:[eE][-+]?\d+)?', str(a['branch_time_in_parent'])).group())
    units = str(a.get('parent_time_units', '')).strip()
    cal = pi_times[0].calendar
    units = re.sub(r'(\d{4}-\d{1,2}-\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})', r'\1 \2:\3:\4', units)
    d = cftime.num2date(bt, units, calendar=cal)
    return d.year + (d.dayofyr - 1) / (360 if cal == '360_day' else 365), units, cal


def drift_fit(pi, y0, y1):
    """Linear and quadratic fits (in mm, centred year) on piControl years [y0, y1]."""
    s = pi.loc[y0:y1].dropna()
    t = s.index.values.astype(float)
    tc = t.mean()
    p1 = np.polyfit(t - tc, s.values * 1e3, 1)
    p2 = np.polyfit(t - tc, s.values * 1e3, 2)
    return p1, p2, tc, len(s), (int(t.min()), int(t.max()))


# ---------------------------------------------------------------- tas

def tas_cache():
    if not GMST_CSV.exists():
        return pd.DataFrame()
    d = pd.read_csv(GMST_CSV, usecols=['source_id', 'member_id', 'experiment_id', 'year', 'gmst_K'])
    return d[d.experiment_id.isin(['historical', 'ssp245'])]


def tas_global_annual(model, member, exp, pcat):
    """Area-weighted (cos lat) global annual-mean Amon tas from Pangeo, as in
    gcm_greenland_temperature.py (read lazily, years TAS_YEARS)."""
    c = pcat[(pcat.source_id == model) & (pcat.experiment_id == exp) & (pcat.member_id == member)
             & (pcat.table_id == 'Amon') & (pcat.variable_id == 'tas')]
    if c.empty:
        return None, 'no Pangeo Amon tas'
    row = c.sort_values('version').iloc[-1]
    ds = xr.open_zarr(https(row.zstore), consolidated=True, decode_times=DECODE)
    tas = ds['tas']
    yrs = np.array([t.year for t in tas['time'].values])
    keep = np.where((yrs >= TAS_YEARS[0]) & (yrs <= TAS_YEARS[1]))[0]
    tas = tas.isel(time=keep)
    yrs = yrs[keep]
    w = np.cos(np.deg2rad(ds['lat']))
    w = (w / w.mean()).broadcast_like(tas.isel(time=0))
    g = ((tas * w).sum(['lat', 'lon']) / w.sum()).values
    s = pd.Series(g, index=yrs)
    n = s.groupby(level=0).count()
    return s.groupby(level=0).mean()[n == 12], f'Pangeo {https(row.zstore)}'


def tas_esgf_annual(model, member, exp):
    """Global annual-mean tas from ESGF OPeNDAP (cos-lat weights, years TAS_YEARS);
    used only when Pangeo has no Amon tas for the run."""
    q = dict(project='CMIP6', source_id=model, experiment_id=exp, member_id=member,
             variable_id='tas', table_id='Amon', type='File', latest='true', distrib='true',
             limit=500, format='application/solr+json')
    files = {}
    for d in esgf_query(q):
        dods = [u.split('|')[0] for u in d['url'] if u.endswith('|OPENDAP')]
        if dods:
            u = dods[0][:-5] if dods[0].endswith('.html') else dods[0]
            files.setdefault(str(d.get('version', '')), {}).setdefault(d['title'], {})[
                d['data_node']] = u
    if not files:
        return None, 'no ESGF Amon tas'
    ver = sorted(files)[-1]
    parts, used = [], []
    for title, nodes in sorted(files[ver].items()):
        m = re.search(r'_(\d{6})-(\d{6})\.nc$', title)
        if m and (int(m.group(2)[:4]) < TAS_YEARS[0] or int(m.group(1)[:4]) > TAS_YEARS[1]):
            continue
        order = sorted(nodes, key=lambda n: NODE_PREF.index(n) if n in NODE_PREF else 99)
        last = None
        for n in order:
            try:
                ds = xr.open_dataset(nodes[n], decode_times=DECODE)
                yrs = np.array([t.year for t in ds['time'].values])
                k = np.where((yrs >= TAS_YEARS[0]) & (yrs <= TAS_YEARS[1]))[0]
                w = np.cos(np.deg2rad(ds['lat'].values))[:, None]
                g = []
                for b0 in range(0, len(k), 60):      # small OPeNDAP requests
                    x = ds['tas'].isel(time=k[b0:b0 + 60]).values.astype(float)
                    x = np.where(np.abs(x) > 1e10, np.nan, x)
                    g.extend((x * w).sum(axis=(1, 2)) / (w * np.ones(x.shape[1:])).sum())
                g = np.array(g)
                if not np.all((g > 200) & (g < 330)):
                    raise RuntimeError(f'implausible global-mean tas from {n}')
                parts.append(pd.Series(g, index=yrs[k]))
                used.append(nodes[n])
                last = None
                break
            except Exception as e:  # noqa: BLE001
                last = e
        if last is not None:
            raise RuntimeError(f'{title}: {last}')
    s = pd.concat(parts)
    n = s.groupby(level=0).count()
    return s.groupby(level=0).mean()[n == 12], f'ESGF OPeNDAP v{ver} ({len(used)} files)'


# ---------------------------------------------------------------- one model

def process_model(model, pcat, emem, dl, tcache):
    def members(exp):
        pm = set(pcat[(pcat.source_id == model) & (pcat.experiment_id == exp)
                      & (pcat.variable_id == 'zostoga')].member_id)
        return pm | emem.get((model, exp), set())

    hist_m, ssp_m, pi_m = members('historical'), members('ssp245'), members('piControl')
    if not pi_m:
        return None, 'no piControl zostoga (Pangeo or ESGF)'
    cands = sorted(hist_m, key=lambda m: (m not in ssp_m,) + member_key(m))
    tried = []
    for mem in cands[:8]:
        try:
            hds, hsrc = open_zostoga(model, 'historical', mem, pcat, dl)
            parent = str(hds.attrs.get('parent_variant_label', mem)).strip()
            if parent not in pi_m:
                tried.append(f'{mem} (parent {parent} not available)')
                continue
            return process_member(model, mem, parent, hds, hsrc, tried, ssp_m, pcat, emem, dl,
                                  tcache), None
        except Exception as e:  # noqa: BLE001
            tried.append(f'{mem} ({type(e).__name__}: {str(e)[:150]})')
    return None, 'no usable member: ' + '; '.join(tried)


def process_member(model, mem, parent, hds, hsrc, tried, ssp_m, pcat, emem, dl, tcache):
    info = dict(source_id=model, member_id=mem, picontrol_member=parent, hist_source=hsrc,
                members_tried_before=list(tried))
    h, _ = annual(hds)
    sp = None
    if mem in ssp_m:
        sds, ssrc = open_zostoga(model, 'ssp245', mem, pcat, dl)
        sp, _ = annual(sds)
        info['ssp245_source'] = ssrc
        info['ssp245_parent_variant_label'] = str(sds.attrs.get('parent_variant_label', ''))
        info['ssp245_parent_ok'] = info['ssp245_parent_variant_label'] == mem
    pds, psrc = open_zostoga(model, 'piControl', parent, pcat, dl)
    pi, pt = annual(pds)
    info['picontrol_source'] = psrc
    info['picontrol_years'] = [int(pi.index.min()), int(pi.index.max())]

    # alignment
    hist0 = int(h.index.min())

    def locate(pi, pt):
        by, units, cal = branch_year(hds, pt)
        off = int(np.floor(by + 1e-6)) - hist0
        seg = (HIST_SPAN[0] + off, HIST_SPAN[1] + off)
        return by, units, cal, off, seg, bool(pi.index.min() <= seg[0] <= pi.index.max())

    try:
        inside = locate(pi, pt)[-1]
    except Exception:  # noqa: BLE001
        inside = False
    if not inside and psrc.startswith('Pangeo') and parent in emem.get((model, 'piControl'), ()):
        # Pangeo may hold only part of the piControl run; try the ESGF copy
        try:
            pds2, psrc2 = open_zostoga(model, 'piControl', parent, pcat, dl, force_esgf=True)
            pi2, pt2 = annual(pds2)
            if locate(pi2, pt2)[-1]:
                pi, pt, psrc = pi2, pt2, psrc2
                info['picontrol_source'] = psrc + ' (Pangeo copy did not contain the branch point)'
                info['picontrol_years'] = [int(pi.index.min()), int(pi.index.max())]
        except Exception as e:  # noqa: BLE001
            info['esgf_picontrol_retry_error'] = f'{type(e).__name__}: {e}'
    try:
        by, units, cal, off, seg, inside = locate(pi, pt)
        info.update(branch_time_in_parent=str(hds.attrs['branch_time_in_parent']),
                    parent_time_units=units, picontrol_calendar=cal, branch_year=round(by, 2))
        if not inside:
            off = None
    except Exception as e:  # noqa: BLE001
        info['branch_error'] = f'{type(e).__name__}: {e}'
        inside, off, seg = False, None, None
    if inside:
        y0, y1 = seg[0], min(seg[1], int(pi.index.max()))
        info['drift_mode'] = 'parallel segment'
        if y1 < seg[1]:
            info['drift_mode'] = 'parallel segment (truncated)'
    else:
        y0, y1 = int(pi.index.min()), int(pi.index.max())
        info['drift_mode'] = 'whole piControl (branch point outside piControl)'
    p1, p2, tc, n, span = drift_fit(pi, y0, y1)
    info.update(drift_fit_years=list(span), drift_fit_n=n, piC_offset_years=off,
                drift_lin_mm_per_yr=float(p1[0]), drift_quad_coefs=[float(x) for x in p2])
    info['picontrol_curvature_mm_per_yr2'] = float(2 * p2[0])

    # historical + ssp245 on one axis
    z = h.copy()
    if sp is not None:
        sp = sp[sp.index > z.index.max()]
        jump = float((sp.iloc[0] - z.iloc[-1]) * 1e3)
        d1 = np.diff(z.loc[2000:].values * 1e3)
        info['join_jump_mm'] = jump
        info['join_typical_step_mm'] = float(np.median(d1))
        info['join_step_std_mm'] = float(np.std(d1))
        bad = abs(jump - np.median(d1)) > JOIN_TOL * np.std(d1)
        if bad and abs(sp.iloc[0]) < 0.005:
            # ssp245 zostoga re-referenced to zero at the start of 2015 (e.g. MRI-ESM2-0):
            # restore continuity with the median 2000-2014 annual step
            shift = z.iloc[-1] + np.median(d1) / 1e3 - sp.iloc[0]
            sp = sp + shift
            info['join_rereferenced_shift_mm'] = float(shift * 1e3)
        elif bad:
            info['join_flag'] = 'discontinuous; excluded from post-2014 metrics'
        z = pd.concat([z, sp])
    yrs = z.index.values
    if off is None:
        off = int(np.round(tc)) - int(np.mean(HIST_SPAN))  # centred mapping for whole-run fit
        info['piC_offset_years_fallback'] = off
    tp = yrs + off - tc
    dl_mm = np.polyval(p1, tp)
    dq_mm = np.polyval(p2, tp)
    df = pd.DataFrame({'year': yrs, 'experiment': np.where(yrs <= h.index.max(), 'historical',
                                                           'ssp245'),
                       'zostoga_raw_mm': z.values * 1e3, 'drift_lin_mm': dl_mm,
                       'drift_quad_mm': dq_mm})
    df['zostoga_mm'] = df.zostoga_raw_mm - df.drift_lin_mm
    df['zostoga_quad_mm'] = df.zostoga_raw_mm - df.drift_quad_mm
    df = df[(df.year >= OUT_YEARS[0]) & (df.year <= OUT_YEARS[1])]

    # tas
    tas = pd.Series(dtype=float)
    tsrc = []
    for exp in (['historical', 'ssp245'] if sp is not None else ['historical']):
        c = tcache[(tcache.source_id == model) & (tcache.member_id == mem)
                   & (tcache.experiment_id == exp)] if len(tcache) else tcache
        if len(c):
            s = c.set_index('year').gmst_K
            tsrc.append(f'{exp}: {GMST_CSV.relative_to(ROOT)}')
        else:
            tc_path = dl / 'tas_global_annual' / f'{model}_{mem}_{exp}.csv'
            if tc_path.exists():
                t = pd.read_csv(tc_path, index_col=0)
                s, src = t['gmst_K'], t['source'].iloc[0] + ' (cached)'
            else:
                s, src = tas_global_annual(model, mem, exp, pcat)
            if s is None:
                try:
                    s, src = tas_esgf_annual(model, mem, exp)
                except Exception as e:  # noqa: BLE001
                    s, src = None, f'{src}; ESGF OPeNDAP failed ({type(e).__name__}: {e})'
            if s is None:
                tsrc.append(f'{exp}: {src}')
                continue
            if not tc_path.exists():
                tc_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({'gmst_K': s, 'source': src}).to_csv(tc_path)
            tsrc.append(f'{exp}: {src}')
        tas = pd.concat([tas, s[~s.index.isin(tas.index)]])
    info['tas_source'] = tsrc
    df['tas_K'] = df.year.map(tas)
    df['qc_flag'] = 'ok'
    tr = _trend(np.arange(1955, 2015.), df.set_index('year').zostoga_mm.loc[1955:2014].values)[0]
    if abs(tr) > QC_MAX_TREND:
        df['qc_flag'] = 'excluded_implausible'
    elif 'join_flag' in info:
        df.loc[df.year > h.index.max(), 'qc_flag'] = 'post2014_excluded_discontinuous'
    elif 'join_rereferenced_shift_mm' in info:
        df.loc[df.year > h.index.max(), 'qc_flag'] = 'ssp245_rereferenced'
    df.insert(0, 'picontrol_member', parent)
    df.insert(0, 'member_id', mem)
    df.insert(0, 'source_id', model)
    df['drift_mode'] = info['drift_mode']
    return df, info


def cmd_fetch(a):
    dl = Path(a.download_dir)
    dl.mkdir(parents=True, exist_ok=True)
    pcat = load_pangeo(dl)
    emem = esgf_members(dl)
    tcache = tas_cache()
    hist_models = ({m for (m, e) in emem if e == 'historical'}
                   | set(pcat[(pcat.variable_id == 'zostoga')
                              & (pcat.experiment_id == 'historical')].source_id))
    if a.models and Path(a.out_csv).exists():
        # --models writes only the listed models; it does not merge into an existing CSV
        raise SystemExit(f'{a.out_csv} exists; with --models write to a separate --out-csv')
    models = sorted(hist_models if not a.models else set(a.models))
    frames, infos, skipped = [], {}, {}

    def run(m):
        try:
            return m, *process_model(m, pcat, emem, dl, tcache)
        except Exception as e:  # noqa: BLE001
            return m, None, f'{type(e).__name__}: {e} | {traceback.format_exc(limit=2)[-300:]}'

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for m, res, err in ex.map(run, models):
            if res is None:
                skipped[m] = err
                log(f'SKIP {m}: {err}')
                continue
            df, info = res
            frames.append(df)
            infos[m] = info
            log(f'OK {m} {info["member_id"]} piC {info["picontrol_member"]} '
                f'{info["drift_mode"]}, drift {info["drift_lin_mm_per_yr"]:+.3f} mm/yr, '
                f'join {info.get("join_jump_mm", float("nan")):+.2f} mm')
    out = pd.concat(frames, ignore_index=True)
    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out_csv, index=False, float_format='%.4f')
    meta = Path(a.download_dir) / 'fetch_provenance.json'
    meta.write_text(json.dumps(dict(accessed=time.strftime('%Y-%m-%d'), models=infos,
                                    skipped=skipped), indent=1, default=str))
    log(f'wrote {a.out_csv} ({len(infos)} models); provenance {meta}')


# ---------------------------------------------------------------- analysis

def _trend(t, y):
    """OLS trend with a 90% interval; AR(1)-adjusted effective sample size
    (same as scripts/thermosteric_record_sensitivity.py)."""
    X = np.c_[np.ones_like(t), t]
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ c
    r1 = np.corrcoef(r[:-1], r[1:])[0, 1]
    n = len(t)
    neff = max(3.0, n * (1 - r1) / (1 + r1))
    # Santer et al. (2008): residual variance and t quantile on neff - 2 dof
    dof = neff - 2
    se = np.sqrt(r @ r / dof / ((t - t.mean())**2).sum())
    return float(c[1]), float(student_t.ppf(0.95, dof) * se)


def q3(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.median(x)), float(np.percentile(x, 5)), float(np.percentile(x, 95))]


TREND_WINDOWS = [(1955, 2012), (1970, 2005), (2005, 2024), (2013, 2024)]
BELOW_2000 = (0.117, 0.037)       # mm/yr, 1 sigma (Desbruyeres et al. 2016; production values)
NOAA_SCALE = 1.14                 # Li et al. (2022), production calibration
PD_WINDOWS = ((1955, 1974), (2005, 2024))
PD_TREND = (1971, 2014)
PD_FUTURE = ((1995, 2014), (2081, 2100))
QC_MAX_TREND = 5.0                # mm/yr
JOIN_TOL = 4.0                    # flag a historical->ssp245 jump beyond 4 std of annual steps


def load_cmip(path, prov):
    d = pd.read_csv(path)
    Z, ZQ, T = {}, {}, {}
    for m, g in d.groupby('source_id'):
        g = g.set_index('year')
        Z[m], ZQ[m], T[m] = g.zostoga_mm, g.zostoga_quad_mm, g.tas_K
    # plausibility: a drift-corrected 1955-2014 trend beyond QC_MAX_TREND (an order of
    # magnitude above the observed ~0.5-1 mm/yr) marks a defective series
    qc_bad = {}
    for m in list(Z):
        tr = _trend(np.arange(1955, 2015.), Z[m].loc[1955:2014].values)[0]
        if abs(tr) > QC_MAX_TREND:
            qc_bad[m] = tr
            del Z[m], ZQ[m], T[m]
    full = {m for m in Z if Z[m].index.max() >= 2024}
    join_bad = set()
    for m, i in prov['models'].items():
        if 'join_flag' in i:
            join_bad.add(m)
    return d, Z, ZQ, T, full - join_bad, join_bad, qc_bad


def win(s, w):
    s = s.loc[w[0]:w[1]].dropna()
    return s if len(s) == w[1] - w[0] + 1 else None


def berkeley_annual():
    d = pd.read_hdf(H5_PROC, 'harmonized/df_berkeley_h')
    yr = d.index.year
    n = d.groupby(yr).temperature.count()
    a = d.groupby(yr).temperature.mean()
    return a[n == 12]


def production():
    with h5py.File(H5_COMP, 'r') as f:
        py = f['proj_years'][:].astype(int)
        g = f['ocean/projections/SSP2-4.5']
        full = g['samples'][:] * 1e3
        upper = g['upper_samples'][:] * 1e3
        two = g['twolayer_samples'][:] * 1e3       # 0-2000 m (full minus the below-2000 m rate)
    return py, full, upper, two


def ar6_ssp245_berkeley():
    """AR6 SSP2-4.5 median GMST on the Berkeley basis (notebooks/ar6_gmst.py:
    mid-year stamps, offset = AR6 minus Berkeley over 2015-2024), by year."""
    ar = pd.read_hdf(H5_PROC, 'projections/temp/SSP2_4_5')
    ar = pd.Series(ar['temperature'].values, index=ar['decimal_year'].astype(int).values)
    be = berkeley_annual()
    be = be - be.loc[1995:2005].mean()
    off = ar.loc[2015:2024].mean() - be.loc[2015:2024].mean()
    s = pd.concat([be.loc[:2024], ar.loc[2025:] - off])
    s.loc[2100] = s.loc[2099]            # forcing held flat to 2100, as in production
    return s, float(off)


def cmd_analyze(a):
    dl = Path(a.download_dir)
    prov = json.loads((dl / 'fetch_provenance.json').read_text())
    d, Z, ZQ, T, post, join_bad, qc_bad = load_cmip(a.out_csv, prov)
    models = sorted(Z)
    obs = json.loads(OBS_JSON.read_text())
    S = dict(n_models=len(models), models=models, skipped=prov['skipped'],
             post2014_models=sorted(post), join_flagged=sorted(join_bad),
             n_reaching_2024=int(sum(Z[m].index.max() >= 2024 for m in models)),
             qc_excluded_1955_2014_trend_mm_per_yr=qc_bad,
             members={m: [prov['models'][m]['member_id'], prov['models'][m]['picontrol_member']]
                      for m in models})

    # ---- drift diagnostics
    dr = {}
    for m in models:
        i = prov['models'][m]
        lin = _trend(np.arange(1955, 2015.), Z[m].loc[1955:2014].values)[0]
        quad = _trend(np.arange(1955, 2015.), ZQ[m].loc[1955:2014].values)[0]
        raw = d[(d.source_id == m) & d.year.between(1955, 2014)].zostoga_raw_mm.values
        rawt = _trend(np.arange(1955, 2015.), raw)[0]
        dr[m] = dict(mode=i['drift_mode'], fit_years=i['drift_fit_years'],
                     drift_lin_mm_per_yr=i['drift_lin_mm_per_yr'],
                     trend_1955_2014_raw=rawt, trend_1955_2014_lin=lin,
                     trend_1955_2014_quad=quad, quad_minus_lin=quad - lin)
    S['drift'] = dict(
        per_model=dr,
        n_parallel=sum(v['mode'].startswith('parallel') for v in dr.values()),
        mode_counts={k: sum(v['mode'] == k for v in dr.values())
                     for k in sorted({v['mode'] for v in dr.values()})},
        drift_lin_mm_per_yr=q3([v['drift_lin_mm_per_yr'] for v in dr.values()]),
        abs_drift_over_trend=q3([abs(v['drift_lin_mm_per_yr']) / v['trend_1955_2014_lin']
                                 for v in dr.values()]),
        quad_minus_lin_1955_2014=q3([v['quad_minus_lin'] for v in dr.values()]),
        max_abs_quad_minus_lin=max(abs(v['quad_minus_lin']) for v in dr.values()),
        trend_1955_2014_lin=q3([v['trend_1955_2014_lin'] for v in dr.values()]),
        trend_1955_2014_quad=q3([v['trend_1955_2014_quad'] for v in dr.values()]))

    # ---- multi-model band
    yrs = np.arange(1955, 2025)
    band = {}
    for lab, w in (('rebased_1995_2005', (1995, 2005)), ('rebased_2005_2009', (2005, 2009))):
        M = np.array([Z[m].reindex(yrs).values - Z[m].loc[w[0]:w[1]].mean() for m in sorted(post)])
        band[lab] = dict(years=yrs.tolist(), median=np.median(M, 0).round(3).tolist(),
                         p5=np.percentile(M, 5, 0).round(3).tolist(),
                         p95=np.percentile(M, 95, 0).round(3).tolist(), n=len(M))
    S['band'] = band

    # ---- trends
    py, full, upper, two = production()
    tr = {}
    for w in TREND_WINDOWS:
        key = f'{w[0]}-{w[1]}'
        mm = [m for m in models if (w[1] <= 2014 or m in post) and win(Z[m], w) is not None]
        per = {m: _trend(np.arange(w[0], w[1] + 1.), win(Z[m], w).values) for m in mm}
        k = (py >= w[0]) & (py <= w[1])
        t = py[k].astype(float)
        X = np.c_[np.ones_like(t), t - t.mean()]
        sl_f = np.linalg.lstsq(X, full[:, k].T, rcond=None)[0][1]
        sl_u = np.linalg.lstsq(X, upper[:, k].T, rcond=None)[0][1]
        sl_2 = np.linalg.lstsq(X, two[:, k].T, rcond=None)[0][1]
        no_mri = [v[0] for m, v in per.items() if m != 'MRI-ESM2-0']
        o = obs['trends_mm_per_yr']
        row = dict(n=len(mm), per_model=per, cmip6=q3([v[0] for v in per.values()]),
                   cmip6_without_MRI_ESM2_0=q3(no_mri),
                   production_full=q3(sl_f), production_0_700=q3(sl_u),
                   production_0_2000=q3(sl_2),
                   obs_0_700={r: o[f'0-700m_{key}'][r] for r in ('noaa', 'en4', 'iap')})
        row['obs_0_700']['noaa_x1.14'] = [x * NOAA_SCALE for x in o[f'0-700m_{key}']['noaa']]
        if f'0-2000m_{key}' in o:
            o2 = {r: o[f'0-2000m_{key}'][r] for r in ('noaa', 'en4', 'iap')}
            o2['noaa_x1.14'] = [x * NOAA_SCALE for x in o2['noaa']]
            row['obs_0_2000'] = o2
            row['obs_full_depth_est'] = {
                r: [v[0] + BELOW_2000[0], float(np.hypot(v[1], 1.645 * BELOW_2000[1]))]
                for r, v in o2.items()}
        tr[key] = row
    S['trends_mm_per_yr'] = tr

    # ---- per-degree rise
    be = berkeley_annual()
    pdg = {}
    (a0, a1), (b0, b1) = PD_WINDOWS
    per = {}
    for m in models:
        z, t = Z[m], T[m]
        ws = [win(x, w) for x in (z, t) for w in PD_WINDOWS]
        if m not in post or any(v is None for v in ws):
            continue
        dH = ws[1].mean() - ws[0].mean()
        dT = ws[3].mean() - ws[2].mean()
        per[m] = dict(dH_mm=float(dH), dT_K=float(dT), mm_per_K=float(dH / dT))
    i0 = (py >= a0) & (py <= a1)
    i1 = (py >= b0) & (py <= b1)
    dHp = full[:, i1].mean(1) - full[:, i0].mean(1)
    dTb = float(be.loc[b0:b1].mean() - be.loc[a0:a1].mean())
    dHu = upper[:, i1].mean(1) - upper[:, i0].mean(1)
    pdg['window_means'] = dict(
        windows=[list(PD_WINDOWS[0]), list(PD_WINDOWS[1])], per_model=per, n=len(per),
        cmip6_mm_per_K=q3([v['mm_per_K'] for v in per.values()]),
        cmip6_dH_mm=q3([v['dH_mm'] for v in per.values()]),
        cmip6_dT_K=q3([v['dT_K'] for v in per.values()]),
        production_dH_mm=q3(dHp), berkeley_dT_K=dTb, production_mm_per_K=q3(dHp / dTb),
        production_0_700_mm_per_K=q3(dHu / dTb))
    # 1971-2014: ratio of OLS trends
    w = PD_TREND
    per2 = {}
    tt = np.arange(w[0], w[1] + 1.)
    for m in models:
        if win(Z[m], w) is None or win(T[m], w) is None:
            continue
        sh = _trend(tt, win(Z[m], w).values)[0]
        st = _trend(tt, win(T[m], w).values)[0]
        per2[m] = dict(dHdt=sh, dTdt=st, mm_per_K=sh / st)
    k = (py >= w[0]) & (py <= w[1])
    X = np.c_[np.ones(k.sum()), py[k] - py[k].mean()]
    shp = np.linalg.lstsq(X, full[:, k].T, rcond=None)[0][1]
    stb = _trend(tt, be.loc[w[0]:w[1]].values)[0]
    pdg['trend_ratio_1971_2014'] = dict(
        per_model=per2, n=len(per2), cmip6_mm_per_K=q3([v['mm_per_K'] for v in per2.values()]),
        cmip6_dHdt=q3([v['dHdt'] for v in per2.values()]),
        cmip6_dTdt=q3([v['dTdt'] for v in per2.values()]),
        production_dHdt=q3(shp), berkeley_dTdt=stb, production_mm_per_K=q3(shp / stb))
    # future: 2081-2100 minus 1995-2014 under SSP2-4.5
    gm, off = ar6_ssp245_berkeley()
    (f0, f1), (g0, g1) = PD_FUTURE
    per3 = {}
    for m in post:
        ws = [win(x, w) for x in (Z[m], T[m]) for w in PD_FUTURE]
        if any(v is None for v in ws):
            continue
        dH = ws[1].mean() - ws[0].mean()
        dT = ws[3].mean() - ws[2].mean()
        per3[m] = dict(dH_mm=float(dH), dT_K=float(dT), mm_per_K=float(dH / dT))
    j0 = (py >= f0) & (py <= f1)
    j1 = (py >= g0) & (py <= g1)
    dHf = full[:, j1].mean(1) - full[:, j0].mean(1)
    dTf = float(gm.loc[g0:g1].mean() - gm.loc[f0:f1].mean())
    # Each production member follows its own AR6 percentile path (warming_paths,
    # same seed and member order as component_ocean); its warming is the median
    # path's plus its own offset, which is zero through 2024 and held flat to 2100.
    from warming_paths import warming_z, ssp_row_offsets
    ar = pd.read_hdf(H5_PROC, 'projections/temp/SSP2_4_5')
    offs = pd.DataFrame(ssp_row_offsets(ar, warming_z(full.shape[0])).T,
                        index=ar['decimal_year'].astype(int).values)
    offs.loc[2100] = offs.loc[2099]
    dTi = dTf + offs.loc[g0:g1].mean().values - offs.reindex(range(f0, f1 + 1)).fillna(0).mean().values
    pdg['ssp245_2081_2100_vs_1995_2014'] = dict(
        per_model=per3, n=len(per3), cmip6_mm_per_K=q3([v['mm_per_K'] for v in per3.values()]),
        cmip6_dH_mm=q3([v['dH_mm'] for v in per3.values()]),
        cmip6_dT_K=q3([v['dT_K'] for v in per3.values()]),
        production_dH_mm=q3(dHf), ar6_median_dT_K=dTf, ar6_offset_K=off,
        production_dT_K=q3(dTi), production_mm_per_K=q3(dHf / dTi),
        production_mm_per_K_at_median_dT=q3(dHf / dTf),
        note='production_mm_per_K divides each member by its own AR6 percentile-path '
             'warming; the _at_median_dT variant divides every member by the median')

    # AR6 reference (local FACTS files): thermal expansion samples (oceandynamics-tlm,
    # ssp245, base year 2005 = 1995-2014 mean, decadal) and the AR6 two-layer model GMST
    # samples (twolayer_SSPs.h5, annual from 1750). Samples are not assumed paired, so
    # the per-degree value is a ratio of medians.
    ar6 = {}
    try:
        fz = AR6_DIR / 'global/full_sample_components/oceandynamics-tlm-oceandynamics-ssp245_globalsl.nc'
        with xr.open_dataset(fz) as ds:
            yz = ds['years'].values
            hz = ds['sea_level_change'].values.squeeze().astype(float)
            included = ds.attrs.get('comment', '')
        ya = np.arange(g0, g1 + 1)
        h8 = np.array([np.interp(ya, yz, r).mean() for r in hz])   # 2081-2100 mean, mm rel. 2005
        h21 = hz[:, list(yz).index(2100)]
        with h5py.File(AR6_DIR / 'tlmdata/twolayer_SSPs.h5', 'r') as f:
            Tt = f['ssp245/surface_temperature'][:]
            ecs6, tcr6 = f['ssp245/ECS'][:], f['ssp245/TCR'][:]
        yt = np.arange(1750, 1750 + Tt.shape[0])
        dT6 = Tt[(yt >= g0) & (yt <= g1)].mean(0) - Tt[(yt >= f0) & (yt <= f1)].mean(0)
        ar6 = dict(source=str(fz.relative_to(ROOT)), included_models_comment=included,
                   dH_2081_2100_mm=q3(h8), dH_2100_mm=q3(h21), tlm_dT_K=q3(dT6),
                   mm_per_K_ratio_of_medians=float(np.median(h8) / np.median(dT6)),
                   tlm_ECS=q3(ecs6), tlm_TCR=q3(tcr6))
        k0 = (py >= f0) & (py <= f1)
        i21 = int(np.where(py == 2100)[0][0])
        ar6['production_dH_2100_rel_1995_2014_mm'] = q3(full[:, i21] - full[:, k0].mean(1))
        ar6['production_dH_2081_2100_rel_1995_2014_mm'] = q3(dHf)
    except Exception as e:  # noqa: BLE001
        ar6 = dict(error=f'{type(e).__name__}: {e}')
    pdg['ar6_reference'] = ar6

    # ECS (Zelinka et al. 2020, GitHub JSON)
    ecs_path = dl / 'zelinka_ecs.json'
    if not ecs_path.exists():
        urllib.request.urlretrieve(ECS_URL, ecs_path)
    E = json.loads(ecs_path.read_text())['CMIP6']
    ecs = {}
    for m in models:
        if m in E:
            mem = prov['models'][m]['member_id']
            v = E[m].get(mem) or E[m][sorted(E[m], key=member_key)[0]]
            ecs[m] = float(v['ECS'])
    pdg['ecs_zelinka2020'] = ecs
    from scipy import stats
    pm = pdg['window_means']['per_model']
    mm = [m for m in pm if m in ecs]
    x = np.array([ecs[m] for m in mm])
    rho_ecs = stats.spearmanr(x, [pm[m]['mm_per_K'] for m in mm])
    dTs = np.array([pm[m]['dT_K'] for m in pm])
    dHs = np.array([pm[m]['dH_mm'] for m in pm])
    rk = np.array([pm[m]['mm_per_K'] for m in pm])
    lr = stats.linregress(dTs, dHs)
    pdg['cross_model'] = dict(
        spearman_mm_per_K_vs_ECS=[float(rho_ecs.statistic), float(rho_ecs.pvalue), len(mm)],
        spearman_mm_per_K_vs_dT=[float(stats.spearmanr(dTs, rk).statistic),
                                 float(stats.spearmanr(dTs, rk).pvalue), len(pm)],
        spearman_dH_vs_ECS=[float(stats.spearmanr(x, [pm[m]['dH_mm'] for m in mm]).statistic),
                            float(stats.spearmanr(x, [pm[m]['dH_mm'] for m in mm]).pvalue)],
        spearman_dT_vs_ECS=[float(stats.spearmanr(x, [pm[m]['dT_K'] for m in mm]).statistic),
                            float(stats.spearmanr(x, [pm[m]['dT_K'] for m in mm]).pvalue)],
        ols_dH_on_dT=dict(slope_mm_per_K=float(lr.slope), slope_90=float(student_t.ppf(0.95, len(dTs) - 2) * lr.stderr),
                          intercept_mm=float(lr.intercept), r=float(lr.rvalue),
                          p=float(lr.pvalue)))
    S['per_degree'] = pdg
    Path(a.summary_json).write_text(json.dumps(S, indent=1))
    log(f'wrote {a.summary_json}')
    make_figure(S, obs, py, full, upper, a.figure)


def make_figure(S, obs, py, full, upper, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    C = dict(noaa='#2a78d6', en4='#eb6834', iap='#1baf7a', prod='#6250d6', cmip='#8a8986',
             ink='#0b0b0b', ink2='#52514e', grid='#e4e3df')
    plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': C['ink2'], 'axes.labelcolor': C['ink'],
                         'xtick.color': C['ink2'], 'ytick.color': C['ink2'],
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 6.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
    b = S['band']['rebased_1995_2005']
    y = np.array(b['years'])
    ax.fill_between(y, b['p5'], b['p95'], color=C['cmip'], alpha=0.28, lw=0,
                    label=f'CMIP6 full depth, 5–95% (n={b["n"]})')
    ax.plot(y, b['median'], color=C['cmip'], lw=2, label='CMIP6 full depth, median')
    for r, lab in (('noaa', 'NOAA 0–700 m'), ('en4', 'EN4 0–700 m'), ('iap', 'IAP 0–700 m')):
        s = obs['series'][r]
        yy = np.array(s['years_700'])
        k = (yy >= 1955) & (yy <= 2024)
        ax.plot(yy[k], np.array(s['h700_mm'])[k], color=C[r], lw=1.3, label=lab)
    s = obs['series']['noaa']
    yy = np.array(s['years_700'])
    k = (yy >= 1955) & (yy <= 2024)
    ax.plot(yy[k], NOAA_SCALE * np.array(s['h700_mm'])[k], color=C['noaa'], lw=1.3, ls='--',
            label='NOAA 0–700 m ×1.14')
    kk = (py >= 1955) & (py <= 2024)
    pf = np.median(full[:, kk], 0)
    ref = (py[kk] >= 1995) & (py[kk] <= 2005)
    ax.plot(py[kk], pf - pf[ref].mean(), color=C['prod'], lw=2,
            label='This study, full depth (median)')
    pu = np.median(upper[:, kk], 0)
    ax.plot(py[kk], pu - pu[ref].mean(), color=C['prod'], lw=1.3, ls=':',
            label='This study, 0–700 m (median)')
    ax.axhline(0, color=C['grid'], lw=0.8, zorder=0)
    ax.set_xlim(1955, 2024.5)
    ax.set_ylabel('Thermosteric sea level (mm, rel. 1995–2005)')
    ax.set_title('(a) Global thermosteric sea level', loc='left', color=C['ink'])
    ax.legend(frameon=False, fontsize=7.3, loc='upper left')
    ax.grid(axis='y', color=C['grid'], lw=0.6)

    pd1 = S['per_degree']['window_means']
    pd2 = S['per_degree']['trend_ratio_1971_2014']
    ms = sorted(pd1['per_model'], key=lambda m: pd1['per_model'][m]['mm_per_K'])
    yv = np.arange(len(ms))
    p = pd1['production_mm_per_K']
    bx.axvspan(p[1], p[2], color=C['prod'], alpha=0.15, lw=0)
    bx.axvline(p[0], color=C['prod'], lw=2,
               label=f'This study, 1955–74→2005–24: {p[0]:.0f} [{p[1]:.0f}, {p[2]:.0f}] mm/°C')
    p2 = pd2['production_mm_per_K']
    bx.axvline(p2[0], color=C['prod'], lw=1.3, ls='--',
               label=f'This study, 1971–2014 trend ratio: {p2[0]:.0f} [{p2[1]:.0f}, {p2[2]:.0f}] mm/°C')
    bx.scatter([pd1['per_model'][m]['mm_per_K'] for m in ms], yv, s=26, color=C['ink'],
               zorder=3, label='CMIP6, 1955–74→2005–24')
    x2 = [pd2['per_model'][m]['mm_per_K'] if m in pd2['per_model'] else np.nan for m in ms]
    bx.scatter(x2, yv, s=26, facecolors='none', edgecolors=C['ink2'], zorder=3,
               label='CMIP6, 1971–2014 trend ratio')
    bx.set_yticks(yv)
    bx.set_yticklabels(ms, fontsize=6.6)
    bx.set_xlabel('Thermosteric rise per degree of GMST (mm/°C), full depth')
    bx.set_title('(b) Historical rise per degree of warming', loc='left', color=C['ink'])
    bx.grid(axis='x', color=C['grid'], lw=0.6)
    bx.legend(frameon=False, fontsize=6.8, loc='upper center', bbox_to_anchor=(0.45, -0.12), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    log(f'wrote {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fetch')
    f.add_argument('--download-dir', required=True)
    f.add_argument('--out-csv', required=True)
    f.add_argument('--models', nargs='*')
    f.add_argument('--workers', type=int, default=6)
    g = sub.add_parser('analyze')
    g.add_argument('--out-csv', required=True)
    g.add_argument('--download-dir', required=True)
    g.add_argument('--summary-json', required=True)
    g.add_argument('--figure', required=True)
    a = ap.parse_args()
    if a.cmd == 'fetch':
        cmd_fetch(a)
    else:
        cmd_analyze(a)


if __name__ == '__main__':
    main()
