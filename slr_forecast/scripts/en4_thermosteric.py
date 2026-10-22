"""Global thermosteric sea level from the EN4.2.2 objective analyses (TEOS-10).

Input to the calibration-record sensitivity test of the two-layer
thermosteric model (scripts/thermosteric_record_sensitivity.py): EN4 in place
of the NOAA (Levitus et al. 2012) records.  Writes only to the paths given on
the command line.

Method
------
For each monthly analysis, potential temperature (K) and practical salinity
are converted to Absolute Salinity SA and Conservative Temperature CT with
gsw, at pressure p(z, lat).  Salinity is held at a fixed climatology SA_c
(EN4 1971-2000 mean, the period EN4 relaxes to; Good et al. 2013, para 28),
so the thermosteric height is

    h_T(t) = sum_cells A dz [rho(SA_c, CT_c, p) / rho(SA_c, CT(t), p) - 1] / A_surface

(the specific-volume ratio; equal to -drho/rho to first order).  The
halosteric term uses rho(SA(t), CT_c) instead (information only).  A cell
enters only if it is wet in every month of the climatology window.  Layer
integrals use the EN4 depth_bnds, with the level containing the layer base
cut there (700 m inside 595.9-725.2 m; 2000 m leaves a 0.59 m sliver of the
level starting at 1999.41 m).  Columns shallower than the layer base are
integrated to their deepest wet level.

Per-level quantities are stored per unit thickness, so any layer can be formed
afterwards:  h_layer = sum_k dz_layer[k] * hT_lev[k].

Regions (areas divided by the region's surface area):
  global : every wet surface cell
  li     : columns with a wet level below 700 m, south of 66 N (the area of
           Li et al. 2022, about 89% of the ocean)

Subcommands
-----------
  clim     climatological mean potential temperature and salinity, 1971-2000
           and 1995-2005 (pass 1)
  series   per-year monthly per-level thermosteric, halosteric and CT anomaly
           sums; annual-mean CT anomaly fields 0-700 m (pass 2; resumable,
           skips years already written)
  annual   annual-mean layer series (0-700 m, 0-2000 m) of thermosteric and
           halosteric height (mm) and heat content (ZJ, rho0 * cp0 * CT), for
           complete years, written to one CSV (pass 3)

Usage
-----
  python -I scripts/en4_thermosteric.py clim   --zip-dirs DIR [DIR ...] --out OUT
  python -I scripts/en4_thermosteric.py series --zip-dirs DIR [DIR ...] --out OUT [--workers N]
  python -I scripts/en4_thermosteric.py annual --out OUT --csv data/processed/en4_thermosteric_annual.csv

When the same year is present in several directories, the archive with the
most monthly members wins (ties: the later directory).  Files ending in .part
or .lock are ignored.
"""
import argparse
import re
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gsw
import netCDF4 as nc
import numpy as np

Z_MAX = 2000.0          # m, deepest layer base used
H_UPPER = 700.0         # m
CLIM_WINDOWS = {'7100': (1971, 2000), '9505': (1995, 2005)}
LI_LAT_MAX = 66.0


def find_archives(zip_dirs):
    """{year: (path, member names)} for complete EN4 analysis archives."""
    found = {}
    pat = re.compile(r'EN\.4\.2\.2\.analyses\.g10\.(\d{4})\.zip$')
    for d in zip_dirs:
        for p in sorted(Path(d).iterdir()):
            m = pat.match(p.name)
            if not m:
                continue
            try:
                with zipfile.ZipFile(p) as zf:
                    names = sorted(n for n in zf.namelist() if n.endswith('.nc'))
            except zipfile.BadZipFile:
                print(f'  skip (bad zip): {p}', file=sys.stderr)
                continue
            yr = int(m.group(1))
            if yr not in found or len(names) >= len(found[yr][1]):
                found[yr] = (p, names)
    return dict(sorted(found.items()))


def _grid(zip_path, name):
    with zipfile.ZipFile(zip_path) as zf:
        ds = nc.Dataset('g.nc', memory=zf.read(name))
        depth = ds.variables['depth'][:].data.astype(float)
        bnds = ds.variables['depth_bnds'][:].data.astype(float)
        lat = ds.variables['lat'][:].data.astype(float)
        lon = ds.variables['lon'][:].data.astype(float)
        ds.close()
    k = int(np.searchsorted(bnds[:, 0], Z_MAX))     # levels starting above Z_MAX
    return depth[:k], bnds[:k], lat, lon


def _read(zf, name, k):
    ds = nc.Dataset('m.nc', memory=zf.read(name))
    pt = ds.variables['temperature'][0, :k].filled(np.nan).astype(float) - 273.15
    sp = ds.variables['salinity'][0, :k].filled(np.nan).astype(float)
    ds.close()
    return pt, sp


def cmd_clim(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arch = find_archives(args.zip_dirs)
    yr0 = next(iter(arch))
    depth, bnds, lat, lon = _grid(arch[yr0][0], arch[yr0][1][0])
    k = len(depth)
    res = {'depth': depth, 'depth_bnds': bnds, 'lat': lat, 'lon': lon}
    for tag, (y0, y1) in CLIM_WINDOWS.items():
        s_pt = np.zeros((k, lat.size, lon.size))
        s_sp = np.zeros_like(s_pt)
        cnt = np.zeros_like(s_pt)
        nmon = 0
        for yr in range(y0, y1 + 1):
            if yr not in arch or len(arch[yr][1]) != 12:
                raise SystemExit(f'climatology {tag}: year {yr} missing or incomplete')
            with zipfile.ZipFile(arch[yr][0]) as zf:
                for name in arch[yr][1]:
                    pt, sp = _read(zf, name, k)
                    ok = np.isfinite(pt) & np.isfinite(sp)
                    s_pt += np.where(ok, pt, 0)
                    s_sp += np.where(ok, sp, 0)
                    cnt += ok
                    nmon += 1
            print(f'  clim {tag}: {yr}', flush=True)
        wet = cnt == nmon
        res[f'pt_{tag}'] = np.where(wet, s_pt / np.maximum(cnt, 1), np.nan)
        res[f'sp_{tag}'] = np.where(wet, s_sp / np.maximum(cnt, 1), np.nan)
        res[f'nmon_{tag}'] = nmon
        res[f'partial_wet_{tag}'] = int(((cnt > 0) & ~wet).sum())
    np.savez_compressed(out / 'en4_clim.npz', **res)
    print(f'wrote {out / "en4_clim.npz"}')


def _setup(clim_path):
    c = np.load(clim_path)
    depth, bnds, lat, lon = c['depth'], c['depth_bnds'], c['lat'], c['lon']
    p = gsw.p_from_z(-depth[:, None, None], lat[None, :, None]) * np.ones((1, 1, lon.size))
    lon3, lat3 = np.meshgrid(lon, lat)
    sa_c = gsw.SA_from_SP(c['sp_7100'], p, lon3[None], lat3[None])
    ct_c = gsw.CT_from_pt(sa_c, c['pt_7100'])
    rho_c = gsw.rho(sa_c, ct_c, p)
    wet = np.isfinite(rho_c)
    area = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size))  # relative
    k700 = int(np.searchsorted(bnds[:, 0], H_UPPER)) - 1    # level containing 700 m
    deep700 = wet[k700 + 1] if k700 + 1 < wet.shape[0] else wet[k700]
    regions = {
        'global': wet[0],
        'li': wet[0] & deep700 & (lat3 < LI_LAT_MAX),
    }
    return dict(depth=depth, bnds=bnds, lat=lat, lon=lon, p=p, lon3=lon3, lat3=lat3,
                sa_c=sa_c, ct_c=ct_c, rho_c=rho_c, wet=wet, area=area, regions=regions,
                k_up=k700 + 1)


_S = None


def _init(clim_path):
    global _S
    _S = _setup(clim_path)


def _year(job):
    yr, zpath, names, out_dir = job
    S = _S
    k = S['depth'].size
    nreg = len(S['regions'])
    hT = np.zeros((len(names), nreg, k))
    hH = np.zeros_like(hT)
    dct = np.zeros_like(hT)
    areas = np.zeros((nreg, k))
    surf = np.zeros(nreg)
    for r, m in enumerate(S['regions'].values()):
        surf[r] = (S['area'] * m).sum()
        areas[r] = (S['area'][None] * (m[None] & S['wet'])).sum(axis=(1, 2))
    ku = S['k_up']
    ct_ann = np.zeros((ku,) + S['area'].shape)
    n_fill = 0
    with zipfile.ZipFile(zpath) as zf:
        for i, name in enumerate(names):
            pt, sp = _read(zf, name, k)
            sa = gsw.SA_from_SP(sp, S['p'], S['lon3'][None], S['lat3'][None])
            ct = gsw.CT_from_pt(sa, pt)
            ok = S['wet'] & np.isfinite(ct) & np.isfinite(sa)
            n_fill += int((S['wet'] & ~ok).sum())
            ct = np.where(ok, ct, S['ct_c'])
            sa = np.where(ok, sa, S['sa_c'])
            vT = S['rho_c'] / gsw.rho(S['sa_c'], ct, S['p']) - 1.0
            vH = S['rho_c'] / gsw.rho(sa, S['ct_c'], S['p']) - 1.0
            d = ct - S['ct_c']
            for r, m in enumerate(S['regions'].values()):
                w = S['area'][None] * (m[None] & S['wet'])
                hT[i, r] = np.nansum(vT * w, axis=(1, 2)) / surf[r]
                hH[i, r] = np.nansum(vH * w, axis=(1, 2)) / surf[r]
                dct[i, r] = np.nansum(d * w, axis=(1, 2)) / np.maximum(areas[r], 1e-30)
            ct_ann += np.where(S['wet'][:ku], d[:ku], 0.0)
    ct_ann = np.where(S['wet'][:ku], ct_ann / len(names), np.nan).astype(np.float32)
    np.savez_compressed(Path(out_dir) / f'{yr}.npz', hT=hT, hH=hH, dct=dct, areas=areas,
                        surf=surf, n_months=len(names), n_fill=n_fill, source=str(zpath),
                        members=np.array(names), regions=np.array(list(S['regions'])))
    np.save(Path(out_dir) / f'ctanom_0_700_{yr}.npy', ct_ann)
    return yr, len(names), n_fill


def cmd_series(args):
    out = Path(args.out)
    ydir = out / 'years'
    ydir.mkdir(parents=True, exist_ok=True)
    arch = find_archives(args.zip_dirs)
    jobs = [(yr, str(p), names, str(ydir)) for yr, (p, names) in arch.items()
            if yr >= args.first_year and not (ydir / f'{yr}.npz').exists()]
    print(f'{len(jobs)} years to process: {[j[0] for j in jobs]}', flush=True)
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(str(out / 'en4_clim.npz'),)) as ex:
        for yr, nm, nf in ex.map(_year, jobs):
            print(f'  {yr}: {nm} months, {nf} wet cell-months filled with climatology',
                  flush=True)


RHO0 = 1025.0                    # kg m-3, reference density for heat content
CP0 = 3991.86795711963          # J kg-1 K-1, TEOS-10 cp0; cp0 * CT is potential enthalpy
DEG2_M2 = (np.pi / 180 * 6.371e6)**2   # _setup areas are cos(lat) in square degrees


def cmd_annual(args):
    """Annual-mean layer series from the per-year files of `series`."""
    import pandas as pd
    out = Path(args.out)
    bnds = np.load(out / 'en4_clim.npz')['depth_bnds']
    dz = {H: np.minimum(bnds[:, 1], H) - np.minimum(bnds[:, 0], H) for H in (700.0, 2000.0)}
    rows = []
    for f in sorted((out / 'years').glob('[0-9][0-9][0-9][0-9].npz')):
        z = np.load(f)
        if int(z['n_months']) != 12:
            continue
        r = {'year': int(f.stem), 'n_fill': int(z['n_fill'])}
        for ri, rn in enumerate(z['regions']):
            sfx = '' if rn == 'global' else f'_{rn}'
            for H in (700, 2000):
                d = dz[float(H)]
                r[f'h{H}_mm{sfx}'] = (z['hT'][:, ri] @ d).mean() * 1e3
                r[f'hH{H}_mm{sfx}'] = (z['hH'][:, ri] @ d).mean() * 1e3
                r[f'ohc{H}_zj{sfx}'] = (RHO0 * CP0 * (z['dct'][:, ri] * z['areas'][ri] * DEG2_M2 * d)
                                        .sum(1).mean() / 1e21)
        rows.append(r)
    pd.DataFrame(rows).set_index('year').to_csv(args.csv, float_format='%.6g')
    print(f'wrote {args.csv} ({len(rows)} years, {rows[0]["year"]}-{rows[-1]["year"]})')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('clim', 'series'):
        s = sub.add_parser(name)
        s.add_argument('--zip-dirs', nargs='+', required=True)
        s.add_argument('--out', required=True)
        s.add_argument('--workers', type=int, default=6)
        s.add_argument('--first-year', type=int, default=1950)
    s = sub.add_parser('annual')
    s.add_argument('--out', required=True)
    s.add_argument('--csv', required=True)
    args = ap.parse_args()
    {'clim': cmd_clim, 'series': cmd_series, 'annual': cmd_annual}[args.cmd](args)


if __name__ == '__main__':
    main()
