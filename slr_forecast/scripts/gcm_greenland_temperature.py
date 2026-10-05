"""Annual GMST and Greenland near-surface temperature for the CMIP6 runs that
drive the MARv3.12 PROTECT ensemble (see extract_mar_protect_smb.py).

Source: Pangeo CMIP6 cloud archive (Amon tas, fx sftlf), read as zarr over
HTTPS from storage.googleapis.com.  Greenland temperature follows Noel et al.
(2021, Text S1): 2-m air temperature averaged over grid cells with at least
50% land between 72W-12W and 59N-85N, area weighted.  GMST is the
area-weighted global mean.

Output: data/raw/ice_sheets/greenland/mar_protect/gcm_temperature.csv
  columns: gcm, member, experiment, year, gmst_K, tgris_K
  Runs already in the file are skipped and new runs are appended, so
  existing values are not recomputed.  Members follow the tas files in the
  PROTECT directory.

Usage:  python scripts/gcm_greenland_temperature.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
CATALOG = 'https://storage.googleapis.com/cmip6/pangeo-cmip6.csv'
OUT = ROOT / 'data/raw/ice_sheets/greenland/mar_protect/gcm_temperature.csv'

RUNS = [('CESM2', 'r11i1p1f1', ['historical', 'ssp126', 'ssp245', 'ssp585']),
        ('MPI-ESM1-2-HR', 'r1i1p1f1', ['historical', 'ssp126', 'ssp245', 'ssp585']),
        ('NorESM2-MM', 'r1i1p1f1', ['historical', 'ssp245', 'ssp585']),
        ('UKESM1-0-LL', 'r1i1p1f2', ['historical', 'ssp245', 'ssp585']),
        ('CNRM-CM6-1', 'r1i1p1f2', ['historical', 'ssp585']),
        ('CNRM-ESM2-1', 'r1i1p1f2', ['historical', 'ssp585']),
        ('IPSL-CM6A-LR', 'r1i1p1f1', ['historical', 'ssp585'])]


def https(zstore):
    return zstore.replace('gs://', 'https://storage.googleapis.com/')


def open_zarr(zstore):
    return xr.open_zarr(https(zstore), consolidated=True, decode_times=True)


def area_weights(ds):
    lat = ds['lat']
    w = np.cos(np.deg2rad(lat))
    return w / w.mean()


def main():
    cat = pd.read_csv(CATALOG, usecols=['source_id', 'experiment_id', 'member_id',
                                        'table_id', 'variable_id', 'zstore', 'version'])
    old = pd.read_csv(OUT) if OUT.exists() else None
    have = set() if old is None else set(zip(old.gcm, old.member, old.experiment))
    rows = []
    for gcm, member, exps in RUNS:
        exps = [e for e in exps if (gcm, member, e) not in have]
        if not exps:
            continue
        lf = cat[(cat.source_id == gcm) & (cat.variable_id == 'sftlf') & (cat.table_id == 'fx')]
        if lf.empty:
            raise RuntimeError(f'no sftlf for {gcm}')
        lf = lf.sort_values('member_id')
        sftlf = open_zarr(lf.iloc[0].zstore)['sftlf'].load()
        if float(sftlf.max()) > 1.5:
            sftlf = sftlf / 100.0
        for exp in exps:
            c = cat[(cat.source_id == gcm) & (cat.experiment_id == exp) & (cat.member_id == member)
                    & (cat.table_id == 'Amon') & (cat.variable_id == 'tas')]
            if c.empty:
                raise RuntimeError(f'missing tas {gcm} {exp} {member}')
            c = c.sort_values('version')
            ds = open_zarr(c.iloc[-1].zstore)
            tas = ds['tas']
            yrs = tas['time'].dt.year
            keep = (yrs >= 1950) & (yrs <= 2100)
            tas = tas.isel(time=np.where(keep.values)[0])
            w = area_weights(ds)
            lon = ds['lon'] % 360
            box = ((ds['lat'] >= 59) & (ds['lat'] <= 85)
                   & (lon >= 288) & (lon <= 348) & (sftlf.reindex_like(tas.isel(time=0), method='nearest') >= 0.5))
            wb = (w * box).broadcast_like(tas.isel(time=0))
            wg = w.broadcast_like(tas.isel(time=0))
            gmst = (tas * wg).sum(['lat', 'lon']) / wg.sum()
            tg = (tas * wb).sum(['lat', 'lon']) / wb.sum()
            ann = pd.DataFrame({'gmst_K': gmst.groupby('time.year').mean().values,
                                'tgris_K': tg.groupby('time.year').mean().values},
                               index=np.unique(tas['time'].dt.year.values))
            for y, r in ann.iterrows():
                rows.append((gcm, member, exp, int(y), r.gmst_K, r.tgris_K))
            print(f'{gcm} {exp}: {ann.index.min()}-{ann.index.max()}, '
                  f'{int(wb.astype(bool).sum())} Greenland cells', flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows, columns=['gcm', 'member', 'experiment', 'year', 'gmst_K', 'tgris_K'])
    if new.empty:
        print('nothing new to add')
        return
    new.to_csv(OUT, mode='a', header=old is None, index=False, float_format='%.4f')
    print(f'appended {len(new)} rows to {OUT}')


if __name__ == '__main__':
    main()
