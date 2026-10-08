"""Greenland summer (JJA) near-surface temperature for the CMIP6 runs that
drive the MARv3.12 PROTECT ensemble.

Companion to gcm_greenland_temperature.py (same runs, catalog and Greenland
box: grid cells with at least 50% land between 72W-12W and 59N-85N, area
weighted), but averaged over June-August instead of the calendar year.
Used by hindcast_greenland_t_emulator.py.

Output: data/raw/ice_sheets/greenland/mar_protect/gcm_temperature_jja.csv
  columns: gcm, member, experiment, year, tgris_jja_K
  Runs already in the file are skipped.

Usage:  python scripts/gcm_greenland_jja_temperature.py
"""

import numpy as np
import pandas as pd

from gcm_greenland_temperature import CATALOG, OUT as OUT_ANN, RUNS, open_zarr, area_weights

OUT = OUT_ANN.with_name('gcm_temperature_jja.csv')


def main():
    cat = pd.read_csv(CATALOG, usecols=['source_id', 'experiment_id', 'member_id',
                                        'table_id', 'variable_id', 'zstore', 'version'])
    old = pd.read_csv(OUT) if OUT.exists() else None
    have = set() if old is None else set(zip(old.gcm, old.member, old.experiment))
    for gcm, member, exps in RUNS:
        exps = [e for e in exps if (gcm, member, e) not in have]
        if not exps:
            continue
        lf = cat[(cat.source_id == gcm) & (cat.variable_id == 'sftlf') & (cat.table_id == 'fx')]
        sftlf = open_zarr(lf.sort_values('member_id').iloc[0].zstore)['sftlf'].load()
        if float(sftlf.max()) > 1.5:
            sftlf = sftlf / 100.0
        for exp in exps:
            c = cat[(cat.source_id == gcm) & (cat.experiment_id == exp) & (cat.member_id == member)
                    & (cat.table_id == 'Amon') & (cat.variable_id == 'tas')]
            if c.empty:
                raise RuntimeError(f'missing tas {gcm} {exp} {member}')
            ds = open_zarr(c.sort_values('version').iloc[-1].zstore)
            tas = ds['tas']
            yrs = tas['time'].dt.year
            tas = tas.isel(time=np.where(((yrs >= 1950) & (yrs <= 2100)).values)[0])
            w = area_weights(ds)
            lon = ds['lon'] % 360
            lf_g = sftlf.reindex_like(tas.isel(time=0), method='nearest')
            box = ((ds['lat'] >= 59) & (ds['lat'] <= 85) & (lon >= 288) & (lon <= 348)
                   & (lf_g >= 0.5))
            # Subset to the box's bounding rows/columns before loading
            ii = np.where(box.any('lon').values)[0]
            jj = np.where(box.any('lat').values)[0]
            sel = dict(lat=slice(ii.min(), ii.max() + 1), lon=slice(jj.min(), jj.max() + 1))
            t = tas.isel(**sel)
            wb = (w.isel(lat=sel['lat']) * box.isel(**sel)).broadcast_like(t.isel(time=0))
            jja = t.isel(time=np.where(t['time'].dt.month.isin([6, 7, 8]).values)[0]).load()
            tg = (jja * wb).sum(['lat', 'lon']) / wb.sum()
            ann = tg.groupby('time.year').mean().to_series()
            rows = pd.DataFrame({'gcm': gcm, 'member': member, 'experiment': exp,
                                 'year': ann.index.astype(int), 'tgris_jja_K': ann.values})
            rows.to_csv(OUT, mode='a', header=not OUT.exists(), index=False, float_format='%.4f')
            print(f'{gcm} {exp}: {ann.index.min()}-{ann.index.max()}, '
                  f'{int(box.sum())} Greenland cells', flush=True)


if __name__ == '__main__':
    main()
