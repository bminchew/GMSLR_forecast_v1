"""Extract annual ice-sheet SMB from the MARv3.12 multi-GCM ensemble (PROTECT).

Source: http://ftp.climato.be/fettweis/MARv3.12/Greenland/PROTECT/ -- one
monthly file per year (~1 GB each) on the ISMIP6 1-km grid (EPSG:3413).  Only
the SMB variable is read, via HTTP byte-range requests (fsspec + h5py), so
each year costs ~0.1 GB of transfer instead of ~1 GB.

Integration follows integrate_glaude2024_smb.py: Noel et al. (2021)
Promicemask mapped to the ISMIP6 grid, 3 = ice sheet (the emulator target),
1-3 = with peripheral glaciers (comparison only), true cell areas.  Validated
against the Glaude et al. (2024) MAR series (same MAR version, CESM2-Leo
forcing): 354.5 vs 353.6, -4.7 vs -4.4, -1300.7 vs -1294.0 Gt/yr in 2000, 2050
and 2090.

Output (appended row by row; reruns skip finished run-years):
  data/raw/ice_sheets/greenland/mar_protect/mar_protect_integrated_smb.csv
  columns: run, gcm, scenario, year, sheet, with_periph   (Gt/yr, mass gain)

Usage:
  python scripts/extract_mar_protect_smb.py [--workers 4] [--tier 1|2|3]
Tiers: 1 = GCMs with low-emission runs, 1980-2100;
       2 = SSP5-8.5-only GCMs, 1980-2100;
       3 = earlier historical years (1950-1979) for all GCMs.
"""

import argparse
import csv
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fsspec
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import integrate_glaude2024_smb as I

BASE = 'http://ftp.climato.be/fettweis/MARv3.12/Greenland/PROTECT'
OUT = I.ROOT / 'data/raw/ice_sheets/greenland/mar_protect/mar_protect_integrated_smb.csv'

# (directory prefix, GCM label, scenarios, first historical year on server)
TIER1 = [('CESM2-CMIP6', 'CESM2', ['ssp126', 'ssp245', 'ssp585'], 1948),
         ('MPI-ESM1-2-HR', 'MPI-ESM1-2-HR', ['ssp126', 'ssp245', 'ssp585'], 1950),
         ('NorESM2', 'NorESM2-MM', ['ssp245', 'ssp585'], 1950),
         ('UKESM1-0-LL-CMIP6', 'UKESM1-0-LL', ['ssp245', 'ssp585'], 1949)]
TIER2 = [('CNRM-CM6', 'CNRM-CM6-1', ['ssp585'], 1950),
         ('CNRM-ESM2', 'CNRM-ESM2-1', ['ssp585'], 1950),
         ('IPSL-CM6A-LR', 'IPSL-CM6A-LR', ['ssp585'], 1948),
         ('UKESM1-0-LL-Robin', 'UKESM1-0-LL (Robin)', ['ssp585'], 1960),
         ('CESM2-Leo', 'CESM2 (Leo branch)', ['ssp585'], 1950)]

_lock = threading.Lock()


def tasks(tier):
    out = []
    groups = TIER1 if tier == 1 else TIER2 if tier == 2 else TIER1 + TIER2
    for pre, gcm, scen, h0 in groups:
        if tier in (1, 2):
            out += [(f'{pre}-histo', gcm, 'historical', y) for y in range(1980, 2015)]
            for s in scen:
                out += [(f'{pre}-{s}', gcm, s, y) for y in range(2015, 2101)]
        else:
            out += [(f'{pre}-histo', gcm, 'historical', y) for y in range(max(h0, 1950), 1980)]
    return out


def done_keys():
    if not OUT.exists():
        return set()
    with open(OUT) as f:
        return {(r['run'], int(r['year'])) for r in csv.DictReader(f)}


def setup_grid():
    """Mask and areas on the ISMIP6 grid (taken from one MAR file)."""
    run = 'CESM2-CMIP6-ssp126'
    with fsspec.open(f'{BASE}/{run}/MARv3.12-monthly-{run}-2015.nc', 'rb') as f:
        h = h5py.File(f, 'r')
        X = h['x'][:].astype(float); Y = h['y'][:].astype(float)
    xc, yc, mr = I.racmo_grid()
    mask = I.map_mask(xc, yc, mr, X, Y)
    area = 1.0 / I.areal_scale(*np.meshgrid(X, Y))
    return X, Y, mask == 3, (mask >= 1) & (mask <= 3), area


def extract(task, grid, retries=4):
    run, gcm, scen, yr = task
    X, Y, sheet, periph, area = grid
    url = f'{BASE}/{run}/MARv3.12-monthly-{run}-{yr}.nc'
    for k in range(retries):
        try:
            with fsspec.open(url, 'rb', block_size=8 * 2**20) as f:
                h = h5py.File(f, 'r')
                if not (np.array_equal(h['x'][:], X) and np.array_equal(h['y'][:], Y)):
                    raise ValueError(f'grid mismatch in {url}')
                a = h['SMB'][:].astype(float)
            a = np.where(np.abs(a) > 1e30, np.nan, a)
            if a.shape[0] != 12:
                raise ValueError(f'{a.shape[0]} months in {url}')
            ann = a.sum(axis=0) * area                      # mm w.e. km^2 per year
            return (run, gcm, scen, yr,
                    np.nansum(ann[sheet]) * I.MM_KM2_TO_GT,
                    np.nansum(ann[periph]) * I.MM_KM2_TO_GT,
                    float(np.isnan(a[:, sheet]).mean()))
        except Exception as e:                             # network hiccups
            if k == retries - 1:
                raise
            time.sleep(30 * (k + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--tier', type=int, default=1)
    args = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tasks(args.tier) if (t[0], t[3]) not in done_keys()]
    print(f'tier {args.tier}: {len(todo)} run-years to extract', flush=True)
    grid = setup_grid()
    new = not OUT.exists()
    t0 = time.time(); n = 0
    with open(OUT, 'a', newline='') as fo:
        w = csv.writer(fo)
        if new:
            w.writerow(['run', 'gcm', 'scenario', 'year', 'sheet', 'with_periph', 'nan_frac'])
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(extract, t, grid): t for t in todo}
            for fu in as_completed(futs):
                try:
                    r = fu.result()
                except Exception as e:
                    print(f'FAILED {futs[fu]}: {e}', flush=True)
                    continue
                with _lock:
                    w.writerow([r[0], r[1], r[2], r[3], f'{r[4]:.2f}', f'{r[5]:.2f}', f'{r[6]:.4f}'])
                    fo.flush()
                n += 1
                if n % 10 == 0:
                    el = time.time() - t0
                    print(f'{n}/{len(todo)} done, {el / n:.0f} s per run-year, '
                          f'~{(len(todo) - n) * el / n / 3600:.1f} h left', flush=True)
    print('finished', flush=True)


if __name__ == '__main__':
    main()
