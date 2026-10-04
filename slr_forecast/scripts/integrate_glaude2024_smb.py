"""Integrate the Glaude et al. (2024) 1-km SMB grids to annual Greenland totals.

Inputs (Zenodo 10.5281/zenodo.13270554), kept outside Dropbox:
  MAR_SMB_1km.nc     ISMIP6 1-km grid (EPSG:3413), 1950-2100, mm w.e./yr,
                     forced by 'CESM2-Leo' (the branch run that also forced
                     the Noel et al. 2021 RACMO run)
  HIRHAM_SMB_1km.nc  ISMIP6 1-km grid, 1971-2100, mm w.e. accumulated per
                     year, forced by CESM2 historical/ssp585 r1i1p1f1
  RACMO_SMB_1km.nc   Noel 1-km EPSG:3413 grid (x/y are cell corners, offset
                     500 m from centres), 1950-2099
Mask: Promicemask from Noel et al. (2021) (Zenodo 4289959) on the RACMO grid;
3 = ice sheet, 1-2 = peripheral glaciers and ice caps.  Mapped to the ISMIP6
grid by nearest cell centre.  Only the ice sheet (3) enters the emulator,
because GlaMBIE already counts RGI region 05; the with-periphery total is kept
for comparison with the published Glaude numbers.

HIRHAM's missing margin cells (about 3% of the ice sheet) are filled from the
nearest valid cell; 'sheet_unfilled' keeps the raw total.

Cell areas are true areas, 1 km^2 divided by the EPSG:3413 areal scale factor.

Output: data/raw/ice_sheets/greenland/glaude2024/glaude2024_integrated_smb.csv
(Gt/yr, mass-gain convention).

Usage:  python scripts/integrate_glaude2024_smb.py
"""

from pathlib import Path

import h5py
import netCDF4 as nc
import numpy as np
import pandas as pd
from pyproj import Proj

ROOT = Path(__file__).resolve().parents[1]
GRID_DIR = Path('/Users/minchew/Documents/Research/SLR/forecasting/'
                'global_simple_v1/slr_forecast/data/raw/glaude2024')
MASK = ROOT / 'data/raw/ice_sheets/greenland/noel2021/Icemask_Topography_lon_lat_average_1km_GrIS.nc'
OUT = ROOT / 'data/raw/ice_sheets/greenland/glaude2024/glaude2024_integrated_smb.csv'
P3413 = Proj('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +ellps=WGS84')
MM_KM2_TO_GT = 1e-6


def areal_scale(X, Y):
    lon, lat = P3413(X, Y, inverse=True)
    return P3413.get_factors(lon, lat).areal_scale


def racmo_grid():
    m = nc.Dataset(MASK)
    xc = m['x'][:].astype(float) - 500.0
    yc = m['y'][:].astype(float) - 500.0
    mask = np.asarray(m['Promicemask'][:])
    return xc, yc, mask


def map_mask(xc, yc, mask, X, Y):
    """Nearest-centre lookup of the RACMO-grid mask at ISMIP6 centres."""
    dx = np.median(np.diff(xc)); dy = np.median(np.diff(yc))
    j = np.rint((X - xc[0]) / dx).astype(int)
    i = np.rint((Y - yc[0]) / dy).astype(int)
    ok_j = (j >= 0) & (j < len(xc)); ok_i = (i >= 0) & (i < len(yc))
    out = np.zeros((len(Y), len(X)))
    sub = mask[np.ix_(i[ok_i], j[ok_j])]
    out[np.ix_(np.where(ok_i)[0], np.where(ok_j)[0])] = sub
    return out


def integrate(get_field, nt, mask, area, fill_test, nn_fill=False):
    """Annual totals over the ice sheet and ice sheet + periphery.  With
    nn_fill, missing cells take the value of the nearest valid cell (used for
    HIRHAM, whose bilinear remap leaves the outermost 1-6 km of the margin
    empty); the unfilled ice-sheet total is returned as well."""
    from scipy import ndimage
    rows = []
    sheet, periph = mask == 3, (mask >= 1) & (mask <= 3)
    for k in range(nt):
        f = get_field(k).astype(float)
        bad = fill_test(f)
        f = np.where(bad, np.nan, f)
        raw_sheet = np.nansum((f * area)[sheet]) * MM_KM2_TO_GT
        if nn_fill:
            idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
            f = f[tuple(idx)]
        w = f * area
        rows.append((np.nansum(w[sheet]) * MM_KM2_TO_GT,
                     np.nansum(w[periph]) * MM_KM2_TO_GT,
                     1.0 - np.mean(bad[sheet]), raw_sheet))
    return np.array(rows)


def main():
    xc, yc, mask_r = racmo_grid()
    out = {}

    # RACMO on its own grid
    d = nc.Dataset(GRID_DIR / 'RACMO_SMB_1km.nc')
    t = d['time'][:]
    years_r = 1950 + np.floor(np.asarray(t) / 365.2425).astype(int)
    XX, YY = np.meshgrid(xc, yc)
    area_r = 1.0 / areal_scale(XX, YY)
    v = d['SMB']; v.set_auto_mask(False)
    r = integrate(lambda k: v[k], len(t), mask_r, area_r, lambda f: np.abs(f) > 1e20)
    out['RACMO'] = pd.DataFrame(r, index=years_r, columns=['sheet', 'with_periph', 'coverage', 'sheet_unfilled'])

    # ISMIP6 grid (MAR, HIRHAM)
    fm = h5py.File(GRID_DIR / 'MAR_SMB_1km.nc', 'r')
    X = fm['x'][:].astype(float); Y = fm['y'][:].astype(float)
    mask_i = map_mask(xc, yc, mask_r, X, Y)
    XX, YY = np.meshgrid(X, Y)
    area_i = 1.0 / areal_scale(XX, YY)
    print(f'ice-sheet area: RACMO grid {area_r[mask_r == 3].sum():.0f} km2, '
          f'ISMIP6 grid {area_i[mask_i == 3].sum():.0f} km2')

    years_m = (1900 + np.floor(fm['time'][:] + 0.5)).astype(int)
    r = integrate(lambda k: fm['SMB'][k], len(years_m), mask_i, area_i, lambda f: np.abs(f) > 1e30)
    out['MAR'] = pd.DataFrame(r, index=years_m, columns=['sheet', 'with_periph', 'coverage', 'sheet_unfilled'])

    fh = h5py.File(GRID_DIR / 'HIRHAM_SMB_1km.nc', 'r')
    tb = fh['time_bnds'][:, 0]
    years_h = (1950 + np.floor((tb + 31) / 365.0)).astype(int)   # days since 1949-12-01, 365_day
    r = integrate(lambda k: fh['SMB'][k, 0], len(years_h), mask_i, area_i, lambda f: f <= -9998,
                  nn_fill=True)
    out['HIRHAM'] = pd.DataFrame(r, index=years_h, columns=['sheet', 'with_periph', 'coverage', 'sheet_unfilled'])

    df = pd.concat(out, axis=1)
    df.columns = [f'{a}_{b}' for a, b in df.columns]
    df.index.name = 'year'
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, float_format='%.2f')
    print(f'saved {OUT}')
    for mname in ['RACMO', 'MAR', 'HIRHAM']:
        s = out[mname]
        print(f'{mname:6s} {s.index.min()}-{s.index.max()}  coverage min {s["coverage"].min():.4f}  '
              f'2080-2099 mean: sheet {s.loc[2080:2099, "sheet"].mean():.0f}, '
              f'with periph {s.loc[2080:2099, "with_periph"].mean():.0f} Gt/yr;  '
              f'1980-1999 mean sheet {s.loc[1980:1999, "sheet"].mean():.0f};  '
              f'2080-2099 unfilled sheet {s.loc[2080:2099, "sheet_unfilled"].mean():.0f}')


if __name__ == '__main__':
    main()
