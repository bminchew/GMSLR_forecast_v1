"""Greenland-only cell mask for the Berkeley Earth 1-degree grid.

The Greenland regional surface temperature (component_greenland.ipynb) was
the land average over the box 60-84 N, 73-12 W, which also takes in parts
of Baffin Island, Ellesmere Island and Iceland (~13% of the weight). This
script lists the grid cells of that box whose centre lies within the
Greenland outline of Natural Earth (1:10m admin-1, admin == 'Greenland'),
buffered by 0.5 degree so coastal cells are kept. The reader
(read_berkeley_earth_gridded, cell_mask=...) then keeps only those cells,
with the same land-fraction threshold and weights as before.

Writes data/processed/greenland_mask_berkeley_1deg.csv (lat, lon of the
included cell centres). One-time; not part of run_pipeline.sh (needs the
Natural Earth shapefile that cartopy caches).

Run from the repository root:  python scripts/make_greenland_mask.py
"""

from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
GRIDDED = ROOT / 'data' / 'raw' / 'gmst' / 'berkEarth_Global_TAVG_Gridded_1deg.nc'
OUT = ROOT / 'data' / 'processed' / 'greenland_mask_berkeley_1deg.csv'
LAT_BOUNDS, LON_BOUNDS = (60.0, 84.0), (-73.0, -12.0)   # reader defaults
BUFFER_DEG = 0.5


def main():
    shp = shpreader.natural_earth(resolution='10m', category='cultural',
                                  name='admin_1_states_provinces_lakes')
    outline = unary_union([r.geometry for r in shpreader.Reader(shp).records()
                           if r.attributes.get('admin') == 'Greenland'])
    region = outline.buffer(BUFFER_DEG)

    with nc.Dataset(str(GRIDDED)) as ds:
        lat = np.asarray(ds.variables['latitude'][:])
        lon = np.asarray(ds.variables['longitude'][:])
    lat_b = lat[(lat >= LAT_BOUNDS[0]) & (lat <= LAT_BOUNDS[1])]
    lon_b = lon[(lon >= LON_BOUNDS[0]) & (lon <= LON_BOUNDS[1])]
    cells = [(y, x) for y in lat_b for x in lon_b if region.contains(Point(x, y))]
    df = pd.DataFrame(cells, columns=['lat', 'lon'])
    df.to_csv(OUT, index=False)
    print(f'{len(df)} of {len(lat_b) * len(lon_b)} box cells inside the buffered '
          f'Greenland outline -> {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
