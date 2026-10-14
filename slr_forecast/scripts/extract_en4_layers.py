"""Extract EN4 Greenland-peripheral ocean temperature for alternative layers.

One-time extraction for the depth-layer sensitivity test in
component_greenland.ipynb (§6). The production forcing averages 200-500 m,
the characteristic grounding-line depth range of Greenland tidewater
glaciers (Slater et al., 2019, 2020; Morlighem et al., 2017). The test
shifts that 300 m window 100 m shallower (100-400 m) and 100 m deeper
(300-600 m), with the same region (58-80 N, 75-5 W) and the same reader.

The production layer is re-extracted first and compared with the series in
slr_processed_data.h5, so the alternative layers are built exactly as the
production one was. Writes en4_greenland_peripheral_{lo}_{hi}m.csv to
data/processed. Not part of run_pipeline.sh (reads ~50 yearly archives).

Run from the repository root:  python scripts/extract_en4_layers.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from slr_forecast.readers.ocean_temp import read_en4_regional  # noqa: E402

EN4_DIR = ROOT / 'data' / 'raw' / 'ocean_temp' / 'en4'
OUT_DIR = ROOT / 'data' / 'processed'
H5 = OUT_DIR / 'slr_processed_data.h5'
LAYERS = [(200, 500), (100, 400), (300, 600)]


def main():
    prod = pd.read_hdf(H5, 'harmonized/df_en4_greenland_200_500m')
    for lo, hi in LAYERS:
        df = read_en4_regional(EN4_DIR, depth_bounds=(float(lo), float(hi)))
        if (lo, hi) == (200, 500):
            a = df['temperature'] - df['temperature'].mean()
            b = prod['temperature'].reindex(df.index)
            b = b - b.mean()
            diff = np.nanmax(np.abs(a.values - b.values))
            print(f'200-500 m re-extraction vs production (demeaned): '
                  f'max |diff| {diff:.2e} degC over {b.notna().sum()} months')
            if not diff < 1e-6:
                raise SystemExit('Re-extraction does not reproduce the production series.')
            continue
        out = OUT_DIR / f'en4_greenland_peripheral_{lo}_{hi}m.csv'
        df.to_csv(out)
        print(f'{lo}-{hi} m: {len(df)} months, '
              f'{df["decimal_year"].iloc[0]:.1f}-{df["decimal_year"].iloc[-1]:.1f} -> {out.name}')


if __name__ == '__main__':
    main()
