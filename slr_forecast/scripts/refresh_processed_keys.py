"""Regenerate selected keys of data/processed/slr_processed_data.h5 with the current readers.

slr_processed_data.h5 is built by notebooks/read_process_datafiles.ipynb, which is not part of
run_pipeline.sh. Tables stored before the following reader fixes differ from the current readers:

- Berkeley Earth: temperature_sigma = unc / 1.645 before the fix; the reader uses unc / 1.96
  (raw/df_berkeley, harmonized/df_berkeley_h, kinematics/gmst/berkeley).
- IMBIE WAIS (2021): negative sigma before the np.abs fix (raw/df_imbie_wais, harmonized/df_imbie_wais_h).
- Dangendorf (2024): negative gia_sigma (raw/df_dangendorf, harmonized/df_dangendorf_h).
- Horwath (2022): negative steric_deep_sigma, signed as in the source file (raw/df_horwath,
  harmonized/df_horwath_h).
- IPCC AR6 observed GMSL: gmsl_sigma was half the 17-83% range; it is now that half-range / 0.9542
  (raw/df_ipcc_observed_gmsl, harmonized/df_ipcc_observed_gmsl_h, kinematics/gmsl/ipcc).

This script rebuilds only those keys, following read_process_datafiles.ipynb (read, convert to standard
units, harmonize to 1995-2005; kinematics on the unharmonized series with a 30-yr span). It compares every
column with the stored table and refuses to write if any non-sigma column changes. Differences at
round-off level (max |diff| <= 1e-12, or a one-point tie in n_effective) are reported as noise; --write
still refuses to overwrite a stored key when any non-sigma column differs at all, and prints the largest
difference, unless --accept-differences is given. Larger non-sigma differences always stop the script.

It also writes the keys that read_process_datafiles.ipynb does not write (its mode='w' save drops them),
so run it with --write after every rebuild of the file:

- NOAA yearly thermosteric sea level, read as component_ocean.ipynb reads it
  (harmonized/df_noaa_tsl_0_700m, read by tests/test_ocean.py; harmonized/df_noaa_tsl_0_2000m, which no
  pipeline or test code reads and is kept for completeness next to the 0-700 m key; the pipeline reads
  the NOAA .dat files directly).
- EN4.2.2 monthly regional means, 1970-01 to 2021-06 (harmonized/df_en4_greenland_200_500m, the
  Greenland discharge forcing; harmonized/df_en4_global_0_700m, component_ocean diagnostic plots).
  These read 52 yearly EN4 archives (about 15 GB of data/raw/ocean_temp/en4, about 1 min per key);
  --skip-en4 leaves them out.

A key absent from the store is added without comparison.

Usage:
    python scripts/refresh_processed_keys.py            # dry run: print the comparison only
    python scripts/refresh_processed_keys.py --write    # replace the keys in place
    python scripts/refresh_processed_keys.py --write --skip-en4   # without the two EN4 keys
    python scripts/refresh_processed_keys.py --write --accept-differences   # also overwrite noise-level changes
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'notebooks'))

from slr_data_readers import (  # noqa: E402
    read_berkeley_earth, read_dangendorf2024_fields, read_imbie_west_antarctica, read_horwath2022,
    read_ipcc_ar6_observed_gmsl,
    convert_to_standard_units, datetime_to_decimal_year,
)
from slr_analysis import harmonize_baseline, compute_kinematics  # noqa: E402
from slr_forecast.readers.forcing import read_noaa_thermosteric_yearly  # noqa: E402
from slr_forecast.readers.ocean_temp import read_en4_regional  # noqa: E402

H5 = ROOT / 'data' / 'processed' / 'slr_processed_data.h5'
RAW = ROOT / 'data' / 'raw'
BASELINE = (1995, 2005)
SPAN_YEARS = 30
# Years of the stored EN4 keys. data/raw/ocean_temp/en4 also holds 1900-1969 archives, which the
# reader would include without year_range.
EN4_YEARS = (1970, 2021)


def _decimal_year(df):
    if 'decimal_year' in df.columns:
        return df['decimal_year'].values
    return np.array([datetime_to_decimal_year(t) for t in df.index])


def build(include_en4=True):
    """Return {key: DataFrame} for the keys this script refreshes."""
    out = {}

    df_b = convert_to_standard_units(
        read_berkeley_earth(RAW / 'gmst' / 'berkEarth_globalmean_airTaboveseaice.txt'))
    out['raw/df_berkeley'] = df_b
    out['harmonized/df_berkeley_h'] = harmonize_baseline(df_b, 'temperature', BASELINE)
    out['kinematics/gmst/berkeley'] = compute_kinematics(
        time=_decimal_year(df_b), value=df_b['temperature'].values,
        sigma=df_b['temperature_sigma'].values, span_years=SPAN_YEARS).to_dataframe()

    df_w = convert_to_standard_units(read_imbie_west_antarctica(
        RAW / 'ice_sheets' / 'antarctica' / 'imbie_west_antarctica_2021_mm.csv'))
    out['raw/df_imbie_wais'] = df_w
    out['harmonized/df_imbie_wais_h'] = harmonize_baseline(df_w, 'cumulative_mass_balance', BASELINE)

    df_d = convert_to_standard_units(read_dangendorf2024_fields(RAW / 'gmslr'))
    out['raw/df_dangendorf'] = df_d
    out['harmonized/df_dangendorf_h'] = harmonize_baseline(df_d, 'gmsl', BASELINE)

    df_h = convert_to_standard_units(read_horwath2022(RAW / 'gmslr' / 'horwath2021_ESACCI_SLBC_v2.2.csv'))
    out['raw/df_horwath'] = df_h
    out['harmonized/df_horwath_h'] = harmonize_baseline(df_h, 'gmsl', BASELINE)

    df_i = convert_to_standard_units(read_ipcc_ar6_observed_gmsl(
        RAW / 'gmslr' / 'ipccar6_global_sea_level_observed.csv'))
    out['raw/df_ipcc_observed_gmsl'] = df_i
    out['harmonized/df_ipcc_observed_gmsl_h'] = harmonize_baseline(df_i, 'gmsl', BASELINE)
    out['kinematics/gmsl/ipcc'] = compute_kinematics(
        time=_decimal_year(df_i), value=df_i['gmsl'].values,
        sigma=df_i['gmsl_sigma'].values, span_years=SPAN_YEARS).to_dataframe()

    # Keys not written by read_process_datafiles.ipynb
    for layer in ('0-700m', '0-2000m'):
        out[f'harmonized/df_noaa_tsl_{layer.replace("-", "_")}'] = read_noaa_thermosteric_yearly(
            RAW / 'steric' / f'noaa_tsl_yearly_{layer}.dat')
    if include_en4:
        en4_dir = RAW / 'ocean_temp' / 'en4'
        # Reader defaults: 58-80 N, 75-5 W, 200-500 m (component_greenland.ipynb)
        out['harmonized/df_en4_greenland_200_500m'] = read_en4_regional(en4_dir, year_range=EN4_YEARS)
        # Same arguments as the fallback in component_ocean.ipynb
        out['harmonized/df_en4_global_0_700m'] = read_en4_regional(
            en4_dir, lat_bounds=(-90.0, 90.0), lon_bounds=(-180.0, 180.0),
            depth_bounds=(0.0, 700.0), year_range=EN4_YEARS)
    return out


def compare(key, old, new):
    """Print column-by-column differences.

    Return (ok, max_nonsigma_diff). ok is False if the index or the set of columns differs, or a
    non-sigma column changes beyond round-off noise. max_nonsigma_diff is the largest |difference| in
    any non-sigma column, noise included (0.0 when every non-sigma column is identical).
    """
    ok = True
    max_nonsigma = 0.0
    if not old.index.equals(new.index):
        print(f'  {key}: INDEX DIFFERS ({len(old)} vs {len(new)} rows)')
        return False, np.inf
    cols = sorted(set(old.columns) | set(new.columns))
    for c in cols:
        if c not in old.columns or c not in new.columns:
            print(f'  {key}: column {c!r} only in {"new" if c in new.columns else "old"}')
            ok = False
            continue
        a = pd.to_numeric(old[c], errors='coerce').to_numpy(float)
        b = pd.to_numeric(new[c], errors='coerce').to_numpy(float)
        same_nan = np.array_equal(np.isnan(a), np.isnan(b))
        d = np.nanmax(np.abs(a - b)) if np.isfinite(a - b).any() else 0.0
        if d == 0.0 and same_nan:
            continue
        is_sigma = 'sigma' in c or c.endswith('_se')
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = np.nanmedian(a / b) if is_sigma else np.nan
        # Round-off in rate/accel, and a one-point tie at the kernel-window edge in n_effective,
        # are not value changes.
        noise = same_nan and (d <= 1e-12 or (c == 'n_effective' and d <= 1))
        tag = 'sigma' if is_sigma else ('noise' if noise else 'VALUE')
        print(f'  {key}: {tag} column {c!r} max|diff| = {d:.3e}'
              + (f', median old/new = {ratio:.4f}' if is_sigma else '')
              + ('' if same_nan else ', NaN pattern differs'))
        if not is_sigma:
            max_nonsigma = max(max_nonsigma, d if same_nan else np.inf)
        if not (is_sigma or noise):
            ok = False
    return ok, max_nonsigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true', help='replace the keys in place')
    ap.add_argument('--skip-en4', action='store_true',
                    help='leave out the two EN4 keys (they need data/raw/ocean_temp/en4)')
    ap.add_argument('--accept-differences', action='store_true',
                    help='with --write, also overwrite keys whose non-sigma columns differ at '
                         'round-off level')
    args = ap.parse_args()

    new = build(include_en4=not args.skip_en4)
    all_ok = True
    max_nonsigma = 0.0
    with pd.HDFStore(str(H5), mode='r') as store:
        for key, df in new.items():
            print(f'{key}:')
            if key not in store:
                print('  absent from the store; will be added')
                continue
            ok, d = compare(key, store[key], df)
            all_ok &= ok
            max_nonsigma = max(max_nonsigma, d)

    if not all_ok:
        print('\nNon-sigma columns differ; not writing. Inspect before regenerating.')
        sys.exit(1)
    if not args.write:
        print('\nDry run: only sigma columns differ beyond round-off. Re-run with --write to replace '
              'the keys.')
        if max_nonsigma > 0:
            print(f'Non-sigma columns differ by up to {max_nonsigma:.3e}; --write will also need '
                  '--accept-differences.')
        return
    if max_nonsigma > 0 and not args.accept_differences:
        print(f'\nNon-sigma columns differ by up to {max_nonsigma:.3e} (round-off level); not writing. '
              'Re-run with --write --accept-differences to replace the keys anyway.')
        sys.exit(1)
    with pd.HDFStore(str(H5), mode='a', complevel=5, complib='blosc:zstd') as store:
        for key, df in new.items():
            store.put(key, df)
    print(f'\nReplaced {len(new)} keys in {H5}')


if __name__ == '__main__':
    main()
