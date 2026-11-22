"""Global ocean heat content from IAP v4.2 (Cheng et al.), monthly, 1950 onward.

Input to the calibration-record sensitivity test of the two-layer thermosteric
model (scripts/thermosteric_record_sensitivity.py).  For each month the IAP
v4.2 0-6000 m heat-content anomaly file (1 degree grid, J m-2 per layer) is
downloaded, integrated over the globe for the 0-300, 0-700, 0-2000 and
0-6000 m layers, and deleted.  One row per month is appended to the CSV, so
the run can be resumed; months already in the CSV are skipped.

Usage
-----
  python -I scripts/iap_ohc_global.py --csv data/processed/iap_v42_ohc_global_monthly.csv \
      --tmp DIR [--first-year 1950] [--last-year 2025]
"""
import argparse
import http.client
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import netCDF4 as nc
import numpy as np

URL = ('http://www.ocean.iap.ac.cn/ftp/cheng/IAPv4.2_Ocean_Heat_Content_anomaly_0_6000m/'
       'Monthly/OHCa_IAP_0_6000m_year_{y}_month_{m}.nc')
LAYERS = ('OHC300', 'OHC700', 'OHC2000', 'OHC6000')
R_EARTH = 6.371e6


def integrate(path):
    """Global integrals (ZJ) of the IAP heat-content layers in one monthly file."""
    d = nc.Dataset(path)
    lat = d['lat'][:].astype(float)
    area = (np.deg2rad(1.0) * R_EARTH**2
            * (np.sin(np.deg2rad(lat + 0.5)) - np.sin(np.deg2rad(lat - 0.5))))[:, None]
    out = []
    for v in LAYERS:
        x = np.ma.filled(d[v][:].astype(float), np.nan)
        x[np.abs(x) > 1e15] = np.nan          # fill values
        out.append(np.nansum(x * area) / 1e21)
    d.close()
    return out


def download(url, dest, timeout=300, attempts=4):
    """Download url to dest. Return False if the server reports it missing (HTTP 4xx).

    Transient failures (network errors, HTTP 5xx) are retried up to `attempts`
    times in all; the file is written under a temporary name and renamed when
    complete, so an interrupted download never leaves a partial file at dest.
    """
    part = dest.with_name(dest.name + '.part')
    for k in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, part.open('wb') as fh:
                shutil.copyfileobj(r, fh)
            part.replace(dest)
            return True
        except urllib.error.HTTPError as e:
            part.unlink(missing_ok=True)
            if e.code < 500:
                return False
            err = e
        except (OSError, http.client.HTTPException) as e:   # network errors, timeouts, truncated reads
            part.unlink(missing_ok=True)
            err = e
        if k < attempts - 1:
            time.sleep(2 ** k)
    print(f'  download failed after {attempts} attempts: {err}', file=sys.stderr)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--tmp', required=True)
    ap.add_argument('--first-year', type=int, default=1950)
    ap.add_argument('--last-year', type=int, default=2025)
    a = ap.parse_args()
    csv, tmp = Path(a.csv), Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    done = set()
    if csv.exists():
        done = {line.split(',')[0] for line in csv.read_text().splitlines()[1:]}
    else:
        csv.write_text('yyyymm,' + ','.join(f'{v.lower()}_zj' for v in LAYERS) + '\n')
    for y in range(a.first_year, a.last_year + 1):
        for m in range(1, 13):
            key = f'{y}{m:02d}'
            if key in done:
                continue
            f = tmp / f'iap_{key}.nc'
            if not download(URL.format(y=y, m=f'{m:02d}'), f):
                print(f'  {key}: not available', file=sys.stderr)
                continue
            vals = integrate(f)
            f.unlink()
            with csv.open('a') as fh:
                fh.write(key + ',' + ','.join(f'{v:.4f}' for v in vals) + '\n')
            print(f'  {key}', flush=True)


if __name__ == '__main__':
    main()
