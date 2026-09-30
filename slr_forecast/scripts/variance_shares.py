"""Shares of the 2100 forecast variance by component (main text and supplement).

Definition, used in both documents: at 2100, for each warming trajectory,
the total is the unblended sum of the component models (thermosteric,
glaciers, Greenland, EAIS, Antarctic Peninsula, WAIS), without terrestrial
water storage. Each share is var(group) / var(total), where a group's variance
is that of its per-sample sum, so it includes the covariance among the group's
members from the shared warming path. WAIS is drawn independently of the
other components, so the WAIS share and the calibrated share nearly sum to one
with EAIS and the Peninsula taking the remainder.

Groups:
  calibrated  thermosteric + glaciers + Greenland (the GMST-sensitive
              components with calibrated sensitivities)
  wais        WAIS mixture at the fast-WAIS weighting, p(S1) = 0.10
              (wais_2k/full_samples, the samples summed in the forecast)

Run from the repository root:  python scripts/variance_shares.py
"""

import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
H5 = ROOT / 'data' / 'processed' / 'component_results.h5'
SSPS = {'SSP1-2.6': '2 C', 'SSP2-4.5': '3 C', 'SSP3-7.0': '4 C'}
YEAR = 2100.0


def at_year(years, arr, year=YEAR):
    return arr[:, int(np.argmin(np.abs(years - year)))]


def main():
    with h5py.File(H5, 'r') as f:
        py = f['proj_years'][:]
        wy = f['wais_2k/years'][:]
        wais = at_year(wy, f['wais_2k/full_samples'][:])
        for ssp, label in SSPS.items():
            comp = {c: at_year(py, f[f'{c}/projections/{ssp}/samples'][:])
                    for c in ['ocean', 'glacier', 'greenland', 'eais', 'apeninsula']}
            calibrated = comp['ocean'] + comp['glacier'] + comp['greenland']
            total = calibrated + comp['eais'] + comp['apeninsula'] + wais
            v = np.var(total, ddof=1)
            print(f'{label} ({ssp}): WAIS {100 * np.var(wais, ddof=1) / v:5.1f}%   '
                  f'calibrated {100 * np.var(calibrated, ddof=1) / v:4.2f}%')


if __name__ == '__main__':
    sys.exit(main())
