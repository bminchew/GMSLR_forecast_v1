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
              (wais_2k/full_samples, the samples summed in the forecast;
              the realized S1 fraction of that draw is printed, 0.0975 in
              the 2026-10-18 run, SF-12)

A second table repeats the WAIS share as p(S1) varies. The WAIS variance at
weight p is that of the two-branch mixture, from the S1 branch
(wais_2k/s1_samples) and the S2 members of the full mixture
(wais_2k/full_samples where scenario_idx == 1):
  p var(S1) + (1 - p) var(S2) + p (1 - p) (mean(S1) - mean(S2))^2.
WAIS is independent of the other components, so the total variance is this
plus the variance of their sum. p(S1) = 0.9 is the slow-WAIS presentation case;
it shows that a 10% chance of fast loss is enough for WAIS to dominate the
variance (abstract).

Run from the repository root:  python scripts/variance_shares.py
"""

import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
H5 = ROOT / 'data' / 'processed' / 'component_results.h5'
SSPS = {'SSP1-2.6': '1.8 C', 'SSP2-4.5': '2.7 C', 'SSP3-7.0': '3.6 C'}
YEAR = 2100.0
P_S1 = [1.0, 0.9, 0.5, 0.1]


def at_year(years, arr, year=YEAR):
    return arr[:, int(np.argmin(np.abs(years - year)))]


def main():
    with h5py.File(H5, 'r') as f:
        py = f['proj_years'][:]
        wy = f['wais_2k/years'][:]
        wais = at_year(wy, f['wais_2k/full_samples'][:])
        print(f"Realized S1 fraction of wais_2k/full_samples: "
              f"{np.mean(f['wais_2k/scenario_idx'][:] == 0):.4f}")
        for ssp, label in SSPS.items():
            comp = {c: at_year(py, f[f'{c}/projections/{ssp}/samples'][:])
                    for c in ['ocean', 'glacier', 'greenland', 'eais', 'apeninsula']}
            calibrated = comp['ocean'] + comp['glacier'] + comp['greenland']
            total = calibrated + comp['eais'] + comp['apeninsula'] + wais
            v = np.var(total, ddof=1)
            print(f'{label} ({ssp}): WAIS {100 * np.var(wais, ddof=1) / v:5.1f}%   '
                  f'calibrated {100 * np.var(calibrated, ddof=1) / v:4.2f}%')

        # WAIS share as a function of the slow-WAIS weight p(S1)
        s1 = at_year(wy, f['wais_2k/s1_samples'][:])
        s2 = wais[f['wais_2k/scenario_idx'][:].astype(int) == 1]
        print('\nWAIS share of 2100 variance vs p(S1)')
        print(f'{"p(S1)":>6}' + ''.join(f'{lab:>9}' for lab in SSPS.values()))
        for p in P_S1:
            row = f'{p:6.2f}'
            for ssp in SSPS:
                other = sum(at_year(py, f[f'{c}/projections/{ssp}/samples'][:])
                            for c in ['ocean', 'glacier', 'greenland', 'eais', 'apeninsula'])
                vw = (p * np.var(s1, ddof=1) + (1 - p) * np.var(s2, ddof=1)
                      + p * (1 - p) * (s1.mean() - s2.mean()) ** 2)
                row += f'{100 * vw / (vw + np.var(other, ddof=1)):8.1f}%'
            print(row)


if __name__ == '__main__':
    sys.exit(main())
