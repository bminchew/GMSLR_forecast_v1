#!/usr/bin/env python3
"""
Transient sea-level sensitivity (TSLS) of the glacier and thermosteric projections.

Follows the definition of Grinsted et al. (2022, Earth's Future,
doi:10.1029/2022EF002696): across emission scenarios, regress the 2051-2100
mean contribution rate on the 2051-2100 mean GMST. The slope, in
mm yr^-1 degC^-1, is the TSLS. Grinsted et al. report 1.5 +/- 0.2 (steric)
and 0.7 +/- 0.1 (glaciers) for CMIP6-forced projections, 90% ranges.

The same metric is applied to the IPCC AR6 medians as a check. Rates are
differences of levels, so the per-component rebase offset in
ipcc_distributions.h5 cancels.

Scenarios: SSP1-2.6, SSP2-4.5, SSP3-7.0 (the 2, 3, 4 degC cases of the paper).
The GMST grid ends in 2099, so the temperature mean spans 2051-2099.

Usage:
    python scripts/transient_sensitivity.py

Output:
    printed summary only
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROCESSED = Path(__file__).resolve().parents[1] / 'data' / 'processed'
SCENARIOS = {'SSP1-2.6': 'SSP1_2_6', 'SSP2-4.5': 'SSP2_4_5', 'SSP3-7.0': 'SSP3_7_0'}
M_TO_MM = 1000.0


def mean_gmst_2051_2100(key):
    """Mean of the median GMST trajectory over 2051-2100 (degC)."""
    df = pd.read_hdf(PROCESSED / 'slr_processed_data.h5', f'projections/temp/{key}')
    years = df['decimal_year'].values
    return df['temperature'].values[(years >= 2051) & (years <= 2100)].mean()


def main():
    T = np.array([mean_gmst_2051_2100(k) for k in SCENARIOS.values()])
    print('2051-2100 mean GMST (degC):', dict(zip(SCENARIOS, T.round(2))))

    with h5py.File(PROCESSED / 'component_results.h5', 'r') as f:
        t = f['glacier/projections/SSP2-4.5/projection_times'][:]
        i2015, i2050, i2100 = (np.argmin(abs(t - y)) for y in (2015, 2050, 2100))

        for comp, keys in [('ocean', ['samples', 'upper_samples']), ('glacier', ['samples'])]:
            for key in keys:
                # Rate from the median trajectories (samples are not paired
                # draw-by-draw across scenarios, so no per-draw slope)
                rates = []
                for s in SCENARIOS:
                    S = f[f'{comp}/projections/{s}/{key}'][:] * M_TO_MM
                    rates.append((np.median(S[:, i2100]) - np.median(S[:, i2050])) / 50.0)
                slope = np.polyfit(T, rates, 1)[0]
                print(f'{comp:8s} {key:14s} rates {np.round(rates, 2)} mm/yr  TSLS {slope:.2f}')

        # 2015-2100 contributions, for comparison with Rounce et al. (2023),
        # Marzeion et al. (2020), and Edwards et al. (2021), all relative to 2015
        for comp in ['glacier', 'greenland']:
            for s in SCENARIOS:
                S = f[f'{comp}/projections/{s}/samples'][:] * M_TO_MM
                p5, p50, p95 = np.percentile(S[:, i2100] - S[:, i2015], [5, 50, 95])
                print(f'{comp} {s} 2015-2100: {p50:.0f} [{p5:.0f}, {p95:.0f}] mm')

    with h5py.File(PROCESSED / 'ipcc_distributions.h5', 'r') as g:
        for comp in ['oceandynamics', 'glaciers']:
            rates = []
            for s in SCENARIOS:
                y = g[f'params/{comp}/{s}/years'][:]
                q = g[f'params/{comp}/{s}/q50'][:]
                rates.append((np.interp(2100, y, q) - np.interp(2050, y, q)) / 50.0)
            slope = np.polyfit(T, rates, 1)[0]
            print(f'AR6 {comp:14s} rates {np.round(rates, 2)} mm/yr  TSLS {slope:.2f}')


if __name__ == '__main__':
    main()
