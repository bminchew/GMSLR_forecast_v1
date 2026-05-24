"""
Compute fractional risk attribution and save to component_results.h5.

Risk = E[D(S)] where D is the Jevrejeva (2018) damage function.
Each component's attributed risk = (mean SLR_comp / mean SLR_total) × E[D(total)].

Usage:
    python compute_shapley_risk.py

Writes to: data/processed/component_results.h5 under group 'risk_shapley/'
"""
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slr_forecast.config import PROCESSED_DATA_DIR

H5_COMP = PROCESSED_DATA_DIR / 'component_results.h5'

HORIZONS = [2050, 2075, 2100, 2150]
SSPS = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0']
WARMING_LABELS = {'SSP1-2.6': '2°C', 'SSP2-4.5': '3°C', 'SSP3-7.0': '4°C'}
COMPONENTS = ['ocean', 'glacier', 'greenland', 'apeninsula', 'wais', 'eais']
COMP_LABELS = {
    'ocean': 'Thermosteric', 'glacier': 'Glaciers',
    'greenland': 'Greenland', 'apeninsula': 'Peninsula',
    'wais': 'WAIS', 'eais': 'EAIS',
}

# ── Vectorized damage function (numpy piecewise, no Python loop) ──
_SLR_KNOTS = np.array([0.00, 0.20, 0.52, 0.63, 0.86, 1.80])
_COST_KNOTS = np.array([0.0, 1000.0, 10200.0, 11700.0, 14000.0, 27000.0])

def damage_vec(slr_m):
    """Vectorized Jevrejeva (2018) damage function. Input/output: arrays."""
    slr_m = np.asarray(slr_m, dtype=float)
    result = np.interp(slr_m, _SLR_KNOTS, _COST_KNOTS)
    # Extrapolate beyond 1.80 m with final slope
    slope = (_COST_KNOTS[-1] - _COST_KNOTS[-2]) / (_SLR_KNOTS[-1] - _SLR_KNOTS[-2])
    mask = slr_m > _SLR_KNOTS[-1]
    result[mask] = _COST_KNOTS[-1] + slope * (slr_m[mask] - _SLR_KNOTS[-1])
    return result


def main():
    comp_time = np.arange(1950, 2151, dtype=float)

    print('Computing fractional risk attribution ...')

    all_risk_comp = {}    # {ssp: {comp: {yr: val}}}
    all_risk_total = {}   # {ssp: {yr: val}}

    for ssp in SSPS:
        warming = WARMING_LABELS[ssp]

        with h5py.File(str(H5_COMP), 'r') as hf:
            blended_yr = hf['blended/forecast_years'][:]

            # Component mean SLR at each horizon
            comp_mean = {}
            for comp in COMPONENTS:
                comp_mean[comp] = {}
                if comp == 'wais':
                    s = hf['wais_2k/full_samples'][:]
                    t = hf['wais_2k/years'][:]
                else:
                    s = hf[f'{comp}/projections/{ssp}/samples'][:]
                    t = comp_time
                for yr in HORIZONS:
                    iy = np.argmin(np.abs(t - yr))
                    comp_mean[comp][yr] = np.mean(s[:, iy])

            # Total SLR samples → E[D(total)]
            total_risk = {}
            total_mean_slr = {}
            for yr in HORIZONS:
                iy = np.argmin(np.abs(blended_yr - yr))
                samples = hf[f'blended/{ssp}/samples'][:, iy]
                total_risk[yr] = np.mean(damage_vec(samples))
                total_mean_slr[yr] = np.mean(samples)

        # Fractional attribution: mean is additive, so fractions sum to 1
        risk_comp = {}
        for comp in COMPONENTS:
            risk_comp[comp] = {}
            for yr in HORIZONS:
                frac = comp_mean[comp][yr] / total_mean_slr[yr]
                risk_comp[comp][yr] = frac * total_risk[yr]

        all_risk_comp[ssp] = risk_comp
        all_risk_total[ssp] = total_risk

        # Print
        print(f'\n=== {ssp} ({warming}) — Expected damages ($B/yr) ===')
        print(f'{"":>12s}', ''.join(f'{yr:>8d}' for yr in HORIZONS))
        print(f'{"Total":>12s}', ''.join(f'{total_risk[yr]:>8.0f}' for yr in HORIZONS))
        frac_sum = {yr: sum(risk_comp[c][yr] for c in COMPONENTS) for yr in HORIZONS}
        print(f'{"Frac sum":>12s}', ''.join(f'{frac_sum[yr]:>8.0f}' for yr in HORIZONS))
        print()
        for comp in COMPONENTS:
            lab = COMP_LABELS[comp]
            print(f'{lab:>12s}', ''.join(f'{risk_comp[comp][yr]:>8.0f}' for yr in HORIZONS))

    # ── Write to HDF5 ──
    with h5py.File(str(H5_COMP), 'a') as hf:
        grp_name = 'risk_shapley'
        if grp_name in hf:
            del hf[grp_name]
        g = hf.create_group(grp_name)
        g.attrs['ssps'] = ','.join(SSPS)
        g.attrs['warming_labels'] = ','.join(WARMING_LABELS[s] for s in SSPS)
        g.attrs['damage_function'] = 'Jevrejeva2018'
        g.attrs['method'] = 'Fractional attribution: (mean SLR_comp / mean SLR_total) × E[D(total)]'
        g.create_dataset('horizons', data=np.array(HORIZONS))
        g.create_dataset('components', data=np.array(COMPONENTS, dtype='S'))
        g.create_dataset('ssps', data=np.array(SSPS, dtype='S'))

        for ssp in SSPS:
            sg = g.create_group(ssp)
            attr_arr = np.array([[all_risk_comp[ssp][c][yr] for yr in HORIZONS]
                                  for c in COMPONENTS])
            sg.create_dataset('shapley_values', data=attr_arr)
            sg.create_dataset('risk_total_kde',
                              data=np.array([all_risk_total[ssp][yr] for yr in HORIZONS]))

    print(f'\nWritten to {H5_COMP} under group "{grp_name}/"')


if __name__ == '__main__':
    main()
